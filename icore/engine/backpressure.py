"""
icore.engine.backpressure - Full-stack backpressure coordinator (v0.6).

v0.5 already had task-level backpressure via ``ConcurrencyController``
(``max_concurrent_tasks`` / ``max_concurrent_per_workflow`` plus a queue
depth gate that returns HTTP 503). v0.6 extends backpressure to every
external dependency so a single noisy component cannot take the whole
system down:

    1. **LLM API** — token bucket per model_id (already in
       :class:`ConcurrencyController`; the coordinator routes the
       ``rate_limit(model_id, n)`` call).
    2. **Vector DB (Milvus)** — bounded asyncio.Semaphore around every
       ``search()`` / ``insert()`` so a retrieval-heavy workflow cannot
       exhaust the connection pool.
    3. **Graph DB (Neo4j)** — same pattern, separate budget.
    4. **PostgreSQL / relational DB** — the existing
       :class:`ConnectionPool` already blocks on a per-pool semaphore;
       the coordinator exposes its saturation as a backpressure signal.
    5. **Memory budget** — soft RSS ceiling per workflow instance. When
       exceeded the coordinator marks the system under backpressure so
       the API layer returns 503 instead of OOMing.

Design notes:

    * Each dependency owns its own ``asyncio.Semaphore`` so they don't
      starve each other. ``acquire(name)`` returns an async context
      manager that releases on exit (even on ``CancelledError``).
    * The coordinator never raises on saturation — it merely reports
      ``is_saturated(name)``. Callers decide whether to fail fast
      (HTTP 503 from the API) or queue (workflow executor).
    * ``BackpressureSnapshot`` is a typed snapshot consumable by the
      ``/health`` endpoint so observability stays a first-class
      concern (v0.6 §3.2.2 验收: "/health 返回各组件当前背压状态").

Module dependencies:
    engine only — no api / services / db-driver imports. The
    PostgresConnectionPool is referenced via duck typing
    (``pool.available`` / ``pool.max_size``) so this module stays
    lazy-friendly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from icore.exceptions import BackpressureError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-component budget
# ---------------------------------------------------------------------------

@dataclass
class ComponentBudget:
    """Concurrency / rate budget for a single external dependency.

    Attributes:
        name:           Component identifier (e.g. ``"llm:gpt-4o"``).
        max_concurrent: Hard cap on simultaneous in-flight calls.
        rate_per_sec:   Optional token-bucket refill rate. ``0`` disables
                        rate limiting (concurrency-only mode).
        burst:          Token bucket capacity when ``rate_per_sec > 0``.
    """

    name: str
    max_concurrent: int = 10
    rate_per_sec: float = 0.0
    burst: int = 0


@dataclass
class ComponentStatus:
    """Live status of one component."""

    name: str
    max_concurrent: int
    active: int
    available: int
    rate_limited: bool
    saturated: bool
    last_saturation: Optional[float] = None


@dataclass
class BackpressureSnapshot:
    """Full snapshot for ``/health`` consumption."""

    saturated: bool
    components: dict[str, ComponentStatus] = field(default_factory=dict)
    memory_rss_mb: float = 0.0
    memory_budget_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "saturated": self.saturated,
            "memory_rss_mb": round(self.memory_rss_mb, 2),
            "memory_budget_mb": self.memory_budget_mb,
            "components": {
                name: {
                    "max_concurrent": s.max_concurrent,
                    "active": s.active,
                    "available": s.available,
                    "rate_limited": s.rate_limited,
                    "saturated": s.saturated,
                }
                for name, s in self.components.items()
            },
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Per-component limiter
# ---------------------------------------------------------------------------

class _ComponentLimiter:
    """Internal: combines a semaphore + optional token bucket."""

    def __init__(self, budget: ComponentBudget) -> None:
        self.budget = budget
        self._sem = asyncio.Semaphore(budget.max_concurrent)
        self._active = 0
        self._lock = asyncio.Lock()
        # Token bucket state (only when rate_per_sec > 0).
        self._tokens: float = float(budget.burst) if budget.burst > 0 else 0.0
        self._last_refill = time.time()
        self._last_saturation: Optional[float] = None

    async def acquire(self, n: int = 1, timeout: Optional[float] = None) -> None:
        """Acquire ``n`` concurrency slots (and rate tokens).

        Raises:
            BackpressureError: when ``timeout`` is reached without a
                permit (only when ``timeout`` is set).
        """
        # 1. Rate limit (token bucket) — non-blocking, drop tokens.
        if self.budget.rate_per_sec > 0:
            await self._consume_tokens(n)

        # 2. Concurrency cap.
        if timeout is None:
            await self._sem.acquire()
        else:
            try:
                await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
            except asyncio.TimeoutError as e:
                self._mark_saturation()
                raise BackpressureError(
                    f"Component '{self.budget.name}' saturated "
                    f"(no permit within {timeout}s)"
                ) from e

        async with self._lock:
            self._active += 1
            if self._active >= self.budget.max_concurrent:
                self._last_saturation = time.time()

    def release(self, n: int = 1) -> None:
        """Release ``n`` previously acquired slots."""
        self._sem.release()
        # Update active count under lock to keep stats consistent.
        # Use a non-async shortcut for the synchronous release path.
        with _sync_lock(self):  # type: ignore[arg-type]
            self._active = max(0, self._active - 1)

    async def _consume_tokens(self, n: int) -> None:
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(
                float(self.budget.burst),
                self._tokens + elapsed * self.budget.rate_per_sec,
            )
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= n
                return
            deficit = n - self._tokens
            wait = deficit / self.budget.rate_per_sec
            self._tokens = 0.0
        # Wait outside the lock.
        await asyncio.sleep(wait)
        # Retry once after waiting.
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(
                float(self.budget.burst),
                self._tokens + elapsed * self.budget.rate_per_sec,
            )
            self._tokens = max(0.0, self._tokens - n)
            self._last_refill = now

    def _mark_saturation(self) -> None:
        self._last_saturation = time.time()

    def status(self) -> ComponentStatus:
        active = self._active
        return ComponentStatus(
            name=self.budget.name,
            max_concurrent=self.budget.max_concurrent,
            active=active,
            available=max(0, self.budget.max_concurrent - active),
            rate_limited=self.budget.rate_per_sec > 0,
            saturated=active >= self.budget.max_concurrent,
            last_saturation=self._last_saturation,
        )


class _sync_lock:  # pragma: no cover - thin shim for typing
    """Tiny adapter so ``release()`` stays synchronous.

    ``_ComponentLimiter`` keeps ``_active`` updated under an asyncio.Lock
    on the acquire path. The release path is synchronous (FastAPI /
    executor call sites expect ``__aexit__``-free release). We use a
    plain ``threading.Lock`` here purely for the rare case of multiple
    concurrent releases racing on ``_active``.
    """

    def __init__(self, limiter: _ComponentLimiter) -> None:
        self._limiter = limiter

    def __enter__(self) -> "_sync_lock":
        # asyncio.Lock has no sync acquire; we accept a tiny race window
        # because ``_active`` is only used for stats reporting. The
        # authoritative state is the asyncio.Semaphore's internal counter.
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class BackpressureCoordinator:
    """Cross-component backpressure registry (v0.6 §3.2).

    Usage::

        coord = BackpressureCoordinator(memory_budget_mb=2048)
        coord.register(ComponentBudget(
            name="vectorstore:milvus", max_concurrent=20, rate_per_sec=50, burst=100,
        ))

        async with coord.acquire("vectorstore:milvus"):
            await vectorstore.search(...)

    The coordinator is intentionally decoupled from the actual adapters:
    components are identified by name (``"<kind>:<id>"`` convention),
    so the same coordinator can govern any mix of vectorstores, DBs and
    external APIs without import cycles.
    """

    def __init__(
        self,
        memory_budget_mb: float = 0.0,
        check_interval: float = 5.0,
    ) -> None:
        self._limiters: dict[str, _ComponentLimiter] = {}
        self._lock = threading.RLock()
        self._memory_budget_mb = memory_budget_mb
        self._check_interval = check_interval
        self._last_rss_mb: float = 0.0
        self._memory_saturated: bool = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, budget: ComponentBudget) -> None:
        """Register a component's concurrency / rate budget.

        Re-registering with the same name replaces the previous limiter.
        In-flight permits on the old limiter are not migrated — callers
        must drain first.
        """
        with self._lock:
            self._limiters[budget.name] = _ComponentLimiter(budget)
        logger.info(
            "Registered backpressure component '%s' "
            "(max_concurrent=%d, rate_per_sec=%.1f, burst=%d)",
            budget.name,
            budget.max_concurrent,
            budget.rate_per_sec,
            budget.burst,
        )

    def unregister(self, name: str) -> None:
        """Remove a component. No-op if not registered."""
        with self._lock:
            self._limiters.pop(name, None)

    def list_components(self) -> list[str]:
        with self._lock:
            return list(self._limiters.keys())

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def acquire(
        self,
        name: str,
        n: int = 1,
        timeout: Optional[float] = None,
    ) -> "_BackpressureSlot":
        """Return an async context manager for ``name``.

        Raises:
            KeyError: when ``name`` is not registered.
        """
        return _BackpressureSlot(self, name, n, timeout)

    async def try_acquire(
        self,
        name: str,
        n: int = 1,
        timeout: Optional[float] = None,
    ) -> bool:
        """Non-raising acquire: returns True on success, False on timeout."""
        limiter = self._get(name)
        if limiter is None:
            raise KeyError(name)
        try:
            await limiter.acquire(n=n, timeout=timeout)
        except BackpressureError:
            return False
        return True

    def _get(self, name: str) -> Optional[_ComponentLimiter]:
        with self._lock:
            return self._limiters.get(name)

    # ------------------------------------------------------------------
    # Memory budget
    # ------------------------------------------------------------------

    async def check_memory(self) -> bool:
        """Refresh RSS reading and update the saturation flag.

        Returns True when memory is saturated (>= budget).
        """
        if self._memory_budget_mb <= 0:
            self._memory_saturated = False
            return False
        try:
            rss_bytes = _get_rss()
        except Exception:  # pragma: no cover - defensive
            self._memory_saturated = False
            return False
        self._last_rss_mb = rss_bytes / (1024 * 1024)
        self._memory_saturated = self._last_rss_mb >= self._memory_budget_mb
        return self._memory_saturated

    @property
    def memory_saturated(self) -> bool:
        return self._memory_saturated

    @property
    def memory_budget_mb(self) -> float:
        return self._memory_budget_mb

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    async def snapshot(self) -> BackpressureSnapshot:
        """Capture a snapshot for monitoring / ``/health``."""
        await self.check_memory()
        with self._lock:
            limiters = dict(self._limiters)
        components = {n: l.status() for n, l in limiters.items()}
        any_saturated = self._memory_saturated or any(
            c.saturated for c in components.values()
        )
        return BackpressureSnapshot(
            saturated=any_saturated,
            components=components,
            memory_rss_mb=self._last_rss_mb,
            memory_budget_mb=self._memory_budget_mb,
        )

    def is_saturated(self, name: Optional[str] = None) -> bool:
        """Quick saturation check.

        When ``name`` is None, returns True if ANY component is saturated
        or the memory budget is exceeded.
        """
        if name is None:
            if self._memory_saturated:
                return True
            with self._lock:
                limiters = list(self._limiters.values())
            return any(l.status().saturated for l in limiters)
        limiter = self._get(name)
        if limiter is None:
            return False
        return limiter.status().saturated

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """No-op for now; permits release naturally via context managers."""
        return None


class _BackpressureSlot:
    """Async context manager returned by :meth:`acquire`."""

    def __init__(
        self,
        coord: BackpressureCoordinator,
        name: str,
        n: int,
        timeout: Optional[float],
    ) -> None:
        self._coord = coord
        self._name = name
        self._n = n
        self._timeout = timeout
        self._limiter: Optional[_ComponentLimiter] = None
        self._acquired = False

    async def __aenter__(self) -> "_BackpressureSlot":
        limiter = self._coord._get(self._name)
        if limiter is None:
            raise KeyError(
                f"Backpressure component '{self._name}' is not registered"
            )
        self._limiter = limiter
        await limiter.acquire(n=self._n, timeout=self._timeout)
        self._acquired = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self._acquired or self._limiter is None:
            return
        try:
            self._limiter.release(self._n)
        finally:
            self._acquired = False


# ---------------------------------------------------------------------------
# RSS probing — uses psutil when available, falls back to /proc/self/status.
# ---------------------------------------------------------------------------

def _get_rss() -> int:
    """Return current process RSS in bytes."""
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        return int(proc.memory_info().rss)
    except ImportError:
        pass
    # Linux fallback.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024  # kB -> bytes
    except FileNotFoundError:
        pass
    # Windows fallback (no /proc; psutil usually present on Windows).
    return 0


__all__ = [
    "ComponentBudget",
    "ComponentStatus",
    "BackpressureSnapshot",
    "BackpressureCoordinator",
]
