"""
icore.security.rate_limiter - Generic keyed rate limiting primitives.

Provides a **keyed fixed-window quota** primitive (ICORE-ISSUE-005):

    - ``RateLimitDecision``:  Immutable check result
                               (``allowed`` / ``limit`` / ``used`` / ``remaining`` /
                               ``reset_at`` / ``retry_after``).
    - ``BaseRateLimiter``:    Abstract interface: ``check`` / ``peek`` / ``reset``.
    - ``MemoryRateLimiter``:  Single-process backend (development / testing).
    - ``RedisRateLimiter``:   Cross-instance backend; ``INCR`` + ``PEXPIRE``
                               made atomic by a single Lua script.
    - ``create_rate_limiter``: Factory (``backend="memory" | "redis"``).

Semantic boundary (deliberate — ICORE-ISSUE-005 §4/§5.2):

    - "How to count" is a platform capability (this module); "what the
      numbers mean and what to do on violation" is the caller's business.
    - ``check()`` is **non-blocking**: a denied decision returns
      immediately. Callers wanting queue semantics can loop on
      ``retry_after`` themselves.
    - Fixed window counted **from the first hit** (INCR + TTL semantics);
      the window length is supplied per call. Timezone-aligned "natural
      day" resets are the caller's concern.
    - Denied hits **still count** (prevents free probing under rejection
      storms); ``used`` may therefore exceed ``limit``.
    - Redis failures **propagate** — the degradation posture (fail-open
      with a MemoryRateLimiter vs fail-closed) is a business decision and
      belongs to the caller.

This primitive intentionally **coexists with** (and does not replace)
``engine.concurrency_control.TokenBucket`` (per-``model_id`` smooth-rate
blocking acquire) and ``BackpressureCoordinator`` (in-flight concurrency
budget). Different semantic families for different jobs.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from icore.observability import get_metrics_registry

__all__ = [
    "RateLimitDecision",
    "BaseRateLimiter",
    "MemoryRateLimiter",
    "RedisRateLimiter",
    "create_rate_limiter",
]

logger = logging.getLogger(__name__)

#: Prometheus counter registered for rate limit decisions. Labels are
#: ``(backend, outcome)`` only — keyed limits are deliberately NOT a label
#: (user ids / IPs would explode the series cardinality).
_METRIC_CHECKS = "icore_rate_limit_checks_total"


def _validate_limit(limit: int) -> None:
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit!r}")


def _validate_window(window: float) -> None:
    if window <= 0:
        raise ValueError(f"window must be > 0 (seconds), got {window!r}")


@dataclass(frozen=True)
class RateLimitDecision:
    """Outcome of a rate limit ``check`` / ``peek``.

    Attributes:
        allowed:     Whether the hit is within quota.
        limit:       The quota supplied by the caller.
        used:        Units consumed in the current window (**including
                     the hit being judged** — denied hits still count).
        remaining:   Remaining quota (``>= 0``, clamped).
        reset_at:    Epoch seconds when the window resets
                     (``0.0`` when no window is active).
        retry_after: Seconds until reset — set only when
                     ``allowed`` is ``False``.
    """

    allowed: bool
    limit: int
    used: int
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None


def _make_decision(
    count: int, limit: int, reset_at: float, now: float
) -> RateLimitDecision:
    """Build a decision from a (possibly over-limit) count."""
    allowed = count <= limit
    retry_after: Optional[float] = None
    if not allowed:
        retry_after = max(0.0, reset_at - now)
    return RateLimitDecision(
        allowed=allowed,
        limit=limit,
        used=count,
        remaining=max(0, limit - count),
        reset_at=reset_at,
        retry_after=retry_after,
    )


def _record_check(backend: str, decision: RateLimitDecision) -> None:
    """Increment the rate limit checks counter (never breaks limiting)."""
    try:
        registry = get_metrics_registry()
        counter = registry.create_counter(
            _METRIC_CHECKS,
            "Total rate limit checks (keyed fixed-window primitive)",
            ("backend", "outcome"),
        )
        counter.inc(
            backend=backend,
            outcome="allowed" if decision.allowed else "denied",
        )
    except Exception:  # noqa: BLE001 — observability must not break limiting
        logger.warning("Failed to record rate limit metric", exc_info=True)


class BaseRateLimiter(ABC):
    """Abstract keyed fixed-window rate limiter (ICORE-ISSUE-005 §5.1)."""

    @abstractmethod
    async def check(
        self, key: str, *, limit: int, window: float
    ) -> RateLimitDecision:
        """Consume one unit for ``key`` and return the decision.

        **Non-blocking**: a denied decision returns immediately — the
        limiter never waits for quota to free up. Key and window are
        supplied by the caller; the primitive carries no business
        numbers and no timezone.
        """

    @abstractmethod
    async def peek(self, key: str, *, limit: int) -> RateLimitDecision:
        """Read-only quota query for ``key`` (does not consume)."""

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Clear the counter for ``key`` (ops / testing)."""


class MemoryRateLimiter(BaseRateLimiter):
    """Single-process fixed-window limiter.

    Counters live in process memory. The dict mutations in ``check`` /
    ``peek`` contain no ``await`` points, so they are atomic within a
    single event loop. Use :class:`RedisRateLimiter` when counts must be
    shared across instances.

    Args:
        cleanup_interval: Minimum seconds between opportunistic sweeps
            that drop expired windows (default 60; ``0`` disables — the
            caller then manages ``cleanup()`` explicitly). Expired
            windows are always lazily replaced on the next ``check``.
    """

    def __init__(self, *, cleanup_interval: float = 60.0) -> None:
        # key -> (count, reset_at epoch seconds)
        self._windows: dict[str, tuple[int, float]] = {}
        self._cleanup_interval = max(0.0, cleanup_interval)
        self._last_cleanup = time.time()

    async def check(
        self, key: str, *, limit: int, window: float
    ) -> RateLimitDecision:
        _validate_limit(limit)
        _validate_window(window)
        now = time.time()
        self._maybe_cleanup(now)
        entry = self._windows.get(key)
        if entry is None or entry[1] <= now:
            # First hit of a new window: count starts at 1.
            count, reset_at = 1, now + window
        else:
            count, reset_at = entry[0] + 1, entry[1]
        self._windows[key] = (count, reset_at)
        decision = _make_decision(count, limit, reset_at, now)
        _record_check("memory", decision)
        return decision

    async def peek(self, key: str, *, limit: int) -> RateLimitDecision:
        _validate_limit(limit)
        now = time.time()
        entry = self._windows.get(key)
        if entry is None or entry[1] <= now:
            # No active window: nothing used, nothing to reset.
            return RateLimitDecision(
                allowed=True,
                limit=limit,
                used=0,
                remaining=limit,
                reset_at=0.0,
                retry_after=None,
            )
        return _make_decision(entry[0], limit, entry[1], now)

    async def reset(self, key: str) -> None:
        self._windows.pop(key, None)

    def cleanup(self) -> int:
        """Drop expired windows; return the number of keys freed."""
        return self._cleanup(time.time())

    # -- internal ----------------------------------------------------------

    def _maybe_cleanup(self, now: float) -> None:
        if (
            self._cleanup_interval <= 0
            or (now - self._last_cleanup) < self._cleanup_interval
        ):
            return
        self._last_cleanup = now
        self._cleanup(now)

    def _cleanup(self, now: float) -> int:
        expired = [
            k for k, (_, reset_at) in self._windows.items() if reset_at <= now
        ]
        for k in expired:
            self._windows.pop(k, None)
        return len(expired)


class RedisRateLimiter(BaseRateLimiter):
    """Redis-backed fixed-window limiter shared across instances.

    ``check`` runs a single Lua script so that ``INCR`` and the first
    ``PEXPIRE`` are **atomic** — the "counter without TTL leaks Redis
    memory" race (``INCR`` then ``EXPIRE`` as separate commands) cannot
    occur. A key found without TTL (e.g. written by an older tool) is
    repaired by the same script.

    The Redis client must be supplied by the caller (mirrors the
    ``RedisLock`` injection pattern; typically built from
    ``settings.concurrency.redis_url``)::

        import redis.asyncio as aioredis
        client = aioredis.from_url(redis_url)
        limiter = RedisRateLimiter(client)

    **Failure posture**: Redis errors propagate to the caller. Whether
    to fail-open (fall back to a MemoryRateLimiter) or fail-closed is a
    business decision (ICORE-ISSUE-005 §5.2.6) and is not hidden here.
    """

    #: Atomic INCR + PEXPIRE (first hit or TTL-less key) + read-back.
    _CHECK_SCRIPT = """
        local count = redis.call("INCR", KEYS[1])
        local ttl = redis.call("PTTL", KEYS[1])
        if count == 1 or ttl < 0 then
            redis.call("PEXPIRE", KEYS[1], ARGV[1])
            ttl = tonumber(ARGV[1])
        end
        return {count, ttl}
    """

    #: Read-only GET + PTTL (never increments).
    _PEEK_SCRIPT = """
        local count = tonumber(redis.call("GET", KEYS[1]) or "0")
        local ttl = redis.call("PTTL", KEYS[1])
        return {count, ttl}
    """

    def __init__(self, client: Any, *, prefix: str = "icore:rate_limit:") -> None:
        self._client = client
        self._prefix = prefix

    async def check(
        self, key: str, *, limit: int, window: float
    ) -> RateLimitDecision:
        _validate_limit(limit)
        _validate_window(window)
        now = time.time()
        ttl_ms = int(window * 1000)
        count, ttl = await self._client.eval(
            self._CHECK_SCRIPT, 1, self._k(key), ttl_ms
        )
        reset_at = now + float(ttl) / 1000.0
        decision = _make_decision(int(count), limit, reset_at, now)
        _record_check("redis", decision)
        return decision

    async def peek(self, key: str, *, limit: int) -> RateLimitDecision:
        _validate_limit(limit)
        now = time.time()
        count, ttl = await self._client.eval(self._PEEK_SCRIPT, 1, self._k(key))
        count = int(count)
        if count == 0 or int(ttl) <= 0:
            # Key absent (GET nil / PTTL -2) or TTL-less (PTTL -1):
            # treat as "no active window".
            return RateLimitDecision(
                allowed=True,
                limit=limit,
                used=0,
                remaining=limit,
                reset_at=0.0,
                retry_after=None,
            )
        reset_at = now + float(ttl) / 1000.0
        return _make_decision(count, limit, reset_at, now)

    async def reset(self, key: str) -> None:
        await self._client.delete(self._k(key))

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"


def create_rate_limiter(
    backend: str = "memory",
    *,
    redis_url: Optional[str] = None,
    prefix: str = "icore:rate_limit:",
    cleanup_interval: float = 60.0,
) -> BaseRateLimiter:
    """Build a rate limiter (ICORE-ISSUE-005 §5.1 factory).

    Args:
        backend:          ``"memory"`` (single process) or ``"redis"``
                          (cross-instance shared counts).
        redis_url:        Required for the redis backend when
                          ``ICORE_CONCURRENCY_REDIS_URL`` is unset.
        prefix:           Redis key namespace (redis backend only).
        cleanup_interval: Memory backend expired-window sweep interval.

    The redis driver is imported lazily, so this module (and
    ``icore.security`` as a whole) stays importable without ``redis``
    installed.
    """
    if backend == "memory":
        return MemoryRateLimiter(cleanup_interval=cleanup_interval)
    if backend == "redis":
        from icore.config import get_settings

        url = redis_url or get_settings().concurrency.redis_url
        if not url:
            raise ValueError(
                "redis backend requires redis_url (or "
                "ICORE_CONCURRENCY_REDIS_URL) to be set"
            )
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "redis backend requires the 'redis' package: pip install redis"
            ) from exc
        return RedisRateLimiter(aioredis.from_url(url), prefix=prefix)
    raise ValueError(
        f"Unknown rate limiter backend: {backend!r} (use 'memory' or 'redis')"
    )
