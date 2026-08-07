"""
icore.engine.graceful_degradation - Degradation matrix coordinator (v0.6 §3.3.2).

When an infrastructure component fails, the system should *degrade* to a
workable fallback rather than crash. This module formalises the
degradation matrix from the design doc:

    =============  ============================================
    Component      Fallback
    =============  ============================================
    LLM API        fallback model → ``NoAvailableModelError``
    Milvus         ``InMemoryVectorStore`` (or BM25 keyword search)
    Neo4j          ``InMemoryGraphStore``
    Redis          ``MemoryLock`` + in-memory idempotency cache
    PostgreSQL     local JSONL audit log (no persistence guarantees)
    Object store   inline base64 in API response
    =============  ============================================

Design:

    * Each component has a *primary* provider and an optional *fallback*
      provider (already-built instances). The coordinator tracks the
      current mode (``"primary" | "degraded"``) per component.
    * ``mark_failed(name)`` triggers degradation — the primary is
      swapped out for the fallback. ``mark_recovered(name)`` restores
      the primary.
    * A background ``watch()`` task periodically health-checks the
      primary; on success it auto-recovers. This makes degradation
      self-healing.
    * ``DegradationSnapshot`` is consumed by ``/health`` so the API
      surfaces which components are degraded (v0.6 §3.3.3 验收:
      "降级时 /health 返回 status: degraded 并列出哪些组件降级了").

Module dependencies:
    engine only — no api / db / models imports. Callers pass already-
    built provider instances; the coordinator never imports adapters.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DegradationState(str, Enum):
    """Per-component degradation state."""

    PRIMARY = "primary"    # Using the primary provider.
    DEGRADED = "degraded"  # Using the fallback provider.
    UNAVAILABLE = "unavailable"  # No provider available (hard failure).


@dataclass
class ComponentState:
    """State of one component."""

    name: str
    state: DegradationState = DegradationState.PRIMARY
    failure_count: int = 0
    last_failure: Optional[float] = None
    last_recovery: Optional[float] = None
    last_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure,
            "last_recovery": self.last_recovery,
            "last_error": self.last_error,
        }


@dataclass
class DegradationSnapshot:
    """Full snapshot for ``/health``."""

    any_degraded: bool
    components: dict[str, ComponentState] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "any_degraded": self.any_degraded,
            "components": {
                n: c.to_dict() for n, c in self.components.items()
            },
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Provider binding
# ---------------------------------------------------------------------------

@dataclass
class _ProviderBinding:
    """Internal: primary + fallback for one component."""

    name: str
    primary: Any
    fallback: Optional[Any]
    health_check: Optional[Callable[[], Awaitable[bool]]]
    failure_threshold: int = 3  # failures before degradation
    recovery_interval: float = 30.0  # seconds between recovery probes


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class GracefulDegradationCoordinator:
    """Coordinates per-component graceful degradation (v0.6 §3.3.2).

    Usage::

        coord = GracefulDegradationCoordinator()
        coord.register(
            name="vectorstore",
            primary=milvus_adapter,
            fallback=in_memory_store,
            health_check=lambda: milvus_adapter.health_check(),
        )

        # The active provider for any component:
        provider = coord.get("vectorstore")

        # Auto-recovery: a background task polls the primary and
        # restores it when healthy.
        await coord.start_watch()

    Failure tracking:

        Callers wrap risky operations in ``with coord.guard("vectorstore")``
        — on exception the failure counter is bumped and the component
        degrades after ``failure_threshold`` consecutive failures.

        Callers can also explicitly ``mark_failed(name, error)`` /
        ``mark_recovered(name)`` if they have richer signal (e.g. the
        circuit breaker's OPEN state).
    """

    def __init__(self) -> None:
        self._bindings: dict[str, _ProviderBinding] = {}
        self._states: dict[str, ComponentState] = {}
        self._lock = threading.RLock()
        self._watch_task: Optional[asyncio.Task[None]] = None
        self._stop_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        primary: Any,
        fallback: Optional[Any] = None,
        health_check: Optional[Callable[[], Awaitable[bool]]] = None,
        *,
        failure_threshold: int = 3,
        recovery_interval: float = 30.0,
    ) -> None:
        """Register a component's primary + fallback providers.

        Args:
            name:               Component identifier.
            primary:            Primary provider instance.
            fallback:           Optional fallback. If None, the
                                component becomes UNAVAILABLE on
                                primary failure rather than degraded.
            health_check:       Optional async callable returning True
                                when the primary is healthy. Used by
                                the auto-recovery watcher.
            failure_threshold:  Consecutive failures before degradation.
            recovery_interval:  Seconds between recovery probes.
        """
        with self._lock:
            self._bindings[name] = _ProviderBinding(
                name=name,
                primary=primary,
                fallback=fallback,
                health_check=health_check,
                failure_threshold=failure_threshold,
                recovery_interval=recovery_interval,
            )
            self._states[name] = ComponentState(name=name)
        logger.info(
            "Registered degradation component '%s' "
            "(threshold=%d, recovery_interval=%.1fs, has_fallback=%s)",
            name,
            failure_threshold,
            recovery_interval,
            fallback is not None,
        )

    def unregister(self, name: str) -> None:
        with self._lock:
            self._bindings.pop(name, None)
            self._states.pop(name, None)

    def list_components(self) -> list[str]:
        with self._lock:
            return list(self._bindings.keys())

    # ------------------------------------------------------------------
    # Provider access
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Return the currently-active provider for ``name``.

        Raises:
            KeyError: when ``name`` is not registered.
            RuntimeError: when the component is UNAVAILABLE (no
                fallback) and the primary has failed.
        """
        with self._lock:
            binding = self._bindings.get(name)
            state = self._states.get(name)
        if binding is None:
            raise KeyError(name)
        if state is None:  # pragma: no cover - defensive
            return binding.primary
        if state.state == DegradationState.PRIMARY:
            return binding.primary
        if state.state == DegradationState.DEGRADED:
            if binding.fallback is None:
                # Should not happen — UNAVAILABLE takes precedence —
                # but guard anyway.
                raise RuntimeError(
                    f"Component '{name}' is degraded but has no fallback"
                )
            return binding.fallback
        # UNAVAILABLE
        raise RuntimeError(
            f"Component '{name}' is unavailable (primary failed, no fallback)"
        )

    def get_state(self, name: str) -> ComponentState:
        """Return the current state record for ``name``."""
        with self._lock:
            state = self._states.get(name)
        if state is None:
            raise KeyError(name)
        return state

    # ------------------------------------------------------------------
    # Failure / recovery signalling
    # ------------------------------------------------------------------

    def mark_failed(self, name: str, error: Optional[str] = None) -> None:
        """Record a failure for ``name``; degrade when threshold reached."""
        with self._lock:
            binding = self._bindings.get(name)
            state = self._states.get(name)
            if binding is None or state is None:
                return
            state.failure_count += 1
            state.last_failure = time.time()
            if error:
                state.last_error = str(error)[:200]
            if state.failure_count >= binding.failure_threshold:
                if binding.fallback is not None:
                    state.state = DegradationState.DEGRADED
                    logger.warning(
                        "Component '%s' degraded to fallback after %d "
                        "failures (last_error=%s)",
                        name,
                        state.failure_count,
                        state.last_error,
                    )
                else:
                    state.state = DegradationState.UNAVAILABLE
                    logger.error(
                        "Component '%s' unavailable after %d failures "
                        "(no fallback configured; last_error=%s)",
                        name,
                        state.failure_count,
                        state.last_error,
                    )

    def mark_recovered(self, name: str) -> None:
        """Mark ``name`` as recovered; restore primary provider."""
        with self._lock:
            state = self._states.get(name)
            if state is None:
                return
            state.failure_count = 0
            state.last_error = None
            state.last_recovery = time.time()
            if state.state != DegradationState.PRIMARY:
                logger.info(
                    "Component '%s' recovered to primary", name
                )
            state.state = DegradationState.PRIMARY

    def reset(self, name: str) -> None:
        """Reset state without changing provider (testing helper)."""
        with self._lock:
            state = self._states.get(name)
            if state is None:
                return
            state.failure_count = 0
            state.last_error = None
            state.state = DegradationState.PRIMARY

    # ------------------------------------------------------------------
    # Context manager guard
    # ------------------------------------------------------------------

    def guard(self, name: str) -> "_DegradationGuard":
        """Return a context manager that auto-marks failures.

        Usage::

            with coord.guard("vectorstore"):
                results = await vectorstore.search(...)
        """
        return _DegradationGuard(self, name)

    # ------------------------------------------------------------------
    # Auto-recovery watcher
    # ------------------------------------------------------------------

    async def start_watch(self) -> None:
        """Start the background recovery watcher."""
        if self._watch_task is not None:
            return
        self._stop_event = asyncio.Event()
        self._watch_task = asyncio.create_task(
            self._watch_loop(), name="icore-degradation-watch"
        )
        logger.info("Degradation watcher started")

    async def stop_watch(self) -> None:
        """Stop the background recovery watcher."""
        if self._watch_task is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        self._watch_task.cancel()
        try:
            await self._watch_task
        except asyncio.CancelledError:
            pass
        self._watch_task = None
        self._stop_event = None
        logger.info("Degradation watcher stopped")

    async def _watch_loop(self) -> None:
        """Background loop: probe primary health for degraded components."""
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    bindings = list(self._bindings.values())
                    states = {n: s for n, s in self._states.items()}
                for binding in bindings:
                    if binding.health_check is None:
                        continue
                    state = states.get(binding.name)
                    if state is None:
                        continue
                    if state.state != DegradationState.DEGRADED:
                        continue
                    try:
                        ok = await binding.health_check()
                    except Exception as e:
                        logger.debug(
                            "Recovery probe for '%s' failed: %s",
                            binding.name,
                            e,
                        )
                        continue
                    if ok:
                        self.mark_recovered(binding.name)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Degradation watcher iteration failed")
            # Poll at the shortest recovery interval across components.
            with self._lock:
                intervals = [
                    b.recovery_interval for b in self._bindings.values()
                ]
            wait = min(intervals, default=30.0)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait)
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> DegradationSnapshot:
        """Capture a snapshot for monitoring / ``/health``."""
        with self._lock:
            states = {n: _clone_state(s) for n, s in self._states.items()}
        any_degraded = any(
            s.state != DegradationState.PRIMARY for s in states.values()
        )
        return DegradationSnapshot(
            any_degraded=any_degraded,
            components=states,
        )

    def is_degraded(self, name: Optional[str] = None) -> bool:
        """Quick degradation check.

        When ``name`` is None, returns True if ANY component is degraded.
        """
        with self._lock:
            if name is not None:
                state = self._states.get(name)
                return state is not None and state.state != DegradationState.PRIMARY
            return any(
                s.state != DegradationState.PRIMARY
                for s in self._states.values()
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self.stop_watch()


def _clone_state(state: ComponentState) -> ComponentState:
    """Deep-copy a ComponentState for snapshot consistency."""
    return ComponentState(
        name=state.name,
        state=state.state,
        failure_count=state.failure_count,
        last_failure=state.last_failure,
        last_recovery=state.last_recovery,
        last_error=state.last_error,
    )


class _DegradationGuard:
    """Sync context manager that records failures for ``name``."""

    def __init__(
        self, coord: GracefulDegradationCoordinator, name: str
    ) -> None:
        self._coord = coord
        self._name = name

    def __enter__(self) -> "_DegradationGuard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self._coord.mark_failed(self._name, str(exc))
        return None


__all__ = [
    "DegradationState",
    "ComponentState",
    "DegradationSnapshot",
    "GracefulDegradationCoordinator",
]
