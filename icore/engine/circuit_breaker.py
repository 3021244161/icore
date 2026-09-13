"""
icore.engine.circuit_breaker - Per-model circuit breaker.

A circuit breaker protects downstream services (LLM APIs, vector DBs,
etc.) from cascading failures by failing fast when a target is
clearly unhealthy. Each ``CircuitBreaker`` instance guards one
resource (typically one ``model_id``).

State machine::

    CLOSED    --(failure_threshold reached)--> OPEN
    OPEN      --(recovery_timeout elapsed)----> HALF_OPEN
    HALF_OPEN --(half_open_max_requests successes)--> CLOSED
    HALF_OPEN --(any failure)----------------------> OPEN

The breaker is async-safe (single asyncio.Lock) and exposes a
``call(coro_factory)`` helper that wraps a coroutine with state
transitions.

Usage::

    cb = CircuitBreaker(name="gpt-4o", failure_threshold=5)
    try:
        result = await cb.call(lambda: adapter.chat(messages))
    except CircuitBreakerOpenError:
        # fail fast / fall back
        ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from icore.exceptions import CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Breaker state."""

    CLOSED = "closed"        # Normal operation; calls pass through.
    OPEN = "open"            # Tripped; calls fail fast.
    HALF_OPEN = "half_open"  # Recovery probe; limited calls allowed.


@dataclass
class CircuitBreaker:
    """
    Per-resource circuit breaker.

    Two CLOSED-state trip criteria are supported (ICORE-ISSUE-006):

        - **Consecutive-failure mode** (default): trips when
          ``failure_count >= failure_threshold``. Legacy behavior,
          unchanged and zero overhead (no samples recorded) when
          ``failure_rate_threshold`` is ``None``.
        - **Windowed-failure-rate mode** (optional): when
          ``failure_rate_threshold`` is set, the CLOSED-state trip
          criterion switches to "failure rate within the sliding
          ``failure_rate_window`` **reaches or exceeds** the threshold,
          with at least ``min_samples`` observations in the window".
          ``min_samples`` guards against small-sample false trips;
          the windowed rate is far more robust to traffic bursts than
          consecutive counting (a handful of failures during a
          low-traffic period no longer trips the breaker).

    HALF_OPEN semantics (probe successes close, any probe failure
    re-opens) are identical in both modes.

    Attributes:
        name:                       Identifier (usually model_id).
        failure_threshold:          Consecutive failures in CLOSED that
                                    trip the breaker (consecutive mode
                                    only; ignored in rate mode).
        failure_rate_threshold:    Optional failure-rate threshold in
                                    ``(0, 1]`` (e.g. ``0.8``). ``None``
                                    keeps the consecutive-failure
                                    semantics (default).
        failure_rate_window:       Sliding window length in seconds for
                                    rate mode (default 30).
        min_samples:                Minimum observations required in the
                                    window before the rate can trip the
                                    breaker (default 10).
        recovery_timeout:           Seconds OPEN waits before HALF_OPEN.
        half_open_max_requests:     Successful probes in HALF_OPEN that
                                    transition back to CLOSED.
        state:                      Current state.
        failure_count:              Consecutive failure counter (CLOSED).
        last_failure_time:          Unix timestamp of last failure.
        half_open_successes:        Successful probes while HALF_OPEN.
        half_open_failures:         Failed probes while HALF_OPEN.
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_requests: int = 3
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_successes: int = 0
    half_open_failures: int = 0
    # ICORE-ISSUE-006: 可选「时间窗失败率」触发口径。None = 既有
    # 连续计数语义（零开销，不记录样本）；设置后 CLOSED 态触发判定
    # 切换为窗口失败率 >= 阈值（样本数达到 min_samples 才参与判定）。
    # 刻意放在既有字段之后：位置参数顺序对既有调用方完全不变。
    failure_rate_threshold: Optional[float] = None
    failure_rate_window: float = 30.0
    min_samples: int = 10
    _samples: deque = field(default_factory=deque, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.failure_rate_threshold is not None and not (
            0 < self.failure_rate_threshold <= 1
        ):
            raise ValueError(
                "failure_rate_threshold must be in (0, 1] or None, "
                f"got {self.failure_rate_threshold!r}"
            )
        if self.failure_rate_window <= 0:
            raise ValueError(
                "failure_rate_window must be > 0 (seconds), "
                f"got {self.failure_rate_window!r}"
            )
        if self.min_samples < 1:
            raise ValueError(
                f"min_samples must be >= 1, got {self.min_samples!r}"
            )

    async def call(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Execute ``coro_factory()`` under breaker protection.

        Args:
            coro_factory: Zero-arg callable returning an awaitable.

        Returns:
            The awaitable's result.

        Raises:
            CircuitBreakerOpenError: Breaker is OPEN.
            Exception: Re-raises any exception from the awaitable
                       after recording the failure.
        """
        await self._pre_call()
        try:
            result = await coro_factory()
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _pre_call(self) -> None:
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if (
                    time.time() - self.last_failure_time
                    >= self.recovery_timeout
                ):
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_successes = 0
                    self.half_open_failures = 0
                    logger.info(
                        "Circuit '%s' -> HALF_OPEN (probing)", self.name
                    )
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker for '{self.name}' is OPEN"
                    )

    async def _on_success(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if (
                    self.half_open_successes
                    >= self.half_open_max_requests
                ):
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(
                        "Circuit '%s' -> CLOSED (recovered)", self.name
                    )
            elif self.state == CircuitState.CLOSED:
                # Reset on any success to require *consecutive* failures.
                self.failure_count = 0
                # ICORE-ISSUE-006: 成功样本稀释窗口失败率。样本凑满
                # min_samples 的瞬间可能恰好是成功调用，故此处同样
                # 判定（语义：窗口失败率达到阈值即熔断，无论末样本）。
                self._record_sample(False)
                if self._should_trip():
                    self.state = CircuitState.OPEN
                    logger.warning(
                        "Circuit '%s' -> OPEN (%s)",
                        self.name,
                        self._trip_reason(),
                    )

    async def _on_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self._record_sample(True)
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_failures += 1
                self.state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' -> OPEN (half-open probe failed)",
                    self.name,
                )
            elif (
                self.state == CircuitState.CLOSED
                and self._should_trip()
            ):
                self.state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' -> OPEN (%s)",
                    self.name,
                    self._trip_reason(),
                )

    # -- ICORE-ISSUE-006: windowed failure-rate bookkeeping --------------

    def _record_sample(self, is_failure: bool) -> None:
        """Append a ``(timestamp, outcome)`` sample (rate mode only)."""
        if self.failure_rate_threshold is None:
            return
        self._samples.append((time.time(), is_failure))

    def _prune_samples(self, now: float) -> None:
        """Drop samples older than the sliding window."""
        horizon = now - self.failure_rate_window
        while self._samples and self._samples[0][0] <= horizon:
            self._samples.popleft()

    def _should_trip(self) -> bool:
        """CLOSED-state trip criterion (consecutive OR windowed rate).

        Consecutive mode: ``failure_count >= failure_threshold``.
        Rate mode: windowed failure rate **>=** ``failure_rate_threshold``
        with at least ``min_samples`` observations (else never trips).
        """
        if self.failure_rate_threshold is None:
            return self.failure_count >= self.failure_threshold
        self._prune_samples(time.time())
        if len(self._samples) < self.min_samples:
            return False
        failures = sum(1 for _, failed in self._samples if failed)
        return failures / len(self._samples) >= self.failure_rate_threshold

    def _trip_reason(self) -> str:
        if self.failure_rate_threshold is None:
            return f"{self.failure_count} consecutive failures"
        failures = sum(1 for _, failed in self._samples if failed)
        return (
            f"window failure rate "
            f"{failures / max(len(self._samples), 1):.1%} "
            f"({failures}/{len(self._samples)} within "
            f"{self.failure_rate_window}s)"
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of breaker state."""
        snap: dict[str, Any] = {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "half_open_successes": self.half_open_successes,
            "half_open_failures": self.half_open_failures,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "failure_rate_threshold": self.failure_rate_threshold,
            "failure_rate_window": self.failure_rate_window,
            "min_samples": self.min_samples,
        }
        if self.failure_rate_threshold is not None:
            self._prune_samples(time.time())
            failures = sum(1 for _, failed in self._samples if failed)
            snap["window_samples"] = len(self._samples)
            snap["window_failures"] = failures
            snap["window_failure_rate"] = round(
                failures / max(len(self._samples), 1), 4
            )
        return snap


class CircuitBreakerRegistry:
    """
    Per-model registry of circuit breakers.

    Each model_id gets its own breaker (lazily created on first
    access) so that one unhealthy model does not trip breakers for
    unrelated models.

    Usage::

        registry = CircuitBreakerRegistry()
        cb = registry.get("gpt-4o")
        result = await cb.call(lambda: adapter.chat(messages))
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 3,
        failure_rate_threshold: Optional[float] = None,
        failure_rate_window: float = 30.0,
        min_samples: int = 10,
    ) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_requests = half_open_max_requests
        # ICORE-ISSUE-006: 透传可选「时间窗失败率」口径给新建 breaker。
        self._failure_rate_threshold = failure_rate_threshold
        self._failure_rate_window = failure_rate_window
        self._min_samples = min_samples
        self._lock = asyncio.Lock()

    async def get(self, name: str) -> CircuitBreaker:
        """Get or lazily create a breaker for ``name``."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=self._failure_threshold,
                    recovery_timeout=self._recovery_timeout,
                    half_open_max_requests=self._half_open_max_requests,
                    failure_rate_threshold=self._failure_rate_threshold,
                    failure_rate_window=self._failure_rate_window,
                    min_samples=self._min_samples,
                )
            return self._breakers[name]

    def snapshot(self) -> list[dict[str, Any]]:
        """List state of all registered breakers."""
        return [cb.snapshot() for cb in self._breakers.values()]

    async def reset(self, name: str | None = None) -> None:
        """Reset one or all breakers (force CLOSED)."""
        async with self._lock:
            if name is None:
                for cb in self._breakers.values():
                    cb.state = CircuitState.CLOSED
                    cb.failure_count = 0
                    cb.half_open_successes = 0
                    cb.half_open_failures = 0
            elif name in self._breakers:
                cb = self._breakers[name]
                cb.state = CircuitState.CLOSED
                cb.failure_count = 0
                cb.half_open_successes = 0
                cb.half_open_failures = 0


__all__ = [
    "CircuitState",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
]
