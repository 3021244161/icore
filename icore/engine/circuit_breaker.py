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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

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

    Attributes:
        name:                       Identifier (usually model_id).
        failure_threshold:          Consecutive failures in CLOSED that
                                    trip the breaker.
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
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

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

    async def _on_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_failures += 1
                self.state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' -> OPEN (half-open probe failed)",
                    self.name,
                )
            elif (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.failure_threshold
            ):
                self.state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' -> OPEN (%d consecutive failures)",
                    self.name,
                    self.failure_count,
                )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "half_open_successes": self.half_open_successes,
            "half_open_failures": self.half_open_failures,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


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
    ) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_requests = half_open_max_requests
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
