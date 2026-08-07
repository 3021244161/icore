"""
icore.engine.backoff - Configurable backoff strategies for retries.

Provides four backoff strategies that can be plugged into the existing
``_retry_async`` helper used by the executor and the circuit-breaker
call path:

    - CONSTANT:           ``base_delay`` every attempt
    - LINEAR:             ``base_delay * attempt``
    - EXPONENTIAL:        ``base_delay * 2 ** (attempt - 1)``
    - EXPONENTIAL_JITTER: EXPONENTIAL + random jitter (+/- 25%)

All strategies cap the delay at ``max_delay`` so a misconfigured
``base_delay`` cannot stall the worker for minutes.

Usage::

    from icore.engine.backoff import BackoffStrategy, compute_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return await call()
        except Exception:
            if attempt == max_attempts:
                raise
            delay = compute_delay(
                BackoffStrategy.EXPONENTIAL_JITTER,
                base_delay=0.5,
                attempt=attempt,
                max_delay=30.0,
            )
            await asyncio.sleep(delay)
"""

from __future__ import annotations

import asyncio
import logging
import random
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackoffStrategy(str, Enum):
    """Backoff strategy selector.

    The string values are stable identifiers used in YAML config
    (``retry.backoff: exp_jitter``) and in Prometheus labels.
    """

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exp_jitter"


def compute_delay(
    strategy: BackoffStrategy,
    base_delay: float,
    attempt: int,
    max_delay: float = 60.0,
) -> float:
    """
    Compute the sleep delay for the given attempt.

    Args:
        strategy:   Backoff strategy enum value.
        base_delay: Base delay in seconds (attempt 1 = base_delay for
                    CONSTANT / EXPONENTIAL, base_delay * 1 for LINEAR).
        attempt:    1-based attempt index (1 = first retry).
        max_delay:  Upper bound on the returned delay.

    Returns:
        Delay in seconds (float). Always <= ``max_delay``.
    """
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")
    if attempt < 1:
        raise ValueError("attempt must be >= 1")

    if strategy == BackoffStrategy.CONSTANT:
        delay = base_delay
    elif strategy == BackoffStrategy.LINEAR:
        delay = base_delay * attempt
    elif strategy == BackoffStrategy.EXPONENTIAL:
        delay = base_delay * (2 ** (attempt - 1))
    elif strategy == BackoffStrategy.EXPONENTIAL_JITTER:
        # Full exponential then +/- 25% jitter.
        exp = base_delay * (2 ** (attempt - 1))
        jitter = exp * 0.25
        delay = exp + random.uniform(-jitter, jitter)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown backoff strategy: {strategy}")

    # Never negative (jitter can produce negative values when exp == 0).
    return max(0.0, min(delay, max_delay))


async def retry_with_backoff(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_on: Optional[tuple[type[BaseException], ...]] = None,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """
    Retry an async callable with the given backoff strategy.

    Args:
        coro_factory: Zero-arg callable returning a fresh awaitable.
                      Called once per attempt (so retries get a new
                      coroutine, not a re-await of the same one).
        max_attempts: Total attempts including the first. ``1`` = no retries.
        strategy:     Backoff strategy.
        base_delay:   Base delay in seconds.
        max_delay:    Cap on per-attempt delay.
        retry_on:     Optional tuple of exception types that should trigger
                      a retry. ``None`` (default) retries on any Exception.
        on_retry:     Optional callback invoked before each retry sleep
                      with ``(attempt, exception)``. Useful for logging
                      or metrics.

    Returns:
        The result of ``coro_factory()`` on the first successful attempt.

    Raises:
        The last exception raised by ``coro_factory()`` after all retries
        are exhausted (or immediately if ``retry_on`` excludes it).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            # If caller specified a whitelist and this exc is not in it, raise.
            if retry_on is not None and not isinstance(exc, retry_on):
                raise
            # No more retries left.
            if attempt >= max_attempts:
                raise
            if on_retry is not None:
                try:
                    on_retry(attempt, exc)
                except Exception:  # pragma: no cover - callback must not break retry
                    logger.warning("on_retry callback raised", exc_info=True)
            delay = compute_delay(
                strategy, base_delay, attempt, max_delay
            )
            logger.debug(
                "retry attempt %d/%d after %.3fs (strategy=%s, exc=%s)",
                attempt,
                max_attempts,
                delay,
                strategy.value,
                type(exc).__name__,
            )
            await asyncio.sleep(delay)
    # Should be unreachable, but satisfy type checker.
    assert last_exc is not None
    raise last_exc


__all__ = [
    "BackoffStrategy",
    "compute_delay",
    "retry_with_backoff",
]
