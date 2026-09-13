"""
Tests for icore.engine.circuit_breaker.

Covers:
    - CircuitBreaker state transitions:
        CLOSED -> OPEN (after failure_threshold)
        OPEN   -> HALF_OPEN (after recovery_timeout)
        HALF_OPEN -> CLOSED (after half_open_max_requests successes)
        HALF_OPEN -> OPEN  (on any failure)
    - Success in CLOSED resets failure_count
    - call() re-raises exceptions after recording failure
    - CircuitBreakerOpenError raised when calling OPEN breaker
    - Registry: lazy creation, snapshot, reset
"""

from __future__ import annotations

import asyncio
import time

import pytest

from icore.engine.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)
from icore.exceptions import CircuitBreakerOpenError


# ---------------------------------------------------------------------------
# CircuitBreaker state machine
# ---------------------------------------------------------------------------

class TestCircuitBreakerStates:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(name="m1")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    async def test_success_in_closed_resets_count(self):
        cb = CircuitBreaker(name="m1", failure_threshold=3)

        async def ok():
            return "ok"

        # Pre-poison failure_count
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        assert cb.failure_count == 2

        # A success resets it
        result = await cb.call(ok)
        assert result == "ok"
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    async def test_closed_to_open_after_threshold(self):
        cb = CircuitBreaker(name="m1", failure_threshold=3)

        for i in range(3):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError(f"fail-{i}")))

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    async def test_open_raises_circuit_breaker_open_error(self):
        cb = CircuitBreaker(name="m1", failure_threshold=1)

        with pytest.raises(ValueError):
            await cb.call(_raise(ValueError("first fail")))
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(_noop)  # should not even invoke the factory

    async def test_open_to_half_open_after_timeout(self):
        cb = CircuitBreaker(
            name="m1",
            failure_threshold=1,
            recovery_timeout=0.05,
        )
        with pytest.raises(ValueError):
            await cb.call(_raise(ValueError("fail")))
        assert cb.state == CircuitState.OPEN

        # Wait past recovery_timeout
        await asyncio.sleep(0.06)

        # Next call should transition to HALF_OPEN and execute
        result = await cb.call(_return("recovered"))
        assert result == "recovered"
        # Still HALF_OPEN (need half_open_max_requests successes)
        assert cb.state == CircuitState.HALF_OPEN

    async def test_half_open_to_closed_after_successes(self):
        cb = CircuitBreaker(
            name="m1",
            failure_threshold=1,
            recovery_timeout=0.01,
            half_open_max_requests=3,
        )
        with pytest.raises(ValueError):
            await cb.call(_raise(ValueError("fail")))
        await asyncio.sleep(0.02)

        # Drive three successful probes
        for _ in range(3):
            await cb.call(_return("ok"))

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    async def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(
            name="m1",
            failure_threshold=1,
            recovery_timeout=0.01,
            half_open_max_requests=3,
        )
        with pytest.raises(ValueError):
            await cb.call(_raise(ValueError("fail")))
        await asyncio.sleep(0.02)

        # First probe succeeds (transitions to HALF_OPEN)
        await cb.call(_return("ok"))
        assert cb.state == CircuitState.HALF_OPEN

        # Failure during HALF_OPEN immediately reverts to OPEN
        with pytest.raises(ValueError):
            await cb.call(_raise(ValueError("boom")))
        assert cb.state == CircuitState.OPEN

    async def test_call_re_raises_original_exception(self):
        cb = CircuitBreaker(name="m1", failure_threshold=5)

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            await cb.call(_raise(CustomError("custom")))


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestCircuitBreakerSnapshot:
    def test_snapshot_returns_state_dict(self):
        cb = CircuitBreaker(
            name="m1",
            failure_threshold=7,
            recovery_timeout=42.0,
        )
        snap = cb.snapshot()
        assert snap["name"] == "m1"
        assert snap["state"] == "closed"
        assert snap["failure_threshold"] == 7
        assert snap["recovery_timeout"] == 42.0
        assert snap["failure_count"] == 0

    def test_snapshot_reflects_failures(self):
        cb = CircuitBreaker(name="m1", failure_threshold=10)
        # Manually bump failure_count to avoid async setup
        cb.failure_count = 5
        snap = cb.snapshot()
        assert snap["failure_count"] == 5


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestCircuitBreakerRegistry:
    async def test_get_creates_lazily(self):
        reg = CircuitBreakerRegistry()
        cb1 = await reg.get("m1")
        cb2 = await reg.get("m1")
        assert cb1 is cb2

    async def test_get_different_names_returns_different_breakers(self):
        reg = CircuitBreakerRegistry()
        cb1 = await reg.get("m1")
        cb2 = await reg.get("m2")
        assert cb1 is not cb2
        assert cb1.name == "m1"
        assert cb2.name == "m2"

    async def test_get_uses_configured_defaults(self):
        reg = CircuitBreakerRegistry(
            failure_threshold=10,
            recovery_timeout=99.0,
            half_open_max_requests=7,
        )
        cb = await reg.get("m1")
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 99.0
        assert cb.half_open_max_requests == 7

    async def test_snapshot_lists_all_breakers(self):
        reg = CircuitBreakerRegistry()
        await reg.get("m1")
        await reg.get("m2")
        snap = reg.snapshot()
        assert len(snap) == 2
        names = {s["name"] for s in snap}
        assert names == {"m1", "m2"}

    async def test_reset_all_breakers(self):
        reg = CircuitBreakerRegistry(failure_threshold=1)
        cb1 = await reg.get("m1")
        cb2 = await reg.get("m2")
        # Trip both
        with pytest.raises(ValueError):
            await cb1.call(_raise(ValueError("x")))
        with pytest.raises(ValueError):
            await cb2.call(_raise(ValueError("x")))
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        await reg.reset()
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED
        assert cb1.failure_count == 0

    async def test_reset_single_breaker(self):
        reg = CircuitBreakerRegistry(failure_threshold=1)
        cb1 = await reg.get("m1")
        cb2 = await reg.get("m2")
        with pytest.raises(ValueError):
            await cb1.call(_raise(ValueError("x")))
        with pytest.raises(ValueError):
            await cb2.call(_raise(ValueError("x")))

        await reg.reset("m1")
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.OPEN

    async def test_reset_unknown_name_is_noop(self):
        reg = CircuitBreakerRegistry()
        # Should not raise
        await reg.reset("nonexistent")


# ---------------------------------------------------------------------------
# ICORE-ISSUE-006: windowed failure-rate trip criterion
# ---------------------------------------------------------------------------

class TestWindowedFailureRate:
    """ICORE-ISSUE-006: 可选「时间窗失败率」触发口径。

    设置 failure_rate_threshold 后 CLOSED 态判定从「连续 N 次失败」
    切换为「滑动窗口内失败率 >= 阈值且样本 >= min_samples」；
    未设置时既有连续计数语义不变（由 TestCircuitBreakerStates 守护）。
    """

    async def test_rate_mode_trips_at_threshold(self):
        cb = CircuitBreaker(
            name="m1",
            failure_threshold=100,  # 连续计数口径刻意调高：比率模式替代它
            failure_rate_threshold=0.8,
            failure_rate_window=30.0,
            min_samples=10,
        )
        for _ in range(8):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        # 样本不足 min_samples：即使 8/8 全失败也不触发。
        assert cb.state == CircuitState.CLOSED
        # 2 个成功样本凑满 10 样本：8/10 = 80% >= 0.8 → OPEN
        # （样本凑满瞬间可能是成功调用，成功后同样判定）。
        await cb.call(_noop)
        assert cb.state == CircuitState.CLOSED
        await cb.call(_noop)
        assert cb.state == CircuitState.OPEN

    async def test_rate_mode_min_samples_guards_small_samples(self):
        cb = CircuitBreaker(
            name="m1",
            failure_rate_threshold=0.5,
            failure_rate_window=30.0,
            min_samples=10,
        )
        for _ in range(9):  # 9 样本 100% 失败，但 < min_samples
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        assert cb.state == CircuitState.CLOSED

    async def test_rate_mode_replaces_consecutive_mode(self):
        # 比率模式下连续计数不再触发：failure_threshold=3、连续 3 次
        # 失败（样本 3 < min_samples=10）不 OPEN。
        cb = CircuitBreaker(
            name="m1",
            failure_threshold=3,
            failure_rate_threshold=0.99,
            min_samples=10,
        )
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        assert cb.state == CircuitState.CLOSED

    async def test_rate_mode_success_dilutes_window(self):
        cb = CircuitBreaker(
            name="m1",
            failure_rate_threshold=0.8,
            failure_rate_window=30.0,
            min_samples=10,
        )
        for _ in range(5):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        for _ in range(5):
            await cb.call(_noop)
        # 5/10 = 50% < 80%：成功样本稀释，不触发。
        assert cb.state == CircuitState.CLOSED

    async def test_rate_mode_window_expiry_resets_samples(self):
        cb = CircuitBreaker(
            name="m1",
            failure_rate_threshold=0.5,
            failure_rate_window=0.05,
            min_samples=2,
            half_open_max_requests=1,
        )
        # 第一窗口内 100% 失败且样本足够 → 触发。
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        assert cb.state == CircuitState.OPEN
        # 恢复后旧样本全部过期（window 极短），窗口从零开始。
        await asyncio.sleep(0.07)
        cb.state = CircuitState.HALF_OPEN
        await cb.call(_noop)  # probe success -> CLOSED
        snap = cb.snapshot()
        assert snap["window_samples"] <= 1
        assert cb.state == CircuitState.CLOSED

    async def test_rate_mode_half_open_semantics_unchanged(self):
        cb = CircuitBreaker(
            name="m1",
            failure_rate_threshold=0.5,
            failure_rate_window=30.0,
            min_samples=2,
            recovery_timeout=0.05,
        )
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.07)
        # HALF_OPEN probe 失败立即 OPEN（无需样本判定，语义不变）。
        with pytest.raises(ValueError):
            await cb.call(_raise(ValueError("probe boom")))
        assert cb.state == CircuitState.OPEN
        assert cb.half_open_failures == 1

    async def test_invalid_rate_config_raises(self):
        with pytest.raises(ValueError):
            CircuitBreaker(name="m1", failure_rate_threshold=0.0)
        with pytest.raises(ValueError):
            CircuitBreaker(name="m1", failure_rate_threshold=1.5)
        with pytest.raises(ValueError):
            CircuitBreaker(name="m1", failure_rate_threshold=0.8, failure_rate_window=0.0)
        with pytest.raises(ValueError):
            CircuitBreaker(name="m1", failure_rate_threshold=0.8, min_samples=0)

    async def test_snapshot_exposes_window_stats(self):
        cb = CircuitBreaker(
            name="m1",
            failure_rate_threshold=0.8,
            min_samples=10,
        )
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        snap = cb.snapshot()
        assert snap["failure_rate_threshold"] == 0.8
        assert snap["failure_rate_window"] == 30.0
        assert snap["min_samples"] == 10
        assert snap["window_samples"] == 3
        assert snap["window_failures"] == 3
        assert snap["window_failure_rate"] == 1.0

    async def test_default_mode_records_no_samples(self):
        # 未设比率口径：零开销，不记录样本。
        cb = CircuitBreaker(name="m1", failure_threshold=2)
        with pytest.raises(ValueError):
            await cb.call(_raise(ValueError("boom")))
        assert len(cb._samples) == 0
        snap = cb.snapshot()
        assert "window_samples" not in snap

    async def test_registry_passes_rate_config(self):
        reg = CircuitBreakerRegistry(
            failure_rate_threshold=0.5,
            failure_rate_window=30.0,
            min_samples=2,
        )
        cb = await reg.get("m1")
        assert cb.failure_rate_threshold == 0.5
        assert cb.min_samples == 2
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_raise(ValueError("boom")))
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _noop():
    return None


def _return(value):
    async def _factory():
        return value
    return _factory


def _raise(exc):
    async def _factory():
        raise exc
    return _factory
