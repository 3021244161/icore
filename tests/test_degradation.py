"""
Tests for v0.6 GracefulDegradationCoordinator (v0.6 §3.3.2).

Covers:
    * Registration with / without fallback
    * mark_failed threshold -> DEGRADED / UNAVAILABLE
    * mark_recovered restores primary
    * guard() context manager auto-marks failures
    * get() returns primary / fallback / raises RuntimeError when unavailable
    * Snapshot structure for /health
    * Auto-recovery watcher: degraded -> healthy primary -> mark_recovered
    * is_degraded quick check
    * reset() helper
    * KeyError on unknown component
"""

from __future__ import annotations

import asyncio

import pytest

from icore.engine.graceful_degradation import (
    DegradationSnapshot,
    DegradationState,
    GracefulDegradationCoordinator,
)


# ===========================================================================
# Test fixtures (provider stubs)
# ===========================================================================

class _Stub:
    """A stub provider identified by a label."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"<Stub {self.label}>"


async def _healthy_probe() -> bool:
    return True


async def _unhealthy_probe() -> bool:
    return False


# ===========================================================================
# Registration
# ===========================================================================

class TestRegistration:
    def test_register_with_fallback_lists_component(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(
            name="vectorstore",
            primary=_Stub("milvus"),
            fallback=_Stub("in_memory"),
            health_check=_healthy_probe,
        )
        assert "vectorstore" in coord.list_components()

    def test_register_without_fallback_lists_component(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(name="lock", primary=_Stub("redis_lock"))
        assert "lock" in coord.list_components()

    def test_unregister_removes_component(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(name="x", primary=_Stub("p"))
        coord.unregister("x")
        assert "x" not in coord.list_components()
        # Unregistering an unknown name is a no-op.
        coord.unregister("nonexistent")


# ===========================================================================
# Failure tracking
# ===========================================================================

class TestFailureTracking:
    def test_below_threshold_stays_primary(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(
            name="v",
            primary=_Stub("p"),
            fallback=_Stub("f"),
            failure_threshold=3,
        )
        coord.mark_failed("v", "transient")
        coord.mark_failed("v", "transient")
        state = coord.get_state("v")
        assert state.state == DegradationState.PRIMARY
        assert state.failure_count == 2
        assert state.last_error == "transient"

    def test_threshold_reached_with_fallback_degrades(self) -> None:
        coord = GracefulDegradationCoordinator()
        primary = _Stub("p")
        fallback = _Stub("f")
        coord.register(
            name="v",
            primary=primary,
            fallback=fallback,
            failure_threshold=3,
        )
        for _ in range(3):
            coord.mark_failed("v", "boom")
        state = coord.get_state("v")
        assert state.state == DegradationState.DEGRADED
        # Active provider is now the fallback.
        assert coord.get("v") is fallback

    def test_threshold_reached_without_fallback_marks_unavailable(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(
            name="v",
            primary=_Stub("p"),
            fallback=None,
            failure_threshold=2,
        )
        coord.mark_failed("v", "x")
        coord.mark_failed("v", "x")
        state = coord.get_state("v")
        assert state.state == DegradationState.UNAVAILABLE
        with pytest.raises(RuntimeError):
            coord.get("v")

    def test_mark_recovered_restores_primary(self) -> None:
        coord = GracefulDegradationCoordinator()
        primary = _Stub("p")
        fallback = _Stub("f")
        coord.register(
            name="v",
            primary=primary,
            fallback=fallback,
            failure_threshold=1,
        )
        coord.mark_failed("v", "boom")
        assert coord.get_state("v").state == DegradationState.DEGRADED
        coord.mark_recovered("v")
        state = coord.get_state("v")
        assert state.state == DegradationState.PRIMARY
        assert state.failure_count == 0
        assert state.last_error is None
        assert state.last_recovery is not None
        assert coord.get("v") is primary

    def test_reset_clears_state_without_changing_provider(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(
            name="v",
            primary=_Stub("p"),
            fallback=_Stub("f"),
            failure_threshold=5,
        )
        coord.mark_failed("v", "x")
        coord.mark_failed("v", "y")
        coord.reset("v")
        state = coord.get_state("v")
        assert state.state == DegradationState.PRIMARY
        assert state.failure_count == 0


# ===========================================================================
# Guard context manager
# ===========================================================================

class TestGuard:
    def test_guard_no_exception_does_not_mark_failed(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(
            name="v",
            primary=_Stub("p"),
            fallback=_Stub("f"),
            failure_threshold=1,
        )
        with coord.guard("v"):
            pass
        assert coord.get_state("v").failure_count == 0
        assert coord.get_state("v").state == DegradationState.PRIMARY

    def test_guard_exception_marks_failed(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(
            name="v",
            primary=_Stub("p"),
            fallback=_Stub("f"),
            failure_threshold=1,
        )
        with pytest.raises(ValueError):
            with coord.guard("v"):
                raise ValueError("boom")
        state = coord.get_state("v")
        assert state.failure_count == 1
        assert state.state == DegradationState.DEGRADED
        assert "boom" in (state.last_error or "")


# ===========================================================================
# Snapshot / observability
# ===========================================================================

class TestSnapshot:
    def test_snapshot_when_all_primary(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(name="a", primary=_Stub("p1"), fallback=_Stub("f1"))
        coord.register(name="b", primary=_Stub("p2"))
        snap = coord.snapshot()
        assert isinstance(snap, DegradationSnapshot)
        assert snap.any_degraded is False
        assert set(snap.components.keys()) == {"a", "b"}
        d = snap.to_dict()
        assert d["any_degraded"] is False
        assert "components" in d
        assert "timestamp" in d
        for name, c in d["components"].items():
            assert {"name", "state", "failure_count", "last_failure", "last_recovery", "last_error"} <= set(c.keys())

    def test_snapshot_when_one_degraded(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(
            name="a",
            primary=_Stub("p1"),
            fallback=_Stub("f1"),
            failure_threshold=1,
        )
        coord.register(name="b", primary=_Stub("p2"))
        coord.mark_failed("a", "boom")
        snap = coord.snapshot()
        assert snap.any_degraded is True
        assert snap.components["a"].state == DegradationState.DEGRADED
        assert snap.components["b"].state == DegradationState.PRIMARY

    def test_is_degraded_helper(self) -> None:
        coord = GracefulDegradationCoordinator()
        coord.register(
            name="a",
            primary=_Stub("p1"),
            fallback=_Stub("f1"),
            failure_threshold=1,
        )
        assert coord.is_degraded() is False
        assert coord.is_degraded("a") is False
        coord.mark_failed("a", "x")
        assert coord.is_degraded() is True
        assert coord.is_degraded("a") is True


# ===========================================================================
# Auto-recovery watcher
# ===========================================================================

class TestAutoRecovery:
    async def test_watch_recovers_when_primary_health_returns_true(self) -> None:
        coord = GracefulDegradationCoordinator()
        primary = _Stub("p")
        fallback = _Stub("f")
        coord.register(
            name="v",
            primary=primary,
            fallback=fallback,
            health_check=_healthy_probe,
            failure_threshold=1,
            recovery_interval=0.05,
        )
        coord.mark_failed("v", "transient")
        assert coord.get_state("v").state == DegradationState.DEGRADED

        await coord.start_watch()
        try:
            # Wait for the watcher to probe and recover.
            for _ in range(50):
                if coord.get_state("v").state == DegradationState.PRIMARY:
                    break
                await asyncio.sleep(0.05)
            assert coord.get_state("v").state == DegradationState.PRIMARY
            assert coord.get("v") is primary
        finally:
            await coord.stop_watch()

    async def test_watch_does_not_recover_when_probe_fails(self) -> None:
        coord = GracefulDegradationCoordinator()
        primary = _Stub("p")
        fallback = _Stub("f")
        coord.register(
            name="v",
            primary=primary,
            fallback=fallback,
            health_check=_unhealthy_probe,
            failure_threshold=1,
            recovery_interval=0.05,
        )
        coord.mark_failed("v", "x")
        await coord.start_watch()
        try:
            await asyncio.sleep(0.2)
            assert coord.get_state("v").state == DegradationState.DEGRADED
            assert coord.get("v") is fallback
        finally:
            await coord.stop_watch()

    async def test_start_stop_idempotent(self) -> None:
        coord = GracefulDegradationCoordinator()
        await coord.start_watch()
        await coord.start_watch()  # no-op
        await coord.stop_watch()
        await coord.stop_watch()  # no-op


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_get_unknown_component_raises_keyerror(self) -> None:
        coord = GracefulDegradationCoordinator()
        with pytest.raises(KeyError):
            coord.get("nonexistent")

    def test_get_state_unknown_component_raises_keyerror(self) -> None:
        coord = GracefulDegradationCoordinator()
        with pytest.raises(KeyError):
            coord.get_state("nonexistent")

    def test_mark_failed_unknown_component_silent(self) -> None:
        coord = GracefulDegradationCoordinator()
        # Should not raise.
        coord.mark_failed("nonexistent", "x")
        coord.mark_recovered("nonexistent")
        coord.reset("nonexistent")

    async def test_close_stops_watcher(self) -> None:
        coord = GracefulDegradationCoordinator()
        await coord.start_watch()
        await coord.close()
        assert coord._watch_task is None
