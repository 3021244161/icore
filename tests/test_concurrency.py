"""
Smoke tests for icore.engine concurrency control components.

Covers:
    - TokenBucket: acquire, refill, capacity, invalid args
    - ConcurrencyController: acquire/release, per-workflow isolation,
      backpressure detection, stats, rate limiting
    - TaskQueue (memory backend): enqueue/dequeue, priority ordering,
      size, clear, max_size overflow, lifecycle
    - TaskInstanceManager: create, dedup, state transitions, cancel,
      cleanup TTL, list/count queries
    - BackpressureError: attributes

These tests are deterministic and offline (no Redis, no network).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from icore.engine.concurrency_control import (
    BackpressureError,
    ConcurrencyController,
    ConcurrencyStats,
    TokenBucket,
)
from icore.engine.instance_manager import TaskInstance, TaskInstanceManager
from icore.engine.states import TaskState
from icore.engine.task_queue import QueueFullError, TaskItem, TaskQueue


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_invalid_args_rejected(self):
        with pytest.raises(ValueError):
            TokenBucket(capacity=0)
        with pytest.raises(ValueError):
            TokenBucket(capacity=10, refill_rate=0)

    @pytest.mark.asyncio
    async def test_acquire_within_capacity_does_not_block(self):
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        # Should consume 3 tokens instantly (bucket starts full)
        start = time.time()
        await bucket.acquire(n=3)
        elapsed = time.time() - start
        assert elapsed < 0.1
        assert bucket.available_tokens == pytest.approx(2.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_acquire_zero_or_negative_is_noop(self):
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        await bucket.acquire(n=0)
        assert bucket.available_tokens == pytest.approx(5.0, abs=0.01)
        await bucket.acquire(n=-3)
        assert bucket.available_tokens == pytest.approx(5.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_acquire_blocks_then_refills(self):
        # capacity=2, refill 10 tokens/sec -> 0.1s per token
        bucket = TokenBucket(capacity=2, refill_rate=10.0)
        await bucket.acquire(n=2)  # drain
        # Need 1 more token -> should wait ~0.1s
        start = time.time()
        await bucket.acquire(n=1)
        elapsed = time.time() - start
        assert elapsed >= 0.05  # at least some wait
        assert elapsed < 0.5  # but not too long


# ---------------------------------------------------------------------------
# ConcurrencyController
# ---------------------------------------------------------------------------

class TestConcurrencyController:
    def test_defaults_and_properties(self):
        ctrl = ConcurrencyController(
            max_concurrent_tasks=100,
            max_concurrent_per_workflow=20,
            backpressure_threshold=500,
        )
        assert ctrl.max_concurrent == 100
        assert ctrl.max_per_workflow == 20
        assert ctrl.backpressure_threshold == 500
        assert ctrl.active_count == 0

    @pytest.mark.asyncio
    async def test_acquire_release_updates_active_count(self):
        ctrl = ConcurrencyController(max_concurrent_tasks=10, max_concurrent_per_workflow=5)
        assert ctrl.active_count == 0
        async with ctrl.acquire("wf_a"):
            assert ctrl.active_count == 1
        assert ctrl.active_count == 0

    @pytest.mark.asyncio
    async def test_per_workflow_stats_tracked(self):
        ctrl = ConcurrencyController(max_concurrent_tasks=10, max_concurrent_per_workflow=5)
        async with ctrl.acquire("wf_a"):
            stats = await ctrl.get_stats()
            assert stats.active_global == 1
            assert stats.active_per_wf.get("wf_a") == 1
            assert stats.max_global == 10
            assert stats.max_per_workflow == 5
            assert stats.backpressure is False

    @pytest.mark.asyncio
    async def test_release_on_exception(self):
        ctrl = ConcurrencyController(max_concurrent_tasks=10, max_concurrent_per_workflow=5)
        with pytest.raises(RuntimeError):
            async with ctrl.acquire("wf_x"):
                raise RuntimeError("boom")
        # Slot must have been released
        assert ctrl.active_count == 0

    @pytest.mark.asyncio
    async def test_backpressure_when_active_reaches_max(self):
        ctrl = ConcurrencyController(
            max_concurrent_tasks=2,
            max_concurrent_per_workflow=2,
        )
        # Occupy both global slots
        slot1 = ctrl.acquire("wf_a")
        slot2 = ctrl.acquire("wf_b")
        await slot1.__aenter__()
        await slot2.__aenter__()
        try:
            assert await ctrl.is_backpressure(None) is True
        finally:
            await slot1.__aexit__(None, None, None)
            await slot2.__aexit__(None, None, None)
        assert await ctrl.is_backpressure(None) is False

    @pytest.mark.asyncio
    async def test_backpressure_triggered_by_queue_depth(self):
        ctrl = ConcurrencyController(
            max_concurrent_tasks=100,
            max_concurrent_per_workflow=20,
            backpressure_threshold=3,
        )
        queue = TaskQueue(backend="memory")
        await queue.start()
        try:
            for i in range(3):
                await queue.enqueue(f"t-{i}")
            assert await ctrl.is_backpressure(queue) is True
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_rate_limiting_registration_and_use(self):
        ctrl = ConcurrencyController()
        assert ctrl.has_rate_limiter("gpt-4o") is False
        bucket = ctrl.register_rate_limit("gpt-4o", capacity=5, refill_rate=10.0)
        assert ctrl.has_rate_limiter("gpt-4o") is True
        assert bucket is not None
        # Acquire should not block (bucket starts full)
        await ctrl.rate_limit("gpt-4o", n=1)
        # Unknown model is a no-op
        await ctrl.rate_limit("unknown-model", n=1)

    @pytest.mark.asyncio
    async def test_concurrent_acquires_serialize_under_global_limit(self):
        """Multiple coroutines acquiring should be bounded by max_concurrent."""
        ctrl = ConcurrencyController(
            max_concurrent_tasks=2,
            max_concurrent_per_workflow=10,
        )
        peak = 0

        async def worker():
            nonlocal peak
            async with ctrl.acquire("wf"):
                cur = ctrl.active_count
                if cur > peak:
                    peak = cur
                await asyncio.sleep(0.05)

        await asyncio.gather(*(worker() for _ in range(6)))
        assert peak <= 2
        assert ctrl.active_count == 0

    @pytest.mark.asyncio
    async def test_per_workflow_semaphore_isolates_workflows(self):
        """Per-workflow limit caps concurrent same-name executions."""
        ctrl = ConcurrencyController(
            max_concurrent_tasks=100,
            max_concurrent_per_workflow=2,
        )
        peak_wf = 0
        in_flight = 0

        async def worker():
            nonlocal peak_wf, in_flight
            async with ctrl.acquire("wf_iso"):
                in_flight += 1
                if in_flight > peak_wf:
                    peak_wf = in_flight
                await asyncio.sleep(0.05)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(6)))
        assert peak_wf <= 2

    def test_backpressure_error_attributes(self):
        err = BackpressureError("overloaded", retry_after=10)
        assert str(err) == "overloaded"
        assert err.retry_after == 10
        # Default retry_after
        err2 = BackpressureError("again")
        assert err2.retry_after == 5


# ---------------------------------------------------------------------------
# TaskQueue (memory backend)
# ---------------------------------------------------------------------------

class TestTaskQueue:
    @pytest.mark.asyncio
    async def test_enqueue_dequeue_fifo_same_priority(self):
        q = TaskQueue(backend="memory")
        await q.start()
        try:
            await q.enqueue("t1")
            await q.enqueue("t2")
            await q.enqueue("t3")
            assert await q.size() == 3
            first = await q.dequeue()
            second = await q.dequeue()
            third = await q.dequeue()
            assert [first.task_id, second.task_id, third.task_id] == [
                "t1",
                "t2",
                "t3",
            ]
            assert await q.size() == 0
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        q = TaskQueue(backend="memory")
        await q.start()
        try:
            # Lower priority value = dequeued first
            await q.enqueue("low", priority=10)
            await q.enqueue("urgent", priority=-1)
            await q.enqueue("normal", priority=0)
            order = [item.task_id for item in [
                await q.dequeue(),
                await q.dequeue(),
                await q.dequeue(),
            ]]
            assert order == ["urgent", "normal", "low"]
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_metadata_carried_through(self):
        q = TaskQueue(backend="memory")
        await q.start()
        try:
            await q.enqueue(
                "t-meta",
                priority=5,
                metadata={"workflow": "wf_x", "user": "alice"},
            )
            item = await q.dequeue()
            assert item.metadata["workflow"] == "wf_x"
            assert item.metadata["user"] == "alice"
            assert item.priority == 5
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_dequeue_timeout_raises(self):
        q = TaskQueue(backend="memory")
        await q.start()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await q.dequeue(timeout=0.1)
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_max_size_overflow_raises(self):
        q = TaskQueue(backend="memory", max_size=2)
        await q.start()
        try:
            await q.enqueue("t1")
            await q.enqueue("t2")
            with pytest.raises(QueueFullError):
                await q.enqueue("t3")
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_clear_removes_all(self):
        q = TaskQueue(backend="memory")
        await q.start()
        try:
            await q.enqueue("t1")
            await q.enqueue("t2")
            removed = await q.clear()
            assert removed == 2
            assert await q.size() == 0
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_lifecycle_idempotent(self):
        q = TaskQueue(backend="memory")
        assert q.is_started is False
        await q.start()
        assert q.is_started is True
        await q.start()  # no-op
        assert q.is_started is True
        await q.stop()
        assert q.is_started is False
        await q.stop()  # no-op

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError):
            TaskQueue(backend="kafka")

    def test_redis_requires_url(self):
        with pytest.raises(ValueError):
            TaskQueue(backend="redis", redis_url=None)

    def test_taskitem_ordering(self):
        # Higher priority (lower number) first; FIFO within same priority
        a = TaskItem(priority=1, seq=1, task_id="a")
        b = TaskItem(priority=1, seq=2, task_id="b")
        c = TaskItem(priority=0, seq=3, task_id="c")
        ordered = sorted([a, b, c])
        assert [i.task_id for i in ordered] == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# TaskInstanceManager
# ---------------------------------------------------------------------------

class TestTaskInstanceManager:
    @pytest.mark.asyncio
    async def test_create_and_get(self):
        mgr = TaskInstanceManager(ttl=3600)
        inst = await mgr.create("t1", "document_summary", {"text": "hi"})
        assert inst.task_id == "t1"
        assert inst.workflow_name == "document_summary"
        assert inst.state == TaskState.PENDING
        assert inst.is_active is True
        assert inst.is_terminal is False

        fetched = await mgr.get("t1")
        assert fetched is inst

    @pytest.mark.asyncio
    async def test_create_duplicate_active_raises(self):
        mgr = TaskInstanceManager()
        await mgr.create("dup", "wf", {})
        with pytest.raises(ValueError):
            await mgr.create("dup", "wf", {})

    @pytest.mark.asyncio
    async def test_terminal_duplicate_replaces(self):
        mgr = TaskInstanceManager()
        await mgr.create("redo", "wf", {})
        await mgr.update_state("redo", TaskState.COMPLETED)
        # Should be allowed (re-run)
        new_inst = await mgr.create("redo", "wf", {"v": 2})
        assert new_inst.params == {"v": 2}
        assert new_inst.state == TaskState.PENDING

    @pytest.mark.asyncio
    async def test_state_transitions(self):
        mgr = TaskInstanceManager()
        await mgr.create("t", "wf", {})
        await mgr.update_state("t", TaskState.RUNNING)
        assert (await mgr.get("t")).state == TaskState.RUNNING
        await mgr.update_state("t", TaskState.COMPLETED, result={"summary": "ok"})
        inst = await mgr.get("t")
        assert inst.state == TaskState.COMPLETED
        assert inst.is_terminal is True
        assert inst.result == {"summary": "ok"}

    @pytest.mark.asyncio
    async def test_update_unknown_task_returns_none(self):
        mgr = TaskInstanceManager()
        result = await mgr.update_state("ghost", TaskState.RUNNING)
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_active_task(self):
        mgr = TaskInstanceManager()
        await mgr.create("c", "wf", {})
        assert await mgr.cancel("c") is True
        inst = await mgr.get("c")
        assert inst.state == TaskState.CANCELLED
        assert inst.is_terminal is True
        assert await mgr.is_cancelled("c") is True

    @pytest.mark.asyncio
    async def test_cancel_terminal_or_unknown_returns_false(self):
        mgr = TaskInstanceManager()
        await mgr.create("done", "wf", {})
        await mgr.update_state("done", TaskState.COMPLETED)
        assert await mgr.cancel("done") is False
        assert await mgr.cancel("ghost") is False

    @pytest.mark.asyncio
    async def test_list_and_count(self):
        mgr = TaskInstanceManager()
        await mgr.create("a", "wf", {})
        await mgr.create("b", "wf", {})
        await mgr.create("c", "wf", {})
        await mgr.update_state("c", TaskState.COMPLETED)

        active = await mgr.list_active()
        all_ = await mgr.list_all()
        assert len(active) == 2
        assert len(all_) == 3
        assert await mgr.count_active() == 2

        counts = await mgr.count_by_state()
        assert counts.get("PENDING") == 2
        assert counts.get("COMPLETED") == 1

    @pytest.mark.asyncio
    async def test_exists_and_remove(self):
        mgr = TaskInstanceManager()
        await mgr.create("x", "wf", {})
        assert await mgr.exists("x") is True
        assert await mgr.exists("y") is False
        assert await mgr.remove("x") is True
        assert await mgr.remove("x") is False

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_terminal(self):
        # Use TTL=0 so any terminal instance is immediately eligible
        mgr = TaskInstanceManager(ttl=0)
        await mgr.create("old", "wf", {})
        await mgr.update_state("old", TaskState.COMPLETED)
        # Force updated_at into the past
        inst = await mgr.get("old")
        inst.updated_at = time.time() - 100
        removed = await mgr.cleanup()
        assert removed == 1
        assert await mgr.exists("old") is False

    @pytest.mark.asyncio
    async def test_clear_all(self):
        mgr = TaskInstanceManager()
        await mgr.create("a", "wf", {})
        await mgr.create("b", "wf", {})
        n = await mgr.clear_all()
        assert n == 2
        assert await mgr.count_active() == 0

    def test_instance_to_dict(self):
        inst = TaskInstance(task_id="t", workflow_name="wf", params={"k": 1})
        d = inst.to_dict()
        assert d["task_id"] == "t"
        assert d["workflow_name"] == "wf"
        assert d["state"] == "PENDING"
        assert d["is_terminal"] is False
        assert "cancel_event" not in d  # excluded

    def test_concurrency_stats_dataclass(self):
        stats = ConcurrencyStats(
            active_global=3,
            active_per_wf={"wf_a": 2, "wf_b": 1},
            queue_depth=5,
            max_global=100,
            max_per_workflow=20,
            backpressure=False,
        )
        assert stats.active_global == 3
        assert stats.queue_depth == 5
        assert stats.backpressure is False
        assert stats.timestamp > 0  # auto-set
