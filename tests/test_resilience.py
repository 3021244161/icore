"""
Tests for v0.6 structured resilience: BackoffStrategy / DeadLetterQueue / Saga.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

from icore.core.models import BaseTaskOutput
from icore.core.task_context import TaskContext
from icore.engine.backoff import BackoffStrategy, compute_delay, retry_with_backoff
from icore.engine.dead_letter_queue import (
    DLQEntry,
    DeadLetterQueue,
    InMemoryDLQBackend,
    PostgresDLQBackend,
)
from icore.engine.saga import SagaStep, SagaWorkflow
from icore.exceptions import (
    DeadLetterQueueError,
    SagaCompensationError,
)


# ===========================================================================
# BackoffStrategy
# ===========================================================================

class TestBackoffStrategy:
    def test_constant_strategy(self) -> None:
        for attempt in range(1, 5):
            assert compute_delay(BackoffStrategy.CONSTANT, 1.0, attempt) == 1.0

    def test_linear_strategy(self) -> None:
        assert compute_delay(BackoffStrategy.LINEAR, 1.0, 1) == 1.0
        assert compute_delay(BackoffStrategy.LINEAR, 1.0, 2) == 2.0
        assert compute_delay(BackoffStrategy.LINEAR, 1.0, 3) == 3.0

    def test_exponential_strategy(self) -> None:
        assert compute_delay(BackoffStrategy.EXPONENTIAL, 1.0, 1) == 1.0
        assert compute_delay(BackoffStrategy.EXPONENTIAL, 1.0, 2) == 2.0
        assert compute_delay(BackoffStrategy.EXPONENTIAL, 1.0, 3) == 4.0
        assert compute_delay(BackoffStrategy.EXPONENTIAL, 1.0, 4) == 8.0

    def test_exponential_jitter_within_bounds(self) -> None:
        base = 1.0
        for attempt in range(1, 6):
            delay = compute_delay(
                BackoffStrategy.EXPONENTIAL_JITTER, base, attempt
            )
            exp = base * (2 ** (attempt - 1))
            # jitter is +/- 25% of the exponential value.
            assert (exp * 0.75) <= delay <= (exp * 1.25)

    def test_max_delay_caps_result(self) -> None:
        # Attempt 10 with base 1 -> 2^9 = 512; capped at 60.
        assert compute_delay(
            BackoffStrategy.EXPONENTIAL, 1.0, 10, max_delay=60.0
        ) == 60.0

    def test_negative_base_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_delay(BackoffStrategy.CONSTANT, -1.0, 1)

    def test_attempt_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_delay(BackoffStrategy.CONSTANT, 1.0, 0)

    def test_jitter_never_negative(self) -> None:
        # base 0 with jitter should be 0 (not negative).
        for _ in range(20):
            assert compute_delay(
                BackoffStrategy.EXPONENTIAL_JITTER, 0.0, 1
            ) == 0.0

    @pytest.mark.asyncio
    async def test_retry_with_backoff_succeeds_first_try(self) -> None:
        calls = 0

        async def factory() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_with_backoff(
            factory,
            max_attempts=3,
            strategy=BackoffStrategy.CONSTANT,
            base_delay=0.01,
        )
        assert result == "ok"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retry_with_backoff_retries_then_succeeds(self) -> None:
        calls = 0

        async def factory() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("transient")
            return "ok"

        result = await retry_with_backoff(
            factory,
            max_attempts=5,
            strategy=BackoffStrategy.CONSTANT,
            base_delay=0.001,
        )
        assert result == "ok"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_exhausts_and_raises(self) -> None:
        calls = 0

        async def factory() -> str:
            nonlocal calls
            calls += 1
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await retry_with_backoff(
                factory,
                max_attempts=3,
                strategy=BackoffStrategy.CONSTANT,
                base_delay=0.001,
            )
        assert calls == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_respects_retry_on_whitelist(self) -> None:
        calls = 0

        async def factory() -> str:
            nonlocal calls
            calls += 1
            # ValueError is in the whitelist; KeyError is not.
            if calls == 1:
                raise ValueError("retry me")
            raise KeyError("do not retry me")

        with pytest.raises(KeyError):
            await retry_with_backoff(
                factory,
                max_attempts=5,
                strategy=BackoffStrategy.CONSTANT,
                base_delay=0.001,
                retry_on=(ValueError,),
            )
        # First call (ValueError) was retried; second call (KeyError)
        # raised immediately.
        assert calls == 2

    @pytest.mark.asyncio
    async def test_retry_with_backoff_invokes_on_retry_callback(self) -> None:
        attempts_seen: list[int] = []
        excs_seen: list[BaseException] = []

        def on_retry(attempt: int, exc: BaseException) -> None:
            attempts_seen.append(attempt)
            excs_seen.append(exc)

        async def factory() -> str:
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError):
            await retry_with_backoff(
                factory,
                max_attempts=3,
                strategy=BackoffStrategy.CONSTANT,
                base_delay=0.001,
                on_retry=on_retry,
            )
        # callback fires before retries (attempts 1 and 2).
        assert attempts_seen == [1, 2]
        assert len(excs_seen) == 2
        assert all(isinstance(e, RuntimeError) for e in excs_seen)

    @pytest.mark.asyncio
    async def test_retry_with_backoff_invalid_max_attempts(self) -> None:
        async def factory() -> str:
            return "ok"

        with pytest.raises(ValueError):
            await retry_with_backoff(factory, max_attempts=0)


# ===========================================================================
# DeadLetterQueue
# ===========================================================================

class TestInMemoryDLQBackend:
    @pytest.mark.asyncio
    async def test_enqueue_and_get(self) -> None:
        backend = InMemoryDLQBackend()
        entry = DLQEntry(
            id="e1",
            workflow_name="wf",
            task_id="t1",
            node_id="n1",
            task_name="task",
            params={"x": 1},
            error="boom",
            enqueued_at=time.time(),
        )
        await backend.enqueue(entry)
        fetched = await backend.get("e1")
        assert fetched is not None
        assert fetched.id == "e1"
        assert fetched.error == "boom"
        # get on missing id returns None
        assert await backend.get("missing") is None

    @pytest.mark.asyncio
    async def test_list_entries_filters_and_sorts(self) -> None:
        backend = InMemoryDLQBackend()
        now = time.time()
        e1 = DLQEntry(id="e1", workflow_name="wf_a", task_id="t1", node_id="n1", enqueued_at=now)
        e2 = DLQEntry(id="e2", workflow_name="wf_b", task_id="t2", node_id="n2", enqueued_at=now + 1)
        e3 = DLQEntry(id="e3", workflow_name="wf_a", task_id="t3", node_id="n3", enqueued_at=now + 2)
        # Mark e3 as replayed.
        e3.replayed = True
        for e in (e1, e2, e3):
            await backend.enqueue(e)

        # Default: exclude replayed.
        result = await backend.list_entries()
        assert {e.id for e in result} == {"e1", "e2"}
        # Sorted ascending by enqueued_at.
        assert [e.id for e in result] == ["e1", "e2"]

        # Filter by workflow.
        result = await backend.list_entries(workflow_name="wf_a")
        assert [e.id for e in result] == ["e1"]

        # Include replayed.
        result = await backend.list_entries(include_replayed=True)
        assert {e.id for e in result} == {"e1", "e2", "e3"}

        # Limit.
        result = await backend.list_entries(include_replayed=True, limit=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_mark_replayed(self) -> None:
        backend = InMemoryDLQBackend()
        e = DLQEntry(id="e1", workflow_name="wf", task_id="t1", node_id="n1", enqueued_at=time.time())
        await backend.enqueue(e)
        await backend.mark_replayed("e1")
        fetched = await backend.get("e1")
        assert fetched is not None
        assert fetched.replayed is True
        assert fetched.replayed_at > 0

    @pytest.mark.asyncio
    async def test_increment_attempts(self) -> None:
        backend = InMemoryDLQBackend()
        e = DLQEntry(id="e1", workflow_name="wf", task_id="t1", node_id="n1", enqueued_at=time.time())
        await backend.enqueue(e)
        await backend.increment_attempts("e1")
        await backend.increment_attempts("e1")
        fetched = await backend.get("e1")
        assert fetched is not None
        assert fetched.attempts == 2

    @pytest.mark.asyncio
    async def test_purge_removes_old_entries(self) -> None:
        backend = InMemoryDLQBackend()
        now = time.time()
        old = DLQEntry(id="old", workflow_name="wf", task_id="t1", node_id="n1", enqueued_at=now - 1000)
        recent = DLQEntry(id="recent", workflow_name="wf", task_id="t2", node_id="n2", enqueued_at=now)
        await backend.enqueue(old)
        await backend.enqueue(recent)
        # Purge entries older than 100s.
        n = await backend.purge(100.0)
        assert n == 1
        assert await backend.get("old") is None
        assert await backend.get("recent") is not None

    @pytest.mark.asyncio
    async def test_get_on_missing_returns_none(self) -> None:
        backend = InMemoryDLQBackend()
        assert await backend.get("does-not-exist") is None


class TestDeadLetterQueueFacade:
    @pytest.mark.asyncio
    async def test_enqueue_creates_entry_with_uuid(self) -> None:
        dlq = DeadLetterQueue()
        entry = await dlq.enqueue(
            workflow_name="wf",
            task_id="t1",
            node_id="n1",
            task_name="task",
            params={"k": "v"},
            error="boom",
        )
        assert entry.id  # UUID string
        assert entry.workflow_name == "wf"
        assert entry.task_id == "t1"
        assert entry.error == "boom"
        assert entry.params == {"k": "v"}
        assert entry.replayed is False

    @pytest.mark.asyncio
    async def test_list_returns_enqueued_entries(self) -> None:
        dlq = DeadLetterQueue()
        await dlq.enqueue(workflow_name="wf", task_id="t1", node_id="n1", error="e1")
        await dlq.enqueue(workflow_name="wf", task_id="t2", node_id="n2", error="e2")
        entries = await dlq.list()
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_replay_calls_registered_handler(self) -> None:
        dlq = DeadLetterQueue()
        entry = await dlq.enqueue(
            workflow_name="wf", task_id="t1", node_id="n1", params={"q": "hello"}
        )
        captured: dict[str, Any] = {}

        async def handler(e: DLQEntry) -> str:
            captured["entry"] = e
            return "replayed-ok"

        dlq.register_replay_handler("wf", handler)
        result = await dlq.replay(entry.id)
        assert result == "replayed-ok"
        assert captured["entry"].id == entry.id

        # Entry is marked as replayed.
        fetched = await dlq.get(entry.id)
        assert fetched is not None
        assert fetched.replayed is True
        assert fetched.attempts == 1

    @pytest.mark.asyncio
    async def test_replay_unknown_entry_raises(self) -> None:
        dlq = DeadLetterQueue()
        with pytest.raises(DeadLetterQueueError, match="not found"):
            await dlq.replay("nonexistent")

    @pytest.mark.asyncio
    async def test_replay_without_handler_raises(self) -> None:
        dlq = DeadLetterQueue()
        entry = await dlq.enqueue(
            workflow_name="wf_without_handler", task_id="t1", node_id="n1"
        )
        with pytest.raises(DeadLetterQueueError, match="No replay handler"):
            await dlq.replay(entry.id)

    @pytest.mark.asyncio
    async def test_replay_failure_increments_attempts_and_reraises(self) -> None:
        dlq = DeadLetterQueue()
        entry = await dlq.enqueue(workflow_name="wf", task_id="t1", node_id="n1")

        async def handler(e: DLQEntry) -> None:
            raise RuntimeError("replay failed")

        dlq.register_replay_handler("wf", handler)
        with pytest.raises(RuntimeError, match="replay failed"):
            await dlq.replay(entry.id)

        fetched = await dlq.get(entry.id)
        assert fetched is not None
        assert fetched.attempts == 1
        # Not marked as replayed since the handler raised.
        assert fetched.replayed is False

    @pytest.mark.asyncio
    async def test_purge_uses_default_ttl(self) -> None:
        dlq = DeadLetterQueue(default_ttl_seconds=0.0)
        # Default TTL = 0 means everything is older than 0 seconds and
        # will be purged.
        await dlq.enqueue(workflow_name="wf", task_id="t1", node_id="n1")
        await dlq.enqueue(workflow_name="wf", task_id="t2", node_id="n2")
        n = await dlq.purge()
        assert n == 2
        assert await dlq.list() == []

    @pytest.mark.asyncio
    async def test_to_dict_and_from_dict_roundtrip(self) -> None:
        entry = DLQEntry(
            id="e1",
            workflow_name="wf",
            task_id="t1",
            node_id="n1",
            task_name="task",
            params={"a": 1},
            error="boom",
            enqueued_at=12345.0,
            replayed=True,
            replayed_at=13000.0,
            attempts=2,
            metadata={"source": "test"},
        )
        d = entry.to_dict()
        restored = DLQEntry.from_dict(d)
        assert restored.id == entry.id
        assert restored.workflow_name == entry.workflow_name
        assert restored.params == entry.params
        assert restored.metadata == entry.metadata
        assert restored.replayed is True
        assert restored.attempts == 2


class TestPostgresDLQBackendFallback:
    """Verify the Postgres backend gracefully falls back to memory when
    asyncpg is unavailable or the DSN is unreachable."""

    @pytest.mark.asyncio
    async def test_falls_back_to_memory_when_asyncpg_missing(self) -> None:
        # Force ImportError on `import asyncpg` to simulate absent driver.
        backend = PostgresDLQBackend(dsn="postgres://nobody@localhost/x")
        with patch("builtins.__import__", side_effect=ImportError("no asyncpg")):
            # _get_pool will return None and we'll fall back to memory.
            entry = DLQEntry(
                id="e1",
                workflow_name="wf",
                task_id="t1",
                node_id="n1",
                enqueued_at=time.time(),
            )
            await backend.enqueue(entry)
            fetched = await backend.get("e1")
        assert fetched is not None
        assert fetched.id == "e1"
        await backend.close()

    @pytest.mark.asyncio
    async def test_list_uses_shadow_memory(self) -> None:
        backend = PostgresDLQBackend(dsn="postgres://nobody@localhost/x")
        await backend.enqueue(
            DLQEntry(
                id="e1",
                workflow_name="wf",
                task_id="t1",
                node_id="n1",
                enqueued_at=time.time(),
            )
        )
        entries = await backend.list_entries()
        assert len(entries) == 1
        assert entries[0].id == "e1"
        await backend.close()


# ===========================================================================
# SagaWorkflow
# ===========================================================================

class TestSagaWorkflow:
    @pytest.mark.asyncio
    async def test_saga_success_path(self) -> None:
        """All three steps succeed; no compensation runs."""
        from icore.workflows.examples.order_processing import (
            OrderProcessingSaga,
            reset_state,
            _STATE,
        )

        reset_state()
        saga = OrderProcessingSaga()
        ctx = TaskContext(task_id="t1", workflow_id="w1")

        result = await saga.execute(
            ctx,
            {
                "order_id": "ord-1",
                "quantity": 2,
                "amount": 99.5,
            },
        )

        assert result.is_success
        assert _STATE["inventory_reservations"] == {"ord-1": 2}
        assert _STATE["charges"] == {"ord-1": 99.5}
        assert _STATE["notifications_sent"] == ["ord-1"]

    @pytest.mark.asyncio
    async def test_saga_failure_triggers_compensation(self) -> None:
        """charge_payment fails -> reserve_inventory is compensated."""
        from icore.workflows.examples.order_processing import (
            OrderProcessingSaga,
            reset_state,
            _STATE,
        )

        reset_state()
        saga = OrderProcessingSaga()
        ctx = TaskContext(task_id="t1", workflow_id="w1")

        # Input builder for charge_payment pulls `fail` from params.
        result = await saga.execute(
            ctx,
            {
                "order_id": "ord-2",
                "quantity": 1,
                "amount": 50.0,
                "fail": True,
            },
        )

        # The saga returns the failed step's output (status="error").
        assert not result.is_success
        assert "Payment charge failed" in (result.error or "")

        # Compensation: inventory reservation must have been released.
        assert "ord-2" not in _STATE["inventory_reservations"]
        # Charge was never recorded (step 2 failed).
        assert "ord-2" not in _STATE["charges"]
        # Step 3 never ran.
        assert "ord-2" not in _STATE["notifications_sent"]

    @pytest.mark.asyncio
    async def test_saga_no_steps_returns_failure(self) -> None:
        class EmptySaga(SagaWorkflow):
            name = "empty_saga"

            def define(self):  # type: ignore[override]
                # Don't add any steps; just return an empty DAG path.
                return self.build_dag()  # Will raise ValueError.

        saga = EmptySaga()
        ctx = TaskContext(task_id="t1", workflow_id="w1")
        # build_dag raises because no steps registered.
        with pytest.raises(ValueError, match="no steps"):
            saga.define()

    @pytest.mark.asyncio
    async def test_saga_compensator_failure_raises_saga_error(self) -> None:
        """If a compensator itself raises, SagaCompensationError is raised."""

        async def good_step(ctx, inp):
            return BaseTaskOutput.success()

        async def failing_step(ctx, inp):
            return BaseTaskOutput.failure("step failed")

        async def bad_compensator(ctx, saga_ctx, step_output):
            raise RuntimeError("compensator boom")

        class MiniSaga(SagaWorkflow):
            name = "mini_saga"

            def define(self):  # type: ignore[override]
                self.add_saga_step(
                    node_id="step1",
                    task_name="noop_success",
                    compensate=bad_compensator,
                )
                self.add_saga_step(
                    node_id="step2",
                    task_name="noop_failure",
                )
                return self.build_dag()

        # Register tasks for the saga.
        from icore.core.base_task import BaseTask
        from icore.core.models import BaseTaskInput
        from icore.core.registry import register_task

        @register_task("noop_success")
        class NoopSuccessTask(BaseTask):
            name = "noop_success"
            input_model = BaseTaskInput

            async def prepare(self, ctx):
                pass

            async def execute(self, ctx, inp):
                return BaseTaskOutput.success(value="step1 done")

            async def cleanup(self, ctx):
                pass

        @register_task("noop_failure")
        class NoopFailureTask(BaseTask):
            name = "noop_failure"
            input_model = BaseTaskInput

            async def prepare(self, ctx):
                pass

            async def execute(self, ctx, inp):
                return BaseTaskOutput.failure("step2 failed")

            async def cleanup(self, ctx):
                pass

        saga = MiniSaga()
        ctx = TaskContext(task_id="t1", workflow_id="w1")

        # The compensation failure happens *after* the failed step's
        # output is computed. We expect SagaCompensationError to be
        # raised from _compensate_backward. The execute() method does
        # not catch it, so it propagates.
        with pytest.raises(SagaCompensationError, match="Compensation failed"):
            await saga.execute(ctx, {})

    @pytest.mark.asyncio
    async def test_saga_step_without_compensator_skipped(self) -> None:
        """Steps with compensate=None are skipped during rollback."""

        from icore.core.base_task import BaseTask
        from icore.core.models import BaseTaskInput
        from icore.core.registry import register_task

        @register_task("saga_success_task_v2")
        class SuccessTask(BaseTask):
            name = "saga_success_task_v2"
            input_model = BaseTaskInput

            async def prepare(self, ctx):
                pass

            async def execute(self, ctx, inp):
                return BaseTaskOutput.success()

            async def cleanup(self, ctx):
                pass

        @register_task("saga_failure_task_v2")
        class FailureTask(BaseTask):
            name = "saga_failure_task_v2"
            input_model = BaseTaskInput

            async def prepare(self, ctx):
                pass

            async def execute(self, ctx, inp):
                return BaseTaskOutput.failure("nope")

            async def cleanup(self, ctx):
                pass

        compensated: list[str] = []

        async def compensator(ctx, saga_ctx, step_output):
            compensated.append("ran")

        class SkipSaga(SagaWorkflow):
            name = "skip_saga"

            def define(self):  # type: ignore[override]
                # Step 1 has a compensator; step 2 has none; step 3 fails.
                self.add_saga_step(
                    node_id="s1",
                    task_name="saga_success_task_v2",
                    compensate=compensator,
                )
                self.add_saga_step(
                    node_id="s2",
                    task_name="saga_success_task_v2",
                    compensate=None,  # no compensator
                )
                self.add_saga_step(
                    node_id="s3",
                    task_name="saga_failure_task_v2",
                )
                return self.build_dag()

        saga = SkipSaga()
        ctx = TaskContext(task_id="t1", workflow_id="w1")
        result = await saga.execute(ctx, {})
        assert not result.is_success
        # Only step 1's compensator ran; step 2 has none.
        assert compensated == ["ran"]

    def test_saga_step_dataclass_defaults(self) -> None:
        step = SagaStep(node_id="n", task_name="t")
        assert step.compensate is None
        assert step.retries == 0
        assert step.timeout is None
