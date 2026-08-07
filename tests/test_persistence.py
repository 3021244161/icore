"""
Tests for v0.6 workflow persistence module.

Covers:
    - ExecutionStatus enum + dataclass (de)serialization round-trips
    - InMemoryPersistenceBackend CRUD / filtering / pagination
    - PostgresPersistenceBackend graceful degradation (asyncpg missing /
      DSN unreachable / write no-op on DB failure)
    - WorkflowPersistenceManager full lifecycle (create → start →
      complete / fail), checkpoint save/load, resume (FAILED/PAUSED
      resumable, COMPLETED rejected), task execution recording, history
      queries.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

from icore.exceptions import WorkflowPersistenceError, WorkflowResumeError
from icore.persistence import (
    BasePersistenceBackend,
    ExecutionStatus,
    InMemoryPersistenceBackend,
    PostgresPersistenceBackend,
    TaskExecution,
    WorkflowExecution,
    WorkflowPersistenceManager,
)


# ===========================================================================
# ExecutionStatus enum
# ===========================================================================

class TestExecutionStatus:
    def test_status_values(self) -> None:
        assert ExecutionStatus.PENDING.value == "PENDING"
        assert ExecutionStatus.RUNNING.value == "RUNNING"
        assert ExecutionStatus.COMPLETED.value == "COMPLETED"
        assert ExecutionStatus.FAILED.value == "FAILED"
        assert ExecutionStatus.CANCELLED.value == "CANCELLED"
        assert ExecutionStatus.PAUSED.value == "PAUSED"

    def test_status_is_str_enum(self) -> None:
        # str Enum: each member is also a str.
        assert isinstance(ExecutionStatus.PENDING, str)
        assert ExecutionStatus.PENDING == "PENDING"

    def test_status_from_value(self) -> None:
        assert ExecutionStatus("RUNNING") is ExecutionStatus.RUNNING
        assert ExecutionStatus("PAUSED") is ExecutionStatus.PAUSED


# ===========================================================================
# WorkflowExecution dataclass
# ===========================================================================

class TestWorkflowExecutionDataclass:
    def test_defaults(self) -> None:
        e = WorkflowExecution(id="e1", task_id="t1", workflow_name="wf")
        assert e.params == {}
        assert e.status is ExecutionStatus.PENDING
        assert e.started_at is None
        assert e.finished_at is None
        assert e.error is None
        assert e.checkpoint == {}
        assert e.result is None
        assert e.metadata == {}
        assert e.created_at > 0
        assert e.updated_at > 0

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        e = WorkflowExecution(
            id="e1",
            task_id="t1",
            workflow_name="wf",
            params={"q": "hello"},
            status=ExecutionStatus.COMPLETED,
            started_at=1000.0,
            finished_at=2000.0,
            error=None,
            checkpoint={"completed_nodes": ["n1"], "current_wave": 2},
            result={"answer": "world"},
            created_at=100.0,
            updated_at=200.0,
            metadata={"source": "test"},
        )
        d = e.to_dict()
        # status serialized as string value
        assert d["status"] == "COMPLETED"
        restored = WorkflowExecution.from_dict(d)
        assert restored.id == e.id
        assert restored.task_id == e.task_id
        assert restored.workflow_name == e.workflow_name
        assert restored.params == e.params
        assert restored.status is ExecutionStatus.COMPLETED
        assert restored.started_at == e.started_at
        assert restored.finished_at == e.finished_at
        assert restored.checkpoint == e.checkpoint
        assert restored.result == e.result
        assert restored.created_at == e.created_at
        assert restored.updated_at == e.updated_at
        assert restored.metadata == e.metadata

    def test_from_dict_handles_missing_optional_fields(self) -> None:
        restored = WorkflowExecution.from_dict({
            "id": "e1",
            "task_id": "t1",
            "workflow_name": "wf",
        })
        assert restored.params == {}
        assert restored.status is ExecutionStatus.PENDING
        assert restored.checkpoint == {}
        assert restored.result is None
        assert restored.metadata == {}

    def test_from_dict_handles_none_json_fields(self) -> None:
        # DB rows may yield None for default-jsonb columns in edge cases.
        restored = WorkflowExecution.from_dict({
            "id": "e1",
            "task_id": "t1",
            "workflow_name": "wf",
            "params": None,
            "checkpoint": None,
            "metadata": None,
        })
        assert restored.params == {}
        assert restored.checkpoint == {}
        assert restored.metadata == {}

    def test_independent_default_instances(self) -> None:
        # Mutable defaults must not be shared across instances.
        a = WorkflowExecution(id="a", task_id="t", workflow_name="w")
        b = WorkflowExecution(id="b", task_id="t", workflow_name="w")
        a.params["x"] = 1
        a.checkpoint["y"] = 2
        assert b.params == {}
        assert b.checkpoint == {}


# ===========================================================================
# TaskExecution dataclass
# ===========================================================================

class TestTaskExecutionDataclass:
    def test_defaults(self) -> None:
        t = TaskExecution(
            id="te1", execution_id="e1", node_id="n1", task_name="task"
        )
        assert t.status is ExecutionStatus.PENDING
        assert t.input == {}
        assert t.output is None
        assert t.retry_count == 0
        assert t.error is None
        assert t.started_at is None
        assert t.finished_at is None

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        t = TaskExecution(
            id="te1",
            execution_id="e1",
            node_id="n1",
            task_name="task",
            status=ExecutionStatus.FAILED,
            input={"x": 1},
            output=None,
            started_at=10.0,
            finished_at=20.0,
            retry_count=3,
            error="boom",
        )
        d = t.to_dict()
        assert d["status"] == "FAILED"
        restored = TaskExecution.from_dict(d)
        assert restored.id == t.id
        assert restored.execution_id == t.execution_id
        assert restored.node_id == t.node_id
        assert restored.task_name == t.task_name
        assert restored.status is ExecutionStatus.FAILED
        assert restored.input == t.input
        assert restored.output is None
        assert restored.retry_count == 3
        assert restored.error == "boom"

    def test_from_dict_missing_status_defaults_to_pending(self) -> None:
        restored = TaskExecution.from_dict({
            "id": "te1",
            "execution_id": "e1",
            "node_id": "n1",
            "task_name": "task",
        })
        assert restored.status is ExecutionStatus.PENDING
        assert restored.retry_count == 0


# ===========================================================================
# InMemoryPersistenceBackend
# ===========================================================================

class TestInMemoryPersistenceBackend:
    @pytest.mark.asyncio
    async def test_save_and_get_execution(self) -> None:
        backend = InMemoryPersistenceBackend()
        e = WorkflowExecution(id="e1", task_id="t1", workflow_name="wf")
        await backend.save_execution(e)
        fetched = await backend.get_execution("e1")
        assert fetched is not None
        assert fetched.id == "e1"
        assert fetched.workflow_name == "wf"

    @pytest.mark.asyncio
    async def test_get_execution_missing_returns_none(self) -> None:
        backend = InMemoryPersistenceBackend()
        assert await backend.get_execution("does-not-exist") is None

    @pytest.mark.asyncio
    async def test_save_execution_upserts(self) -> None:
        backend = InMemoryPersistenceBackend()
        e = WorkflowExecution(id="e1", task_id="t1", workflow_name="wf")
        await backend.save_execution(e)
        e.status = ExecutionStatus.RUNNING
        await backend.save_execution(e)
        fetched = await backend.get_execution("e1")
        assert fetched is not None
        assert fetched.status is ExecutionStatus.RUNNING

    @pytest.mark.asyncio
    async def test_update_execution_status(self) -> None:
        backend = InMemoryPersistenceBackend()
        e = WorkflowExecution(id="e1", task_id="t1", workflow_name="wf")
        await backend.save_execution(e)
        old_updated = e.updated_at
        await asyncio.sleep(0.001)
        await backend.update_execution_status(
            "e1",
            ExecutionStatus.FAILED,
            error="boom",
            result={"partial": True},
            checkpoint={"completed_nodes": ["n1"]},
        )
        fetched = await backend.get_execution("e1")
        assert fetched is not None
        assert fetched.status is ExecutionStatus.FAILED
        assert fetched.error == "boom"
        assert fetched.result == {"partial": True}
        assert fetched.checkpoint == {"completed_nodes": ["n1"]}
        assert fetched.updated_at > old_updated

    @pytest.mark.asyncio
    async def test_update_execution_status_missing_is_noop(self) -> None:
        backend = InMemoryPersistenceBackend()
        # Should not raise.
        await backend.update_execution_status(
            "missing", ExecutionStatus.RUNNING
        )

    @pytest.mark.asyncio
    async def test_update_execution_status_partial_only_sets_provided(self) -> None:
        backend = InMemoryPersistenceBackend()
        e = WorkflowExecution(
            id="e1",
            task_id="t1",
            workflow_name="wf",
            error="pre-existing",
            result={"r": 1},
        )
        await backend.save_execution(e)
        # Only update checkpoint; error and result should be untouched.
        await backend.update_execution_status(
            "e1", ExecutionStatus.RUNNING, checkpoint={"wave": 1}
        )
        fetched = await backend.get_execution("e1")
        assert fetched is not None
        assert fetched.checkpoint == {"wave": 1}
        assert fetched.error == "pre-existing"
        assert fetched.result == {"r": 1}

    @pytest.mark.asyncio
    async def test_list_executions_filter_by_workflow_name(self) -> None:
        backend = InMemoryPersistenceBackend()
        e1 = WorkflowExecution(id="e1", task_id="t1", workflow_name="wf_a", created_at=1.0)
        e2 = WorkflowExecution(id="e2", task_id="t2", workflow_name="wf_b", created_at=2.0)
        e3 = WorkflowExecution(id="e3", task_id="t3", workflow_name="wf_a", created_at=3.0)
        for e in (e1, e2, e3):
            await backend.save_execution(e)
        result = await backend.list_executions(workflow_name="wf_a")
        assert {e.id for e in result} == {"e1", "e3"}

    @pytest.mark.asyncio
    async def test_list_executions_filter_by_status(self) -> None:
        backend = InMemoryPersistenceBackend()
        e1 = WorkflowExecution(id="e1", task_id="t1", workflow_name="wf", created_at=1.0)
        e1.status = ExecutionStatus.COMPLETED
        e2 = WorkflowExecution(id="e2", task_id="t2", workflow_name="wf", created_at=2.0)
        e2.status = ExecutionStatus.FAILED
        e3 = WorkflowExecution(id="e3", task_id="t3", workflow_name="wf", created_at=3.0)
        e3.status = ExecutionStatus.COMPLETED
        for e in (e1, e2, e3):
            await backend.save_execution(e)
        result = await backend.list_executions(status=ExecutionStatus.COMPLETED)
        assert {e.id for e in result} == {"e1", "e3"}

    @pytest.mark.asyncio
    async def test_list_executions_filter_by_time_range(self) -> None:
        backend = InMemoryPersistenceBackend()
        e1 = WorkflowExecution(id="e1", task_id="t1", workflow_name="wf", created_at=10.0)
        e2 = WorkflowExecution(id="e2", task_id="t2", workflow_name="wf", created_at=20.0)
        e3 = WorkflowExecution(id="e3", task_id="t3", workflow_name="wf", created_at=30.0)
        for e in (e1, e2, e3):
            await backend.save_execution(e)
        # since=15, until=25 -> only e2
        result = await backend.list_executions(since=15.0, until=25.0)
        assert [e.id for e in result] == ["e2"]
        # since=20 inclusive -> e2, e3
        result = await backend.list_executions(since=20.0)
        assert {e.id for e in result} == {"e2", "e3"}
        # until=20 inclusive -> e1, e2
        result = await backend.list_executions(until=20.0)
        assert {e.id for e in result} == {"e1", "e2"}

    @pytest.mark.asyncio
    async def test_list_executions_pagination(self) -> None:
        backend = InMemoryPersistenceBackend()
        for i in range(5):
            await backend.save_execution(
                WorkflowExecution(
                    id=f"e{i}", task_id=f"t{i}", workflow_name="wf",
                    created_at=float(i),
                )
            )
        # limit + offset
        page1 = await backend.list_executions(limit=2, offset=0)
        page2 = await backend.list_executions(limit=2, offset=2)
        page3 = await backend.list_executions(limit=2, offset=4)
        assert [e.id for e in page1] == ["e0", "e1"]
        assert [e.id for e in page2] == ["e2", "e3"]
        assert [e.id for e in page3] == ["e4"]
        # offset beyond range -> empty
        assert await backend.list_executions(limit=10, offset=100) == []

    @pytest.mark.asyncio
    async def test_list_executions_sorted_oldest_first(self) -> None:
        backend = InMemoryPersistenceBackend()
        # Insert out of order.
        for i in [3, 1, 2, 0]:
            await backend.save_execution(
                WorkflowExecution(
                    id=f"e{i}", task_id="t", workflow_name="wf",
                    created_at=float(i),
                )
            )
        result = await backend.list_executions()
        assert [e.id for e in result] == ["e0", "e1", "e2", "e3"]

    @pytest.mark.asyncio
    async def test_delete_execution_returns_true_and_removes(self) -> None:
        backend = InMemoryPersistenceBackend()
        e = WorkflowExecution(id="e1", task_id="t1", workflow_name="wf")
        await backend.save_execution(e)
        assert await backend.delete_execution("e1") is True
        assert await backend.get_execution("e1") is None

    @pytest.mark.asyncio
    async def test_delete_execution_missing_returns_false(self) -> None:
        backend = InMemoryPersistenceBackend()
        assert await backend.delete_execution("missing") is False

    @pytest.mark.asyncio
    async def test_delete_execution_cascades_to_task_executions(self) -> None:
        backend = InMemoryPersistenceBackend()
        await backend.save_execution(
            WorkflowExecution(id="e1", task_id="t1", workflow_name="wf")
        )
        await backend.save_task_execution(
            TaskExecution(id="te1", execution_id="e1", node_id="n1", task_name="task")
        )
        await backend.delete_execution("e1")
        tasks = await backend.get_task_executions("e1")
        assert tasks == []

    @pytest.mark.asyncio
    async def test_save_and_get_task_executions(self) -> None:
        backend = InMemoryPersistenceBackend()
        await backend.save_execution(
            WorkflowExecution(id="e1", task_id="t1", workflow_name="wf")
        )
        t1 = TaskExecution(id="te1", execution_id="e1", node_id="n1", task_name="a")
        t2 = TaskExecution(id="te2", execution_id="e1", node_id="n2", task_name="b")
        await backend.save_task_execution(t1)
        await backend.save_task_execution(t2)
        tasks = await backend.get_task_executions("e1")
        assert {t.id for t in tasks} == {"te1", "te2"}

    @pytest.mark.asyncio
    async def test_save_task_execution_upserts(self) -> None:
        backend = InMemoryPersistenceBackend()
        t = TaskExecution(id="te1", execution_id="e1", node_id="n1", task_name="a")
        await backend.save_task_execution(t)
        t.status = ExecutionStatus.COMPLETED
        await backend.save_task_execution(t)
        tasks = await backend.get_task_executions("e1")
        assert len(tasks) == 1
        assert tasks[0].status is ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_task_executions_missing_returns_empty(self) -> None:
        backend = InMemoryPersistenceBackend()
        assert await backend.get_task_executions("missing") == []


# ===========================================================================
# PostgresPersistenceBackend - graceful degradation
# ===========================================================================

class TestPostgresPersistenceBackendFallback:
    """Verify the Postgres backend falls back to memory when asyncpg is
    unavailable or the DSN is unreachable. No real DB is required."""

    @pytest.mark.asyncio
    async def test_falls_back_to_memory_when_asyncpg_missing(self) -> None:
        backend = PostgresPersistenceBackend(dsn="postgres://nobody@localhost/x")
        with patch("builtins.__import__", side_effect=ImportError("no asyncpg")):
            # _get_pool will return None and we fall back to memory.
            e = WorkflowExecution(
                id="e1", task_id="t1", workflow_name="wf", created_at=1.0
            )
            await backend.save_execution(e)
            fetched = await backend.get_execution("e1")
        assert fetched is not None
        assert fetched.id == "e1"
        await backend.close()

    @pytest.mark.asyncio
    async def test_dsn_unreachable_falls_back_to_memory(self) -> None:
        # Without patching, asyncpg may or may not be installed. If it's
        # installed, create_pool will fail to connect (unreachable DSN)
        # and fall back. If not installed, ImportError path is taken.
        # Either way, the backend must not raise and must serve reads
        # from the in-memory shadow.
        backend = PostgresPersistenceBackend(
            dsn="postgres://nobody:nopass@127.0.0.1:1/none"
        )
        e = WorkflowExecution(
            id="e1", task_id="t1", workflow_name="wf", created_at=1.0
        )
        # save_execution must not raise even if DB is unreachable.
        await backend.save_execution(e)
        fetched = await backend.get_execution("e1")
        assert fetched is not None
        assert fetched.id == "e1"
        await backend.close()

    @pytest.mark.asyncio
    async def test_write_operations_are_noop_on_db_failure(self) -> None:
        # All write ops should succeed (via shadow) without raising.
        backend = PostgresPersistenceBackend(
            dsn="postgres://nobody:nopass@127.0.0.1:1/none"
        )
        e = WorkflowExecution(
            id="e1", task_id="t1", workflow_name="wf", created_at=1.0
        )
        await backend.save_execution(e)
        # update_execution_status should not raise.
        await backend.update_execution_status(
            "e1", ExecutionStatus.RUNNING, checkpoint={"wave": 1}
        )
        # save_task_execution should not raise.
        await backend.save_task_execution(
            TaskExecution(id="te1", execution_id="e1", node_id="n1", task_name="a")
        )
        # delete should not raise.
        assert await backend.delete_execution("e1") is True
        await backend.close()

    @pytest.mark.asyncio
    async def test_list_serves_from_shadow(self) -> None:
        backend = PostgresPersistenceBackend(
            dsn="postgres://nobody:nopass@127.0.0.1:1/none"
        )
        await backend.save_execution(
            WorkflowExecution(
                id="e1", task_id="t1", workflow_name="wf", created_at=1.0
            )
        )
        await backend.save_execution(
            WorkflowExecution(
                id="e2", task_id="t2", workflow_name="wf", created_at=2.0
            )
        )
        result = await backend.list_executions()
        assert {e.id for e in result} == {"e1", "e2"}
        await backend.close()

    @pytest.mark.asyncio
    async def test_get_execution_missing_returns_none_when_db_down(self) -> None:
        backend = PostgresPersistenceBackend(
            dsn="postgres://nobody:nopass@127.0.0.1:1/none"
        )
        # Not in shadow, DB unreachable -> None.
        assert await backend.get_execution("never-saved") is None
        await backend.close()

    @pytest.mark.asyncio
    async def test_task_executions_served_from_shadow(self) -> None:
        backend = PostgresPersistenceBackend(
            dsn="postgres://nobody:nopass@127.0.0.1:1/none"
        )
        await backend.save_execution(
            WorkflowExecution(
                id="e1", task_id="t1", workflow_name="wf", created_at=1.0
            )
        )
        await backend.save_task_execution(
            TaskExecution(id="te1", execution_id="e1", node_id="n1", task_name="a")
        )
        tasks = await backend.get_task_executions("e1")
        assert len(tasks) == 1
        assert tasks[0].id == "te1"
        await backend.close()

    @pytest.mark.asyncio
    async def test_close_with_no_pool_is_safe(self) -> None:
        backend = PostgresPersistenceBackend(
            dsn="postgres://nobody:nopass@127.0.0.1:1/none"
        )
        # close() before any pool creation must not raise.
        await backend.close()


# ===========================================================================
# WorkflowPersistenceManager
# ===========================================================================

class TestWorkflowPersistenceManager:
    @pytest.mark.asyncio
    async def test_create_execution_returns_pending_with_uuid(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {"q": "hello"})
        assert e.id  # UUID string
        assert e.task_id == "t1"
        assert e.workflow_name == "wf"
        assert e.params == {"q": "hello"}
        assert e.status is ExecutionStatus.PENDING
        assert e.started_at is None
        # Persisted: fetchable.
        fetched = await mgr.get_execution(e.id)
        assert fetched is not None
        assert fetched.id == e.id

    @pytest.mark.asyncio
    async def test_start_execution_sets_running_and_started_at(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        await mgr.start_execution(e.id)
        fetched = await mgr.get_execution(e.id)
        assert fetched is not None
        assert fetched.status is ExecutionStatus.RUNNING
        assert fetched.started_at is not None
        assert fetched.started_at > 0

    @pytest.mark.asyncio
    async def test_complete_execution_full_flow(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        await mgr.start_execution(e.id)
        await mgr.complete_execution(e.id, {"answer": "42"})
        fetched = await mgr.get_execution(e.id)
        assert fetched is not None
        assert fetched.status is ExecutionStatus.COMPLETED
        assert fetched.finished_at is not None
        assert fetched.finished_at >= (fetched.started_at or 0)
        assert fetched.result == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_fail_execution_flow(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        await mgr.start_execution(e.id)
        await mgr.fail_execution(e.id, "model timeout")
        fetched = await mgr.get_execution(e.id)
        assert fetched is not None
        assert fetched.status is ExecutionStatus.FAILED
        assert fetched.finished_at is not None
        assert fetched.error == "model timeout"

    @pytest.mark.asyncio
    async def test_start_execution_missing_raises_persistence_error(self) -> None:
        mgr = WorkflowPersistenceManager()
        with pytest.raises(WorkflowPersistenceError, match="not found"):
            await mgr.start_execution("missing")

    @pytest.mark.asyncio
    async def test_complete_execution_missing_raises_persistence_error(self) -> None:
        mgr = WorkflowPersistenceManager()
        with pytest.raises(WorkflowPersistenceError, match="not found"):
            await mgr.complete_execution("missing", {})

    @pytest.mark.asyncio
    async def test_fail_execution_missing_raises_persistence_error(self) -> None:
        mgr = WorkflowPersistenceManager()
        with pytest.raises(WorkflowPersistenceError, match="not found"):
            await mgr.fail_execution("missing", "boom")

    @pytest.mark.asyncio
    async def test_save_and_get_checkpoint(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        checkpoint = {
            "completed_nodes": ["n1", "n2"],
            "current_wave": 2,
            "node_outputs": {"n1": {"v": 1}},
        }
        await mgr.save_checkpoint(e.id, checkpoint)
        loaded = await mgr.get_checkpoint(e.id)
        assert loaded == checkpoint
        # Mutating the returned dict must not affect stored checkpoint.
        loaded["completed_nodes"].append("n3")
        loaded2 = await mgr.get_checkpoint(e.id)
        assert loaded2 == checkpoint

    @pytest.mark.asyncio
    async def test_get_checkpoint_missing_returns_none(self) -> None:
        mgr = WorkflowPersistenceManager()
        assert await mgr.get_checkpoint("missing") is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_missing_raises_persistence_error(self) -> None:
        mgr = WorkflowPersistenceManager()
        with pytest.raises(WorkflowPersistenceError, match="not found"):
            await mgr.save_checkpoint("missing", {"wave": 1})

    @pytest.mark.asyncio
    async def test_resume_execution_from_failed(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        await mgr.start_execution(e.id)
        await mgr.save_checkpoint(e.id, {"completed_nodes": ["n1"]})
        await mgr.fail_execution(e.id, "boom")
        resumed = await mgr.resume_execution(e.id)
        assert resumed.id == e.id
        assert resumed.status is ExecutionStatus.PENDING
        assert resumed.error is None
        # Checkpoint preserved across resume.
        assert resumed.checkpoint == {"completed_nodes": ["n1"]}

    @pytest.mark.asyncio
    async def test_resume_execution_from_paused(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        # Manually mark as PAUSED via backend to simulate the intermediate state.
        e_obj = await mgr.get_execution(e.id)
        assert e_obj is not None
        e_obj.status = ExecutionStatus.PAUSED
        await mgr._backend.save_execution(e_obj)
        resumed = await mgr.resume_execution(e.id)
        assert resumed.status is ExecutionStatus.PENDING

    @pytest.mark.asyncio
    async def test_resume_completed_raises_resume_error(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        await mgr.start_execution(e.id)
        await mgr.complete_execution(e.id, {"r": 1})
        with pytest.raises(WorkflowResumeError, match="Cannot resume"):
            await mgr.resume_execution(e.id)

    @pytest.mark.asyncio
    async def test_resume_pending_raises_resume_error(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        with pytest.raises(WorkflowResumeError, match="Cannot resume"):
            await mgr.resume_execution(e.id)

    @pytest.mark.asyncio
    async def test_resume_missing_raises_resume_error(self) -> None:
        mgr = WorkflowPersistenceManager()
        with pytest.raises(WorkflowResumeError, match="not found"):
            await mgr.resume_execution("missing")

    @pytest.mark.asyncio
    async def test_record_task_execution(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        te = await mgr.record_task_execution(
            execution_id=e.id,
            node_id="n1",
            task_name="generator",
            status=ExecutionStatus.COMPLETED,
            input={"q": "hello"},
            output={"a": "world"},
        )
        assert te.id  # UUID
        assert te.execution_id == e.id
        assert te.node_id == "n1"
        assert te.task_name == "generator"
        assert te.status is ExecutionStatus.COMPLETED
        assert te.input == {"q": "hello"}
        assert te.output == {"a": "world"}
        assert te.started_at is not None
        assert te.finished_at is not None
        # Persisted.
        tasks = await mgr.get_task_executions(e.id)
        assert len(tasks) == 1
        assert tasks[0].id == te.id

    @pytest.mark.asyncio
    async def test_record_task_execution_with_error_and_retry(self) -> None:
        mgr = WorkflowPersistenceManager()
        e = await mgr.create_execution("t1", "wf", {})
        te = await mgr.record_task_execution(
            execution_id=e.id,
            node_id="n1",
            task_name="generator",
            status=ExecutionStatus.FAILED,
            error="timeout",
            retry_count=2,
        )
        assert te.status is ExecutionStatus.FAILED
        assert te.error == "timeout"
        assert te.retry_count == 2
        assert te.finished_at is not None

    @pytest.mark.asyncio
    async def test_get_task_executions_missing_returns_empty(self) -> None:
        mgr = WorkflowPersistenceManager()
        assert await mgr.get_task_executions("missing") == []

    @pytest.mark.asyncio
    async def test_get_history_filter_by_workflow(self) -> None:
        mgr = WorkflowPersistenceManager()
        e1 = await mgr.create_execution("t1", "wf_a", {})
        await asyncio.sleep(0.001)
        e2 = await mgr.create_execution("t2", "wf_b", {})
        result = await mgr.get_history(workflow_name="wf_a")
        assert {e.id for e in result} == {e1.id}

    @pytest.mark.asyncio
    async def test_get_history_filter_by_status(self) -> None:
        mgr = WorkflowPersistenceManager()
        e1 = await mgr.create_execution("t1", "wf", {})
        e2 = await mgr.create_execution("t2", "wf", {})
        await mgr.start_execution(e1.id)
        await mgr.complete_execution(e1.id, {})
        result = await mgr.get_history(status=ExecutionStatus.COMPLETED)
        assert {e.id for e in result} == {e1.id}
        result = await mgr.get_history(status=ExecutionStatus.PENDING)
        assert {e.id for e in result} == {e2.id}

    @pytest.mark.asyncio
    async def test_get_history_time_range(self) -> None:
        mgr = WorkflowPersistenceManager()
        e1 = await mgr.create_execution("t1", "wf", {})
        t1 = e1.created_at
        await asyncio.sleep(0.01)
        e2 = await mgr.create_execution("t2", "wf", {})
        t2 = e2.created_at
        # since=t2 -> only e2
        result = await mgr.get_history(since=t2)
        assert {e.id for e in result} == {e2.id}
        # until=t1 -> only e1
        result = await mgr.get_history(until=t1)
        assert {e.id for e in result} == {e1.id}

    @pytest.mark.asyncio
    async def test_get_history_pagination(self) -> None:
        mgr = WorkflowPersistenceManager()
        ids = []
        for i in range(5):
            e = await mgr.create_execution(f"t{i}", "wf", {})
            ids.append(e.id)
        page1 = await mgr.get_history(limit=2, offset=0)
        page2 = await mgr.get_history(limit=2, offset=2)
        page3 = await mgr.get_history(limit=2, offset=4)
        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1
        # No overlap across pages.
        all_ids = {e.id for e in (page1 + page2 + page3)}
        assert all_ids == set(ids)

    @pytest.mark.asyncio
    async def test_get_execution_missing_returns_none(self) -> None:
        mgr = WorkflowPersistenceManager()
        assert await mgr.get_execution("missing") is None

    @pytest.mark.asyncio
    async def test_default_backend_is_in_memory(self) -> None:
        mgr = WorkflowPersistenceManager()
        assert isinstance(mgr.backend, InMemoryPersistenceBackend)

    @pytest.mark.asyncio
    async def test_accepts_custom_backend(self) -> None:
        backend = InMemoryPersistenceBackend()
        mgr = WorkflowPersistenceManager(backend=backend)
        assert mgr.backend is backend
        e = await mgr.create_execution("t1", "wf", {})
        # Stored in the custom backend.
        assert await backend.get_execution(e.id) is not None

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self) -> None:
        mgr = WorkflowPersistenceManager()
        await mgr.close()


# ===========================================================================
# BasePersistenceBackend is abstract
# ===========================================================================

class TestBasePersistenceBackendAbstract:
    def test_cannot_instantiate_abstract_backend(self) -> None:
        with pytest.raises(TypeError):
            BasePersistenceBackend()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self) -> None:
        # A partial implementation cannot be instantiated.
        class Partial(BasePersistenceBackend):
            async def save_execution(self, exec):
                pass

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]
