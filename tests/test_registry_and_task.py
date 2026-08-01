"""
Tests for icore.core.registry and icore.core.base_task.

Covers:
    - TaskRegistry register/get/list/contains/unregister/clear
    - register_task decorator + global task_registry
    - Thread-safety (RLock) - basic smoke (no thread stress)
    - BaseTask lifecycle: validate/prepare/execute/cleanup
    - BaseTaskOutput success/failure factories + is_success
    - TaskContext: set/get model_manager & db_manager; metadata accessors
"""

from __future__ import annotations

import pytest

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import TaskRegistry, register_task, task_registry
from icore.core.task_context import TaskContext, _BoundDB


# ---------------------------------------------------------------------------
# TaskRegistry
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_default_singleton(self):
        a = TaskRegistry.default()
        b = TaskRegistry.default()
        assert a is b

    def test_register_and_get(self):
        reg = TaskRegistry()
        reg.register("noop", _NoopTask)
        assert reg.contains("noop")
        assert reg.get("noop") is _NoopTask

    def test_get_unknown_raises_keyerror(self):
        reg = TaskRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("ghost")

    def test_register_requires_subclass(self):
        reg = TaskRegistry()
        with pytest.raises(TypeError, match="subclass of BaseTask"):
            reg.register("bad", object)  # type: ignore[arg-type]

    def test_register_rejects_empty_name(self):
        reg = TaskRegistry()
        with pytest.raises(ValueError, match="cannot be empty"):
            reg.register("", _NoopTask)

    def test_register_overwrites(self):
        reg = TaskRegistry()
        reg.register("t", _NoopTask)
        reg.register("t", _EchoTask)
        assert reg.get("t") is _EchoTask

    def test_unregister(self):
        reg = TaskRegistry()
        reg.register("t", _NoopTask)
        reg.unregister("t")
        assert not reg.contains("t")
        with pytest.raises(KeyError):
            reg.unregister("t")

    def test_clear(self):
        reg = TaskRegistry()
        reg.register("a", _NoopTask)
        reg.register("b", _EchoTask)
        reg.clear()
        assert len(reg) == 0

    def test_list_tasks_sorted(self):
        reg = TaskRegistry()
        reg.register("zeta", _NoopTask)
        reg.register("alpha", _EchoTask)
        assert reg.list_tasks() == ["alpha", "zeta"]

    def test_contains_dunder(self):
        reg = TaskRegistry()
        reg.register("t", _NoopTask)
        assert "t" in reg
        assert "ghost" not in reg

    def test_len_dunder(self):
        reg = TaskRegistry()
        assert len(reg) == 0
        reg.register("a", _NoopTask)
        assert len(reg) == 1


# ---------------------------------------------------------------------------
# register_task decorator
# ---------------------------------------------------------------------------

class TestRegisterTaskDecorator:
    def test_decorator_registers_and_returns_class(self):
        @register_task("decorator_test_task", registry=task_registry)
        class MyTask(_NoopTask):
            pass

        # Returned unchanged
        assert MyTask.__name__ == "MyTask"
        # Registered
        assert task_registry.contains("decorator_test_task")
        assert task_registry.get("decorator_test_task") is MyTask

    def test_decorator_sets_name_attr_if_empty(self):
        @register_task("auto_name_task", registry=task_registry)
        class MyTask(BaseTask):
            async def prepare(self, ctx):
                pass

            async def execute(self, ctx, inp):
                return BaseTaskOutput.success()

        assert MyTask.name == "auto_name_task"


# ---------------------------------------------------------------------------
# BaseTaskOutput
# ---------------------------------------------------------------------------

class TestBaseTaskOutput:
    def test_success_factory(self):
        out = BaseTaskOutput.success(a=1, b="x")
        assert out.is_success is True
        assert out.error is None
        assert out.data == {"a": 1, "b": "x"}

    def test_failure_factory(self):
        out = BaseTaskOutput.failure("boom", debug="trace")
        assert out.is_success is False
        assert out.error == "boom"
        assert out.data == {"debug": "trace"}

    def test_default_status_success(self):
        out = BaseTaskOutput()
        assert out.status == "success"
        assert out.is_success is True


# ---------------------------------------------------------------------------
# BaseTask lifecycle
# ---------------------------------------------------------------------------

class TestBaseTaskLifecycle:
    async def test_validate_default_returns_true(self):
        task = _NoopTask()
        inp = BaseTaskInput()
        assert task.validate(inp) is True

    async def test_execute_noop(self):
        task = _NoopTask()
        ctx = TaskContext(task_id="t1")
        inp = BaseTaskInput()
        out = await task.execute(ctx, inp)
        assert out.is_success is True
        assert out.data == {"ran": True}

    async def test_cleanup_default_noop(self):
        task = _NoopTask()
        ctx = TaskContext(task_id="t1")
        # Should not raise
        await task.cleanup(ctx)

    def test_repr_includes_name(self):
        task = _NoopTask()
        assert "NoopTask" in repr(task)
        assert "_noop" in repr(task)


# ---------------------------------------------------------------------------
# TaskContext
# ---------------------------------------------------------------------------

class TestTaskContext:
    def test_basic_fields(self):
        ctx = TaskContext(task_id="t1", workflow_id="w1")
        assert ctx.task_id == "t1"
        assert ctx.workflow_id == "w1"
        assert ctx.model_id is None
        assert ctx.stream is False
        assert ctx.metadata == {}

    def test_metadata_accessors(self):
        ctx = TaskContext(task_id="t1")
        ctx.set_metadata("trace_id", "abc")
        assert ctx.get_metadata("trace_id") == "abc"
        assert ctx.get_metadata("missing", "default") == "default"

    def test_get_model_adapter_without_manager_raises(self):
        ctx = TaskContext(task_id="t1")
        with pytest.raises(RuntimeError, match="No ModelManager"):
            ctx.get_model_adapter()

    def test_get_db_without_manager_raises(self):
        ctx = TaskContext(task_id="t1")
        with pytest.raises(RuntimeError, match="No DBManager"):
            ctx.get_db("main")

    def test_set_and_get_model_adapter(self, fake_model_manager):
        ctx = TaskContext(task_id="t1", model_id="fake-model")
        ctx.set_model_manager(fake_model_manager)
        adapter = ctx.get_model_adapter()
        assert adapter.model_id == "fake-model"

    def test_get_model_adapter_auto_routes_when_id_none(
        self, fake_model_manager
    ):
        ctx = TaskContext(task_id="t1")
        ctx.set_model_manager(fake_model_manager)
        adapter = ctx.get_model_adapter()
        # Router was configured to default to fake-model
        assert adapter.model_id == "fake-model"

    def test_set_and_get_db(self):
        class FakeDBManager:
            async def query(self, name, sql, params=None):
                return [{"name": name, "sql": sql}]

            async def execute(self, name, sql, params=None):
                return 1

            def connection_ctx(self, name):
                class _Ctx:
                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *args):
                        pass

                return _Ctx()

        mgr = FakeDBManager()
        ctx = TaskContext(task_id="t1")
        ctx.set_db_manager(mgr)
        db = ctx.get_db("main_db")
        assert isinstance(db, _BoundDB)
        assert db.name == "main_db"

    async def test_bound_db_query_delegates(self):
        class FakeDBManager:
            def __init__(self):
                self.calls = []

            async def query(self, name, sql, params=None):
                self.calls.append(("query", name, sql, params))
                return [{"row": 1}]

            async def execute(self, name, sql, params=None):
                self.calls.append(("execute", name, sql, params))
                return 42

            def connection_ctx(self, name):
                raise NotImplementedError

        mgr = FakeDBManager()
        bound = _BoundDB(mgr, "main")
        rows = await bound.query("SELECT 1")
        assert rows == [{"row": 1}]
        affected = await bound.execute("DELETE FROM t")
        assert affected == 42
        assert mgr.calls[0][0] == "query"
        assert mgr.calls[1][0] == "execute"

    async def test_bound_db_fetch_one(self):
        class FakeDBManager:
            async def query(self, name, sql, params=None):
                return [{"a": 1}, {"a": 2}]

            async def execute(self, name, sql, params=None):
                return 0

            def connection_ctx(self, name):
                raise NotImplementedError

        bound = _BoundDB(FakeDBManager(), "main")
        row = await bound.fetch_one("SELECT a FROM t")
        assert row == {"a": 1}

    async def test_bound_db_fetch_one_empty(self):
        class FakeDBManager:
            async def query(self, name, sql, params=None):
                return []

            async def execute(self, name, sql, params=None):
                return 0

            def connection_ctx(self, name):
                raise NotImplementedError

        bound = _BoundDB(FakeDBManager(), "main")
        assert await bound.fetch_one("SELECT a FROM t") is None

    def test_repr(self):
        ctx = TaskContext(task_id="t1", workflow_id="w1", model_id="m1")
        r = repr(ctx)
        assert "t1" in r
        assert "w1" in r
        assert "m1" in r


# ---------------------------------------------------------------------------
# Fixtures: simple test tasks
# ---------------------------------------------------------------------------

class _NoopTask(BaseTask):
    name = "_noop"
    description = "No-op task for tests"

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        return BaseTaskOutput.success(ran=True)


class _EchoTask(BaseTask):
    name = "_echo"
    description = "Echo task for tests"

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        return BaseTaskOutput.success(echo=inp.model_dump())
