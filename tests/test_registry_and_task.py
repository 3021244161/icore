"""
Tests for icore.core.registry and icore.core.base_task.

Covers:
    - TaskRegistry register/get/list/contains/unregister/clear
    - register_task decorator + global task_registry
    - Thread-safety (RLock) - basic smoke (no thread stress)
    - BaseTask lifecycle: validate/prepare/execute/cleanup
    - BaseTaskOutput success/failure factories + is_success
    - TaskContext: set/get model_manager & db_manager; metadata accessors
    - BaseTaskInput enum contract (ICORE-ISSUE-001): default value
      coercion, strict_enums opt-in, extra/validate_assignment semantics
"""

from __future__ import annotations

import enum
import json

import pytest
from pydantic import ValidationError

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


# ---------------------------------------------------------------------------
# BaseTaskInput enum contract (ICORE-ISSUE-001)
# ---------------------------------------------------------------------------

class _Policy(str, enum.Enum):
    A = "a"
    B = "b"


class _Color(enum.Enum):  # 纯 Enum（非 str/int 基类）
    RED = 1
    GREEN = 2


class _PolicyInput(BaseTaskInput):
    """默认契约：use_enum_values=True，枚举降级为 .value。"""

    policy: _Policy = _Policy.A


class _ColorInput(BaseTaskInput):
    color: _Color = _Color.RED


class _StrictPolicyInput(BaseTaskInput, strict_enums=True):
    """opt-in 强类型契约：保留枚举成员。"""

    policy: _Policy = _Policy.A


class TestTaskInputEnumContract:
    """ICORE-ISSUE-001：枚举字段在任务输入中的类型契约（回归守护）。"""

    def test_default_coerces_enum_to_value(self):
        # 默认契约固化：枚举字段存储 .value（str），不是枚举成员。
        inp = _PolicyInput(policy=_Policy.B)
        assert type(inp.policy) is str
        # is 陷阱：静默失效（zGo 事故根因）
        assert inp.policy is not _Policy.B
        # 值比较仍成立
        assert inp.policy == _Policy.B

    def test_default_pure_enum_is_json_safe(self):
        # 默认值的存在理由：纯 Enum 输入降级后可被 json.dumps
        # （persistence 层 json.dumps(task_exec.input) 无 default=str 兜底）。
        inp = _ColorInput(color=_Color.GREEN)
        assert json.dumps({"color": inp.color}) == '{"color": 2}'

    def test_strict_enums_preserves_member(self):
        # opt-in 开关：保留枚举成员，is 比较成立。
        inp = _StrictPolicyInput(policy=_Policy.B)
        assert inp.policy is _Policy.B

    def test_strict_enums_does_not_leak_to_base_or_siblings(self):
        # 开关只作用于声明类：基类与兄弟类保持默认降级行为。
        assert BaseTaskInput.model_config["use_enum_values"] is True
        assert _PolicyInput(policy=_Policy.B).policy is not _Policy.B

    def test_strict_enums_dict_construction_yields_member(self):
        # 引擎路径：从 dict 构造输入（executor / agent 均如此），
        # strict 模式下校验把 str 还原为枚举成员。
        inp = _StrictPolicyInput.model_validate({"policy": "b"})
        assert inp.policy is _Policy.B

    def test_strict_enums_roundtrip_model_dump(self):
        inp = _StrictPolicyInput(policy=_Policy.B)
        restored = _StrictPolicyInput.model_validate(inp.model_dump())
        assert restored.policy is _Policy.B

    def test_strict_enums_assignment_validates(self):
        # validate_assignment 语义：构造后赋值重新校验。
        inp = _StrictPolicyInput(policy=_Policy.A)
        inp.policy = _Policy.B
        assert inp.policy is _Policy.B
        with pytest.raises(ValidationError):
            inp.policy = "not-a-policy"

    def test_default_assignment_validates(self):
        inp = _PolicyInput(policy=_Policy.A)
        with pytest.raises(ValidationError):
            inp.policy = "not-a-policy"

    def test_extra_fields_allowed(self):
        # extra="allow" 语义：未知字段透传保留，不报错。
        inp = _PolicyInput(policy=_Policy.A, tenant="zgo")
        assert inp.tenant == "zgo"

    def test_consumer_normalization_pattern(self):
        # 消费方归一化模式（zGo 修复写法）：默认契约下两态兼容。
        inp = _PolicyInput(policy=_Policy.B)
        assert _Policy(inp.policy) is _Policy.B
