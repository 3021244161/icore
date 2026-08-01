"""
Tests for icore.engine.executor.WorkflowExecutor.

Covers:
    - Linear DAG execution (chunk -> summarize -> merge pattern)
    - Diamond DAG with parallel wave execution
    - Task failure propagation (downstream nodes skipped)
    - Retry logic on execute() exceptions
    - Timeout enforcement via node.timeout
    - Conditional branching (edge conditions evaluated)
    - Sub-workflow delegation
    - input_builder vs auto-merge input construction
    - get_terminal_output aggregation
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest
from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import TaskRegistry, register_task, task_registry
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG
from icore.engine.executor import WorkflowExecutionResult, WorkflowExecutor
from icore.engine.registry import (
    WorkflowRegistry,
    register_workflow,
    workflow_registry,
)


# ---------------------------------------------------------------------------
# Test tasks
# ---------------------------------------------------------------------------

class _Counter:
    """Shared mutable counter for verifying retry behavior."""
    def __init__(self) -> None:
        self.calls = 0
        self.fail_until = 0


class _AddInput(BaseTaskInput):
    a: int
    b: int


class _AddTask(BaseTask):
    name: ClassVar[str] = "test_add"
    description: ClassVar[str] = "Add two numbers"
    input_model: ClassVar[type[BaseTaskInput]] = _AddInput

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        return BaseTaskOutput.success(sum=inp.a + inp.b)


class _FlakyInput(BaseTaskInput):
    counter_key: str
    fail_until: int = 2


class _FlakyTask(BaseTask):
    """Fails N times then succeeds - for testing retry logic."""
    name: ClassVar[str] = "test_flaky"
    description: ClassVar[str] = "Fails N times then succeeds"
    input_model: ClassVar[type[BaseTaskInput]] = _FlakyInput

    _counters: ClassVar[dict[str, int]] = {}

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        key = inp.counter_key
        _FlakyTask._counters[key] = _FlakyTask._counters.get(key, 0) + 1
        n = _FlakyTask._counters[key]
        if n <= inp.fail_until:
            raise RuntimeError(f"flaky failure #{n}")
        return BaseTaskOutput.success(call_count=n)


class _SlowInput(BaseTaskInput):
    delay: float = 1.0


class _SlowTask(BaseTask):
    name: ClassVar[str] = "test_slow"
    description: ClassVar[str] = "Sleeps for inp.delay seconds"
    input_model: ClassVar[type[BaseTaskInput]] = _SlowInput

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        await asyncio.sleep(inp.delay)
        return BaseTaskOutput.success(slept=inp.delay)


class _ClassifyInput(BaseTaskInput):
    label: str


class _ClassifyTask(BaseTask):
    """Pass-through task that echoes the input label."""
    name: ClassVar[str] = "test_classify"
    description: ClassVar[str] = "Echoes label for branching"
    input_model: ClassVar[type[BaseTaskInput]] = _ClassifyInput

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        return BaseTaskOutput.success(label=inp.label)


class _RouteInput(BaseTaskInput):
    pass


class _RouteA(BaseTask):
    name: ClassVar[str] = "test_route_a"
    description: ClassVar[str] = "Branch A"
    input_model: ClassVar[type[BaseTaskInput]] = _RouteInput

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        return BaseTaskOutput.success(branch="A")


class _RouteB(BaseTask):
    name: ClassVar[str] = "test_route_b"
    description: ClassVar[str] = "Branch B"
    input_model: ClassVar[type[BaseTaskInput]] = _RouteInput

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        return BaseTaskOutput.success(branch="B")


class _FailingTask(BaseTask):
    name: ClassVar[str] = "test_failing"
    description: ClassVar[str] = "Always fails"
    input_model: ClassVar[type[BaseTaskInput]] = BaseTaskInput

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        raise RuntimeError("always fails")


# ---------------------------------------------------------------------------
# Register test tasks once
# ---------------------------------------------------------------------------

_register_called = False


def _ensure_test_tasks_registered() -> None:
    global _register_called
    if _register_called:
        return
    _register_called = True
    # Only register if not already present
    for cls in (_AddTask, _FlakyTask, _SlowTask, _ClassifyTask,
                _RouteA, _RouteB, _FailingTask):
        if not task_registry.contains(cls.name):
            task_registry.register(cls.name, cls)


_ensure_test_tasks_registered()


# ---------------------------------------------------------------------------
# Helper: build a TaskContext with managers attached
# ---------------------------------------------------------------------------

def _make_ctx(model_manager=None, db_manager=None) -> TaskContext:
    ctx = TaskContext(task_id="t-root", workflow_id="wf-test")
    if model_manager is not None:
        ctx.set_model_manager(model_manager)
    if db_manager is not None:
        ctx.set_db_manager(db_manager)
    return ctx


# ---------------------------------------------------------------------------
# WorkflowExecutionResult
# ---------------------------------------------------------------------------

class TestWorkflowExecutionResult:
    def test_is_success_default_pending(self):
        r = WorkflowExecutionResult()
        assert r.is_success is False

    def test_is_success_when_completed(self):
        r = WorkflowExecutionResult()
        r.workflow_state = type(
            "S", (), {"COMPLETED": "completed"}
        )().COMPLETED
        # The actual enum value is "completed"
        from icore.engine.states import WorkflowState
        r.workflow_state = WorkflowState.COMPLETED
        assert r.is_success is True

    def test_get_terminal_output_single(self):
        r = WorkflowExecutionResult()
        out = BaseTaskOutput.success(x=1)
        r.node_outputs["only"] = out
        assert r.get_terminal_output(["only"]) is out

    def test_get_terminal_output_none_active(self):
        r = WorkflowExecutionResult()
        r.error = "everything failed"
        out = r.get_terminal_output([])
        assert out.is_success is False
        assert "everything failed" in out.error

    def test_get_terminal_output_merged(self):
        r = WorkflowExecutionResult()
        r.node_outputs["a"] = BaseTaskOutput.success(x=1)
        r.node_outputs["b"] = BaseTaskOutput.success(y=2)
        out = r.get_terminal_output(["a", "b"])
        assert out.is_success is True
        # Multiple terminal nodes -> data merged under "results" key
        assert out.data["results"]["a"] == {"x": 1}
        assert out.data["results"]["b"] == {"y": 2}

    def test_get_terminal_output_with_one_failure(self):
        r = WorkflowExecutionResult()
        r.node_outputs["a"] = BaseTaskOutput.success(x=1)
        r.node_outputs["b"] = BaseTaskOutput.failure("boom")
        out = r.get_terminal_output(["a", "b"])
        assert out.is_success is False
        # Merged data is still present under "results"
        assert "a" in out.data["results"]
        assert "b" in out.data["results"]


# ---------------------------------------------------------------------------
# Linear DAG execution
# ---------------------------------------------------------------------------

class TestExecutorLinear:
    async def test_simple_two_node_chain(self):
        """A -> B: A produces a, b; B uses input_builder to add them."""
        dag = DAG()
        dag.add_node("a", task_name="test_add")
        dag.add_node(
            "b",
            task_name="test_add",
            input_builder=lambda params, upstream: _AddInput(
                a=upstream["a"].data["sum"],
                b=10,
            ),
        )
        dag.add_edge("a", "b")

        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(dag, ctx, {"a": 5, "b": 7})

        assert result.is_success is True
        # Final output from terminal node 'b' = 5 + 7 + 10 = 22
        assert result.data["sum"] == 22

    async def test_auto_merge_input_no_builder(self):
        """Without input_builder, executor merges params + upstream data."""
        dag = DAG()
        dag.add_node("a", task_name="test_add")
        dag.add_node("b", task_name="test_add")
        dag.add_edge("a", "b")

        executor = WorkflowExecutor()
        ctx = _make_ctx()
        # params provide a=3, b=4 -> a's output sum=7
        # b's input merges params (a=3, b=4) with a's output (sum=7)
        # _AddInput needs a, b -> from params: a=3, b=4 -> sum=7
        result = await executor.run(dag, ctx, {"a": 3, "b": 4})

        assert result.is_success is True
        # b's output: a=3, b=4 -> 7
        assert result.data["sum"] == 7


# ---------------------------------------------------------------------------
# Diamond DAG (parallel wave)
# ---------------------------------------------------------------------------

class TestExecutorDiamond:
    async def test_diamond_parallel_branches(self):
        """A -> B, A -> C, B -> D, C -> D.

        B and C must run in the same wave (parallel). D merges.
        """
        dag = DAG()
        dag.add_node("start", task_name="test_add")  # produces sum
        dag.add_node(
            "b",
            task_name="test_add",
            input_builder=lambda params, upstream: _AddInput(
                a=upstream["start"].data["sum"], b=100
            ),
        )
        dag.add_node(
            "c",
            task_name="test_add",
            input_builder=lambda params, upstream: _AddInput(
                a=upstream["start"].data["sum"], b=200
            ),
        )
        dag.add_node(
            "d",
            task_name="test_add",
            input_builder=lambda params, upstream: _AddInput(
                a=upstream["b"].data["sum"],
                b=upstream["c"].data["sum"],
            ),
        )
        dag.add_edge("start", "b")
        dag.add_edge("start", "c")
        dag.add_edge("b", "d")
        dag.add_edge("c", "d")

        executor = WorkflowExecutor()
        ctx = _make_ctx()
        # start: a=1, b=2 -> 3
        # b: 3 + 100 = 103
        # c: 3 + 200 = 203
        # d: 103 + 203 = 306
        result = await executor.run(dag, ctx, {"a": 1, "b": 2})
        assert result.is_success is True
        assert result.data["sum"] == 306


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------

class TestExecutorFailure:
    async def test_failure_skips_downstream(self):
        """Failing node -> downstream nodes should be skipped."""
        dag = DAG()
        dag.add_node("bad", task_name="test_failing")
        dag.add_node(
            "after",
            task_name="test_add",
            input_builder=lambda params, upstream: _AddInput(a=1, b=2),
        )
        dag.add_edge("bad", "after")

        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(dag, ctx, {})

        # Workflow should fail
        assert result.is_success is False
        assert "Workflow failed" in (result.error or "") or \
               "always fails" in (result.error or "")

    async def test_unknown_task_returns_failure(self):
        dag = DAG()
        dag.add_node("ghost", task_name="does_not_exist")
        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(dag, ctx, {})
        assert result.is_success is False
        assert "not registered" in (result.error or "").lower() or \
               "does_not_exist" in (result.error or "")


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestExecutorRetry:
    async def test_retry_succeeds_after_failures(self):
        """Flaky task fails twice, succeeds on third attempt (retries=2)."""
        # Reset counter for this test
        _FlakyTask._counters["retry_test"] = 0

        dag = DAG()
        dag.add_node(
            "flaky",
            task_name="test_flaky",
            retries=2,  # 2 retries = 3 total attempts
        )

        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(
            dag, ctx, {"counter_key": "retry_test", "fail_until": 2}
        )

        assert result.is_success is True
        assert result.data["call_count"] == 3

    async def test_retry_exhausted_returns_failure(self):
        """Flaky task fails 5 times, retries=2 -> still fails."""
        _FlakyTask._counters["exhaust_test"] = 0

        dag = DAG()
        dag.add_node(
            "flaky",
            task_name="test_flaky",
            retries=2,
        )

        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(
            dag, ctx, {"counter_key": "exhaust_test", "fail_until": 5}
        )

        assert result.is_success is False
        # Should have been called 3 times (1 + 2 retries)
        assert _FlakyTask._counters["exhaust_test"] == 3


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestExecutorTimeout:
    async def test_timeout_marks_failure(self):
        dag = DAG()
        dag.add_node(
            "slow",
            task_name="test_slow",
            timeout=0.1,  # 100ms timeout
        )
        # Input asks for 1-second sleep
        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(dag, ctx, {"delay": 1.0})

        assert result.is_success is False
        assert "timed out" in (result.error or "").lower() or \
               "timeout" in (result.error or "").lower()

    async def test_no_timeout_completes(self):
        dag = DAG()
        dag.add_node("slow", task_name="test_slow")
        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(dag, ctx, {"delay": 0.05})
        assert result.is_success is True
        assert result.data["slept"] == 0.05


# ---------------------------------------------------------------------------
# Conditional branching
# ---------------------------------------------------------------------------

class TestExecutorConditional:
    async def test_conditional_branch_taken(self):
        """classify -> route_a (if label=='A'), classify -> route_b (if label=='B')"""
        dag = DAG()
        dag.add_node("classify", task_name="test_classify")
        dag.add_node(
            "route_a",
            task_name="test_route_a",
            input_builder=lambda params, upstream: _RouteInput(),
        )
        dag.add_node(
            "route_b",
            task_name="test_route_b",
            input_builder=lambda params, upstream: _RouteInput(),
        )
        # Conditional edges
        dag.add_edge(
            "classify", "route_a",
            condition=lambda out: out.data.get("label") == "A",
        )
        dag.add_edge(
            "classify", "route_b",
            condition=lambda out: out.data.get("label") == "B",
        )

        executor = WorkflowExecutor()
        ctx = _make_ctx()

        # Take branch A
        result = await executor.run(dag, ctx, {"label": "A"})
        assert result.is_success is True
        assert result.data["branch"] == "A"

        # Take branch B (separate run)
        result_b = await executor.run(dag, ctx, {"label": "B"})
        assert result_b.is_success is True
        assert result_b.data["branch"] == "B"


# ---------------------------------------------------------------------------
# Sub-workflow
# ---------------------------------------------------------------------------

class _SubWorkflowInput(BaseTaskInput):
    value: int


class _SubWorkflowOutput(BaseTaskInput):
    pass


class _SubAddTask(BaseTask):
    """Adds 100 to upstream value."""
    name: ClassVar[str] = "test_sub_add"
    description: ClassVar[str] = "Add 100 to value"
    input_model: ClassVar[type[BaseTaskInput]] = _SubWorkflowInput

    async def prepare(self, ctx):
        pass

    async def execute(self, ctx, inp):
        return BaseTaskOutput.success(result=inp.value + 100)


if not task_registry.contains("test_sub_add"):
    task_registry.register("test_sub_add", _SubAddTask)


@register_workflow("test_sub_wf")
class _SubWorkflow(BaseWorkflow):
    name: ClassVar[str] = "test_sub_wf"
    description: ClassVar[str] = "Adds 100 to value"

    def define(self) -> DAG:
        dag = DAG()
        dag.add_node("inner_add", task_name="test_sub_add")
        return dag


class TestExecutorSubworkflow:
    async def test_subworkflow_invocation(self):
        """Top-level node delegates to a registered sub-workflow."""
        dag = DAG()
        dag.add_node(
            "sub",
            is_subworkflow=True,
            workflow_name="test_sub_wf",
        )

        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(dag, ctx, {"value": 5})

        assert result.is_success is True
        assert result.data["result"] == 105

    async def test_unknown_subworkflow_returns_failure(self):
        dag = DAG()
        dag.add_node(
            "sub",
            is_subworkflow=True,
            workflow_name="does_not_exist",
        )

        executor = WorkflowExecutor()
        ctx = _make_ctx()
        result = await executor.run(dag, ctx, {})
        assert result.is_success is False
