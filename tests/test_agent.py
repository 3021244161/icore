"""
Tests for icore.engine.agent — the v0.6 multi-Agent collaboration framework.

Coverage:
    - AgentConfig defaults / custom values / nested supervisor_agents
    - AgentExecutionResult construction
    - _parse_llm_response: YAML block, JSON block, plain JSON, FINISH
      fallback, empty response
    - _validate_output: valid, missing required, wrong type, non-dict
    - REACT: single-iteration FINISH, multi-iteration tool calls,
      max_iterations, max_tool_calls, unregistered task (graceful),
      action not in available_tasks, output_schema pass/fail, LLM error,
      model_id override, no-available-tasks immediate FINISH
    - SUPERVISOR: parallel sub-agents, result aggregation, no sub-agents
    - SWARM: transfer chain, max_transfers, no members
    - Integration: full REACT workflow, full SUPERVISOR workflow, agent
      node in DAG metadata

All tests are offline: a scripted FakeModelAdapter drives the Agent's
reasoning; leaf tasks are pure Python.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task, task_registry
from icore.core.task_context import TaskContext
from icore.engine.agent import (
    AgentConfig,
    AgentExecutionResult,
    AgentMode,
    AgentNodeExecutor,
)
from icore.models.manager import ModelManager
from tests.conftest import FakeModelAdapter, make_model_config

# Import the demo module so its @register_task / @register_workflow
# decorators run and the workflows become discoverable.
import icore.workflows.examples.agent_demo  # noqa: F401


# ===========================================================================
# Test tasks (tools the agents can call)
# ===========================================================================

class _AddInput(BaseTaskInput):
    a: int = 0
    b: int = 0


@register_task("agent_test_add")
class _AddTask(BaseTask):
    name: ClassVar[str] = "agent_test_add"
    description: ClassVar[str] = "Add two integers"
    input_model: ClassVar[type[BaseTaskInput]] = _AddInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: _AddInput) -> BaseTaskOutput:
        return BaseTaskOutput.success(sum=inp.a + inp.b)

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


class _EchoInput(BaseTaskInput):
    message: str = ""


@register_task("agent_test_echo")
class _EchoTask(BaseTask):
    name: ClassVar[str] = "agent_test_echo"
    description: ClassVar[str] = "Echo a message"
    input_model: ClassVar[type[BaseTaskInput]] = _EchoInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: _EchoInput) -> BaseTaskOutput:
        return BaseTaskOutput.success(echo=inp.message)

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ===========================================================================
# Test helpers
# ===========================================================================

def _manager_with_responder(responder):
    """Build a ModelManager backed by a FakeModelAdapter whose chat()
    delegates to ``responder(messages)``."""
    config = make_model_config("fake-model")
    adapter = FakeModelAdapter(config, responder=responder)
    mgr = ModelManager()
    mgr.register_adapter("fake-model", adapter)
    mgr._router.set_default_model_id("fake-model")
    mgr._auto_routing = True
    return mgr, adapter


def _sequence_responder(responses):
    """Return a responder that yields ``responses`` in order, then a
    default FINISH."""
    state = {"idx": 0}

    def responder(messages):
        if state["idx"] < len(responses):
            r = responses[state["idx"]]
            state["idx"] += 1
            return r
        return '```yaml\naction: FINISH\nfinal_answer: {"done": true}\n```'

    return responder


def _ctx(mgr, model_id=None) -> TaskContext:
    ctx = TaskContext(task_id="t-agent", workflow_id="wf-agent")
    ctx.set_model_manager(mgr)
    if model_id is not None:
        ctx.model_id = model_id
    return ctx


# ===========================================================================
# AgentConfig
# ===========================================================================

class TestAgentConfig:
    def test_defaults(self):
        c = AgentConfig()
        assert c.mode == AgentMode.REACT
        assert c.goal == ""
        assert c.max_iterations == 10
        assert c.max_tool_calls == 20
        assert c.available_tasks == []
        assert c.output_schema is None
        assert c.supervisor_agents == []
        assert c.swarm_max_transfers == 5
        assert c.model_id is None
        assert c.temperature == 0.7

    def test_custom_values(self):
        c = AgentConfig(
            mode=AgentMode.SUPERVISOR,
            goal="review code",
            max_iterations=3,
            max_tool_calls=5,
            available_tasks=["style_check"],
            output_schema={"required": ["x"]},
            swarm_max_transfers=2,
            model_id="gpt-4o",
            temperature=0.1,
        )
        assert c.mode == AgentMode.SUPERVISOR
        assert c.max_iterations == 3
        assert c.available_tasks == ["style_check"]
        assert c.output_schema == {"required": ["x"]}
        assert c.model_id == "gpt-4o"
        assert c.temperature == 0.1

    def test_supervisor_agents_nesting(self):
        child = AgentConfig(goal="child", available_tasks=["t1"])
        parent = AgentConfig(
            mode=AgentMode.SUPERVISOR,
            goal="parent",
            supervisor_agents=[child],
        )
        assert len(parent.supervisor_agents) == 1
        assert parent.supervisor_agents[0].goal == "child"
        assert parent.supervisor_agents[0].available_tasks == ["t1"]

    def test_available_tasks_default_empty(self):
        c = AgentConfig()
        assert c.available_tasks == []
        assert isinstance(c.available_tasks, list)


# ===========================================================================
# AgentExecutionResult
# ===========================================================================

class TestAgentExecutionResult:
    def test_construction(self):
        out = BaseTaskOutput.success(x=1)
        r = AgentExecutionResult(
            output=out, iterations=2, tool_calls=1,
            trace=[{"step": 1}], finish_reason="completed",
        )
        assert r.output is out
        assert r.iterations == 2
        assert r.tool_calls == 1
        assert r.finish_reason == "completed"
        assert r.transfer_target is None

    def test_finish_reason_values(self):
        for reason in ("completed", "max_iterations", "max_tool_calls",
                       "error", "max_transfers", "transfer"):
            out = BaseTaskOutput.success()
            r = AgentExecutionResult(
                output=out, iterations=0, tool_calls=0,
                finish_reason=reason,
            )
            assert r.finish_reason == reason


# ===========================================================================
# _parse_llm_response
# ===========================================================================

class TestParseLLMResponse:
    def setup_method(self):
        self.ex = AgentNodeExecutor()

    def test_yaml_block(self):
        resp = (
            '```yaml\n'
            'thought: adding numbers\n'
            'action: agent_test_add\n'
            'action_input: {"a": 1, "b": 2}\n'
            '```'
        )
        p = self.ex._parse_llm_response(resp)
        assert p["thought"] == "adding numbers"
        assert p["action"] == "agent_test_add"
        assert p["action_input"] == {"a": 1, "b": 2}
        assert p["final_answer"] is None

    def test_json_block(self):
        resp = (
            '```json\n'
            '{"thought": "hi", "action": "agent_test_add", '
            '"action_input": {"a": 5, "b": 6}}\n'
            '```'
        )
        p = self.ex._parse_llm_response(resp)
        assert p["thought"] == "hi"
        assert p["action"] == "agent_test_add"
        assert p["action_input"]["a"] == 5

    def test_plain_json(self):
        resp = '{"thought": "x", "action": "FINISH", "final_answer": {"r": 1}}'
        p = self.ex._parse_llm_response(resp)
        assert p["action"] == "FINISH"
        assert p["final_answer"] == {"r": 1}

    def test_finish_keyword_fallback(self):
        resp = "I am done. FINISH with result."
        p = self.ex._parse_llm_response(resp)
        assert p["action"] == "FINISH"

    def test_empty_response(self):
        p = self.ex._parse_llm_response("")
        assert p["action"] == "FINISH"
        p2 = self.ex._parse_llm_response("   ")
        assert p2["action"] == "FINISH"


# ===========================================================================
# _validate_output
# ===========================================================================

class TestValidateOutput:
    def setup_method(self):
        self.ex = AgentNodeExecutor()

    def test_valid(self):
        schema = {
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        assert self.ex._validate_output({"name": "bob", "age": 30}, schema)

    def test_missing_required(self):
        schema = {"required": ["name", "age"]}
        assert not self.ex._validate_output({"name": "bob"}, schema)

    def test_wrong_type(self):
        schema = {
            "properties": {"age": {"type": "integer"}},
        }
        assert not self.ex._validate_output({"age": "thirty"}, schema)

    def test_not_dict(self):
        schema = {"required": ["x"]}
        assert not self.ex._validate_output("not a dict", schema)
        assert not self.ex._validate_output([1, 2], schema)

    def test_bool_not_integer(self):
        # bool is a subclass of int in Python; ensure it's rejected as
        # an integer per JSON Schema semantics.
        schema = {"properties": {"flag": {"type": "integer"}}}
        assert not self.ex._validate_output({"flag": True}, schema)


# ===========================================================================
# REACT execution
# ===========================================================================

class TestReactExecution:
    @pytest.mark.asyncio
    async def test_single_iteration_finish(self):
        responses = [
            '```yaml\nthought: done\naction: FINISH\n'
            'final_answer: {"result": 42}\n```'
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test", available_tasks=["agent_test_add"],
            max_iterations=5,
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "completed"
        assert result.iterations == 1
        assert result.tool_calls == 0
        assert result.output.is_success
        assert result.output.data["result"] == 42

    @pytest.mark.asyncio
    async def test_multiple_iterations_with_tool_calls(self):
        responses = [
            '```yaml\nthought: add\naction: agent_test_add\n'
            'action_input: {"a": 1, "b": 2}\n```',
            '```yaml\nthought: echo\naction: agent_test_echo\n'
            'action_input: {"message": "hi"}\n```',
            '```yaml\nthought: done\naction: FINISH\n'
            'final_answer: {"total": 3, "msg": "hi"}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test",
            available_tasks=["agent_test_add", "agent_test_echo"],
            max_iterations=5,
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "completed"
        assert result.iterations == 3
        assert result.tool_calls == 2
        assert result.output.data["total"] == 3

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        # LLM always wants to call a tool, never FINISH.
        responses = [
            '```yaml\naction: agent_test_add\n'
            'action_input: {"a": 1, "b": 2}\n```',
            '```yaml\naction: agent_test_add\n'
            'action_input: {"a": 3, "b": 4}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test", available_tasks=["agent_test_add"],
            max_iterations=2, max_tool_calls=20,
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "max_iterations"
        assert result.iterations == 2
        assert result.tool_calls == 2

    @pytest.mark.asyncio
    async def test_max_tool_calls(self):
        responses = [
            '```yaml\naction: agent_test_add\n'
            'action_input: {"a": 1, "b": 2}\n```',
            '```yaml\naction: agent_test_add\n'
            'action_input: {"a": 3, "b": 4}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test", available_tasks=["agent_test_add"],
            max_iterations=10, max_tool_calls=1,
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "max_tool_calls"
        assert result.tool_calls == 1
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_unregistered_task_graceful(self):
        # available_tasks lists a name that is NOT registered.
        responses = [
            '```yaml\naction: __not_registered__\naction_input: {}\n```',
            '```yaml\naction: FINISH\nfinal_answer: {"ok": true}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test", available_tasks=["__not_registered__"],
            max_iterations=5,
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        # The error observation must not crash the agent; it FINISHes next.
        assert result.finish_reason == "completed"
        assert result.tool_calls == 0
        assert result.iterations == 2
        # The trace should record the error observation.
        assert "Error" in result.trace[0]["observation"]

    @pytest.mark.asyncio
    async def test_action_not_in_available_tasks(self):
        responses = [
            '```yaml\naction: agent_test_add\n'
            'action_input: {"a": 1, "b": 2}\n```',
            '```yaml\naction: FINISH\nfinal_answer: {"ok": true}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test", available_tasks=["agent_test_echo"],
            max_iterations=5,
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        # agent_test_add is not in available_tasks -> error observation,
        # tool_calls stays 0, then FINISH.
        assert result.finish_reason == "completed"
        assert result.tool_calls == 0
        assert "not in available_tasks" in result.trace[0]["observation"]

    @pytest.mark.asyncio
    async def test_output_schema_validation_failure(self):
        responses = [
            '```yaml\naction: FINISH\nfinal_answer: {"name": "bob"}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test", available_tasks=[], max_iterations=5,
            output_schema={
                "required": ["name", "age"],
                "properties": {"age": {"type": "integer"}},
            },
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "error"
        assert not result.output.is_success
        assert "schema" in result.output.error.lower()

    @pytest.mark.asyncio
    async def test_output_schema_validation_success(self):
        responses = [
            '```yaml\naction: FINISH\n'
            'final_answer: {"name": "bob", "age": 30}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test", available_tasks=[], max_iterations=5,
            output_schema={
                "required": ["name", "age"],
                "properties": {"age": {"type": "integer"}},
            },
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "completed"
        assert result.output.data["name"] == "bob"
        assert result.output.data["age"] == 30

    @pytest.mark.asyncio
    async def test_llm_call_error(self):
        def raising_responder(messages):
            raise RuntimeError("LLM is down")
        mgr, _ = _manager_with_responder(raising_responder)
        config = AgentConfig(
            goal="test", available_tasks=["agent_test_add"],
            max_iterations=5,
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "error"
        assert not result.output.is_success
        assert "LLM call failed" in result.output.error

    @pytest.mark.asyncio
    async def test_model_id_override(self):
        # Two adapters: default echoes "default", "pro" echoes "pro".
        def default_responder(messages):
            return '```yaml\naction: FINISH\nfinal_answer: {"from": "default"}\n```'

        def pro_responder(messages):
            return '```yaml\naction: FINISH\nfinal_answer: {"from": "pro"}\n```'

        config_default = make_model_config("fake-model")
        adapter_default = FakeModelAdapter(config_default, responder=default_responder)
        config_pro = make_model_config("pro-model")
        adapter_pro = FakeModelAdapter(config_pro, responder=pro_responder)

        mgr = ModelManager()
        mgr.register_adapter("fake-model", adapter_default)
        mgr.register_adapter("pro-model", adapter_pro)
        mgr._router.set_default_model_id("fake-model")
        mgr._auto_routing = True

        config = AgentConfig(
            goal="test", available_tasks=[], max_iterations=5,
            model_id="pro-model",
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "completed"
        assert result.output.data["from"] == "pro"

    @pytest.mark.asyncio
    async def test_no_available_tasks_immediate_finish(self):
        responses = [
            '```yaml\naction: FINISH\nfinal_answer: {"ok": true}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(goal="test", available_tasks=[], max_iterations=3)
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "completed"
        assert result.iterations == 1


# ===========================================================================
# SUPERVISOR execution
# ===========================================================================

def _supervisor_responder():
    """Content-aware responder: routes by agent identity (goal in the
    system prompt). Order-independent so parallel sub-agents work."""
    def responder(messages):
        sys_content = messages[0]["content"] if messages else ""
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        history_empty = "(none)" in user_content

        if "supervisor agent" in sys_content.lower():
            return (
                '```json\n{"assignments": ['
                '{"agent_index": 0, "sub_input": {}}, '
                '{"agent_index": 1, "sub_input": {}}, '
                '{"agent_index": 2, "sub_input": {}}'
                ']}\n```'
            )
        if "StyleAgent" in sys_content:
            if history_empty:
                return ('```yaml\naction: style_check\n'
                        'action_input: {"code": "x = 1"}\n```')
            return '```yaml\naction: FINISH\nfinal_answer: {"style": "pass"}\n```'
        if "SecurityAgent" in sys_content:
            if history_empty:
                return ('```yaml\naction: security_check\n'
                        'action_input: {"code": "x = 1"}\n```')
            return '```yaml\naction: FINISH\nfinal_answer: {"security": "pass"}\n```'
        if "PerfAgent" in sys_content:
            if history_empty:
                return ('```yaml\naction: perf_check\n'
                        'action_input: {"code": "x = 1"}\n```')
            return '```yaml\naction: FINISH\nfinal_answer: {"perf": "pass"}\n```'
        return '```yaml\naction: FINISH\nfinal_answer: {"done": true}\n```'
    return responder


class TestSupervisorExecution:
    @pytest.mark.asyncio
    async def test_parallel_sub_agents(self):
        mgr, _ = _manager_with_responder(_supervisor_responder())
        config = AgentConfig(
            mode=AgentMode.SUPERVISOR,
            goal="review code",
            available_tasks=[],
            supervisor_agents=[
                AgentConfig(goal="StyleAgent", available_tasks=["style_check"],
                            max_iterations=3),
                AgentConfig(goal="SecurityAgent", available_tasks=["security_check"],
                            max_iterations=3),
                AgentConfig(goal="PerfAgent", available_tasks=["perf_check"],
                            max_iterations=3),
            ],
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr), params={"code": "x=1"})
        assert result.finish_reason == "completed"
        # All three sub-agents ran.
        agg = result.output.data["results"]
        assert "StyleAgent" in agg
        assert "SecurityAgent" in agg
        assert "PerfAgent" in agg
        # Each sub-agent made exactly 1 tool call.
        assert result.tool_calls == 3
        # Each sub-agent ran 2 iterations (call tool + FINISH).
        assert result.iterations == 6

    @pytest.mark.asyncio
    async def test_aggregates_results(self):
        mgr, _ = _manager_with_responder(_supervisor_responder())
        config = AgentConfig(
            mode=AgentMode.SUPERVISOR,
            goal="review",
            supervisor_agents=[
                AgentConfig(goal="StyleAgent", available_tasks=["style_check"]),
                AgentConfig(goal="SecurityAgent", available_tasks=["security_check"]),
            ],
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        agg = result.output.data["results"]
        assert agg["StyleAgent"]["style"] == "pass"
        assert agg["SecurityAgent"]["security"] == "pass"

    @pytest.mark.asyncio
    async def test_no_sub_agents_error(self):
        mgr, _ = _manager_with_responder(_supervisor_responder())
        config = AgentConfig(mode=AgentMode.SUPERVISOR, goal="review")
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "error"
        assert not result.output.is_success
        assert "no supervisor_agents" in result.output.error


# ===========================================================================
# SWARM execution
# ===========================================================================

def _swarm_transfer_responder():
    """AgentA calls a tool then transfers to AgentB; AgentB calls a tool
    then FINISHes.

    Note: the swarm system prompt lists ALL peers (e.g. "['AgentA',
    'AgentB']"), so a naive ``"AgentA" in sys_content`` check would match
    AgentB too. We match on the goal line ``"Your goal: AgentA"`` to
    distinguish which agent is currently running.
    """
    def responder(messages):
        sys_content = messages[0]["content"] if messages else ""
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        history_empty = "(none)" in user_content

        if "Your goal: AgentA" in sys_content:
            if history_empty:
                return ('```yaml\naction: agent_test_add\n'
                        'action_input: {"a": 1, "b": 2}\n```')
            return ('```yaml\naction: TRANSFER\n'
                    'action_input: {"target": "AgentB"}\n```')
        if "Your goal: AgentB" in sys_content:
            if history_empty:
                return ('```yaml\naction: agent_test_add\n'
                        'action_input: {"a": 3, "b": 4}\n```')
            return '```yaml\naction: FINISH\nfinal_answer: {"final": "done"}\n```'
        return '```yaml\naction: FINISH\n```'
    return responder


class TestSwarmExecution:
    @pytest.mark.asyncio
    async def test_simple_transfer_chain(self):
        mgr, _ = _manager_with_responder(_swarm_transfer_responder())
        config = AgentConfig(
            mode=AgentMode.SWARM,
            goal="swarm",
            swarm_max_transfers=5,
            supervisor_agents=[
                AgentConfig(goal="AgentA", available_tasks=["agent_test_add"],
                            max_iterations=5),
                AgentConfig(goal="AgentB", available_tasks=["agent_test_add"],
                            max_iterations=5),
            ],
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "completed"
        assert result.output.data["final"] == "done"
        # AgentA: 1 tool call; AgentB: 1 tool call.
        assert result.tool_calls == 2
        # The trace records both agents.
        agents = [e["agent"] for e in result.trace if e.get("phase") == "swarm_agent"]
        assert "AgentA" in agents
        assert "AgentB" in agents

    @pytest.mark.asyncio
    async def test_max_transfers(self):
        # Both agents always TRANSFER to each other -> infinite loop,
        # bounded by swarm_max_transfers=1.
        # Use "Your goal:" prefix to avoid matching peer names in the
        # swarm peers list (see _swarm_transfer_responder docstring).
        def loop_responder(messages):
            sys_content = messages[0]["content"] if messages else ""
            if "Your goal: AgentA" in sys_content:
                return ('```yaml\naction: TRANSFER\n'
                    'action_input: {"target": "AgentB"}\n```')
            if "Your goal: AgentB" in sys_content:
                return ('```yaml\naction: TRANSFER\n'
                    'action_input: {"target": "AgentA"}\n```')
            return '```yaml\naction: FINISH\n```'

        mgr, _ = _manager_with_responder(loop_responder)
        config = AgentConfig(
            mode=AgentMode.SWARM,
            goal="swarm",
            swarm_max_transfers=1,
            supervisor_agents=[
                AgentConfig(goal="AgentA", available_tasks=[], max_iterations=3),
                AgentConfig(goal="AgentB", available_tasks=[], max_iterations=3),
            ],
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "max_transfers"
        assert not result.output.is_success
        assert "max_transfers" in result.output.error.lower() or "swarm_max_transfers" in result.output.error

    @pytest.mark.asyncio
    async def test_no_members_error(self):
        mgr, _ = _manager_with_responder(_swarm_transfer_responder())
        config = AgentConfig(mode=AgentMode.SWARM, goal="swarm")
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "error"
        assert not result.output.is_success
        assert "no swarm members" in result.output.error.lower() or "swarm members" in result.output.error.lower()


# ===========================================================================
# Integration: demo workflows
# ===========================================================================

def _data_analysis_responder():
    """Drive the REACT data-analysis agent: call analyze_trend, then FINISH."""
    def responder(messages):
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        if "(none)" in user_content:
            return (
                '```yaml\naction: analyze_trend\n'
                'action_input: {"data": ['
                '{"month": "2026-01", "sales": 100},'
                '{"month": "2026-02", "sales": 200}'
                ']}\n```'
            )
        return ('```yaml\naction: FINISH\n'
                'final_answer: {"trend": "up", "slope": 100.0}\n```')
    return responder


class TestIntegration:
    @pytest.mark.asyncio
    async def test_react_workflow_end_to_end(self):
        from icore.workflows.examples.agent_demo import (
            DataAnalysisAgentWorkflow,
        )
        mgr, _ = _manager_with_responder(_data_analysis_responder())
        wf = DataAnalysisAgentWorkflow()
        out = await wf.execute(_ctx(mgr), {"csv_path": "test.csv"})
        assert out.is_success
        # The agent_render_report task renders the agent's analysis dict.
        assert "report" in out.data
        assert "trend" in out.data["report"] or "up" in out.data["report"]

    @pytest.mark.asyncio
    async def test_supervisor_workflow_end_to_end(self):
        from icore.workflows.examples.agent_demo import (
            CodeReviewSupervisorWorkflow,
        )
        mgr, _ = _manager_with_responder(_supervisor_responder())
        wf = CodeReviewSupervisorWorkflow()
        out = await wf.execute(_ctx(mgr), {"code": "x = 1"})
        assert out.is_success
        # The aggregate task produces a summary + total_issues.
        assert "summary" in out.data
        assert "total_issues" in out.data
        assert "StyleAgent" in out.data["reviewers"]

    def test_agent_node_in_dag_metadata(self):
        from icore.workflows.examples.agent_demo import (
            DataAnalysisAgentWorkflow,
        )
        wf = DataAnalysisAgentWorkflow()
        dag = wf.define()
        assert dag.validate()

        node = dag.get_node("agent_decide")
        assert node.metadata["is_agent"] is True
        cfg = node.metadata["agent_config"]
        assert cfg.mode == AgentMode.REACT
        assert "analyze_trend" in cfg.available_tasks

        # The agent node sits between load_data and agent_render_report.
        assert "load_data" in dag.get_predecessors("agent_decide")
        assert "agent_render_report" in dag.get_successors("agent_decide")

    @pytest.mark.asyncio
    async def test_supervisor_workflow_registered(self):
        from icore.engine.registry import workflow_registry
        assert "data_analysis_agent" in workflow_registry.list_workflows()
        assert "code_review_supervisor" in workflow_registry.list_workflows()

    @pytest.mark.asyncio
    async def test_react_trace_recorded(self):
        """The REACT trace captures every think/act/observe step."""
        responses = [
            '```yaml\nthought: first\naction: agent_test_add\n'
            'action_input: {"a": 1, "b": 2}\n```',
            '```yaml\nthought: second\naction: agent_test_echo\n'
            'action_input: {"message": "hi"}\n```',
            '```yaml\nthought: done\naction: FINISH\n'
            'final_answer: {"ok": true}\n```',
        ]
        mgr, _ = _manager_with_responder(_sequence_responder(responses))
        config = AgentConfig(
            goal="test",
            available_tasks=["agent_test_add", "agent_test_echo"],
            max_iterations=5,
        )
        result = await AgentNodeExecutor().execute(config, _ctx(mgr))
        assert result.finish_reason == "completed"
        assert len(result.trace) == 3
        assert result.trace[0]["action"] == "agent_test_add"
        assert result.trace[0]["thought"] == "first"
        assert "OK:" in result.trace[0]["observation"]
        assert result.trace[2]["action"] == "FINISH"
