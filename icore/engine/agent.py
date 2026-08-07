"""
icore.engine.agent - Multi-Agent collaboration framework (v0.6).

Design philosophy (docs/12 §2.2.1):
    icore's core is always deterministic workflows. An Agent does not
    replace the DAG; it is a special node *within* the DAG — when a step
    cannot be determined in advance, an Agent makes the dynamic decision.

Core constraints:
    - Agent nodes and ordinary Task nodes are equal citizens in the DAG:
      both have a ``node_id``, both receive upstream output, both pass
      output downstream.
    - An Agent's "tools" are registered ordinary Tasks (via
      :class:`TaskRegistry`). There is no separate tool system — every
      tool call is a normal ``task.prepare -> execute -> cleanup`` cycle,
      enjoying the same lifecycle as a DAG node.
    - Agent boundaries are constrained by configurable hard limits
      (``max_iterations`` / ``max_tool_calls`` / ``swarm_max_transfers``)
      so an Agent can never run away.

Three modes share one executor; only the inner loop differs:

    REACT       — single Agent: think -> act -> observe -> reflect -> FINISH
    SUPERVISOR  — supervisor LLM decomposes the task; sub-agents (each a
                  REACT) run in parallel; results are aggregated.
    SWARM       — agents hand control to one another via ``TRANSFER`` until
                  one emits ``FINISH`` or the transfer budget is exhausted.

This module uses only the standard library ``logging`` (never loguru) and
does not depend on SQLAlchemy. YAML parsing lazy-imports ``yaml`` (PyYAML
is a declared runtime dependency, but lazy import keeps the module
importable even if the package is absent).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from icore.core.models import BaseTaskOutput
from icore.core.registry import task_registry as _default_task_registry
from icore.core.task_context import TaskContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class AgentMode(str, Enum):
    """Agent collaboration mode.

    REACT       — single autonomous agent (think/act/observe loop).
    SUPERVISOR  — supervisor decomposes; sub-agents run in parallel.
    SWARM       — agents transfer control to one another.
    """

    REACT = "react"
    SUPERVISOR = "supervisor"
    SWARM = "swarm"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Configuration for an Agent node.

    A config instance is attached to a DAG node via the explicit
    ``is_agent`` / ``agent_config`` keyword arguments on
    :meth:`DAG.add_node` (v0.6). For backward compatibility, the
    executor also checks ``node.metadata`` for these values.

    Attributes:
        mode:               Collaboration mode.
        goal:               High-level goal; sent to the LLM as the
                            system prompt.
        max_iterations:     Hard limit on think-act rounds.
        max_tool_calls:     Hard limit on Task invocations.
        available_tasks:    Registered task names the Agent may call as
                            tools. Must exist in ``TaskRegistry`` at
                            execution time.
        output_schema:      Optional JSON Schema (subset) used to
                            validate the Agent's final output. Validation
                            failure => node failure.
        supervisor_agents:  Sub-agent configs (SUPERVISOR mode) or swarm
                            member configs (SWARM mode).
        swarm_max_transfers: Hard limit on control transfers (SWARM).
        model_id:           Explicit model to use; ``None`` = ctx default.
        temperature:        LLM sampling temperature.
        force_json_output:  Force JSON response format from the LLM.
        timeout:            Optional per-agent timeout in seconds. When
                            set, the entire agent execution is wrapped in
                            ``asyncio.wait_for``. ``None`` = no timeout.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: AgentMode = AgentMode.REACT
    goal: str = ""
    max_iterations: int = 10
    max_tool_calls: int = 20
    available_tasks: list[str] = Field(default_factory=list)
    output_schema: Optional[dict[str, Any]] = None
    supervisor_agents: list["AgentConfig"] = Field(default_factory=list)
    swarm_max_transfers: int = 5
    model_id: Optional[str] = None
    temperature: float = 0.7
    # v0.6: 强制模型按 JSON 格式输出（透传 response_format 给适配器）。
    # 当为 True 时，_call_llm 会传 response_format={"type": "json_object"}，
    # 减少解析失败概率。需要模型支持（GPT-4o / DeepSeek 等原生支持）。
    force_json_output: bool = False
    # v0.6: Agent 执行超时（秒）。设置后 execute() 用 asyncio.wait_for 包裹
    # 实际执行逻辑，超时返回 failure 结果。None 表示不限制。
    timeout: Optional[float] = None


# Resolve the forward reference for ``supervisor_agents``.
AgentConfig.model_rebuild()


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

@dataclass
class AgentExecutionResult:
    """Outcome of a single Agent execution.

    Attributes:
        output:           Final ``BaseTaskOutput`` (success or failure).
        iterations:       Number of think-act rounds executed.
        tool_calls:       Number of Task invocations made.
        trace:            Full execution trajectory (one dict per step).
        finish_reason:    Why the agent stopped. One of
                          ``completed`` / ``max_iterations`` /
                          ``max_tool_calls`` / ``error`` /
                          ``max_transfers`` (SWARM) / ``transfer``
                          (internal SWARM handoff sentinel).
        transfer_target:  When ``finish_reason == "transfer"``, the name
                          of the agent that control was handed to.
    """

    output: BaseTaskOutput
    iterations: int
    tool_calls: int
    trace: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "completed"
    transfer_target: Optional[str] = None


# ---------------------------------------------------------------------------
# Sentinel action names
# ---------------------------------------------------------------------------

_FINISH = "FINISH"
_TRANSFER = "TRANSFER"

# Regexes for extracting fenced code blocks from LLM responses.
_YAML_BLOCK_RE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class AgentNodeExecutor:
    """Unified executor for all three Agent modes.

    The executor is agnostic about DAG integration: callers (typically a
    workflow's ``execute()`` override) detect agent nodes via
    ``node.is_agent`` (v0.6 explicit field) or ``node.metadata["is_agent"]``
    (backward compat) and delegate to :meth:`execute`.

    Args:
        task_registry:     ``TaskRegistry`` for tool lookup. Defaults to
                           the global default registry.
        workflow_executor: Optional ``WorkflowExecutor`` reference,
                           reserved for future sub-workflow tool support.
    """

    def __init__(
        self,
        task_registry: Any = None,
        workflow_executor: Any = None,
    ) -> None:
        self._task_registry = (
            task_registry if task_registry is not None else _default_task_registry
        )
        self._workflow_executor = workflow_executor

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        config: AgentConfig,
        ctx: TaskContext,
        upstream_output: BaseTaskOutput | None = None,
        params: dict[str, Any] | None = None,
        *,
        node_timeout: float | None = None,
    ) -> AgentExecutionResult:
        """Dispatch to the mode-specific executor.

        When a timeout is in effect (either ``node_timeout`` or
        ``config.timeout``), the entire agent execution is wrapped in
        :func:`asyncio.wait_for`. On timeout, a failure result is
        returned instead of raising.

        Observability (v0.6): a tracing span is opened for the agent
        execution, and the ``icore_task_duration_seconds`` histogram is
        observed on completion.

        Args:
            config:          Agent configuration.
            ctx:             TaskContext with an injected ModelManager.
            upstream_output: Output of the upstream DAG node (if any).
            params:          Workflow-level input parameters.
            node_timeout:    Optional timeout override (seconds). Takes
                             precedence over ``config.timeout``.

        Returns:
            :class:`AgentExecutionResult` describing the outcome.
        """
        params = params or {}
        timeout = node_timeout or config.timeout

        # Lazy import observability to avoid potential circular dependency
        # (AGENTS.md §10: agent.py should not hard-import observability at
        # module level).
        from icore.observability import get_metrics_registry, start_span

        _start = time.monotonic()
        with start_span(f"agent.execute.{config.mode.value}") as span:
            span.set_attribute("agent.mode", config.mode.value)
            span.set_attribute("agent.goal", config.goal)

            if timeout:
                try:
                    result = await asyncio.wait_for(
                        self._do_execute(
                            config, ctx, upstream_output, params
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Agent timed out after %.1fs (mode=%s)",
                        timeout,
                        config.mode.value,
                    )
                    result = AgentExecutionResult(
                        output=BaseTaskOutput.failure(
                            f"Agent timed out after {timeout}s"
                        ),
                        iterations=0,
                        tool_calls=0,
                        finish_reason="error",
                    )
            else:
                result = await self._do_execute(
                    config, ctx, upstream_output, params
                )

        # Record metrics (best-effort; never fail the agent for metrics).
        try:
            _duration = time.monotonic() - _start
            get_metrics_registry().get_histogram(
                "icore_task_duration_seconds"
            ).observe(
                _duration, task_name=f"agent_{config.mode.value}"
            )
        except Exception:  # noqa: BLE001 - metrics 不得影响核心逻辑
            pass

        return result

    async def _do_execute(
        self,
        config: AgentConfig,
        ctx: TaskContext,
        upstream_output: BaseTaskOutput | None,
        params: dict[str, Any],
    ) -> AgentExecutionResult:
        """Internal dispatch to the mode-specific executor."""
        if config.mode == AgentMode.REACT:
            return await self._execute_react(
                config, ctx, upstream_output, params
            )
        if config.mode == AgentMode.SUPERVISOR:
            return await self._execute_supervisor(
                config, ctx, upstream_output, params
            )
        if config.mode == AgentMode.SWARM:
            return await self._execute_swarm(
                config, ctx, upstream_output, params
            )
        return AgentExecutionResult(
            output=BaseTaskOutput.failure(
                f"Unknown agent mode: {config.mode}"
            ),
            iterations=0,
            tool_calls=0,
            finish_reason="error",
        )

    # ------------------------------------------------------------------
    # REACT
    # ------------------------------------------------------------------

    async def _execute_react(
        self,
        config: AgentConfig,
        ctx: TaskContext,
        upstream_output: BaseTaskOutput | None,
        params: dict[str, Any],
        swarm_ctx: dict[str, Any] | None = None,
    ) -> AgentExecutionResult:
        """Run the REACT think-act-observe loop.

        ``swarm_ctx`` is supplied only when invoked from
        :meth:`_execute_swarm`; it enables the ``TRANSFER`` action and
        carries the agent name -> config map.
        """
        trace: list[dict[str, Any]] = []
        tool_calls = 0
        history: list[dict[str, Any]] = []

        system_prompt = self._build_react_system_prompt(config, swarm_ctx)
        input_context = self._build_input_context(upstream_output, params)

        for iteration in range(1, config.max_iterations + 1):
            user_prompt = self._build_react_user_prompt(
                input_context, history, swarm_ctx
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # 1. Ask the LLM what to do next.
            try:
                raw_response = await self._call_llm(ctx, config, messages)
            except Exception as e:
                logger.error("Agent LLM call failed: %s", e)
                trace.append({
                    "iteration": iteration,
                    "error": f"LLM call failed: {e}",
                })
                return AgentExecutionResult(
                    output=BaseTaskOutput.failure(f"LLM call failed: {e}"),
                    iterations=iteration,
                    tool_calls=tool_calls,
                    trace=trace,
                    finish_reason="error",
                )

            parsed = self._parse_llm_response(raw_response)
            thought = parsed.get("thought", "")
            action = parsed.get("action", "")
            action_input = parsed.get("action_input") or {}
            final_answer = parsed.get("final_answer")

            step: dict[str, Any] = {
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "final_answer": final_answer,
            }

            # 2. FINISH -> validate + return.
            if self._is_finish(action):
                result_data = self._extract_final_data(
                    final_answer, action_input
                )
                if config.output_schema is not None and not self._validate_output(
                    result_data, config.output_schema
                ):
                    step["observation"] = (
                        "output_schema validation failed"
                    )
                    trace.append(step)
                    return AgentExecutionResult(
                        output=BaseTaskOutput.failure(
                            "Agent output failed schema validation"
                        ),
                        iterations=iteration,
                        tool_calls=tool_calls,
                        trace=trace,
                        finish_reason="error",
                    )
                trace.append(step)
                output = self._make_output(result_data)
                return AgentExecutionResult(
                    output=output,
                    iterations=iteration,
                    tool_calls=tool_calls,
                    trace=trace,
                    finish_reason="completed",
                )

            # 3. TRANSFER (swarm only).
            if self._is_transfer(action) and swarm_ctx is not None:
                target = None
                if isinstance(action_input, dict):
                    target = action_input.get("target")
                agent_map = swarm_ctx["agent_map"]
                if target and target in agent_map:
                    step["transfer_target"] = target
                    trace.append(step)
                    payload = action_input.get("payload") or {}
                    return AgentExecutionResult(
                        output=self._make_output(payload),
                        iterations=iteration,
                        tool_calls=tool_calls,
                        trace=trace,
                        finish_reason="transfer",
                        transfer_target=target,
                    )
                observation = (
                    f"Error: cannot transfer to '{target}'. Available "
                    f"agents: {list(agent_map.keys())}"
                )
                history.append({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                })
                step["observation"] = observation
                trace.append(step)
                continue

            # 3b. v0.6: 解析失败自愈反馈。当 action 为空字符串时（_parse_llm_response
            # 的兜底分支），把"无法解析"反馈塞回 history，让模型下一轮自纠正，
            # 避免重复同一问题直到 max_iterations 后静默失败。
            if not action:
                _parse_feedback = (
                    "ERROR: Your previous response could not be parsed. "
                    "Please respond with a JSON object containing "
                    "'thought', 'action', 'action_input', and optionally "
                    "'final_answer'. Example:\n"
                    '{"thought": "...", "action": "task_name", '
                    '"action_input": {"key": "value"}}\n'
                    'Use "FINISH" as the action to end.'
                )
                history.append({
                    "thought": thought or raw_response[:200],
                    "action": action,
                    "action_input": action_input,
                    "observation": _parse_feedback,
                })
                step["observation"] = _parse_feedback
                step["parse_error"] = True
                trace.append(step)
                continue

            # 4. Tool call. Check the tool-call budget first.
            if tool_calls >= config.max_tool_calls:
                step["observation"] = (
                    f"max_tool_calls ({config.max_tool_calls}) reached"
                )
                trace.append(step)
                return AgentExecutionResult(
                    output=BaseTaskOutput.failure(
                        f"Agent reached max_tool_calls "
                        f"({config.max_tool_calls})"
                    ),
                    iterations=iteration,
                    tool_calls=tool_calls,
                    trace=trace,
                    finish_reason="max_tool_calls",
                )

            observation = await self._invoke_tool(
                config, action, action_input, ctx
            )
            if observation.startswith("OK:"):
                tool_calls += 1

            history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            })
            step["observation"] = observation
            trace.append(step)

        # 5. Iteration budget exhausted.
        return AgentExecutionResult(
            output=BaseTaskOutput.failure(
                f"Agent reached max_iterations ({config.max_iterations})"
            ),
            iterations=config.max_iterations,
            tool_calls=tool_calls,
            trace=trace,
            finish_reason="max_iterations",
        )

    # ------------------------------------------------------------------
    # SUPERVISOR
    # ------------------------------------------------------------------

    async def _execute_supervisor(
        self,
        config: AgentConfig,
        ctx: TaskContext,
        upstream_output: BaseTaskOutput | None,
        params: dict[str, Any],
    ) -> AgentExecutionResult:
        """Supervisor decomposes the task; sub-agents run in parallel."""
        trace: list[dict[str, Any]] = []
        sub_agents = config.supervisor_agents

        if not sub_agents:
            return AgentExecutionResult(
                output=BaseTaskOutput.failure(
                    "SUPERVISOR agent has no supervisor_agents configured"
                ),
                iterations=0,
                tool_calls=0,
                trace=trace,
                finish_reason="error",
            )

        # 1. Supervisor LLM decides which sub-agents to invoke.
        system_prompt = self._build_supervisor_system_prompt(
            config, sub_agents
        )
        user_prompt = self._build_input_context(upstream_output, params)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_response = await self._call_llm(ctx, config, messages)
        except Exception as e:
            logger.error("Supervisor LLM call failed: %s", e)
            return AgentExecutionResult(
                output=BaseTaskOutput.failure(
                    f"Supervisor LLM call failed: {e}"
                ),
                iterations=1,
                tool_calls=0,
                trace=trace,
                finish_reason="error",
            )

        assignments = self._parse_supervisor_response(
            raw_response, sub_agents
        )
        trace.append({
            "phase": "supervisor_decompose",
            "assignments": [
                {"agent_index": i, "sub_input": si}
                for i, (_, si) in enumerate(assignments)
            ],
        })

        # 2. Run sub-agents in parallel (each is a REACT).
        sub_tasks = [
            self._execute_react(ac, ctx, upstream_output, {**params, **sub_in})
            for ac, sub_in in assignments
        ]
        sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)

        # 3. Aggregate.
        aggregated: dict[str, Any] = {}
        total_iterations = 0
        total_tool_calls = 0
        any_failed = False
        for i, res in enumerate(sub_results):
            ac = assignments[i][0]
            name = ac.goal or f"agent_{i}"
            if isinstance(res, Exception):
                aggregated[name] = {"error": str(res)}
                any_failed = True
                # 异常分支也追加 trace 条目，格式与 failure 分支一致
                trace.append({
                    "phase": "sub_agent",
                    "name": name,
                    "iterations": 0,
                    "tool_calls": 0,
                    "finish_reason": "error",
                    "error": str(res),
                    "exception_type": type(res).__name__,
                })
            else:
                aggregated[name] = res.output.data
                total_iterations += res.iterations
                total_tool_calls += res.tool_calls
                trace.append({
                    "phase": "sub_agent",
                    "name": name,
                    "iterations": res.iterations,
                    "tool_calls": res.tool_calls,
                    "finish_reason": res.finish_reason,
                    "output": res.output.data,
                    "sub_trace": res.trace,
                })
                if not res.output.is_success:
                    any_failed = True

        if any_failed:
            output = BaseTaskOutput.failure(
                "One or more sub-agents failed", results=aggregated
            )
        else:
            output = BaseTaskOutput.success(results=aggregated)

        return AgentExecutionResult(
            output=output,
            iterations=total_iterations,
            tool_calls=total_tool_calls,
            trace=trace,
            finish_reason="completed",
        )

    # ------------------------------------------------------------------
    # SWARM
    # ------------------------------------------------------------------

    async def _execute_swarm(
        self,
        config: AgentConfig,
        ctx: TaskContext,
        upstream_output: BaseTaskOutput | None,
        params: dict[str, Any],
    ) -> AgentExecutionResult:
        """Agents hand control to one another until FINISH or budget hit."""
        trace: list[dict[str, Any]] = []
        swarm_agents = config.supervisor_agents

        if not swarm_agents:
            return AgentExecutionResult(
                output=BaseTaskOutput.failure(
                    "SWARM agent has no swarm members configured "
                    "(set supervisor_agents)"
                ),
                iterations=0,
                tool_calls=0,
                trace=trace,
                finish_reason="error",
            )

        # Build the agent name -> config map. The first member starts.
        agent_map: dict[str, AgentConfig] = {}
        order: list[str] = []
        for i, ac in enumerate(swarm_agents):
            name = ac.goal or f"agent_{i}"
            agent_map[name] = ac
            order.append(name)

        current_name = order[0]
        current_agent = agent_map[current_name]
        current_input = upstream_output
        transfers = 0
        total_iterations = 0
        total_tool_calls = 0

        while True:
            swarm_ctx = {"agent_map": agent_map}
            result = await self._execute_react(
                current_agent,
                ctx,
                current_input,
                params,
                swarm_ctx=swarm_ctx,
            )
            total_iterations += result.iterations
            total_tool_calls += result.tool_calls
            trace.append({
                "phase": "swarm_agent",
                "agent": current_name,
                "iterations": result.iterations,
                "tool_calls": result.tool_calls,
                "finish_reason": result.finish_reason,
                "sub_trace": result.trace,
            })

            if result.finish_reason == "transfer":
                if transfers >= config.swarm_max_transfers:
                    return AgentExecutionResult(
                        output=BaseTaskOutput.failure(
                            f"Agent reached swarm_max_transfers "
                            f"({config.swarm_max_transfers})"
                        ),
                        iterations=total_iterations,
                        tool_calls=total_tool_calls,
                        trace=trace,
                        finish_reason="max_transfers",
                    )
                transfers += 1
                current_name = result.transfer_target  # type: ignore[assignment]
                current_agent = agent_map[current_name]
                current_input = result.output
                continue

            # completed / max_iterations / max_tool_calls / error
            return AgentExecutionResult(
                output=result.output,
                iterations=total_iterations,
                tool_calls=total_tool_calls,
                trace=trace,
                finish_reason=result.finish_reason,
            )

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    async def _invoke_tool(
        self,
        config: AgentConfig,
        action: str,
        action_input: Any,
        ctx: TaskContext,
    ) -> str:
        """Invoke a registered Task as an Agent tool.

        Returns an observation string. Observations starting with ``OK:``
        count against the tool-call budget; those starting with ``Error:``
        do not (the tool never ran or raised).
        """
        if not action:
            return "Error: empty action"

        if action not in config.available_tasks:
            return (
                f"Error: task '{action}' is not in available_tasks "
                f"({config.available_tasks})"
            )

        try:
            output = await self._call_tool(action, ctx, action_input)
        except KeyError:
            return f"Error: task '{action}' is not registered"
        except Exception as e:
            return f"Error calling task '{action}': {type(e).__name__}: {e}"

        if output.is_success:
            return "OK: " + json.dumps(output.data, default=str)
        return f"Error: {output.error}"

    async def _call_tool(
        self,
        task_name: str,
        ctx: TaskContext,
        task_input: Any,
    ) -> BaseTaskOutput:
        """Instantiate and run a registered task end-to-end.

        Mirrors the lifecycle used by ``WorkflowExecutor`` for ordinary
        DAG nodes: ``prepare -> execute -> cleanup``.
        """
        if not isinstance(task_input, dict):
            task_input = {}

        task_cls = self._task_registry.get(task_name)
        task = task_cls()
        try:
            await task.prepare(ctx)
            input_obj = task.input_model(**task_input)
            output = await task.execute(ctx, input_obj)
        finally:
            try:
                await task.cleanup(ctx)
            except Exception as e:
                logger.warning(
                    "Cleanup failed for agent tool '%s': %s", task_name, e
                )
        return output

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        ctx: TaskContext,
        config: AgentConfig,
        messages: list[dict[str, str]],
    ) -> str:
        """Call the LLM and return the assistant content string.

        v0.6: 当 ``config.force_json_output`` 为 True 时，透传
        ``response_format={"type": "json_object"}`` 给适配器，
        强制模型按 JSON 格式输出，减少解析失败概率。
        """
        adapter = self._get_adapter(ctx, config)
        call_kwargs: dict[str, Any] = {
            "temperature": config.temperature,
        }
        if getattr(config, "force_json_output", False):
            call_kwargs["response_format"] = {"type": "json_object"}
        response = await adapter.chat(
            messages=messages, **call_kwargs
        )
        return response.get("content", "")

    def _get_adapter(self, ctx: TaskContext, config: AgentConfig) -> Any:
        """Resolve the model adapter, honouring ``config.model_id``.

        Reads the injected ModelManager reference (does not use
        ``object.__setattr__``). When ``config.model_id`` is set and
        differs from the context's, the manager is asked for that
        specific adapter; otherwise the context's default is used.
        """
        if config.model_id:
            manager = getattr(ctx, "_model_manager", None)
            if manager is not None:
                try:
                    return manager.get_adapter(config.model_id)
                except Exception:
                    logger.warning(
                        "Failed to get adapter for model_id='%s'; "
                        "falling back to ctx default",
                        config.model_id,
                    )
        return ctx.get_model_adapter()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_react_system_prompt(
        self,
        config: AgentConfig,
        swarm_ctx: dict[str, Any] | None,
    ) -> str:
        """Build the REACT system prompt (goal + tools + format)."""
        task_desc = self._build_task_descriptions(config)
        prompt = (
            f"You are an autonomous agent. Your goal: {config.goal or '(unspecified)'}\n\n"
            f"Available tasks (call as tools):\n{task_desc}\n\n"
            "At each step, reason about what to do, then either call a "
            "task or finish.\n\n"
            "Respond in YAML format:\n"
            "```yaml\n"
            "thought: <your reasoning>\n"
            "action: <task_name or FINISH>\n"
            "action_input: <JSON-like dict passed to the task, or null "
            "if FINISH>\n"
            "final_answer: <only when action=FINISH; a JSON object or "
            "string with the final result>\n"
            "```\n"
        )
        if swarm_ctx is not None:
            peers = list(swarm_ctx["agent_map"].keys())
            prompt += (
                "\nYou are part of a swarm. You may also hand control "
                f"to another agent: {peers}\n"
                "To hand off, use action: TRANSFER and "
                "action_input: {\"target\": \"<agent_name>\"}\n"
            )
        return prompt

    def _build_react_user_prompt(
        self,
        input_context: str,
        history: list[dict[str, Any]],
        swarm_ctx: dict[str, Any] | None,
    ) -> str:
        """Build the per-iteration user prompt (input + history)."""
        parts = [f"Input:\n{input_context}"]
        parts.append(f"\nHistory:\n{self._format_history(history)}")
        if swarm_ctx is not None:
            parts.append(
                "\nDecide: call a task, TRANSFER to another agent, or "
                "FINISH with a final answer."
            )
        else:
            parts.append("\nWhat should you do next?")
        return "\n".join(parts)

    def _build_task_descriptions(self, config: AgentConfig) -> str:
        """List available tasks with their descriptions (if registered)."""
        if not config.available_tasks:
            return "  (no tasks available — emit FINISH immediately)"
        lines: list[str] = []
        for name in config.available_tasks:
            desc = name
            try:
                task_cls = self._task_registry.get(name)
                d = getattr(task_cls, "description", "") or ""
                desc = f"{name}: {d}" if d else name
            except KeyError:
                desc = f"{name} (WARNING: not registered)"
            lines.append(f"  - {desc}")
        return "\n".join(lines)

    def _build_input_context(
        self,
        upstream_output: BaseTaskOutput | None,
        params: dict[str, Any],
    ) -> str:
        """Serialise upstream output + params into the user prompt body."""
        parts: list[str] = []
        if params:
            parts.append(
                "Params: " + json.dumps(params, default=str)
            )
        if upstream_output is not None:
            data = (
                upstream_output.data
                if isinstance(upstream_output, BaseTaskOutput)
                else upstream_output
            )
            parts.append(
                "Upstream output: " + json.dumps(data, default=str)
            )
        return "\n".join(parts) if parts else "(no input)"

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        """Render the think/act/observe history for the next prompt."""
        if not history:
            return "(none)"
        lines: list[str] = []
        for i, h in enumerate(history, 1):
            obs = h.get("observation", "")
            if len(obs) > 500:
                obs = obs[:500] + "...(truncated)"
            lines.append(f"  Step {i}:")
            lines.append(f"    Thought: {h.get('thought', '')}")
            lines.append(f"    Action: {h.get('action', '')}")
            lines.append(
                f"    Action Input: "
                f"{json.dumps(h.get('action_input') or {}, default=str)}"
            )
            lines.append(f"    Observation: {obs}")
        return "\n".join(lines)

    def _build_supervisor_system_prompt(
        self,
        config: AgentConfig,
        sub_agents: list[AgentConfig],
    ) -> str:
        """Build the supervisor's decomposition prompt."""
        lines: list[str] = []
        for i, ac in enumerate(sub_agents):
            name = ac.goal or f"agent_{i}"
            lines.append(
                f"  {i}: name={name}, "
                f"available_tasks={ac.available_tasks}"
            )
        agent_list = "\n".join(lines)
        return (
            f"You are a supervisor agent. Your goal: {config.goal or '(unspecified)'}\n\n"
            f"You manage {len(sub_agents)} sub-agents:\n{agent_list}\n\n"
            "Given the input, decide which sub-agents to invoke and what "
            "sub-task input to pass each. Respond in JSON:\n"
            "```json\n"
            "{\n"
            '  "assignments": [\n'
            '    {"agent_index": 0, "sub_input": {}},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "```\n"
            "sub_input is merged with the original input; use {} to pass "
            "the original input unchanged. Only invoke relevant sub-agents."
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response into a structured dict.

        Supports two fenced formats and two fallbacks:
            - ```yaml ... ``` block  (parsed with PyYAML, lazy-imported)
            - ```json ... ``` block  (parsed with json)
            - whole-response JSON
            - whole-response YAML
            - plain text containing FINISH

        Returns a dict with keys: ``thought``, ``action``,
        ``action_input``, ``final_answer``.
        """
        if not response or not response.strip():
            return {
                "thought": "",
                "action": _FINISH,
                "action_input": {},
                "final_answer": None,
            }

        # 1. Fenced YAML block.
        m = _YAML_BLOCK_RE.search(response)
        if m:
            parsed = self._safe_yaml_load(m.group(1))
            if isinstance(parsed, dict):
                return self._normalize_parsed(parsed)

        # 2. Fenced JSON block.
        m = _JSON_BLOCK_RE.search(response)
        if m:
            parsed = self._safe_json_load(m.group(1))
            if isinstance(parsed, dict):
                return self._normalize_parsed(parsed)

        # 3. Whole response as JSON.
        parsed = self._safe_json_load(response)
        if isinstance(parsed, dict):
            return self._normalize_parsed(parsed)

        # 4. Whole response as YAML.
        parsed = self._safe_yaml_load(response)
        if isinstance(parsed, dict):
            return self._normalize_parsed(parsed)

        # 5. Plain-text fallback: look for FINISH / TRANSFER keywords.
        upper = response.upper()
        if _FINISH in upper:
            return {
                "thought": response,
                "action": _FINISH,
                "action_input": {},
                "final_answer": response,
            }
        if _TRANSFER in upper:
            return {
                "thought": response,
                "action": _TRANSFER,
                "action_input": {},
                "final_answer": None,
            }

        # 6. Last resort: treat as a thought with no actionable parse.
        return {
            "thought": response,
            "action": "",
            "action_input": {},
            "final_answer": None,
        }

    def _normalize_parsed(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Normalise a parsed dict to the canonical schema."""
        action = parsed.get("action") or parsed.get("task") or ""
        action_input = parsed.get("action_input")
        if action_input is None:
            action_input = parsed.get("input") or {}
        return {
            "thought": parsed.get("thought", ""),
            "action": action,
            "action_input": action_input,
            "final_answer": parsed.get("final_answer"),
        }

    def _parse_supervisor_response(
        self,
        response: str,
        sub_agents: list[AgentConfig],
    ) -> list[tuple[AgentConfig, dict[str, Any]]]:
        """Parse the supervisor's assignment list.

        Falls back to "invoke every sub-agent with the original input"
        if parsing fails.
        """
        default = [(ac, {}) for ac in sub_agents]

        parsed: Any = None
        m = _JSON_BLOCK_RE.search(response)
        if m:
            parsed = self._safe_json_load(m.group(1))
        if not isinstance(parsed, dict):
            parsed = self._safe_json_load(response)

        if not isinstance(parsed, dict) or "assignments" not in parsed:
            return default

        assignments: list[tuple[AgentConfig, dict[str, Any]]] = []
        for a in parsed["assignments"]:
            if not isinstance(a, dict):
                continue
            idx = a.get("agent_index", 0)
            sub_input = a.get("sub_input", {})
            if not isinstance(sub_input, dict):
                sub_input = {}
            if isinstance(idx, int) and 0 <= idx < len(sub_agents):
                assignments.append((sub_agents[idx], sub_input))

        return assignments if assignments else default

    # ------------------------------------------------------------------
    # Output extraction & validation
    # ------------------------------------------------------------------

    def _extract_final_data(
        self,
        final_answer: Any,
        action_input: Any,
    ) -> dict[str, Any]:
        """Extract the agent's final result as a dict."""
        # Dict final_answer is used directly.
        if isinstance(final_answer, dict):
            return final_answer

        # String final_answer: try JSON, else wrap under "answer".
        if isinstance(final_answer, str):
            stripped = final_answer.strip()
            parsed = self._safe_json_load(stripped)
            if isinstance(parsed, dict):
                return parsed
            return {"answer": final_answer}

        # Fall back to action_input, then a default.
        if isinstance(action_input, dict) and action_input:
            return action_input
        return {"answer": "completed"}

    def _make_output(self, data: dict[str, Any]) -> BaseTaskOutput:
        """Build a success ``BaseTaskOutput`` from a dict."""
        return BaseTaskOutput.success(**data)

    def _validate_output(
        self,
        output: Any,
        schema: dict[str, Any],
    ) -> bool:
        """Validate ``output`` against a JSON-Schema subset.

        Supports ``required`` and ``properties.<field>.type``. No
        external ``jsonschema`` dependency.
        """
        if not isinstance(output, dict):
            return False

        required = schema.get("required", []) or []
        for name in required:
            if name not in output:
                return False

        properties = schema.get("properties", {}) or {}
        for field_name, field_schema in properties.items():
            if field_name not in output:
                continue
            if not isinstance(field_schema, dict):
                continue
            expected = field_schema.get("type")
            if expected and not self._check_type(output[field_name], expected):
                return False
        return True

    def _check_type(self, value: Any, expected: str) -> bool:
        """Check ``value`` against a JSON-Schema type name."""
        # bool is a subclass of int in Python; handle it first.
        if expected in ("boolean", "bool"):
            return isinstance(value, bool)
        if expected in ("integer", "int"):
            return isinstance(value, int) and not isinstance(value, bool)
        if expected in ("number",):
            return isinstance(value, (int, float)) and not isinstance(
                value, bool
            )
        type_map = {
            "string": str,
            "str": str,
            "object": dict,
            "dict": dict,
            "array": list,
            "list": list,
            "float": float,
        }
        py_type = type_map.get(expected)
        if py_type is None:
            # Unknown type constraint — be permissive.
            return True
        return isinstance(value, py_type)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_finish(action: str) -> bool:
        return action.strip().upper() == _FINISH

    @staticmethod
    def _is_transfer(action: str) -> bool:
        return action.strip().upper() == _TRANSFER

    @staticmethod
    def _safe_json_load(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _safe_yaml_load(text: str) -> Any:
        try:
            import yaml  # lazy import; PyYAML is a runtime dep
        except Exception:
            return AgentNodeExecutor._fallback_yaml_load(text)
        try:
            return yaml.safe_load(text)
        except Exception:
            return None

    @staticmethod
    def _fallback_yaml_load(text: str) -> Any:
        """Minimal YAML parser for ``key: value`` lines (no PyYAML).

        Handles only the flat mapping used by the REACT prompt; values
        that look like JSON flow mappings are delegated to ``json``.
        """
        result: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value in ("null", "~", ""):
                result[key] = None
            elif value.lower() in ("true", "false"):
                result[key] = value.lower() == "true"
            elif value.startswith("{") or value.startswith("["):
                parsed = AgentNodeExecutor._safe_json_load(value)
                result[key] = parsed if parsed is not None else value
            else:
                # Strip surrounding quotes if present.
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                result[key] = value
        return result

    def __repr__(self) -> str:
        return (
            f"AgentNodeExecutor("
            f"task_registry={self._task_registry!r})"
        )
