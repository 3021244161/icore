"""
icore.workflows.examples.agent_demo - v0.6 multi-Agent demo workflows.

Two end-to-end examples that embed an Agent node inside a DAG, showing
that Agent nodes and ordinary Task nodes are equal citizens (same
``node_id``, same upstream/downstream data flow) — only the dispatch
differs.

Example 1 — ``data_analysis_agent`` (REACT):
    load_csv_data -> [Agent: agent_decide] -> generate_report

    The agent dynamically decides which analysis tasks to invoke
    (``analyze_trend`` / ``detect_anomalies`` / ``segment_customers``)
    based on the loaded data. The DAG marks the agent node via
    ``metadata={"is_agent": True, "agent_config": ...}`` (the existing
    ``DAGNode`` has no dedicated ``is_agent`` field, so metadata is the
    integration point). The workflow's ``execute()`` override detects the
    marker and routes the node through ``AgentNodeExecutor``; ordinary
    nodes run via the standard task lifecycle.

Example 2 — ``code_review_supervisor`` (SUPERVISOR):
    A supervisor LLM decomposes a code-review request and dispatches
    three sub-agents in parallel — StyleAgent, SecurityAgent, PerfAgent
    — each of which is a REACT agent calling its own check task. The
    supervisor aggregates the sub-agent results.

All leaf tasks are pure Python (no LLM calls) so the workflows are
fully testable offline with a scripted FakeModelAdapter driving only
the Agent's own reasoning.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task, task_registry
from icore.core.task_context import TaskContext
from icore.engine.agent import (
    AgentConfig,
    AgentMode,
    AgentNodeExecutor,
)
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG
from icore.engine.registry import register_workflow

logger = logging.getLogger(__name__)


# ===========================================================================
# Example 1 — data analysis tasks (tools for the REACT agent)
# ===========================================================================

class LoadCsvInput(BaseTaskInput):
    csv_path: str = Field(default="data.csv", description="Path to the CSV file")


@register_task("load_csv_data")
class LoadCsvDataTask(BaseTask):
    """Load CSV rows (deterministic fixture; no real I/O)."""

    name: ClassVar[str] = "load_csv_data"
    description: ClassVar[str] = "Load CSV rows for analysis"
    input_model: ClassVar[type[BaseTaskInput]] = LoadCsvInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: LoadCsvInput) -> BaseTaskOutput:
        # Deterministic fixture so the demo is reproducible offline.
        rows = [
            {"month": "2026-01", "sales": 100},
            {"month": "2026-02", "sales": 120},
            {"month": "2026-03", "sales": 450},
            {"month": "2026-04", "sales": 130},
        ]
        return BaseTaskOutput.success(
            rows=rows,
            row_count=len(rows),
            source=inp.csv_path,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


class AnalyzeTrendInput(BaseTaskInput):
    data: list[dict[str, Any]] = Field(default_factory=list)


@register_task("analyze_trend")
class AnalyzeTrendTask(BaseTask):
    """Analyse the overall sales trend from the loaded rows."""

    name: ClassVar[str] = "analyze_trend"
    description: ClassVar[str] = "Analyse sales trend over time"
    input_model: ClassVar[type[BaseTaskInput]] = AnalyzeTrendInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: AnalyzeTrendInput) -> BaseTaskOutput:
        if not inp.data:
            return BaseTaskOutput.failure("no data to analyse")
        sales = [r.get("sales", 0) for r in inp.data]
        slope = (sales[-1] - sales[0]) / max(len(sales) - 1, 1)
        direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
        return BaseTaskOutput.success(
            trend=direction,
            slope=round(slope, 2),
            sample_size=len(sales),
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


class DetectAnomaliesInput(BaseTaskInput):
    data: list[dict[str, Any]] = Field(default_factory=list)


@register_task("detect_anomalies")
class DetectAnomaliesTask(BaseTask):
    """Detect anomalous rows (sales > 2x the mean)."""

    name: ClassVar[str] = "detect_anomalies"
    description: ClassVar[str] = "Detect anomalous sales rows"
    input_model: ClassVar[type[BaseTaskInput]] = DetectAnomaliesInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: DetectAnomaliesInput) -> BaseTaskOutput:
        if not inp.data:
            return BaseTaskOutput.failure("no data to scan")
        sales = [r.get("sales", 0) for r in inp.data]
        mean = sum(sales) / len(sales)
        anomalies = [
            r for r in inp.data if r.get("sales", 0) > 2 * mean
        ]
        return BaseTaskOutput.success(
            anomalies=anomalies,
            count=len(anomalies),
            mean=round(mean, 2),
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


class SegmentCustomersInput(BaseTaskInput):
    data: list[dict[str, Any]] = Field(default_factory=list)


@register_task("segment_customers")
class SegmentCustomersTask(BaseTask):
    """Segment the loaded rows into high/low value buckets."""

    name: ClassVar[str] = "segment_customers"
    description: ClassVar[str] = "Segment customers by sales value"
    input_model: ClassVar[type[BaseTaskInput]] = SegmentCustomersInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: SegmentCustomersInput) -> BaseTaskOutput:
        if not inp.data:
            return BaseTaskOutput.failure("no data to segment")
        threshold = 200
        high = [r for r in inp.data if r.get("sales", 0) >= threshold]
        low = [r for r in inp.data if r.get("sales", 0) < threshold]
        return BaseTaskOutput.success(
            segments={"high": len(high), "low": len(low)},
            threshold=threshold,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


class GenerateReportInput(BaseTaskInput):
    analysis: dict[str, Any] = Field(default_factory=dict)


@register_task("agent_render_report")
class GenerateReportTask(BaseTask):
    """Render the agent's analysis dict into a textual report."""

    name: ClassVar[str] = "agent_render_report"
    description: ClassVar[str] = "Render analysis results into a report"
    input_model: ClassVar[type[BaseTaskInput]] = GenerateReportInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: GenerateReportInput) -> BaseTaskOutput:
        text = "Data Analysis Report\n"
        text += "=" * 20 + "\n"
        for key, value in inp.analysis.items():
            text += f"- {key}: {value}\n"
        return BaseTaskOutput.success(
            report=text,
            section_count=len(inp.analysis),
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ===========================================================================
# Example 1 — REACT data-analysis workflow
# ===========================================================================

@register_workflow("data_analysis_agent")
class DataAnalysisAgentWorkflow(BaseWorkflow):
    """DAG: load_data -> agent_decide -> generate_report.

    ``agent_decide`` is an Agent node (REACT). Because ``DAGNode`` has no
    dedicated ``is_agent`` field, the node is marked via ``metadata``;
    the workflow's ``execute()`` override detects the marker and routes
    the node through ``AgentNodeExecutor`` while ordinary nodes run the
    standard task lifecycle.
    """

    name: ClassVar[str] = "data_analysis_agent"
    description: ClassVar[str] = (
        "REACT agent that dynamically chooses analysis tasks for "
        "loaded CSV data, then renders a report"
    )

    def define(self) -> DAG:
        dag = DAG()

        dag.add_node("load_data", task_name="load_csv_data")

        # Agent node: the placeholder task_name satisfies DAGNode's
        # non-empty validation; execute() intercepts via metadata.
        # DAG.add_node() captures extra kwargs into ``metadata`` via
        # ``**metadata``, so we pass is_agent / agent_config directly
        # (not wrapped in a metadata={...} dict, which would nest them).
        dag.add_node(
            "agent_decide",
            task_name="_agent_node_placeholder",
            is_agent=True,
            agent_config=AgentConfig(
                mode=AgentMode.REACT,
                goal="分析加载的数据，决定调用哪些 task 完成分析",
                available_tasks=[
                    "analyze_trend",
                    "detect_anomalies",
                    "segment_customers",
                ],
                max_iterations=5,
                max_tool_calls=10,
            ),
        )

        dag.add_node(
            "agent_render_report",
            task_name="agent_render_report",
            input_builder=lambda params, upstream: GenerateReportInput(
                analysis=upstream["agent_decide"].data
                if "agent_decide" in upstream
                else {}
            ),
        )

        dag.add_edge("load_data", "agent_decide")
        dag.add_edge("agent_decide", "agent_render_report")
        return dag

    async def execute(
        self, ctx: TaskContext, params: dict[str, Any]
    ) -> BaseTaskOutput:
        """Walk the DAG in topological order; dispatch agent nodes to
        ``AgentNodeExecutor`` and ordinary nodes through the normal task
        lifecycle."""
        dag = self.define()
        dag.validate()
        agent_executor = AgentNodeExecutor()

        node_outputs: dict[str, BaseTaskOutput] = {}
        for node_id in dag.topological_sort():
            node = dag.get_node(node_id)
            upstream = {
                pid: node_outputs[pid]
                for pid in dag.get_predecessors(node_id)
                if pid in node_outputs
            }

            if node.metadata.get("is_agent"):
                config: AgentConfig = node.metadata["agent_config"]
                upstream_out = next(iter(upstream.values()), None)
                result = await agent_executor.execute(
                    config, ctx, upstream_out, params
                )
                node_outputs[node_id] = result.output
                logger.info(
                    "Agent node '%s' finished: reason=%s, iters=%d, tools=%d",
                    node_id,
                    result.finish_reason,
                    result.iterations,
                    result.tool_calls,
                )
                continue

            node_outputs[node_id] = await self._run_ordinary_node(
                node, ctx, params, upstream
            )

        terminals = dag.get_terminal_nodes()
        if len(terminals) == 1:
            return node_outputs[terminals[0]]
        # Multiple terminal nodes: merge under "results".
        return BaseTaskOutput.success(
            results={nid: node_outputs[nid].data for nid in terminals}
        )

    @staticmethod
    async def _run_ordinary_node(
        node, ctx: TaskContext, params: dict[str, Any], upstream
    ) -> BaseTaskOutput:
        """Run a non-agent DAG node through the standard task lifecycle."""
        task_cls = task_registry.get(node.task_name)
        task = task_cls()
        if node.input_builder is not None:
            inp = node.input_builder(params, upstream)
        else:
            merged = dict(params)
            for out in upstream.values():
                if out.is_success and out.data:
                    merged.update(out.data)
            try:
                inp = task_cls.input_model(**merged)
            except Exception:
                inp = BaseTaskInput(**merged)
        try:
            await task.prepare(ctx)
            return await task.execute(ctx, inp)
        finally:
            try:
                await task.cleanup(ctx)
            except Exception:
                pass


# ===========================================================================
# Example 2 — code-review tasks (tools for the supervisor's sub-agents)
# ===========================================================================

class CodeReviewInput(BaseTaskInput):
    code: str = Field(default="", description="Source code to review")


@register_task("style_check")
class StyleCheckTask(BaseTask):
    """Static style check (looks for long lines)."""

    name: ClassVar[str] = "style_check"
    description: ClassVar[str] = "Check code style (line length)"
    input_model: ClassVar[type[BaseTaskInput]] = CodeReviewInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: CodeReviewInput) -> BaseTaskOutput:
        long_lines = [
            i + 1 for i, line in enumerate(inp.code.splitlines())
            if len(line) > 79
        ]
        return BaseTaskOutput.success(
            issues=long_lines,
            issue_count=len(long_lines),
            verdict="pass" if not long_lines else "fail",
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


@register_task("security_check")
class SecurityCheckTask(BaseTask):
    """Naive security scan (looks for 'eval' / 'exec')."""

    name: ClassVar[str] = "security_check"
    description: ClassVar[str] = "Scan code for unsafe APIs"
    input_model: ClassVar[type[BaseTaskInput]] = CodeReviewInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: CodeReviewInput) -> BaseTaskOutput:
        unsafe = []
        for i, line in enumerate(inp.code.splitlines()):
            if "eval(" in line or "exec(" in line:
                unsafe.append(i + 1)
        return BaseTaskOutput.success(
            unsafe_lines=unsafe,
            issue_count=len(unsafe),
            verdict="pass" if not unsafe else "fail",
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


@register_task("perf_check")
class PerfCheckTask(BaseTask):
    """Naive performance hint (nested loops)."""

    name: ClassVar[str] = "perf_check"
    description: ClassVar[str] = "Flag nested loops as perf hotspots"
    input_model: ClassVar[type[BaseTaskInput]] = CodeReviewInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: CodeReviewInput) -> BaseTaskOutput:
        hotspots = []
        for i, line in enumerate(inp.code.splitlines()):
            if "for " in line and line.lstrip().startswith("for "):
                hotspots.append(i + 1)
        return BaseTaskOutput.success(
            hotspots=hotspots,
            issue_count=len(hotspots),
            verdict="pass" if not hotspots else "review",
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


@register_task("aggregate_review")
class AggregateReviewTask(BaseTask):
    """Combine sub-agent outputs into a single review summary."""

    name: ClassVar[str] = "aggregate_review"
    description: ClassVar[str] = "Aggregate code-review sub-results"
    input_model: ClassVar[type[BaseTaskInput]] = BaseTaskInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: BaseTaskInput) -> BaseTaskOutput:
        results = getattr(inp, "results", None) or {}
        total = sum(
            (v.get("issue_count", 0) if isinstance(v, dict) else 0)
            for v in results.values()
        )
        return BaseTaskOutput.success(
            summary=f"{total} total issues across {len(results)} reviewers",
            total_issues=total,
            reviewers=list(results.keys()),
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ===========================================================================
# Example 2 — SUPERVISOR code-review workflow
# ===========================================================================

@register_workflow("code_review_supervisor")
class CodeReviewSupervisorWorkflow(BaseWorkflow):
    """Supervisor dispatches StyleAgent / SecurityAgent / PerfAgent in
    parallel; results are aggregated.

    The supervisor config's ``supervisor_agents`` carries the three
    sub-agent configs. Each sub-agent is a REACT agent whose
    ``available_tasks`` contains exactly one check task.
    """

    name: ClassVar[str] = "code_review_supervisor"
    description: ClassVar[str] = (
        "SUPERVISOR code review: parallel StyleAgent + SecurityAgent "
        "+ PerfAgent, then aggregate"
    )

    # Sub-agent configs reused by define() and execute().
    _STYLE_AGENT = AgentConfig(
        mode=AgentMode.REACT,
        goal="StyleAgent",
        available_tasks=["style_check"],
        max_iterations=3,
        max_tool_calls=3,
    )
    _SECURITY_AGENT = AgentConfig(
        mode=AgentMode.REACT,
        goal="SecurityAgent",
        available_tasks=["security_check"],
        max_iterations=3,
        max_tool_calls=3,
    )
    _PERF_AGENT = AgentConfig(
        mode=AgentMode.REACT,
        goal="PerfAgent",
        available_tasks=["perf_check"],
        max_iterations=3,
        max_tool_calls=3,
    )

    def _supervisor_config(self) -> AgentConfig:
        return AgentConfig(
            mode=AgentMode.SUPERVISOR,
            goal="评审给定代码：委派 StyleAgent、SecurityAgent、PerfAgent 并行检查",
            available_tasks=[],
            max_iterations=1,
            max_tool_calls=0,
            supervisor_agents=[
                self._STYLE_AGENT,
                self._SECURITY_AGENT,
                self._PERF_AGENT,
            ],
        )

    def define(self) -> DAG:
        dag = DAG()
        dag.add_node(
            "review",
            task_name="_agent_node_placeholder",
            is_agent=True,
            agent_config=self._supervisor_config(),
        )
        dag.add_node("aggregate", task_name="aggregate_review")
        dag.add_edge("review", "aggregate")
        return dag

    async def execute(
        self, ctx: TaskContext, params: dict[str, Any]
    ) -> BaseTaskOutput:
        """Run the supervisor agent, then aggregate the results."""
        dag = self.define()
        dag.validate()
        agent_executor = AgentNodeExecutor()

        # 1. Supervisor agent (decomposes + runs sub-agents in parallel).
        review_result = await agent_executor.execute(
            self._supervisor_config(), ctx, None, params
        )

        # 2. Aggregate task: feed the supervisor's results dict.
        agg_task = AggregateReviewTask()
        agg_input = BaseTaskInput(results=review_result.output.data.get("results", {}))
        try:
            await agg_task.prepare(ctx)
            agg_out = await agg_task.execute(ctx, agg_input)
        finally:
            try:
                await agg_task.cleanup(ctx)
            except Exception:
                pass
        return agg_out
