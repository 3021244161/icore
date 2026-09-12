"""
icore.engine.executor - Workflow execution engine.

WorkflowExecutor takes a DAG and executes its tasks in topological order,
managing the full lifecycle of each task (prepare -> execute -> cleanup),
passing outputs between tasks, handling conditional branching, sub-workflow
invocation, retries, and timeouts.

Execution model:
    1. Validate the DAG (cycle detection, structural integrity)
    2. Compute execution waves (groups of nodes whose predecessors are done)
    3. For each wave, execute all non-skipped nodes concurrently (asyncio.gather)
    4. After each wave, evaluate conditional edges to determine which
       successor nodes should be skipped
    5. Track node outputs and states
    6. Return aggregated result from terminal nodes

Sub-workflow support:
    A DAG node with is_subworkflow=True delegates to another registered
    workflow. The executor looks it up in WorkflowRegistry, instantiates it,
    and calls its execute() method with the sub-workflow's parameters.

Conditional branching:
    DAG edges can carry a condition predicate (EdgeCondition). After a
    node completes, each outgoing conditional edge is evaluated against
    the node's output. If the condition returns False, the target node
    is marked SKIPPED.

    Skip semantics (v0.6.x, ICORE-ISSUE-002 —— 支持菱形分支汇合):

    - A node runs when at least one incoming edge is "active" — i.e. its
      predecessor completed AND the edge is unconditional or its condition
      evaluates to True.
    - A node is skipped when NO incoming edge is active (all predecessors
      skipped, or all completed predecessors' conditions are False) —
      AND-join truncation. Linear chains keep the legacy cascade: the
      single skipped predecessor is "all predecessors".
    - A condition-skipped sibling does NOT force-skip a fan-in join
      (diamond merge); it merely contributes no output.
    - A FAILED predecessor still force-skips its downstream cone
      (``_mark_downstream_skipped``), so errors stay loud even when a
      sibling branch completed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.task_context import TaskContext
from icore.engine.dag import DAG, DAGNode
from icore.engine.states import TaskState, WorkflowState
from icore.observability import get_metrics_registry, start_span

if TYPE_CHECKING:
    from icore.core.registry import TaskRegistry
    from icore.engine.registry import WorkflowRegistry

logger = logging.getLogger(__name__)


class WorkflowExecutionResult:
    """
    Holds the complete result of a workflow execution.

    Tracks per-node states, outputs, skipped nodes, and the overall
    workflow state. Used internally by the executor and can be returned
    to callers for detailed inspection.

    Attributes:
        workflow_state:  Final state of the workflow instance.
        node_outputs:    Mapping of node_id -> BaseTaskOutput for completed nodes.
        node_states:     Mapping of node_id -> TaskState for all nodes.
        error:           Error message if the workflow failed.
        skipped_nodes:   Set of node_ids that were skipped (conditional or failed upstream).
    """

    def __init__(self) -> None:
        self.workflow_state: WorkflowState = WorkflowState.PENDING
        self.node_outputs: dict[str, BaseTaskOutput] = {}
        self.node_states: dict[str, TaskState] = {}
        self.error: str | None = None
        self.skipped_nodes: set[str] = set()

    @property
    def is_success(self) -> bool:
        """True if the workflow completed successfully."""
        return self.workflow_state == WorkflowState.COMPLETED

    def get_terminal_output(
        self, terminal_nodes: list[str]
    ) -> BaseTaskOutput:
        """
        Extract the final output from terminal nodes.

        .. note::

            The return structure is **asymmetric** depending on the number
            of terminal nodes that actually produced output. Callers that
            need uniform handling should branch on ``len(terminal_nodes)``
            or normalise the result afterwards:

            - **Single terminal node** (``len(active) == 1``): returns the
              node's own ``BaseTaskOutput`` *unchanged*. ``output.data`` is
              therefore whatever the task produced (e.g. ``{"summary": ...}``)
              and there is **no** ``results`` wrapper.
            - **Multiple terminal nodes** (``len(active) > 1``): returns a
              freshly-built ``BaseTaskOutput`` whose ``data`` is wrapped as
              ``{"results": {node_id_1: <data_1>, node_id_2: <data_2>, ...}}``.
              Each value under ``results`` is the original node's ``data``
              dict, keyed by its node ID.
            - **No terminal nodes** (all skipped, or none produced output):
              returns a ``BaseTaskOutput.failure(...)`` describing the cause.

        Args:
            terminal_nodes: List of terminal node IDs from the DAG.

        Returns:
            BaseTaskOutput containing the aggregated result. See the note
            above for the single-vs-multiple return-shape difference.
        """
        # Filter to only nodes that actually produced output
        active = [
            nid for nid in terminal_nodes if nid in self.node_outputs
        ]

        if not active:
            if self.error:
                return BaseTaskOutput.failure(self.error)
            return BaseTaskOutput.failure(
                "No terminal nodes produced output"
            )

        if len(active) == 1:
            return self.node_outputs[active[0]]

        # Multiple terminal nodes: merge their data under their node_id
        merged_data: dict[str, Any] = {}
        any_failed = False
        for nid in active:
            out = self.node_outputs[nid]
            merged_data[nid] = out.data
            if not out.is_success:
                any_failed = True

        if any_failed:
            return BaseTaskOutput.failure(
                "One or more terminal nodes failed",
                results=merged_data,
            )
        return BaseTaskOutput.success(results=merged_data)


class WorkflowExecutor:
    """
    Executes workflow DAGs in topological order with concurrency.

    The executor is the core runtime component of the workflow engine.
    It takes a validated DAG and a parent TaskContext, then orchestrates
    the execution of each node:

        1. Build the task input from workflow params + upstream outputs
        2. Create a child TaskContext (with unique task_id)
        3. Inject model_manager and db_manager from the parent context
        4. Run the task lifecycle: validate -> prepare -> execute -> cleanup
        5. Handle retries, timeouts, and conditional branching
        6. Collect outputs for downstream nodes

    Nodes in the same "execution wave" (same topological depth) run
    concurrently via asyncio.gather, maximizing throughput for
    independent tasks.

    Attributes:
        _task_registry:     TaskRegistry for looking up task classes.
        _workflow_registry:  WorkflowRegistry for sub-workflow lookup.
    """

    def __init__(
        self,
        task_registry: TaskRegistry | None = None,
        workflow_registry: WorkflowRegistry | None = None,
        *,
        persistence_manager: Any = None,
        dlq: Any = None,
        task_queue: Any = None,
    ) -> None:
        """
        Initialize the executor.

        Args:
            task_registry:     Optional TaskRegistry instance. If None,
                               the global default TaskRegistry is used.
            workflow_registry: Optional WorkflowRegistry instance. If None,
                               the global default WorkflowRegistry is used.
            persistence_manager: Optional WorkflowPersistenceManager (v0.6).
                               When wired AND ``run(..., execution_id=...)``
                               is passed, each node's result is recorded
                               and a checkpoint is saved after every wave.
            dlq:               Optional DeadLetterQueue (v0.6). When wired,
                               nodes that exhaust all retries are enqueued
                               for later replay.
            task_queue:        Optional TaskQueue (v0.6 可观测性). When wired,
                               its depth is exported via the
                               ``icore_queue_depth`` gauge at run() start.
        """
        if task_registry is not None:
            self._task_registry = task_registry
        else:
            from icore.core.registry import task_registry as _default_tr

            self._task_registry = _default_tr

        if workflow_registry is not None:
            self._workflow_registry = workflow_registry
        else:
            from icore.engine.registry import (
                workflow_registry as _default_wr,
            )

            self._workflow_registry = _default_wr

        # v0.6 optional wirings (default None -> no-op, preserves the
        # infrastructure-free test path).
        self._persistence = persistence_manager
        self._dlq = dlq
        self._task_queue = task_queue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        dag: DAG,
        ctx: TaskContext,
        params: dict[str, Any],
        *,
        execution_id: str | None = None,
        workflow_name: str = "",
        checkpoint: dict[str, Any] | None = None,
    ) -> BaseTaskOutput:
        """
        Execute a workflow DAG and return the aggregated result.

        This is the main entry point, called by BaseWorkflow.execute().

        Args:
            dag:    The validated task dependency graph.
            ctx:    Parent TaskContext (has model_manager, db_manager injected).
            params: Workflow input parameters (dict from API request).
            execution_id: Optional v0.6 persistence execution ID. When
                    provided AND ``self._persistence`` is wired, each
                    node's result is recorded and a checkpoint is saved
                    after every wave (enables断点续跑).
            workflow_name: Optional workflow name for DLQ metadata.
            checkpoint: Optional resume checkpoint (v0.6). When provided,
                    nodes listed in ``checkpoint["completed_nodes"]`` are
                    skipped and their state is marked COMPLETED, enabling
                    断点续跑 from a previously failed/paused execution.

        Returns:
            BaseTaskOutput with the aggregated result from terminal nodes.
        """
        # 1. Validate the DAG
        dag.validate()

        # 2. Extract managers from parent context
        model_manager = self._get_manager(ctx, "_model_manager")
        db_manager = self._get_manager(ctx, "_db_manager")

        # 3. Initialize execution result
        result = WorkflowExecutionResult()
        result.workflow_state = WorkflowState.RUNNING

        # 3a. Resume from checkpoint: pre-populate skipped_nodes and
        #     node_states so the wave scheduler skips completed nodes.
        if checkpoint:
            completed = checkpoint.get("completed_nodes", [])
            result.skipped_nodes.update(completed)
            for nid in completed:
                result.node_states[nid] = TaskState.COMPLETED
            logger.info(
                "Resuming from checkpoint: %d completed node(s) skipped",
                len(completed),
            )

        # 4. Get execution waves for parallel scheduling
        waves = dag.get_execution_waves()

        # 可观测性：活跃实例数 +1（结束时在 finally 中 -1）
        try:
            get_metrics_registry().get_gauge(
                "icore_active_instances"
            ).inc()
        except Exception:  # noqa: BLE001 — metrics 不得影响核心逻辑
            pass
        # 可观测性：如有 task_queue，导出队列深度
        if self._task_queue is not None:
            try:
                _qsize = await self._task_queue.size()
                get_metrics_registry().get_gauge(
                    "icore_queue_depth"
                ).set(_qsize, queue_name="default")
            except Exception:  # noqa: BLE001
                pass

        try:
            with start_span(
                "workflow.execute", workflow_name=workflow_name
            ):
                # 5. Execute wave by wave
                for wave in waves:
                    # Filter out skipped nodes
                    executable = [
                        nid for nid in wave if nid not in result.skipped_nodes
                    ]

                    if not executable:
                        continue

                    # Check conditional edges for each node in this wave
                    to_execute: list[str] = []
                    for node_id in executable:
                        if self._should_skip_node(dag, node_id, result):
                            result.skipped_nodes.add(node_id)
                            result.node_states[node_id] = TaskState.SKIPPED
                            logger.debug(
                                "Node '%s' skipped (conditional or failed upstream)",
                                node_id,
                            )
                        else:
                            to_execute.append(node_id)

                    if not to_execute:
                        continue

                    # Execute all nodes in this wave concurrently
                    tasks = [
                        self._execute_node(
                            dag.get_node(nid),
                            dag,
                            ctx,
                            params,
                            result,
                            model_manager,
                            db_manager,
                        )
                        for nid in to_execute
                    ]
                    outputs = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    for node_id, output in zip(to_execute, outputs):
                        if isinstance(output, Exception):
                            logger.error(
                                "Node '%s' raised exception: %s", node_id, output
                            )
                            result.node_outputs[node_id] = BaseTaskOutput.failure(
                                str(output)
                            )
                            result.node_states[node_id] = TaskState.FAILED
                        else:
                            result.node_outputs[node_id] = output
                            result.node_states[node_id] = (
                                TaskState.COMPLETED
                                if output.is_success
                                else TaskState.FAILED
                            )

                        # v0.6: record per-node execution + enqueue failed nodes.
                        node_state = result.node_states[node_id]
                        node_obj = dag.get_node(node_id)
                        task_name = getattr(node_obj, "task_name", "") or ""
                        await self._record_node(
                            execution_id=execution_id,
                            node_id=node_id,
                            task_name=task_name,
                            state=node_state,
                            output=result.node_outputs.get(node_id),
                            workflow_name=workflow_name,
                            ctx=ctx,
                        )

                        # If a node failed, mark all downstream nodes as skipped
                        if result.node_states[node_id] == TaskState.FAILED:
                            self._mark_downstream_skipped(
                                dag, node_id, result.skipped_nodes
                            )

                    # v0.6: save checkpoint after each wave (enables resume).
                    await self._save_checkpoint(execution_id, result)

                # 6. Determine final workflow state
                any_failed = any(
                    s == TaskState.FAILED for s in result.node_states.values()
                )
                if any_failed:
                    result.workflow_state = WorkflowState.FAILED
                    failed_nodes = [
                        nid
                        for nid, s in result.node_states.items()
                        if s == TaskState.FAILED
                    ]
                    result.error = (
                        f"Workflow failed: nodes {failed_nodes} did not complete"
                    )
                    logger.warning("Workflow failed: %s", failed_nodes)
                else:
                    result.workflow_state = WorkflowState.COMPLETED
                    logger.info("Workflow completed successfully")

                # 7. Return terminal output
                return result.get_terminal_output(dag.get_terminal_nodes())
        finally:
            # 可观测性：活跃实例数 -1
            try:
                get_metrics_registry().get_gauge(
                    "icore_active_instances"
                ).dec()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------

    async def _execute_node(
        self,
        node: DAGNode,
        dag: DAG,
        parent_ctx: TaskContext,
        params: dict[str, Any],
        result: WorkflowExecutionResult,
        model_manager: Any,
        db_manager: Any,
    ) -> BaseTaskOutput:
        """
        Execute a single DAG node (task or sub-workflow).

        Handles:
            - Input building (custom input_builder or auto-merge)
            - TaskContext creation and manager injection
            - Lifecycle: validate -> prepare -> execute -> cleanup
            - Retries on failure
            - Per-node timeout

        Args:
            node:          The DAGNode to execute.
            dag:           The full DAG (for predecessor lookups).
            parent_ctx:    Parent TaskContext (source of managers).
            params:        Workflow input parameters.
            result:        Accumulated execution result (for upstream outputs).
            model_manager: ModelManager instance (from parent context).
            db_manager:    DBManager instance (from parent context).

        Returns:
            BaseTaskOutput from the task or sub-workflow.
        """
        result.node_states[node.node_id] = TaskState.PENDING

        # Collect upstream outputs for input building
        upstream_outputs = self._collect_upstream_outputs(
            dag, node.node_id, result
        )

        # 可观测性：节点级 tracer span + 执行耗时 histogram
        _node_start = time.monotonic()
        _task_name = getattr(node, "task_name", "") or ""
        _model_id = getattr(node, "model_id", "") or ""
        try:
            with start_span(
                f"node.{node.node_id}",
                node_id=node.node_id,
                task_name=_task_name,
                model_id=_model_id,
            ):
                # Route to agent, sub-workflow or task execution
                if getattr(node, "is_agent", False):
                    return await self._execute_agent(
                        node, parent_ctx, params, upstream_outputs,
                        model_manager, db_manager,
                    )

                if node.is_subworkflow:
                    return await self._execute_subworkflow(
                        node, parent_ctx, params, upstream_outputs,
                        model_manager, db_manager,
                    )

                return await self._execute_task(
                    node, parent_ctx, params, upstream_outputs,
                    result, model_manager, db_manager,
                )
        finally:
            # 可观测性：记录节点执行耗时（task_name 作为标签）
            try:
                _duration = time.monotonic() - _node_start
                get_metrics_registry().get_histogram(
                    "icore_task_duration_seconds"
                ).observe(_duration, task_name=_task_name)
            except Exception:  # noqa: BLE001 — metrics 不得影响核心逻辑
                pass

    async def _execute_task(
        self,
        node: DAGNode,
        parent_ctx: TaskContext,
        params: dict[str, Any],
        upstream_outputs: dict[str, BaseTaskOutput],
        result: WorkflowExecutionResult,
        model_manager: Any,
        db_manager: Any,
    ) -> BaseTaskOutput:
        """
        Execute a regular task node with full lifecycle management.

        Implements retry logic: if execute() fails and retries > 0,
        the task is re-instantiated and re-executed up to retries times.
        """
        # 保护：is_agent=True 的节点不应落入普通 task 路径。
        if getattr(node, "is_agent", False):
            return BaseTaskOutput.failure(
                f"Node '{node.node_id}' is an Agent node; it must be "
                f"routed to AgentNodeExecutor (engine does this "
                f"automatically). Falling back to task execution is "
                f"not allowed."
            )

        # Look up the task class
        try:
            task_cls = self._task_registry.get(node.task_name)
        except KeyError as e:
            return BaseTaskOutput.failure(
                f"Task '{node.task_name}' is not registered: {e}"
            )

        # Build the input
        task_input = self._build_input(
            task_cls, node, params, upstream_outputs
        )

        # Retry loop
        last_error: str | None = None
        max_attempts = node.retries + 1

        for attempt in range(1, max_attempts + 1):
            logger.debug(
                "Executing task '%s' (node='%s', attempt %d/%d)",
                node.task_name,
                node.node_id,
                attempt,
                max_attempts,
            )

            # Fresh task instance for each attempt
            task = task_cls()

            # Create child TaskContext
            node_ctx = self._create_node_context(
                node, parent_ctx, model_manager, db_manager
            )

            try:
                # 1. Validate input
                if not task.validate(task_input):
                    last_error = (
                        f"Validation failed for task '{node.task_name}'"
                    )
                    result_output = BaseTaskOutput.failure(last_error)
                    if attempt < max_attempts:
                        logger.warning(
                            "Validation failed (attempt %d), retrying...",
                            attempt,
                        )
                        continue
                    return result_output

                # 2. Prepare
                result.node_states[node.node_id] = TaskState.RUNNING
                await task.prepare(node_ctx)

                # 3. Execute (with optional timeout)
                if node.timeout is not None:
                    result_output = await asyncio.wait_for(
                        task.execute(node_ctx, task_input),
                        timeout=node.timeout,
                    )
                else:
                    result_output = await task.execute(
                        node_ctx, task_input
                    )

                # 4. Cleanup (always called, like try/finally)
                try:
                    await task.cleanup(node_ctx)
                except Exception as cleanup_err:
                    logger.warning(
                        "Cleanup failed for task '%s': %s",
                        node.task_name,
                        cleanup_err,
                    )

                return result_output

            except asyncio.TimeoutError:
                last_error = (
                    f"Task '{node.task_name}' timed out after "
                    f"{node.timeout}s"
                )
                logger.warning(last_error)
                # Run cleanup even on timeout
                try:
                    await task.cleanup(node_ctx)
                except Exception:
                    pass

                if attempt < max_attempts:
                    logger.info("Retrying after timeout (attempt %d)...", attempt)
                    continue

            except asyncio.CancelledError:
                result.node_states[node.node_id] = TaskState.CANCELLED
                try:
                    await task.cleanup(node_ctx)
                except Exception:
                    pass
                raise

            except Exception as e:
                last_error = (
                    f"Task '{node.task_name}' failed: {type(e).__name__}: {e}"
                )
                logger.error(last_error, exc_info=True)
                # Run cleanup even on failure
                try:
                    await task.cleanup(node_ctx)
                except Exception:
                    pass

                if attempt < max_attempts:
                    logger.info("Retrying after error (attempt %d)...", attempt)
                    continue

        # All retries exhausted
        return BaseTaskOutput.failure(last_error or "Unknown error")

    async def _execute_agent(
        self,
        node: DAGNode,
        parent_ctx: TaskContext,
        params: dict[str, Any],
        upstream_outputs: dict[str, BaseTaskOutput],
        model_manager: Any,
        db_manager: Any,
    ) -> BaseTaskOutput:
        """
        Execute an Agent node through ``AgentNodeExecutor``.

        v0.6: Agent 节点是一等公民。``_execute_node`` 检测
        ``node.is_agent`` 后路由到这里，走与普通节点相同的引擎
        能力（child context / retry / timeout / persistence / DLQ /
        metrics）。上游输出取第一个前驱节点的结果作为
        ``upstream_output``（与 agent_demo 原实现一致）。
        """
        from icore.engine.agent import AgentNodeExecutor

        config = getattr(node, "agent_config", None)
        if config is None:
            config = node.metadata.get("agent_config") if node.metadata else None
        if config is None:
            return BaseTaskOutput.failure(
                f"Agent node '{node.node_id}' has no agent_config"
            )

        upstream_out = next(iter(upstream_outputs.values()), None)
        executor = AgentNodeExecutor(task_registry=self._task_registry)

        # Retry loop (same semantics as _execute_task: retries on failure).
        last_error: str | None = None
        max_attempts = node.retries + 1

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "Executing agent node '%s' (mode=%s, attempt %d/%d)",
                node.node_id,
                config.mode.value,
                attempt,
                max_attempts,
            )
            node_ctx = self._create_node_context(
                node, parent_ctx, model_manager, db_manager
            )
            try:
                agent_result = await executor.execute(
                    config,
                    node_ctx,
                    upstream_output=upstream_out,
                    params=params,
                    node_timeout=node.timeout,
                )
                output = agent_result.output
                if output is not None and output.is_success:
                    return output
                last_error = (
                    output.error
                    if output is not None and output.error
                    else f"Agent node '{node.node_id}' failed "
                    f"(reason={agent_result.finish_reason})"
                )
                if attempt < max_attempts:
                    logger.warning(
                        "Agent node '%s' failed (attempt %d), retrying: %s",
                        node.node_id,
                        attempt,
                        last_error,
                    )
                    continue
                return BaseTaskOutput.failure(last_error)
            except asyncio.TimeoutError:
                last_error = (
                    f"Agent node '{node.node_id}' timed out after "
                    f"{node.timeout}s"
                )
                logger.warning(last_error)
                if attempt < max_attempts:
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = (
                    f"Agent node '{node.node_id}' failed: "
                    f"{type(e).__name__}: {e}"
                )
                logger.error(last_error, exc_info=True)
                if attempt < max_attempts:
                    continue

        return BaseTaskOutput.failure(last_error or "Unknown error")

    async def _execute_subworkflow(
        self,
        node: DAGNode,
        parent_ctx: TaskContext,
        params: dict[str, Any],
        upstream_outputs: dict[str, BaseTaskOutput],
        model_manager: Any,
        db_manager: Any,
    ) -> BaseTaskOutput:
        """
        Execute a sub-workflow node.

        Looks up the workflow class from WorkflowRegistry, builds its
        input parameters from upstream outputs, and delegates execution
        to the sub-workflow's execute() method (which in turn calls
        this executor recursively).
        """
        try:
            wf_cls = self._workflow_registry.get(node.workflow_name)
        except KeyError as e:
            return BaseTaskOutput.failure(
                f"Sub-workflow '{node.workflow_name}' is not registered: {e}"
            )

        # Build sub-workflow params from upstream outputs
        sub_params = self._build_subworkflow_params(
            params, upstream_outputs
        )

        # Instantiate and execute
        wf = wf_cls()

        # Create child context for the sub-workflow
        sub_ctx = self._create_node_context(
            node, parent_ctx, model_manager, db_manager
        )

        logger.info(
            "Invoking sub-workflow '%s' (node='%s')",
            node.workflow_name,
            node.node_id,
        )

        try:
            return await wf.execute(sub_ctx, sub_params)
        except Exception as e:
            logger.error(
                "Sub-workflow '%s' failed: %s",
                node.workflow_name,
                e,
                exc_info=True,
            )
            return BaseTaskOutput.failure(
                f"Sub-workflow '{node.workflow_name}' failed: {e}"
            )

    # ------------------------------------------------------------------
    # Input building
    # ------------------------------------------------------------------

    def _build_input(
        self,
        task_cls: type,
        node: DAGNode,
        params: dict[str, Any],
        upstream_outputs: dict[str, BaseTaskOutput],
    ) -> BaseTaskInput:
        """
        Build the task input from workflow params and upstream outputs.

        If the node has a custom input_builder, use it. Otherwise,
        auto-merge: start with workflow params, then overlay upstream
        outputs' data, and construct the task's input_model.

        Args:
            task_cls:        The BaseTask subclass (has input_model class attr).
            node:            The DAGNode (may have input_builder).
            params:          Workflow-level input parameters.
            upstream_outputs:  Mapping of predecessor node_id -> output.

        Returns:
            A BaseTaskInput (or subclass) instance.
        """
        if node.input_builder is not None:
            return node.input_builder(params, upstream_outputs)

        # Auto-merge: workflow params as base, upstream data overlays
        merged: dict[str, Any] = dict(params)
        for pred_id, output in upstream_outputs.items():
            if output.is_success and output.data:
                merged.update(output.data)

        # Construct the task's input model
        input_model = getattr(task_cls, "input_model", BaseTaskInput)
        try:
            return input_model(**merged)
        except Exception as e:
            logger.warning(
                "Failed to construct input model %s from merged params: %s. "
                "Falling back to BaseTaskInput.",
                input_model.__name__,
                e,
            )
            return BaseTaskInput(**merged)

    def _build_subworkflow_params(
        self,
        params: dict[str, Any],
        upstream_outputs: dict[str, BaseTaskOutput],
    ) -> dict[str, Any]:
        """
        Build parameters for a sub-workflow invocation.

        Merges workflow-level params with upstream outputs' data.
        """
        merged = dict(params)
        for pred_id, output in upstream_outputs.items():
            if output.is_success and output.data:
                merged.update(output.data)
        return merged

    # ------------------------------------------------------------------
    # Context creation
    # ------------------------------------------------------------------

    def _create_node_context(
        self,
        node: DAGNode,
        parent_ctx: TaskContext,
        model_manager: Any,
        db_manager: Any,
    ) -> TaskContext:
        """
        Create a child TaskContext for a node execution.

        The child context inherits workflow_id, callback_url, stream,
        and metadata from the parent, but gets:
            - A unique task_id (parent_task_id:node_id)
            - model_id from node config (or parent's)

        v0.5: Also propagates vectorstore / graphstore / media_processor /
        lock / circuit_breaker from the parent context so that tasks
        using these new components can access them.
        """
        child_task_id = f"{parent_ctx.task_id}:{node.node_id}"

        child_ctx = TaskContext(
            task_id=child_task_id,
            workflow_id=parent_ctx.workflow_id,
            model_id=node.model_id or parent_ctx.model_id,
            callback_url=parent_ctx.callback_url,
            stream=parent_ctx.stream,
            metadata=dict(parent_ctx.metadata),
        )

        # Inject managers from parent
        if model_manager is not None:
            child_ctx.set_model_manager(model_manager)
        if db_manager is not None:
            child_ctx.set_db_manager(db_manager)

        # v0.5: Propagate new injectable components from parent context.
        if parent_ctx.has_vectorstore():
            child_ctx.set_vectorstore(parent_ctx.get_vectorstore())
        if parent_ctx.has_graphstore():
            child_ctx.set_graphstore(parent_ctx.get_graphstore())
        if parent_ctx.has_media_processor():
            child_ctx.set_media_processor(parent_ctx.get_media_processor())
        if parent_ctx.has_lock():
            child_ctx.set_lock(parent_ctx.get_lock())
        if parent_ctx.has_circuit_breaker():
            child_ctx.set_circuit_breaker(parent_ctx.get_circuit_breaker())
        # v0.6: Propagate objectstore so multi-node workflows (e.g.
        # report_export's upload_to_store node) can access it.
        if parent_ctx.has_objectstore():
            child_ctx.set_objectstore(parent_ctx.get_objectstore())

        return child_ctx

    # ------------------------------------------------------------------
    # v0.6 persistence + DLQ helpers (no-op when not wired)
    # ------------------------------------------------------------------

    async def _record_node(
        self,
        *,
        execution_id: str | None,
        node_id: str,
        task_name: str,
        state: TaskState,
        output: BaseTaskOutput | None,
        workflow_name: str,
        ctx: TaskContext,
    ) -> None:
        """Record a node's execution to persistence + enqueue on failure.

        Both persistence and DLQ are optional wirings; when not wired this
        method is a no-op, preserving the infrastructure-free test path.
        """
        # Map executor TaskState -> persistence ExecutionStatus string.
        status_map = {
            TaskState.COMPLETED: "completed",
            TaskState.FAILED: "failed",
            TaskState.SKIPPED: "skipped",
            TaskState.CANCELLED: "cancelled",
            TaskState.RUNNING: "running",
            TaskState.PENDING: "pending",
        }
        status = status_map.get(state, "unknown")

        # 1. Record to persistence backend.
        if self._persistence is not None and execution_id:
            try:
                output_data = (
                    output.data if output is not None else None
                )
                error_msg = (
                    output.error
                    if output is not None and not output.is_success
                    else None
                )
                await self._persistence.record_task_execution(
                    execution_id=execution_id,
                    node_id=node_id,
                    task_name=task_name,
                    status=status,
                    output=output_data,
                    error=error_msg,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "persistence record_task_execution failed for "
                    "node '%s': %s",
                    node_id,
                    e,
                )

        # 2. Enqueue to DLQ when the node ultimately failed.
        if (
            self._dlq is not None
            and state == TaskState.FAILED
            and output is not None
            and not output.is_success
        ):
            try:
                await self._dlq.enqueue(
                    workflow_name=workflow_name or ctx.workflow_id,
                    task_id=ctx.task_id,
                    node_id=node_id,
                    task_name=task_name,
                    params=None,
                    error=output.error or "unknown error",
                    metadata={"execution_id": execution_id},
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "DLQ enqueue failed for node '%s': %s", node_id, e
                )

    async def _save_checkpoint(
        self,
        execution_id: str | None,
        result: WorkflowExecutionResult,
    ) -> None:
        """Save a checkpoint after each wave (enables resume)."""
        if self._persistence is None or not execution_id:
            return
        try:
            checkpoint = {
                "completed_nodes": [
                    nid
                    for nid, s in result.node_states.items()
                    if s == TaskState.COMPLETED
                ],
                "failed_nodes": [
                    nid
                    for nid, s in result.node_states.items()
                    if s == TaskState.FAILED
                ],
                "skipped_nodes": sorted(result.skipped_nodes),
                "workflow_state": result.workflow_state.value
                if hasattr(result.workflow_state, "value")
                else str(result.workflow_state),
            }
            await self._persistence.save_checkpoint(execution_id, checkpoint)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("save_checkpoint failed: %s", e)

    # ------------------------------------------------------------------
    # Conditional branching & skip propagation
    # ------------------------------------------------------------------

    def _should_skip_node(
        self,
        dag: DAG,
        node_id: str,
        result: WorkflowExecutionResult,
    ) -> bool:
        """
        Check if a node should be skipped based on conditional edges.

        Skip semantics (v0.6.x, ICORE-ISSUE-002 —— 支持菱形分支汇合):

            1. Any predecessor FAILED -> skip（失败传播保持"响亮"：另一条
               分支成功也不掩盖错误；通常已由 ``_mark_downstream_skipped``
               传递性预标记，此处为防御性兜底）。
            2. ALL predecessors skipped（没有一个完成）-> skip（AND-join
               截断：没有任何输入路径。线性链上唯一前驱被跳过即"全部
               被跳过"，与旧语义兼容）。
            3. 在完成的前驱中：至少一条入边无条件或条件为 True ->
               执行。**被条件跳过的兄弟分支不会强制跳过汇合节点**
               （菱形汇合），它只是不贡献输出。

        Unconditional edges (condition=None) always permit execution.
        A node with at least one unconditional edge from a completed
        predecessor is NOT skipped, even if other conditional edges
        evaluate to False.
        """
        incoming_edges = dag.get_edges_to(node_id)

        if not incoming_edges:
            # No predecessors -> start node, never skip
            return False

        has_unconditional_pass = False
        has_conditional_fail = False
        has_completed_pred = False
        has_failed_pred = False

        for edge in incoming_edges:
            pred_id = edge.source
            pred_state = result.node_states.get(pred_id, TaskState.PENDING)

            if pred_state == TaskState.FAILED:
                # ICORE-ISSUE-002: 失败仍然传播，即使其他前驱已成功完成
                has_failed_pred = True
            elif pred_state == TaskState.SKIPPED:
                # ICORE-ISSUE-002: 条件未选中的分支不再强制跳过后继——
                # 它仅仅不贡献输出。是否跳过由"是否存在已完成前驱"
                # （AND-join）统一判定，见下方。
                pass
            else:
                # Predecessor completed (or defensively PENDING)
                has_completed_pred = True

                # Check conditional edges
                if edge.condition is not None:
                    pred_output = result.node_outputs.get(pred_id)
                    if pred_output is not None:
                        try:
                            if edge.condition(pred_output):
                                has_unconditional_pass = True
                            else:
                                has_conditional_fail = True
                        except Exception as e:
                            logger.warning(
                                "Condition evaluation failed on edge "
                                "%s -> %s: %s. Treating as not-matching.",
                                edge.source,
                                edge.target,
                                e,
                            )
                            has_conditional_fail = True
                else:
                    # Unconditional edge from a completed predecessor
                    has_unconditional_pass = True

        # Failure propagation stays loud regardless of sibling successes.
        if has_failed_pred:
            return True

        # AND-join truncation: no predecessor produced output -> no input
        # path -> skip. (Linear chains: the single skipped predecessor is
        # "all predecessors", preserving legacy cascade behaviour.)
        if not has_completed_pred:
            return True

        # If there's at least one unconditional pass, don't skip
        if has_unconditional_pass:
            return False

        # All edges are conditional and at least one failed
        return has_conditional_fail

    def _mark_downstream_skipped(
        self,
        dag: DAG,
        failed_node_id: str,
        skipped: set[str],
    ) -> None:
        """
        Recursively mark all downstream nodes as skipped.

        When a node fails, all of its transitive successors are
        skipped because their input depends on the failed node.
        """
        to_process = list(dag.get_successors(failed_node_id))
        while to_process:
            nid = to_process.pop(0)
            if nid not in skipped:
                skipped.add(nid)
                to_process.extend(dag.get_successors(nid))

    # ------------------------------------------------------------------
    # Upstream output collection
    # ------------------------------------------------------------------

    def _collect_upstream_outputs(
        self,
        dag: DAG,
        node_id: str,
        result: WorkflowExecutionResult,
    ) -> dict[str, BaseTaskOutput]:
        """
        Collect outputs from all predecessor nodes.

        Args:
            dag:     The full DAG.
            node_id:  The node whose predecessors' outputs are needed.
            result:  The execution result holding node outputs.

        Returns:
            Mapping of predecessor node_id -> BaseTaskOutput.
        """
        predecessors = dag.get_predecessors(node_id)
        outputs: dict[str, BaseTaskOutput] = {}
        for pred_id in predecessors:
            if pred_id in result.node_outputs:
                outputs[pred_id] = result.node_outputs[pred_id]
        return outputs

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _get_manager(ctx: TaskContext, attr: str) -> Any:
        """
        Safely extract a manager reference from a TaskContext.

        Uses getattr with a fallback to None, since the private
        attributes may not be set if the parent context was created
        without manager injection (e.g., in tests).
        """
        try:
            return getattr(ctx, attr, None)
        except AttributeError:
            return None

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return (
            f"WorkflowExecutor("
            f"task_registry={self._task_registry!r}, "
            f"workflow_registry={self._workflow_registry!r})"
        )
