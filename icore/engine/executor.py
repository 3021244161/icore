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
    is marked SKIPPED. A node is skipped if ANY of its incoming
    conditional edges evaluates to False; unconditional edges always
    permit execution.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Optional

from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.task_context import TaskContext
from icore.engine.dag import DAG, DAGNode
from icore.engine.states import TaskState, WorkflowState

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
    ) -> None:
        """
        Initialize the executor.

        Args:
            task_registry:     Optional TaskRegistry instance. If None,
                               the global default TaskRegistry is used.
            workflow_registry: Optional WorkflowRegistry instance. If None,
                               the global default WorkflowRegistry is used.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        dag: DAG,
        ctx: TaskContext,
        params: dict[str, Any],
    ) -> BaseTaskOutput:
        """
        Execute a workflow DAG and return the aggregated result.

        This is the main entry point, called by BaseWorkflow.execute().

        Args:
            dag:    The validated task dependency graph.
            ctx:    Parent TaskContext (has model_manager, db_manager injected).
            params: Workflow input parameters (dict from API request).

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

        # 4. Get execution waves for parallel scheduling
        waves = dag.get_execution_waves()

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

                # If a node failed, mark all downstream nodes as skipped
                if result.node_states[node_id] == TaskState.FAILED:
                    self._mark_downstream_skipped(
                        dag, node_id, result.skipped_nodes
                    )

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

        # Route to sub-workflow or task execution
        if node.is_subworkflow:
            return await self._execute_subworkflow(
                node, parent_ctx, params, upstream_outputs,
                model_manager, db_manager,
            )

        return await self._execute_task(
            node, parent_ctx, params, upstream_outputs,
            result, model_manager, db_manager,
        )

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

        return child_ctx

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

        A node is skipped if:
            1. Any predecessor was skipped or failed (propagation), OR
            2. Any conditional incoming edge's condition evaluates to False

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

        for edge in incoming_edges:
            pred_id = edge.source
            pred_state = result.node_states.get(pred_id, TaskState.PENDING)

            # If predecessor was skipped or failed, skip this node
            if pred_state in (TaskState.SKIPPED, TaskState.FAILED):
                return True

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
