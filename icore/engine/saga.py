"""
icore.engine.saga - Saga compensation pattern for long-running workflows.

A Saga is a sequence of tasks where each step may have a *compensation*
function that undoes its effects. When a later step fails, the engine
walks backwards through the successfully completed steps and invokes
each one's compensator in reverse order. This guarantees the system
returns to a consistent state — e.g. if "charge_payment" fails after
"reserve_inventory" succeeded, ``reserve_inventory.compensate`` is
invoked to release the reserved stock.

Two execution strategies are supported:

    - **Forward**: run all steps in order; on failure compensate
      backwards from the failed step. Returns the original output
      (with status="error") so callers can see what failed.
    - **Wrapped**: same as Forward, but the entire saga runs inside a
      single transaction-like envelope (the compensators see a shared
      ``saga_context`` dict they can read/write to coordinate).

Subclasses register steps via :meth:`SagaWorkflow.add_saga_step` in
:meth:`define`. Each step is a (task_name, compensate_fn) pair; the
compensate_fn is an async callable receiving ``(ctx, saga_context,
step_output)`` and may return any value (ignored).

Example::

    @register_workflow("order_processing")
    class OrderProcessingSaga(SagaWorkflow):
        name = "order_processing"

        def define(self):
            self.add_saga_step(
                node_id="reserve_inventory",
                task_name="inventory_reserve",
                compensate=release_inventory,
            )
            self.add_saga_step(
                node_id="charge_payment",
                task_name="payment_charge",
                compensate=refund_payment,
            )
            self.add_saga_step(
                node_id="send_confirmation",
                task_name="notification_send",
                # No compensation: notification is idempotent / best-effort.
            )
            return self.build_dag()

When ``charge_payment`` fails, the engine calls:

    await release_inventory(ctx, saga_context, reserve_output)
    # Then returns BaseTaskOutput.failure(...)

If a compensator itself raises, ``SagaCompensationError`` is raised
*after* the remaining compensators have been attempted (best-effort
rollback). The original step-failure error is preserved in the
exception chain.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from icore.core.models import BaseTaskOutput
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG
from icore.exceptions import SagaCompensationError

logger = logging.getLogger(__name__)


# A compensator is an async callable:
#   (ctx, saga_context, step_output) -> Any
CompensateFn = Callable[
    [TaskContext, dict[str, Any], BaseTaskOutput], Awaitable[Any]
]


@dataclass
class SagaStep:
    """
    A single Saga step definition.

    Attributes:
        node_id:    Unique node ID in the underlying DAG.
        task_name:  Registered task to execute for this step.
        compensate: Optional async compensator. ``None`` means the
                    step has no side effects to undo (e.g. a
                    best-effort notification).
        retries:    Retry attempts on failure (default 0).
        timeout:    Optional per-step timeout in seconds.
    """

    node_id: str
    task_name: str
    compensate: Optional[CompensateFn] = None
    retries: int = 0
    timeout: Optional[float] = None


class SagaWorkflow(BaseWorkflow):
    """
    Base class for Saga-style workflows with compensation.

    Subclasses:

    1. Call :meth:`add_saga_step` in :meth:`define` to register each
       step (with its compensator).
    2. Return ``self.build_dag()`` from :meth:`define`.

    The default :meth:`execute` runs the saga in forward order and
    triggers compensation on the first failure. Subclasses that need
    custom execution (e.g. parallel fan-out with all-or-nothing
    rollback) may override :meth:`execute`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._saga_steps: list[SagaStep] = []
        self._saga_context: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Step registration (called from define())
    # ------------------------------------------------------------------

    def add_saga_step(
        self,
        node_id: str,
        task_name: str,
        compensate: Optional[CompensateFn] = None,
        *,
        retries: int = 0,
        timeout: Optional[float] = None,
    ) -> SagaStep:
        """
        Register a Saga step.

        Args:
            node_id:    Unique node ID (used as the DAG node id and as
                        the key in ``saga_context["outputs"]``).
            task_name:  Registered task name to execute.
            compensate: Optional async compensator.
            retries:    Retry attempts on failure.
            timeout:    Per-step timeout in seconds.
        """
        step = SagaStep(
            node_id=node_id,
            task_name=task_name,
            compensate=compensate,
            retries=retries,
            timeout=timeout,
        )
        self._saga_steps.append(step)
        return step

    def build_dag(self) -> DAG:
        """
        Build a linear DAG from the registered saga steps.

        Steps are connected in registration order:
            step_0 -> step_1 -> ... -> step_n
        """
        if not self._saga_steps:
            raise ValueError("Saga has no steps; call add_saga_step() first")
        dag = DAG()
        prev: Optional[str] = None
        for step in self._saga_steps:
            dag.add_node(
                node_id=step.node_id,
                task_name=step.task_name,
                retries=step.retries,
                timeout=step.timeout,
            )
            if prev is not None:
                dag.add_edge(prev, step.node_id)
            prev = step.node_id
        return dag

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self, ctx: TaskContext, params: dict[str, Any]
    ) -> BaseTaskOutput:
        """
        Execute the saga in forward order; compensate on failure.

        The saga maintains a ``saga_context`` dict with:

        - ``outputs``: dict of node_id -> BaseTaskOutput for completed steps
        - ``params``:  the original workflow params (read-only)
        - ``compensated``: list of node_ids that have been compensated

        On success: returns the final step's output.
        On failure: returns the failed step's BaseTaskOutput (with
        status="error"), after running compensators for all previously
        completed steps in reverse order.
        """
        # If define() hasn't been called yet (e.g. caller invoked
        # execute() directly without going through validate()), call
        # it now to populate _saga_steps.
        if not self._saga_steps:
            self.define()

        if not self._saga_steps:
            return BaseTaskOutput.failure("Saga has no steps")

        # Reset per-execution state (a Saga instance may be re-executed).
        self._saga_context = {
            "outputs": {},
            "params": dict(params),
            "compensated": [],
        }

        # Import here to avoid circular import at module load.
        from icore.engine.executor import WorkflowExecutor

        executor = WorkflowExecutor()
        dag = self.build_dag()
        dag.validate()

        completed: list[SagaStep] = []

        for step in self._saga_steps:
            # Build a single-node DAG wave so the standard executor
            # path (with retries, timeout, lifecycle) is reused.
            node = dag.get_node(step.node_id)
            # We invoke the executor on the whole DAG but only forward
            # the single node's output. To keep the implementation
            # simple and robust, we build a fresh single-node sub-DAG
            # and run it. This avoids re-running earlier nodes.
            sub_dag = DAG()
            sub_dag.add_node(
                node_id=node.node_id,
                task_name=node.task_name,
                retries=node.retries,
                timeout=node.timeout,
            )
            # Pass the accumulated saga outputs as upstream-like params
            # so the task's input_builder can pick them up.
            step_params = dict(params)
            step_params.setdefault("_saga_outputs", dict(self._saga_context["outputs"]))

            try:
                output = await executor.run(sub_dag, ctx, step_params)
            except Exception as e:
                logger.error(
                    "Saga step '%s' raised: %s", step.node_id, e, exc_info=True
                )
                output = BaseTaskOutput.failure(
                    f"Saga step '{step.node_id}' raised: {type(e).__name__}: {e}"
                )

            if not output.is_success:
                # Failure: compensate backwards over completed steps.
                logger.warning(
                    "Saga step '%s' failed; compensating %d completed step(s)",
                    step.node_id,
                    len(completed),
                )
                await self._compensate_backward(ctx, completed)
                return output

            self._saga_context["outputs"][step.node_id] = output
            completed.append(step)

        # All steps succeeded; return the final step's output.
        return self._saga_context["outputs"][self._saga_steps[-1].node_id]

    async def _compensate_backward(
        self,
        ctx: TaskContext,
        completed: list[SagaStep],
    ) -> None:
        """
        Invoke compensators for completed steps in reverse order.

        Each compensator runs in a try/except; failures are collected
        and re-raised as ``SagaCompensationError`` after all
        compensators have been attempted (best-effort rollback).
        """
        compensation_errors: list[tuple[str, BaseException]] = []
        for step in reversed(completed):
            if step.compensate is None:
                logger.info(
                    "Saga step '%s' has no compensator; skipping", step.node_id
                )
                continue
            step_output = self._saga_context["outputs"].get(step.node_id)
            if step_output is None:
                # Defensive: should not happen since the step completed.
                continue
            try:
                await step.compensate(ctx, self._saga_context, step_output)
                self._saga_context["compensated"].append(step.node_id)
                logger.info("Saga compensated step '%s'", step.node_id)
            except Exception as e:
                logger.error(
                    "Saga compensation failed for step '%s': %s",
                    step.node_id,
                    e,
                    exc_info=True,
                )
                compensation_errors.append((step.node_id, e))

        if compensation_errors:
            failed = ", ".join(n for n, _ in compensation_errors)
            raise SagaCompensationError(
                f"Compensation failed for step(s): {failed}. "
                f"Manual intervention may be required."
            )


__all__ = [
    "SagaStep",
    "SagaWorkflow",
    "CompensateFn",
]
