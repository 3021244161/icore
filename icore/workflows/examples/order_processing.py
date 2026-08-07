"""
icore.workflows.examples.order_processing - Saga example: order processing.

A canonical Saga demo: an order goes through three steps, each with a
compensator. If any step fails, the engine rolls back the completed
steps in reverse order.

Steps:
    1. reserve_inventory:   Reserve stock for the order.
                            Compensate: release the reservation.
    2. charge_payment:      Charge the customer's payment method.
                            Compensate: refund the charge.
    3. send_confirmation:   Email the customer a confirmation.
                            (No compensation — notification is best-effort.)

The tasks are deliberately tiny in-memory fakes so the example can run
end-to-end in tests without any real payment / inventory system.

DAG (linear, built by SagaWorkflow.build_dag()):

    reserve_inventory -> charge_payment -> send_confirmation

On failure of step 2, the saga:

    1. Invokes release_inventory(ctx, saga_context, reserve_output)
    2. Returns BaseTaskOutput.failure("...")

The saga_context dict carries state between forward and compensation
phases, so compensators can inspect what the forward step did.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task
from icore.core.task_context import TaskContext
from icore.engine.registry import register_workflow
from icore.engine.saga import SagaWorkflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A shared in-memory "system" so the saga has visible side effects that
# compensators can undo. In a real deployment these would be external
# services (inventory DB, payment gateway, email provider).
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {
    "inventory_reservations": {},   # order_id -> qty reserved
    "charges": {},                  # order_id -> amount charged
    "notifications_sent": [],       # list of order_ids
}


def reset_state() -> None:
    """Reset the shared side-effect state (used by tests)."""
    _STATE["inventory_reservations"].clear()
    _STATE["charges"].clear()
    _STATE["notifications_sent"].clear()


# ---------------------------------------------------------------------------
# Task input models
# ---------------------------------------------------------------------------

class ReserveInventoryInput(BaseTaskInput):
    order_id: str = Field(description="Unique order ID")
    quantity: int = Field(default=1, ge=1, description="Units to reserve")


class ChargePaymentInput(BaseTaskInput):
    order_id: str = Field(description="Unique order ID")
    amount: float = Field(ge=0.0, description="Amount to charge")
    # When True, the charge fails (used by tests to trigger compensation).
    fail: bool = Field(default=False)


class SendConfirmationInput(BaseTaskInput):
    order_id: str = Field(description="Unique order ID")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@register_task("inventory_reserve")
class ReserveInventoryTask(BaseTask):
    """Reserve inventory for an order (fake in-memory side effect)."""

    name: ClassVar[str] = "inventory_reserve"
    description: ClassVar[str] = "Reserve inventory units for an order"
    input_model: ClassVar[type[BaseTaskInput]] = ReserveInventoryInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(
        self, ctx: TaskContext, inp: ReserveInventoryInput
    ) -> BaseTaskOutput:
        _STATE["inventory_reservations"][inp.order_id] = inp.quantity
        logger.info(
            "Reserved %d unit(s) for order %s", inp.quantity, inp.order_id
        )
        return BaseTaskOutput.success(
            order_id=inp.order_id,
            reserved_quantity=inp.quantity,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


@register_task("payment_charge")
class ChargePaymentTask(BaseTask):
    """Charge the customer's payment method (fake in-memory side effect)."""

    name: ClassVar[str] = "payment_charge"
    description: ClassVar[str] = "Charge the customer's payment method"
    input_model: ClassVar[type[BaseTaskInput]] = ChargePaymentInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(
        self, ctx: TaskContext, inp: ChargePaymentInput
    ) -> BaseTaskOutput:
        if inp.fail:
            return BaseTaskOutput.failure(
                f"Payment charge failed for order {inp.order_id} "
                f"(simulated failure)"
            )
        _STATE["charges"][inp.order_id] = inp.amount
        logger.info(
            "Charged %.2f for order %s", inp.amount, inp.order_id
        )
        return BaseTaskOutput.success(
            order_id=inp.order_id,
            charged_amount=inp.amount,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


@register_task("notification_send")
class SendConfirmationTask(BaseTask):
    """Send a confirmation email (fake in-memory side effect)."""

    name: ClassVar[str] = "notification_send"
    description: ClassVar[str] = "Send order confirmation email"
    input_model: ClassVar[type[BaseTaskInput]] = SendConfirmationInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(
        self, ctx: TaskContext, inp: SendConfirmationInput
    ) -> BaseTaskOutput:
        _STATE["notifications_sent"].append(inp.order_id)
        logger.info("Sent confirmation for order %s", inp.order_id)
        return BaseTaskOutput.success(order_id=inp.order_id, notified=True)

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Compensators (async callables, not registered as tasks)
# ---------------------------------------------------------------------------

async def release_inventory(
    ctx: TaskContext,
    saga_context: dict[str, Any],
    step_output: BaseTaskOutput,
) -> None:
    """Compensator for reserve_inventory: release the reserved stock."""
    order_id = step_output.data.get("order_id")
    if order_id is None:
        return
    removed = _STATE["inventory_reservations"].pop(order_id, None)
    logger.info(
        "Compensated reserve_inventory: released reservation for order %s (was %s)",
        order_id,
        removed,
    )


async def refund_payment(
    ctx: TaskContext,
    saga_context: dict[str, Any],
    step_output: BaseTaskOutput,
) -> None:
    """Compensator for charge_payment: refund the charged amount."""
    order_id = step_output.data.get("order_id")
    if order_id is None:
        return
    removed = _STATE["charges"].pop(order_id, None)
    logger.info(
        "Compensated charge_payment: refunded order %s (was %s)",
        order_id,
        removed,
    )


# ---------------------------------------------------------------------------
# Saga workflow
# ---------------------------------------------------------------------------

@register_workflow("order_processing")
class OrderProcessingSaga(SagaWorkflow):
    """
    Order processing Saga with compensation.

    DAG:
        reserve_inventory -> charge_payment -> send_confirmation

    If charge_payment fails, release_inventory is invoked to undo the
    reservation. If send_confirmation fails (rare), both
    refund_payment and release_inventory are invoked in reverse order.
    """

    name: ClassVar[str] = "order_processing"
    description: ClassVar[str] = (
        "Saga: reserve_inventory -> charge_payment -> send_confirmation "
        "with compensation on failure"
    )

    def define(self):  # type: ignore[override]
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
            compensate=None,
        )
        return self.build_dag()
