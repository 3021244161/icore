"""
icore.workflows.examples.fraud_detection - Event-driven fraud detection workflow.

An event-driven workflow triggered by Kafka order messages. It detects
potentially fraudulent orders by checking risk indicators and generating
an alert report via LLM.

Pipeline:
    1. receive_order:   Parse the incoming Kafka order message
    2. check_risk:      Evaluate risk indicators (amount threshold, frequency)
    3. generate_alert:  Produce a fraud risk assessment report via LLM

DAG:
    receive_order -> check_risk -> generate_alert

This demonstrates:
    - Event-driven workflow pattern (Kafka trigger -> workflow)
    - Non-LLM data parsing + rule-based risk checking
    - LLM-based report generation
    - Input builders passing data between tasks
    - Registration via @register_workflow / @register_task

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.fraud_detection import FraudDetectionWorkflow

    wf = FraudDetectionWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # Inject managers before execution...
    result = await wf.execute(ctx, {
        "order_id": "ORD-001",
        "user_id": "u123",
        "amount": 15000,
        "currency": "CNY",
    })
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG
from icore.engine.registry import register_workflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Input Models
# ---------------------------------------------------------------------------

class ReceiveOrderInput(BaseTaskInput):
    """Input for the order-receiving task (from Kafka message)."""

    order_id: str = Field(description="Unique order identifier")
    user_id: str = Field(description="User who placed the order")
    amount: float = Field(description="Order amount")
    currency: str = Field(default="CNY", description="Currency code")
    timestamp: str = Field(default="", description="Order timestamp (ISO 8601)")
    ip_address: str = Field(default="", description="Client IP address")
    items: list[str] = Field(
        default_factory=list,
        description="List of item IDs in the order",
    )


class CheckRiskInput(BaseTaskInput):
    """Input for the risk-checking task."""

    order_id: str = Field(description="Order identifier")
    user_id: str = Field(description="User who placed the order")
    amount: float = Field(description="Order amount")
    currency: str = Field(default="CNY", description="Currency code")
    timestamp: str = Field(default="", description="Order timestamp")
    ip_address: str = Field(default="", description="Client IP address")
    items: list[str] = Field(
        default_factory=list,
        description="List of item IDs in the order",
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Risk score threshold for flagging as fraudulent",
    )


class GenerateAlertInput(BaseTaskInput):
    """Input for the alert-generation task."""

    order_id: str = Field(description="Order identifier")
    user_id: str = Field(description="User who placed the order")
    amount: float = Field(description="Order amount")
    currency: str = Field(description="Currency code")
    risk_score: float = Field(description="Calculated risk score (0.0-1.0)")
    risk_factors: list[str] = Field(
        description="List of triggered risk factor descriptions",
    )
    is_fraudulent: bool = Field(
        description="Whether the order exceeded the fraud threshold",
    )


# ---------------------------------------------------------------------------
# Task: Receive Order (parse Kafka message)
# ---------------------------------------------------------------------------

@register_task("receive_order")
class ReceiveOrderTask(BaseTask):
    """
    Parses the incoming order message from the Kafka trigger.

    This is a non-LLM task: it normalizes the raw message payload into
    a structured order record for downstream risk analysis.
    """

    name: ClassVar[str] = "receive_order"
    description: ClassVar[str] = "Parse incoming Kafka order message into structured data"
    input_model: ClassVar[type[BaseTaskInput]] = ReceiveOrderInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        """No external resources needed for parsing."""
        pass

    async def execute(
        self, ctx: TaskContext, inp: ReceiveOrderInput
    ) -> BaseTaskOutput:
        """Normalize the order data from the Kafka message."""
        logger.info(
            "Received order %s from user %s (amount=%.2f %s)",
            inp.order_id,
            inp.user_id,
            inp.amount,
            inp.currency,
        )

        return BaseTaskOutput.success(
            order_id=inp.order_id,
            user_id=inp.user_id,
            amount=inp.amount,
            currency=inp.currency,
            timestamp=inp.timestamp,
            ip_address=inp.ip_address,
            items=inp.items,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Check Risk (rule-based)
# ---------------------------------------------------------------------------

@register_task("check_risk")
class CheckRiskTask(BaseTask):
    """
    Evaluates risk indicators for the order using rule-based checks.

    Risk factors checked:
        - Amount exceeds high-value threshold
        - Number of items is unusually high
        - IP address is empty or suspicious
        - Currency mismatch (non-CNY for domestic orders)

    The risk score is a weighted sum of triggered factors, normalized
    to [0.0, 1.0].
    """

    name: ClassVar[str] = "check_risk"
    description: ClassVar[str] = "Evaluate risk indicators for fraud detection"
    input_model: ClassVar[type[BaseTaskInput]] = CheckRiskInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    # Risk factor weights (sum = 1.0)
    _AMOUNT_WEIGHT: ClassVar[float] = 0.4
    _ITEM_COUNT_WEIGHT: ClassVar[float] = 0.2
    _IP_WEIGHT: ClassVar[float] = 0.2
    _CURRENCY_WEIGHT: ClassVar[float] = 0.2

    # Thresholds
    _HIGH_AMOUNT_THRESHOLD: ClassVar[float] = 10000.0
    _HIGH_ITEM_COUNT: ClassVar[int] = 20

    async def prepare(self, ctx: TaskContext) -> None:
        """No external resources needed for rule-based checks."""
        pass

    async def execute(
        self, ctx: TaskContext, inp: CheckRiskInput
    ) -> BaseTaskOutput:
        """Evaluate risk indicators and compute a risk score."""
        risk_factors: list[str] = []
        score = 0.0

        # Factor 1: High-value amount
        if inp.amount >= self._HIGH_AMOUNT_THRESHOLD:
            risk_factors.append(
                f"High-value order: {inp.amount:.2f} {inp.currency} "
                f"(threshold: {self._HIGH_AMOUNT_THRESHOLD})"
            )
            score += self._AMOUNT_WEIGHT

        # Factor 2: Unusually many items
        if len(inp.items) >= self._HIGH_ITEM_COUNT:
            risk_factors.append(
                f"Unusually high item count: {len(inp.items)} "
                f"(threshold: {self._HIGH_ITEM_COUNT})"
            )
            score += self._ITEM_COUNT_WEIGHT

        # Factor 3: Missing or suspicious IP
        if not inp.ip_address:
            risk_factors.append("Missing IP address")
            score += self._IP_WEIGHT

        # Factor 4: Non-CNY currency (domestic fraud heuristic)
        if inp.currency != "CNY":
            risk_factors.append(
                f"Non-domestic currency: {inp.currency}"
            )
            score += self._CURRENCY_WEIGHT

        is_fraudulent = score >= inp.threshold

        logger.info(
            "Risk check for order %s: score=%.2f, factors=%d, fraudulent=%s",
            inp.order_id,
            score,
            len(risk_factors),
            is_fraudulent,
        )

        return BaseTaskOutput.success(
            order_id=inp.order_id,
            user_id=inp.user_id,
            amount=inp.amount,
            currency=inp.currency,
            risk_score=round(score, 4),
            risk_factors=risk_factors,
            is_fraudulent=is_fraudulent,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Generate Alert (LLM)
# ---------------------------------------------------------------------------

@register_task("generate_alert")
class GenerateAlertTask(BaseTask):
    """
    Generates a fraud risk assessment report via the LLM model adapter.

    This task demonstrates LLM usage: it obtains the model adapter from
    TaskContext, sends the risk assessment data with a prompt, and
    produces a structured alert report.
    """

    name: ClassVar[str] = "generate_alert"
    description: ClassVar[str] = "Generate a fraud risk assessment report via LLM"
    input_model: ClassVar[type[BaseTaskInput]] = GenerateAlertInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any  # Model adapter (set in prepare)

    async def prepare(self, ctx: TaskContext) -> None:
        """Obtain the model adapter from context."""
        self._model = ctx.get_model_adapter()
        logger.debug("GenerateAlertTask prepared with model adapter")

    async def execute(
        self, ctx: TaskContext, inp: GenerateAlertInput
    ) -> BaseTaskOutput:
        """Generate a fraud alert report via the LLM."""
        factors_text = (
            "\n".join(f"  - {f}" for f in inp.risk_factors)
            if inp.risk_factors
            else "  (no risk factors triggered)"
        )

        prompt = (
            f"You are a fraud detection analyst. Based on the following "
            f"risk assessment, generate a concise alert report.\n\n"
            f"Order ID: {inp.order_id}\n"
            f"User ID: {inp.user_id}\n"
            f"Amount: {inp.amount:.2f} {inp.currency}\n"
            f"Risk Score: {inp.risk_score:.2f}\n"
            f"Fraudulent: {'YES' if inp.is_fraudulent else 'NO'}\n"
            f"Risk Factors:\n{factors_text}\n\n"
            f"Please provide:\n"
            f"1. A summary of the risk assessment\n"
            f"2. Recommended actions (block, review, or allow)\n"
            f"3. Additional investigation steps if needed"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional fraud detection assistant. "
                    "Generate clear, actionable risk assessment reports."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._model.chat(messages=messages)
            report = response.get("content", "")
            logger.info(
                "Generated alert report for order %s (%d chars)",
                inp.order_id,
                len(report),
            )
            return BaseTaskOutput.success(
                order_id=inp.order_id,
                report=report,
                risk_score=inp.risk_score,
                is_fraudulent=inp.is_fraudulent,
                risk_factors=inp.risk_factors,
            )
        except Exception as e:
            logger.error("Alert generation failed for order %s: %s", inp.order_id, e)
            return BaseTaskOutput.failure(f"Alert generation failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        """Release model adapter reference."""
        self._model = None


# ---------------------------------------------------------------------------
# Workflow: Fraud Detection
# ---------------------------------------------------------------------------

@register_workflow("fraud_detection")
class FraudDetectionWorkflow(BaseWorkflow):
    """
    Event-driven fraud detection workflow.

    Triggered by Kafka ``new_orders`` topic (see config/triggers.yaml).
    Parses the order message, evaluates risk indicators, and generates
    an LLM-based alert report.

    Pipeline:
        1. receive_order:   Parse Kafka order message -> structured order
        2. check_risk:      Evaluate risk factors -> risk score + factors
        3. generate_alert:  LLM generates fraud assessment report

    DAG:
        receive_order -> check_risk -> generate_alert
    """

    name: ClassVar[str] = "fraud_detection"
    description: ClassVar[str] = (
        "Event-driven fraud detection: parse order, check risk, "
        "generate alert report"
    )

    def define(self) -> DAG:
        """Build the receive_order -> check_risk -> generate_alert DAG."""
        dag = DAG()

        # Node 1: Receive and parse the order message
        dag.add_node(
            node_id="receive_order",
            task_name="receive_order",
        )

        # Node 2: Check risk indicators
        # Input builder: takes the parsed order data and constructs
        # the CheckRiskInput with the threshold from params
        dag.add_node(
            node_id="check_risk",
            task_name="check_risk",
            input_builder=lambda params, upstream: CheckRiskInput(
                order_id=upstream["receive_order"].data["order_id"],
                user_id=upstream["receive_order"].data["user_id"],
                amount=upstream["receive_order"].data["amount"],
                currency=upstream["receive_order"].data["currency"],
                timestamp=upstream["receive_order"].data["timestamp"],
                ip_address=upstream["receive_order"].data["ip_address"],
                items=upstream["receive_order"].data["items"],
                threshold=params.get("threshold", 0.8),
            ),
        )

        # Node 3: Generate alert report via LLM
        # Input builder: takes the risk assessment and constructs
        # the GenerateAlertInput
        dag.add_node(
            node_id="generate_alert",
            task_name="generate_alert",
            input_builder=lambda params, upstream: GenerateAlertInput(
                order_id=upstream["check_risk"].data["order_id"],
                user_id=upstream["check_risk"].data["user_id"],
                amount=upstream["check_risk"].data["amount"],
                currency=upstream["check_risk"].data["currency"],
                risk_score=upstream["check_risk"].data["risk_score"],
                risk_factors=upstream["check_risk"].data["risk_factors"],
                is_fraudulent=upstream["check_risk"].data["is_fraudulent"],
            ),
        )

        # Linear dependency chain
        dag.add_edge("receive_order", "check_risk")
        dag.add_edge("check_risk", "generate_alert")

        return dag
