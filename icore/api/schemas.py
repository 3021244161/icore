"""
icore.api.schemas - Pydantic v2 request/response models for the API layer.

These schemas define the contract between external callers and the icore
platform. They are automatically documented in the Swagger/OpenAPI page.

Three primary schemas:
    - InvokeRequest:  The main request body for POST /invoke
    - InvokeResponse: The response for non-streaming invocations
    - HealthResponse: The response for GET /health

Design notes:
    - extra="allow": Accepts extra fields so future extensions don't
      break existing clients.
    - task_id is optional: if not provided, the API auto-generates a UUID.
    - params is a free-form dict: the workflow validates it against its
      own input schema at runtime.
    - metadata is free-form: used for tracing, user_id, etc.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class InvokeRequest(BaseModel):
    """
    Request body for POST /invoke - the main API interface.

    This single endpoint handles all workflow invocations. The caller
    specifies which workflow to run, passes its parameters, and
    optionally configures streaming, callback, and model selection.

    Attributes:
        workflow_name: Name of the registered workflow to invoke.
        params:        Workflow input parameters (validated by the
                       workflow's own input schema at runtime).
        task_id:        Unique task instance ID. Auto-generated (UUID)
                        if not provided by the caller.
        callback_url:  URL to POST results to upon completion. When set,
                       the API returns immediately with status="running"
                       and delivers results asynchronously.
        model_id:       Explicit LLM model ID. If None, the workflow
                       uses ModelRouter for auto-routing.
        stream:         If True, results are streamed via SSE. If False,
                       the API waits for completion and returns JSON.
        metadata:       Free-form metadata (trace_id, user_id, etc.).
        resume_from:    断点续跑的 execution_id（v0.6）。传入后从 checkpoint
                       恢复执行，跳过已完成节点。需要 persistence 已接线。
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    workflow_name: str = Field(
        ...,
        description="Name of the registered workflow to invoke",
        min_length=1,
        examples=["document_summary"],
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow input parameters",
        examples=[{"document": "Long text to summarize..."}],
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Unique task instance ID (auto-generated if not provided)",
        examples=["task-550e8400-e29b-41d4-a716-446655440000"],
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Callback URL for async result delivery",
        examples=["https://example.com/callback"],
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Explicit model ID (None = auto-route via ModelRouter)",
        examples=["gpt-4o"],
    )
    stream: bool = Field(
        default=False,
        description="If True, stream results via SSE; if False, return JSON",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata (trace_id, user_id, etc.)",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Idempotency key. Repeated requests with the same key "
            "return the original result without re-executing the "
            "workflow. Keys are cached for 24 hours."
        ),
        examples=["idem-9f3c-4a1e-8b2d-1f0e2a3b4c5d"],
    )
    resume_from: Optional[str] = Field(
        default=None,
        description=(
            "断点续跑：传入一个已存在（FAILED 或 PAUSED）的 execution_id，"
            "API 会从其 checkpoint 恢复执行。需要 persistence 已接线，"
            "否则返回 422。"
        ),
        examples=["exec-550e8400-e29b-41d4-a716-446655440000"],
    )


class InvokeResponse(BaseModel):
    """
    Response body for POST /invoke (non-streaming mode).

    When stream=False and no callback_url is set, the API waits for
    the workflow to complete and returns this response.

    When callback_url is set, the API returns immediately with
    status="running" and delivers the final result to the callback URL.

    Attributes:
        task_id:  The task instance ID (echoed from request or auto-generated).
        status:   Execution status: "success", "error", or "running".
        result:   Workflow result data (present when status="success").
        error:    Error message (present when status="error").
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    task_id: str = Field(
        ...,
        description="Task instance ID",
    )
    status: str = Field(
        default="success",
        description="Execution status: 'success', 'error', or 'running'",
    )
    result: Optional[dict[str, Any]] = Field(
        default=None,
        description="Workflow result data (present when status='success')",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (present when status='error')",
    )


class HealthResponse(BaseModel):
    """
    Response body for GET /health.

    A simple health check endpoint that returns system status and
    version information. No authentication required.

    v0.5 enhancement: When infrastructure components are wired, the
    response also includes a ``components`` dict with per-component
    health status. The overall ``status`` is ``"healthy"`` only when
    all required components are healthy; otherwise it is ``"degraded"``.

    Attributes:
        status:     System health status: "healthy" or "degraded".
        version:    icore version string.
        timestamp:  ISO 8601 timestamp of the health check.
        components: Per-component health snapshot (v0.5). Keys are
                    component names (``"db"``, ``"model"``, ``"vectorstore"``,
                    ``"graphstore"``); values are dicts with at least a
                    ``"status"`` key.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    status: str = Field(
        default="healthy",
        description="System health status",
    )
    version: str = Field(
        default="1.0.0",
        description="icore version",
    )
    timestamp: str = Field(
        default="",
        description="ISO 8601 timestamp of the health check",
    )
    components: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-component health snapshot (v0.5)",
    )
