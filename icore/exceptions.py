"""
icore.exceptions - Unified exception hierarchy for the entire icore platform.

All icore-raised errors inherit from ``ICoreError``, which carries:

    - ``code``:        Stable error code (e.g. ``"E-WF-001"``) for
                       programmatic matching by clients.
    - ``http_status``: Suggested HTTP status code for the API layer.
    - ``retryable``:   Whether the operation can reasonably be retried.

The exception tree is split into two branches:

    1. Business exceptions (4xx) — caller did something wrong.
    2. System exceptions  (5xx) — infrastructure failure, often retryable.

The API layer's global exception handler translates any ``ICoreError``
into a structured JSON response with ``error`` / ``code`` / ``detail`` /
``retryable`` / ``task_id`` fields (plus ``Retry-After`` when retryable).
"""

from __future__ import annotations

from typing import Optional


class ICoreError(Exception):
    """
    Base exception for every error raised by icore.

    Subclasses set the class-level attributes ``code``, ``http_status``
    and ``retryable`` so callers can match on a stable error code
    instead of exception type.

    Attributes:
        code:        Stable error code (e.g. ``"E-INTERNAL"``).
        http_status: Suggested HTTP status code for API responses.
        retryable:   Whether retrying the same operation could succeed.
    """

    code: str = "E-INTERNAL"
    http_status: int = 500
    retryable: bool = False

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)
        self.detail: str = detail

    def to_dict(self, task_id: Optional[str] = None) -> dict:
        """Serialize to a JSON-friendly dict for API responses."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "detail": self.detail or str(self),
            "retryable": self.retryable,
            "task_id": task_id,
        }


# ---------------------------------------------------------------------------
# Business exceptions (4xx) - caller error
# ---------------------------------------------------------------------------

class WorkflowNotFoundError(ICoreError):
    """Raised when a requested workflow_name is not registered."""

    code = "E-WF-001"
    http_status = 404


class TaskNotFoundError(ICoreError):
    """Raised when a task_id cannot be found in the instance manager."""

    code = "E-TASK-001"
    http_status = 404


class ValidationError(ICoreError):
    """Raised when input or model output fails validation."""

    code = "E-VAL-001"
    http_status = 422


class MediaUnsupportedError(ICoreError):
    """Raised when a media type/source is not supported by a processor."""

    code = "E-MEDIA-001"
    http_status = 400


# ---------------------------------------------------------------------------
# System exceptions (5xx) - infrastructure failure
# ---------------------------------------------------------------------------

class DatabaseConnectionError(ICoreError):
    """Raised when a database connection cannot be established or is lost."""

    code = "E-DB-001"
    http_status = 503
    retryable = True


class ModelAPIError(ICoreError):
    """Raised when an LLM API call fails (network, auth, 5xx, etc.)."""

    code = "E-MODEL-001"
    http_status = 502
    retryable = True

    def __init__(
        self,
        detail: str = "",
        *,
        model_id: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(detail)
        self.model_id = model_id
        self.status_code = status_code


class ModelTimeoutError(ICoreError):
    """Raised when an LLM API call times out."""

    code = "E-MODEL-002"
    http_status = 504
    retryable = True

    def __init__(
        self,
        detail: str = "",
        *,
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__(detail)
        self.model_id = model_id


class VectorStoreError(ICoreError):
    """Raised by vector-store adapters (Milvus, etc.)."""

    code = "E-VEC-001"
    http_status = 503
    retryable = True


class ObjectStoreError(ICoreError):
    """Raised by object-store adapters (MinIO, S3, etc.)."""

    code = "E-OBJ-001"
    http_status = 503
    retryable = True


class GraphStoreError(ICoreError):
    """Raised by graph-store adapters (Neo4j, etc.)."""

    code = "E-GRAPH-001"
    http_status = 503
    retryable = True


class CircuitBreakerOpenError(ICoreError):
    """Raised when a circuit breaker is OPEN and refuses a call."""

    code = "E-CB-001"
    http_status = 503
    retryable = True


class BackpressureError(ICoreError):
    """Raised when the system is under backpressure and rejects work."""

    code = "E-BP-001"
    http_status = 503
    retryable = True

    def __init__(
        self,
        detail: str = "System is under backpressure",
        *,
        retry_after: int = 5,
    ) -> None:
        super().__init__(detail)
        self.retry_after = retry_after


class ConflictError(ICoreError):
    """Raised on concurrent modification / lock acquisition failure."""

    code = "E-CONFLICT-001"
    http_status = 409
    retryable = True


class NoAvailableModelError(ICoreError):
    """Raised when all models (incl. fallbacks) are exhausted."""

    code = "E-MODEL-003"
    http_status = 503
    retryable = True

    def __init__(self, detail: str = "") -> None:
        # Backward-compatible signature: detail is positional.
        msg = "No available model found by auto-router"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class ModelNotFoundError(ICoreError):
    """Raised when a requested model_id is not registered."""

    code = "E-MODEL-004"
    http_status = 404

    def __init__(
        self,
        model_id: str,
        available: Optional[list[str]] = None,
    ) -> None:
        msg = f"Model '{model_id}' is not registered"
        if available:
            msg += f". Available: {available}"
        super().__init__(msg)
        self.model_id = model_id


class ModelUnhealthyError(ICoreError):
    """Raised when a model is registered but currently unhealthy."""

    code = "E-MODEL-005"
    http_status = 503
    retryable = True

    def __init__(self, model_id: str, detail: str = "") -> None:
        msg = f"Model '{model_id}' is unhealthy"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.model_id = model_id


class ModelUnsupportedError(ICoreError):
    """Raised when a model lacks a requested capability (e.g. vision)."""

    code = "E-MODEL-006"
    http_status = 400


# ---------------------------------------------------------------------------
# v0.6: Resilience / auth / persistence exceptions
# ---------------------------------------------------------------------------

class DeadLetterQueueError(ICoreError):
    """Raised by the Dead Letter Queue on replay / persistence failure."""

    code = "E-DLQ-001"
    http_status = 503
    retryable = True


class SagaCompensationError(ICoreError):
    """Raised when a Saga compensation (rollback) step fails."""

    code = "E-SAGA-001"
    http_status = 500


class PromptInjectionError(ICoreError):
    """Raised when prompt injection is detected (input rejected)."""

    code = "E-SEC-001"
    http_status = 400


class AuthenticationError(ICoreError):
    """Raised when API Key / JWT authentication fails."""

    code = "E-AUTH-001"
    http_status = 401


class AuthorizationError(ICoreError):
    """Raised when an authenticated principal lacks required role."""

    code = "E-AUTH-002"
    http_status = 403


class WorkflowPersistenceError(ICoreError):
    """Raised when workflow execution state cannot be persisted."""

    code = "E-PERSIST-001"
    http_status = 503
    retryable = True


class WorkflowResumeError(ICoreError):
    """Raised when a workflow cannot be resumed from a checkpoint."""

    code = "E-PERSIST-002"
    http_status = 409


__all__ = [
    "ICoreError",
    # Business (4xx)
    "WorkflowNotFoundError",
    "TaskNotFoundError",
    "ValidationError",
    "MediaUnsupportedError",
    "ModelNotFoundError",
    "ModelUnsupportedError",
    "PromptInjectionError",
    "AuthenticationError",
    "AuthorizationError",
    # System (5xx)
    "DatabaseConnectionError",
    "ModelAPIError",
    "ModelTimeoutError",
    "ModelUnhealthyError",
    "VectorStoreError",
    "ObjectStoreError",
    "GraphStoreError",
    "CircuitBreakerOpenError",
    "BackpressureError",
    "ConflictError",
    "NoAvailableModelError",
    # v0.6
    "DeadLetterQueueError",
    "SagaCompensationError",
    "WorkflowPersistenceError",
    "WorkflowResumeError",
]
