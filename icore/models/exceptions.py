"""
icore.models.exceptions - Exception hierarchy for the model layer.

All model-related errors inherit from ModelError so callers can catch
the entire family with a single except clause.

Exception hierarchy:
    ModelError
    ├── ModelNotFoundError      - model_id not registered
    ├── ModelUnhealthyError     - model registered but failed health check
    ├── NoAvailableModelError   - auto-routing found no suitable model
    └── ModelAPIError           - LLM API call failed (network, auth, etc.)
"""

from __future__ import annotations


class ModelError(Exception):
    """Base exception for all model-layer errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model_id is not registered in ModelManager."""

    def __init__(self, model_id: str, available: list[str] | None = None) -> None:
        msg = f"Model '{model_id}' is not registered"
        if available:
            msg += f". Available: {available}"
        super().__init__(msg)
        self.model_id = model_id


class ModelUnhealthyError(ModelError):
    """Raised when a model is registered but currently unhealthy."""

    def __init__(self, model_id: str, detail: str = "") -> None:
        msg = f"Model '{model_id}' is unhealthy"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.model_id = model_id


class NoAvailableModelError(ModelError):
    """Raised when auto-routing cannot find any available model."""

    def __init__(self, detail: str = "") -> None:
        msg = "No available model found by auto-router"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class ModelAPIError(ModelError):
    """
    Raised when an LLM API call fails.

    Wraps the underlying HTTP/network error with context about which
    model and endpoint was being called.

    Attributes:
        model_id:   The model that was being called.
        status_code: HTTP status code (if applicable).
        detail:     Error detail message.
    """

    def __init__(
        self,
        model_id: str,
        detail: str = "",
        status_code: int | None = None,
    ) -> None:
        msg = f"API error for model '{model_id}'"
        if status_code is not None:
            msg += f" (HTTP {status_code})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)
        self.model_id = model_id
        self.status_code = status_code
