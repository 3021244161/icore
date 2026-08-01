"""
icore.core.models - Base input/output models for tasks.

Every concrete task in icore defines its own input and output classes by
subclassing BaseTaskInput and BaseTaskOutput respectively. This provides
strong typing, automatic validation via Pydantic v2, and seamless
integration with FastAPI's auto-generated OpenAPI schemas.

Key design decisions:
    - extra="allow": Subclasses can freely add fields; extra fields from
      API requests are preserved rather than rejected.
    - arbitrary_types_allowed=True: Allows non-Pydantic-native types in
      fields (e.g. custom objects, numpy arrays).
    - use_enum_values=True: Enum fields are serialized as their values,
      making JSON responses clean.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseTaskInput(BaseModel):
    """
    Base class for all task inputs.

    Concrete tasks subclass this and add their own fields:

        class SummaryTaskInput(BaseTaskInput):
            document: str = Field(description="Document text to summarize")
            max_length: int = Field(default=500)

    The model_config allows extra fields (so API callers can pass additional
    metadata) and arbitrary types (so tasks can accept complex objects).
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
    )


class BaseTaskOutput(BaseModel):
    """
    Base class for all task outputs.

    Every task execution returns a BaseTaskOutput (or subclass). The three
    standard fields provide a uniform interface for the workflow engine
    and API layer to determine success/failure and access results.

    Attributes:
        status: "success" if the task completed normally, "error" if it
                failed. This is a quick boolean-ish check for consumers.
        error:  Error message when status="error", None when status="success".
        data:   Business data produced by the task. The structure is defined
                by the concrete task; the engine and API layer treat it as
                an opaque dict for serialization purposes.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    status: str = Field(
        default="success",
        description="Execution status: 'success' or 'error'",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status='error', else None",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Business data produced by the task",
    )

    @property
    def is_success(self) -> bool:
        """Quick check if the task succeeded."""
        return self.status == "success"

    @classmethod
    def success(cls, **data: Any) -> BaseTaskOutput:
        """Create a success output with the given data fields."""
        return cls(status="success", error=None, data=dict(data))

    @classmethod
    def failure(cls, error: str, **data: Any) -> BaseTaskOutput:
        """Create a failure output with the given error message."""
        return cls(status="error", error=error, data=dict(data))
