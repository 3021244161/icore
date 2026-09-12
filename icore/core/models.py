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
      making JSON responses clean. NOTE: under this default, enum fields
      are stored as their ``.value`` (``str``/``int``), NOT enum members --
      see the BaseTaskInput docstring for the full contract (including the
      ``strict_enums`` opt-in switch) before branching on enum identity.
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

    Model configuration semantics:

        - ``extra="allow"``: unknown fields passed at construction are kept
          as attributes instead of being rejected. The API layer can
          therefore forward extra metadata without every task declaring it.
        - ``arbitrary_types_allowed=True``: fields may use non-Pydantic
          types (custom objects); they are assigned without validation.
        - ``use_enum_values=True`` (**default**): enum fields are stored as
          their ``.value`` (``str`` / ``int``), never as enum members.
          Rationale: task inputs are JSON-serialised by the persistence
          layer (``json.dumps`` without a ``default=`` fallback), so raw
          ``Enum`` members would crash persistence for non-``str``/``int``
          based enums.
          **Contract caveat (ICORE-ISSUE-001):** downstream code must use
          value comparison (``==``) or normalise via ``MyEnum(inp.field)``
          -- ``inp.field is MyEnum.X`` is silently ``False`` under this
          default config.
        - ``validate_assignment=True``: assigning to a field after
          construction re-runs validation; an invalid value raises
          ``ValidationError`` instead of being silently accepted.

    Opting into strong-typed enums (``strict_enums``):

        Tasks whose logic branches on enum identity declare their input
        class with the ``strict_enums`` class keyword:

            class MyInput(BaseTaskInput, strict_enums=True):
                policy: Policy = Policy.A

            MyInput(policy=Policy.B).policy is Policy.B   # True

        The flag flips ``use_enum_values`` to ``False`` for that class
        only -- the base class and sibling classes keep the JSON-safe
        default. With ``strict_enums=True``, prefer ``StrEnum`` /
        ``IntEnum`` (or serialise via ``model_dump(mode="json")``), since
        plain ``Enum`` members are not directly JSON-serialisable and the
        persistence layer has no ``default=`` fallback.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
    )

    def __init_subclass__(
        cls, strict_enums: bool = False, **kwargs: Any
    ) -> None:
        """
        Class-creation hook implementing the ``strict_enums`` switch.

        Pydantic's metaclass merges parent and child ``model_config`` into
        a fresh dict exposed as ``cls.model_config`` *before* ``type.__new__
        `` fires this hook, and builds the validation schema from that same
        dict *afterwards* -- so flipping the key in place here is both
        safe (the parent's config dict is never touched) and effective
        (the schema is generated with the flipped value). See
        ICORE-ISSUE-001 (zGo) for the motivation.

        Args:
            strict_enums: When True, keep enum members instead of coercing
                them to their ``.value`` for this class only.
            **kwargs: Forwarded to ``super().__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        if strict_enums and cls.model_config.get("use_enum_values"):
            cls.model_config["use_enum_values"] = False


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
