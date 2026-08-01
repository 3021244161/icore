"""
icore.core.base_task - Abstract base class for all icore tasks.

BaseTask defines the execution contract for every task in the platform.
Concrete tasks subclass BaseTask and implement the abstract methods:

    - prepare():  Initialize resources before execution
    - execute():  Core business logic (LLM call, DB query, etc.)
    - cleanup():  Release resources (always called, like try/finally)

The non-abstract validate() method provides default Pydantic validation
using the task's input_model. Subclasses can override for custom logic.

Lifecycle (managed by WorkflowExecutor):
    1. validate(inp)     -> check input is valid
    2. prepare(ctx)      -> set up resources
    3. execute(ctx, inp) -> run task, produce output
    4. cleanup(ctx)      -> tear down resources (always called)

Example::

    class SummaryInput(BaseTaskInput):
        text: str = Field(description="Text to summarize")

    @register_task("summary")
    class SummaryTask(BaseTask):
        name = "summary"
        input_model = SummaryInput
        output_model = BaseTaskOutput

        async def prepare(self, ctx):
            self._model = ctx.get_model_adapter()

        async def execute(self, ctx, inp):
            result = await self._model.chat(
                messages=[{"role": "user", "content": inp.text}]
            )
            return BaseTaskOutput.success(summary=result["content"])
"""

from __future__ import annotations

import abc
import logging
from typing import ClassVar

from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.task_context import TaskContext

logger = logging.getLogger(__name__)


class BaseTask(abc.ABC):
    """
    Abstract base class for all tasks.

    Subclasses must:
        1. Set class attributes: name, description, input_model, output_model
        2. Implement abstract methods: prepare(), execute()

    Subclasses may:
        - Override validate() for custom input validation
        - Override cleanup() to release resources (default is no-op)

    Class Attributes:
        name:          Short unique identifier for this task type.
        description:   Human-readable description (shown in Swagger docs).
        input_model:   The Pydantic model class for task input.
        output_model:  The Pydantic model class for task output.
    """

    # --- Class-level metadata (set by subclasses) ---

    name: ClassVar[str] = ""
    """Unique name for this task type (used in TaskRegistry)."""

    description: ClassVar[str] = ""
    """Human-readable description of what this task does."""

    input_model: ClassVar[type[BaseTaskInput]] = BaseTaskInput
    """The Pydantic model class that defines this task's input schema."""

    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput
    """The Pydantic model class that defines this task's output schema."""

    # ------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def prepare(self, ctx: TaskContext) -> None:
        """
        Prepare resources before execution.

        Called by the engine before execute(). Use this to:
            - Obtain model adapters: model = ctx.get_model_adapter()
            - Open database connections: db = ctx.get_db("main")
            - Load configuration or warm caches
            - Validate runtime context (beyond input validation)

        Args:
            ctx: The task execution context with injected dependencies.

        Raises:
            Exception: If preparation fails, the task transitions to
                       FAILED state and execute() is not called.
        """
        ...

    @abc.abstractmethod
    async def execute(
        self, ctx: TaskContext, inp: BaseTaskInput
    ) -> BaseTaskOutput:
        """
        Execute the core task logic.

        This is the main method where the task's business logic lives.
        It receives the prepared context and validated input, and must
        return a BaseTaskOutput (or subclass) instance.

        For streaming tasks (ctx.stream == True), this method may return
        an async generator that yields BaseTaskOutput chunks. The engine
        detects this and handles SSE streaming accordingly.

        Args:
            ctx: The task execution context with injected dependencies.
            inp: The validated input model instance (type is input_model).

        Returns:
            BaseTaskOutput containing the task results.

        Raises:
            Exception: Any exception transitions the task to FAILED state.
                       The exception message is captured in output.error.
        """
        ...

    async def cleanup(self, ctx: TaskContext) -> None:
        """
        Clean up resources after execution.

        Called by the engine AFTER execute(), regardless of whether
        execute() succeeded or raised an exception (like try/finally).
        Use this to:
            - Close database connections
            - Release locks or semaphores
            - Flush buffers
            - Clean up temporary files

        This method must NOT raise exceptions. If cleanup fails, log the
        error but do not propagate it - the task result should reflect
        the execute() outcome, not the cleanup outcome.

        The default implementation is a no-op. Subclasses override this
        only when they have resources to release.

        Args:
            ctx: The task execution context.
        """
        pass  # no-op by default; override if resources need cleanup

    # ------------------------------------------------------------------
    # Non-abstract methods - can be overridden but not required
    # ------------------------------------------------------------------

    def validate(self, inp: BaseTaskInput) -> bool:
        """
        Validate the input before execution.

        Default implementation re-validates the input using Pydantic.
        Since input is already a Pydantic model instance (validated at
        construction), this is effectively a no-op that returns True.

        Subclasses can override this to add custom validation logic:

            def validate(self, inp: MyInput) -> bool:
                if not super().validate(inp):
                    return False
                return len(inp.text) <= self.max_input_length

        Args:
            inp: The input model instance to validate.

        Returns:
            True if the input is valid, False otherwise.
        """
        try:
            # Re-validate using Pydantic (catches any mutations)
            inp.model_validate(inp.model_dump())
            return True
        except Exception as e:
            logger.warning(
                "Validation failed for task %s: %s", self.name, e
            )
            return False

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return f"{self.__class__.__name__}(name={self.name!r})"

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.description}" if self.description else self.name
