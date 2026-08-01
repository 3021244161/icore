"""
icore.engine.base_workflow - Abstract base class for all workflows.

BaseWorkflow defines the contract for workflow definitions. A workflow is
a composable unit of work that orchestrates one or more tasks through a
DAG (Directed Acyclic Graph). Workflows can also invoke other workflows
as sub-workflows, enabling hierarchical composition.

The typical development flow is:

    1. Subclass BaseWorkflow
    2. Implement define() to build the task DAG
    3. Register the workflow with @register_workflow("name")
    4. The engine/API invokes execute() which delegates to WorkflowExecutor

BaseWorkflow uses the Template Method pattern:
    - define() is abstract (subclasses must implement)
    - execute() has a default implementation that delegates to WorkflowExecutor
    - validate() has a default implementation that validates the DAG

Example::

    from icore.engine.base_workflow import BaseWorkflow
    from icore.engine.registry import register_workflow
    from icore.engine.dag import DAG

    @register_workflow("document_summary")
    class DocumentSummaryWorkflow(BaseWorkflow):
        name = "document_summary"
        description = "Summarize a large document via chunking"

        def define(self) -> DAG:
            dag = DAG()
            dag.add_node("chunk", task_name="text_chunker")
            dag.add_node("summarize", task_name="summarizer")
            dag.add_node("merge", task_name="merger")
            dag.add_edge("chunk", "summarize")
            dag.add_edge("summarize", "merge")
            return dag
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.task_context import TaskContext
from icore.engine.dag import DAG
from icore.engine.states import WorkflowState

if TYPE_CHECKING:
    # Avoid circular import at runtime; only for type checking.
    from icore.engine.executor import WorkflowExecutor

logger = logging.getLogger(__name__)


class BaseWorkflow(abc.ABC):
    """
    Abstract base class for all workflows.

    A workflow defines a DAG of tasks and orchestrates their execution.
    Subclasses must implement define() to build the task dependency graph.
    The default execute() implementation delegates to WorkflowExecutor.

    Class Attributes:
        name:        Short unique identifier for this workflow type.
        description: Human-readable description (shown in Swagger docs).
    """

    # --- Class-level metadata (set by subclasses) ---

    name: ClassVar[str] = ""
    """Unique name for this workflow type (used in WorkflowRegistry)."""

    description: ClassVar[str] = ""
    """Human-readable description of what this workflow does."""

    # ------------------------------------------------------------------
    # Abstract method - must be implemented by subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def define(self) -> DAG:
        """
        Define the workflow's task DAG.

        Subclasses build a DAG by adding nodes (tasks) and edges
        (dependencies) using the DAG API:

            dag = DAG()
            dag.add_node("step1", task_name="task_a")
            dag.add_node("step2", task_name="task_b")
            dag.add_edge("step1", "step2")
            return dag

        The DAG may include:
            - Conditional edges (for branching logic)
            - Sub-workflow nodes (is_subworkflow=True)
            - Input builders (for custom data mapping between tasks)

        Returns:
            A validated DAG instance describing the task execution plan.
        """
        ...

    # ------------------------------------------------------------------
    # Default implementations - can be overridden but not required
    # ------------------------------------------------------------------

    async def execute(
        self, ctx: TaskContext, params: dict[str, Any]
    ) -> BaseTaskOutput:
        """
        Execute the workflow.

        Default implementation:
            1. Calls define() to get the DAG
            2. Delegates execution to WorkflowExecutor
            3. Returns the final output from terminal nodes

        Subclasses may override this for custom execution logic (e.g.,
        pre-processing params, post-processing results, custom error
        handling), but most workflows should use the default.

        Args:
            ctx:    TaskContext with injected dependencies (model_manager,
                    db_manager, task_id, workflow_id, etc.).
            params: Workflow input parameters (dict from API request).

        Returns:
            BaseTaskOutput containing the aggregated result of the workflow.
        """
        from icore.engine.executor import WorkflowExecutor

        dag = self.define()
        executor = WorkflowExecutor()
        return await executor.run(dag, ctx, params)

    def validate(self) -> bool:
        """
        Validate the workflow definition.

        Default implementation:
            1. Builds the DAG via define()
            2. Calls DAG.validate() to check for cycles, missing nodes, etc.

        Returns:
            True if the workflow DAG is valid.

        Raises:
            DAGValidationError: If the DAG contains cycles or structural issues.
        """
        dag = self.define()
        return dag.validate()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return f"{self.__class__.__name__}(name={self.name!r})"

    def __str__(self) -> str:
        """String representation."""
        return (
            f"{self.name}: {self.description}"
            if self.description
            else self.name
        )
