"""
icore.engine.registry - Workflow registration and discovery.

WorkflowRegistry provides a thread-safe registry for workflow classes.
Workflows are registered by name and can be looked up at runtime by the
API layer and by the executor (for sub-workflow invocation).

Two registration modes:
    1. Decorator: @register_workflow("workflow_name") on the class definition
    2. Explicit:  WorkflowRegistry.default().register("workflow_name", WFClass)

The registry mirrors the design of TaskRegistry (icore.core.registry) for
consistency. It uses a reentrant lock (RLock) for thread safety.

Example::

    @register_workflow("document_summary")
    class DocumentSummaryWorkflow(BaseWorkflow):
        name = "document_summary"
        ...

    # Lookup
    wf_cls = WorkflowRegistry.default().get("document_summary")
    wf = wf_cls()
    result = await wf.execute(ctx, params)
"""

from __future__ import annotations

import threading
from typing import ClassVar

from icore.engine.base_workflow import BaseWorkflow


class WorkflowRegistry:
    """
    Thread-safe registry for workflow classes.

    Workflows are stored as a mapping from name -> workflow class. The
    registry is a singleton per instance, but multiple registry instances
    can exist (e.g. for testing isolation).

    Attributes:
        _workflows: Internal dict mapping workflow name to workflow class.
        _lock:      Reentrant lock for thread-safe operations.
    """

    # Class-level default registry instance
    _default: ClassVar[WorkflowRegistry | None] = None
    _default_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._workflows: dict[str, type[BaseWorkflow]] = {}
        self._lock = threading.RLock()

    @classmethod
    def default(cls) -> WorkflowRegistry:
        """
        Get the global default WorkflowRegistry instance (singleton).

        Creates the instance on first call, then returns the same
        instance on all subsequent calls.

        Returns:
            The singleton WorkflowRegistry instance.
        """
        if cls._default is None:
            with cls._default_lock:
                # Double-checked locking for thread safety
                if cls._default is None:
                    cls._default = cls()
        return cls._default

    def register(self, name: str, workflow_cls: type[BaseWorkflow]) -> None:
        """
        Register a workflow class under the given name.

        If a workflow with the same name is already registered, it will be
        overwritten (last-write-wins). This allows hot-reloading of
        workflow definitions during development.

        Args:
            name:         The unique name to register the workflow under.
            workflow_cls: The BaseWorkflow subclass to register.

        Raises:
            TypeError:  If workflow_cls is not a subclass of BaseWorkflow.
            ValueError: If name is empty.
        """
        if not name:
            raise ValueError("Workflow name cannot be empty")
        if not (
            isinstance(workflow_cls, type)
            and issubclass(workflow_cls, BaseWorkflow)
        ):
            raise TypeError(
                f"workflow_cls must be a subclass of BaseWorkflow, "
                f"got {workflow_cls}"
            )

        with self._lock:
            self._workflows[name] = workflow_cls

    def get(self, name: str) -> type[BaseWorkflow]:
        """
        Look up a registered workflow class by name.

        Args:
            name: The registered workflow name.

        Returns:
            The BaseWorkflow subclass registered under this name.

        Raises:
            KeyError: If no workflow is registered under this name.
        """
        with self._lock:
            if name not in self._workflows:
                raise KeyError(
                    f"Workflow '{name}' is not registered. "
                    f"Available workflows: {self.list_workflows()}"
                )
            return self._workflows[name]

    def list_workflows(self) -> list[str]:
        """
        List all registered workflow names.

        Returns:
            A sorted list of all registered workflow names.
        """
        with self._lock:
            return sorted(self._workflows.keys())

    def contains(self, name: str) -> bool:
        """
        Check if a workflow is registered under the given name.

        Args:
            name: The workflow name to check.

        Returns:
            True if the workflow is registered, False otherwise.
        """
        with self._lock:
            return name in self._workflows

    def unregister(self, name: str) -> None:
        """
        Remove a workflow from the registry.

        Primarily useful for testing and hot-reload scenarios.

        Args:
            name: The workflow name to remove.

        Raises:
            KeyError: If the name is not registered.
        """
        with self._lock:
            if name not in self._workflows:
                raise KeyError(f"Workflow '{name}' is not registered")
            del self._workflows[name]

    def clear(self) -> None:
        """Remove all registered workflows (useful for testing)."""
        with self._lock:
            self._workflows.clear()

    def __len__(self) -> int:
        """Number of registered workflows."""
        with self._lock:
            return len(self._workflows)

    def __contains__(self, name: str) -> bool:
        """Support `in` operator for checking registration."""
        return self.contains(name)

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return f"WorkflowRegistry(workflows={len(self)})"


# ---------------------------------------------------------------------------
# Module-level convenience instance and decorator
# ---------------------------------------------------------------------------

#: The global default WorkflowRegistry instance (lazy-initialized via
#: WorkflowRegistry.default())
workflow_registry = WorkflowRegistry.default()


def register_workflow(
    name: str, registry: WorkflowRegistry | None = None
) -> type:
    """
    Decorator to register a workflow class.

    Usage::

        @register_workflow("document_summary")
        class DocumentSummaryWorkflow(BaseWorkflow):
            name = "document_summary"
            ...

    Args:
        name:    The name to register the workflow under.
        registry: Optional registry instance. Defaults to the global
                  default registry.

    Returns:
        A decorator function that registers the class and returns it
        unchanged.
    """
    reg = registry if registry is not None else workflow_registry

    def decorator(cls: type[BaseWorkflow]) -> type[BaseWorkflow]:
        reg.register(name, cls)
        # Ensure the class's name attribute matches the registered name
        # if not explicitly set by the subclass
        if not cls.name:
            cls.name = name
        return cls

    return decorator
