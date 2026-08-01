"""
icore.core.registry - Task registration and discovery.

TaskRegistry provides a thread-safe registry for task classes. Tasks are
registered by name and can be looked up at runtime by the workflow engine.

Two registration modes:
    1. Decorator: @register_task("task_name") on the class definition
    2. Explicit:  TaskRegistry.default().register("task_name", TaskClass)

The registry uses a reentrant lock (RLock) for thread safety, since
registration may happen during module import in multiple threads.

Example::

    @register_task("text_classification")
    class TextClassificationTask(BaseTask):
        name = "text_classification"
        ...

    # Lookup
    task_cls = TaskRegistry.default().get("text_classification")
"""

from __future__ import annotations

import threading
from typing import ClassVar

from icore.core.base_task import BaseTask


class TaskRegistry:
    """
    Thread-safe registry for task classes.

    Tasks are stored as a mapping from name -> task class. The registry
    is a singleton per instance, but multiple registry instances can
    exist (e.g. for testing isolation).

    Attributes:
        _tasks: Internal dict mapping task name to task class.
        _lock:  Reentrant lock for thread-safe operations.
    """

    # Class-level default registry instance
    _default: ClassVar[TaskRegistry | None] = None
    _default_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tasks: dict[str, type[BaseTask]] = {}
        self._lock = threading.RLock()

    @classmethod
    def default(cls) -> TaskRegistry:
        """
        Get the global default TaskRegistry instance (singleton).

        Creates the instance on first call, then returns the same
        instance on all subsequent calls.

        Returns:
            The singleton TaskRegistry instance.
        """
        if cls._default is None:
            with cls._default_lock:
                # Double-checked locking for thread safety
                if cls._default is None:
                    cls._default = cls()
        return cls._default

    def register(self, name: str, task_cls: type[BaseTask]) -> None:
        """
        Register a task class under the given name.

        If a task with the same name is already registered, it will be
        overwritten (last-write-wins). This allows hot-reloading of
        task definitions during development.

        Args:
            name:     The unique name to register the task under.
            task_cls: The BaseTask subclass to register.

        Raises:
            TypeError: If task_cls is not a subclass of BaseTask.
            ValueError: If name is empty.
        """
        if not name:
            raise ValueError("Task name cannot be empty")
        if not (isinstance(task_cls, type) and issubclass(task_cls, BaseTask)):
            raise TypeError(
                f"task_cls must be a subclass of BaseTask, got {task_cls}"
            )

        with self._lock:
            self._tasks[name] = task_cls

    def get(self, name: str) -> type[BaseTask]:
        """
        Look up a registered task class by name.

        Args:
            name: The registered task name.

        Returns:
            The BaseTask subclass registered under this name.

        Raises:
            KeyError: If no task is registered under this name.
        """
        with self._lock:
            if name not in self._tasks:
                raise KeyError(
                    f"Task '{name}' is not registered. "
                    f"Available tasks: {self.list_tasks()}"
                )
            return self._tasks[name]

    def list_tasks(self) -> list[str]:
        """
        List all registered task names.

        Returns:
            A sorted list of all registered task names.
        """
        with self._lock:
            return sorted(self._tasks.keys())

    def contains(self, name: str) -> bool:
        """
        Check if a task is registered under the given name.

        Args:
            name: The task name to check.

        Returns:
            True if the task is registered, False otherwise.
        """
        with self._lock:
            return name in self._tasks

    def unregister(self, name: str) -> None:
        """
        Remove a task from the registry.

        Primarily useful for testing and hot-reload scenarios.

        Args:
            name: The task name to remove.

        Raises:
            KeyError: If the name is not registered.
        """
        with self._lock:
            if name not in self._tasks:
                raise KeyError(f"Task '{name}' is not registered")
            del self._tasks[name]

    def clear(self) -> None:
        """Remove all registered tasks (useful for testing)."""
        with self._lock:
            self._tasks.clear()

    def __len__(self) -> int:
        """Number of registered tasks."""
        with self._lock:
            return len(self._tasks)

    def __contains__(self, name: str) -> bool:
        """Support `in` operator for checking registration."""
        return self.contains(name)

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return f"TaskRegistry(tasks={len(self)})"


# ---------------------------------------------------------------------------
# Module-level convenience instance and decorator
# ---------------------------------------------------------------------------

#: The global default TaskRegistry instance (lazy-initialized via TaskRegistry.default())
task_registry = TaskRegistry.default()


def register_task(name: str, registry: TaskRegistry | None = None) -> type:
    """
    Decorator to register a task class.

    Usage::

        @register_task("text_classification")
        class TextClassificationTask(BaseTask):
            name = "text_classification"
            ...

    Args:
        name:    The name to register the task under.
        registry: Optional registry instance. Defaults to the global
                  default registry.

    Returns:
        A decorator function that registers the class and returns it
        unchanged.
    """
    reg = registry if registry is not None else task_registry

    def decorator(cls: type[BaseTask]) -> type[BaseTask]:
        reg.register(name, cls)
        # Ensure the class's name attribute matches the registered name
        # if not explicitly set by the subclass
        if not cls.name:
            cls.name = name
        return cls

    return decorator
