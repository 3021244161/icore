"""
icore.core - Core task abstraction layer.

This package defines the foundational abstractions that all icore tasks build upon:
    - BaseTask:       Abstract base class for all tasks
    - BaseTaskInput:  Base Pydantic model for task inputs
    - BaseTaskOutput: Base Pydantic model for task outputs
    - TaskContext:    Dependency injection container for task execution
    - TaskRegistry:   Registry for task class discovery
    - register_task:  Decorator for declarative task registration
"""

from __future__ import annotations

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import TaskRegistry, register_task, task_registry
from icore.core.task_context import TaskContext

__all__ = [
    "BaseTask",
    "BaseTaskInput",
    "BaseTaskOutput",
    "TaskContext",
    "TaskRegistry",
    "register_task",
    "task_registry",
]
