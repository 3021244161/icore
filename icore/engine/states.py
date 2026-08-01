"""
icore.engine.states - State enums for tasks and workflows.

Defines the lifecycle states for both task instances and workflow instances.
These states are used by the WorkflowExecutor and TaskInstanceManager to
track execution progress and drive state transitions.

State machines:
    Task:      PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED
    Workflow:  PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED

These enums replace the string constants referenced in icore.core (which
uses bare strings to avoid circular imports). The engine module owns the
canonical enum definitions.
"""

from __future__ import annotations

from enum import Enum


class TaskState(str, Enum):
    """
    Lifecycle states for a task instance.

    Attributes:
        PENDING:   Task instance created, waiting for execution.
        RUNNING:   Task is currently executing (execute() in progress).
        COMPLETED: Task finished successfully, output is available.
        FAILED:    Task execution failed (exception or validation error).
        CANCELLED: Task was cancelled (timeout or external signal).
        SKIPPED:   Task was not executed due to conditional branching
                   (an upstream edge condition evaluated to False) or
                   because a predecessor failed or was skipped.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal(self) -> bool:
        """True if this state is terminal (no further transitions)."""
        return self in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.SKIPPED,
        )

    @property
    def is_success(self) -> bool:
        """True if this state represents a successful completion."""
        return self == TaskState.COMPLETED


class WorkflowState(str, Enum):
    """
    Lifecycle states for a workflow instance.

    Attributes:
        PENDING:   Workflow instance created, waiting for execution.
        RUNNING:   Workflow is executing its task DAG.
        COMPLETED: All tasks in the workflow finished successfully.
        FAILED:    One or more tasks failed (failure propagation policy applied).
        CANCELLED: Workflow was cancelled (timeout or external signal).
        SKIPPED:   Workflow was not executed (e.g. as a sub-workflow whose
                   upstream condition evaluated to False).
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal(self) -> bool:
        """True if this state is terminal (no further transitions)."""
        return self in (
            WorkflowState.COMPLETED,
            WorkflowState.FAILED,
            WorkflowState.CANCELLED,
            WorkflowState.SKIPPED,
        )

    @property
    def is_success(self) -> bool:
        """True if this state represents a successful completion."""
        return self == WorkflowState.COMPLETED
