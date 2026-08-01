"""
icore.engine.instance_manager - Task instance lifecycle tracking.

Manages the lifecycle of all task instances running in the system.
Each workflow invocation creates a TaskInstance with a unique task_id.
The manager tracks the instance through its lifecycle states and
provides cleanup of stale terminal instances.

The instance manager is the system's source of truth for "what is
currently running". The API layer uses it to:
    - Check if a task_id already exists (deduplication)
    - Query the status of a running/completed task
    - List all active tasks for monitoring
    - Cancel a running task
    - Clean up old completed/failed instances to reclaim memory

Thread safety:
    All public methods are async and use an asyncio.Lock to protect
    the internal instance dict. This is correct for asyncio's
    single-threaded event loop model - the lock prevents concurrent
    coroutine interleaving during await points.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from icore.engine.states import TaskState

logger = logging.getLogger(__name__)


@dataclass
class TaskInstance:
    """
    A runtime instance of a task (workflow invocation).

    Created when a workflow is invoked via the API. Tracks the full
    lifecycle from creation through completion, including the result
    and any error information.

    Attributes:
        task_id:       Unique identifier for this instance (from API request).
        workflow_name: Name of the workflow being executed.
        params:        Input parameters passed to the workflow.
        state:         Current lifecycle state (TaskState enum).
        created_at:     Unix timestamp of creation.
        updated_at:    Unix timestamp of last state update.
        result:        The final BaseTaskOutput (or None if not yet terminal).
        error:         Error message if the task failed (or None).
        cancel_event:  asyncio.Event set when the task is cancelled,
                       allowing the executor to check for cancellation.
    """

    task_id: str
    workflow_name: str
    params: dict[str, Any] = field(default_factory=dict)
    state: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Any = None
    error: Optional[str] = None
    cancel_event: asyncio.Event = field(
        default_factory=asyncio.Event, repr=False
    )

    @property
    def is_terminal(self) -> bool:
        """True if the instance is in a terminal state."""
        return self.state.is_terminal

    @property
    def is_active(self) -> bool:
        """True if the instance is still running (not terminal)."""
        return not self.state.is_terminal

    @property
    def elapsed(self) -> float:
        """Seconds since the instance was created."""
        return time.time() - self.created_at

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a dict (for API responses / monitoring).

        Excludes the cancel_event (not serializable) and the raw result
        (which may contain non-serializable objects).
        """
        return {
            "task_id": self.task_id,
            "workflow_name": self.workflow_name,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed": round(self.elapsed, 3),
            "error": self.error,
            "is_terminal": self.is_terminal,
        }


class TaskInstanceManager:
    """
    Manages the lifecycle of all task instances.

    The manager is the central registry for task instances. It is used
    by the API layer to track invocations, by the concurrency controller
    to count active instances, and by monitoring endpoints to report
    system status.

    Usage:

        manager = TaskInstanceManager(ttl=3600)
        await manager.create("task-1", "document_summary", {"text": "..."})
        await manager.update_state("task-1", TaskState.RUNNING)
        instance = await manager.get("task-1")
        await manager.update_state("task-1", TaskState.COMPLETED, result=output)
        await manager.cleanup()  # remove instances older than TTL

    Concurrency model:
        All operations are guarded by an asyncio.Lock to ensure
        consistency across coroutine yields. The lock is reentrant-safe
        because each method acquires, does its work (no await on other
        locked methods), and releases.

    Attributes:
        _instances:  Dict mapping task_id -> TaskInstance.
        _lock:       asyncio.Lock protecting _instances.
        _ttl:        Time-to-live for terminal instances (seconds).
    """

    def __init__(self, ttl: int = 3600) -> None:
        """
        Initialize the instance manager.

        Args:
            ttl: Time-to-live for terminal instances in seconds.
                 Instances in terminal states (COMPLETED, FAILED,
                 CANCELLED, SKIPPED) older than this are eligible for
                 cleanup. Default: 3600 (1 hour).
        """
        self._instances: dict[str, TaskInstance] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._ttl: int = ttl

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def create(
        self,
        task_id: str,
        workflow_name: str,
        params: Optional[dict[str, Any]] = None,
    ) -> TaskInstance:
        """
        Create a new task instance record.

        Args:
            task_id:       Unique identifier (from API request).
            workflow_name: Name of the workflow to execute.
            params:        Input parameters for the workflow.

        Returns:
            The created TaskInstance.

        Raises:
            ValueError: If a task with this task_id already exists
                        (duplicate submission).
        """
        async with self._lock:
            if task_id in self._instances:
                existing = self._instances[task_id]
                if not existing.is_terminal:
                    raise ValueError(
                        f"Task '{task_id}' already exists and is "
                        f"in state {existing.state.value}"
                    )
                # Terminal duplicate: replace (re-run)
                logger.info(
                    "Replacing terminal task '%s' (was %s) with new instance",
                    task_id,
                    existing.state.value,
                )

            instance = TaskInstance(
                task_id=task_id,
                workflow_name=workflow_name,
                params=params or {},
                state=TaskState.PENDING,
            )
            self._instances[task_id] = instance
            logger.info(
                "Created task instance '%s' (workflow='%s')",
                task_id,
                workflow_name,
            )
            return instance

    async def get(self, task_id: str) -> Optional[TaskInstance]:
        """
        Get a task instance by ID.

        Args:
            task_id: The unique task identifier.

        Returns:
            The TaskInstance, or None if not found.
        """
        async with self._lock:
            return self._instances.get(task_id)

    async def update_state(
        self,
        task_id: str,
        state: TaskState,
        result: Any = None,
        error: Optional[str] = None,
    ) -> Optional[TaskInstance]:
        """
        Update the state of a task instance.

        Args:
            task_id: The unique task identifier.
            state:   The new TaskState.
            result:  Optional result (set when state is COMPLETED).
            error:   Optional error message (set when state is FAILED).

        Returns:
            The updated TaskInstance, or None if not found.
        """
        async with self._lock:
            instance = self._instances.get(task_id)
            if instance is None:
                logger.warning(
                    "Attempted to update unknown task '%s' to state %s",
                    task_id,
                    state.value,
                )
                return None

            instance.state = state
            instance.updated_at = time.time()
            if result is not None:
                instance.result = result
            if error is not None:
                instance.error = error

            logger.debug(
                "Task '%s' state -> %s", task_id, state.value
            )
            return instance

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a task instance.

        Sets the cancel_event so the executor can detect the cancellation
        and abort the running coroutine. Marks the instance as CANCELLED
        if it was still active.

        Args:
            task_id: The unique task identifier.

        Returns:
            True if the task was cancelled, False if not found or
            already terminal.
        """
        async with self._lock:
            instance = self._instances.get(task_id)
            if instance is None:
                logger.warning(
                    "Attempted to cancel unknown task '%s'", task_id
                )
                return False

            if instance.is_terminal:
                logger.info(
                    "Task '%s' is already terminal (%s), cannot cancel",
                    task_id,
                    instance.state.value,
                )
                return False

            # Signal the cancel event so the executor can abort
            instance.cancel_event.set()
            instance.state = TaskState.CANCELLED
            instance.updated_at = time.time()
            logger.info("Cancelled task '%s'", task_id)
            return True

    async def remove(self, task_id: str) -> bool:
        """
        Remove a task instance from tracking.

        Args:
            task_id: The unique task identifier.

        Returns:
            True if removed, False if not found.
        """
        async with self._lock:
            if task_id in self._instances:
                del self._instances[task_id]
                logger.debug("Removed task instance '%s'", task_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def list_active(self) -> list[TaskInstance]:
        """
        List all non-terminal (active) task instances.

        Returns:
            List of TaskInstance objects that are still running.
        """
        async with self._lock:
            return [
                inst
                for inst in self._instances.values()
                if inst.is_active
            ]

    async def list_all(self) -> list[TaskInstance]:
        """
        List all task instances (active and terminal).

        Returns:
            List of all TaskInstance objects.
        """
        async with self._lock:
            return list(self._instances.values())

    async def count_active(self) -> int:
        """Return the number of active (non-terminal) instances."""
        async with self._lock:
            return sum(
                1 for inst in self._instances.values() if inst.is_active
            )

    async def count_by_state(self) -> dict[str, int]:
        """
        Count instances grouped by state.

        Returns:
            Dict mapping state name -> count.
        """
        async with self._lock:
            counts: dict[str, int] = {}
            for inst in self._instances.values():
                key = inst.state.value
                counts[key] = counts.get(key, 0) + 1
            return counts

    async def exists(self, task_id: str) -> bool:
        """Check if a task_id exists in the manager."""
        async with self._lock:
            return task_id in self._instances

    async def is_cancelled(self, task_id: str) -> bool:
        """
        Check if a task has been cancelled.

        Used by the executor to detect cancellation during execution.
        """
        async with self._lock:
            instance = self._instances.get(task_id)
            if instance is None:
                return False
            return instance.cancel_event.is_set()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> int:
        """
        Remove terminal instances older than the TTL.

        This reclaims memory from completed/failed/cancelled tasks
        that are no longer needed for status queries.

        Returns:
            Number of instances removed.
        """
        now = time.time()
        removed = 0

        async with self._lock:
            to_remove: list[str] = []
            for task_id, inst in self._instances.items():
                if inst.is_terminal and (now - inst.updated_at) > self._ttl:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._instances[task_id]
                removed += 1

        if removed > 0:
            logger.info(
                "Cleaned up %d terminal task instances (TTL=%ds)",
                removed,
                self._ttl,
            )
        return removed

    async def clear_all(self) -> int:
        """
        Remove ALL instances (including active ones).

        **Dangerous**: This should only be used during shutdown or
        testing. Active instances will be orphaned.

        Returns:
            Number of instances removed.
        """
        async with self._lock:
            count = len(self._instances)
            self._instances.clear()
            logger.warning("Cleared all %d task instances", count)
            return count

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ttl(self) -> int:
        """Return the TTL for terminal instances (seconds)."""
        return self._ttl

    @property
    def total_count(self) -> int:
        """Return the total number of tracked instances (no lock needed for int read)."""
        return len(self._instances)
