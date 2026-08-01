"""
icore.engine - Workflow engine layer.

This package provides the workflow orchestration engine for icore:
    - BaseWorkflow:       Abstract base class for workflow definitions
    - DAG:                Directed Acyclic Graph for task scheduling
    - DAGNode / DAGEdge:  Graph components (nodes and edges)
    - WorkflowExecutor:   Runs DAGs with concurrency, retries, and timeouts
    - WorkflowRegistry:   Thread-safe registry for workflow classes
    - WorkflowState:      Workflow lifecycle state enum
    - TaskState:          Task lifecycle state enum (with SKIPPED)
    - TaskQueue:          Async task queue (in-memory / Redis)
    - TaskInstanceManager: Manages task instance lifecycle and cleanup
    - ConcurrencyController: Global/per-workflow semaphores, backpressure, rate limiting

Module dependency:
    engine depends on core (BaseTask, TaskContext, models)
    engine does NOT depend on db, models, api, or services
"""

from __future__ import annotations

from icore.engine.base_workflow import BaseWorkflow
from icore.engine.concurrency_control import (
    BackpressureError,
    ConcurrencyController,
    ConcurrencyStats,
    TokenBucket,
)
from icore.engine.dag import (
    DAG,
    DAGEdge,
    DAGNode,
    DAGValidationError,
    EdgeCondition,
    InputBuilder,
)
from icore.engine.executor import WorkflowExecutionResult, WorkflowExecutor
from icore.engine.instance_manager import TaskInstance, TaskInstanceManager
from icore.engine.registry import (
    WorkflowRegistry,
    register_workflow,
    workflow_registry,
)
from icore.engine.states import TaskState, WorkflowState
from icore.engine.task_queue import QueueFullError, TaskItem, TaskQueue

__all__ = [
    # Base workflow
    "BaseWorkflow",
    # DAG
    "DAG",
    "DAGNode",
    "DAGEdge",
    "DAGValidationError",
    "EdgeCondition",
    "InputBuilder",
    # Executor
    "WorkflowExecutor",
    "WorkflowExecutionResult",
    # Registry
    "WorkflowRegistry",
    "register_workflow",
    "workflow_registry",
    # States
    "WorkflowState",
    "TaskState",
    # Task queue (US-007)
    "TaskQueue",
    "TaskItem",
    "QueueFullError",
    # Instance manager (US-007)
    "TaskInstanceManager",
    "TaskInstance",
    # Concurrency control (US-007)
    "ConcurrencyController",
    "ConcurrencyStats",
    "TokenBucket",
    "BackpressureError",
]
