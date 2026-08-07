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
    - CircuitBreaker:     Per-resource circuit breaker (CLOSED/OPEN/HALF_OPEN)
    - v0.6 resilience:    BackoffStrategy / DeadLetterQueue / SagaWorkflow

Module dependency:
    engine depends on core (BaseTask, TaskContext, models)
    engine does NOT depend on db, models, api, or services
"""

from __future__ import annotations

from icore.engine.base_workflow import BaseWorkflow
from icore.engine.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)
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

# v0.6: resilience components
from icore.engine.backoff import BackoffStrategy, compute_delay, retry_with_backoff
from icore.engine.dead_letter_queue import (
    BaseDLQBackend,
    DLQEntry,
    DeadLetterQueue,
    InMemoryDLQBackend,
    PostgresDLQBackend,
)
from icore.engine.saga import SagaStep, SagaWorkflow

# v0.6: multi-Agent collaboration framework
from icore.engine.agent import (
    AgentConfig,
    AgentExecutionResult,
    AgentMode,
    AgentNodeExecutor,
)

# v0.6: full-stack backpressure + hot reload + graceful degradation
from icore.engine.backpressure import (
    BackpressureCoordinator,
    BackpressureSnapshot,
    ComponentBudget,
    ComponentStatus,
)
from icore.engine.graceful_degradation import (
    ComponentState as DegradationComponentState,
    DegradationSnapshot,
    DegradationState,
    GracefulDegradationCoordinator,
)
from icore.engine.hot_reload import (
    HotReloadCoordinator,
    ReloadCallback,
    ReloadEvent,
)

__all__ = [
    # Base workflow
    "BaseWorkflow",
    "SagaWorkflow",
    "SagaStep",
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
    # Circuit breaker (v0.5)
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
    # v0.6: resilience
    "BackoffStrategy",
    "compute_delay",
    "retry_with_backoff",
    "DLQEntry",
    "BaseDLQBackend",
    "InMemoryDLQBackend",
    "PostgresDLQBackend",
    "DeadLetterQueue",
    # v0.6: multi-Agent framework
    "AgentMode",
    "AgentConfig",
    "AgentExecutionResult",
    "AgentNodeExecutor",
    # v0.6: full-stack backpressure + hot reload + graceful degradation
    "BackpressureCoordinator",
    "BackpressureSnapshot",
    "ComponentBudget",
    "ComponentStatus",
    "DegradationState",
    "DegradationSnapshot",
    "DegradationComponentState",
    "GracefulDegradationCoordinator",
    "HotReloadCoordinator",
    "ReloadCallback",
    "ReloadEvent",
]
