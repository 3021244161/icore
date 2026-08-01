"""
icore - Enterprise-level LLM Workflow Orchestration Platform.

A coding-driven workflow capability platform that orchestrates LLM-based tasks
through DAG-based workflows, with multi-model routing, multi-database support,
and multiple service exposure protocols.

Core abstractions:
    - BaseTask:        Abstract base class for all tasks (icore.core.base_task)
    - BaseWorkflow:    Abstract base class for all workflows (icore.engine.base_workflow)
    - BaseConnector:   Abstract base class for database connectors (icore.db.base_connector)
    - BaseModelAdapter: Abstract base class for LLM model adapters (icore.models.base_adapter)
    - BaseServiceExposer: Abstract base class for service exposers (icore.services.base_exposer)
"""

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task, task_registry
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.registry import register_workflow, workflow_registry

#: Package version (PEP 440).
__version__ = "1.0.0"

#: Alias matching the design-doc public API (``icore.registry``).
registry = task_registry

__all__ = [
    "__version__",
    # Core task abstractions
    "BaseTask",
    "BaseTaskInput",
    "BaseTaskOutput",
    "TaskContext",
    "register_task",
    "task_registry",
    "registry",
    # Workflow abstractions
    "BaseWorkflow",
    "register_workflow",
    "workflow_registry",
]
