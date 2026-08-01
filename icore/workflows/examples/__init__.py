"""
icore.workflows.examples - Sample workflow implementations.

These examples demonstrate how to use the icore platform:
    - Defining custom task input classes (extending BaseTaskInput)
    - Implementing tasks (extending BaseTask)
    - Composing tasks into a DAG workflow (extending BaseWorkflow)
    - Using model adapters via TaskContext
    - Using database connections via TaskContext
    - Registering workflows with @register_workflow

Examples:
    document_summary:  Multi-task chunk -> summarize -> merge pipeline
    entity_extraction:  Extract -> normalize -> format pipeline
    weekly_report:      DB query -> LLM generate -> format pipeline

Importing this package eagerly imports all example modules so that their
``@register_task`` / ``@register_workflow`` decorators execute and the
examples become discoverable through the default registries::

    import icore.workflows.examples  # noqa: F401  (registers all examples)
    from icore.engine.registry import workflow_registry
    assert "document_summary" in workflow_registry.list_workflows()
"""

from icore.workflows.examples import (  # noqa: F401
    document_summary,
    entity_extraction,
    weekly_report,
)

__all__ = ["document_summary", "entity_extraction", "weekly_report"]
