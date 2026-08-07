"""
icore.workflows.examples - Sample workflow implementations.

These examples demonstrate how to use the icore platform:
    - Defining custom task input classes (extending BaseTaskInput)
    - Implementing tasks (extending BaseTask)
    - Composing tasks into a DAG workflow (extending BaseWorkflow)
    - Using model adapters via TaskContext
    - Using database connections via TaskContext
    - Using vector store / graph store / media processor via TaskContext (v0.5)
    - Registering workflows with @register_workflow

Examples:
    document_summary:  Multi-task chunk -> summarize -> merge pipeline
    entity_extraction: Extract -> normalize -> format pipeline
    fraud_detection:   Event-driven order -> risk check -> alert (Kafka)
    weekly_report:      DB query -> LLM generate -> format pipeline
    rag_qa:             Embed query -> retrieve docs -> generate answer (v0.5)
    knowledge_graph:    Extract -> build graph -> GraphRAG query (v0.5)
    multimodal:         Image caption + OCR summary workflows (v0.5)
    agent_demo:         REACT + SUPERVISOR Agent nodes inside a DAG (v0.6)

Importing this package eagerly imports all example modules so that their
``@register_task`` / ``@register_workflow`` decorators execute and the
examples become discoverable through the default registries::

    import icore.workflows.examples  # noqa: F401  (registers all examples)
    from icore.engine.registry import workflow_registry
    assert "document_summary" in workflow_registry.list_workflows()
"""

from icore.workflows.examples import (  # noqa: F401
    agent_demo,
    document_summary,
    entity_extraction,
    fraud_detection,
    knowledge_graph,
    multimodal,
    rag_qa,
    report_export,
    svg_flow,
    weekly_report,
)

__all__ = [
    "agent_demo",
    "document_summary",
    "entity_extraction",
    "fraud_detection",
    "knowledge_graph",
    "multimodal",
    "rag_qa",
    "report_export",
    "svg_flow",
    "weekly_report",
]
