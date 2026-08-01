"""
icore.workflows.examples.entity_extraction - Entity-relationship extraction.

A workflow that reads document content and extracts entity relationships:

    1. Extract:   Identify entities and raw relationships from the text via LLM
    2. Normalize: Clean and normalize the extracted entities/relationships
    3. Format:    Structure the results into a knowledge graph format

This demonstrates:
    - Custom task input classes for each stage
    - LLM-based entity extraction with structured prompts
    - Post-processing task (normalize) without LLM
    - Formatting task to produce structured output
    - DAG with linear chain: extract -> normalize -> format

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.entity_extraction import EntityExtractionWorkflow

    wf = EntityExtractionWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # Inject managers before execution...
    result = await wf.execute(ctx, {"text": "Alice works at Acme Corp..."})
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar

from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG
from icore.engine.registry import register_workflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Input Models
# ---------------------------------------------------------------------------

class ExtractEntitiesInput(BaseTaskInput):
    """Input for the entity extraction task."""

    text: str = Field(description="Document text to extract entities from")
    entity_types: list[str] = Field(
        default_factory=lambda: ["person", "organization", "location", "date", "event"],
        description="Entity types to extract",
    )


class NormalizeEntitiesInput(BaseTaskInput):
    """Input for the entity normalization task."""

    raw_entities: list[dict[str, Any]] = Field(
        description="Raw entity dicts from extraction (name, type, mentions)",
    )
    raw_relationships: list[dict[str, Any]] = Field(
        description="Raw relationship dicts (subject, predicate, object)",
    )


class FormatEntitiesInput(BaseTaskInput):
    """Input for the entity formatting task."""

    entities: list[dict[str, Any]] = Field(
        description="Normalized entity list",
    )
    relationships: list[dict[str, Any]] = Field(
        description="Normalized relationship list",
    )
    output_format: str = Field(
        default="json",
        description="Output format: 'json' or 'cytoscape'",
    )


# ---------------------------------------------------------------------------
# Task: Entity Extraction (LLM)
# ---------------------------------------------------------------------------

@register_task("extract_entities")
class ExtractEntitiesTask(BaseTask):
    """
    Extracts entities and relationships from text using the LLM.

    Sends a structured prompt to the LLM asking it to identify entities
    (people, organizations, locations, etc.) and their relationships
    in JSON format.
    """

    name: ClassVar[str] = "extract_entities"
    description: ClassVar[str] = "Extract entities and relationships from text via LLM"
    input_model: ClassVar[type[BaseTaskInput]] = ExtractEntitiesInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        """Obtain the model adapter."""
        self._model = ctx.get_model_adapter()

    async def execute(
        self, ctx: TaskContext, inp: ExtractEntitiesInput
    ) -> BaseTaskOutput:
        """Extract entities and relationships via LLM."""
        if not inp.text:
            return BaseTaskOutput.failure("Input text is empty")

        entity_types_str = ", ".join(inp.entity_types)

        prompt = (
            f"Analyze the following text and extract entities and their "
            f"relationships.\n\n"
            f"Entity types to identify: {entity_types_str}\n\n"
            f"Return a JSON object with this structure:\n"
            f'{{"entities": [{{"name": "...", "type": "...", "mentions": [...]}}], '
            f'"relationships": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}\n\n'
            f"Text:\n{inp.text}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an entity-relationship extraction system. "
                    "Always respond with valid JSON only, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._model.chat(messages=messages)
            content = response.get("content", "")

            # Parse the LLM's JSON response
            parsed = self._parse_json_response(content)

            if parsed is None:
                return BaseTaskOutput.failure(
                    "Failed to parse LLM response as JSON",
                    raw_response=content,
                )

            entities = parsed.get("entities", [])
            relationships = parsed.get("relationships", [])

            logger.info(
                "Extracted %d entities and %d relationships",
                len(entities),
                len(relationships),
            )

            return BaseTaskOutput.success(
                raw_entities=entities,
                raw_relationships=relationships,
            )
        except Exception as e:
            logger.error("Entity extraction failed: %s", e)
            return BaseTaskOutput.failure(f"Extraction failed: {e}")

    def _parse_json_response(self, content: str) -> dict[str, Any] | None:
        """Attempt to parse JSON from the LLM response, handling markdown."""
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding the first { ... } block
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    async def cleanup(self, ctx: TaskContext) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Task: Entity Normalization (no LLM)
# ---------------------------------------------------------------------------

@register_task("normalize_entities")
class NormalizeEntitiesTask(BaseTask):
    """
    Normalizes extracted entities and relationships.

    This is a pure-processing task (no LLM) that:
        - Deduplicates entities by name (case-insensitive)
        - Normalizes entity types to lowercase
        - Removes empty or whitespace-only names
        - Deduplicates relationships
    """

    name: ClassVar[str] = "normalize_entities"
    description: ClassVar[str] = "Normalize and deduplicate extracted entities"
    input_model: ClassVar[type[BaseTaskInput]] = NormalizeEntitiesInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(
        self, ctx: TaskContext, inp: NormalizeEntitiesInput
    ) -> BaseTaskOutput:
        """Normalize entities and relationships."""
        # Deduplicate entities by normalized name
        seen_names: dict[str, dict[str, Any]] = {}
        for entity in inp.raw_entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key not in seen_names:
                normalized = {
                    "name": name,
                    "type": str(entity.get("type", "unknown")).lower().strip(),
                    "mentions": entity.get("mentions", [name]),
                }
                seen_names[key] = normalized
            else:
                # Merge mentions
                existing_mentions = seen_names[key].get("mentions", [])
                new_mentions = entity.get("mentions", [])
                seen_names[key]["mentions"] = list(
                    set(existing_mentions + new_mentions)
                )

        entities = list(seen_names.values())

        # Deduplicate relationships
        seen_rels: set[str] = set()
        relationships: list[dict[str, Any]] = []
        for rel in inp.raw_relationships:
            subject = str(rel.get("subject", "")).strip()
            predicate = str(rel.get("predicate", "")).strip().lower()
            obj = str(rel.get("object", "")).strip()
            if not subject or not predicate or not obj:
                continue
            key = f"{subject.lower()}|{predicate}|{obj.lower()}"
            if key not in seen_rels:
                seen_rels.add(key)
                relationships.append(
                    {"subject": subject, "predicate": predicate, "object": obj}
                )

        logger.info(
            "Normalized to %d entities (from %d) and %d relationships (from %d)",
            len(entities),
            len(inp.raw_entities),
            len(relationships),
            len(inp.raw_relationships),
        )

        return BaseTaskOutput.success(
            entities=entities,
            relationships=relationships,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Entity Formatting (no LLM)
# ---------------------------------------------------------------------------

@register_task("format_entities")
class FormatEntitiesTask(BaseTask):
    """
    Formats normalized entities and relationships into structured output.

    Supports two output formats:
        - 'json':       Standard JSON with entities and relationships
        - 'cytoscape':  Cytoscape.js elements format for graph visualization
    """

    name: ClassVar[str] = "format_entities"
    description: ClassVar[str] = "Format entities into structured output (JSON or Cytoscape)"
    input_model: ClassVar[type[BaseTaskInput]] = FormatEntitiesInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(
        self, ctx: TaskContext, inp: FormatEntitiesInput
    ) -> BaseTaskOutput:
        """Format entities and relationships into the requested format."""
        if inp.output_format == "cytoscape":
            # Cytoscape.js elements format
            elements: list[dict[str, Any]] = []

            # Add nodes (entities)
            for entity in inp.entities:
                elements.append(
                    {
                        "data": {
                            "id": entity["name"],
                            "label": entity["name"],
                            "type": entity["type"],
                        }
                    }
                )

            # Add edges (relationships)
            for i, rel in enumerate(inp.relationships):
                elements.append(
                    {
                        "data": {
                            "id": f"edge_{i}",
                            "source": rel["subject"],
                            "target": rel["object"],
                            "label": rel["predicate"],
                        }
                    }
                )

            logger.info(
                "Formatted %d entities + %d relationships as Cytoscape elements",
                len(inp.entities),
                len(inp.relationships),
            )
            return BaseTaskOutput.success(
                format="cytoscape",
                elements=elements,
                node_count=len(inp.entities),
                edge_count=len(inp.relationships),
            )

        # Default: JSON format
        result = {
            "entities": inp.entities,
            "relationships": inp.relationships,
            "stats": {
                "entity_count": len(inp.entities),
                "relationship_count": len(inp.relationships),
            },
        }

        logger.info(
            "Formatted %d entities + %d relationships as JSON",
            len(inp.entities),
            len(inp.relationships),
        )
        return BaseTaskOutput.success(
            format="json",
            result=result,
            entity_count=len(inp.entities),
            relationship_count=len(inp.relationships),
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Workflow: Entity Extraction
# ---------------------------------------------------------------------------

@register_workflow("entity_extraction")
class EntityExtractionWorkflow(BaseWorkflow):
    """
    Multi-task workflow for entity-relationship extraction.

    Pipeline:
        1. extract_entities:    Extract raw entities/relationships via LLM
        2. normalize_entities:   Deduplicate and normalize
        3. format_entities:      Format into JSON or Cytoscape

    DAG:
        extract -> normalize -> format
    """

    name: ClassVar[str] = "entity_extraction"
    description: ClassVar[str] = (
        "Extract entity relationships from document text, normalize, "
        "and format into structured output"
    )

    def define(self) -> DAG:
        """Build the extract -> normalize -> format DAG."""
        dag = DAG()

        # Node 1: Extract entities via LLM
        dag.add_node(
            node_id="extract",
            task_name="extract_entities",
        )

        # Node 2: Normalize the extracted entities
        dag.add_node(
            node_id="normalize",
            task_name="normalize_entities",
            input_builder=lambda params, upstream: NormalizeEntitiesInput(
                raw_entities=upstream["extract"].data.get("raw_entities", []),
                raw_relationships=upstream["extract"].data.get("raw_relationships", []),
            ),
        )

        # Node 3: Format the normalized entities
        dag.add_node(
            node_id="format",
            task_name="format_entities",
            input_builder=lambda params, upstream: FormatEntitiesInput(
                entities=upstream["normalize"].data.get("entities", []),
                relationships=upstream["normalize"].data.get("relationships", []),
                output_format=params.get("output_format", "json"),
            ),
        )

        # Linear chain
        dag.add_edge("extract", "normalize")
        dag.add_edge("normalize", "format")

        return dag
