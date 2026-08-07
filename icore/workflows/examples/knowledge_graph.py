"""
icore.workflows.examples.knowledge_graph - Knowledge graph construction & query.

A workflow that builds a knowledge graph from raw text and answers
questions over it (GraphRAG):

    1. Extract:   Use LLM to extract entities + relationships from text
    2. Build:     Upsert the entities/relationships into the graph store
                  (uses a distributed lock to serialize concurrent writes)
    3. Query:     Run a Cypher query to retrieve a subgraph, then ask
                  the LLM to synthesize an answer grounded in the graph

This demonstrates:
    - GraphStore integration via TaskContext.get_graphstore()
    - Distributed lock via TaskContext.get_lock() (concurrent write safety)
    - LLM entity extraction (reused from entity_extraction workflow)
    - Cypher query + LLM-grounded answer synthesis (GraphRAG)

DAG:
    extract -> build_graph -> query_graph

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.knowledge_graph import KnowledgeGraphWorkflow

    wf = KnowledgeGraphWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # Inject managers + graphstore + lock before execution...
    result = await wf.execute(ctx, {
        "text": "Alice works at Acme Corp. Acme Corp is based in Seattle.",
        "question": "Where is Acme Corp based?",
        "depth": 2,
    })
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
from icore.graphstore import GraphEdge, GraphNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Input Models
# ---------------------------------------------------------------------------

class ExtractEntitiesForGraphInput(BaseTaskInput):
    """Input for the LLM entity extraction task."""

    text: str = Field(description="Document text to extract entities from")
    entity_types: list[str] = Field(
        default_factory=lambda: ["person", "organization", "location", "event"],
        description="Entity types to identify",
    )


class BuildGraphInput(BaseTaskInput):
    """Input for the graph upsert task."""

    entities: list[dict[str, Any]] = Field(
        description="Entities to upsert (each must have name + type)",
    )
    relationships: list[dict[str, Any]] = Field(
        description="Relationships to upsert (subject, predicate, object)",
    )
    use_lock: bool = Field(
        default=True,
        description=(
            "Whether to acquire a distributed lock during upsert to "
            "serialize concurrent graph writes"
        ),
    )


class QueryGraphInput(BaseTaskInput):
    """Input for the graph query task."""

    question: str = Field(description="The user's question to answer")
    entities: list[dict[str, Any]] = Field(
        description="Entities extracted from the source text (used to seed the query)",
    )
    depth: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Subgraph depth (number of hops from seed entities)",
    )


# ---------------------------------------------------------------------------
# Task: Extract Entities (LLM)
# ---------------------------------------------------------------------------

@register_task("extract_entities_for_graph")
class ExtractEntitiesForGraphTask(BaseTask):
    """
    Extracts entities and relationships from text using the LLM.

    Sends a structured prompt to the LLM asking it to identify entities
    and relationships in JSON format. The output is consumed by the
    ``build_graph`` task to upsert into the graph store.
    """

    name: ClassVar[str] = "extract_entities_for_graph"
    description: ClassVar[str] = (
        "Extract entities and relationships from text via LLM (for graph build)"
    )
    input_model: ClassVar[type[BaseTaskInput]] = ExtractEntitiesForGraphInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        self._model = ctx.get_model_adapter()

    async def execute(
        self, ctx: TaskContext, inp: ExtractEntitiesForGraphInput
    ) -> BaseTaskOutput:
        if not inp.text:
            return BaseTaskOutput.failure("Input text is empty")

        entity_types_str = ", ".join(inp.entity_types)
        prompt = (
            f"Analyze the following text and extract entities and their "
            f"relationships.\n\n"
            f"Entity types to identify: {entity_types_str}\n\n"
            f"Return a JSON object with this structure:\n"
            f'{{"entities": [{{"name": "...", "type": "..."}}], '
            f'"relationships": [{{"subject": "...", "predicate": "...", '
            f'"object": "..."}}]}}\n\n'
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
            parsed = self._parse_json_response(content)

            if parsed is None:
                return BaseTaskOutput.failure(
                    "Failed to parse LLM response as JSON",
                    raw_response=content,
                )

            entities = parsed.get("entities", [])
            relationships = parsed.get("relationships", [])

            logger.info(
                "Extracted %d entities and %d relationships for graph build",
                len(entities),
                len(relationships),
            )

            return BaseTaskOutput.success(
                entities=entities,
                relationships=relationships,
            )
        except Exception as e:
            logger.error("Entity extraction (graph) failed: %s", e)
            return BaseTaskOutput.failure(f"Extraction failed: {e}")

    def _parse_json_response(self, content: str) -> dict[str, Any] | None:
        """Parse JSON from the LLM response, handling markdown wrapping."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
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
# Task: Build Graph (GraphStore + Lock)
# ---------------------------------------------------------------------------

@register_task("build_graph")
class BuildGraphTask(BaseTask):
    """
    Upserts extracted entities and relationships into the graph store.

    This task demonstrates:
        - GraphStore integration via ``ctx.get_graphstore()``
        - Distributed lock usage via ``ctx.get_lock()`` to serialize
          concurrent graph modifications (preventing partial-upsert races)

    Entities are stored as nodes labeled by their type; relationships
    are stored as directed edges typed by their predicate.
    """

    name: ClassVar[str] = "build_graph"
    description: ClassVar[str] = (
        "Upsert extracted entities + relationships into the graph store"
    )
    input_model: ClassVar[type[BaseTaskInput]] = BuildGraphInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(
        self, ctx: TaskContext, inp: BuildGraphInput
    ) -> BaseTaskOutput:
        if not inp.entities and not inp.relationships:
            return BaseTaskOutput.success(
                upserted_nodes=0,
                upserted_edges=0,
                note="No entities or relationships to upsert",
            )

        try:
            graphstore = ctx.get_graphstore()
        except RuntimeError as e:
            return BaseTaskOutput.failure(str(e))

        # Build GraphNode list from extracted entities.
        nodes: list[GraphNode] = []
        for ent in inp.entities:
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            ent_type = str(ent.get("type", "entity")).strip().lower() or "entity"
            nodes.append(
                GraphNode(
                    id=name,
                    labels=[ent_type],
                    properties={
                        "name": name,
                        "type": ent_type,
                        **(ent.get("mentions") or {}),
                    },
                )
            )

        # Build GraphEdge list from extracted relationships.
        edges: list[GraphEdge] = []
        for rel in inp.relationships:
            subject = str(rel.get("subject", "")).strip()
            predicate = str(rel.get("predicate", "")).strip().upper()
            obj = str(rel.get("object", "")).strip()
            if not subject or not predicate or not obj:
                continue
            edges.append(
                GraphEdge(
                    source_id=subject,
                    target_id=obj,
                    type=predicate,
                    properties={"predicate": predicate.lower()},
                )
            )

        async def _do_upsert() -> tuple[list[str], int]:
            node_ids = await graphstore.upsert_nodes(nodes)
            await graphstore.upsert_edges(edges)
            return node_ids, len(edges)

        try:
            if inp.use_lock and ctx.has_lock():
                lock = ctx.get_lock()
                async with lock.lock("graph:upsert:build_graph"):
                    node_ids, edge_count = await _do_upsert()
            else:
                node_ids, edge_count = await _do_upsert()

            logger.info(
                "Upserted %d nodes + %d edges into graph store (lock=%s)",
                len(node_ids),
                edge_count,
                inp.use_lock and ctx.has_lock(),
            )

            return BaseTaskOutput.success(
                upserted_nodes=len(node_ids),
                upserted_edges=edge_count,
                node_ids=node_ids,
                # Echo entities so the downstream query_graph task can
                # access them without a direct edge from extract.
                entities=inp.entities,
            )
        except Exception as e:
            logger.error("Graph upsert failed: %s", e)
            return BaseTaskOutput.failure(f"Graph build failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Query Graph (GraphStore + LLM)
# ---------------------------------------------------------------------------

@register_task("query_graph")
class QueryGraphTask(BaseTask):
    """
    Queries the graph store for a subgraph and synthesizes an answer via LLM.

    This is a GraphRAG-style task: it pulls a depth-N subgraph around the
    extracted entities and asks the LLM to ground its answer in the
    retrieved graph structure (nodes + edges).
    """

    name: ClassVar[str] = "query_graph"
    description: ClassVar[str] = (
        "Query graph store for a subgraph and synthesize an answer via LLM"
    )
    input_model: ClassVar[type[BaseTaskInput]] = QueryGraphInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        self._model = ctx.get_model_adapter()

    async def execute(
        self, ctx: TaskContext, inp: QueryGraphInput
    ) -> BaseTaskOutput:
        if not inp.question:
            return BaseTaskOutput.failure("Question is empty")

        try:
            graphstore = ctx.get_graphstore()
        except RuntimeError as e:
            return BaseTaskOutput.failure(str(e))

        # Seed the subgraph query with the extracted entity names.
        seed_ids: list[str] = []
        for ent in inp.entities:
            name = str(ent.get("name", "")).strip()
            if name:
                seed_ids.append(name)

        try:
            if seed_ids:
                subgraph = await graphstore.get_subgraph(
                    node_ids=seed_ids,
                    depth=inp.depth,
                )
            else:
                # Fall back to listing all nodes (depth-0 subgraph).
                subgraph = await graphstore.get_subgraph(
                    node_ids=[],
                    depth=0,
                )
        except Exception as e:
            logger.error("Subgraph retrieval failed: %s", e)
            return BaseTaskOutput.failure(f"Graph query failed: {e}")

        # Format the subgraph into a textual context for the LLM.
        nodes = subgraph.get("nodes", []) if isinstance(subgraph, dict) else []
        edges = subgraph.get("edges", []) if isinstance(subgraph, dict) else []

        if not nodes and not edges:
            context_block = "[Graph is empty. No entities or relationships found.]"
        else:
            node_lines: list[str] = []
            for n in nodes:
                node_id = n.get("id", "?") if isinstance(n, dict) else "?"
                labels = n.get("labels", []) if isinstance(n, dict) else []
                props = n.get("properties", {}) if isinstance(n, dict) else {}
                node_lines.append(
                    f"- Node {node_id} (labels={labels}, props={props})"
                )
            edge_lines: list[str] = []
            for e in edges:
                src = e.get("source_id", "?") if isinstance(e, dict) else "?"
                tgt = e.get("target_id", "?") if isinstance(e, dict) else "?"
                etype = e.get("type", "?") if isinstance(e, dict) else "?"
                edge_lines.append(f"- {src} -[{etype}]-> {tgt}")
            context_block = (
                "Nodes:\n" + "\n".join(node_lines)
                + "\n\nEdges:\n" + "\n".join(edge_lines)
            )

        prompt = (
            f"Answer the user's question based on the knowledge graph below. "
            f"The graph is represented as nodes and edges.\n\n"
            f"Knowledge graph:\n{context_block}\n\n"
            f"User question: {inp.question}\n\n"
            f"Answer:"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a GraphRAG assistant. Ground your answer in "
                    "the provided graph structure and cite the relationships "
                    "you used."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._model.chat(messages=messages)
            answer = response.get("content", "")

            logger.info(
                "Generated GraphRAG answer (%d chars, %d nodes + %d edges in subgraph)",
                len(answer),
                len(nodes),
                len(edges),
            )

            return BaseTaskOutput.success(
                answer=answer,
                question=inp.question,
                node_count=len(nodes),
                edge_count=len(edges),
                subgraph=subgraph if isinstance(subgraph, dict) else {},
            )
        except Exception as e:
            logger.error("GraphRAG answer generation failed: %s", e)
            return BaseTaskOutput.failure(f"Answer generation failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Workflow: Knowledge Graph (GraphRAG)
# ---------------------------------------------------------------------------

@register_workflow("knowledge_graph")
class KnowledgeGraphWorkflow(BaseWorkflow):
    """
    Knowledge graph construction + GraphRAG QA workflow.

    Pipeline:
        1. extract_entities_for_graph:  Extract entities + relationships via LLM
        2. build_graph:                 Upsert them into the graph store
                                        (under a distributed lock)
        3. query_graph:                  Query a subgraph and synthesize an
                                        answer via LLM (GraphRAG)

    DAG:
        extract -> build_graph -> query_graph

    This workflow demonstrates graphstore + lock + LLM integration: the
    first task extracts entities via LLM, the second upserts them into
    the graph store under a distributed lock (concurrent-write safety),
    and the third queries a depth-N subgraph and asks the LLM to
    synthesize a grounded answer.
    """

    name: ClassVar[str] = "knowledge_graph"
    description: ClassVar[str] = (
        "Extract entities via LLM -> upsert into graph store (with lock) -> "
        "query subgraph and synthesize GraphRAG answer"
    )

    def define(self) -> DAG:
        """Build the extract -> build -> query DAG."""
        dag = DAG()

        # Node 1: Extract entities and relationships from text
        dag.add_node(
            node_id="extract",
            task_name="extract_entities_for_graph",
        )

        # Node 2: Build the graph by upserting entities + relationships
        dag.add_node(
            node_id="build_graph",
            task_name="build_graph",
            input_builder=lambda params, upstream: BuildGraphInput(
                entities=upstream["extract"].data.get("entities", []),
                relationships=upstream["extract"].data.get("relationships", []),
                use_lock=params.get("use_lock", True),
            ),
        )

        # Node 3: Query the graph and synthesize an answer.
        # Note: entities come from workflow params (re-extracted by the
        # query task itself is not ideal); we instead pass the entities
        # through build_graph's output (which echoes them) since
        # build_graph is the direct predecessor of query_graph.
        # However, build_graph doesn't echo entities — so we use params
        # to carry the question, and build_graph passes through the
        # extracted entities via its success output.
        dag.add_node(
            node_id="query_graph",
            task_name="query_graph",
            input_builder=lambda params, upstream: QueryGraphInput(
                question=params.get("question", ""),
                entities=upstream["build_graph"].data.get("entities", []),
                depth=params.get("depth", 2),
            ),
        )

        # Linear dependency chain
        dag.add_edge("extract", "build_graph")
        dag.add_edge("build_graph", "query_graph")

        return dag
