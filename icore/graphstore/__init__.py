"""
icore.graphstore - Graph database abstractions and adapters.

Provides:
    - ``GraphNode``:        Dataclass for a graph node.
    - ``GraphEdge``:        Dataclass for a graph edge.
    - ``BaseGraphStore``:   Abstract interface for graph stores.
    - ``Neo4jAdapter``:     Neo4j-backed implementation (lazy import).
    - ``InMemoryGraphStore``: Pure-Python in-memory implementation
                                used by tests and small deployments.

Module dependency:
    graphstore depends on icore.exceptions only. The Neo4j driver
    (``neo4j``) is imported lazily so this package is importable
    without neo4j installed.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from icore.exceptions import GraphStoreError

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """
    A graph node.

    Attributes:
        id:         Unique node identifier.
        labels:     Node labels (e.g. ``["Person", "Employee"]``).
        properties: Free-form properties dict.
    """

    id: str
    labels: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """
    A directed graph edge.

    Attributes:
        source_id: Source node ID.
        target_id: Target node ID.
        type:      Relationship type (e.g. ``"KNOWS"``).
        properties: Free-form properties dict.
    """

    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


class BaseGraphStore(ABC):
    """
    Abstract graph store interface.

    Implementations: ``Neo4jAdapter``, ``InMemoryGraphStore``.
    TaskContext injects a concrete instance accessible via
    ``ctx.get_graphstore()``.
    """

    @abstractmethod
    async def upsert_nodes(
        self, nodes: list[GraphNode]
    ) -> list[str]:
        """Insert or update nodes by id. Return the list of node IDs."""
        raise NotImplementedError

    @abstractmethod
    async def upsert_edges(self, edges: list[GraphEdge]) -> None:
        """Insert or update edges. Idempotent on (source, target, type)."""
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        cypher: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return rows as dicts."""
        raise NotImplementedError

    @abstractmethod
    async def get_subgraph(
        self,
        node_ids: list[str],
        depth: int = 1,
    ) -> dict[str, Any]:
        """
        Return the subgraph centered on ``node_ids`` up to ``depth`` hops.

        Returns a dict with ``nodes`` and ``edges`` lists.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(self, ids: list[str]) -> int:
        """Delete nodes (and their incident edges). Return count deleted."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """Lightweight connectivity check."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release connection resources. Default no-op."""
        pass


# ---------------------------------------------------------------------------
# In-memory implementation (offline tests, small deployments)
# ---------------------------------------------------------------------------


class InMemoryGraphStore(BaseGraphStore):
    """
    Pure-Python in-memory graph store.

    Stores nodes keyed by id and edges in a list. Suitable for unit
    tests, demos, and small graphs (<10k nodes).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._lock = asyncio.Lock()

    async def upsert_nodes(
        self, nodes: list[GraphNode]
    ) -> list[str]:
        async with self._lock:
            for n in nodes:
                existing = self._nodes.get(n.id)
                if existing is None:
                    self._nodes[n.id] = n
                else:
                    # Merge labels (union, preserve order).
                    merged_labels = list(existing.labels)
                    for lab in n.labels:
                        if lab not in merged_labels:
                            merged_labels.append(lab)
                    merged_props = dict(existing.properties)
                    merged_props.update(n.properties)
                    self._nodes[n.id] = GraphNode(
                        id=n.id,
                        labels=merged_labels,
                        properties=merged_props,
                    )
            return [n.id for n in nodes]

    async def upsert_edges(self, edges: list[GraphEdge]) -> None:
        async with self._lock:
            for e in edges:
                # Replace existing edge with same (source, target, type).
                self._edges[:] = [
                    x
                    for x in self._edges
                    if not (
                        x.source_id == e.source_id
                        and x.target_id == e.target_id
                        and x.type == e.type
                    )
                ]
                self._edges.append(e)

    async def query(
        self,
        cypher: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Very small Cypher subset for tests.

        Supports:
            - ``MATCH (n) RETURN n``
            - ``MATCH (n:Label) RETURN n``
            - ``MATCH (n) WHERE n.id = $id RETURN n``
            - ``MATCH (n)-[r]->(m) RETURN n, r, m``
        """
        async with self._lock:
            nodes = list(self._nodes.values())
            edges = list(self._edges)

        cypher_stripped = cypher.strip()

        # Edge traversal: MATCH (n)-[r]->(m) RETURN n, r, m
        # Check this BEFORE the simple node match because the edge
        # pattern also starts with "MATCH (n)" and contains "RETURN n".
        if ")-[r]->(" in cypher_stripped and "RETURN n, r, m" in cypher_stripped:
            result: list[dict[str, Any]] = []
            for e in edges:
                src = self._nodes.get(e.source_id)
                tgt = self._nodes.get(e.target_id)
                if src is None or tgt is None:
                    continue
                result.append(
                    {
                        "n": {
                            "id": src.id,
                            "labels": list(src.labels),
                            "properties": dict(src.properties),
                        },
                        "r": {
                            "type": e.type,
                            "properties": dict(e.properties),
                        },
                        "m": {
                            "id": tgt.id,
                            "labels": list(tgt.labels),
                            "properties": dict(tgt.properties),
                        },
                    }
                )
            return result

        # Whole-graph node return.
        # Match patterns:
        #   MATCH (n) RETURN n
        #   MATCH (n:Label) RETURN n
        #   MATCH (n) WHERE n.k = $p RETURN n
        # We detect this by checking that the MATCH clause closes
        # immediately after the variable (with optional :Label).
        if cypher_stripped.startswith("MATCH (n") and "RETURN n" in cypher_stripped:
            # Extract the part within the first (...) of the MATCH clause.
            # For "MATCH (n)" → "n"; for "MATCH (n:Label)" → "n:Label".
            match_head = cypher_stripped[len("MATCH "):].lstrip()
            # match_head now starts with "(...".
            closing_paren = match_head.find(")")
            if closing_paren == -1:
                logger.debug(
                    "InMemoryGraphStore: malformed MATCH clause: %s", cypher
                )
                return []
            inner = match_head[:closing_paren].lstrip("(")
            label_filter: Optional[str] = None
            if ":" in inner:
                # e.g. "n:Label" → label "Label"
                label_filter = inner.split(":", 1)[1].strip()

            prop_filter_key = None
            prop_filter_val = None
            if "WHERE" in cypher_stripped:
                where_clause = cypher_stripped.split("WHERE", 1)[1]
                where_clause = where_clause.split("RETURN", 1)[0]
                if "=" in where_clause and "$" in where_clause:
                    key_part, val_part = where_clause.split("=", 1)
                    key_part = key_part.replace("n.", "").strip()
                    val_part = val_part.strip()
                    if val_part.startswith("$"):
                        param_name = val_part[1:].strip()
                        param_vals = params or {}
                        prop_filter_key = key_part
                        prop_filter_val = param_vals.get(param_name)

            result = []
            for node in nodes:
                if label_filter and label_filter not in node.labels:
                    continue
                if (
                    prop_filter_key
                    and node.properties.get(prop_filter_key) != prop_filter_val
                ):
                    continue
                result.append(
                    {
                        "n": {
                            "id": node.id,
                            "labels": list(node.labels),
                            "properties": dict(node.properties),
                        }
                    }
                )
            return result

        # Unknown query: return empty list rather than failing.
        logger.debug("InMemoryGraphStore: unsupported Cypher: %s", cypher)
        return []

    async def get_subgraph(
        self,
        node_ids: list[str],
        depth: int = 1,
    ) -> dict[str, Any]:
        async with self._lock:
            visited_ids: set[str] = set(node_ids)
            frontier: set[str] = set(node_ids)

            # BFS to discover all reachable nodes within `depth` hops.
            for _ in range(max(depth, 0)):
                next_frontier: set[str] = set()
                for e in self._edges:
                    if e.source_id in frontier and e.target_id not in visited_ids:
                        next_frontier.add(e.target_id)
                        visited_ids.add(e.target_id)
                    elif e.target_id in frontier and e.source_id not in visited_ids:
                        next_frontier.add(e.source_id)
                        visited_ids.add(e.source_id)
                if not next_frontier:
                    break
                frontier = next_frontier

            # Collect ALL edges between visited nodes (complete subgraph).
            # This includes edges between seed nodes that BFS traversal
            # would miss (since both endpoints were already visited).
            sub_edges = [
                e
                for e in self._edges
                if e.source_id in visited_ids and e.target_id in visited_ids
            ]

            sub_nodes = [
                self._nodes[nid]
                for nid in visited_ids
                if nid in self._nodes
            ]
            return {
                "nodes": [
                    {
                        "id": n.id,
                        "labels": list(n.labels),
                        "properties": dict(n.properties),
                    }
                    for n in sub_nodes
                ],
                "edges": [
                    {
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "type": e.type,
                        "properties": dict(e.properties),
                    }
                    for e in sub_edges
                ],
            }

    async def delete_nodes(self, ids: list[str]) -> int:
        async with self._lock:
            target = set(ids)
            before = len(self._nodes)
            for i in ids:
                self._nodes.pop(i, None)
            self._edges[:] = [
                e
                for e in self._edges
                if e.source_id not in target and e.target_id not in target
            ]
            return before - len(self._nodes)

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Neo4j adapter (lazy import)
# ---------------------------------------------------------------------------


class Neo4jAdapter(BaseGraphStore):
    """
    Neo4j graph store adapter.

    Uses ``neo4j.AsyncGraphDatabase`` for native async access. The
    driver is imported lazily so this module is importable without
    neo4j installed.

    Supports:
        - Connection pooling (driver-level).
        - Parameterized Cypher (anti-injection).
        - Auto transactions (``run()``) and explicit transactions.
        - Multi-database selection (enterprise edition).
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "neo4j",
        database: str = "neo4j",
        max_connection_pool_size: int = 10,
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._max_pool = max_connection_pool_size
        self._driver: Any = None
        self._lock = asyncio.Lock()

    async def _get_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import AsyncGraphDatabase  # type: ignore
        except ImportError as e:
            raise ImportError(
                "neo4j is required for Neo4jAdapter. "
                "Install with: pip install neo4j"
            ) from e
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
            max_connection_pool_size=self._max_pool,
        )
        return self._driver

    async def upsert_nodes(
        self, nodes: list[GraphNode]
    ) -> list[str]:
        if not nodes:
            return []
        driver = await self._get_driver()
        try:
            # UNWIND + MERGE for idempotent upsert.
            labels_for_all = set()
            for n in nodes:
                labels_for_all.update(n.labels)
            # Use a generic MERGE on id; set labels viaAPOC if needed.
            # For simplicity, MERGE on (id) only and set properties.
            cypher = (
                "UNWIND $rows AS row "
                "MERGE (n:Entity {id: row.id}) "
                "SET n += row.properties "
                "RETURN n.id AS id"
            )
            rows = [
                {"id": n.id, "properties": n.properties}
                for n in nodes
            ]

            async def _run() -> list[dict[str, Any]]:
                result_list: list[dict[str, Any]] = []
                async with driver.session(database=self._database) as session:
                    res = await session.run(cypher, rows=rows)
                    async for r in res:
                        result_list.append(dict(r))
                return result_list

            records = await _run()
            ids = [r.get("id") for r in records]
            return [i for i in ids if i is not None]
        except Exception as e:
            raise GraphStoreError(f"Neo4j upsert_nodes failed: {e}") from e

    async def upsert_edges(self, edges: list[GraphEdge]) -> None:
        if not edges:
            return
        driver = await self._get_driver()
        try:
            # Note: relationship type cannot be parameterized in Cypher.
            # Group by type to issue one UNWIND per type.
            by_type: dict[str, list[GraphEdge]] = {}
            for e in edges:
                by_type.setdefault(e.type, []).append(e)
            statements: list[tuple[str, dict[str, Any]]] = []
            for rel_type, group in by_type.items():
                safe_type = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in rel_type
                ) or "REL"
                cypher = (
                    "UNWIND $rows AS row "
                    "MATCH (s:Entity {id: row.source_id}) "
                    "MATCH (t:Entity {id: row.target_id}) "
                    f"MERGE (s)-[r:{safe_type}]->(t) "
                    "SET r += row.properties"
                )
                rows = [
                    {
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "properties": e.properties,
                    }
                    for e in group
                ]
                statements.append((cypher, {"rows": rows}))

            async def _run() -> None:
                async with driver.session(database=self._database) as session:
                    for cypher, params in statements:
                        await session.run(cypher, **params)

            await _run()
        except Exception as e:
            raise GraphStoreError(f"Neo4j upsert_edges failed: {e}") from e

    async def query(
        self,
        cypher: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        driver = await self._get_driver()
        try:

            async def _run() -> list[dict[str, Any]]:
                result_list: list[dict[str, Any]] = []
                async with driver.session(database=self._database) as session:
                    res = await session.run(cypher, **(params or {}))
                    async for r in res:
                        # Convert neo4j Record to dict.
                        try:
                            result_list.append(dict(r))
                        except Exception:
                            result_list.append({"value": str(r)})
                return result_list

            return await _run()
        except Exception as e:
            raise GraphStoreError(f"Neo4j query failed: {e}") from e

    async def get_subgraph(
        self,
        node_ids: list[str],
        depth: int = 1,
    ) -> dict[str, Any]:
        if not node_ids:
            return {"nodes": [], "edges": []}
        driver = await self._get_driver()
        try:
            cypher = (
                "MATCH path = (n:Entity)-[*0..%d]-(m) "
                "WHERE n.id IN $ids "
                "WITH nodes(path) AS ns, relationships(path) AS rs "
                "UNWIND ns AS node "
                "WITH collect(DISTINCT node) AS ns, rs "
                "UNWIND rs AS rel "
                "WITH ns, collect(DISTINCT rel) AS rs "
                "RETURN ns, rs"
            ) % max(depth, 0)

            async def _run() -> dict[str, Any]:
                async with driver.session(database=self._database) as session:
                    res = await session.run(cypher, ids=list(node_ids))
                    nodes_out: list[dict[str, Any]] = []
                    edges_out: list[dict[str, Any]] = []
                    async for r in res:
                        ns = r.get("ns") or []
                        rs = r.get("rs") or []
                        for node in ns:
                            nodes_out.append(
                                {
                                    "id": node.get("id"),
                                    "labels": list(node.labels) if hasattr(node, "labels") else [],
                                    "properties": dict(node),
                                }
                            )
                        for rel in rs:
                            edges_out.append(
                                {
                                    "source_id": rel.start_node.get("id") if hasattr(rel, "start_node") else None,
                                    "target_id": rel.end_node.get("id") if hasattr(rel, "end_node") else None,
                                    "type": rel.type if hasattr(rel, "type") else None,
                                    "properties": dict(rel),
                                }
                            )
                    return {"nodes": nodes_out, "edges": edges_out}

            return await _run()
        except Exception as e:
            raise GraphStoreError(
                f"Neo4j get_subgraph failed: {e}"
            ) from e

    async def delete_nodes(self, ids: list[str]) -> int:
        if not ids:
            return 0
        driver = await self._get_driver()
        try:
            cypher = (
                "MATCH (n:Entity) WHERE n.id IN $ids "
                "DETACH DELETE n "
                "RETURN count(n) AS deleted"
            )

            async def _run() -> int:
                async with driver.session(database=self._database) as session:
                    res = await session.run(cypher, ids=list(ids))
                    count = 0
                    async for r in res:
                        count = int(r.get("deleted", 0))
                    return count

            return await _run()
        except Exception as e:
            raise GraphStoreError(
                f"Neo4j delete_nodes failed: {e}"
            ) from e

    async def health_check(self) -> bool:
        try:
            driver = await self._get_driver()

            async def _run() -> bool:
                async with driver.session(database=self._database) as session:
                    res = await session.run("RETURN 1 AS ok")
                    async for _ in res:
                        return True
                return True

            return await _run()
        except Exception as e:
            logger.debug("Neo4j health check failed: %s", e)
            return False

    async def close(self) -> None:
        if self._driver is not None:
            try:
                await self._driver.close()
            except Exception as e:  # pragma: no cover
                logger.warning("Neo4j close failed: %s", e)
            finally:
                self._driver = None


__all__ = [
    "GraphNode",
    "GraphEdge",
    "BaseGraphStore",
    "InMemoryGraphStore",
    "Neo4jAdapter",
]
