"""
Tests for icore.graphstore - Graph database abstractions.

Covers:
    - GraphNode / GraphEdge dataclass defaults
    - BaseGraphStore abstractness
    - InMemoryGraphStore:
        * upsert_nodes (insert + merge)
        * upsert_edges (idempotent on source/target/type)
        * query (Cypher subset: MATCH (n) RETURN n)
        * query with WHERE param filter
        * query edge traversal (n)-[r]->(m)
        * get_subgraph (depth=1, depth=2)
        * delete_nodes (cascades to edges)
        * health_check
    - Neo4jAdapter importability without neo4j installed (lazy import)
"""

from __future__ import annotations

import pytest

from icore.graphstore import (
    BaseGraphStore,
    GraphEdge,
    GraphNode,
    InMemoryGraphStore,
    Neo4jAdapter,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestGraphNode:
    def test_required_id(self):
        n = GraphNode(id="n1")
        assert n.id == "n1"
        assert n.labels == []
        assert n.properties == {}

    def test_with_labels_and_props(self):
        n = GraphNode(
            id="n1",
            labels=["Person"],
            properties={"name": "Alice"},
        )
        assert n.labels == ["Person"]
        assert n.properties == {"name": "Alice"}


class TestGraphEdge:
    def test_required_fields(self):
        e = GraphEdge(source_id="a", target_id="b", type="KNOWS")
        assert e.source_id == "a"
        assert e.target_id == "b"
        assert e.type == "KNOWS"
        assert e.properties == {}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TestBaseGraphStore:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseGraphStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# InMemoryGraphStore - node upsert
# ---------------------------------------------------------------------------

class TestInMemoryGraphStoreNodes:
    async def test_upsert_nodes_returns_ids(self):
        gs = InMemoryGraphStore()
        ids = await gs.upsert_nodes(
            [
                GraphNode(id="n1", labels=["Person"]),
                GraphNode(id="n2", labels=["Person"]),
            ]
        )
        assert ids == ["n1", "n2"]

    async def test_upsert_nodes_merges_labels_and_props(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [GraphNode(id="n1", labels=["Person"], properties={"age": 30})]
        )
        await gs.upsert_nodes(
            [
                GraphNode(
                    id="n1",
                    labels=["Employee"],
                    properties={"name": "Alice"},
                )
            ]
        )
        results = await gs.query("MATCH (n) RETURN n")
        assert len(results) == 1
        node = results[0]["n"]
        assert set(node["labels"]) == {"Person", "Employee"}
        # age kept, name added
        assert node["properties"]["age"] == 30
        assert node["properties"]["name"] == "Alice"

    async def test_upsert_empty_list(self):
        gs = InMemoryGraphStore()
        ids = await gs.upsert_nodes([])
        assert ids == []


# ---------------------------------------------------------------------------
# InMemoryGraphStore - edge upsert
# ---------------------------------------------------------------------------

class TestInMemoryGraphStoreEdges:
    async def test_upsert_edges_idempotent(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [GraphNode(id="a"), GraphNode(id="b")]
        )
        edge = GraphEdge(source_id="a", target_id="b", type="KNOWS")
        await gs.upsert_edges([edge])
        await gs.upsert_edges([edge])
        results = await gs.query("MATCH (n)-[r]->(m) RETURN n, r, m")
        assert len(results) == 1

    async def test_upsert_edges_replaces_on_property_change(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [GraphNode(id="a"), GraphNode(id="b")]
        )
        await gs.upsert_edges(
            [GraphEdge("a", "b", "KNOWS", properties={"since": 2020})]
        )
        await gs.upsert_edges(
            [GraphEdge("a", "b", "KNOWS", properties={"since": 2024})]
        )
        results = await gs.query("MATCH (n)-[r]->(m) RETURN n, r, m")
        assert len(results) == 1
        assert results[0]["r"]["properties"]["since"] == 2024


# ---------------------------------------------------------------------------
# InMemoryGraphStore - query
# ---------------------------------------------------------------------------

class TestInMemoryGraphStoreQuery:
    async def test_query_all_nodes(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [
                GraphNode(id="n1", labels=["Person"]),
                GraphNode(id="n2", labels=["Person"]),
            ]
        )
        results = await gs.query("MATCH (n) RETURN n")
        assert len(results) == 2

    async def test_query_filter_by_label(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [
                GraphNode(id="n1", labels=["Person"]),
                GraphNode(id="n2", labels=["Company"]),
            ]
        )
        results = await gs.query("MATCH (n:Person) RETURN n")
        assert len(results) == 1
        assert results[0]["n"]["id"] == "n1"

    async def test_query_filter_by_param(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [
                GraphNode(id="n1", properties={"name": "Alice"}),
                GraphNode(id="n2", properties={"name": "Bob"}),
            ]
        )
        results = await gs.query(
            "MATCH (n) WHERE n.name = $name RETURN n",
            params={"name": "Alice"},
        )
        assert len(results) == 1
        assert results[0]["n"]["id"] == "n1"

    async def test_query_edge_traversal(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [GraphNode(id="a"), GraphNode(id="b")]
        )
        await gs.upsert_edges(
            [GraphEdge("a", "b", "KNOWS", properties={"weight": 1})]
        )
        results = await gs.query("MATCH (n)-[r]->(m) RETURN n, r, m")
        assert len(results) == 1
        assert results[0]["n"]["id"] == "a"
        assert results[0]["m"]["id"] == "b"
        assert results[0]["r"]["type"] == "KNOWS"

    async def test_query_unsupported_returns_empty(self):
        gs = InMemoryGraphStore()
        results = await gs.query("SOME UNKNOWN CYPHER")
        assert results == []


# ---------------------------------------------------------------------------
# InMemoryGraphStore - subgraph
# ---------------------------------------------------------------------------

class TestInMemoryGraphStoreSubgraph:
    async def test_get_subgraph_depth_1(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [GraphNode(id="a"), GraphNode(id="b"), GraphNode(id="c")]
        )
        await gs.upsert_edges(
            [
                GraphEdge("a", "b", "KNOWS"),
                GraphEdge("b", "c", "KNOWS"),
            ]
        )
        sg = await gs.get_subgraph(["a"], depth=1)
        ids = {n["id"] for n in sg["nodes"]}
        assert "a" in ids
        assert "b" in ids
        # depth=1 should NOT reach c
        assert "c" not in ids
        assert len(sg["edges"]) == 1

    async def test_get_subgraph_depth_2(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [GraphNode(id="a"), GraphNode(id="b"), GraphNode(id="c")]
        )
        await gs.upsert_edges(
            [
                GraphEdge("a", "b", "KNOWS"),
                GraphEdge("b", "c", "KNOWS"),
            ]
        )
        sg = await gs.get_subgraph(["a"], depth=2)
        ids = {n["id"] for n in sg["nodes"]}
        assert ids == {"a", "b", "c"}
        assert len(sg["edges"]) == 2

    async def test_get_subgraph_empty_ids(self):
        gs = InMemoryGraphStore()
        sg = await gs.get_subgraph([], depth=1)
        assert sg["nodes"] == []
        assert sg["edges"] == []

    async def test_get_subgraph_unknown_node(self):
        gs = InMemoryGraphStore()
        sg = await gs.get_subgraph(["ghost"], depth=1)
        assert sg["nodes"] == []
        assert sg["edges"] == []


# ---------------------------------------------------------------------------
# InMemoryGraphStore - delete
# ---------------------------------------------------------------------------

class TestInMemoryGraphStoreDelete:
    async def test_delete_nodes_returns_count(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [GraphNode(id="a"), GraphNode(id="b")]
        )
        removed = await gs.delete_nodes(["a"])
        assert removed == 1
        results = await gs.query("MATCH (n) RETURN n")
        assert len(results) == 1
        assert results[0]["n"]["id"] == "b"

    async def test_delete_nodes_cascades_to_edges(self):
        gs = InMemoryGraphStore()
        await gs.upsert_nodes(
            [GraphNode(id="a"), GraphNode(id="b")]
        )
        await gs.upsert_edges([GraphEdge("a", "b", "KNOWS")])
        await gs.delete_nodes(["a"])
        results = await gs.query("MATCH (n)-[r]->(m) RETURN n, r, m")
        assert results == []

    async def test_delete_unknown_returns_zero(self):
        gs = InMemoryGraphStore()
        removed = await gs.delete_nodes(["ghost"])
        assert removed == 0

    async def test_health_check(self):
        gs = InMemoryGraphStore()
        assert await gs.health_check() is True

    async def test_close_is_noop(self):
        gs = InMemoryGraphStore()
        await gs.close()


# ---------------------------------------------------------------------------
# Neo4jAdapter (lazy import)
# ---------------------------------------------------------------------------

class TestNeo4jAdapterLazyImport:
    def test_module_importable_without_neo4j(self):
        from icore.graphstore import Neo4jAdapter  # noqa: F401
        assert Neo4jAdapter is not None

    async def test_get_driver_raises_without_neo4j(self):
        adapter = Neo4jAdapter()
        adapter._driver = None
        try:
            import neo4j  # type: ignore  # noqa: F401
            pytest.skip("neo4j is installed; cannot test ImportError path")
        except ImportError:
            with pytest.raises(ImportError):
                await adapter._get_driver()

    def test_constructor_defaults(self):
        a = Neo4jAdapter()
        assert a._uri == "bolt://localhost:7687"
        assert a._username == "neo4j"
        assert a._password == "neo4j"
        assert a._database == "neo4j"
        assert a._max_pool == 10

    def test_constructor_overrides(self):
        a = Neo4jAdapter(
            uri="bolt://neo4j.example.com:7687",
            username="user",
            password="pass",
            database="prod",
            max_connection_pool_size=50,
        )
        assert a._uri == "bolt://neo4j.example.com:7687"
        assert a._username == "user"
        assert a._password == "pass"
        assert a._database == "prod"
        assert a._max_pool == 50
