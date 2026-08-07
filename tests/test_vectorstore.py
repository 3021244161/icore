"""
Tests for icore.vectorstore - Vector database abstractions.

Covers:
    - VectorDocument dataclass defaults
    - BaseVectorStore abstractness
    - InMemoryVectorStore:
        * create_collection / drop_collection
        * insert (with upsert semantics)
        * search (cosine similarity ranking + top_k)
        * search with filter_expr (== / > / and)
        * delete by ids
        * health_check
        * empty-collection search
    - _cosine helper (zero vectors, mismatched lengths)
    - _eval_simple_filter helper (== / > / AND / key existence)
    - MilvusAdapter importability without pymilvus (lazy import)
"""

from __future__ import annotations

import asyncio

import pytest

from icore.exceptions import VectorStoreError
from icore.vectorstore import (
    BaseVectorStore,
    InMemoryVectorStore,
    MilvusAdapter,
    VectorDocument,
    _cosine,
    _eval_simple_filter,
)


# ---------------------------------------------------------------------------
# VectorDocument
# ---------------------------------------------------------------------------

class TestVectorDocument:
    def test_required_fields(self):
        d = VectorDocument(id="d1", vector=[0.1, 0.2])
        assert d.id == "d1"
        assert d.vector == [0.1, 0.2]

    def test_defaults(self):
        d = VectorDocument(id="d1", vector=[1.0])
        assert d.metadata == {}
        assert d.text is None

    def test_equality(self):
        a = VectorDocument(id="d1", vector=[1.0], metadata={"k": "v"})
        b = VectorDocument(id="d1", vector=[1.0], metadata={"k": "v"})
        assert a == b


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TestBaseVectorStore:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------

class TestInMemoryVectorStoreCRUD:
    async def test_create_collection_idempotent(self):
        vs = InMemoryVectorStore()
        await vs.create_collection("c1", dim=3)
        await vs.create_collection("c1", dim=3)  # no error
        assert await vs.health_check() is True

    async def test_drop_collection(self):
        vs = InMemoryVectorStore()
        await vs.create_collection("c1", dim=3)
        await vs.drop_collection("c1")
        # Inserting into dropped collection re-creates it (setdefault).
        await vs.insert("c1", [VectorDocument(id="x", vector=[1.0])])
        results = await vs.search("c1", [1.0])
        assert len(results) == 1

    async def test_insert_returns_ids(self):
        vs = InMemoryVectorStore()
        docs = [
            VectorDocument(id="d1", vector=[1.0, 0.0, 0.0]),
            VectorDocument(id="d2", vector=[0.0, 1.0, 0.0]),
        ]
        ids = await vs.insert("c1", docs)
        assert ids == ["d1", "d2"]

    async def test_insert_upsert_replaces_existing(self):
        vs = InMemoryVectorStore()
        await vs.insert("c1", [VectorDocument(id="d1", vector=[1.0, 0.0])])
        await vs.insert("c1", [VectorDocument(id="d1", vector=[0.0, 1.0])])
        results = await vs.search("c1", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].id == "d1"
        # The replaced vector should now be [0.0, 1.0]
        assert results[0].vector == [0.0, 1.0]

    async def test_delete_returns_count(self):
        vs = InMemoryVectorStore()
        await vs.insert(
            "c1",
            [
                VectorDocument(id="d1", vector=[1.0]),
                VectorDocument(id="d2", vector=[2.0]),
                VectorDocument(id="d3", vector=[3.0]),
            ],
        )
        removed = await vs.delete("c1", ["d1", "d2", "missing"])
        assert removed == 2
        results = await vs.search("c1", [10.0])
        assert {r.id for r in results} == {"d3"}

    async def test_delete_from_unknown_collection(self):
        vs = InMemoryVectorStore()
        removed = await vs.delete("ghost", ["x"])
        assert removed == 0


class TestInMemoryVectorStoreSearch:
    async def test_search_ranks_by_cosine(self):
        vs = InMemoryVectorStore()
        await vs.insert(
            "c1",
            [
                VectorDocument(id="a", vector=[1.0, 0.0]),
                VectorDocument(id="b", vector=[0.0, 1.0]),
                VectorDocument(id="c", vector=[0.7, 0.7]),
            ],
        )
        results = await vs.search("c1", [1.0, 0.0], top_k=3)
        assert results[0].id == "a"  # exact match → highest cosine
        # b should be last (orthogonal)
        assert results[-1].id == "b"

    async def test_search_top_k_limit(self):
        vs = InMemoryVectorStore()
        await vs.insert(
            "c1",
            [
                VectorDocument(id=str(i), vector=[float(i)])
                for i in range(10)
            ],
        )
        results = await vs.search("c1", [10.0], top_k=3)
        assert len(results) == 3

    async def test_search_empty_collection(self):
        vs = InMemoryVectorStore()
        results = await vs.search("ghost", [1.0, 2.0])
        assert results == []

    async def test_search_with_filter_expr_equality(self):
        vs = InMemoryVectorStore()
        await vs.insert(
            "c1",
            [
                VectorDocument(id="a", vector=[1.0], metadata={"tag": "news"}),
                VectorDocument(id="b", vector=[1.0], metadata={"tag": "blog"}),
            ],
        )
        results = await vs.search(
            "c1", [1.0], filter_expr='tag == "news"'
        )
        assert len(results) == 1
        assert results[0].id == "a"

    async def test_search_with_filter_expr_numeric(self):
        vs = InMemoryVectorStore()
        await vs.insert(
            "c1",
            [
                VectorDocument(id="a", vector=[1.0], metadata={"ts": 100}),
                VectorDocument(id="b", vector=[1.0], metadata={"ts": 500}),
            ],
        )
        results = await vs.search("c1", [1.0], filter_expr="ts > 200")
        assert len(results) == 1
        assert results[0].id == "b"

    async def test_search_with_filter_expr_and(self):
        vs = InMemoryVectorStore()
        await vs.insert(
            "c1",
            [
                VectorDocument(
                    id="a", vector=[1.0], metadata={"tag": "news", "ts": 500}
                ),
                VectorDocument(
                    id="b", vector=[1.0], metadata={"tag": "news", "ts": 100}
                ),
                VectorDocument(
                    id="c", vector=[1.0], metadata={"tag": "blog", "ts": 500}
                ),
            ],
        )
        results = await vs.search(
            "c1",
            [1.0],
            filter_expr='tag == "news" and ts > 200',
        )
        assert len(results) == 1
        assert results[0].id == "a"

    async def test_search_with_filter_missing_key_excluded(self):
        vs = InMemoryVectorStore()
        await vs.insert(
            "c1",
            [
                VectorDocument(id="a", vector=[1.0], metadata={"tag": "news"}),
                VectorDocument(id="b", vector=[1.0]),  # no tag
            ],
        )
        results = await vs.search("c1", [1.0], filter_expr='tag == "news"')
        assert len(results) == 1


class TestInMemoryVectorStoreHealth:
    async def test_health_check_returns_true(self):
        vs = InMemoryVectorStore()
        assert await vs.health_check() is True

    async def test_close_is_noop(self):
        vs = InMemoryVectorStore()
        await vs.close()  # should not raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestCosineHelper:
    def test_identical_vectors(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_empty_vectors(self):
        assert _cosine([], []) == 0.0

    def test_mismatched_lengths(self):
        assert _cosine([1.0], [1.0, 0.0]) == 0.0


class TestEvalSimpleFilter:
    def test_empty_expr_returns_true(self):
        assert _eval_simple_filter("", {"a": 1}) is True

    def test_equality_string(self):
        assert _eval_simple_filter('a == "x"', {"a": "x"}) is True
        assert _eval_simple_filter('a == "y"', {"a": "x"}) is False

    def test_equality_int(self):
        assert _eval_simple_filter("a == 1", {"a": 1}) is True
        assert _eval_simple_filter("a == 2", {"a": 1}) is False

    def test_greater_than(self):
        assert _eval_simple_filter("a > 5", {"a": 10}) is True
        assert _eval_simple_filter("a > 10", {"a": 10}) is False

    def test_and_conjunction(self):
        expr = 'a == "x" and b > 5'
        assert _eval_simple_filter(expr, {"a": "x", "b": 10}) is True
        assert _eval_simple_filter(expr, {"a": "x", "b": 1}) is False
        assert _eval_simple_filter(expr, {"a": "y", "b": 10}) is False

    def test_missing_key_returns_false(self):
        assert _eval_simple_filter("missing == 1", {"a": 1}) is False

    def test_key_existence(self):
        # No operator → treats as 'key exists' filter.
        assert _eval_simple_filter("tag", {"tag": "news"}) is True
        assert _eval_simple_filter("tag", {"other": 1}) is False


# ---------------------------------------------------------------------------
# MilvusAdapter (lazy import path)
# ---------------------------------------------------------------------------

class TestMilvusAdapterLazyImport:
    def test_module_importable_without_pymilvus(self):
        # The fact that this test file imports cleanly already proves it,
        # but we add an explicit assertion for clarity.
        from icore.vectorstore import MilvusAdapter  # noqa: F401
        assert MilvusAdapter is not None

    async def test_get_client_raises_without_pymilvus(self):
        # Force the lazy-import path by constructing an adapter whose
        # _client is None. We can't actually uninstall pymilvus in CI,
        # so this test only runs when the import fails.
        adapter = MilvusAdapter()
        adapter._client = None
        try:
            import pymilvus  # type: ignore  # noqa: F401
            pytest.skip("pymilvus is installed; cannot test ImportError path")
        except ImportError:
            with pytest.raises(ImportError):
                await adapter._get_client()

    def test_constructor_defaults(self):
        a = MilvusAdapter()
        assert a._host == "localhost"
        assert a._port == 19530
        assert a._db_name == "icore"
        assert a._default_index_type == "IVF_FLAT"

    def test_constructor_overrides(self):
        a = MilvusAdapter(
            host="milvus.example.com",
            port=9091,
            db_name="prod",
            default_index_type="HNSW",
            default_metric="L2",
            pool_size=20,
        )
        assert a._host == "milvus.example.com"
        assert a._port == 9091
        assert a._db_name == "prod"
        assert a._default_index_type == "HNSW"
        assert a._default_metric == "L2"
        assert a._pool_size == 20
