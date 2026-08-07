"""
Tests for icore.retrieval - hybrid retrieval (Dense + Sparse + Rerank).

Covers:
    - ``BM25Retriever``: tokenize (en/cn), index, search, score calc,
      empty corpus, repeated tokens, parameters validation, missing
      before index
    - ``reciprocal_rank_fusion``: ordering, ID merging across lists,
      weight influence, empty inputs, parameter validation
    - ``IdentityReranker`` / ``LLMReranker``: basic rerank, JSON parse
      fallback, LLM failure graceful fallback
    - ``CrossEncoderReranker``: lazy import path (no httpx installed)
    - ``HybridRetriever``: end-to-end, dense-only, sparse-only,
      rerank on/off, collection switching, BM25 / reranker failure
      graceful degradation
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from icore.exceptions import ValidationError, VectorStoreError
from icore.retrieval import (
    BM25Retriever,
    BaseReranker,
    CrossEncoderReranker,
    HybridRetriever,
    IdentityReranker,
    LLMReranker,
    reciprocal_rank_fusion,
)
from icore.vectorstore import InMemoryVectorStore, VectorDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    doc_id: str, text: str, vector: list[float] | None = None
) -> VectorDocument:
    return VectorDocument(
        id=doc_id,
        vector=vector or [0.0],
        text=text,
    )


# ---------------------------------------------------------------------------
# BM25Retriever - tokenize
# ---------------------------------------------------------------------------


class TestBM25Tokenize:
    def test_english_lowercased(self) -> None:
        bm25 = BM25Retriever()
        tokens = bm25._tokenize("Hello World Foo")
        assert tokens == ["hello", "world", "foo"]

    def test_punctuation_dropped(self) -> None:
        bm25 = BM25Retriever()
        tokens = bm25._tokenize("Hello, world! Foo-bar baz.")
        # "Foo-bar" splits into "foo", "bar" because - is not in [A-Za-z0-9_]
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens

    def test_chinese_per_char(self) -> None:
        bm25 = BM25Retriever()
        tokens = bm25._tokenize("机器学习很有趣")
        assert tokens == ["机", "器", "学", "习", "很", "有", "趣"]

    def test_mixed_cn_en(self) -> None:
        bm25 = BM25Retriever()
        tokens = bm25._tokenize("RAG 检索增强生成")
        assert "rag" in tokens
        assert "检" in tokens
        assert "索" in tokens
        assert "增" in tokens

    def test_empty_string(self) -> None:
        bm25 = BM25Retriever()
        assert bm25._tokenize("") == []
        assert bm25._tokenize(None) == []  # type: ignore[arg-type]

    def test_preserves_duplicates(self) -> None:
        """Token list should preserve duplicates (for tf counting)."""
        bm25 = BM25Retriever()
        tokens = bm25._tokenize("foo foo bar")
        assert tokens.count("foo") == 2
        assert tokens.count("bar") == 1


# ---------------------------------------------------------------------------
# BM25Retriever - index + search
# ---------------------------------------------------------------------------


class TestBM25IndexSearch:
    def test_index_requires_non_empty(self) -> None:
        bm25 = BM25Retriever()
        with pytest.raises(ValidationError):
            bm25.index([])

    def test_search_before_index_raises(self) -> None:
        bm25 = BM25Retriever()
        with pytest.raises(ValidationError):
            asyncio.run(bm25.search("foo"))

    def test_search_returns_empty_for_no_match(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "alpha beta")])
        results = asyncio.run(bm25.search("zzz"))
        assert results == []

    def test_basic_search_orders_by_relevance(self) -> None:
        bm25 = BM25Retriever()
        bm25.index(
            [
                _make_doc("d1", "machine learning is fun"),
                _make_doc("d2", "deep learning models"),
                _make_doc("d3", "cooking recipes"),
            ]
        )
        results = asyncio.run(bm25.search("machine learning", top_k=3))
        # d1 contains both "machine" and "learning" → highest score
        assert results[0].id == "d1"
        # d3 has no matching tokens → excluded
        ids = [r.id for r in results]
        assert "d3" not in ids

    def test_search_top_k_limit(self) -> None:
        bm25 = BM25Retriever()
        bm25.index(
            [_make_doc(f"d{i}", f"doc number {i}") for i in range(5)]
        )
        results = asyncio.run(bm25.search("doc", top_k=2))
        assert len(results) == 2

    def test_search_metadata_contains_bm25_score(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "hello world")])
        results = asyncio.run(bm25.search("hello", top_k=1))
        assert results[0].metadata["bm25_score"] > 0

    def test_search_invalid_top_k(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "hello")])
        with pytest.raises(ValueError):
            asyncio.run(bm25.search("hello", top_k=0))

    def test_index_replaces_previous(self) -> None:
        """Calling index() twice should replace, not append."""
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "alpha")])
        bm25.index([_make_doc("d2", "beta")])
        # Searching for alpha should now return empty (d1 gone)
        results = asyncio.run(bm25.search("alpha"))
        assert results == []
        # Searching for beta returns d2
        results = asyncio.run(bm25.search("beta"))
        assert len(results) == 1
        assert results[0].id == "d2"

    def test_doc_with_none_text_skipped(self) -> None:
        bm25 = BM25Retriever()
        bm25.index(
            [
                VectorDocument(id="d1", vector=[0.0], text=None),
                _make_doc("d2", "hello world"),
            ]
        )
        results = asyncio.run(bm25.search("hello"))
        assert len(results) == 1
        assert results[0].id == "d2"


# ---------------------------------------------------------------------------
# BM25Retriever - score calculation
# ---------------------------------------------------------------------------


class TestBM25Score:
    def test_score_zero_for_no_overlap(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "alpha beta")])
        # Token not in index → score 0
        assert bm25._bm25_score(["zzz"], "d1") == 0.0

    def test_score_positive_for_match(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "alpha beta")])
        score = bm25._bm25_score(["alpha"], "d1")
        assert score > 0.0

    def test_score_zero_for_unknown_doc(self) -> None:
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "alpha")])
        assert bm25._bm25_score(["alpha"], "ghost") == 0.0

    def test_more_specific_doc_scores_higher(self) -> None:
        """A doc with rare matching terms should score higher than
        a doc where the term is diluted by many other tokens."""
        bm25 = BM25Retriever(k1=1.5, b=0.75)
        bm25.index(
            [
                _make_doc("d1", "python python python"),  # 3 tokens, all match
                _make_doc(
                    "d2",
                    "python java c++ rust go javascript typescript "
                    "kotlin scala haskell",
                ),  # 11 tokens, only 1 matches "python"
            ]
        )
        results = asyncio.run(bm25.search("python", top_k=2))
        # d1 has higher density → should rank first
        assert results[0].id == "d1"

    def test_repeated_query_token_increases_score(self) -> None:
        """Query with repeated token should score higher than single occurrence."""
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "alpha alpha alpha")])
        single = bm25._bm25_score(["alpha"], "d1")
        triple = bm25._bm25_score(["alpha", "alpha", "alpha"], "d1")
        assert triple > single

    def test_constructor_validates_params(self) -> None:
        with pytest.raises(ValueError):
            BM25Retriever(k1=-1.0)
        with pytest.raises(ValueError):
            BM25Retriever(b=1.5)
        with pytest.raises(ValueError):
            BM25Retriever(b=-0.1)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


class TestRRF:
    def test_fuses_and_orders_by_score(self) -> None:
        dense = [_make_doc("a", "x"), _make_doc("b", "y"), _make_doc("c", "z")]
        sparse = [_make_doc("b", "y"), _make_doc("a", "x"), _make_doc("d", "w")]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        ids = [d.id for d, _ in fused]
        # a and b appear in both → highest scores
        assert "a" in ids[:2]
        assert "b" in ids[:2]
        # d only in sparse, c only in dense → lower scores
        assert "c" in ids
        assert "d" in ids

    def test_doc_in_both_lists_gets_higher_score(self) -> None:
        dense = [_make_doc("shared", "x"), _make_doc("only_dense", "y")]
        sparse = [_make_doc("shared", "x"), _make_doc("only_sparse", "z")]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        by_id = {d.id: s for d, s in fused}
        # shared appears in both lists → higher fused score
        assert by_id["shared"] > by_id["only_dense"]
        assert by_id["shared"] > by_id["only_sparse"]

    def test_k_smaller_increases_score_gap(self) -> None:
        """Smaller k → larger gap between rank 1 and rank 2."""
        dense = [_make_doc("a", "x"), _make_doc("b", "y")]
        sparse: list[VectorDocument] = []
        fused_k1 = reciprocal_rank_fusion(dense, sparse, k=1)
        fused_k60 = reciprocal_rank_fusion(dense, sparse, k=60)
        # With sparse empty, only dense contributes; both lists have a at rank 1
        score_a_k1 = dict((d.id, s) for d, s in fused_k1)["a"]
        score_a_k60 = dict((d.id, s) for d, s in fused_k60)["a"]
        # 1 / (1+1) = 0.5; 1 / (60+1) ≈ 0.0164
        assert score_a_k1 > score_a_k60

    def test_weight_influence(self) -> None:
        """Skewed weights favor one retriever's ordering."""
        dense = [_make_doc("a", "x"), _make_doc("b", "y")]
        sparse = [_make_doc("b", "y"), _make_doc("a", "x")]
        # All weight on dense → a (dense rank 1) should win
        fused = reciprocal_rank_fusion(dense, sparse, k=60, weights=(1.0, 0.0))
        assert fused[0][0].id == "a"
        # All weight on sparse → b (sparse rank 1) should win
        fused = reciprocal_rank_fusion(dense, sparse, k=60, weights=(0.0, 1.0))
        assert fused[0][0].id == "b"

    def test_empty_inputs(self) -> None:
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_one_empty_list(self) -> None:
        dense = [_make_doc("a", "x"), _make_doc("b", "y")]
        fused = reciprocal_rank_fusion(dense, [], k=60)
        assert len(fused) == 2
        assert [d.id for d, _ in fused] == ["a", "b"]

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([], [], k=0)

    def test_invalid_weights(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([], [], weights=(-0.1, 0.5))
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([], [], weights=(0.0, 0.0))

    def test_fused_score_non_negative(self) -> None:
        dense = [_make_doc("a", "x")]
        sparse = [_make_doc("a", "x")]
        fused = reciprocal_rank_fusion(dense, sparse)
        assert all(score >= 0 for _, score in fused)


# ---------------------------------------------------------------------------
# IdentityReranker / LLMReranker
# ---------------------------------------------------------------------------


class _FakeModelAdapter:
    """Test fake model adapter for LLMReranker (no network)."""

    def __init__(self, response_content: str) -> None:
        self._response_content = response_content
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(messages)
        return {
            "content": self._response_content,
            "role": "assistant",
            "model": "fake",
            "usage": {"total_tokens": 1},
            "finish_reason": "stop",
        }


class TestIdentityReranker:
    async def test_returns_input_unchanged(self) -> None:
        reranker = IdentityReranker()
        docs = [_make_doc(f"d{i}", f"text {i}") for i in range(5)]
        result = await reranker.rerank("query", docs, top_k=3)
        assert len(result) == 3
        assert [d.id for d in result] == ["d0", "d1", "d2"]

    async def test_top_k_larger_than_input(self) -> None:
        reranker = IdentityReranker()
        docs = [_make_doc("d0", "x")]
        result = await reranker.rerank("query", docs, top_k=10)
        assert len(result) == 1


class TestLLMReranker:
    async def test_rerank_orders_by_llm_scores(self) -> None:
        # LLM says d2 > d0 > d1
        llm_response = (
            '[{"doc_id": "d0", "score": 0.5}, '
            '{"doc_id": "d1", "score": 0.2}, '
            '{"doc_id": "d2", "score": 0.9}]'
        )
        adapter = _FakeModelAdapter(llm_response)
        reranker = LLMReranker(adapter)
        docs = [_make_doc(f"d{i}", f"text {i}") for i in range(3)]
        result = await reranker.rerank("query", docs, top_k=3)
        assert [d.id for d in result] == ["d2", "d0", "d1"]

    async def test_rerank_respects_top_k(self) -> None:
        llm_response = (
            '[{"doc_id": "d0", "score": 0.1}, {"doc_id": "d1", "score": 0.9}]'
        )
        adapter = _FakeModelAdapter(llm_response)
        reranker = LLMReranker(adapter)
        docs = [_make_doc(f"d{i}", f"text {i}") for i in range(2)]
        result = await reranker.rerank("query", docs, top_k=1)
        assert len(result) == 1
        assert result[0].id == "d1"

    async def test_rerank_empty_documents(self) -> None:
        adapter = _FakeModelAdapter("[]")
        reranker = LLMReranker(adapter)
        result = await reranker.rerank("query", [], top_k=3)
        assert result == []

    async def test_rerank_invalid_top_k(self) -> None:
        adapter = _FakeModelAdapter("[]")
        reranker = LLMReranker(adapter)
        docs = [_make_doc("d0", "x")]
        result = await reranker.rerank("query", docs, top_k=0)
        assert result == []

    async def test_rerank_falls_back_on_unparseable_output(self) -> None:
        # LLM returns garbage → should fall back to original order
        adapter = _FakeModelAdapter("I cannot help with that.")
        reranker = LLMReranker(adapter)
        docs = [_make_doc(f"d{i}", f"text {i}") for i in range(3)]
        result = await reranker.rerank("query", docs, top_k=2)
        # Falls back to input order
        assert [d.id for d in result] == ["d0", "d1"]

    async def test_rerank_falls_back_on_chat_exception(self) -> None:
        class FailingAdapter:
            async def chat(self, messages, **kwargs):
                raise RuntimeError("LLM down")

        reranker = LLMReranker(FailingAdapter())  # type: ignore[arg-type]
        docs = [_make_doc(f"d{i}", f"text {i}") for i in range(3)]
        result = await reranker.rerank("query", docs, top_k=2)
        # Falls back to input order
        assert [d.id for d in result] == ["d0", "d1"]

    async def test_rerank_extracts_json_array_from_text(self) -> None:
        """LLM wraps JSON in natural language — should still parse."""
        llm_response = (
            "Here are the scores:\n"
            '[{"doc_id": "d0", "score": 0.9}, {"doc_id": "d1", "score": 0.3}]\n'
            "Hope that helps!"
        )
        adapter = _FakeModelAdapter(llm_response)
        reranker = LLMReranker(adapter)
        docs = [_make_doc(f"d{i}", f"text {i}") for i in range(2)]
        result = await reranker.rerank("query", docs, top_k=2)
        assert result[0].id == "d0"

    def test_constructor_requires_adapter(self) -> None:
        with pytest.raises(ValueError):
            LLMReranker(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CrossEncoderReranker - lazy import
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    def test_constructor_validates(self) -> None:
        with pytest.raises(ValueError):
            CrossEncoderReranker(endpoint="", api_key="x")
        with pytest.raises(ValueError):
            CrossEncoderReranker(endpoint=None, api_key="x")  # type: ignore[arg-type]

    async def test_rerank_empty_documents(self) -> None:
        reranker = CrossEncoderReranker(
            endpoint="http://localhost:8080/score", api_key="x"
        )
        result = await reranker.rerank("query", [], top_k=3)
        assert result == []

    async def test_raises_without_httpx(self) -> None:
        """When httpx is not installed, rerank should raise ImportError."""
        reranker = CrossEncoderReranker(
            endpoint="http://localhost:8080/score", api_key="x"
        )
        docs = [_make_doc("d0", "hello")]
        with patch("builtins.__import__", side_effect=ImportError("no httpx")):
            with pytest.raises(ImportError):
                await reranker.rerank("query", docs, top_k=1)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


async def _seed_vectorstore() -> InMemoryVectorStore:
    vs = InMemoryVectorStore()
    docs = [
        _make_doc("d1", "machine learning intro", [1.0, 0.0]),
        _make_doc("d2", "deep learning models", [0.9, 0.1]),
        _make_doc("d3", "cooking recipes guide", [0.0, 1.0]),
        _make_doc("d4", "learning python programming", [0.8, 0.2]),
    ]
    await vs.insert("default", docs)
    return vs


# Same docs seeded into a fresh BM25 index — shared by HybridRetriever tests.
_SEED_DOCS = [
    _make_doc("d1", "machine learning intro", [1.0, 0.0]),
    _make_doc("d2", "deep learning models", [0.9, 0.1]),
    _make_doc("d3", "cooking recipes guide", [0.0, 1.0]),
    _make_doc("d4", "learning python programming", [0.8, 0.2]),
]


class TestHybridRetriever:
    def test_constructor_validates(self) -> None:
        with pytest.raises(ValueError):
            HybridRetriever(vectorstore=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            HybridRetriever(
                vectorstore=InMemoryVectorStore(), rrf_k=0
            )
        with pytest.raises(ValueError):
            HybridRetriever(
                vectorstore=InMemoryVectorStore(), dense_weight=-0.1, sparse_weight=0.5
            )
        with pytest.raises(ValueError):
            HybridRetriever(
                vectorstore=InMemoryVectorStore(), collection=""
            )

    async def test_end_to_end_dense_plus_sparse_with_rerank(self) -> None:
        vs = await _seed_vectorstore()
        bm25 = BM25Retriever()
        # BM25 needs the same docs text as the vectorstore
        bm25.index(_SEED_DOCS)
        # LLM reranker returns d2 > d1
        llm_response = (
            '[{"doc_id": "d2", "score": 0.95}, {"doc_id": "d1", "score": 0.7}, '
            '{"doc_id": "d4", "score": 0.4}]'
        )
        reranker = LLMReranker(_FakeModelAdapter(llm_response))
        hybrid = HybridRetriever(
            vectorstore=vs,
            bm25=bm25,
            reranker=reranker,
            dense_weight=0.5,
            sparse_weight=0.5,
        )
        results = await hybrid.retrieve(
            query="learning",
            query_vector=[1.0, 0.0],
            top_k=4,
            rerank_top_k=2,
            use_sparse=True,
            use_rerank=True,
        )
        # After rerank, top should be d2 (LLM scored highest)
        assert results[0].id == "d2"
        assert len(results) == 2

    async def test_dense_only(self) -> None:
        vs = await _seed_vectorstore()
        hybrid = HybridRetriever(vectorstore=vs)  # no bm25 / no reranker
        results = await hybrid.retrieve(
            query="learning",
            query_vector=[1.0, 0.0],
            top_k=3,
            rerank_top_k=3,
            use_sparse=False,
            use_rerank=False,
        )
        # d1 has exact vector match → rank 1
        assert results[0].id == "d1"
        assert len(results) == 3

    async def test_dense_only_with_bm25_disabled(self) -> None:
        """Even with BM25 configured, use_sparse=False → no sparse path."""
        vs = await _seed_vectorstore()
        bm25 = BM25Retriever()
        bm25.index([_make_doc("d1", "machine learning", [1.0, 0.0])])
        hybrid = HybridRetriever(vectorstore=vs, bm25=bm25)
        results = await hybrid.retrieve(
            query="cooking",
            query_vector=[0.0, 1.0],
            top_k=2,
            rerank_top_k=2,
            use_sparse=False,
            use_rerank=False,
        )
        # Pure dense → d3 (vector [0,1] match)
        assert "d3" in [r.id for r in results]

    async def test_rerank_off(self) -> None:
        vs = await _seed_vectorstore()
        bm25 = BM25Retriever()
        bm25.index(_SEED_DOCS)
        # Use a reranker that would obviously reorder — verify it's bypassed
        reranker = LLMReranker(
            _FakeModelAdapter('[{"doc_id": "d3", "score": 1.0}]')
        )
        hybrid = HybridRetriever(vectorstore=vs, bm25=bm25, reranker=reranker)
        results = await hybrid.retrieve(
            query="learning",
            query_vector=[1.0, 0.0],
            top_k=3,
            rerank_top_k=3,
            use_rerank=False,
        )
        # rerank bypassed → top should be dense winner (d1) not d3
        assert results[0].id == "d1"

    async def test_collection_switching(self) -> None:
        vs = InMemoryVectorStore()
        await vs.insert("collection_a", [_make_doc("a1", "x", [1.0])])
        await vs.insert("collection_b", [_make_doc("b1", "y", [1.0])])
        hybrid = HybridRetriever(vectorstore=vs, collection="collection_a")
        # Initially searches collection_a
        results = await hybrid.retrieve(
            query="x", query_vector=[1.0], top_k=5, rerank_top_k=5,
            use_sparse=False, use_rerank=False,
        )
        assert results[0].id == "a1"
        # Switch collection
        hybrid.collection = "collection_b"
        results = await hybrid.retrieve(
            query="y", query_vector=[1.0], top_k=5, rerank_top_k=5,
            use_sparse=False, use_rerank=False,
        )
        assert results[0].id == "b1"

    async def test_collection_setter_validates(self) -> None:
        vs = InMemoryVectorStore()
        hybrid = HybridRetriever(vectorstore=vs, collection="default")
        with pytest.raises(ValueError):
            hybrid.collection = ""

    async def test_sparse_failure_falls_back_to_dense(self) -> None:
        """If BM25.search raises, dense results should still come back."""
        vs = await _seed_vectorstore()

        class _BrokenBM25(BM25Retriever):
            async def search(self, query, top_k=10):  # type: ignore[override]
                raise RuntimeError("BM25 broken")

        bm25 = _BrokenBM25()
        bm25.index([_make_doc("d1", "x", [1.0])])
        hybrid = HybridRetriever(vectorstore=vs, bm25=bm25)
        # Should not raise; falls back to dense-only
        results = await hybrid.retrieve(
            query="learning",
            query_vector=[1.0, 0.0],
            top_k=3,
            rerank_top_k=3,
            use_sparse=True,
            use_rerank=False,
        )
        assert results[0].id == "d1"

    async def test_rerank_failure_falls_back_to_fused_order(self) -> None:
        vs = await _seed_vectorstore()

        class _FailingReranker(BaseReranker):
            async def rerank(self, query, documents, top_k=5):
                raise RuntimeError("reranker broken")

        hybrid = HybridRetriever(vectorstore=vs, reranker=_FailingReranker())
        results = await hybrid.retrieve(
            query="learning",
            query_vector=[1.0, 0.0],
            top_k=3,
            rerank_top_k=3,
            use_sparse=False,
            use_rerank=True,
        )
        # Falls back to fused (dense) order — d1 is dense winner
        assert results[0].id == "d1"

    async def test_dense_failure_raises(self) -> None:
        """If vectorstore.search fails, retrieve should raise VectorStoreError."""

        class _BrokenVectorStore(InMemoryVectorStore):
            async def search(self, collection, query_vector, top_k=10, filter_expr=None):  # type: ignore[override]
                raise RuntimeError("vectorstore broken")

        hybrid = HybridRetriever(vectorstore=_BrokenVectorStore())
        with pytest.raises(VectorStoreError):
            await hybrid.retrieve(
                query="x",
                query_vector=[1.0],
                top_k=3,
                rerank_top_k=3,
                use_sparse=False,
                use_rerank=False,
            )

    async def test_invalid_top_k(self) -> None:
        vs = await _seed_vectorstore()
        hybrid = HybridRetriever(vectorstore=vs)
        with pytest.raises(ValueError):
            await hybrid.retrieve(
                query="x", query_vector=[1.0], top_k=0, rerank_top_k=1
            )
        with pytest.raises(ValueError):
            await hybrid.retrieve(
                query="x", query_vector=[1.0], top_k=1, rerank_top_k=0
            )

    async def test_no_bm25_configured_with_use_sparse_true(self) -> None:
        """use_sparse=True but no BM25 → behaves like dense-only, no crash."""
        vs = await _seed_vectorstore()
        hybrid = HybridRetriever(vectorstore=vs)  # no bm25
        results = await hybrid.retrieve(
            query="learning",
            query_vector=[1.0, 0.0],
            top_k=3,
            rerank_top_k=3,
            use_sparse=True,
            use_rerank=False,
        )
        assert results[0].id == "d1"


# ---------------------------------------------------------------------------
# BaseReranker abstractness
# ---------------------------------------------------------------------------


class TestBaseReranker:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]
