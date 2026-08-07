"""
Tests for icore.cache - LLM semantic cache module (v0.6).

Covers:
    - CacheEntry dataclass + is_expired()
    - L1Cache: set/get, miss, TTL expiry, LRU eviction, invalidate,
      clear, size, model isolation, hit_count, LRU access-order
    - L2Cache: set/get, similar hit, dissimilar miss, missing
      collection, invalidate, clear, size, model isolation, TTL expiry
    - SemanticCache: L1 hit, L1-miss→L2-hit→backfill, both-miss,
      double-write, stats (hits/misses/hit_rate/l1_hits/l2_hits),
      reset_stats, enable_l1=False, enable_l2=False
    - CachedModelAdapter: cache hit (no underlying call), miss → write
      → second-call hit, embed passthrough, health_check, stats
    - Helpers: _md5_hash, _cosine, _serialize_prompt
    - Factory: create_semantic_cache with/without vectorstore
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

import pytest

from icore.cache import (
    CacheEntry,
    CachedModelAdapter,
    L1Cache,
    L2Cache,
    SemanticCache,
    create_semantic_cache,
    _cosine,
    _md5_hash,
    _serialize_prompt,
)
from icore.vectorstore import InMemoryVectorStore, VectorDocument

from tests.conftest import FakeModelAdapter, make_model_config


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# A deterministic embedding function mapping known prompts to hand-picked
# vectors so that cosine similarity is fully predictable.

_KNOWN_EMBEDDINGS: dict[str, list[float]] = {
    "what is icore":           [1.0, 0.0, 0.0, 0.0],
    "tell me about icore":     [0.98, 0.1, 0.0, 0.0],   # cosine ≈ 0.998 with above
    "explain icore platform":  [0.95, 0.2, 0.05, 0.0],   # cosine ≈ 0.978 with above
    "hello world":             [0.0, 1.0, 0.0, 0.0],   # orthogonal
    "goodbye universe":        [0.0, 0.0, 1.0, 0.0],   # orthogonal
    "completely different":    [0.0, 0.0, 0.0, 1.0],   # orthogonal
}


async def _test_embed(text: str) -> list[float]:
    """Deterministic test embedding with known cosine relationships."""
    if text in _KNOWN_EMBEDDINGS:
        return list(_KNOWN_EMBEDDINGS[text])
    # Unknown prompt → deterministic 4-d vector from md5.
    h = hashlib.md5(text.encode("utf-8")).digest()
    return [b / 255.0 for b in h[:4]]


# ---------------------------------------------------------------------------
# Helpers (_md5_hash / _cosine / _serialize_prompt)
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_md5_hash_deterministic(self):
        assert _md5_hash("hello") == _md5_hash("hello")

    def test_md5_hash_differs(self):
        assert _md5_hash("hello") != _md5_hash("world")

    def test_cosine_identical(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_mismatched_length(self):
        assert _cosine([1.0, 0.0], [1.0]) == 0.0

    def test_cosine_zero_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_serialize_prompt_stable(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _serialize_prompt(msgs) == _serialize_prompt(msgs)

    def test_serialize_prompt_ignores_temperature(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _serialize_prompt(msgs, temperature=0.1) == \
            _serialize_prompt(msgs, temperature=0.9)

    def test_serialize_prompt_includes_tools(self):
        msgs = [{"role": "user", "content": "hi"}]
        a = _serialize_prompt(msgs, tools=[{"name": "t"}])
        b = _serialize_prompt(msgs, tools=[{"name": "t"}])
        assert a == b
        c = _serialize_prompt(msgs, tools=[{"name": "other"}])
        assert a != c


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

class TestCacheEntry:
    def test_defaults(self):
        e = CacheEntry(
            prompt="p", response="r", prompt_hash="h",
            model_id="m", created_at=time.time(),
        )
        assert e.hit_count == 0
        assert e.ttl_seconds == 3600.0
        assert e.metadata == {}

    def test_is_expired_false_when_fresh(self):
        e = CacheEntry(
            prompt="p", response="r", prompt_hash="h",
            model_id="m", created_at=time.time(), ttl_seconds=3600,
        )
        assert e.is_expired() is False

    def test_is_expired_true_when_old(self):
        e = CacheEntry(
            prompt="p", response="r", prompt_hash="h",
            model_id="m", created_at=time.time() - 100, ttl_seconds=1,
        )
        assert e.is_expired() is True

    def test_is_expired_false_when_no_ttl(self):
        e = CacheEntry(
            prompt="p", response="r", prompt_hash="h",
            model_id="m", created_at=time.time() - 9999, ttl_seconds=0,
        )
        assert e.is_expired() is False


# ---------------------------------------------------------------------------
# L1Cache
# ---------------------------------------------------------------------------

class TestL1Cache:
    async def test_set_get_basic(self):
        cache = L1Cache()
        entry = await cache.set("hello", "world", "model-A")
        assert isinstance(entry, CacheEntry)
        got = await cache.get("hello", "model-A")
        assert got is not None
        assert got.response == "world"
        assert got.prompt == "hello"
        assert got.model_id == "model-A"

    async def test_get_missing_returns_none(self):
        cache = L1Cache()
        assert await cache.get("nope", "model-A") is None

    async def test_ttl_expired_returns_none(self):
        cache = L1Cache()
        await cache.set("p1", "r1", "m1", ttl=0.05)
        await asyncio.sleep(0.1)
        assert await cache.get("p1", "m1") is None

    async def test_lru_eviction(self):
        cache = L1Cache(max_size=3)
        await cache.set("p1", "r1", "m1")
        await cache.set("p2", "r2", "m1")
        await cache.set("p3", "r3", "m1")
        # Access p1 → make it most-recently-used
        await cache.get("p1", "m1")
        # Insert p4 → evict LRU (p2)
        await cache.set("p4", "r4", "m1")
        assert await cache.size() == 3
        assert await cache.get("p2", "m1") is None   # evicted
        assert await cache.get("p1", "m1") is not None
        assert await cache.get("p3", "m1") is not None
        assert await cache.get("p4", "m1") is not None

    async def test_lru_eviction_without_access(self):
        cache = L1Cache(max_size=2)
        await cache.set("p1", "r1", "m1")
        await cache.set("p2", "r2", "m1")
        await cache.set("p3", "r3", "m1")  # evicts p1
        assert await cache.get("p1", "m1") is None
        assert await cache.get("p2", "m1") is not None
        assert await cache.get("p3", "m1") is not None

    async def test_invalidate(self):
        cache = L1Cache()
        await cache.set("p1", "r1", "m1")
        assert await cache.invalidate("p1", "m1") is True
        assert await cache.get("p1", "m1") is None
        # Invalidating a non-existent entry returns False
        assert await cache.invalidate("p1", "m1") is False

    async def test_clear(self):
        cache = L1Cache()
        await cache.set("p1", "r1", "m1")
        await cache.set("p2", "r2", "m1")
        count = await cache.clear()
        assert count == 2
        assert await cache.size() == 0

    async def test_size(self):
        cache = L1Cache()
        assert await cache.size() == 0
        await cache.set("p1", "r1", "m1")
        assert await cache.size() == 1
        await cache.set("p2", "r2", "m1")
        assert await cache.size() == 2

    async def test_model_isolation(self):
        cache = L1Cache()
        await cache.set("same prompt", "response-A", "model-A")
        await cache.set("same prompt", "response-B", "model-B")
        a = await cache.get("same prompt", "model-A")
        b = await cache.get("same prompt", "model-B")
        assert a is not None and b is not None
        assert a.response == "response-A"
        assert b.response == "response-B"

    async def test_hit_count_increment(self):
        cache = L1Cache()
        await cache.set("p1", "r1", "m1")
        e1 = await cache.get("p1", "m1")
        assert e1 is not None and e1.hit_count == 1
        e2 = await cache.get("p1", "m1")
        assert e2 is not None and e2.hit_count == 2

    async def test_overwrite_updates_entry(self):
        cache = L1Cache()
        await cache.set("p1", "old", "m1")
        await cache.set("p1", "new", "m1")
        got = await cache.get("p1", "m1")
        assert got is not None and got.response == "new"
        assert got.hit_count == 1  # fresh entry after overwrite


# ---------------------------------------------------------------------------
# L2Cache
# ---------------------------------------------------------------------------

class TestL2Cache:
    async def test_set_get_exact(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed, similarity_threshold=0.95)
        await l2.set("what is icore", "It is a platform.", "model-A")
        got = await l2.get("what is icore", "model-A")
        assert got is not None
        assert got.response == "It is a platform."
        assert got.model_id == "model-A"

    async def test_similar_hit(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed, similarity_threshold=0.95)
        await l2.set("what is icore", "It is a platform.", "model-A")
        # "tell me about icore" has cosine ≈ 0.998 with "what is icore"
        got = await l2.get("tell me about icore", "model-A")
        assert got is not None
        assert got.response == "It is a platform."
        assert "score" in got.metadata

    async def test_dissimilar_miss(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed, similarity_threshold=0.95)
        await l2.set("what is icore", "It is a platform.", "model-A")
        # "hello world" is orthogonal → below threshold
        got = await l2.get("hello world", "model-A")
        assert got is None

    async def test_missing_collection_graceful(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed, collection="nonexistent")
        # get without prior set → search returns []
        got = await l2.get("anything", "model-A")
        assert got is None

    async def test_invalidate(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed)
        await l2.set("p1", "r1", "m1")
        assert await l2.invalidate("p1", "m1") is True
        assert await l2.get("p1", "m1") is None
        # Invalidating non-existent → False
        assert await l2.invalidate("p1", "m1") is False

    async def test_clear(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed)
        await l2.set("p1", "r1", "m1")
        await l2.set("p2", "r2", "m1")
        count = await l2.clear()
        assert count == 2
        assert await l2.size() == 0
        # get after clear returns None
        assert await l2.get("p1", "m1") is None

    async def test_size(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed)
        assert await l2.size() == 0
        await l2.set("p1", "r1", "m1")
        assert await l2.size() == 1
        await l2.set("p2", "r2", "m1")
        assert await l2.size() == 2
        # Upsert: same prompt+model → no size increase
        await l2.set("p1", "r1-updated", "m1")
        assert await l2.size() == 2

    async def test_model_isolation(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed)
        await l2.set("what is icore", "response-A", "model-A")
        await l2.set("what is icore", "response-B", "model-B")
        a = await l2.get("what is icore", "model-A")
        b = await l2.get("what is icore", "model-B")
        assert a is not None and b is not None
        assert a.response == "response-A"
        assert b.response == "response-B"

    async def test_ttl_expiry(self):
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed, default_ttl=0.05)
        await l2.set("p1", "r1", "m1", ttl=0.05)
        await asyncio.sleep(0.1)
        assert await l2.get("p1", "m1") is None

    async def test_create_collection_called(self):
        """L2Cache.set should call create_collection (wrapped in try/except)."""
        vs = InMemoryVectorStore()
        l2 = L2Cache(vs, _test_embed, collection="my_coll")
        await l2.set("p1", "r1", "m1")
        # Collection should exist now
        assert "my_coll" in vs._collections


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class TestSemanticCache:
    async def test_l1_hit(self):
        sc = create_semantic_cache(enable_l1=True, enable_l2=False)
        await sc.set("p1", "r1", "m1")
        got = await sc.get("p1", "m1")
        assert got is not None
        assert got.response == "r1"
        assert sc.hits == 1
        assert sc.l1_hits == 1
        assert sc.l2_hits == 0
        assert sc.misses == 0

    async def test_l1_miss_l2_hit_backfill(self):
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
            enable_l1=True, enable_l2=True,
        )
        # Write directly to L2, bypassing L1.
        await sc._l2.set("what is icore", "It is a platform.", "m1")
        assert await sc._l1.size() == 0
        # get → L1 miss → L2 hit → backfill to L1
        got = await sc.get("what is icore", "m1")
        assert got is not None
        assert got.response == "It is a platform."
        assert sc.l1_hits == 0
        assert sc.l2_hits == 1
        assert sc.misses == 0
        # L1 should now have the backfilled entry.
        assert await sc._l1.size() == 1
        # Second get → L1 hit
        got2 = await sc.get("what is icore", "m1")
        assert got2 is not None
        assert sc.l1_hits == 1

    async def test_both_miss(self):
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
            enable_l1=True, enable_l2=True,
        )
        got = await sc.get("nonexistent", "m1")
        assert got is None
        assert sc.misses == 1
        assert sc.hits == 0

    async def test_double_write(self):
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
            enable_l1=True, enable_l2=True,
        )
        await sc.set("p1", "r1", "m1")
        # L1 should have it
        assert await sc._l1.size() == 1
        # L2 should have it
        assert await sc._l2.size() == 1

    async def test_stats_hit_rate(self):
        sc = create_semantic_cache(enable_l1=True, enable_l2=False)
        # 2 hits, 1 miss → hit_rate = 2/3
        await sc.set("p1", "r1", "m1")
        await sc.get("p1", "m1")   # hit
        await sc.get("p1", "m1")   # hit
        await sc.get("p2", "m1")   # miss
        assert sc.hits == 2
        assert sc.misses == 1
        assert sc.hit_rate == pytest.approx(2 / 3)

    async def test_hit_rate_zero_when_empty(self):
        sc = create_semantic_cache(enable_l1=True, enable_l2=False)
        assert sc.hit_rate == 0.0

    async def test_reset_stats(self):
        sc = create_semantic_cache(enable_l1=True, enable_l2=False)
        await sc.set("p1", "r1", "m1")
        await sc.get("p1", "m1")   # hit
        await sc.get("p2", "m1")   # miss
        assert sc.hits > 0
        sc.reset_stats()
        assert sc.hits == 0
        assert sc.misses == 0
        assert sc.l1_hits == 0
        assert sc.l2_hits == 0

    async def test_disable_l1(self):
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
            enable_l1=False, enable_l2=True,
        )
        assert sc.l1 is None
        assert sc.l2 is not None
        await sc.set("what is icore", "r1", "m1")
        # Only L2 should have it
        assert await sc._l2.size() == 1
        got = await sc.get("what is icore", "m1")
        assert got is not None
        assert sc.l1_hits == 0
        assert sc.l2_hits == 1

    async def test_disable_l2(self):
        sc = create_semantic_cache(
            enable_l1=True, enable_l2=False,
        )
        assert sc.l1 is not None
        assert sc.l2 is None
        await sc.set("p1", "r1", "m1")
        got = await sc.get("p1", "m1")
        assert got is not None
        assert sc.l1_hits == 1
        assert sc.l2_hits == 0

    async def test_invalidate_both_layers(self):
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
            enable_l1=True, enable_l2=True,
        )
        await sc.set("p1", "r1", "m1")
        result = await sc.invalidate("p1", "m1")
        assert result is True
        assert await sc.get("p1", "m1") is None

    async def test_clear_both_layers(self):
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
            enable_l1=True, enable_l2=True,
        )
        await sc.set("p1", "r1", "m1")
        await sc.set("p2", "r2", "m1")
        total = await sc.clear()
        # 2 entries in L1 + 2 entries in L2 = 4
        assert total == 4
        assert await sc._l1.size() == 0
        assert await sc._l2.size() == 0

    async def test_l2_backfill_does_not_double_count(self):
        """L2 hit should backfill to L1 without counting as an L1 hit."""
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
            enable_l1=True, enable_l2=True,
        )
        await sc._l2.set("p1", "r1", "m1")
        await sc.get("p1", "m1")  # L2 hit
        assert sc.hits == 1
        assert sc.l2_hits == 1
        assert sc.l1_hits == 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_factory_no_vectorstore_disables_l2(self):
        sc = create_semantic_cache(enable_l1=True, enable_l2=True)
        assert sc.l1 is not None
        assert sc.l2 is None  # auto-disabled

    def test_factory_with_vectorstore_enables_l2(self):
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
        )
        assert sc.l1 is not None
        assert sc.l2 is not None

    def test_factory_disable_both(self):
        sc = create_semantic_cache(enable_l1=False, enable_l2=False)
        assert sc.l1 is None
        assert sc.l2 is None

    def test_factory_custom_params(self):
        vs = InMemoryVectorStore()
        sc = create_semantic_cache(
            vectorstore=vs, embed_fn=_test_embed,
            similarity_threshold=0.8,
            l1_max_size=50,
            default_ttl=60.0,
        )
        assert sc.l1._max_size == 50
        assert sc.l1._default_ttl == 60.0
        assert sc.l2._similarity_threshold == 0.8
        assert sc.l2._default_ttl == 60.0


# ---------------------------------------------------------------------------
# CachedModelAdapter
# ---------------------------------------------------------------------------

class TestCachedModelAdapter:
    async def test_miss_then_hit(self):
        adapter = FakeModelAdapter(make_model_config("test-model"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "test-model")

        messages = [{"role": "user", "content": "hello"}]
        r1 = await cached.chat(messages)
        assert r1["cached"] is False
        assert len(adapter.calls) == 1

        # Second call with identical messages → cache hit
        r2 = await cached.chat(messages)
        assert r2["cached"] is True
        assert len(adapter.calls) == 1  # adapter NOT called again
        assert r2["content"] == r1["content"]

    async def test_hit_does_not_call_adapter(self):
        adapter = FakeModelAdapter(make_model_config("m1"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "m1")

        msgs = [{"role": "user", "content": "ping"}]
        await cached.chat(msgs)
        assert len(adapter.calls) == 1

        await cached.chat(msgs)
        assert len(adapter.calls) == 1  # still 1 — served from cache

    async def test_different_messages_are_separate_entries(self):
        adapter = FakeModelAdapter(make_model_config("m1"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "m1")

        await cached.chat([{"role": "user", "content": "a"}])
        await cached.chat([{"role": "user", "content": "b"}])
        assert len(adapter.calls) == 2  # two distinct prompts

    async def test_embed_passthrough(self):
        adapter = FakeModelAdapter(make_model_config("m1"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "m1")

        result = await cached.embed(["hello", "world"])
        # FakeModelAdapter.embed returns [[float(len(t))]]
        assert result == [[5.0], [5.0]]

    async def test_health_check_passthrough(self):
        adapter = FakeModelAdapter(make_model_config("m1"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "m1")

        assert await cached.health_check() is True

    async def test_close_passthrough(self):
        adapter = FakeModelAdapter(make_model_config("m1"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "m1")

        # Should not raise
        await cached.close()

    async def test_stats_after_multiple_calls(self):
        adapter = FakeModelAdapter(make_model_config("m1"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "m1")

        msgs_a = [{"role": "user", "content": "a"}]
        msgs_b = [{"role": "user", "content": "b"}]

        await cached.chat(msgs_a)   # miss
        await cached.chat(msgs_a)   # hit
        await cached.chat(msgs_b)   # miss
        await cached.chat(msgs_b)   # hit
        await cached.chat(msgs_a)   # hit

        assert cache.hits == 3
        assert cache.misses == 2
        assert len(adapter.calls) == 2  # only 2 actual model calls

    async def test_model_id_property(self):
        adapter = FakeModelAdapter(make_model_config("my-model"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "my-model")
        assert cached.model_id == "my-model"

    async def test_tools_affect_cache_key(self):
        """Different `tools` kwarg should produce different cache entries."""
        adapter = FakeModelAdapter(make_model_config("m1"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "m1")

        msgs = [{"role": "user", "content": "hi"}]
        await cached.chat(msgs, tools=[{"name": "t1"}])
        # Same messages but different tools → different key → miss
        await cached.chat(msgs, tools=[{"name": "t2"}])
        assert len(adapter.calls) == 2

    async def test_temperature_does_not_affect_cache_key(self):
        """temperature should be ignored in the cache key."""
        adapter = FakeModelAdapter(make_model_config("m1"))
        cache = create_semantic_cache(enable_l1=True, enable_l2=False)
        cached = CachedModelAdapter(adapter, cache, "m1")

        msgs = [{"role": "user", "content": "hi"}]
        await cached.chat(msgs, temperature=0.1)
        await cached.chat(msgs, temperature=0.9)  # same key → hit
        assert len(adapter.calls) == 1
