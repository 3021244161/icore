"""
Tests for icore.api.idempotency - Idempotency cache.

Covers:
    - BaseIdempotencyCache abstractness
    - MemoryIdempotencyCache:
        * set / get round-trip
        * get returns None for missing key
        * get returns None for expired key (TTL enforcement)
        * delete returns True/False appropriately
        * clear removes all entries and returns count
        * evict_expired removes only expired entries
        * TTL default is 24h (DEFAULT_TTL)
    - create_idempotency_cache factory:
        * memory backend
        * redis backend falls back to memory with warning
"""

from __future__ import annotations

import asyncio
import time

import pytest

from icore.api.idempotency import (
    DEFAULT_TTL,
    BaseIdempotencyCache,
    MemoryIdempotencyCache,
    create_idempotency_cache,
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TestBaseIdempotencyCache:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseIdempotencyCache()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MemoryIdempotencyCache - basic CRUD
# ---------------------------------------------------------------------------

class TestMemoryIdempotencyCacheCRUD:
    async def test_set_and_get(self):
        cache = MemoryIdempotencyCache()
        await cache.set("k1", {"result": "ok"})
        value = await cache.get("k1")
        assert value == {"result": "ok"}

    async def test_get_missing_returns_none(self):
        cache = MemoryIdempotencyCache()
        assert await cache.get("ghost") is None

    async def test_set_overwrites_existing(self):
        cache = MemoryIdempotencyCache()
        await cache.set("k1", {"v": 1})
        await cache.set("k1", {"v": 2})
        value = await cache.get("k1")
        assert value == {"v": 2}

    async def test_delete_existing_returns_true(self):
        cache = MemoryIdempotencyCache()
        await cache.set("k1", "v1")
        assert await cache.delete("k1") is True
        assert await cache.get("k1") is None

    async def test_delete_missing_returns_false(self):
        cache = MemoryIdempotencyCache()
        assert await cache.delete("ghost") is False

    async def test_clear_removes_all(self):
        cache = MemoryIdempotencyCache()
        await cache.set("k1", "v1")
        await cache.set("k2", "v2")
        count = await cache.clear()
        assert count == 2
        assert await cache.get("k1") is None
        assert await cache.get("k2") is None

    async def test_clear_on_empty_returns_zero(self):
        cache = MemoryIdempotencyCache()
        count = await cache.clear()
        assert count == 0


# ---------------------------------------------------------------------------
# MemoryIdempotencyCache - TTL
# ---------------------------------------------------------------------------

class TestMemoryIdempotencyCacheTTL:
    async def test_expired_entry_returns_none(self):
        cache = MemoryIdempotencyCache()
        await cache.set("k1", "v1", ttl=1)
        # Manually expire by rewriting the entry's timestamp.
        async with cache._lock:
            expiry, value = cache._store["k1"]
            cache._store["k1"] = (time.time() - 1, value)
        assert await cache.get("k1") is None

    async def test_get_lazy_eviction_removes_entry(self):
        cache = MemoryIdempotencyCache()
        await cache.set("k1", "v1", ttl=1)
        async with cache._lock:
            expiry, value = cache._store["k1"]
            cache._store["k1"] = (time.time() - 1, value)
        await cache.get("k1")  # triggers eviction
        async with cache._lock:
            assert "k1" not in cache._store

    async def test_evict_expired_only_removes_expired(self):
        cache = MemoryIdempotencyCache()
        await cache.set("fresh", "v1", ttl=3600)
        await cache.set("stale", "v2", ttl=1)
        async with cache._lock:
            expiry, value = cache._store["stale"]
            cache._store["stale"] = (time.time() - 1, value)
        removed = await cache.evict_expired()
        assert removed == 1
        assert await cache.get("fresh") == "v1"
        assert await cache.get("stale") is None

    async def test_evict_expired_returns_zero_when_none_expired(self):
        cache = MemoryIdempotencyCache()
        await cache.set("k1", "v1", ttl=3600)
        removed = await cache.evict_expired()
        assert removed == 0


# ---------------------------------------------------------------------------
# Default TTL constant
# ---------------------------------------------------------------------------

class TestDefaultTTL:
    def test_default_ttl_is_24_hours(self):
        assert DEFAULT_TTL == 86400

    async def test_set_uses_default_ttl_when_omitted(self):
        cache = MemoryIdempotencyCache()
        await cache.set("k1", "v1")
        async with cache._lock:
            expiry, _ = cache._store["k1"]
        # Expiry should be ~now + 24h
        assert expiry > time.time() + 86300


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestCreateIdempotencyCache:
    def test_memory_backend(self):
        cache = create_idempotency_cache(backend="memory")
        assert isinstance(cache, MemoryIdempotencyCache)

    def test_redis_backend_falls_back_to_memory(self):
        cache = create_idempotency_cache(
            backend="redis", redis_url="redis://localhost:6379"
        )
        # Should fall back to memory implementation.
        assert isinstance(cache, MemoryIdempotencyCache)

    def test_redis_backend_without_url_uses_memory(self):
        cache = create_idempotency_cache(backend="redis")
        assert isinstance(cache, MemoryIdempotencyCache)

    def test_unknown_backend_uses_memory(self):
        cache = create_idempotency_cache(backend="unknown")
        assert isinstance(cache, MemoryIdempotencyCache)

    def test_custom_ttl(self):
        cache = create_idempotency_cache(backend="memory", ttl=60)
        assert cache._ttl == 60
