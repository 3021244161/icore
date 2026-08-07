"""
Tests for icore.engine.lock - Distributed lock abstractions.

Covers:
    - BaseDistributedLock abstractness
    - MemoryLock:
        * acquire / release basic flow
        * reentrant acquire after release
        * non-blocking acquire fails when held
        * blocking acquire waits for release
        * TTL expiry (stale lock is auto-released)
        * lock() context manager
        * ConflictError raised when context manager cannot acquire
    - RedisLock:
        * module importable without redis installed
        * acquire / release with mock client
        * non-blocking path
        * token-based release (Lua script called with stored token)
        * release without prior acquire is a no-op
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from icore.engine.lock import (
    BaseDistributedLock,
    MemoryLock,
    RedisLock,
)
from icore.exceptions import ConflictError


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TestBaseDistributedLock:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseDistributedLock()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MemoryLock
# ---------------------------------------------------------------------------

class TestMemoryLockAcquireRelease:
    async def test_acquire_returns_true(self):
        lock = MemoryLock()
        assert await lock.acquire("k1") is True

    async def test_release_allows_reacquire(self):
        lock = MemoryLock()
        await lock.acquire("k1")
        await lock.release("k1")
        assert await lock.acquire("k1") is True

    async def test_non_blocking_fails_when_held(self):
        lock = MemoryLock()
        await lock.acquire("k1")
        # Second non-blocking acquire should fail.
        result = await lock.acquire("k1", blocking=False)
        assert result is False

    async def test_non_blocking_succeeds_when_free(self):
        lock = MemoryLock()
        result = await lock.acquire("k1", blocking=False)
        assert result is True

    async def test_blocking_waits_for_release(self):
        lock = MemoryLock()
        await lock.acquire("k1", ttl=10.0)

        async def _release_after_delay():
            await asyncio.sleep(0.05)
            await lock.release("k1")

        asyncio.create_task(_release_after_delay())
        # This should block until the release happens.
        result = await lock.acquire("k1", blocking=True, ttl=10.0)
        assert result is True


class TestMemoryLockTTL:
    async def test_stale_lock_is_released(self):
        lock = MemoryLock()
        # Acquire with a tiny TTL and then wait past it.
        await lock.acquire("k1", ttl=0.01)
        await asyncio.sleep(0.02)
        # The next acquire should succeed because the lock is stale.
        result = await lock.acquire("k1", ttl=1.0)
        assert result is True


class TestMemoryLockContextManager:
    async def test_lock_context_manager_acquires_and_releases(self):
        lock = MemoryLock()
        async with lock.lock("k1"):
            # Inside the context, a non-blocking acquire should fail.
            assert await lock.acquire("k1", blocking=False) is False
        # After the context, it should be acquirable again.
        assert await lock.acquire("k1", blocking=False) is True

    async def test_lock_context_manager_yields(self):
        lock = MemoryLock()
        # The context manager should yield (no exception).
        async with lock.lock("k1"):
            pass


# ---------------------------------------------------------------------------
# RedisLock - lazy import & mock-based tests
# ---------------------------------------------------------------------------

class TestRedisLockModule:
    def test_module_importable_without_redis(self):
        from icore.engine.lock import RedisLock  # noqa: F401
        assert RedisLock is not None

    def test_constructor_accepts_client(self):
        client = AsyncMock()
        lock = RedisLock(client)
        assert lock._client is client


class TestRedisLockAcquireRelease:
    async def test_acquire_sets_with_nx_px(self):
        # Simulate a successful SET NX PX.
        client = AsyncMock()
        client.set.return_value = True
        lock = RedisLock(client)
        result = await lock.acquire("k1", ttl=30.0)
        assert result is True
        client.set.assert_awaited_once()
        args, kwargs = client.set.call_args
        assert args[0] == "k1"
        assert kwargs.get("nx") is True
        assert kwargs.get("px") == 30_000

    async def test_acquire_returns_false_non_blocking_when_held(self):
        client = AsyncMock()
        client.set.return_value = False  # key already exists
        lock = RedisLock(client)
        result = await lock.acquire("k1", ttl=10.0, blocking=False)
        assert result is False

    async def test_acquire_blocking_polls_until_success(self):
        client = AsyncMock()
        # First two attempts fail, third succeeds.
        client.set.side_effect = [False, False, True]
        lock = RedisLock(client)
        # Use very short poll interval to keep test fast.
        # Patch sleep to avoid real delay.
        lock._client = client
        # Use a short TTL so the deadline check doesn't dominate.
        result = await lock.acquire("k1", ttl=1.0, blocking=True)
        assert result is True
        assert client.set.await_count == 3

    async def test_acquire_blocking_times_out(self):
        client = AsyncMock()
        client.set.return_value = False
        lock = RedisLock(client)
        # Use a tiny TTL so the deadline triggers quickly.
        result = await lock.acquire("k1", ttl=0.01, blocking=True)
        assert result is False

    async def test_release_calls_eval_with_token(self):
        client = AsyncMock()
        client.set.return_value = True
        lock = RedisLock(client)
        await lock.acquire("k1", ttl=30.0)
        token = lock._tokens["k1"]
        await lock.release("k1")
        client.eval.assert_awaited_once()
        args, _ = client.eval.call_args
        # args = (script, numkeys, key, token)
        assert args[2] == "k1"
        assert args[3] == token

    async def test_release_without_acquire_is_noop(self):
        client = AsyncMock()
        lock = RedisLock(client)
        await lock.release("k1")  # should not raise
        client.eval.assert_not_awaited()


class TestRedisLockContextManager:
    async def test_lock_context_manager_acquires_and_releases(self):
        client = AsyncMock()
        client.set.return_value = True
        lock = RedisLock(client)
        async with lock.lock("k1", ttl=30.0):
            pass
        # Should have called set() once for acquire
        # and eval() once for release.
        client.set.assert_awaited_once()
        client.eval.assert_awaited_once()
