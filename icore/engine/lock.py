"""
icore.engine.lock - Distributed lock abstractions.

Provides:
    - ``BaseDistributedLock``: Abstract interface.
    - ``MemoryLock``:          Single-process asyncio implementation.
    - ``RedisLock``:           Redis-backed distributed lock using
                                SET NX PX + token-checked Lua release.

All locks expose a uniform ``async with lock.lock(key, ttl): ...``
context manager that raises ``ConflictError`` when the lock cannot
be acquired (non-blocking mode).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from icore.exceptions import ConflictError

logger = logging.getLogger(__name__)


class BaseDistributedLock(ABC):
    """Abstract distributed lock interface."""

    @abstractmethod
    async def acquire(
        self,
        key: str,
        ttl: float = 30.0,
        blocking: bool = True,
    ) -> bool:
        """
        Acquire the lock for ``key``.

        Args:
            key:      Lock name (e.g. ``"graph:upsert:task-123"``).
            ttl:      Time-to-live in seconds (prevents permanent
                      deadlock if the holder crashes).
            blocking: If True, wait until the lock is acquired (up to
                      ``ttl``). If False, fail immediately when the
                      lock is held.

        Returns:
            True if the lock was acquired.

        Raises:
            ConflictError: In non-blocking mode, when the lock is held.
        """
        raise NotImplementedError

    @abstractmethod
    async def release(self, key: str) -> None:
        """Release the lock for ``key``."""
        raise NotImplementedError

    @asynccontextmanager
    async def lock(
        self,
        key: str,
        ttl: float = 30.0,
    ) -> AsyncIterator[None]:
        """
        Context-manager wrapper around acquire/release.

        Usage::

            async with ctx.get_lock().lock("graph:upsert:task-123"):
                await graphstore.upsert_nodes(nodes)

        Raises:
            ConflictError: If the lock cannot be acquired.
        """
        acquired = await self.acquire(key, ttl)
        if not acquired:
            raise ConflictError(f"Failed to acquire lock: {key}")
        try:
            yield
        finally:
            try:
                await self.release(key)
            except Exception as e:  # pragma: no cover - release errors
                logger.warning(
                    "Failed to release lock '%s': %s", key, e
                )


class MemoryLock(BaseDistributedLock):
    """
    Single-process asyncio distributed lock.

    Implements the same interface as ``RedisLock`` using
    ``asyncio.Lock`` plus an expiry timestamp so that TTL semantics
    are respected (a crashed holder's lock auto-expires).

    Suitable for development / single-node deployments where a real
    Redis is unavailable.
    """

    def __init__(self) -> None:
        # key -> (asyncio.Lock, expiry_timestamp)
        self._locks: dict[str, tuple[asyncio.Lock, float]] = {}
        self._meta_lock = asyncio.Lock()

    async def acquire(
        self,
        key: str,
        ttl: float = 30.0,
        blocking: bool = True,
    ) -> bool:
        async with self._meta_lock:
            entry = self._locks.get(key)
            now = time.monotonic()
            if entry is not None and entry[1] < now:
                # Stale lock (TTL expired); treat as released.
                self._locks.pop(key, None)
                entry = None

            if entry is None:
                lock = asyncio.Lock()
                self._locks[key] = (lock, now + ttl)
            else:
                lock = entry[0]

        if blocking:
            await lock.acquire()
            # Refresh expiry on successful acquire.
            async with self._meta_lock:
                self._locks[key] = (lock, time.monotonic() + ttl)
            return True

        if lock.locked():
            return False
        await lock.acquire()
        async with self._meta_lock:
            self._locks[key] = (lock, time.monotonic() + ttl)
        return True

    async def release(self, key: str) -> None:
        async with self._meta_lock:
            entry = self._locks.get(key)
        if entry is None:
            return
        lock = entry[0]
        if lock.locked():
            lock.release()
        # Do not pop the entry: the next acquire can reuse the same
        # asyncio.Lock (otherwise blocked waiters would hang forever).
        async with self._meta_lock:
            self._locks[key] = (lock, time.monotonic() + 30.0)


class RedisLock(BaseDistributedLock):
    """
    Redis-backed distributed lock.

    Uses ``SET key token NX PX ttl`` for atomic acquire and a Lua
    script for token-checked release (prevents releasing another
    holder's lock). The ``redis`` driver is imported lazily so this
    module is importable without ``redis`` installed.

    The Redis client must be supplied by the caller (typically
    built from ``settings.concurrency.redis_url``)::

        import redis.asyncio as aioredis
        client = aioredis.from_url(redis_url)
        lock = RedisLock(client)
    """

    #: Lua script for token-checked release.
    _RELEASE_SCRIPT = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._tokens: dict[str, str] = {}

    async def acquire(
        self,
        key: str,
        ttl: float = 30.0,
        blocking: bool = True,
    ) -> bool:
        token = secrets.token_hex(16)
        ttl_ms = int(ttl * 1000)
        # Poll cadence for blocking acquire.
        poll_interval = max(0.05, min(0.5, ttl / 10))
        deadline = time.monotonic() + ttl

        while True:
            ok = await self._client.set(key, token, nx=True, px=ttl_ms)
            if ok:
                self._tokens[key] = token
                return True
            if not blocking:
                return False
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll_interval)

    async def release(self, key: str) -> None:
        token = self._tokens.pop(key, None)
        if token is None:
            return
        try:
            await self._client.eval(
                self._RELEASE_SCRIPT, 1, key, token
            )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("Redis release failed for '%s': %s", key, e)


__all__ = [
    "BaseDistributedLock",
    "MemoryLock",
    "RedisLock",
]
