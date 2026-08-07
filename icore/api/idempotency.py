"""
icore.api.idempotency - In-memory idempotency cache.

Stores the JSON-serializable response of completed /invoke requests
keyed by caller-supplied ``idempotency_key``. Repeated requests with
the same key return the cached response without re-executing the
workflow.

Two backends are supported:

    - ``memory`` (default): asyncio.Lock-guarded dict with TTL eviction.
    - ``redis`` (planned):  Redis with TTL semantics.

Cache entries auto-expire after ``ttl`` seconds (default 86400 = 24h,
matching the design doc).
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Default TTL = 24 hours (matches design doc §7.3.3).
DEFAULT_TTL = 86400


class BaseIdempotencyCache(ABC):
    """Abstract idempotency cache."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Return cached response, or None if missing/expired."""
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
        """Cache ``value`` under ``key`` with ``ttl`` seconds."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove ``key``. Return True if removed."""
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> int:
        """Remove all entries. Return count removed."""
        raise NotImplementedError


class MemoryIdempotencyCache(BaseIdempotencyCache):
    """
    Single-process idempotency cache with TTL eviction.

    Entries are stored as ``(expiry_ts, value)`` tuples. Lazy
    expiration on read; eager expiration on ``clear()``.
    """

    def __init__(self, ttl: int = DEFAULT_TTL) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expiry, value = entry
            if expiry < time.time():
                self._store.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
        async with self._lock:
            self._store[key] = (time.time() + ttl, value)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._store:
                self._store.pop(key, None)
                return True
            return False

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    async def evict_expired(self) -> int:
        """Eagerly remove all expired entries. Returns count removed."""
        now = time.time()
        async with self._lock:
            to_remove = [
                k for k, (exp, _) in self._store.items() if exp < now
            ]
            for k in to_remove:
                self._store.pop(k, None)
        return len(to_remove)


def create_idempotency_cache(
    backend: str = "memory",
    redis_url: Optional[str] = None,
    ttl: int = DEFAULT_TTL,
) -> BaseIdempotencyCache:
    """
    Factory used by ``bootstrap`` to build the cache.

    For ``redis``, a real Redis-backed implementation would go here;
    for now we fall back to memory to keep the test suite offline.
    """
    if backend == "redis" and redis_url:
        logger.warning(
            "Redis idempotency cache not yet implemented; "
            "falling back to memory backend."
        )
    return MemoryIdempotencyCache(ttl=ttl)


__all__ = [
    "DEFAULT_TTL",
    "BaseIdempotencyCache",
    "MemoryIdempotencyCache",
    "create_idempotency_cache",
]
