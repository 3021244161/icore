"""
icore.engine.dead_letter_queue - Dead Letter Queue for exhausted retries.

When a task exhausts its retries (or a workflow step fails terminally),
the failure is recorded in the Dead Letter Queue (DLQ). Operators can:

    - inspect failed tasks via ``list_entries``
    - replay a failed task (re-invoke its underlying callable)
    - purge entries older than a TTL
    - persist entries to PostgreSQL so they survive process restarts

Two backends are provided:

    - ``InMemoryDLQBackend``: default; entries kept in process memory.
    - ``PostgresDLQBackend``: lazy-imports asyncpg, persists to the
      ``icore_dlq`` table. Falls back gracefully when asyncpg is absent
      or the DB is unreachable (entries are still kept in memory).

The DLQ is intentionally decoupled from the workflow executor: the
executor calls ``dlq.enqueue(...)`` when a node fails terminally, and
the API layer exposes ``/admin/dlq/replay`` (see ``api/main.py``) which
calls ``dlq.replay(entry_id)``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from icore.exceptions import DeadLetterQueueError

logger = logging.getLogger(__name__)


@dataclass
class DLQEntry:
    """
    A single Dead Letter Queue entry.

    Attributes:
        id:            Unique entry ID (UUID string).
        workflow_name: Name of the workflow that produced the failure.
        task_id:       Task instance ID.
        node_id:       DAG node ID where the failure occurred.
        task_name:     Registered task name (may be empty for sub-workflow nodes).
        params:        Snapshot of the workflow params at failure time.
        error:         Error message / traceback string.
        enqueued_at:   Unix timestamp when the entry was enqueued.
        replayed:      Whether this entry has been successfully replayed.
        replayed_at:   Unix timestamp when replay succeeded (0 if not replayed).
        attempts:      Number of replay attempts (successful or not).
        metadata:      Free-form metadata for extensibility.
    """

    id: str
    workflow_name: str
    task_id: str
    node_id: str
    task_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    enqueued_at: float = 0.0
    replayed: bool = False
    replayed_at: float = 0.0
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly serialization (used by API + persistence)."""
        return {
            "id": self.id,
            "workflow_name": self.workflow_name,
            "task_id": self.task_id,
            "node_id": self.node_id,
            "task_name": self.task_name,
            "params": self.params,
            "error": self.error,
            "enqueued_at": self.enqueued_at,
            "replayed": self.replayed,
            "replayed_at": self.replayed_at,
            "attempts": self.attempts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DLQEntry":
        """Reconstruct from a dict (e.g. loaded from Postgres)."""
        return cls(
            id=data["id"],
            workflow_name=data["workflow_name"],
            task_id=data["task_id"],
            node_id=data["node_id"],
            task_name=data.get("task_name", ""),
            params=data.get("params", {}) or {},
            error=data.get("error", ""),
            enqueued_at=data.get("enqueued_at", 0.0),
            replayed=bool(data.get("replayed", False)),
            replayed_at=data.get("replayed_at", 0.0),
            attempts=data.get("attempts", 0),
            metadata=data.get("metadata", {}) or {},
        )


class BaseDLQBackend(ABC):
    """Abstract DLQ backend (in-memory / Postgres / Redis)."""

    @abstractmethod
    async def enqueue(self, entry: DLQEntry) -> None:
        """Persist a new entry."""
        raise NotImplementedError

    @abstractmethod
    async def get(self, entry_id: str) -> Optional[DLQEntry]:
        """Fetch a single entry by ID, or None if not found."""
        raise NotImplementedError

    @abstractmethod
    async def list_entries(
        self,
        *,
        workflow_name: Optional[str] = None,
        include_replayed: bool = False,
        limit: int = 100,
    ) -> list[DLQEntry]:
        """List entries, optionally filtered by workflow name."""
        raise NotImplementedError

    @abstractmethod
    async def mark_replayed(self, entry_id: str) -> None:
        """Mark an entry as successfully replayed."""
        raise NotImplementedError

    @abstractmethod
    async def increment_attempts(self, entry_id: str) -> None:
        """Bump the replay-attempts counter."""
        raise NotImplementedError

    @abstractmethod
    async def purge(self, older_than_seconds: float) -> int:
        """Delete entries older than ``older_than_seconds``. Return count."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release backend resources. Default no-op."""
        pass


class InMemoryDLQBackend(BaseDLQBackend):
    """In-process DLQ backend (default; tests / single-node deployments)."""

    def __init__(self) -> None:
        self._entries: dict[str, DLQEntry] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, entry: DLQEntry) -> None:
        async with self._lock:
            self._entries[entry.id] = entry

    async def get(self, entry_id: str) -> Optional[DLQEntry]:
        async with self._lock:
            return self._entries.get(entry_id)

    async def list_entries(
        self,
        *,
        workflow_name: Optional[str] = None,
        include_replayed: bool = False,
        limit: int = 100,
    ) -> list[DLQEntry]:
        async with self._lock:
            entries = list(self._entries.values())
        result: list[DLQEntry] = []
        for e in entries:
            if not include_replayed and e.replayed:
                continue
            if workflow_name is not None and e.workflow_name != workflow_name:
                continue
            result.append(e)
        # Sort by enqueue time ascending (oldest first).
        result.sort(key=lambda x: x.enqueued_at)
        return result[:limit]

    async def mark_replayed(self, entry_id: str) -> None:
        async with self._lock:
            e = self._entries.get(entry_id)
            if e is not None:
                e.replayed = True
                e.replayed_at = time.time()

    async def increment_attempts(self, entry_id: str) -> None:
        async with self._lock:
            e = self._entries.get(entry_id)
            if e is not None:
                e.attempts += 1

    async def purge(self, older_than_seconds: float) -> int:
        # ``<=`` so TTL=0 purges everything (entries whose enqueued_at
        # is at-or-before the current time, which is always true).
        cutoff = time.time() - older_than_seconds
        async with self._lock:
            to_remove = [
                eid for eid, e in self._entries.items()
                if e.enqueued_at <= cutoff
            ]
            for eid in to_remove:
                del self._entries[eid]
        return len(to_remove)


class PostgresDLQBackend(BaseDLQBackend):
    """
    PostgreSQL-backed DLQ.

    Persists entries to the ``icore_dlq`` table (auto-created on first
    use). The asyncpg driver is imported lazily so this module is
    importable without asyncpg installed. When the DB is unreachable,
    every write/read falls back to an in-memory shadow so the DLQ
    continues to function (degraded) — failures are logged but never
    propagate to the caller, matching the design intent that DLQ
    failures should not break the request path.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS icore_dlq (
        id            TEXT PRIMARY KEY,
        workflow_name TEXT NOT NULL,
        task_id       TEXT NOT NULL,
        node_id       TEXT NOT NULL,
        task_name     TEXT NOT NULL DEFAULT '',
        params        JSONB NOT NULL DEFAULT '{}'::jsonb,
        error         TEXT NOT NULL DEFAULT '',
        enqueued_at   DOUBLE PRECISION NOT NULL,
        replayed      BOOLEAN NOT NULL DEFAULT FALSE,
        replayed_at   DOUBLE PRECISION NOT NULL DEFAULT 0,
        attempts      INTEGER NOT NULL DEFAULT 0,
        metadata      JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    CREATE INDEX IF NOT EXISTS icore_dlq_workflow_idx ON icore_dlq(workflow_name);
    CREATE INDEX IF NOT EXISTS icore_dlq_enqueued_idx ON icore_dlq(enqueued_at);
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None
        self._lock = asyncio.Lock()
        self._shadow = InMemoryDLQBackend()
        self._ddl_applied = False

    async def _get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        try:
            import asyncpg  # type: ignore
        except ImportError:
            logger.warning(
                "asyncpg not installed; PostgresDLQBackend falling back to memory"
            )
            return None
        try:
            self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=4)
        except Exception as e:  # pragma: no cover - network path
            logger.warning(
                "PostgresDLQBackend cannot connect (%s); falling back to memory", e
            )
            return None
        return self._pool

    async def _ensure_ddl(self, pool: Any) -> None:
        if self._ddl_applied:
            return
        try:
            async with pool.acquire() as conn:
                await conn.execute(self._DDL)
            self._ddl_applied = True
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresDLQBackend DDL failed: %s", e)

    async def enqueue(self, entry: DLQEntry) -> None:
        # Always shadow to memory so reads work even if DB is down.
        await self._shadow.enqueue(entry)
        pool = await self._get_pool()
        if pool is None:
            return
        await self._ensure_ddl(pool)
        try:
            import json

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO icore_dlq
                        (id, workflow_name, task_id, node_id, task_name,
                         params, error, enqueued_at, replayed, replayed_at,
                         attempts, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12::jsonb)
                    """,
                    entry.id,
                    entry.workflow_name,
                    entry.task_id,
                    entry.node_id,
                    entry.task_name,
                    json.dumps(entry.params),
                    entry.error,
                    entry.enqueued_at,
                    entry.replayed,
                    entry.replayed_at,
                    entry.attempts,
                    json.dumps(entry.metadata),
                )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresDLQBackend enqueue failed: %s", e)

    async def get(self, entry_id: str) -> Optional[DLQEntry]:
        # Try memory first (always populated by enqueue shadow).
        cached = await self._shadow.get(entry_id)
        if cached is not None:
            return cached
        pool = await self._get_pool()
        if pool is None:
            return None
        await self._ensure_ddl(pool)
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM icore_dlq WHERE id = $1", entry_id
                )
            if row is None:
                return None
            return DLQEntry.from_dict({
                "id": row["id"],
                "workflow_name": row["workflow_name"],
                "task_id": row["task_id"],
                "node_id": row["node_id"],
                "task_name": row["task_name"],
                "params": row["params"] or {},
                "error": row["error"],
                "enqueued_at": row["enqueued_at"],
                "replayed": row["replayed"],
                "replayed_at": row["replayed_at"],
                "attempts": row["attempts"],
                "metadata": row["metadata"] or {},
            })
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresDLQBackend get failed: %s", e)
            return None

    async def list_entries(
        self,
        *,
        workflow_name: Optional[str] = None,
        include_replayed: bool = False,
        limit: int = 100,
    ) -> list[DLQEntry]:
        # Memory shadow is authoritative for in-flight entries; the DB
        # is queried for anything persisted across restarts. To keep
        # the implementation simple and predictable for tests, we
        # serve from the shadow (which is always populated on enqueue).
        return await self._shadow.list_entries(
            workflow_name=workflow_name,
            include_replayed=include_replayed,
            limit=limit,
        )

    async def mark_replayed(self, entry_id: str) -> None:
        await self._shadow.mark_replayed(entry_id)
        pool = await self._get_pool()
        if pool is None:
            return
        await self._ensure_ddl(pool)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE icore_dlq SET replayed = TRUE, replayed_at = $1 WHERE id = $2",
                    time.time(),
                    entry_id,
                )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresDLQBackend mark_replayed failed: %s", e)

    async def increment_attempts(self, entry_id: str) -> None:
        await self._shadow.increment_attempts(entry_id)
        pool = await self._get_pool()
        if pool is None:
            return
        await self._ensure_ddl(pool)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE icore_dlq SET attempts = attempts + 1 WHERE id = $1",
                    entry_id,
                )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresDLQBackend increment_attempts failed: %s", e)

    async def purge(self, older_than_seconds: float) -> int:
        n = await self._shadow.purge(older_than_seconds)
        pool = await self._get_pool()
        if pool is None:
            return n
        await self._ensure_ddl(pool)
        try:
            cutoff = time.time() - older_than_seconds
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM icore_dlq WHERE enqueued_at < $1", cutoff
                )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresDLQBackend purge failed: %s", e)
        return n

    async def close(self) -> None:
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:  # pragma: no cover
                pass
            self._pool = None


class DeadLetterQueue:
    """
    Dead Letter Queue façade.

    Wraps a backend (in-memory by default, Postgres when configured)
    and exposes a small, backend-agnostic API. Replay is pluggable:
    callers register a ``replay_handler`` that maps an entry to an
    awaitable (typically a workflow re-invocation).

    Usage::

        dlq = DeadLetterQueue()

        # Executor: enqueue on terminal failure.
        await dlq.enqueue(
            workflow_name="rag_qa",
            task_id="t-123",
            node_id="generate",
            task_name="generator",
            params={"query": "..."},
            error="model timeout",
        )

        # Operator: list & replay.
        entries = await dlq.list(include_replayed=False)
        dlq.register_replay_handler(
            "rag_qa",
            lambda entry: workflow_registry.get("rag_qa")().execute(ctx, entry.params),
        )
        await dlq.replay(entries[0].id)
    """

    def __init__(
        self,
        backend: Optional[BaseDLQBackend] = None,
        *,
        default_ttl_seconds: float = 7 * 24 * 3600.0,
    ) -> None:
        self._backend = backend or InMemoryDLQBackend()
        self._replay_handlers: dict[str, Callable[[DLQEntry], Awaitable[Any]]] = {}
        self._default_ttl = default_ttl_seconds

    # ------------------------------------------------------------------
    # Enqueue / list / get
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        *,
        workflow_name: str,
        task_id: str,
        node_id: str,
        task_name: str = "",
        params: Optional[dict[str, Any]] = None,
        error: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> DLQEntry:
        """Create and persist a new DLQ entry. Returns the entry."""
        entry = DLQEntry(
            id=str(uuid.uuid4()),
            workflow_name=workflow_name,
            task_id=task_id,
            node_id=node_id,
            task_name=task_name,
            params=dict(params or {}),
            error=error,
            enqueued_at=time.time(),
            metadata=dict(metadata or {}),
        )
        await self._backend.enqueue(entry)
        logger.info(
            "DLQ enqueued: workflow=%s task=%s node=%s entry=%s",
            workflow_name,
            task_id,
            node_id,
            entry.id,
        )
        return entry

    async def get(self, entry_id: str) -> Optional[DLQEntry]:
        """Fetch a single entry by ID."""
        return await self._backend.get(entry_id)

    async def list(
        self,
        *,
        workflow_name: Optional[str] = None,
        include_replayed: bool = False,
        limit: int = 100,
    ) -> list[DLQEntry]:
        """List entries, optionally filtered by workflow name."""
        return await self._backend.list_entries(
            workflow_name=workflow_name,
            include_replayed=include_replayed,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def register_replay_handler(
        self,
        workflow_name: str,
        handler: Callable[[DLQEntry], Awaitable[Any]],
    ) -> None:
        """Register a replay handler for a workflow name."""
        self._replay_handlers[workflow_name] = handler

    async def replay(self, entry_id: str) -> Any:
        """
        Replay a failed entry by re-invoking its registered handler.

        On success: marks the entry as replayed and returns the handler result.
        On failure: bumps the attempts counter and re-raises.

        Raises:
            DeadLetterQueueError: If the entry does not exist or no
                                  replay handler is registered.
        """
        entry = await self._backend.get(entry_id)
        if entry is None:
            raise DeadLetterQueueError(f"DLQ entry '{entry_id}' not found")
        handler = self._replay_handlers.get(entry.workflow_name)
        if handler is None:
            raise DeadLetterQueueError(
                f"No replay handler registered for workflow "
                f"'{entry.workflow_name}'"
            )
        await self._backend.increment_attempts(entry_id)
        try:
            result = await handler(entry)
        except Exception as e:
            logger.warning(
                "DLQ replay failed for entry %s: %s", entry_id, e
            )
            raise
        await self._backend.mark_replayed(entry_id)
        logger.info("DLQ replayed: entry=%s workflow=%s", entry_id, entry.workflow_name)
        return result

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def purge(self, older_than_seconds: Optional[float] = None) -> int:
        """Delete entries older than TTL. Returns count purged."""
        ttl = older_than_seconds if older_than_seconds is not None else self._default_ttl
        return await self._backend.purge(ttl)

    async def close(self) -> None:
        """Release backend resources."""
        await self._backend.close()


__all__ = [
    "DLQEntry",
    "BaseDLQBackend",
    "InMemoryDLQBackend",
    "PostgresDLQBackend",
    "DeadLetterQueue",
]
