"""
icore.persistence - Workflow execution persistence (v0.6).

Records every workflow execution (and its per-node task executions) so
that runs can be inspected after the fact, resumed from a checkpoint
after a crash, and replayed for auditing.

Two backends are provided, mirroring the DLQ design:

    - ``InMemoryPersistenceBackend``: default; records kept in process
      memory. Suitable for tests and single-node deployments.
    - ``PostgresPersistenceBackend``: lazy-imports asyncpg, persists to
      the ``workflow_executions`` / ``task_executions`` tables. Falls
      back gracefully to an in-memory shadow when asyncpg is absent or
      the DB is unreachable — writes are still recorded in memory and
      reads are served from the shadow, so the persistence layer never
      breaks the request path.

``WorkflowPersistenceManager`` wraps a backend and exposes a small,
backend-agnostic high-level API (create / start / complete / fail /
checkpoint / resume / history).

Design notes:
    - All timestamps are Unix epochs (``time.time()``); no datetime.
    - JSON fields (params / checkpoint / result / metadata / input /
      output) are serialized with ``json.dumps`` and stored as JSONB.
    - asyncpg is imported lazily so this module imports cleanly without
      the driver installed.
    - No SQLAlchemy, no loguru — only stdlib ``logging``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from icore.exceptions import WorkflowPersistenceError, WorkflowResumeError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class ExecutionStatus(str, Enum):
    """Lifecycle status for a workflow / task execution.

    ``PAUSED`` is an intermediate state used for checkpoint-and-resume:
    a paused execution can be resumed, whereas a ``COMPLETED`` one
    cannot.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorkflowExecution:
    """
    A single workflow execution record.

    Attributes:
        id:            Unique execution ID (UUID string).
        task_id:       Correlates with the API layer's task_id.
        workflow_name: Registered workflow name.
        params:        Snapshot of invocation params.
        status:        Current :class:`ExecutionStatus`.
        started_at:    Unix epoch when execution entered RUNNING.
        finished_at:   Unix epoch when execution reached a terminal state.
        error:         Error message if status is FAILED.
        checkpoint:    Resume checkpoint. Structure::

                           {"completed_nodes": [...],
                            "current_wave": int,
                            "node_outputs": {...}}

        result:        Terminal output (for COMPLETED runs).
        created_at:    Unix epoch when the record was created.
        updated_at:    Unix epoch of the last write.
        metadata:      Free-form extensibility metadata.
    """

    id: str
    task_id: str
    workflow_name: str
    params: dict[str, Any] = field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    checkpoint: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly serialization (used by API + persistence)."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "workflow_name": self.workflow_name,
            "params": self.params,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "checkpoint": self.checkpoint,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowExecution":
        """Reconstruct from a dict (e.g. loaded from Postgres)."""
        return cls(
            id=data["id"],
            task_id=data["task_id"],
            workflow_name=data["workflow_name"],
            params=dict(data.get("params") or {}),
            status=ExecutionStatus(data.get("status", ExecutionStatus.PENDING.value)),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            error=data.get("error"),
            checkpoint=dict(data.get("checkpoint") or {}),
            result=data.get("result"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class TaskExecution:
    """
    A single task (DAG node) execution within a :class:`WorkflowExecution`.

    Attributes:
        id:           Unique task execution ID (UUID string).
        execution_id: Foreign key to ``WorkflowExecution.id``.
        node_id:      DAG node ID.
        task_name:    Registered task name.
        status:       Current :class:`ExecutionStatus`.
        input:        Node input snapshot.
        output:       Node output (None until completed).
        started_at:   Unix epoch when the node started running.
        finished_at:  Unix epoch when the node reached a terminal state.
        retry_count:  Number of retries for this node.
        error:        Error message if the node failed.
    """

    id: str
    execution_id: str
    node_id: str
    task_name: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    input: dict[str, Any] = field(default_factory=dict)
    output: Optional[dict[str, Any]] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    retry_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly serialization."""
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "task_name": self.task_name,
            "status": self.status.value,
            "input": self.input,
            "output": self.output,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "retry_count": self.retry_count,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskExecution":
        """Reconstruct from a dict."""
        return cls(
            id=data["id"],
            execution_id=data["execution_id"],
            node_id=data["node_id"],
            task_name=data["task_name"],
            status=ExecutionStatus(data.get("status", ExecutionStatus.PENDING.value)),
            input=dict(data.get("input") or {}),
            output=data.get("output"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            retry_count=data.get("retry_count", 0),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class BasePersistenceBackend(ABC):
    """Abstract persistence backend (in-memory / Postgres / Redis)."""

    @abstractmethod
    async def save_execution(self, exec: WorkflowExecution) -> None:
        """Insert or update (upsert) a workflow execution record."""
        raise NotImplementedError

    @abstractmethod
    async def get_execution(self, exec_id: str) -> Optional[WorkflowExecution]:
        """Fetch a single execution by ID, or None if not found."""
        raise NotImplementedError

    @abstractmethod
    async def update_execution_status(
        self,
        exec_id: str,
        status: ExecutionStatus,
        error: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        checkpoint: Optional[dict[str, Any]] = None,
    ) -> None:
        """Partial update of an execution's status / error / result / checkpoint."""
        raise NotImplementedError

    @abstractmethod
    async def list_executions(
        self,
        *,
        workflow_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WorkflowExecution]:
        """List executions, optionally filtered, with pagination."""
        raise NotImplementedError

    @abstractmethod
    async def delete_execution(self, exec_id: str) -> bool:
        """Delete an execution (and its task executions). Return whether it existed."""
        raise NotImplementedError

    @abstractmethod
    async def save_task_execution(self, task_exec: TaskExecution) -> None:
        """Insert or update (upsert) a task execution record."""
        raise NotImplementedError

    @abstractmethod
    async def get_task_executions(self, execution_id: str) -> list[TaskExecution]:
        """List all task executions belonging to a workflow execution."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release backend resources."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------

class InMemoryPersistenceBackend(BasePersistenceBackend):
    """In-process persistence backend (default; tests / single-node)."""

    def __init__(self) -> None:
        self._executions: dict[str, WorkflowExecution] = {}
        # task executions grouped by execution_id for O(1) lookup
        self._task_executions: dict[str, list[TaskExecution]] = {}
        self._lock = asyncio.Lock()

    async def save_execution(self, exec: WorkflowExecution) -> None:
        async with self._lock:
            self._executions[exec.id] = exec

    async def get_execution(self, exec_id: str) -> Optional[WorkflowExecution]:
        async with self._lock:
            return self._executions.get(exec_id)

    async def update_execution_status(
        self,
        exec_id: str,
        status: ExecutionStatus,
        error: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        checkpoint: Optional[dict[str, Any]] = None,
    ) -> None:
        async with self._lock:
            e = self._executions.get(exec_id)
            if e is None:
                return
            e.status = status
            e.updated_at = time.time()
            if error is not None:
                e.error = error
            if result is not None:
                e.result = result
            if checkpoint is not None:
                e.checkpoint = dict(checkpoint)

    async def list_executions(
        self,
        *,
        workflow_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WorkflowExecution]:
        async with self._lock:
            items = list(self._executions.values())
        result: list[WorkflowExecution] = []
        for e in items:
            if workflow_name is not None and e.workflow_name != workflow_name:
                continue
            if status is not None and e.status != status:
                continue
            if since is not None and e.created_at < since:
                continue
            if until is not None and e.created_at > until:
                continue
            result.append(e)
        # Stable order: oldest first.
        result.sort(key=lambda x: (x.created_at, x.id))
        if offset > 0:
            result = result[offset:]
        return result[:limit]

    async def delete_execution(self, exec_id: str) -> bool:
        async with self._lock:
            existed = exec_id in self._executions
            if existed:
                del self._executions[exec_id]
            # Cascade-delete child task executions.
            self._task_executions.pop(exec_id, None)
        return existed

    async def save_task_execution(self, task_exec: TaskExecution) -> None:
        async with self._lock:
            bucket = self._task_executions.setdefault(task_exec.execution_id, [])
            for i, t in enumerate(bucket):
                if t.id == task_exec.id:
                    bucket[i] = task_exec
                    break
            else:
                bucket.append(task_exec)

    async def get_task_executions(self, execution_id: str) -> list[TaskExecution]:
        async with self._lock:
            return list(self._task_executions.get(execution_id, []))

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------

def _json_loads(val: Any) -> Any:
    """Decode a JSONB value returned by asyncpg.

    asyncpg returns ``json`` / ``jsonb`` as ``str`` by default; if a
    codec is registered the value may already be parsed. Handle both.
    """
    if val is None:
        return None
    if isinstance(val, str):
        return json.loads(val) if val else None
    return val


class PostgresPersistenceBackend(BasePersistenceBackend):
    """
    PostgreSQL-backed persistence.

    Persists to ``workflow_executions`` / ``task_executions`` (auto-created
    on first use). The asyncpg driver is imported lazily so this module is
    importable without asyncpg installed. When the DB is unreachable every
    write falls back to an in-memory shadow (and is then best-effort
    mirrored to the DB), and every read is served from the shadow when the
    DB is unavailable — failures are logged but never propagate, matching
    the design intent that persistence must not break the request path.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS workflow_executions (
        id            TEXT PRIMARY KEY,
        task_id       TEXT NOT NULL,
        workflow_name TEXT NOT NULL,
        params        JSONB NOT NULL DEFAULT '{}'::jsonb,
        status        TEXT NOT NULL,
        started_at    DOUBLE PRECISION,
        finished_at   DOUBLE PRECISION,
        error         TEXT,
        checkpoint    JSONB NOT NULL DEFAULT '{}'::jsonb,
        result        JSONB,
        created_at    DOUBLE PRECISION NOT NULL,
        updated_at    DOUBLE PRECISION NOT NULL,
        metadata      JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    CREATE INDEX IF NOT EXISTS wf_exec_workflow_idx ON workflow_executions(workflow_name);
    CREATE INDEX IF NOT EXISTS wf_exec_status_idx ON workflow_executions(status);
    CREATE INDEX IF NOT EXISTS wf_exec_task_idx ON workflow_executions(task_id);
    CREATE INDEX IF NOT EXISTS wf_exec_started_idx ON workflow_executions(started_at);

    CREATE TABLE IF NOT EXISTS task_executions (
        id            TEXT PRIMARY KEY,
        execution_id  TEXT NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
        node_id       TEXT NOT NULL,
        task_name     TEXT NOT NULL,
        status        TEXT NOT NULL,
        input         JSONB NOT NULL DEFAULT '{}'::jsonb,
        output        JSONB,
        started_at    DOUBLE PRECISION,
        finished_at   DOUBLE PRECISION,
        retry_count   INTEGER NOT NULL DEFAULT 0,
        error         TEXT
    );
    CREATE INDEX IF NOT EXISTS task_exec_execution_idx ON task_executions(execution_id);
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None
        self._lock = asyncio.Lock()
        self._shadow = InMemoryPersistenceBackend()
        self._ddl_applied = False

    async def _get_pool(self) -> Any:
        """Lazily create the asyncpg pool. Returns None on any failure."""
        if self._pool is not None:
            return self._pool
        try:
            import asyncpg  # type: ignore
        except ImportError:
            logger.warning(
                "asyncpg not installed; PostgresPersistenceBackend falling back to memory"
            )
            return None
        try:
            self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=4)
        except Exception as e:  # pragma: no cover - network path
            logger.warning(
                "PostgresPersistenceBackend cannot connect (%s); falling back to memory", e
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
            logger.warning("PostgresPersistenceBackend DDL failed: %s", e)

    # ------------------------------------------------------------------
    # Workflow executions
    # ------------------------------------------------------------------

    async def save_execution(self, exec: WorkflowExecution) -> None:
        # Always shadow so reads work even if the DB is down.
        await self._shadow.save_execution(exec)
        pool = await self._get_pool()
        if pool is None:
            return
        await self._ensure_ddl(pool)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO workflow_executions
                        (id, task_id, workflow_name, params, status, started_at,
                         finished_at, error, checkpoint, result, created_at,
                         updated_at, metadata)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9::jsonb,
                            $10::jsonb, $11, $12, $13::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        task_id       = EXCLUDED.task_id,
                        workflow_name = EXCLUDED.workflow_name,
                        params        = EXCLUDED.params,
                        status        = EXCLUDED.status,
                        started_at    = EXCLUDED.started_at,
                        finished_at   = EXCLUDED.finished_at,
                        error         = EXCLUDED.error,
                        checkpoint    = EXCLUDED.checkpoint,
                        result        = EXCLUDED.result,
                        updated_at    = EXCLUDED.updated_at,
                        metadata      = EXCLUDED.metadata
                    """,
                    exec.id,
                    exec.task_id,
                    exec.workflow_name,
                    json.dumps(exec.params),
                    exec.status.value,
                    exec.started_at,
                    exec.finished_at,
                    exec.error,
                    json.dumps(exec.checkpoint),
                    json.dumps(exec.result) if exec.result is not None else None,
                    exec.created_at,
                    exec.updated_at,
                    json.dumps(exec.metadata),
                )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresPersistenceBackend save_execution failed: %s", e)

    async def get_execution(self, exec_id: str) -> Optional[WorkflowExecution]:
        # Try shadow first (always populated by save_execution).
        cached = await self._shadow.get_execution(exec_id)
        if cached is not None:
            return cached
        pool = await self._get_pool()
        if pool is None:
            return None
        await self._ensure_ddl(pool)
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM workflow_executions WHERE id = $1", exec_id
                )
            if row is None:
                return None
            return WorkflowExecution.from_dict({
                "id": row["id"],
                "task_id": row["task_id"],
                "workflow_name": row["workflow_name"],
                "params": _json_loads(row["params"]) or {},
                "status": row["status"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "error": row["error"],
                "checkpoint": _json_loads(row["checkpoint"]) or {},
                "result": _json_loads(row["result"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": _json_loads(row["metadata"]) or {},
            })
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresPersistenceBackend get_execution failed: %s", e)
            return None

    async def update_execution_status(
        self,
        exec_id: str,
        status: ExecutionStatus,
        error: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        checkpoint: Optional[dict[str, Any]] = None,
    ) -> None:
        await self._shadow.update_execution_status(
            exec_id, status, error=error, result=result, checkpoint=checkpoint
        )
        pool = await self._get_pool()
        if pool is None:
            return
        await self._ensure_ddl(pool)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE workflow_executions
                    SET status = $2,
                        error = COALESCE($3, error),
                        result = COALESCE($4::jsonb, result),
                        checkpoint = COALESCE($5::jsonb, checkpoint),
                        updated_at = $6
                    WHERE id = $1
                    """,
                    exec_id,
                    status.value,
                    error,
                    json.dumps(result) if result is not None else None,
                    json.dumps(checkpoint) if checkpoint is not None else None,
                    time.time(),
                )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresPersistenceBackend update_execution_status failed: %s", e)

    async def list_executions(
        self,
        *,
        workflow_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WorkflowExecution]:
        # Serve from the shadow (always populated on save), matching the
        # DLQ pattern — keeps behaviour simple and predictable for tests.
        return await self._shadow.list_executions(
            workflow_name=workflow_name,
            status=status,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )

    async def delete_execution(self, exec_id: str) -> bool:
        existed = await self._shadow.delete_execution(exec_id)
        pool = await self._get_pool()
        if pool is None:
            return existed
        await self._ensure_ddl(pool)
        try:
            async with pool.acquire() as conn:
                # task_executions cascade-deleted via FK.
                await conn.execute(
                    "DELETE FROM workflow_executions WHERE id = $1", exec_id
                )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresPersistenceBackend delete_execution failed: %s", e)
        return existed

    # ------------------------------------------------------------------
    # Task executions
    # ------------------------------------------------------------------

    async def save_task_execution(self, task_exec: TaskExecution) -> None:
        await self._shadow.save_task_execution(task_exec)
        pool = await self._get_pool()
        if pool is None:
            return
        await self._ensure_ddl(pool)
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO task_executions
                        (id, execution_id, node_id, task_name, status, input,
                         output, started_at, finished_at, retry_count, error)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $10, $11)
                    ON CONFLICT (id) DO UPDATE SET
                        node_id      = EXCLUDED.node_id,
                        task_name    = EXCLUDED.task_name,
                        status       = EXCLUDED.status,
                        input        = EXCLUDED.input,
                        output       = EXCLUDED.output,
                        started_at   = EXCLUDED.started_at,
                        finished_at  = EXCLUDED.finished_at,
                        retry_count  = EXCLUDED.retry_count,
                        error        = EXCLUDED.error
                    """,
                    task_exec.id,
                    task_exec.execution_id,
                    task_exec.node_id,
                    task_exec.task_name,
                    task_exec.status.value,
                    json.dumps(task_exec.input),
                    json.dumps(task_exec.output) if task_exec.output is not None else None,
                    task_exec.started_at,
                    task_exec.finished_at,
                    task_exec.retry_count,
                    task_exec.error,
                )
        except Exception as e:  # pragma: no cover - network path
            logger.warning("PostgresPersistenceBackend save_task_execution failed: %s", e)

    async def get_task_executions(self, execution_id: str) -> list[TaskExecution]:
        return await self._shadow.get_task_executions(execution_id)

    async def close(self) -> None:
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:  # pragma: no cover
                pass
            self._pool = None


# ---------------------------------------------------------------------------
# Manager façade
# ---------------------------------------------------------------------------

class WorkflowPersistenceManager:
    """
    Workflow persistence façade.

    Wraps a backend (in-memory by default, Postgres when configured) and
    exposes a backend-agnostic high-level API for the engine / API layer.

    Usage::

        mgr = WorkflowPersistenceManager()
        exec_obj = await mgr.create_execution("t-1", "rag_qa", {"q": "..."})
        await mgr.start_execution(exec_obj.id)
        ...
        await mgr.save_checkpoint(exec_obj.id, {"completed_nodes": ["retrieve"]})
        ...
        await mgr.complete_execution(exec_obj.id, {"answer": "..."})

    On crash recovery::

        exec_obj = await mgr.resume_execution(exec_id)
        # exec_obj.checkpoint contains the resume point.
    """

    def __init__(self, backend: Optional[BasePersistenceBackend] = None) -> None:
        self._backend: BasePersistenceBackend = (
            backend if backend is not None else InMemoryPersistenceBackend()
        )

    @property
    def backend(self) -> BasePersistenceBackend:
        """Expose the underlying backend (for admin / introspection)."""
        return self._backend

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def create_execution(
        self,
        task_id: str,
        workflow_name: str,
        params: dict[str, Any],
    ) -> WorkflowExecution:
        """Create and persist a new PENDING execution record."""
        exec_obj = WorkflowExecution(
            id=str(uuid.uuid4()),
            task_id=task_id,
            workflow_name=workflow_name,
            params=dict(params),
        )
        await self._backend.save_execution(exec_obj)
        logger.info(
            "Persistence: created execution %s (workflow=%s task=%s)",
            exec_obj.id, workflow_name, task_id,
        )
        return exec_obj

    async def start_execution(self, exec_id: str) -> None:
        """Mark an execution as RUNNING and record ``started_at``."""
        exec_obj = await self._backend.get_execution(exec_id)
        if exec_obj is None:
            raise WorkflowPersistenceError(f"Execution '{exec_id}' not found")
        exec_obj.status = ExecutionStatus.RUNNING
        exec_obj.started_at = time.time()
        exec_obj.updated_at = time.time()
        await self._backend.save_execution(exec_obj)

    async def complete_execution(
        self, exec_id: str, result: dict[str, Any]
    ) -> None:
        """Mark an execution as COMPLETED and record ``finished_at`` + result."""
        exec_obj = await self._backend.get_execution(exec_id)
        if exec_obj is None:
            raise WorkflowPersistenceError(f"Execution '{exec_id}' not found")
        exec_obj.status = ExecutionStatus.COMPLETED
        exec_obj.finished_at = time.time()
        exec_obj.updated_at = time.time()
        exec_obj.result = dict(result)
        await self._backend.save_execution(exec_obj)

    async def fail_execution(self, exec_id: str, error: str) -> None:
        """Mark an execution as FAILED and record ``finished_at`` + error."""
        exec_obj = await self._backend.get_execution(exec_id)
        if exec_obj is None:
            raise WorkflowPersistenceError(f"Execution '{exec_id}' not found")
        exec_obj.status = ExecutionStatus.FAILED
        exec_obj.finished_at = time.time()
        exec_obj.updated_at = time.time()
        exec_obj.error = error
        await self._backend.save_execution(exec_obj)

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    async def save_checkpoint(
        self, exec_id: str, checkpoint: dict[str, Any]
    ) -> None:
        """Persist a resume checkpoint (completed_nodes / current_wave / node_outputs)."""
        exec_obj = await self._backend.get_execution(exec_id)
        if exec_obj is None:
            raise WorkflowPersistenceError(f"Execution '{exec_id}' not found")
        await self._backend.update_execution_status(
            exec_id, exec_obj.status, checkpoint=dict(checkpoint)
        )
        # Keep the in-memory reference in sync for backends that return
        # the live object (e.g. the shadow inside Postgres backend).
        exec_obj.checkpoint = dict(checkpoint)

    async def get_checkpoint(self, exec_id: str) -> Optional[dict[str, Any]]:
        """Return the checkpoint dict, or None if the execution does not exist."""
        exec_obj = await self._backend.get_execution(exec_id)
        if exec_obj is None:
            return None
        return dict(exec_obj.checkpoint)

    async def resume_execution(self, exec_id: str) -> WorkflowExecution:
        """
        Resume an execution from its checkpoint.

        1. Load the execution.
        2. Validate the status is FAILED or PAUSED.
        3. Reset to PENDING so the executor can reschedule.
        4. Return the execution (with its checkpoint).

        Raises:
            WorkflowResumeError: If the execution does not exist or is
                not in a resumable state.
        """
        exec_obj = await self._backend.get_execution(exec_id)
        if exec_obj is None:
            raise WorkflowResumeError(f"Execution '{exec_id}' not found")
        if exec_obj.status not in (ExecutionStatus.FAILED, ExecutionStatus.PAUSED):
            raise WorkflowResumeError(
                f"Cannot resume execution in status '{exec_obj.status.value}'"
            )
        exec_obj.status = ExecutionStatus.PENDING
        exec_obj.error = None
        exec_obj.updated_at = time.time()
        await self._backend.save_execution(exec_obj)
        logger.info("Persistence: resumed execution %s", exec_id)
        return exec_obj

    # ------------------------------------------------------------------
    # Task executions
    # ------------------------------------------------------------------

    async def record_task_execution(
        self,
        execution_id: str,
        node_id: str,
        task_name: str,
        status: ExecutionStatus,
        input: Optional[dict[str, Any]] = None,
        output: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        retry_count: int = 0,
    ) -> TaskExecution:
        """Create and persist a task (node) execution record."""
        task_exec = TaskExecution(
            id=str(uuid.uuid4()),
            execution_id=execution_id,
            node_id=node_id,
            task_name=task_name,
            status=status,
            input=dict(input or {}),
            output=dict(output) if output is not None else None,
            error=error,
            retry_count=retry_count,
            started_at=time.time() if status != ExecutionStatus.PENDING else None,
            finished_at=time.time() if status in (
                ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED
            ) else None,
        )
        await self._backend.save_task_execution(task_exec)
        return task_exec

    # ------------------------------------------------------------------
    # Query / history
    # ------------------------------------------------------------------

    async def get_history(
        self,
        *,
        workflow_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WorkflowExecution]:
        """List executions matching the given filters, paginated."""
        return await self._backend.list_executions(
            workflow_name=workflow_name,
            status=status,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )

    async def get_execution(self, exec_id: str) -> Optional[WorkflowExecution]:
        """Fetch a single execution by ID."""
        return await self._backend.get_execution(exec_id)

    async def get_task_executions(self, exec_id: str) -> list[TaskExecution]:
        """List all task executions belonging to a workflow execution."""
        return await self._backend.get_task_executions(exec_id)

    async def close(self) -> None:
        """Release backend resources."""
        await self._backend.close()


__all__ = [
    "ExecutionStatus",
    "WorkflowExecution",
    "TaskExecution",
    "BasePersistenceBackend",
    "InMemoryPersistenceBackend",
    "PostgresPersistenceBackend",
    "WorkflowPersistenceManager",
]
