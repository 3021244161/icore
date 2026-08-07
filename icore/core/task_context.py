"""
icore.core.task_context - Dependency injection container for task execution.

TaskContext is created by the workflow engine before executing a task and
passed to the task's prepare(), execute(), and cleanup() methods. It carries
all runtime context needed by the task:

    - Task identification (task_id, workflow_id)
    - Model selection (model_id - None means auto-route)
    - Callback configuration (callback_url)
    - Streaming control (stream flag)
    - Arbitrary metadata (trace_id, user_id, etc.)

It also provides factory methods to obtain runtime dependencies:

    - get_model_adapter():    Returns the LLM model adapter (from ModelManager)
    - get_db(name):           Returns a database connector (from DBManager)
    - get_vectorstore():      Returns the BaseVectorStore (v0.5)
    - get_graphstore():       Returns the BaseGraphStore (v0.5)
    - get_media_processor():  Returns the MediaProcessorRegistry (v0.5)
    - get_lock():             Returns the BaseDistributedLock (v0.5)
    - get_circuit_breaker():  Returns the CircuitBreakerRegistry (v0.5)

IMPORTANT: The actual ModelManager and DBManager are implemented in later
stories (US-003, US-005). To avoid circular dependencies, TaskContext uses
duck typing at runtime: it stores references to managers via private
attributes and calls their methods by convention. Type annotations use
TYPE_CHECKING imports for IDE support without runtime import cost.

Injection convention (AGENTS.md §4.1):
    Managers are injected via official setter methods (``set_*``). The
    setters use ``object.__setattr__`` because TaskContext is a Pydantic
    frozen-ish model — this is the *only* place that pattern is allowed.
    Tasks must NEVER call ``object.__setattr__`` on the context; they
    use the ``get_*`` accessors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    # These imports are only for type checking; actual implementations
    # come from US-003 (db.manager.DBManager) and US-005 (models.manager.ModelManager).
    # We use TYPE_CHECKING to avoid circular imports at runtime.
    from icore.db.manager import DBManager  # noqa: F401
    from icore.engine.circuit_breaker import CircuitBreakerRegistry  # noqa: F401
    from icore.engine.lock import BaseDistributedLock  # noqa: F401
    from icore.graphstore import BaseGraphStore  # noqa: F401
    from icore.media import MediaProcessorRegistry  # noqa: F401
    from icore.models.manager import ModelManager  # noqa: F401
    from icore.vectorstore import BaseVectorStore  # noqa: F401


class TaskContext(BaseModel):
    """
    Dependency injection container passed to every task execution.

    The engine creates a TaskContext for each task instance, populating it
    with the request parameters (task_id, model_id, callback_url, etc.)
    and injecting references to the ModelManager and DBManager.

    Tasks should NOT instantiate dependencies directly. Instead, they use
    the factory methods get_model_adapter() and get_db() to obtain what
    they need. This makes tasks testable (inject mock managers) and
    decoupled from the concrete infrastructure.

    Attributes:
        task_id:      Unique identifier for this task instance.
        workflow_id:  Identifier for the parent workflow instance.
        model_id:     Explicit model ID to use, or None for auto-routing.
        callback_url: URL to POST results to upon completion (None = no callback).
        stream:       Whether to stream results back (SSE) vs return all at once.
        metadata:     Free-form metadata dict for tracing, user info, etc.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    # --- Core context fields ---
    task_id: str = Field(description="Unique task instance ID")
    workflow_id: str = Field(default="", description="Parent workflow instance ID")
    model_id: Optional[str] = Field(
        default=None,
        description="Explicit model ID (None = auto-route via ModelRouter)",
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Callback URL for async result delivery",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream results (SSE) or return at once",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata (trace_id, user_id, etc.)",
    )

    # --- Injected manager references (not serialized) ---
    # These are set by the engine before task execution. They are excluded
    # from serialization to avoid leaking internal infrastructure into API
    # responses. We use private attributes with leading underscore.
    _model_manager: Any = None  # type: ignore[assignment]
    _db_manager: Any = None  # type: ignore[assignment]
    # v0.5 new injectable components.
    _vectorstore: Any = None  # type: ignore[assignment]
    _graphstore: Any = None  # type: ignore[assignment]
    _media_processor: Any = None  # type: ignore[assignment]
    _lock: Any = None  # type: ignore[assignment]
    _circuit_breaker: Any = None  # type: ignore[assignment]
    _objectstore: Any = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Manager injection methods (called by the workflow engine)
    # ------------------------------------------------------------------

    def set_model_manager(self, manager: Any) -> None:
        """Inject the ModelManager instance (called by engine)."""
        object.__setattr__(self, "_model_manager", manager)

    def set_db_manager(self, manager: Any) -> None:
        """Inject the DBManager instance (called by engine)."""
        object.__setattr__(self, "_db_manager", manager)

    def set_vectorstore(self, vectorstore: Any) -> None:
        """Inject the ``BaseVectorStore`` instance (v0.5, called by engine)."""
        object.__setattr__(self, "_vectorstore", vectorstore)

    def set_graphstore(self, graphstore: Any) -> None:
        """Inject the ``BaseGraphStore`` instance (v0.5, called by engine)."""
        object.__setattr__(self, "_graphstore", graphstore)

    def set_media_processor(self, registry: Any) -> None:
        """Inject the ``MediaProcessorRegistry`` instance (v0.5, called by engine)."""
        object.__setattr__(self, "_media_processor", registry)

    def set_lock(self, lock: Any) -> None:
        """Inject the ``BaseDistributedLock`` instance (v0.5, called by engine)."""
        object.__setattr__(self, "_lock", lock)

    def set_circuit_breaker(self, registry: Any) -> None:
        """Inject the ``CircuitBreakerRegistry`` instance (v0.5, called by engine)."""
        object.__setattr__(self, "_circuit_breaker", registry)

    def set_objectstore(self, store: Any) -> None:
        """Inject the ``BaseObjectStore`` instance (called by engine)."""
        object.__setattr__(self, "_objectstore", store)

    # ------------------------------------------------------------------
    # Child context creation (v0.6)
    # ------------------------------------------------------------------

    def create_child(self) -> "TaskContext":
        """创建一个继承父上下文所有已注入资源的子上下文 (v0.6)。

        子上下文是一个全新的 ``TaskContext`` 实例，复制父上下文的核心
        标识字段（task_id / workflow_id / model_id / callback_url /
        stream / metadata），并通过官方 setter 注入父上下文已注入的全部
        运行时资源（model_manager / db_manager / vectorstore /
        graphstore / objectstore / media_processor / lock /
        circuit_breaker）。子上下文可独立设置属性，互不影响父上下文。

        Returns:
            一个继承父上下文资源的新 ``TaskContext`` 实例。
        """
        child = TaskContext(
            task_id=self.task_id,
            workflow_id=self.workflow_id,
            model_id=self.model_id,
            callback_url=self.callback_url,
            stream=self.stream,
            metadata=dict(self.metadata),
        )
        # 通过官方 setter 注入（不用 object.__setattr__，AGENTS.md §4.1）。
        if self._model_manager is not None:
            child.set_model_manager(self._model_manager)
        if self._db_manager is not None:
            child.set_db_manager(self._db_manager)
        if self.has_vectorstore():
            child.set_vectorstore(self.get_vectorstore())
        if self.has_graphstore():
            child.set_graphstore(self.get_graphstore())
        if self.has_objectstore():
            child.set_objectstore(self.get_objectstore())
        if self.has_media_processor():
            child.set_media_processor(self.get_media_processor())
        if self.has_lock():
            child.set_lock(self.get_lock())
        if self.has_circuit_breaker():
            child.set_circuit_breaker(self.get_circuit_breaker())
        return child

    # ------------------------------------------------------------------
    # Dependency factory methods (called by tasks)
    # ------------------------------------------------------------------

    def get_model_adapter(self) -> Any:
        """
        Get the LLM model adapter for this task.

        If model_id is set, returns that specific model's adapter.
        If model_id is None, delegates to ModelRouter for auto-routing.

        Returns:
            A model adapter instance (duck-typed: has chat(), stream_chat(), embed()).

        Raises:
            RuntimeError: If no ModelManager has been injected.
        """
        if self._model_manager is None:
            raise RuntimeError(
                "No ModelManager injected into TaskContext. "
                "The workflow engine must call ctx.set_model_manager() "
                "before task execution."
            )
        return self._model_manager.get_adapter(self.model_id)

    def get_db(self, name: str) -> Any:
        """
        Get a database access handle bound to a named connection.

        The returned handle is pool-backed: each ``query()`` / ``execute()``
        call acquires a connection from the pool, runs the statement, and
        releases it automatically (via ``DBManager.connection_ctx()``).
        Callers therefore never need to manage acquire/release themselves,
        which matches the design intent that "the connection pool
        auto-manages reuse and recycling".

        For multi-statement transactions on the same connection, use
        ``handle.connection_ctx()`` as an async context manager::

            db = ctx.get_db("main_db")
            async with db.connection_ctx() as conn:
                rows = await conn.query("SELECT ...")

        Args:
            name: The registered name of the database connection
                  (e.g. "main_db", "analytics").

        Returns:
            A ``_BoundDB`` handle (duck-typed: has async query(),
            execute(), fetch_one(), fetch_all(), connection_ctx()).

        Raises:
            RuntimeError: If no DBManager has been injected.
        """
        if self._db_manager is None:
            raise RuntimeError(
                "No DBManager injected into TaskContext. "
                "The workflow engine must call ctx.set_db_manager() "
                "before task execution."
            )
        return _BoundDB(self._db_manager, name)

    # ------------------------------------------------------------------
    # v0.5: New component accessors
    # ------------------------------------------------------------------

    def get_vectorstore(self) -> Any:
        """
        Return the injected ``BaseVectorStore`` instance.

        Tasks call this to insert / search / delete vector documents::

            vs = ctx.get_vectorstore()
            hits = await vs.search("doc_embeddings", query_vec, top_k=5)

        Raises:
            RuntimeError: If no vectorstore has been injected. Callers
                          that don't need vector search should guard
                          with ``has_vectorstore()``.
        """
        if self._vectorstore is None:
            raise RuntimeError(
                "No VectorStore injected into TaskContext. "
                "The workflow engine must call ctx.set_vectorstore() "
                "before task execution, or this workflow does not "
                "require vector search."
            )
        return self._vectorstore

    def has_vectorstore(self) -> bool:
        """Whether a vectorstore has been injected."""
        return self._vectorstore is not None

    def get_graphstore(self) -> Any:
        """
        Return the injected ``BaseGraphStore`` instance.

        Tasks call this to upsert nodes/edges and run Cypher queries::

            gs = ctx.get_graphstore()
            await gs.upsert_nodes(nodes)

        Raises:
            RuntimeError: If no graphstore has been injected.
        """
        if self._graphstore is None:
            raise RuntimeError(
                "No GraphStore injected into TaskContext. "
                "The workflow engine must call ctx.set_graphstore() "
                "before task execution, or this workflow does not "
                "require graph operations."
            )
        return self._graphstore

    def has_graphstore(self) -> bool:
        """Whether a graphstore has been injected."""
        return self._graphstore is not None

    def get_media_processor(self) -> Any:
        """
        Return the injected ``MediaProcessorRegistry``.

        Tasks obtain a specific processor via ``registry.get(media_file)``::

            registry = ctx.get_media_processor()
            processor = registry.get(media_file)
            meta = await processor.extract_metadata(media_file)

        Raises:
            RuntimeError: If no media processor has been injected.
        """
        if self._media_processor is None:
            raise RuntimeError(
                "No MediaProcessorRegistry injected into TaskContext. "
                "The workflow engine must call ctx.set_media_processor() "
                "before task execution, or this workflow does not "
                "require media processing."
            )
        return self._media_processor

    def has_media_processor(self) -> bool:
        """Whether a media processor registry has been injected."""
        return self._media_processor is not None

    def get_lock(self) -> Any:
        """
        Return the injected ``BaseDistributedLock`` instance.

        Tasks use it to serialize concurrent modifications::

            async with ctx.get_lock().lock("graph:upsert:task-123"):
                await ctx.get_graphstore().upsert_nodes(nodes)

        Raises:
            RuntimeError: If no lock has been injected.
        """
        if self._lock is None:
            raise RuntimeError(
                "No DistributedLock injected into TaskContext. "
                "The workflow engine must call ctx.set_lock() "
                "before task execution, or this workflow does not "
                "require distributed locking."
            )
        return self._lock

    def has_lock(self) -> bool:
        """Whether a distributed lock has been injected."""
        return self._lock is not None

    def get_circuit_breaker(self) -> Any:
        """
        Return the injected ``CircuitBreakerRegistry``.

        Tasks can protect downstream calls via::

            registry = ctx.get_circuit_breaker()
            cb = await registry.get(model_id)
            result = await cb.call(lambda: adapter.chat(messages))

        Raises:
            RuntimeError: If no circuit-breaker registry has been injected.
        """
        if self._circuit_breaker is None:
            raise RuntimeError(
                "No CircuitBreakerRegistry injected into TaskContext. "
                "The workflow engine must call ctx.set_circuit_breaker() "
                "before task execution, or this workflow does not "
                "require circuit-breaker protection."
            )
        return self._circuit_breaker

    def has_circuit_breaker(self) -> bool:
        """Whether a circuit-breaker registry has been injected."""
        return self._circuit_breaker is not None

    def get_objectstore(self) -> Any:
        """
        Get the injected ``BaseObjectStore`` (MinIO / S3 / in-memory).

        Example::

            store = ctx.get_objectstore()
            ref = await store.put("reports", "result.json", json_bytes)
            url = await store.presigned_get_url("reports", "result.json")

        Raises:
            RuntimeError: If no object store has been injected.
        """
        if self._objectstore is None:
            raise RuntimeError(
                "No object store has been injected. "
                "The workflow engine must call ctx.set_objectstore() "
                "during bootstrapping. Call has_objectstore() to check "
                "before accessing."
            )
        return self._objectstore

    def has_objectstore(self) -> bool:
        """Whether an object store has been injected."""
        return self._objectstore is not None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata dict with a default."""
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a value in metadata dict (mutable during execution)."""
        self.metadata[key] = value

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return (
            f"TaskContext(task_id={self.task_id!r}, "
            f"workflow_id={self.workflow_id!r}, "
            f"model_id={self.model_id!r}, "
            f"stream={self.stream})"
        )


class _BoundDB:
    """
    Pool-backed handle bound to a named database connection.

    Returned by :meth:`TaskContext.get_db`. Each convenience method
    (``query`` / ``execute`` / ``fetch_one`` / ``fetch_all``) acquires a
    connection from the underlying ``DBManager`` pool, runs the statement,
    and releases the connection automatically — so callers never deal with
    acquire/release and cannot leak connections.

    For multiple statements that must share one connection, use
    :meth:`connection_ctx` as an async context manager.

    This class is duck-typed against ``DBManager`` (no runtime import) to
    preserve the ``core`` → ``db`` layering (core must not depend on db at
    import time).
    """

    __slots__ = ("_manager", "_name")

    def __init__(self, manager: Any, name: str) -> None:
        self._manager = manager
        self._name = name

    @property
    def name(self) -> str:
        """The bound connection name."""
        return self._name

    async def query(
        self,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Acquire, run a SELECT, and release. Returns rows as dicts."""
        return await self._manager.query(self._name, sql, params)

    async def execute(
        self,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> int:
        """Acquire, run INSERT/UPDATE/DELETE/DDL, and release. Returns affected rows."""
        return await self._manager.execute(self._name, sql, params)

    async def fetch_all(
        self,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Alias for :meth:`query`."""
        return await self._manager.query(self._name, sql, params)

    async def fetch_one(
        self,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> dict[str, Any] | None:
        """Run a SELECT and return the first row, or None if no rows."""
        rows = await self._manager.query(self._name, sql, params)
        return rows[0] if rows else None

    def connection_ctx(self) -> Any:
        """
        Return an async context manager yielding a pooled connection.

        Use this when several statements must run on the same connection::

            async with db.connection_ctx() as conn:
                await conn.execute("INSERT ...")
                rows = await conn.query("SELECT ...")
        """
        return self._manager.connection_ctx(self._name)

    def __repr__(self) -> str:
        return f"_BoundDB(name={self._name!r})"
