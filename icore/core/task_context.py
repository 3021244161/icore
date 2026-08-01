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

    - get_model_adapter(): Returns the LLM model adapter (from ModelManager)
    - get_db(name):        Returns a database connector (from DBManager)

IMPORTANT: The actual ModelManager and DBManager are implemented in later
stories (US-003, US-005). To avoid circular dependencies, TaskContext uses
duck typing at runtime: it stores references to managers via private
attributes and calls their methods by convention. Type annotations use
TYPE_CHECKING imports for IDE support without runtime import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    # These imports are only for type checking; actual implementations
    # come from US-003 (db.manager.DBManager) and US-005 (models.manager.ModelManager).
    # We use TYPE_CHECKING to avoid circular imports at runtime.
    from icore.db.manager import DBManager  # noqa: F401
    from icore.models.manager import ModelManager  # noqa: F401


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

    # ------------------------------------------------------------------
    # Manager injection methods (called by the workflow engine)
    # ------------------------------------------------------------------

    def set_model_manager(self, manager: Any) -> None:
        """Inject the ModelManager instance (called by engine)."""
        object.__setattr__(self, "_model_manager", manager)

    def set_db_manager(self, manager: Any) -> None:
        """Inject the DBManager instance (called by engine)."""
        object.__setattr__(self, "_db_manager", manager)

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
