"""
icore.db.base_connector - Abstract base class for all database connectors.

Defines the unified async interface that all database adapters must implement.
Upper layers (Task / TaskContext) depend on this abstraction, not on specific
adapter implementations.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class BaseConnector(abc.ABC):
    """
    Abstract base class for all database connectors.

    Every concrete database adapter (PostgreSQL, Oracle, Hive, etc.) must
    inherit from this class and implement all abstract methods.

    The connector provides a unified async interface for connecting,
    querying, executing statements, and closing connections.  Upper-layer
    code interacts exclusively through this interface, achieving decoupling
    from specific database drivers.

    Subclasses must set the class-level ``db_type`` attribute and provide
    a ``name`` property that identifies the logical connection.

    Usage::

        class MyConnector(BaseConnector):
            db_type = "mydb"
            ...
    """

    #: Database type identifier (e.g. ``"postgresql"``, ``"oracle"``).
    #: Subclasses MUST override this.
    db_type: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Logical connection name (e.g. ``"main_db"``)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether the underlying connection is currently active."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Abstract async methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def connect(self) -> None:
        """
        Establish the database connection.

        Creates the underlying driver connection object and authenticates.
        Called lazily by the connection pool on first ``acquire()``.

        Raises:
            ConnectionError: If the connection attempt fails.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect and release all underlying resources.

        Closes the driver connection and frees associated socket / memory
        resources.  Safe to call multiple times (subsequent calls are no-ops).
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def execute(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> int:
        """
        Execute a non-query SQL statement (INSERT/UPDATE/DELETE/DDL).

        Uses parameterized binding to prevent SQL injection.  The placeholder
        syntax depends on the underlying driver; adapters translate the
        ``params`` tuple to the driver-specific format.

        Args:
            sql: SQL statement with parameterized placeholders.
            params: Positional parameter values for binding.

        Returns:
            Number of rows affected by the statement.

        Raises:
            QueryError: If execution fails.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def query(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT query and return rows as dictionaries.

        Each row is converted to a ``dict`` mapping column name to value,
        regardless of the native row type returned by the driver.  This
        ensures upper-layer code never depends on driver-specific row
        types.

        Args:
            sql: SELECT statement with parameterized placeholders.
            params: Positional parameter values for binding.

        Returns:
            A list of dictionaries, one per row.  Empty list if no rows.

        Raises:
            QueryError: If the query fails.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the connection (logical close).

        Semantically, this is the hook called when the connection pool
        reclaims a connection.  By default it should behave like
        ``disconnect()``, but subclasses may override to implement custom
        cleanup logic (e.g. resetting session state before returning to
        pool).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # v0.5: Optimistic-lock helper (default implementation)
    # ------------------------------------------------------------------

    async def execute_with_version(
        self,
        sql: str,
        params: dict[str, Any],
        expected_version: int,
    ) -> int:
        """
        Execute an UPDATE guarded by an optimistic version check.

        Caller writes SQL like::

            UPDATE entities
            SET name = :name, version = version + 1
            WHERE id = :id AND version = :expected_version

        and passes ``expected_version`` separately. The base connector
        injects ``expected_version`` into ``params`` (if absent) and
        delegates to ``execute()``. Returns the affected-rows count:
        ``0`` means a version conflict (another writer committed first).

        Adapters with native optimistic-lock support may override this
        to push the check into a single server-side statement.

        Note:
            ``params`` uses named placeholders (``:name`` style) to
            match the SQL example above. Concrete adapters that use
            positional placeholders (``$1``, ``?``) must override this
            method to translate the dict to their driver's expected
            format, or supply a positional SQL string.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            f"execute_with_version(); override it to enable "
            f"optimistic locking."
        )
