"""
icore.db.manager - Multi-datasource database manager.

``DBManager`` is the top-level entry point for database access in icore.
It manages multiple named database connections, each backed by its own
``ConnectionPool``.  Upper layers (Task / TaskContext) obtain connectors
through ``DBManager.get_connection(name)``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from icore.db.adapters import (
    HiveConnector,
    MySQLConnector,
    OracleConnector,
    PostgreSQLConnector,
)
from icore.db.base_connector import BaseConnector
from icore.db.connection_pool import ConnectionPool

if TYPE_CHECKING:
    from icore.config import DatabaseConnectionConfig

logger = logging.getLogger(__name__)


class DBManager:
    """
    Multi-datasource database connection manager.

    Manages a collection of named database connections, each with its own
    ``ConnectionPool``.  Connections are created lazily: ``register()``
    only stores configuration; the pool and first connection are created
    on the first ``get_connection()`` call.

    Usage::

        from icore.config import DatabaseConnectionConfig
        from icore.db.manager import DBManager

        mgr = DBManager()
        mgr.register("main_db", DatabaseConnectionConfig(
            db_type="postgresql",
            host="localhost", port=5432,
            username="user", password="pass",
            database="mydb",
        ))

        rows = await mgr.query("main_db", "SELECT 1 AS val")

    Attributes:
        _configs: Maps connection name -> ``DatabaseConnectionConfig``.
        _pools: Maps connection name -> ``ConnectionPool`` (lazily created).
        _adapter_map: Maps ``db_type`` string -> adapter class.
    """

    def __init__(self) -> None:
        self._configs: dict[str, DatabaseConnectionConfig] = {}
        self._pools: dict[str, ConnectionPool] = {}
        self._adapter_map: dict[str, type[BaseConnector]] = {
            "postgresql": PostgreSQLConnector,
            "oracle": OracleConnector,
            "hive": HiveConnector,
            "mysql": MySQLConnector,
        }

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, config: DatabaseConnectionConfig) -> None:
        """
        Register a named database connection configuration.

        Stores the config for lazy pool creation.  If a connection with
        the same name already exists, it is replaced (existing pool is
        NOT closed automatically - call ``close_all()`` first if needed).

        Args:
            name: Logical connection name (e.g. ``"main_db"``).
            config: Database connection configuration.
        """
        if config.db_type not in self._adapter_map:
            raise ValueError(
                f"Unsupported database type: '{config.db_type}'. "
                f"Supported types: {list(self._adapter_map.keys())}"
            )

        self._configs[name] = config
        logger.info(
            "Registered database connection '%s' (type=%s, host=%s:%d)",
            name,
            config.db_type,
            config.host,
            config.port,
        )

    def register_adapter(
        self, db_type: str, adapter_class: type[BaseConnector]
    ) -> None:
        """
        Register a custom database adapter class.

        Allows extending ``DBManager`` with new database types without
        modifying the source code.

        Args:
            db_type: Database type identifier (must match config.db_type).
            adapter_class: A ``BaseConnector`` subclass.
        """
        if not issubclass(adapter_class, BaseConnector):
            raise TypeError(
                f"adapter_class must be a subclass of BaseConnector, "
                f"got {adapter_class}"
            )
        self._adapter_map[db_type] = adapter_class
        logger.info("Registered custom adapter for db_type='%s'", db_type)

    # ------------------------------------------------------------------
    # Connection access
    # ------------------------------------------------------------------

    async def get_connection(self, name: str) -> BaseConnector:
        """
        Get a connection from the named connection pool.

        If the pool for this name does not exist yet, it is created
        lazily using the registered configuration.  The connection must
        be released back to the pool via ``release_connection()`` or
        the context manager ``connection_ctx()``.

        Args:
            name: The registered connection name.

        Returns:
            A ``BaseConnector`` instance ready for use.

        Raises:
            KeyError: If no connection with this name is registered.
        """
        if name not in self._configs:
            raise KeyError(
                f"No database connection registered with name '{name}'. "
                f"Registered: {list(self._configs.keys())}"
            )

        # Lazy pool creation
        if name not in self._pools:
            self._create_pool(name)

        return await self._pools[name].acquire()

    async def release_connection(
        self, name: str, conn: BaseConnector
    ) -> None:
        """
        Return a connection to its pool.

        Args:
            name: The connection name used to acquire it.
            conn: The ``BaseConnector`` instance to release.
        """
        if name not in self._pools:
            logger.warning(
                "Pool for '%s' not found; disconnecting connection", name
            )
            await conn.disconnect()
            return
        await self._pools[name].release(conn)

    def connection_ctx(self, name: str):
        """
        Return an async context manager for acquiring and releasing a
        connection.

        Usage::

            async with mgr.connection_ctx("main_db") as conn:
                rows = await conn.query("SELECT 1")

        Ensures the connection is always released, even on exception.

        Args:
            name: The registered connection name.

        Returns:
            An async context manager yielding a ``BaseConnector``.
        """
        mgr = self

        class _ConnectionCtx:
            async def __aenter__(self) -> BaseConnector:
                self._conn = await mgr.get_connection(name)
                return self._conn

            async def __aexit__(self, *args: Any) -> None:
                await mgr.release_connection(name, self._conn)

        return _ConnectionCtx()

    # ------------------------------------------------------------------
    # Convenience query
    # ------------------------------------------------------------------

    async def query(
        self,
        name: str,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convenience method: acquire connection, execute query, release.

        This wraps the acquire -> query -> release cycle in a single call.
        For multiple operations on the same connection, use
        ``connection_ctx()`` instead.

        Args:
            name: The registered connection name.
            sql: SELECT statement with parameterized placeholders.
            params: Positional parameter values.

        Returns:
            A list of result rows as dictionaries.
        """
        async with self.connection_ctx(name) as conn:
            return await conn.query(sql, params)

    async def execute(
        self,
        name: str,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> int:
        """
        Convenience method: acquire connection, execute statement, release.

        Args:
            name: The registered connection name.
            sql: INSERT/UPDATE/DELETE/DDL statement.
            params: Positional parameter values.

        Returns:
            Number of rows affected.
        """
        async with self.connection_ctx(name) as conn:
            return await conn.execute(sql, params)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """Close all connection pools and disconnect all connections."""
        for name, pool in self._pools.items():
            try:
                await pool.close_all()
                logger.info("Closed connection pool for '%s'", name)
            except Exception as e:
                logger.error("Error closing pool '%s': %s", name, e)
        self._pools.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_pool(self, name: str) -> None:
        """Create a connection pool for the named connection."""
        config = self._configs[name]
        adapter_class = self._adapter_map[config.db_type]

        def factory() -> BaseConnector:
            return adapter_class(config)

        self._pools[name] = ConnectionPool(
            connector_factory=factory,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
        )
        logger.info(
            "Created connection pool for '%s' (pool_size=%d, max_overflow=%d)",
            name,
            config.pool_size,
            config.max_overflow,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def registered_names(self) -> list[str]:
        """List of registered connection names."""
        return list(self._configs.keys())

    @property
    def supported_types(self) -> list[str]:
        """List of supported database types."""
        return list(self._adapter_map.keys())

    def get_pool_stats(self, name: str) -> dict[str, int]:
        """
        Get pool statistics for a named connection.

        Returns:
            Dict with keys: ``pool_size``, ``max_overflow``,
            ``available``, ``in_use``, ``total_created``.
        """
        if name not in self._pools:
            raise KeyError(f"No pool for connection '{name}'")

        pool = self._pools[name]
        return {
            "pool_size": pool.pool_size,
            "max_overflow": pool.max_overflow,
            "available": pool.available_count,
            "in_use": pool.in_use_count,
            "total_created": pool.total_created,
        }
