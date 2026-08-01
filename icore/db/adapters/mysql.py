"""
icore.db.adapters.mysql - MySQL database connector adapter.

Uses ``aiomysql`` as the underlying async driver.  The driver is imported
lazily inside ``connect()`` so that the module can be imported without
``aiomysql`` being installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from icore.db.base_connector import BaseConnector

if TYPE_CHECKING:
    from icore.config import DatabaseConnectionConfig

logger = logging.getLogger(__name__)


class MySQLConnector(BaseConnector):
    """
    MySQL database connector using ``aiomysql``.

    Provides the unified ``BaseConnector`` interface backed by aiomysql's
    native async connection.  Query results are converted from aiomysql
    ``dict``-style cursors to plain ``dict`` for driver-agnostic
    consumption.

    Args:
        config: Database connection configuration (host, port, credentials,
            pool settings, extra params).
    """

    db_type = "mysql"

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        self._config = config
        self._conn: Any = None  # aiomysql.Connection, lazily set

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Logical connection name (uses database name from config)."""
        return self._config.database or f"mysql_{self._config.host}"

    @property
    def is_connected(self) -> bool:
        """Whether the aiomysql connection is active."""
        if self._conn is None:
            return False
        try:
            # aiomysql connection has .closed attribute (True when closed)
            return not bool(getattr(self._conn, "closed", True))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish an aiomysql connection."""
        if self._conn is not None:
            logger.debug("MySQLConnector already connected")
            return

        try:
            import aiomysql  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "aiomysql is required for MySQLConnector. "
                "Install with: pip install aiomysql"
            ) from e

        logger.debug(
            "Connecting to MySQL at %s:%d/%s",
            self._config.host,
            self._config.port,
            self._config.database,
        )

        self._conn = await aiomysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.username,
            password=self._config.password,
            db=self._config.database,
            **self._config.extra_params,
        )

        logger.info("MySQL connection established: %s", self.name)

    async def disconnect(self) -> None:
        """Close the aiomysql connection."""
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception as e:
            logger.warning("Error closing MySQL connection: %s", e)
        finally:
            self._conn = None

    # ------------------------------------------------------------------
    # Query / Execute
    # ------------------------------------------------------------------

    async def execute(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> int:
        """
        Execute a non-query statement.

        aiomysql uses ``%s`` style placeholders.  An implicit commit is
        performed after execution for DML statements.
        """
        if self._conn is None:
            raise RuntimeError("MySQLConnector is not connected")

        # Use DictCursor for consistent row format
        async with self._conn.cursor() as cursor:
            await cursor.execute(sql, params or ())
            await self._conn.commit()
            return cursor.rowcount or 0

    async def query(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT query.

        Uses a ``DictCursor`` so rows are returned as dictionaries directly,
        matching the unified ``BaseConnector`` interface without manual
        conversion.
        """
        if self._conn is None:
            raise RuntimeError("MySQLConnector is not connected")

        try:
            import aiomysql  # type: ignore[import-untyped]
        except ImportError:
            # Fallback: use default cursor and convert manually
            async with self._conn.cursor() as cursor:
                await cursor.execute(sql, params or ())
                columns = (
                    [desc[0] for desc in cursor.description]
                    if cursor.description
                    else []
                )
                rows = await cursor.fetchall()
                if columns:
                    return [dict(zip(columns, row)) for row in rows]
                return []
        else:
            # Use DictCursor for direct dict rows
            async with self._conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, params or ())
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Close (logical)
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Logical close - delegates to ``disconnect()``."""
        await self.disconnect()
