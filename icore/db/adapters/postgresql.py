"""
icore.db.adapters.postgresql - PostgreSQL database connector adapter.

Uses ``asyncpg`` as the underlying async driver.  The driver is imported
lazily inside ``connect()`` so that the module can be imported without
``asyncpg`` being installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from icore.db.base_connector import BaseConnector

if TYPE_CHECKING:
    from icore.config import DatabaseConnectionConfig

logger = logging.getLogger(__name__)


class PostgreSQLConnector(BaseConnector):
    """
    PostgreSQL database connector using ``asyncpg``.

    Provides the unified ``BaseConnector`` interface backed by asyncpg's
    native async connection.  Query results are converted from asyncpg
    ``Record`` objects to plain ``dict`` for driver-agnostic consumption.

    Args:
        config: Database connection configuration (host, port, credentials,
            pool settings, extra params).
    """

    db_type = "postgresql"

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        self._config = config
        self._conn: Any = None  # asyncpg.Connection, lazily set

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Logical connection name (uses database name from config)."""
        return self._config.database or f"pg_{self._config.host}"

    @property
    def is_connected(self) -> bool:
        """Whether the asyncpg connection is active."""
        if self._conn is None:
            return False
        # asyncpg Connection.is_closed() returns True when closed
        try:
            return not bool(self._conn.is_closed())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish an asyncpg connection."""
        if self._conn is not None:
            logger.debug("PostgreSQLConnector already connected")
            return

        try:
            import asyncpg  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "asyncpg is required for PostgreSQLConnector. "
                "Install with: pip install asyncpg"
            ) from e

        logger.debug(
            "Connecting to PostgreSQL at %s:%d/%s",
            self._config.host,
            self._config.port,
            self._config.database,
        )

        self._conn = await asyncpg.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.username,
            password=self._config.password,
            database=self._config.database,
            **self._config.extra_params,
        )

        logger.info("PostgreSQL connection established: %s", self.name)

    async def disconnect(self) -> None:
        """Close the asyncpg connection."""
        if self._conn is None:
            return
        try:
            await self._conn.close()
        except Exception as e:
            logger.warning("Error closing PostgreSQL connection: %s", e)
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

        asyncpg uses ``$1, $2, ...`` style placeholders.  The ``params``
        tuple is unpacked as positional arguments.
        """
        if self._conn is None:
            raise RuntimeError("PostgreSQLConnector is not connected")

        result = await self._conn.execute(sql, *(params or ()))
        # asyncpg execute returns a string like "INSERT 0 5"
        # Parse the affected row count from the last token
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def query(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT query.

        asyncpg ``Record`` objects are converted to ``dict`` for
        driver-agnostic consumption.
        """
        if self._conn is None:
            raise RuntimeError("PostgreSQLConnector is not connected")

        rows = await self._conn.fetch(sql, *(params or ()))
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Close (logical)
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Logical close - delegates to ``disconnect()``."""
        await self.disconnect()
