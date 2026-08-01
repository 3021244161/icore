"""
icore.db.adapters.oracle - Oracle database connector adapter.

Uses ``oracledb`` (the successor to cx_Oracle) in async mode as the
underlying driver.  The driver is imported lazily inside ``connect()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from icore.db.base_connector import BaseConnector

if TYPE_CHECKING:
    from icore.config import DatabaseConnectionConfig

logger = logging.getLogger(__name__)


class OracleConnector(BaseConnector):
    """
    Oracle database connector using ``oracledb`` async mode.

    Provides the unified ``BaseConnector`` interface backed by oracledb's
    async connection.  Oracle DSN can be constructed from host/port and
    service_name (in ``extra_params``) or a full DSN string.

    Args:
        config: Database connection configuration.
    """

    db_type = "oracle"

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        self._config = config
        self._conn: Any = None  # oracledb.AsyncConnection, lazily set

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Logical connection name (uses database/service name from config)."""
        service = self._config.extra_params.get(
            "service_name", self._config.database
        )
        return service or f"oracle_{self._config.host}"

    @property
    def is_connected(self) -> bool:
        """Whether the oracledb connection is active."""
        if self._conn is None:
            return False
        try:
            # oracledb connection has .cursor() that raises if closed
            # A simpler check: the connection object exists and isn't closed
            return not bool(getattr(self._conn, "_closed", False))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish an oracledb async connection."""
        if self._conn is not None:
            logger.debug("OracleConnector already connected")
            return

        try:
            import oracledb  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "oracledb is required for OracleConnector. "
                "Install with: pip install oracledb"
            ) from e

        # Build DSN: prefer service_name from extra_params
        service_name = self._config.extra_params.get(
            "service_name", self._config.database
        )
        dsn = oracledb.makedsn(
            self._config.host,
            self._config.port,
            service_name=service_name,
        )

        logger.debug(
            "Connecting to Oracle at %s:%d/%s",
            self._config.host,
            self._config.port,
            service_name,
        )

        self._conn = await oracledb.connect_async(
            user=self._config.username,
            password=self._config.password,
            dsn=dsn,
        )

        logger.info("Oracle connection established: %s", self.name)

    async def disconnect(self) -> None:
        """Close the oracledb connection."""
        if self._conn is None:
            return
        try:
            await self._conn.close()
        except Exception as e:
            logger.warning("Error closing Oracle connection: %s", e)
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

        oracledb uses ``:1, :2, ...`` or named (``:name``) style
        placeholders.  For simplicity, the adapter translates positional
        ``$1, $2`` style to ``:1, :2`` before executing.
        """
        if self._conn is None:
            raise RuntimeError("OracleConnector is not connected")

        # Translate $1 -> :1 style placeholders
        translated_sql = self._translate_placeholders(sql)

        cursor = self._conn.cursor()
        try:
            await cursor.execute(translated_sql, params or ())
            await self._conn.commit()
            return cursor.rowcount or 0
        finally:
            cursor.close()

    async def query(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT query.

        oracledb cursor rows are converted to ``dict`` using column
        descriptions from ``cursor.description``.
        """
        if self._conn is None:
            raise RuntimeError("OracleConnector is not connected")

        # Translate $1 -> :1 style placeholders
        translated_sql = self._translate_placeholders(sql)

        cursor = self._conn.cursor()
        try:
            await cursor.execute(translated_sql, params or ())
            columns = [col[0] for col in cursor.description]
            rows = await cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_placeholders(sql: str) -> str:
        """
        Translate ``$1, $2, ...`` style placeholders to Oracle's
        ``:1, :2, ...`` style.

        This allows upper-layer code to use a consistent placeholder
        convention while the adapter handles driver-specific syntax.
        """
        import re

        return re.sub(r"\$(\d+)", r":\1", sql)

    # ------------------------------------------------------------------
    # Close (logical)
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Logical close - delegates to ``disconnect()``."""
        await self.disconnect()
