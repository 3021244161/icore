"""
icore.db.adapters.hive - Hive database connector adapter.

Uses ``pyhive`` (which wraps the HiveServer2 Thrift API) as the underlying
driver.  Since pyhive is synchronous, operations are wrapped with
``asyncio.to_thread()`` / ``run_in_executor()`` to provide an async
interface without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from icore.db.base_connector import BaseConnector

if TYPE_CHECKING:
    from icore.config import DatabaseConnectionConfig

logger = logging.getLogger(__name__)


class HiveConnector(BaseConnector):
    """
    Hive database connector using ``pyhive.hive``.

    Because pyhive is a synchronous driver, all blocking calls are
    delegated to a thread pool via ``asyncio.to_thread()``.  Hive queries
    typically have high latency (seconds to minutes), so callers should
    configure appropriate timeouts.

    Args:
        config: Database connection configuration.  The ``database`` field
            maps to the Hive database/schema name.  ``extra_params`` can
            include ``auth_mechanism``, ``kerberos_service_name``, etc.
    """

    db_type = "hive"

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        self._config = config
        self._conn: Any = None  # pyhive.hive.Connection, lazily set

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Logical connection name (uses database from config)."""
        return self._config.database or f"hive_{self._config.host}"

    @property
    def is_connected(self) -> bool:
        """Whether the pyhive connection is active."""
        if self._conn is None:
            return False
        try:
            # pyhive connections don't have an explicit is_closed method,
            # but we can check if the transport is still open
            transport = getattr(self._conn, "_transport", None)
            if transport is not None:
                return bool(getattr(transport, "isOpen", lambda: True)())
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish a pyhive Hive connection (in a thread)."""
        if self._conn is not None:
            logger.debug("HiveConnector already connected")
            return

        try:
            from pyhive import hive  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "pyhive is required for HiveConnector. "
                "Install with: pip install pyhive"
            ) from e

        logger.debug(
            "Connecting to Hive at %s:%d/%s",
            self._config.host,
            self._config.port,
            self._config.database,
        )

        extra = self._config.extra_params

        # pyhive.hive.connect is synchronous; run in thread pool
        self._conn = await asyncio.to_thread(
            hive.connect,
            host=self._config.host,
            port=self._config.port,
            username=self._config.username,
            password=self._config.password,
            database=self._config.database,
            auth=extra.get("auth_mechanism", "NONE"),
            kerberos_service_name=extra.get("kerberos_service_name"),
            **{
                k: v
                for k, v in extra.items()
                if k not in ("auth_mechanism", "kerberos_service_name")
            },
        )

        logger.info("Hive connection established: %s", self.name)

    async def disconnect(self) -> None:
        """Close the pyhive Hive connection (in a thread)."""
        if self._conn is None:
            return
        try:
            await asyncio.to_thread(self._conn.close)
        except Exception as e:
            logger.warning("Error closing Hive connection: %s", e)
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

        pyhive uses ``%s`` style placeholders.  Operations are run in a
        thread to avoid blocking the async event loop.

        Returns:
            For Hive, affected row count is often unavailable; returns 0
            when the driver does not report it.
        """
        if self._conn is None:
            raise RuntimeError("HiveConnector is not connected")

        cursor = self._conn.cursor()
        try:
            await asyncio.to_thread(cursor.execute, sql, params or ())
            return cursor.rowcount or 0
        finally:
            cursor.close()

    async def query(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT query.

        pyhive cursor rows are converted to ``dict`` using column
        descriptions from ``cursor.description``.  Operations are run in a
        thread to avoid blocking the async event loop.
        """
        if self._conn is None:
            raise RuntimeError("HiveConnector is not connected")

        cursor = self._conn.cursor()
        try:
            await asyncio.to_thread(cursor.execute, sql, params or ())
            columns = (
                [desc[0] for desc in cursor.description]
                if cursor.description
                else []
            )
            rows = await asyncio.to_thread(cursor.fetchall)
            if columns:
                return [dict(zip(columns, row)) for row in rows]
            return [dict(enumerate(row)) for row in rows]
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Close (logical)
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Logical close - delegates to ``disconnect()``."""
        await self.disconnect()
