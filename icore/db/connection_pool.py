"""
icore.db.connection_pool - Async connection pool manager.

Manages a pool of ``BaseConnector`` instances, reusing connections to
avoid the overhead of frequent connect/disconnect cycles.  Uses
``asyncio.Semaphore`` to enforce maximum concurrent connections and
``asyncio.Queue`` to track available (idle) connections.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from icore.db.base_connector import BaseConnector

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Asynchronous connection pool for ``BaseConnector`` instances.

    The pool lazily creates connections on demand (up to
    ``pool_size + max_overflow``) and reuses released connections.
    A ``Semaphore`` limits concurrent access; callers block on
    ``acquire()`` when the pool is exhausted.

    Usage::

        pool = ConnectionPool(
            connector_factory=lambda: PostgreSQLConnector(config),
            pool_size=10,
            max_overflow=20,
        )
        async with pool:
            conn = await pool.acquire()
            try:
                rows = await conn.query("SELECT 1")
            finally:
                await pool.release(conn)

    Args:
        connector_factory: Callable that returns a new, unconnected
            ``BaseConnector`` instance.  Called lazily when the pool
            needs to create a new connection.
        pool_size: Base number of connections maintained in the pool.
        max_overflow: Additional connections allowed beyond ``pool_size``
            under load.  Total max connections = ``pool_size + max_overflow``.
    """

    def __init__(
        self,
        connector_factory: Callable[[], BaseConnector],
        pool_size: int = 10,
        max_overflow: int = 20,
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if max_overflow < 0:
            raise ValueError("max_overflow must be >= 0")

        self._connector_factory: Callable[[], BaseConnector] = connector_factory
        self._pool_size: int = pool_size
        self._max_overflow: int = max_overflow

        # Total capacity = pool_size + max_overflow
        self._max_connections: int = pool_size + max_overflow

        # Semaphore limits concurrent connections
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self._max_connections
        )

        # Available (idle) connections
        self._available: asyncio.Queue[BaseConnector] = asyncio.Queue()

        # Connections currently in use
        self._in_use: set[BaseConnector] = set()

        # Total connections created (including in-use + available)
        self._total_created: int = 0

        # Whether the pool has been closed
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def pool_size(self) -> int:
        """Configured base pool size."""
        return self._pool_size

    @property
    def max_overflow(self) -> int:
        """Configured max overflow connections."""
        return self._max_overflow

    @property
    def max_connections(self) -> int:
        """Total max connections (pool_size + max_overflow)."""
        return self._max_connections

    @property
    def available_count(self) -> int:
        """Number of idle connections currently in the pool."""
        return self._available.qsize()

    @property
    def in_use_count(self) -> int:
        """Number of connections currently checked out."""
        return len(self._in_use)

    @property
    def total_created(self) -> int:
        """Total connections created since pool init."""
        return self._total_created

    @property
    def is_closed(self) -> bool:
        """Whether the pool has been closed."""
        return self._closed

    # ------------------------------------------------------------------
    # Acquire / Release
    # ------------------------------------------------------------------

    async def acquire(self) -> BaseConnector:
        """
        Acquire a connection from the pool.

        If an idle connection is available, it is returned immediately.
        Otherwise, if under the max connection limit, a new connection is
        created and connected.  If at capacity, the call blocks until a
        connection is released back to the pool.

        Returns:
            A connected ``BaseConnector`` instance ready for use.

        Raises:
            RuntimeError: If the pool has been closed.
        """
        if self._closed:
            raise RuntimeError("ConnectionPool is closed")

        # Acquire a permit (blocks if at capacity)
        await self._semaphore.acquire()

        try:
            # Try to get an idle connection from the queue
            try:
                conn = self._available.get_nowait()
                logger.debug(
                    "Reusing idle connection (available=%d, in_use=%d)",
                    self.available_count,
                    self.in_use_count + 1,
                )
            except asyncio.QueueEmpty:
                # No idle connection; create a new one
                conn = self._connector_factory()
                await conn.connect()
                self._total_created += 1
                logger.debug(
                    "Created new connection (total=%d, in_use=%d)",
                    self._total_created,
                    self.in_use_count + 1,
                )

            self._in_use.add(conn)
            return conn
        except Exception:
            # If anything went wrong, release the permit
            self._semaphore.release()
            raise

    async def release(self, conn: BaseConnector) -> None:
        """
        Return a connection to the pool for reuse.

        The connection is placed back into the available queue.  If the
        connection has been disconnected, it is discarded instead.

        Args:
            conn: The ``BaseConnector`` previously obtained via
                ``acquire()``.
        """
        if self._closed:
            # Pool is closing; just disconnect the connection
            await conn.disconnect()
            self._in_use.discard(conn)
            return

        if conn not in self._in_use:
            # Connection not tracked by this pool; just disconnect
            await conn.disconnect()
            return

        self._in_use.discard(conn)

        if conn.is_connected:
            # Return to the available queue for reuse
            await self._available.put(conn)
            logger.debug(
                "Connection released (available=%d, in_use=%d)",
                self.available_count,
                self.in_use_count,
            )
        else:
            # Connection is dead; discard and release permit
            logger.debug("Discarding dead connection on release")
            try:
                await conn.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting dead connection: %s", e)

        # Release the semaphore permit
        self._semaphore.release()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """
        Close all connections in the pool.

        Disconnects both idle (available) and in-use connections.  After
        calling this, the pool is marked as closed and ``acquire()`` will
        raise ``RuntimeError``.
        """
        self._closed = True

        # Close all idle connections
        idle_conns: list[BaseConnector] = []
        while not self._available.empty():
            try:
                idle_conns.append(self._available.get_nowait())
            except asyncio.QueueEmpty:
                break

        for conn in idle_conns:
            try:
                await conn.disconnect()
            except Exception as e:
                logger.warning("Error closing idle connection: %s", e)

        # Close all in-use connections
        in_use_conns = list(self._in_use)
        self._in_use.clear()
        for conn in in_use_conns:
            try:
                await conn.disconnect()
            except Exception as e:
                logger.warning("Error closing in-use connection: %s", e)

        # Release all semaphore permits
        for _ in range(self._max_connections):
            self._semaphore.release()

        logger.info(
            "ConnectionPool closed (was managing %d total connections)",
            self._total_created,
        )

    # ------------------------------------------------------------------
    # Async context manager (pool lifecycle)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> ConnectionPool:
        """Enter async context; returns the pool itself."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context; closes all connections."""
        await self.close_all()
