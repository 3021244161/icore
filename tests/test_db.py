"""
Smoke tests for icore.db - database connectors, connection pool, and manager.

Covers:
    - BaseConnector: abstract interface enforcement
    - All 4 adapters (PostgreSQL/MySQL/Oracle/Hive): instantiation,
      db_type, name derivation, is_connected before connect, lazy driver
      import error handling
    - ConnectionPool: acquire/release, idle reuse, max capacity, close_all,
      closed-pool rejection, dead-connection discard on release
    - DBManager: register, supported_types, register_adapter (custom +
      invalid), lazy pool creation, get_connection KeyError on unknown,
      pool stats, connection_ctx, close_all

These tests are deterministic and offline. No real database connections
are made; adapter.connect() is exercised via a fake connector for the
pool, and real adapters are only checked at the import/config level
(lazy driver import is validated to raise ImportError when the driver
is absent).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from icore.config import DatabaseConnectionConfig
from icore.db.base_connector import BaseConnector
from icore.db.connection_pool import ConnectionPool
from icore.db.manager import DBManager
from icore.db.adapters import (
    HiveConnector,
    MySQLConnector,
    OracleConnector,
    PostgreSQLConnector,
)


# ---------------------------------------------------------------------------
# Fake connector for pool tests (avoids needing a real DB)
# ---------------------------------------------------------------------------

class _FakeConnector(BaseConnector):
    """In-memory connector that mimics connect/disconnect/query."""

    db_type = "fake"

    def __init__(self, name: str = "fake") -> None:
        self._name = name
        self._connected = False
        self.disconnect_calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self.disconnect_calls += 1

    async def execute(self, sql, params=None) -> int:
        return 1

    async def query(self, sql, params=None):
        return [{"sql": sql, "params": list(params or ())}]

    async def close(self) -> None:
        await self.disconnect()


# ---------------------------------------------------------------------------
# BaseConnector abstract enforcement
# ---------------------------------------------------------------------------

class TestBaseConnector:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseConnector()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self):
        # Missing implementations -> TypeError
        with pytest.raises(TypeError):
            class _Incomplete(BaseConnector):
                db_type = "incomplete"
                name = "x"  # noqa: F841

            _Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Adapter instantiation & metadata (no real connections)
# ---------------------------------------------------------------------------

class TestAdaptersInstantiation:
    def _cfg(self, db_type: str, **kw) -> DatabaseConnectionConfig:
        return DatabaseConnectionConfig(
            db_type=db_type,
            host="localhost",
            port=5432,
            username="user",
            password="pass",
            database="testdb",
            **kw,
        )

    def test_postgresql_metadata(self):
        cfg = self._cfg("postgresql")
        c = PostgreSQLConnector(cfg)
        assert c.db_type == "postgresql"
        assert c.name == "testdb"
        assert c.is_connected is False

    def test_postgresql_name_falls_back_to_host(self):
        cfg = DatabaseConnectionConfig(
            db_type="postgresql", host="db.host", database=""
        )
        c = PostgreSQLConnector(cfg)
        assert c.name == "pg_db.host"

    def test_mysql_metadata(self):
        cfg = self._cfg("mysql")
        c = MySQLConnector(cfg)
        assert c.db_type == "mysql"
        assert c.is_connected is False

    def test_oracle_metadata(self):
        cfg = self._cfg("oracle")
        c = OracleConnector(cfg)
        assert c.db_type == "oracle"
        assert c.is_connected is False

    def test_hive_metadata(self):
        cfg = self._cfg("hive")
        c = HiveConnector(cfg)
        assert c.db_type == "hive"
        assert c.is_connected is False

    @pytest.mark.asyncio
    async def test_postgresql_connect_without_driver_raises_importerror(self):
        # asyncpg may or may not be installed in the test env. If it is
        # installed, this test is skipped (cannot easily uninstall). We
        # only assert the lazy-import path is exercised: connect() must
        # either raise ImportError (driver absent) or succeed (present).
        cfg = self._cfg("postgresql")
        c = PostgreSQLConnector(cfg)
        try:
            import asyncpg  # noqa: F401
            asyncpg_present = True
        except ImportError:
            asyncpg_present = False

        if asyncpg_present:
            pytest.skip("asyncpg is installed; cannot test ImportError path")
        else:
            with pytest.raises(ImportError, match="asyncpg"):
                await c.connect()

    @pytest.mark.asyncio
    async def test_postgresql_query_before_connect_raises(self):
        cfg = self._cfg("postgresql")
        c = PostgreSQLConnector(cfg)
        with pytest.raises(RuntimeError, match="not connected"):
            await c.query("SELECT 1")

    @pytest.mark.asyncio
    async def test_postgresql_execute_before_connect_raises(self):
        cfg = self._cfg("postgresql")
        c = PostgreSQLConnector(cfg)
        with pytest.raises(RuntimeError, match="not connected"):
            await c.execute("INSERT INTO t VALUES (1)")

    @pytest.mark.asyncio
    async def test_disconnect_is_safe_when_not_connected(self):
        cfg = self._cfg("postgresql")
        c = PostgreSQLConnector(cfg)
        # Should be a no-op, not raise
        await c.disconnect()
        await c.close()


# ---------------------------------------------------------------------------
# ConnectionPool
# ---------------------------------------------------------------------------

class TestConnectionPool:
    @pytest.mark.asyncio
    async def test_acquire_creates_and_connects(self):
        pool = ConnectionPool(
            connector_factory=lambda: _FakeConnector("c1"),
            pool_size=2,
            max_overflow=3,
        )
        async with pool:
            conn = await pool.acquire()
            try:
                assert conn.is_connected is True
                assert pool.in_use_count == 1
                assert pool.total_created == 1
            finally:
                await pool.release(conn)
            assert pool.in_use_count == 0
            assert pool.available_count == 1

    @pytest.mark.asyncio
    async def test_release_reuses_idle_connection(self):
        pool = ConnectionPool(
            connector_factory=lambda: _FakeConnector(),
            pool_size=2,
        )
        async with pool:
            conn1 = await pool.acquire()
            await pool.release(conn1)
            conn2 = await pool.acquire()
            try:
                assert conn2 is conn1  # reused
                assert pool.total_created == 1
            finally:
                await pool.release(conn2)

    @pytest.mark.asyncio
    async def test_max_connections_enforced_by_blocking(self):
        pool = ConnectionPool(
            connector_factory=lambda: _FakeConnector(),
            pool_size=1,
            max_overflow=1,
        )
        async with pool:
            c1 = await pool.acquire()
            c2 = await pool.acquire()
            assert pool.in_use_count == 2
            assert pool.max_connections == 2

            # Third acquire should block; release one then it should proceed
            async def release_after_delay():
                await asyncio.sleep(0.05)
                await pool.release(c2)

            asyncio.create_task(release_after_delay())
            c3 = await pool.acquire()  # should unblock after release
            assert c3 is c2  # got the released one back
            await pool.release(c1)
            await pool.release(c3)

    @pytest.mark.asyncio
    async def test_release_dead_connection_discarded(self):
        pool = ConnectionPool(
            connector_factory=lambda: _FakeConnector(),
            pool_size=2,
        )
        async with pool:
            conn = await pool.acquire()
            # Simulate the connection dying while in use
            await conn.disconnect()
            assert conn.is_connected is False
            await pool.release(conn)
            # Should be discarded, not returned to available
            assert pool.available_count == 0

    @pytest.mark.asyncio
    async def test_close_all_closes_connections(self):
        pool = ConnectionPool(
            connector_factory=lambda: _FakeConnector(),
            pool_size=2,
        )
        c1 = await pool.acquire()
        c2 = await pool.acquire()
        await pool.release(c1)  # c1 idle, c2 in-use
        await pool.close_all()
        assert pool.is_closed is True
        assert c1.disconnect_calls >= 1
        assert c2.disconnect_calls >= 1

    @pytest.mark.asyncio
    async def test_acquire_after_close_raises(self):
        pool = ConnectionPool(
            connector_factory=lambda: _FakeConnector(),
            pool_size=1,
        )
        await pool.close_all()
        with pytest.raises(RuntimeError, match="closed"):
            await pool.acquire()

    def test_invalid_pool_args(self):
        with pytest.raises(ValueError):
            ConnectionPool(connector_factory=lambda: _FakeConnector(), pool_size=0)
        with pytest.raises(ValueError):
            ConnectionPool(
                connector_factory=lambda: _FakeConnector(),
                pool_size=1,
                max_overflow=-1,
            )

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_on_exit(self):
        pool = ConnectionPool(
            connector_factory=lambda: _FakeConnector(),
            pool_size=2,
        )
        async with pool:
            conn = await pool.acquire()
            await pool.release(conn)
        # After exit, pool should be closed
        assert pool.is_closed is True


# ---------------------------------------------------------------------------
# DBManager
# ---------------------------------------------------------------------------

class TestDBManager:
    def test_supported_types(self):
        mgr = DBManager()
        types = mgr.supported_types
        assert set(types) == {"postgresql", "oracle", "hive", "mysql"}

    def test_register_stores_config(self):
        mgr = DBManager()
        cfg = DatabaseConnectionConfig(
            db_type="postgresql", host="h", database="db"
        )
        mgr.register("main", cfg)
        assert "main" in mgr.registered_names

    def test_register_unsupported_type_raises(self):
        mgr = DBManager()
        cfg = DatabaseConnectionConfig(db_type="cassandra", database="x")
        with pytest.raises(ValueError, match="Unsupported"):
            mgr.register("cass", cfg)

    def test_register_custom_adapter(self):
        mgr = DBManager()
        mgr.register_adapter("fake", _FakeConnector)
        assert "fake" in mgr.supported_types

        cfg = DatabaseConnectionConfig(db_type="fake", database="fdb")
        mgr.register("fake_conn", cfg)
        assert "fake_conn" in mgr.registered_names

    def test_register_invalid_adapter_rejected(self):
        mgr = DBManager()
        with pytest.raises(TypeError):
            mgr.register_adapter("bad", dict)  # not a BaseConnector subclass

    @pytest.mark.asyncio
    async def test_get_connection_unknown_name_raises(self):
        mgr = DBManager()
        with pytest.raises(KeyError):
            await mgr.get_connection("ghost")

    @pytest.mark.asyncio
    async def test_get_connection_creates_pool_lazily(self):
        mgr = DBManager()
        mgr.register_adapter("fake", _FakeConnector)
        cfg = DatabaseConnectionConfig(db_type="fake", database="fdb")
        mgr.register("c", cfg)

        # Pool not created yet
        assert "c" not in mgr._pools
        conn = await mgr.get_connection("c")
        try:
            assert conn.is_connected is True
            # Pool now exists
            assert "c" in mgr._pools
        finally:
            await mgr.release_connection("c", conn)

    @pytest.mark.asyncio
    async def test_connection_ctx_releases_on_exit(self):
        mgr = DBManager()
        mgr.register_adapter("fake", _FakeConnector)
        cfg = DatabaseConnectionConfig(db_type="fake", database="fdb")
        mgr.register("c", cfg)

        async with mgr.connection_ctx("c") as conn:
            assert conn.is_connected is True
        # After ctx exit, connection should be back in pool (available)
        pool = mgr._pools["c"]
        assert pool.in_use_count == 0
        assert pool.available_count == 1

    @pytest.mark.asyncio
    async def test_connection_ctx_releases_on_exception(self):
        mgr = DBManager()
        mgr.register_adapter("fake", _FakeConnector)
        cfg = DatabaseConnectionConfig(db_type="fake", database="fdb")
        mgr.register("c", cfg)

        with pytest.raises(RuntimeError):
            async with mgr.connection_ctx("c") as conn:
                assert conn.is_connected is True
                raise RuntimeError("boom")
        pool = mgr._pools["c"]
        assert pool.in_use_count == 0

    @pytest.mark.asyncio
    async def test_query_and_execute_convenience(self):
        mgr = DBManager()
        mgr.register_adapter("fake", _FakeConnector)
        cfg = DatabaseConnectionConfig(db_type="fake", database="fdb")
        mgr.register("c", cfg)

        rows = await mgr.query("c", "SELECT 1", ())
        assert rows == [{"sql": "SELECT 1", "params": []}]
        affected = await mgr.execute("c", "INSERT INTO t VALUES (1)", ())
        assert affected == 1

    @pytest.mark.asyncio
    async def test_get_pool_stats(self):
        mgr = DBManager()
        mgr.register_adapter("fake", _FakeConnector)
        cfg = DatabaseConnectionConfig(
            db_type="fake", database="fdb", pool_size=3, max_overflow=2
        )
        mgr.register("c", cfg)

        # Trigger lazy pool creation
        conn = await mgr.get_connection("c")
        stats = mgr.get_pool_stats("c")
        assert stats["pool_size"] == 3
        assert stats["max_overflow"] == 2
        assert stats["in_use"] == 1
        assert stats["total_created"] == 1
        await mgr.release_connection("c", conn)

    def test_get_pool_stats_unknown_raises(self):
        mgr = DBManager()
        with pytest.raises(KeyError):
            mgr.get_pool_stats("ghost")

    @pytest.mark.asyncio
    async def test_close_all(self):
        mgr = DBManager()
        mgr.register_adapter("fake", _FakeConnector)
        cfg = DatabaseConnectionConfig(db_type="fake", database="fdb")
        mgr.register("c1", cfg)
        mgr.register("c2", cfg)

        await mgr.get_connection("c1")
        await mgr.get_connection("c2")
        await mgr.close_all()
        assert mgr._pools == {}

    @pytest.mark.asyncio
    async def test_release_to_unknown_pool_disconnects(self):
        mgr = DBManager()
        conn = _FakeConnector("orphan")
        await conn.connect()
        # No pool registered for this name -> should disconnect, not crash
        await mgr.release_connection("unknown", conn)
        assert conn.is_connected is False
