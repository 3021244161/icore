"""
icore.db - Database connection layer for icore.

Provides a unified async interface for connecting to multiple heterogeneous
databases (PostgreSQL, Oracle, Hive, etc.) through adapter classes and
connection pooling.

Key classes:
    - BaseConnector:  Abstract base class for all database connectors.
    - ConnectionPool: Async connection pool manager.
    - DBManager:      Multi-datasource manager (top-level entry point).
    - Adapters:       PostgreSQLConnector, OracleConnector, HiveConnector.

Usage::

    from icore.db.manager import DBManager
    from icore.config import DatabaseConnectionConfig

    mgr = DBManager()
    mgr.register("main_db", DatabaseConnectionConfig(
        db_type="postgresql", host="localhost", port=5432,
        username="user", password="pass", database="mydb",
    ))

    rows = await mgr.query("main_db", "SELECT 1 AS val")
"""

from __future__ import annotations

from icore.db.base_connector import BaseConnector
from icore.db.connection_pool import ConnectionPool
from icore.db.exceptions import (
    ConnectionError,
    DatabaseError,
    PoolExhaustedError,
    QueryError,
)
from icore.db.manager import DBManager

__all__ = [
    "BaseConnector",
    "ConnectionPool",
    "DBManager",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "PoolExhaustedError",
]
