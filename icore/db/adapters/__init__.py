"""
icore.db.adapters - Database adapter implementations.

Each adapter implements the ``BaseConnector`` interface for a specific
database driver.  Importing this package does not require any DB driver
to be installed; drivers are imported lazily inside ``connect()``.
"""

from __future__ import annotations

from icore.db.adapters.hive import HiveConnector
from icore.db.adapters.mysql import MySQLConnector
from icore.db.adapters.oracle import OracleConnector
from icore.db.adapters.postgresql import PostgreSQLConnector

__all__ = [
    "PostgreSQLConnector",
    "OracleConnector",
    "HiveConnector",
    "MySQLConnector",
]
