"""
icore.db.exceptions - Exception hierarchy for the database layer.

All database-related errors inherit from ``DatabaseError``.  Upper-layer
code can catch ``DatabaseError`` to handle any database failure, or catch
specific subtypes for finer-grained handling.

Exception hierarchy::

    DatabaseError
    ├── ConnectionError      # connection establishment / maintenance failed
    ├── QueryError           # query execution failed
    └── PoolExhaustedError  # connection pool has no available connections
"""

from __future__ import annotations


class DatabaseError(Exception):
    """
    Base exception for all database-related errors.

    Catch this to handle any failure originating from the database layer.
    """

    def __init__(self, message: str = "", *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause: Exception | None = cause


class ConnectionError(DatabaseError):
    """
    Raised when a database connection cannot be established or has been lost.

    Common scenarios:
        - Network unreachable / wrong host or port
        - Authentication failure (bad credentials)
        - Connection dropped mid-operation
        - Driver import failure
    """


class QueryError(DatabaseError):
    """
    Raised when a SQL query or statement fails during execution.

    Common scenarios:
        - Syntax error in SQL
        - Constraint violation (unique, foreign key, not null)
        - Permission denied
        - Timeout during query execution
    """


class PoolExhaustedError(DatabaseError):
    """
    Raised when the connection pool has no available connections and the
    acquire timeout has elapsed.

    This typically indicates that the pool is undersized or that connections
    are being held too long by callers.
    """
