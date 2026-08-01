"""
icore.engine.task_queue - Async task queue with priority support.

Provides an async task queue that manages pending task execution requests.
Supports two backends:

    1. In-memory (default): Uses asyncio.PriorityQueue for fast, local
       queuing. Suitable for single-process deployments.
    2. Redis (optional): Uses redis.asyncio (formerly aioredis) for
       distributed queuing across multiple workers. The Redis driver is
       imported lazily inside methods, so this module can be imported
       without redis installed.

The queue is consumed by the WorkflowExecutor's concurrency layer:
requests are enqueued when a new workflow invocation arrives, and
dequeued by worker coroutines when concurrency slots are available.

Priority semantics:
    Lower priority values are dequeued first (standard min-heap behavior
    in asyncio.PriorityQueue). The default priority is 0. Negative values
    are allowed for urgent tasks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass  # reserved for future type-only imports

logger = logging.getLogger(__name__)


@dataclass(order=True)
class TaskItem:
    """
    A single item in the task queue.

    The ``order`` field (priority) is the first field so that
    ``asyncio.PriorityQueue`` and ``heapq`` sort by priority,
    then by sequence number (FIFO within the same priority).

    Attributes:
        priority:  Lower values are dequeued first (default 0).
        seq:       Monotonic sequence number for FIFO ordering within
                   the same priority level.
        task_id:   Unique identifier for this task instance.
        created_at: Unix timestamp when the item was enqueued.
        metadata:  Arbitrary metadata carried with the task
                   (e.g., workflow_name, params, model_id, callback_url).
    """

    priority: int
    seq: int
    task_id: str = field(compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


class TaskQueue:
    """
    Async task queue with priority support and pluggable backends.

    Usage (in-memory):

        queue = TaskQueue(backend="memory")
        await queue.enqueue("task-1", priority=0)
        item = await queue.dequeue()
        print(item.task_id)  # "task-1"

    Usage (Redis):

        queue = TaskQueue(backend="redis", redis_url="redis://localhost:6379")
        await queue.start()
        await queue.enqueue("task-1", priority=0)
        item = await queue.dequeue()
        await queue.stop()

    The Redis backend uses a sorted set (ZADD) for priority ordering
    and a list (LPUSH/BRPOP) for blocking dequeue. Alternatively, a
    simpler LPUSH/RPOP approach with priority buckets can be used.
    This implementation uses ZADD for O(log N) priority insert and
    ZPOPMIN for O(log N) priority dequeue.

    Attributes:
        _backend:        "memory" or "redis".
        _max_size:       Maximum queue depth (None = unlimited).
        _seq_counter:    Monotonic counter for FIFO within priority.
        _pq:             asyncio.PriorityQueue (memory backend).
        _redis:          Redis client (redis backend, lazily created).
        _redis_url:      Redis connection URL.
        _lock:           asyncio.Lock for thread-safe sequence assignment.
        _started:        Whether the backend has been initialized.
    """

    def __init__(
        self,
        backend: str = "memory",
        max_size: Optional[int] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the task queue.

        Args:
            backend:  "memory" (default) or "redis".
            max_size: Maximum number of items in the queue.
                      None means unlimited. When full, enqueue() raises
                      QueueFullError (a RuntimeError subclass).
            redis_url: Redis connection URL (required if backend="redis").

        Raises:
            ValueError: If backend is "redis" but redis_url is None.
        """
        if backend not in ("memory", "redis"):
            raise ValueError(
                f"Unsupported backend '{backend}'. Use 'memory' or 'redis'."
            )
        if backend == "redis" and redis_url is None:
            raise ValueError("redis_url is required when backend='redis'")

        self._backend: str = backend
        self._max_size: Optional[int] = max_size
        self._seq_counter: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

        # Memory backend
        self._pq: asyncio.PriorityQueue[TaskItem] = asyncio.PriorityQueue()

        # Redis backend (lazy)
        self._redis: Any = None
        self._redis_url: Optional[str] = redis_url
        self._redis_key: str = "icore:task_queue"
        self._started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Initialize the backend.

        For the memory backend, this is a no-op.
        For the Redis backend, this creates the async Redis connection.

        Must be called before enqueue/dequeue when using Redis.
        """
        if self._started:
            return
        if self._backend == "redis":
            await self._init_redis()
        self._started = True
        logger.info("TaskQueue started (backend=%s)", self._backend)

    async def stop(self) -> None:
        """
        Clean up backend resources.

        For the memory backend, this clears the queue.
        For the Redis backend, this closes the connection.
        """
        if not self._started:
            return
        if self._backend == "redis" and self._redis is not None:
            await self._redis.close()
            self._redis = None
        self._started = False
        logger.info("TaskQueue stopped (backend=%s)", self._backend)

    async def _init_redis(self) -> None:
        """Create the async Redis connection (lazy import)."""
        try:
            import redis.asyncio as aioredis  # type: ignore
        except ImportError:
            try:
                import aioredis  # type: ignore
                aioredis = aioredis  # legacy package name
            except ImportError:
                raise ImportError(
                    "Neither 'redis' nor 'aioredis' is installed. "
                    "Install with: pip install redis>=4.2"
                )
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        task_id: str,
        priority: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TaskItem:
        """
        Add a task to the queue.

        Args:
            task_id:  Unique identifier for the task instance.
            priority: Lower values are processed first (default 0).
            metadata: Optional metadata to carry with the task.

        Returns:
            The created TaskItem.

        Raises:
            RuntimeError: If the queue is full (max_size exceeded) or
                          the backend is not started.
        """
        if not self._started:
            await self.start()

        # Assign a monotonic sequence for FIFO within same priority
        async with self._lock:
            self._seq_counter += 1
            seq = self._seq_counter

        item = TaskItem(
            priority=priority,
            seq=seq,
            task_id=task_id,
            metadata=metadata or {},
        )

        if self._backend == "memory":
            await self._enqueue_memory(item)
        else:
            await self._enqueue_redis(item)

        logger.debug(
            "Enqueued task '%s' (priority=%d, seq=%d)",
            task_id,
            priority,
            seq,
        )
        return item

    async def dequeue(self, timeout: Optional[float] = None) -> TaskItem:
        """
        Get the next task from the queue (highest priority first).

        Blocks until a task is available or timeout expires.

        Args:
            timeout: Maximum seconds to wait. None = block forever.

        Returns:
            The next TaskItem.

        Raises:
            asyncio.TimeoutError: If timeout expires with no item.
            RuntimeError: If the backend is not started.
        """
        if not self._started:
            await self.start()

        if self._backend == "memory":
            return await self._dequeue_memory(timeout)
        return await self._dequeue_redis(timeout)

    async def size(self) -> int:
        """
        Return the current number of items in the queue.

        Returns:
            Queue depth (int).
        """
        if self._backend == "memory":
            return self._pq.qsize()
        if self._redis is not None:
            return await self._redis.zcard(self._redis_key)
        return 0

    async def clear(self) -> int:
        """
        Remove all pending tasks from the queue.

        Returns:
            Number of items removed.
        """
        if self._backend == "memory":
            count = self._pq.qsize()
            while not self._pq.empty():
                try:
                    self._pq.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.info("Cleared %d items from memory queue", count)
            return count
        if self._redis is not None:
            count = await self._redis.zcard(self._redis_key)
            await self._redis.delete(self._redis_key)
            logger.info("Cleared %d items from Redis queue", count)
            return count
        return 0

    @property
    def backend(self) -> str:
        """Return the backend type ('memory' or 'redis')."""
        return self._backend

    @property
    def is_started(self) -> bool:
        """Return True if the queue backend has been initialized."""
        return self._started

    # ------------------------------------------------------------------
    # Memory backend implementation
    # ------------------------------------------------------------------

    async def _enqueue_memory(self, item: TaskItem) -> None:
        """Enqueue to the in-memory PriorityQueue."""
        if self._max_size is not None:
            current = self._pq.qsize()
            if current >= self._max_size:
                raise QueueFullError(
                    f"Task queue is full (max_size={self._max_size})"
                )
        await self._pq.put(item)

    async def _dequeue_memory(
        self, timeout: Optional[float]
    ) -> TaskItem:
        """Dequeue from the in-memory PriorityQueue."""
        if timeout is not None:
            try:
                return await asyncio.wait_for(
                    self._pq.get(), timeout=timeout
                )
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"Dequeue timed out after {timeout}s"
                )
        return await self._pq.get()

    # ------------------------------------------------------------------
    # Redis backend implementation
    # ------------------------------------------------------------------

    async def _enqueue_redis(self, item: TaskItem) -> None:
        """
        Enqueue to Redis using a sorted set.

        The member is a JSON-serialized TaskItem; the score is computed
        from priority and sequence to preserve FIFO within the same
        priority: score = priority * 1e10 + seq.
        """
        import json

        if self._max_size is not None:
            current = await self._redis.zcard(self._redis_key)
            if current >= self._max_size:
                raise QueueFullError(
                    f"Task queue is full (max_size={self._max_size})"
                )

        score = item.priority * 10_000_000_000 + item.seq
        member = json.dumps({
            "task_id": item.task_id,
            "priority": item.priority,
            "seq": item.seq,
            "created_at": item.created_at,
            "metadata": item.metadata,
        })
        await self._redis.zadd(self._redis_key, {member: score})

    async def _dequeue_redis(
        self, timeout: Optional[float]
    ) -> TaskItem:
        """
        Dequeue from Redis using ZPOPMIN (atomic, blocking).

        Uses BZPOPMIN if timeout is specified, otherwise ZPOPMIN
        with a retry loop.
        """
        import json

        if timeout is not None and timeout > 0:
            # BZPOPMIN returns (key, member, score) tuple or None on timeout
            result = await self._redis.bzpopmin(
                self._redis_key, timeout=int(timeout)
            )
            if result is None:
                raise asyncio.TimeoutError(
                    f"Dequeue timed out after {timeout}s"
                )
            # result is (key, member, score)
            _, member, _ = result
            data = json.loads(member)
        else:
            # Non-blocking: ZPOPMIN
            result = await self._redis.zpopmin(self._redis_key, count=1)
            if not result:
                # Poll until something is available
                poll_interval = 0.1
                while True:
                    await asyncio.sleep(poll_interval)
                    result = await self._redis.zpopmin(
                        self._redis_key, count=1
                    )
                    if result:
                        break
            member, _ = result[0]
            data = json.loads(member)

        return TaskItem(
            priority=data["priority"],
            seq=data["seq"],
            task_id=data["task_id"],
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


class QueueFullError(RuntimeError):
    """Raised when the task queue has reached its max_size limit."""

    pass
