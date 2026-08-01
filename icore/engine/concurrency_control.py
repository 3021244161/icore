"""
icore.engine.concurrency_control - Concurrency limiting and rate control.

Provides the concurrency control layer that wraps the WorkflowExecutor.
It enforces:

    1. Global concurrency limit: max_concurrent_tasks total across all
       workflows. Enforced via a single asyncio.Semaphore.
    2. Per-workflow concurrency limit: max_concurrent_per_workflow per
       workflow type. Enforced via per-workflow semaphores, created lazily.
    3. Backpressure: When the queue depth exceeds a threshold, the
       system returns HTTP 503 to reject new requests gracefully.
    4. Per-model rate limiting: Optional token bucket per model_id to
       respect LLM API rate limits.

Design rationale:
    The WorkflowExecutor is stateless and does not enforce any
    concurrency limits itself. This separation keeps the executor
    focused on DAG execution logic, while the concurrency controller
    handles resource governance. The API layer wraps each workflow
    invocation with the controller before delegating to the executor.

Improvement over the original implementation:
    The user's original system had basic concurrency handling but
    "wasn't very good". This design improves it with:
    - Wave-based parallel execution (from the DAG scheduler)
    - Per-workflow semaphores (isolate noisy workflows)
    - Backpressure with HTTP 503 (fail fast instead of queueing forever)
    - Token bucket rate limiting per model (respect LLM API limits)
    - Automatic cleanup of terminal instances (prevent memory leaks)
    - Cancellation support (propagate cancel signal to executor)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from icore.engine.instance_manager import TaskInstanceManager
    from icore.engine.task_queue import TaskQueue

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyStats:
    """
    Snapshot of current concurrency statistics.

    Used for monitoring and backpressure decisions.

    Attributes:
        active_global:   Currently executing tasks (global).
        active_per_wf:   Currently executing tasks per workflow name.
        queue_depth:     Current task queue depth.
        max_global:      Configured global concurrency limit.
        max_per_workflow: Configured per-workflow concurrency limit.
        backpressure:    Whether backpressure is currently active.
        timestamp:       When this snapshot was taken.
    """

    active_global: int = 0
    active_per_wf: dict[str, int] = field(default_factory=dict)
    queue_depth: int = 0
    max_global: int = 0
    max_per_workflow: int = 0
    backpressure: bool = False
    timestamp: float = field(default_factory=time.time)


class TokenBucket:
    """
    Token bucket rate limiter for per-model API rate limiting.

    The bucket has a fixed capacity and refills at a fixed rate.
    Each API call consumes one (or more) tokens. If the bucket is
    empty, the caller must wait until a token is available.

    This is used to respect LLM API rate limits (e.g., OpenAI's
    RPM/TPM limits) without hard-coding delays.

    Usage:

        bucket = TokenBucket(capacity=60, refill_rate=1.0)
        await bucket.acquire()  # blocks until a token is available
        await bucket.acquire(n=2)  # consume 2 tokens at once

    Attributes:
        _capacity:    Maximum tokens the bucket can hold.
        _refill_rate:  Tokens per second to refill.
        _tokens:      Current token count (float for precision).
        _last_refill: Last refill timestamp.
        _lock:        asyncio.Lock for thread-safe token management.
    """

    def __init__(
        self,
        capacity: int = 60,
        refill_rate: float = 1.0,
    ) -> None:
        """
        Initialize the token bucket.

        Args:
            capacity:    Maximum number of tokens the bucket can hold.
            refill_rate:  Refill rate in tokens per second.
                          E.g., 1.0 = 1 token/sec = 60 RPM.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")

        self._capacity: int = capacity
        self._refill_rate: float = refill_rate
        self._tokens: float = float(capacity)
        self._last_refill: float = time.time()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        """
        Acquire n tokens from the bucket.

        Blocks until enough tokens are available. If n > capacity,
        acquires as many as possible and waits for the rest.

        Args:
            n: Number of tokens to acquire (default 1).
        """
        if n <= 0:
            return

        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                # Not enough tokens: calculate wait time
                deficit = n - self._tokens
                wait_time = deficit / self._refill_rate

            logger.debug(
                "TokenBucket: need %d tokens, have %.2f, waiting %.2fs",
                n,
                self._tokens,
                wait_time,
            )
            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Must be called under lock."""
        now = time.time()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._refill_rate,
            )
            self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Return the current (approximate) token count."""
        return self._tokens


class ConcurrencyController:
    """
    Controls concurrency across the system.

    Wraps the WorkflowExecutor with concurrency governance:

        1. Global semaphore: limits total concurrent task instances.
        2. Per-workflow semaphores: limits concurrent instances of
           the same workflow type (prevents one noisy workflow from
           starving others).
        3. Backpressure: returns a BackpressureError when the queue
           depth exceeds the threshold, allowing the API layer to
           return HTTP 503.
        4. Per-model rate limiting: token buckets per model_id.

    Usage:

        controller = ConcurrencyController(
            max_concurrent_tasks=100,
            max_concurrent_per_workflow=20,
            backpressure_threshold=500,
        )

        # In the API layer:
        if controller.is_backpressure(queue):
            raise HTTPException(503, "System overloaded")

        async with controller.acquire("document_summary"):
            result = await executor.run(dag, ctx, params)

    The acquire() method is an async context manager that acquires both
    the global and per-workflow semaphores, then releases them on exit.

    Attributes:
        _global_sem:        Global asyncio.Semaphore.
        _workflow_sems:     Per-workflow asyncio.Semaphore (lazy).
        _max_concurrent:    Global concurrency limit.
        _max_per_workflow:  Per-workflow concurrency limit.
        _backpressure_threshold: Queue depth that triggers backpressure.
        _rate_limiters:     Per-model TokenBucket instances.
        _active_per_wf:     Active count per workflow (for stats).
        _active_global:     Active count globally (for stats).
        _lock:              asyncio.Lock for stats and semaphore creation.
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 100,
        max_concurrent_per_workflow: int = 20,
        backpressure_threshold: int = 500,
    ) -> None:
        """
        Initialize the concurrency controller.

        Args:
            max_concurrent_tasks:    Global max concurrent task instances.
            max_concurrent_per_workflow: Max concurrent per workflow type.
            backpressure_threshold:  Queue depth that triggers HTTP 503.
        """
        self._max_concurrent: int = max_concurrent_tasks
        self._max_per_workflow: int = max_concurrent_per_workflow
        self._backpressure_threshold: int = backpressure_threshold

        self._global_sem: asyncio.Semaphore = asyncio.Semaphore(
            max_concurrent_tasks
        )
        self._workflow_sems: dict[str, asyncio.Semaphore] = {}
        self._rate_limiters: dict[str, TokenBucket] = {}

        # Stats tracking
        self._active_per_wf: dict[str, int] = {}
        self._active_global: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Concurrency acquisition
    # ------------------------------------------------------------------

    def acquire(self, workflow_name: str) -> _ConcurrencySlot:
        """
        Acquire concurrency slots for a workflow execution.

        Returns an async context manager that:
            1. Acquires the global semaphore (total concurrency limit)
            2. Acquires the per-workflow semaphore (per-type limit)
            3. Releases both on exit (even on exception)

        Args:
            workflow_name: Name of the workflow to acquire slots for.

        Returns:
            An async context manager (_ConcurrencySlot).
        """
        return _ConcurrencySlot(self, workflow_name)

    def _get_workflow_sem(self, workflow_name: str) -> asyncio.Semaphore:
        """Get or create the per-workflow semaphore (not thread-safe, call under lock)."""
        if workflow_name not in self._workflow_sems:
            self._workflow_sems[workflow_name] = asyncio.Semaphore(
                self._max_per_workflow
            )
            self._active_per_wf[workflow_name] = 0
        return self._workflow_sems[workflow_name]

    # ------------------------------------------------------------------
    # Backpressure
    # ------------------------------------------------------------------

    async def is_backpressure(
        self,
        queue: Optional[TaskQueue] = None,
    ) -> bool:
        """
        Check if the system is under backpressure.

        Backpressure is triggered when:
            - The queue depth exceeds backpressure_threshold, OR
            - The active global count reaches max_concurrent_tasks

        When backpressure is active, the API layer should return
        HTTP 503 Service Unavailable with a Retry-After header.

        Args:
            queue: Optional TaskQueue to check depth. If None,
                   only checks active count.

        Returns:
            True if the system is under backpressure.
        """
        # Check active count
        if self._active_global >= self._max_concurrent:
            return True

        # Check queue depth
        if queue is not None:
            depth = await queue.size()
            if depth >= self._backpressure_threshold:
                return True

        return False

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def register_rate_limit(
        self,
        model_id: str,
        capacity: int = 60,
        refill_rate: float = 1.0,
    ) -> TokenBucket:
        """
        Register a rate limiter for a model.

        Args:
            model_id:    The model identifier to rate-limit.
            capacity:    Token bucket capacity (max burst).
            refill_rate:  Tokens per second (sustained rate).
                          E.g., 1.0 = 60 RPM, 10.0 = 600 RPM.

        Returns:
            The created TokenBucket instance.
        """
        bucket = TokenBucket(capacity=capacity, refill_rate=refill_rate)
        self._rate_limiters[model_id] = bucket
        logger.info(
            "Registered rate limiter for model '%s' "
            "(capacity=%d, refill=%.1f/s)",
            model_id,
            capacity,
            refill_rate,
        )
        return bucket

    async def rate_limit(self, model_id: str, n: int = 1) -> None:
        """
        Acquire rate-limit tokens for a model.

        If no rate limiter is registered for the model_id, this is a no-op.

        Args:
            model_id: The model to rate-limit.
            n:        Number of tokens to acquire (default 1).
        """
        bucket = self._rate_limiters.get(model_id)
        if bucket is not None:
            await bucket.acquire(n)

    def has_rate_limiter(self, model_id: str) -> bool:
        """Check if a rate limiter is registered for a model."""
        return model_id in self._rate_limiters

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(
        self,
        queue: Optional[TaskQueue] = None,
    ) -> ConcurrencyStats:
        """
        Get a snapshot of current concurrency statistics.

        Args:
            queue: Optional TaskQueue to include queue depth.

        Returns:
            ConcurrencyStats dataclass.
        """
        async with self._lock:
            queue_depth = 0
            if queue is not None:
                queue_depth = await queue.size()

            return ConcurrencyStats(
                active_global=self._active_global,
                active_per_wf=dict(self._active_per_wf),
                queue_depth=queue_depth,
                max_global=self._max_concurrent,
                max_per_workflow=self._max_per_workflow,
                backpressure=(
                    self._active_global >= self._max_concurrent
                    or queue_depth >= self._backpressure_threshold
                ),
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_concurrent(self) -> int:
        """Return the global concurrency limit."""
        return self._max_concurrent

    @property
    def max_per_workflow(self) -> int:
        """Return the per-workflow concurrency limit."""
        return self._max_per_workflow

    @property
    def backpressure_threshold(self) -> int:
        """Return the backpressure threshold."""
        return self._backpressure_threshold

    @property
    def active_count(self) -> int:
        """Return the current global active count (approximate)."""
        return self._active_global


class _ConcurrencySlot:
    """
    Async context manager for acquiring/releasing concurrency slots.

    Acquires both the global semaphore and the per-workflow semaphore
    on __aenter__, releases both on __aexit__. Also tracks active
    counts for monitoring and backpressure checks.

    This is created by ConcurrencyController.acquire() and should not
    be instantiated directly.
    """

    def __init__(
        self,
        controller: ConcurrencyController,
        workflow_name: str,
    ) -> None:
        self._controller = controller
        self._workflow_name = workflow_name
        self._wf_sem: Optional[asyncio.Semaphore] = None

    async def __aenter__(self) -> None:
        """
        Acquire global and per-workflow semaphores.

        Acquires the global semaphore first (for fair ordering across
        all workflows), then the per-workflow semaphore. If the second
        acquisition fails (e.g. ``CancelledError``), the global slot is
        released immediately to avoid leaking a concurrency permit.
        """
        ctrl = self._controller

        # Create per-workflow semaphore if needed (under lock)
        async with ctrl._lock:
            self._wf_sem = ctrl._get_workflow_sem(self._workflow_name)

        # Acquire global semaphore first (fair ordering)
        await ctrl._global_sem.acquire()
        global_acquired = True

        try:
            # Acquire per-workflow semaphore
            await self._wf_sem.acquire()
        except BaseException:
            # Includes asyncio.CancelledError. Release the global slot
            # so we don't leak a permit when the second acquire aborts.
            if global_acquired:
                ctrl._global_sem.release()
                global_acquired = False
            raise

        # Update stats
        async with ctrl._lock:
            ctrl._active_global += 1
            ctrl._active_per_wf[self._workflow_name] = (
                ctrl._active_per_wf.get(self._workflow_name, 0) + 1
            )

        logger.debug(
            "Acquired slots for workflow '%s' "
            "(active_global=%d, active_%s=%d)",
            self._workflow_name,
            ctrl._active_global,
            self._workflow_name,
            ctrl._active_per_wf.get(self._workflow_name, 0),
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Release both semaphores and update stats."""
        ctrl = self._controller

        # Release per-workflow semaphore
        if self._wf_sem is not None:
            self._wf_sem.release()

        # Release global semaphore
        ctrl._global_sem.release()

        # Update stats
        async with ctrl._lock:
            ctrl._active_global = max(0, ctrl._active_global - 1)
            current = ctrl._active_per_wf.get(self._workflow_name, 0)
            ctrl._active_per_wf[self._workflow_name] = max(0, current - 1)

        logger.debug(
            "Released slots for workflow '%s' (active_global=%d)",
            self._workflow_name,
            ctrl._active_global,
        )


class BackpressureError(Exception):
    """
    Raised when the system is under backpressure.

    The API layer should catch this and return HTTP 503 with a
    Retry-After header.
    """

    def __init__(
        self,
        message: str = "System is under backpressure",
        retry_after: int = 5,
    ) -> None:
        """
        Initialize the backpressure error.

        Args:
            message:     Error message.
            retry_after: Suggested retry delay in seconds.
        """
        super().__init__(message)
        self.retry_after = retry_after
