"""
icore.api.callback - Async callback result delivery manager.

CallbackManager is responsible for delivering workflow results to
external systems via HTTP POST. It is used when the caller provides
a ``callback_url`` in the InvokeRequest - in that case, the API
returns immediately with status="running" and the result is
delivered asynchronously when the workflow completes.

Key characteristics:
    - Non-blocking: Uses asyncio.create_task() so the caller is not
      blocked waiting for the callback to complete.
    - Error-resilient: Failures are logged but never crash the main
      workflow execution.
    - Configurable timeout: Uses the API request_timeout from settings.
    - Retries: Optional retry with backoff for transient failures.

Usage::

    manager = CallbackManager()
    manager.deliver(
        url="https://example.com/callback",
        payload={"task_id": "abc", "status": "success", "result": {...}},
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class CallbackManager:
    """
    Manages async delivery of workflow results to callback URLs.

    The manager uses httpx.AsyncClient for non-blocking HTTP calls.
    Delivery is fire-and-forget: the caller schedules the delivery
    and continues immediately. Errors are logged but do not propagate
    to the caller.

    Thread safety:
        Individual deliveries are async tasks. The manager itself is
        safe to use from multiple coroutines concurrently.

    Attributes:
        _timeout:     Timeout in seconds for each callback request.
        _max_retries: Maximum number of retry attempts on failure.
        _retry_delay: Base delay (seconds) for exponential backoff.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize the CallbackManager.

        Args:
            timeout:     Timeout in seconds for each HTTP POST request.
            max_retries: Max retry attempts on transient failures.
            retry_delay: Base delay for exponential backoff (seconds).
        """
        self._timeout: int = timeout
        self._max_retries: int = max_retries
        self._retry_delay: float = retry_delay

    def deliver(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> asyncio.Task[None]:
        """
        Schedule async delivery of a payload to a callback URL.

        This method is non-blocking: it creates an asyncio task and
        returns immediately. The task handles retries and error logging.

        Args:
            url:     The callback URL to POST to.
            payload: The JSON body to send (typically an InvokeResponse dict).

        Returns:
            The asyncio.Task for the delivery (can be awaited if needed,
            but typically fire-and-forget).
        """
        task = asyncio.create_task(
            self._deliver_with_retry(url, payload)
        )
        return task

    async def deliver_and_wait(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> bool:
        """
        Deliver a payload and wait for completion.

        Unlike ``deliver()``, this method blocks until the delivery
        completes (or exhausts retries). Returns the success status.

        Args:
            url:     The callback URL to POST to.
            payload: The JSON body to send.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        return await self._deliver_with_retry(url, payload)

    async def _deliver_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> bool:
        """
        Attempt delivery with exponential backoff retry.

        Retries on connection errors and 5xx responses. Does NOT retry
        on 4xx responses (client errors are not transient).

        Args:
            url:     The callback URL.
            payload: The JSON body.

        Returns:
            True if delivery succeeded, False if all retries exhausted.
        """
        last_error: Optional[str] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout
                ) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )

                    # Success: 2xx status
                    if response.status_code < 300:
                        logger.info(
                            "Callback delivered to %s (attempt %d, "
                            "status=%d)",
                            url,
                            attempt,
                            response.status_code,
                        )
                        return True

                    # Client error (4xx): don't retry
                    if 400 <= response.status_code < 500:
                        logger.error(
                            "Callback to %s failed with client error "
                            "%d (not retrying): %s",
                            url,
                            response.status_code,
                            response.text[:200],
                        )
                        return False

                    # Server error (5xx): retry
                    last_error = (
                        f"HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    logger.warning(
                        "Callback to %s got %d (attempt %d/%d), "
                        "retrying...",
                        url,
                        response.status_code,
                        attempt,
                        self._max_retries,
                    )

            except httpx.TimeoutException:
                last_error = f"Timeout after {self._timeout}s"
                logger.warning(
                    "Callback to %s timed out (attempt %d/%d)",
                    url,
                    attempt,
                    self._max_retries,
                )
            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                logger.warning(
                    "Callback to %s connection failed (attempt %d/%d): %s",
                    url,
                    attempt,
                    self._max_retries,
                    e,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.error(
                    "Callback to %s failed (attempt %d/%d): %s",
                    url,
                    attempt,
                    self._max_retries,
                    e,
                    exc_info=True,
                )

            # Exponential backoff before next retry
            if attempt < self._max_retries:
                delay = self._retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        logger.error(
            "Callback to %s failed after %d attempts: %s",
            url,
            self._max_retries,
            last_error,
        )
        return False

    def __repr__(self) -> str:
        return (
            f"CallbackManager(timeout={self._timeout}, "
            f"max_retries={self._max_retries})"
        )
