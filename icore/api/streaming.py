"""
icore.api.streaming - SSE (Server-Sent Events) streaming response handler.

SSEStreamHandler converts an async generator of workflow execution events
into a FastAPI-compatible StreamingResponse with SSE formatting.

SSE format:
    - Data events:     ``data: {json}\\n\\n``
    - Error events:    ``event: error\\ndata: {json}\\n\\n``
    - End marker:      ``data: [DONE]\\n\\n``

The handler wraps the workflow execution in an async generator that:
    1. Yields progress events as SSE data events
    2. Catches exceptions and sends an error event before closing
    3. Always sends [DONE] as the final event

Usage in the API layer::

    handler = SSEStreamHandler()
    return StreamingResponse(
        handler.stream(workflow, ctx, params),
        media_type="text/event-stream",
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional

from fastapi.responses import StreamingResponse

from icore.core.models import BaseTaskOutput
from icore.core.task_context import TaskContext

logger = logging.getLogger(__name__)

# Sentinel for end-of-stream marker
_DONE_MARKER = "data: [DONE]\n\n"


class SSEStreamHandler:
    """
    SSE streaming handler for workflow execution results.

    This handler wraps a workflow execution and produces an SSE-formatted
    async generator suitable for FastAPI's StreamingResponse.

    The handler supports two streaming modes:

    1. **Progress streaming**: Yields intermediate events as the
       workflow DAG executes (node started, node completed, etc.).
       The final result is yielded as the last data event before [DONE].

    2. **Token streaming**: If the workflow's tasks support streaming
       (via model_adapter.stream_chat()), intermediate tokens can be
       forwarded as SSE events. This is controlled by the stream flag
       in TaskContext.

    Attributes:
        _buffer_size:  Max events buffered before flushing.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        """
        Initialize the SSE handler.

        Args:
            buffer_size: Max events to buffer internally (for flow control).
        """
        self._buffer_size = buffer_size

    def create_response(
        self,
        workflow: Any,
        ctx: TaskContext,
        params: dict[str, Any],
        callback_manager: Optional[Any] = None,
        instance_manager: Optional[Any] = None,
    ) -> StreamingResponse:
        """
        Create a FastAPI StreamingResponse for SSE streaming.

        Args:
            workflow:         The BaseWorkflow instance to execute.
            ctx:              TaskContext with execution context.
            params:           Workflow input parameters.
            callback_manager: Optional CallbackManager for result delivery.
            instance_manager: Optional TaskInstanceManager for lifecycle
                              tracking (PENDING -> RUNNING -> COMPLETED|FAILED).

        Returns:
            A StreamingResponse with media_type="text/event-stream".
        """
        return StreamingResponse(
            self.stream(
                workflow, ctx, params, callback_manager, instance_manager
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def stream(
        self,
        workflow: Any,
        ctx: TaskContext,
        params: dict[str, Any],
        callback_manager: Optional[Any] = None,
        instance_manager: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate SSE-formatted events from workflow execution.

        This is the core async generator that produces SSE events:

        1. Yield a "started" event with task_id
        2. Execute the workflow
        3. Yield intermediate events (if the workflow provides them)
        4. Yield the final result as a data event
        5. If callback_url is set, schedule async delivery
        6. Yield [DONE] to signal end of stream

        On error:
        - Send an error event with the error message
        - Still send [DONE] to close the stream

        Args:
            workflow:         The BaseWorkflow instance.
            ctx:              TaskContext for this execution.
            params:           Workflow parameters.
            callback_manager: Optional CallbackManager.
            instance_manager: Optional TaskInstanceManager. When provided,
                              the instance is advanced RUNNING -> terminal.

        Yields:
            SSE-formatted strings (``data: {json}\\n\\n``).
        """
        # 1. Send start event
        start_event = {
            "task_id": ctx.task_id,
            "workflow_id": ctx.workflow_id,
            "status": "running",
            "timestamp": self._timestamp(),
        }
        yield self._format_data(start_event)

        # Advance instance state to RUNNING if tracked
        if instance_manager is not None:
            from icore.engine.states import TaskState
            await instance_manager.update_state(ctx.task_id, TaskState.RUNNING)

        # 2. Execute workflow with error handling
        try:
            result: BaseTaskOutput = await workflow.execute(ctx, params)

            # Advance instance state to terminal if tracked
            if instance_manager is not None:
                from icore.engine.states import TaskState
                new_state = (
                    TaskState.COMPLETED
                    if result.is_success
                    else TaskState.FAILED
                )
                await instance_manager.update_state(
                    ctx.task_id, new_state, result=result,
                )

            # 3. Send final result event
            result_event: dict[str, Any] = {
                "task_id": ctx.task_id,
                "status": result.status,
                "data": result.data,
            }
            if result.error:
                result_event["error"] = result.error
            yield self._format_data(result_event)

            # 4. If callback is configured, deliver result async
            if ctx.callback_url and callback_manager is not None:
                callback_manager.deliver(
                    ctx.callback_url,
                    {
                        "task_id": ctx.task_id,
                        "status": result.status,
                        "result": result.data,
                        "error": result.error,
                    },
                )

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled for task %s", ctx.task_id)
            if instance_manager is not None:
                from icore.engine.states import TaskState
                await instance_manager.update_state(
                    ctx.task_id, TaskState.CANCELLED,
                )
            yield self._format_error(
                "Stream cancelled",
                {"task_id": ctx.task_id},
            )

        except Exception as e:
            logger.error(
                "Workflow execution error in SSE stream: %s", e, exc_info=True
            )
            if instance_manager is not None:
                from icore.engine.states import TaskState
                await instance_manager.update_state(
                    ctx.task_id, TaskState.FAILED,
                )
            error_event = {
                "task_id": ctx.task_id,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            yield self._format_error(
                f"Workflow execution failed: {e}",
                error_event,
            )

        # 5. Always send [DONE] to close the stream
        yield _DONE_MARKER

    # ------------------------------------------------------------------
    # SSE formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_data(data: Any) -> str:
        """
        Format a data event in SSE format.

        Args:
            data: Data to send (will be JSON-serialized).

        Returns:
            SSE-formatted string: ``data: {json}\\n\\n``
        """
        return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    @staticmethod
    def _format_error(message: str, data: Optional[dict] = None) -> str:
        """
        Format an error event in SSE format.

        Error events use the SSE ``event`` field to signal error type:

            event: error
            data: {json}

        Args:
            message: Error message.
            data:    Optional additional data dict.

        Returns:
            SSE-formatted error event string.
        """
        payload = {"error": message}
        if data:
            payload.update(data)
        return (
            f"event: error\n"
            f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
        )

    @staticmethod
    def _timestamp() -> str:
        """Get current ISO 8601 timestamp."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        return f"SSEStreamHandler(buffer_size={self._buffer_size})"
