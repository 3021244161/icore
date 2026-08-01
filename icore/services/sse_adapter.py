"""
icore.services.sse_adapter - SSE (Server-Sent Events) streaming exposer.

Exposes workflows as SSE streaming endpoints. This adapter wraps
workflow execution results into Server-Sent Events (SSE) format,
enabling clients to receive real-time streaming updates.

Use this when you want to:
    - Stream LLM token-by-token output to web clients
    - Provide real-time progress updates during long-running workflows
    - Integrate with EventSource-compatible frontend frameworks
    - Expose workflows via SSE without FastAPI's built-in StreamingResponse

Design:
    The SSEExposer produces SSE-formatted output:
        event: progress
        data: {"step": 1, "message": "Processing..."}

        event: result
        data: {"status": "success", "summary": "..."}

        event: done
        data: {}

    Each workflow execution generates a sequence of SSE events:
    1. "start" event on execution begin
    2. "progress" events during multi-step execution
    3. "result" event with final output
    4. "done" event signaling stream end
    5. "error" event if execution fails

Usage:
    exposer = SSEExposer()
    exposer.expose(DocumentSummaryWorkflow)

    async for event in exposer.stream("document_summary", params, ctx):
        print(event)

    # Or integrate with FastAPI:
    @app.get("/stream/{workflow_name}")
    async def stream_workflow(workflow_name: str, ...):
        exposer = get_sse_exposer()
        return StreamingResponse(
            exposer.stream_events(workflow_name, params, ctx),
            media_type="text/event-stream",
        )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator

from icore.core.task_context import TaskContext
from icore.services.base_exposer import BaseServiceExposer, ServiceDef

if TYPE_CHECKING:
    from icore.engine.base_workflow import BaseWorkflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE Event Types
# ---------------------------------------------------------------------------

@dataclass
class SSEEvent:
    """
    A single Server-Sent Event.

    Attributes:
        event:   Event type name (e.g., "start", "progress", "result", "done", "error").
        data:    Event payload as a JSON string.
        id:      Optional event ID for EventSource last-event-id tracking.
        retry:   Optional retry interval in milliseconds.
    """

    event: str
    data: str
    id: str | None = None
    retry: int | None = None

    def to_sse_string(self) -> str:
        """
        Format this event as an SSE-compliant string.

        The format follows the SSE specification:
            event: <type>\\n
            data: <json>\\n
            [id: <id>\\n]
            \\n

        Returns:
            SSE-formatted string ready to be sent over HTTP.
        """
        lines: list[str] = []
        lines.append(f"event: {self.event}")
        lines.append(f"data: {self.data}")
        if self.id is not None:
            lines.append(f"id: {self.id}")
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        lines.append("")  # Empty line signals end of event
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SSE Exposer
# ---------------------------------------------------------------------------

class SSEExposer(BaseServiceExposer):
    """
    Exposes workflows as SSE (Server-Sent Events) streaming endpoints.

    Converts workflow execution into a sequence of SSE events that
    clients can consume via EventSource. Supports both synchronous
    result delivery and streaming progress updates for long-running
    or multi-step workflows.

    Event types emitted:
        - "start":     Emitted when execution begins.
        - "progress":  Emitted for each intermediate step (if supported).
        - "result":    Emitted with the final workflow output.
        - "done":      Emitted after result, signaling stream end.
        - "error":     Emitted if execution fails.
    """

    name: str = "sse"
    protocol: str = "sse"

    def __init__(
        self,
        model_manager: Any = None,
        db_manager: Any = None,
        heartbeat_interval: float = 15.0,
    ) -> None:
        """
        Initialise the SSE exposer.

        Args:
            model_manager:     Optional ModelManager for dependency injection.
            db_manager:        Optional DBManager for dependency injection.
            heartbeat_interval: Seconds between heartbeat keep-alive events.
        """
        self._services: dict[str, ServiceDef] = {}
        self._workflow_classes: dict[str, type[BaseWorkflow]] = {}
        self._model_manager = model_manager
        self._db_manager = db_manager
        self._heartbeat_interval = heartbeat_interval

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_model_manager(self, model_manager: Any) -> None:
        """Set the ModelManager for dependency injection."""
        self._model_manager = model_manager

    def set_db_manager(self, db_manager: Any) -> None:
        """Set the DBManager for dependency injection."""
        self._db_manager = db_manager

    # ------------------------------------------------------------------
    # Expose
    # ------------------------------------------------------------------

    async def expose(
        self, workflow_cls: type[BaseWorkflow]
    ) -> ServiceDef:
        """
        Expose a workflow for SSE streaming.

        Args:
            workflow_cls: The BaseWorkflow subclass to expose.

        Returns:
            A ServiceDef describing the SSE endpoint.
        """
        name = workflow_cls.name or workflow_cls.__name__.lower()
        service = ServiceDef(
            name=name,
            description=workflow_cls.description or "",
            workflow_name=workflow_cls.name,
            input_schema={},
            output_schema={
                "type": "text/event-stream",
                "description": "SSE stream of workflow events",
            },
            protocol="sse",
            metadata={
                "source": "icore-sse-exposer",
                "heartbeat_interval": self._heartbeat_interval,
            },
        )

        self._services[name] = service
        self._workflow_classes[name] = workflow_cls

        logger.info(
            "Exposed workflow '%s' as SSE endpoint", name
        )
        return service

    async def list_services(self) -> list[ServiceDef]:
        """List all SSE-exposed workflows."""
        return list(self._services.values())

    async def call_service(
        self, name: str, params: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        """
        Execute a workflow via SSE (non-streaming fallback).

        Args:
            name:   Workflow name.
            params: Parameters dict.
            ctx:    TaskContext with injected dependencies.

        Returns:
            Dict with "status", "data", and optionally "error" keys.
        """
        if name not in self._workflow_classes:
            raise KeyError(
                f"SSE endpoint '{name}' not found. "
                f"Available: {list(self._workflow_classes)}"
            )

        wf_cls = self._workflow_classes[name]

        if self._model_manager is not None:
            try:
                _ = ctx.get_model_adapter()
            except RuntimeError:
                ctx.set_model_manager(self._model_manager)

        if self._db_manager is not None:
            try:
                _ = ctx.get_db("default")
            except RuntimeError:
                ctx.set_db_manager(self._db_manager)

        wf = wf_cls()
        try:
            output = await wf.execute(ctx, params)
            return {
                "status": output.status,
                "data": output.data,
                "error": output.error if output.status == "error" else None,
            }
        except Exception as exc:
            logger.exception(
                "SSE workflow '%s' execution failed", name
            )
            return {
                "status": "error",
                "data": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # SSE Streaming
    # ------------------------------------------------------------------

    async def stream_events(
        self,
        workflow_name: str,
        params: dict[str, Any],
        ctx: TaskContext,
    ) -> AsyncIterator[str]:
        """
        Stream workflow execution as SSE events.

        This is the main streaming method. It yields SSE-formatted
        event strings that can be consumed by a FastAPI
        StreamingResponse or any async SSE consumer.

        The stream includes:
            1. "start" event
            2. Heartbeat events during execution
            3. "progress" events (if the workflow supports them)
            4. "result" event
            5. "done" event

        On failure, the stream emits an "error" event and closes.

        Args:
            workflow_name: Name of the workflow to execute.
            params:        Parameters for workflow execution.
            ctx:           TaskContext with injected dependencies.

        Yields:
            SSE-formatted event strings.
        """
        event_id = 0

        # Inject managers if needed
        if self._model_manager is not None:
            try:
                _ = ctx.get_model_adapter()
            except RuntimeError:
                ctx.set_model_manager(self._model_manager)

        if self._db_manager is not None:
            try:
                _ = ctx.get_db("default")
            except RuntimeError:
                ctx.set_db_manager(self._db_manager)

        wf_cls = self._workflow_classes.get(workflow_name)
        if wf_cls is None:
            yield self._error_event(
                event_id, f"Workflow '{workflow_name}' not found"
            )
            return

        try:
            # Emit "start" event
            yield self._event_to_sse(
                SSEEvent(
                    event="start",
                    data=json.dumps({
                        "workflow": workflow_name,
                        "task_id": ctx.task_id,
                        "timestamp": time.time(),
                    }),
                    id=str(event_id),
                )
            )
            event_id += 1

            # Emit initial heartbeat
            yield self._heartbeat_event(event_id)
            event_id += 1

            # Execute the workflow
            wf = wf_cls()

            if ctx.stream:
                # Streaming mode: the workflow yields chunks
                result = await wf.execute(ctx, params)
                if hasattr(result, "__aiter__"):
                    async for chunk in result:
                        progress = self._progress_event(
                            event_id, "stream_chunk", chunk.data
                        )
                        yield self._event_to_sse(progress)
                        event_id += 1
                output = (
                    result
                    if not hasattr(result, "__aiter__")
                    else None
                )
            else:
                # Non-streaming: execute and return result
                output = await wf.execute(ctx, params)

            # Emit "result" event
            if output is not None:
                result_data = json.dumps({
                    "status": output.status,
                    "data": output.data,
                    "error": output.error,
                })
                yield self._event_to_sse(
                    SSEEvent(
                        event="result",
                        data=result_data,
                        id=str(event_id),
                    )
                )
                event_id += 1

        except Exception as exc:
            logger.exception(
                "SSE stream for '%s' failed", workflow_name
            )
            yield self._error_event(event_id, str(exc))
            return

        # Emit "done" event
        yield self._event_to_sse(
            SSEEvent(
                event="done",
                data=json.dumps({"completed": True}),
                id=str(event_id),
            )
        )

    async def stream(
        self,
        workflow_name: str,
        params: dict[str, Any],
        ctx: TaskContext,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream workflow execution as Python dicts (non-SSE format).

        Convenience method for programmatic consumers. Yields dicts
        with "event" and "data" keys instead of SSE strings.

        Args:
            workflow_name: Name of the workflow to execute.
            params:        Parameters for workflow execution.
            ctx:           TaskContext with injected dependencies.

        Yields:
            Dicts with "event" (str) and "data" (dict) keys.
        """
        async for sse_str in self.stream_events(workflow_name, params, ctx):
            # Parse the SSE string back into a dict
            lines = sse_str.strip().split("\n")
            result: dict[str, Any] = {}
            for line in lines:
                if line.startswith("event: "):
                    result["event"] = line[7:]
                elif line.startswith("data: "):
                    try:
                        result["data"] = json.loads(line[6:])
                    except json.JSONDecodeError:
                        result["data"] = line[6:]
            if result:
                yield result

    # ------------------------------------------------------------------
    # SSE Event Builders
    # ------------------------------------------------------------------

    @staticmethod
    def _event_to_sse(event: SSEEvent) -> str:
        """Convert an SSEEvent to an SSE-formatted string."""
        return event.to_sse_string()

    @staticmethod
    def _start_event(event_id: int, workflow_name: str, task_id: str) -> SSEEvent:
        """Create a "start" event."""
        return SSEEvent(
            event="start",
            data=json.dumps({
                "workflow": workflow_name,
                "task_id": task_id,
                "timestamp": time.time(),
            }),
            id=str(event_id),
        )

    @staticmethod
    def _progress_event(
        event_id: int, step: str, data: dict[str, Any]
    ) -> SSEEvent:
        """Create a "progress" event."""
        return SSEEvent(
            event="progress",
            data=json.dumps({
                "step": step,
                "data": data,
                "timestamp": time.time(),
            }),
            id=str(event_id),
        )

    @staticmethod
    def _error_event(event_id: int, error: str) -> str:
        """Create an "error" event and return as SSE string."""
        event = SSEEvent(
            event="error",
            data=json.dumps({"error": error, "timestamp": time.time()}),
            id=str(event_id),
        )
        return event.to_sse_string()

    @staticmethod
    def _heartbeat_event(event_id: int) -> str:
        """Create a heartbeat comment event."""
        return f": heartbeat {event_id}\n\n"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SSEExposer(endpoints={len(self._workflow_classes)}, "
            f"heartbeat={self._heartbeat_interval}s)"
        )
