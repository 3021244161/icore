"""
icore.observability.middleware - FastAPI/Starlette observability middleware.

Provides two ASGI middlewares:

    - ``RequestIDMiddleware``: injects an ``X-Request-ID`` header on every
      request (reusing an incoming one when present, otherwise generating a
      fresh UUID). The id is stored on ``request.state.request_id`` and
      echoed back on the response.
    - ``MetricsMiddleware``: records ``icore_invoke_total`` (Counter) and
      ``icore_invoke_duration_seconds`` (Histogram) around each request,
      tagged with the workflow name extracted from the ``POST /invoke``
      JSON body (``"unknown"`` otherwise).

Both middlewares are implemented as **pure ASGI** classes (``__init__(app,
...)`` + ``__call__(scope, receive, send)``) rather than
``BaseHTTPMiddleware``. The pure-ASGI form is used deliberately for
``MetricsMiddleware`` because it must read the request body to extract
``workflow_name`` and then replay it to the downstream app — something that
is fiddly and error-prone with ``BaseHTTPMiddleware``.

Both middlewares accept an optional ``registry`` argument; when omitted
``MetricsMiddleware`` falls back to
:func:`icore.observability.get_metrics_registry`. They do not modify
``icore/api/main.py`` or ``icore/bootstrap.py`` — callers wire them up
explicitly via ``app.add_middleware(...)``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

from icore.observability import MetricsRegistry, get_metrics_registry

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = b"x-request-id"
_REQUEST_ID_HEADER_STR = "X-Request-ID"


class RequestIDMiddleware:
    """Pure-ASGI middleware that injects an ``X-Request-ID`` header.

    If the incoming request carries an ``X-Request-ID`` header it is reused;
    otherwise a fresh UUID4 is generated. The id is exposed to downstream
    handlers via ``request.state.request_id`` (i.e. ``scope["state"]``) and
    echoed on the response.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id: Optional[str] = None
        for name, value in scope.get("headers", []):
            if name.lower() == _REQUEST_ID_HEADER:
                request_id = value.decode("latin-1")
                break
        if not request_id:
            request_id = str(uuid.uuid4())

        # Expose to downstream handlers via Starlette's request state.
        state = scope.setdefault("state", {})
        state["request_id"] = request_id

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = [
                    (n, v)
                    for (n, v) in message.get("headers", [])
                    if n.lower() != _REQUEST_ID_HEADER
                ]
                headers.append(
                    (_REQUEST_ID_HEADER, request_id.encode("latin-1"))  # type: ignore[union-attr]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class MetricsMiddleware:
    """Pure-ASGI middleware recording invocation count and latency.

    For ``POST /invoke`` the middleware reads the JSON body to extract
    ``workflow_name`` (so the ``icore_invoke_total`` /
    ``icore_invoke_duration_seconds`` metrics are tagged with it). The body
    is replayed to the downstream app so handlers see it intact.

    Status mapping: HTTP status code ``< 400`` counts as ``success``;
    anything ``>= 400`` (and any raised exception) counts as ``error``.
    """

    def __init__(
        self,
        app: Any,
        registry: Optional[MetricsRegistry] = None,
    ) -> None:
        self.app = app
        self._registry = registry

    def _registry_or_default(self) -> MetricsRegistry:
        return self._registry if self._registry is not None else get_metrics_registry()

    @staticmethod
    def _extract_workflow_name(body: bytes) -> str:
        if not body:
            return "unknown"
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                wf = data.get("workflow_name")
                if isinstance(wf, str) and wf:
                    return wf
        except (ValueError, TypeError):
            pass
        return "unknown"

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        registry = self._registry_or_default()
        workflow_name = "unknown"

        # Only buffer the body for the invoke endpoint; reading the body for
        # every request would be wasteful and could break streaming.
        if scope.get("method") == "POST" and scope.get("path") == "/invoke":
            body = b""
            more_body = True
            while more_body:
                message = await receive()
                if message.get("type") != "http.request":
                    # http.disconnect or unexpected message; stop reading.
                    break
                body += message.get("body", b"")
                more_body = message.get("more_body", False)

            workflow_name = self._extract_workflow_name(body)

            body_replayed = False

            async def replay_receive() -> dict[str, Any]:
                nonlocal body_replayed
                if not body_replayed:
                    body_replayed = True
                    return {
                        "type": "http.request",
                        "body": body,
                        "more_body": False,
                    }
                # Subsequent reads (e.g. waiting for disconnect) delegate to
                # the original receive.
                return await receive()

            downstream_receive: Callable[[], Awaitable[dict[str, Any]]] = replay_receive
        else:
            downstream_receive = receive

        start = time.monotonic()
        status_code_holder = {"code": 200}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status_code_holder["code"] = message.get("status", 200)
            await send(message)

        status = "success"
        try:
            await self.app(scope, downstream_receive, send_wrapper)
            if status_code_holder["code"] >= 400:
                status = "error"
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.monotonic() - start
            try:
                registry.get_counter("icore_invoke_total").inc(
                    workflow_name=workflow_name, status=status
                )
                registry.get_histogram("icore_invoke_duration_seconds").observe(
                    duration, workflow_name=workflow_name
                )
            except Exception:  # pragma: no cover - never let metrics break a response
                logger.warning(
                    "Failed to record request metrics (workflow=%s, status=%s)",
                    workflow_name,
                    status,
                    exc_info=True,
                )


__all__ = [
    "RequestIDMiddleware",
    "MetricsMiddleware",
]
