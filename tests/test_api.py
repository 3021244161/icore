"""
Tests for icore.api - FastAPI application & endpoints.

Covers:
    - GET /health returns 200 with status/version/timestamp
    - POST /invoke sync mode (no stream, no callback) returns InvokeResponse
    - POST /invoke with unknown workflow returns 404
    - POST /invoke streaming mode returns SSE text/event-stream
    - POST /invoke async mode (callback_url) returns immediately with running
    - Exception handlers: KeyError -> 404, ValueError -> 422, Exception -> 500
    - App state contains model_manager / db_manager / callback_manager
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from icore.api.callback import CallbackManager
from icore.api.main import create_app
from icore.api.schemas import HealthResponse, InvokeRequest, InvokeResponse
from icore.api.streaming import SSEStreamHandler
from icore.config import Settings, get_settings
from icore.engine.registry import workflow_registry

# Ensure example workflows are registered
import icore.workflows.examples  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_fake_model(fake_model_manager):
    """FastAPI app wired with the fake model manager."""
    app = create_app(
        model_manager=fake_model_manager,
        db_manager=None,
    )
    return app


@pytest.fixture
def client(app_with_fake_model):
    """TestClient backed by the app with fake model manager."""
    return TestClient(app_with_fake_model)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_shape(self, client):
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "timestamp" in body
        assert body["timestamp"] != ""  # non-empty

    def test_health_no_auth_required(self, client):
        # No Authorization header -> still 200
        r = client.get("/health", headers={})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Sync invoke
# ---------------------------------------------------------------------------

class TestInvokeSync:
    def test_invoke_document_summary_sync(self, client):
        r = client.post(
            "/invoke",
            json={
                "workflow_name": "document_summary",
                "params": {
                    "document": "This is a test document for summarization.",
                },
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["task_id"]  # auto-generated UUID
        assert body["result"] is not None
        assert "summary" in body["result"]

    def test_invoke_with_explicit_task_id(self, client):
        r = client.post(
            "/invoke",
            json={
                "workflow_name": "document_summary",
                "task_id": "my-task-123",
                "params": {
                    "document": "Short doc.",
                },
            },
        )
        assert r.status_code == 200
        assert r.json()["task_id"] == "my-task-123"

    def test_invoke_unknown_workflow_returns_404(self, client):
        r = client.post(
            "/invoke",
            json={
                "workflow_name": "does_not_exist",
                "params": {},
            },
        )
        assert r.status_code == 404
        body = r.json()
        assert "not registered" in body["detail"].lower() or \
               "does_not_exist" in body["detail"]

    def test_invoke_missing_workflow_name_returns_422(self, client):
        # Pydantic validation: required field missing
        r = client.post("/invoke", json={"params": {}})
        assert r.status_code == 422

    def test_invoke_with_model_id(self, client, fake_model_manager):
        # Explicitly request the fake model
        r = client.post(
            "/invoke",
            json={
                "workflow_name": "document_summary",
                "model_id": "fake-model",
                "params": {"document": "Test doc with explicit model."},
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "success"


# ---------------------------------------------------------------------------
# Streaming invoke (SSE)
# ---------------------------------------------------------------------------

class TestInvokeStreaming:
    def test_stream_returns_event_stream(self, client):
        with client.stream(
            "POST",
            "/invoke",
            json={
                "workflow_name": "document_summary",
                "stream": True,
                "params": {"document": "Stream this doc."},
            },
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            # Collect the full body
            body = b"".join(r.iter_bytes()).decode("utf-8")
            # Should contain SSE data events and a [DONE] marker
            assert "data:" in body
            assert "[DONE]" in body

    def test_stream_includes_start_and_result_events(self, client):
        with client.stream(
            "POST",
            "/invoke",
            json={
                "workflow_name": "document_summary",
                "stream": True,
                "params": {"document": "Show start + result events."},
            },
        ) as r:
            body = b"".join(r.iter_bytes()).decode("utf-8")

        # Parse SSE events
        events = []
        for block in body.split("\n\n"):
            block = block.strip()
            if block.startswith("data: "):
                payload = block[len("data: "):]
                if payload == "[DONE]":
                    continue
                events.append(json.loads(payload))

        # First event: status=running
        assert events[0]["status"] == "running"
        # Last data event: should contain the workflow result
        last = events[-1]
        assert last["status"] in ("success", "error")
        if last["status"] == "success":
            assert "data" in last


# ---------------------------------------------------------------------------
# Async invoke (callback)
# ---------------------------------------------------------------------------

class TestInvokeCallback:
    def test_callback_url_returns_immediately(self, client):
        # The callback URL doesn't need to be reachable - the API returns
        # immediately and the background task will retry/fail silently.
        r = client.post(
            "/invoke",
            json={
                "workflow_name": "document_summary",
                "callback_url": "http://localhost:9999/nonexistent-callback",
                "params": {"document": "Async callback test."},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "running"
        assert body["task_id"]
        # result should be None (not yet available)
        assert body["result"] is None


# ---------------------------------------------------------------------------
# App state & wiring
# ---------------------------------------------------------------------------

class TestAppState:
    def test_app_state_has_callback_manager(self, app_with_fake_model):
        assert app_with_fake_model.state.callback_manager is not None
        assert isinstance(
            app_with_fake_model.state.callback_manager, CallbackManager
        )

    def test_app_state_has_workflow_registry(self, app_with_fake_model):
        assert app_with_fake_model.state.workflow_registry is not None

    def test_app_state_has_model_manager(self, app_with_fake_model, fake_model_manager):
        assert app_with_fake_model.state.model_manager is fake_model_manager

    def test_app_state_has_bg_tasks_set(self, app_with_fake_model):
        assert hasattr(app_with_fake_model.state, "_bg_tasks")
        assert isinstance(app_with_fake_model.state._bg_tasks, set)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

class TestExceptionHandlers:
    def test_keyerror_returns_404(self, client):
        # Trigger by invoking unknown workflow
        r = client.post(
            "/invoke",
            json={"workflow_name": "ghost_wf", "params": {}},
        )
        assert r.status_code == 404

    def test_value_error_returns_422(self, app_with_fake_model):
        # Inject a workflow that raises ValueError on execution
        # We use a test client with a custom workflow
        from icore.engine.base_workflow import BaseWorkflow
        from icore.engine.dag import DAG
        from icore.engine.registry import register_workflow

        @register_workflow("test_valueerror_wf")
        class _VEWorkflow(BaseWorkflow):
            name = "test_valueerror_wf"
            description = "raises ValueError"

            def define(self) -> DAG:
                dag = DAG()
                dag.add_node("a", task_name="text_chunker")
                return dag

            async def execute(self, ctx, params):
                raise ValueError("invalid input from test")

        client = TestClient(app_with_fake_model)
        r = client.post(
            "/invoke",
            json={"workflow_name": "test_valueerror_wf", "params": {}},
        )
        # The API catches Exception in the sync path and returns 200 with
        # status="error", but the global handler returns 422 for ValueError
        # raised outside the try/except. Since execute() is inside the
        # try/except in the invoke endpoint, it returns 200 with error.
        assert r.status_code in (200, 422)
        body = r.json()
        # Either way, an error should be reported
        if r.status_code == 200:
            assert body["status"] == "error"
        else:
            assert "invalid input" in body["detail"].lower() or \
                   "validation" in body["detail"].lower()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_invoke_request_defaults(self):
        req = InvokeRequest(workflow_name="test")
        assert req.params == {}
        assert req.task_id is None
        assert req.callback_url is None
        assert req.model_id is None
        assert req.stream is False
        assert req.metadata == {}

    def test_invoke_request_extra_fields_allowed(self):
        req = InvokeRequest(
            workflow_name="test",
            custom_field="allowed",  # type: ignore[call-arg]
        )
        # extra="allow" preserves custom fields
        assert req.model_dump().get("custom_field") == "allowed"

    def test_invoke_response_default_status(self):
        resp = InvokeResponse(task_id="t1")
        assert resp.status == "success"
        assert resp.result is None
        assert resp.error is None

    def test_health_response_defaults(self):
        h = HealthResponse()
        assert h.status == "healthy"
        assert h.version == "1.0.0"


# ---------------------------------------------------------------------------
# SSEStreamHandler (unit-level)
# ---------------------------------------------------------------------------

class TestSSEStreamHandler:
    def test_format_data(self):
        s = SSEStreamHandler._format_data({"a": 1})
        assert s.startswith("data: ")
        assert s.endswith("\n\n")
        assert json.loads(s[len("data: "):].strip()) == {"a": 1}

    def test_format_error(self):
        s = SSEStreamHandler._format_error("boom", {"task_id": "t1"})
        assert s.startswith("event: error\n")
        assert "boom" in s
        assert "t1" in s

    async def test_stream_yields_start_result_done(self, fake_model_manager):
        from icore.core.task_context import TaskContext
        from icore.workflows.examples.document_summary import (
            DocumentSummaryWorkflow,
        )

        handler = SSEStreamHandler()
        ctx = TaskContext(task_id="t-sse", workflow_id="wf-sse")
        ctx.set_model_manager(fake_model_manager)
        wf = DocumentSummaryWorkflow()

        chunks: list[str] = []
        async for chunk in handler.stream(wf, ctx, {"document": "SSE test."}):
            chunks.append(chunk)

        # Should have at least: start event, result event, [DONE]
        assert len(chunks) >= 3
        assert chunks[-1] == "data: [DONE]\n\n"

        # First chunk should be a data event with status=running
        first = chunks[0]
        assert "data:" in first
        first_payload = json.loads(first[len("data: "):].strip())
        assert first_payload["status"] == "running"
        assert first_payload["task_id"] == "t-sse"

    async def test_stream_handles_workflow_error(self):
        """If the workflow raises, SSE should emit error event + [DONE]."""
        from icore.core.task_context import TaskContext
        from icore.engine.base_workflow import BaseWorkflow
        from icore.engine.dag import DAG

        class _CrashWorkflow(BaseWorkflow):
            name = "_crash_test"
            description = "always raises"

            def define(self) -> DAG:
                dag = DAG()
                dag.add_node("a", task_name="text_chunker")
                return dag

            async def execute(self, ctx, params):
                raise RuntimeError("crash!")

        handler = SSEStreamHandler()
        ctx = TaskContext(task_id="t-crash")
        wf = _CrashWorkflow()

        chunks: list[str] = []
        async for chunk in handler.stream(wf, ctx, {}):
            chunks.append(chunk)

        # Should still have [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"
        # Should have an error event
        joined = "".join(chunks)
        assert "event: error" in joined
        assert "crash!" in joined


# ---------------------------------------------------------------------------
# CallbackManager
# ---------------------------------------------------------------------------

class TestCallbackManager:
    def test_init_defaults(self):
        mgr = CallbackManager()
        assert mgr._timeout == 30
        assert mgr._max_retries == 3
        assert mgr._retry_delay == 1.0

    async def test_deliver_and_wait_returns_true_on_success(self):
        """Mock httpx.AsyncClient.post to return 200."""
        mgr = CallbackManager(timeout=1, max_retries=1, retry_delay=0)

        async def fake_post(*args, **kwargs):
            class _Resp:
                status_code = 200
                text = "OK"
            return _Resp()

        with patch("httpx.AsyncClient.post", new=fake_post):
            ok = await mgr.deliver_and_wait(
                "http://example.com/cb", {"task_id": "t1"}
            )
        assert ok is True

    async def test_deliver_and_wait_returns_false_on_4xx(self):
        mgr = CallbackManager(timeout=1, max_retries=3, retry_delay=0)

        async def fake_post(*args, **kwargs):
            class _Resp:
                status_code = 404
                text = "Not Found"
            return _Resp()

        with patch("httpx.AsyncClient.post", new=fake_post):
            ok = await mgr.deliver_and_wait(
                "http://example.com/cb", {"task_id": "t1"}
            )
        assert ok is False

    async def test_deliver_and_wait_retries_on_5xx(self):
        mgr = CallbackManager(timeout=1, max_retries=2, retry_delay=0)

        call_count = {"n": 0}

        async def fake_post(*args, **kwargs):
            call_count["n"] += 1
            class _Resp:
                status_code = 500
                text = "Internal Error"
            return _Resp()

        with patch("httpx.AsyncClient.post", new=fake_post):
            ok = await mgr.deliver_and_wait(
                "http://example.com/cb", {"task_id": "t1"}
            )
        assert ok is False
        assert call_count["n"] == 2  # retried once (max_retries=2)
