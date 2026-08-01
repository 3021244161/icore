"""
Smoke tests for icore.services - service exposers.

Covers all 4 exposer implementations:
    - ToolServiceExposer: expose, list, call_service with model injection,
      OpenAI function-calling tool definitions, execute_tool_call
    - SSEExposer: expose, list, call_service, stream_events emits
      start/result/done events, SSEEvent formatting
    - StreamlitExposer: expose, list, call_service, generate_app_code
      produces runnable Python
    - MCPServiceExposer: expose, list, call_service (without starting the
      MCP server - no `mcp` package required for these paths)

Key regression guard:
    All exposers must use TaskContext.set_model_manager() /
    set_db_manager() for dependency injection - never the
    ``object.__setattr__`` private-field hack. A dedicated test asserts
    the injected manager is reachable via ctx.get_model_adapter().

These tests use the FakeModelAdapter (no network) and the example
DocumentSummaryWorkflow (already registered via @register_workflow).
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest

from icore.core.task_context import TaskContext
from icore.engine.registry import workflow_registry
from icore.services.mcp_server import MCPServiceExposer
from icore.services.sse_adapter import SSEEvent, SSEExposer
from icore.services.streamlit_app import StreamlitExposer
from icore.services.tool_service import ToolServiceExposer

# Ensure example workflows are registered
import icore.workflows.examples  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def document_summary_workflow_cls():
    """Get the DocumentSummaryWorkflow class from the registry."""
    return workflow_registry.get("document_summary")


@pytest.fixture
def tool_exposer(fake_model_manager, document_summary_workflow_cls):
    exposer = ToolServiceExposer(model_manager=fake_model_manager)
    return exposer, document_summary_workflow_cls


# ---------------------------------------------------------------------------
# ToolServiceExposer
# ---------------------------------------------------------------------------

class TestToolServiceExposer:
    @pytest.mark.asyncio
    async def test_expose_returns_servicedef(
        self, tool_exposer
    ):
        exposer, wf_cls = tool_exposer
        svc = await exposer.expose(wf_cls)
        assert svc.name == "document_summary"
        assert svc.protocol == "tool"
        assert svc.workflow_name == "document_summary"
        assert svc.description  # non-empty
        assert "function" in svc.input_schema

    @pytest.mark.asyncio
    async def test_list_services(self, tool_exposer):
        exposer, wf_cls = tool_exposer
        await exposer.expose(wf_cls)
        services = await exposer.list_services()
        assert len(services) == 1
        assert services[0].name == "document_summary"

    @pytest.mark.asyncio
    async def test_call_service_with_model_injection(
        self, tool_exposer, fake_model_manager
    ):
        exposer, wf_cls = tool_exposer
        await exposer.expose(wf_cls)

        ctx = TaskContext(
            task_id="t-tool-1",
            workflow_id="wf-tool-1",
        )
        # ctx has NO model manager before call - exposer should inject
        result = await exposer.call_service(
            "document_summary",
            {"document": "Hello world document for tool service."},
            ctx,
        )
        assert result["status"] == "success"
        assert result["data"] is not None
        # Regression: injection used the setter, not object.__setattr__
        # If the setter was used, get_model_adapter() must succeed.
        adapter = ctx.get_model_adapter()
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_call_service_unknown_name_raises(self, tool_exposer):
        exposer, _ = tool_exposer
        ctx = TaskContext(task_id="t", workflow_id="wf")
        with pytest.raises(KeyError):
            await exposer.call_service("ghost", {}, ctx)

    @pytest.mark.asyncio
    async def test_call_service_swallows_execution_exception(
        self, tool_exposer
    ):
        exposer, wf_cls = tool_exposer
        await exposer.expose(wf_cls)
        # Pass invalid params (missing 'document') -> workflow raises,
        # exposer catches and returns error dict
        ctx = TaskContext(task_id="t-err", workflow_id="wf-err")
        # Inject model so we get past model setup
        ctx.set_model_manager(exposer._model_manager)
        result = await exposer.call_service("document_summary", {}, ctx)
        assert result["status"] == "error"
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_get_tool_definitions_openai_format(self, tool_exposer):
        exposer, wf_cls = tool_exposer
        await exposer.expose(wf_cls)
        defs = exposer.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "document_summary"

    @pytest.mark.asyncio
    async def test_execute_tool_call_creates_context(
        self, tool_exposer
    ):
        exposer, wf_cls = tool_exposer
        await exposer.expose(wf_cls)
        result = await exposer.execute_tool_call(
            "document_summary",
            {"document": "Tool call test."},
        )
        assert result["status"] == "success"

    def test_repr(self, tool_exposer):
        exposer, _ = tool_exposer
        assert "ToolServiceExposer" in repr(exposer)

    def test_setters_work(self, fake_model_manager):
        exposer = ToolServiceExposer()
        exposer.set_model_manager(fake_model_manager)
        exposer.set_db_manager(None)
        assert exposer._model_manager is fake_model_manager
        assert exposer._db_manager is None


# ---------------------------------------------------------------------------
# SSEExposer
# ---------------------------------------------------------------------------

class TestSSEExposer:
    @pytest.mark.asyncio
    async def test_expose_and_list(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = SSEExposer(model_manager=fake_model_manager)
        svc = await exposer.expose(document_summary_workflow_cls)
        assert svc.protocol == "sse"
        assert svc.name == "document_summary"
        services = await exposer.list_services()
        assert len(services) == 1

    @pytest.mark.asyncio
    async def test_call_service_returns_result(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = SSEExposer(model_manager=fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        ctx = TaskContext(task_id="t-sse-1", workflow_id="wf-sse-1")
        result = await exposer.call_service(
            "document_summary",
            {"document": "SSE call test."},
            ctx,
        )
        assert result["status"] == "success"
        # Regression: setter-based injection
        assert ctx.get_model_adapter() is not None

    @pytest.mark.asyncio
    async def test_stream_events_emits_start_result_done(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = SSEExposer(model_manager=fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        ctx = TaskContext(
            task_id="t-sse-stream",
            workflow_id="wf-sse-stream",
        )
        events = []
        async for sse_str in exposer.stream_events(
            "document_summary",
            {"document": "Stream test document."},
            ctx,
        ):
            events.append(sse_str)

        # Concatenate and check event types are present
        blob = "\n".join(events)
        assert "event: start" in blob
        assert "event: result" in blob
        assert "event: done" in blob

    @pytest.mark.asyncio
    async def test_stream_events_unknown_workflow_emits_error(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = SSEExposer(model_manager=fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        ctx = TaskContext(task_id="t", workflow_id="wf")
        events = []
        async for s in exposer.stream_events("ghost", {}, ctx):
            events.append(s)
        blob = "\n".join(events)
        assert "event: error" in blob

    @pytest.mark.asyncio
    async def test_stream_yields_dicts(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = SSEExposer(model_manager=fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        ctx = TaskContext(task_id="t", workflow_id="wf")
        dicts = []
        async for d in exposer.stream(
            "document_summary",
            {"document": "Dict stream test."},
            ctx,
        ):
            dicts.append(d)
        events = [d["event"] for d in dicts if "event" in d]
        assert "start" in events
        assert "done" in events

    def test_sse_event_formatting(self):
        evt = SSEEvent(event="result", data='{"k": 1}', id="42")
        s = evt.to_sse_string()
        assert "event: result" in s
        assert 'data: {"k": 1}' in s
        assert "id: 42" in s

    def test_repr(self, fake_model_manager):
        exposer = SSEExposer(model_manager=fake_model_manager)
        assert "SSEExposer" in repr(exposer)


# ---------------------------------------------------------------------------
# StreamlitExposer
# ---------------------------------------------------------------------------

class TestStreamlitExposer:
    @pytest.mark.asyncio
    async def test_expose_and_list(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = StreamlitExposer(model_manager=fake_model_manager)
        svc = await exposer.expose(document_summary_workflow_cls)
        assert svc.protocol == "streamlit"
        assert svc.name == "document_summary"
        assert "display_name" in svc.metadata
        services = await exposer.list_services()
        assert len(services) == 1

    @pytest.mark.asyncio
    async def test_call_service_with_injection(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = StreamlitExposer(model_manager=fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        ctx = TaskContext(task_id="t-ui-1", workflow_id="wf-ui-1")
        result = await exposer.call_service(
            "document_summary",
            {"document": "UI call test."},
            ctx,
        )
        assert result["status"] == "success"
        # Regression: setter-based injection
        assert ctx.get_model_adapter() is not None

    @pytest.mark.asyncio
    async def test_call_service_unknown_raises(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = StreamlitExposer(model_manager=fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        ctx = TaskContext(task_id="t", workflow_id="wf")
        with pytest.raises(KeyError):
            await exposer.call_service("ghost", {}, ctx)

    @pytest.mark.asyncio
    async def test_generate_app_code_is_valid_python(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = StreamlitExposer(model_manager=fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        code = exposer.generate_app_code()
        assert isinstance(code, str)
        assert "streamlit" in code.lower()
        # Must be syntactically valid Python
        compile(code, "<generated_app>", "exec")

    def test_setters_work(self, fake_model_manager):
        exposer = StreamlitExposer()
        exposer.set_model_manager(fake_model_manager)
        exposer.set_db_manager(None)
        assert exposer._model_manager is fake_model_manager


# ---------------------------------------------------------------------------
# MCPServiceExposer (without starting the server - no mcp package needed)
# ---------------------------------------------------------------------------

class TestMCPServiceExposer:
    @pytest.mark.asyncio
    async def test_expose_and_list(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = MCPServiceExposer()
        exposer.set_model_manager(fake_model_manager)
        svc = await exposer.expose(document_summary_workflow_cls)
        assert svc.protocol == "mcp"
        # MCP tool names are sanitised to snake_case
        assert svc.name == "document_summary"
        services = await exposer.list_services()
        assert len(services) == 1

    @pytest.mark.asyncio
    async def test_call_service_with_injection(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = MCPServiceExposer()
        exposer.set_model_manager(fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)

        ctx = TaskContext(task_id="t-mcp-1", workflow_id="wf-mcp-1")
        result = await exposer.call_service(
            "document_summary",
            {"document": "MCP call test."},
            ctx,
        )
        assert result["status"] == "success"
        # Regression: setter-based injection (not object.__setattr__)
        assert ctx.get_model_adapter() is not None

    @pytest.mark.asyncio
    async def test_call_service_unknown_raises(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = MCPServiceExposer()
        exposer.set_model_manager(fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        ctx = TaskContext(task_id="t", workflow_id="wf")
        with pytest.raises(KeyError):
            await exposer.call_service("ghost", {}, ctx)

    @pytest.mark.asyncio
    async def test_call_service_swallows_exception(
        self, fake_model_manager, document_summary_workflow_cls
    ):
        exposer = MCPServiceExposer()
        exposer.set_model_manager(fake_model_manager)
        await exposer.expose(document_summary_workflow_cls)
        ctx = TaskContext(task_id="t-err", workflow_id="wf-err")
        ctx.set_model_manager(fake_model_manager)
        # Missing 'document' param -> workflow raises -> caught
        result = await exposer.call_service("document_summary", {}, ctx)
        assert result["status"] == "error"
        assert result["error"] is not None

    def test_setters_work(self, fake_model_manager):
        exposer = MCPServiceExposer()
        exposer.set_model_manager(fake_model_manager)
        exposer.set_db_manager(None)
        assert exposer._model_manager is fake_model_manager
        assert exposer._db_manager is None

    def test_sanitise_name_helper(self):
        from icore.services.mcp_server import _sanitise_name
        assert _sanitise_name("Document Summary") == "document_summary"
        assert _sanitise_name("My-Workflow") == "my_workflow"
        assert _sanitise_name("already_snake") == "already_snake"

    def test_build_input_schema_falls_back_to_freeform(self):
        from icore.services.mcp_server import _build_input_schema

        class _NoModel:
            name = "x"

        schema = _build_input_schema(_NoModel)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is True

    def test_build_input_schema_uses_pydantic_model(self):
        from icore.services.mcp_server import _build_input_schema
        from icore.core.models import BaseTaskInput

        class _SchemaInput(BaseTaskInput):
            text: str

        class _WithModel:
            name = "x"
            input_model = _SchemaInput

        schema = _build_input_schema(_WithModel)
        assert "properties" in schema
        assert "text" in schema["properties"]


# ---------------------------------------------------------------------------
# Regression: no object.__setattr__ hack in any exposer
# ---------------------------------------------------------------------------

class TestNoSetattrHack:
    """Ensure all service exposers inject managers via the official setters.

    This guards against the anti-pattern identified in the code review
    (object.__setattr__(ctx, "_model_manager", ...)) regressing.
    """

    def test_source_has_no_object_setattr_in_services(self):
        import icore.services.mcp_server as mcp_mod
        import icore.services.sse_adapter as sse_mod
        import icore.services.streamlit_app as st_mod
        import icore.services.tool_service as tool_mod
        import inspect

        for mod in (mcp_mod, sse_mod, st_mod, tool_mod):
            src = inspect.getsource(mod)
            assert "object.__setattr__" not in src, (
                f"{mod.__name__} still uses object.__setattr__; "
                "use TaskContext.set_model_manager() / set_db_manager() instead"
            )
