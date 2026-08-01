"""
icore.services.mcp_server - MCP (Model Context Protocol) service exposer.

Converts registered workflows into MCP tools, making them available to
AI coding assistants (Claude, Cursor, etc.) and other MCP-compatible clients.

Uses the ``mcp`` SDK (FastMCP) for server setup. If the SDK is not installed,
the module imports gracefully and raises a clear error at call time.

Design:
    Each registered workflow becomes one MCP tool:
    - Tool name:   workflow.name (with sanitisation)
    - Description: workflow.description
    - Parameters:  JSON Schema derived from workflow input expectations
    - Handler:     Creates TaskContext, executes workflow, returns result

Usage:
    exposer = MCPServiceExposer()
    exposer.expose(DocumentSummaryWorkflow)

    # Run the MCP server (stdio transport by default)
    await exposer.start()
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from icore.core.task_context import TaskContext
from icore.services.base_exposer import BaseServiceExposer, ServiceDef

if TYPE_CHECKING:
    from icore.engine.base_workflow import BaseWorkflow

logger = logging.getLogger(__name__)


class MCPServiceExposer(BaseServiceExposer):
    """
    Exposes workflows as MCP (Model Context Protocol) tools.

    Wraps each registered workflow as an MCP tool. When the tool is
    called by an MCP client (e.g., Claude Desktop), the exposer:
        1. Creates a TaskContext from the tool parameters
        2. Instantiates the workflow
        3. Calls workflow.execute(ctx, params)
        4. Returns the result as a JSON-serialisable dict

    The MCP server runs on stdio transport by default, which is the
    standard for MCP server integration.
    """

    name: str = "mcp"
    protocol: str = "mcp"

    def __init__(self) -> None:
        """Initialise the MCP exposer."""
        self._services: dict[str, ServiceDef] = {}
        self._workflow_classes: dict[str, type[BaseWorkflow]] = {}
        self._model_manager: Any = None
        self._db_manager: Any = None

    # ------------------------------------------------------------------
    # Configuration (must be called before expose)
    # ------------------------------------------------------------------

    def set_model_manager(self, model_manager: Any) -> None:
        """
        Set the ModelManager for injected dependencies.

        The ModelManager is injected into TaskContext so workflows
        can access LLM adapters.
        """
        self._model_manager = model_manager

    def set_db_manager(self, db_manager: Any) -> None:
        """
        Set the DBManager for injected dependencies.

        The DBManager is injected into TaskContext so workflows
        can access database connections.
        """
        self._db_manager = db_manager

    # ------------------------------------------------------------------
    # Expose
    # ------------------------------------------------------------------

    async def expose(
        self, workflow_cls: type[BaseWorkflow]
    ) -> ServiceDef:
        """
        Expose a workflow as an MCP tool.

        Args:
            workflow_cls: The BaseWorkflow subclass to expose.

        Returns:
            A ServiceDef describing the exposed MCP tool.
        """
        name = _sanitise_name(workflow_cls.name or workflow_cls.__name__)

        service = ServiceDef(
            name=name,
            description=workflow_cls.description or "",
            workflow_name=workflow_cls.name,
            input_schema=_build_input_schema(workflow_cls),
            output_schema=_build_output_schema(),
            protocol="mcp",
            metadata={"source": "icore-mcp-exposer"},
        )

        self._services[name] = service
        self._workflow_classes[name] = workflow_cls

        logger.info("Exposed workflow '%s' as MCP tool '%s'", workflow_cls.name, name)
        return service

    async def list_services(self) -> list[ServiceDef]:
        """List all exposed MCP tools."""
        return list(self._services.values())

    async def call_service(
        self, name: str, params: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        """
        Call an exposed MCP tool.

        Args:
            name:   Tool name.
            params: Parameters for the workflow invocation.
            ctx:    TaskContext with injected dependencies.

        Returns:
            Dict with "status", "data", and optionally "error" keys.

        Raises:
            KeyError: If no tool exists with the given name.
        """
        if name not in self._workflow_classes:
            raise KeyError(
                f"MCP tool '{name}' not found. "
                f"Available: {list(self._workflow_classes)}"
            )

        workflow_cls = self._workflow_classes[name]

        # Inject managers if the context doesn't already have them
        if self._model_manager is not None:
            try:
                ctx.get_model_adapter()
            except RuntimeError:
                ctx.set_model_manager(self._model_manager)

        if self._db_manager is not None:
            try:
                ctx.get_db("default")
            except RuntimeError:
                ctx.set_db_manager(self._db_manager)

        wf = workflow_cls()
        try:
            output = await wf.execute(ctx, params)
            return {
                "status": output.status,
                "data": output.data,
                "error": output.error if output.status == "error" else None,
            }
        except Exception as exc:
            logger.exception("MCP tool '%s' execution failed", name)
            return {
                "status": "error",
                "data": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the MCP server using stdio transport.

        Registers all exposed tools with the MCP server and starts
        listening. This is a blocking operation (the MCP SDK handles
        the event loop).

        Requires the ``mcp`` package to be installed.
        """
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError:
            raise ImportError(
                "The 'mcp' package is required for MCPServiceExposer. "
                "Install it with: pip install mcp"
            )

        server = FastMCP("icore-mcp-server")

        for name, wf_cls in self._workflow_classes.items():
            service = self._services[name]

            # Create a closure that captures name
            def make_handler(wf_name: str):
                async def handler(**kwargs: Any) -> str:
                    # Create context for this invocation
                    ctx = TaskContext(
                        task_id=f"mcp:{wf_name}:{_short_id()}",
                        workflow_id=wf_name,
                        metadata={"source": "mcp", "raw_params": kwargs},
                    )
                    if self._model_manager is not None:
                        ctx.set_model_manager(self._model_manager)
                    if self._db_manager is not None:
                        ctx.set_db_manager(self._db_manager)

                    result = await self.call_service(wf_name, kwargs, ctx)
                    return json.dumps(result, ensure_ascii=False, default=str)

                return handler

            server.tool(
                name=name,
                description=service.description,
            )(make_handler(name))

            logger.info("Registered MCP tool: %s", name)

        logger.info("Starting MCP server with %d tools", len(self._workflow_classes))
        await server.run_stdio_async()

    async def stop(self) -> None:
        """Stop the MCP server (no-op for stdio transport)."""
        logger.info("Stopping MCP exposer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_name(name: str) -> str:
    """
    Sanitise a workflow name for use as an MCP tool name.

    MCP tool names should be snake_case with no special characters.
    """
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _build_input_schema(workflow_cls: type[BaseWorkflow]) -> dict[str, Any]:
    """
    Build a JSON Schema for the workflow's input.

    Attempts to derive from the workflow's input_model or from a
    generic params object.
    """
    # Check if the workflow has an input_model attribute
    input_model = getattr(workflow_cls, "input_model", None)
    if input_model is not None and hasattr(input_model, "model_json_schema"):
        return input_model.model_json_schema()

    # Fallback: free-form parameters
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
        "description": "Workflow input parameters",
    }


def _build_output_schema() -> dict[str, Any]:
    """Build a JSON Schema for workflow output."""
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "data": {"type": "object"},
            "error": {"type": "string"},
        },
    }


def _short_id() -> str:
    """Generate a short random ID."""
    from uuid import uuid4
    return uuid4().hex[:8]
