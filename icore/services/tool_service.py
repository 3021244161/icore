"""
icore.services.tool_service - Tool service exposer.

Wraps registered workflows as callable tool functions. This is the simplest
service exposure pattern: each workflow becomes a Python function that
accepts parameters as a dict and returns results.

Use this when you want to:
    - Call workflows programmatically from other Python code
    - Expose workflows as REST API tools (via FastAPI dependency injection)
    - Register workflows as OpenAI/GPT function calling tools
    - Build a simple tool registry for agent-based systems

Design:
    Each registered workflow becomes one ToolFunction:
    - Function name: workflow.name
    - Description:   workflow.description
    - Parameters:    dict[str, Any] (free-form or typed via input_model)
    - Handler:       Creates TaskContext, executes workflow, returns dict result

Usage:
    exposer = ToolServiceExposer(model_manager=mm, db_manager=dm)
    exposer.expose(DocumentSummaryWorkflow)

    result = await exposer.call_service("document_summary",
                                        params={"text": "..."},
                                        ctx=ctx)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from icore.core.task_context import TaskContext
from icore.services.base_exposer import BaseServiceExposer, ServiceDef

if TYPE_CHECKING:
    from icore.engine.base_workflow import BaseWorkflow

logger = logging.getLogger(__name__)

# Type alias for a callable tool function
ToolCallable = Callable[..., Coroutine[Any, Any, dict[str, Any]]]


class ToolServiceExposer(BaseServiceExposer):
    """
    Exposes workflows as callable tool functions.

    Each workflow is wrapped as a ToolFunction that can be called
    programmatically or registered with external tool registries
    (OpenAI function calling, LangChain tools, etc.).

    The exposer maintains a registry of ToolDef dataclass-like
    structures that describe each tool's name, description, and
    parameter schema, compatible with the OpenAI function calling
    format.
    """

    name: str = "tool"
    protocol: str = "tool"

    def __init__(
        self,
        model_manager: Any = None,
        db_manager: Any = None,
    ) -> None:
        """
        Initialise the tool service exposer.

        Args:
            model_manager: Optional ModelManager for dependency injection.
            db_manager:    Optional DBManager for dependency injection.
        """
        self._services: dict[str, ServiceDef] = {}
        self._workflow_classes: dict[str, type[BaseWorkflow]] = {}
        self._model_manager = model_manager
        self._db_manager = db_manager

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
        Expose a workflow as a tool function.

        Args:
            workflow_cls: The BaseWorkflow subclass to expose.

        Returns:
            A ServiceDef describing the tool function.
        """
        name = workflow_cls.name or workflow_cls.__name__.lower()

        # Build input schema (OpenAI function calling compatible)
        input_schema = self._build_tool_schema(workflow_cls)

        service = ServiceDef(
            name=name,
            description=workflow_cls.description or "",
            workflow_name=workflow_cls.name,
            input_schema=input_schema,
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "data": {"type": "object"},
                    "error": {"type": "string"},
                },
            },
            protocol="tool",
            metadata={
                "source": "icore-tool-exposer",
                "tool_type": "function",
            },
        )

        self._services[name] = service
        self._workflow_classes[name] = workflow_cls

        logger.info("Exposed workflow '%s' as tool '%s'", workflow_cls.name, name)
        return service

    async def list_services(self) -> list[ServiceDef]:
        """List all exposed tool functions."""
        return list(self._services.values())

    async def call_service(
        self, name: str, params: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        """
        Call an exposed tool function.

        Args:
            name:   Tool name (workflow name).
            params: Parameters dict for the workflow.
            ctx:    TaskContext with injected dependencies.

        Returns:
            Dict with "status", "data", and optionally "error" keys.

        Raises:
            KeyError: If no tool with the given name exists.
        """
        if name not in self._workflow_classes:
            raise KeyError(
                f"Tool '{name}' not found. "
                f"Available: {list(self._workflow_classes)}"
            )

        workflow_cls = self._workflow_classes[name]

        # Inject managers into context if needed
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

        wf = workflow_cls()
        try:
            output = await wf.execute(ctx, params)
            return {
                "status": output.status,
                "data": output.data,
                "error": output.error if output.status == "error" else None,
            }
        except Exception as exc:
            logger.exception("Tool '%s' execution failed", name)
            return {
                "status": "error",
                "data": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Tool Schema Builder (OpenAI function calling format)
    # ------------------------------------------------------------------

    def _build_tool_schema(
        self, workflow_cls: type[BaseWorkflow]
    ) -> dict[str, Any]:
        """
        Build an OpenAI function calling compatible schema.

        Returns a dict with:
            type: "function"
            function:
                name: ...
                description: ...
                parameters: {type: "object", properties: {...}}
        """
        param_schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

        # Try to derive from input_model
        input_model = getattr(workflow_cls, "input_model", None)
        if (
            input_model is not None
            and hasattr(input_model, "model_json_schema")
            and input_model is not type(None)
        ):
            try:
                json_schema = input_model.model_json_schema()
                # Remove JSON Schema keys that are not in OpenAI format
                param_schema["properties"] = json_schema.get(
                    "properties", {}
                )
                param_schema["required"] = json_schema.get(
                    "required", []
                )
            except Exception:
                param_schema = {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                }

        return {
            "type": "function",
            "function": {
                "name": workflow_cls.name or workflow_cls.__name__,
                "description": workflow_cls.description or "",
                "parameters": param_schema,
            },
        }

    # ------------------------------------------------------------------
    # OpenAI Function Calling Integration
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get all tools in OpenAI function calling format.

        Returns a list suitable for use as the ``tools`` parameter
        in OpenAI Chat Completions API calls.

        Returns:
            List of tool definitions in OpenAI format.
        """
        tools: list[dict[str, Any]] = []
        for name in self._workflow_classes:
            wf_cls = self._workflow_classes[name]
            tools.append(self._build_tool_schema(wf_cls))
        return tools

    async def execute_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool call from an OpenAI function calling response.

        Args:
            tool_name: The function name from the tool call.
            arguments: The parsed arguments dict from the tool call.

        Returns:
            A dict result suitable for returning to the OpenAI API.
        """
        from uuid import uuid4

        ctx = TaskContext(
            task_id=f"tool:{tool_name}:{uuid4().hex[:8]}",
            workflow_id=tool_name,
            metadata={"source": "openai_function_calling"},
        )

        return await self.call_service(tool_name, arguments, ctx)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ToolServiceExposer(tools={len(self._workflow_classes)})"
        )
