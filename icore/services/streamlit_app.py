"""
icore.services.streamlit_app - Streamlit UI exposer.

Generates a Streamlit web application that provides an interactive UI
for executing registered workflows. Each workflow becomes a page or
section of the Streamlit app with form inputs and result display.

Use this when you want to:
    - Provide a quick web UI for testing workflows
    - Create internal tools with minimal frontend effort
    - Demonstrate workflows to non-technical stakeholders
    - Build admin panels for workflow monitoring

Design:
    The StreamlitExposer generates a Streamlit app (app.py) that:
    1. Shows a sidebar listing all registered workflows
    2. Renders input forms based on each workflow's input_model
    3. Executes workflows on form submission
    4. Displays results (text, JSON, charts as appropriate)
    5. Supports streaming display via st.write_stream()

Usage:
    exposer = StreamlitExposer(model_manager=mm, db_manager=dm)
    exposer.expose(DocumentSummaryWorkflow)
    exposer.expose(EntityExtractionWorkflow)

    # Generate the app code and save
    app_code = exposer.generate_app_code()
    # Or run directly: streamlit run the generated app
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


class StreamlitExposer(BaseServiceExposer):
    """
    Exposes workflows through a Streamlit web UI.

    Generates a Streamlit app that renders an interactive interface
    for each registered workflow. Users can select a workflow from
    the sidebar, fill in parameters via auto-generated form inputs,
    and execute workflows with real-time streaming results.
    """

    name: str = "streamlit"
    protocol: str = "streamlit"

    def __init__(
        self,
        model_manager: Any = None,
        db_manager: Any = None,
    ) -> None:
        """
        Initialise the Streamlit exposer.

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
        Expose a workflow for Streamlit UI.

        Args:
            workflow_cls: The BaseWorkflow subclass to expose.

        Returns:
            A ServiceDef describing the Streamlit interface.
        """
        name = workflow_cls.name or workflow_cls.__name__.lower()
        service = ServiceDef(
            name=name,
            description=workflow_cls.description or "",
            workflow_name=workflow_cls.name,
            input_schema=self._derive_input_form(workflow_cls),
            output_schema={},
            protocol="streamlit",
            metadata={
                "source": "icore-streamlit-exposer",
                "display_name": getattr(workflow_cls, "display_name", name.replace("_", " ").title()),
            },
        )

        self._services[name] = service
        self._workflow_classes[name] = workflow_cls

        logger.info(
            "Exposed workflow '%s' for Streamlit UI", name
        )
        return service

    async def list_services(self) -> list[ServiceDef]:
        """List all Streamlit-exposed workflows."""
        return list(self._services.values())

    async def call_service(
        self, name: str, params: dict[str, Any], ctx: Any
    ) -> dict[str, Any]:
        """
        Call a workflow from the Streamlit UI.

        Args:
            name:   Workflow name.
            params: Form parameters as a dict.
            ctx:    TaskContext with injected dependencies.

        Returns:
            Dict with "status", "data", and optionally "error" keys.
        """
        if name not in self._workflow_classes:
            raise KeyError(
                f"Workflow '{name}' not found. "
                f"Available: {list(self._workflow_classes)}"
            )

        wf_cls = self._workflow_classes[name]

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
                "Streamlit workflow '%s' execution failed", name
            )
            return {
                "status": "error",
                "data": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # App Code Generation
    # ------------------------------------------------------------------

    def generate_app_code(self) -> str:
        """
        Generate the complete Streamlit app Python code.

        The generated code is a standalone Streamlit app that can
        be saved to a file and run with ``streamlit run``.

        Returns:
            A string containing the complete app code.
        """
        lines: list[str] = []
        lines.extend(self._header())
        lines.extend(self._sidebar_code())
        lines.extend(self._main_area_code())
        return "\n".join(lines)

    def _header(self) -> list[str]:
        """Generate the imports and page config section."""
        return [
            "# Auto-generated icore Streamlit App",
            "# Generated by StreamlitExposer",
            "",
            '"""icore Workflow UI - Interactive workflow execution interface."""',
            "",
            "from __future__ import annotations",
            "",
            "import json",
            "import asyncio",
            "import sys",
            "from pathlib import Path",
            "from uuid import uuid4",
            "",
            "import streamlit as st",
            "",
            'st.set_page_config(',
            '    page_title="icore Workflow UI",',
            '    page_icon="🤖",',
            '    layout="wide",',
            ")",
            "",
            "# Add icore to path (adjust as needed)",
            "ICORE_PATH = str(Path(__file__).resolve().parent.parent)",
            "if ICORE_PATH not in sys.path:",
            "    sys.path.insert(0, ICORE_PATH)",
            "",
            "from icore.core.task_context import TaskContext",
            "",
            "# --- Workflow registry ---",
            f"_WORKFLOW_MAP = {json.dumps(self._workflow_map(), indent=4)}",
            f"_SERVICE_MAP = {json.dumps(self._service_map(), indent=4)}",
            "",
            "",
            "def _import_workflow(name: str):",
            '    """Import a workflow class by module path."""',
            "    import importlib",
            "    module_path, class_name = _WORKFLOW_MAP[name]",
            "    mod = importlib.import_module(module_path)",
            "    return getattr(mod, class_name)",
            "",
            "",
            "def _get_model_manager():",
            '    """Get ModelManager from session state or create default."""',
            "    if '_model_manager' not in st.session_state:",
            "        try:",
            "            from icore.models.manager import ModelManager",
            "            st.session_state._model_manager = ModelManager()",
            "        except ImportError:",
            "            st.session_state._model_manager = None",
            "    return st.session_state._model_manager",
            "",
            "",
            "def _get_db_manager():",
            '    """Get DBManager from session state or create default."""',
            "    if '_db_manager' not in st.session_state:",
            "        try:",
            "            from icore.db.manager import DBManager",
            "            st.session_state._db_manager = DBManager()",
            "        except ImportError:",
            "            st.session_state._db_manager = None",
            "    return st.session_state._db_manager",
            "",
        ]

    def _sidebar_code(self) -> list[str]:
        """Generate the sidebar with workflow selector."""
        return [
            "",
            "# --- Sidebar ---",
            "st.sidebar.title('icore Workflows')",
            "st.sidebar.markdown('---')",
            "",
            "workflow_names = list(_SERVICE_MAP.keys())",
            "selected_name = st.sidebar.selectbox(",
            "    'Select Workflow',",
            "    workflow_names,",
            "    format_func=lambda n: _SERVICE_MAP[n].get('display_name', n),",
            ")",
            "",
            "st.sidebar.markdown('---')",
            "st.sidebar.info(",
            "    '**icore** - Enterprise LLM Workflow Platform\\n\\n'",
            "    'Select a workflow and fill in parameters to execute.',",
            ")",
        ]

    def _main_area_code(self) -> list[str]:
        """Generate the main content area."""
        return [
            "",
            "# --- Main Area ---",
            "if selected_name:",
            "    service = _SERVICE_MAP[selected_name]",
            "    st.title(service.get('display_name', selected_name))",
            "    st.markdown(service.get('description', ''))",
            "    st.markdown('---')",
            "",
            "    # Render input form",
            "    input_schema = service.get('input_schema', {})",
            "    properties = input_schema.get('properties', {})",
            "    required = set(input_schema.get('required', []))",
            "",
            "    if not properties:",
            "        st.info('This workflow has no input parameters.')",
            "        params = {}",
            "    else:",
            '        st.subheader("Parameters")',
            "        params = {}",
            "        cols = st.columns(2)",
            "        for i, (param_name, param_schema) in enumerate(properties.items()):",
            "            col = cols[i % 2]",
            "            with col:",
            "                label = param_name",
            "                if param_name in required:",
            "                    label += ' *'",
            "                default = param_schema.get('default', '')",
            '                description = param_schema.get("description", "")',
            "                param_type = param_schema.get('type', 'string')",
            "",
            "                if param_type == 'string':",
            "                    params[param_name] = st.text_input(",
            "                        label, value=default, help=description",
            "                    )",
            "                elif param_type in ('integer', 'number'):",
            "                    params[param_name] = st.number_input(",
            "                        label, value=default or 0, help=description",
            "                    )",
            '                elif param_type == "boolean":',
            "                    params[param_name] = st.checkbox(",
            "                        label, value=default or False, help=description",
            "                    )",
            "                else:",
            "                    params[param_name] = st.text_area(",
            "                        label, value=str(default) if default else '',",
            "                        help=description",
            "                    )",
            "",
            '    st.markdown("---")',
            "",
            "    # Execute button",
            '    col_exec, col_stream = st.columns([1, 1])',
            "    with col_exec:",
            '        execute_clicked = st.button("▶ Execute", type="primary", use_container_width=True)',
            "    with col_stream:",
            '        stream_mode = st.checkbox("Stream output", value=False)',
            "",
            "    if execute_clicked:",
            '        with st.spinner("Executing workflow..."):',
            "            try:",
            "                workflow_cls = _import_workflow(selected_name)",
            "                ctx = TaskContext(",
            "                    task_id=f'st:{selected_name}:{uuid4().hex[:8]}',",
            "                    workflow_id=selected_name,",
            '                    stream=stream_mode,',
            '                    metadata={"source": "streamlit"},',
            "                )",
            "",
            "                mm = _get_model_manager()",
            "                dm = _get_db_manager()",
            "                if mm is not None:",
            "                    ctx.set_model_manager(mm)",
            "                if dm is not None:",
            "                    ctx.set_db_manager(dm)",
            "",
            "                wf = workflow_cls()",
            "                output = asyncio.run(wf.execute(ctx, params))",
            "",
            '                if output.status == "success":',
            '                    st.success("Workflow completed successfully")',
            '                    with st.expander("Results", expanded=True):',
            "                        st.json(output.data)",
            "                else:",
            '                    st.error(f"Workflow failed: {output.error}")',
            "",
            "            except Exception as e:",
            "                st.exception(e)",
            "",
            "    st.markdown('---')",
            '    st.caption("Generated by icore StreamlitExposer")',
        ]

    def _workflow_map(self) -> dict[str, list[str]]:
        """Build a map of workflow name -> [module_path, class_name]."""
        result: dict[str, list[str]] = {}
        for name, wf_cls in self._workflow_classes.items():
            module = wf_cls.__module__
            class_name = wf_cls.__name__
            result[name] = [module, class_name]
        return result

    def _service_map(self) -> dict[str, dict[str, Any]]:
        """Build a map of workflow name -> service metadata."""
        result: dict[str, dict[str, Any]] = {}
        for name, service in self._services.items():
            result[name] = {
                "display_name": service.metadata.get("display_name", name),
                "description": service.description,
                "input_schema": service.input_schema,
            }
        return result

    # ------------------------------------------------------------------
    # Input Form Derivation
    # ------------------------------------------------------------------

    def _derive_input_form(
        self, workflow_cls: type[BaseWorkflow]
    ) -> dict[str, Any]:
        """
        Derive an input form schema from the workflow's input_model.

        Uses Pydantic's model_json_schema() to get field descriptions,
        types, defaults, and required status. This is used by the
        Streamlit UI to auto-generate form inputs.

        Returns:
            A dict with "properties" and "required" keys suitable
            for Streamlit form generation.
        """
        input_model = getattr(workflow_cls, "input_model", None)
        if (
            input_model is not None
            and hasattr(input_model, "model_json_schema")
            and input_model is not type(None)
        ):
            try:
                schema = input_model.model_json_schema()
                return {
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                }
            except Exception as e:
                logger.warning(
                    "Failed to derive input schema for %s: %s",
                    workflow_cls.name,
                    e,
                )

        return {"properties": {}, "required": []}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StreamlitExposer(workflows={len(self._workflow_classes)})"
        )
