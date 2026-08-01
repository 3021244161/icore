"""
icore.services - Service exposure layer.

Converts registered workflows into consumable services through multiple
standard protocols: MCP (Model Context Protocol), Tool Service, Streamlit
UI, and SSE streaming. Each protocol has a dedicated exposer implementing
the BaseServiceExposer interface.

Core abstractions:
    - ServiceDef:            Standardised service definition (dataclass)
    - BaseServiceExposer:    Abstract base class for all exposers
    - MCPServiceExposer:     Exposes workflows as MCP tools
    - ToolServiceExposer:    Exposes workflows as callable tool functions
    - StreamlitExposer:      Generates Streamlit UI for workflows
    - SSEExposer:            Exposes workflows as SSE streaming endpoints
"""

from __future__ import annotations

from icore.services.base_exposer import BaseServiceExposer, ServiceDef
from icore.services.mcp_server import MCPServiceExposer
from icore.services.tool_service import ToolServiceExposer
from icore.services.streamlit_app import StreamlitExposer
from icore.services.sse_adapter import SSEExposer

__all__ = [
    "BaseServiceExposer",
    "ServiceDef",
    "MCPServiceExposer",
    "ToolServiceExposer",
    "StreamlitExposer",
    "SSEExposer",
]
