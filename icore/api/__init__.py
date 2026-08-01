"""
icore.api - API access layer.

This package provides the HTTP API layer for icore, built on FastAPI.

Two endpoints only:
    - GET  /health  - System health check (no params)
    - POST /invoke  - Main interface (workflow invocation)

Key components:
    - create_app():       Factory function to build the FastAPI app
    - InvokeRequest:      Request schema for POST /invoke
    - InvokeResponse:     Response schema for non-streaming results
    - HealthResponse:     Response schema for GET /health
    - CallbackManager:    Async result delivery to callback URLs
    - SSEStreamHandler:   SSE streaming response handler

Module dependency:
    api depends on engine (WorkflowRegistry, BaseWorkflow)
    api depends on core (TaskContext, BaseTaskOutput)
    api depends on config (APISettings, Settings)
"""

from __future__ import annotations

from icore.api.callback import CallbackManager
from icore.api.main import create_app
from icore.api.schemas import HealthResponse, InvokeRequest, InvokeResponse
from icore.api.streaming import SSEStreamHandler

__all__ = [
    "create_app",
    "InvokeRequest",
    "InvokeResponse",
    "HealthResponse",
    "CallbackManager",
    "SSEStreamHandler",
]
