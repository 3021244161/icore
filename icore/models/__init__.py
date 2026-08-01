"""
icore.models - LLM model management layer.

This package manages multiple LLM model adapters, auto-routing, health
checks, and fallback chains.

Public API:
    - BaseModelAdapter:          Abstract base class for all adapters
    - OpenAICompatibleAdapter:   Default OpenAI-compatible adapter
    - ModelManager:              Multi-model coordinator (get_adapter, register, health)
    - ModelRouter:               Auto-routing engine
    - ModelHealthStatus:         Health status Pydantic model
    - ChatMessage/ChatRequest/EmbeddingRequest: Request/response types

Exceptions:
    - ModelError:                Base exception for all model errors
    - ModelNotFoundError:        Raised when model_id is not registered
    - ModelUnhealthyError:       Raised when model is unhealthy with no fallback
    - NoAvailableModelError:     Raised when auto-routing finds no model
    - ModelAPIError:             Raised when LLM API call fails

Config types (re-exported from icore.config):
    - ModelConfig, ModelSettings, RoutingRule
"""

from __future__ import annotations

from icore.models.base_adapter import BaseModelAdapter
from icore.models.config import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
    ModelHealthStatus,
)
from icore.models.exceptions import (
    ModelAPIError,
    ModelError,
    ModelNotFoundError,
    ModelUnhealthyError,
    NoAvailableModelError,
)
from icore.models.manager import ModelManager
from icore.models.openai_adapter import OpenAICompatibleAdapter
from icore.models.router import ModelRouter

# Re-export config types for convenience
from icore.config import ModelConfig, ModelSettings, RoutingRule

__all__ = [
    # Base classes
    "BaseModelAdapter",
    "OpenAICompatibleAdapter",
    # Management
    "ModelManager",
    "ModelRouter",
    "ModelHealthStatus",
    # Request/response types
    "ChatMessage",
    "ChatRequest",
    "EmbeddingRequest",
    # Exceptions
    "ModelError",
    "ModelAPIError",
    "ModelNotFoundError",
    "ModelUnhealthyError",
    "NoAvailableModelError",
    # Config re-exports
    "ModelConfig",
    "ModelSettings",
    "RoutingRule",
]
