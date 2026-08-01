"""
icore.models.config - Model-related configuration and request/response types.

This module re-exports the core model configuration classes defined in
``icore.config`` and adds request/response data models specific to the
LLM adapter layer (chat messages, chat requests, embedding requests).

Re-exports from icore.config:
    - ModelConfig:      Configuration for a single LLM model instance.
    - RoutingRule:      A single auto-routing rule.
    - ModelSettings:    Top-level model settings (models, routing rules, etc.)

Additional models defined here:
    - ChatMessage:      A single chat message (role + content).
    - ChatRequest:      A structured chat completion request.
    - EmbeddingRequest: A structured embedding request.
    - ModelHealthStatus: Health status of a model (model_id + healthy flag).

These types are used by BaseModelAdapter implementations and ModelManager
to standardise request/response shapes across different LLM providers.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# Re-export core config classes so consumers can import everything from one place.
from icore.config import ModelConfig, ModelSettings, RoutingRule

__all__ = [
    "ModelConfig",
    "RoutingRule",
    "ModelSettings",
    "ChatMessage",
    "ChatRequest",
    "EmbeddingRequest",
    "ModelHealthStatus",
]


# ---------------------------------------------------------------------------
# Chat message and request models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """
    A single message in a chat conversation.

    This mirrors the OpenAI chat message format but is provider-agnostic.
    Adapters are responsible for translating this into the specific
    format required by their backend API.

    Attributes:
        role:    The role of the message sender.
                 Common values: "system", "user", "assistant", "tool".
        content: The text content of the message.
        name:    Optional name for the sender (used in multi-turn conversations).
        tool_call_id: Optional tool call ID (for function/tool calling).
    """

    model_config = {"extra": "allow"}

    role: str = Field(description="Message role: system, user, assistant, or tool")
    content: str = Field(default="", description="Message text content")
    name: Optional[str] = Field(default=None, description="Optional sender name")
    tool_call_id: Optional[str] = Field(
        default=None, description="Tool call ID for function calling"
    )


class ChatRequest(BaseModel):
    """
    A structured chat completion request.

    This is the canonical request shape that BaseModelAdapter.chat()
    and stream_chat() accept internally. Callers may also pass a plain
    list[dict] for convenience; adapters convert it to this model.

    Attributes:
        messages:     The conversation messages.
        model:        Override the model name (defaults to adapter's model_name).
        temperature:  Sampling temperature override.
        max_tokens:   Maximum tokens to generate.
        top_p:        Nucleus sampling parameter.
        stop:         Stop sequences.
        stream:       Whether to stream the response.
        extra_params: Provider-specific extra parameters passed through.
    """

    model_config = {"extra": "allow"}

    messages: list[ChatMessage] = Field(
        description="Conversation messages"
    )
    model: Optional[str] = Field(
        default=None, description="Model name override"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, description="Max tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling"
    )
    stop: Optional[list[str]] = Field(
        default=None, description="Stop sequences"
    )
    stream: bool = Field(
        default=False, description="Whether to stream the response"
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific extra parameters",
    )


# ---------------------------------------------------------------------------
# Embedding request model
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    """
    A structured embedding request.

    Attributes:
        input:    Text or list of texts to embed.
        model:    Override the model name for embedding.
        dimensions: Desired embedding dimensions (if supported).
    """

    model_config = {"extra": "allow"}

    input: str | list[str] = Field(
        description="Text or list of texts to embed"
    )
    model: Optional[str] = Field(
        default=None, description="Model name override for embeddings"
    )
    dimensions: Optional[int] = Field(
        default=None, ge=1, description="Desired embedding dimensions"
    )


# ---------------------------------------------------------------------------
# Model health status
# ---------------------------------------------------------------------------


class ModelHealthStatus(BaseModel):
    """
    Health status of a single model.

    Used by ModelManager.health_check_all() to report the health of
    every registered model.

    Attributes:
        model_id:   The model's unique identifier.
        healthy:    Whether the model is currently available.
        latency_ms: Response latency in milliseconds (None if unhealthy).
        error:      Error message if unhealthy, None if healthy.
        checked_at: ISO timestamp of the last health check.
    """

    model_config = {"extra": "allow"}

    model_id: str = Field(description="Model identifier")
    healthy: bool = Field(description="Whether the model is available")
    latency_ms: Optional[float] = Field(
        default=None, description="Response latency in milliseconds"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if unhealthy"
    )
    checked_at: Optional[str] = Field(
        default=None, description="ISO timestamp of last check"
    )
