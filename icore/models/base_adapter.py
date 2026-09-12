"""
icore.models.base_adapter - Abstract base class for LLM model adapters.

BaseModelAdapter defines the contract for all LLM interactions in icore.
Concrete adapters (e.g. OpenAICompatibleAdapter) implement the abstract
methods to communicate with specific model APIs.

The three core methods are:
    - chat():         Non-streaming chat completion (returns full response)
    - stream_chat():  Streaming chat completion (yields tokens one by one)
    - embed():        Text embedding (returns float vectors)

Additionally, every adapter must implement:
    - health_check(): Lightweight connectivity test
    - close():        Release underlying resources (HTTP clients, etc.)

Design principles:
    - All methods are async (I/O-bound LLM API calls)
    - stream_chat returns an AsyncIterator, not a coroutine
    - Unified response format in chat() return dict
    - Adapter instances are cached by ModelManager (one per model_id)
"""

from __future__ import annotations

import abc
from typing import Any, AsyncIterator

from icore.config import ModelConfig


class BaseModelAdapter(abc.ABC):
    """
    Abstract base class for LLM model adapters.

    Subclasses must implement all abstract methods. The ``config``
    attribute holds the ModelConfig instance provided at construction
    time, which contains API endpoint, key, retry settings, etc.

    Attributes:
        config:    The ModelConfig for this adapter's model.
        model_id:  Shorthand for config.model_id.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the adapter with a model configuration.

        Args:
            config: The ModelConfig containing API endpoint, key, and
                    behavior settings for this model.
        """
        self.config: ModelConfig = config
        self.model_id: str = config.model_id

    # ------------------------------------------------------------------
    # Core LLM interaction methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform a non-streaming chat completion.

        Args:
            messages: OpenAI-format message list, e.g.::

                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ]

            **kwargs: Additional API parameters (temperature, max_tokens,
                      top_p, etc.). These override the defaults in
                      ModelConfig for this call only.

        Returns:
            A dict with the following structure::

                {
                    "content": "Generated text",
                    "role": "assistant",
                    "model": "gpt-4o",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    "finish_reason": "stop",
                }

        Raises:
            Exception: If the API call fails after retries.
        """
        ...

    @abc.abstractmethod
    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        on_usage: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Perform a streaming chat completion.

        Yields content tokens as they arrive from the model. The caller
        iterates over the returned async iterator to receive tokens
        incrementally (suitable for SSE streaming).

        v0.6.x (ICORE-ISSUE-003): implementations SHOULD support the
        optional ``on_usage`` callback — invoked at most once, after the
        stream has been fully consumed, with the provider's usage dict
        for the call (``prompt_tokens`` / ``completion_tokens`` /
        ``total_tokens``, plus provider extensions such as DeepSeek's
        ``prompt_cache_hit_tokens``). Implementations that cannot obtain
        streaming usage may simply never invoke the callback; callers
        must treat the absence of the callback as "usage unavailable".
        Implementations must pop ``on_usage`` out of kwargs (it must
        never leak into the HTTP payload).

        Args:
            messages: OpenAI-format message list.
            on_usage: Optional callback receiving the final usage dict
                once the stream completes.
            **kwargs: Additional API parameters.

        Yields:
            str: Content tokens/chunks from the model, one at a time.

        Raises:
            Exception: If the API call fails after retries.
        """
        ...

    @abc.abstractmethod
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of input strings to embed.

        Returns:
            A list of embedding vectors, one per input text. Each
            vector is a list of floats.

        Raises:
            Exception: If the API call fails after retries.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Perform a lightweight health check on the model.

        This should be a fast, low-cost operation that verifies the
        model is reachable and the API key is valid. Typical
        implementations send a GET /models request or a minimal chat
        completion.

        Returns:
            True if the model is healthy and reachable, False otherwise.
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Release underlying resources.

        Called by ModelManager.close_all() during shutdown. Implementations
        should close HTTP clients, connection pools, etc.

        This method must not raise exceptions.
        """
        ...

    # ------------------------------------------------------------------
    # v0.5: Multimodal extensions (default to "not supported")
    # ------------------------------------------------------------------

    @property
    def supports_vision(self) -> bool:
        """Whether this adapter accepts image inputs. Default False."""
        return False

    @property
    def supports_audio(self) -> bool:
        """Whether this adapter accepts audio inputs. Default False."""
        return False

    async def chat_with_media(
        self,
        prompt: str,
        media_files: list[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Multimodal chat completion.

        Args:
            prompt:       User text prompt.
            media_files:  List of ``MediaFile`` instances.
            **kwargs:     Additional API parameters.

        Raises:
            NotImplementedError: If the model does not support vision.
        """
        raise NotImplementedError(
            f"Model '{self.model_id}' does not support multimodal input"
        )

    async def embed_multimodal(
        self,
        inputs: list[Any],
    ) -> list[list[float]]:
        """
        Multimodal embedding (e.g. CLIP image/text alignment).

        Default implementation delegates text inputs to ``embed()``
        and rejects media inputs.
        """
        text_inputs: list[str] = []
        for item in inputs:
            if isinstance(item, str):
                text_inputs.append(item)
            else:
                raise NotImplementedError(
                    f"Model '{self.model_id}' does not support "
                    f"multimodal embedding (input type: {type(item).__name__})"
                )
        return await self.embed(text_inputs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return f"{self.__class__.__name__}(model_id={self.model_id!r})"
