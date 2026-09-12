"""
icore.models.openai_adapter - OpenAI-compatible LLM adapter implementation.

OpenAICompatibleAdapter is the default implementation of BaseModelAdapter,
supporting any API that follows the OpenAI Chat Completions specification.
This includes:
    - OpenAI (GPT-4o, GPT-4o-mini, etc.)
    - Azure OpenAI (with custom api_base)
    - DeepSeek, Moonshot, ZhipuAI, etc.
    - Self-hosted models via vLLM, Ollama, LM Studio, etc.

Key design decisions:
    - httpx.AsyncClient for async HTTP with connection pooling
    - Lazy import of httpx (module importable without httpx installed)
    - Exponential backoff retry for transient errors (429, 5xx, timeout)
    - No retry for client errors (400, 401, 404) - they won't succeed
    - SSE stream parsing for stream_chat() yielding content tokens
    - Unified response dict format in chat()
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from icore.config import ModelConfig
from icore.models.base_adapter import BaseModelAdapter
from icore.models.exceptions import ModelAPIError
from icore.observability import get_metrics_registry, start_span

logger = logging.getLogger(__name__)


# HTTP status codes that warrant retry (transient errors)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatibleAdapter(BaseModelAdapter):
    """
    LLM adapter for OpenAI-compatible API endpoints.

    Uses httpx.AsyncClient for async HTTP communication with connection
    pool reuse. The client is created lazily on first use and cached
    for the lifetime of the adapter instance.

    Args:
        config: ModelConfig with api_base, api_key, model_name, etc.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client: Any = None  # httpx.AsyncClient (lazy init)
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Lazy httpx client management
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        """
        Get or create the httpx.AsyncClient instance.

        httpx is imported lazily so this module can be imported without
        httpx installed. The client is created on first use and reused
        for all subsequent calls (connection pool reuse).

        Returns:
            The httpx.AsyncClient instance.

        Raises:
            ImportError: If httpx is not installed.
        """
        if self._client is not None and not self._closed:
            return self._client

        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "httpx is required for OpenAICompatibleAdapter. "
                "Install it with: pip install httpx"
            ) from e

        self._client = httpx.AsyncClient(
            base_url=self.config.api_base.rstrip("/"),
            headers=self._build_headers(),
            timeout=self.config.request_timeout,
        )
        self._closed = False
        return self._client

    def _build_headers(self) -> dict[str, str]:
        """Build the HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Build the request payload for chat completions.

        Merges ModelConfig defaults with call-specific kwargs.
        Call kwargs take precedence over config defaults.

        v0.6: 显式透传 ``response_format`` / ``tools`` / ``tool_choice``
        结构化输出参数。这些参数由调用方（如 AgentNodeExecutor）传入，
        用于强制模型按 JSON 格式输出，减少解析失败概率。

        v0.6.x (ICORE-ISSUE-003): 流式请求默认附加
        ``stream_options={"include_usage": True}``（OpenAI 兼容协议，
        provider 在流末尾的 usage-only chunk 返回精确用量，含 DeepSeek
        的 prompt_cache_hit_tokens 等扩展字段）。调用方显式传入
        ``stream_options`` 时以调用方为准（可关闭）。
        """
        payload: dict[str, Any] = {
            "model": kwargs.pop("model_name", self.config.model_name),
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "temperature": kwargs.pop("temperature", self.config.temperature),
        }
        if stream:
            payload["stream"] = True
            payload.setdefault("stream_options", {"include_usage": True})
        # v0.6: 显式提取结构化输出参数（避免被其他 kwargs 覆盖）。
        for _structured_key in ("response_format", "tools", "tool_choice"):
            if _structured_key in kwargs:
                payload[_structured_key] = kwargs.pop(_structured_key)
        # Merge any remaining kwargs (top_p, frequency_penalty, etc.)
        payload.update(kwargs)
        return payload

    def _record_usage_metrics(self, usage: dict[str, Any]) -> None:
        """
        Record token usage metrics for a completed model call.

        v0.6.x: shared by both chat() and stream_chat() (ICORE-ISSUE-003 —
        the streaming path previously recorded NO token metrics at all).
        Best-effort: never raises into the caller.
        """
        try:
            _reg = get_metrics_registry()
            _reg.get_counter("icore_model_tokens_total").inc(
                model_id=self.model_id,
                type="prompt",
                n=usage.get("prompt_tokens", 0),
            )
            _reg.get_counter("icore_model_tokens_total").inc(
                model_id=self.model_id,
                type="completion",
                n=usage.get("completion_tokens", 0),
            )
            prompt_cache_hit = usage.get("prompt_cache_hit_tokens", 0)
            prompt_cache_miss = usage.get("prompt_cache_miss_tokens", 0)
            if prompt_cache_hit:
                _reg.get_counter(
                    "icore_model_prompt_cache_hit_tokens_total"
                ).inc(model_id=self.model_id, n=prompt_cache_hit)
            if prompt_cache_miss:
                _reg.get_counter(
                    "icore_model_prompt_cache_miss_tokens_total"
                ).inc(model_id=self.model_id, n=prompt_cache_miss)
        except Exception:  # noqa: BLE001 - metrics 不得影响核心逻辑
            pass

    async def _retry_async(
        self,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an async function with exponential backoff retry.

        Retries on:
            - httpx.TimeoutException
            - httpx.HTTPStatusError with status in _RETRYABLE_STATUS_CODES
            - asyncio.TimeoutError

        Does NOT retry on:
            - ModelAPIError with non-retryable status codes
            - Other unexpected exceptions

        Args:
            fn:  The async callable to execute.
            *args: Positional arguments passed to fn.
            **kwargs: Keyword arguments passed to fn.

        Returns:
            The return value of fn.

        Raises:
            ModelAPIError: If all retries are exhausted on retryable errors.
            Exception: For non-retryable errors, re-raised immediately.
        """
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except ModelAPIError as e:
                if e.status_code not in _RETRYABLE_STATUS_CODES:
                    raise
                last_exc = e
            except Exception as e:
                # Check if it's a network/timeout error
                exc_name = type(e).__name__
                if exc_name in ("TimeoutException", "ConnectError",
                                "ReadError", "RemoteProtocolError",
                                "TimeoutError", "asyncio.TimeoutError"):
                    last_exc = e
                else:
                    raise

            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    "API call to '%s' failed (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    self.model_id,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    last_exc,
                )
                await asyncio.sleep(delay)

        raise ModelAPIError(
            status_code=0,
            detail=f"Max retries ({max_retries}) exhausted: {last_exc}",
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------
    # BaseModelAdapter implementation
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform a non-streaming chat completion."""
        client = await self._get_client()
        payload = self._build_payload(messages, stream=False, **kwargs)

        async def _do_request() -> dict[str, Any]:
            resp = await client.post("/chat/completions", json=payload)
            if resp.status_code != 200:
                error_detail = "Unknown error"
                try:
                    error_body = resp.json()
                    error_detail = error_body.get("error", {}).get(
                        "message", str(error_body)
                    )
                except Exception:
                    error_detail = resp.text
                raise ModelAPIError(
                    status_code=resp.status_code,
                    detail=error_detail,
                    model_id=self.model_id,
                )
            return resp.json()

        # v0.6 可观测性：tracer span + token / error 指标
        with start_span("model.call") as _span:
            _span.set_attribute("model_id", self.model_id)
            try:
                data = await self._retry_async(_do_request)
            except Exception:
                try:
                    get_metrics_registry().get_counter(
                        "icore_model_errors_total"
                    ).inc(model_id=self.model_id)
                except Exception:  # noqa: BLE001 - metrics 不得影响核心逻辑
                    pass
                raise

            # Normalize to unified format
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            # v0.6.x: DeepSeek 等提供方返回的上下文缓存命中/未命中 token
            # （prompt_cache_hit_tokens / prompt_cache_miss_tokens）。多轮对话
            # 中前缀命中缓存可显著降低输入成本，透传给上层并记录指标以便观测。
            prompt_cache_hit_tokens = usage.get("prompt_cache_hit_tokens", 0)
            prompt_cache_miss_tokens = usage.get("prompt_cache_miss_tokens", 0)

            _span.set_attribute("prompt_tokens", prompt_tokens)
            _span.set_attribute("completion_tokens", completion_tokens)
            _span.set_attribute(
                "prompt_cache_hit_tokens", prompt_cache_hit_tokens
            )
            _span.set_attribute(
                "prompt_cache_miss_tokens", prompt_cache_miss_tokens
            )

            self._record_usage_metrics(usage)

            return {
                "content": message.get("content", ""),
                "role": message.get("role", "assistant"),
                "model": data.get("model", self.config.model_name),
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
                    "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
                },
                "finish_reason": choice.get("finish_reason", "stop"),
            }

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        on_usage: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Perform a streaming chat completion.

        Yields content tokens as they arrive from the API. Uses SSE
        stream parsing to extract delta.content from each chunk.

        v0.6.x (ICORE-ISSUE-003): 流式用量透出。

        - 请求默认附加 ``stream_options={"include_usage": True}``，OpenAI
          兼容协议随后在**流末尾的 usage-only chunk**（``choices=[]``）返回
          该次调用的精确用量（含 ``prompt_tokens`` / ``completion_tokens``
          及 DeepSeek 的 ``prompt_cache_hit_tokens`` 等 provider 扩展字段）。
        - 调用方可传入 ``on_usage`` 回调：流被**完整消费**后回调一次，
          入参为 provider 返回的原始 usage dict（拷贝）。回调抛出的异常
          仅记录日志，不影响流本身。
        - 若 provider 未返回 usage、流中途出错或调用方提前 break，
          ``on_usage`` 不会被调用（精确记账依赖完整消费）。
        - 无论是否传 ``on_usage``，捕获到的用量都会记录到 token 指标
          （``icore_model_tokens_total`` 等）。
        """
        client = await self._get_client()
        payload = self._build_payload(messages, stream=True, **kwargs)

        # 每次尝试（含重试）独立收集 usage；仅成功完成的流会触发回调。
        usage_holder: dict[str, Any] = {}

        async def _do_stream() -> AsyncIterator[str]:
            usage_holder.clear()
            async with client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    error_detail = body.decode("utf-8", errors="replace")
                    try:
                        error_json = json.loads(error_detail)
                        error_detail = error_json.get("error", {}).get(
                            "message", error_detail
                        )
                    except Exception:
                        pass
                    raise ModelAPIError(
                        status_code=resp.status_code,
                        detail=error_detail,
                        model_id=self.model_id,
                    )

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Skipping unparseable SSE line: %s", data_str
                        )
                        continue

                    # ICORE-ISSUE-003: usage-only chunk（choices=[]）在旧实现
                    # 被 `if not choices: continue` 丢弃——先捕获 usage。
                    # 部分 provider 也会把 usage 附在普通 chunk 上，统一
                    # 以"最后一个非空 usage"为准。
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage_holder["usage"] = chunk_usage

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

        # Note: streaming retry wraps the entire stream, which means if
        # the connection drops mid-stream, the retry will restart from
        # the beginning. This is acceptable for LLM use cases since
        # partial outputs are not useful.
        async for token in self._retry_stream(_do_stream):
            yield token

        # 流被完整消费：记录指标 + 触发一次性回调。
        captured = usage_holder.get("usage")
        if captured:
            self._record_usage_metrics(captured)
            if on_usage is not None:
                try:
                    on_usage(dict(captured))
                except Exception as e:  # noqa: BLE001 - 回调不得影响流
                    logger.warning(
                        "on_usage callback failed for '%s': %s",
                        self.model_id,
                        e,
                    )

    async def _retry_stream(
        self,
        stream_fn: Any,
    ) -> AsyncIterator[str]:
        """
        Wrap a streaming function with retry logic.

        Retries the initial connection on transient errors. Once tokens
        start flowing, no retry is attempted (partial output is discarded
        on error, and the caller will see an exception).
        """
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                async for token in stream_fn():
                    yield token
                return  # Stream completed successfully
            except ModelAPIError as e:
                if e.status_code not in _RETRYABLE_STATUS_CODES:
                    raise
                last_exc = e
            except Exception as e:
                exc_name = type(e).__name__
                if exc_name in ("TimeoutException", "ConnectError",
                                "ReadError", "RemoteProtocolError",
                                "TimeoutError", "asyncio.TimeoutError"):
                    last_exc = e
                else:
                    raise

            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    "Stream to '%s' failed (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    self.model_id,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    last_exc,
                )
                await asyncio.sleep(delay)

        raise ModelAPIError(
            status_code=0,
            detail=f"Max retries ({max_retries}) exhausted: {last_exc}",
            model_id=self.model_id,
        )

    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        client = await self._get_client()

        async def _do_request() -> list[list[float]]:
            resp = await client.post(
                "/embeddings",
                json={
                    "model": self.config.model_name,
                    "input": texts,
                },
            )
            if resp.status_code != 200:
                error_detail = "Unknown error"
                try:
                    error_body = resp.json()
                    error_detail = error_body.get("error", {}).get(
                        "message", str(error_body)
                    )
                except Exception:
                    error_detail = resp.text
                raise ModelAPIError(
                    status_code=resp.status_code,
                    detail=error_detail,
                    model_id=self.model_id,
                )
            data = resp.json()
            # Sort by index to ensure order matches input
            embeddings_data = sorted(
                data.get("data", []), key=lambda x: x.get("index", 0)
            )
            return [item["embedding"] for item in embeddings_data]

        return await self._retry_async(_do_request)

    async def health_check(self) -> bool:
        """
        Perform a lightweight health check.

        Tries a GET /models request first (fastest). If the endpoint
        doesn't support it, falls back to a minimal chat completion.
        """
        try:
            client = await self._get_client()
            resp = await client.get("/models")
            if resp.status_code == 200:
                return True
            # If /models not supported, try a minimal chat
            if resp.status_code == 404:
                return await self._minimal_chat_health()
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Health check failed for '%s': %s", self.model_id, e)
            return False

    async def _minimal_chat_health(self) -> bool:
        """Fallback health check via minimal chat completion."""
        try:
            result = await asyncio.wait_for(
                self.chat(
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                ),
                timeout=10,
            )
            return bool(result.get("content"))
        except Exception as e:
            logger.debug(
                "Minimal chat health check failed for '%s': %s",
                self.model_id,
                e,
            )
            return False

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None and not self._closed:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning("Error closing client for '%s': %s", self.model_id, e)
            finally:
                self._client = None
                self._closed = True
