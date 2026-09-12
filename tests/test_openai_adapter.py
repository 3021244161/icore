"""
Tests for icore.models.openai_adapter (adapter-level contracts).

Covers (v0.6.x, ICORE-ISSUE-003 —— 流式 usage 透出):
    - _build_payload: streaming payloads default to
      stream_options={"include_usage": True}; caller may override;
      non-streaming payloads are untouched
    - stream_chat: yields content tokens and invokes on_usage exactly
      once with the provider's final usage dict (including DeepSeek's
      prompt_cache_hit_tokens), taken from the usage-only trailing
      chunk (choices=[]) that the old parser dropped
    - stream_chat: usage attached to a regular content chunk is also
      captured (last non-null usage wins)
    - stream_chat: no usage chunk -> callback never fires
    - stream_chat: callback exceptions are swallowed
    - stream_chat: captured usage is recorded into token metrics
    - FakeModelAdapter (conftest): honours the same on_usage contract
"""

from __future__ import annotations

import json
from typing import Any

from icore.config import ModelConfig
from icore.models.openai_adapter import OpenAICompatibleAdapter
from icore.observability import (
    _reset_singleton_for_testing,
    get_metrics_registry,
)


# ---------------------------------------------------------------------------
# Fake streaming client harness
# ---------------------------------------------------------------------------

class _FakeStreamResp:
    """Mimics an httpx streaming response with canned SSE lines."""

    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self._lines = lines

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamClient:
    """Mimics httpx.AsyncClient.stream() and captures sent payloads."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.payloads: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, json: dict[str, Any] | None = None):
        self.payloads.append(json)

        class _CM:
            async def __aenter__(self) -> _FakeStreamResp:
                return _FakeStreamResp(outer._lines)

            async def __aexit__(self, *args: Any) -> bool:
                return False

        outer = self
        return _CM()


def _make_adapter(lines: list[str]) -> tuple[OpenAICompatibleAdapter, _FakeStreamClient]:
    cfg = ModelConfig(
        model_id="deepseek",
        model_name="deepseek-chat",
        api_base="http://localhost/v1",
        api_key="k",
    )
    adapter = OpenAICompatibleAdapter(cfg)
    client = _FakeStreamClient(lines)

    async def _fake_client() -> _FakeStreamClient:
        return client

    adapter._get_client = _fake_client  # type: ignore[assignment]
    return adapter, client


def _sse_lines(*chunks: dict[str, Any], with_done: bool = True) -> list[str]:
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    if with_done:
        lines.append("data: [DONE]")
    return lines


_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "prompt_cache_hit_tokens": 80,
    "prompt_cache_miss_tokens": 20,
}


# ---------------------------------------------------------------------------
# _build_payload: stream_options
# ---------------------------------------------------------------------------

class TestBuildPayloadStreamOptions:
    def test_stream_payload_defaults_to_include_usage(self):
        adapter = OpenAICompatibleAdapter(_cfg())
        payload = adapter._build_payload(
            [{"role": "user", "content": "hi"}], stream=True
        )
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}

    def test_non_stream_payload_has_no_stream_options(self):
        adapter = OpenAICompatibleAdapter(_cfg())
        payload = adapter._build_payload(
            [{"role": "user", "content": "hi"}], stream=False
        )
        assert "stream_options" not in payload
        assert "stream" not in payload

    def test_caller_stream_options_override_default(self):
        adapter = OpenAICompatibleAdapter(_cfg())
        payload = adapter._build_payload(
            [{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": False},
        )
        assert payload["stream_options"] == {"include_usage": False}

    def test_structured_keys_still_extracted_with_stream_options(self):
        adapter = OpenAICompatibleAdapter(_cfg())
        payload = adapter._build_payload(
            [{"role": "user", "content": "hi"}],
            stream=True,
            response_format={"type": "json_object"},
        )
        assert payload["response_format"] == {"type": "json_object"}


def _cfg() -> ModelConfig:
    return ModelConfig(
        model_id="deepseek",
        model_name="deepseek-chat",
        api_base="http://localhost/v1",
        api_key="k",
    )


# ---------------------------------------------------------------------------
# stream_chat: on_usage contract
# ---------------------------------------------------------------------------

class TestStreamChatUsageCallback:
    async def test_usage_only_final_chunk_invokes_callback(self):
        """issue 复现核心：末尾 usage-only chunk（choices=[]）不得再被丢弃。"""
        _reset_singleton_for_testing()
        lines = _sse_lines(
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [], "usage": dict(_USAGE)},
        )
        adapter, _client = _make_adapter(lines)
        received: list[dict[str, Any]] = []
        tokens: list[str] = []

        async for tok in adapter.stream_chat(
            [{"role": "user", "content": "hi"}],
            on_usage=received.append,
        ):
            tokens.append(tok)

        assert tokens == ["Hel", "lo"]
        assert received == [_USAGE]

    async def test_payload_sent_includes_stream_options(self):
        lines = _sse_lines({"choices": [{"delta": {"content": "x"}}]})
        adapter, client = _make_adapter(lines)

        async for _ in adapter.stream_chat([{"role": "user", "content": "hi"}]):
            pass

        assert client.payloads
        assert client.payloads[0]["stream_options"] == {"include_usage": True}

    async def test_usage_attached_to_content_chunk_captured(self):
        """部分 provider 把 usage 附在普通 chunk 上：最后一个非空 usage 生效。"""
        _reset_singleton_for_testing()
        lines = _sse_lines(
            {"choices": [{"delta": {"content": "a"}}],
             "usage": {"prompt_tokens": 1}},
            {"choices": [{"delta": {"content": "b"}}],
             "usage": dict(_USAGE)},
        )
        adapter, _client = _make_adapter(lines)
        received: list[dict[str, Any]] = []

        async for _ in adapter.stream_chat(
            [{"role": "user", "content": "hi"}], on_usage=received.append
        ):
            pass

        assert received == [_USAGE]

    async def test_no_usage_chunk_callback_never_fires(self):
        lines = _sse_lines({"choices": [{"delta": {"content": "x"}}]})
        adapter, _client = _make_adapter(lines)
        received: list[dict[str, Any]] = []

        async for _ in adapter.stream_chat(
            [{"role": "user", "content": "hi"}], on_usage=received.append
        ):
            pass

        assert received == []

    async def test_callback_exception_swallowed(self):
        lines = _sse_lines(
            {"choices": [{"delta": {"content": "x"}}]},
            {"choices": [], "usage": dict(_USAGE)},
        )
        adapter, _client = _make_adapter(lines)

        def _boom(usage: dict[str, Any]) -> None:
            raise RuntimeError("callback boom")

        tokens: list[str] = []
        async for tok in adapter.stream_chat(
            [{"role": "user", "content": "hi"}], on_usage=_boom
        ):
            tokens.append(tok)

        # 流本身不受回调异常影响
        assert tokens == ["x"]

    async def test_stream_usage_records_token_metrics(self):
        """流式路径此前完全不记 token 指标——现在与 chat() 对齐。"""
        _reset_singleton_for_testing()
        lines = _sse_lines(
            {"choices": [{"delta": {"content": "x"}}]},
            {"choices": [], "usage": dict(_USAGE)},
        )
        adapter, _client = _make_adapter(lines)

        async for _ in adapter.stream_chat([{"role": "user", "content": "hi"}]):
            pass

        reg = get_metrics_registry()
        assert reg.get_counter("icore_model_tokens_total").get(
            model_id="deepseek", type="prompt"
        ) == 100.0
        assert reg.get_counter("icore_model_tokens_total").get(
            model_id="deepseek", type="completion"
        ) == 50.0
        assert reg.get_counter(
            "icore_model_prompt_cache_hit_tokens_total"
        ).get(model_id="deepseek") == 80.0
        assert reg.get_counter(
            "icore_model_prompt_cache_miss_tokens_total"
        ).get(model_id="deepseek") == 20.0

    async def test_on_usage_does_not_leak_into_payload(self):
        """on_usage 是适配器层回调，绝不能进入 HTTP payload。"""
        lines = _sse_lines({"choices": [{"delta": {"content": "x"}}]})
        adapter, client = _make_adapter(lines)

        async for _ in adapter.stream_chat(
            [{"role": "user", "content": "hi"}], on_usage=lambda u: None
        ):
            pass

        assert client.payloads
        assert "on_usage" not in client.payloads[0]


# ---------------------------------------------------------------------------
# FakeModelAdapter honours the same contract (conftest)
# ---------------------------------------------------------------------------

class TestFakeAdapterStreamUsageContract:
    async def test_fake_adapter_invokes_on_usage_after_stream(self):
        from tests.conftest import FakeModelAdapter, make_model_config

        adapter = FakeModelAdapter(make_model_config())
        received: list[dict[str, Any]] = []
        tokens: list[str] = []

        async for tok in adapter.stream_chat(
            [{"role": "user", "content": "hello world"}],
            on_usage=received.append,
        ):
            tokens.append(tok)

        assert tokens == ["hello ", "world "]
        assert received == [{
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }]

    async def test_fake_adapter_without_callback_unchanged(self):
        from tests.conftest import FakeModelAdapter, make_model_config

        adapter = FakeModelAdapter(make_model_config())
        tokens: list[str] = []
        async for tok in adapter.stream_chat(
            [{"role": "user", "content": "hello"}]
        ):
            tokens.append(tok)

        assert tokens == ["hello "]
