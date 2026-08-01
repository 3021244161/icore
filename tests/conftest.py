"""
Shared pytest fixtures for icore.

Provides a ``FakeModelAdapter`` that mimics an LLM without network access,
so the full workflow pipeline (chunk -> summarize -> merge, etc.) can be
exercised end-to-end in tests deterministically and offline.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from icore.config import ModelConfig
from icore.models.base_adapter import BaseModelAdapter


# ---------------------------------------------------------------------------
# Fake model adapter
# ---------------------------------------------------------------------------

class FakeModelAdapter(BaseModelAdapter):
    """Deterministic, offline LLM adapter for tests.

    ``chat()`` echoes the last user message prefixed with ``SUMMARY:`` so
    tests can assert data flow through multi-task DAGs.
    """

    def __init__(self, config: ModelConfig, *, responder: Any = None) -> None:
        super().__init__(config)
        self._responder = responder
        self.calls: list[list[dict[str, str]]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(messages)
        if self._responder is not None:
            content = self._responder(messages)
        else:
            # Echo the last user message
            user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    break
            content = f"SUMMARY: {user_msg[:80]}"
        return {
            "content": content,
            "role": "assistant",
            "model": self.model_id,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "finish_reason": "stop",
        }

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            for m in reversed(messages):
                if m.get("role") == "user":
                    for tok in (m.get("content", "")).split():
                        yield tok + " "
                    return

        return _gen()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def make_model_config(model_id: str = "fake-model") -> ModelConfig:
    """Build a minimal valid ModelConfig for tests."""
    return ModelConfig(
        model_id=model_id,
        model_name=model_id,
        api_base="http://localhost/v1",
        api_key="test-key",
        max_tokens=1024,
        enabled=True,
        tags=["cheap", "fast", "capable"],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_model_config() -> ModelConfig:
    return make_model_config()


@pytest.fixture
def fake_adapter(fake_model_config: ModelConfig) -> FakeModelAdapter:
    return FakeModelAdapter(fake_model_config)


@pytest.fixture
def fake_model_manager(fake_adapter: FakeModelAdapter):
    """A real ModelManager with a single fake adapter registered + default."""
    from icore.models.manager import ModelManager

    mgr = ModelManager()
    mgr.register_adapter("fake-model", fake_adapter)
    # Make auto-routing resolve to the fake model
    mgr._router.set_default_model_id("fake-model")
    mgr._auto_routing = True
    return mgr
