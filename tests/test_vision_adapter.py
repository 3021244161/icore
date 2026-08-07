"""
Tests for icore.models.vision_adapter - Multimodal vision LLM adapter.

Covers:
    - VisionModelAdapter inherits from OpenAICompatibleAdapter
    - supports_vision property is True
    - chat_with_media:
        * builds OpenAI Vision message format (image_url content blocks)
        * delegates to parent chat() with composed messages
        * raises ValidationError when video MediaFile is supplied
        * raises ValidationError when audio MediaFile is supplied
        * raises ValidationError for unsupported media_type
        * raises ValidationError when neither prompt nor media_files given
        * image_detail kwarg is consumed (not forwarded to chat)
        * multiple images produce multiple image_url blocks
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest

from icore.config import ModelConfig
from icore.exceptions import ValidationError
from icore.media import MediaFile, MediaSource, MediaType
from icore.models.openai_adapter import OpenAICompatibleAdapter
from icore.models.vision_adapter import VisionModelAdapter


# ---------------------------------------------------------------------------
# Construction & inheritance
# ---------------------------------------------------------------------------

class TestVisionModelAdapterBasics:
    def test_inherits_from_openai_adapter(self):
        adapter = VisionModelAdapter(_make_config())
        assert isinstance(adapter, OpenAICompatibleAdapter)

    def test_supports_vision_is_true(self):
        adapter = VisionModelAdapter(_make_config())
        assert adapter.supports_vision is True

    def test_parent_supports_vision_is_false(self):
        # Sanity check: the base adapter should NOT support vision.
        parent = OpenAICompatibleAdapter(_make_config())
        assert parent.supports_vision is False


# ---------------------------------------------------------------------------
# chat_with_media - happy path
# ---------------------------------------------------------------------------

class TestChatWithMediaHappyPath:
    async def test_single_image_calls_chat_with_composed_messages(self):
        adapter = VisionModelAdapter(_make_config())
        # Mock the parent chat() to capture the messages we compose.
        captured: dict = {}

        async def _fake_chat(messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return {
                "content": "vision-result",
                "role": "assistant",
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "finish_reason": "stop",
            }

        with patch.object(adapter, "chat", _fake_chat):
            mf = MediaFile(
                media_type=MediaType.IMAGE,
                source_type=MediaSource.BYTES,
                source=b"fake-image-bytes",
            )
            result = await adapter.chat_with_media("What is this?", [mf])

        assert result["content"] == "vision-result"
        messages = captured["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        # Should contain a text block + an image_url block.
        types = [block["type"] for block in content]
        assert "text" in types
        assert "image_url" in types

    async def test_multiple_images_create_multiple_blocks(self):
        adapter = VisionModelAdapter(_make_config())
        captured: dict = {}

        async def _fake_chat(messages, **kwargs):
            captured["messages"] = messages
            return {"content": "ok", "role": "assistant"}

        with patch.object(adapter, "chat", _fake_chat):
            files = [
                MediaFile(
                    media_type=MediaType.IMAGE,
                    source_type=MediaSource.BYTES,
                    source=b"img1",
                ),
                MediaFile(
                    media_type=MediaType.IMAGE,
                    source_type=MediaSource.BYTES,
                    source=b"img2",
                ),
            ]
            await adapter.chat_with_media("Compare", files)

        content = captured["messages"][0]["content"]
        image_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(image_blocks) == 2

    async def test_image_url_block_uses_data_url_format(self):
        adapter = VisionModelAdapter(_make_config())
        captured: dict = {}

        async def _fake_chat(messages, **kwargs):
            captured["messages"] = messages
            return {"content": "ok", "role": "assistant"}

        with patch.object(adapter, "chat", _fake_chat):
            mf = MediaFile(
                media_type=MediaType.IMAGE,
                source_type=MediaSource.BYTES,
                source=b"img-data",
                metadata={"mime_type": "image/png"},
            )
            await adapter.chat_with_media("desc", [mf])

        content = captured["messages"][0]["content"]
        image_block = next(b for b in content if b["type"] == "image_url")
        url = image_block["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # The base64 portion should decode back to the original bytes.
        b64_part = url.split(",", 1)[1]
        assert base64.b64decode(b64_part) == b"img-data"

    async def test_image_detail_consumed_not_forwarded(self):
        adapter = VisionModelAdapter(_make_config())
        captured: dict = {}

        async def _fake_chat(messages, **kwargs):
            captured["kwargs"] = kwargs
            return {"content": "ok", "role": "assistant"}

        with patch.object(adapter, "chat", _fake_chat):
            mf = MediaFile(
                media_type=MediaType.IMAGE,
                source_type=MediaSource.BYTES,
                source=b"x",
            )
            await adapter.chat_with_media(
                "p", [mf], image_detail="high", max_tokens=100
            )
        # image_detail should NOT be in forwarded kwargs.
        assert "image_detail" not in captured["kwargs"]
        # max_tokens SHOULD be forwarded.
        assert captured["kwargs"].get("max_tokens") == 100

    async def test_no_prompt_still_works_with_image(self):
        adapter = VisionModelAdapter(_make_config())
        captured: dict = {}

        async def _fake_chat(messages, **kwargs):
            captured["messages"] = messages
            return {"content": "ok", "role": "assistant"}

        with patch.object(adapter, "chat", _fake_chat):
            mf = MediaFile(
                media_type=MediaType.IMAGE,
                source_type=MediaSource.BYTES,
                source=b"x",
            )
            await adapter.chat_with_media("", [mf])
        content = captured["messages"][0]["content"]
        # No text block when prompt is empty string.
        text_blocks = [b for b in content if b["type"] == "text"]
        assert len(text_blocks) == 0
        image_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(image_blocks) == 1


# ---------------------------------------------------------------------------
# chat_with_media - error paths
# ---------------------------------------------------------------------------

class TestChatWithMediaErrors:
    async def test_video_raises_validation_error(self):
        adapter = VisionModelAdapter(_make_config())
        mf = MediaFile(
            media_type=MediaType.VIDEO,
            source_type=MediaSource.LOCAL,
            source="/tmp/clip.mp4",
        )
        with pytest.raises(ValidationError) as exc_info:
            await adapter.chat_with_media("p", [mf])
        assert "Video" in str(exc_info.value) or "video" in str(exc_info.value)

    async def test_audio_raises_validation_error(self):
        adapter = VisionModelAdapter(_make_config())
        mf = MediaFile(
            media_type=MediaType.AUDIO,
            source_type=MediaSource.LOCAL,
            source="/tmp/audio.mp3",
        )
        with pytest.raises(ValidationError) as exc_info:
            await adapter.chat_with_media("p", [mf])
        assert "Audio" in str(exc_info.value) or "audio" in str(exc_info.value)

    async def test_empty_prompt_and_empty_files_raises(self):
        adapter = VisionModelAdapter(_make_config())
        with pytest.raises(ValidationError):
            await adapter.chat_with_media("", [])

    async def test_unsupported_media_type_raises(self):
        adapter = VisionModelAdapter(_make_config())
        # PDF is a MediaType but not supported by vision adapter.
        mf = MediaFile(
            media_type=MediaType.PDF,
            source_type=MediaSource.BYTES,
            source=b"x",
        )
        with pytest.raises(ValidationError):
            await adapter.chat_with_media("p", [mf])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> ModelConfig:
    return ModelConfig(
        model_id="gpt-4o",
        model_name="gpt-4o",
        api_base="http://localhost/v1",
        api_key="test-key",
        max_tokens=1024,
        enabled=True,
        tags=["capable"],
    )
