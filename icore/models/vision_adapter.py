"""
icore.models.vision_adapter - Multimodal (vision) LLM adapter.

Extends ``OpenAICompatibleAdapter`` with vision input support. Builds
the OpenAI Vision API message format (``image_url`` content blocks)
from ``MediaFile`` instances and delegates the actual HTTP call to
the parent ``chat()``.

Supports:
    - GPT-4V / GPT-4o / GPT-4 Turbo Vision (OpenAI compatible)
    - Claude 3.5 Sonnet vision (OpenAI compatible via proxy / Anthropic API)
    - Gemini Pro Vision (OpenAI compatible proxy)

The vision adapter refuses video and audio inputs; callers must
pre-process video frames (``VideoProcessor.extract_keyframes``) and
audio (``AudioProcessor.transcribe``) before passing them in.
"""

from __future__ import annotations

import logging
from typing import Any

from icore.exceptions import ValidationError
from icore.media import MediaFile, MediaType
from icore.models.openai_adapter import OpenAICompatibleAdapter

logger = logging.getLogger(__name__)


class VisionModelAdapter(OpenAICompatibleAdapter):
    """
    Multimodal vision model adapter.

    Inherits all text/chat/embed behaviour from ``OpenAICompatibleAdapter``
    and adds ``chat_with_media()`` for image inputs.
    """

    @property
    def supports_vision(self) -> bool:
        """This adapter accepts image inputs."""
        return True

    async def chat_with_media(
        self,
        prompt: str,
        media_files: list[MediaFile],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Multimodal chat completion.

        Constructs an OpenAI Vision-style ``user`` message whose
        ``content`` is a list of text and image_url blocks, then
        delegates to the parent ``chat()``.

        Args:
            prompt:       User text prompt.
            media_files:  List of ``MediaFile`` instances (images only;
                          video/audio must be pre-processed).
            **kwargs:     Additional API parameters (max_tokens,
                          temperature, image_detail, etc.).

        Returns:
            The unified chat response dict from ``chat()``.

        Raises:
            ValidationError: If a video/audio MediaFile is supplied
                             without pre-processing.
        """
        if not prompt and not media_files:
            raise ValidationError(
                "chat_with_media requires at least a prompt or media input"
            )

        image_detail = kwargs.pop("image_detail", "auto")

        content: list[dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})

        for mf in media_files:
            if mf.media_type == MediaType.IMAGE:
                b64 = await mf.to_base64()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mf.mime_type};base64,{b64}",
                            "detail": image_detail,
                        },
                    }
                )
            elif mf.media_type == MediaType.VIDEO:
                raise ValidationError(
                    "Video files must be pre-processed into frames. "
                    "Use VideoProcessor.extract_keyframes() first."
                )
            elif mf.media_type == MediaType.AUDIO:
                raise ValidationError(
                    "Audio files must be transcribed first. "
                    "Use AudioProcessor.transcribe() then pass text."
                )
            else:
                raise ValidationError(
                    f"Unsupported media_type for vision adapter: {mf.media_type}"
                )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": content}
        ]
        return await self.chat(messages, **kwargs)


__all__ = ["VisionModelAdapter"]
