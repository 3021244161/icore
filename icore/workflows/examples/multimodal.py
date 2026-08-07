"""
icore.workflows.examples.multimodal - Multimodal LLM workflows.

This module provides two multimodal example workflows:

    1. ``ImageCaptionWorkflow`` — Send an image (URL / path / bytes) to a
       vision-capable LLM and produce a textual description.

    2. ``OcrSummaryWorkflow``   — Extract text from an image via OCR
       (ImageProcessor), then summarize it via a text LLM.

Both workflows demonstrate:
    - ``MediaFile`` abstraction across LOCAL / URL / BYTES sources
    - ``VisionModelAdapter.chat_with_media()`` for image inputs
    - ``ImageProcessor`` (Pillow + pytesseract) for OCR
    - DAG composition with media handling tasks

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.multimodal import ImageCaptionWorkflow

    wf = ImageCaptionWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # Inject managers + media_processor before execution...
    result = await wf.execute(ctx, {
        "image_source": "https://example.com/cat.jpg",
        "image_source_type": "url",
        "prompt": "Describe this image in detail.",
    })
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG
from icore.engine.registry import register_workflow
from icore.media import MediaFile, MediaSource, MediaType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared: Build a MediaFile from workflow params
# ---------------------------------------------------------------------------

def _build_media_file(
    image_source: str | bytes,
    image_source_type: str,
    metadata: dict[str, Any] | None = None,
) -> MediaFile:
    """Construct a ``MediaFile`` for an image given source + source_type."""
    source_type_map = {
        "local": MediaSource.LOCAL,
        "url": MediaSource.URL,
        "base64": MediaSource.BASE64,
        "bytes": MediaSource.BYTES,
    }
    if image_source_type not in source_type_map:
        raise ValueError(
            f"Unsupported image_source_type: {image_source_type}. "
            f"Must be one of: {list(source_type_map)}"
        )
    return MediaFile(
        media_type=MediaType.IMAGE,
        source_type=source_type_map[image_source_type],
        source=image_source,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Workflow 1: Image Caption (vision LLM)
# ---------------------------------------------------------------------------

class ImageCaptionInput(BaseTaskInput):
    """Input for the image caption workflow."""

    image_source: str | bytes = Field(
        description="Image source (path / URL / base64 string / raw bytes)",
    )
    image_source_type: str = Field(
        default="url",
        description=(
            "Source type: 'local' (file path), 'url' (HTTP URL), "
            "'base64' (base64-encoded string), or 'bytes' (raw bytes)"
        ),
    )
    prompt: str = Field(
        default="Describe this image in detail.",
        description="The text prompt to send along with the image",
    )
    image_detail: str = Field(
        default="auto",
        description="OpenAI Vision API detail level: low / high / auto",
    )


@register_task("describe_image")
class DescribeImageTask(BaseTask):
    """
    Sends an image to a vision-capable LLM and returns its description.

    This task demonstrates multimodal model usage: it builds a
    ``MediaFile`` from the workflow params and calls
    ``VisionModelAdapter.chat_with_media()``. If the active model adapter
    does not support vision (``supports_vision == False``), the task
    fails with a clear error message.
    """

    name: ClassVar[str] = "describe_image"
    description: ClassVar[str] = (
        "Send an image to a vision LLM and return a textual description"
    )
    input_model: ClassVar[type[BaseTaskInput]] = ImageCaptionInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        self._model = ctx.get_model_adapter()

    async def execute(
        self, ctx: TaskContext, inp: ImageCaptionInput
    ) -> BaseTaskOutput:
        # Build the MediaFile from inputs.
        try:
            media_file = _build_media_file(
                inp.image_source,
                inp.image_source_type,
            )
        except ValueError as e:
            return BaseTaskOutput.failure(str(e))

        # Verify the active model supports vision inputs.
        supports_vision = getattr(self._model, "supports_vision", False)
        if not supports_vision:
            return BaseTaskOutput.failure(
                f"Model '{getattr(self._model, 'model_id', '?')}' does not "
                f"support vision input. Configure a vision-capable model "
                f"(e.g. model_type: 'vision')."
            )

        # Call the multimodal API.
        chat_with_media = getattr(self._model, "chat_with_media", None)
        if chat_with_media is None:
            return BaseTaskOutput.failure(
                "Model adapter lacks chat_with_media() method"
            )

        try:
            async with media_file:
                response = await chat_with_media(
                    prompt=inp.prompt,
                    media_files=[media_file],
                    image_detail=inp.image_detail,
                )
            caption = response.get("content", "")
            logger.info(
                "Generated image caption (%d chars) via vision model",
                len(caption),
            )
            return BaseTaskOutput.success(
                caption=caption,
                prompt=inp.prompt,
                image_source_type=inp.image_source_type,
            )
        except Exception as e:
            logger.error("Image captioning failed: %s", e)
            return BaseTaskOutput.failure(f"Vision call failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        self._model = None


@register_workflow("image_caption")
class ImageCaptionWorkflow(BaseWorkflow):
    """
    Single-task workflow that describes an image via a vision LLM.

    Pipeline:
        1. describe_image:  Send image + prompt to vision model -> caption

    This is the simplest multimodal workflow: it demonstrates how to
    construct a ``MediaFile`` and pass it to a ``VisionModelAdapter``
    via ``chat_with_media()``.
    """

    name: ClassVar[str] = "image_caption"
    description: ClassVar[str] = (
        "Describe an image using a vision-capable LLM"
    )

    def define(self) -> DAG:
        dag = DAG()
        dag.add_node(
            node_id="describe_image",
            task_name="describe_image",
        )
        return dag


# ---------------------------------------------------------------------------
# Workflow 2: OCR + Summary (ImageProcessor + text LLM)
# ---------------------------------------------------------------------------

class OcrExtractInput(BaseTaskInput):
    """Input for the OCR extraction task."""

    image_source: str | bytes = Field(
        description="Image source (path / URL / base64 / raw bytes)",
    )
    image_source_type: str = Field(
        default="local",
        description="Source type: local / url / base64 / bytes",
    )
    ocr_lang: str = Field(
        default="eng",
        description="OCR language code (e.g. 'eng', 'chi_sim')",
    )


class SummarizeOcrInput(BaseTaskInput):
    """Input for the OCR-summary task."""

    ocr_text: str = Field(description="Text extracted via OCR")
    max_length: int = Field(
        default=300,
        ge=50,
        le=2000,
        description="Maximum summary length (words)",
    )


@register_task("ocr_extract")
class OcrExtractTask(BaseTask):
    """
    Extracts text from an image via OCR.

    This task demonstrates media processor integration: it obtains the
    ``MediaProcessorRegistry`` from TaskContext, gets the registered
    ``ImageProcessor``, and calls its ``ocr()`` method.

    Note: OCR requires the optional ``pytesseract`` + ``Pillow`` packages.
    If they are not installed, the task fails with a clear ImportError.
    """

    name: ClassVar[str] = "ocr_extract"
    description: ClassVar[str] = "Extract text from an image via OCR"
    input_model: ClassVar[type[BaseTaskInput]] = OcrExtractInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(
        self, ctx: TaskContext, inp: OcrExtractInput
    ) -> BaseTaskOutput:
        try:
            media_file = _build_media_file(
                inp.image_source,
                inp.image_source_type,
            )
        except ValueError as e:
            return BaseTaskOutput.failure(str(e))

        try:
            registry = ctx.get_media_processor()
        except RuntimeError as e:
            return BaseTaskOutput.failure(str(e))

        processor = registry.get(media_file)

        try:
            async with media_file:
                ocr_text = await processor.ocr(media_file, lang=inp.ocr_lang)
            ocr_text = (ocr_text or "").strip()

            logger.info(
                "OCR extracted %d chars from image (lang=%s)",
                len(ocr_text),
                inp.ocr_lang,
            )

            return BaseTaskOutput.success(
                ocr_text=ocr_text,
                ocr_lang=inp.ocr_lang,
                image_source_type=inp.image_source_type,
            )
        except ImportError as e:
            logger.error("OCR dependencies missing: %s", e)
            return BaseTaskOutput.failure(
                f"OCR dependency missing: {e}. "
                f"Install pytesseract + Pillow to use this workflow."
            )
        except Exception as e:
            logger.error("OCR extraction failed: %s", e)
            return BaseTaskOutput.failure(f"OCR failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


@register_task("summarize_ocr")
class SummarizeOcrTask(BaseTask):
    """
    Summarizes OCR-extracted text via a text LLM.

    Sends the OCR text to the LLM with a summarization prompt and
    returns a concise summary. Useful for scanned documents, receipts,
    whiteboard photos, etc.
    """

    name: ClassVar[str] = "summarize_ocr"
    description: ClassVar[str] = "Summarize OCR-extracted text via LLM"
    input_model: ClassVar[type[BaseTaskInput]] = SummarizeOcrInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        self._model = ctx.get_model_adapter()

    async def execute(
        self, ctx: TaskContext, inp: SummarizeOcrInput
    ) -> BaseTaskOutput:
        if not inp.ocr_text:
            return BaseTaskOutput.success(
                summary="[No text was extracted by OCR; nothing to summarize.]",
                ocr_text_length=0,
            )

        prompt = (
            f"Please summarize the following OCR-extracted text in no more "
            f"than {inp.max_length} words. Correct obvious OCR errors and "
            f"preserve the key information:\n\n{inp.ocr_text}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional summarization assistant for "
                    "OCR-extracted text. Fix OCR errors when obvious."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._model.chat(messages=messages)
            summary = response.get("content", "")
            logger.info(
                "Summarized OCR text (%d chars -> %d chars summary)",
                len(inp.ocr_text),
                len(summary),
            )
            return BaseTaskOutput.success(
                summary=summary,
                ocr_text_length=len(inp.ocr_text),
            )
        except Exception as e:
            logger.error("OCR summary failed: %s", e)
            return BaseTaskOutput.failure(f"Summary failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        self._model = None


@register_workflow("ocr_summary")
class OcrSummaryWorkflow(BaseWorkflow):
    """
    OCR + LLM summary workflow.

    Pipeline:
        1. ocr_extract:   Extract text from an image via OCR
        2. summarize_ocr: Summarize the OCR text via LLM

    DAG:
        ocr_extract -> summarize_ocr

    This workflow demonstrates media processor + LLM integration: the
    first task uses the ``ImageProcessor`` to extract text from an
    image (scanned document, receipt, whiteboard photo, etc.), and the
    second uses a text LLM to summarize the extracted text.
    """

    name: ClassVar[str] = "ocr_summary"
    description: ClassVar[str] = (
        "Extract text from an image via OCR -> summarize via LLM"
    )

    def define(self) -> DAG:
        dag = DAG()

        dag.add_node(
            node_id="ocr_extract",
            task_name="ocr_extract",
        )

        dag.add_node(
            node_id="summarize_ocr",
            task_name="summarize_ocr",
            input_builder=lambda params, upstream: SummarizeOcrInput(
                ocr_text=upstream["ocr_extract"].data.get("ocr_text", ""),
                max_length=params.get("max_length", 300),
            ),
        )

        dag.add_edge("ocr_extract", "summarize_ocr")
        return dag
