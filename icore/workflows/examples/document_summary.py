"""
icore.workflows.examples.document_summary - Large document summarization.

A multi-task workflow that summarizes a large document by:
    1. Chunking the document into smaller pieces
    2. Summarizing each chunk via LLM
    3. Merging the chunk summaries into a final summary

This demonstrates:
    - Custom task input classes (DocumentSummaryInput, etc.)
    - Two concrete tasks (TextChunkerTask, SummarizeChunkTask, MergeSummaryTask)
    - DAG composition with linear dependency chain
    - Model adapter usage via TaskContext.get_model_adapter()
    - Input builders to pass data between tasks

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.document_summary import DocumentSummaryWorkflow

    wf = DocumentSummaryWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # Inject managers before execution...
    result = await wf.execute(ctx, {"document": "Long text here..."})
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Input Models
# ---------------------------------------------------------------------------

class ChunkDocumentInput(BaseTaskInput):
    """Input for the text chunking task."""

    document: str = Field(description="The full document text to chunk")
    chunk_size: int = Field(
        default=2000,
        ge=100,
        description="Maximum characters per chunk",
    )
    overlap: int = Field(
        default=100,
        ge=0,
        description="Character overlap between adjacent chunks",
    )


class SummarizeChunkInput(BaseTaskInput):
    """Input for the per-chunk summarization task."""

    chunks: list[str] = Field(
        description="List of text chunks to summarize individually",
    )
    max_length_per_chunk: int = Field(
        default=500,
        ge=50,
        description="Maximum summary length per chunk (in words)",
    )


class MergeSummaryInput(BaseTaskInput):
    """Input for the merge task."""

    chunk_summaries: list[str] = Field(
        description="List of per-chunk summaries to merge",
    )
    final_max_length: int = Field(
        default=1000,
        ge=100,
        description="Maximum final summary length (in words)",
    )


# ---------------------------------------------------------------------------
# Task: Text Chunking
# ---------------------------------------------------------------------------

@register_task("text_chunker")
class TextChunkerTask(BaseTask):
    """
    Splits a large document into overlapping text chunks.

    This is a non-LLM task: it performs pure text processing to break
    the document into manageable pieces for downstream summarization.
    """

    name: ClassVar[str] = "text_chunker"
    description: ClassVar[str] = "Split a large document into overlapping text chunks"
    input_model: ClassVar[type[BaseTaskInput]] = ChunkDocumentInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        """No external resources needed for chunking."""
        pass

    async def execute(
        self, ctx: TaskContext, inp: ChunkDocumentInput
    ) -> BaseTaskOutput:
        """Split the document into overlapping chunks."""
        document = inp.document
        chunk_size = inp.chunk_size
        overlap = inp.overlap

        if not document:
            return BaseTaskOutput.failure("Document is empty")

        chunks: list[str] = []
        start = 0
        while start < len(document):
            end = start + chunk_size
            chunk = document[start:end]
            chunks.append(chunk)
            # Move start forward, accounting for overlap
            start = end - overlap if end < len(document) else len(document)

        logger.info(
            "Chunked document into %d pieces (chunk_size=%d, overlap=%d)",
            len(chunks),
            chunk_size,
            overlap,
        )

        return BaseTaskOutput.success(
            chunks=chunks,
            chunk_count=len(chunks),
            original_length=len(document),
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Chunk Summarization (LLM)
# ---------------------------------------------------------------------------

@register_task("summarize_chunk")
class SummarizeChunkTask(BaseTask):
    """
    Summarizes each chunk via the LLM model adapter.

    This task demonstrates LLM usage: it obtains the model adapter from
    TaskContext, sends each chunk to the model with a summarization prompt,
    and collects the results.

    For streaming mode (ctx.stream == True), this task could yield
    incremental results as each chunk is summarized.
    """

    name: ClassVar[str] = "summarize_chunk"
    description: ClassVar[str] = "Summarize each text chunk via LLM"
    input_model: ClassVar[type[BaseTaskInput]] = SummarizeChunkInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any  # Model adapter (set in prepare)

    async def prepare(self, ctx: TaskContext) -> None:
        """Obtain the model adapter from context."""
        self._model = ctx.get_model_adapter()
        logger.debug("SummarizeChunkTask prepared with model adapter")

    async def execute(
        self, ctx: TaskContext, inp: SummarizeChunkInput
    ) -> BaseTaskOutput:
        """Summarize each chunk via the LLM."""
        if not inp.chunks:
            return BaseTaskOutput.failure("No chunks to summarize")

        summaries: list[str] = []
        max_len = inp.max_length_per_chunk

        for i, chunk in enumerate(inp.chunks):
            prompt = (
                f"Please summarize the following text in no more than "
                f"{max_len} words. Preserve the key information:\n\n"
                f"{chunk}"
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional text summarization assistant.",
                },
                {"role": "user", "content": prompt},
            ]

            try:
                response = await self._model.chat(messages=messages)
                summary = response.get("content", "")
                summaries.append(summary)
                logger.debug(
                    "Summarized chunk %d/%d (%d chars -> %d chars)",
                    i + 1,
                    len(inp.chunks),
                    len(chunk),
                    len(summary),
                )
            except Exception as e:
                logger.error("Failed to summarize chunk %d: %s", i + 1, e)
                # Use a placeholder so the merge task can still proceed
                summaries.append(f"[Summary failed for chunk {i + 1}: {e}]")

        return BaseTaskOutput.success(
            chunk_summaries=summaries,
            total_chunks=len(summaries),
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        """Release model adapter reference."""
        self._model = None


# ---------------------------------------------------------------------------
# Task: Merge Summaries (LLM)
# ---------------------------------------------------------------------------

@register_task("merge_summary")
class MergeSummaryTask(BaseTask):
    """
    Merges per-chunk summaries into a cohesive final summary via LLM.

    This task sends all chunk summaries to the LLM with a merge prompt,
    producing a single coherent summary of the entire document.
    """

    name: ClassVar[str] = "merge_summary"
    description: ClassVar[str] = "Merge per-chunk summaries into a final summary"
    input_model: ClassVar[type[BaseTaskInput]] = MergeSummaryInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        """Obtain the model adapter."""
        self._model = ctx.get_model_adapter()

    async def execute(
        self, ctx: TaskContext, inp: MergeSummaryInput
    ) -> BaseTaskOutput:
        """Merge chunk summaries into a final summary."""
        if not inp.chunk_summaries:
            return BaseTaskOutput.failure("No chunk summaries to merge")

        # Combine all summaries into a single prompt
        combined = "\n\n---\n\n".join(
            f"[Section {i + 1}]\n{s}"
            for i, s in enumerate(inp.chunk_summaries)
        )

        prompt = (
            f"Below are summaries of different sections of a document. "
            f"Please merge them into a single coherent summary of no more "
            f"than {inp.final_max_length} words. Remove redundancies and "
            f"ensure logical flow:\n\n{combined}"
        )
        messages = [
            {
                "role": "system",
                "content": "You are a professional document summarization assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._model.chat(messages=messages)
            final_summary = response.get("content", "")
            logger.info(
                "Merged %d chunk summaries into final summary (%d chars)",
                len(inp.chunk_summaries),
                len(final_summary),
            )
            return BaseTaskOutput.success(
                summary=final_summary,
                merged_from=len(inp.chunk_summaries),
            )
        except Exception as e:
            logger.error("Merge summarization failed: %s", e)
            return BaseTaskOutput.failure(f"Merge failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Workflow: Document Summary
# ---------------------------------------------------------------------------

@register_workflow("document_summary")
class DocumentSummaryWorkflow(BaseWorkflow):
    """
    Multi-task workflow for large document summarization.

    Pipeline:
        1. text_chunker:  Split document -> chunks
        2. summarize_chunk: Summarize each chunk via LLM
        3. merge_summary: Merge chunk summaries -> final summary

    DAG:
        chunk -> summarize -> merge
    """

    name: ClassVar[str] = "document_summary"
    description: ClassVar[str] = (
        "Summarize a large document via chunking, per-chunk LLM "
        "summarization, and merge"
    )

    def define(self) -> DAG:
        """Build the chunk -> summarize -> merge DAG."""
        dag = DAG()

        # Node 1: Chunk the document
        dag.add_node(
            node_id="chunk",
            task_name="text_chunker",
        )

        # Node 2: Summarize each chunk
        # Input builder: takes upstream output (chunks) and constructs
        # the SummarizeChunkInput with the chunk list
        dag.add_node(
            node_id="summarize",
            task_name="summarize_chunk",
            input_builder=lambda params, upstream: SummarizeChunkInput(
                chunks=upstream["chunk"].data["chunks"],
                max_length_per_chunk=params.get("max_length_per_chunk", 500),
            ),
        )

        # Node 3: Merge summaries
        # Input builder: takes the summarized chunks and constructs
        # the MergeSummaryInput
        dag.add_node(
            node_id="merge",
            task_name="merge_summary",
            input_builder=lambda params, upstream: MergeSummaryInput(
                chunk_summaries=upstream["summarize"].data["chunk_summaries"],
                final_max_length=params.get("final_max_length", 1000),
            ),
        )

        # Linear dependency chain
        dag.add_edge("chunk", "summarize")
        dag.add_edge("summarize", "merge")

        return dag
