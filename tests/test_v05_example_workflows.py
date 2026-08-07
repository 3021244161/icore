"""
End-to-end tests for the v0.5 example workflows: RAG QA, Knowledge Graph,
and Multimodal (image caption + OCR summary).

These tests exercise the new v0.5 components (vectorstore, graphstore,
media processor, distributed lock, vision adapter) end-to-end through
the workflow pipeline, using in-memory implementations and a fake
vision model adapter (no network access required).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from icore.core.task_context import TaskContext
from icore.engine.registry import workflow_registry
from icore.graphstore import InMemoryGraphStore, GraphNode, GraphEdge
from icore.engine.lock import MemoryLock
from icore.media import (
    MediaFile,
    MediaSource,
    MediaType,
    MediaProcessorRegistry,
)
from icore.models.vision_adapter import VisionModelAdapter
from icore.vectorstore import InMemoryVectorStore, VectorDocument
from tests.conftest import FakeModelAdapter, make_model_config


# Ensure example workflows are imported/registered.
import icore.workflows.examples  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx_with_model(model_manager) -> TaskContext:
    ctx = TaskContext(task_id="t-test", workflow_id="wf-test")
    ctx.set_model_manager(model_manager)
    return ctx


def _ctx_with_model_and_vectorstore(model_manager, vectorstore) -> TaskContext:
    ctx = TaskContext(task_id="t-test", workflow_id="wf-test")
    ctx.set_model_manager(model_manager)
    ctx.set_vectorstore(vectorstore)
    return ctx


def _ctx_with_model_graphstore_lock(
    model_manager, graphstore, lock
) -> TaskContext:
    ctx = TaskContext(task_id="t-test", workflow_id="wf-test")
    ctx.set_model_manager(model_manager)
    ctx.set_graphstore(graphstore)
    ctx.set_lock(lock)
    return ctx


def _ctx_with_model_and_media(model_manager, media_registry) -> TaskContext:
    ctx = TaskContext(task_id="t-test", workflow_id="wf-test")
    ctx.set_model_manager(model_manager)
    ctx.set_media_processor(media_registry)
    return ctx


class _FakeVisionAdapter(FakeModelAdapter):
    """FakeModelAdapter that also reports supports_vision=True."""

    @property
    def supports_vision(self) -> bool:
        return True

    async def chat_with_media(
        self,
        prompt: str,
        media_files: list[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Reuse the responder if set, else echo prompt.
        if self._responder is not None:
            content = self._responder({"prompt": prompt, "media_files": media_files})
        else:
            n_images = sum(
                1 for mf in media_files if getattr(mf, "media_type", None) == MediaType.IMAGE
            )
            content = f"VISION[{n_images}img]: {prompt[:50]}"
        return {
            "content": content,
            "role": "assistant",
            "model": self.model_id,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "finish_reason": "stop",
        }


# ---------------------------------------------------------------------------
# Registration smoke tests
# ---------------------------------------------------------------------------

class TestV05Registration:
    def test_rag_qa_workflow_registered(self):
        names = workflow_registry.list_workflows()
        assert "rag_qa" in names

    def test_knowledge_graph_workflow_registered(self):
        names = workflow_registry.list_workflows()
        assert "knowledge_graph" in names

    def test_multimodal_workflows_registered(self):
        names = workflow_registry.list_workflows()
        assert "image_caption" in names
        assert "ocr_summary" in names

    def test_rag_qa_tasks_registered(self):
        from icore.core.registry import task_registry
        assert "embed_query" in task_registry
        assert "retrieve_docs" in task_registry
        assert "generate_rag_answer" in task_registry

    def test_knowledge_graph_tasks_registered(self):
        from icore.core.registry import task_registry
        assert "extract_entities_for_graph" in task_registry
        assert "build_graph" in task_registry
        assert "query_graph" in task_registry

    def test_multimodal_tasks_registered(self):
        from icore.core.registry import task_registry
        assert "describe_image" in task_registry
        assert "ocr_extract" in task_registry
        assert "summarize_ocr" in task_registry


# ---------------------------------------------------------------------------
# RAG QA workflow
# ---------------------------------------------------------------------------

class TestRAGQAWorkflow:
    async def test_dag_validates(self):
        from icore.workflows.examples.rag_qa import RAGQAWorkflow
        wf = RAGQAWorkflow()
        assert wf.validate() is True

    async def test_end_to_end_with_retrieved_docs(self, fake_model_manager):
        from icore.workflows.examples.rag_qa import RAGQAWorkflow

        # Build an in-memory vector store and seed it with documents.
        vs = InMemoryVectorStore()
        await vs.create_collection("doc_embeddings", dim=4)
        await vs.insert(
            "doc_embeddings",
            [
                VectorDocument(
                    id="d1",
                    vector=[1.0, 0.0, 0.0, 0.0],
                    text="Refund policy: refunds within 30 days.",
                    metadata={"source": "policy.md"},
                ),
                VectorDocument(
                    id="d2",
                    vector=[0.0, 1.0, 0.0, 0.0],
                    text="Shipping takes 3-5 business days.",
                    metadata={"source": "shipping.md"},
                ),
            ],
        )

        # Build a fake adapter whose embed() returns a vector that
        # matches d1 (highest cosine similarity).
        config = make_model_config("fake-model")
        adapter = FakeModelAdapter(
            config,
            responder=lambda msgs: "SUMMARY: refund answer",
        )

        async def _embed(texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        adapter.embed = _embed
        fake_model_manager._adapters["fake-model"] = adapter

        wf = RAGQAWorkflow()
        ctx = _ctx_with_model_and_vectorstore(fake_model_manager, vs)

        result = await wf.execute(
            ctx,
            {
                "question": "What is the refund policy?",
                "collection": "doc_embeddings",
                "top_k": 2,
            },
        )

        assert result.is_success is True, f"Failed: {result.error}"
        assert "answer" in result.data
        assert result.data["question"] == "What is the refund policy?"
        # Should have retrieved at least 1 doc.
        assert result.data["context_count"] >= 1
        # The most-similar doc should be d1.
        assert "d1" in result.data.get("used_context_ids", [])

    async def test_no_vectorstore_injected_fails_gracefully(
        self, fake_model_manager
    ):
        from icore.workflows.examples.rag_qa import RAGQAWorkflow

        wf = RAGQAWorkflow()
        # No vectorstore injected.
        ctx = _ctx_with_model(fake_model_manager)

        result = await wf.execute(
            ctx,
            {"question": "What is the refund policy?"},
        )

        # The retrieve_docs task should fail because no vectorstore.
        assert result.is_success is False

    async def test_empty_question_fails_gracefully(self, fake_model_manager):
        from icore.workflows.examples.rag_qa import RAGQAWorkflow

        vs = InMemoryVectorStore()
        wf = RAGQAWorkflow()
        ctx = _ctx_with_model_and_vectorstore(fake_model_manager, vs)

        result = await wf.execute(ctx, {"question": ""})
        assert result.is_success is False

    async def test_no_retrieved_docs_still_answers(self, fake_model_manager):
        from icore.workflows.examples.rag_qa import RAGQAWorkflow

        # Empty vector store — no documents will be retrieved.
        vs = InMemoryVectorStore()
        await vs.create_collection("empty", dim=4)

        # Adapter that returns a vector that won't match anything.
        config = make_model_config("fake-model")
        adapter = FakeModelAdapter(config)

        async def _embed(texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        adapter.embed = _embed
        fake_model_manager._adapters["fake-model"] = adapter

        wf = RAGQAWorkflow()
        ctx = _ctx_with_model_and_vectorstore(fake_model_manager, vs)

        result = await wf.execute(
            ctx,
            {"question": "anything", "collection": "empty", "top_k": 5},
        )

        assert result.is_success is True
        assert result.data["context_count"] == 0
        assert "answer" in result.data


# ---------------------------------------------------------------------------
# Knowledge Graph workflow
# ---------------------------------------------------------------------------

class TestKnowledgeGraphWorkflow:
    async def test_dag_validates(self):
        from icore.workflows.examples.knowledge_graph import (
            KnowledgeGraphWorkflow,
        )
        wf = KnowledgeGraphWorkflow()
        assert wf.validate() is True

    async def test_end_to_end_build_and_query(self, fake_model_manager):
        from icore.workflows.examples.knowledge_graph import (
            KnowledgeGraphWorkflow,
        )

        # LLM returns entities + relationships as JSON.
        extraction_payload = {
            "entities": [
                {"name": "Alice", "type": "person"},
                {"name": "Acme Corp", "type": "organization"},
                {"name": "Seattle", "type": "location"},
            ],
            "relationships": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"},
                {"subject": "Acme Corp", "predicate": "based_in", "object": "Seattle"},
            ],
        }

        # First call returns the extraction JSON; subsequent calls
        # return answer text.
        call_count = {"n": 0}

        def responder(messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return json.dumps(extraction_payload)
            return "Acme Corp is based in Seattle."

        config = make_model_config("fake-model")
        adapter = FakeModelAdapter(config, responder=responder)
        fake_model_manager._adapters["fake-model"] = adapter

        gs = InMemoryGraphStore()
        lock = MemoryLock()
        wf = KnowledgeGraphWorkflow()
        ctx = _ctx_with_model_graphstore_lock(fake_model_manager, gs, lock)

        result = await wf.execute(
            ctx,
            {
                "text": "Alice works at Acme Corp. Acme Corp is based in Seattle.",
                "question": "Where is Acme Corp based?",
                "depth": 2,
            },
        )

        assert result.is_success is True, f"Failed: {result.error}"
        assert "answer" in result.data
        # The graph should have been built.
        assert result.data["node_count"] >= 2
        assert result.data["edge_count"] >= 1

        # Verify the graph store now contains the upserted nodes.
        all_nodes = await gs.query("MATCH (n) RETURN n")
        node_ids = {n["n"]["id"] for n in all_nodes}
        assert "Alice" in node_ids
        assert "Acme Corp" in node_ids

    async def test_no_lock_injected_still_works(self, fake_model_manager):
        """The build_graph task should still succeed without a lock."""
        from icore.workflows.examples.knowledge_graph import (
            KnowledgeGraphWorkflow,
        )

        extraction_payload = {
            "entities": [{"name": "X", "type": "thing"}],
            "relationships": [],
        }

        call_count = {"n": 0}

        def responder(messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return json.dumps(extraction_payload)
            return "answer"

        config = make_model_config("fake-model")
        adapter = FakeModelAdapter(config, responder=responder)
        fake_model_manager._adapters["fake-model"] = adapter

        gs = InMemoryGraphStore()
        # Note: NO lock injected.
        ctx = TaskContext(task_id="t", workflow_id="w")
        ctx.set_model_manager(fake_model_manager)
        ctx.set_graphstore(gs)

        wf = KnowledgeGraphWorkflow()
        result = await wf.execute(
            ctx,
            {"text": "X exists.", "question": "what is X?"},
        )
        assert result.is_success is True
        assert result.data["node_count"] >= 1

    async def test_empty_text_fails_gracefully(self, fake_model_manager):
        from icore.workflows.examples.knowledge_graph import (
            KnowledgeGraphWorkflow,
        )

        gs = InMemoryGraphStore()
        lock = MemoryLock()
        wf = KnowledgeGraphWorkflow()
        ctx = _ctx_with_model_graphstore_lock(fake_model_manager, gs, lock)

        result = await wf.execute(
            ctx,
            {"text": "", "question": "anything"},
        )
        assert result.is_success is False

    async def test_no_graphstore_injected_fails(self, fake_model_manager):
        from icore.workflows.examples.knowledge_graph import (
            KnowledgeGraphWorkflow,
        )

        wf = KnowledgeGraphWorkflow()
        ctx = _ctx_with_model(fake_model_manager)
        # No graphstore injected.
        result = await wf.execute(
            ctx,
            {"text": "Alice works at Acme.", "question": "who?"},
        )
        # extract succeeds but build_graph fails.
        assert result.is_success is False


# ---------------------------------------------------------------------------
# Multimodal: Image Caption workflow
# ---------------------------------------------------------------------------

class TestImageCaptionWorkflow:
    async def test_dag_validates(self):
        from icore.workflows.examples.multimodal import ImageCaptionWorkflow
        wf = ImageCaptionWorkflow()
        assert wf.validate() is True

    async def test_end_to_end_with_vision_adapter(self, fake_model_manager):
        from icore.workflows.examples.multimodal import ImageCaptionWorkflow

        # Replace the fake adapter with a fake vision adapter.
        vision_adapter = _FakeVisionAdapter(make_model_config("fake-model"))
        fake_model_manager._adapters["fake-model"] = vision_adapter

        wf = ImageCaptionWorkflow()
        ctx = _ctx_with_model(fake_model_manager)

        result = await wf.execute(
            ctx,
            {
                "image_source": b"fake-image-bytes",
                "image_source_type": "bytes",
                "prompt": "Describe this image.",
            },
        )

        assert result.is_success is True, f"Failed: {result.error}"
        assert "caption" in result.data
        assert "VISION" in result.data["caption"]

    async def test_non_vision_model_fails_clearly(self, fake_model_manager):
        """If the model lacks supports_vision, the task should fail clearly."""
        from icore.workflows.examples.multimodal import ImageCaptionWorkflow

        # Default FakeModelAdapter does NOT support vision.
        wf = ImageCaptionWorkflow()
        ctx = _ctx_with_model(fake_model_manager)

        result = await wf.execute(
            ctx,
            {
                "image_source": b"img",
                "image_source_type": "bytes",
                "prompt": "describe",
            },
        )

        assert result.is_success is False
        assert "vision" in result.error.lower()

    async def test_invalid_source_type_fails(self, fake_model_manager):
        from icore.workflows.examples.multimodal import ImageCaptionWorkflow

        vision_adapter = _FakeVisionAdapter(make_model_config("fake-model"))
        fake_model_manager._adapters["fake-model"] = vision_adapter

        wf = ImageCaptionWorkflow()
        ctx = _ctx_with_model(fake_model_manager)

        result = await wf.execute(
            ctx,
            {
                "image_source": "foo",
                "image_source_type": "invalid_type",
                "prompt": "describe",
            },
        )

        assert result.is_success is False
        assert "image_source_type" in result.error or "unsupported" in result.error.lower()


# ---------------------------------------------------------------------------
# Multimodal: OCR Summary workflow
# ---------------------------------------------------------------------------

class TestOcrSummaryWorkflow:
    async def test_dag_validates(self):
        from icore.workflows.examples.multimodal import OcrSummaryWorkflow
        wf = OcrSummaryWorkflow()
        assert wf.validate() is True

    async def test_end_to_end_with_mocked_ocr(self, fake_model_manager):
        """OCR requires pytesseract; we mock the ImageProcessor to avoid
        the heavy dependency in tests."""
        from icore.workflows.examples.multimodal import OcrSummaryWorkflow
        from icore.media import ImageProcessor

        # Build a media registry with a mocked ImageProcessor.
        registry = MediaProcessorRegistry()
        image_proc = ImageProcessor()

        # Mock the ocr() method to return a fixed string.
        async def _fake_ocr(file, lang="eng"):
            return "Invoice #1234. Total: $99.99. Date: 2026-08-01."

        image_proc.ocr = _fake_ocr
        registry.register(MediaType.IMAGE, image_proc)

        wf = OcrSummaryWorkflow()
        ctx = _ctx_with_model_and_media(fake_model_manager, registry)

        result = await wf.execute(
            ctx,
            {
                "image_source": b"fake-invoice-image",
                "image_source_type": "bytes",
                "ocr_lang": "eng",
                "max_length": 100,
            },
        )

        assert result.is_success is True, f"Failed: {result.error}"
        assert "summary" in result.data
        # FakeModelAdapter echoes "SUMMARY: ..."
        assert "SUMMARY:" in result.data["summary"]

    async def test_no_media_processor_fails(self, fake_model_manager):
        from icore.workflows.examples.multimodal import OcrSummaryWorkflow

        wf = OcrSummaryWorkflow()
        # No media processor injected.
        ctx = _ctx_with_model(fake_model_manager)

        result = await wf.execute(
            ctx,
            {
                "image_source": b"img",
                "image_source_type": "bytes",
            },
        )

        assert result.is_success is False

    async def test_empty_ocr_text_still_succeeds(self, fake_model_manager):
        """When OCR returns empty text, the summary task should still
        return a graceful message."""
        from icore.workflows.examples.multimodal import OcrSummaryWorkflow
        from icore.media import ImageProcessor

        registry = MediaProcessorRegistry()
        image_proc = ImageProcessor()

        async def _empty_ocr(file, lang="eng"):
            return ""

        image_proc.ocr = _empty_ocr
        registry.register(MediaType.IMAGE, image_proc)

        wf = OcrSummaryWorkflow()
        ctx = _ctx_with_model_and_media(fake_model_manager, registry)

        result = await wf.execute(
            ctx,
            {
                "image_source": b"blank-image",
                "image_source_type": "bytes",
            },
        )

        assert result.is_success is True
        assert "No text" in result.data["summary"] or "nothing" in result.data["summary"].lower()

    async def test_invalid_source_type_fails(self, fake_model_manager):
        from icore.workflows.examples.multimodal import OcrSummaryWorkflow
        from icore.media import ImageProcessor

        registry = MediaProcessorRegistry()
        registry.register(MediaType.IMAGE, ImageProcessor())

        wf = OcrSummaryWorkflow()
        ctx = _ctx_with_model_and_media(fake_model_manager, registry)

        result = await wf.execute(
            ctx,
            {
                "image_source": "x",
                "image_source_type": "bad_type",
            },
        )

        assert result.is_success is False
