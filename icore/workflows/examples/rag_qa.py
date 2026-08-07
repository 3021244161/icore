"""
icore.workflows.examples.rag_qa - Retrieval-Augmented Generation QA.

A workflow that answers a user question by retrieving relevant document
chunks from a vector store and feeding them to the LLM as context:

    1. Embed:        Convert the user's question into a vector via LLM
    2. Retrieve:     Search the vector store for the top-K most similar
                     documents (RAG context)
    3. Generate:     Send the question + retrieved context to the LLM
                     and produce a grounded answer

This demonstrates:
    - VectorStore integration via TaskContext.get_vectorstore()
    - ModelAdapter.embed() for query vectorization
    - LLM generation grounded by retrieved context
    - DAG composition: embed_query -> retrieve_docs -> generate_answer

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.rag_qa import RAGQAWorkflow

    wf = RAGQAWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # Inject managers + vectorstore before execution...
    result = await wf.execute(ctx, {
        "question": "What is the refund policy?",
        "collection": "doc_embeddings",
        "top_k": 5,
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Input Models
# ---------------------------------------------------------------------------

class EmbedQueryInput(BaseTaskInput):
    """Input for the query embedding task."""

    question: str = Field(description="The user's question to embed")
    model_id: str | None = Field(
        default=None,
        description="Optional override model for embedding",
    )


class RetrieveDocsInput(BaseTaskInput):
    """Input for the document retrieval task."""

    query_vector: list[float] = Field(
        description="Embedding vector of the user's question",
    )
    collection: str = Field(
        default="doc_embeddings",
        description="Vector store collection to search",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of documents to retrieve",
    )
    filter_expr: str | None = Field(
        default=None,
        description="Optional metadata filter expression",
    )
    question: str | None = Field(
        default=None,
        description=(
            "Original user question text. Required when use_hybrid=True "
            "for BM25 sparse retrieval; ignored by pure dense search."
        ),
    )
    use_hybrid: bool = Field(
        default=False,
        description=(
            "Whether to use HybridRetriever (Dense + BM25 + Rerank). "
            "When True, requires bm25_retriever and reranker to be "
            "available via ctx.metadata; falls back to pure dense "
            "search if either is missing."
        ),
    )


class GenerateAnswerInput(BaseTaskInput):
    """Input for the answer generation task."""

    question: str = Field(description="The user's original question")
    contexts: list[dict[str, Any]] = Field(
        description="Retrieved document chunks used as grounding context",
    )
    max_context_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of context chunks to include in the prompt",
    )


# ---------------------------------------------------------------------------
# Task: Embed Query (LLM)
# ---------------------------------------------------------------------------

@register_task("embed_query")
class EmbedQueryTask(BaseTask):
    """
    Embeds the user's question into a vector via the LLM model adapter.

    This task obtains the model adapter from TaskContext and calls its
    ``embed()`` method to convert the question text into a float vector
    suitable for vector store similarity search.
    """

    name: ClassVar[str] = "embed_query"
    description: ClassVar[str] = "Embed the user question into a vector"
    input_model: ClassVar[type[BaseTaskInput]] = EmbedQueryInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        """Obtain the model adapter from context."""
        self._model = ctx.get_model_adapter()
        logger.debug("EmbedQueryTask prepared with model adapter")

    async def execute(
        self, ctx: TaskContext, inp: EmbedQueryInput
    ) -> BaseTaskOutput:
        """Embed the question via the LLM adapter."""
        if not inp.question:
            return BaseTaskOutput.failure("Question is empty")

        try:
            vectors = await self._model.embed([inp.question])
            if not vectors:
                return BaseTaskOutput.failure(
                    "Model returned no embedding vectors"
                )
            query_vector = vectors[0]
            logger.info(
                "Embedded question (%d chars -> %d-dim vector)",
                len(inp.question),
                len(query_vector),
            )
            return BaseTaskOutput.success(
                query_vector=query_vector,
                question=inp.question,
            )
        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            return BaseTaskOutput.failure(f"Embedding failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Task: Retrieve Documents (VectorStore)
# ---------------------------------------------------------------------------

@register_task("retrieve_docs")
class RetrieveDocsTask(BaseTask):
    """
    Retrieves relevant document chunks from the vector store.

    This task demonstrates vector store integration: it obtains a
    ``BaseVectorStore`` instance via ``ctx.get_vectorstore()`` and
    performs an ANN similarity search using the query vector.
    """

    name: ClassVar[str] = "retrieve_docs"
    description: ClassVar[str] = "Retrieve documents from vector store"
    input_model: ClassVar[type[BaseTaskInput]] = RetrieveDocsInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        """No persistent state needed; vectorstore is fetched per-call."""
        pass

    async def execute(
        self, ctx: TaskContext, inp: RetrieveDocsInput
    ) -> BaseTaskOutput:
        """Search the vector store for similar documents."""
        if not inp.query_vector:
            return BaseTaskOutput.failure("Query vector is empty")

        try:
            vectorstore = ctx.get_vectorstore()
        except RuntimeError as e:
            return BaseTaskOutput.failure(str(e))

        # 当 use_hybrid=True 且 ctx.metadata 中提供了 BM25Retriever 与
        # Reranker 时，走混合检索（Dense + BM25 → RRF → Rerank）；否则
        # 退回到纯向量检索（向后兼容）。
        hybrid_retriever = self._maybe_build_hybrid_retriever(
            ctx, vectorstore, inp
        )

        try:
            if hybrid_retriever is not None:
                hits = await hybrid_retriever.retrieve(
                    query=inp.question or "",
                    query_vector=inp.query_vector,
                    top_k=inp.top_k,
                    rerank_top_k=inp.top_k,
                )
            else:
                hits = await vectorstore.search(
                    collection=inp.collection,
                    query_vector=inp.query_vector,
                    top_k=inp.top_k,
                    filter_expr=inp.filter_expr,
                )

            # Convert VectorDocument dataclass to plain dict for
            # downstream consumption (serialization-friendly).
            contexts: list[dict[str, Any]] = []
            for hit in hits:
                contexts.append(
                    {
                        "id": getattr(hit, "id", None),
                        "text": getattr(hit, "text", None)
                        or getattr(hit, "metadata", {}).get("text", ""),
                        "metadata": dict(getattr(hit, "metadata", {}) or {}),
                        "score": getattr(hit, "score", None),
                    }
                )

            logger.info(
                "Retrieved %d documents from '%s' (top_k=%d, hybrid=%s)",
                len(contexts),
                inp.collection,
                inp.top_k,
                hybrid_retriever is not None,
            )

            return BaseTaskOutput.success(
                contexts=contexts,
                retrieved_count=len(contexts),
                collection=inp.collection,
            )
        except Exception as e:
            logger.error("Document retrieval failed: %s", e)
            return BaseTaskOutput.failure(f"Retrieval failed: {e}")

    @staticmethod
    def _maybe_build_hybrid_retriever(
        ctx: TaskContext,
        vectorstore: Any,
        inp: "RetrieveDocsInput",
    ) -> Any:
        """
        根据输入与上下文决定是否构建 HybridRetriever。

        构建条件（全部满足才走混合检索）：
            1. ``inp.use_hybrid`` 为 True
            2. ``ctx.metadata`` 中存在 ``bm25_retriever`` 与 ``reranker``
            3. HybridRetriever 构造成功

        任一条件不满足时返回 None，调用方退回纯向量检索。

        Args:
            ctx:         任务上下文。
            vectorstore: 已注入的向量库实例。
            inp:         检索任务输入。

        Returns:
            HybridRetriever 实例或 None。
        """
        if not inp.use_hybrid:
            return None

        bm25 = ctx.get_metadata("bm25_retriever")
        reranker = ctx.get_metadata("reranker")
        if bm25 is None or reranker is None:
            logger.warning(
                "retrieve_docs: use_hybrid=True but bm25_retriever/"
                "reranker not found in ctx.metadata; "
                "falling back to pure dense search"
            )
            return None

        try:
            from icore.retrieval import HybridRetriever
        except ImportError:
            logger.warning(
                "retrieve_docs: icore.retrieval.HybridRetriever import "
                "failed; falling back to pure dense search"
            )
            return None

        try:
            retriever = HybridRetriever(
                vectorstore=vectorstore,
                bm25=bm25,
                reranker=reranker,
                collection=inp.collection,
            )
            logger.info(
                "retrieve_docs: HybridRetriever enabled "
                "(collection='%s')",
                inp.collection,
            )
            return retriever
        except Exception as e:
            logger.warning(
                "retrieve_docs: failed to build HybridRetriever "
                "(%s); falling back to pure dense search",
                e,
            )
            return None

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Generate Answer (LLM)
# ---------------------------------------------------------------------------

@register_task("generate_rag_answer")
class GenerateAnswerTask(BaseTask):
    """
    Generates a grounded answer from the retrieved context via LLM.

    Combines the user's question with the retrieved document chunks
    into a single prompt, then asks the LLM to produce a concise,
    context-grounded answer. The prompt explicitly instructs the model
    to cite which chunks it used.
    """

    name: ClassVar[str] = "generate_rag_answer"
    description: ClassVar[str] = (
        "Generate a context-grounded answer via LLM"
    )
    input_model: ClassVar[type[BaseTaskInput]] = GenerateAnswerInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        """Obtain the model adapter."""
        self._model = ctx.get_model_adapter()

    async def execute(
        self, ctx: TaskContext, inp: GenerateAnswerInput
    ) -> BaseTaskOutput:
        """Generate the final answer grounded by retrieved context."""
        if not inp.question:
            return BaseTaskOutput.failure("Question is empty")

        # Truncate to max_context_chunks to keep prompt size bounded.
        selected = inp.contexts[: inp.max_context_chunks]

        if not selected:
            # No context — still answer, but flag the lack of grounding.
            context_block = (
                "[No relevant documents were retrieved. Answer based on "
                "general knowledge and clearly state the limitation.]"
            )
        else:
            chunks: list[str] = []
            for i, doc in enumerate(selected, 1):
                text = doc.get("text") or doc.get("metadata", {}).get("text", "")
                doc_id = doc.get("id", f"chunk-{i}")
                chunks.append(f"[Context {i}] (id={doc_id}):\n{text}")
            context_block = "\n\n".join(chunks)

        prompt = (
            f"Answer the user's question based on the retrieved context "
            f"below. If the context is insufficient, say so explicitly. "
            f"Cite the context indices you used.\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            f"User question: {inp.question}\n\n"
            f"Answer:"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-augmented generation assistant. "
                    "Always ground your answer in the provided context "
                    "and cite which chunks you used."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._model.chat(messages=messages)
            answer = response.get("content", "")

            logger.info(
                "Generated RAG answer (%d chars, %d context chunks used)",
                len(answer),
                len(selected),
            )

            return BaseTaskOutput.success(
                answer=answer,
                question=inp.question,
                context_count=len(selected),
                used_context_ids=[d.get("id") for d in selected],
            )
        except Exception as e:
            logger.error("RAG answer generation failed: %s", e)
            return BaseTaskOutput.failure(f"Answer generation failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Workflow: RAG QA
# ---------------------------------------------------------------------------

@register_workflow("rag_qa")
class RAGQAWorkflow(BaseWorkflow):
    """
    Retrieval-Augmented Generation QA workflow.

    Pipeline:
        1. embed_query:      Convert the question to a vector via LLM
        2. retrieve_docs:    Search the vector store for top-K similar docs
        3. generate_answer:  Generate a grounded answer via LLM

    DAG:
        embed_query -> retrieve_docs -> generate_answer

    This workflow demonstrates vectorstore + LLM integration: the first
    task embeds the question using the LLM, the second retrieves relevant
    document chunks from the vector store, and the third generates a
    grounded answer using both the question and the retrieved context.
    """

    name: ClassVar[str] = "rag_qa"
    description: ClassVar[str] = (
        "Embed question -> retrieve relevant docs from vector store -> "
        "generate grounded answer via LLM"
    )

    def define(self) -> DAG:
        """Build the embed -> retrieve -> generate DAG."""
        dag = DAG()

        # Node 1: Embed the question
        dag.add_node(
            node_id="embed_query",
            task_name="embed_query",
        )

        # Node 2: Retrieve relevant documents
        # Input builder: takes the embedded vector + workflow params.
        # question 与 use_hybrid 仅在启用混合检索时被使用；缺省时
        # 走纯向量检索，保持向后兼容。
        dag.add_node(
            node_id="retrieve_docs",
            task_name="retrieve_docs",
            input_builder=lambda params, upstream: RetrieveDocsInput(
                query_vector=upstream["embed_query"].data["query_vector"],
                collection=params.get("collection", "doc_embeddings"),
                top_k=params.get("top_k", 5),
                filter_expr=params.get("filter_expr"),
                question=upstream["embed_query"].data.get("question")
                or params.get("question"),
                use_hybrid=bool(params.get("use_hybrid", False)),
            ),
        )

        # Node 3: Generate the grounded answer
        # Input builder: combines question + retrieved contexts.
        # Note: question comes from workflow params (not upstream) because
        # generate_answer's only direct predecessor is retrieve_docs.
        dag.add_node(
            node_id="generate_answer",
            task_name="generate_rag_answer",
            input_builder=lambda params, upstream: GenerateAnswerInput(
                question=params.get("question", ""),
                contexts=upstream["retrieve_docs"].data.get("contexts", []),
                max_context_chunks=params.get("max_context_chunks", 5),
            ),
        )

        # Linear dependency chain
        dag.add_edge("embed_query", "retrieve_docs")
        dag.add_edge("retrieve_docs", "generate_answer")

        return dag
