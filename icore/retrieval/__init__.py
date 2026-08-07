"""
icore.retrieval - v0.6 混合检索模块（Dense + Sparse + Rerank）。

提供四大组件，组合使用即可构建生产级 RAG 检索管线：

    1. ``BM25Retriever``        - 纯 Python BM25 稀疏检索，零外部依赖
    2. ``reciprocal_rank_fusion`` - RRF 算法，融合 Dense / Sparse 结果
    3. ``BaseReranker``         - 重排序抽象基类
       - ``IdentityReranker``         - no-op，原样返回
       - ``LLMReranker``              - 调用注入的 ``BaseModelAdapter``，让 LLM 直接打分
       - ``CrossEncoderReranker``     - 调用 sentence-transformers 风格的 Cross-Encoder 服务
                                       （``httpx`` 懒导入，未装时模块仍可 import）
    4. ``HybridRetriever``      - 组合 ``BaseVectorStore`` + ``BM25Retriever`` + ``Reranker``

模块依赖：
    仅依赖标准库 + ``icore.vectorstore`` + ``icore.exceptions``。
    ``httpx`` 在 ``CrossEncoderReranker`` 中懒导入，不影响模块 import。

设计原则：
    - **不引入** whoosh / rank-bm25 / sentence-transformers / Jinja2 等第三方库
    - BM25 用 ``collections.Counter`` + ``dict`` 实现，参考 Robertson 等人的原始公式
    - RRF 用 Cormack 等 (2009) 的标准公式 ``1 / (k + rank)``
    - 所有日志走标准库 ``logging``（**不**使用 loguru）
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Optional

from icore.exceptions import ValidationError, VectorStoreError
from icore.vectorstore import BaseVectorStore, VectorDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 稀疏检索
# ---------------------------------------------------------------------------


# CJK 统一表意文字范围（用于中文按字符切分）
_CJK_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)
# 英文 token：字母 + 数字 + 下划线，最长 64（避免超长 token）
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class BM25Retriever:
    """
    纯 Python BM25 稀疏检索器。

    使用 Okapi BM25 算法（Robertson & Zaragoza, 2009）::

        score(q, d) = Σ_t IDF(t) · (f(t,d) · (k1+1)) / (f(t,d) + k1 · (1-b + b·|d|/avgdl))
        IDF(t)     = log( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )

    其中：
        - ``N``       文档总数
        - ``n(t)``    含 term t 的文档数（document frequency）
        - ``f(t,d)``  term t 在文档 d 中的频次（term frequency）
        - ``|d|``     文档 d 的长度（token 数）
        - ``avgdl``   全体文档平均长度
        - ``k1``      词频饱和参数（默认 1.5）
        - ``b``       长度归一化参数（默认 0.75）

    所有索引状态（``_docs`` / ``_inverted`` / ``_idf``）由 ``index()`` 一次性构建，
    支持多次 ``search()`` 调用。线程安全由调用方保证（与 vectorstore 一致）。

    Attributes:
        k1: 词频饱和参数。越大则 tf 越不容易饱和。
        b:  长度归一化参数。0 = 不归一化，1 = 完全归一化。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be in [0, 1], got {b}")
        self.k1: float = k1
        self.b: float = b

        # doc_id -> VectorDocument（仅保存元信息，不保存 vector）
        self._docs: dict[str, VectorDocument] = {}
        # doc_id -> Counter[token, tf]
        self._tf: dict[str, Counter] = {}
        # doc_id -> 文档长度（token 数）
        self._doc_len: dict[str, int] = {}
        # token -> set(doc_id)（倒排索引）
        self._inverted: dict[str, set[str]] = {}
        # token -> IDF 值
        self._idf: dict[str, float] = {}
        # 平均文档长度
        self._avgdl: float = 0.0
        # 文档总数
        self._n_docs: int = 0

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def index(self, documents: list[VectorDocument]) -> None:
        """
        构建 BM25 倒排索引和 IDF 表。

        重复调用会覆盖已有索引（不做增量合并）。文档 ID 重复时后到覆盖。

        Args:
            documents: 待索引的文档列表。使用每个文档的 ``text`` 字段
                       作为语料；``text`` 为 None 的文档被跳过。

        Raises:
            ValidationError: documents 为空。
        """
        if not documents:
            raise ValidationError("BM25Retriever.index requires non-empty documents")

        # 重置索引
        self._docs = {}
        self._tf = {}
        self._doc_len = {}
        self._inverted = {}
        self._idf = {}

        total_len = 0
        for doc in documents:
            text = doc.text or ""
            tokens = self._tokenize(text)
            tf = Counter(tokens)

            self._docs[doc.id] = doc
            self._tf[doc.id] = tf
            self._doc_len[doc.id] = len(tokens)
            total_len += len(tokens)

            for token in tf.keys():
                self._inverted.setdefault(token, set()).add(doc.id)

        self._n_docs = len(self._docs)
        self._avgdl = (total_len / self._n_docs) if self._n_docs else 0.0

        # 计算 IDF（标准 Okapi BM25 形式，加 1 防止负值）
        for token, doc_ids in self._inverted.items():
            df = len(doc_ids)
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf[token] = idf

        logger.debug(
            "BM25 index built: %d docs, %d unique tokens, avgdl=%.2f",
            self._n_docs,
            len(self._inverted),
            self._avgdl,
        )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 10) -> list[VectorDocument]:
        """
        对 query 执行 BM25 检索，返回按分数降序的 top_k 文档。

        Args:
            query:  查询字符串。会被 ``_tokenize`` 切分。
            top_k:  返回前 K 个结果。若不足 K 则返回全部。

        Returns:
            ``VectorDocument`` 列表，按 BM25 分数降序。每个返回的
            ``VectorDocument`` 的 ``metadata`` 中会写入本次查询的
            ``bm25_score`` 字段（float）。

        Raises:
            ValidationError: 索引为空或 top_k <= 0。
        """
        if not self._docs:
            raise ValidationError("BM25Retriever.index must be called before search")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[str, float]] = []
        # 仅对 query 中出现的、且在索引里有 posting list 的 token 计算候选
        candidate_doc_ids: set[str] = set()
        for tok in set(query_tokens):
            posting = self._inverted.get(tok)
            if posting:
                candidate_doc_ids.update(posting)

        for doc_id in candidate_doc_ids:
            score = self._bm25_score(query_tokens, doc_id)
            scores.append((doc_id, score))

        # Sort by score desc; on ties, break by doc_id asc so the order
        # is deterministic across runs (avoids flaky RRF fusion results
        # when multiple docs have identical BM25 scores).
        scores.sort(key=lambda x: (-x[1], x[0]))
        top = scores[:top_k]

        results: list[VectorDocument] = []
        for doc_id, score in top:
            original = self._docs[doc_id]
            # 复制 metadata，避免污染索引中的原始对象
            new_meta = dict(original.metadata)
            new_meta["bm25_score"] = score
            results.append(
                VectorDocument(
                    id=original.id,
                    vector=original.vector,
                    metadata=new_meta,
                    text=original.text,
                )
            )
        return results

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """
        简单分词：
            - 英文 / 数字：按 ``[A-Za-z0-9_]+`` 切分，统一小写
            - 中文（CJK）：每个字符作为一个 token
            - 标点 / 空白：丢弃

        这样混合中英文文本可以同时被索引和检索。

        Args:
            text: 原始文本。

        Returns:
            token 列表，可能含重复（保留词频信息）。
        """
        if not text:
            return []
        tokens: list[str] = []
        # 先抽 CJK 单字
        for ch in text:
            if _CJK_PATTERN.match(ch):
                tokens.append(ch)
        # 再抽英文 / 数字 token
        for m in _WORD_PATTERN.findall(text):
            tokens.append(m.lower())
        return tokens

    def _bm25_score(self, query_tokens: list[str], doc_id: str) -> float:
        """
        计算单个文档相对 query 的 BM25 分数。

        Args:
            query_tokens: 已分词的 query token 列表（保留重复）。
            doc_id:       目标文档 ID。

        Returns:
            BM25 分数（float，>=0）。若文档不存在返回 0.0。
        """
        if doc_id not in self._tf:
            return 0.0

        tf = self._tf[doc_id]
        doc_len = self._doc_len[doc_id]
        # 防止 avgdl=0（所有文档都是空字符串）的退化情况
        denom_len_norm = 1 - self.b + self.b * (
            doc_len / self._avgdl if self._avgdl > 0 else 0.0
        )

        # query 中每个 token 都贡献一次（保留 query 中的词频信息）
        score = 0.0
        for token in query_tokens:
            idf = self._idf.get(token)
            if idf is None:
                continue
            f = tf.get(token, 0)
            if f == 0:
                continue
            score += idf * (f * (self.k1 + 1)) / (f + self.k1 * denom_len_norm)
        return score


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    dense_results: list[VectorDocument],
    sparse_results: list[VectorDocument],
    k: int = 60,
    weights: tuple[float, float] = (0.5, 0.5),
) -> list[tuple[VectorDocument, float]]:
    """
    Reciprocal Rank Fusion（RRF）算法。

    标准公式（Cormack, Clarke & Büttcher, 2009）::

        RRF(d) = Σ_i  w_i / (k + rank_i(d))

    其中 ``rank_i(d)`` 是文档 d 在第 i 路检索结果中的排名（从 1 开始）；
    未出现的文档不参与该路的求和。

    相同 ID 的文档在两路结果中均出现时，两路贡献相加，最终输出
    该 ID 第一次出现时的 ``VectorDocument``（dense 优先）。

    Args:
        dense_results:   Dense 检索结果（已按相关度降序）。
        sparse_results:  Sparse 检索结果（已按相关度降序）。
        k:               RRF 平滑常数（默认 60，标准经验值）。
        weights:         ``(dense_weight, sparse_weight)``，归一化后使用。

    Returns:
        ``(VectorDocument, fused_score)`` 列表，按 fused_score 降序。
        fused_score >= 0。

    Raises:
        ValueError: k <= 0 或 weights 含负值。
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    w_dense, w_sparse = weights
    if w_dense < 0 or w_sparse < 0:
        raise ValueError(f"RRF weights must be non-negative, got {weights}")
    total_w = w_dense + w_sparse
    if total_w <= 0:
        raise ValueError("RRF weights sum must be positive")
    w_dense /= total_w
    w_sparse /= total_w

    scores: dict[str, float] = {}
    docs: dict[str, VectorDocument] = {}

    # Dense 路
    for rank, doc in enumerate(dense_results, start=1):
        if doc.id in scores:
            scores[doc.id] += w_dense / (k + rank)
        else:
            scores[doc.id] = w_dense / (k + rank)
            docs[doc.id] = doc

    # Sparse 路
    for rank, doc in enumerate(sparse_results, start=1):
        if doc.id in scores:
            scores[doc.id] += w_sparse / (k + rank)
        else:
            scores[doc.id] = w_sparse / (k + rank)
            docs[doc.id] = doc

    fused = [(docs[doc_id], score) for doc_id, score in scores.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Reranker 抽象 + 实现
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """
    重排序器抽象基类。

    所有重排序器接收一个 query 和一组候选文档，返回按相关性降序的
    ``top_k`` 文档。子类只需实现 ``rerank``。
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[VectorDocument],
        top_k: int = 5,
    ) -> list[VectorDocument]:
        """重排序候选文档，返回 top_k 最相关者。"""
        raise NotImplementedError


class IdentityReranker(BaseReranker):
    """
    No-op reranker，直接返回输入的前 top_k 个文档。

    适用于：
        - 关闭 rerank 的场景（``HybridRetriever`` 默认行为）
        - 测试 / 基线对照
    """

    async def rerank(
        self,
        query: str,
        documents: list[VectorDocument],
        top_k: int = 5,
    ) -> list[VectorDocument]:
        return list(documents[:top_k])


class LLMReranker(BaseReranker):
    """
    用 LLM 直接给候选文档打分的重排序器。

    通过注入的 ``BaseModelAdapter`` 调用 LLM，让模型输出形如::

        [{"doc_id": "1", "score": 0.95}, ...]

    的 JSON，然后按 score 降序返回前 ``top_k``。

    评分 prompt 中明确告知 LLM 输出**纯 JSON 数组**，并附带 query 与
    候选文档（编号 + 截断文本）。对 LLM 输出做容错解析：先尝试整体
    JSON 解析；失败则用正则提取第一个 ``[...`` JSON 数组。

    Attributes:
        model_adapter: 实现 ``chat()`` 的 ``BaseModelAdapter`` 实例。
        max_chars:     每个候选文档送入 LLM 的最大字符数（防止 prompt 过长）。
    """

    def __init__(
        self,
        model_adapter: Any,
        *,
        max_chars: int = 500,
    ) -> None:
        if model_adapter is None:
            raise ValueError("model_adapter is required")
        self._adapter = model_adapter
        self._max_chars = max_chars

    async def rerank(
        self,
        query: str,
        documents: list[VectorDocument],
        top_k: int = 5,
    ) -> list[VectorDocument]:
        if not documents:
            return []
        if top_k <= 0:
            return []

        # 构造候选文档清单
        lines: list[str] = []
        for idx, doc in enumerate(documents, start=1):
            text = (doc.text or "").strip()
            if len(text) > self._max_chars:
                text = text[: self._max_chars] + "..."
            lines.append(f"[{idx}] doc_id={doc.id}\n{text}")
        candidates_block = "\n\n".join(lines)

        prompt = (
            "You are a relevance scoring engine. Given a query and a list of "
            "candidate documents, output a JSON array of objects ranking each "
            "document by relevance to the query. Each object must have "
            '"doc_id" (string) and "score" (float in [0, 1]).\n\n'
            f"Query: {query}\n\n"
            f"Candidates:\n{candidates_block}\n\n"
            "Output ONLY the JSON array, no extra text. Example format:\n"
            '[{"doc_id": "1", "score": 0.95}, {"doc_id": "2", "score": 0.42}]\n'
        )

        try:
            response = await self._adapter.chat(
                [{"role": "user", "content": prompt}]
            )
        except Exception as e:
            logger.warning("LLMReranker chat failed: %s; falling back to original order", e)
            return list(documents[:top_k])

        content = (response or {}).get("content", "") if isinstance(response, dict) else ""
        scored = self._parse_llm_scores(content, documents)

        if not scored:
            # 解析失败：降级为原顺序
            logger.warning("LLMReranker failed to parse LLM output; keeping original order")
            return list(documents[:top_k])

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    def _parse_llm_scores(
        self,
        content: str,
        documents: list[VectorDocument],
    ) -> list[tuple[VectorDocument, float]]:
        """
        从 LLM 输出中解析 ``[{doc_id, score}, ...]``。

        解析失败时返回空列表（调用方降级处理）。
        """
        if not content:
            return []

        # 1) 直接尝试整段 JSON 解析
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return self._build_scored(parsed, documents)
        except json.JSONDecodeError:
            pass

        # 2) 用正则抽第一个 JSON 数组
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return self._build_scored(parsed, documents)
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def _build_scored(
        parsed: list[Any],
        documents: list[VectorDocument],
    ) -> list[tuple[VectorDocument, float]]:
        """根据解析出的 list 构造 ``(doc, score)`` 列表。"""
        doc_by_id = {d.id: d for d in documents}
        scored: list[tuple[VectorDocument, float]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("doc_id")
            score = item.get("score")
            if doc_id is None or score is None:
                continue
            doc = doc_by_id.get(str(doc_id))
            if doc is None:
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            scored.append((doc, score_f))
        return scored


class CrossEncoderReranker(BaseReranker):
    """
    调用 Cross-Encoder 服务的重排序器。

    设计参考 ``sentence-transformers`` 的 ``CrossEncoder.predict`` 接口。
    用户可指向任意兼容该接口的 HTTP 服务（自建推理服务 / Cohere Rerank /
    Jina Rerank 等）。

    HTTP 协议（POST ``endpoint``）::

        请求体:
        {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "pairs": [[query, doc_text_1], [query, doc_text_2], ...]
        }
        响应体:
        {
            "scores": [0.95, 0.42, ...]
        }

    ``httpx`` 懒导入：模块可在未安装 httpx 时被 import，仅在
    ``rerank()`` 被调用且 httpx 缺失时抛 ``ImportError``。

    Attributes:
        endpoint:  Cross-Encoder 服务 URL。
        api_key:   Bearer 鉴权 token（可空）。
        model:     模型名（透传给服务）。
        timeout:   单次请求超时秒数。
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        timeout: float = 30.0,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def rerank(
        self,
        query: str,
        documents: list[VectorDocument],
        top_k: int = 5,
    ) -> list[VectorDocument]:
        if not documents:
            return []
        if top_k <= 0:
            return []

        try:
            import httpx  # type: ignore
        except ImportError as e:
            raise ImportError(
                "httpx is required for CrossEncoderReranker. "
                "Install with: pip install httpx"
            ) from e

        pairs = [
            [query, (d.text or "")[:2000]]
            for d in documents
        ]
        payload = {
            "model": self._model,
            "pairs": pairs,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            raise VectorStoreError(f"CrossEncoder rerank failed: {e}") from e

        scores_raw = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(scores_raw, list) or len(scores_raw) != len(documents):
            raise VectorStoreError(
                "CrossEncoder returned malformed scores: "
                f"expected {len(documents)} floats, got {type(scores_raw).__name__}"
            )

        scored: list[tuple[VectorDocument, float]] = []
        for doc, raw in zip(documents, scores_raw):
            try:
                s = float(raw)
            except (TypeError, ValueError) as e:
                raise VectorStoreError(
                    f"CrossEncoder score not float: {raw!r}"
                ) from e
            scored.append((doc, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# 混合检索器
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    混合检索器：Dense + Sparse → RRF 融合 → Rerank。

    工作流::

        ┌─────────────┐   query_vector   ┌─────────────────┐
        │ query       │ ───────────────► │ vectorstore     │ ──► dense_results
        │ (text+vec)  │                  │ .search()       │
        └─────┬───────┘                  └─────────────────┘
              │ query text
              ▼
        ┌─────────────┐                  ┌─────────────────┐
        │ BM25        │ ───────────────► │ bm25.search()   │ ──► sparse_results
        │ (optional)  │                  └─────────────────┘
        └─────────────┘

                  dense_results + sparse_results
                              │
                              ▼
                  ┌──────────────────────┐
                  │ reciprocal_rank_     │ ──► fused [(doc, score), ...]
                  │ fusion (RRF)        │
                  └──────────────────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │ reranker.rerank()    │ ──► final top_k
                  │ (optional)           │
                  └──────────────────────┘

    Attributes:
        vectorstore:    底层向量库（必须）。
        bm25:           BM25 检索器；None 表示不使用 sparse 检索。
        reranker:       重排序器；None 表示不使用 rerank。
        dense_weight:   RRF 中 dense 路权重（与 sparse_weight 一起归一化）。
        sparse_weight:  RRF 中 sparse 路权重。
        rrf_k:          RRF 平滑常数。
        collection:     默认向量库 collection 名。
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        bm25: Optional[BM25Retriever] = None,
        reranker: Optional[BaseReranker] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        collection: str = "default",
    ) -> None:
        if vectorstore is None:
            raise ValueError("vectorstore is required")
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("dense_weight / sparse_weight must be non-negative")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not collection:
            raise ValueError("collection must be non-empty")

        self._vectorstore = vectorstore
        self._bm25 = bm25
        self._reranker = reranker
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._rrf_k = rrf_k
        self._collection = collection

    @property
    def collection(self) -> str:
        """当前默认 collection 名。"""
        return self._collection

    @collection.setter
    def collection(self, value: str) -> None:
        if not value:
            raise ValueError("collection must be non-empty")
        self._collection = value

    async def retrieve(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        rerank_top_k: int = 5,
        use_sparse: bool = True,
        use_rerank: bool = True,
    ) -> list[VectorDocument]:
        """
        执行混合检索。

        步骤：
            1. Dense 检索（``vectorstore.search``），取 ``top_k`` 个候选。
            2. Sparse 检索（``bm25.search``），取 ``top_k`` 个候选
               （仅在 ``use_sparse=True`` 且配置了 BM25 时启用）。
            3. RRF 融合 dense + sparse 结果。
            4. 若开启 rerank 且配置了 reranker，对融合结果前
               ``top_k`` 个 rerank，取 ``rerank_top_k`` 个最终结果。

        Args:
            query:         用户查询文本。
            query_vector:   用户查询向量（embedding）。
            top_k:         初检每路取的候选数 + RRF 融合候选数。
            rerank_top_k:  rerank 后返回的最终数量。
            use_sparse:    是否启用 sparse 检索。
            use_rerank:    是否启用 rerank。

        Returns:
            ``VectorDocument`` 列表，按相关度降序。

        Raises:
            ValueError: top_k / rerank_top_k <= 0。
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be positive")

        # 1. Dense 检索
        try:
            dense_results = await self._vectorstore.search(
                self._collection, query_vector, top_k=top_k
            )
        except Exception as e:
            raise VectorStoreError(f"Dense retrieval failed: {e}") from e

        # 2. Sparse 检索（可选）
        sparse_results: list[VectorDocument] = []
        if use_sparse and self._bm25 is not None:
            try:
                sparse_results = await self._bm25.search(query, top_k=top_k)
            except Exception as e:
                # sparse 失败不致命，降级为只用 dense
                logger.warning("BM25 sparse retrieval failed: %s; falling back to dense-only", e)
                sparse_results = []

        # 3. RRF 融合
        if sparse_results:
            fused = reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                k=self._rrf_k,
                weights=(self._dense_weight, self._sparse_weight),
            )
            candidates = [doc for doc, _ in fused]
        else:
            candidates = dense_results

        # 4. Rerank（可选）
        if use_rerank and self._reranker is not None and candidates:
            # 仅对前 top_k 个候选做 rerank（控制成本）
            rerank_input = candidates[:top_k]
            try:
                return await self._reranker.rerank(
                    query, rerank_input, top_k=rerank_top_k
                )
            except Exception as e:
                logger.warning("Rerank failed: %s; falling back to fused order", e)
                return candidates[:rerank_top_k]

        return candidates[:rerank_top_k]


__all__ = [
    "BM25Retriever",
    "reciprocal_rank_fusion",
    "BaseReranker",
    "IdentityReranker",
    "LLMReranker",
    "CrossEncoderReranker",
    "HybridRetriever",
]
