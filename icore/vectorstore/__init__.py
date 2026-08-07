"""
icore.vectorstore - Vector database abstractions and adapters.

Provides:
    - ``VectorDocument``:  Dataclass for a vectorized document.
    - ``BaseVectorStore``: Abstract interface for vector stores.
    - ``MilvusAdapter``:   Milvus-backed implementation (lazy import).
    - ``InMemoryVectorStore``: Pure-Python in-memory implementation
                                used by tests and small deployments.

Module dependency:
    vectorstore depends on icore.exceptions only. The Milvus driver
    (``pymilvus``) is imported lazily so this package is importable
    without pymilvus installed.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from icore.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """
    A vectorized document.

    Attributes:
        id:       Unique document ID.
        vector:   Embedding vector (list of floats).
        metadata: Free-form metadata dict (tags, source, timestamps).
        text:     Optional original text used to generate the vector.
    """

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None


class BaseVectorStore(ABC):
    """
    Abstract vector store interface.

    Implementations: ``MilvusAdapter``, ``InMemoryVectorStore``.
    TaskContext injects a concrete instance accessible via
    ``ctx.get_vectorstore()``.
    """

    @abstractmethod
    async def insert(
        self, collection: str, docs: list[VectorDocument]
    ) -> list[str]:
        """Insert documents, returning the list of inserted IDs."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[VectorDocument]:
        """ANN similarity search."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> int:
        """Delete documents by ID. Return count deleted."""
        raise NotImplementedError

    @abstractmethod
    async def create_collection(
        self, name: str, dim: int, index_type: str = "IVF_FLAT"
    ) -> None:
        """Create a collection (with index config)."""
        raise NotImplementedError

    @abstractmethod
    async def drop_collection(self, name: str) -> None:
        """Drop a collection."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """Lightweight connectivity check."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release connection resources. Default no-op."""
        pass


# ---------------------------------------------------------------------------
# In-memory implementation (offline tests, small deployments)
# ---------------------------------------------------------------------------


class InMemoryVectorStore(BaseVectorStore):
    """
    Pure-Python in-memory vector store.

    Performs exact (brute-force) cosine similarity search. Suitable
    for unit tests, demos, and small datasets (<10k vectors).
    """

    def __init__(self) -> None:
        # collection -> list[VectorDocument]
        self._collections: dict[str, list[VectorDocument]] = {}
        self._lock = asyncio.Lock()

    async def insert(
        self, collection: str, docs: list[VectorDocument]
    ) -> list[str]:
        async with self._lock:
            store = self._collections.setdefault(collection, [])
            # De-dup by id (upsert semantics)
            existing_ids = {d.id for d in store}
            inserted: list[str] = []
            for d in docs:
                if d.id in existing_ids:
                    # Replace existing
                    store[:] = [
                        d if x.id == d.id else x for x in store
                    ]
                else:
                    store.append(d)
                inserted.append(d.id)
            return inserted

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[VectorDocument]:
        async with self._lock:
            store = list(self._collections.get(collection, []))

        if not store:
            return []

        # Apply metadata filter (very small expression subset: k == v)
        if filter_expr:
            store = [
                d
                for d in store
                if _eval_simple_filter(filter_expr, d.metadata)
            ]

        # Cosine similarity
        scored = [
            (_cosine(query_vector, d.vector), d) for d in store
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    async def delete(self, collection: str, ids: list[str]) -> int:
        async with self._lock:
            store = self._collections.get(collection, [])
            target = set(ids)
            before = len(store)
            store[:] = [d for d in store if d.id not in target]
            removed = before - len(store)
            return removed

    async def create_collection(
        self, name: str, dim: int, index_type: str = "IVF_FLAT"
    ) -> None:
        async with self._lock:
            if name not in self._collections:
                self._collections[name] = []

    async def drop_collection(self, name: str) -> None:
        async with self._lock:
            self._collections.pop(name, None)

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Milvus adapter (lazy import)
# ---------------------------------------------------------------------------


class MilvusAdapter(BaseVectorStore):
    """
    Milvus vector store adapter.

    Uses ``pymilvus.MilvusClient`` for synchronous-ish CRUD over a
    Milvus server. The driver is imported lazily so this module is
    importable without pymilvus installed.

    Supports:
        - Auto collection creation (schema + index)
        - Batch insert + flush
        - ANN search (IVF_FLAT / HNSW / IVF_PQ)
        - Scalar filter expressions (Milvus native syntax)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        db_name: str = "icore",
        default_index_type: str = "IVF_FLAT",
        default_metric: str = "COSINE",
        pool_size: int = 10,
    ) -> None:
        self._host = host
        self._port = port
        self._db_name = db_name
        self._default_index_type = default_index_type
        self._default_metric = default_metric
        self._pool_size = pool_size
        self._client: Any = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pymilvus is required for MilvusAdapter. "
                "Install with: pip install pymilvus"
            ) from e
        # MilvusClient is thread-safe; one instance is enough.
        self._client = MilvusClient(
            uri=f"http://{self._host}:{self._port}",
            db_name=self._db_name,
        )
        return self._client

    async def insert(
        self, collection: str, docs: list[VectorDocument]
    ) -> list[str]:
        client = await self._get_client()
        try:
            data = [
                {
                    "id": d.id,
                    "vector": d.vector,
                    "text": d.text or "",
                    **d.metadata,
                }
                for d in docs
            ]
            result = await asyncio.to_thread(
                client.insert, collection_name=collection, data=data
            )
            return [d.id for d in docs]
        except Exception as e:
            raise VectorStoreError(f"Milvus insert failed: {e}") from e

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[VectorDocument]:
        client = await self._get_client()
        try:
            kwargs: dict[str, Any] = {
                "collection_name": collection,
                "data": [query_vector],
                "limit": top_k,
                "output_fields": ["text", "*"],
            }
            if filter_expr:
                kwargs["filter"] = filter_expr
            results = await asyncio.to_thread(client.search, **kwargs)
            out: list[VectorDocument] = []
            if not results:
                return out
            for hit in results[0]:
                entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
                hit_id = (
                    hit.get("id")
                    if isinstance(hit, dict)
                    else getattr(hit, "id", None)
                )
                vector = entity.get("vector", []) or []
                text = entity.get("text")
                metadata = {
                    k: v
                    for k, v in entity.items()
                    if k not in ("vector", "text")
                }
                out.append(
                    VectorDocument(
                        id=str(hit_id),
                        vector=list(vector),
                        metadata=metadata,
                        text=text,
                    )
                )
            return out
        except Exception as e:
            raise VectorStoreError(f"Milvus search failed: {e}") from e

    async def delete(self, collection: str, ids: list[str]) -> int:
        client = await self._get_client()
        try:
            await asyncio.to_thread(
                client.delete,
                collection_name=collection,
                ids=ids,
            )
            return len(ids)
        except Exception as e:
            raise VectorStoreError(f"Milvus delete failed: {e}") from e

    async def create_collection(
        self, name: str, dim: int, index_type: str = "IVF_FLAT"
    ) -> None:
        client = await self._get_client()
        try:
            from pymilvus import DataType  # type: ignore

            schema = client.create_schema(
                auto_id=False, enable_dynamic_field=True
            )
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type=index_type or self._default_index_type,
                metric_type=self._default_metric,
            )
            await asyncio.to_thread(
                client.create_collection,
                collection_name=name,
                schema=schema,
                index_params=index_params,
            )
        except Exception as e:
            raise VectorStoreError(
                f"Milvus create_collection failed: {e}"
            ) from e

    async def drop_collection(self, name: str) -> None:
        client = await self._get_client()
        try:
            await asyncio.to_thread(
                client.drop_collection, collection_name=name
            )
        except Exception as e:
            raise VectorStoreError(
                f"Milvus drop_collection failed: {e}"
            ) from e

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            await asyncio.to_thread(client.list_collections)
            return True
        except Exception as e:
            logger.debug("Milvus health check failed: %s", e)
            return False

    async def close(self) -> None:
        if self._client is not None:
            try:
                close = getattr(self._client, "close", None)
                if close is not None:
                    await asyncio.to_thread(close)
            except Exception as e:  # pragma: no cover
                logger.warning("Milvus close failed: %s", e)
            finally:
                self._client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _eval_simple_filter(expr: str, metadata: dict[str, Any]) -> bool:
    """
    Tiny evaluator for ``key == value`` and ``key > value`` style filters.

    Supports AND conjunction. The full Milvus expression syntax is
    delegated to the Milvus adapter itself; this helper only serves
    the in-memory implementation.
    """
    if not expr:
        return True
    for clause in expr.split(" and "):
        clause = clause.strip()
        if not clause:
            continue
        for op in ("==", ">=", "<=", ">", "<"):
            if op in clause:
                k, v = clause.split(op, 1)
                k = k.strip().strip('"').strip("'")
                v = v.strip().strip('"').strip("'")
                actual = metadata.get(k)
                if actual is None:
                    return False
                try:
                    actual_t = type(v)(actual) if v.replace("-", "").replace(".", "").isdigit() else actual
                    v_t = type(actual_t)(v) if isinstance(actual_t, (int, float)) else v
                except Exception:
                    actual_t = actual
                    v_t = v
                if op == "==" and not (actual_t == v_t or str(actual_t) == str(v_t)):
                    return False
                if op == ">" and not (float(actual_t) > float(v_t)):
                    return False
                if op == "<" and not (float(actual_t) < float(v_t)):
                    return False
                if op == ">=" and not (float(actual_t) >= float(v_t)):
                    return False
                if op == "<=" and not (float(actual_t) <= float(v_t)):
                    return False
                break
        else:
            # No operator matched; treat as 'key exists' filter.
            if clause.strip() not in metadata:
                return False
    return True


__all__ = [
    "VectorDocument",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "MilvusAdapter",
]
