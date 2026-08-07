"""
icore.cache - LLM semantic cache module (v0.6).

Provides a two-level cache for LLM responses:

    - **L1 (L1Cache)**:      Exact-match cache keyed by ``md5(prompt)``.
                              Backed by an in-process ``OrderedDict`` with
                              LRU eviction and per-entry TTL.
    - **L2 (L2Cache)**:      Semantic-similarity cache backed by a
                              ``BaseVectorStore``.  Prompts are embedded
                              and compared via cosine similarity; entries
                              above ``similarity_threshold`` are returned
                              as hits.
    - **SemanticCache**:     Combines L1 + L2.  L1 is checked first
                              (fast path); on L1 miss L2 is queried and
                              a hit is back-filled into L1.

Additionally, ``CachedModelAdapter`` decorates any ``BaseModelAdapter``
so that ``chat()`` transparently checks the cache before calling the
underlying model, and writes the response back on a miss.

Design notes:
    - Uses **only** the standard-library ``logging`` module (no loguru).
    - No ORM / SQLAlchemy — the vector store abstraction handles
      persistence; L1 is pure in-memory.
    - All external dependencies (vector stores, embed functions) are
      injected, keeping this module dependency-free at import time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from icore.models.base_adapter import BaseModelAdapter
from icore.vectorstore import BaseVectorStore, VectorDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5_hash(text: str) -> str:
    """Return the hexadecimal MD5 digest of *text*."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity for two equal-length float vectors.

    Returns ``0.0`` for empty / mismatched-length / zero-norm inputs.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _serialize_prompt(
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> str:
    """
    Serialise *messages* + output-affecting kwargs into a stable string.

    Only ``tools``, ``tool_choice`` and ``response_format`` are included
    from *kwargs*; parameters such as ``temperature`` or ``max_tokens``
    that influence generation without changing the semantic meaning of
    the prompt are deliberately ignored so that semantically identical
    requests share a cache key.
    """
    relevant_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ("tools", "tool_choice", "response_format")
    }
    payload = {"messages": messages, "kwargs": relevant_kwargs}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """
    A single cached prompt → response mapping.

    Attributes:
        prompt:       The original prompt string (or serialised key).
        response:     The cached response string.
        prompt_hash:  ``md5(prompt)`` hex digest — used as the L1 key.
        model_id:     The model this entry belongs to.
        created_at:   Unix timestamp (``time.time()``) of creation.
        hit_count:    Number of times this entry was returned as a hit.
        ttl_seconds:  Time-to-live in seconds (``<= 0`` means no expiry).
        metadata:     Free-form metadata (e.g. similarity score for L2).
    """

    prompt: str
    response: str
    prompt_hash: str
    model_id: str
    created_at: float
    hit_count: int = 0
    ttl_seconds: float = 3600.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Return ``True`` if the TTL has elapsed since ``created_at``."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds


# ---------------------------------------------------------------------------
# L1Cache — exact match (md5 hash)
# ---------------------------------------------------------------------------

class L1Cache:
    """
    L1 = exact-match cache.

    Keyed by ``(md5(prompt), model_id)``.  Uses an ``OrderedDict`` so
    that LRU eviction is O(1): ``get`` moves the entry to the end
    (most-recently-used), and when ``size > max_size`` the oldest
    entry (least-recently-used) is popped.

    Each entry has its own TTL; expired entries are lazily evicted on
    ``get``.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
    ) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()

    async def get(self, prompt: str, model_id: str) -> Optional[CacheEntry]:
        """Return a non-expired entry for *prompt*/*model_id*, or ``None``."""
        key = (_md5_hash(prompt), model_id)
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            # Lazy eviction of expired entries.
            del self._store[key]
            return None
        entry.hit_count += 1
        self._store.move_to_end(key)
        return entry

    async def set(
        self,
        prompt: str,
        response: str,
        model_id: str,
        ttl: Optional[float] = None,
    ) -> CacheEntry:
        """Insert or update an entry, evicting LRU victims if needed."""
        prompt_hash = _md5_hash(prompt)
        key = (prompt_hash, model_id)
        entry = CacheEntry(
            prompt=prompt,
            response=response,
            prompt_hash=prompt_hash,
            model_id=model_id,
            created_at=time.time(),
            hit_count=0,
            ttl_seconds=ttl if ttl is not None else self._default_ttl,
        )
        self._store[key] = entry  # __setitem__ places at end
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)
        return entry

    async def invalidate(self, prompt: str, model_id: str) -> bool:
        """Remove the entry for *prompt*/*model_id*; return ``True`` if it existed."""
        key = (_md5_hash(prompt), model_id)
        if key in self._store:
            del self._store[key]
            return True
        return False

    async def clear(self) -> int:
        """Remove all entries; return the number removed."""
        count = len(self._store)
        self._store.clear()
        return count

    async def size(self) -> int:
        """Return the current number of entries."""
        return len(self._store)


# ---------------------------------------------------------------------------
# L2Cache — semantic similarity (vector ANN)
# ---------------------------------------------------------------------------

class L2Cache:
    """
    L2 = semantic-similarity cache.

    Uses a ``BaseVectorStore`` to store prompt embeddings.  On ``get``
    the query prompt is embedded, the vector store is searched for the
    top-k nearest neighbours, and each candidate is checked for:

        1. Same ``model_id``.
        2. Cosine similarity ≥ ``similarity_threshold``.
        3. Not expired (TTL).

    The prompt's embedding vector is stored both in the
    ``VectorDocument.vector`` field and in ``metadata["vector"]`` so
    that cosine similarity can be recomputed reliably regardless of
    whether the vector store returns the vector on search.
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        collection: str = "semantic_cache",
        similarity_threshold: float = 0.95,
        default_ttl: float = 3600.0,
    ) -> None:
        self._vectorstore = vectorstore
        self._embed_fn = embed_fn
        self._collection = collection
        self._similarity_threshold = similarity_threshold
        self._default_ttl = default_ttl
        self._ids: set[str] = set()
        self._collection_ready = False

    async def _ensure_collection(self) -> None:
        """Create the vector collection if it does not yet exist."""
        if self._collection_ready:
            return
        try:
            await self._vectorstore.create_collection(self._collection, 0)
        except Exception as exc:  # noqa: BLE001 — best-effort create
            logger.debug(
                "L2 create_collection('%s') failed (may already exist): %s",
                self._collection,
                exc,
            )
        self._collection_ready = True

    async def get(
        self,
        prompt: str,
        model_id: str,
    ) -> Optional[CacheEntry]:
        """
        Search for a semantically similar cached entry.

        Returns the first candidate (highest similarity) that matches
        *model_id*, passes the threshold, and is not expired.
        """
        try:
            query_vec = await self._embed_fn(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.debug("L2 get: embed failed: %s", exc)
            return None

        try:
            candidates = await self._vectorstore.search(
                self._collection,
                query_vec,
                top_k=10,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("L2 get: search failed: %s", exc)
            return None

        for doc in candidates:
            meta = doc.metadata or {}
            if meta.get("model_id") != model_id:
                continue
            cached_vec = meta.get("vector") or doc.vector
            score = _cosine(query_vec, cached_vec)
            if score < self._similarity_threshold:
                continue
            created_at = float(meta.get("created_at", 0.0))
            ttl = float(meta.get("ttl_seconds", self._default_ttl))
            if ttl > 0 and (time.time() - created_at) > ttl:
                continue  # expired — skip
            cached_prompt = meta.get("prompt", "")
            return CacheEntry(
                prompt=cached_prompt,
                response=meta.get("response", ""),
                prompt_hash=_md5_hash(cached_prompt),
                model_id=model_id,
                created_at=created_at,
                hit_count=int(meta.get("hit_count", 0)),
                ttl_seconds=ttl,
                metadata={"score": score, "doc_id": doc.id},
            )
        return None

    async def set(
        self,
        prompt: str,
        response: str,
        model_id: str,
        ttl: Optional[float] = None,
    ) -> CacheEntry:
        """Embed *prompt* and store the entry in the vector store."""
        await self._ensure_collection()
        ttl_val = ttl if ttl is not None else self._default_ttl
        prompt_hash = _md5_hash(prompt)
        now = time.time()
        doc_id = _md5_hash(f"{prompt}:{model_id}")

        try:
            vec = await self._embed_fn(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("L2 set: embed failed, entry not persisted: %s", exc)
            return CacheEntry(
                prompt=prompt,
                response=response,
                prompt_hash=prompt_hash,
                model_id=model_id,
                created_at=now,
                ttl_seconds=ttl_val,
            )

        meta: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "model_id": model_id,
            "created_at": now,
            "ttl_seconds": ttl_val,
            "hit_count": 0,
            "vector": vec,
            "prompt_hash": prompt_hash,
        }
        doc = VectorDocument(
            id=doc_id,
            vector=vec,
            metadata=meta,
            text=prompt,
        )
        try:
            await self._vectorstore.insert(self._collection, [doc])
            self._ids.add(doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("L2 set: insert failed: %s", exc)

        return CacheEntry(
            prompt=prompt,
            response=response,
            prompt_hash=prompt_hash,
            model_id=model_id,
            created_at=now,
            ttl_seconds=ttl_val,
        )

    async def invalidate(self, prompt: str, model_id: str) -> bool:
        """Delete the entry for *prompt*/*model_id*; return ``True`` if deleted."""
        doc_id = _md5_hash(f"{prompt}:{model_id}")
        self._ids.discard(doc_id)
        try:
            deleted = await self._vectorstore.delete(self._collection, [doc_id])
            return deleted > 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("L2 invalidate failed: %s", exc)
            return False

    async def clear(self) -> int:
        """Drop the entire collection; return the number of entries removed."""
        count = len(self._ids)
        try:
            await self._vectorstore.drop_collection(self._collection)
        except Exception as exc:  # noqa: BLE001
            logger.debug("L2 clear: drop_collection failed: %s", exc)
        self._ids.clear()
        self._collection_ready = False
        return count

    async def size(self) -> int:
        """Return the number of entries tracked by this cache."""
        return len(self._ids)


# ---------------------------------------------------------------------------
# SemanticCache — L1 + L2 combination
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Two-level semantic cache combining L1 (exact) and L2 (semantic).

    Lookup order: L1 → L2.  On an L1 miss but L2 hit, the entry is
    back-filled into L1 so subsequent identical lookups hit L1 directly.

    Use ``enable_l1`` / ``enable_l2`` to selectively disable a layer.
    """

    def __init__(
        self,
        l1: Optional[L1Cache] = None,
        l2: Optional[L2Cache] = None,
        enable_l1: bool = True,
        enable_l2: bool = True,
    ) -> None:
        self._l1: Optional[L1Cache] = l1 if enable_l1 else None
        self._l2: Optional[L2Cache] = l2 if enable_l2 else None
        self._enable_l1 = enable_l1
        self._enable_l2 = enable_l2
        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._l1_hits: int = 0
        self._l2_hits: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def l1(self) -> Optional[L1Cache]:
        return self._l1

    @property
    def l2(self) -> Optional[L2Cache]:
        return self._l2

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def get(
        self,
        prompt: str,
        model_id: str,
    ) -> Optional[CacheEntry]:
        """
        Look up *prompt* in L1 then L2.

        On L1 hit: increment L1 stats and return.
        On L2 hit:  increment L2 stats, back-fill L1, and return.
        On full miss: increment miss counter and return ``None``.
        """
        # L1 — fast path
        if self._l1 is not None:
            entry = await self._l1.get(prompt, model_id)
            if entry is not None:
                self._hits += 1
                self._l1_hits += 1
                return entry

        # L2 — semantic path
        if self._l2 is not None:
            entry = await self._l2.get(prompt, model_id)
            if entry is not None:
                self._hits += 1
                self._l2_hits += 1
                # Back-fill to L1 for future fast-path hits.
                if self._l1 is not None:
                    await self._l1.set(
                        entry.prompt,
                        entry.response,
                        model_id,
                        entry.ttl_seconds,
                    )
                return entry

        # Miss
        self._misses += 1
        return None

    async def set(
        self,
        prompt: str,
        response: str,
        model_id: str,
        ttl: Optional[float] = None,
    ) -> CacheEntry:
        """Write *prompt* → *response* into both L1 and L2 (double-write)."""
        entry: Optional[CacheEntry] = None
        if self._l1 is not None:
            entry = await self._l1.set(prompt, response, model_id, ttl)
        if self._l2 is not None:
            l2_entry = await self._l2.set(prompt, response, model_id, ttl)
            if entry is None:
                entry = l2_entry
        if entry is None:
            # Both layers disabled — create a transient entry.
            entry = CacheEntry(
                prompt=prompt,
                response=response,
                prompt_hash=_md5_hash(prompt),
                model_id=model_id,
                created_at=time.time(),
            )
        return entry

    async def invalidate(self, prompt: str, model_id: str) -> bool:
        """Remove the entry from both layers; ``True`` if either had it."""
        l1_ok = False
        l2_ok = False
        if self._l1 is not None:
            l1_ok = await self._l1.invalidate(prompt, model_id)
        if self._l2 is not None:
            l2_ok = await self._l2.invalidate(prompt, model_id)
        return l1_ok or l2_ok

    async def clear(self) -> int:
        """Clear both layers; return the total number of entries removed."""
        total = 0
        if self._l1 is not None:
            total += await self._l1.clear()
        if self._l2 is not None:
            total += await self._l2.clear()
        return total

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Total hits (L1 + L2)."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total misses."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Hit rate in ``[0.0, 1.0]``; ``0.0`` when no lookups yet."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def l1_hits(self) -> int:
        """Number of L1 hits."""
        return self._l1_hits

    @property
    def l2_hits(self) -> int:
        """Number of L2 hits."""
        return self._l2_hits

    def reset_stats(self) -> None:
        """Reset all hit / miss counters to zero."""
        self._hits = 0
        self._misses = 0
        self._l1_hits = 0
        self._l2_hits = 0


# ---------------------------------------------------------------------------
# CachedModelAdapter — decorator for BaseModelAdapter
# ---------------------------------------------------------------------------

class CachedModelAdapter:
    """
    Decorator that wraps a ``BaseModelAdapter`` with a ``SemanticCache``.

    ``chat()`` serialises the messages into a stable prompt key, checks
    the cache, and on a hit returns the stored response (with
    ``cached = True``).  On a miss the underlying adapter is called,
    the response is written back to the cache, and ``cached = False``
    is returned.

    ``embed()``, ``health_check()`` and ``close()`` are transparently
    delegated to the wrapped adapter.
    """

    def __init__(
        self,
        adapter: BaseModelAdapter,
        cache: SemanticCache,
        model_id: str,
    ) -> None:
        self._adapter = adapter
        self._cache = cache
        self._model_id = model_id

    # ------------------------------------------------------------------
    # Convenience properties delegating to the wrapped adapter
    # ------------------------------------------------------------------

    @property
    def config(self) -> Any:
        return self._adapter.config

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def cache(self) -> SemanticCache:
        return self._cache

    @property
    def adapter(self) -> BaseModelAdapter:
        return self._adapter

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Cache-aware chat completion.

        Flow:
            1. Serialise *messages* + output-affecting kwargs → prompt key.
            2. ``cache.get(prompt, model_id)``.
            3. Hit  → deserialise cached response, tag ``cached=True``.
            4. Miss → call ``adapter.chat()``, write to cache, tag
                      ``cached=False``.
        """
        prompt = _serialize_prompt(messages, **kwargs)
        entry = await self._cache.get(prompt, self._model_id)
        if entry is not None:
            try:
                response = json.loads(entry.response)
                if not isinstance(response, dict):
                    response = {"content": entry.response}
            except (json.JSONDecodeError, TypeError):
                response = {"content": entry.response}
            response["cached"] = True
            try:
                from icore.observability import get_metrics_registry
                get_metrics_registry().get_counter(
                    "icore_cache_hits_total"
                ).inc(cache_name="semantic_cache")
            except Exception:
                pass
            return response

        # Miss - call the underlying model.
        response = await self._adapter.chat(messages, **kwargs)
        try:
            await self._cache.set(
                prompt,
                json.dumps(response, ensure_ascii=False),
                self._model_id,
            )
        except Exception as exc:  # noqa: BLE001 - cache write must not break chat
            logger.warning("CachedModelAdapter: cache.set failed: %s", exc)
        response["cached"] = False
        try:
            from icore.observability import get_metrics_registry
            get_metrics_registry().get_counter(
                "icore_cache_misses_total"
            ).inc(cache_name="semantic_cache")
        except Exception:
            pass
        return response

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Delegate to the wrapped adapter (embeddings are not cached)."""
        return await self._adapter.embed(texts)

    async def health_check(self) -> bool:
        """Delegate to the wrapped adapter."""
        return await self._adapter.health_check()

    async def close(self) -> None:
        """Delegate to the wrapped adapter."""
        await self._adapter.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_semantic_cache(
    vectorstore: Optional[BaseVectorStore] = None,
    embed_fn: Optional[Callable[[str], Awaitable[list[float]]]] = None,
    *,
    enable_l1: bool = True,
    enable_l2: bool = True,
    similarity_threshold: float = 0.95,
    l1_max_size: int = 1000,
    default_ttl: float = 3600.0,
) -> SemanticCache:
    """
    Convenience factory for building a ``SemanticCache``.

    If *vectorstore* or *embed_fn* is ``None`` the L2 layer is
    automatically disabled (L2 cannot function without both).
    """
    l1: Optional[L1Cache] = None
    if enable_l1:
        l1 = L1Cache(max_size=l1_max_size, default_ttl=default_ttl)

    effective_enable_l2 = (
        enable_l2 and vectorstore is not None and embed_fn is not None
    )
    l2: Optional[L2Cache] = None
    if effective_enable_l2:
        assert vectorstore is not None and embed_fn is not None  # for type-checkers
        l2 = L2Cache(
            vectorstore=vectorstore,
            embed_fn=embed_fn,
            similarity_threshold=similarity_threshold,
            default_ttl=default_ttl,
        )

    return SemanticCache(
        l1=l1,
        l2=l2,
        enable_l1=enable_l1,
        enable_l2=effective_enable_l2,
    )


__all__ = [
    "CacheEntry",
    "L1Cache",
    "L2Cache",
    "SemanticCache",
    "CachedModelAdapter",
    "create_semantic_cache",
]
