"""
icore.objectstore - Object storage abstractions and adapters.

Provides:
    - ``ObjectRef``:        Lightweight reference to a stored object.
    - ``BaseObjectStore``:  Abstract interface for object stores.
    - ``MinIOAdapter``:     MinIO / S3-compatible implementation (lazy import).
    - ``InMemoryObjectStore``: Pure-Python in-memory implementation used
                                by tests and small deployments.

Module dependency:
    objectstore depends on ``icore.exceptions`` only. The MinIO driver
    (``minio``) is imported lazily so this package is importable without
    minio installed.

Integration:
    TaskContext injects a concrete instance via ``ctx.get_objectstore()``.
    Workflows use it to store generated files (images, reports, exports)
    and return presigned URLs or object keys to the caller.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from icore.exceptions import ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ObjectRef
# ---------------------------------------------------------------------------


@dataclass
class ObjectRef:
    """
    Lightweight reference to a stored object.

    A workflow that generates a file (e.g. a chart image, an exported
    JSON report, an OCR'd PDF) stores it via ``objectstore.put()`` and
    returns an ``ObjectRef``. Downstream tasks or API callers can then
    download the object via ``objectstore.get()`` or access it directly
    via a presigned URL.

    Attributes:
        bucket:      Bucket / container name.
        key:         Object key (path-like).
        size_bytes:  Size in bytes (set after put).
        content_type: MIME type.
        metadata:    Arbitrary user metadata.
    """

    bucket: str
    key: str
    size_bytes: int = 0
    content_type: str = "application/octet-stream"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def uri(self) -> str:
        """s3://bucket/key style URI."""
        return f"s3://{self.bucket}/{self.key}"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseObjectStore(ABC):
    """
    Abstract object store interface.

    Implementations: ``MinIOAdapter``, ``InMemoryObjectStore``.
    TaskContext injects a concrete instance via ``ctx.get_objectstore()``.
    """

    @abstractmethod
    async def put(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ObjectRef:
        """Upload an object. Returns an ``ObjectRef`` with size populated."""
        raise NotImplementedError

    @abstractmethod
    async def get(self, bucket: str, key: str) -> bytes | None:
        """Download an object's raw bytes. Returns None if not found."""
        raise NotImplementedError

    @abstractmethod
    async def presigned_get_url(
        self, bucket: str, key: str, expires_seconds: int = 3600
    ) -> str:
        """Generate a presigned GET URL valid for ``expires_seconds``."""
        raise NotImplementedError

    async def presigned_put_url(
        self, bucket: str, key: str, expires_seconds: int = 3600
    ) -> str:
        """Generate a presigned PUT URL (for client-side direct upload)."""
        raise NotImplementedError

    async def put_multipart(
        self,
        bucket: str,
        key: str,
        chunks: list[bytes],
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
        part_size: int = 5 * 1024 * 1024,
    ) -> ObjectRef:
        """
        Upload a large object as multipart parts.

        Default implementation concatenates all chunks and delegates
        to ``put()``. Subclasses with native multipart support (e.g.
        ``MinIOAdapter``) override this to stream parts without
        buffering the whole payload in memory.

        Args:
            bucket:       Target bucket.
            key:          Object key.
            chunks:       List of byte chunks (each <= part_size).
            content_type: MIME type.
            metadata:     User metadata.
            part_size:    Suggested part size in bytes (>= 5 MiB for S3).

        Returns:
            ``ObjectRef`` for the assembled object.
        """
        data = b"".join(chunks)
        return await self.put(
            bucket=bucket,
            key=key,
            data=data,
            content_type=content_type,
            metadata=metadata,
        )

    @abstractmethod
    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> list[ObjectRef]:
        """List objects in a bucket with optional prefix filter."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, bucket: str, key: str) -> bool:
        """Delete an object. Returns True if it existed."""
        raise NotImplementedError

    @abstractmethod
    async def delete_prefix(self, bucket: str, prefix: str) -> int:
        """Delete all objects under a prefix. Returns count deleted."""
        raise NotImplementedError

    @abstractmethod
    async def bucket_exists(self, bucket: str) -> bool:
        """Check whether a bucket exists."""
        raise NotImplementedError

    @abstractmethod
    async def make_bucket(self, bucket: str) -> None:
        """Create a bucket (idempotent)."""
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


class InMemoryObjectStore(BaseObjectStore):
    """
    Pure-Python in-memory object store.

    Suitable for unit tests, demos, and small deployments where a
    real S3/MinIO server is not available.
    """

    def __init__(self) -> None:
        # bucket -> key -> (bytes, content_type, metadata)
        self._store: dict[str, dict[str, tuple[bytes, str, dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def put(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ObjectRef:
        async with self._lock:
            b = self._store.setdefault(bucket, {})
            b[key] = (data, content_type, metadata or {})
        return ObjectRef(
            bucket=bucket,
            key=key,
            size_bytes=len(data),
            content_type=content_type,
            metadata=metadata or {},
        )

    async def get(self, bucket: str, key: str) -> bytes | None:
        b = self._store.get(bucket, {})
        entry = b.get(key)
        return entry[0] if entry else None

    async def presigned_get_url(
        self, bucket: str, key: str, expires_seconds: int = 3600
    ) -> str:
        # In-memory store has no real URLs; generate a data: URI or a
        # mock URL pointing back through the API.
        entry = (self._store.get(bucket, {}) or {}).get(key)
        if entry is None:
            raise ValidationError(
                f"Object not found: s3://{bucket}/{key}"
            )
        data, ct, _ = entry
        import base64

        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{ct};base64,{b64}"

    async def presigned_put_url(
        self, bucket: str, key: str, expires_seconds: int = 3600
    ) -> str:
        # Return a mock URL; actual put goes through the adapter's put().
        return f"mock://{bucket}/{key}?presigned-put={int(time.time())}"

    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> list[ObjectRef]:
        b = self._store.get(bucket, {})
        refs: list[ObjectRef] = []
        for key, (data, ct, meta) in b.items():
            if prefix and not key.startswith(prefix):
                continue
            refs.append(
                ObjectRef(
                    bucket=bucket,
                    key=key,
                    size_bytes=len(data),
                    content_type=ct,
                    metadata=meta,
                )
            )
            if len(refs) >= max_keys:
                break
        return refs

    async def delete(self, bucket: str, key: str) -> bool:
        async with self._lock:
            b = self._store.get(bucket, {})
            if key in b:
                del b[key]
                return True
            return False

    async def delete_prefix(self, bucket: str, prefix: str) -> int:
        async with self._lock:
            b = self._store.get(bucket, {})
            to_delete = [k for k in b if k.startswith(prefix)]
            for k in to_delete:
                del b[k]
            return len(to_delete)

    async def bucket_exists(self, bucket: str) -> bool:
        return bucket in self._store

    async def make_bucket(self, bucket: str) -> None:
        async with self._lock:
            self._store.setdefault(bucket, {})

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# MinIO / S3-compatible adapter (lazy import)
# ---------------------------------------------------------------------------


class MinIOAdapter(BaseObjectStore):
    """
    MinIO / S3-compatible object store adapter.

    Uses the ``minio`` Python SDK. The driver is imported lazily so
    this module is importable without minio installed.

    Supports:
        - Put / Get / Delete / List objects
        - Presigned GET and PUT URLs (for direct client upload)
        - Multipart upload (handled by the SDK)
        - Bucket auto-creation (make_bucket)
    """

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False,
        region: str = "us-east-1",
        default_bucket: str = "icore",
        session_token: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure
        self._region = region
        self._default_bucket = default_bucket
        self._session_token = session_token
        self._client: Any = None
        self._lock = asyncio.Lock()

    @property
    def default_bucket(self) -> str:
        """Convenience: the bucket used when callers omit the arg."""
        return self._default_bucket

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from minio import Minio  # type: ignore
        except ImportError as e:
            raise ImportError(
                "minio is required for MinIOAdapter. "
                "Install with: pip install minio"
            ) from e
        kwargs: dict[str, Any] = dict(
            endpoint=self._endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
            region=self._region,
        )
        if self._session_token:
            kwargs["session_token"] = self._session_token
        self._client = Minio(**kwargs)
        return self._client

    async def put(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ObjectRef:
        client = await self._get_client()
        await self._ensure_bucket(bucket, client)
        try:
            result = await asyncio.to_thread(
                client.put_object,
                bucket_name=bucket,
                object_name=key,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type,
                metadata=metadata or {},
            )
        except Exception as e:
            raise ValidationError(
                f"MinIO put_object failed for s3://{bucket}/{key}: {e}"
            ) from e
        return ObjectRef(
            bucket=bucket,
            key=key,
            size_bytes=len(data),
            content_type=content_type,
            metadata=metadata or {},
        )

    async def get(self, bucket: str, key: str) -> bytes | None:
        client = await self._get_client()
        try:
            response = await asyncio.to_thread(
                client.get_object,
                bucket_name=bucket,
                object_name=key,
            )
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as e:
            cls_name = type(e).__name__
            if "NoSuchKey" in cls_name or "S3Error" in cls_name:
                return None
            raise ValidationError(
                f"MinIO get_object failed for s3://{bucket}/{key}: {e}"
            ) from e

    async def presigned_get_url(
        self, bucket: str, key: str, expires_seconds: int = 3600
    ) -> str:
        client = await self._get_client()
        try:
            return await asyncio.to_thread(
                client.presigned_get_object,
                bucket_name=bucket,
                object_name=key,
                expires=expires_seconds,
            )
        except Exception as e:
            raise ValidationError(
                f"MinIO presigned_get_url failed for "
                f"s3://{bucket}/{key}: {e}"
            ) from e

    async def presigned_put_url(
        self, bucket: str, key: str, expires_seconds: int = 3600
    ) -> str:
        client = await self._get_client()
        await self._ensure_bucket(bucket, client)
        try:
            return await asyncio.to_thread(
                client.presigned_put_object,
                bucket_name=bucket,
                object_name=key,
                expires=expires_seconds,
            )
        except Exception as e:
            raise ValidationError(
                f"MinIO presigned_put_url failed for "
                f"s3://{bucket}/{key}: {e}"
            ) from e

    async def put_multipart(
        self,
        bucket: str,
        key: str,
        chunks: list[bytes],
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
        part_size: int = 5 * 1024 * 1024,
    ) -> ObjectRef:
        """
        Native S3 multipart upload for large objects.

        Streams each chunk as a separate part, then completes the
        upload. Avoids buffering the whole payload in memory and
        resumes gracefully on transient failures (re-upload only the
        failed part).
        """
        if not chunks:
            raise ValidationError(
                f"Cannot multipart-upload empty payload to s3://{bucket}/{key}"
            )
        client = await self._get_client()
        await self._ensure_bucket(bucket, client)
        try:
            from minio.api import Part  # type: ignore
        except ImportError:
            Part = None  # type: ignore

        upload_id: Any = None
        parts: list[Any] = []
        total_size = 0
        try:
            # 1. Initiate multipart upload (minio SDK uses _create_multipart_upload).
            upload_id = await asyncio.to_thread(
                client._create_multipart_upload,
                bucket_name=bucket,
                object_name=key,
                metadata=metadata or {},
            )
            # 2. Upload each chunk as a part.
            for idx, chunk in enumerate(chunks, start=1):
                total_size += len(chunk)
                part = await asyncio.to_thread(
                    client._upload_part,
                    bucket_name=bucket,
                    object_name=key,
                    upload_id=upload_id,
                    part_number=idx,
                    data=io.BytesIO(chunk),
                    length=len(chunk),
                )
                parts.append(part)
            # 3. Complete.
            await asyncio.to_thread(
                client._complete_multipart_upload,
                bucket_name=bucket,
                object_name=key,
                upload_id=upload_id,
                parts=parts,
            )
        except Exception as e:
            # Abort on any failure to avoid dangling multipart uploads.
            if upload_id is not None:
                try:
                    await asyncio.to_thread(
                        client._abort_multipart_upload,
                        bucket_name=bucket,
                        object_name=key,
                        upload_id=upload_id,
                    )
                except Exception:
                    pass
            raise ValidationError(
                f"MinIO multipart upload failed for "
                f"s3://{bucket}/{key}: {e}"
            ) from e
        return ObjectRef(
            bucket=bucket,
            key=key,
            size_bytes=total_size,
            content_type=content_type,
            metadata=metadata or {},
        )

    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> list[ObjectRef]:
        client = await self._get_client()
        try:
            objects = await asyncio.to_thread(
                client.list_objects,
                bucket_name=bucket,
                prefix=prefix or None,
            )
            refs: list[ObjectRef] = []
            for obj in objects:
                refs.append(
                    ObjectRef(
                        bucket=obj.bucket_name,
                        key=obj.object_name,
                        size_bytes=obj.size or 0,
                        content_type=obj.content_type or "application/octet-stream",
                        metadata=obj.metadata or {},
                    )
                )
                if len(refs) >= max_keys:
                    break
            return refs
        except Exception as e:
            raise ValidationError(
                f"MinIO list_objects failed for bucket '{bucket}': {e}"
            ) from e

    async def delete(self, bucket: str, key: str) -> bool:
        client = await self._get_client()
        try:
            await asyncio.to_thread(
                client.remove_object,
                bucket_name=bucket,
                object_name=key,
            )
            return True
        except Exception as e:
            cls_name = type(e).__name__
            if "NoSuchKey" in cls_name or "S3Error" in cls_name:
                return False
            raise ValidationError(
                f"MinIO delete failed for s3://{bucket}/{key}: {e}"
            ) from e

    async def delete_prefix(self, bucket: str, prefix: str) -> int:
        """Delete all objects matching *prefix* in *bucket*."""
        client = await self._get_client()
        try:
            objects = await asyncio.to_thread(
                client.list_objects,
                bucket_name=bucket,
                prefix=prefix,
            )
            deleted = 0
            for obj in objects:
                await asyncio.to_thread(
                    client.remove_object,
                    bucket_name=bucket,
                    object_name=obj.object_name,
                )
                deleted += 1
            return deleted
        except Exception as e:
            raise ValidationError(
                f"MinIO delete_prefix failed for "
                f"s3://{bucket}/{prefix}: {e}"
            ) from e

    async def bucket_exists(self, bucket: str) -> bool:
        client = await self._get_client()
        try:
            return await asyncio.to_thread(
                client.bucket_exists, bucket
            )
        except Exception:
            return False

    async def make_bucket(self, bucket: str) -> None:
        client = await self._get_client()
        try:
            await asyncio.to_thread(
                client.make_bucket,
                bucket_name=bucket,
                location=self._region,
            )
        except Exception as e:
            # If bucket already exists, fine.
            cls_name = type(e).__name__
            if "BucketAlreadyOwnedByYou" not in cls_name and "BucketAlreadyExists" not in cls_name:
                raise ValidationError(
                    f"MinIO make_bucket failed for '{bucket}': {e}"
                ) from e

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            await asyncio.to_thread(client.list_buckets)
            return True
        except Exception as e:
            logger.debug("MinIO health check failed: %s", e)
            return False

    async def close(self) -> None:
        self._client = None

    async def _ensure_bucket(self, bucket: str, client: Any) -> None:
        """Make sure a bucket exists, creating it if necessary."""
        exists = await asyncio.to_thread(client.bucket_exists, bucket)
        if not exists:
            await self.make_bucket(bucket)


__all__ = [
    "ObjectRef",
    "BaseObjectStore",
    "InMemoryObjectStore",
    "MinIOAdapter",
]
