"""
Tests for icore.objectstore - object storage abstraction.

Covers:
    - ``ObjectRef`` dataclass (uri property, default factory for metadata)
    - ``InMemoryObjectStore`` happy path + error path for every method
    - ``BaseObjectStore.put_multipart`` default fallback (concatenation)
    - ``MinIOAdapter`` lazy-import failure, error wrapping, multipart
      abort on failure (mocked SDK — no real MinIO server required)
    - Integration with ``report_export`` workflow (fallback path when
      no objectstore is configured)
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any
from unittest.mock import MagicMock

import pytest

from icore.core.task_context import TaskContext
from icore.exceptions import ValidationError
from icore.objectstore import (
    BaseObjectStore,
    InMemoryObjectStore,
    MinIOAdapter,
    ObjectRef,
)


# ---------------------------------------------------------------------------
# ObjectRef
# ---------------------------------------------------------------------------


class TestObjectRef:
    def test_uri_property(self) -> None:
        ref = ObjectRef(bucket="reports", key="2026/q3.json")
        assert ref.uri == "s3://reports/2026/q3.json"

    def test_default_metadata_is_empty_dict(self) -> None:
        ref = ObjectRef(bucket="b", key="k")
        assert ref.metadata == {}
        # Each instance gets its own dict (not shared)
        ref2 = ObjectRef(bucket="b", key="k")
        ref.metadata["x"] = 1
        assert "x" not in ref2.metadata

    def test_size_bytes_defaults_to_zero(self) -> None:
        ref = ObjectRef(bucket="b", key="k")
        assert ref.size_bytes == 0


# ---------------------------------------------------------------------------
# InMemoryObjectStore
# ---------------------------------------------------------------------------


class TestInMemoryObjectStore:
    @pytest.fixture
    def store(self) -> InMemoryObjectStore:
        return InMemoryObjectStore()

    async def test_put_returns_objectref_with_size(self, store: InMemoryObjectStore) -> None:
        data = b"hello world"
        ref = await store.put("bucket1", "key1", data, content_type="text/plain")
        assert isinstance(ref, ObjectRef)
        assert ref.bucket == "bucket1"
        assert ref.key == "key1"
        assert ref.size_bytes == len(data)
        assert ref.content_type == "text/plain"
        assert ref.uri == "s3://bucket1/key1"

    async def test_put_with_metadata(self, store: InMemoryObjectStore) -> None:
        ref = await store.put(
            "b", "k", b"data",
            metadata={"author": "icore", "version": "1"},
        )
        assert ref.metadata == {"author": "icore", "version": "1"}

    async def test_get_returns_bytes(self, store: InMemoryObjectStore) -> None:
        await store.put("b", "k", b"payload")
        result = await store.get("b", "k")
        assert result == b"payload"

    async def test_get_returns_none_when_missing(self, store: InMemoryObjectStore) -> None:
        result = await store.get("missing", "key")
        assert result is None
        result = await store.get("b", "missing-key")
        assert result is None

    async def test_get_after_delete_returns_none(self, store: InMemoryObjectStore) -> None:
        await store.put("b", "k", b"data")
        await store.delete("b", "k")
        assert await store.get("b", "k") is None

    async def test_delete_returns_true_when_existed(self, store: InMemoryObjectStore) -> None:
        await store.put("b", "k", b"data")
        assert await store.delete("b", "k") is True
        # Second delete returns False (already gone)
        assert await store.delete("b", "k") is False

    async def test_delete_returns_false_when_missing(self, store: InMemoryObjectStore) -> None:
        assert await store.delete("nope", "k") is False

    async def test_list_objects_empty_bucket(self, store: InMemoryObjectStore) -> None:
        await store.make_bucket("empty")
        refs = await store.list_objects("empty")
        assert refs == []

    async def test_list_objects_with_prefix(self, store: InMemoryObjectStore) -> None:
        await store.put("b", "reports/2026/q1.json", b"1")
        await store.put("b", "reports/2026/q2.json", b"2")
        await store.put("b", "reports/2025/q4.json", b"3")
        await store.put("b", "other/data.csv", b"4")

        refs = await store.list_objects("b", prefix="reports/2026/")
        keys = sorted(r.key for r in refs)
        assert keys == ["reports/2026/q1.json", "reports/2026/q2.json"]

    async def test_list_objects_max_keys(self, store: InMemoryObjectStore) -> None:
        for i in range(5):
            await store.put("b", f"k{i}", b"x")
        refs = await store.list_objects("b", max_keys=3)
        assert len(refs) == 3

    async def test_delete_prefix_returns_count(self, store: InMemoryObjectStore) -> None:
        await store.put("b", "logs/2026/a.log", b"1")
        await store.put("b", "logs/2026/b.log", b"2")
        await store.put("b", "logs/2025/c.log", b"3")
        await store.put("b", "other.txt", b"4")

        count = await store.delete_prefix("b", prefix="logs/2026/")
        assert count == 2
        # Verify deletion
        remaining = await store.list_objects("b")
        keys = sorted(r.key for r in remaining)
        assert keys == ["logs/2025/c.log", "other.txt"]

    async def test_delete_prefix_no_match_returns_zero(self, store: InMemoryObjectStore) -> None:
        await store.put("b", "k", b"v")
        count = await store.delete_prefix("b", prefix="nonexistent/")
        assert count == 0

    async def test_bucket_exists(self, store: InMemoryObjectStore) -> None:
        assert await store.bucket_exists("missing") is False
        await store.make_bucket("exists")
        assert await store.bucket_exists("exists") is True

    async def test_make_bucket_idempotent(self, store: InMemoryObjectStore) -> None:
        await store.make_bucket("b")
        await store.make_bucket("b")  # should not raise
        assert await store.bucket_exists("b") is True

    async def test_health_check_always_true(self, store: InMemoryObjectStore) -> None:
        assert await store.health_check() is True

    async def test_presigned_get_url_returns_data_uri(self, store: InMemoryObjectStore) -> None:
        data = b"hello"
        await store.put("b", "k", data, content_type="text/plain")
        url = await store.presigned_get_url("b", "k")
        assert url.startswith("data:text/plain;base64,")
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == data

    async def test_presigned_get_url_missing_object_raises(self, store: InMemoryObjectStore) -> None:
        with pytest.raises(ValidationError):
            await store.presigned_get_url("b", "missing")

    async def test_presigned_put_url_returns_mock_url(self, store: InMemoryObjectStore) -> None:
        url = await store.presigned_put_url("b", "k")
        assert url.startswith("mock://b/k")

    async def test_put_multipart_default_concatenates(self, store: InMemoryObjectStore) -> None:
        chunks = [b"chunk1-", b"chunk2-", b"chunk3"]
        ref = await store.put_multipart("b", "k", chunks, content_type="text/plain")
        assert ref.size_bytes == len(b"chunk1-chunk2-chunk3")
        data = await store.get("b", "k")
        assert data == b"chunk1-chunk2-chunk3"

    async def test_put_overwrites_existing(self, store: InMemoryObjectStore) -> None:
        await store.put("b", "k", b"old")
        await store.put("b", "k", b"new")
        assert await store.get("b", "k") == b"new"


# ---------------------------------------------------------------------------
# BaseObjectStore abstract behaviour
# ---------------------------------------------------------------------------


class TestBaseObjectStoreContract:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            BaseObjectStore()  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self) -> None:
        # InMemoryObjectStore implements all abstract methods.
        store = InMemoryObjectStore()
        assert isinstance(store, BaseObjectStore)


# ---------------------------------------------------------------------------
# MinIOAdapter (no real MinIO server — uses mocked SDK)
# ---------------------------------------------------------------------------


class TestMinIOAdapterLazyImport:
    def test_init_does_not_import_minio(self) -> None:
        # Constructing the adapter must not require the minio package.
        # We force-import to verify this property even if minio is installed.
        adapter = MinIOAdapter(endpoint="x:9000")
        assert adapter._client is None

    async def test_get_client_raises_import_error_when_minio_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = MinIOAdapter()

        async def _fake_import_minio() -> None:
            raise ImportError("minio not installed")

        # Patch the import inside the adapter's _get_client.
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "minio":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="minio is required"):
            await adapter._get_client()


class TestMinIOAdapterWithMockedClient:
    """
    Mock the minio SDK client to exercise MinIOAdapter error paths
    without a live MinIO server.
    """

    @pytest.fixture
    def adapter_with_mock(self) -> tuple[MinIOAdapter, MagicMock]:
        adapter = MinIOAdapter()
        client = MagicMock()
        # Pre-inject the mock client so _get_client short-circuits.
        adapter._client = client  # type: ignore[assignment]
        return adapter, client

    async def test_put_ensures_bucket_and_calls_put_object(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.bucket_exists.return_value = True
        client.put_object.return_value = MagicMock()

        ref = await adapter.put("b", "k", b"data", content_type="text/plain")
        assert ref.bucket == "b"
        assert ref.key == "k"
        assert ref.size_bytes == 4
        client.put_object.assert_called_once()

    async def test_put_creates_bucket_if_missing(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        # First call returns False (bucket missing); make_bucket then succeeds.
        client.bucket_exists.return_value = False
        client.make_bucket.return_value = None
        client.put_object.return_value = MagicMock()

        await adapter.put("b", "k", b"data")
        client.make_bucket.assert_called_once()

    async def test_put_wraps_sdk_error_in_validation_error(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.bucket_exists.return_value = True
        client.put_object.side_effect = RuntimeError("network down")

        with pytest.raises(ValidationError, match="MinIO put_object failed"):
            await adapter.put("b", "k", b"data")

    async def test_get_returns_none_on_s3_error(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock

        class FakeS3Error(Exception):
            pass

        # Simulate NoSuchKey via S3Error class name
        err = FakeS3Error("NoSuchKey")
        client.get_object.side_effect = err
        result = await adapter.get("b", "missing")
        assert result is None

    async def test_get_wraps_unexpected_error(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.get_object.side_effect = RuntimeError("auth failed")

        with pytest.raises(ValidationError, match="MinIO get_object failed"):
            await adapter.get("b", "k")

    async def test_presigned_get_url_wraps_error(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.presigned_get_object.side_effect = RuntimeError("signing failed")

        with pytest.raises(ValidationError, match="presigned_get_url failed"):
            await adapter.presigned_get_url("b", "k")

    async def test_presigned_put_url_ensures_bucket(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.bucket_exists.return_value = True
        client.presigned_put_object.return_value = "https://signed.url/put"

        url = await adapter.presigned_put_url("b", "k", expires_seconds=600)
        assert url == "https://signed.url/put"
        client.presigned_put_object.assert_called_once()

    async def test_list_objects_wraps_error(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.list_objects.side_effect = RuntimeError("list failed")

        with pytest.raises(ValidationError, match="MinIO list_objects failed"):
            await adapter.list_objects("b")

    async def test_delete_returns_false_on_s3_error(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock

        class FakeS3Error(Exception):
            pass

        client.remove_object.side_effect = FakeS3Error("S3Error: NoSuchKey")
        assert await adapter.delete("b", "missing") is False

    async def test_delete_wraps_unexpected_error(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.remove_object.side_effect = RuntimeError("timeout")

        with pytest.raises(ValidationError, match="MinIO delete failed"):
            await adapter.delete("b", "k")

    async def test_bucket_exists_returns_false_on_exception(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.bucket_exists.side_effect = RuntimeError("unreachable")
        assert await adapter.bucket_exists("b") is False

    async def test_make_bucket_ignores_already_exists(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock

        class BucketAlreadyOwnedByYou(Exception):
            pass

        client.make_bucket.side_effect = BucketAlreadyOwnedByYou("already mine")
        # Should not raise — already-owned is idempotent.
        await adapter.make_bucket("b")

    async def test_make_bucket_wraps_other_errors(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.make_bucket.side_effect = RuntimeError("forbidden")

        with pytest.raises(ValidationError, match="MinIO make_bucket failed"):
            await adapter.make_bucket("b")

    async def test_health_check_returns_false_on_failure(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.list_buckets.side_effect = RuntimeError("unreachable")
        assert await adapter.health_check() is False

    async def test_health_check_returns_true_when_ok(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.list_buckets.return_value = []
        assert await adapter.health_check() is True

    async def test_close_resets_client(self, adapter_with_mock: tuple[MinIOAdapter, MagicMock]) -> None:
        adapter, _ = adapter_with_mock
        await adapter.close()
        assert adapter._client is None

    async def test_put_multipart_empty_chunks_raises(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, _ = adapter_with_mock
        with pytest.raises(ValidationError, match="empty payload"):
            await adapter.put_multipart("b", "k", [])

    async def test_put_multipart_aborts_on_failure(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.bucket_exists.return_value = True
        client._create_multipart_upload.return_value = "upload-id-123"
        client._upload_part.side_effect = RuntimeError("part upload failed")
        client._abort_multipart_upload.return_value = None

        with pytest.raises(ValidationError, match="multipart upload failed"):
            await adapter.put_multipart("b", "k", [b"chunk1", b"chunk2"])

        # Abort must have been called to clean up the dangling upload.
        client._abort_multipart_upload.assert_called_once()

    async def test_put_multipart_success(
        self,
        adapter_with_mock: tuple[MinIOAdapter, MagicMock],
    ) -> None:
        adapter, client = adapter_with_mock
        client.bucket_exists.return_value = True
        client._create_multipart_upload.return_value = "upload-id"
        client._upload_part.side_effect = [
            MagicMock(part_number=1),
            MagicMock(part_number=2),
        ]
        client._complete_multipart_upload.return_value = None

        ref = await adapter.put_multipart(
            "b", "k", [b"chunk1", b"chunk2"], content_type="video/mp4"
        )
        assert ref.size_bytes == len(b"chunk1chunk2")
        assert ref.content_type == "video/mp4"
        assert client._upload_part.call_count == 2
        client._complete_multipart_upload.assert_called_once()

    async def test_default_bucket_property(self) -> None:
        adapter = MinIOAdapter(default_bucket="mybucket")
        assert adapter.default_bucket == "mybucket"


# ---------------------------------------------------------------------------
# report_export workflow integration (fallback path)
# ---------------------------------------------------------------------------


class TestReportExportFallback:
    """The report_export workflow must work without an objectstore."""

    def test_workflow_imports_cleanly(self) -> None:
        from icore.workflows.examples.report_export import ReportExportWorkflow
        assert ReportExportWorkflow.name == "report_export"

    async def test_workflow_fallback_when_no_objectstore(self) -> None:
        """When ctx.has_objectstore() is False, upload_to_store returns inline base64."""
        from icore.workflows.examples.report_export import ReportExportWorkflow

        wf = ReportExportWorkflow()
        ctx = TaskContext(task_id="t1", workflow_id="w1")
        # NOTE: No objectstore injected — has_objectstore() returns False.

        result = await wf.execute(
            ctx,
            {
                "data": [{"month": "Jan", "sales": 12000},
                          {"month": "Feb", "sales": 15000}],
                "format": "json",
                "bucket": "reports",
            },
        )
        assert result.is_success
        # Fallback path: file_url is None, inline_data_b64 is populated.
        assert result.data.get("file_url") is None
        assert "inline_data_b64" in result.data
        # Sanity: base64 decodes to JSON bytes.
        decoded = base64.b64decode(result.data["inline_data_b64"])
        assert b"Jan" in decoded or b"sales" in decoded

    async def test_workflow_with_in_memory_objectstore(self) -> None:
        """When an InMemoryObjectStore is injected, a presigned URL is returned."""
        from icore.workflows.examples.report_export import ReportExportWorkflow

        wf = ReportExportWorkflow()
        ctx = TaskContext(task_id="t2", workflow_id="w2")
        ctx.set_objectstore(InMemoryObjectStore())

        result = await wf.execute(
            ctx,
            {
                "data": [{"month": "Jan", "sales": 12000}],
                "format": "csv",
                "bucket": "reports",
            },
        )
        assert result.is_success
        assert result.data.get("file_url") is not None
        assert result.data["file_url"].startswith("data:")
        assert "object_ref" in result.data
        assert result.data["object_ref"]["bucket"] == "reports"
