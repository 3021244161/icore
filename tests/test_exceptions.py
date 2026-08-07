"""
Tests for icore.exceptions - Unified exception hierarchy.

Covers:
    - ICoreError base class attributes (code, http_status, retryable)
    - All business (4xx) and system (5xx) subclasses
    - to_dict() serialization
    - Constructor signatures (positional + keyword args)
    - Exception is raisable and catchable as ICoreError
    - Backward compatibility: BackpressureError carries retry_after
"""

from __future__ import annotations

import pytest

from icore.exceptions import (
    BackpressureError,
    CircuitBreakerOpenError,
    ConflictError,
    DatabaseConnectionError,
    GraphStoreError,
    ICoreError,
    MediaUnsupportedError,
    ModelAPIError,
    ModelNotFoundError,
    ModelTimeoutError,
    ModelUnhealthyError,
    ModelUnsupportedError,
    NoAvailableModelError,
    TaskNotFoundError,
    ValidationError,
    VectorStoreError,
    WorkflowNotFoundError,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TestICoreErrorBase:
    def test_base_attributes(self):
        assert ICoreError.code == "E-INTERNAL"
        assert ICoreError.http_status == 500
        assert ICoreError.retryable is False

    def test_init_with_detail(self):
        err = ICoreError("something broke")
        assert err.detail == "something broke"
        assert str(err) == "something broke"

    def test_init_without_detail(self):
        err = ICoreError()
        assert err.detail == ""

    def test_to_dict_basic(self):
        err = ICoreError("boom")
        d = err.to_dict(task_id="t-123")
        assert d["error"] == "ICoreError"
        assert d["code"] == "E-INTERNAL"
        assert d["detail"] == "boom"
        assert d["retryable"] is False
        assert d["task_id"] == "t-123"

    def test_to_dict_without_task_id(self):
        err = ICoreError("boom")
        d = err.to_dict()
        assert d["task_id"] is None

    def test_raisable_and_catchable(self):
        with pytest.raises(ICoreError):
            raise ICoreError("raise me")

    def test_subclass_caught_by_base(self):
        with pytest.raises(ICoreError):
            raise WorkflowNotFoundError("missing")


# ---------------------------------------------------------------------------
# Business exceptions (4xx)
# ---------------------------------------------------------------------------

class TestBusinessExceptions:
    def test_workflow_not_found(self):
        err = WorkflowNotFoundError("nope")
        assert err.code == "E-WF-001"
        assert err.http_status == 404
        assert err.retryable is False
        assert err.detail == "nope"

    def test_task_not_found(self):
        err = TaskNotFoundError("t-1")
        assert err.code == "E-TASK-001"
        assert err.http_status == 404

    def test_validation_error(self):
        err = ValidationError("bad input")
        assert err.code == "E-VAL-001"
        assert err.http_status == 422

    def test_media_unsupported_error(self):
        err = MediaUnsupportedError("unsupported type")
        assert err.code == "E-MEDIA-001"
        assert err.http_status == 400

    def test_model_not_found_includes_model_id(self):
        err = ModelNotFoundError("gpt-99", available=["gpt-4o"])
        assert err.code == "E-MODEL-004"
        assert err.http_status == 404
        assert err.model_id == "gpt-99"
        assert "gpt-99" in str(err)
        assert "gpt-4o" in str(err)

    def test_model_not_found_no_available(self):
        err = ModelNotFoundError("gpt-99")
        assert err.model_id == "gpt-99"
        assert "Available" not in str(err)

    def test_model_unsupported_error(self):
        err = ModelUnsupportedError("no vision")
        assert err.code == "E-MODEL-006"
        assert err.http_status == 400


# ---------------------------------------------------------------------------
# System exceptions (5xx, mostly retryable)
# ---------------------------------------------------------------------------

class TestSystemExceptions:
    def test_database_connection_error(self):
        err = DatabaseConnectionError("conn lost")
        assert err.code == "E-DB-001"
        assert err.http_status == 503
        assert err.retryable is True

    def test_model_api_error_with_kwargs(self):
        err = ModelAPIError(
            "500 from openai",
            model_id="gpt-4o",
            status_code=500,
        )
        assert err.code == "E-MODEL-001"
        assert err.http_status == 502
        assert err.retryable is True
        assert err.model_id == "gpt-4o"
        assert err.status_code == 500

    def test_model_api_error_without_kwargs(self):
        err = ModelAPIError("generic")
        assert err.model_id is None
        assert err.status_code is None

    def test_model_timeout_error(self):
        err = ModelTimeoutError("timed out", model_id="gpt-4o")
        assert err.code == "E-MODEL-002"
        assert err.http_status == 504
        assert err.retryable is True
        assert err.model_id == "gpt-4o"

    def test_model_unhealthy_error(self):
        err = ModelUnhealthyError("gpt-4o", detail="disabled")
        assert err.code == "E-MODEL-005"
        assert err.http_status == 503
        assert err.retryable is True
        assert err.model_id == "gpt-4o"
        assert "disabled" in str(err)

    def test_vector_store_error(self):
        err = VectorStoreError("milvus down")
        assert err.code == "E-VEC-001"
        assert err.http_status == 503
        assert err.retryable is True

    def test_graph_store_error(self):
        err = GraphStoreError("neo4j down")
        assert err.code == "E-GRAPH-001"
        assert err.http_status == 503
        assert err.retryable is True

    def test_circuit_breaker_open_error(self):
        err = CircuitBreakerOpenError("breaker open")
        assert err.code == "E-CB-001"
        assert err.http_status == 503
        assert err.retryable is True

    def test_backpressure_error_default_retry_after(self):
        err = BackpressureError()
        assert err.code == "E-BP-001"
        assert err.http_status == 503
        assert err.retryable is True
        assert err.retry_after == 5

    def test_backpressure_error_custom_retry_after(self):
        err = BackpressureError("overloaded", retry_after=10)
        assert err.retry_after == 10
        assert "overloaded" in str(err)

    def test_conflict_error(self):
        err = ConflictError("lock taken")
        assert err.code == "E-CONFLICT-001"
        assert err.http_status == 409
        assert err.retryable is True

    def test_no_available_model_error_default_message(self):
        err = NoAvailableModelError()
        assert err.code == "E-MODEL-003"
        assert err.http_status == 503
        assert err.retryable is True
        assert "No available model" in str(err)

    def test_no_available_model_error_with_detail(self):
        err = NoAvailableModelError("all fallbacks exhausted")
        assert "all fallbacks exhausted" in str(err)


# ---------------------------------------------------------------------------
# to_dict for retryable errors
# ---------------------------------------------------------------------------

class TestToDictSerialization:
    def test_retryable_error_to_dict(self):
        err = ModelAPIError("500", model_id="gpt-4o", status_code=500)
        d = err.to_dict(task_id="t-1")
        assert d["retryable"] is True
        assert d["code"] == "E-MODEL-001"
        assert d["task_id"] == "t-1"

    def test_business_error_to_dict(self):
        err = WorkflowNotFoundError("wf-foo")
        d = err.to_dict()
        assert d["retryable"] is False
        assert d["code"] == "E-WF-001"
        assert d["error"] == "WorkflowNotFoundError"

    def test_class_name_in_to_dict(self):
        err = ConflictError("conflict")
        d = err.to_dict()
        assert d["error"] == "ConflictError"
