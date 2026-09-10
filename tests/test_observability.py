"""
Tests for icore.observability (v0.6 可观测性体系).

Covers:
    - Counter inc / get / labels / error paths
    - Histogram observe / get_stats / buckets / error paths
    - Gauge set / inc / dec / get / error paths
    - MetricsRegistry: default metrics, create/get, render_prometheus format,
      reset, singleton
    - JsonFormatter: valid JSON, required fields, context fields, extra
      passthrough, exception
    - setup_observability: handler wiring, idempotency, log_dir, plain formatter
    - Span / trace_context: creation, nesting + trace_id propagation,
      exception capture + reraise, attributes, events, current_span
    - RequestIDMiddleware: generates id, preserves incoming id, sets state
    - MetricsMiddleware: counts invoke_total, records duration, error status,
      extracts workflow_name, body replayed to handler
    - prometheus_response: content type + body
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from icore.observability import (
    DEFAULT_HISTOGRAM_BUCKETS,
    Counter,
    Gauge,
    Histogram,
    JsonFormatter,
    MetricsRegistry,
    Span,
    TraceContext,
    current_span,
    get_metrics_registry,
    get_span_id,
    get_trace_id,
    setup_observability,
    start_span,
    trace_context,
)
from icore.observability import (
    _reset_singleton_for_testing,
)
from icore.observability.middleware import (
    MetricsMiddleware,
    RequestIDMiddleware,
)
from icore.observability.prometheus_endpoint import prometheus_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_registry() -> MetricsRegistry:
    """A brand-new MetricsRegistry (not the singleton)."""
    return MetricsRegistry()


@pytest.fixture(autouse=True)
def _isolate_trace_context():
    """Reset the global trace_context stack between tests."""
    trace_context.reset()
    yield
    trace_context.reset()


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Snapshot/restore root logger handlers + level around each test."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)


# ===========================================================================
# Counter
# ===========================================================================

class TestCounter:
    def test_inc_default_increments_by_one(self):
        c = Counter("c", "help", ("k",))
        c.inc(k="a")
        assert c.get(k="a") == 1.0

    def test_inc_with_explicit_amount(self):
        c = Counter("c", "help", ("k",))
        c.inc(5.0, k="a")
        c.inc(2.0, k="a")
        assert c.get(k="a") == 7.0

    def test_inc_accumulates_across_label_values(self):
        c = Counter("c", "help", ("k",))
        c.inc(k="a")
        c.inc(k="b")
        c.inc(k="a")
        assert c.get(k="a") == 2.0
        assert c.get(k="b") == 1.0

    def test_get_unseen_label_returns_zero(self):
        c = Counter("c", "help", ("k",))
        assert c.get(k="missing") == 0.0

    def test_inc_negative_raises(self):
        c = Counter("c", "help", ("k",))
        with pytest.raises(ValueError):
            c.inc(-1.0, k="a")

    def test_inc_wrong_labels_raises(self):
        c = Counter("c", "help", ("k",))
        with pytest.raises(ValueError):
            c.inc(other="x")


# ===========================================================================
# Histogram
# ===========================================================================

class TestHistogram:
    def test_observe_records_count_and_sum(self):
        h = Histogram("h", "help", ("k",))
        h.observe(0.1, k="a")
        h.observe(0.3, k="a")
        stats = h.get_stats(k="a")
        assert stats["count"] == 2
        assert abs(stats["sum"] - 0.4) < 1e-9

    def test_observe_buckets_are_cumulative(self):
        h = Histogram("h", "help", ("k",), buckets=(0.1, 0.5, 1.0))
        h.observe(0.05, k="a")   # <= 0.1, <= 0.5, <= 1.0
        h.observe(0.2, k="a")    # <= 0.5, <= 1.0
        h.observe(2.0, k="a")    # none
        stats = h.get_stats(k="a")
        bucket_dict = dict(stats["buckets"])
        assert bucket_dict[0.1] == 1
        assert bucket_dict[0.5] == 2
        assert bucket_dict[1.0] == 2
        assert stats["count"] == 3

    def test_get_stats_unseen_returns_zeros(self):
        h = Histogram("h", "help", ("k",))
        stats = h.get_stats(k="missing")
        assert stats["count"] == 0
        assert stats["sum"] == 0.0
        assert all(c == 0 for _, c in stats["buckets"])

    def test_default_buckets_match_spec(self):
        assert DEFAULT_HISTOGRAM_BUCKETS == (
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        )

    def test_observe_wrong_labels_raises(self):
        h = Histogram("h", "help", ("k",))
        with pytest.raises(ValueError):
            h.observe(0.1, other="x")


# ===========================================================================
# Gauge
# ===========================================================================

class TestGauge:
    def test_set_replaces_value(self):
        g = Gauge("g", "help", ("k",))
        g.set(10.0, k="a")
        g.set(3.0, k="a")
        assert g.get(k="a") == 3.0

    def test_inc_and_dec_adjust_value(self):
        g = Gauge("g", "help", ("k",))
        g.inc(5.0, k="a")
        g.dec(2.0, k="a")
        assert g.get(k="a") == 3.0
        g.inc(k="a")  # default +1
        g.dec(k="a")  # default -1
        assert g.get(k="a") == 3.0

    def test_get_unseen_returns_zero(self):
        g = Gauge("g", "help", ("k",))
        assert g.get(k="missing") == 0.0

    def test_wrong_labels_raises(self):
        g = Gauge("g", "help", ("k",))
        with pytest.raises(ValueError):
            g.set(1.0, other="x")


# ===========================================================================
# MetricsRegistry
# ===========================================================================

class TestMetricsRegistry:
    def test_default_metrics_registered(self, fresh_registry):
        # The 12 v0.6 metrics must be pre-registered with the right types.
        for n in [
            "icore_invoke_total",
            "icore_model_tokens_total",
            "icore_model_errors_total",
            "icore_model_prompt_cache_hit_tokens_total",
            "icore_model_prompt_cache_miss_tokens_total",
            "icore_cache_hits_total",
            "icore_cache_misses_total",
        ]:
            assert isinstance(fresh_registry.get_counter(n), Counter)
        for n in [
            "icore_invoke_duration_seconds",
            "icore_task_duration_seconds",
            "icore_vectorstore_latency_seconds",
            "icore_graphstore_latency_seconds",
        ]:
            assert isinstance(fresh_registry.get_histogram(n), Histogram)
        for n in [
            "icore_circuit_breaker_state",
            "icore_queue_depth",
            "icore_active_instances",
        ]:
            assert isinstance(fresh_registry.get_gauge(n), Gauge)

    def test_default_metric_count_is_at_least_ten(self, fresh_registry):
        out = fresh_registry.render_prometheus()
        # Count TYPE declarations; v0.6 spec requires >= 10 metrics.
        type_lines = [l for l in out.splitlines() if l.startswith("# TYPE ")]
        assert len(type_lines) >= 10

    def test_create_counter_returns_existing_on_reregister(self, fresh_registry):
        c1 = fresh_registry.create_counter("x", "h", ("a",))
        c2 = fresh_registry.create_counter("x", "h", ("a",))
        assert c1 is c2

    def test_get_counter_missing_raises(self, fresh_registry):
        with pytest.raises(KeyError):
            fresh_registry.get_counter("nope")

    def test_render_prometheus_has_help_and_type(self, fresh_registry):
        fresh_registry.get_counter("icore_invoke_total").inc(
            workflow_name="wf", status="success"
        )
        out = fresh_registry.render_prometheus()
        assert "# HELP icore_invoke_total" in out
        assert "# TYPE icore_invoke_total counter" in out

    def test_render_prometheus_counter_line_format(self, fresh_registry):
        fresh_registry.get_counter("icore_invoke_total").inc(
            workflow_name="rag_qa", status="success"
        )
        out = fresh_registry.render_prometheus()
        line = (
            'icore_invoke_total{workflow_name="rag_qa",status="success"} 1'
        )
        assert line in out

    def test_render_prometheus_histogram_lines(self, fresh_registry):
        fresh_registry.get_histogram("icore_invoke_duration_seconds").observe(
            0.05, workflow_name="wf"
        )
        out = fresh_registry.render_prometheus()
        assert "# TYPE icore_invoke_duration_seconds histogram" in out
        assert "icore_invoke_duration_seconds_count" in out
        assert "icore_invoke_duration_seconds_sum" in out
        # +Inf bucket equals the count.
        assert 'icore_invoke_duration_seconds_bucket{workflow_name="wf",le="+Inf"} 1' in out

    def test_render_prometheus_gauge_line(self, fresh_registry):
        fresh_registry.get_gauge("icore_queue_depth").set(7, queue_name="default")
        out = fresh_registry.render_prometheus()
        assert "# TYPE icore_queue_depth gauge" in out
        assert 'icore_queue_depth{queue_name="default"} 7' in out

    def test_render_label_value_escaping(self, fresh_registry):
        fresh_registry.get_gauge("icore_queue_depth").set(
            1, queue_name='a"b\\c'
        )
        out = fresh_registry.render_prometheus()
        assert 'queue_name="a\\"b\\\\c"' in out

    def test_reset_clears_values(self, fresh_registry):
        c = fresh_registry.get_counter("icore_invoke_total")
        c.inc(workflow_name="wf", status="success")
        assert c.get(workflow_name="wf", status="success") == 1.0
        fresh_registry.reset()
        assert c.get(workflow_name="wf", status="success") == 0.0


# ===========================================================================
# Singleton
# ===========================================================================

class TestSingleton:
    def test_get_metrics_registry_returns_same_instance(self):
        _reset_singleton_for_testing()
        try:
            a = get_metrics_registry()
            b = get_metrics_registry()
            assert a is b
        finally:
            _reset_singleton_for_testing()

    def test_singleton_has_default_metrics(self):
        _reset_singleton_for_testing()
        try:
            reg = get_metrics_registry()
            assert isinstance(reg.get_counter("icore_invoke_total"), Counter)
            assert isinstance(
                reg.get_histogram("icore_invoke_duration_seconds"), Histogram
            )
            assert isinstance(reg.get_gauge("icore_active_instances"), Gauge)
        finally:
            _reset_singleton_for_testing()


# ===========================================================================
# JsonFormatter
# ===========================================================================

class TestJsonFormatter:
    def _make_record(
        self,
        msg: str = "hello",
        args: tuple = (),
        level: int = logging.INFO,
        exc_info: Any = None,
        extra: dict | None = None,
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name="icore.test",
            level=level,
            pathname=__file__,
            lineno=10,
            msg=msg,
            args=args,
            exc_info=exc_info,
        )
        if extra:
            for k, v in extra.items():
                setattr(record, k, v)
        return record

    def test_output_is_valid_json(self):
        line = JsonFormatter().format(self._make_record("hi %s", ("there",)))
        data = json.loads(line)  # must not raise
        assert data["message"] == "hi there"

    def test_has_required_fields(self):
        line = JsonFormatter().format(self._make_record())
        data = json.loads(line)
        for field in ("timestamp", "level", "logger", "message"):
            assert field in data
        assert data["level"] == "INFO"
        assert data["logger"] == "icore.test"
        assert data["message"] == "hello"
        # ISO 8601 with Z suffix.
        assert re.match(r"^\d{4}-\d{2}-\d{2}T.+Z$", data["timestamp"])

    def test_context_fields_from_extra(self):
        line = JsonFormatter().format(
            self._make_record(
                extra={"trace_id": "abc123", "task_id": "t-1",
                       "workflow_name": "rag_qa"}
            )
        )
        data = json.loads(line)
        assert data["trace_id"] == "abc123"
        assert data["task_id"] == "t-1"
        assert data["workflow_name"] == "rag_qa"

    def test_arbitrary_extra_passthrough(self):
        line = JsonFormatter().format(
            self._make_record(extra={"user_id": 42, "component": "engine"})
        )
        data = json.loads(line)
        assert data["user_id"] == 42
        assert data["component"] == "engine"

    def test_exception_field_present(self):
        try:
            raise ValueError("boom")
        except ValueError:
            record = self._make_record(
                "oops", level=logging.ERROR, exc_info=sys.exc_info()
            )
        data = json.loads(JsonFormatter().format(record))
        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "boom" in data["exception"]


# ===========================================================================
# setup_observability
# ===========================================================================

class TestSetupObservability:
    def test_adds_tagged_stream_handler(self):
        root = setup_observability(level=logging.DEBUG, json_logs=True)
        tagged = [
            h for h in root.handlers
            if getattr(h, "_icore_observability_tag", None) == "stream"
        ]
        assert len(tagged) == 1
        assert isinstance(tagged[0].formatter, JsonFormatter)
        assert root.level == logging.DEBUG

    def test_idempotent_does_not_stack_handlers(self):
        setup_observability()
        setup_observability()
        setup_observability()
        stream_handlers = [
            h for h in logging.getLogger().handlers
            if getattr(h, "_icore_observability_tag", None) == "stream"
        ]
        assert len(stream_handlers) == 1

    def test_log_dir_creates_file_handler(self, tmp_path):
        log_dir = tmp_path / "logs"
        setup_observability(json_logs=True, log_dir=str(log_dir))
        file_handlers = [
            h for h in logging.getLogger().handlers
            if getattr(h, "_icore_observability_tag", None) == "file"
        ]
        assert len(file_handlers) == 1
        assert log_dir.exists()
        # Emit a record and confirm it lands in the file.
        logging.getLogger("icore.test").info("written")
        file_handlers[0].flush()
        contents = (log_dir / "icore.log").read_text(encoding="utf-8")
        assert "written" in contents

    def test_plain_formatter_when_json_disabled(self):
        root = setup_observability(json_logs=False)
        tagged = [
            h for h in root.handlers
            if getattr(h, "_icore_observability_tag", None) == "stream"
        ]
        assert not isinstance(tagged[0].formatter, JsonFormatter)


# ===========================================================================
# Span / trace_context
# ===========================================================================

class TestSpan:
    def test_start_span_yields_span_with_ids(self):
        with start_span("op") as span:
            assert isinstance(span, Span)
            assert span.name == "op"
            assert span.trace_id
            assert span.span_id
            assert span.parent_id == ""
            assert span.status == "ok"
            assert span.start_time > 0
            assert span.end_time == 0.0
        assert span.ended
        assert span.duration >= 0.0

    def test_set_attribute_and_add_event(self):
        with start_span("op", foo="bar") as span:
            assert span.attributes == {"foo": "bar"}
            span.set_attribute("count", 3)
            span.add_event("checkpoint", phase="mid")
            assert span.attributes["count"] == 3
            assert len(span.events) == 1
            assert span.events[0][0] == "checkpoint"
            assert span.events[0][1] == {"phase": "mid"}

    def test_nested_spans_share_trace_id(self):
        with start_span("parent") as parent:
            parent_trace = parent.trace_id
            assert get_trace_id() == parent_trace
            assert current_span() is parent
            with start_span("child") as child:
                assert child.trace_id == parent_trace
                assert child.parent_id == parent.span_id
                assert current_span() is child
                with start_span("grandchild") as gc:
                    assert gc.trace_id == parent_trace
                    assert gc.parent_id == child.span_id
                    assert current_span() is gc
                # Back to child after grandchild exits.
                assert current_span() is child
            # Back to parent after child exits.
            assert current_span() is parent

    def test_exception_recorded_and_reraised(self):
        with pytest.raises(RuntimeError, match="boom"):
            with start_span("op") as span:
                raise RuntimeError("boom")
        assert span.status == "error"
        assert span.ended
        assert isinstance(span.exception, RuntimeError)
        assert any(name == "exception" for name, _, _ in span.events)

    def test_current_span_none_outside(self):
        assert current_span() is None
        assert get_trace_id() is None
        assert get_span_id() is None

    def test_separate_traces_get_distinct_trace_ids(self):
        with start_span("a") as sa:
            tid_a = sa.trace_id
        with start_span("b") as sb:
            tid_b = sb.trace_id
        assert tid_a != tid_b

    def test_trace_context_reset_clears_stack(self):
        with start_span("op"):
            assert current_span() is not None
        trace_context.reset()
        assert current_span() is None


# ===========================================================================
# Middleware
# ===========================================================================

def _build_app(registry: MetricsRegistry) -> FastAPI:
    """FastAPI app wired with both observability middlewares."""
    app = FastAPI()

    @app.post("/invoke")
    async def invoke(request: Request) -> dict[str, Any]:
        data = await request.json()
        return {
            "workflow_name": data.get("workflow_name"),
            "echo": data,
            "request_id": request.state.request_id,
        }

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/boom")
    async def boom() -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": "boom"})

    # add_middleware: last added runs outermost. Add Metrics first so it
    # wraps the request alongside RequestID (order does not matter for
    # these tests).
    app.add_middleware(MetricsMiddleware, registry=registry)
    app.add_middleware(RequestIDMiddleware)
    return app


class TestRequestIDMiddleware:
    def test_generates_id_when_absent(self):
        app = _build_app(MetricsRegistry())
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        rid = r.headers.get("X-Request-ID")
        assert rid and len(rid) > 0

    def test_preserves_incoming_id(self):
        app = _build_app(MetricsRegistry())
        client = TestClient(app)
        r = client.get("/health", headers={"X-Request-ID": "fixed-id-123"})
        assert r.headers.get("X-Request-ID") == "fixed-id-123"

    def test_sets_state_request_id(self):
        app = _build_app(MetricsRegistry())
        client = TestClient(app)
        r = client.post(
            "/invoke",
            json={"workflow_name": "wf"},
            headers={"X-Request-ID": "rid-abc"},
        )
        assert r.status_code == 200
        assert r.json()["request_id"] == "rid-abc"


class TestMetricsMiddleware:
    def test_records_invoke_total_success(self):
        reg = MetricsRegistry()
        client = TestClient(_build_app(reg))
        client.post("/invoke", json={"workflow_name": "rag_qa"})
        val = reg.get_counter("icore_invoke_total").get(
            workflow_name="rag_qa", status="success"
        )
        assert val == 1.0

    def test_records_invoke_duration(self):
        reg = MetricsRegistry()
        client = TestClient(_build_app(reg))
        client.post("/invoke", json={"workflow_name": "wf"})
        stats = reg.get_histogram("icore_invoke_duration_seconds").get_stats(
            workflow_name="wf"
        )
        assert stats["count"] == 1
        assert stats["sum"] >= 0.0

    def test_error_status_on_500(self):
        reg = MetricsRegistry()
        client = TestClient(_build_app(reg))
        client.get("/boom")
        err = reg.get_counter("icore_invoke_total").get(
            workflow_name="unknown", status="error"
        )
        assert err == 1.0

    def test_extracts_workflow_name_from_body(self):
        reg = MetricsRegistry()
        client = TestClient(_build_app(reg))
        r = client.post(
            "/invoke", json={"workflow_name": "report_export", "params": {}}
        )
        assert r.status_code == 200
        # Body is replayed intact to the handler.
        assert r.json()["echo"]["workflow_name"] == "report_export"
        assert reg.get_counter("icore_invoke_total").get(
            workflow_name="report_export", status="success"
        ) == 1.0

    def test_unknown_workflow_name_when_body_invalid(self):
        reg = MetricsRegistry()
        # raise_server_exceptions=False so the server-side JSONDecodeError is
        # surfaced as a 500 response (middleware still records the metric).
        client = TestClient(_build_app(reg), raise_server_exceptions=False)
        r = client.post(
            "/invoke",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 500
        # Invalid JSON -> recorded as error with unknown workflow_name.
        assert reg.get_counter("icore_invoke_total").get(
            workflow_name="unknown", status="error"
        ) == 1.0

    def test_get_request_does_not_record_invoke(self):
        # GET /health should still record metrics tagged workflow_name=unknown.
        reg = MetricsRegistry()
        client = TestClient(_build_app(reg))
        client.get("/health")
        assert reg.get_counter("icore_invoke_total").get(
            workflow_name="unknown", status="success"
        ) == 1.0


# ===========================================================================
# prometheus_endpoint
# ===========================================================================

class TestPrometheusEndpoint:
    def test_returns_plain_text_with_correct_content_type(self):
        reg = MetricsRegistry()
        reg.get_counter("icore_invoke_total").inc(
            workflow_name="wf", status="success"
        )
        resp = prometheus_response(registry=reg)
        assert resp.media_type == "text/plain; version=0.0.4; charset=utf-8"
        body = resp.body.decode("utf-8") if isinstance(resp.body, bytes) else resp.body
        assert "# HELP icore_invoke_total" in body
        assert 'icore_invoke_total{workflow_name="wf",status="success"} 1' in body

    def test_response_body_is_str_or_bytes(self):
        resp = prometheus_response(registry=MetricsRegistry())
        assert isinstance(resp.body, (bytes, str))


# ===========================================================================
# Prompt cache token metrics (DeepSeek prompt_cache_hit_tokens)
# ===========================================================================

class TestPromptCacheTokenMetrics:
    """openai_adapter.chat() 解析并记录提供方上下文缓存 token 指标。"""

    @staticmethod
    def _resp_with_usage(usage: dict[str, int]):
        class _Resp:
            status_code = 200

            @staticmethod
            def json() -> dict[str, Any]:
                return {
                    "choices": [
                        {
                            "message": {"content": "hi", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                    "model": "deepseek-chat",
                    "usage": usage,
                }

        return _Resp()

    @staticmethod
    def _adapter_with_fake_client(resp: Any):
        from icore.config import ModelConfig
        from icore.models.openai_adapter import OpenAICompatibleAdapter

        cfg = ModelConfig(
            model_id="deepseek",
            model_name="deepseek-chat",
            api_base="http://localhost/v1",
            api_key="k",
        )
        adapter = OpenAICompatibleAdapter(cfg)

        async def _fake_client():
            class _Client:
                async def post(self, *args: Any, **kwargs: Any):
                    return resp

            return _Client()

        adapter._get_client = _fake_client  # type: ignore[assignment]
        return adapter

    async def test_chat_exposes_prompt_cache_tokens(self):
        _reset_singleton_for_testing()
        adapter = self._adapter_with_fake_client(
            self._resp_with_usage({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
            })
        )

        result = await adapter.chat([{"role": "user", "content": "hi"}])

        assert result["usage"]["prompt_cache_hit_tokens"] == 80
        assert result["usage"]["prompt_cache_miss_tokens"] == 20

    async def test_chat_records_prompt_cache_metrics(self):
        _reset_singleton_for_testing()
        adapter = self._adapter_with_fake_client(
            self._resp_with_usage({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
            })
        )

        await adapter.chat([{"role": "user", "content": "hi"}])

        reg = get_metrics_registry()
        assert reg.get_counter(
            "icore_model_prompt_cache_hit_tokens_total"
        ).get(model_id="deepseek") == 80.0
        assert reg.get_counter(
            "icore_model_prompt_cache_miss_tokens_total"
        ).get(model_id="deepseek") == 20.0

    async def test_chat_without_cache_fields_returns_zero(self):
        _reset_singleton_for_testing()
        adapter = self._adapter_with_fake_client(
            self._resp_with_usage({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            })
        )

        result = await adapter.chat([{"role": "user", "content": "hi"}])

        assert result["usage"]["prompt_cache_hit_tokens"] == 0
        assert result["usage"]["prompt_cache_miss_tokens"] == 0
        hit = get_metrics_registry().get_counter(
            "icore_model_prompt_cache_hit_tokens_total"
        ).get(model_id="deepseek")
        assert hit == 0.0
