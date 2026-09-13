"""
icore.observability - v0.6 可观测性体系（Metrics / Tracing / 结构化日志）。

提供三大能力，全部基于 Python 标准库 ``logging`` 实现，**不依赖
loguru / opentelemetry / prometheus_client**：

    1. **Metrics**：``MetricsRegistry`` 单例注册中心，支持
       ``Counter`` / ``Histogram`` / ``Gauge`` 三种指标类型，预置 14 个
       v0.6 指标，并通过 ``render_prometheus()`` 输出 Prometheus text
       exposition format。
    2. **Structured Logging**：``JsonFormatter`` 输出结构化 JSON 日志，
       透传 trace_id / task_id / workflow_name 等上下文字段以及任意
       ``extra`` 字段；``setup_observability()`` 一键配置。
    3. **Tracing**：``trace_context`` + ``start_span()`` 实现轻量级
       span 树（嵌套 + 异常捕获），不依赖 OpenTelemetry。

模块依赖：
    仅依赖标准库 + ``icore`` 自身。线程安全使用 ``threading.Lock``（指标
    操作是同步的，无需 asyncio.Lock 的开销与 await 污染）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

__all__ = [
    # Metric types
    "Counter",
    "Histogram",
    "Gauge",
    "DEFAULT_HISTOGRAM_BUCKETS",
    # Registry
    "MetricsRegistry",
    "get_metrics_registry",
    # Logging
    "JsonFormatter",
    "setup_observability",
    # Tracing
    "Span",
    "TraceContext",
    "trace_context",
    "start_span",
    "current_span",
    "get_trace_id",
    "get_span_id",
]

logger = logging.getLogger(__name__)

# ============================================================================
# Helpers
# ============================================================================

# Default Prometheus-style histogram buckets (seconds), matching the v0.6
# design spec: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10].
DEFAULT_HISTOGRAM_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


def _new_trace_id() -> str:
    """Generate a new trace id (32-char hex)."""
    return uuid.uuid4().hex


def _new_span_id() -> str:
    """Generate a new span id (16-char hex)."""
    return uuid.uuid4().hex[:16]


def _escape_label_value(value: Any) -> str:
    """Escape a label value for Prometheus text exposition format."""
    s = str(value)
    # Order matters: escape backslash first.
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    return s


def _format_value(value: float) -> str:
    """Format a metric value for Prometheus output.

    Whole-valued floats render as integers (``42`` not ``42.0``); other
    floats render via ``repr`` to preserve precision.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ============================================================================
# Metric types
# ============================================================================


class Counter:
    """Monotonically increasing counter.

    Counters only go up (``inc`` with a positive ``n``). Use ``inc(**labels)``
    to record one observation for a label combination, and ``get(**labels)``
    to read the current value (0.0 when unseen).
    """

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.help = help_text
        self.labelnames: tuple[str, ...] = tuple(labelnames)
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, Any]) -> tuple[str, ...]:
        if set(labels.keys()) != set(self.labelnames):
            raise ValueError(
                f"Counter '{self.name}' labels mismatch: expected "
                f"{self.labelnames}, got {tuple(labels.keys())}"
            )
        return tuple(str(labels[n]) for n in self.labelnames)

    def inc(self, n: float = 1.0, **labels: Any) -> None:
        """Increment the counter by ``n`` (must be >= 0) for the label set."""
        if n < 0:
            raise ValueError(
                f"Counter '{self.name}' cannot be incremented by a negative value"
            )
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(n)

    def get(self, **labels: Any) -> float:
        """Return the current counter value for the label set (0.0 if unseen)."""
        key = self._key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _format_labels(self, key: tuple[str, ...]) -> str:
        if not self.labelnames:
            return ""
        parts = [
            f'{name}="{_escape_label_value(val)}"'
            for name, val in zip(self.labelnames, key)
        ]
        return "{" + ",".join(parts) + "}"

    def _render_lines(self) -> list[str]:
        with self._lock:
            items = list(self._values.items())
        lines: list[str] = []
        if not items and not self.labelnames:
            # Emit a zero line for labelless counters with no observations.
            lines.append(f"{self.name} 0")
            return lines
        for key, val in items:
            lines.append(f"{self.name}{self._format_labels(key)} {_format_value(val)}")
        return lines


class Histogram:
    """Histogram tracking count, sum, and cumulative bucket counts.

    Buckets default to ``DEFAULT_HISTOGRAM_BUCKETS``. ``observe(value)``
    records one observation; ``get_stats()`` returns ``{count, sum, buckets}``
    where ``buckets`` is a list of ``(upper_bound, cumulative_count)`` pairs.
    """

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_HISTOGRAM_BUCKETS,
    ) -> None:
        self.name = name
        self.help = help_text
        self.labelnames: tuple[str, ...] = tuple(labelnames)
        self.buckets: tuple[float, ...] = tuple(sorted(buckets))
        self._data: dict[tuple[str, ...], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, Any]) -> tuple[str, ...]:
        if set(labels.keys()) != set(self.labelnames):
            raise ValueError(
                f"Histogram '{self.name}' labels mismatch: expected "
                f"{self.labelnames}, got {tuple(labels.keys())}"
            )
        return tuple(str(labels[n]) for n in self.labelnames)

    def _empty_entry(self) -> dict[str, Any]:
        return {
            "count": 0,
            "sum": 0.0,
            "buckets": [0] * len(self.buckets),
        }

    def observe(self, value: float, **labels: Any) -> None:
        """Record a single observation ``value`` for the label set."""
        key = self._key(labels)
        v = float(value)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                entry = self._empty_entry()
                self._data[key] = entry
            entry["count"] += 1
            entry["sum"] += v
            # Cumulative buckets: every bucket whose le >= value increments.
            for i, upper in enumerate(self.buckets):
                if v <= upper:
                    entry["buckets"][i] += 1

    def get_stats(self, **labels: Any) -> dict[str, Any]:
        """Return ``{count, sum, buckets}`` for the label set.

        ``buckets`` is a list of ``(upper_bound, cumulative_count)`` tuples.
        Unseen label sets return zeros.
        """
        key = self._key(labels)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "buckets": [(b, 0) for b in self.buckets],
                }
            return {
                "count": entry["count"],
                "sum": entry["sum"],
                "buckets": [
                    (b, c) for b, c in zip(self.buckets, entry["buckets"])
                ],
            }

    def _format_labels(self, key: tuple[str, ...], extra: list[tuple[str, str]]) -> str:
        parts: list[str] = []
        for name, val in zip(self.labelnames, key):
            parts.append(f'{name}="{_escape_label_value(val)}"')
        for name, val in extra:
            parts.append(f'{name}="{_escape_label_value(val)}"')
        if not parts:
            return ""
        return "{" + ",".join(parts) + "}"

    def _render_lines(self) -> list[str]:
        with self._lock:
            items = list(self._data.items())
        lines: list[str] = []
        if not items and not self.labelnames:
            # Emit a zero set for labelless histograms with no observations.
            label_str = self._format_labels((), [("le", "+Inf")])
            lines.append(f"{self.name}_bucket{label_str} 0")
            lines.append(f"{self.name}_sum 0")
            lines.append(f"{self.name}_count 0")
            return lines
        for key, entry in items:
            cumulative = 0
            for upper, count in zip(self.buckets, entry["buckets"]):
                cumulative = count  # already cumulative
                label_str = self._format_labels(
                    key, [("le", _format_value(upper))]
                )
                lines.append(f"{self.name}_bucket{label_str} {_format_value(cumulative)}")
            # +Inf bucket == total count.
            label_str = self._format_labels(key, [("le", "+Inf")])
            lines.append(
                f"{self.name}_bucket{label_str} {_format_value(entry['count'])}"
            )
            label_str = self._format_labels(key, [])
            lines.append(f"{self.name}_sum{label_str} {_format_value(entry['sum'])}")
            lines.append(
                f"{self.name}_count{label_str} {_format_value(entry['count'])}"
            )
        return lines


class Gauge:
    """Gauge that can go up and down.

    ``set`` replaces the value; ``inc`` / ``dec`` adjust it; ``get`` reads it.
    """

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.help = help_text
        self.labelnames: tuple[str, ...] = tuple(labelnames)
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, Any]) -> tuple[str, ...]:
        if set(labels.keys()) != set(self.labelnames):
            raise ValueError(
                f"Gauge '{self.name}' labels mismatch: expected "
                f"{self.labelnames}, got {tuple(labels.keys())}"
            )
        return tuple(str(labels[n]) for n in self.labelnames)

    def set(self, value: float, **labels: Any) -> None:
        """Set the gauge to ``value`` for the label set."""
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, n: float = 1.0, **labels: Any) -> None:
        """Increment the gauge by ``n`` for the label set."""
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(n)

    def dec(self, n: float = 1.0, **labels: Any) -> None:
        """Decrement the gauge by ``n`` for the label set."""
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - float(n)

    def get(self, **labels: Any) -> float:
        """Return the current gauge value for the label set (0.0 if unseen)."""
        key = self._key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _format_labels(self, key: tuple[str, ...]) -> str:
        if not self.labelnames:
            return ""
        parts = [
            f'{name}="{_escape_label_value(val)}"'
            for name, val in zip(self.labelnames, key)
        ]
        return "{" + ",".join(parts) + "}"

    def _render_lines(self) -> list[str]:
        with self._lock:
            items = list(self._values.items())
        lines: list[str] = []
        if not items and not self.labelnames:
            lines.append(f"{self.name} 0")
            return lines
        for key, val in items:
            lines.append(f"{self.name}{self._format_labels(key)} {_format_value(val)}")
        return lines


# ============================================================================
# MetricsRegistry (singleton)
# ============================================================================

# (metric_name, kind) pairs in registration order for deterministic rendering.
_METRIC_KIND_COUNTER = "counter"
_METRIC_KIND_HISTOGRAM = "histogram"
_METRIC_KIND_GAUGE = "gauge"


class MetricsRegistry:
    """Singleton metrics registry.

    Holds all Counter / Histogram / Gauge instances and pre-registers the
    default icore metrics on construction. ``render_prometheus()`` produces
    the full Prometheus text exposition payload.
    """

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Gauge] = {}
        self._order: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._register_default_metrics()

    # -- registration ------------------------------------------------------

    def create_counter(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
    ) -> Counter:
        """Register and return a Counter. Re-registration returns existing."""
        with self._lock:
            existing = self._counters.get(name)
            if existing is not None:
                return existing
            c = Counter(name, help_text, labelnames)
            self._counters[name] = c
            self._order.append((name, _METRIC_KIND_COUNTER))
            return c

    def create_histogram(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_HISTOGRAM_BUCKETS,
    ) -> Histogram:
        """Register and return a Histogram. Re-registration returns existing."""
        with self._lock:
            existing = self._histograms.get(name)
            if existing is not None:
                return existing
            h = Histogram(name, help_text, labelnames, buckets)
            self._histograms[name] = h
            self._order.append((name, _METRIC_KIND_HISTOGRAM))
            return h

    def create_gauge(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
    ) -> Gauge:
        """Register and return a Gauge. Re-registration returns existing."""
        with self._lock:
            existing = self._gauges.get(name)
            if existing is not None:
                return existing
            g = Gauge(name, help_text, labelnames)
            self._gauges[name] = g
            self._order.append((name, _METRIC_KIND_GAUGE))
            return g

    def get_counter(self, name: str) -> Counter:
        try:
            return self._counters[name]
        except KeyError:
            raise KeyError(f"Counter '{name}' is not registered")

    def get_histogram(self, name: str) -> Histogram:
        try:
            return self._histograms[name]
        except KeyError:
            raise KeyError(f"Histogram '{name}' is not registered")

    def get_gauge(self, name: str) -> Gauge:
        try:
            return self._gauges[name]
        except KeyError:
            raise KeyError(f"Gauge '{name}' is not registered")

    # -- rendering ---------------------------------------------------------

    def render_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        # Snapshot the order under the lock so iteration is stable.
        with self._lock:
            order = list(self._order)
            counters = dict(self._counters)
            histograms = dict(self._histograms)
            gauges = dict(self._gauges)
        for name, kind in order:
            if kind == _METRIC_KIND_COUNTER:
                metric: Any = counters.get(name)
                type_str = "counter"
            elif kind == _METRIC_KIND_HISTOGRAM:
                metric = histograms.get(name)
                type_str = "histogram"
            else:
                metric = gauges.get(name)
                type_str = "gauge"
            if metric is None:
                continue
            # HELP and TYPE lines.
            help_escaped = metric.help.replace("\\", "\\\\").replace("\n", "\\n")
            lines.append(f"# HELP {name} {help_escaped}")
            lines.append(f"# TYPE {name} {type_str}")
            lines.extend(metric._render_lines())
        return "\n".join(lines) + ("\n" if lines else "")

    # -- default metrics ---------------------------------------------------

    def _register_default_metrics(self) -> None:
        """Register the 15 default icore metrics."""
        self.create_counter(
            "icore_invoke_total",
            "Total number of workflow invocations",
            ("workflow_name", "status"),
        )
        self.create_histogram(
            "icore_invoke_duration_seconds",
            "End-to-end workflow invocation latency in seconds",
            ("workflow_name",),
        )
        self.create_histogram(
            "icore_task_duration_seconds",
            "Per-task execution latency in seconds",
            ("task_name",),
        )
        self.create_counter(
            "icore_model_tokens_total",
            "Total LLM tokens consumed",
            ("model_id", "type"),
        )
        self.create_counter(
            "icore_model_errors_total",
            "Total LLM model call errors",
            ("model_id",),
        )
        # v0.6.x: 提供方上下文缓存（如 DeepSeek 的 prompt_cache_hit_tokens）
        # 命中/未命中 token 计数。多轮对话中前缀命中缓存可显著降低输入成本，
        # 这两个指标用于观测缓存命中率与成本节省效果。
        self.create_counter(
            "icore_model_prompt_cache_hit_tokens_total",
            "Total prompt tokens served from provider context cache",
            ("model_id",),
        )
        self.create_counter(
            "icore_model_prompt_cache_miss_tokens_total",
            "Total prompt tokens that missed provider context cache",
            ("model_id",),
        )
        self.create_gauge(
            "icore_circuit_breaker_state",
            "Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
            ("model_id",),
        )
        self.create_gauge(
            "icore_queue_depth",
            "Task queue depth",
            ("queue_name",),
        )
        self.create_gauge(
            "icore_active_instances",
            "Number of active workflow instances",
            (),
        )
        self.create_histogram(
            "icore_vectorstore_latency_seconds",
            "Vector store operation latency in seconds",
            ("collection",),
        )
        self.create_histogram(
            "icore_graphstore_latency_seconds",
            "Graph store operation latency in seconds",
            ("operation",),
        )
        self.create_counter(
            "icore_cache_hits_total",
            "Total cache hits",
            ("cache_name",),
        )
        self.create_counter(
            "icore_cache_misses_total",
            "Total cache misses",
            ("cache_name",),
        )
        # ICORE-ISSUE-005: 通用限流原语（键控固定窗口）判定计数。
        # label 刻意只有 backend/outcome——user_id/ip 等高基数限流键
        # 绝不能作为 label，否则 Prometheus 序列会爆炸。
        self.create_counter(
            "icore_rate_limit_checks_total",
            "Total rate limit checks (keyed fixed-window primitive)",
            ("backend", "outcome"),
        )

    def reset(self) -> None:
        """Reset all metric values to zero (keeps metric definitions).

        Primarily intended for tests; production code should not call this.
        """
        with self._lock:
            for c in self._counters.values():
                with c._lock:
                    c._values.clear()
            for h in self._histograms.values():
                with h._lock:
                    h._data.clear()
            for g in self._gauges.values():
                with g._lock:
                    g._values.clear()


# Module-level singleton -----------------------------------------------------

_registry: Optional[MetricsRegistry] = None
_registry_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Return the process-wide singleton ``MetricsRegistry``.

    Lazily constructed on first call and thread-safe.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = MetricsRegistry()
    return _registry


def _reset_singleton_for_testing() -> None:
    """Reset the module-level singleton (tests only)."""
    global _registry
    with _registry_lock:
        _registry = None


# ============================================================================
# Structured JSON logging
# ============================================================================

# Standard LogRecord attribute names that should NOT be forwarded as extra
# fields. Python 3.12 adds ``taskName``; it is included for safety.
_LOGRECORD_RESERVED: frozenset[str] = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)

# Observability context fields pulled from LogRecord attributes (set via
# ``logger.info(..., extra={...})`` or context propagation).
_CONTEXT_FIELDS: tuple[str, ...] = (
    "trace_id",
    "task_id",
    "workflow_name",
    "span_id",
    "request_id",
)


class JsonFormatter(logging.Formatter):
    """Logging formatter that emits one JSON object per log record.

    Output fields: ``timestamp``, ``level``, ``logger``, ``message``, plus
    any present context field (``trace_id`` / ``task_id`` / ``workflow_name``
    / ``span_id`` / ``request_id``) and any non-standard ``extra`` fields
    passed to the logging call. ``exception`` is added when ``exc_info`` is
    set.
    """

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict[str, Any] = {
            "timestamp": dt.isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Known context fields.
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        # Arbitrary extra fields (anything not reserved and not a context
        # field and not private).
        for key, value in record.__dict__.items():
            if key in _LOGRECORD_RESERVED:
                continue
            if key in _CONTEXT_FIELDS:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_observability(
    level: int = logging.INFO,
    json_logs: bool = True,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """Configure root logging with icore's observability handlers.

    Idempotent: repeated calls do not stack duplicate handlers. When
    ``json_logs`` is True a :class:`JsonFormatter` is used; otherwise a
    plain human-readable formatter is used. When ``log_dir`` is provided, an
    additional :class:`logging.FileHandler` writes to ``<log_dir>/icore.log``.

    Returns the configured root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter: logging.Formatter
    if json_logs:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )

    def _ensure_handler(
        handler: logging.Handler, tag: str
    ) -> None:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        handler._icore_observability_tag = tag  # type: ignore[attr-defined]
        # Avoid stacking duplicates.
        has = any(
            getattr(h, "_icore_observability_tag", None) == tag
            for h in root.handlers
        )
        if not has:
            root.addHandler(handler)

    _ensure_handler(logging.StreamHandler(), "stream")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        _ensure_handler(
            logging.FileHandler(os.path.join(log_dir, "icore.log"), encoding="utf-8"),
            "file",
        )
    return root


# ============================================================================
# Tracing (lightweight, dependency-free)
# ============================================================================


@dataclass
class Span:
    """A single tracing span.

    Attributes:
        name:        Span/human-readable operation name.
        attributes:  Free-form key/value attributes set on the span.
        trace_id:    Trace id shared by all spans in one trace tree.
        span_id:     Unique id of this span.
        parent_id:   Span id of the parent (empty for root spans).
        start_time:  ``time.time()`` when the span started (0 if not started).
        end_time:    ``time.time()`` when the span ended (0 if not ended).
        status:      ``"ok"`` or ``"error"``.
        exception:   The exception recorded on the span, if any.
        events:      List of ``(name, attributes, timestamp)`` events.
    """

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    span_id: str = ""
    parent_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"
    exception: Optional[BaseException] = None
    events: list[tuple[str, dict[str, Any], float]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set or update a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, **attributes: Any) -> None:
        """Record an event on the span."""
        self.events.append((name, dict(attributes), time.time()))

    def record_exception(self, exc: BaseException) -> None:
        """Mark the span as failed and record the exception."""
        self.status = "error"
        self.exception = exc
        self.events.append(
            ("exception", {"type": type(exc).__name__, "message": str(exc)}, time.time())
        )

    @property
    def duration(self) -> float:
        """Span duration in seconds (0.0 if not yet ended)."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def ended(self) -> bool:
        return self.end_time > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the span to a JSON-friendly dict."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "attributes": dict(self.attributes),
            "events": [
                {"name": n, "attributes": a, "timestamp": t}
                for n, a, t in self.events
            ],
        }


class TraceContext:
    """Lightweight tracer backed by a ``ContextVar`` span stack.

    Spans are scoped via ``with trace_context.start_span(...) as span:`` and
    nest naturally: a child span inherits its parent's ``trace_id`` and
    records the parent's ``span_id``.
    """

    def __init__(self) -> None:
        self._stack: ContextVar[tuple[Span, ...]] = ContextVar(
            "icore_trace_stack", default=()
        )

    def current_span(self) -> Optional[Span]:
        """Return the currently active span, or ``None``."""
        stack = self._stack.get()
        return stack[-1] if stack else None

    def current_trace_id(self) -> Optional[str]:
        """Return the active trace id, or ``None`` outside any span."""
        span = self.current_span()
        return span.trace_id if span else None

    @contextmanager
    def start_span(
        self, name: str, **attributes: Any
    ) -> Iterator[Span]:
        """Open a new span, yielding it as a context manager.

        Exceptions raised inside the ``with`` block are recorded on the span
        and re-raised. The span is ended (``end_time`` set) on exit.
        """
        parent = self.current_span()
        trace_id = parent.trace_id if parent is not None else _new_trace_id()
        span = Span(
            name=name,
            attributes=dict(attributes),
            trace_id=trace_id,
            span_id=_new_span_id(),
            parent_id=parent.span_id if parent is not None else "",
        )
        span.start_time = time.time()
        token = self._stack.set(self._stack.get() + (span,))
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            span.record_exception(exc)
            raise
        finally:
            span.end_time = time.time()
            self._stack.reset(token)

    def reset(self) -> None:
        """Clear the current span stack (tests / context resets)."""
        self._stack.set(())


# Module-level tracer + convenience function.
trace_context = TraceContext()


def start_span(name: str, **attributes: Any):
    """Open a span on the default :data:`trace_context`.

    Usage::

        with start_span("workflow.execute", workflow_name="rag_qa") as span:
            ...
    """
    return trace_context.start_span(name, **attributes)


def current_span() -> Optional[Span]:
    """Return the currently active span on the default tracer."""
    return trace_context.current_span()


def get_trace_id() -> Optional[str]:
    """Return the active trace id on the default tracer (``None`` if none)."""
    return trace_context.current_trace_id()


def get_span_id() -> Optional[str]:
    """Return the active span id on the default tracer (``None`` if none)."""
    span = current_span()
    return span.span_id if span else None
