"""
icore.observability.prometheus_endpoint - Prometheus /metrics response helper.

Provides :func:`prometheus_response`, a small helper that renders the
process-wide :class:`~icore.observability.MetricsRegistry` to the Prometheus
text exposition format and wraps it in a ``text/plain`` Starlette
:class:`~starlette.responses.PlainTextResponse`.

Intended usage (by callers wiring their own ``/metrics`` route; this module
does NOT add a route to ``icore/api/main.py``)::

    from fastapi import FastAPI
    from icore.observability.prometheus_endpoint import prometheus_response

    app = FastAPI()

    @app.get("/metrics")
    async def metrics():
        return prometheus_response()
"""

from __future__ import annotations

from starlette.responses import PlainTextResponse

from icore.observability import MetricsRegistry, get_metrics_registry

# Prometheus text exposition format content type (version 0.0.4).
_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def prometheus_response(
    registry: MetricsRegistry | None = None,
) -> PlainTextResponse:
    """Return a ``PlainTextResponse`` containing the Prometheus metrics dump.

    Args:
        registry: Optional ``MetricsRegistry`` to render. Defaults to the
            process-wide singleton returned by
            :func:`icore.observability.get_metrics_registry`.
    """
    reg = registry if registry is not None else get_metrics_registry()
    body = reg.render_prometheus()
    return PlainTextResponse(body, media_type=_PROMETHEUS_CONTENT_TYPE)


__all__ = ["prometheus_response"]
