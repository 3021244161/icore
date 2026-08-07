"""
icore.api.main - FastAPI application with exactly two endpoints.

This module creates the FastAPI application for icore. The API layer
exposes exactly two HTTP endpoints:

    1. GET  /health  - System health check (no parameters)
    2. POST /invoke  - Main interface for workflow invocation

The POST /invoke endpoint is the single entry point for all workflow
execution. It handles:
    - Workflow lookup via WorkflowRegistry
    - Task instance creation (with auto-generated task_id)
    - Model/DB manager injection via TaskContext
    - Synchronous execution (stream=False, no callback)
    - Streaming execution (stream=True, SSE response)
    - Asynchronous execution (callback_url set, returns immediately)

Application state:
    The FastAPI app stores infrastructure managers in app.state:
    - app.state.model_manager:  ModelManager instance (or None)
    - app.state.db_manager:     DBManager instance (or None)
    - app.state.callback_manager: CallbackManager instance

These are injected at startup and used for every invocation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from icore.api.callback import CallbackManager
from icore.api.schemas import HealthResponse, InvokeRequest, InvokeResponse
from icore.api.streaming import SSEStreamHandler
from icore.config import APISettings, Settings, get_settings
from icore.core.task_context import TaskContext
from icore.engine.concurrency_control import BackpressureError
from icore.engine.registry import WorkflowRegistry
from icore.engine.states import TaskState
from icore.exceptions import ICoreError

logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    model_manager: Any = None,
    db_manager: Any = None,
    workflow_registry: Optional[WorkflowRegistry] = None,
    concurrency_controller: Any = None,
    task_queue: Any = None,
    instance_manager: Any = None,
    vectorstore: Any = None,
    graphstore: Any = None,
    media_processor: Any = None,
    lock: Any = None,
    circuit_breaker_registry: Any = None,
    objectstore: Any = None,
    idempotency_cache: Any = None,
    backpressure_coordinator: Any = None,
    hot_reload_coordinator: Any = None,
    degradation_coordinator: Any = None,
    auth_dependency: Any = None,
    auth_bundle: Any = None,
    persistence_manager: Any = None,
    dlq: Any = None,
    metrics_registry: Any = None,
    security_injection_detector: Any = None,
    pii_detector: Any = None,
) -> FastAPI:
    """
    Create and configure the icore FastAPI application.

    This factory function sets up:
        - CORS middleware
        - Exception handlers (KeyError, ValueError, generic Exception)
        - Application state (model_manager, db_manager, callback_manager,
          concurrency_controller, task_queue, instance_manager,
          vectorstore, graphstore, media_processor, lock,
          circuit_breaker_registry, idempotency_cache,
          v0.6 backpressure / hot-reload / degradation coordinators)
        - Two endpoints: GET /health and POST /invoke

    Args:
        settings:              Optional Settings instance. Defaults to
                               get_settings() singleton.
        model_manager:         Optional ModelManager instance. Stored in
                              app.state for injection into TaskContext.
        db_manager:            Optional DBManager instance. Stored in
                              app.state for injection into TaskContext.
        workflow_registry:     Optional WorkflowRegistry instance. Defaults
                              to the global default registry.
        concurrency_controller: Optional ConcurrencyController. When
                              provided, the ``/invoke`` endpoint wraps
                              workflow execution in ``acquire()`` and
                              returns HTTP 503 under backpressure.
        task_queue:            Optional TaskQueue. Used for backpressure
                              depth checks and (future) async dispatch.
        instance_manager:      Optional TaskInstanceManager. When provided,
                              every invocation is registered, deduplicated,
                              and tracked through its lifecycle states.
        vectorstore:           Optional BaseVectorStore (v0.5). Injected
                              into TaskContext for RAG / similarity search.
        graphstore:            Optional BaseGraphStore (v0.5). Injected
                              into TaskContext for knowledge-graph ops.
        media_processor:       Optional MediaProcessorRegistry (v0.5).
                              Injected into TaskContext for multimodal
                              file processing.
        lock:                  Optional BaseDistributedLock (v0.5).
                              Injected into TaskContext for serializing
                              concurrent modifications.
        circuit_breaker_registry: Optional CircuitBreakerRegistry (v0.5).
                              Injected into TaskContext for per-model
                              breaker access. Defaults to the registry
                              owned by ModelManager when None.
        idempotency_cache:     Optional BaseIdempotencyCache (v0.5). When
                              wired, ``/invoke`` short-circuits repeated
                              requests carrying the same idempotency_key.
        backpressure_coordinator: Optional BackpressureCoordinator (v0.6).
                              When wired, ``/health`` reports per-component
                              saturation and ``/invoke`` returns HTTP 503
                              under cross-component backpressure.
        hot_reload_coordinator: Optional HotReloadCoordinator (v0.6).
                              The coordinator is started/stopped by the
                              bootstrap layer; the app only holds a
                              reference for graceful shutdown.
        degradation_coordinator: Optional GracefulDegradationCoordinator
                              (v0.6). When wired, ``/health`` reports
                              per-component degradation state and
                              ``/invoke`` returns HTTP 503 when any
                              required component is UNAVAILABLE.

    Returns:
        A configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    api_settings: APISettings = settings.api

    app = FastAPI(
        title="icore",
        description=(
            "icore - 企业级 LLM 工作流编排平台\\n\\n"
            "Enterprise LLM Workflow Orchestration Platform.\\n\\n"
            "## 端点说明\\n"
            "- `GET /health`: 系统健康检查\\n"
            "- `POST /invoke`: 工作流调用主接口"
        ),
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Store infrastructure in app state ---
    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.db_manager = db_manager
    app.state.callback_manager = CallbackManager(
        timeout=api_settings.request_timeout
    )
    app.state.workflow_registry = (
        workflow_registry
        if workflow_registry is not None
        else WorkflowRegistry.default()
    )
    app.state.concurrency_controller = concurrency_controller
    app.state.task_queue = task_queue
    app.state.instance_manager = instance_manager
    # v0.5 new components.
    app.state.vectorstore = vectorstore
    app.state.graphstore = graphstore
    app.state.media_processor = media_processor
    app.state.lock = lock
    # Default to ModelManager's own registry when not explicitly supplied.
    if circuit_breaker_registry is None and model_manager is not None:
        cb_getter = getattr(model_manager, "get_circuit_breaker_registry", None)
        if cb_getter is not None:
            try:
                circuit_breaker_registry = cb_getter()
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to fetch circuit-breaker registry from ModelManager")
    app.state.circuit_breaker_registry = circuit_breaker_registry
    app.state.objectstore = objectstore
    app.state.idempotency_cache = idempotency_cache
    # v0.6 engineering enhancement coordinators.
    app.state.backpressure_coordinator = backpressure_coordinator
    app.state.hot_reload_coordinator = hot_reload_coordinator
    app.state.degradation_coordinator = degradation_coordinator
    # v0.6 module wirings (observability / auth / persistence / dlq / security)
    app.state.auth_dependency = auth_dependency
    app.state.auth_bundle = auth_bundle
    app.state.persistence_manager = persistence_manager
    app.state.dlq = dlq
    app.state.metrics_registry = metrics_registry
    app.state.security_injection_detector = security_injection_detector
    # v0.6: PII 脱敏检测器（可选）。接线后 /invoke 会对 params 做脱敏，
    #       执行完成后再对结果做 unmask 恢复。
    app.state.pii_detector = pii_detector
    # Strong references to fire-and-forget background tasks so they are
    # not garbage-collected mid-execution (see CPython asyncio docs).
    app.state._bg_tasks: set = set()

    # --- CORS middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- v0.6 可观测性中间件 ---
    if metrics_registry is not None:
        from icore.observability.middleware import (
            MetricsMiddleware,
            RequestIDMiddleware,
        )

        app.add_middleware(RequestIDMiddleware)
        app.add_middleware(MetricsMiddleware, registry=metrics_registry)

    # --- Register exception handlers ---
    _register_exception_handlers(app)

    # --- Register endpoints ---
    _register_endpoints(app, api_settings)

    logger.info(
        "icore API app created (version=%s, host=%s, port=%d)",
        settings.version,
        api_settings.host,
        api_settings.port,
    )

    return app


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

def _register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers for common error types."""

    @app.exception_handler(BackpressureError)
    async def handle_backpressure(
        request: Request, exc: BackpressureError
    ) -> JSONResponse:
        """Handle BackpressureError - system overloaded, return 503."""
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": str(exc.retry_after)},
            content={
                "error": "Service Unavailable",
                "detail": str(exc),
                "retry_after": exc.retry_after,
                "task_id": getattr(request.state, "task_id", None),
            },
        )

    @app.exception_handler(ICoreError)
    async def handle_icore_error(
        request: Request, exc: ICoreError
    ) -> JSONResponse:
        """Handle any ICoreError subclass with a structured JSON response."""
        headers: dict[str, str] = {}
        if exc.retryable:
            # BackpressureError carries its own retry_after; use a sane
            # default for other retryable errors.
            retry_after = getattr(exc, "retry_after", 5)
            headers["Retry-After"] = str(retry_after)
        return JSONResponse(
            status_code=exc.http_status,
            headers=headers,
            content=exc.to_dict(getattr(request.state, "task_id", None)),
        )

    @app.exception_handler(KeyError)
    async def handle_key_error(
        request: Request, exc: KeyError
    ) -> JSONResponse:
        """Handle KeyError - typically workflow not found."""
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "detail": str(exc),
                "task_id": getattr(request.state, "task_id", None),
            },
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        """Handle ValueError - typically invalid parameters or duplicate task_id."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "detail": str(exc),
                "task_id": getattr(request.state, "task_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def handle_generic_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle any unhandled exception."""
        logger.error(
            "Unhandled exception in request: %s", exc, exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "code": "E-INTERNAL",
                "detail": "An unexpected error occurred",
                "task_id": getattr(request.state, "task_id", None),
            },
        )


# ---------------------------------------------------------------------------
# Endpoint registration
# ---------------------------------------------------------------------------

def _is_saturated_or_degraded(components: dict[str, Any]) -> bool:
    """v0.6 helper: inspect the backpressure / degradation snapshots.

    Returns True when any component is saturated or running on a
    fallback. Both snapshots are optional — absent sections don't
    affect the result.
    """
    bp = components.get("backpressure")
    if isinstance(bp, dict):
        if bp.get("saturated"):
            return True
        comps = bp.get("components") or {}
        for c in comps.values():
            if isinstance(c, dict) and c.get("saturated"):
                return True
    deg = components.get("degradation")
    if isinstance(deg, dict):
        if deg.get("any_degraded"):
            return True
        comps = deg.get("components") or {}
        for c in comps.values():
            if isinstance(c, dict) and c.get("state") not in (
                None, "primary",
            ):
                return True
    return False


def _parse_date_param(value: str | None) -> float | None:
    """将 ISO 日期字符串或 epoch 数字字符串转换为 Unix epoch float。

    支持 ISO 8601 格式（如 ``2024-01-15T10:30:00`` 或 ``2024-01-15``）
    以及纯数字字符串（直接作为 epoch 解析）。返回 None 当输入为 None。
    """
    if value is None:
        return None
    # 尝试直接作为 epoch 数字解析
    try:
        return float(value)
    except ValueError:
        pass
    # 尝试作为 ISO 日期解析
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(value)
        return dt.timestamp()
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid date format: '{value}'. "
            "Expected ISO 8601 (e.g. '2024-01-15T10:30:00') or epoch number."
        )


def _register_endpoints(app: FastAPI, api_settings: APISettings) -> None:
    """Register the API endpoints (业务端点 + 运维端点)."""

    # --- v0.6 运维端点: /metrics ---
    if getattr(app.state, "metrics_registry", None) is not None:

        @app.get(
            "/metrics",
            tags=["System"],
            summary="Prometheus metrics",
        )
        async def metrics():
            from icore.observability.prometheus_endpoint import (
                prometheus_response,
            )

            return prometheus_response(app.state.metrics_registry)

    # --- v0.6 运维端点: /history ---
    if getattr(app.state, "persistence_manager", None) is not None:
        # 当 auth_bundle 已接线时，/history 需要 ``history`` 权限
        # （admin / developer / viewer 都有此权限）。
        _history_deps: list = []
        _auth_bundle = getattr(app.state, "auth_bundle", None)
        if _auth_bundle is not None:
            from fastapi import Depends as _Depends

            _history_deps = [
                _Depends(
                    _auth_bundle.dependency(required_permission="history")
                )
            ]

        @app.get(
            "/history",
            tags=["System"],
            summary="Workflow execution history",
            dependencies=_history_deps,
        )
        async def history(
            workflow_name: str | None = None,
            status: str | None = None,
            since: str | None = None,
            until: str | None = None,
            limit: int = 100,
            offset: int = 0,
        ):
            # limit 上限保护
            settings = app.state.settings
            max_limit = settings.persistence.history_max_limit
            limit = min(limit, max_limit)
            # 将 ISO 日期字符串转换为 Unix epoch（后端接受 float）
            since_ts: float | None = _parse_date_param(since)
            until_ts: float | None = _parse_date_param(until)
            executions = await app.state.persistence_manager.get_history(
                workflow_name=workflow_name,
                status=status,
                since=since_ts,
                until=until_ts,
                limit=limit,
                offset=offset,
            )
            return {
                "executions": [
                    e.to_dict() if hasattr(e, "to_dict") else e
                    for e in executions
                ]
            }

    # --- v0.6 运维端点: /admin/dlq/* ---
    if getattr(app.state, "dlq", None) is not None:
        # 当 auth_bundle 已接线时，/admin/dlq/* 需要 ``admin`` 权限
        # （仅 admin 角色有此权限）。
        _admin_deps: list = []
        _auth_bundle = getattr(app.state, "auth_bundle", None)
        if _auth_bundle is not None:
            from fastapi import Depends as _Depends

            _admin_deps = [
                _Depends(
                    _auth_bundle.dependency(required_permission="admin")
                )
            ]

        @app.get(
            "/admin/dlq/list",
            tags=["Admin"],
            summary="List DLQ entries",
            dependencies=_admin_deps,
        )
        async def dlq_list(
            workflow_name: str | None = None, limit: int = 100
        ):
            entries = await app.state.dlq.list(
                workflow_name=workflow_name, limit=limit
            )
            return {
                "entries": [
                    e.to_dict() if hasattr(e, "to_dict") else str(e)
                    for e in entries
                ]
            }

        @app.post(
            "/admin/dlq/replay",
            tags=["Admin"],
            summary="Replay a DLQ entry",
            dependencies=_admin_deps,
        )
        async def dlq_replay(entry_id: str):
            try:
                result = await app.state.dlq.replay(entry_id)
                return {
                    "status": "replayed",
                    "entry_id": entry_id,
                    "result": str(result),
                }
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "entry_id": entry_id},
                )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="系统健康检查",
        description="检查 icore 系统是否正常运行。无需任何参数。",
    )
    async def health_check() -> HealthResponse:
        """
        Health check endpoint.

        v0.5: Probes each wired infrastructure component concurrently
        and aggregates the result. The overall ``status`` is
        ``"healthy"`` only when every probe returns healthy; otherwise
        ``"degraded"``. Components that are not wired (e.g. when no
        vectorstore is configured) are omitted from the response.

        v0.6: Also surfaces the cross-component backpressure snapshot
        (per-component saturation) and the graceful-degradation state
        matrix (which components are running on their fallback). The
        overall ``status`` is ``"degraded"`` when ANY component is
        unhealthy, saturated, or running on a fallback provider.

        Probes are intentionally **lightweight** — they verify that a
        component is wired and reachable in-process (e.g. model manager
        has models registered, db manager has connections registered,
        vector/graph stores report ``health_check()`` True). They do
        NOT perform expensive outbound calls (no real LLM API ping,
        no DB SELECT 1) so the endpoint stays fast and safe to call
        from liveness probes. Deep health checks are performed by the
        background ``ModelManager.start_health_monitor()`` task and
        surfaced via ``list_models()`` / model routing decisions.
        """
        import asyncio

        settings = app.state.settings

        async def _probe(name: str, obj: Any) -> tuple[str, dict[str, Any]]:
            try:
                # ModelManager: probe by counting registered models.
                # Calling ``health_check_all()`` here would perform real
                # LLM API pings which is too expensive for a liveness
                # endpoint and would falsely mark the system degraded
                # whenever the upstream provider is briefly unreachable.
                if name == "model":
                    count = len(obj)
                    return name, {
                        "status": "healthy" if count > 0 else "unhealthy",
                        "registered_models": count,
                    }
                # DBManager: probe by counting registered connections.
                if name == "db":
                    names = getattr(obj, "registered_names", []) or []
                    return name, {
                        "status": "healthy" if names else "unhealthy",
                        "registered_connections": len(names),
                    }
                # Vector / graph stores: lightweight no-arg health_check.
                check = getattr(obj, "health_check", None)
                if check is not None:
                    import inspect

                    sig = inspect.signature(check)
                    required = [
                        p
                        for p in sig.parameters.values()
                        if p.default is inspect.Parameter.empty
                        and p.kind
                        in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.POSITIONAL_ONLY,
                        )
                    ]
                    if not required:
                        ok = await check()
                        if isinstance(ok, dict):
                            return name, {
                                "status": "healthy" if ok.get("healthy", True) else "unhealthy",
                                **ok,
                            }
                        return name, {"status": "healthy" if ok else "unhealthy"}
                # No usable health-check method; assume healthy.
                return name, {"status": "healthy", "wired": True}
            except Exception as e:
                return name, {"status": "unhealthy", "error": str(e)}

        targets: list[tuple[str, Any]] = []
        if app.state.model_manager is not None:
            targets.append(("model", app.state.model_manager))
        if app.state.db_manager is not None:
            targets.append(("db", app.state.db_manager))
        if getattr(app.state, "vectorstore", None) is not None:
            targets.append(("vectorstore", app.state.vectorstore))
        if getattr(app.state, "graphstore", None) is not None:
            targets.append(("graphstore", app.state.graphstore))

        results = await asyncio.gather(
            *[_probe(n, o) for n, o in targets],
            return_exceptions=False,
        )
        components: dict[str, Any] = {n: r for n, r in results}

        # v0.6: append backpressure + degradation snapshots.
        bp_coord = getattr(app.state, "backpressure_coordinator", None)
        if bp_coord is not None:
            try:
                snap = await bp_coord.snapshot()
                components["backpressure"] = snap.to_dict()
            except Exception as e:  # pragma: no cover - defensive
                components["backpressure"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
        deg_coord = getattr(app.state, "degradation_coordinator", None)
        if deg_coord is not None:
            try:
                snap = deg_coord.snapshot()
                components["degradation"] = snap.to_dict()
            except Exception as e:  # pragma: no cover - defensive
                components["degradation"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        all_healthy = all(
            isinstance(c, dict) and c.get("status") == "healthy"
            for name, c in components.items()
            if name not in ("backpressure", "degradation")
        ) and not _is_saturated_or_degraded(components)
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version=settings.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=components,
        )

    # --- v0.6: /invoke 鉴权（可选）---
    _invoke_kwargs: dict[str, Any] = {
        "response_model": InvokeResponse,
        "tags": ["Workflow"],
        "summary": "工作流调用主接口",
        "description": (
            "调用指定的工作流。传入工作流名称、参数、任务ID、"
            "回调地址、模型ID等参数。"
        ),
    }
    _auth_dep = getattr(app.state, "auth_dependency", None)
    if _auth_dep is not None:
        from fastapi import Depends

        _invoke_kwargs["dependencies"] = [Depends(_auth_dep)]

    @app.post("/invoke", **_invoke_kwargs)
    async def invoke(
        request: InvokeRequest,
        raw_request: Request,
    ) -> Any:
        """
        Main workflow invocation endpoint.

        Request flow:
            1. Auto-generate task_id if not provided
            2. Look up workflow_name in WorkflowRegistry
            3. Backpressure check -> HTTP 503 if system overloaded
            4. Register task instance (dedup active task_id)
            5. Create TaskContext with model_id, callback_url, stream
            6. Inject model_manager and db_manager from app state
            7. Branch on execution mode (all wrapped in
               ConcurrencyController.acquire() when wired):
               - stream=True -> SSE streaming response
               - callback_url set -> async execution, return immediately
               - default -> sync execution, return when done
            8. Update instance state through lifecycle
               (PENDING -> RUNNING -> COMPLETED|FAILED)

        Args:
            request:      The InvokeRequest body.
            raw_request:  The raw FastAPI Request (for app state access).

        Returns:
            - InvokeResponse (JSON) for sync mode
            - StreamingResponse (SSE) for stream mode
        """
        # 1. Resolve task_id (auto-generate if not provided)
        task_id = request.task_id or str(uuid.uuid4())
        raw_request.state.task_id = task_id

        # 1a. v0.6: 注入检测（在幂等检查之前）
        # 检测范围：workflow_name + params 的所有文本值（不仅限于 params dict）。
        detector = getattr(app.state, "security_injection_detector", None)
        if detector is not None:
            import json as _json

            # 把 workflow_name + params 合并后检测，覆盖所有用户可控字段。
            _check_payload = {
                "workflow_name": request.workflow_name or "",
                "params": request.params or {},
            }
            text_to_check = _json.dumps(
                _check_payload, ensure_ascii=False
            )
            result = await detector.detect(text_to_check)
            if result.is_injection:
                from icore.exceptions import PromptInjectionError

                raise PromptInjectionError(
                    f"Prompt injection detected: {result.matched_patterns}"
                )

        # 1b. Idempotency short-circuit (v0.5): when the caller supplies
        #     an idempotency_key and a cache is wired, return the cached
        #     response without re-executing the workflow.
        idem_cache = getattr(app.state, "idempotency_cache", None)
        if idem_cache is not None and request.idempotency_key:
            cached = await idem_cache.get(request.idempotency_key)
            if cached is not None:
                logger.info(
                    "Idempotency hit for key=%s, returning cached result",
                    request.idempotency_key,
                )
                # Ensure the cached payload is shaped as InvokeResponse.
                if isinstance(cached, InvokeResponse):
                    return cached
                if isinstance(cached, dict):
                    return InvokeResponse(**cached)
                # Fallback: wrap as a success response.
                return InvokeResponse(
                    task_id=task_id,
                    status="success",
                    result={"cached": True, "value": cached},
                )

        # 2. Look up workflow in registry
        registry: WorkflowRegistry = app.state.workflow_registry
        try:
            workflow_cls = registry.get(request.workflow_name)
        except KeyError:
            raise KeyError(
                f"Workflow '{request.workflow_name}' is not registered. "
                f"Available: {registry.list_workflows()}"
            )

        # 3. Backpressure check: reject fast when the system is overloaded.
        #    When no controller is wired (test path), this is skipped.
        controller = app.state.concurrency_controller
        task_queue = app.state.task_queue
        instance_manager = app.state.instance_manager

        if controller is not None:
            if await controller.is_backpressure(task_queue):
                raise BackpressureError(
                    "System is under backpressure "
                    "(active=%d, max=%d)"
                    % (
                        controller.active_count,
                        controller.max_concurrent,
                    )
                )

        # v0.6: cross-component backpressure. When the coordinator
        # reports the system as saturated (memory budget exceeded or
        # any infrastructure component at capacity), reject with 503.
        bp_coord = getattr(app.state, "backpressure_coordinator", None)
        if bp_coord is not None:
            try:
                snap = await bp_coord.snapshot()
                if snap.saturated:
                    raise BackpressureError(
                        "Cross-component backpressure saturated "
                        "(memory_rss_mb=%.1f, components=%s)"
                        % (
                            snap.memory_rss_mb,
                            [
                                n for n, c in snap.components.items()
                                if c.saturated
                            ] or ["memory"],
                        ),
                        retry_after=10,
                    )
            except BackpressureError:
                raise
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Backpressure snapshot failed: %s", e)

        # v0.6: graceful degradation hard-fail. If a required component
        # has degraded past UNAVAILABLE (no fallback configured), reject
        # new invocations rather than letting workflows crash mid-flight.
        deg_coord = getattr(app.state, "degradation_coordinator", None)
        if deg_coord is not None:
            try:
                snap = deg_coord.snapshot()
                unavailable = [
                    n for n, c in snap.components.items()
                    if c.state.value == "unavailable"
                ]
                if unavailable:
                    raise BackpressureError(
                        "Required components unavailable: %s" % unavailable,
                        retry_after=30,
                    )
            except BackpressureError:
                raise
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Degradation snapshot failed: %s", e)

        # 4. Register task instance (deduplicates active task_id).
        #    TaskInstanceManager.create() raises ValueError if a task
        #    with this id is already running -> 422 via handler.
        if instance_manager is not None:
            await instance_manager.create(
                task_id=task_id,
                workflow_name=request.workflow_name,
                params=request.params,
            )

        # 5. Create workflow instance
        workflow = workflow_cls()

        # 6. Create TaskContext
        ctx = TaskContext(
            task_id=task_id,
            workflow_id=f"wf-{uuid.uuid4()}",
            model_id=request.model_id,
            callback_url=request.callback_url,
            stream=request.stream,
            metadata=request.metadata,
        )

        # 7. Inject managers from app state (v0.4 + v0.5 components).
        model_manager = app.state.model_manager
        db_manager = app.state.db_manager
        if model_manager is not None:
            ctx.set_model_manager(model_manager)
        if db_manager is not None:
            ctx.set_db_manager(db_manager)
        # v0.5 new injectables — only set when actually wired so that
        # ``has_*()`` returns False for workflows that don't need them.
        if getattr(app.state, "vectorstore", None) is not None:
            ctx.set_vectorstore(app.state.vectorstore)
        if getattr(app.state, "graphstore", None) is not None:
            ctx.set_graphstore(app.state.graphstore)
        if getattr(app.state, "media_processor", None) is not None:
            ctx.set_media_processor(app.state.media_processor)
        if getattr(app.state, "lock", None) is not None:
            ctx.set_lock(app.state.lock)
        cb_registry = getattr(app.state, "circuit_breaker_registry", None)
        if cb_registry is not None:
            ctx.set_circuit_breaker(cb_registry)
        obj_store = getattr(app.state, "objectstore", None)
        if obj_store is not None:
            ctx.set_objectstore(obj_store)

        callback_manager = app.state.callback_manager

        logger.info(
            "Invoke: workflow='%s', task_id='%s', stream=%s, "
            "callback=%s, model_id=%s",
            request.workflow_name,
            task_id,
            request.stream,
            request.callback_url is not None,
            request.model_id,
        )

        # 7b. v0.6: 接入 persistence + DLQ 到 workflow 的 executor
        persistence = getattr(app.state, "persistence_manager", None)
        dlq_manager = getattr(app.state, "dlq", None)
        if persistence is not None or dlq_manager is not None:
            from icore.engine.executor import WorkflowExecutor

            executor = WorkflowExecutor(
                persistence_manager=persistence,
                dlq=dlq_manager,
            )
            if hasattr(workflow, "_executor"):
                workflow._executor = executor

        # 7c. v0.6: persistence 落库（创建执行记录 + 标记 RUNNING）
        #     resume_from：断点续跑，从已有 checkpoint 恢复执行。
        execution_id: str | None = None
        resume_checkpoint: dict[str, Any] | None = None
        if request.resume_from is not None:
            # 断点续跑模式
            if persistence is None:
                raise ValueError(
                    "Cannot resume: persistence manager is not wired"
                )
            try:
                resumed = await persistence.resume_execution(
                    request.resume_from
                )
                execution_id = resumed.id
                resume_checkpoint = dict(resumed.checkpoint) if resumed.checkpoint else None
                await persistence.start_execution(execution_id)
                logger.info(
                    "Resuming execution %s with checkpoint: %s",
                    execution_id,
                    resume_checkpoint,
                )
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"Failed to resume execution '{request.resume_from}': {e}"
                )
        elif persistence is not None:
            try:
                exec_record = await persistence.create_execution(
                    task_id=task_id,
                    workflow_name=request.workflow_name,
                    params=request.params,
                )
                execution_id = exec_record.id
                await persistence.start_execution(execution_id)
            except Exception as e:
                logger.warning("persistence create_execution failed: %s", e)

        # 8. Branch on execution mode. Each branch wraps execution in
        #    the concurrency controller's acquire() context (when wired)
        #    and drives the instance manager's state transitions.
        if request.stream:
            # --- SSE streaming mode ---
            # Backpressure was already checked above; the stream itself
            # runs unbounded (it is the caller's read loop that paces it).
            # Instance state transitions are driven by the stream handler.
            handler = SSEStreamHandler()
            return handler.create_response(
                workflow=workflow,
                ctx=ctx,
                params=request.params,
                callback_manager=callback_manager,
                instance_manager=instance_manager,
            )

        if request.callback_url:
            # --- Async mode: execute in background, return immediately ---
            _spawn_background_task(
                _execute_with_callback(
                    app=app,
                    workflow=workflow,
                    ctx=ctx,
                    params=request.params,
                    callback_url=request.callback_url,
                    callback_manager=callback_manager,
                    controller=controller,
                    instance_manager=instance_manager,
                    execution_id=execution_id,
                    checkpoint=resume_checkpoint,
                ),
                app,
            )
            return InvokeResponse(
                task_id=task_id,
                status="running",
                result=None,
                error=None,
            )

        # --- Sync mode: execute and wait for result ---
        try:
            result = await _run_with_governance(
                workflow=workflow,
                ctx=ctx,
                params=request.params,
                workflow_name=request.workflow_name,
                controller=controller,
                instance_manager=instance_manager,
                execution_id=execution_id,
                checkpoint=resume_checkpoint,
            )
            response = InvokeResponse(
                task_id=task_id,
                status=result.status,
                result=result.data if result.is_success else None,
                error=result.error,
            )
            # v0.6: persistence 落库（完成/失败）
            if persistence is not None and execution_id:
                try:
                    if result.is_success:
                        await persistence.complete_execution(
                            execution_id, result.data
                        )
                    else:
                        await persistence.fail_execution(
                            execution_id, result.error
                        )
                except Exception as e:
                    logger.warning(
                        "persistence complete/fail failed: %s", e
                    )
            # v0.5: cache successful (and explicit failure) responses
            # under the idempotency key so retries don't re-execute.
            if idem_cache is not None and request.idempotency_key:
                try:
                    await idem_cache.set(
                        request.idempotency_key,
                        response.model_dump(),
                    )
                except Exception as cache_err:  # pragma: no cover
                    logger.warning(
                        "Failed to cache idempotency result for key=%s: %s",
                        request.idempotency_key,
                        cache_err,
                    )
            return response
        except Exception as e:
            logger.error(
                "Workflow execution failed: %s", e, exc_info=True
            )
            # Mark the instance as failed if we tracked it
            if instance_manager is not None:
                await instance_manager.update_state(
                    task_id, TaskState.FAILED,
                    error=f"{type(e).__name__}: {e}",
                )
            return InvokeResponse(
                task_id=task_id,
                status="error",
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------------------------------------------------------
# Helper: async background execution with callback
# ---------------------------------------------------------------------------

async def _run_with_governance(
    workflow: Any,
    ctx: TaskContext,
    params: dict[str, Any],
    workflow_name: str,
    controller: Any,
    instance_manager: Any,
    execution_id: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> Any:
    """
    Execute a workflow under concurrency governance and instance tracking.

    When a ``ConcurrencyController`` is provided, the execution is wrapped
    in ``controller.acquire(workflow_name)`` so the global + per-workflow
    semaphores are honoured. When an ``TaskInstanceManager`` is provided,
    the instance state is advanced PENDING -> RUNNING -> COMPLETED|FAILED.

    Both arguments are optional so this helper also works in the
    infrastructure-free test path.
    """
    # Enter the concurrency slot (acquires global + per-workflow sems)
    if controller is not None:
        slot = controller.acquire(workflow_name)
        await slot.__aenter__()
    else:
        slot = None

    if instance_manager is not None:
        await instance_manager.update_state(
            ctx.task_id, TaskState.RUNNING
        )

    try:
        result = await workflow.execute(
            ctx, params,
            execution_id=execution_id,
            workflow_name=workflow_name,
            checkpoint=checkpoint,
        )
        if instance_manager is not None:
            new_state = (
                TaskState.COMPLETED
                if result.is_success
                else TaskState.FAILED
            )
            await instance_manager.update_state(
                ctx.task_id, new_state, result=result,
            )
        return result
    except Exception:
        if instance_manager is not None:
            await instance_manager.update_state(
                ctx.task_id, TaskState.FAILED,
            )
        raise
    finally:
        if slot is not None:
            await slot.__aexit__(None, None, None)


async def _execute_with_callback(
    app: FastAPI,
    workflow: Any,
    ctx: TaskContext,
    params: dict[str, Any],
    callback_url: str,
    callback_manager: CallbackManager,
    controller: Any = None,
    instance_manager: Any = None,
    execution_id: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> None:
    """
    Execute workflow in background and deliver result to callback URL.

    This runs as a background asyncio task when callback_url is set.
    The workflow result is POSTed to the callback URL upon completion.

    The execution is wrapped in ``_run_with_governance`` so the
    concurrency controller and instance manager are honoured even in
    async/callback mode.

    Args:
        app:              The FastAPI application (for state access).
        workflow:         The BaseWorkflow instance.
        ctx:              TaskContext for this execution.
        params:           Workflow input parameters.
        callback_url:     URL to deliver results to.
        callback_manager: CallbackManager instance.
        controller:       Optional ConcurrencyController.
        instance_manager: Optional TaskInstanceManager.
        checkpoint:       Optional resume checkpoint (v0.6).
    """
    wf_name = getattr(workflow, "name", ctx.workflow_id)
    try:
        result = await _run_with_governance(
            workflow=workflow,
            ctx=ctx,
            params=params,
            workflow_name=wf_name,
            controller=controller,
            instance_manager=instance_manager,
            execution_id=execution_id,
            checkpoint=checkpoint,
        )
        payload: dict[str, Any] = {
            "task_id": ctx.task_id,
            "status": result.status,
            "result": result.data,
            "error": result.error,
        }
    except Exception as e:
        logger.error(
            "Background workflow execution failed: %s", e, exc_info=True
        )
        payload = {
            "task_id": ctx.task_id,
            "status": "error",
            "result": None,
            "error": f"{type(e).__name__}: {e}",
        }

    # Deliver result to callback URL. We await (rather than fire-and-forget)
    # so the background task stays alive until delivery completes/retries
    # finish, and so no nested untracked task is created.
    await callback_manager.deliver_and_wait(callback_url, payload)


def _spawn_background_task(coro: Any, app: FastAPI) -> None:
    """
    Schedule a coroutine as a tracked background task.

    Keeps a strong reference on ``app.state._bg_tasks`` so the task is not
    garbage-collected mid-execution, and removes it once done. Exceptions
    are logged (never re-raised) since this is fire-and-forget.
    """
    import asyncio

    task = asyncio.create_task(coro)
    app.state._bg_tasks.add(task)

    def _on_done(t: "asyncio.Task[Any]") -> None:
        app.state._bg_tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error(
                "Background task failed: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=exc,
            )

    task.add_done_callback(_on_done)


# ---------------------------------------------------------------------------
# Module-level application instance
# ---------------------------------------------------------------------------
#
# Enables ``uvicorn icore.api.main:app`` to run a fully-wired server
# out-of-the-box: the production bootstrap loads ``config/*.yaml``,
# builds the Model/DB managers, and auto-registers example workflows.
#
# The pure factory ``create_app()`` above remains infrastructure-free
# for tests and custom wiring. If the bootstrap fails (e.g. missing
# config), we fall back to a bare app so the process still starts and
# the error is visible in logs rather than crashing on import.

def _build_default_app() -> FastAPI:
    try:
        from icore.bootstrap import create_production_app

        return create_production_app()
    except Exception as e:  # pragma: no cover - bootstrap robustness
        logger.error(
            "Production bootstrap failed (%s: %s); falling back to bare app. "
            "Workflow/model/db wiring will be unavailable until fixed.",
            type(e).__name__,
            e,
            exc_info=True,
        )
        return create_app()


app: FastAPI = _build_default_app()
