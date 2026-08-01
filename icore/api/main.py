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

logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    model_manager: Any = None,
    db_manager: Any = None,
    workflow_registry: Optional[WorkflowRegistry] = None,
    concurrency_controller: Any = None,
    task_queue: Any = None,
    instance_manager: Any = None,
) -> FastAPI:
    """
    Create and configure the icore FastAPI application.

    This factory function sets up:
        - CORS middleware
        - Exception handlers (KeyError, ValueError, generic Exception)
        - Application state (model_manager, db_manager, callback_manager,
          concurrency_controller, task_queue, instance_manager)
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
                "error": "Internal Server Error",
                "detail": str(exc),
                "task_id": getattr(request.state, "task_id", None),
            },
        )


# ---------------------------------------------------------------------------
# Endpoint registration
# ---------------------------------------------------------------------------

def _register_endpoints(app: FastAPI, api_settings: APISettings) -> None:
    """Register the two API endpoints."""

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

        Returns system status, version, and current timestamp.
        No parameters required.
        """
        settings = app.state.settings
        return HealthResponse(
            status="healthy",
            version=settings.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.post(
        "/invoke",
        response_model=InvokeResponse,
        tags=["Workflow"],
        summary="工作流调用主接口",
        description=(
            "调用指定的工作流。传入工作流名称、参数、任务ID、"
            "回调地址、模型ID等参数。"
        ),
    )
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

        # 7. Inject managers from app state
        model_manager = app.state.model_manager
        db_manager = app.state.db_manager
        if model_manager is not None:
            ctx.set_model_manager(model_manager)
        if db_manager is not None:
            ctx.set_db_manager(db_manager)

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
            )
            return InvokeResponse(
                task_id=task_id,
                status=result.status,
                result=result.data if result.is_success else None,
                error=result.error,
            )
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
        result = await workflow.execute(ctx, params)
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
