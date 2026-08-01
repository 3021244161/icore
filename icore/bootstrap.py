"""
icore.bootstrap - Production application bootstrap.

Wires the icore FastAPI application to its runtime infrastructure by
loading YAML configuration, building the ``ModelManager`` and
``DBManager``, auto-registering workflow modules, and installing
lifecycle hooks (health monitor start / graceful shutdown).

This is the layer that makes ``uvicorn icore.api.main:app`` (and
``python -m icore``) work out-of-the-box: the pure factory
``icore.api.main.create_app()`` is intentionally infrastructure-free
for testability; this module supplies the production wiring.

Configuration sources (precedence, highest last):
    1. Built-in defaults (``icore.config.Settings``)
    2. ``.env`` file
    3. Environment variables (``ICORE_*`` prefix)
    4. YAML files in ``settings.config_dir`` (``models.yaml``,
       ``databases.yaml``) — values may reference env vars via
       ``${VAR}`` placeholders, which are interpolated at load time.

The YAML ``models:`` section uses the dict key as ``model_id``::

    models:
      gpt-4o:            # <- this key becomes ModelConfig.model_id
        model_name: gpt-4o
        api_base: https://api.openai.com/v1
        api_key: ${OPENAI_API_KEY}

Similarly, ``databases.connections:`` uses the dict key as the
connection name passed to ``DBManager.register(name, config)``.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from icore.config import (
    DatabaseConnectionConfig,
    ModelConfig,
    RoutingRule,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)

#: Matches ``${VAR_NAME}`` placeholders in YAML string values.
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


# ---------------------------------------------------------------------------
# YAML loading with ${VAR} interpolation
# ---------------------------------------------------------------------------

def _interpolate(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with env values."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), ""), value
        )
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file with ``${VAR}`` env interpolation.

    Returns an empty dict if the file does not exist or PyYAML is not
    installed. Non-dict YAML documents are also returned as ``{}``.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("Config file not found, skipping: %s", p)
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        logger.warning("PyYAML not installed; cannot load %s", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return _interpolate(data)


# ---------------------------------------------------------------------------
# Manager construction
# ---------------------------------------------------------------------------

def build_model_manager(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build a ``ModelManager`` from ``config/models.yaml``.

    The YAML dict key for each model is injected as ``model_id`` (the
    field is required on ``ModelConfig`` but omitted from the YAML body
    by convention). ``${VAR}`` placeholders are interpolated.

    Returns an empty manager if the config file is absent.
    """
    from icore.models.manager import ModelManager

    settings = settings or get_settings()
    cfg_path = Path(settings.config_dir) / "models.yaml"
    data = load_yaml_config(cfg_path)

    models: dict[str, ModelConfig] = {}
    for model_id, raw in (data.get("models") or {}).items():
        if not isinstance(raw, dict):
            continue
        body = dict(raw)
        body["model_id"] = model_id  # dict key is the canonical id
        try:
            models[model_id] = ModelConfig(**body)
        except Exception as e:
            logger.warning(
                "Skipping model '%s' (invalid config): %s", model_id, e
            )

    routing_rules: list[RoutingRule] = []
    for raw_rule in (data.get("routing_rules") or []):
        if not isinstance(raw_rule, dict):
            continue
        try:
            routing_rules.append(RoutingRule(**raw_rule))
        except Exception as e:
            logger.warning("Skipping routing rule %r: %s", raw_rule, e)

    # Build a ModelSettings populated from YAML (env vars still override
    # via BaseSettings, but explicit YAML values take precedence).
    from icore.config import ModelSettings

    model_settings = ModelSettings(
        default_model_id=data.get("default_model_id"),
        enable_auto_routing=data.get("enable_auto_routing", True),
        models=models,
        routing_rules=routing_rules,
        health_check_interval=data.get("health_check_interval", 60),
    )
    mgr = ModelManager(model_settings)
    logger.info(
        "Built ModelManager: %d model(s) registered, auto_routing=%s",
        len(models),
        model_settings.enable_auto_routing,
    )
    return mgr


def build_db_manager(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build a ``DBManager`` from ``config/databases.yaml``.

    The YAML dict key for each connection is used as the connection
    name (passed to ``DBManager.register``). ``${VAR}`` placeholders
    are interpolated.

    Returns an empty manager if the config file is absent.
    """
    from icore.db.manager import DBManager

    settings = settings or get_settings()
    cfg_path = Path(settings.config_dir) / "databases.yaml"
    data = load_yaml_config(cfg_path)

    mgr = DBManager()
    for name, raw in (data.get("connections") or {}).items():
        if not isinstance(raw, dict):
            continue
        try:
            mgr.register(name, DatabaseConnectionConfig(**raw))
        except Exception as e:
            logger.warning(
                "Skipping database connection '%s' (invalid config): %s",
                name,
                e,
            )
    logger.info("Built DBManager: %d connection(s) registered", len(mgr.registered_names))
    return mgr


# ---------------------------------------------------------------------------
# Workflow auto-registration
# ---------------------------------------------------------------------------

def autoregister_workflows(
    settings: Optional[Settings] = None,
) -> list[str]:
    """
    Import configured workflow modules so their ``@register_task`` /
    ``@register_workflow`` decorators execute.

    Returns the list of module names that were imported successfully.
    Controlled by ``settings.autoregister_workflows`` (default True) and
    ``settings.workflow_modules`` (default ``["icore.workflows.examples"]``).
    """
    settings = settings or get_settings()
    imported: list[str] = []
    if not settings.autoregister_workflows:
        return imported

    for mod_name in settings.workflow_modules:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
            logger.debug("Auto-registered workflow module: %s", mod_name)
        except Exception as e:
            logger.warning(
                "Failed to import workflow module '%s': %s", mod_name, e
            )
    return imported


# ---------------------------------------------------------------------------
# Production app factory
# ---------------------------------------------------------------------------

def build_concurrency_controller(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the ``ConcurrencyController`` from ``settings.concurrency``.

    The controller enforces global / per-workflow semaphores and exposes
    a backpressure signal consumed by the API layer (HTTP 503).

    Returns:
        A ``ConcurrencyController`` instance.
    """
    from icore.engine.concurrency_control import ConcurrencyController

    settings = settings or get_settings()
    cc = settings.concurrency
    controller = ConcurrencyController(
        max_concurrent_tasks=cc.max_concurrent_tasks,
        max_concurrent_per_workflow=cc.max_concurrent_per_workflow,
        backpressure_threshold=cc.backpressure_threshold,
    )
    logger.info(
        "Built ConcurrencyController "
        "(max_global=%d, max_per_workflow=%d, backpressure=%d)",
        controller.max_concurrent,
        controller.max_per_workflow,
        controller.backpressure_threshold,
    )
    return controller


def build_task_queue(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the ``TaskQueue`` from ``settings.concurrency``.

    Supports the ``memory`` (default) and ``redis`` backends. The Redis
    driver is imported lazily inside ``TaskQueue`` so this never fails
    at import time even when ``redis`` is not installed.
    """
    from icore.engine.task_queue import TaskQueue

    settings = settings or get_settings()
    cc = settings.concurrency
    queue = TaskQueue(
        backend=cc.task_queue_backend,
        redis_url=cc.redis_url,
    )
    logger.info("Built TaskQueue (backend=%s)", cc.task_queue_backend)
    return queue


def build_instance_manager(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the ``TaskInstanceManager`` for task lifecycle tracking.

    The manager is the system's source of truth for "what is currently
    running" and powers deduplication, status queries, and cleanup of
    terminal instances.
    """
    from icore.engine.instance_manager import TaskInstanceManager

    settings = settings or get_settings()
    # TTL is tied to the configured task timeout so terminal instances
    # linger long enough for status polling but don't leak forever.
    ttl = max(settings.concurrency.task_timeout * 6, 3600)
    mgr = TaskInstanceManager(ttl=ttl)
    logger.info("Built TaskInstanceManager (ttl=%ds)", ttl)
    return mgr


def create_production_app(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the fully-wired FastAPI application for production.

    Steps:
        1. Auto-register workflow modules (populates Task/Workflow registries)
        2. Build ModelManager from ``config/models.yaml``
        3. Build DBManager from ``config/databases.yaml``
        4. Build ConcurrencyController / TaskQueue / TaskInstanceManager
           from ``settings.concurrency``
        5. Call ``icore.api.main.create_app()`` with managers injected
        6. Install startup/shutdown lifecycle hooks:
           - startup: start model health monitor, start task queue,
                     start periodic instance cleanup
           - shutdown: stop queue, close model adapters + DB pools,
                      cancel pending background tasks

    Args:
        settings: Optional Settings override. Defaults to ``get_settings()``.

    Returns:
        A configured FastAPI application with infrastructure attached.
    """
    from icore.api.main import create_app

    settings = settings or get_settings()

    # 1. Register workflows (must happen before the app serves requests)
    autoregister_workflows(settings)

    # 2-4. Build infrastructure managers from YAML + settings
    model_manager = build_model_manager(settings)
    db_manager = build_db_manager(settings)
    concurrency_controller = build_concurrency_controller(settings)
    task_queue = build_task_queue(settings)
    instance_manager = build_instance_manager(settings)

    # 5. Create the FastAPI app with managers injected
    app = create_app(
        settings=settings,
        model_manager=model_manager,
        db_manager=db_manager,
        concurrency_controller=concurrency_controller,
        task_queue=task_queue,
        instance_manager=instance_manager,
    )

    # 6. Lifecycle hooks
    @app.on_event("startup")
    async def _on_startup() -> None:
        try:
            await task_queue.start()
        except Exception as e:
            logger.warning("Failed to start task queue: %s", e)
        try:
            if model_manager is not None and len(model_manager) > 0:
                await model_manager.start_health_monitor()
        except Exception as e:
            logger.warning("Failed to start model health monitor: %s", e)

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        # Cancel any pending background tasks
        bg_tasks = getattr(app.state, "_bg_tasks", set())
        for t in list(bg_tasks):
            if not t.done():
                t.cancel()
        # Stop the task queue
        try:
            await task_queue.stop()
        except Exception as e:
            logger.warning("Error stopping task queue: %s", e)
        # Graceful close
        if model_manager is not None:
            try:
                await model_manager.close_all()
            except Exception as e:
                logger.warning("Error closing model manager: %s", e)
        if db_manager is not None:
            try:
                await db_manager.close_all()
            except Exception as e:
                logger.warning("Error closing db manager: %s", e)

    return app
