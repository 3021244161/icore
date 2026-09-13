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
    2. ``.env`` file (cwd-relative, read-only)
    3. Environment variables (``ICORE_*`` prefix)
    4. YAML files in ``settings.config_dir`` (``models.yaml``,
       ``databases.yaml``) — values may reference env vars via
       ``${VAR}`` placeholders, which are interpolated at load time
       against ``os.environ`` plus the read-only ``.env`` view
       (ICORE-ISSUE-004: the ``.env`` file is **never** written into
       ``os.environ``, so importing ``icore.api`` has no environment
       side effects).

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

def _interpolate(
    value: Any,
    env: Optional[dict[str, str]] = None,
) -> Any:
    """Recursively replace ``${VAR}`` placeholders with env values.

    ``env`` defaults to ``os.environ``. ``load_yaml_config`` passes a
    merged read-only view (``.env`` file + process environment) so
    placeholders resolve without ever writing into ``os.environ``
    (ICORE-ISSUE-004).
    """
    if isinstance(value, str):
        source = os.environ if env is None else env
        return _ENV_PATTERN.sub(
            lambda m: source.get(m.group(1), ""), value
        )
    if isinstance(value, dict):
        return {k: _interpolate(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, env) for v in value]
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file with ``${VAR}`` env interpolation.

    Returns an empty dict if the file does not exist or PyYAML is not
    installed. Non-dict YAML documents are also returned as ``{}``.

    ``${VAR}`` placeholders resolve against the union of a local
    ``.env`` file and the process environment (env vars take
    precedence), matching the previous ``load_dotenv(override=False)``
    semantics.

    v0.6.x (ICORE-ISSUE-004): the ``.env`` file is resolved from the
    **current working directory** and read **read-only** via
    ``dotenv_values()`` — it is never written into ``os.environ``.
    Importing ``icore.api.*`` therefore no longer has environment side
    effects. (Previously ``load_dotenv()`` silently exported every
    missing key from the located ``.env`` into the process environment,
    breaking consumer config isolation and test reproducibility; and
    its ``find_dotenv()`` stack-walk resolved a different file
    depending on tooling such as ``coverage``.)
    """
    # ICORE-ISSUE-004: .env 只读参与 ${VAR} 插值，绝不写入 os.environ。
    # 旧实现在 import 链上（api/main.py 模块级 app 构建）调用 load_dotenv()
    # 把 .env 的所有缺失键静默注入进程环境；且 find_dotenv() 按调用栈
    # 定位文件、结果不确定。现改为 cwd 相对路径 + dotenv_values 只读合并。
    file_env: dict[str, str] = {}
    try:
        from dotenv import dotenv_values  # type: ignore
    except ImportError:
        pass
    else:
        try:
            file_env = {
                k: v
                for k, v in dotenv_values(".env").items()
                if v is not None
            }
        except Exception:  # noqa: BLE001 — .env 读取失败不阻断配置装载
            file_env = {}

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
    # 进程环境变量优先于 .env 文件值（等价旧 load_dotenv 的 override=False）。
    return _interpolate(data, {**file_env, **dict(os.environ)})


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


# ---------------------------------------------------------------------------
# v0.5: New infrastructure builders
# ---------------------------------------------------------------------------

def build_vectorstore(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the ``BaseVectorStore`` from ``config/vectorstore.yaml``.

    Supports ``backend: milvus`` (lazy-loaded) and falls back to an
    in-memory store when the config file is absent (useful for tests
    / local dev). The Milvus driver is imported lazily by the adapter
    so this never fails at import time.
    """
    from icore.vectorstore import InMemoryVectorStore, MilvusAdapter

    settings = settings or get_settings()
    cfg_path = Path(settings.config_dir) / "vectorstore.yaml"
    data = load_yaml_config(cfg_path)
    backend = (data.get("backend") or "memory").lower()

    if backend == "milvus":
        milvus_cfg = data.get("milvus") or {}
        adapter = MilvusAdapter(
            host=milvus_cfg.get("host", "localhost"),
            port=int(milvus_cfg.get("port", 19530)),
            db_name=milvus_cfg.get("db_name", "icore"),
            default_index_type=milvus_cfg.get(
                "default_index_type", "IVF_FLAT"
            ),
            default_metric=milvus_cfg.get("default_metric", "COSINE"),
            pool_size=int(milvus_cfg.get("pool_size", 10)),
        )
        logger.info(
            "Built MilvusAdapter (host=%s, port=%s, db=%s)",
            milvus_cfg.get("host", "localhost"),
            milvus_cfg.get("port", 19530),
            milvus_cfg.get("db_name", "icore"),
        )
        return adapter

    # Default: in-memory (offline / development).
    logger.info("Built InMemoryVectorStore (backend=%s)", backend)
    return InMemoryVectorStore()


def build_graphstore(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the ``BaseGraphStore`` from ``config/graphstore.yaml``.

    Supports ``backend: neo4j`` (lazy-loaded) and falls back to an
    in-memory store when the config file is absent.
    """
    from icore.graphstore import InMemoryGraphStore, Neo4jAdapter

    settings = settings or get_settings()
    cfg_path = Path(settings.config_dir) / "graphstore.yaml"
    data = load_yaml_config(cfg_path)
    backend = (data.get("backend") or "memory").lower()

    if backend == "neo4j":
        neo4j_cfg = data.get("neo4j") or {}
        adapter = Neo4jAdapter(
            uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
            username=neo4j_cfg.get("username", "neo4j"),
            password=neo4j_cfg.get("password", "neo4j"),
            database=neo4j_cfg.get("database", "neo4j"),
            max_connection_pool_size=int(
                neo4j_cfg.get("max_connection_pool_size", 10)
            ),
        )
        logger.info(
            "Built Neo4jAdapter (uri=%s, database=%s)",
            neo4j_cfg.get("uri", "bolt://localhost:7687"),
            neo4j_cfg.get("database", "neo4j"),
        )
        return adapter

    logger.info("Built InMemoryGraphStore (backend=%s)", backend)
    return InMemoryGraphStore()


def build_media_processor(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the default ``MediaProcessorRegistry``.

    Heavy backends (Pillow / ffmpeg / pydub) are imported lazily by
    each processor so this factory is always cheap. Configuration is
    currently hard-coded to the defaults; a ``config/media.yaml`` could
    be added later for fine-grained control.
    """
    from icore.media import create_default_media_registry

    settings = settings or get_settings()
    registry = create_default_media_registry()
    logger.info(
        "Built MediaProcessorRegistry (types=%s)",
        [t.value for t in registry.list_types()],
    )
    return registry


def build_lock(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the ``BaseDistributedLock`` from ``settings.concurrency``.

    Uses ``RedisLock`` when ``task_queue_backend == "redis"`` and a
    Redis URL is configured; otherwise falls back to ``MemoryLock``
    for single-process deployments.
    """
    from icore.engine.lock import MemoryLock, RedisLock

    settings = settings or get_settings()
    cc = settings.concurrency
    if cc.task_queue_backend == "redis" and cc.redis_url:
        try:
            import redis.asyncio as aioredis  # type: ignore

            client = aioredis.from_url(cc.redis_url)
            lock = RedisLock(client)
            logger.info("Built RedisLock (redis_url=%s)", cc.redis_url)
            return lock
        except ImportError:
            logger.warning(
                "redis driver not installed; falling back to MemoryLock"
            )
        except Exception as e:  # pragma: no cover - network path
            logger.warning(
                "Failed to build RedisLock (%s); falling back to MemoryLock",
                e,
            )

    logger.info("Built MemoryLock (single-process)")
    return MemoryLock()


def build_idempotency_cache(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the ``BaseIdempotencyCache`` from ``settings.concurrency``.

    Memory backend is the default; a Redis-backed implementation can
    be added later by setting ``task_queue_backend="redis"`` once the
    Redis variant is implemented.
    """
    from icore.api.idempotency import create_idempotency_cache

    settings = settings or get_settings()
    cc = settings.concurrency
    cache = create_idempotency_cache(
        backend=cc.task_queue_backend,
        redis_url=cc.redis_url,
    )
    logger.info("Built IdempotencyCache (backend=%s)", cc.task_queue_backend)
    return cache


def build_objectstore(
    settings: Optional[Settings] = None,
) -> "Any":
    """
    Build the ``BaseObjectStore`` from ``config/objectstore.yaml``.

    When ``backend == "minio"`` the ``MinIOAdapter`` is used (minio SDK
    is imported lazily). Otherwise falls back to ``InMemoryObjectStore``.
    """
    from icore.objectstore import InMemoryObjectStore, MinIOAdapter

    settings = settings or get_settings()
    cfg_path = Path(settings.config_dir) / "objectstore.yaml"
    data = load_yaml_config(cfg_path)

    backend = (data.get("backend") or "memory").lower()
    if backend == "minio":
        minio_cfg = data.get("minio", {})
        store = MinIOAdapter(
            endpoint=minio_cfg.get("endpoint", "localhost:9000"),
            access_key=minio_cfg.get("access_key", "minioadmin"),
            secret_key=minio_cfg.get("secret_key", "minioadmin"),
            secure=bool(minio_cfg.get("secure", False)),
            region=minio_cfg.get("region", "us-east-1"),
            default_bucket=minio_cfg.get("default_bucket", "icore"),
        )
    else:
        store = InMemoryObjectStore()
    logger.info("Built ObjectStore (backend=%s)", backend)
    return store


# ---------------------------------------------------------------------------
# v0.6: Engineering enhancement builders
# ---------------------------------------------------------------------------

def build_backpressure_coordinator(
    settings: Optional[Settings] = None,
) -> "Any":
    """Build the ``BackpressureCoordinator`` (v0.6 §3.2).

    Reads the v0.6 ``settings.backpressure`` block and registers a
    budget for every LLM / vectorstore / graphstore / db component
    detected in the current app. Per-model budgets are registered
    lazily by ``ModelManager`` itself; here we register the static
    infrastructure budgets (vectorstore / graphstore / db).
    """
    from icore.engine.backpressure import (
        BackpressureCoordinator,
        ComponentBudget,
    )

    settings = settings or get_settings()
    bp = settings.backpressure
    coord = BackpressureCoordinator(
        memory_budget_mb=bp.memory_budget_mb,
        check_interval=bp.check_interval,
    )
    if bp.enabled:
        coord.register(
            ComponentBudget(
                name="vectorstore:default",
                max_concurrent=bp.vector_max_concurrent,
            )
        )
        coord.register(
            ComponentBudget(
                name="graphstore:default",
                max_concurrent=bp.graph_max_concurrent,
            )
        )
        coord.register(
            ComponentBudget(
                name="db:default",
                max_concurrent=bp.db_max_concurrent,
            )
        )
    logger.info(
        "Built BackpressureCoordinator (memory_budget_mb=%.1f, components=%s)",
        bp.memory_budget_mb,
        coord.list_components(),
    )
    return coord


def build_hot_reload_coordinator(
    settings: Optional[Settings] = None,
) -> "Any":
    """Build the ``HotReloadCoordinator`` (v0.6 §3.3.1).

    The coordinator is constructed but **not** started here. The
    caller (``create_production_app``) wires reload callbacks and
    starts/stops the observer during FastAPI lifecycle.
    """
    from icore.engine.hot_reload import HotReloadCoordinator

    settings = settings or get_settings()
    hr = settings.hot_reload
    if not hr.enabled:
        logger.info("Hot-reload disabled by settings")
        return None
    coord = HotReloadCoordinator(
        config_dir=settings.config_dir,
        poll_interval=hr.poll_interval,
        use_watchdog=hr.use_watchdog,
    )
    logger.info(
        "Built HotReloadCoordinator (config_dir=%s, watchdog=%s)",
        settings.config_dir,
        hr.use_watchdog,
    )
    return coord


def build_degradation_coordinator(
    settings: Optional[Settings] = None,
) -> "Any":
    """Build the ``GracefulDegradationCoordinator`` (v0.6 §3.3.2).

    Like the hot-reload coordinator, this is built but not yet
    populated with provider bindings. The caller wires primary +
    fallback pairs after constructing the other managers.
    """
    from icore.engine.graceful_degradation import (
        GracefulDegradationCoordinator,
    )

    settings = settings or get_settings()
    if not settings.degradation.enabled:
        logger.info("Graceful degradation disabled by settings")
        return None
    coord = GracefulDegradationCoordinator()
    logger.info("Built GracefulDegradationCoordinator")
    return coord


def _wire_degradation_providers(
    coord: Any,
    *,
    model_manager: Any,
    db_manager: Any,
    vectorstore: Any,
    graphstore: Any,
    lock: Any,
    objectstore: Any,
    settings: Settings,
) -> None:
    """Wire primary + fallback provider pairs into the coordinator.

    Each infrastructure component that has a built in-memory fallback
    is registered. Components without a fallback (e.g. ModelManager)
    are skipped — the existing circuit breaker + NoAvailableModelError
    flow already covers them.
    """
    if coord is None:
        return
    failure_threshold = settings.degradation.failure_threshold
    recovery_interval = settings.degradation.recovery_interval

    # Vectorstore: Milvus → InMemoryVectorStore.
    if vectorstore is not None:
        try:
            from icore.vectorstore import InMemoryVectorStore

            fallback_vs = InMemoryVectorStore()
            primary_health = getattr(vectorstore, "health_check", None)

            async def _vs_health() -> bool:
                if primary_health is None:
                    return True
                try:
                    return bool(await primary_health())
                except Exception:
                    return False

            coord.register(
                name="vectorstore",
                primary=vectorstore,
                fallback=fallback_vs,
                health_check=_vs_health,
                failure_threshold=failure_threshold,
                recovery_interval=recovery_interval,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to wire vectorstore degradation: %s", e)

    # Graphstore: Neo4j → InMemoryGraphStore.
    if graphstore is not None:
        try:
            from icore.graphstore import InMemoryGraphStore

            fallback_gs = InMemoryGraphStore()
            primary_health = getattr(graphstore, "health_check", None)

            async def _gs_health() -> bool:
                if primary_health is None:
                    return True
                try:
                    return bool(await primary_health())
                except Exception:
                    return False

            coord.register(
                name="graphstore",
                primary=graphstore,
                fallback=fallback_gs,
                health_check=_gs_health,
                failure_threshold=failure_threshold,
                recovery_interval=recovery_interval,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to wire graphstore degradation: %s", e)

    # Lock: Redis → Memory.
    if lock is not None:
        try:
            from icore.engine.lock import MemoryLock

            fallback_lock = MemoryLock()
            primary_health = getattr(lock, "health_check", None)

            async def _lock_health() -> bool:
                if primary_health is None:
                    return True
                try:
                    return bool(await primary_health())
                except Exception:
                    return False

            coord.register(
                name="lock",
                primary=lock,
                fallback=fallback_lock,
                health_check=_lock_health,
                failure_threshold=failure_threshold,
                recovery_interval=recovery_interval,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to wire lock degradation: %s", e)


# ---------------------------------------------------------------------------
# v0.6: Module builders (observability / auth / persistence / dlq /
#        security / triggers / cache)
# ---------------------------------------------------------------------------

def build_observability(
    settings: Optional[Settings] = None,
) -> "Any":
    """构建可观测性（v0.6 §2.1）：metrics registry。

    当 ``settings.observability.enabled`` 为 True 时：
        - 返回 ``get_metrics_registry()`` 单例
        - 日志配置（``setup_observability``）延迟到 lifespan startup，
          避免在 import 时修改全局 logging 状态影响测试

    否则返回 None。
    """
    settings = settings or get_settings()
    if not settings.observability.enabled:
        logger.info("Observability disabled by settings")
        return None
    from icore.observability import get_metrics_registry

    registry = get_metrics_registry()
    logger.info(
        "Built Observability (json_logs=%s, log_dir=%s)",
        settings.observability.json_logs,
        settings.observability.log_dir or "(stdout)",
    )
    return registry


def build_auth_dependency(
    settings: Optional[Settings] = None,
) -> "Any":
    """构建鉴权依赖 + AuthBundle（v0.6 §2.6）。

    当 ``settings.auth.enabled`` 为 True 时：
        - 从 ``config_dir / api_keys_file`` 加载用户
        - 构造 APIKeyAuth（和 JWTAuth，当 jwt_secret 非空时）
        - 构建 AuthBundle（含 api_key_auth / jwt_auth / rbac / exempt_paths）
        - 调 ``bundle.dependency()`` 生成 /invoke 的 dependency callable

    返回 ``(dependency, auth_bundle)`` 元组。当 auth 关闭时返回
    ``(None, None)``。
    """
    settings = settings or get_settings()
    if not settings.auth.enabled:
        logger.info("Auth disabled by settings")
        return None, None
    from icore.auth import (
        APIKeyAuth,
        AuthBundle,
        JWTAuth,
        RBAC,
        create_auth_dependency,
        load_users_yaml,
    )

    api_keys_path = Path(settings.config_dir) / settings.auth.api_keys_file
    users_data = load_users_yaml(api_keys_path)

    # 解析 users.yaml -> {api_key: user_id} + {user_id: role}
    api_users: dict[str, str] = {}
    user_roles: dict[str, str] = {}
    for user_id, info in (users_data.get("users") or {}).items():
        if not isinstance(info, dict):
            continue
        api_key = info.get("api_key")
        role = info.get("role", "developer")
        if api_key:
            api_users[api_key] = user_id
            user_roles[user_id] = role

    api_key_auth = APIKeyAuth(api_users, user_roles=user_roles)

    jwt_auth = None
    if settings.auth.jwt_secret:
        jwt_auth = JWTAuth(
            secret=settings.auth.jwt_secret,
            algorithm=settings.auth.jwt_algorithm,
        )

    exempt_paths = {
        "/health",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    bundle = AuthBundle(
        api_key_auth=api_key_auth,
        jwt_auth=jwt_auth,
        rbac=RBAC(),
        exempt_paths=exempt_paths,
    )
    dependency = bundle.dependency(
        required_permission=settings.auth.required_permission
    )
    logger.info(
        "Built Auth dependency (mode=%s, users=%d, jwt=%s)",
        settings.auth.mode,
        len(api_users),
        jwt_auth is not None,
    )
    return dependency, bundle


def build_persistence_manager(
    settings: Optional[Settings] = None,
) -> "Any":
    """构建工作流持久化管理器（v0.6 §2.4）。

    当 ``settings.persistence.enabled`` 为 True 时，根据 backend 选择
    InMemory 或 Postgres 后端。否则返回 None。
    """
    settings = settings or get_settings()
    if not settings.persistence.enabled:
        logger.info("Persistence disabled by settings")
        return None
    from icore.persistence import (
        InMemoryPersistenceBackend,
        PostgresPersistenceBackend,
        WorkflowPersistenceManager,
    )

    backend_str = settings.persistence.backend.lower()
    if backend_str == "postgres":
        backend = PostgresPersistenceBackend(
            settings.persistence.postgres_dsn
        )
    else:
        backend = InMemoryPersistenceBackend()

    mgr = WorkflowPersistenceManager(backend)
    logger.info("Built PersistenceManager (backend=%s)", backend_str)
    return mgr


def build_dlq(
    settings: Optional[Settings] = None,
) -> "Any":
    """构建死信队列（v0.6 §3.1.2）。

    当 ``settings.dlq.enabled`` 为 True 时，根据 backend 选择
    InMemory 或 Postgres 后端。否则返回 None。
    """
    settings = settings or get_settings()
    if not settings.dlq.enabled:
        logger.info("DLQ disabled by settings")
        return None
    from icore.engine.dead_letter_queue import (
        DeadLetterQueue,
        InMemoryDLQBackend,
        PostgresDLQBackend,
    )

    backend_str = settings.dlq.backend.lower()
    if backend_str == "postgres":
        backend = PostgresDLQBackend(settings.dlq.postgres_dsn)
    else:
        backend = InMemoryDLQBackend()

    dlq = DeadLetterQueue(
        backend,
        default_ttl_seconds=settings.dlq.default_ttl_seconds,
    )
    logger.info("Built DeadLetterQueue (backend=%s)", backend_str)
    return dlq


def build_security(
    settings: Optional[Settings] = None,
) -> "Any":
    """构建安全检测器（v0.6 §3.5）。

    当 ``settings.security.injection_detection`` 为 True 时，构造
    ``InjectionDetector``。当 ``settings.security.pii_masking`` 为 True
    时，构造 ``PIIDetector``。

    返回 ``(injection_detector, pii_detector)`` 元组。当对应功能关闭时
    对应元素为 None。
    """
    settings = settings or get_settings()
    injection_detector = None
    pii_detector = None

    if settings.security.injection_detection:
        from icore.security import InjectionDetector

        injection_detector = InjectionDetector(
            threshold=settings.security.injection_threshold,
        )
        logger.info(
            "Built InjectionDetector (threshold=%.2f)",
            settings.security.injection_threshold,
        )

    if settings.security.pii_masking:
        from icore.security import PIIDetector

        # 解析启用的 PII 类别（逗号分隔），空字符串表示全部启用。
        raw_cats = settings.security.pii_enabled_categories.strip()
        enabled_cats = None
        if raw_cats:
            enabled_cats = [
                c.strip() for c in raw_cats.split(",") if c.strip()
            ]
        pii_detector = PIIDetector(enabled_categories=enabled_cats)
        logger.info(
            "Built PIIDetector (categories=%s)",
            enabled_cats or "all",
        )

    return injection_detector, pii_detector


def build_trigger_manager(
    settings: Optional[Settings] = None,
    workflow_registry: Any = None,
    callback_manager: Any = None,
) -> "Any":
    """构建触发器管理器（v0.6 §2.5）。

    当 ``settings.triggers.enabled`` 为 True 时：
        - 构造 ``TriggerManager``
        - 从 ``config_dir / triggers.yaml`` 加载配置并注册
        - 返回 manager（start_all 在 lifespan startup 调）

    否则返回 None。
    """
    settings = settings or get_settings()
    if not settings.triggers.enabled:
        logger.info("Triggers disabled by settings")
        return None
    from icore.triggers import TriggerManager

    mgr = TriggerManager(workflow_registry, callback_manager)
    cfg_path = Path(settings.config_dir) / settings.triggers.config_file
    if cfg_path.exists():
        try:
            import asyncio

            # load_from_yaml 是 async，但此处可能在无 event loop 时调用
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果 loop 已在运行，用 ensure_future 延迟加载
                    asyncio.ensure_future(mgr.load_from_yaml(cfg_path))
                else:
                    loop.run_until_complete(mgr.load_from_yaml(cfg_path))
            except RuntimeError:
                # 无 event loop，创建新的
                asyncio.run(mgr.load_from_yaml(cfg_path))
        except Exception as e:
            logger.warning("Failed to load triggers from %s: %s", cfg_path, e)
    else:
        logger.warning(
            "Triggers enabled but %s not found", cfg_path
        )
    logger.info("Built TriggerManager (config=%s)", cfg_path)
    return mgr


def build_semantic_cache(
    settings: Optional[Settings] = None,
    model_manager: Any = None,
    vectorstore: Any = None,
) -> "Any":
    """构建语义缓存（v0.6 §2.3）。

    当 ``settings.cache.enabled`` 为 True 时：
        - 构造 ``SemanticCache``（L1 精确匹配，L2 暂不接）
        - 调 ``model_manager.set_semantic_cache(cache)`` 注入
        - 返回 cache

    否则返回 None。
    """
    settings = settings or get_settings()
    if not settings.cache.enabled:
        logger.info("Semantic cache disabled by settings")
        return None
    from icore.cache import create_semantic_cache

    # L2 需要 embedding model，暂不接入
    cache = create_semantic_cache(
        vectorstore=None,
        embed_fn=None,
        enable_l1=settings.cache.enable_l1,
        enable_l2=False,
        similarity_threshold=settings.cache.similarity_threshold,
        l1_max_size=settings.cache.l1_max_size,
        default_ttl=settings.cache.default_ttl,
    )
    if model_manager is not None:
        model_manager.set_semantic_cache(cache)
    logger.info(
        "Built SemanticCache (l1=%s, l2=False)",
        settings.cache.enable_l1,
    )
    return cache


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
        5. v0.5: Build VectorStore / GraphStore / MediaProcessorRegistry /
           DistributedLock / IdempotencyCache / ObjectStore
           from their respective configs
        6. v0.6: Build BackpressureCoordinator / HotReloadCoordinator /
           GracefulDegradationCoordinator and wire degradation providers
        7. Call ``icore.api.main.create_app()`` with all managers injected
        8. Install startup/shutdown lifecycle hooks:
           - startup: start model health monitor, start task queue,
                      start degradation watcher, start hot-reload observer
           - shutdown: cancel pending background tasks, close every
                      manager that owns external resources, stop the
                      hot-reload observer and degradation watcher

    Args:
        settings: Optional Settings override. Defaults to ``get_settings()``.

    Returns:
        A configured FastAPI application with infrastructure attached.
    """
    from icore.api.main import create_app

    settings = settings or get_settings()

    # 1. Register workflows (must happen before the app serves requests)
    autoregister_workflows(settings)

    # 2-5. Build infrastructure managers from YAML + settings
    model_manager = build_model_manager(settings)
    db_manager = build_db_manager(settings)
    concurrency_controller = build_concurrency_controller(settings)
    task_queue = build_task_queue(settings)
    instance_manager = build_instance_manager(settings)
    # v0.5 new components.
    vectorstore = build_vectorstore(settings)
    graphstore = build_graphstore(settings)
    media_processor = build_media_processor(settings)
    lock = build_lock(settings)
    idempotency_cache = build_idempotency_cache(settings)
    objectstore = build_objectstore(settings)
    # v0.6 engineering enhancements.
    backpressure_coord = build_backpressure_coordinator(settings)
    hot_reload_coord = build_hot_reload_coordinator(settings)
    degradation_coord = build_degradation_coordinator(settings)
    _wire_degradation_providers(
        degradation_coord,
        model_manager=model_manager,
        db_manager=db_manager,
        vectorstore=vectorstore,
        graphstore=graphstore,
        lock=lock,
        objectstore=objectstore,
        settings=settings,
    )

    # v0.6 module wirings (所有默认关闭或用 InMemory)
    metrics_registry = build_observability(settings)
    auth_dependency, auth_bundle = build_auth_dependency(settings)
    persistence_manager = build_persistence_manager(settings)
    dlq_manager = build_dlq(settings)
    security_injection_detector, security_pii_detector = build_security(settings)
    from icore.engine.registry import WorkflowRegistry as _WR

    trigger_manager = build_trigger_manager(
        settings,
        workflow_registry=_WR.default(),
        callback_manager=None,
    )
    semantic_cache = build_semantic_cache(
        settings, model_manager=model_manager, vectorstore=vectorstore
    )

    # 6. Create the FastAPI app with managers injected
    app = create_app(
        settings=settings,
        model_manager=model_manager,
        db_manager=db_manager,
        concurrency_controller=concurrency_controller,
        task_queue=task_queue,
        instance_manager=instance_manager,
        vectorstore=vectorstore,
        graphstore=graphstore,
        media_processor=media_processor,
        lock=lock,
        idempotency_cache=idempotency_cache,
        objectstore=objectstore,
        backpressure_coordinator=backpressure_coord,
        hot_reload_coordinator=hot_reload_coord,
        degradation_coordinator=degradation_coord,
        auth_dependency=auth_dependency,
        auth_bundle=auth_bundle,
        persistence_manager=persistence_manager,
        dlq=dlq_manager,
        metrics_registry=metrics_registry,
        security_injection_detector=security_injection_detector,
        pii_detector=security_pii_detector,
    )

    # 7. Wire hot-reload callbacks (models.yaml / prompts/*).
    if hot_reload_coord is not None:
        _wire_hot_reload_callbacks(
            hot_reload_coord,
            model_manager=model_manager,
            settings=settings,
        )
        # v0.6: triggers.yaml 变化时重建 TriggerManager
        if trigger_manager is not None:
            hot_reload_coord.set_trigger_manager(trigger_manager)
            try:
                hot_reload_coord.register(
                    "triggers.yaml", hot_reload_coord._reload_triggers
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Failed to register triggers.yaml hot-reload watcher"
                )

    # 8. Lifecycle hooks
    @app.on_event("startup")
    async def _on_startup() -> None:
        # v0.6: 配置结构化日志（在 startup 而非 import 时调用，
        #       避免修改全局 logging 状态影响测试）
        if metrics_registry is not None:
            try:
                from icore.observability import setup_observability
                import logging as _logging

                level_map = {
                    "DEBUG": _logging.DEBUG,
                    "INFO": _logging.INFO,
                    "WARNING": _logging.WARNING,
                    "ERROR": _logging.ERROR,
                }
                level = level_map.get(
                    settings.observability.log_level.upper(),
                    _logging.INFO,
                )
                setup_observability(
                    level=level,
                    json_logs=settings.observability.json_logs,
                    log_dir=settings.observability.log_dir or None,
                )
            except Exception as e:
                logger.warning("Failed to setup observability: %s", e)
        try:
            await task_queue.start()
        except Exception as e:
            logger.warning("Failed to start task queue: %s", e)
        try:
            if model_manager is not None and len(model_manager) > 0:
                await model_manager.start_health_monitor()
        except Exception as e:
            logger.warning("Failed to start model health monitor: %s", e)
        # v0.6: start degradation auto-recovery watcher.
        if degradation_coord is not None:
            try:
                await degradation_coord.start_watch()
            except Exception as e:
                logger.warning("Failed to start degradation watcher: %s", e)
        # v0.6: start hot-reload observer.
        if hot_reload_coord is not None:
            try:
                await hot_reload_coord.start()
            except Exception as e:
                logger.warning("Failed to start hot-reload observer: %s", e)
        # v0.6: start trigger manager.
        if trigger_manager is not None:
            try:
                await trigger_manager.start_all()
            except Exception as e:
                logger.warning("Failed to start trigger manager: %s", e)

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
        # v0.6: stop hot-reload observer and degradation watcher first so
        # they don't try to mutate managers we're about to close.
        if hot_reload_coord is not None:
            try:
                await hot_reload_coord.stop()
            except Exception as e:
                logger.warning("Error stopping hot-reload observer: %s", e)
        if degradation_coord is not None:
            try:
                await degradation_coord.close()
            except Exception as e:
                logger.warning("Error stopping degradation watcher: %s", e)
        # v0.6: stop trigger manager.
        if trigger_manager is not None:
            try:
                await trigger_manager.stop_all()
            except Exception as e:
                logger.warning("Error stopping trigger manager: %s", e)
        # v0.6: close persistence manager + DLQ.
        if persistence_manager is not None:
            try:
                await persistence_manager.close()
            except Exception as e:
                logger.warning("Error closing persistence manager: %s", e)
        if dlq_manager is not None:
            try:
                close = getattr(dlq_manager, "close", None)
                if close is not None:
                    result = close()
                    if hasattr(result, "__await__"):
                        await result
            except Exception as e:
                logger.warning("Error closing DLQ: %s", e)
        # v0.5: graceful close of every component that owns resources.
        # Each close is wrapped in try/except so one failing close
        # doesn't prevent the others from running.
        closables: list[tuple[str, Any]] = [
            ("model_manager", model_manager),
            ("db_manager", db_manager),
            ("vectorstore", getattr(app.state, "vectorstore", None)),
            ("graphstore", getattr(app.state, "graphstore", None)),
            ("objectstore", getattr(app.state, "objectstore", None)),
            ("backpressure_coordinator", backpressure_coord),
        ]
        for name, obj in closables:
            if obj is None:
                continue
            close = getattr(obj, "close", None) or getattr(
                obj, "close_all", None
            )
            if close is None:
                continue
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning("Error closing %s: %s", name, e)
        logger.info("Shutdown complete")

    return app


def _wire_hot_reload_callbacks(
    coord: Any,
    *,
    model_manager: Any,
    settings: Settings,
) -> None:
    """Register reload callbacks for known config files.

    Currently wires:
        - ``models.yaml`` → rebuild ModelManager (re-register models +
          routing rules without disturbing in-flight adapters).
        - ``prompts/**/*.yaml`` → reload into the global PromptManager
          (when one is configured).
    """
    def _reload_models(path: Any) -> None:
        if model_manager is None:
            return
        try:
            from icore.bootstrap import load_yaml_config
            from icore.config import ModelConfig, ModelSettings, RoutingRule

            data = load_yaml_config(path)
            models: dict[str, ModelConfig] = {}
            for model_id, raw in (data.get("models") or {}).items():
                if not isinstance(raw, dict):
                    continue
                body = dict(raw)
                body["model_id"] = model_id
                try:
                    models[model_id] = ModelConfig(**body)
                except Exception as e:
                    logger.warning(
                        "Hot reload: skipping model '%s' (invalid): %s",
                        model_id, e,
                    )
            routing_rules: list[RoutingRule] = []
            for raw_rule in (data.get("routing_rules") or []):
                if not isinstance(raw_rule, dict):
                    continue
                try:
                    routing_rules.append(RoutingRule(**raw_rule))
                except Exception as e:
                    logger.warning(
                        "Hot reload: skipping routing rule %r: %s",
                        raw_rule, e,
                    )
            ms = ModelSettings(
                default_model_id=data.get("default_model_id"),
                enable_auto_routing=data.get("enable_auto_routing", True),
                models=models,
                routing_rules=routing_rules,
                health_check_interval=data.get(
                    "health_check_interval", 60
                ),
            )
            # Re-init the manager in place — register() clears cached
            # adapters so the next get_adapter() re-creates with new
            # config.
            for cfg in models.values():
                model_manager.register(cfg)
            if hasattr(model_manager, "_router"):
                model_manager._router.set_rules(routing_rules)
                default_id = ms.default_model_id
                if default_id:
                    model_manager._router.set_default_model_id(default_id)
            logger.info(
                "Hot reload: reloaded %d model(s) from %s",
                len(models), path,
            )
        except Exception:
            logger.exception(
                "Hot reload of models.yaml failed (path=%s)", path
            )

    try:
        coord.register("models.yaml", _reload_models)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to register models.yaml hot-reload watcher")
