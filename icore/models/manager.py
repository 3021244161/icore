"""
icore.models.manager - Multi-model LLM adapter manager.

ModelManager is the top-level entry point for LLM access in icore.
It manages multiple model adapters, handles lazy adapter creation,
delegates auto-routing to ModelRouter, and provides health monitoring.

Upper layers (Task / TaskContext) obtain adapters through
``ModelManager.get_adapter()``::

    adapter = manager.get_adapter("gpt-4o")       # explicit
    adapter = manager.get_adapter(None)            # auto-route

The manager is thread-safe and supports dynamic registration of new
models at runtime.

Integration with TaskContext:
    TaskContext.get_model_adapter() calls manager.get_adapter(model_id).
    When model_id is None, the manager delegates to ModelRouter which
    selects the best model based on routing rules, health, and cost.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from icore.config import ModelConfig, ModelSettings
from icore.exceptions import (
    CircuitBreakerOpenError,
    ModelAPIError,
    ModelNotFoundError,
    ModelTimeoutError,
    ModelUnhealthyError,
    ModelUnsupportedError,
    NoAvailableModelError,
)
from icore.models.base_adapter import BaseModelAdapter
from icore.models.config import ModelHealthStatus
from icore.models.openai_adapter import OpenAICompatibleAdapter
from icore.models.router import ModelRouter
from icore.models.vision_adapter import VisionModelAdapter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelInfo - metadata about a registered model
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """
    Lightweight metadata about a registered model.

    Used by ModelManager.list_models() and ModelRouter for routing
    decisions. This is a plain dataclass (not Pydantic) for speed.

    Attributes:
        model_id:           Unique model identifier.
        model_name:         Model name for API calls (e.g. "gpt-4o").
        enabled:            Whether the model is enabled for use.
        healthy:            Whether the model passed its last health check.
        tags:               Routing tags (e.g. ["cheap", "fast", "capable"]).
        cost_per_1k_input:  Cost per 1k input tokens.
        cost_per_1k_output: Cost per 1k output tokens.
        has_adapter:        Whether the adapter has been created (cached).
    """

    model_id: str
    model_name: str
    enabled: bool
    healthy: bool = True
    tags: list[str] = field(default_factory=list)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    has_adapter: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for router consumption."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "enabled": self.enabled,
            "healthy": self.healthy,
            "tags": list(self.tags),
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "has_adapter": self.has_adapter,
        }


# ---------------------------------------------------------------------------
# Internal health tracking
# ---------------------------------------------------------------------------

@dataclass
class _ModelHealth:
    """Internal health tracking for a single model."""

    healthy: bool = True
    consecutive_failures: int = 0
    latency_ms: float | None = None
    last_error: str | None = None
    last_checked: str | None = None


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Multi-model LLM adapter manager.

    Manages a collection of model adapters, each backed by a
    ``ModelConfig``. Adapters are created lazily on first access and
    cached for subsequent calls.

    When auto-routing is enabled and ``get_adapter(None)`` is called,
    the manager delegates to ``ModelRouter`` to select the best model
    based on routing rules, task characteristics, and model availability.

    Thread safety:
        All public methods are protected by a reentrant lock.

    Usage::

        from icore.config import ModelConfig
        from icore.models.manager import ModelManager

        mgr = ModelManager()
        mgr.register(ModelConfig(
            model_id="gpt-4o",
            model_name="gpt-4o",
            api_base="https://api.openai.com",
            api_key="sk-...",
            tags=["capable", "balanced"],
        ))

        # Explicit model
        adapter = mgr.get_adapter("gpt-4o")
        result = await adapter.chat([{"role": "user", "content": "Hi"}])

        # Auto-routed model
        adapter = mgr.get_adapter(None)

    Attributes:
        _configs:        Maps model_id -> ModelConfig.
        _adapters:       Maps model_id -> BaseModelAdapter (lazily created).
        _router:         ModelRouter instance for auto-routing.
        _health:         Maps model_id -> _ModelHealth tracking.
        _lock:           Reentrant lock for thread safety.
        _auto_routing:   Whether auto-routing is enabled.
    """

    # Adapter class registry: model_type -> adapter_class.
    # v0.5: includes "vision" for multimodal vision-capable models.
    _ADAPTER_MAP: dict[str, type[BaseModelAdapter]] = {
        "openai": OpenAICompatibleAdapter,
        "vision": VisionModelAdapter,
    }

    def __init__(
        self,
        model_settings: Optional[ModelSettings] = None,
    ) -> None:
        """
        Initialize the manager, optionally from ModelSettings.

        Args:
            model_settings: If provided, registers all models and
                            routing rules from the settings. If None,
                            the manager starts empty and models must be
                            registered manually.
        """
        self._configs: dict[str, ModelConfig] = {}
        self._adapters: dict[str, BaseModelAdapter] = {}
        self._health: dict[str, _ModelHealth] = {}
        self._lock = threading.RLock()
        self._auto_routing: bool = True
        self._health_interval: int = 60
        self._health_check_task: Optional[asyncio.Task[None]] = None

        # v0.6: 语义缓存（默认 None，不包装 adapter）
        self._semantic_cache: Any = None

        # v0.5: per-model circuit-breaker registry (lazy import to
        # avoid an engine->models dependency cycle).
        from icore.engine.circuit_breaker import CircuitBreakerRegistry

        self._circuit_breakers: CircuitBreakerRegistry = (
            CircuitBreakerRegistry()
        )

        # Build router (it needs a reference to this manager)
        self._router = ModelRouter(self)

        if model_settings is not None:
            self._init_from_settings(model_settings)

    def _init_from_settings(self, settings: ModelSettings) -> None:
        """Initialize from ModelSettings: register models + routing rules."""
        self._auto_routing = settings.enable_auto_routing
        self._health_interval = settings.health_check_interval

        for model_id, config in settings.models.items():
            self._configs[model_id] = config
            self._health[model_id] = _ModelHealth()
            logger.info("Registered model '%s' (%s)", model_id, config.model_name)

        # Configure router
        self._router.set_rules(list(settings.routing_rules))
        if settings.default_model_id:
            self._router.set_default_model_id(settings.default_model_id)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, config: ModelConfig) -> None:
        """
        Register a model configuration.

        The adapter is created lazily on first ``get_adapter()`` call.
        If a model with the same ID already exists, it is replaced
        (existing cached adapter is cleared).

        Args:
            config: The ModelConfig for this model.
        """
        with self._lock:
            model_id = config.model_id
            self._configs[model_id] = config
            self._health[model_id] = _ModelHealth()

            # Clear cached adapter if re-registering
            if model_id in self._adapters:
                self._adapters.pop(model_id)
                logger.info(
                    "Re-registering model '%s'; cleared cached adapter",
                    model_id,
                )

            logger.info(
                "Registered model '%s' (name=%s, tags=%s, enabled=%s)",
                model_id, config.model_name, config.tags, config.enabled,
            )

    def register_adapter(
        self,
        model_id: str,
        adapter: BaseModelAdapter,
    ) -> None:
        """
        Register a pre-built adapter instance.

        This bypasses lazy creation - the adapter is stored directly.
        Useful for custom adapter implementations or testing.

        Args:
            model_id:  The model ID to register under.
            adapter:   A BaseModelAdapter instance.
        """
        if not isinstance(adapter, BaseModelAdapter):
            raise TypeError(
                f"adapter must be a BaseModelAdapter instance, "
                f"got {type(adapter)}"
            )
        with self._lock:
            self._adapters[model_id] = adapter
            self._configs[model_id] = adapter.config
            self._health[model_id] = _ModelHealth()
            logger.info(
                "Registered pre-built adapter for model '%s'", model_id
            )

    def register_adapter_type(
        self,
        model_type: str,
        adapter_class: type[BaseModelAdapter],
    ) -> None:
        """
        Register a custom adapter class for a model type.

        This allows extending ModelManager with new LLM providers
        (e.g. Anthropic, Cohere) without modifying source code.

        Args:
            model_type:    Provider type identifier (e.g. "anthropic").
            adapter_class: A BaseModelAdapter subclass.
        """
        if not issubclass(adapter_class, BaseModelAdapter):
            raise TypeError(
                f"adapter_class must be a subclass of BaseModelAdapter, "
                f"got {adapter_class}"
            )
        with self._lock:
            self._ADAPTER_MAP[model_type] = adapter_class
            logger.info("Registered adapter type '%s' -> %s",
                        model_type, adapter_class.__name__)

    async def unregister(self, model_id: str) -> None:
        """
        Remove a model from the manager.

        Closes the adapter if it was created, then removes config and
        cached adapter.

        Args:
            model_id: The model ID to remove.
        """
        with self._lock:
            self._configs.pop(model_id, None)
            self._health.pop(model_id, None)
            adapter = self._adapters.pop(model_id, None)

        # Close adapter outside lock to avoid blocking.
        # ``unregister`` is async so the adapter's async ``close()`` is
        # awaited properly (no fire-and-forget task that may be GC'd or
        # skipped when no event loop is running).
        if adapter is not None and hasattr(adapter, "close"):
            try:
                await adapter.close()
            except Exception as e:
                logger.warning(
                    "Failed to close adapter for '%s': %s", model_id, e
                )
            logger.info("Unregistered and closed model '%s'", model_id)

    # ------------------------------------------------------------------
    # Adapter access
    # ------------------------------------------------------------------

    def set_semantic_cache(self, cache: Any) -> None:
        """注入语义缓存（v0.6）。

        设置后，``get_adapter()`` 返回的 adapter 会被
        ``CachedModelAdapter`` 包装，使 LLM 调用走缓存。
        传 ``None`` 可清除缓存包装。
        """
        with self._lock:
            self._semantic_cache = cache
            # 清除已缓存的 adapter，使下次 get_adapter 重新创建并包装
            if cache is not None:
                self._adapters.clear()

    def get_adapter(
        self,
        model_id: Optional[str] = None,
        *,
        task_type: Optional[str] = None,
        complexity: Optional[str] = None,
        cost_constraint: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> BaseModelAdapter:
        """
        Get a model adapter by ID or via auto-routing.

        If ``model_id`` is provided and the model exists, returns its
        adapter directly. If ``model_id`` is None and auto-routing is
        enabled, delegates to ModelRouter to select the best model.

        Args:
            model_id:        Explicit model ID, or None for auto-routing.
            task_type:       Task category hint for routing (optional).
            complexity:      Task complexity hint: "low" | "medium" | "high".
            cost_constraint: Max cost per 1k tokens for routing.
            tags:            Tag hints for routing (e.g. ["fast", "cheap"]).

        Returns:
            A BaseModelAdapter instance ready for use.

        Raises:
            ModelNotFoundError:     If model_id is not registered.
            NoAvailableModelError:  If auto-routing fails.
            ModelUnhealthyError:    If model is explicitly requested but unhealthy.
        """
        with self._lock:
            # Explicit model request
            if model_id is not None:
                adapter = self._get_or_create_adapter(model_id)
                actual_id = model_id
            else:
                # Auto-routing
                if not self._auto_routing:
                    raise NoAvailableModelError(
                        "Auto-routing is disabled and no model_id specified"
                    )

                selected_id = self._router.auto_route(
                    task_type=task_type,
                    max_cost=cost_constraint,
                    tags=tags,
                )
                if selected_id is None:
                    raise NoAvailableModelError(
                        f"task_type={task_type}, tags={tags}"
                    )
                logger.info(
                    "Auto-routed to model '%s' (task_type=%s, complexity=%s)",
                    selected_id, task_type, complexity,
                )
                adapter = self._get_or_create_adapter(selected_id)
                actual_id = selected_id

            # v0.6: 如果已注入语义缓存且该 adapter 尚未被包装，
            #       用 CachedModelAdapter 包装并缓存。
            if self._semantic_cache is not None:
                from icore.cache import CachedModelAdapter

                if not isinstance(adapter, CachedModelAdapter):
                    adapter = CachedModelAdapter(
                        adapter=adapter,
                        cache=self._semantic_cache,
                        model_id=actual_id,
                    )
                    self._adapters[actual_id] = adapter

            return adapter

    def get_model(self, model_id: str) -> BaseModelAdapter:
        """
        Alias for get_adapter(model_id) with a required model_id.

        Args:
            model_id: The model ID to retrieve.

        Returns:
            A BaseModelAdapter instance.
        """
        return self.get_adapter(model_id)

    def _get_or_create_adapter(self, model_id: str) -> BaseModelAdapter:
        """
        Get a cached adapter or create one lazily.

        Args:
            model_id: The model ID.

        Returns:
            A BaseModelAdapter instance.

        Raises:
            ModelNotFoundError:  If model_id is not registered.
            ModelUnhealthyError:  If model is registered but unhealthy.
        """
        if model_id not in self._configs:
            raise ModelNotFoundError(
                model_id, available=list(self._configs.keys())
            )

        config = self._configs[model_id]
        if not config.enabled:
            raise ModelUnhealthyError(model_id, "model is disabled")

        # Return cached adapter if available
        if model_id in self._adapters:
            return self._adapters[model_id]

        # Select adapter class by config.model_type (v0.5).
        # Defaults to OpenAI-compatible for unknown types.
        model_type = getattr(config, "model_type", "openai") or "openai"
        adapter_class = self._ADAPTER_MAP.get(
            model_type, OpenAICompatibleAdapter
        )
        adapter = adapter_class(config)
        self._adapters[model_id] = adapter
        logger.debug(
            "Created adapter for model '%s' (type=%s, class=%s)",
            model_id, model_type, adapter_class.__name__,
        )
        return adapter

    # ------------------------------------------------------------------
    # Router access
    # ------------------------------------------------------------------

    def get_router(self) -> ModelRouter:
        """Get the ModelRouter instance."""
        return self._router

    def set_router(self, router: ModelRouter) -> None:
        """Set a custom ModelRouter instance."""
        with self._lock:
            self._router = router

    # ------------------------------------------------------------------
    # Introspection (used by ModelRouter)
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        """
        List all models with their configuration and health info.

        Returns a list of dicts (not ModelInfo objects) because the
        ModelRouter accesses fields via ``dict.get()``.

        Each dict has keys: model_id, model_name, enabled, healthy,
        tags, cost_per_1k_input, cost_per_1k_output, has_adapter.

        Returns:
            List of model info dicts.
        """
        with self._lock:
            result: list[dict[str, Any]] = []
            for model_id, config in self._configs.items():
                health = self._health.get(model_id, _ModelHealth())
                result.append({
                    "model_id": model_id,
                    "model_name": config.model_name,
                    "enabled": config.enabled,
                    "healthy": health.healthy,
                    "tags": list(config.tags),
                    "cost_per_1k_input": config.cost_per_1k_input,
                    "cost_per_1k_output": config.cost_per_1k_output,
                    "has_adapter": model_id in self._adapters,
                })
            return result

    def get_model_info(self, model_id: str) -> ModelInfo:
        """
        Get detailed info about a specific model.

        Args:
            model_id: The model ID.

        Returns:
            A ModelInfo dataclass instance.

        Raises:
            ModelNotFoundError: If model_id is not registered.
        """
        with self._lock:
            if model_id not in self._configs:
                raise ModelNotFoundError(
                    model_id, available=list(self._configs.keys())
                )
            config = self._configs[model_id]
            health = self._health.get(model_id, _ModelHealth())
            return ModelInfo(
                model_id=model_id,
                model_name=config.model_name,
                enabled=config.enabled,
                healthy=health.healthy,
                tags=list(config.tags),
                cost_per_1k_input=config.cost_per_1k_input,
                cost_per_1k_output=config.cost_per_1k_output,
                has_adapter=model_id in self._adapters,
            )

    def get_all_health_status(self) -> dict[str, _ModelHealth]:
        """
        Get health status for all models.

        Used by ModelRouter for health-aware routing decisions.

        Returns:
            Dict mapping model_id -> _ModelHealth (with
            consecutive_failures attribute used by router scoring).
        """
        with self._lock:
            return dict(self._health)

    @property
    def is_auto_routing_enabled(self) -> bool:
        """Whether auto-routing is enabled."""
        return self._auto_routing

    def set_auto_routing(self, enabled: bool) -> None:
        """Enable or disable auto-routing."""
        with self._lock:
            self._auto_routing = enabled

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self, model_id: str) -> ModelHealthStatus:
        """
        Check the health of a single model.

        Updates internal health tracking and returns a ModelHealthStatus.

        Args:
            model_id: The model ID to check.

        Returns:
            A ModelHealthStatus instance.
        """
        with self._lock:
            if model_id not in self._configs:
                return ModelHealthStatus(
                    model_id=model_id,
                    healthy=False,
                    error=f"Model '{model_id}' not registered",
                )
            config = self._configs[model_id]
            if not config.enabled:
                return ModelHealthStatus(
                    model_id=model_id,
                    healthy=False,
                    error="Model is disabled",
                )

        try:
            adapter = self._get_or_create_adapter(model_id)
            start = time.monotonic()
            is_healthy = await adapter.health_check()
            latency_ms = (time.monotonic() - start) * 1000

            # Update internal health tracking
            with self._lock:
                health = self._health.setdefault(model_id, _ModelHealth())
                if is_healthy:
                    health.healthy = True
                    health.consecutive_failures = 0
                    health.latency_ms = round(latency_ms, 2)
                    health.last_error = None
                else:
                    health.healthy = False
                    health.consecutive_failures += 1
                    health.last_error = "Health check returned False"
                health.last_checked = time.strftime("%Y-%m-%dT%H:%M:%S")

            return ModelHealthStatus(
                model_id=model_id,
                healthy=is_healthy,
                latency_ms=round(latency_ms, 2) if is_healthy else None,
                error=None if is_healthy else "Health check returned False",
                checked_at=health.last_checked,
            )

        except Exception as e:
            logger.error("Health check failed for '%s': %s", model_id, e)
            with self._lock:
                health = self._health.setdefault(model_id, _ModelHealth())
                health.healthy = False
                health.consecutive_failures += 1
                health.last_error = str(e)
                health.last_checked = time.strftime("%Y-%m-%dT%H:%M:%S")

            return ModelHealthStatus(
                model_id=model_id,
                healthy=False,
                error=str(e),
                checked_at=health.last_checked,
            )

    async def health_check_all(self) -> list[ModelHealthStatus]:
        """
        Check the health of all registered models concurrently.

        Runs all health checks in parallel using asyncio.gather().

        Returns:
            A list of ModelHealthStatus, one per registered model.
        """
        with self._lock:
            model_ids = list(self._configs.keys())

        if not model_ids:
            return []

        tasks = [self.health_check(mid) for mid in model_ids]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        healthy = sum(1 for r in results if r.healthy)
        logger.info(
            "Health check complete: %d/%d models healthy",
            healthy, len(results),
        )
        return list(results)

    async def start_health_monitor(self) -> None:
        """
        Start a background task that periodically checks model health.

        The task runs ``health_check_all()`` every ``_health_interval``
        seconds. To stop it, call ``stop_health_monitor()``.
        """
        if self._health_check_task is not None:
            logger.warning("Health monitor already running")
            return

        async def _monitor_loop() -> None:
            logger.info(
                "Started health monitor (interval=%ds)", self._health_interval
            )
            while True:
                try:
                    await self.health_check_all()
                except Exception as e:
                    logger.error("Health monitor error: %s", e)
                await asyncio.sleep(self._health_interval)

        self._health_check_task = asyncio.create_task(_monitor_loop())

    def stop_health_monitor(self) -> None:
        """Stop the background health monitor task."""
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            self._health_check_task = None
            logger.info("Stopped health monitor")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """Close all cached adapters and release resources."""
        self.stop_health_monitor()
        with self._lock:
            adapters = list(self._adapters.values())
            self._adapters.clear()

        for adapter in adapters:
            if hasattr(adapter, "close"):
                try:
                    await adapter.close()  # type: ignore[attr-defined]
                except Exception as e:
                    logger.error("Error closing adapter: %s", e)
        logger.info("Closed all model adapters")

    # ------------------------------------------------------------------
    # v0.5: Circuit-breaker + fallback chain
    # ------------------------------------------------------------------

    async def call_with_fallback(
        self,
        model_id: Optional[str],
        fn: Callable[[BaseModelAdapter], Awaitable[Any]],
        fallback_model_ids: Optional[list[str]] = None,
    ) -> Any:
        """
        Invoke ``fn(adapter)`` through the per-model circuit breaker,
        falling back to ``fallback_model_ids`` in order on breaker-open
        or model-API errors.

        Args:
            model_id:            Primary model ID. None uses the
                                 default routed model.
            fn:                  Callable taking an adapter and
                                 returning an awaitable.
            fallback_model_ids:  Optional fallback chain.

        Raises:
            NoAvailableModelError: If every model (primary + fallbacks)
                                   is unavailable.
        """
        # Resolve primary model_id (auto-route when None).
        if model_id is None:
            with self._lock:
                default_id = self._router._default_model_id
            if default_id is None:
                # Fall back to the first registered model if any.
                with self._lock:
                    ids = list(self._configs.keys())
                if not ids:
                    raise NoAvailableModelError("no models registered")
                default_id = ids[0]
            model_id = default_id

        # Try primary.
        cb = await self._circuit_breakers.get(model_id)
        try:
            return await cb.call(lambda: fn(self.get_adapter(model_id)))
        except CircuitBreakerOpenError:
            logger.warning(
                "Circuit breaker open for primary '%s'", model_id
            )
        except (ModelAPIError, ModelTimeoutError) as e:
            logger.warning(
                "Primary model '%s' failed (%s), trying fallbacks",
                model_id,
                type(e).__name__,
            )

        # Fallback chain.
        for fb_id in fallback_model_ids or []:
            try:
                adapter = self.get_adapter(fb_id)
            except (ModelNotFoundError, ModelUnhealthyError) as e:
                logger.warning(
                    "Fallback model '%s' unavailable: %s", fb_id, e
                )
                continue
            cb = await self._circuit_breakers.get(fb_id)
            try:
                return await cb.call(lambda: fn(adapter))
            except Exception as e:
                logger.warning(
                    "Fallback model '%s' failed: %s", fb_id, e
                )
                continue

        raise NoAvailableModelError(
            f"All models exhausted (primary={model_id}, "
            f"fallbacks={fallback_model_ids})"
        )

    def get_circuit_breaker_registry(self) -> Any:
        """Return the per-model ``CircuitBreakerRegistry``."""
        return self._circuit_breakers

    # ------------------------------------------------------------------
    # v0.5: Vision-capable adapter access
    # ------------------------------------------------------------------

    def get_vision_adapter(self, model_id: Optional[str] = None) -> Any:
        """
        Return an adapter that supports vision input.

        Args:
            model_id: Explicit model ID. If None, searches registered
                      models for the first ``supports_vision`` one.

        Raises:
            ModelUnsupportedError: If no vision-capable model is found.
        """
        adapter = self.get_adapter(model_id)
        if not getattr(adapter, "supports_vision", False):
            raise ModelUnsupportedError(
                f"Model '{adapter.model_id}' does not support vision input"
            )
        return adapter

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of registered models."""
        return len(self._configs)

    def __contains__(self, model_id: str) -> bool:
        """Check if a model is registered."""
        return model_id in self._configs

    def __repr__(self) -> str:
        return (
            f"ModelManager(models={len(self._configs)}, "
            f"cached_adapters={len(self._adapters)}, "
            f"auto_routing={self._auto_routing})"
        )
