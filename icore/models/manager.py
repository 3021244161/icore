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
from typing import TYPE_CHECKING, Any, Optional

from icore.config import ModelConfig, ModelSettings
from icore.models.base_adapter import BaseModelAdapter
from icore.models.config import ModelHealthStatus
from icore.models.exceptions import (
    ModelNotFoundError,
    ModelUnhealthyError,
    NoAvailableModelError,
)
from icore.models.openai_adapter import OpenAICompatibleAdapter
from icore.models.router import ModelRouter

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
    # Currently only OpenAI-compatible, but extensible for future providers.
    _ADAPTER_MAP: dict[str, type[BaseModelAdapter]] = {
        "openai": OpenAICompatibleAdapter,
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
                return self._get_or_create_adapter(model_id)

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
            return self._get_or_create_adapter(selected_id)

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

        # Create adapter (default: OpenAI-compatible)
        adapter_class = self._ADAPTER_MAP.get("openai", OpenAICompatibleAdapter)
        adapter = adapter_class(config)
        self._adapters[model_id] = adapter
        logger.debug("Created adapter for model '%s' (%s)",
                     model_id, adapter_class.__name__)
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
