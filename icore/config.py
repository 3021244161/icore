"""
icore.config - Global configuration management.

Configuration is loaded from two sources (later sources override earlier ones):
    1. YAML config files (config/*.yaml)
    2. Environment variables (highest priority, 12-Factor App compliant)

All settings are strongly typed via Pydantic Settings, ensuring type safety
and automatic validation at startup.

Usage:
    from icore.config import settings
    print(settings.app_name)           # "icore"
    print(settings.api_host)            # "0.0.0.0"
    print(settings.api_port)            # 8000
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-settings: API / Server
# ---------------------------------------------------------------------------

class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_API_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="API bind host")
    port: int = Field(default=8000, description="API bind port")
    workers: int = Field(default=1, ge=1, description="Uvicorn worker count")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins",
    )
    request_timeout: int = Field(
        default=300, ge=1, description="Request timeout in seconds"
    )


# ---------------------------------------------------------------------------
# Sub-settings: Concurrency
# ---------------------------------------------------------------------------

class ConcurrencySettings(BaseSettings):
    """Concurrency and task queue configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_CONCURRENCY_",
        env_file=".env",
        extra="ignore",
    )

    max_concurrent_tasks: int = Field(
        default=100, ge=1, description="Global max concurrent task instances"
    )
    max_concurrent_per_workflow: int = Field(
        default=20, ge=1, description="Max concurrent instances per workflow type"
    )
    task_queue_backend: str = Field(
        default="memory",
        description="Queue backend: 'memory' or 'redis'",
    )
    redis_url: Optional[str] = Field(
        default=None, description="Redis URL for distributed queue (if backend='redis')"
    )
    task_timeout: int = Field(
        default=600, ge=1, description="Default task execution timeout in seconds"
    )
    backpressure_threshold: int = Field(
        default=500, ge=1,
        description="Queue depth threshold to trigger backpressure (HTTP 503)",
    )


# ---------------------------------------------------------------------------
# Sub-settings: Database
# ---------------------------------------------------------------------------

class DatabaseConnectionConfig(BaseSettings):
    """Configuration for a single database connection."""

    model_config = SettingsConfigDict(extra="allow")

    db_type: str = Field(description="Database type: postgresql, oracle, hive, mysql")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    username: str = Field(default="")
    password: str = Field(default="")
    database: str = Field(default="")
    pool_size: int = Field(default=10, ge=1, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, description="Max overflow connections")
    pool_recycle: int = Field(
        default=3600, ge=1, description="Connection recycle time in seconds"
    )
    extra_params: Dict[str, Any] = Field(
        default_factory=dict, description="Extra connection parameters"
    )


class DatabaseSettings(BaseSettings):
    """Top-level database configuration managing multiple named connections."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_DB_",
        env_file=".env",
        extra="ignore",
    )

    default_connection: str = Field(
        default="default", description="Name of the default DB connection"
    )
    connections: Dict[str, DatabaseConnectionConfig] = Field(
        default_factory=dict,
        description="Named database connections (name -> config)",
    )


# ---------------------------------------------------------------------------
# Sub-settings: Model / LLM
# ---------------------------------------------------------------------------

class ModelConfig(BaseSettings):
    """Configuration for a single LLM model."""

    model_config = SettingsConfigDict(extra="allow")

    model_id: str = Field(description="Unique model identifier")
    model_name: str = Field(description="Model name for API calls (e.g. gpt-4o)")
    api_base: str = Field(description="OpenAI-compatible API base URL")
    api_key: str = Field(description="API key")
    model_type: str = Field(
        default="openai",
        description=(
            "Adapter type selector. v0.5 supports 'openai' (default, "
            "text-only) and 'vision' (multimodal vision-capable). "
            "Custom types can be registered via "
            "ModelManager.register_adapter_type()."
        ),
    )
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    request_timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0.0)
    cost_per_1k_input: float = Field(default=0.0, ge=0.0, description="Cost per 1k input tokens")
    cost_per_1k_output: float = Field(default=0.0, ge=0.0, description="Cost per 1k output tokens")
    enabled: bool = Field(default=True)
    tags: list[str] = Field(default_factory=list, description="Tags for routing (e.g. 'cheap', 'fast', 'capable')")


class RoutingRule(BaseSettings):
    """A single auto-routing rule."""

    model_config = SettingsConfigDict(extra="allow")

    task_type: Optional[str] = Field(default=None, description="Match by task type")
    max_cost: Optional[float] = Field(default=None, description="Max acceptable cost")
    preferred_tags: list[str] = Field(default_factory=list)
    fallback_model_id: Optional[str] = Field(default=None)


class ModelSettings(BaseSettings):
    """Top-level model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_MODEL_",
        env_file=".env",
        extra="ignore",
    )

    default_model_id: Optional[str] = Field(
        default=None, description="Default model ID when none specified"
    )
    models: Dict[str, ModelConfig] = Field(
        default_factory=dict, description="Registered models (model_id -> config)"
    )
    routing_rules: list[RoutingRule] = Field(
        default_factory=list, description="Auto-routing rules"
    )
    enable_auto_routing: bool = Field(
        default=True, description="Enable automatic model routing"
    )
    health_check_interval: int = Field(
        default=60, ge=1, description="Model health check interval in seconds"
    )


# ---------------------------------------------------------------------------
# Sub-settings: Logging
# ---------------------------------------------------------------------------

class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_LOG_",
        env_file=".env",
        extra="ignore",
    )

    level: str = Field(default="INFO", description="Log level: DEBUG/INFO/WARNING/ERROR")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        description="Loguru format string",
    )
    log_dir: Optional[str] = Field(default=None, description="Log file directory (None = stdout only)")
    rotation: str = Field(default="100 MB", description="Log rotation size")
    retention: str = Field(default="30 days", description="Log retention period")


# ---------------------------------------------------------------------------
# Sub-settings: v0.6 Engineering enhancements
# ---------------------------------------------------------------------------

class BackpressureSettings(BaseSettings):
    """v0.6 §3.2 full-stack backpressure configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_BACKPRESSURE_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True, description="Enable the BackpressureCoordinator"
    )
    memory_budget_mb: float = Field(
        default=0.0,
        description=(
            "Soft RSS ceiling in MB. 0 disables memory-based backpressure. "
            "When exceeded, /health returns 'saturated' and /invoke "
            "returns HTTP 503."
        ),
    )
    check_interval: float = Field(
        default=5.0, description="Seconds between RSS saturation checks"
    )
    # Per-component budgets (model_id / collection name is the unit).
    llm_max_concurrent: int = Field(
        default=20, description="Max concurrent calls per LLM model_id"
    )
    vector_max_concurrent: int = Field(
        default=20, description="Max concurrent Milvus queries"
    )
    graph_max_concurrent: int = Field(
        default=20, description="Max concurrent Neo4j queries"
    )
    db_max_concurrent: int = Field(
        default=20, description="Max concurrent DB queries per connection"
    )


class HotReloadSettings(BaseSettings):
    """v0.6 §3.3.1 configuration hot reload."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_HOT_RELOAD_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable hot reload")
    use_watchdog: bool = Field(
        default=True,
        description=(
            "Try to use the watchdog library for file watching. "
            "Falls back to mtime polling when watchdog is unavailable."
        ),
    )
    poll_interval: float = Field(
        default=1.0, description="Polling interval in seconds (fallback mode)"
    )
    debounce_sec: float = Field(
        default=0.5, description="Debounce window for change events"
    )


class DegradationSettings(BaseSettings):
    """v0.6 §3.3.2 graceful degradation."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_DEGRADATION_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True, description="Enable graceful degradation coordinator"
    )
    failure_threshold: int = Field(
        default=3,
        description="Consecutive failures before degrading to fallback",
    )
    recovery_interval: float = Field(
        default=30.0,
        description="Seconds between primary health probes (auto-recovery)",
    )


# ---------------------------------------------------------------------------
# Sub-settings: v0.6 module wirings (auth / observability / persistence /
# dlq / cache / security / triggers). All default OFF or InMemory so the
# existing test path (infrastructure-free) stays unaffected.
# ---------------------------------------------------------------------------

class AuthSettings(BaseSettings):
    """v0.6 §2.6 authentication / RBAC configuration.

    Disabled by default to preserve the existing infrastructure-free test
    path. Enable in production via ``ICORE_AUTH_ENABLED=true``.
    """

    model_config = SettingsConfigDict(
        env_prefix="ICORE_AUTH_", env_file=".env", extra="ignore"
    )

    enabled: bool = Field(default=False, description="Enable API auth on /invoke")
    mode: str = Field(
        default="api_key",
        description="Auth mode: 'api_key' or 'jwt' or 'both'",
    )
    api_keys_file: str = Field(
        default="users.yaml",
        description="YAML file under config_dir mapping user -> api_key",
    )
    jwt_secret: str = Field(
        default="",
        description="HS256 signing secret for JWT (empty disables JWT)",
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expires_in: int = Field(default=3600, description="JWT lifetime in seconds")
    required_permission: str = Field(
        default="invoke",
        description="RBAC permission required to call /invoke",
    )


class ObservabilitySettings(BaseSettings):
    """v0.6 §2.1 observability (metrics / tracing / structured logs)."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_OBSERVABILITY_", env_file=".env", extra="ignore"
    )

    enabled: bool = Field(
        default=True,
        description=(
            "Enable observability: install MetricsMiddleware + "
            "RequestIDMiddleware and expose GET /metrics. When False, "
            "no middleware is installed and /metrics returns 404."
        ),
    )
    json_logs: bool = Field(
        default=False,
        description=(
            "Emit structured JSON logs via JsonFormatter. False keeps "
            "the default logging format (zero-dependency, AGENTS.md §2)."
        ),
    )
    log_level: str = Field(default="INFO")
    log_dir: str = Field(
        default="", description="Optional log directory for file handlers"
    )
    metrics_path: str = Field(default="/metrics")


class PersistenceSettings(BaseSettings):
    """v0.6 §2.4 workflow execution persistence."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_PERSISTENCE_", env_file=".env", extra="ignore"
    )

    enabled: bool = Field(
        default=True,
        description=(
            "Persist every /invoke execution + per-node checkpoint. "
            "Uses InMemory backend by default; switch to 'postgres' for "
            "production durability."
        ),
    )
    backend: str = Field(
        default="memory",
        description="Backend: 'memory' or 'postgres'",
    )
    postgres_dsn: str = Field(
        default="",
        description="PostgreSQL DSN when backend='postgres'",
    )
    history_default_limit: int = Field(
        default=100, description="Default page size for /history"
    )
    history_max_limit: int = Field(
        default=500, description="Maximum page size for /history"
    )


class DLQSettings(BaseSettings):
    """v0.6 §3.1.2 dead letter queue."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_DLQ_", env_file=".env", extra="ignore"
    )

    enabled: bool = Field(
        default=True,
        description=(
            "Enqueue nodes that exhausted all retries into the DLQ. "
            "InMemory backend by default."
        ),
    )
    backend: str = Field(
        default="memory",
        description="Backend: 'memory' or 'postgres'",
    )
    postgres_dsn: str = Field(
        default="", description="PostgreSQL DSN when backend='postgres'"
    )
    default_ttl_seconds: float = Field(
        default=7 * 24 * 3600.0,
        description="Default TTL for DLQ entries (7 days)",
    )


class CacheSettings(BaseSettings):
    """v0.6 §2.3 LLM response semantic cache."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_CACHE_", env_file=".env", extra="ignore"
    )

    enabled: bool = Field(
        default=False,
        description=(
            "Wrap model adapters with CachedModelAdapter. Disabled by "
            "default to avoid changing LLM behavior in tests."
        ),
    )
    enable_l1: bool = Field(default=True, description="Enable L1 exact match")
    enable_l2: bool = Field(
        default=False,
        description=(
            "Enable L2 vector similarity. Requires a vectorstore + "
            "embed_fn; auto-disabled when those are unavailable."
        ),
    )
    similarity_threshold: float = Field(default=0.95)
    l1_max_size: int = Field(default=1000)
    default_ttl: float = Field(default=3600.0)


class SecuritySettings(BaseSettings):
    """v0.6 §3.5 prompt injection + PII masking."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_SECURITY_", env_file=".env", extra="ignore"
    )

    injection_detection: bool = Field(
        default=False,
        description=(
            "Scan invoke params for prompt injection; reject with 400. "
            "Disabled by default to avoid false positives in tests."
        ),
    )
    injection_threshold: float = Field(
        default=0.7, description="Confidence threshold for injection rejection"
    )
    pii_masking: bool = Field(
        default=False,
        description=(
            "Mask PII in invoke params before sending to LLM, then unmask "
            "in the response. Disabled by default."
        ),
    )
    pii_enabled_categories: str = Field(
        default="",
        description=(
            "Comma-separated PII categories to enable (empty = all). "
            "Categories: phone, id_card, email, api_key, bank_card"
        ),
    )


class TriggersSettings(BaseSettings):
    """v0.6 §2.5 message queue triggers."""

    model_config = SettingsConfigDict(
        env_prefix="ICORE_TRIGGERS_", env_file=".env", extra="ignore"
    )

    enabled: bool = Field(
        default=False,
        description=(
            "Build + start TriggerManager from config/triggers.yaml. "
            "Disabled by default; enable only when triggers.yaml exists."
        ),
    )
    config_file: str = Field(
        default="triggers.yaml",
        description="YAML file under config_dir listing triggers",
    )


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Top-level icore application settings.

    Aggregates all sub-settings into a single configuration object.
    Loaded once and cached via ``get_settings()``.
    """

    model_config = SettingsConfigDict(
        env_prefix="ICORE_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = Field(default="icore", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    config_dir: str = Field(
        default=str(Path(__file__).resolve().parent.parent / "config"),
        description="YAML config directory",
    )
    workflow_modules: list[str] = Field(
        default_factory=lambda: ["icore.workflows.examples"],
        description=(
            "Modules to import at startup so their @register_task / "
            "@register_workflow decorators run. Override via "
            "ICORE_WORKFLOW_MODULES (comma-separated)."
        ),
    )
    autoregister_workflows: bool = Field(
        default=True,
        description="Import workflow_modules at startup to populate registries",
    )

    # Sub-settings
    api: APISettings = Field(default_factory=APISettings)
    concurrency: ConcurrencySettings = Field(default_factory=ConcurrencySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    # v0.6 engineering enhancements
    backpressure: BackpressureSettings = Field(default_factory=BackpressureSettings)
    hot_reload: HotReloadSettings = Field(default_factory=HotReloadSettings)
    degradation: DegradationSettings = Field(default_factory=DegradationSettings)
    # v0.6 module wirings (all default OFF / InMemory so existing tests stay green)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    persistence: PersistenceSettings = Field(default_factory=PersistenceSettings)
    dlq: DLQSettings = Field(default_factory=DLQSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    triggers: TriggersSettings = Field(default_factory=TriggersSettings)

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> Dict[str, Any]:
        """
        Load a YAML config file and return it as a dict.

        This is a helper for merging YAML-based configuration on top of
        environment-variable-based configuration. Returns an empty dict
        if the file does not exist or PyYAML is not installed.
        """
        path = Path(yaml_path)
        if not path.exists():
            return {}
        try:
            import yaml  # type: ignore
        except ImportError:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the cached singleton Settings instance.

    Configuration precedence (highest priority last):
        1. Built-in defaults
        2. .env file
        3. Environment variables (ICORE_* prefix)

    Returns:
        Settings: The global settings instance.
    """
    return Settings()
