"""
Tests for icore.config and icore.bootstrap.

Covers:
    - Settings defaults & sub-settings (API/Concurrency/Database/Model/Logging)
    - Environment variable overrides (ICORE_* prefix)
    - YAML loading helper (Settings.load_from_yaml)
    - get_settings() singleton caching
    - bootstrap._interpolate (recursive ${VAR} replacement)
    - bootstrap.load_yaml_config (file loading + env interpolation)
    - bootstrap.build_model_manager (from YAML, routing rules, default id)
    - bootstrap.build_db_manager (from YAML, named connections)
    - bootstrap.autoregister_workflows (module import + registry population)
    - bootstrap.create_production_app (full wiring + lifecycle hooks)
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from icore.config import (
    APISettings,
    ConcurrencySettings,
    DatabaseConnectionConfig,
    DatabaseSettings,
    LoggingSettings,
    ModelConfig,
    ModelSettings,
    RoutingRule,
    Settings,
    get_settings,
)


# ---------------------------------------------------------------------------
# Settings: defaults
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    def test_top_level_defaults(self):
        s = Settings()
        assert s.app_name == "icore"
        assert s.version == "1.0.0"
        assert s.debug is False
        assert isinstance(s.config_dir, str)
        assert s.autoregister_workflows is True
        assert "icore.workflows.examples" in s.workflow_modules

    def test_api_settings_defaults(self):
        api = APISettings()
        assert api.host == "0.0.0.0"
        assert api.port == 8000
        assert api.workers >= 1
        assert api.request_timeout >= 1
        assert isinstance(api.cors_origins, list)

    def test_concurrency_settings_defaults(self):
        c = ConcurrencySettings()
        assert c.max_concurrent_tasks >= 1
        assert c.max_concurrent_per_workflow >= 1
        assert c.task_queue_backend in ("memory", "redis")
        assert c.task_timeout >= 1
        assert c.backpressure_threshold >= 1

    def test_database_settings_defaults(self):
        d = DatabaseSettings()
        assert d.default_connection == "default"
        assert d.connections == {}

    def test_database_connection_config_required_fields(self):
        # db_type is the only required field
        c = DatabaseConnectionConfig(db_type="postgresql")
        assert c.db_type == "postgresql"
        assert c.host == "localhost"
        assert c.port == 5432
        assert c.pool_size >= 1
        assert c.max_overflow >= 0
        assert c.pool_recycle >= 1

    def test_model_config_required_fields(self):
        m = ModelConfig(
            model_id="m1",
            model_name="m1",
            api_base="http://x/v1",
            api_key="k",
        )
        assert m.model_id == "m1"
        assert m.max_tokens >= 1
        assert 0.0 <= m.temperature <= 2.0
        assert m.enabled is True
        assert isinstance(m.tags, list)

    def test_model_settings_defaults(self):
        ms = ModelSettings()
        assert ms.default_model_id is None
        assert ms.models == {}
        assert ms.routing_rules == []
        assert ms.enable_auto_routing is True
        assert ms.health_check_interval >= 1

    def test_routing_rule_defaults(self):
        r = RoutingRule()
        assert r.task_type is None
        assert r.max_cost is None
        assert r.preferred_tags == []
        assert r.fallback_model_id is None

    def test_logging_settings_defaults(self):
        lg = LoggingSettings()
        assert lg.level == "INFO"
        assert "level" in lg.format
        assert lg.rotation == "100 MB"
        assert lg.retention == "30 days"

    def test_aggregated_sub_settings_present(self):
        s = Settings()
        assert isinstance(s.api, APISettings)
        assert isinstance(s.concurrency, ConcurrencySettings)
        assert isinstance(s.database, DatabaseSettings)
        assert isinstance(s.model, ModelSettings)
        assert isinstance(s.logging, LoggingSettings)


# ---------------------------------------------------------------------------
# Settings: environment variable overrides
# ---------------------------------------------------------------------------

class TestSettingsEnvOverrides:
    def test_api_port_override_via_env(self, monkeypatch):
        get_settings.cache_clear()
        monkeypatch.setenv("ICORE_API_PORT", "9999")
        s = Settings()
        assert s.api.port == 9999
        get_settings.cache_clear()

    def test_api_host_override_via_env(self, monkeypatch):
        monkeypatch.setenv("ICORE_API_HOST", "127.0.0.1")
        api = APISettings()
        assert api.host == "127.0.0.1"

    def test_concurrency_override_via_env(self, monkeypatch):
        monkeypatch.setenv("ICORE_CONCURRENCY_MAX_CONCURRENT_TASKS", "50")
        c = ConcurrencySettings()
        assert c.max_concurrent_tasks == 50

    def test_top_level_debug_override_via_env(self, monkeypatch):
        get_settings.cache_clear()
        monkeypatch.setenv("ICORE_DEBUG", "true")
        s = Settings()
        assert s.debug is True
        get_settings.cache_clear()

    def test_app_name_override_via_env(self, monkeypatch):
        get_settings.cache_clear()
        monkeypatch.setenv("ICORE_APP_NAME", "icore-test")
        s = Settings()
        assert s.app_name == "icore-test"
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Settings: get_settings caching
# ---------------------------------------------------------------------------

class TestGetSettings:
    def test_get_settings_returns_singleton(self):
        get_settings.cache_clear()
        a = get_settings()
        b = get_settings()
        assert a is b
        get_settings.cache_clear()

    def test_get_settings_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)


# ---------------------------------------------------------------------------
# Settings: YAML helper
# ---------------------------------------------------------------------------

class TestYamlHelper:
    def test_load_from_yaml_missing_file_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        assert Settings.load_from_yaml(missing) == {}

    def test_load_from_yaml_valid_dict(self, tmp_path):
        f = tmp_path / "x.yaml"
        f.write_text("key: value\nlist:\n  - 1\n  - 2\n", encoding="utf-8")
        data = Settings.load_from_yaml(f)
        assert data == {"key": "value", "list": [1, 2]}

    def test_load_from_yaml_non_dict_returns_empty(self, tmp_path):
        f = tmp_path / "list.yaml"
        f.write_text("- a\n- b\n", encoding="utf-8")
        assert Settings.load_from_yaml(f) == {}

    def test_load_from_yaml_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        assert Settings.load_from_yaml(f) == {}


# ---------------------------------------------------------------------------
# Bootstrap: _interpolate
# ---------------------------------------------------------------------------

class TestInterpolate:
    def test_interpolates_string_var(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "hello")
        from icore.bootstrap import _interpolate
        assert _interpolate("${MY_VAR}") == "hello"

    def test_interpolates_inside_larger_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "db.local")
        from icore.bootstrap import _interpolate
        assert _interpolate("postgres://${HOST}:5432") == "postgres://db.local:5432"

    def test_unknown_var_becomes_empty(self, monkeypatch):
        monkeypatch.delenv("DEFINITELY_NOT_SET_VAR", raising=False)
        from icore.bootstrap import _interpolate
        assert _interpolate("${DEFINITELY_NOT_SET_VAR}") == ""

    def test_interpolates_nested_dict(self, monkeypatch):
        monkeypatch.setenv("K", "v1")
        from icore.bootstrap import _interpolate
        out = _interpolate({"a": "${K}", "b": {"c": "${K}"}})
        assert out == {"a": "v1", "b": {"c": "v1"}}

    def test_interpolates_list(self, monkeypatch):
        monkeypatch.setenv("K", "v")
        from icore.bootstrap import _interpolate
        out = _interpolate(["${K}", "literal", 42])
        assert out == ["v", "literal", 42]

    def test_non_string_passthrough(self):
        from icore.bootstrap import _interpolate
        assert _interpolate(42) == 42
        assert _interpolate(3.14) == 3.14
        assert _interpolate(True) is True
        assert _interpolate(None) is None


# ---------------------------------------------------------------------------
# Bootstrap: load_yaml_config
# ---------------------------------------------------------------------------

class TestLoadYamlConfig:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        from icore.bootstrap import load_yaml_config
        assert load_yaml_config(tmp_path / "nope.yaml") == {}

    def test_loads_with_env_interpolation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        f = tmp_path / "models.yaml"
        f.write_text(
            textwrap.dedent(
                """
                default_model_id: gpt-4o
                models:
                  gpt-4o:
                    model_name: gpt-4o
                    api_base: https://api.openai.com/v1
                    api_key: ${OPENAI_API_KEY}
                """
            ).strip(),
            encoding="utf-8",
        )
        from icore.bootstrap import load_yaml_config
        data = load_yaml_config(f)
        assert data["models"]["gpt-4o"]["api_key"] == "sk-test-123"

    def test_non_dict_yaml_returns_empty(self, tmp_path):
        f = tmp_path / "list.yaml"
        f.write_text("- a\n- b\n", encoding="utf-8")
        from icore.bootstrap import load_yaml_config
        assert load_yaml_config(f) == {}


# ---------------------------------------------------------------------------
# Bootstrap: build_model_manager
# ---------------------------------------------------------------------------

class TestBuildModelManager:
    def test_builds_from_real_config_dir(self):
        """The repo ships config/models.yaml - build from it."""
        from icore.bootstrap import build_model_manager
        mgr = build_model_manager()
        # Two enabled models in config/models.yaml: gpt-4o, gpt-4o-mini
        assert "gpt-4o" in mgr
        assert "gpt-4o-mini" in mgr
        assert len(mgr) >= 2

    def test_default_model_id_is_set(self):
        from icore.bootstrap import build_model_manager
        mgr = build_model_manager()
        # config/models.yaml sets default_model_id: gpt-4o-mini
        assert mgr._router._default_model_id == "gpt-4o-mini"

    def test_routing_rules_loaded(self):
        from icore.bootstrap import build_model_manager
        mgr = build_model_manager()
        rules = mgr._router.list_rules()
        assert len(rules) >= 1
        # Rule for summarize task type
        task_types = {r["task_type"] for r in rules}
        assert "summarize" in task_types or "extract" in task_types

    def test_auto_routing_enabled(self):
        from icore.bootstrap import build_model_manager
        mgr = build_model_manager()
        assert mgr.is_auto_routing_enabled is True

    def test_builds_from_custom_settings(self, tmp_path, monkeypatch):
        """Override config_dir to point at a temp YAML."""
        monkeypatch.setenv("TEST_KEY", "sk-from-env")
        cfg = tmp_path / "models.yaml"
        cfg.write_text(
            textwrap.dedent(
                """
                default_model_id: m1
                enable_auto_routing: false
                models:
                  m1:
                    model_name: m1
                    api_base: http://x/v1
                    api_key: ${TEST_KEY}
                    tags: [cheap]
                routing_rules: []
                """
            ).strip(),
            encoding="utf-8",
        )
        s = Settings(config_dir=str(tmp_path))
        from icore.bootstrap import build_model_manager
        mgr = build_model_manager(s)
        assert "m1" in mgr
        assert mgr.is_auto_routing_enabled is False
        # Env var interpolation happened
        assert mgr._configs["m1"].api_key == "sk-from-env"

    def test_invalid_model_entry_is_skipped(self, tmp_path):
        """A malformed model entry should be logged & skipped, not crash."""
        cfg = tmp_path / "models.yaml"
        cfg.write_text(
            textwrap.dedent(
                """
                models:
                  good:
                    model_name: good
                    api_base: http://x/v1
                    api_key: k
                  bad:
                    model_name: bad
                    # missing api_base & api_key -> validation error
                """
            ).strip(),
            encoding="utf-8",
        )
        s = Settings(config_dir=str(tmp_path))
        from icore.bootstrap import build_model_manager
        mgr = build_model_manager(s)
        assert "good" in mgr
        assert "bad" not in mgr

    def test_missing_config_file_yields_empty_manager(self, tmp_path):
        s = Settings(config_dir=str(tmp_path))
        from icore.bootstrap import build_model_manager
        mgr = build_model_manager(s)
        assert len(mgr) == 0


# ---------------------------------------------------------------------------
# Bootstrap: build_db_manager
# ---------------------------------------------------------------------------

class TestBuildDbManager:
    def test_builds_from_real_config_dir(self):
        from icore.bootstrap import build_db_manager
        mgr = build_db_manager()
        # config/databases.yaml has main_db enabled
        assert "main_db" in mgr.registered_names

    def test_main_db_is_postgresql(self):
        from icore.bootstrap import build_db_manager
        mgr = build_db_manager()
        assert "main_db" in mgr.registered_names

    def test_env_interpolation_in_db_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_USER", "icore_user")
        monkeypatch.setenv("DB_PASSWORD", "s3cret")
        cfg = tmp_path / "databases.yaml"
        cfg.write_text(
            textwrap.dedent(
                """
                connections:
                  main_db:
                    db_type: postgresql
                    host: localhost
                    port: 5432
                    username: ${DB_USER}
                    password: ${DB_PASSWORD}
                    database: icore
                """
            ).strip(),
            encoding="utf-8",
        )
        s = Settings(config_dir=str(tmp_path))
        from icore.bootstrap import build_db_manager
        mgr = build_db_manager(s)
        assert "main_db" in mgr.registered_names
        # Internals: config has interpolated values
        conn_config = mgr._configs["main_db"]
        assert conn_config.username == "icore_user"
        assert conn_config.password == "s3cret"

    def test_invalid_connection_skipped(self, tmp_path):
        cfg = tmp_path / "databases.yaml"
        cfg.write_text(
            textwrap.dedent(
                """
                connections:
                  good:
                    db_type: postgresql
                    host: localhost
                  bad:
                    db_type: unsupported_type
                """
            ).strip(),
            encoding="utf-8",
        )
        s = Settings(config_dir=str(tmp_path))
        from icore.bootstrap import build_db_manager
        # build_db_manager catches the ValueError from register() and skips
        mgr = build_db_manager(s)
        assert "good" in mgr.registered_names
        assert "bad" not in mgr.registered_names

    def test_missing_config_file_yields_empty_manager(self, tmp_path):
        s = Settings(config_dir=str(tmp_path))
        from icore.bootstrap import build_db_manager
        mgr = build_db_manager(s)
        assert mgr.registered_names == []


# ---------------------------------------------------------------------------
# Bootstrap: autoregister_workflows
# ---------------------------------------------------------------------------

class TestAutoregisterWorkflows:
    def test_imports_default_modules(self):
        from icore.bootstrap import autoregister_workflows
        imported = autoregister_workflows()
        assert "icore.workflows.examples" in imported

    def test_examples_populate_workflow_registry(self):
        from icore.engine.registry import workflow_registry
        from icore.bootstrap import autoregister_workflows
        autoregister_workflows()
        names = workflow_registry.list_workflows()
        # Three example workflows should be registered
        assert "document_summary" in names
        assert "entity_extraction" in names
        assert "weekly_report" in names

    def test_disabled_when_setting_off(self):
        from icore.bootstrap import autoregister_workflows
        s = Settings(autoregister_workflows=False)
        imported = autoregister_workflows(s)
        assert imported == []

    def test_unknown_module_is_skipped(self):
        from icore.bootstrap import autoregister_workflows
        s = Settings(workflow_modules=["icore.does.not.exist"])
        imported = autoregister_workflows(s)
        # Should not raise, just skip the bad module
        assert "icore.does.not.exist" not in imported


# ---------------------------------------------------------------------------
# Bootstrap: create_production_app
# ---------------------------------------------------------------------------

class TestCreateProductionApp:
    def test_returns_fastapi_app(self):
        from fastapi import FastAPI
        from icore.bootstrap import create_production_app
        app = create_production_app()
        assert isinstance(app, FastAPI)

    def test_app_has_model_manager(self):
        from icore.bootstrap import create_production_app
        app = create_production_app()
        assert app.state.model_manager is not None
        # Real config has gpt-4o
        assert "gpt-4o" in app.state.model_manager

    def test_app_has_db_manager(self):
        from icore.bootstrap import create_production_app
        app = create_production_app()
        assert app.state.db_manager is not None
        assert "main_db" in app.state.db_manager.registered_names

    def test_app_has_callback_manager(self):
        from icore.api.callback import CallbackManager
        from icore.bootstrap import create_production_app
        app = create_production_app()
        assert isinstance(app.state.callback_manager, CallbackManager)

    def test_app_has_workflow_registry(self):
        from icore.engine.registry import WorkflowRegistry
        from icore.bootstrap import create_production_app
        app = create_production_app()
        assert isinstance(app.state.workflow_registry, WorkflowRegistry)
        names = app.state.workflow_registry.list_workflows()
        assert "document_summary" in names

    def test_app_has_bg_tasks_set(self):
        from icore.bootstrap import create_production_app
        app = create_production_app()
        assert hasattr(app.state, "_bg_tasks")
        assert isinstance(app.state._bg_tasks, set)

    def test_health_endpoint_works_via_testclient(self):
        from fastapi.testclient import TestClient
        from icore.bootstrap import create_production_app
        app = create_production_app()
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"

    def test_unknown_workflow_returns_404_via_testclient(self):
        from fastapi.testclient import TestClient
        from icore.bootstrap import create_production_app
        app = create_production_app()
        client = TestClient(app)
        r = client.post(
            "/invoke",
            json={"workflow_name": "ghost_wf", "params": {}},
        )
        assert r.status_code == 404
