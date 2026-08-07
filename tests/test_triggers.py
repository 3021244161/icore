"""
Tests for icore.triggers - message-queue-driven workflow triggers (v0.6).

Coverage:
    - TriggerConfig: defaults, effective_name derivation, extra-field tolerance
    - WorkflowTrigger._decode_message: dict / bytes / str / JSON / invalid
    - InMemoryTrigger: full lifecycle, message processing, result publish,
      multiple messages, parallel triggers, error handling (unknown workflow,
      workflow exception), idempotent start/stop, params merge
    - KafkaTrigger: lazy-import failure graceful degradation, consumer start
      failure no-raise, config propagation, stop-when-not-started
    - RabbitMQTrigger: lazy-import failure, connection failure no-raise,
      config propagation, stop-when-not-started
    - TriggerManager: register / list / get / start_all / stop_all,
      disabled trigger exclusion, YAML load (valid / missing / invalid entry),
      unknown-type rejection, is_running property

All tests are offline (no real Kafka / RabbitMQ broker). aiokafka / aio_pika
are simulated via ``sys.modules`` patching where needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from icore.core.models import BaseTaskOutput
from icore.core.task_context import TaskContext
from icore.triggers import (
    InMemoryTrigger,
    KafkaTrigger,
    RabbitMQTrigger,
    TriggerConfig,
    TriggerManager,
    WorkflowTrigger,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _FakeWorkflow:
    """Records every execute() call and returns a success output.

    ``__call__`` makes this instance behave like a workflow *class*: the
    trigger's ``_invoke_workflow`` does ``wf_cls = registry.get(name)``
    then ``wf = wf_cls()``. By returning ``self`` from ``__call__``, the
    same instance is used (and its ``calls`` list is observable by tests).
    """

    def __init__(self, name: str = "test_wf") -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def __call__(self) -> "_FakeWorkflow":
        return self

    async def execute(
        self, ctx: TaskContext, params: dict[str, Any]
    ) -> BaseTaskOutput:
        self.calls.append({"ctx": ctx, "params": params})
        return BaseTaskOutput.success(
            workflow=self.name,
            received=params,
        )


class _FailingWorkflow:
    """Always raises on execute()."""

    def __call__(self) -> "_FailingWorkflow":
        return self

    async def execute(
        self, ctx: TaskContext, params: dict[str, Any]
    ) -> BaseTaskOutput:
        raise RuntimeError("workflow exploded")


class _FakeWorkflowRegistry:
    """Minimal registry: get(name) returns a stored workflow or raises KeyError."""

    def __init__(self) -> None:
        self._workflows: dict[str, Any] = {}

    def register(self, name: str, wf: Any) -> None:
        self._workflows[name] = wf

    def get(self, name: str) -> Any:
        if name not in self._workflows:
            raise KeyError(f"Workflow '{name}' is not registered")
        return self._workflows[name]

    def list_workflows(self) -> list[str]:
        return sorted(self._workflows.keys())


def _make_registry(*workflows: tuple[str, Any]) -> _FakeWorkflowRegistry:
    reg = _FakeWorkflowRegistry()
    for name, wf in workflows:
        reg.register(name, wf)
    return reg


def _inmem_config(**overrides: Any) -> TriggerConfig:
    defaults: dict[str, Any] = {
        "type": "inmemory",
        "workflow": "test_wf",
        "name": "test_trig",
    }
    defaults.update(overrides)
    return TriggerConfig(**defaults)


def _kafka_config(**overrides: Any) -> TriggerConfig:
    defaults: dict[str, Any] = {
        "type": "kafka",
        "workflow": "test_wf",
        "name": "kafka_trig",
        "topic": "test_topic",
        "brokers": ["localhost:9092"],
        "group_id": "test-group",
    }
    defaults.update(overrides)
    return TriggerConfig(**defaults)


def _rabbit_config(**overrides: Any) -> TriggerConfig:
    defaults: dict[str, Any] = {
        "type": "rabbitmq",
        "workflow": "test_wf",
        "name": "rmq_trig",
        "amqp_url": "amqp://guest:guest@localhost/",
        "queue_name": "test_queue",
    }
    defaults.update(overrides)
    return TriggerConfig(**defaults)


# ---------------------------------------------------------------------------
# TriggerConfig
# ---------------------------------------------------------------------------

class TestTriggerConfig:
    def test_defaults(self) -> None:
        cfg = TriggerConfig(type="inmemory", workflow="wf")
        assert cfg.topic == ""
        assert cfg.group_id == "icore-trigger-group"
        assert cfg.brokers == []
        assert cfg.amqp_url == ""
        assert cfg.queue_name == ""
        assert cfg.exchange == ""
        assert cfg.params == {}
        assert cfg.result_topic == ""
        assert cfg.enabled is True
        assert cfg.auto_commit is True
        assert cfg.name == ""

    def test_effective_name_explicit(self) -> None:
        cfg = TriggerConfig(
            type="kafka", workflow="wf", name="my_trigger"
        )
        assert cfg.effective_name() == "my_trigger"

    def test_effective_name_derived_from_topic(self) -> None:
        cfg = TriggerConfig(type="kafka", workflow="fraud", topic="orders")
        assert cfg.effective_name() == "fraud:kafka:orders"

    def test_effective_name_derived_from_queue(self) -> None:
        cfg = TriggerConfig(
            type="rabbitmq", workflow="docs", queue_name="doc_q"
        )
        assert cfg.effective_name() == "docs:rabbitmq:doc_q"

    def test_effective_name_falls_back_to_workflow(self) -> None:
        cfg = TriggerConfig(type="inmemory", workflow="wf")
        assert cfg.effective_name() == "wf:inmemory:wf"

    def test_extra_fields_ignored(self) -> None:
        # Unknown YAML keys should be silently dropped, not raise.
        cfg = TriggerConfig(
            type="inmemory", workflow="wf", unknown_field="ignored"  # type: ignore[call-arg]
        )
        assert cfg.workflow == "wf"


# ---------------------------------------------------------------------------
# WorkflowTrigger._decode_message
# ---------------------------------------------------------------------------

class TestDecodeMessage:
    def test_dict_passthrough_copied(self) -> None:
        original = {"a": 1}
        decoded = WorkflowTrigger._decode_message(original)
        assert decoded == original
        assert decoded is not original  # must be a copy

    def test_none_returns_empty(self) -> None:
        assert WorkflowTrigger._decode_message(None) == {}

    def test_bytes_valid_json(self) -> None:
        decoded = WorkflowTrigger._decode_message(b'{"key": "value"}')
        assert decoded == {"key": "value"}

    def test_str_valid_json(self) -> None:
        decoded = WorkflowTrigger._decode_message('{"x": 42}')
        assert decoded == {"x": 42}

    def test_invalid_json_wrapped_as_payload(self) -> None:
        decoded = WorkflowTrigger._decode_message("not json at all")
        assert decoded == {"payload": "not json at all"}

    def test_bytes_invalid_utf8_fallback(self) -> None:
        decoded = WorkflowTrigger._decode_message(b"\xff\xfe not utf8")
        # Should not raise; falls back to {"payload": ...}
        assert "payload" in decoded

    def test_json_array_wrapped(self) -> None:
        decoded = WorkflowTrigger._decode_message("[1, 2, 3]")
        assert decoded == {"payload": [1, 2, 3]}

    def test_empty_string_returns_empty(self) -> None:
        assert WorkflowTrigger._decode_message("") == {}
        assert WorkflowTrigger._decode_message("   ") == {}


# ---------------------------------------------------------------------------
# InMemoryTrigger
# ---------------------------------------------------------------------------

class TestInMemoryTrigger:
    @pytest.mark.asyncio
    async def test_basic_invocation(self) -> None:
        wf = _FakeWorkflow()
        reg = _make_registry(("test_wf", wf))
        trig = InMemoryTrigger(_inmem_config(), reg)

        await trig.start()
        assert trig.is_running is True

        await trig.push({"hello": "world"})
        await trig._queue.join()

        assert len(wf.calls) == 1
        assert wf.calls[0]["params"]["hello"] == "world"
        # TaskContext metadata should record the trigger source.
        ctx: TaskContext = wf.calls[0]["ctx"]
        assert ctx.metadata["source"] == "trigger"
        assert ctx.metadata["trigger_type"] == "inmemory"

        await trig.stop()

    @pytest.mark.asyncio
    async def test_result_topic_publishes(self) -> None:
        wf = _FakeWorkflow()
        reg = _make_registry(("test_wf", wf))
        config = _inmem_config(result_topic="results")
        trig = InMemoryTrigger(config, reg)

        await trig.start()
        await trig.push({"x": 1})
        await trig._queue.join()

        # _publish_result should have recorded the BaseTaskOutput dump.
        assert len(trig._published_results) == 1
        published = trig._published_results[0]
        assert isinstance(published, dict)
        assert published["status"] == "success"
        assert "received" in published["data"]

        await trig.stop()

    @pytest.mark.asyncio
    async def test_no_result_topic_no_publish(self) -> None:
        wf = _FakeWorkflow()
        reg = _make_registry(("test_wf", wf))
        trig = InMemoryTrigger(_inmem_config(), reg)  # no result_topic

        await trig.start()
        await trig.push({"x": 1})
        await trig._queue.join()

        assert trig._published_results == []
        await trig.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self) -> None:
        trig = InMemoryTrigger(
            _inmem_config(), _make_registry(("test_wf", _FakeWorkflow()))
        )
        await trig.start()
        assert trig.is_running is True
        await trig.stop()
        assert trig.is_running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self) -> None:
        trig = InMemoryTrigger(
            _inmem_config(), _make_registry(("test_wf", _FakeWorkflow()))
        )
        await trig.start()
        task1 = trig._consume_task
        await trig.start()  # second start is a no-op
        assert trig._consume_task is task1
        await trig.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent_when_not_started(self) -> None:
        trig = InMemoryTrigger(
            _inmem_config(), _make_registry(("test_wf", _FakeWorkflow()))
        )
        # stop() before start() should not raise.
        await trig.stop()
        assert trig.is_running is False

    @pytest.mark.asyncio
    async def test_multiple_messages(self) -> None:
        wf = _FakeWorkflow()
        reg = _make_registry(("test_wf", wf))
        trig = InMemoryTrigger(_inmem_config(), reg)

        await trig.start()
        for i in range(5):
            await trig.push({"index": i})
        await trig._queue.join()

        assert len(wf.calls) == 5
        for i, call in enumerate(wf.calls):
            assert call["params"]["index"] == i

        await trig.stop()

    @pytest.mark.asyncio
    async def test_workflow_not_registered_continues(self) -> None:
        # Empty registry: workflow lookup will raise KeyError.
        reg = _FakeWorkflowRegistry()
        trig = InMemoryTrigger(_inmem_config(), reg)

        await trig.start()
        await trig.push({"x": 1})
        await trig._queue.join()

        # Consume loop should have swallowed the error and kept running.
        assert trig.is_running is True
        await trig.stop()

    @pytest.mark.asyncio
    async def test_workflow_exception_continues(self) -> None:
        reg = _make_registry(("test_wf", _FailingWorkflow()))
        trig = InMemoryTrigger(_inmem_config(), reg)

        await trig.start()
        await trig.push({"x": 1})
        await trig._queue.join()

        # The consume loop must not crash; trigger stays running.
        assert trig.is_running is True
        await trig.stop()

    @pytest.mark.asyncio
    async def test_params_merged_with_message(self) -> None:
        wf = _FakeWorkflow()
        reg = _make_registry(("test_wf", wf))
        config = _inmem_config(params={"threshold": 0.8, "region": "us"})
        trig = InMemoryTrigger(config, reg)

        await trig.start()
        # Message overrides "region" but keeps "threshold".
        await trig.push({"region": "eu", "order_id": 42})
        await trig._queue.join()

        params = wf.calls[0]["params"]
        assert params["threshold"] == 0.8  # from config.params
        assert params["region"] == "eu"    # from message (overrides)
        assert params["order_id"] == 42     # from message

        await trig.stop()

    @pytest.mark.asyncio
    async def test_multiple_triggers_parallel(self) -> None:
        wf1 = _FakeWorkflow(name="wf1")
        wf2 = _FakeWorkflow(name="wf2")
        reg = _make_registry(("wf1", wf1), ("wf2", wf2))

        t1 = InMemoryTrigger(
            TriggerConfig(
                type="inmemory", workflow="wf1", name="t1"
            ),
            reg,
        )
        t2 = InMemoryTrigger(
            TriggerConfig(
                type="inmemory", workflow="wf2", name="t2"
            ),
            reg,
        )

        await t1.start()
        await t2.start()

        await t1.push({"n": 1})
        await t2.push({"n": 2})

        await t1._queue.join()
        await t2._queue.join()

        assert len(wf1.calls) == 1
        assert len(wf2.calls) == 1
        assert wf1.calls[0]["params"]["n"] == 1
        assert wf2.calls[0]["params"]["n"] == 2

        await t1.stop()
        await t2.stop()


# ---------------------------------------------------------------------------
# KafkaTrigger
# ---------------------------------------------------------------------------

class TestKafkaTrigger:
    @pytest.mark.asyncio
    async def test_lazy_import_failure(self) -> None:
        """aiokafka not installed → start() logs warning, no raise."""
        trig = KafkaTrigger(_kafka_config(), _FakeWorkflowRegistry())
        # Make aiokafka appear uninstalled.
        with patch.dict(sys.modules, {"aiokafka": None}):
            await trig.start()
        assert trig.is_running is False
        # No resources should be left dangling.
        assert trig._consumer is None

    @pytest.mark.asyncio
    async def test_consumer_start_failure_no_raise(self) -> None:
        """When consumer.start() fails, start() should log warning, not raise."""

        class _FakeConsumer:
            def __init__(self, *a, **kw) -> None:
                pass

            async def start(self) -> None:
                raise OSError("broker unreachable")

            async def stop(self) -> None:
                pass

        class _FakeProducer:
            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        fake_mod = types.ModuleType("aiokafka")
        fake_mod.AIOKafkaConsumer = _FakeConsumer  # type: ignore[attr-defined]
        fake_mod.AIOKafkaProducer = _FakeProducer  # type: ignore[attr-defined]

        trig = KafkaTrigger(_kafka_config(), _FakeWorkflowRegistry())
        with patch.dict(sys.modules, {"aiokafka": fake_mod}):
            await trig.start()
        assert trig.is_running is False
        assert trig._consumer is None

    @pytest.mark.asyncio
    async def test_config_passed_to_consumer(self) -> None:
        """Verify that config values reach the AIOKafkaConsumer constructor."""
        captured: dict[str, Any] = {}

        class _FakeConsumer:
            def __init__(self, topic, **kw) -> None:
                captured["topic"] = topic
                captured.update(kw)

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        class _FakeProducer:
            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        fake_mod = types.ModuleType("aiokafka")
        fake_mod.AIOKafkaConsumer = _FakeConsumer  # type: ignore[attr-defined]
        fake_mod.AIOKafkaProducer = _FakeProducer  # type: ignore[attr-defined]

        config = _kafka_config(
            topic="my_topic",
            brokers=["broker1:9092", "broker2:9092"],
            group_id="my-group",
            auto_commit=False,
        )
        trig = KafkaTrigger(config, _FakeWorkflowRegistry())
        with patch.dict(sys.modules, {"aiokafka": fake_mod}):
            await trig.start()

        assert captured["topic"] == "my_topic"
        assert captured["group_id"] == "my-group"
        assert captured["bootstrap_servers"] == ["broker1:9092", "broker2:9092"]
        assert captured["enable_auto_commit"] is False
        assert trig.is_running is True

        await trig.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self) -> None:
        """stop() before start() should be a safe no-op."""
        trig = KafkaTrigger(_kafka_config(), _FakeWorkflowRegistry())
        await trig.stop()  # should not raise
        assert trig.is_running is False

    @pytest.mark.asyncio
    async def test_full_start_stop_cycle(self) -> None:
        """Verify start/stop lifecycle with a fake consumer."""

        class _FakeConsumer:
            def __init__(self, *a, **kw) -> None:
                self._stopped = False

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                self._stopped = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                # No messages; raise StopAsyncIteration to end iteration.
                raise StopAsyncIteration

        class _FakeProducer:
            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        fake_mod = types.ModuleType("aiokafka")
        fake_mod.AIOKafkaConsumer = _FakeConsumer  # type: ignore[attr-defined]
        fake_mod.AIOKafkaProducer = _FakeProducer  # type: ignore[attr-defined]

        trig = KafkaTrigger(_kafka_config(), _FakeWorkflowRegistry())
        with patch.dict(sys.modules, {"aiokafka": fake_mod}):
            await trig.start()
            assert trig.is_running is True
            await trig.stop()
        assert trig.is_running is False


# ---------------------------------------------------------------------------
# RabbitMQTrigger
# ---------------------------------------------------------------------------

class TestRabbitMQTrigger:
    @pytest.mark.asyncio
    async def test_lazy_import_failure(self) -> None:
        """aio_pika not installed → start() logs warning, no raise."""
        trig = RabbitMQTrigger(_rabbit_config(), _FakeWorkflowRegistry())
        with patch.dict(sys.modules, {"aio_pika": None}):
            await trig.start()
        assert trig.is_running is False

    @pytest.mark.asyncio
    async def test_connection_failure_no_raise(self) -> None:
        """When connect_robust fails, start() should log warning, not raise."""

        class _FakeChannel:
            async def close(self) -> None:
                pass

        class _FakeConnection:
            async def channel(self) -> _FakeChannel:
                return _FakeChannel()

            async def close(self) -> None:
                pass

        fake_mod = types.ModuleType("aio_pika")

        async def _connect_robust(url: str):
            raise OSError("rabbitmq unreachable")

        fake_mod.connect_robust = _connect_robust  # type: ignore[attr-defined]
        fake_mod.Message = lambda **kw: None  # type: ignore[attr-defined]

        trig = RabbitMQTrigger(_rabbit_config(), _FakeWorkflowRegistry())
        with patch.dict(sys.modules, {"aio_pika": fake_mod}):
            await trig.start()
        assert trig.is_running is False

    @pytest.mark.asyncio
    async def test_config_accessible(self) -> None:
        """Verify config values are stored on the trigger instance."""
        config = _rabbit_config(
            amqp_url="amqp://user:pass@rmq:5672/vhost",
            queue_name="my_queue",
            exchange="my_exchange",
        )
        trig = RabbitMQTrigger(config, _FakeWorkflowRegistry())
        assert trig.config.amqp_url == "amqp://user:pass@rmq:5672/vhost"
        assert trig.config.queue_name == "my_queue"
        assert trig.config.exchange == "my_exchange"

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self) -> None:
        """stop() before start() should be a safe no-op."""
        trig = RabbitMQTrigger(_rabbit_config(), _FakeWorkflowRegistry())
        await trig.stop()  # should not raise
        assert trig.is_running is False

    @pytest.mark.asyncio
    async def test_consume_loop_is_noop(self) -> None:
        """RabbitMQ uses callbacks; _consume_loop should return immediately."""
        trig = RabbitMQTrigger(_rabbit_config(), _FakeWorkflowRegistry())
        # Should return without raising.
        await trig._consume_loop()


# ---------------------------------------------------------------------------
# TriggerManager
# ---------------------------------------------------------------------------

class TestTriggerManager:
    def test_register_creates_trigger(self) -> None:
        mgr = TriggerManager(_FakeWorkflowRegistry())
        trig = mgr.register(_inmem_config())
        assert isinstance(trig, InMemoryTrigger)
        assert "test_trig" in mgr.list_triggers()

    def test_register_unknown_type_raises(self) -> None:
        mgr = TriggerManager(_FakeWorkflowRegistry())
        config = TriggerConfig(type="unknown", workflow="wf", name="bad")
        with pytest.raises(ValueError, match="Unknown trigger type"):
            mgr.register(config)

    def test_get_trigger_returns_none_for_unknown(self) -> None:
        mgr = TriggerManager(_FakeWorkflowRegistry())
        assert mgr.get_trigger("nonexistent") is None

    def test_get_trigger_returns_registered(self) -> None:
        mgr = TriggerManager(_FakeWorkflowRegistry())
        trig = mgr.register(_inmem_config())
        assert mgr.get_trigger("test_trig") is trig

    def test_list_triggers_returns_copy(self) -> None:
        mgr = TriggerManager(_FakeWorkflowRegistry())
        mgr.register(_inmem_config())
        listed = mgr.list_triggers()
        listed["injected"] = _inmem_config()  # type: ignore[assignment]
        # Mutating the returned dict should not affect the manager.
        assert "injected" not in mgr.list_triggers()

    @pytest.mark.asyncio
    async def test_start_all_starts_enabled_only(self) -> None:
        wf = _FakeWorkflow()
        reg = _make_registry(("wf", wf))
        mgr = TriggerManager(reg)

        mgr.register(
            TriggerConfig(
                type="inmemory", workflow="wf", name="enabled_trig",
                enabled=True,
            )
        )
        mgr.register(
            TriggerConfig(
                type="inmemory", workflow="wf", name="disabled_trig",
                enabled=False,
            )
        )

        results = await mgr.start_all()
        assert "enabled_trig" in results
        assert results["enabled_trig"] is True
        assert "disabled_trig" not in results
        assert mgr.get_trigger("disabled_trig").is_running is False
        assert mgr.is_running is True

        await mgr.stop_all()
        assert mgr.is_running is False

    @pytest.mark.asyncio
    async def test_start_all_with_message(self) -> None:
        """Integration: start_all → push → workflow invoked → stop_all."""
        wf = _FakeWorkflow()
        reg = _make_registry(("wf", wf))
        mgr = TriggerManager(reg)
        mgr.register(
            TriggerConfig(
                type="inmemory", workflow="wf", name="t1", enabled=True
            )
        )

        await mgr.start_all()
        trig = mgr.get_trigger("t1")
        assert trig is not None
        await trig.push({"k": "v"})
        await trig._queue.join()

        assert len(wf.calls) == 1
        await mgr.stop_all()

    @pytest.mark.asyncio
    async def test_stop_all_idempotent(self) -> None:
        mgr = TriggerManager(_FakeWorkflowRegistry())
        # stop_all on empty manager should not raise.
        await mgr.stop_all()

    @pytest.mark.asyncio
    async def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
triggers:
  - name: t1
    type: inmemory
    workflow: wf1
    enabled: true
    params:
      threshold: 0.9
  - name: t2
    type: inmemory
    workflow: wf2
    enabled: false
"""
        path = tmp_path / "triggers.yaml"
        path.write_text(yaml_content, encoding="utf-8")

        reg = _make_registry(("wf1", _FakeWorkflow()), ("wf2", _FakeWorkflow()))
        mgr = TriggerManager(reg)
        configs = await mgr.load_from_yaml(path)

        assert len(configs) == 2
        assert configs[0].name == "t1"
        assert configs[0].params["threshold"] == 0.9
        assert configs[1].name == "t2"
        assert "t1" in mgr.list_triggers()
        assert "t2" in mgr.list_triggers()

    @pytest.mark.asyncio
    async def test_load_from_yaml_missing_file(self, tmp_path: Path) -> None:
        mgr = TriggerManager(_FakeWorkflowRegistry())
        configs = await mgr.load_from_yaml(tmp_path / "nonexistent.yaml")
        assert configs == []
        assert mgr.list_triggers() == {}

    @pytest.mark.asyncio
    async def test_load_from_yaml_invalid_entry_skipped(
        self, tmp_path: Path
    ) -> None:
        # Missing required "workflow" field → Pydantic validation error.
        yaml_content = """
triggers:
  - name: good
    type: inmemory
    workflow: wf
  - name: bad
    type: inmemory
    # missing workflow
"""
        path = tmp_path / "triggers.yaml"
        path.write_text(yaml_content, encoding="utf-8")

        reg = _make_registry(("wf", _FakeWorkflow()))
        mgr = TriggerManager(reg)
        configs = await mgr.load_from_yaml(path)

        # Only the valid entry should be registered.
        assert len(configs) == 1
        assert configs[0].name == "good"
        assert "good" in mgr.list_triggers()
        assert "bad" not in mgr.list_triggers()

    @pytest.mark.asyncio
    async def test_load_from_yaml_empty_triggers(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("triggers: []\n", encoding="utf-8")
        mgr = TriggerManager(_FakeWorkflowRegistry())
        configs = await mgr.load_from_yaml(path)
        assert configs == []

    @pytest.mark.asyncio
    async def test_is_running_reflects_start_state(self) -> None:
        reg = _make_registry(("wf", _FakeWorkflow()))
        mgr = TriggerManager(reg)
        mgr.register(
            TriggerConfig(
                type="inmemory", workflow="wf", name="t1", enabled=True
            )
        )
        assert mgr.is_running is False
        await mgr.start_all()
        assert mgr.is_running is True
        await mgr.stop_all()
        assert mgr.is_running is False

    @pytest.mark.asyncio
    async def test_load_from_yaml_real_config_file(self) -> None:
        """Load the real config/triggers.yaml shipped with the project."""
        # The real config references workflows that may not be registered
        # in the test environment, so we use a permissive registry.
        reg = _FakeWorkflowRegistry()
        mgr = TriggerManager(reg)

        # Locate the config file relative to this test.
        config_path = (
            Path(__file__).resolve().parent.parent / "config" / "triggers.yaml"
        )
        if not config_path.exists():
            pytest.skip(f"config/triggers.yaml not found at {config_path}")

        configs = await mgr.load_from_yaml(config_path)
        # The shipped config has 3 triggers (fraud, doc, local_test).
        assert len(configs) == 3
        names = {c.name for c in configs}
        assert "fraud_detection_trigger" in names
        assert "doc_ingest_trigger" in names
        assert "local_test_trigger" in names

        # local_test_trigger is disabled in the shipped config.
        local = [c for c in configs if c.name == "local_test_trigger"][0]
        assert local.enabled is False
