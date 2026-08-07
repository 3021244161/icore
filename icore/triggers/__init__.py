"""
icore.triggers - Message-queue-driven workflow triggers (v0.6).

This module provides the ability to start icore workflows in response to
external events arriving on Kafka / RabbitMQ topics (or an in-memory
asyncio.Queue for testing). It is intentionally decoupled from the API
layer so it can be embedded in any process that hosts a WorkflowRegistry.

Design highlights:
    - Lazy imports: ``aiokafka`` / ``aio_pika`` are imported only when the
      corresponding trigger is started, so the module is importable
      without those packages installed.
    - Graceful degradation: when a driver is missing, ``start()`` logs a
      warning and returns without raising; ``is_running`` stays False.
    - The ``InMemoryTrigger`` is always available (used by tests and for
      local development).
    - ``TriggerManager`` reads ``config/triggers.yaml`` and starts every
      enabled trigger in parallel.

Logging uses the stdlib ``logging`` module (no loguru), per AGENTS.md.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from icore.core.task_context import TaskContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TriggerConfig(BaseModel):
    """
    Configuration for a single message-queue trigger.

    Attributes:
        type:         Trigger type: ``"kafka"``, ``"rabbitmq"``, or
                      ``"inmemory"``.
        workflow:     Name of the workflow to invoke when a message arrives.
        topic:        Kafka topic OR RabbitMQ routing key (type-dependent).
        group_id:     Consumer group ID (Kafka).
        brokers:      Kafka bootstrap servers (e.g. ``["localhost:9092"]``).
        amqp_url:     RabbitMQ AMQP URL
                      (e.g. ``"amqp://guest:guest@localhost/"``).
        queue_name:   RabbitMQ queue name.
        exchange:     RabbitMQ exchange name (empty = default exchange).
        params:       Parameter template merged into each incoming message
                      before being passed to the workflow. Incoming message
                      keys take precedence over template keys.
        result_topic: Optional topic to publish the workflow result to.
        enabled:      Whether this trigger should be started by the manager.
        auto_commit:  Whether to auto-commit offsets (Kafka) / ack (RabbitMQ).
        name:         Optional human-readable trigger name. Defaults to
                      ``"<workflow>:<type>:<topic-or-queue>"`` when not set.
                      Used by the ``TriggerManager`` as the registry key.
    """

    model_config = ConfigDict(extra="ignore")

    type: str = Field(description="Trigger type: kafka | rabbitmq | inmemory")
    workflow: str = Field(description="Workflow name to invoke")
    topic: str = Field(default="", description="Kafka topic / RabbitMQ routing key")
    group_id: str = Field(default="icore-trigger-group")
    brokers: list[str] = Field(default_factory=list)
    amqp_url: str = Field(default="")
    queue_name: str = Field(default="")
    exchange: str = Field(default="")
    params: dict[str, Any] = Field(default_factory=dict)
    result_topic: str = Field(default="")
    enabled: bool = True
    auto_commit: bool = True
    name: str = Field(default="", description="Optional trigger name")

    def effective_name(self) -> str:
        """Return the trigger name, deriving a default when ``name`` is empty."""
        if self.name:
            return self.name
        suffix = self.topic or self.queue_name or self.workflow
        return f"{self.workflow}:{self.type}:{suffix}"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class WorkflowTrigger(ABC):
    """
    Abstract base class for all workflow triggers.

    A trigger subscribes to a message source (Kafka / RabbitMQ / in-memory
    queue), deserialises each message into a dict, and invokes the
    configured workflow via the supplied ``workflow_registry``.

    Subclasses MUST implement ``start`` and ``stop``.
    """

    def __init__(
        self,
        config: TriggerConfig,
        workflow_registry: Any,
        callback_manager: Any = None,
    ) -> None:
        self.config = config
        self.workflow_registry = workflow_registry
        self.callback_manager = callback_manager
        self._is_running = False
        self._consume_task: Optional[asyncio.Task[None]] = None
        # Records published results (useful for tests / observability).
        self._published_results: list[Any] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Start consuming messages. Idempotent: starting twice is a no-op."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Stop consuming and release resources. Idempotent."""
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        """Whether the trigger is currently consuming messages."""
        return self._is_running

    # ------------------------------------------------------------------
    # Workflow invocation
    # ------------------------------------------------------------------

    async def _invoke_workflow(
        self,
        message: dict[str, Any],
        raw_message: bytes | None = None,
    ) -> Any:
        """
        Look up the configured workflow and execute it with ``message``.

        The ``TriggerConfig.params`` dict is merged underneath the incoming
        message (incoming keys win) and passed to ``workflow.execute``.
        The ``TaskContext`` is built with a fresh UUID so each invocation
        is independently traceable. No model/db managers are injected —
        triggers are best suited to pure-compute workflows, or workflows
        that handle missing dependencies themselves.

        Args:
            message:     Parsed message payload (already a dict).
            raw_message:  Original bytes (unused by default; available
                          for subclasses that want to log it).

        Returns:
            The ``BaseTaskOutput`` returned by the workflow.

        Raises:
            KeyError:  If the workflow is not registered (propagates so
                       the consume loop can decide whether to ack/skip).
            Exception: Any exception raised by the workflow propagates
                       so callers can commit/ack/reject appropriately.
        """
        wf_cls = self.workflow_registry.get(self.config.workflow)
        wf = wf_cls()
        merged: dict[str, Any] = {**self.config.params, **message}
        ctx = TaskContext(
            task_id=f"trigger-{uuid.uuid4()}",
            workflow_id=f"trigger-wf-{uuid.uuid4()}",
            metadata={
                "source": "trigger",
                "trigger_type": self.config.type,
                "trigger_name": self.config.effective_name(),
            },
        )
        result = await wf.execute(ctx, merged)
        if self.config.result_topic:
            try:
                await self._publish_result(result)
            except Exception:
                logger.warning(
                    "trigger '%s': failed to publish result to '%s'",
                    self.config.effective_name(),
                    self.config.result_topic,
                    exc_info=True,
                )
        return result

    async def _publish_result(self, result: Any) -> None:
        """
        Publish a workflow result to ``config.result_topic``.

        The default implementation records the result in
        ``self._published_results`` (used by tests and the InMemoryTrigger).
        Kafka / RabbitMQ subclasses override this to actually send to the
        broker.
        """
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        self._published_results.append(payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_message(raw: Any) -> dict[str, Any]:
        """
        Decode a raw message into a dict.

        Accepts:
            - dict: shallow-copied and returned.
            - bytes / bytearray: decoded as UTF-8, then parsed as JSON.
              Falls back to ``{"payload": <text>}`` if not valid JSON.
            - str: parsed as JSON. Falls back to ``{"payload": <text>}``.
            - None: empty dict.

        Returns:
            A dict suitable for passing to ``_invoke_workflow``.
        """
        import json

        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except (ValueError, TypeError):
                return {"payload": raw}
            if isinstance(parsed, dict):
                return parsed
            return {"payload": parsed}
        return {"payload": raw}


# ---------------------------------------------------------------------------
# InMemoryTrigger
# ---------------------------------------------------------------------------

class InMemoryTrigger(WorkflowTrigger):
    """
    In-process trigger backed by an ``asyncio.Queue``.

    Used for testing and local development: tests can ``await push(...)``
    a message and assert the workflow was invoked. No external broker.

    Usage::

        trig = InMemoryTrigger(config, workflow_registry)
        await trig.start()
        await trig.push({"order_id": 42})
        # ... workflow fires asynchronously ...
        await trig.stop()
    """

    _STOP_SENTINEL: dict[str, Any] = {"__icore_trigger_stop__": True}

    def __init__(
        self,
        config: TriggerConfig,
        workflow_registry: Any,
        callback_manager: Any = None,
    ) -> None:
        super().__init__(config, workflow_registry, callback_manager)
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def push(self, message: dict[str, Any]) -> None:
        """
        Push a message onto the in-memory queue.

        If the trigger is running, the consume loop will pick it up and
        invoke the workflow. If the trigger is not running, the message
        is queued and will be processed when ``start()`` is called.
        """
        await self._queue.put(message)

    async def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._consume_task = asyncio.create_task(
            self._consume_loop(),
            name=f"inmem-trigger-{self.config.effective_name()}",
        )
        logger.info(
            "InMemoryTrigger '%s' started (workflow=%s)",
            self.config.effective_name(),
            self.config.workflow,
        )

    async def stop(self) -> None:
        if not self._is_running:
            return
        self._is_running = False
        # Wake up the consume loop if it's blocked on queue.get().
        await self._queue.put(self._STOP_SENTINEL)
        if self._consume_task is not None:
            try:
                await asyncio.wait_for(self._consume_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._consume_task = None
        logger.info(
            "InMemoryTrigger '%s' stopped", self.config.effective_name()
        )

    async def _consume_loop(self) -> None:
        while self._is_running:
            try:
                message = await self._queue.get()
            except asyncio.CancelledError:
                break
            if message is self._STOP_SENTINEL:
                break
            try:
                await self._invoke_workflow(message)
            except Exception:
                logger.warning(
                    "InMemoryTrigger '%s' workflow invocation failed",
                    self.config.effective_name(),
                    exc_info=True,
                )
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# KafkaTrigger
# ---------------------------------------------------------------------------

class KafkaTrigger(WorkflowTrigger):
    """
    Kafka-based trigger using ``aiokafka.AIOKafkaConsumer``.

    The ``aiokafka`` package is imported lazily inside ``start()``; if it
    is not installed, ``start()`` logs a warning and returns without
    raising. ``is_running`` stays ``False`` in that case.
    """

    def __init__(
        self,
        config: TriggerConfig,
        workflow_registry: Any,
        callback_manager: Any = None,
    ) -> None:
        super().__init__(config, workflow_registry, callback_manager)
        self._consumer: Any = None
        self._producer: Any = None

    async def start(self) -> None:
        if self._is_running:
            return
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer  # type: ignore
        except ImportError:
            logger.warning(
                "aiokafka not installed; KafkaTrigger '%s' disabled "
                "(workflow=%s, topic=%s)",
                self.config.effective_name(),
                self.config.workflow,
                self.config.topic,
            )
            return
        brokers = self.config.brokers or ["localhost:9092"]
        self._consumer = AIOKafkaConsumer(
            self.config.topic,
            group_id=self.config.group_id,
            bootstrap_servers=brokers,
            enable_auto_commit=self.config.auto_commit,
            value_deserializer=lambda b: b,
        )
        try:
            await self._consumer.start()
        except Exception:
            logger.warning(
                "KafkaTrigger '%s' failed to start consumer "
                "(brokers=%s, topic=%s)",
                self.config.effective_name(),
                brokers,
                self.config.topic,
                exc_info=True,
            )
            self._consumer = None
            return
        if self.config.result_topic:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=brokers,
                value_serializer=lambda v: v.encode("utf-8")
                if isinstance(v, str)
                else v,
            )
            try:
                await self._producer.start()
            except Exception:
                logger.warning(
                    "KafkaTrigger '%s' failed to start result producer",
                    self.config.effective_name(),
                    exc_info=True,
                )
                self._producer = None
        self._is_running = True
        self._consume_task = asyncio.create_task(
            self._consume_loop(),
            name=f"kafka-trigger-{self.config.effective_name()}",
        )
        logger.info(
            "KafkaTrigger '%s' started (topic=%s, group=%s, brokers=%s)",
            self.config.effective_name(),
            self.config.topic,
            self.config.group_id,
            brokers,
        )

    async def stop(self) -> None:
        was_running = self._is_running
        self._is_running = False
        if self._consume_task is not None:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except (asyncio.CancelledError, Exception):
                pass
            self._consume_task = None
        await self._close_resources()
        if was_running:
            logger.info(
                "KafkaTrigger '%s' stopped", self.config.effective_name()
            )

    async def _close_resources(self) -> None:
        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception:
                pass
            self._consumer = None
        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception:
                pass
            self._producer = None

    async def _consume_loop(self) -> None:
        try:
            async for msg in self._consumer:
                if not self._is_running:
                    break
                try:
                    payload = self._decode_message(msg.value)
                    await self._invoke_workflow(payload, raw_message=msg.value)
                    # 手动提交 offset：仅当 auto_commit=False 时由本触发器
                    # 在工作流执行成功后显式 commit；失败时不 commit，
                    # 下次消费会重新拉取该消息（at-least-once 语义）。
                    if not self.config.auto_commit:
                        try:
                            await self._consumer.commit()
                        except Exception:
                            logger.warning(
                                "KafkaTrigger '%s' manual offset commit failed",
                                self.config.effective_name(),
                                exc_info=True,
                            )
                except Exception:
                    logger.warning(
                        "KafkaTrigger '%s' message handling failed; "
                        "offset not committed (will be re-delivered)",
                        self.config.effective_name(),
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                "KafkaTrigger '%s' consume loop error",
                self.config.effective_name(),
                exc_info=True,
            )

    async def _publish_result(self, result: Any) -> None:
        import json

        if self._producer is None:
            await super()._publish_result(result)
            return
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        try:
            if isinstance(payload, (bytes, bytearray)):
                serialized: bytes = bytes(payload)
            elif isinstance(payload, str):
                serialized = payload.encode("utf-8")
            else:
                serialized = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError):
            serialized = str(payload).encode("utf-8")
        await self._producer.send_and_wait(self.config.result_topic, serialized)
        self._published_results.append(payload)


# ---------------------------------------------------------------------------
# RabbitMQTrigger
# ---------------------------------------------------------------------------

class RabbitMQTrigger(WorkflowTrigger):
    """
    RabbitMQ-based trigger using ``aio_pika``.

    The ``aio_pika`` package is imported lazily inside ``start()``; if it
    is not installed, ``start()`` logs a warning and returns without
    raising.
    """

    def __init__(
        self,
        config: TriggerConfig,
        workflow_registry: Any,
        callback_manager: Any = None,
    ) -> None:
        super().__init__(config, workflow_registry, callback_manager)
        self._connection: Any = None
        self._channel: Any = None
        self._queue_obj: Any = None
        self._consumer_tag: Any = None
        self._aio_pika: Any = None

    async def start(self) -> None:
        if self._is_running:
            return
        try:
            import aio_pika  # type: ignore
        except ImportError:
            logger.warning(
                "aio_pika not installed; RabbitMQTrigger '%s' disabled "
                "(workflow=%s, queue=%s)",
                self.config.effective_name(),
                self.config.workflow,
                self.config.queue_name,
            )
            return
        self._aio_pika = aio_pika
        try:
            self._connection = await aio_pika.connect_robust(self.config.amqp_url)
            self._channel = await self._connection.channel()
            self._queue_obj = await self._channel.declare_queue(
                self.config.queue_name, durable=True
            )
        except Exception:
            logger.warning(
                "RabbitMQTrigger '%s' failed to connect "
                "(amqp_url=%s, queue=%s)",
                self.config.effective_name(),
                self.config.amqp_url,
                self.config.queue_name,
                exc_info=True,
            )
            await self._close_resources()
            return

        async def _on_message(message: Any) -> None:
            try:
                async with message.process(ignore_processed=True):
                    payload = self._decode_message(message.body)
                    await self._invoke_workflow(
                        payload, raw_message=message.body
                    )
            except Exception:
                logger.warning(
                    "RabbitMQTrigger '%s' message handling failed",
                    self.config.effective_name(),
                    exc_info=True,
                )

        self._consumer_tag = await self._queue_obj.consume(
            _on_message, no_ack=not self.config.auto_commit
        )
        self._is_running = True
        logger.info(
            "RabbitMQTrigger '%s' started (queue=%s, exchange=%s)",
            self.config.effective_name(),
            self.config.queue_name,
            self.config.exchange or "(default)",
        )

    async def stop(self) -> None:
        was_running = self._is_running
        self._is_running = False
        if self._queue_obj is not None and self._consumer_tag is not None:
            try:
                await self._queue_obj.cancel(self._consumer_tag)
            except Exception:
                pass
            self._consumer_tag = None
        await self._close_resources()
        if was_running:
            logger.info(
                "RabbitMQTrigger '%s' stopped", self.config.effective_name()
            )

    async def _close_resources(self) -> None:
        self._queue_obj = None
        self._channel = None
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                pass
            self._connection = None

    async def _consume_loop(self) -> None:
        # RabbitMQ uses an async callback registered via queue.consume()
        # in start(), so there is no explicit consume-loop coroutine.
        # This method is provided for API symmetry with KafkaTrigger and
        # is intentionally a no-op.
        return

    async def _publish_result(self, result: Any) -> None:
        import json

        if self._channel is None or self._aio_pika is None:
            await super()._publish_result(result)
            return
        payload = result.model_dump() if hasattr(result, "model_dump") else result
        try:
            if isinstance(payload, (bytes, bytearray)):
                body: bytes = bytes(payload)
            elif isinstance(payload, str):
                body = payload.encode("utf-8")
            else:
                body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError):
            body = str(payload).encode("utf-8")
        try:
            await self._channel.default_exchange.publish(
                self._aio_pika.Message(body=body),
                routing_key=self.config.result_topic,
            )
            self._published_results.append(payload)
        except Exception:
            logger.warning(
                "RabbitMQTrigger '%s' failed to publish result",
                self.config.effective_name(),
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# TriggerManager
# ---------------------------------------------------------------------------

class TriggerManager:
    """
    Manages the lifecycle of a set of WorkflowTriggers.

    Usage::

        mgr = TriggerManager(workflow_registry)
        await mgr.load_from_yaml("config/triggers.yaml")
        await mgr.start_all()
        ...
        await mgr.stop_all()

    The manager is decoupled from the API layer; whoever owns it (the
    FastAPI app's lifespan, a CLI command, etc.) is responsible for
    calling ``start_all`` / ``stop_all`` at the right times.
    """

    def __init__(
        self,
        workflow_registry: Any,
        callback_manager: Any = None,
    ) -> None:
        self.workflow_registry = workflow_registry
        self.callback_manager = callback_manager
        self._triggers: dict[str, WorkflowTrigger] = {}
        self._configs: dict[str, TriggerConfig] = {}
        self._is_running = False

    def register(self, config: TriggerConfig) -> WorkflowTrigger:
        """
        Register a trigger config and instantiate the corresponding trigger.

        Returns the created ``WorkflowTrigger``. If a trigger with the
        same effective name already exists, it is replaced.

        Raises:
            ValueError: If ``config.type`` is unknown.
        """
        name = config.effective_name()
        trigger = self._build_trigger(config)
        self._triggers[name] = trigger
        self._configs[name] = config
        logger.info(
            "Registered trigger '%s' (type=%s, workflow=%s)",
            name,
            config.type,
            config.workflow,
        )
        return trigger

    def _build_trigger(self, config: TriggerConfig) -> WorkflowTrigger:
        if config.type == "kafka":
            return KafkaTrigger(
                config, self.workflow_registry, self.callback_manager
            )
        if config.type == "rabbitmq":
            return RabbitMQTrigger(
                config, self.workflow_registry, self.callback_manager
            )
        if config.type == "inmemory":
            return InMemoryTrigger(
                config, self.workflow_registry, self.callback_manager
            )
        raise ValueError(
            f"Unknown trigger type '{config.type}'. "
            f"Expected one of: kafka, rabbitmq, inmemory"
        )

    async def start_all(self) -> dict[str, bool]:
        """
        Start every registered trigger whose ``enabled`` flag is True.

        Returns:
            A mapping of trigger name -> ``is_running`` after ``start()``
            returns. Disabled triggers are excluded from the result.
        """
        results: dict[str, bool] = {}

        async def _one(name: str, trig: WorkflowTrigger) -> tuple[str, bool]:
            try:
                await trig.start()
            except Exception:
                logger.warning(
                    "Trigger '%s' start raised", name, exc_info=True
                )
            return name, trig.is_running

        tasks = [
            _one(name, trig)
            for name, trig in self._triggers.items()
            if self._configs[name].enabled
        ]
        if tasks:
            for name, ok in await asyncio.gather(*tasks):
                results[name] = ok
        self._is_running = any(results.values())
        return results

    async def stop_all(self) -> None:
        """Stop every registered trigger. Idempotent."""

        async def _one(name: str, trig: WorkflowTrigger) -> None:
            try:
                await trig.stop()
            except Exception:
                logger.warning(
                    "Trigger '%s' stop raised", name, exc_info=True
                )

        if self._triggers:
            await asyncio.gather(
                *[_one(n, t) for n, t in self._triggers.items()]
            )
        self._is_running = False

    def get_trigger(self, name: str) -> Optional[WorkflowTrigger]:
        """Return the trigger registered under ``name``, or ``None``."""
        return self._triggers.get(name)

    def list_triggers(self) -> dict[str, TriggerConfig]:
        """Return a shallow copy of the trigger configs map."""
        return dict(self._configs)

    async def load_from_yaml(
        self, path: str | Path
    ) -> list[TriggerConfig]:
        """
        Load trigger configs from a YAML file and register them.

        The YAML file must have a top-level ``triggers`` key containing a
        list of trigger config dicts (see ``config/triggers.yaml`` for
        the schema). Returns the list of registered ``TriggerConfig``
        objects (in YAML order). Entries that fail validation are
        skipped with a warning.

        Args:
            path: Path to the YAML file.

        Returns:
            List of registered ``TriggerConfig`` instances.
        """
        p = Path(path)
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load trigger config. "
                "Install with: pip install PyYAML"
            ) from e
        if not p.exists():
            logger.warning("Triggers YAML not found: %s", p)
            return []
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning("Triggers YAML root must be a mapping: %s", p)
            return []
        entries = data.get("triggers", []) or []
        if not isinstance(entries, list):
            logger.warning("'triggers' must be a list in %s", p)
            return []
        configs: list[TriggerConfig] = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                logger.warning(
                    "triggers[%d] is not a mapping in %s", i, p
                )
                continue
            try:
                cfg = TriggerConfig(**entry)
            except Exception:
                logger.warning(
                    "triggers[%d] failed validation in %s",
                    i,
                    p,
                    exc_info=True,
                )
                continue
            self.register(cfg)
            configs.append(cfg)
        return configs

    @property
    def is_running(self) -> bool:
        """Whether any trigger is currently running."""
        return self._is_running


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "TriggerConfig",
    "WorkflowTrigger",
    "InMemoryTrigger",
    "KafkaTrigger",
    "RabbitMQTrigger",
    "TriggerManager",
]
