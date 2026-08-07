"""
icore.engine.hot_reload - Configuration hot reload (v0.6 §3.3.1).

Watches the YAML configuration directory and reloads affected managers
when files change on disk — no service restart required.

Watched files and their reload actions:

    ===============  =========================================
    File             Reload action
    ===============  =========================================
    models.yaml      Rebuild ``ModelManager`` (re-registers all
                     models + routing rules; in-flight requests
                     keep their already-resolved adapters).
    prompts/*.yaml   Reload into ``PromptManager`` (per-template
                     version bump).
    triggers.yaml    Rebuild ``TriggerManager`` (stop → reload →
                     restart triggers).
    ===============  =========================================

Backend:

    The default backend uses ``watchdog`` (optional dependency). When
    ``watchdog`` is not installed, the coordinator falls back to a
    polling observer (checks mtime every ``poll_interval`` seconds)
    so the system still works in minimal installations.

Threading:

    ``watchdog``'s ``Observer`` runs on its own thread. Reload
    callbacks are scheduled on the asyncio event loop via
    ``loop.call_soon_threadsafe`` so we never touch the managers
    from a foreign thread.

Module dependencies:
    engine + (optional) watchdog. No api / services imports.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reload callback signature
# ---------------------------------------------------------------------------

ReloadCallback = Callable[[Path], "Any"]
"""Reload callback invoked when a watched file changes.

Receives the changed path. May return a coroutine (awaited on the
loop) or a plain value. Exceptions are logged but never propagated
so one failing reload doesn't poison the observer.
"""


# ---------------------------------------------------------------------------
# Reload entry
# ---------------------------------------------------------------------------

@dataclass
class _WatchEntry:
    """One watched file → callback binding."""

    path: Path
    callback: ReloadCallback
    last_mtime: float = 0.0
    debounce_sec: float = 0.5
    last_fired: float = 0.0


@dataclass
class ReloadEvent:
    """A reload event delivered to a callback."""

    path: Path
    event_type: str  # "modified" | "created" | "moved"
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Polling observer (always available, no third-party deps)
# ---------------------------------------------------------------------------

class _PollingObserver:
    """mtime-based polling observer.

    Runs in a background thread, checks every entry's mtime every
    ``poll_interval`` seconds. Fires the callback via the event loop
    when mtime changes.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        poll_interval: float = 1.0,
    ) -> None:
        self._loop = loop
        self._poll_interval = poll_interval
        self._entries: list[_WatchEntry] = []
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def add(self, entry: _WatchEntry) -> None:
        with self._lock:
            try:
                entry.last_mtime = entry.path.stat().st_mtime
            except FileNotFoundError:
                entry.last_mtime = 0.0
            self._entries.append(entry)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="icore-hot-reload", daemon=True
        )
        self._thread.start()
        logger.info(
            "Hot-reload polling observer started (interval=%.1fs)",
            self._poll_interval,
        )

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self._poll_interval * 2)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                entries = list(self._entries)
            for entry in entries:
                if self._stop.is_set():
                    break
                try:
                    mtime = entry.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if mtime <= entry.last_mtime:
                    continue
                # Debounce: skip if we just fired for this entry.
                now = time.time()
                if now - entry.last_fired < entry.debounce_sec:
                    entry.last_mtime = mtime
                    continue
                entry.last_fired = now
                entry.last_mtime = mtime
                self._loop.call_soon_threadsafe(
                    self._dispatch, entry
                )
            self._stop.wait(self._poll_interval)

    def _dispatch(self, entry: _WatchEntry) -> None:
        """Run on the loop thread."""
        try:
            result = entry.callback(entry.path)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result, loop=self._loop)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Hot-reload callback for %s raised", entry.path
            )


# ---------------------------------------------------------------------------
# Watchdog observer (lazy import; falls back to polling)
# ---------------------------------------------------------------------------

class _WatchdogObserver:
    """``watchdog``-backed observer; falls back to polling on import error."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        poll_interval: float = 1.0,
    ) -> None:
        self._loop = loop
        self._poll_interval = poll_interval
        self._entries: list[_WatchEntry] = []
        self._lock = threading.RLock()
        self._observer: Any = None
        self._handler: Any = None

    def add(self, entry: _WatchEntry) -> None:
        with self._lock:
            try:
                entry.last_mtime = entry.path.stat().st_mtime
            except FileNotFoundError:
                entry.last_mtime = 0.0
            self._entries.append(entry)
            # Register a watch for the parent directory if observer is running.
            if self._observer is not None:
                self._observer.schedule(
                    self._handler,
                    str(entry.path.parent),
                    recursive=False,
                )

    def start(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore
            from watchdog.observers import Observer  # type: ignore
        except ImportError:
            logger.info(
                "watchdog not installed; falling back to polling observer"
            )
            # Replace ourselves with a polling observer.
            self._fallback = _PollingObserver(
                self._loop, self._poll_interval
            )
            for entry in self._entries:
                self._fallback.add(entry)
            self._fallback.start()
            self._observer = None
            return

        outer = self

        class _Handler(FileSystemEventHandler):  # type: ignore[misc]
            def on_modified(self, event: Any) -> None:
                if event.is_directory:
                    return
                outer._on_file_event(Path(event.src_path), "modified")

            def on_created(self, event: Any) -> None:
                if event.is_directory:
                    return
                outer._on_file_event(Path(event.src_path), "created")

            def on_moved(self, event: Any) -> None:
                if event.is_directory:
                    return
                outer._on_file_event(Path(event.dest_path), "moved")

        self._handler = _Handler()
        self._observer = Observer()
        watched_dirs: set[Path] = set()
        with self._lock:
            for entry in self._entries:
                watched_dirs.add(entry.path.parent)
        for d in watched_dirs:
            self._observer.schedule(
                self._handler, str(d), recursive=False
            )
        self._observer.start()
        logger.info(
            "Hot-reload watchdog observer started (watching %d dir(s))",
            len(watched_dirs),
        )

    def stop(self) -> None:
        if self._observer is None:
            # Maybe we fell back to polling.
            fb = getattr(self, "_fallback", None)
            if fb is not None:
                fb.stop()
            return
        self._observer.stop()
        self._observer.join(timeout=5.0)

    def _on_file_event(self, path: Path, event_type: str) -> None:
        with self._lock:
            entries = [
                e for e in self._entries if e.path == path.resolve()
                or e.path == path
            ]
        now = time.time()
        for entry in entries:
            if now - entry.last_fired < entry.debounce_sec:
                continue
            entry.last_fired = now
            self._loop.call_soon_threadsafe(
                self._dispatch, entry, event_type
            )

    def _dispatch(
        self, entry: _WatchEntry, event_type: str
    ) -> None:  # pragma: no cover - thin shim
        try:
            result = entry.callback(entry.path)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result, loop=self._loop)
        except Exception:
            logger.exception(
                "Hot-reload callback for %s raised", entry.path
            )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class HotReloadCoordinator:
    """Top-level coordinator (v0.6 §3.3).

    Wires file-watch observers to user-supplied reload callbacks.

    Typical use (in ``bootstrap.create_production_app``)::

        coord = HotReloadCoordinator(config_dir=settings.config_dir)
        coord.register("models.yaml", reload_model_manager)
        coord.register("prompts/summarizer/v1.yaml", reload_prompt)
        await coord.start()

        # On shutdown:
        await coord.stop()
    """

    def __init__(
        self,
        config_dir: str | Path,
        *,
        poll_interval: float = 1.0,
        use_watchdog: bool = True,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        trigger_manager: Any = None,
    ) -> None:
        self._config_dir = Path(config_dir)
        self._poll_interval = poll_interval
        self._use_watchdog = use_watchdog
        self._loop = loop  # resolved on start() if None
        self._observer: Optional[Any] = None
        self._started = False
        # v0.6: 可选的 TriggerManager 引用，triggers.yaml 变化时重建
        self._trigger_manager = trigger_manager

    def set_trigger_manager(self, mgr: Any) -> None:
        """注入 TriggerManager，triggers.yaml 变化时自动重建。"""
        self._trigger_manager = mgr

    def register(
        self,
        relative_path: str | Path,
        callback: ReloadCallback,
        *,
        debounce_sec: float = 0.5,
    ) -> None:
        """Register a watch for ``config_dir / relative_path``.

        Safe to call before :meth:`start`; the entry will be picked up
        when the observer starts.
        """
        path = (self._config_dir / relative_path).resolve()
        entry = _WatchEntry(
            path=path,
            callback=callback,
            debounce_sec=debounce_sec,
        )
        if self._observer is not None:
            self._observer.add(entry)
        else:
            self._pending: list[_WatchEntry] = getattr(
                self, "_pending", []
            )
            self._pending.append(entry)

    async def _reload_triggers(self, path: Any) -> None:
        """triggers.yaml 变化时的重建回调（v0.6）。

        当 ``self._trigger_manager`` 不为 None 时，先 ``stop_all()``
        停止所有触发器，再 ``start_all()`` 重启，实现热重建。
        """
        mgr = self._trigger_manager
        if mgr is None:
            return
        logger.info("Hot reload: rebuilding TriggerManager from %s", path)
        try:
            await mgr.stop_all()
        except Exception as e:
            logger.warning("Hot reload: stop_all failed: %s", e)
        try:
            # 重新从 YAML 加载配置并注册
            await mgr.load_from_yaml(path)
        except Exception as e:
            logger.warning("Hot reload: load_from_yaml failed: %s", e)
        try:
            await mgr.start_all()
        except Exception as e:
            logger.warning("Hot reload: start_all failed: %s", e)

    async def start(self) -> None:
        """Start the observer.

        If ``watchdog`` is not installed or ``use_watchdog=False``, a
        polling observer is used instead.
        """
        if self._started:
            return
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        if self._use_watchdog:
            self._observer = _WatchdogObserver(
                self._loop, self._poll_interval
            )
        else:
            self._observer = _PollingObserver(
                self._loop, self._poll_interval
            )
        # Drain pending registrations.
        for entry in getattr(self, "_pending", []):
            self._observer.add(entry)
        self._pending = []
        # ``start()`` is synchronous for both observer types.
        self._observer.start()
        self._started = True
        logger.info(
            "HotReloadCoordinator started (config_dir=%s, watchdog=%s)",
            self._config_dir,
            self._use_watchdog,
        )

    async def stop(self) -> None:
        """Stop the observer. Safe to call multiple times."""
        if not self._started or self._observer is None:
            return
        self._observer.stop()
        self._observer = None
        self._started = False
        logger.info("HotReloadCoordinator stopped")

    @property
    def started(self) -> bool:
        return self._started

    @property
    def config_dir(self) -> Path:
        return self._config_dir


__all__ = [
    "ReloadCallback",
    "ReloadEvent",
    "HotReloadCoordinator",
]
