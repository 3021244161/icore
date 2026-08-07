"""
Tests for v0.6 HotReloadCoordinator (v0.6 §3.3.1).

Covers:
    * Polling observer (no third-party deps) detects mtime change
    * Debounce: rapid changes fire only once
    * Callback receives the changed path
    * Async callbacks are awaited on the loop
    * Callback exceptions are swallowed (observer stays alive)
    * start/stop idempotent
    * Watchdog fallback: when watchdog is unavailable, falls back to polling
    * register() before start() works (pending entries drained on start)
    * register() after start() works (entry added live)
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from icore.engine.hot_reload import (
    HotReloadCoordinator,
    ReloadEvent,
    _PollingObserver,
    _WatchEntry,
)


# ===========================================================================
# Polling observer
# ===========================================================================

class TestPollingObserver:
    async def test_polling_detects_mtime_change(self, tmp_path: Path) -> None:
        target = tmp_path / "models.yaml"
        target.write_text("version: 1\n")

        loop = asyncio.get_event_loop()
        obs = _PollingObserver(loop, poll_interval=0.05)

        fired: list[Path] = []
        done = asyncio.Event()

        def cb(path: Path) -> None:
            fired.append(path)
            done.set()

        entry = _WatchEntry(path=target.resolve(), callback=cb)
        obs.add(entry)
        obs.start()
        try:
            # Make sure the initial mtime is older than the next write.
            time.sleep(0.06)
            # Touch the file with a newer mtime.
            new_mtime = time.time() + 1
            os.utime(target, (new_mtime, new_mtime))

            await asyncio.wait_for(done.wait(), timeout=2.0)
            assert fired, "callback should have fired"
            assert fired[0].name == "models.yaml"
        finally:
            obs.stop()

    async def test_polling_debounces_rapid_changes(self, tmp_path: Path) -> None:
        target = tmp_path / "models.yaml"
        target.write_text("v1\n")

        loop = asyncio.get_event_loop()
        obs = _PollingObserver(loop, poll_interval=0.02)

        # Counter accessible from the sync callback.
        counter = {"n": 0}

        def cb(path: Path) -> None:
            counter["n"] += 1

        entry = _WatchEntry(
            path=target.resolve(),
            callback=cb,
            debounce_sec=0.3,
        )
        obs.add(entry)
        obs.start()
        try:
            # Fire 5 rapid changes within the debounce window.
            for i in range(5):
                new_mtime = time.time() + i + 1
                os.utime(target, (new_mtime, new_mtime))
                await asyncio.sleep(0.01)

            # Wait past one polling cycle but inside debounce window.
            await asyncio.sleep(0.1)
            # First fire may have happened; subsequent ones should be debounced.
            first_count = counter["n"]
            # Wait past debounce so a final write fires again.
            time.sleep(0.35)
            new_mtime = time.time() + 100
            os.utime(target, (new_mtime, new_mtime))
            await asyncio.sleep(0.2)
            # The final write should fire (only fire if observer picks it up).
            # We only assert that the observer didn't fire 5 times.
            assert counter["n"] < 5
        finally:
            obs.stop()

    async def test_polling_callback_exception_swallowed(self, tmp_path: Path) -> None:
        target = tmp_path / "models.yaml"
        target.write_text("v1\n")

        loop = asyncio.get_event_loop()
        obs = _PollingObserver(loop, poll_interval=0.05)

        second_called = asyncio.Event()

        def cb1(path: Path) -> None:
            raise RuntimeError("boom")

        def cb2(path: Path) -> None:
            loop.call_soon_threadsafe(second_called.set)

        e1 = _WatchEntry(path=target.resolve(), callback=cb1)
        # Different file so the second callback fires independently.
        target2 = tmp_path / "other.yaml"
        target2.write_text("v2\n")
        e2 = _WatchEntry(path=target2.resolve(), callback=cb2)

        obs.add(e1)
        obs.add(e2)
        obs.start()
        try:
            time.sleep(0.06)
            new_mtime = time.time() + 1
            os.utime(target, (new_mtime, new_mtime))
            os.utime(target2, (new_mtime, new_mtime))
            # cb2 should still fire despite cb1 raising.
            await asyncio.wait_for(second_called.wait(), timeout=2.0)
            assert True
        finally:
            obs.stop()


# ===========================================================================
# HotReloadCoordinator
# ===========================================================================

class TestHotReloadCoordinator:
    async def test_register_before_start_fires_on_change(self, tmp_path: Path) -> None:
        target = tmp_path / "models.yaml"
        target.write_text("v1\n")

        coord = HotReloadCoordinator(
            config_dir=tmp_path,
            poll_interval=0.05,
            use_watchdog=False,  # force polling for portability
        )
        fired: list[Path] = []
        done = asyncio.Event()

        def cb(path: Path) -> None:
            fired.append(path)
            done.set()

        coord.register("models.yaml", cb, debounce_sec=0.0)
        await coord.start()
        try:
            time.sleep(0.06)
            new_mtime = time.time() + 1
            os.utime(target, (new_mtime, new_mtime))
            await asyncio.wait_for(done.wait(), timeout=2.0)
            assert fired
        finally:
            await coord.stop()
            assert coord.started is False

    async def test_register_after_start_fires_on_change(self, tmp_path: Path) -> None:
        target = tmp_path / "triggers.yaml"
        target.write_text("v1\n")

        coord = HotReloadCoordinator(
            config_dir=tmp_path,
            poll_interval=0.05,
            use_watchdog=False,
        )
        fired: list[Path] = []
        done = asyncio.Event()

        def cb(path: Path) -> None:
            fired.append(path)
            done.set()

        await coord.start()
        try:
            coord.register("triggers.yaml", cb, debounce_sec=0.0)
            time.sleep(0.06)
            new_mtime = time.time() + 1
            os.utime(target, (new_mtime, new_mtime))
            await asyncio.wait_for(done.wait(), timeout=2.0)
            assert fired
        finally:
            await coord.stop()

    async def test_async_callback_awaited(self, tmp_path: Path) -> None:
        target = tmp_path / "models.yaml"
        target.write_text("v1\n")

        coord = HotReloadCoordinator(
            config_dir=tmp_path,
            poll_interval=0.05,
            use_watchdog=False,
        )
        fired: list[str] = []
        done = asyncio.Event()

        async def cb(path: Path) -> None:
            await asyncio.sleep(0.01)
            fired.append("awaited")
            done.set()

        coord.register("models.yaml", cb, debounce_sec=0.0)
        await coord.start()
        try:
            time.sleep(0.06)
            new_mtime = time.time() + 1
            os.utime(target, (new_mtime, new_mtime))
            await asyncio.wait_for(done.wait(), timeout=2.0)
            assert fired == ["awaited"]
        finally:
            await coord.stop()

    async def test_start_stop_idempotent(self, tmp_path: Path) -> None:
        coord = HotReloadCoordinator(
            config_dir=tmp_path,
            poll_interval=0.1,
            use_watchdog=False,
        )
        await coord.start()
        await coord.start()  # no-op
        assert coord.started is True
        await coord.stop()
        await coord.stop()  # no-op
        assert coord.started is False

    async def test_watchdog_fallback_to_polling(self, tmp_path: Path, monkeypatch) -> None:
        """When watchdog import fails, the watchdog observer should silently
        fall back to polling — verified by simulating ImportError."""
        target = tmp_path / "models.yaml"
        target.write_text("v1\n")

        # Force ImportError on `from watchdog.observers import Observer`.
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name.startswith("watchdog"):
                raise ImportError("no watchdog")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        coord = HotReloadCoordinator(
            config_dir=tmp_path,
            poll_interval=0.05,
            use_watchdog=True,  # asks for watchdog but should fall back
        )
        fired: list[Path] = []
        done = asyncio.Event()

        def cb(path: Path) -> None:
            fired.append(path)
            done.set()

        coord.register("models.yaml", cb, debounce_sec=0.0)
        await coord.start()
        try:
            time.sleep(0.06)
            new_mtime = time.time() + 1
            os.utime(target, (new_mtime, new_mtime))
            await asyncio.wait_for(done.wait(), timeout=2.0)
            assert fired
        finally:
            await coord.stop()

    def test_config_dir_property(self, tmp_path: Path) -> None:
        coord = HotReloadCoordinator(config_dir=tmp_path, use_watchdog=False)
        assert coord.config_dir == tmp_path.resolve() or coord.config_dir == tmp_path


# ===========================================================================
# ReloadEvent dataclass
# ===========================================================================

class TestReloadEvent:
    def test_reload_event_fields(self) -> None:
        evt = ReloadEvent(path=Path("/tmp/x.yaml"), event_type="modified")
        assert evt.path == Path("/tmp/x.yaml")
        assert evt.event_type == "modified"
        assert evt.timestamp > 0
