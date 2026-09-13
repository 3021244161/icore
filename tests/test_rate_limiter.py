"""
Tests for icore.security.rate_limiter - Generic keyed rate limiting
primitives (ICORE-ISSUE-005).

Covers:
    - BaseRateLimiter abstractness
    - Interface contract (issue §5.1 shape — zGo depends on it)
    - MemoryRateLimiter:
        * check: first hit / within limit / denied at limit / denied
          hits still count / window expiry resets
        * peek: read-only, unknown key, denied with retry_after
        * reset
        * limit=0 denies everything (but peek on unused key allows)
        * invalid limit/window raise ValueError
        * concurrent checks respect the exact quota
        * expired-window cleanup (manual + throttled)
    - RedisRateLimiter (fake client, no network):
        * check calls the atomic Lua script with prefixed key + TTL ms
        * decision parsing (allowed/denied, reset_at, retry_after)
        * peek is read-only (GET/PTTL script, never INCR)
        * peek missing / TTL-less key -> no active window
        * reset issues DELETE on the prefixed key
        * Redis errors propagate (no silent fallback)
    - create_rate_limiter factory (memory default, unknown backend,
      redis without URL, redis with URL when the driver is installed)
    - Prometheus counter icore_rate_limit_checks_total (backend, outcome)
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import time
from typing import Any

import pytest

from icore.observability import Counter, MetricsRegistry, get_metrics_registry
from icore.security import (
    BaseRateLimiter,
    MemoryRateLimiter,
    RateLimitDecision,
    RedisRateLimiter,
    create_rate_limiter,
)
from icore.security.rate_limiter import _METRIC_CHECKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRedisClient:
    """Minimal async fake mirroring the redis.asyncio calls used."""

    def __init__(self, eval_result: Any = None) -> None:
        self.eval_calls: list[tuple[str, int, tuple[Any, ...]]] = []
        self.deleted: list[str] = []
        self.eval_result: Any = (
            [1, 5000] if eval_result is None else eval_result
        )

    async def eval(
        self, script: str, numkeys: int, *keys_and_args: Any
    ) -> list[Any]:
        self.eval_calls.append((script, numkeys, keys_and_args))
        if isinstance(self.eval_result, Exception):
            raise self.eval_result
        return list(self.eval_result)

    async def delete(self, *keys: str) -> int:
        self.deleted.extend(keys)
        return len(keys)


# ---------------------------------------------------------------------------
# Abstract base + interface contract
# ---------------------------------------------------------------------------

class TestBaseRateLimiter:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseRateLimiter()  # type: ignore[abstract]


class TestIssue005InterfaceContract:
    """ICORE-ISSUE-005 §5.1 — zGo 依赖此接口形态，防签名漂移。"""

    def test_decision_fields(self):
        names = tuple(f.name for f in dataclasses.fields(RateLimitDecision))
        assert names == (
            "allowed",
            "limit",
            "used",
            "remaining",
            "reset_at",
            "retry_after",
        )

    def test_decision_is_frozen(self):
        d = RateLimitDecision(
            allowed=True, limit=5, used=1, remaining=4, reset_at=1.0
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.allowed = False  # type: ignore[misc]

    def test_check_signature(self):
        sig = inspect.signature(BaseRateLimiter.check)
        assert list(sig.parameters) == ["self", "key", "limit", "window"]
        assert sig.parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["window"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_peek_signature(self):
        sig = inspect.signature(BaseRateLimiter.peek)
        assert list(sig.parameters) == ["self", "key", "limit"]
        assert sig.parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_reset_signature(self):
        sig = inspect.signature(BaseRateLimiter.reset)
        assert list(sig.parameters) == ["self", "key"]

    def test_factory_signature(self):
        sig = inspect.signature(create_rate_limiter)
        assert list(sig.parameters) == [
            "backend",
            "redis_url",
            "prefix",
            "cleanup_interval",
        ]
        assert sig.parameters["backend"].default == "memory"
        assert sig.parameters["redis_url"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_instances_expose_issue_api(self):
        rl = MemoryRateLimiter()
        for method in ("check", "peek", "reset"):
            assert callable(getattr(rl, method))


# ---------------------------------------------------------------------------
# MemoryRateLimiter
# ---------------------------------------------------------------------------

class TestMemoryCheck:
    async def test_first_hit_allowed(self):
        rl = MemoryRateLimiter()
        now = time.time()
        d = await rl.check("user:1", limit=5, window=60.0)
        assert d.allowed is True
        assert d.limit == 5
        assert d.used == 1
        assert d.remaining == 4
        assert d.retry_after is None
        assert d.reset_at == pytest.approx(now + 60.0, abs=0.5)

    async def test_counts_up_within_limit(self):
        rl = MemoryRateLimiter()
        for expected_used in (1, 2, 3):
            d = await rl.check("user:1", limit=5, window=60.0)
            assert d.used == expected_used
            assert d.allowed is True

    async def test_denied_at_limit_with_retry_after(self):
        rl = MemoryRateLimiter()
        for _ in range(2):
            await rl.check("user:1", limit=2, window=60.0)
        d = await rl.check("user:1", limit=2, window=60.0)
        assert d.allowed is False
        assert d.used == 3  # 含本次被拒的 hit
        assert d.remaining == 0
        assert d.retry_after is not None and d.retry_after > 0

    async def test_denied_hits_still_count(self):
        # 被拒的请求也占计数：防止拒绝风暴下免费试探。
        rl = MemoryRateLimiter()
        for _ in range(5):
            await rl.check("user:1", limit=2, window=60.0)
        d = await rl.check("user:1", limit=2, window=60.0)
        assert d.allowed is False
        assert d.used == 6

    async def test_independent_keys(self):
        rl = MemoryRateLimiter()
        await rl.check("user:1", limit=1, window=60.0)
        d = await rl.check("user:2", limit=1, window=60.0)
        assert d.allowed is True and d.used == 1

    async def test_limit_zero_denies_all_checks(self):
        rl = MemoryRateLimiter()
        d = await rl.check("k", limit=0, window=60.0)
        assert d.allowed is False
        assert d.remaining == 0

    async def test_window_expiry_resets_counter(self):
        rl = MemoryRateLimiter()
        await rl.check("k", limit=1, window=0.05)
        d1 = await rl.check("k", limit=1, window=0.05)
        assert d1.allowed is False
        await asyncio.sleep(0.07)
        d2 = await rl.check("k", limit=1, window=0.05)
        assert d2.allowed is True
        assert d2.used == 1  # 新窗口重新计数
        assert d2.reset_at > d1.reset_at


class TestMemoryPeek:
    async def test_peek_does_not_consume(self):
        rl = MemoryRateLimiter()
        for _ in range(3):
            await rl.check("user:1", limit=5, window=60.0)
        d = await rl.peek("user:1", limit=5)
        assert d.used == 3
        assert d.remaining == 2
        assert d.allowed is True
        # Repeated peeks never move the counter.
        d2 = await rl.peek("user:1", limit=5)
        assert d2.used == 3

    async def test_peek_unknown_key(self):
        rl = MemoryRateLimiter()
        d = await rl.peek("nobody", limit=5)
        assert d.allowed is True
        assert d.used == 0
        assert d.remaining == 5
        assert d.reset_at == 0.0
        assert d.retry_after is None

    async def test_peek_denied_reports_retry_after(self):
        rl = MemoryRateLimiter()
        for _ in range(3):
            await rl.check("user:1", limit=2, window=60.0)
        d = await rl.peek("user:1", limit=2)
        assert d.allowed is False
        assert d.used == 3
        assert d.retry_after is not None and d.retry_after > 0


class TestMemoryResetAndCleanup:
    async def test_reset_clears_key(self):
        rl = MemoryRateLimiter()
        await rl.check("user:1", limit=1, window=60.0)
        await rl.reset("user:1")
        d = await rl.peek("user:1", limit=1)
        assert d.used == 0
        # New window starts fresh.
        d2 = await rl.check("user:1", limit=1, window=60.0)
        assert d2.used == 1 and d2.allowed is True

    async def test_manual_cleanup_removes_expired(self):
        rl = MemoryRateLimiter(cleanup_interval=0)  # disable auto sweep
        await rl.check("old", limit=5, window=0.05)
        await asyncio.sleep(0.07)
        freed = rl.cleanup()
        assert freed == 1
        assert "old" not in rl._windows

    async def test_manual_cleanup_keeps_active(self):
        rl = MemoryRateLimiter(cleanup_interval=0)
        await rl.check("fresh", limit=5, window=60.0)
        assert rl.cleanup() == 0
        assert "fresh" in rl._windows

    async def test_throttled_auto_cleanup_on_check(self):
        rl = MemoryRateLimiter(cleanup_interval=0.01)
        await rl.check("old", limit=5, window=0.05)
        await asyncio.sleep(0.07)
        # A check on another key triggers the throttled sweep.
        await rl.check("new", limit=5, window=60.0)
        assert "old" not in rl._windows
        assert "new" in rl._windows


class TestMemoryValidation:
    async def test_negative_limit_raises(self):
        rl = MemoryRateLimiter()
        with pytest.raises(ValueError):
            await rl.check("k", limit=-1, window=60.0)
        with pytest.raises(ValueError):
            await rl.peek("k", limit=-1)

    async def test_non_positive_window_raises(self):
        rl = MemoryRateLimiter()
        with pytest.raises(ValueError):
            await rl.check("k", limit=5, window=0.0)
        with pytest.raises(ValueError):
            await rl.check("k", limit=5, window=-1.0)


class TestMemoryConcurrency:
    async def test_concurrent_checks_respect_exact_quota(self):
        # 60 个并发 check、配额 50：恰好 50 allowed / 10 denied，
        # 最终计数 60（被拒也计数）。内存版 check 无内部 await 点，
        # 单事件循环内原子。
        rl = MemoryRateLimiter()
        decisions = await asyncio.gather(
            *(rl.check("shared", limit=50, window=60.0) for _ in range(60))
        )
        allowed = sum(1 for d in decisions if d.allowed)
        assert allowed == 50
        assert all(d.used <= 60 for d in decisions)
        final = await rl.peek("shared", limit=50)
        assert final.used == 60


# ---------------------------------------------------------------------------
# RedisRateLimiter (fake client — no network)
# ---------------------------------------------------------------------------

class TestRedisCheck:
    async def test_first_hit_allowed_and_lua_args(self):
        fake = _FakeRedisClient(eval_result=[1, 5000])
        rl = RedisRateLimiter(fake)
        now = time.time()
        d = await rl.check("user:1", limit=5, window=5.0)
        assert d.allowed is True
        assert d.used == 1
        assert d.remaining == 4
        assert d.retry_after is None
        assert d.reset_at == pytest.approx(now + 5.0, abs=0.5)
        # Lua: single atomic script, 1 key, (prefixed key, ttl_ms).
        assert len(fake.eval_calls) == 1
        script, numkeys, args = fake.eval_calls[0]
        assert "INCR" in script and "PEXPIRE" in script
        assert numkeys == 1
        assert args == ("icore:rate_limit:user:1", 5000)

    async def test_lua_repairs_ttlless_key(self):
        # The script repairs a TTL-less key (count==1 or ttl<0 -> PEXPIRE):
        # guards the "counter without TTL leaks Redis memory" race.
        assert "count == 1 or ttl < 0" in RedisRateLimiter._CHECK_SCRIPT

    async def test_denied_decision_parsed(self):
        fake = _FakeRedisClient(eval_result=[6, 3000])
        rl = RedisRateLimiter(fake)
        d = await rl.check("user:1", limit=5, window=3.0)
        assert d.allowed is False
        assert d.used == 6
        assert d.remaining == 0
        assert d.retry_after is not None
        assert d.retry_after == pytest.approx(3.0, abs=0.5)

    async def test_custom_prefix(self):
        fake = _FakeRedisClient(eval_result=[1, 1000])
        rl = RedisRateLimiter(fake, prefix="zgo:rl:")
        await rl.check("u1", limit=5, window=1.0)
        _, _, args = fake.eval_calls[0]
        assert args[0] == "zgo:rl:u1"

    async def test_validation_matches_memory_backend(self):
        rl = RedisRateLimiter(_FakeRedisClient())
        with pytest.raises(ValueError):
            await rl.check("k", limit=-1, window=60.0)
        with pytest.raises(ValueError):
            await rl.check("k", limit=5, window=0.0)
        with pytest.raises(ValueError):
            await rl.peek("k", limit=-1)

    async def test_redis_error_propagates(self):
        # 失败姿态由调用方决定：原语不吞 Redis 异常（不做静默降级）。
        fake = _FakeRedisClient(eval_result=RuntimeError("redis down"))
        rl = RedisRateLimiter(fake)
        with pytest.raises(RuntimeError, match="redis down"):
            await rl.check("k", limit=5, window=1.0)


class TestRedisPeek:
    async def test_peek_is_read_only(self):
        fake = _FakeRedisClient(eval_result=[3, 2000])
        rl = RedisRateLimiter(fake)
        d = await rl.peek("user:1", limit=5)
        assert d.allowed is True
        assert d.used == 3
        assert d.remaining == 2
        assert d.retry_after is None
        # The peek script never increments.
        script = fake.eval_calls[0][0]
        assert "GET" in script and "INCR" not in script

    async def test_peek_denied_with_retry_after(self):
        fake = _FakeRedisClient(eval_result=[7, 1000])
        rl = RedisRateLimiter(fake)
        d = await rl.peek("user:1", limit=5)
        assert d.allowed is False
        assert d.used == 7
        assert d.retry_after == pytest.approx(1.0, abs=0.5)

    async def test_peek_missing_key(self):
        fake = _FakeRedisClient(eval_result=[0, -2])
        rl = RedisRateLimiter(fake)
        d = await rl.peek("ghost", limit=5)
        assert d.allowed is True
        assert d.used == 0
        assert d.remaining == 5
        assert d.reset_at == 0.0
        assert d.retry_after is None

    async def test_peek_ttlless_key_treated_as_no_window(self):
        fake = _FakeRedisClient(eval_result=[2, -1])
        rl = RedisRateLimiter(fake)
        d = await rl.peek("legacy", limit=5)
        assert d.allowed is True
        assert d.used == 0
        assert d.reset_at == 0.0


class TestRedisReset:
    async def test_reset_deletes_prefixed_key(self):
        fake = _FakeRedisClient()
        rl = RedisRateLimiter(fake)
        await rl.reset("user:1")
        assert fake.deleted == ["icore:rate_limit:user:1"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestCreateRateLimiter:
    def test_default_is_memory(self):
        rl = create_rate_limiter()
        assert isinstance(rl, MemoryRateLimiter)

    def test_explicit_memory(self):
        assert isinstance(create_rate_limiter("memory"), MemoryRateLimiter)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown rate limiter backend"):
            create_rate_limiter("memcached")

    def test_redis_without_url_raises(self):
        # conftest 的 autouse fixture 已隔离 ICORE_* 环境变量，
        # settings.concurrency.redis_url 默认为 None。
        with pytest.raises(ValueError, match="redis_url"):
            create_rate_limiter("redis")

    def test_redis_with_url_builds_limiter(self):
        try:
            import redis.asyncio  # noqa: F401
        except ImportError:
            pytest.skip("redis package not installed")
        rl = create_rate_limiter("redis", redis_url="redis://localhost:6379/0")
        assert isinstance(rl, RedisRateLimiter)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestRateLimiterMetrics:
    """icore_rate_limit_checks_total — 增量断言（registry 是进程级单例）。"""

    def _counter(self) -> Counter:
        return get_metrics_registry().get_counter(_METRIC_CHECKS)

    async def test_check_records_allowed_and_denied(self):
        counter = self._counter()
        before_allowed = counter.get(backend="memory", outcome="allowed")
        before_denied = counter.get(backend="memory", outcome="denied")
        rl = MemoryRateLimiter()
        await rl.check("metrics-ok", limit=5, window=60.0)
        await rl.check("metrics-no", limit=0, window=60.0)
        assert counter.get(backend="memory", outcome="allowed") == (
            before_allowed + 1
        )
        assert counter.get(backend="memory", outcome="denied") == (
            before_denied + 1
        )

    async def test_peek_does_not_record(self):
        counter = self._counter()
        before_allowed = counter.get(backend="memory", outcome="allowed")
        rl = MemoryRateLimiter()
        await rl.check("metrics-peek", limit=5, window=60.0)
        await rl.peek("metrics-peek", limit=5)
        await rl.peek("metrics-peek", limit=5)
        assert counter.get(backend="memory", outcome="allowed") == (
            before_allowed + 1
        )

    async def test_redis_backend_label(self):
        counter = self._counter()
        before = counter.get(backend="redis", outcome="allowed")
        rl = RedisRateLimiter(_FakeRedisClient(eval_result=[1, 5000]))
        await rl.check("k", limit=5, window=5.0)
        assert counter.get(backend="redis", outcome="allowed") == before + 1

    def test_metric_pre_registered_by_default(self):
        # 默认指标清单包含限流计数（新 registry 上可直接 get）。
        reg = MetricsRegistry()
        assert isinstance(
            reg.get_counter(_METRIC_CHECKS), Counter
        )
