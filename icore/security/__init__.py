"""
icore.security - v0.6 安全加固模块。

提供两大能力，全部基于 Python 标准库 + Pydantic 实现，**不引入外部
NLP 库**：

    1. **Prompt 注入检测**：``InjectionDetector`` 采用三层策略
       （关键词匹配 / 启发式规则 / 可选模型检测），返回
       :class:`InjectionResult` 含置信度与脱敏后输入。
    2. **PII 检测与脱敏**：``PIIDetector`` 基于正则匹配自动检测并
       脱敏手机号 / 身份证 / 邮箱 / API Key / 银行卡号，支持
       ``mask → unmask`` 往复恢复原始值。

模块依赖：
    仅依赖标准库（``re`` / ``logging`` / ``inspect``）+ ``icore`` 自身。
    所有正则在线程级安全使用（``re`` 模块本身线程安全）。
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = [
    "InjectionResult",
    "InjectionDetector",
    "PIIDetector",
]

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt Injection Detection
# ============================================================================

#: Default injection keywords (English + Chinese). Each is matched as a
#: case-insensitive substring within the user input.
DEFAULT_KEYWORDS: list[str] = [
    # --- English: explicit injection phrases ---
    "ignore previous instructions",
    "ignore all previous",
    "ignore the above",
    "disregard previous",
    "disregard all previous",
    "forget previous instructions",
    "forget all previous",
    "override previous instructions",
    "override your instructions",
    "override your system",
    "override system prompt",
    "DAN mode",
    "do anything now",
    "system prompt",
    "reveal your instructions",
    "reveal your prompt",
    "show your prompt",
    "show your instructions",
    "show your system prompt",
    "you are GPT",
    "you are ChatGPT",
    "you are now DAN",
    "you are an AI",
    "jailbreak",
    "developer mode",
    "enable developer mode",
    # --- Chinese: explicit injection phrases ---
    "忽略之前",
    "忽略上述",
    "忽略所有",
    "忘记之前",
    "忘记上述",
    "忘记所有",
    "无视之前",
    "无视上述",
    "你是GPT",
    "你是一个AI",
    "你是一个人工智能",
    "脱离设定",
    "越狱",
    "系统提示词",
    "显示你的提示词",
    "覆盖你的指令",
    "覆盖系统提示",
    "开发者模式",
    "开启开发者模式",
]

#: Heuristic patterns: (compiled_regex, human_label, weight).
#: Each pattern targets a specific injection vector (role override,
#: ChatML token smuggling, Llama tag injection, JSON role injection, etc.).
_HEURISTIC_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    # ChatML / OpenAI token smuggling (high risk).
    (re.compile(r"<\|im_start\|>", re.IGNORECASE), "ChatML im_start token", 0.85),
    (re.compile(r"<\|im_end\|>", re.IGNORECASE), "ChatML im_end token", 0.7),
    (re.compile(r"<\|endoftext\|>", re.IGNORECASE), "endoftext token", 0.7),
    # System / role override tags.
    (re.compile(r"<system>", re.IGNORECASE), "<system> tag", 0.8),
    (re.compile(r"</system>", re.IGNORECASE), "</system> tag", 0.6),
    (re.compile(r"<<SYS>>", re.IGNORECASE), "Llama <<SYS>> tag", 0.85),
    (re.compile(r"<</SYS>>", re.IGNORECASE), "Llama <</SYS>> tag", 0.6),
    (re.compile(r"\[INST\]", re.IGNORECASE), "Llama [INST] tag", 0.75),
    (re.compile(r"\[/INST\]", re.IGNORECASE), "Llama [/INST] tag", 0.6),
    # Role:system directive (text and JSON form).
    (re.compile(r"role\s*:\s*system", re.IGNORECASE), "role:system directive", 0.85),
    (
        re.compile(r'"role"\s*:\s*"system"', re.IGNORECASE),
        'JSON "role":"system" injection',
        0.85,
    ),
    # Instruction-override patterns.
    (
        re.compile(r"new\s+system\s+prompt\s*:", re.IGNORECASE),
        "new system prompt directive",
        0.8,
    ),
    (
        re.compile(r"\\n\s*system\\n\s*:", re.IGNORECASE),
        "escaped system role injection",
        0.75,
    ),
]

#: Per-keyword confidence contribution.
_KEYWORD_WEIGHT: float = 0.75
#: Cap for keyword-only confidence accumulation.
_KEYWORD_CAP: float = 0.8
#: Cap for heuristic-only confidence accumulation.
_HEURISTIC_CAP: float = 1.0
#: Rule-based confidence cap (combined keywords + heuristics).
_RULE_CAP: float = 1.0
#: Model detector weight in combined score (model is weighted higher).
_MODEL_WEIGHT: float = 0.7
#: Rule-based weight in combined score.
_RULE_WEIGHT: float = 0.3


@dataclass
class InjectionResult:
    """Result of prompt injection detection.

    Attributes:
        is_injection:      Whether the input was classified as injection
                           (``confidence >= threshold``).
        confidence:        Detection confidence in ``[0.0, 1.0]``.
        matched_patterns:  Human-readable labels for each matched signal
                           (e.g. ``"keyword:ignore previous instructions"``,
                           ``"heuristic:<system> tag"``).
        sanitized_input:   Input with matched keywords/patterns replaced
                           by ``[FILTERED]``.
    """

    is_injection: bool
    confidence: float
    matched_patterns: list[str] = field(default_factory=list)
    sanitized_input: str = ""


class InjectionDetector:
    """Three-layer prompt injection detector.

    Layer 1 — **Keyword matching**: case-insensitive substring match
    against a curated EN+CN keyword list (overridable via ``keywords``).

    Layer 2 — **Heuristic rules**: regex patterns targeting role-override
    tokens (``<system>``, ``<|im_start|>``, ``role:system``, JSON role
    injection, Llama tags, etc.).

    Layer 3 — **Model detection** (optional): an injected ``model_detector``
    object (with a ``detect(user_input) -> float`` method) or callable
    returning a ``[0.0, 1.0]`` confidence. When absent, only layers 1+2
    are used.

    Args:
        keywords:        Custom keyword list. Defaults to
                         :data:`DEFAULT_KEYWORDS`.
        model_detector:  Optional model-based detector. Must be callable
                         or expose a ``detect`` method (sync or async)
                         returning a ``float`` in ``[0.0, 1.0]``.
        threshold:       Confidence threshold above which the input is
                         classified as injection. Defaults to ``0.7``.
    """

    def __init__(
        self,
        *,
        keywords: Optional[list[str]] = None,
        model_detector: Any = None,
        threshold: float = 0.7,
    ) -> None:
        self._keywords: list[str] = list(keywords) if keywords is not None else list(
            DEFAULT_KEYWORDS
        )
        # Pre-compile keyword regexes (case-insensitive, escaped).
        self._keyword_patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(re.escape(kw), re.IGNORECASE), kw) for kw in self._keywords
        ]
        self._model_detector: Any = model_detector
        self._threshold: float = threshold

    async def detect(self, user_input: str) -> InjectionResult:
        """Analyze ``user_input`` for prompt injection.

        Returns an :class:`InjectionResult` with the combined confidence,
        matched pattern labels, and a sanitized copy of the input.
        """
        if not user_input:
            return InjectionResult(
                is_injection=False,
                confidence=0.0,
                matched_patterns=[],
                sanitized_input=user_input,
            )

        matched: list[str] = []
        sanitized = user_input

        # --- Layer 1: keyword matching ---
        keyword_hits = 0
        for pattern, kw in self._keyword_patterns:
            if pattern.search(sanitized):
                matched.append(f"keyword:{kw}")
                keyword_hits += 1
                sanitized = pattern.sub("[FILTERED]", sanitized)

        keyword_score = min(_KEYWORD_CAP, _KEYWORD_WEIGHT * keyword_hits)

        # --- Layer 2: heuristic rules ---
        heuristic_hits = 0
        heuristic_score = 0.0
        for pattern, label, weight in _HEURISTIC_PATTERNS:
            if pattern.search(sanitized):
                matched.append(f"heuristic:{label}")
                heuristic_hits += 1
                heuristic_score += weight
                sanitized = pattern.sub("[FILTERED]", sanitized)

        heuristic_score = min(_HEURISTIC_CAP, heuristic_score)

        rule_score = min(_RULE_CAP, keyword_score + heuristic_score)

        # --- Layer 3: model detection (optional) ---
        model_score: Optional[float] = None
        if self._model_detector is not None:
            model_score = await self._run_model_detector(user_input)

        if model_score is not None:
            confidence = _RULE_WEIGHT * rule_score + _MODEL_WEIGHT * model_score
        else:
            confidence = rule_score

        confidence = max(0.0, min(1.0, confidence))
        is_injection = confidence >= self._threshold

        return InjectionResult(
            is_injection=is_injection,
            confidence=round(confidence, 4),
            matched_patterns=matched,
            sanitized_input=sanitized,
        )

    async def _run_model_detector(self, user_input: str) -> float:
        """Invoke the model detector, returning a clamped ``[0, 1]`` score.

        Supports:
            - Objects with a ``detect(user_input)`` method (sync or async).
            - Plain callables ``(user_input) -> float`` (sync or async).

        Returns ``0.0`` on any failure (logged as a warning).
        """
        detect_fn: Optional[Callable[..., Any]] = getattr(
            self._model_detector, "detect", None
        )
        if detect_fn is None and callable(self._model_detector):
            detect_fn = self._model_detector
        if detect_fn is None:
            return 0.0
        try:
            result = detect_fn(user_input)
            if inspect.isawaitable(result):
                result = await result
            return max(0.0, min(1.0, float(result)))
        except Exception:
            logger.warning(
                "Model detector raised an exception; falling back to 0.0",
                exc_info=True,
            )
            return 0.0


# ============================================================================
# PII Detection & Masking
# ============================================================================

#: All supported PII categories.
PII_CATEGORIES: tuple[str, ...] = (
    "phone",
    "id_card",
    "email",
    "api_key",
    "bank_card",
)

#: Placeholder prefix used in masked text (unique per occurrence).
_PLACEHOLDER_PREFIX = "<<PII"


class PIIDetector:
    """Regex-based PII detector and masker.

    Supports five PII categories:

        - ``phone``     — China mobile numbers (``1[3-9]\\d{9}``)
        - ``id_card``   — China 18-digit ID cards (with date validation)
        - ``email``     — Standard email addresses
        - ``api_key``   — OpenAI-style API keys (``sk-`` prefix)
        - ``bank_card`` — China UnionPay bank cards (``62`` prefix, 16-19 digits)

    Each category uses a display mask format:

        - phone:     ``138****1234``  (first 3 + last 4)
        - id_card:   ``110101****1234`` (first 6 + last 4)
        - email:     ``a***@example.com`` (first char + domain)
        - api_key:   ``sk-****``      (prefix + masked body)
        - bank_card: ``6210****5678`` (first 4 + last 4)

    Args:
        enabled_categories: Categories to detect. ``None`` enables all.
    """

    def __init__(
        self,
        *,
        enabled_categories: Optional[list[str]] = None,
    ) -> None:
        if enabled_categories is None:
            self._enabled = set(PII_CATEGORIES)
        else:
            self._enabled = {
                c for c in enabled_categories if c in PII_CATEGORIES
            }

    # -- public API --------------------------------------------------------

    async def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Detect and mask all PII in ``text``.

        Returns a tuple of:

            - ``masked_text``: text with each PII replaced by a **unique
              placeholder** (``<<PII:phone:0>>``, ``<<PII:email:1>>``, etc.).
            - ``mapping``: ``{placeholder: original_value}`` for restoration.

        The placeholder is unique per occurrence (indexed), so multiple PIIs
        with the same display mask do not collide.
        """
        if not text:
            return text, {}

        # Collect all matches: (start, end, category, original_value, display_mask)
        matches: list[tuple[int, int, str, str, str]] = []
        for category in PII_CATEGORIES:
            if category not in self._enabled:
                continue
            for m in self._find_matches(category, text):
                matches.append(m)

        if not matches:
            return text, {}

        # Sort by start position descending so we can replace right-to-left
        # without shifting indices.
        matches.sort(key=lambda x: x[0], reverse=True)

        masked_text = text
        mapping: dict[str, str] = {}
        counters: dict[str, int] = {}

        for start, end, category, original, display_mask in matches:
            counters[category] = counters.get(category, 0) + 1
            idx = counters[category] - 1
            placeholder = f"{_PLACEHOLDER_PREFIX}:{category}:{idx}>>"
            masked_text = masked_text[:start] + placeholder + masked_text[end:]
            mapping[placeholder] = original

        return masked_text, mapping

    async def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore original PII values by replacing placeholders.

        Placeholders are replaced longest-first to avoid partial-match
        issues (e.g. ``<<PII:phone:0>>`` vs ``<<PII:phone:10>>``).
        """
        if not mapping:
            return text
        result = text
        # Sort by key length descending so longer placeholders are replaced
        # first (prevents ``<<PII:phone:1>>`` from matching inside
        # ``<<PII:phone:10>>``).
        for key in sorted(mapping.keys(), key=len, reverse=True):
            result = result.replace(key, mapping[key])
        return result

    @staticmethod
    def mask_display(category: str, value: str) -> str:
        """Return the human-readable display mask for a PII value.

        This is the "show to user" format (e.g. ``138****1234``). The
        :meth:`mask` method uses unique placeholders internally; call this
        method separately when you need the display format.
        """
        if category == "phone":
            if len(value) >= 7:
                return value[:3] + "****" + value[-4:]
            return "****"
        if category == "id_card":
            if len(value) >= 10:
                return value[:6] + "****" + value[-4:]
            return "****"
        if category == "email":
            at = value.find("@")
            if at > 0:
                local = value[:at]
                domain = value[at:]
                return local[0] + "***" + domain
            return "****"
        if category == "api_key":
            if value.startswith("sk-"):
                return "sk-****"
            return "****"
        if category == "bank_card":
            if len(value) >= 8:
                return value[:4] + "****" + value[-4:]
            return "****"
        return "****"

    # -- internal: regex patterns ------------------------------------------

    @staticmethod
    def _find_matches(
        category: str, text: str
    ) -> list[tuple[int, int, str, str, str]]:
        """Find all matches for ``category`` in ``text``.

        Returns list of ``(start, end, category, original_value, display_mask)``.
        """
        patterns: list[re.Pattern[str]] = []
        if category == "phone":
            # China mobile: 1[3-9] + 9 digits, not surrounded by digits.
            patterns = [re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")]
        elif category == "id_card":
            # 18-digit ID: region(6) + year(19xx/20xx) + month + day + seq + check.
            patterns = [
                re.compile(
                    r"(?<!\d)"
                    r"[1-9]\d{5}"
                    r"(?:19|20)\d{2}"
                    r"(?:0[1-9]|1[0-2])"
                    r"(?:0[1-9]|[12]\d|3[01])"
                    r"\d{3}[\dXx]"
                    r"(?!\d)"
                )
            ]
        elif category == "email":
            patterns = [re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")]
        elif category == "api_key":
            # OpenAI-style: sk- + 20+ alphanumeric chars.
            patterns = [re.compile(r"sk-[A-Za-z0-9]{20,}")]
        elif category == "bank_card":
            # UnionPay: 62 + 14-17 digits (16-19 total), not surrounded by digits.
            patterns = [re.compile(r"(?<!\d)62\d{14,17}(?!\d)")]

        results: list[tuple[int, int, str, str, str]] = []
        seen_spans: set[tuple[int, int]] = set()
        for pattern in patterns:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                # Skip overlapping matches (earlier categories take priority).
                if any(
                    span[0] < s_end and span[1] > s_start
                    for s_start, s_end in seen_spans
                ):
                    continue
                seen_spans.add(span)
                original = m.group()
                display = PIIDetector.mask_display(category, original)
                results.append((m.start(), m.end(), category, original, display))
        return results
