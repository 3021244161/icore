"""
icore.models.router - Automatic model routing engine.

ModelRouter selects the best model for a task when no explicit model_id
is provided (model_id=None). It uses RoutingRule configurations to match
task characteristics against model capabilities and cost constraints.

Routing strategy:
    1. Match rules by task_type
    2. Filter candidates: enabled=True, healthy=True, within max_cost
    3. Score by tag match (0.5) + cost efficiency (0.3) + health (0.2)
    4. Return the highest-scoring model_id
    5. If no match, fall back to default_model_id
    6. If default unhealthy, fall back to rule's fallback_model_id

The router does NOT create adapters - it only returns model_id strings.
ModelManager.get_adapter() then uses the routed ID to retrieve/create
the adapter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from icore.config import RoutingRule
from icore.models.exceptions import NoAvailableModelError

if TYPE_CHECKING:
    from icore.models.manager import ModelManager

logger = logging.getLogger(__name__)


# Scoring weights
_TAG_WEIGHT = 0.5
_COST_WEIGHT = 0.3
_HEALTH_WEIGHT = 0.2


class ModelRouter:
    """
    Automatic model routing engine.

    Uses RoutingRule configurations to select the best model for a task
    when no explicit model_id is specified. The router queries
    ModelManager for available models, their health status, and
    configuration to make routing decisions.

    Attributes:
        _manager:          Reference to the ModelManager (for querying models).
        _rules:            List of RoutingRule configurations.
        _default_model_id: Fallback model when no rule matches.
    """

    def __init__(self, manager: ModelManager | None = None) -> None:
        """
        Initialize the router with an optional reference to its manager.

        Args:
            manager: The ModelManager that owns this router. The router
                     queries the manager for model configs and health.
                     Can be None if the manager will set it later, or
                     if the router operates in a standalone/test mode.
        """
        self._manager: ModelManager | None = manager
        self._rules: list[RoutingRule] = []
        self._default_model_id: str | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_manager(self, manager: ModelManager) -> None:
        """Set the ModelManager reference (for late binding)."""
        self._manager = manager

    def set_rules(self, rules: list[RoutingRule]) -> None:
        """Set the routing rules list."""
        self._rules = list(rules)
        logger.debug("Router configured with %d routing rules", len(self._rules))

    def set_default_model_id(self, model_id: str) -> None:
        """Set the default fallback model ID."""
        self._default_model_id = model_id

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a single routing rule."""
        self._rules.append(rule)

    # ------------------------------------------------------------------
    # Core routing method
    # ------------------------------------------------------------------

    def auto_route(
        self,
        task_type: str | None = None,
        max_cost: float | None = None,
        tags: list[str] | None = None,
    ) -> str | None:
        """
        Auto-route to the best available model.

        Args:
            task_type: Optional task type for rule matching.
            max_cost: Optional max cost constraint (per 1k input tokens).
            tags: Optional preferred tags.

        Returns:
            A model_id string, or None if no model is available.

        Note:
            When called without arguments (the common case from
            ModelManager.get_adapter(None)), it iterates all rules and
            picks the first matching one with available candidates.
        """
        # Try matching rules first
        for rule in self._rules:
            if self._rule_matches(rule, task_type):
                candidate = self._select_from_rule(rule, max_cost, tags)
                if candidate is not None:
                    logger.debug(
                        "Auto-routed to '%s' via rule (task_type=%s)",
                        candidate,
                        rule.task_type,
                    )
                    return candidate

        # No rule matched (or no rules configured) -> try all enabled models
        candidate = self._select_best_general(max_cost, tags)
        if candidate is not None:
            logger.debug("Auto-routed to '%s' (general selection)", candidate)
            return candidate

        # Fall back to default model
        if self._default_model_id is not None:
            if self._is_model_available(self._default_model_id):
                logger.debug(
                    "Auto-routed to default '%s'", self._default_model_id
                )
                return self._default_model_id
            # Default is unhealthy, try its fallback
            fallback = self._find_rule_fallback(self._default_model_id)
            if fallback and self._is_model_available(fallback):
                logger.debug(
                    "Auto-routed to fallback of default '%s'", fallback
                )
                return fallback

        logger.warning("Auto-routing found no available model")
        return None

    # ------------------------------------------------------------------
    # Rule matching
    # ------------------------------------------------------------------

    def _rule_matches(self, rule: RoutingRule, task_type: str | None) -> bool:
        """Check if a routing rule matches the given task type."""
        if rule.task_type is None:
            # Rule with no task_type matches everything (catch-all)
            return True
        if task_type is None:
            # No task type specified, skip rules that require one
            return False
        return rule.task_type == task_type

    def _select_from_rule(
        self,
        rule: RoutingRule,
        max_cost: float | None,
        tags: list[str] | None,
    ) -> str | None:
        """
        Select the best model from candidates matching a rule.

        Combines rule's preferred_tags and max_cost with any caller-specified
        tags and max_cost (caller takes precedence).
        """
        effective_tags = tags if tags is not None else rule.preferred_tags
        effective_cost = max_cost if max_cost is not None else rule.max_cost

        candidates = self._get_candidates(effective_cost)

        if not candidates:
            # No candidates match, try rule's fallback
            if rule.fallback_model_id and self._is_model_available(
                rule.fallback_model_id
            ):
                return rule.fallback_model_id
            return None

        return self._score_and_select(candidates, effective_tags)

    def _select_best_general(
        self,
        max_cost: float | None,
        tags: list[str] | None,
    ) -> str | None:
        """Select the best model without a specific rule match."""
        candidates = self._get_candidates(max_cost)
        if not candidates:
            return None
        return self._score_and_select(candidates, tags)

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------

    def _get_candidates(
        self,
        max_cost: float | None,
    ) -> list[str]:
        """
        Get model_ids that are enabled, healthy, and within cost.

        Args:
            max_cost: Maximum acceptable cost per 1k input tokens.
                      None means no cost constraint.

        Returns:
            List of candidate model_ids.
        """
        if self._manager is None:
            return []
        all_models = self._manager.list_models()
        candidates = []

        for model_info in all_models:
            if not model_info.get("enabled", False):
                continue
            if not model_info.get("healthy", False):
                continue
            if max_cost is not None:
                input_cost = model_info.get("cost_per_1k_input", 0.0)
                if input_cost > max_cost:
                    continue
            candidates.append(model_info["model_id"])

        return candidates

    def _is_model_available(self, model_id: str) -> bool:
        """Check if a model is registered, enabled, and healthy."""
        if self._manager is None:
            return False
        all_models = self._manager.list_models()
        for model_info in all_models:
            if model_info["model_id"] == model_id:
                return (
                    model_info.get("enabled", False)
                    and model_info.get("healthy", False)
                )
        return False

    # ------------------------------------------------------------------
    # Scoring and selection
    # ------------------------------------------------------------------

    def _score_and_select(
        self,
        candidates: list[str],
        preferred_tags: list[str] | None,
    ) -> str | None:
        """
        Score candidates and return the best one.

        Scoring:
            - Tag match (0.5):  How many preferred_tags the model has
            - Cost (0.3):       Lower cost = higher score
            - Health (0.2):     Fewer failures = higher score
        """
        if not candidates:
            return None

        if not preferred_tags:
            preferred_tags = []

        if self._manager is None:
            return candidates[0]

        all_models = {m["model_id"]: m for m in self._manager.list_models()}
        health_statuses = self._manager.get_all_health_status()

        best_id: str | None = None
        best_score = -1.0

        for model_id in candidates:
            model_info = all_models.get(model_id, {})
            health = health_statuses.get(model_id)

            # Tag score: fraction of preferred tags matched
            model_tags = set(model_info.get("tags", []))
            if preferred_tags:
                matched = len(model_tags & set(preferred_tags))
                tag_score = matched / len(preferred_tags)
            else:
                tag_score = 0.5  # Neutral when no tags specified

            # Cost score: lower cost -> higher score (normalized)
            input_cost = model_info.get("cost_per_1k_input", 0.0)
            output_cost = model_info.get("cost_per_1k_output", 0.0)
            total_cost = input_cost + output_cost
            # Normalize: cost of 0 = score 1.0, cost of 0.1+ = score ~0.0
            cost_score = 1.0 / (1.0 + total_cost * 100)

            # Health score: fewer failures = higher score
            if health:
                health_score = 1.0 / (1.0 + health.consecutive_failures)
            else:
                health_score = 0.5

            total_score = (
                _TAG_WEIGHT * tag_score
                + _COST_WEIGHT * cost_score
                + _HEALTH_WEIGHT * health_score
            )

            if total_score > best_score:
                best_score = total_score
                best_id = model_id

        return best_id

    # ------------------------------------------------------------------
    # Fallback chain
    # ------------------------------------------------------------------

    def get_fallback(self, model_id: str) -> str | None:
        """
        Get the fallback model for a given model_id.

        Searches routing rules for one whose task_type matches the
        given model, or returns the rule's fallback_model_id if the
        given model was the rule's primary candidate.

        Since routing rules don't explicitly map model_id -> fallback,
        this searches for any rule with a fallback_model_id and returns
        the first one that's available.

        Args:
            model_id: The model that needs a fallback.

        Returns:
            A fallback model_id, or None if none available.
        """
        # First, check if any rule has this model as primary and a fallback
        for rule in self._rules:
            if rule.fallback_model_id and self._is_model_available(
                rule.fallback_model_id
            ):
                # Return the first available fallback
                return rule.fallback_model_id

        # Try default model as fallback
        if (
            self._default_model_id
            and self._default_model_id != model_id
            and self._is_model_available(self._default_model_id)
        ):
            return self._default_model_id

        return None

    def _find_rule_fallback(self, model_id: str) -> str | None:
        """Find a fallback for the default model specifically."""
        for rule in self._rules:
            if (
                rule.fallback_model_id
                and rule.fallback_model_id != model_id
                and self._is_model_available(rule.fallback_model_id)
            ):
                return rule.fallback_model_id
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_rules(self) -> list[dict[str, object]]:
        """List all routing rules as dicts for inspection/debugging."""
        result = []
        for rule in self._rules:
            result.append(
                {
                    "task_type": rule.task_type,
                    "max_cost": rule.max_cost,
                    "preferred_tags": list(rule.preferred_tags),
                    "fallback_model_id": rule.fallback_model_id,
                }
            )
        return result

    def __repr__(self) -> str:
        return (
            f"ModelRouter(rules={len(self._rules)}, "
            f"default={self._default_model_id!r})"
        )
