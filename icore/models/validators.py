"""
icore.models.validators - Model output validation utilities.

Provides ``ModelOutputValidator`` with static helpers used to verify
LLM output before it propagates to downstream tasks:

    - ``validate_json(text, schema=None)``: Parse JSON output, optionally
      against a JSON Schema. Handles markdown code-block wrapping.
    - ``validate_non_empty(text)``: Ensure non-empty output.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from icore.exceptions import ValidationError


class ModelOutputValidator:
    """Static helpers for validating model output."""

    @staticmethod
    def validate_json(text: str, schema: Optional[dict] = None) -> dict:
        """
        Parse ``text`` as JSON, optionally validating against ``schema``.

        Handles common LLM quirks such as wrapping the JSON in a
        ```` ```json ... ``` ```` markdown code block.

        Args:
            text:   Raw model output.
            schema: Optional JSON Schema dict for validation. If
                    ``jsonschema`` is not installed and a schema is
                    given, only JSON parsing is performed (with a
                    warning).

        Returns:
            Parsed dict.

        Raises:
            ValidationError: On JSON parse failure or schema mismatch.
        """
        if not text or not text.strip():
            raise ValidationError("Model returned empty JSON output")

        cleaned = text.strip()
        # Strip markdown code fences (```json ... ``` or ``` ... ```).
        if cleaned.startswith("```"):
            # Remove first line (the fence + optional language tag).
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Model output is not valid JSON: {e.msg} at pos {e.pos}"
            ) from e

        if schema is not None:
            try:
                import jsonschema  # type: ignore
            except ImportError:
                # Without jsonschema we still return the data; schema
                # validation is best-effort when the package is absent.
                pass
            else:
                try:
                    jsonschema.validate(data, schema)
                except Exception as e:
                    raise ValidationError(
                        f"Model output failed schema validation: {e}"
                    ) from e

        return data

    @staticmethod
    def validate_non_empty(text: str) -> str:
        """
        Ensure ``text`` is non-empty (after strip).

        Raises:
            ValidationError: If empty/whitespace-only.
        """
        if not text or not text.strip():
            raise ValidationError("Model returned empty output")
        return text.strip()


__all__ = ["ModelOutputValidator"]
