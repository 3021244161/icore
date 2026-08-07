"""
Tests for icore.models.validators - Model output validation.

Covers:
    - ModelOutputValidator.validate_json:
        * plain JSON
        * JSON wrapped in ```json ... ``` code fence
        * JSON wrapped in plain ``` ... ``` fence
        * JSON with leading/trailing whitespace
        * Empty / whitespace-only input raises ValidationError
        * Invalid JSON raises ValidationError with position info
        * Schema validation when jsonschema is installed
        * Schema validation skipped gracefully when jsonschema missing
    - ModelOutputValidator.validate_non_empty:
        * non-empty returns stripped text
        * empty raises ValidationError
        * whitespace-only raises ValidationError
"""

from __future__ import annotations

import json

import pytest

from icore.exceptions import ValidationError
from icore.models.validators import ModelOutputValidator


# ---------------------------------------------------------------------------
# validate_json
# ---------------------------------------------------------------------------

class TestValidateJson:
    def test_plain_json(self):
        text = '{"name": "Alice", "age": 30}'
        data = ModelOutputValidator.validate_json(text)
        assert data == {"name": "Alice", "age": 30}

    def test_json_with_whitespace(self):
        text = '  \n  {"k": "v"}  \n  '
        data = ModelOutputValidator.validate_json(text)
        assert data == {"k": "v"}

    def test_json_wrapped_in_json_code_fence(self):
        text = '```json\n{"key": "value"}\n```'
        data = ModelOutputValidator.validate_json(text)
        assert data == {"key": "value"}

    def test_json_wrapped_in_plain_code_fence(self):
        text = '```\n{"key": "value"}\n```'
        data = ModelOutputValidator.validate_json(text)
        assert data == {"key": "value"}

    def test_json_with_array(self):
        text = '[1, 2, 3]'
        data = ModelOutputValidator.validate_json(text)
        assert data == [1, 2, 3]

    def test_empty_input_raises(self):
        with pytest.raises(ValidationError):
            ModelOutputValidator.validate_json("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError):
            ModelOutputValidator.validate_json("   \n  ")

    def test_invalid_json_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ModelOutputValidator.validate_json('{"name": "Alice"')
        # Error message should mention JSON.
        assert "JSON" in str(exc_info.value) or "json" in str(exc_info.value)

    def test_returns_dict_type(self):
        data = ModelOutputValidator.validate_json('{"k": 1}')
        assert isinstance(data, dict)

    def test_returns_list_type(self):
        data = ModelOutputValidator.validate_json('[1, 2, 3]')
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# validate_json with schema
# ---------------------------------------------------------------------------

class TestValidateJsonSchema:
    def test_schema_passes(self):
        try:
            import jsonschema  # type: ignore  # noqa: F401
        except ImportError:
            pytest.skip("jsonschema not installed")
        text = '{"name": "Alice", "age": 30}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = ModelOutputValidator.validate_json(text, schema=schema)
        assert data["name"] == "Alice"
        assert data["age"] == 30

    def test_schema_fails_on_type_mismatch(self):
        try:
            import jsonschema  # type: ignore  # noqa: F401
        except ImportError:
            pytest.skip("jsonschema not installed")
        text = '{"name": "Alice", "age": "thirty"}'  # age is str, not int
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        with pytest.raises(ValidationError):
            ModelOutputValidator.validate_json(text, schema=schema)

    def test_schema_missing_required_field(self):
        try:
            import jsonschema  # type: ignore  # noqa: F401
        except ImportError:
            pytest.skip("jsonschema not installed")
        text = '{"name": "Alice"}'  # missing age
        schema = {
            "type": "object",
            "required": ["name", "age"],
        }
        with pytest.raises(ValidationError):
            ModelOutputValidator.validate_json(text, schema=schema)

    def test_schema_skipped_when_jsonschema_missing(self, monkeypatch):
        # Force jsonschema import to fail.
        import sys
        monkeypatch.setitem(sys.modules, "jsonschema", None)
        text = '{"name": "Alice"}'
        schema = {"type": "object", "required": ["name"]}
        # Should not raise even if schema validation cannot run.
        data = ModelOutputValidator.validate_json(text, schema=schema)
        assert data == {"name": "Alice"}


# ---------------------------------------------------------------------------
# validate_non_empty
# ---------------------------------------------------------------------------

class TestValidateNonEmpty:
    def test_non_empty_text(self):
        assert ModelOutputValidator.validate_non_empty("hello") == "hello"

    def test_strips_whitespace(self):
        assert ModelOutputValidator.validate_non_empty("  hello  \n") == "hello"

    def test_empty_raises(self):
        with pytest.raises(ValidationError):
            ModelOutputValidator.validate_non_empty("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError):
            ModelOutputValidator.validate_non_empty("   \n  ")

    def test_none_raises(self):
        with pytest.raises((ValidationError, AttributeError)):
            ModelOutputValidator.validate_non_empty(None)  # type: ignore[arg-type]
