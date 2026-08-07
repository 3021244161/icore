"""
Tests for icore.prompts - Prompt template management & versioning.

Covers:
    - ``PromptVariable``: type validation / conversion (str/int/float/bool/list),
      unknown type rejection
    - ``PromptTemplate.render``: normal render, missing required, default
      fallback, type coercion, literal brace escape, undeclared placeholder
    - ``PromptManager``: load_from_yaml, save + round-trip, list_versions,
      get_latest_version, render, reload, load_all, ab_test distribution,
      missing template / version errors, missing config dir
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from icore.exceptions import ValidationError
from icore.prompts import PromptManager, PromptTemplate, PromptVariable


# ---------------------------------------------------------------------------
# PromptVariable
# ---------------------------------------------------------------------------


class TestPromptVariable:
    def test_validate_str(self) -> None:
        v = PromptVariable(name="x", type="str")
        assert v.validate_value("hello") == "hello"

    def test_validate_int_from_str(self) -> None:
        v = PromptVariable(name="x", type="int")
        assert v.validate_value("42") == 42
        assert v.validate_value(42) == 42

    def test_validate_float_from_str(self) -> None:
        v = PromptVariable(name="x", type="float")
        assert v.validate_value("3.14") == pytest.approx(3.14)
        assert v.validate_value(3.14) == pytest.approx(3.14)

    def test_validate_bool_from_str(self) -> None:
        v = PromptVariable(name="x", type="bool")
        assert v.validate_value("true") is True
        assert v.validate_value("yes") is True
        assert v.validate_value("1") is True
        assert v.validate_value("false") is False
        assert v.validate_value("no") is False

    def test_validate_bool_from_int(self) -> None:
        v = PromptVariable(name="x", type="bool")
        assert v.validate_value(1) is True
        assert v.validate_value(0) is False

    def test_validate_list(self) -> None:
        v = PromptVariable(name="x", type="list")
        assert v.validate_value([1, 2, 3]) == [1, 2, 3]
        # tuple → list conversion
        assert v.validate_value((1, 2)) == [1, 2]

    def test_validate_unknown_type(self) -> None:
        v = PromptVariable(name="x", type="bogus")
        with pytest.raises(ValidationError):
            v.validate_value("hi")

    def test_validate_int_invalid_string(self) -> None:
        v = PromptVariable(name="x", type="int")
        with pytest.raises(ValidationError):
            v.validate_value("not a number")

    def test_validate_none_returns_none(self) -> None:
        v = PromptVariable(name="x", type="str")
        assert v.validate_value(None) is None

    def test_defaults(self) -> None:
        v = PromptVariable(name="x")
        assert v.type == "str"
        assert v.default is None
        assert v.required is True
        assert v.description == ""


# ---------------------------------------------------------------------------
# PromptTemplate.render
# ---------------------------------------------------------------------------


class TestPromptTemplateRender:
    def test_normal_render(self) -> None:
        tpl = PromptTemplate(
            name="greet",
            version=1,
            template="Hello {name}, you are {age} years old.",
            variables=[
                PromptVariable(name="name", type="str", required=True),
                PromptVariable(name="age", type="int", required=True),
            ],
        )
        out = tpl.render(name="Alice", age=30)
        assert "Alice" in out
        assert "30" in out

    def test_missing_required_raises(self) -> None:
        tpl = PromptTemplate(
            name="greet",
            version=1,
            template="Hello {name}",
            variables=[PromptVariable(name="name", type="str", required=True)],
        )
        with pytest.raises(ValidationError):
            tpl.render()  # name missing

    def test_default_fallback(self) -> None:
        tpl = PromptTemplate(
            name="greet",
            version=1,
            template="Hello {name}",
            variables=[
                PromptVariable(
                    name="name", type="str", default="World", required=False
                )
            ],
        )
        out = tpl.render()  # name not provided → default
        assert "World" in out

    def test_required_with_default_uses_provided(self) -> None:
        """If a var has both default and required=True, kwargs value wins."""
        tpl = PromptTemplate(
            name="greet",
            version=1,
            template="Hello {name}",
            variables=[
                PromptVariable(
                    name="name", type="str", default="World", required=True
                )
            ],
        )
        out = tpl.render(name="Alice")
        assert "Alice" in out

    def test_optional_no_default_keeps_placeholder(self) -> None:
        """Optional var without default → {name} kept as literal."""
        tpl = PromptTemplate(
            name="greet",
            version=1,
            template="Hello {name}",
            variables=[
                PromptVariable(name="name", type="str", required=False)
            ],
        )
        out = tpl.render()
        assert "{name}" in out

    def test_type_coercion_int_to_str(self) -> None:
        tpl = PromptTemplate(
            name="greet",
            version=1,
            template="Count: {n}",
            variables=[PromptVariable(name="n", type="str")],
        )
        out = tpl.render(n=42)
        assert "42" in out

    def test_literal_braces_escaped(self) -> None:
        """{{ and }} should render as literal { and }."""
        tpl = PromptTemplate(
            name="json",
            version=1,
            template='{{"key": "{value}"}}',
            variables=[PromptVariable(name="value", type="str")],
        )
        out = tpl.render(value="hello")
        assert out == '{"key": "hello"}'

    def test_undeclared_placeholder_preserved(self) -> None:
        """A {var} not declared in variables should remain in output."""
        tpl = PromptTemplate(
            name="x",
            version=1,
            template="Hello {declared} and {undeclared}",
            variables=[
                PromptVariable(name="declared", type="str", default="A")
            ],
        )
        out = tpl.render()
        assert "A" in out
        assert "{undeclared}" in out

    def test_template_with_no_variables(self) -> None:
        tpl = PromptTemplate(
            name="static",
            version=1,
            template="Just a static prompt.",
        )
        out = tpl.render()
        assert out == "Just a static prompt."

    def test_template_default_factory(self) -> None:
        """created_at should auto-fill."""
        tpl = PromptTemplate(name="x", version=1, template="hi")
        assert tpl.created_at is not None

    def test_invalid_version_rejected(self) -> None:
        with pytest.raises(Exception):
            PromptTemplate(name="x", version=0, template="hi")


# ---------------------------------------------------------------------------
# PromptManager - load_from_yaml / list_versions / get_latest_version
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_prompts_dir() -> Path:
    """Create a temp config/prompts dir with sample YAML files."""
    tmp = Path(tempfile.mkdtemp(prefix="icore_prompts_"))
    prompts_root = tmp / "prompts"
    prompts_root.mkdir()

    # summarizer/v1
    s1 = prompts_root / "summarizer"
    s1.mkdir()
    (s1 / "v1.yaml").write_text(
        "version: 1\n"
        "template: |\n"
        "  Summarize in {max_words} words:\n"
        "  {document}\n"
        "variables:\n"
        "  - name: max_words\n"
        "    type: int\n"
        "    default: 200\n"
        "    required: false\n"
        "  - name: document\n"
        "    type: str\n"
        "    required: true\n"
        "description: v1\n",
        encoding="utf-8",
    )

    # summarizer/v2
    (s1 / "v2.yaml").write_text(
        "version: 2\n"
        "template: |\n"
        "  Summarize (3 bullets): {document}\n"
        "variables:\n"
        "  - name: document\n"
        "    type: str\n"
        "    required: true\n"
        "description: v2\n",
        encoding="utf-8",
    )

    # rag_answer/v1
    r1 = prompts_root / "rag_answer"
    r1.mkdir()
    (r1 / "v1.yaml").write_text(
        "version: 1\n"
        'template: "Q: {question}\\nA: {answer}"\n'
        "variables:\n"
        "  - name: question\n"
        "    type: str\n"
        "    required: true\n"
        "  - name: answer\n"
        "    type: str\n"
        "    required: true\n",
        encoding="utf-8",
    )

    yield prompts_root

    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def manager(temp_prompts_dir: Path) -> PromptManager:
    return PromptManager(config_dir=temp_prompts_dir)


class TestPromptManagerLoad:
    def test_load_from_yaml_specific_version(self, manager: PromptManager) -> None:
        tpl = manager.load_from_yaml("summarizer", 1)
        assert tpl.name == "summarizer"
        assert tpl.version == 1
        assert tpl.template.startswith("Summarize")
        assert len(tpl.variables) == 2

    def test_load_from_yaml_latest_version(self, manager: PromptManager) -> None:
        tpl = manager.load_from_yaml("summarizer")  # version=None
        assert tpl.version == 2

    def test_load_from_yaml_missing_template(self, manager: PromptManager) -> None:
        with pytest.raises(ValidationError):
            manager.load_from_yaml("ghost")

    def test_load_from_yaml_missing_version(self, manager: PromptManager) -> None:
        with pytest.raises(ValidationError):
            manager.load_from_yaml("summarizer", 99)

    def test_load_from_yaml_caches(self, manager: PromptManager) -> None:
        tpl1 = manager.load_from_yaml("summarizer", 1)
        tpl2 = manager.load_from_yaml("summarizer", 1)
        # Same instance (cached)
        assert tpl1 is tpl2

    def test_load_rag_answer(self, manager: PromptManager) -> None:
        tpl = manager.load_from_yaml("rag_answer", 1)
        assert tpl.name == "rag_answer"
        assert tpl.version == 1


# ---------------------------------------------------------------------------
# PromptManager - save / round-trip
# ---------------------------------------------------------------------------


class TestPromptManagerSave:
    def test_save_creates_file(self, temp_prompts_dir: Path) -> None:
        # Use a fresh subdir to avoid clashing with existing files
        mgr = PromptManager(config_dir=temp_prompts_dir)
        tpl = PromptTemplate(
            name="greet",
            version=1,
            template="Hello {name}",
            variables=[
                PromptVariable(name="name", type="str", required=True)
            ],
            description="test greet",
            tags=["test"],
        )
        path = mgr.save(tpl)
        assert path.exists()
        assert path.name == "v1.yaml"
        assert path.parent.name == "greet"

    def test_save_round_trip(self, temp_prompts_dir: Path) -> None:
        mgr = PromptManager(config_dir=temp_prompts_dir)
        tpl = PromptTemplate(
            name="roundtrip",
            version=3,
            template="Hi {name}, {greeting}",
            variables=[
                PromptVariable(name="name", type="str", required=True),
                PromptVariable(
                    name="greeting", type="str", default="hi", required=False
                ),
            ],
            description="round trip test",
            tags=["rt"],
        )
        mgr.save(tpl)
        # Load back from disk
        loaded = mgr.load_from_yaml("roundtrip", 3)
        assert loaded.version == 3
        assert loaded.template == tpl.template
        assert loaded.description == "round trip test"
        assert loaded.tags == ["rt"]
        assert len(loaded.variables) == 2
        assert loaded.variables[0].name == "name"
        assert loaded.variables[1].default == "hi"

    def test_save_overwrites_existing(self, temp_prompts_dir: Path) -> None:
        mgr = PromptManager(config_dir=temp_prompts_dir)
        tpl_v1 = PromptTemplate(
            name="ow",
            version=1,
            template="v1",
        )
        mgr.save(tpl_v1)
        # Save again with same version, different content
        tpl_v1b = PromptTemplate(
            name="ow",
            version=1,
            template="v1 updated",
        )
        mgr.save(tpl_v1b)
        # Clear cache to force disk read
        mgr._cache.clear()
        loaded = mgr.load_from_yaml("ow", 1)
        assert loaded.template == "v1 updated"

    def test_save_invalid_version_raises(self, temp_prompts_dir: Path) -> None:
        """Pydantic Field(ge=1) rejects version=0 at construction time;
        save() also defends against bad versions via model_construct bypass."""
        # Verify Pydantic rejects version=0 at construction
        with pytest.raises(Exception):
            PromptTemplate(name="x", version=0, template="hi")
        # Verify save() also defends (via model_construct to bypass Pydantic)
        mgr = PromptManager(config_dir=temp_prompts_dir)
        tpl_force = PromptTemplate.model_construct(
            name="x", version=0, template="hi"
        )
        with pytest.raises(ValidationError):
            mgr.save(tpl_force)


# ---------------------------------------------------------------------------
# PromptManager - list_versions / get_latest_version / load_all
# ---------------------------------------------------------------------------


class TestPromptManagerVersions:
    def test_list_versions_sorted(self, manager: PromptManager) -> None:
        versions = manager.list_versions("summarizer")
        assert versions == [1, 2]

    def test_list_versions_unknown_name(self, manager: PromptManager) -> None:
        assert manager.list_versions("ghost") == []

    def test_get_latest_version(self, manager: PromptManager) -> None:
        assert manager.get_latest_version("summarizer") == 2
        assert manager.get_latest_version("rag_answer") == 1

    def test_get_latest_version_unknown(self, manager: PromptManager) -> None:
        assert manager.get_latest_version("ghost") is None

    def test_load_all_returns_all_templates(self, manager: PromptManager) -> None:
        all_tpls = manager.load_all()
        assert "summarizer" in all_tpls
        assert "rag_answer" in all_tpls
        assert len(all_tpls["summarizer"]) == 2
        assert len(all_tpls["rag_answer"]) == 1
        # Sorted by version
        assert [t.version for t in all_tpls["summarizer"]] == [1, 2]

    def test_load_all_clears_cache(self, manager: PromptManager) -> None:
        # Load v1 to populate cache
        manager.load_from_yaml("summarizer", 1)
        assert "summarizer" in manager._cache
        # Modify underlying file
        s1 = manager.config_dir / "summarizer" / "v3.yaml"
        s1.write_text(
            "version: 3\ntemplate: v3\n",
            encoding="utf-8",
        )
        # load_all should pick up the new file
        all_tpls = manager.load_all()
        assert any(t.version == 3 for t in all_tpls["summarizer"])


# ---------------------------------------------------------------------------
# PromptManager - render
# ---------------------------------------------------------------------------


class TestPromptManagerRender:
    def test_render_latest(self, manager: PromptManager) -> None:
        out = manager.render("summarizer", document="hello world")
        # v2 template contains "3 bullets"
        assert "3 bullets" in out
        assert "hello world" in out

    def test_render_specific_version(self, manager: PromptManager) -> None:
        out = manager.render("summarizer", version=1, document="hello world")
        # v1 template contains "Summarize in"
        assert "Summarize in" in out
        # default max_words=200
        assert "200" in out

    def test_render_with_override(self, manager: PromptManager) -> None:
        out = manager.render(
            "summarizer", version=1, document="x", max_words=50
        )
        assert "50" in out

    def test_render_missing_required(self, manager: PromptManager) -> None:
        with pytest.raises(ValidationError):
            manager.render("summarizer", version=1)  # missing document


# ---------------------------------------------------------------------------
# PromptManager - reload
# ---------------------------------------------------------------------------


class TestPromptManagerReload:
    def test_reload_picks_up_new_files(self, manager: PromptManager) -> None:
        # Initially no v3
        assert manager.get_latest_version("summarizer") == 2
        # Add v3
        v3_path = manager.config_dir / "summarizer" / "v3.yaml"
        v3_path.write_text(
            "version: 3\ntemplate: v3 new content\n",
            encoding="utf-8",
        )
        # Reload
        manager.reload()
        # Now latest is 3
        assert manager.get_latest_version("summarizer") == 3

    def test_reload_clears_cache(self, manager: PromptManager) -> None:
        # Load to populate cache
        manager.load_from_yaml("summarizer", 1)
        assert manager._cache
        manager.reload()
        # After reload, cache is repopulated from disk
        assert "summarizer" in manager._cache


# ---------------------------------------------------------------------------
# PromptManager - ab_test
# ---------------------------------------------------------------------------


class TestPromptManagerABTest:
    async def test_ab_test_returns_valid_variant(self, manager: PromptManager) -> None:
        tpl, label = await manager.ab_test("summarizer", v1=1, v2=2, split=0.5)
        assert label in ("v1", "v2")
        assert tpl.version in (1, 2)

    async def test_ab_test_split_0_always_v2(self, manager: PromptManager) -> None:
        """split=0 → P(v1) = 0 → always v2."""
        for _ in range(10):
            _, label = await manager.ab_test("summarizer", v1=1, v2=2, split=0.0)
            assert label == "v2"

    async def test_ab_test_split_1_always_v1(self, manager: PromptManager) -> None:
        """split=1 → P(v1) = 1 → always v1."""
        for _ in range(10):
            _, label = await manager.ab_test("summarizer", v1=1, v2=2, split=1.0)
            assert label == "v1"

    async def test_ab_test_invalid_split(self, manager: PromptManager) -> None:
        with pytest.raises(ValueError):
            await manager.ab_test("summarizer", v1=1, v2=2, split=1.5)
        with pytest.raises(ValueError):
            await manager.ab_test("summarizer", v1=1, v2=2, split=-0.1)

    async def test_ab_test_same_version_raises(self, manager: PromptManager) -> None:
        with pytest.raises(ValueError):
            await manager.ab_test("summarizer", v1=1, v2=1)

    async def test_ab_test_missing_version(self, manager: PromptManager) -> None:
        with pytest.raises(ValidationError):
            await manager.ab_test("summarizer", v1=1, v2=99)

    async def test_ab_test_distribution(self, manager: PromptManager) -> None:
        """With split=0.5, both variants should appear ~50% over 200 trials."""
        random.seed(42)
        labels: list[str] = []
        for _ in range(200):
            _, label = await manager.ab_test("summarizer", v1=1, v2=2, split=0.5)
            labels.append(label)
        counts = Counter(labels)
        # Both variants should appear at least 30% of the time (statistical cushion)
        assert counts["v1"] >= 60
        assert counts["v2"] >= 60


# ---------------------------------------------------------------------------
# PromptManager - edge cases
# ---------------------------------------------------------------------------


class TestPromptManagerEdgeCases:
    def test_missing_config_dir_returns_empty(self, tmp_path: Path) -> None:
        """load_all on a non-existent dir returns empty dict (no error)."""
        mgr = PromptManager(config_dir=tmp_path / "nonexistent")
        assert mgr.load_all() == {}

    def test_missing_config_dir_load_raises(self, tmp_path: Path) -> None:
        """load_from_yaml on a non-existent dir raises ValidationError."""
        mgr = PromptManager(config_dir=tmp_path / "nonexistent")
        with pytest.raises(ValidationError):
            mgr.load_from_yaml("ghost")

    def test_load_malformed_yaml_raises(self, temp_prompts_dir: Path) -> None:
        # Write a YAML that's not a mapping (it's a list)
        bad = temp_prompts_dir / "bad"
        bad.mkdir()
        (bad / "v1.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
        mgr = PromptManager(config_dir=temp_prompts_dir)
        with pytest.raises(ValidationError):
            mgr.load_from_yaml("bad", 1)

    def test_load_yaml_missing_version_field(self, temp_prompts_dir: Path) -> None:
        bad = temp_prompts_dir / "no_version"
        bad.mkdir()
        (bad / "v1.yaml").write_text(
            "template: hi\n", encoding="utf-8"
        )
        mgr = PromptManager(config_dir=temp_prompts_dir)
        with pytest.raises(ValidationError):
            mgr.load_from_yaml("no_version", 1)

    def test_config_dir_property(self, temp_prompts_dir: Path) -> None:
        mgr = PromptManager(config_dir=temp_prompts_dir)
        assert mgr.config_dir == temp_prompts_dir
