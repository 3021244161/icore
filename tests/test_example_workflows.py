"""
End-to-end tests for the three example workflows.

These tests exercise the full pipeline (DAG -> executor -> tasks -> LLM/DB)
using the FakeModelAdapter (no network) and a fake DBManager. They verify:

    - DAG definition produces a valid topological order
    - Registration with @register_workflow works
    - End-to-end execution succeeds and produces expected output shape
    - LLM-dependent tasks (summarize_chunk, merge_summary, extract_entities,
      generate_weekly_report) receive the fake adapter via TaskContext
    - DB-dependent task (query_weekly_tasks) uses ctx.get_db() correctly
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from icore.core.task_context import TaskContext
from icore.engine.registry import workflow_registry
from tests.conftest import FakeModelAdapter, make_model_config


# ---------------------------------------------------------------------------
# Ensure example workflows are imported/registered
# ---------------------------------------------------------------------------

import icore.workflows.examples  # noqa: F401  (registers all examples)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx_with_model(model_manager) -> TaskContext:
    ctx = TaskContext(task_id="t-test", workflow_id="wf-test")
    ctx.set_model_manager(model_manager)
    return ctx


def _ctx_with_model_and_db(model_manager, db_manager) -> TaskContext:
    ctx = TaskContext(task_id="t-test", workflow_id="wf-test")
    ctx.set_model_manager(model_manager)
    ctx.set_db_manager(db_manager)
    return ctx


class _FakeDBManager:
    """Fake DBManager that returns canned rows for `query`."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows if rows is not None else []
        self.calls: list[tuple[str, str, tuple]] = []

    async def query(self, name, sql, params=None):
        self.calls.append((name, sql, params or ()))
        return list(self._rows)

    async def execute(self, name, sql, params=None):
        self.calls.append((name, sql, params or ()))
        return len(self._rows)

    def connection_ctx(self, name):
        mgr = self

        class _Ctx:
            async def __aenter__(self_):
                return mgr

            async def __aexit__(self_, *args):
                pass

        return _Ctx()


# ---------------------------------------------------------------------------
# Registration smoke test
# ---------------------------------------------------------------------------

class TestExampleRegistration:
    def test_all_three_workflows_registered(self):
        names = workflow_registry.list_workflows()
        assert "document_summary" in names
        assert "entity_extraction" in names
        assert "weekly_report" in names

    def test_document_summary_tasks_registered(self):
        from icore.core.registry import task_registry
        assert "text_chunker" in task_registry
        assert "summarize_chunk" in task_registry
        assert "merge_summary" in task_registry

    def test_entity_extraction_tasks_registered(self):
        from icore.core.registry import task_registry
        assert "extract_entities" in task_registry
        assert "normalize_entities" in task_registry
        assert "format_entities" in task_registry

    def test_weekly_report_tasks_registered(self):
        from icore.core.registry import task_registry
        assert "query_weekly_tasks" in task_registry
        assert "generate_weekly_report" in task_registry
        assert "format_weekly_report" in task_registry


# ---------------------------------------------------------------------------
# Document Summary workflow
# ---------------------------------------------------------------------------

class TestDocumentSummaryWorkflow:
    async def test_dag_validates(self):
        from icore.workflows.examples.document_summary import (
            DocumentSummaryWorkflow,
        )
        wf = DocumentSummaryWorkflow()
        assert wf.validate() is True

    async def test_end_to_end_short_document(self, fake_model_manager):
        from icore.workflows.examples.document_summary import (
            DocumentSummaryWorkflow,
        )

        wf = DocumentSummaryWorkflow()
        ctx = _ctx_with_model(fake_model_manager)

        # Short document - should produce 1 chunk
        doc = "This is a short document for testing summarization."
        result = await wf.execute(ctx, {"document": doc})

        assert result.is_success is True, f"Unexpected failure: {result.error}"
        # Final output from merge_summary node
        assert "summary" in result.data
        # FakeModelAdapter echoes "SUMMARY: <text>"
        assert "SUMMARY:" in result.data["summary"]
        assert result.data["merged_from"] >= 1

    async def test_chunking_produces_multiple_chunks(
        self, fake_model_manager
    ):
        from icore.workflows.examples.document_summary import (
            DocumentSummaryWorkflow,
        )

        # Build a fake adapter that records each call so we can count chunks
        call_count = {"n": 0}

        def responder(messages):
            call_count["n"] += 1
            return f"summary-{call_count['n']}"

        config = make_model_config("fake-model")
        adapter = FakeModelAdapter(config, responder=responder)
        # Replace the manager's adapter with our instrumented one
        fake_model_manager._adapters["fake-model"] = adapter

        wf = DocumentSummaryWorkflow()
        ctx = _ctx_with_model(fake_model_manager)

        # Long document -> multiple chunks
        doc = "Lorem ipsum. " * 500  # ~7000 chars
        result = await wf.execute(
            ctx,
            {"document": doc, "chunk_size": 1000, "overlap": 100},
        )

        assert result.is_success is True
        # Should have summarized multiple chunks then merged
        # call_count = num_chunks (summarize_chunk) + 1 (merge_summary)
        assert call_count["n"] >= 3
        assert result.data["merged_from"] >= 2

    async def test_empty_document_fails_gracefully(self, fake_model_manager):
        from icore.workflows.examples.document_summary import (
            DocumentSummaryWorkflow,
        )

        wf = DocumentSummaryWorkflow()
        ctx = _ctx_with_model(fake_model_manager)
        result = await wf.execute(ctx, {"document": ""})

        # The chunker returns failure, which should propagate
        assert result.is_success is False


# ---------------------------------------------------------------------------
# Entity Extraction workflow
# ---------------------------------------------------------------------------

class TestEntityExtractionWorkflow:
    async def test_dag_validates(self):
        from icore.workflows.examples.entity_extraction import (
            EntityExtractionWorkflow,
        )
        wf = EntityExtractionWorkflow()
        assert wf.validate() is True

    async def test_end_to_end_json_output(self, fake_model_manager):
        from icore.workflows.examples.entity_extraction import (
            EntityExtractionWorkflow,
        )

        # Build a fake adapter that returns valid JSON entities
        entities_payload = {
            "entities": [
                {"name": "Alice", "type": "Person", "mentions": ["Alice"]},
                {"name": "Acme Corp", "type": "Organization",
                 "mentions": ["Acme Corp"]},
            ],
            "relationships": [
                {"subject": "Alice", "predicate": "works_at",
                 "object": "Acme Corp"},
            ],
        }

        def responder(messages):
            return json.dumps(entities_payload)

        config = make_model_config("fake-model")
        adapter = FakeModelAdapter(config, responder=responder)
        fake_model_manager._adapters["fake-model"] = adapter

        wf = EntityExtractionWorkflow()
        ctx = _ctx_with_model(fake_model_manager)

        result = await wf.execute(
            ctx,
            {"text": "Alice works at Acme Corp."},
        )

        assert result.is_success is True, f"Failed: {result.error}"
        assert result.data["format"] == "json"
        assert result.data["entity_count"] == 2
        assert result.data["relationship_count"] == 1
        # Verify normalized entity names (case preserved)
        names = {e["name"] for e in result.data["result"]["entities"]}
        assert "Alice" in names
        assert "Acme Corp" in names

    async def test_cytoscape_format(self, fake_model_manager):
        from icore.workflows.examples.entity_extraction import (
            EntityExtractionWorkflow,
        )

        entities_payload = {
            "entities": [
                {"name": "Node1", "type": "person", "mentions": ["Node1"]},
                {"name": "Node2", "type": "person", "mentions": ["Node2"]},
            ],
            "relationships": [
                {"subject": "Node1", "predicate": "knows",
                 "object": "Node2"},
            ],
        }

        def responder(messages):
            return json.dumps(entities_payload)

        config = make_model_config("fake-model")
        adapter = FakeModelAdapter(config, responder=responder)
        fake_model_manager._adapters["fake-model"] = adapter

        wf = EntityExtractionWorkflow()
        ctx = _ctx_with_model(fake_model_manager)

        result = await wf.execute(
            ctx,
            {
                "text": "Node1 knows Node2.",
                "output_format": "cytoscape",
            },
        )

        assert result.is_success is True
        assert result.data["format"] == "cytoscape"
        assert result.data["node_count"] == 2
        assert result.data["edge_count"] == 1
        # Elements should have 2 nodes + 1 edge = 3 elements
        assert len(result.data["elements"]) == 3

    async def test_normalization_deduplicates(self, fake_model_manager):
        """Verify the normalize task dedupes entities by name (case-insensitive)."""
        from icore.workflows.examples.entity_extraction import (
            EntityExtractionWorkflow,
        )

        # Two entities with the same name differing only in case
        entities_payload = {
            "entities": [
                {"name": "alice", "type": "person", "mentions": ["alice"]},
                {"name": "Alice", "type": "Person", "mentions": ["Alice"]},
            ],
            "relationships": [],
        }

        def responder(messages):
            return json.dumps(entities_payload)

        config = make_model_config("fake-model")
        adapter = FakeModelAdapter(config, responder=responder)
        fake_model_manager._adapters["fake-model"] = adapter

        wf = EntityExtractionWorkflow()
        ctx = _ctx_with_model(fake_model_manager)

        result = await wf.execute(ctx, {"text": "alice Alice"})
        assert result.is_success is True
        # After dedup, only one entity should remain
        assert result.data["entity_count"] == 1


# ---------------------------------------------------------------------------
# Weekly Report workflow
# ---------------------------------------------------------------------------

class TestWeeklyReportWorkflow:
    async def test_dag_validates(self):
        from icore.workflows.examples.weekly_report import (
            WeeklyReportWorkflow,
        )
        wf = WeeklyReportWorkflow()
        assert wf.validate() is True

    async def test_end_to_end_with_db_rows(self, fake_model_manager):
        from icore.workflows.examples.weekly_report import (
            WeeklyReportWorkflow,
        )

        # Fake DB returns 2 task rows
        rows = [
            {
                "id": 1,
                "title": "Implement feature A",
                "description": "Built the new A module",
                "status": "done",
                "created_at": "2026-07-25T10:00:00Z",
                "completed_at": "2026-07-26T15:00:00Z",
                "assignee_id": "u123",
            },
            {
                "id": 2,
                "title": "Fix bug B",
                "description": "Fixed the B issue",
                "status": "done",
                "created_at": "2026-07-26T09:00:00Z",
                "completed_at": "2026-07-27T11:00:00Z",
                "assignee_id": "u123",
            },
        ]
        db_mgr = _FakeDBManager(rows=rows)

        wf = WeeklyReportWorkflow()
        ctx = _ctx_with_model_and_db(fake_model_manager, db_mgr)

        result = await wf.execute(
            ctx,
            {"db_connection": "main_db", "user_id": "u123"},
        )

        assert result.is_success is True, f"Failed: {result.error}"
        assert "report" in result.data
        assert result.data["task_count"] == 2
        # Report should contain header
        assert "Weekly Work Report" in result.data["report"]
        # FakeModelAdapter echoes "SUMMARY: ..." for the LLM call
        assert "SUMMARY:" in result.data["report"]

        # Verify DB was queried with the right connection name
        assert len(db_mgr.calls) == 1
        assert db_mgr.calls[0][0] == "main_db"

    async def test_empty_db_returns_no_tasks_message(
        self, fake_model_manager
    ):
        from icore.workflows.examples.weekly_report import (
            WeeklyReportWorkflow,
        )

        db_mgr = _FakeDBManager(rows=[])
        wf = WeeklyReportWorkflow()
        ctx = _ctx_with_model_and_db(fake_model_manager, db_mgr)

        result = await wf.execute(
            ctx,
            {"db_connection": "main_db"},
        )

        assert result.is_success is True
        # When tasks list is empty, generate_weekly_report returns early
        # with a default message
        assert "No tasks" in result.data["report"] or \
               result.data["task_count"] == 0

    async def test_user_filter_passed_to_sql(self, fake_model_manager):
        from icore.workflows.examples.weekly_report import (
            WeeklyReportWorkflow,
        )

        db_mgr = _FakeDBManager(rows=[])
        wf = WeeklyReportWorkflow()
        ctx = _ctx_with_model_and_db(fake_model_manager, db_mgr)

        await wf.execute(
            ctx,
            {"db_connection": "main_db", "user_id": "alice"},
        )

        # Verify the SQL params contain the user_id
        assert len(db_mgr.calls) == 1
        _, _, params = db_mgr.calls[0]
        assert "alice" in params
