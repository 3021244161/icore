"""
icore.workflows.examples.weekly_report - Weekly report generation from DB.

A workflow that queries a database for the past week's tasks and generates
a weekly report via LLM:

    1. Query DB:     Fetch weekly tasks from the database
    2. LLM Generate:  Send the task list to the LLM to generate a narrative report
    3. Format:        Structure the report with metadata (date range, stats)

This demonstrates:
    - Database integration via TaskContext.get_db()
    - Using ctx.get_db() to obtain a DBManager connector
    - LLM generation based on DB query results
    - Custom task input classes
    - DAG composition: query_db -> generate_report -> format_report

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.weekly_report import WeeklyReportWorkflow

    wf = WeeklyReportWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # Inject managers before execution...
    result = await wf.execute(ctx, {
        "db_connection": "main_db",
        "user_id": "u123",
    })
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG
from icore.engine.registry import register_workflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Input Models
# ---------------------------------------------------------------------------

class QueryWeeklyTasksInput(BaseTaskInput):
    """Input for the DB query task."""

    db_connection: str = Field(
        default="main_db",
        description="Name of the registered database connection to use",
    )
    user_id: str = Field(
        default="",
        description="Filter tasks by user ID (empty = all users)",
    )
    days_back: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Number of days to look back from today",
    )


class GenerateReportInput(BaseTaskInput):
    """Input for the LLM report generation task."""

    tasks: list[dict[str, Any]] = Field(
        description="List of task records from the database",
    )
    date_range: str = Field(
        description="Human-readable date range for the report header",
    )
    user_id: str = Field(
        default="",
        description="User ID for the report (empty = all users)",
    )


class FormatReportInput(BaseTaskInput):
    """Input for the report formatting task."""

    report_text: str = Field(description="Generated report text from the LLM")
    date_range: str = Field(description="Date range string")
    task_count: int = Field(description="Number of tasks in the report")
    user_id: str = Field(default="", description="User ID")
    generated_at: str = Field(
        default="",
        description="Timestamp of generation (empty = now)",
    )


# ---------------------------------------------------------------------------
# Task: Query Weekly Tasks (DB)
# ---------------------------------------------------------------------------

@register_task("query_weekly_tasks")
class QueryWeeklyTasksTask(BaseTask):
    """
    Queries the database for tasks completed in the past week.

    This task demonstrates database integration: it obtains a DB
    connector via ``ctx.get_db()``, runs a parameterized SQL query,
    and returns the results as a list of task records.
    """

    name: ClassVar[str] = "query_weekly_tasks"
    description: ClassVar[str] = "Query database for weekly tasks"
    input_model: ClassVar[type[BaseTaskInput]] = QueryWeeklyTasksInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _db: Any  # DBManager reference

    async def prepare(self, ctx: TaskContext) -> None:
        """
        Obtain the DBManager from context.

        Note: ctx.get_db(name) returns a connector instance per-call.
        We store the DBManager reference itself for convenience.
        """
        # The DBManager is injected into ctx._db_manager.
        # We use ctx.get_db(name) at execute time to get a pooled connection.
        pass

    async def execute(
        self, ctx: TaskContext, inp: QueryWeeklyTasksInput
    ) -> BaseTaskOutput:
        """Query the database for tasks in the specified date range."""
        # Calculate date range
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=inp.days_back)
        date_range = f"{start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"

        # Build the SQL query (parameterized)
        if inp.user_id:
            sql = (
                "SELECT id, title, description, status, "
                "created_at, completed_at, assignee_id "
                "FROM tasks "
                "WHERE completed_at >= $1 "
                "AND completed_at <= $2 "
                "AND assignee_id = $3 "
                "ORDER BY completed_at DESC"
            )
            params: tuple[Any, ...] = (
                start_date,
                now,
                inp.user_id,
            )
        else:
            sql = (
                "SELECT id, title, description, status, "
                "created_at, completed_at, assignee_id "
                "FROM tasks "
                "WHERE completed_at >= $1 "
                "AND completed_at <= $2 "
                "ORDER BY completed_at DESC"
            )
            params = (start_date, now)

        # Execute query via the injected DBManager.
        # ctx.get_db(name) returns a pool-backed handle whose query() method
        # acquires/releases a connection automatically (no manual release needed).
        try:
            db = ctx.get_db(inp.db_connection)
        except RuntimeError as e:
            return BaseTaskOutput.failure(str(e))

        try:
            rows = await db.query(sql, params)

            logger.info(
                "Queried %d tasks from '%s' for %s (range: %s)",
                len(rows),
                inp.db_connection,
                f"user '{inp.user_id}'" if inp.user_id else "all users",
                date_range,
            )

            return BaseTaskOutput.success(
                tasks=rows,
                task_count=len(rows),
                date_range=date_range,
            )
        except Exception as e:
            logger.error("Database query failed: %s", e)
            return BaseTaskOutput.failure(
                f"DB query failed: {e}",
                date_range=date_range,
            )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Generate Report (LLM)
# ---------------------------------------------------------------------------

@register_task("generate_weekly_report")
class GenerateWeeklyReportTask(BaseTask):
    """
    Generates a narrative weekly report from task data using the LLM.

    Takes the list of DB task records and asks the LLM to produce
    a well-structured weekly report summarizing accomplishments,
    progress, and notable items.
    """

    name: ClassVar[str] = "generate_weekly_report"
    description: ClassVar[str] = "Generate weekly report text from task data via LLM"
    input_model: ClassVar[type[BaseTaskInput]] = GenerateReportInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    _model: Any

    async def prepare(self, ctx: TaskContext) -> None:
        """Obtain the model adapter."""
        self._model = ctx.get_model_adapter()

    async def execute(
        self, ctx: TaskContext, inp: GenerateReportInput
    ) -> BaseTaskOutput:
        """Generate a weekly report via LLM."""
        if not inp.tasks:
            return BaseTaskOutput.success(
                report_text="No tasks completed in the specified period.",
                task_count=0,
            )

        # Build a summary of tasks for the LLM prompt
        task_lines: list[str] = []
        for i, task in enumerate(inp.tasks, 1):
            title = task.get("title", "Untitled")
            status = task.get("status", "unknown")
            task_lines.append(f"{i}. [{status}] {title}")

        tasks_summary = "\n".join(task_lines)

        prompt = (
            f"Please generate a professional weekly work report based on "
            f"the following completed tasks.\n\n"
            f"Date range: {inp.date_range}\n"
            f"{'User: ' + inp.user_id if inp.user_id else 'All users'}\n\n"
            f"Completed tasks:\n{tasks_summary}\n\n"
            f"Please structure the report with:\n"
            f"1. Overview (summary of the week)\n"
            f"2. Key accomplishments\n"
            f"3. In-progress items\n"
            f"4. Next week's plan"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional weekly report writer. "
                    "Generate clear, concise, and well-structured reports."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._model.chat(messages=messages)
            report_text = response.get("content", "")

            logger.info(
                "Generated weekly report (%d chars) from %d tasks",
                len(report_text),
                len(inp.tasks),
            )

            # Echo date_range back so downstream tasks (format_report) can
            # access it without needing a direct edge from query_db.
            return BaseTaskOutput.success(
                report_text=report_text,
                task_count=len(inp.tasks),
                date_range=inp.date_range,
            )
        except Exception as e:
            logger.error("Report generation failed: %s", e)
            return BaseTaskOutput.failure(f"Report generation failed: {e}")

    async def cleanup(self, ctx: TaskContext) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Task: Format Report (no LLM)
# ---------------------------------------------------------------------------

@register_task("format_weekly_report")
class FormatWeeklyReportTask(BaseTask):
    """
    Formats the generated report with metadata.

    Adds a header with date range, generation timestamp, and task count
    to the LLM-generated report text.
    """

    name: ClassVar[str] = "format_weekly_report"
    description: ClassVar[str] = "Format weekly report with metadata header"
    input_model: ClassVar[type[BaseTaskInput]] = FormatReportInput
    output_model: ClassVar[type[BaseTaskOutput]] = BaseTaskOutput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(
        self, ctx: TaskContext, inp: FormatReportInput
    ) -> BaseTaskOutput:
        """Format the report with a metadata header."""
        generated_at = inp.generated_at or datetime.now(timezone.utc).isoformat()

        header = (
            f"{'=' * 60}\n"
            f"  Weekly Work Report\n"
            f"  Date Range: {inp.date_range}\n"
            f"  {'User: ' + inp.user_id if inp.user_id else 'Scope: All users'}\n"
            f"  Tasks: {inp.task_count}\n"
            f"  Generated: {generated_at}\n"
            f"{'=' * 60}\n\n"
        )

        formatted_report = header + inp.report_text

        logger.info(
            "Formatted weekly report (%d chars, header + %d chars body)",
            len(formatted_report),
            len(inp.report_text),
        )

        return BaseTaskOutput.success(
            report=formatted_report,
            report_length=len(formatted_report),
            task_count=inp.task_count,
            date_range=inp.date_range,
            generated_at=generated_at,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Workflow: Weekly Report
# ---------------------------------------------------------------------------

@register_workflow("weekly_report")
class WeeklyReportWorkflow(BaseWorkflow):
    """
    Multi-task workflow for weekly report generation.

    Pipeline:
        1. query_weekly_tasks:    Query DB for tasks in the past week
        2. generate_weekly_report:  Generate narrative report via LLM
        3. format_weekly_report:   Add metadata header and format

    DAG:
        query_db -> generate_report -> format_report

    This workflow demonstrates DB + LLM integration: the first task
    queries the database, the second uses the LLM to generate a report
    from the query results, and the third formats the output.
    """

    name: ClassVar[str] = "weekly_report"
    description: ClassVar[str] = (
        "Query DB for weekly tasks, generate a narrative report via LLM, "
        "and format with metadata"
    )

    def define(self) -> DAG:
        """Build the query_db -> generate_report -> format_report DAG."""
        dag = DAG()

        # Node 1: Query the database for weekly tasks
        dag.add_node(
            node_id="query_db",
            task_name="query_weekly_tasks",
        )

        # Node 2: Generate the report via LLM
        # Input builder: takes DB query results and constructs GenerateReportInput
        dag.add_node(
            node_id="generate_report",
            task_name="generate_weekly_report",
            input_builder=lambda params, upstream: GenerateReportInput(
                tasks=upstream["query_db"].data.get("tasks", []),
                date_range=upstream["query_db"].data.get("date_range", ""),
                user_id=params.get("user_id", ""),
            ),
        )

        # Node 3: Format the report with metadata
        # Input builder pulls date_range from generate_report's output (which
        # echoes it) rather than from query_db, because query_db is not a
        # direct predecessor of format_report in the DAG.
        dag.add_node(
            node_id="format_report",
            task_name="format_weekly_report",
            input_builder=lambda params, upstream: FormatReportInput(
                report_text=upstream["generate_report"].data.get("report_text", ""),
                date_range=upstream["generate_report"].data.get("date_range", ""),
                task_count=upstream["generate_report"].data.get("task_count", 0),
                user_id=params.get("user_id", ""),
            ),
        )

        # Linear dependency chain
        dag.add_edge("query_db", "generate_report")
        dag.add_edge("generate_report", "format_report")

        return dag
