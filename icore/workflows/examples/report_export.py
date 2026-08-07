"""
icore.workflows.examples.report_export - Report generation with object store export.

A workflow that generates an analytic report (JSON / HTML / CSV) and
stores it in the object store (MinIO / S3), returning a presigned URL
so the caller can download it directly without hitting the API again.

DAG:
    analyze_data ──► generate_report ──► upload_to_store

This demonstrates:
    - ObjectStore integration via TaskContext.get_objectstore()
    - Presigned URL generation for client-side download
    - Workflow result returning a file URL instead of inline data
    - Graceful fallback when no objectstore is configured

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.report_export import ReportExportWorkflow

    wf = ReportExportWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # Inject managers + objectstore before execution...
    result = await wf.execute(ctx, {
        "data": [{"month": "Jan", "sales": 12000}, ...],
        "format": "json",          # "json" | "csv" | "html"
        "bucket": "reports",
        "key_prefix": "monthly/",
    })
    # result.data["file_url"] → presigned URL (valid 24h)
    # result.data["object_ref"] → ObjectRef {bucket, key, size_bytes}
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
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
# Input models
# ---------------------------------------------------------------------------

class AnalyzeDataInput(BaseTaskInput):
    data: list[dict[str, Any]] = Field(description="Raw data rows (list of dicts)")
    aggregation: str = Field(default="sum", description="Aggregation: sum, avg, count")


class GenerateReportInput(BaseTaskInput):
    analysis: dict[str, Any] = Field(description="Analyzed results")
    format: str = Field(default="json", description="Output format: json, csv, html")
    title: str = Field(default="Report", description="Report title")


class UploadToStoreInput(BaseTaskInput):
    report_bytes: bytes = Field(description="Rendered report bytes")
    content_type: str = Field(description="MIME type of the report")
    extension: str = Field(description="File extension (without dot)")
    bucket: str = Field(description="Object store bucket")
    key_prefix: str = Field(default="", description="Key prefix")


# ---------------------------------------------------------------------------
# Task: Analyze Data
# ---------------------------------------------------------------------------

@register_task("analyze_data")
class AnalyzeDataTask(BaseTask):
    name: ClassVar[str] = "analyze_data"
    description: ClassVar[str] = "Aggregate raw tabular data"
    input_model: ClassVar[type[BaseTaskInput]] = AnalyzeDataInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: AnalyzeDataInput) -> BaseTaskOutput:
        if not inp.data:
            return BaseTaskOutput.failure("No data provided")

        keys = inp.data[0].keys()
        result: dict[str, Any] = {"row_count": len(inp.data), "columns": list(keys)}

        for col in keys:
            values = []
            for row in inp.data:
                v = row.get(col)
                if isinstance(v, (int, float)):
                    values.append(v)
            if not values:
                continue
            if inp.aggregation == "sum":
                result[f"{col}_sum"] = sum(values)
            elif inp.aggregation == "avg":
                result[f"{col}_avg"] = sum(values) / len(values)
            else:
                result[f"{col}_count"] = len(values)

        return BaseTaskOutput.success(**result)

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Generate Report
# ---------------------------------------------------------------------------

@register_task("generate_report")
class GenerateReportTask(BaseTask):
    name: ClassVar[str] = "generate_report"
    description: ClassVar[str] = "Render analyzed data as JSON / CSV / HTML"
    input_model: ClassVar[type[BaseTaskInput]] = GenerateReportInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: GenerateReportInput) -> BaseTaskOutput:
        fmt = inp.format.lower()
        if fmt == "json":
            content, ct, ext = self._render_json(inp)
        elif fmt == "csv":
            content, ct, ext = self._render_csv(inp)
        elif fmt == "html":
            content, ct, ext = self._render_html(inp)
        else:
            return BaseTaskOutput.failure(f"Unsupported format: {fmt}")

        return BaseTaskOutput.success(
            report_bytes=content,
            content_type=ct,
            extension=ext,
            size_bytes=len(content),
        )

    def _render_json(self, inp: GenerateReportInput) -> tuple[bytes, str, str]:
        body = {
            "title": inp.title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis": inp.analysis,
        }
        return json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8"), \
            "application/json", "json"

    def _render_csv(self, inp: GenerateReportInput) -> tuple[bytes, str, str]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        for k, v in inp.analysis.items():
            writer.writerow([k, v])
        return buf.getvalue().encode("utf-8"), "text/csv", "csv"

    def _render_html(self, inp: GenerateReportInput) -> tuple[bytes, str, str]:
        rows = "\n".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in inp.analysis.items()
        )
        html = (
            f"<!DOCTYPE html><html><head><title>{inp.title}</title></head>"
            f"<body><h1>{inp.title}</h1><table border='1'>{rows}</table>"
            f"<p>Generated: {datetime.now(timezone.utc).isoformat()}</p>"
            f"</body></html>"
        )
        return html.encode("utf-8"), "text/html", "html"

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Upload to Object Store
# ---------------------------------------------------------------------------

@register_task("upload_to_store")
class UploadToStoreTask(BaseTask):
    name: ClassVar[str] = "upload_to_store"
    description: ClassVar[str] = "Upload rendered report to object store and return presigned URL"
    input_model: ClassVar[type[BaseTaskInput]] = UploadToStoreInput

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: UploadToStoreInput) -> BaseTaskOutput:
        if not ctx.has_objectstore():
            # Graceful fallback: return inline data when no store configured.
            return BaseTaskOutput.success(
                file_url=None,
                file_size=len(inp.report_bytes),
                content_type=inp.content_type,
                inline_data_b64=__import__("base64").b64encode(inp.report_bytes).decode("ascii"),
                note="Object store not configured; returning inline base64.",
            )

        store = ctx.get_objectstore()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"{inp.key_prefix}report-{ts}.{inp.extension}"

        ref = await store.put(
            bucket=inp.bucket,
            key=key,
            data=inp.report_bytes,
            content_type=inp.content_type,
        )

        # Generate a presigned URL valid for 24 hours.
        url = await store.presigned_get_url(
            bucket=inp.bucket,
            key=key,
            expires_seconds=86400,
        )

        logger.info(
            "Report uploaded to s3://%s/%s (%d bytes)",
            inp.bucket, key, ref.size_bytes,
        )
        return BaseTaskOutput.success(
            file_url=url,
            file_size=ref.size_bytes,
            content_type=ref.content_type,
            object_ref={
                "bucket": ref.bucket,
                "key": ref.key,
                "uri": ref.uri,
            },
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@register_workflow("report_export")
class ReportExportWorkflow(BaseWorkflow):
    """
    Generate a report from raw data and store it in MinIO / S3.

    Returns a presigned URL that the caller can use to download the
    report directly from the object store.
    """

    name: ClassVar[str] = "report_export"
    description: ClassVar[str] = (
        "Analyze raw data, generate a report (JSON/CSV/HTML), "
        "upload to object store, return presigned download URL"
    )

    def define(self) -> DAG:
        dag = DAG()

        dag.add_node("analyze", task_name="analyze_data",
                     input_builder=lambda params, up:
                         AnalyzeDataInput(data=params["data"], aggregation=params.get("aggregation", "sum")))

        dag.add_node("generate", task_name="generate_report",
                     input_builder=lambda params, up:
                         GenerateReportInput(
                             analysis=up["analyze"].data,
                             format=params.get("format", "json"),
                             title=params.get("title", "Report"),
                         ))

        dag.add_node("upload", task_name="upload_to_store",
                     input_builder=lambda params, up:
                         UploadToStoreInput(
                             report_bytes=up["generate"].data["report_bytes"],
                             content_type=up["generate"].data["content_type"],
                             extension=up["generate"].data["extension"],
                             bucket=params.get("bucket", "icore"),
                             key_prefix=params.get("key_prefix", ""),
                         ))

        dag.add_edge("analyze", "generate")
        dag.add_edge("generate", "upload")

        return dag


__all__ = ["ReportExportWorkflow"]
