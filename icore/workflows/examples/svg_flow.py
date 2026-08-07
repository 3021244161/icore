"""
icore.workflows.examples.svg_flow - 流程图 SVG 生成 + 视觉自检工作流。

处理逻辑：
    1. 分析文本提到的处理流程（步骤/分支/循环）
    2. 调用 LLM 生成 SVG 流程图代码（提示词从 config/prompts/svg_draw/*.j2 加载）
    3. 把 SVG 渲染成 PNG 图片，**用视觉方式**（多模态模型）检查
       是否有线条重叠、节点重叠、文字溢出等问题（提示词 svg_inspect/*.j2）
    4. 若检查未通过，把检查建议 + SVG 传回"画图"模型继续修（提示词 svg_fix/*.j2），
       循环直到通过或达到最大轮次

所有提示词均从 .j2 配置文件加载（config/prompts/ 目录），不在代码中硬编码。
修改提示词只需改目录里的 .j2 文件，无需改代码。

Usage:
    from icore.core.task_context import TaskContext
    from icore.workflows.examples.svg_flow import SVGFlowWorkflow

    wf = SVGFlowWorkflow()
    ctx = TaskContext(task_id="t1", workflow_id="w1")
    # 注入 model_manager + media_processor 后执行：
    result = await wf.execute(ctx, {
        "text": "用户下单后，系统先校验库存，有货则扣减库存并生成订单，"
                "无货则通知用户缺货，最后发送确认邮件。",
        "max_rounds": 3,
    })
    # result.data["svg_code"]      → 最终 SVG
    # result.data["rounds"]        → 实际迭代轮数
    # result.data["passed"]        → 检查是否通过
    # result.data["inspect_report"]→ 最后一次检查报告
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
from typing import Any, ClassVar, Optional

from pydantic import Field

from icore.core.base_task import BaseTask
from icore.core.models import BaseTaskInput, BaseTaskOutput
from icore.core.registry import register_task
from icore.core.task_context import TaskContext
from icore.engine.base_workflow import BaseWorkflow
from icore.engine.dag import DAG
from icore.engine.registry import register_workflow
from icore.exceptions import ValidationError
from icore.media import MediaFile, MediaSource, MediaType
from icore.prompts import PromptManager

logger = logging.getLogger(__name__)

# 默认 PromptManager 配置目录（与 config.py 默认一致）
_PROMPT_DIR = "config/prompts"


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------

class SVGFlowInput(BaseTaskInput):
    text: str = Field(description="描述处理流程的文本")
    max_rounds: int = Field(
        default=3, ge=1, le=10,
        description="最大 画图→检查→修复 迭代轮数",
    )
    model_id: Optional[str] = Field(
        default=None, description="画图模型（默认用 ctx 默认模型）"
    )


# ---------------------------------------------------------------------------
# Task: Draw SVG (LLM)
# ---------------------------------------------------------------------------

@register_task("svg_draw")
class SVGDrawTask(BaseTask):
    """根据流程文本生成 SVG 流程图代码（提示词来自 svg_draw/*.j2）。"""

    name: ClassVar[str] = "svg_draw"
    description: ClassVar[str] = "Generate SVG flow chart from process text"
    input_model: ClassVar[type[BaseTaskInput]] = BaseTaskInput

    def __init__(self, prompt_manager: Optional[PromptManager] = None) -> None:
        super().__init__()
        self._pm = prompt_manager or PromptManager(_PROMPT_DIR)

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: BaseTaskInput) -> BaseTaskOutput:
        text = getattr(inp, "text", "")
        if not text:
            return BaseTaskOutput.failure("text is required")

        # 从 .j2 配置文件加载提示词（不在代码中硬编码）
        prompt = self._pm.render_j2(
            "svg_draw", version=None, text=text
        )

        adapter = ctx.get_model_adapter()
        response = await adapter.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=8000,
        )
        svg_code = response.get("content", "").strip()
        if not svg_code:
            # 尝试从 reasoning_content 兜底（推理模型可能把答案放这里）
            svg_code = response.get("reasoning_content", "").strip()

        svg_code = _extract_svg(svg_code)
        if not svg_code:
            return BaseTaskOutput.failure(
                "Model returned no valid SVG code",
                raw_response=response.get("content", "")[:2000],
            )

        return BaseTaskOutput.success(svg_code=svg_code)

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Inspect SVG via vision (multimodal)
# ---------------------------------------------------------------------------

@register_task("svg_inspect")
class SVGInspectTask(BaseTask):
    """
    把 SVG 渲染成 PNG，用视觉模型检查布局问题。

    提示词来自 svg_inspect/*.j2。输出 JSON：
        {"passed": bool, "issues": [...], "suggestions": str}
    """

    name: ClassVar[str] = "svg_inspect"
    description: ClassVar[str] = "Visually inspect SVG layout quality"
    input_model: ClassVar[type[BaseTaskInput]] = BaseTaskInput

    def __init__(self, prompt_manager: Optional[PromptManager] = None) -> None:
        super().__init__()
        self._pm = prompt_manager or PromptManager(_PROMPT_DIR)

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: BaseTaskInput) -> BaseTaskOutput:
        svg_code = getattr(inp, "svg_code", "")
        if not svg_code:
            return BaseTaskOutput.failure("svg_code is required")

        # 1. 渲染 SVG → PNG bytes
        png_bytes = await _render_svg_to_png(svg_code)
        if png_bytes is None:
            # 渲染失败：无法视觉检查，保守起见视为不通过，把错误塞进建议
            return BaseTaskOutput.success(
                passed=False,
                issues=[{
                    "type": "render_error",
                    "node_id": None,
                    "description": "SVG 无法渲染成图片（可能语法错误）",
                }],
                suggestions="SVG 代码无法渲染，请重新生成一个语法正确的 SVG。",
                rendered=False,
            )

        # 2. 构造 MediaFile（图片 bytes）
        media_file = MediaFile(
            media_type=MediaType.IMAGE,
            source_type=MediaSource.BYTES,
            source=png_bytes,
            metadata={"mime_type": "image/png", "extension": ".png"},
        )

        # 3. 视觉模型检查（多模态）
        prompt = self._pm.render_j2("svg_inspect", version=None)
        try:
            adapter = ctx.get_model_adapter()
            if getattr(adapter, "supports_vision", False):
                response = await adapter.chat_with_media(
                    prompt=prompt,
                    media_files=[media_file],
                    temperature=0.2,
                    max_tokens=2000,
                )
            else:
                # 非视觉模型：把 SVG 源码作为文本传入（退路）
                response = await adapter.chat(
                    messages=[{
                        "role": "user",
                        "content": prompt + "\n\nSVG code:\n" + svg_code[:8000],
                    }],
                    temperature=0.2,
                    max_tokens=2000,
                )
        except NotImplementedError:
            response = await adapter.chat(
                messages=[{
                    "role": "user",
                    "content": prompt + "\n\nSVG code:\n" + svg_code[:8000],
                }],
                temperature=0.2,
                max_tokens=2000,
            )

        raw = response.get("content", "") or response.get("reasoning_content", "")
        parsed = _parse_inspect_json(raw)

        # 解析失败 → 保守：视为不通过
        if parsed is None:
            return BaseTaskOutput.success(
                passed=False,
                issues=[{
                    "type": "parse_error",
                    "node_id": None,
                    "description": "检查模型未返回可解析的 JSON",
                }],
                suggestions="检查模型输出格式错误，请重新检查。",
                raw_report=raw[:2000],
                rendered=True,
            )

        return BaseTaskOutput.success(
            passed=bool(parsed.get("passed", False)),
            issues=parsed.get("issues", []),
            suggestions=parsed.get("suggestions", ""),
            raw_report=raw[:2000],
            rendered=True,
        )

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Task: Fix SVG (LLM, 循环修复)
# ---------------------------------------------------------------------------

@register_task("svg_fix")
class SVGFixTask(BaseTask):
    """根据检查建议修复 SVG（提示词来自 svg_fix/*.j2）。"""

    name: ClassVar[str] = "svg_fix"
    description: ClassVar[str] = "Fix SVG based on inspection feedback"
    input_model: ClassVar[type[BaseTaskInput]] = BaseTaskInput

    def __init__(self, prompt_manager: Optional[PromptManager] = None) -> None:
        super().__init__()
        self._pm = prompt_manager or PromptManager(_PROMPT_DIR)

    async def prepare(self, ctx: TaskContext) -> None:
        pass

    async def execute(self, ctx: TaskContext, inp: BaseTaskInput) -> BaseTaskOutput:
        svg_code = getattr(inp, "svg_code", "")
        feedback = getattr(inp, "feedback", "")
        if not svg_code:
            return BaseTaskOutput.failure("svg_code is required")

        prompt = self._pm.render_j2(
            "svg_fix", version=None, svg_code=svg_code, feedback=feedback
        )

        adapter = ctx.get_model_adapter()
        response = await adapter.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=8000,
        )
        new_svg = response.get("content", "").strip()
        if not new_svg:
            new_svg = response.get("reasoning_content", "").strip()

        new_svg = _extract_svg(new_svg)
        if not new_svg:
            # 修复失败：返回原 SVG + 标记未通过（避免死循环吞掉结果）
            return BaseTaskOutput.success(
                svg_code=svg_code,
                fix_failed=True,
                raw_response=response.get("content", "")[:2000],
            )

        return BaseTaskOutput.success(svg_code=new_svg, fix_failed=False)

    async def cleanup(self, ctx: TaskContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@register_workflow("svg_flow")
class SVGFlowWorkflow(BaseWorkflow):
    """
    流程图 SVG 生成 + 视觉自检工作流。

    DAG 结构（循环在 execute() 中手动驱动，因为画图→检查→修复是动态循环）：
        draw ──► inspect ──► (passed? 结束 | 否 → fix → inspect ...)
    """

    name: ClassVar[str] = "svg_flow"
    description: ClassVar[str] = (
        "Generate a flow-chart SVG from process text, visually inspect "
        "it with a multimodal model, and auto-fix layout issues"
    )

    def define(self) -> DAG:
        dag = DAG()
        # 画图
        dag.add_node(
            "draw", task_name="svg_draw",
            input_builder=lambda params, up: BaseTaskInput(text=params["text"]),
        )
        # 视觉检查
        dag.add_node(
            "inspect", task_name="svg_inspect",
            input_builder=lambda params, up: BaseTaskInput(
                svg_code=up["draw"].data.get("svg_code", "")
            ),
        )
        dag.add_edge("draw", "inspect")
        return dag

    async def execute(
        self,
        ctx: TaskContext,
        params: dict[str, Any],
        *,
        execution_id: str | None = None,
        workflow_name: str = "",
    ) -> BaseTaskOutput:
        """驱动 画图→检查→修复 循环。

        ``execution_id`` / ``workflow_name`` 是 v0.6 持久化参数，
        本工作流为纯 LLM 循环（不走 DAG executor），接受但忽略。
        """
        text = params.get("text", "")
        if not text:
            return BaseTaskOutput.failure("text is required")
        max_rounds = int(params.get("max_rounds", 3))
        max_rounds = max(1, min(max_rounds, 10))

        draw = SVGDrawTask()
        inspect = SVGInspectTask()
        fix = SVGFixTask()

        # --- Round 1: 画图 ---
        draw_inp = BaseTaskInput(text=text)
        await draw.prepare(ctx)
        draw_out = await draw.execute(ctx, draw_inp)
        await draw.cleanup(ctx)
        if not draw_out.is_success:
            return draw_out
        svg_code = draw_out.data.get("svg_code", "")

        rounds = 1
        last_report: dict[str, Any] = {}

        while rounds <= max_rounds:
            # --- 视觉检查 ---
            insp_inp = BaseTaskInput(svg_code=svg_code)
            await inspect.prepare(ctx)
            insp_out = await inspect.execute(ctx, insp_inp)
            await inspect.cleanup(ctx)

            last_report = insp_out.data
            passed = bool(insp_out.data.get("passed", False))

            if passed:
                return BaseTaskOutput.success(
                    svg_code=svg_code,
                    rounds=rounds,
                    passed=True,
                    inspect_report=last_report,
                )

            # --- 修复 ---
            if rounds >= max_rounds:
                break

            feedback = (
                insp_out.data.get("suggestions", "")
                or json.dumps(insp_out.data.get("issues", []), ensure_ascii=False)
            )
            fix_inp = BaseTaskInput(svg_code=svg_code, feedback=feedback)
            await fix.prepare(ctx)
            fix_out = await fix.execute(ctx, fix_inp)
            await fix.cleanup(ctx)

            if not fix_out.is_success:
                # 修复任务本身失败 → 返回当前 SVG 与状态
                return BaseTaskOutput.failure(
                    "SVG fix task failed",
                    svg_code=svg_code,
                    rounds=rounds,
                    passed=False,
                    inspect_report=last_report,
                )

            new_svg = fix_out.data.get("svg_code", svg_code)
            if fix_out.data.get("fix_failed") or new_svg == svg_code:
                # 修复没有产生新代码 → 停止循环，避免死循环
                logger.warning("SVG fix produced no change; stopping loop")
                break
            svg_code = new_svg
            rounds += 1

        # 到达最大轮次仍未通过
        return BaseTaskOutput.success(
            svg_code=svg_code,
            rounds=rounds,
            passed=False,
            inspect_report=last_report,
            note=f"Reached max_rounds={max_rounds} without passing inspection",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_svg(text: str) -> str:
    """从模型输出中提取 SVG 代码（去代码块/前后杂质）。"""
    if not text:
        return ""
    # 去掉 ```xml / ```svg / ``` 包裹
    m = re.search(r"```(?:xml|svg)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 无代码块：找到 <svg ... </svg>
    m = re.search(r"<svg[\s\S]*?</svg>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return text.strip()


def _parse_inspect_json(text: str) -> Optional[dict[str, Any]]:
    """解析检查模型的 JSON 输出（容忍代码块包裹/前后杂质）。"""
    if not text:
        return None
    cleaned = text.strip()
    m = re.search(r"```json\s*\n(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        # 尝试提取第一个 { ... } 块
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, dict) else None
            except Exception:
                return None
        return None


async def _render_svg_to_png(svg_code: str) -> Optional[bytes]:
    """把 SVG 渲染成 PNG bytes。

    优先用 Playwright + 系统 Chrome/Edge（浏览器渲染 SVG 最准确），
    fallback 到 svglib+reportlab。失败返回 None（不抛异常，让上层
    保守处理）。
    """
    png = await _render_svg_playwright(svg_code)
    if png is not None:
        return png
    return await _render_svg_svglib(svg_code)


def _find_chrome() -> Optional[str]:
    """查找系统 Chrome / Edge 可执行文件。"""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


async def _render_svg_playwright(svg_code: str) -> Optional[bytes]:
    """用 Playwright + 系统浏览器渲染 SVG。"""
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:
        return None

    chrome = _find_chrome()
    if chrome is None:
        return None

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        "<body style='margin:0'>"
        + svg_code
        + "</body></html>"
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path=chrome,
                args=["--no-sandbox", "--disable-gpu"],
            )
            try:
                page = await browser.new_page(
                    viewport={"width": 1600, "height": 1200}
                )
                await page.set_content(html, wait_until="load")
                # 让 SVG 自适应画布
                await page.evaluate(
                    "() => { const s = document.querySelector('svg');"
                    " if (s) { s.style.width = '100%'; s.style.height = 'auto'; } }"
                )
                png = await page.screenshot(type="png")
                return png
            finally:
                await browser.close()
    except Exception as e:
        logger.warning("Playwright SVG render failed: %s", e)
        return None


async def _render_svg_svglib(svg_code: str) -> Optional[bytes]:
    """fallback: svglib + reportlab 渲染（对复杂 CSS 支持较差）。"""
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
    except ImportError:
        logger.warning(
            "No SVG renderer available. Install playwright browsers "
            "(`playwright install chromium`) or `pip install svglib reportlab`"
        )
        return None

    try:
        drawing = svg2rlg(io.StringIO(svg_code))
        if drawing is None:
            return None
        drawing.width = min(drawing.width, 1600)
        drawing.height = min(drawing.height, 1200)
        buf = io.BytesIO()
        renderPM.drawToFile(drawing, buf, fmt="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning("svglib SVG render failed: %s", e)
        return None


__all__ = ["SVGFlowWorkflow", "SVGDrawTask", "SVGInspectTask", "SVGFixTask"]
