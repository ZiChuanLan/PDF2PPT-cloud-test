from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..convert.llm_adapter import LlmLayoutService
from ..logging_config import get_logger
from ..models.job import JobStage
from .debug import _export_layout_assist_debug_images
from .guarded import run_blocking_with_guards
from .layout import (
    _apply_ai_tables,
    _count_layout_assist_page_changes,
    _extract_warning_suffix,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class LayoutAssistStageResult:
    ir: dict[str, Any]
    status: str
    error: str | None
    pages_changed: int
    pages_total: int


def run_layout_assist_stage(
    *,
    ir: dict[str, Any],
    job_id: str,
    enable_layout_assist: bool,
    layout_assist_apply_image_regions: bool,
    input_pdf: Path,
    job_path: Path,
    artifacts_dir: Path,
    scanned_render_dpi: int,
    export_debug_images: bool,
    select_provider: Callable[[], Any | None],
    set_processing_progress: Callable[[JobStage, int, str], None],
    abort_if_cancelled: Callable[..., None],
    heartbeat: Callable[[], None] | None = None,
    heartbeat_interval_s: float = 15.0,
) -> LayoutAssistStageResult:
    layout_assist_status = "disabled"
    layout_assist_error: str | None = None
    layout_assist_pages_changed = 0
    layout_assist_pages_total = 0

    if not enable_layout_assist:
        return LayoutAssistStageResult(
            ir=ir,
            status=layout_assist_status,
            error=layout_assist_error,
            pages_changed=layout_assist_pages_changed,
            pages_total=layout_assist_pages_total,
        )

    set_processing_progress(
        JobStage.layout_assist,
        72,
        "准备执行 AI 版式辅助…",
    )
    abort_if_cancelled(stage=JobStage.layout_assist, message="Job cancelled")

    llm_provider = select_provider()
    if llm_provider:
        before_ai_ir = copy.deepcopy(ir)
        set_processing_progress(
            JobStage.layout_assist,
            74,
            "AI 版式辅助处理中…",
        )
        abort_if_cancelled(stage=JobStage.layout_assist, message="Job cancelled")
        ir = run_blocking_with_guards(
            lambda: LlmLayoutService(llm_provider).enhance_ir(
                ir,
                layout_mode="assist",
                force_ai=True,
                allow_image_regions=bool(layout_assist_apply_image_regions),
            ),
            cancel_check=lambda: abort_if_cancelled(
                stage=JobStage.layout_assist,
                message="Job cancelled",
            ),
            operation_name="layout assist",
            heartbeat=heartbeat,
            heartbeat_interval_s=heartbeat_interval_s,
        )
        abort_if_cancelled(stage=JobStage.layout_assist, message="Job cancelled")

        ir = _apply_ai_tables(ir)
        layout_assist_pages_changed, layout_assist_pages_total = (
            _count_layout_assist_page_changes(before_ai_ir, ir)
        )
        layout_assist_error = _extract_warning_suffix(
            ir.get("warnings") if isinstance(ir.get("warnings"), list) else None,
            prefix="layout_assist_failed:",
        )
        if layout_assist_error:
            layout_assist_status = "failed"
            logger.warning(
                "Layout assist failed and fell back: job=%s error=%s",
                job_id,
                layout_assist_error,
            )
        elif layout_assist_pages_changed > 0:
            layout_assist_status = "applied"
            logger.info(
                "Layout assist applied: job=%s changed_pages=%s/%s",
                job_id,
                layout_assist_pages_changed,
                layout_assist_pages_total,
            )
        else:
            layout_assist_status = "no_change"
            logger.info(
                "Layout assist produced no structural changes: job=%s",
                job_id,
            )

        ir.setdefault("warnings", []).append(
            f"layout_assist_status={layout_assist_status}"
        )
        ir.setdefault("warnings", []).append(
            "layout_assist_pages_changed="
            f"{layout_assist_pages_changed}/{layout_assist_pages_total}"
        )
        # Persist IR after layout assist for debugging.
        (job_path / "ir.ai.json").write_text(
            json.dumps(ir, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        if export_debug_images:
            try:
                debug_result = _export_layout_assist_debug_images(
                    source_pdf=input_pdf,
                    before_ir=before_ai_ir,
                    after_ir=ir,
                    out_dir=artifacts_dir / "layout_assist",
                    render_dpi=max(96, int(scanned_render_dpi)),
                    assist_status=layout_assist_status,
                    assist_error=layout_assist_error,
                )
                ir.setdefault("warnings", []).append(
                    f"layout_assist_debug_pages={int(debug_result.get('pages_exported') or 0)}"
                )
                ir.setdefault("warnings", []).append(
                    "layout_assist_debug_changed_pages="
                    f"{int(debug_result.get('pages_changed') or 0)}"
                )
            except Exception as e:
                logger.warning("Failed to export layout assist debug images: %s", e)

        set_processing_progress(
            JobStage.layout_assist,
            82,
            f"AI 版式辅助完成（变更 {layout_assist_pages_changed}/{layout_assist_pages_total} 页）",
        )
    else:
        layout_assist_status = "skipped_missing_provider"
        logger.info(
            "Layout assist skipped (missing API key or provider): job=%s",
            job_id,
        )
        ir.setdefault("warnings", []).append(
            f"layout_assist_status={layout_assist_status}"
        )
        ir.setdefault("warnings", []).append(
            "layout_assist_pages_changed="
            f"{layout_assist_pages_changed}/{layout_assist_pages_total}"
        )
        set_processing_progress(
            JobStage.layout_assist,
            80,
            "未配置可用 AI 提供方，已跳过版式辅助",
        )

    abort_if_cancelled(stage=JobStage.layout_assist, message="Job cancelled")
    return LayoutAssistStageResult(
        ir=ir,
        status=layout_assist_status,
        error=layout_assist_error,
        pages_changed=layout_assist_pages_changed,
        pages_total=layout_assist_pages_total,
    )
