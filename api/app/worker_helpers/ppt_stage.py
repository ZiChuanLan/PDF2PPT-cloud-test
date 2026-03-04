from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..convert.pptx_generator import generate_pptx_from_ir
from ..models.job import JobStage
from .guarded import run_blocking_with_guards


def _progress_in_span(
    done: int,
    total: int,
    *,
    start: int,
    end: int,
) -> int:
    if total <= 0:
        return int(end)
    ratio = max(0.0, min(1.0, float(done) / float(total)))
    return int(round(float(start) + (float(end - start) * ratio)))


@dataclass(frozen=True)
class PptStageResult:
    worker_compat_mode: bool


def run_ppt_stage(
    *,
    ir: dict[str, Any],
    output_pptx: Path,
    artifacts_dir: Path,
    scanned_render_dpi: int,
    ppt_text_fit_workers: int = 1,
    normalized_text_erase_mode: str,
    normalized_scanned_page_mode: str,
    normalized_image_bg_clear_expand_min_pt: float,
    normalized_image_bg_clear_expand_max_pt: float,
    normalized_image_bg_clear_expand_ratio: float,
    normalized_scanned_image_region_min_area_ratio: float,
    normalized_scanned_image_region_max_area_ratio: float,
    normalized_scanned_image_region_max_aspect_ratio: float,
    set_processing_progress: Callable[[JobStage, int, str], None],
    abort_if_cancelled: Callable[..., None],
    heartbeat: Callable[[], None] | None = None,
    heartbeat_interval_s: float = 15.0,
) -> PptStageResult:
    ppt_page_total = sum(
        1 for page in (ir.get("pages") or []) if isinstance(page, dict)
    )
    set_processing_progress(
        JobStage.pptx_generating,
        84,
        f"开始生成 PPT（共 {ppt_page_total} 页）",
    )
    abort_if_cancelled(stage=JobStage.pptx_generating, message="Job cancelled")

    def _on_ppt_page_done(done: int, total: int) -> None:
        set_processing_progress(
            JobStage.pptx_generating,
            _progress_in_span(done, max(1, total), start=85, end=97),
            f"正在生成 PPT 页面（{done}/{max(1, total)}）",
        )
        abort_if_cancelled(stage=JobStage.pptx_generating, message="Job cancelled")

    generator_params = inspect.signature(generate_pptx_from_ir).parameters
    missing_generator_features: list[str] = []
    if "text_erase_mode" not in generator_params:
        missing_generator_features.append("text_erase_mode")
    if "progress_callback" not in generator_params:
        missing_generator_features.append("progress_callback")
    worker_compat_mode = bool(missing_generator_features)
    generator_kwargs: dict[str, Any] = {
        "artifacts_dir": artifacts_dir,
        "scanned_render_dpi": int(scanned_render_dpi),
    }
    if "text_fit_workers" in generator_params:
        generator_kwargs["text_fit_workers"] = max(1, int(ppt_text_fit_workers))
    if "text_erase_mode" in generator_params:
        generator_kwargs["text_erase_mode"] = normalized_text_erase_mode
    if "scanned_page_mode" in generator_params:
        generator_kwargs["scanned_page_mode"] = normalized_scanned_page_mode
    if "image_bg_clear_expand_min_pt" in generator_params:
        generator_kwargs["image_bg_clear_expand_min_pt"] = (
            normalized_image_bg_clear_expand_min_pt
        )
    if "image_bg_clear_expand_max_pt" in generator_params:
        generator_kwargs["image_bg_clear_expand_max_pt"] = (
            normalized_image_bg_clear_expand_max_pt
        )
    if "image_bg_clear_expand_ratio" in generator_params:
        generator_kwargs["image_bg_clear_expand_ratio"] = (
            normalized_image_bg_clear_expand_ratio
        )
    if "scanned_image_region_min_area_ratio" in generator_params:
        generator_kwargs["scanned_image_region_min_area_ratio"] = (
            normalized_scanned_image_region_min_area_ratio
        )
    if "scanned_image_region_max_area_ratio" in generator_params:
        generator_kwargs["scanned_image_region_max_area_ratio"] = (
            normalized_scanned_image_region_max_area_ratio
        )
    if "scanned_image_region_max_aspect_ratio" in generator_params:
        generator_kwargs["scanned_image_region_max_aspect_ratio"] = (
            normalized_scanned_image_region_max_aspect_ratio
        )
    if "progress_callback" in generator_params:
        generator_kwargs["progress_callback"] = _on_ppt_page_done

    if worker_compat_mode:
        compat_features = ",".join(missing_generator_features)
        ir.setdefault("warnings", []).append(
            f"worker_compat_mode_missing_features={compat_features}"
        )
        set_processing_progress(
            JobStage.pptx_generating,
            84,
            "检测到 worker 兼容模式（旧转换内核），建议升级 worker",
        )
        abort_if_cancelled(stage=JobStage.pptx_generating, message="Job cancelled")

    abort_if_cancelled(stage=JobStage.pptx_generating, message="Job cancelled")
    run_blocking_with_guards(
        lambda: generate_pptx_from_ir(
            ir,
            output_pptx,
            **generator_kwargs,
        ),
        cancel_check=lambda: abort_if_cancelled(
            stage=JobStage.pptx_generating,
            message="Job cancelled",
        ),
        operation_name="ppt generation",
        heartbeat=heartbeat,
        heartbeat_interval_s=heartbeat_interval_s,
    )
    set_processing_progress(
        JobStage.packaging,
        98,
        (
            "正在打包转换结果…（兼容模式，建议升级 worker）"
            if worker_compat_mode
            else "正在打包转换结果…"
        ),
    )
    abort_if_cancelled(stage=JobStage.packaging, message="Job cancelled")
    return PptStageResult(worker_compat_mode=worker_compat_mode)
