from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..convert.pptx_generator import generate_pptx_from_ir
from ..models.job import JobStage
from .guarded import run_blocking_with_guards


_REQUIRED_GENERATOR_FEATURES = (
    "text_erase_mode",
    "progress_callback",
    "export_final_preview_images",
)


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


def _missing_generator_features(
    generator_params: dict[str, inspect.Parameter],
) -> list[str]:
    return [
        feature
        for feature in _REQUIRED_GENERATOR_FEATURES
        if feature not in generator_params
    ]


def _collect_supported_generator_kwargs(
    *,
    generator_params: dict[str, inspect.Parameter],
    candidates: dict[str, Any],
) -> dict[str, Any]:
    return {
        name: value
        for name, value in candidates.items()
        if name in generator_params
    }


def run_ppt_stage(
    *,
    ir: dict[str, Any],
    output_pptx: Path,
    artifacts_dir: Path,
    scanned_render_dpi: int,
    remove_footer_notebooklm: bool,
    normalized_text_erase_mode: str,
    normalized_scanned_page_mode: str,
    normalized_image_bg_clear_expand_min_pt: float,
    normalized_image_bg_clear_expand_max_pt: float,
    normalized_image_bg_clear_expand_ratio: float,
    normalized_scanned_image_region_min_area_ratio: float,
    normalized_scanned_image_region_max_area_ratio: float,
    normalized_scanned_image_region_max_aspect_ratio: float,
    export_final_preview_images: bool,
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

    generator_params = dict(inspect.signature(generate_pptx_from_ir).parameters)
    missing_generator_features = _missing_generator_features(generator_params)
    worker_compat_mode = bool(missing_generator_features)
    generator_kwargs = _collect_supported_generator_kwargs(
        generator_params=generator_params,
        candidates={
            "artifacts_dir": artifacts_dir,
            "scanned_render_dpi": int(scanned_render_dpi),
            "remove_footer_notebooklm": bool(remove_footer_notebooklm),
            "text_erase_mode": normalized_text_erase_mode,
            "scanned_page_mode": normalized_scanned_page_mode,
            "image_bg_clear_expand_min_pt": normalized_image_bg_clear_expand_min_pt,
            "image_bg_clear_expand_max_pt": normalized_image_bg_clear_expand_max_pt,
            "image_bg_clear_expand_ratio": normalized_image_bg_clear_expand_ratio,
            "scanned_image_region_min_area_ratio": normalized_scanned_image_region_min_area_ratio,
            "scanned_image_region_max_area_ratio": normalized_scanned_image_region_max_area_ratio,
            "scanned_image_region_max_aspect_ratio": normalized_scanned_image_region_max_aspect_ratio,
            "export_final_preview_images": bool(export_final_preview_images),
            "progress_callback": _on_ppt_page_done,
        },
    )

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
