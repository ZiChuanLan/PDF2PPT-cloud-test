from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.perf_policies import (
    ArtifactExportSettings,
    RuntimePerformanceSettings,
    resolve_page_artifact_export,
)
from app.worker_helpers import layout_assist_stage, ppt_stage


def test_resolve_page_artifact_export_disables_large_doc_preview() -> None:
    enabled = resolve_page_artifact_export(
        enabled=True,
        total_pages=13,
        max_pages=5,
    )

    assert enabled is False


def test_resolve_page_artifact_export_keeps_small_doc_preview() -> None:
    enabled = resolve_page_artifact_export(
        enabled=True,
        total_pages=3,
        max_pages=5,
    )

    assert enabled is True


def test_artifact_export_settings_resolve_document_policy() -> None:
    policy = ArtifactExportSettings(
        ocr_overlay_images=False,
        layout_assist_debug_images=True,
        final_preview_images=True,
        final_preview_max_pages=5,
    ).resolve_for_parsed_document(parsed_pages=13)

    assert policy.layout_assist_debug_images is True
    assert policy.final_preview_images is False


def test_runtime_performance_settings_loads_from_settings_object() -> None:
    class _Settings:
        ocr_render_dpi = 180
        scanned_render_dpi = 160
        job_keepalive_interval_s = 9
        export_ocr_overlay_images = True
        export_layout_assist_debug_images = False
        export_final_preview_images = True
        export_final_preview_max_pages = 4

    runtime = RuntimePerformanceSettings.from_settings(_Settings())

    assert runtime.ocr_render_dpi == 180
    assert runtime.scanned_render_dpi == 160
    assert runtime.keepalive_interval_s == 9.0
    assert runtime.artifact_exports.ocr_overlay_images is True
    assert runtime.artifact_exports.final_preview_max_pages == 4


def test_run_layout_assist_stage_skips_debug_export_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    called = {"debug_export": 0}

    class _FakeLayoutService:
        def __init__(self, _provider: object) -> None:
            pass

        def enhance_ir(self, ir: dict, **_kwargs: object) -> dict:
            return ir

    def _fake_debug_export(**_kwargs: object) -> dict:
        called["debug_export"] += 1
        return {"pages_exported": 1, "pages_changed": 0}

    monkeypatch.setattr(layout_assist_stage, "LlmLayoutService", _FakeLayoutService)
    monkeypatch.setattr(
        layout_assist_stage,
        "run_blocking_with_guards",
        lambda fn, **_kwargs: fn(),
    )
    monkeypatch.setattr(layout_assist_stage, "_apply_ai_tables", lambda ir: ir)
    monkeypatch.setattr(
        layout_assist_stage,
        "_count_layout_assist_page_changes",
        lambda _before, _after: (0, 1),
    )
    monkeypatch.setattr(
        layout_assist_stage,
        "_extract_warning_suffix",
        lambda _warnings, prefix: None,
    )
    monkeypatch.setattr(
        layout_assist_stage,
        "_export_layout_assist_debug_images",
        _fake_debug_export,
    )

    job_path = tmp_path / "job"
    job_path.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    result = layout_assist_stage.run_layout_assist_stage(
        ir={
            "pages": [
                {
                    "page_index": 0,
                    "page_width_pt": 720.0,
                    "page_height_pt": 540.0,
                    "elements": [],
                }
            ]
        },
        job_id="job-1",
        enable_layout_assist=True,
        layout_assist_apply_image_regions=False,
        input_pdf=tmp_path / "input.pdf",
        job_path=job_path,
        artifacts_dir=artifacts_dir,
        scanned_render_dpi=200,
        export_debug_images=False,
        select_provider=lambda: object(),
        set_processing_progress=lambda *_args, **_kwargs: None,
        abort_if_cancelled=lambda **_kwargs: None,
    )

    assert called["debug_export"] == 0
    assert result.status == "no_change"


def test_run_ppt_stage_forwards_final_preview_flag(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_generate(
        ir: dict,
        output_pptx: Path,
        *,
        artifacts_dir: Path,
        scanned_render_dpi: int,
        remove_footer_notebooklm: bool,
        text_erase_mode: str,
        scanned_page_mode: str,
        image_bg_clear_expand_min_pt: float,
        image_bg_clear_expand_max_pt: float,
        image_bg_clear_expand_ratio: float,
        scanned_image_region_min_area_ratio: float,
        scanned_image_region_max_area_ratio: float,
        scanned_image_region_max_aspect_ratio: float,
        export_final_preview_images: bool,
        progress_callback,
    ) -> Path:
        captured["ir"] = ir
        captured["output_pptx"] = output_pptx
        captured.update(
            {
                "artifacts_dir": artifacts_dir,
                "scanned_render_dpi": scanned_render_dpi,
                "remove_footer_notebooklm": remove_footer_notebooklm,
                "text_erase_mode": text_erase_mode,
                "scanned_page_mode": scanned_page_mode,
                "image_bg_clear_expand_min_pt": image_bg_clear_expand_min_pt,
                "image_bg_clear_expand_max_pt": image_bg_clear_expand_max_pt,
                "image_bg_clear_expand_ratio": image_bg_clear_expand_ratio,
                "scanned_image_region_min_area_ratio": scanned_image_region_min_area_ratio,
                "scanned_image_region_max_area_ratio": scanned_image_region_max_area_ratio,
                "scanned_image_region_max_aspect_ratio": scanned_image_region_max_aspect_ratio,
                "export_final_preview_images": export_final_preview_images,
                "progress_callback": progress_callback,
            }
        )
        return output_pptx

    monkeypatch.setattr(ppt_stage, "generate_pptx_from_ir", _fake_generate)
    monkeypatch.setattr(
        ppt_stage,
        "run_blocking_with_guards",
        lambda fn, **_kwargs: fn(),
    )

    result = ppt_stage.run_ppt_stage(
        ir={"pages": [{"page_index": 0}]},
        output_pptx=tmp_path / "out.pptx",
        artifacts_dir=tmp_path / "artifacts",
        scanned_render_dpi=200,
        remove_footer_notebooklm=False,
        normalized_text_erase_mode="fill",
        normalized_scanned_page_mode="fullpage",
        normalized_image_bg_clear_expand_min_pt=0.35,
        normalized_image_bg_clear_expand_max_pt=1.5,
        normalized_image_bg_clear_expand_ratio=0.012,
        normalized_scanned_image_region_min_area_ratio=0.0025,
        normalized_scanned_image_region_max_area_ratio=0.72,
        normalized_scanned_image_region_max_aspect_ratio=4.8,
        export_final_preview_images=False,
        set_processing_progress=lambda *_args, **_kwargs: None,
        abort_if_cancelled=lambda **_kwargs: None,
    )

    assert result.worker_compat_mode is False
    assert captured["export_final_preview_images"] is False


def test_run_ppt_stage_marks_compat_mode_when_preview_flag_not_supported(
    monkeypatch, tmp_path: Path
) -> None:
    def _legacy_generate(
        ir: dict,
        output_pptx: Path,
        *,
        artifacts_dir: Path,
        scanned_render_dpi: int,
        remove_footer_notebooklm: bool,
        text_erase_mode: str,
        scanned_page_mode: str,
        image_bg_clear_expand_min_pt: float,
        image_bg_clear_expand_max_pt: float,
        image_bg_clear_expand_ratio: float,
        scanned_image_region_min_area_ratio: float,
        scanned_image_region_max_area_ratio: float,
        scanned_image_region_max_aspect_ratio: float,
        progress_callback,
    ) -> Path:
        return output_pptx

    monkeypatch.setattr(ppt_stage, "generate_pptx_from_ir", _legacy_generate)
    monkeypatch.setattr(
        ppt_stage,
        "run_blocking_with_guards",
        lambda fn, **_kwargs: fn(),
    )

    result = ppt_stage.run_ppt_stage(
        ir={"pages": [{"page_index": 0}], "warnings": []},
        output_pptx=tmp_path / "out-legacy.pptx",
        artifacts_dir=tmp_path / "artifacts-legacy",
        scanned_render_dpi=200,
        remove_footer_notebooklm=False,
        normalized_text_erase_mode="fill",
        normalized_scanned_page_mode="fullpage",
        normalized_image_bg_clear_expand_min_pt=0.35,
        normalized_image_bg_clear_expand_max_pt=1.5,
        normalized_image_bg_clear_expand_ratio=0.012,
        normalized_scanned_image_region_min_area_ratio=0.0025,
        normalized_scanned_image_region_max_area_ratio=0.72,
        normalized_scanned_image_region_max_aspect_ratio=4.8,
        export_final_preview_images=False,
        set_processing_progress=lambda *_args, **_kwargs: None,
        abort_if_cancelled=lambda **_kwargs: None,
    )

    assert result.worker_compat_mode is True
