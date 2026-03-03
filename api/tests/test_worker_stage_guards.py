from __future__ import annotations

import time
from types import SimpleNamespace

from app.services import redis_service as redis_service_module
from app.worker_helpers import layout_assist_stage, ppt_stage


def test_refresh_job_ttl_extends_expires_at(monkeypatch) -> None:
    monkeypatch.setattr(
        redis_service_module,
        "get_settings",
        lambda: SimpleNamespace(job_ttl_minutes=60, redis_url="memory://test"),
    )

    service = redis_service_module.RedisService()
    created = service.create_job("job-ttl-refresh")
    old_expires_at = created.expires_at
    time.sleep(0.01)

    refreshed = service.refresh_job_ttl("job-ttl-refresh")
    assert refreshed is not None
    assert refreshed.expires_at > old_expires_at


def test_layout_assist_long_running_call_does_not_timeout(
    monkeypatch, tmp_path
) -> None:
    class _SlowLayoutService:
        def __init__(self, _provider: object) -> None:
            pass

        def enhance_ir(self, ir: dict, **_kwargs):
            time.sleep(0.35)
            return ir

    monkeypatch.setattr(layout_assist_stage, "LlmLayoutService", _SlowLayoutService)
    monkeypatch.setattr(
        layout_assist_stage,
        "_export_layout_assist_debug_images",
        lambda **_kwargs: {"pages_exported": 0, "pages_changed": 0},
    )

    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    job_path = tmp_path / "job"
    job_path.mkdir(parents=True, exist_ok=True)
    artifacts_dir = job_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result = layout_assist_stage.run_layout_assist_stage(
        ir={"pages": [{"page": 1}]},
        job_id="job-layout-timeout",
        enable_layout_assist=True,
        layout_assist_apply_image_regions=False,
        input_pdf=input_pdf,
        job_path=job_path,
        artifacts_dir=artifacts_dir,
        scanned_render_dpi=200,
        select_provider=lambda: object(),
        set_processing_progress=lambda *_args, **_kwargs: None,
        abort_if_cancelled=lambda **_kwargs: None,
        heartbeat=lambda: None,
        heartbeat_interval_s=0.05,
    )

    assert result.status in {"no_change", "applied"}


def test_ppt_stage_long_running_call_does_not_timeout(monkeypatch, tmp_path) -> None:
    def _slow_generate(
        ir: dict,
        output_pptx,
        *,
        artifacts_dir,
        scanned_render_dpi,
        text_erase_mode,
        progress_callback,
        **_kwargs,
    ) -> None:
        _ = ir, artifacts_dir, scanned_render_dpi, text_erase_mode
        time.sleep(0.35)
        if callable(progress_callback):
            progress_callback(1, 1)
        output_pptx.write_bytes(b"pptx")

    monkeypatch.setattr(ppt_stage, "generate_pptx_from_ir", _slow_generate)

    output_pptx = tmp_path / "output.pptx"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result = ppt_stage.run_ppt_stage(
        ir={"pages": [{"page": 1}]},
        output_pptx=output_pptx,
        artifacts_dir=artifacts_dir,
        scanned_render_dpi=200,
        normalized_text_erase_mode="fill",
        normalized_scanned_page_mode="segmented",
        normalized_image_bg_clear_expand_min_pt=0.35,
        normalized_image_bg_clear_expand_max_pt=1.5,
        normalized_image_bg_clear_expand_ratio=0.012,
        normalized_scanned_image_region_min_area_ratio=0.0025,
        normalized_scanned_image_region_max_area_ratio=0.72,
        normalized_scanned_image_region_max_aspect_ratio=4.8,
        set_processing_progress=lambda *_args, **_kwargs: None,
        abort_if_cancelled=lambda **_kwargs: None,
        heartbeat=lambda: None,
        heartbeat_interval_s=0.05,
    )

    assert result.worker_compat_mode is False
    assert output_pptx.exists()
