from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app import worker


class _FakeRedisService:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def is_cancelled(self, job_id: str) -> bool:
        _ = job_id
        return False

    def update_job(self, job_id: str, **kwargs) -> None:
        self.updates.append({"job_id": job_id, **kwargs})

    def refresh_job_ttl(self, job_id: str) -> None:
        _ = job_id


class _FakeSettings:
    redis_url = "memory://"
    job_ttl_minutes = 60
    job_debug_events_limit = 200
    ocr_render_dpi = 200
    scanned_render_dpi = 200
    job_keepalive_interval_s = 15
    export_ocr_overlay_images = False
    export_layout_assist_debug_images = False
    export_final_preview_images = False
    export_final_preview_max_pages = 0


def test_layout_assist_no_longer_forces_ocr_stage(monkeypatch, tmp_path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.pdf").write_bytes(b"%PDF-1.4\n")

    redis_service = _FakeRedisService()
    setup_calls: list[str] = []
    layout_assist_flags: list[bool] = []

    monkeypatch.setattr(worker, "_job_dir", lambda job_id: job_dir)
    monkeypatch.setattr(worker, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(worker, "get_redis_service", lambda: redis_service)
    monkeypatch.setattr(
        worker,
        "parse_pdf_to_ir",
        lambda *args, **kwargs: {
            "source_pdf": str(job_dir / "input.pdf"),
            "warnings": [],
            "pages": [
                {
                    "page_index": 0,
                    "page_width_pt": 100.0,
                    "page_height_pt": 100.0,
                    "has_text_layer": False,
                    "ocr_used": False,
                    "elements": [],
                }
            ],
        },
    )
    monkeypatch.setattr(
        worker,
        "setup_ocr_runtime",
        lambda **kwargs: setup_calls.append("called"),
    )
    monkeypatch.setattr(
        worker,
        "run_layout_assist_stage",
        lambda **kwargs: (
            layout_assist_flags.append(bool(kwargs["enable_layout_assist"]))
            or SimpleNamespace(ir=kwargs["ir"])
        ),
    )

    def _fake_run_ppt_stage(**kwargs):
        kwargs["output_pptx"].write_bytes(b"pptx")
        return SimpleNamespace(worker_compat_mode=False)

    monkeypatch.setattr(worker, "run_ppt_stage", _fake_run_ppt_stage)

    worker.process_pdf_job(
        "job-layout-assist-only",
        parse_provider="local",
        enable_ocr=False,
        enable_layout_assist=True,
    )

    assert setup_calls == []
    assert layout_assist_flags == [False]
    assert (job_dir / "output.pptx").exists()
