from __future__ import annotations

import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pymupdf
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app import main
from app.routers import jobs


class _FakeRedisJob:
    def __init__(self, job_id: str) -> None:
        now = datetime.now(timezone.utc)
        self.job_id = job_id
        self.status = jobs.JobStatus.pending
        self.created_at = now
        self.expires_at = now + timedelta(hours=24)


class _FakeRedisService:
    def __init__(self) -> None:
        self.updated: list[tuple[str, jobs.JobStatus, jobs.JobStage, str]] = []

    def is_memory_backend(self) -> bool:
        return True

    def create_job(self, job_id: str) -> _FakeRedisJob:
        return _FakeRedisJob(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: jobs.JobStatus,
        stage: jobs.JobStage,
        message: str,
    ) -> None:
        self.updated.append((job_id, status, stage, message))


class _FakeThread:
    started = False
    last_kwargs: dict[str, object] | None = None

    def __init__(self, *, target, kwargs, daemon: bool) -> None:
        _FakeThread.last_kwargs = {
            "target": target,
            "kwargs": kwargs,
            "daemon": daemon,
        }

    def start(self) -> None:
        _FakeThread.started = True


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(jobs.router)
    app.add_exception_handler(jobs.AppException, main.app_exception_handler)
    return app


def _build_png_bytes(*, size: tuple[int, int] = (400, 200)) -> bytes:
    image = Image.new("RGBA", size, (0, 128, 255, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    return buffer.getvalue()


def _normalized_job_options() -> SimpleNamespace:
    return SimpleNamespace(
        parse_provider="local",
        provider="openai",
        baidu_doc_parse_type="paddle_vl",
        ocr_provider="auto",
        ocr_ai_provider="auto",
        ocr_ai_chain_mode="direct",
        ocr_ai_layout_model="pp_doclayout_v3",
        ocr_geometry_mode="auto",
        text_erase_mode="fill",
        scanned_page_mode="segmented",
        ppt_generation_mode="standard",
    )


def test_create_job_accepts_png_and_normalizes_to_input_pdf(
    monkeypatch, tmp_path: Path
) -> None:
    _FakeThread.started = False
    _FakeThread.last_kwargs = None
    redis_service = _FakeRedisService()

    def _ensure_job_dir(job_id: str) -> Path:
        target = tmp_path / job_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(
        jobs,
        "validate_and_normalize_job_options",
        lambda **kwargs: _normalized_job_options(),
    )
    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(max_file_mb=20))
    monkeypatch.setattr(jobs, "get_redis_service", lambda: redis_service)
    monkeypatch.setattr(jobs, "ensure_job_dir", _ensure_job_dir)
    monkeypatch.setattr(jobs.threading, "Thread", _FakeThread)

    client = TestClient(_build_test_app())
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("slide.png", _build_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    job_id = str(payload["job_id"])
    input_pdf = tmp_path / job_id / "input.pdf"
    assert input_pdf.exists()
    assert _FakeThread.started is True
    assert _FakeThread.last_kwargs is not None
    assert _FakeThread.last_kwargs["target"] is jobs.process_pdf_job
    assert redis_service.updated == [
        (
            job_id,
            jobs.JobStatus.pending,
            jobs.JobStage.queued,
            "Job queued for processing",
        )
    ]

    doc = pymupdf.open(str(input_pdf))
    try:
        assert doc.page_count == 1
        page = doc.load_page(0)
        assert page.rect.width > 0
        assert page.rect.height > 0
    finally:
        doc.close()


def test_create_job_rejects_unsupported_upload_type(monkeypatch) -> None:
    monkeypatch.setattr(
        jobs,
        "validate_and_normalize_job_options",
        lambda **kwargs: _normalized_job_options(),
    )
    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(max_file_mb=20))
    monkeypatch.setattr(jobs, "get_redis_service", lambda: _FakeRedisService())

    client = TestClient(_build_test_app())
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "validation_error"
    assert "Only PDF, PNG, JPG, JPEG, and WEBP files are supported" in response.json()[
        "message"
    ]
