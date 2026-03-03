from __future__ import annotations

from types import SimpleNamespace

from app import worker as worker_module
from app.models.job import JobStatus


class _FakeRedisService:
    def __init__(self) -> None:
        self.cancelled = False
        self.updates: list[dict[str, object | None]] = []

    def is_cancelled(self, _job_id: str) -> bool:
        return self.cancelled

    def update_job(
        self,
        _job_id: str,
        status=None,
        stage=None,
        progress=None,
        message=None,
        error=None,
    ) -> None:
        self.updates.append(
            {
                "status": status,
                "stage": stage,
                "progress": progress,
                "message": message,
                "error": error,
            }
        )


def test_mineru_cancel_during_poll_marks_job_cancelled(monkeypatch, tmp_path) -> None:
    redis_service = _FakeRedisService()
    job_id = "job-mineru-cancel"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.pdf").write_bytes(b"%PDF-1.4\n%EOF\n")

    monkeypatch.setattr(worker_module, "_job_dir", lambda _job_id: job_dir)
    monkeypatch.setattr(worker_module, "get_redis_service", lambda: redis_service)
    monkeypatch.setattr(
        worker_module,
        "get_settings",
        lambda: SimpleNamespace(ocr_render_dpi=200, scanned_render_dpi=200),
    )
    monkeypatch.setattr(worker_module, "set_job_id", lambda _job_id: None)

    def _fake_parse_pdf_to_ir_with_mineru(*_args, **kwargs):
        cancel_check = kwargs.get("cancel_check")
        if cancel_check is None:
            raise RuntimeError("cancel_check missing")
        redis_service.cancelled = True
        cancel_check()
        raise AssertionError("cancel_check should abort by raising JobCancelledError")

    monkeypatch.setattr(
        worker_module,
        "parse_pdf_to_ir_with_mineru",
        _fake_parse_pdf_to_ir_with_mineru,
    )

    worker_module.process_pdf_job(
        job_id,
        parse_provider="mineru",
        mineru_api_token="token",
    )

    statuses = [entry["status"] for entry in redis_service.updates]
    assert JobStatus.cancelled in statuses
    assert JobStatus.failed not in statuses
