from __future__ import annotations

import asyncio
import io
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from app.models.error import AppException, ErrorCode
from app.models.job import Job, JobStage, JobStatus, LayoutMode
from app.routers import jobs as jobs_router
from app.services import redis_service as redis_service_module


def test_redis_service_falls_back_to_memory_when_redis_unavailable(
    monkeypatch,
) -> None:
    class _BrokenRedisClient:
        def ping(self) -> None:
            raise RuntimeError("dns resolution failed")

    monkeypatch.setattr(
        redis_service_module,
        "get_settings",
        lambda: SimpleNamespace(
            job_ttl_minutes=60,
            redis_url="redis://redis:6379/0",
        ),
    )
    monkeypatch.setattr(
        redis_service_module.redis,
        "from_url",
        lambda *args, **kwargs: _BrokenRedisClient(),
    )

    service = redis_service_module.RedisService()
    assert service.is_memory_backend() is True

    service.create_job("job-fallback")
    assert service.get_job("job-fallback") is not None


def test_list_jobs_skips_rq_queue_lookup_when_memory_backend(monkeypatch) -> None:
    class _MemoryService:
        def is_memory_backend(self) -> bool:
            return True

        def list_jobs(self, *, limit: int = 50):
            _ = limit
            return []

    monkeypatch.setattr(
        jobs_router,
        "get_redis_service",
        lambda: _MemoryService(),
    )

    def _should_not_call_get_redis_connection():
        raise AssertionError("RQ redis connection should not be used in memory backend")

    monkeypatch.setattr(
        jobs_router,
        "get_redis_connection",
        _should_not_call_get_redis_connection,
    )

    response = asyncio.run(jobs_router.list_jobs(limit=20))
    assert response.queue_size == 0
    assert response.returned == 0
    assert response.jobs == []


def test_create_job_rolls_back_metadata_when_enqueue_fails(
    monkeypatch, tmp_path
) -> None:
    class _FakeRedisService:
        def __init__(self) -> None:
            self.jobs: dict[str, Job] = {}
            self.deleted: list[str] = []

        def create_job(self, job_id: str) -> Job:
            now = datetime.now(timezone.utc)
            job = Job(
                job_id=job_id,
                status=JobStatus.pending,
                stage=JobStage.upload_received,
                progress=0,
                created_at=now,
                expires_at=now + timedelta(hours=1),
                message="Job created, waiting to be queued",
                error=None,
                layout_mode=LayoutMode.fidelity,
            )
            self.jobs[job_id] = job
            return job

        def delete_job(self, job_id: str) -> None:
            self.deleted.append(job_id)
            self.jobs.pop(job_id, None)

        def is_memory_backend(self) -> bool:
            return False

        def update_job(self, *_args, **_kwargs):
            raise AssertionError("update_job should not run when enqueue fails")

    class _FailingQueue:
        def __init__(self, *, connection: object) -> None:
            self.connection = connection

        def enqueue(self, *_args, **_kwargs):
            raise RuntimeError("rq enqueue failed")

    fake_service = _FakeRedisService()
    fixed_job_id = "job-enqueue-fail"

    monkeypatch.setattr(
        jobs_router,
        "get_settings",
        lambda: SimpleNamespace(max_file_mb=20, siliconflow_api_key=""),
    )
    monkeypatch.setattr(jobs_router, "get_redis_service", lambda: fake_service)
    monkeypatch.setattr(jobs_router, "Queue", _FailingQueue)
    monkeypatch.setattr(jobs_router, "get_redis_connection", lambda: object())
    monkeypatch.setattr(jobs_router.uuid, "uuid4", lambda: fixed_job_id)

    def _ensure_job_dir(job_id: str):
        path = tmp_path / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(jobs_router, "ensure_job_dir", _ensure_job_dir)

    upload = UploadFile(filename="sample.pdf", file=io.BytesIO(b"%PDF-1.4\n%EOF\n"))

    with pytest.raises(AppException) as exc_info:
        asyncio.run(jobs_router.create_job(file=upload, parse_provider="local"))

    assert exc_info.value.code == ErrorCode.INTERNAL_ERROR.value
    assert fake_service.deleted == [fixed_job_id]
    assert fixed_job_id not in fake_service.jobs


def test_update_job_does_not_overwrite_terminal_status(monkeypatch) -> None:
    monkeypatch.setattr(
        redis_service_module,
        "get_settings",
        lambda: SimpleNamespace(job_ttl_minutes=60, redis_url="memory://test"),
    )

    service = redis_service_module.RedisService()
    job_id = "job-terminal-guard"
    service.create_job(job_id)

    cancelled = service.update_job(
        job_id,
        status=JobStatus.cancelled,
        stage=JobStage.done,
        progress=100,
        message="cancelled",
    )
    assert cancelled is not None
    assert cancelled.status == JobStatus.cancelled

    updated = service.update_job(
        job_id,
        status=JobStatus.pending,
        stage=JobStage.queued,
        progress=1,
        message="queued again",
    )
    assert updated is not None
    assert updated.status == JobStatus.cancelled
    assert updated.stage == JobStage.done
    assert updated.progress == 100
    assert updated.message == "cancelled"
