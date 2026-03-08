from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.models.job import Job, JobStage, JobStatus, LayoutMode
from app.services.job_cleanup import cleanup_expired_jobs, cleanup_job_process_artifacts


class _FakeRedisService:
    def __init__(self, jobs: dict[str, Job]) -> None:
        self.jobs = dict(jobs)
        self.deleted_job_ids: list[str] = []

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def delete_job(self, job_id: str) -> None:
        self.deleted_job_ids.append(job_id)
        self.jobs.pop(job_id, None)


def _make_job(
    *,
    job_id: str,
    status: JobStatus,
    expires_at: datetime,
) -> Job:
    created_at = expires_at - timedelta(hours=1)
    return Job(
        job_id=job_id,
        status=status,
        stage=JobStage.done if status == JobStatus.completed else JobStage.cleanup,
        progress=100 if status != JobStatus.pending else 20,
        created_at=created_at,
        expires_at=expires_at,
        message="test",
        error=None,
        layout_mode=LayoutMode.fidelity,
        debug_events=[],
    )


def test_cleanup_job_process_artifacts_removes_artifacts_dir(tmp_path: Path) -> None:
    job_dir = tmp_path / "job-1"
    artifacts_dir = job_dir / "artifacts" / "page_renders"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "page-0000.png").write_bytes(b"png")

    removed = cleanup_job_process_artifacts(job_dir)

    assert removed is True
    assert not (job_dir / "artifacts").exists()


def test_cleanup_expired_jobs_deletes_terminal_job_by_metadata_expiry(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    job_dir = tmp_path / "job-expired"
    job_dir.mkdir()
    (job_dir / "output.pptx").write_bytes(b"pptx")

    redis_service = _FakeRedisService(
        {
            "job-expired": _make_job(
                job_id="job-expired",
                status=JobStatus.completed,
                expires_at=now - timedelta(minutes=1),
            )
        }
    )

    stats = cleanup_expired_jobs(
        now=now,
        job_root_dir=tmp_path,
        ttl_minutes=1440,
        redis_service=redis_service,
    )

    assert stats["deleted_dirs"] == 1
    assert stats["deleted_metadata"] == 1
    assert not job_dir.exists()
    assert redis_service.deleted_job_ids == ["job-expired"]


def test_cleanup_expired_jobs_keeps_active_job_even_when_directory_is_old(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    job_dir = tmp_path / "job-active"
    job_dir.mkdir()
    (job_dir / "input.pdf").write_bytes(b"pdf")

    old_epoch = (now - timedelta(days=3)).timestamp()
    os.utime(job_dir, (old_epoch, old_epoch))

    redis_service = _FakeRedisService(
        {
            "job-active": _make_job(
                job_id="job-active",
                status=JobStatus.processing,
                expires_at=now + timedelta(hours=1),
            )
        }
    )

    stats = cleanup_expired_jobs(
        now=now,
        job_root_dir=tmp_path,
        ttl_minutes=1440,
        redis_service=redis_service,
    )

    assert stats["deleted_dirs"] == 0
    assert job_dir.exists()
    assert redis_service.deleted_job_ids == []
