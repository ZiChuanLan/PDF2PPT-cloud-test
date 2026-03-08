"""Job artifact retention and expired-job cleanup helpers."""

from __future__ import annotations

import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..job_paths import get_job_root_dir
from ..models.job import JobStatus

logger = logging.getLogger(__name__)

_TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.completed,
        JobStatus.failed,
        JobStatus.cancelled,
    }
)


def cleanup_job_process_artifacts(job_dir: Path) -> bool:
    """Delete process/debug artifact files for a finished job."""

    artifacts_dir = Path(job_dir) / "artifacts"
    if not artifacts_dir.exists():
        return False

    shutil.rmtree(artifacts_dir)
    return True


def cleanup_expired_jobs(
    *,
    now: datetime | None = None,
    job_root_dir: Path | None = None,
    ttl_minutes: int | None = None,
    redis_service: Any | None = None,
) -> dict[str, int]:
    """Delete expired terminal job directories and best-effort metadata."""

    settings = get_settings()
    root = Path(job_root_dir) if job_root_dir is not None else get_job_root_dir()
    ttl_seconds = max(
        60,
        int(
            (
                ttl_minutes
                if ttl_minutes is not None
                else getattr(settings, "job_ttl_minutes", 1440)
            )
            or 1440
        )
        * 60,
    )
    current_time = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    cutoff_epoch = current_time.timestamp() - float(ttl_seconds)

    stats = {
        "scanned": 0,
        "deleted_dirs": 0,
        "deleted_metadata": 0,
    }

    if not root.exists():
        return stats

    for job_dir in root.iterdir():
        if not job_dir.is_dir():
            continue

        stats["scanned"] += 1
        job_id = str(job_dir.name)
        job = None
        if redis_service is not None:
            try:
                job = redis_service.get_job(job_id)
            except Exception:
                logger.warning("Failed to read job metadata during cleanup: %s", job_id, exc_info=True)

        expired = False
        if job is not None:
            if job.status not in _TERMINAL_JOB_STATUSES:
                continue
            expires_at = job.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            else:
                expires_at = expires_at.astimezone(timezone.utc)
            expired = expires_at <= current_time
        else:
            try:
                expired = float(job_dir.stat().st_mtime) <= cutoff_epoch
            except FileNotFoundError:
                continue

        if not expired:
            continue

        try:
            shutil.rmtree(job_dir)
            stats["deleted_dirs"] += 1
        except FileNotFoundError:
            continue
        except Exception:
            logger.warning("Failed to delete expired job directory: %s", job_dir, exc_info=True)
            continue

        if redis_service is not None:
            try:
                redis_service.delete_job(job_id)
                stats["deleted_metadata"] += 1
            except Exception:
                logger.warning("Failed to delete expired job metadata: %s", job_id, exc_info=True)

    return stats


def start_job_cleanup_daemon(*, redis_service: Any | None = None) -> tuple[threading.Event, threading.Thread]:
    """Start a background thread that periodically deletes expired jobs."""

    settings = get_settings()
    interval_seconds = max(
        60,
        int(getattr(settings, "job_cleanup_interval_minutes", 15) or 15) * 60,
    )
    stop_event = threading.Event()

    def _run() -> None:
        logger.info(
            "Job cleanup daemon started (ttl_minutes=%s, interval_minutes=%s)",
            int(getattr(settings, "job_ttl_minutes", 1440) or 1440),
            int(getattr(settings, "job_cleanup_interval_minutes", 15) or 15),
        )
        while not stop_event.is_set():
            try:
                stats = cleanup_expired_jobs(redis_service=redis_service)
                if stats["deleted_dirs"] or stats["deleted_metadata"]:
                    logger.info(
                        "Expired job cleanup removed %s directories and %s metadata entries",
                        stats["deleted_dirs"],
                        stats["deleted_metadata"],
                    )
            except Exception:
                logger.exception("Job cleanup daemon sweep failed")

            stop_event.wait(interval_seconds)

        logger.info("Job cleanup daemon stopped")

    thread = threading.Thread(
        target=_run,
        name="job-cleanup-daemon",
        daemon=True,
    )
    thread.start()
    return stop_event, thread
