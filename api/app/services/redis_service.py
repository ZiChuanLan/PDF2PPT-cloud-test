# pyright: reportMissingImports=false

"""Redis service for job metadata storage."""

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Optional, cast

import redis

from ..config import get_settings
from ..models.job import Job, JobStage, JobStatus, LayoutMode

logger = logging.getLogger(__name__)

_TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.completed,
        JobStatus.failed,
        JobStatus.cancelled,
    }
)


@dataclass(frozen=True)
class _MemValue:
    value: str
    expires_at_epoch: float | None


class _InMemoryRedis:
    """Tiny Redis-like subset for local QA runs.

    Only supports the commands this project uses:
    - get
    - setex
    - delete
    - keys (prefix*)
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, _MemValue] = {}

    def _purge_if_expired(self, key: str) -> None:
        entry = self._data.get(key)
        if not entry:
            return
        exp = entry.expires_at_epoch
        if exp is not None and exp <= time.time():
            self._data.pop(key, None)

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        expires_at = None
        if ttl_seconds is not None and int(ttl_seconds) > 0:
            expires_at = time.time() + int(ttl_seconds)
        with self._lock:
            self._data[str(key)] = _MemValue(
                value=str(value), expires_at_epoch=expires_at
            )

    def get(self, key: str) -> str | None:
        with self._lock:
            self._purge_if_expired(str(key))
            entry = self._data.get(str(key))
            return entry.value if entry else None

    def delete(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                self._data.pop(str(k), None)

    def keys(self, pattern: str) -> list[str]:
        pat = str(pattern)
        if pat.endswith("*"):
            prefix = pat[:-1]
            out: list[str] = []
            with self._lock:
                for k in list(self._data.keys()):
                    self._purge_if_expired(k)
                for k in self._data.keys():
                    if k.startswith(prefix):
                        out.append(k)
            return out

        value = self.get(pat)
        return [pat] if value is not None else []


_memory_redis = _InMemoryRedis()


class RedisService:
    """Service for managing job metadata in Redis."""

    def __init__(self):
        settings = get_settings()
        self.ttl_seconds = settings.job_ttl_minutes * 60
        self._memory_backend = False
        redis_url = str(settings.redis_url or "").strip()
        if redis_url.startswith("memory://"):
            self.redis_client = _memory_redis
            self._memory_backend = True
        else:
            candidate = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            try:
                candidate.ping()
                self.redis_client = candidate
            except Exception as e:
                logger.warning(
                    "Redis unavailable (%s). Falling back to in-memory job store: %s",
                    redis_url or "<empty>",
                    e,
                )
                self.redis_client = _memory_redis
                self._memory_backend = True

    def is_memory_backend(self) -> bool:
        """Whether RedisService currently uses in-memory backend."""
        return bool(self._memory_backend)

    def _job_key(self, job_id: str) -> str:
        """Generate Redis key for job metadata."""
        return f"job:{job_id}"

    def _cancel_key(self, job_id: str) -> str:
        """Generate Redis key for cancellation flag."""
        return f"job:{job_id}:cancel"

    def _persist_job(self, job: Job) -> Job:
        """Persist job metadata while keeping the exposed expiry aligned with TTL."""
        job.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.ttl_seconds
        )
        self.redis_client.setex(
            self._job_key(job.job_id),
            self.ttl_seconds,
            job.model_dump_json(),
        )
        return job

    def create_job(self, job_id: str) -> Job:
        """Create a new job in Redis."""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.ttl_seconds)

        job = Job(
            job_id=job_id,
            status=JobStatus.pending,
            stage=JobStage.upload_received,
            progress=0,
            created_at=now,
            expires_at=expires_at,
            message="Job created, waiting to be queued",
            error=None,
            layout_mode=LayoutMode.fidelity,
        )

        # Store job metadata with TTL
        return self._persist_job(job)

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job metadata from Redis."""
        data = self.redis_client.get(self._job_key(job_id))
        if not data:
            return None

        if not isinstance(data, (str, bytes, bytearray)):
            return None

        return Job.model_validate_json(cast(str | bytes | bytearray, data))

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        stage: Optional[JobStage] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        error: dict[str, Any] | None = None,
    ) -> Optional[Job]:
        """Update job metadata in Redis."""
        job = self.get_job(job_id)
        if not job:
            return None

        # Guard terminal states from being overwritten by stale writers.
        if job.status in _TERMINAL_JOB_STATUSES and status != job.status:
            return job

        # Update fields
        if status is not None:
            job.status = status
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message
        if error is not None:
            job.error = error

        return self._persist_job(job)

    def refresh_job_ttl(self, job_id: str) -> Optional[Job]:
        """Refresh TTL and expiration timestamp without changing job state."""
        job = self.get_job(job_id)
        if not job:
            return None

        return self._persist_job(job)

    def set_cancel_flag(self, job_id: str) -> None:
        """Set cancellation flag for a job."""
        self.redis_client.setex(
            self._cancel_key(job_id),
            self.ttl_seconds,
            "1",
        )

    def is_cancelled(self, job_id: str) -> bool:
        """Check if job has been cancelled."""
        return bool(self.redis_client.get(self._cancel_key(job_id)))

    def delete_job(self, job_id: str) -> None:
        """Delete job metadata from Redis."""
        self.redis_client.delete(self._job_key(job_id))
        self.redis_client.delete(self._cancel_key(job_id))

    def get_all_job_ids(self) -> list[str]:
        """Get all job IDs from Redis."""
        keys = self.redis_client.keys("job:*")
        if not isinstance(keys, list):
            return []
        # Filter out cancel keys and extract job IDs
        job_ids: list[str] = []
        for key in keys:
            key_str = str(key)
            if ":cancel" not in key_str:
                job_id = key_str.replace("job:", "")
                job_ids.append(job_id)
        return job_ids

    def list_jobs(self, *, limit: int = 50) -> list[Job]:
        """List jobs ordered by creation time descending."""
        limit = max(1, int(limit))
        jobs: list[Job] = []
        for job_id in self.get_all_job_ids():
            job = self.get_job(job_id)
            if job is not None:
                jobs.append(job)

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]


# Singleton instance
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """Get or create Redis service singleton."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service


def reset_redis_service() -> None:
    """Reset the RedisService singleton (used by tests)."""
    global _redis_service
    _redis_service = None
