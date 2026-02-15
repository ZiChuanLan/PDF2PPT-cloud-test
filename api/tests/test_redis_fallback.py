from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
