#!/usr/bin/env python3
"""Cleanup script to purge expired job directories."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.job_cleanup import cleanup_expired_jobs as cleanup_expired_job_data
from app.services.redis_service import get_redis_service


def cleanup_expired_jobs():
    """Delete job directories older than TTL."""
    settings = get_settings()
    stats = cleanup_expired_job_data(
        ttl_minutes=settings.job_ttl_minutes,
        redis_service=get_redis_service(),
    )
    print(
        "Cleanup complete. "
        f"Scanned {stats['scanned']} job directories; "
        f"deleted {stats['deleted_dirs']} directories and "
        f"{stats['deleted_metadata']} metadata entries."
    )


if __name__ == "__main__":
    cleanup_expired_jobs()
