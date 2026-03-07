"""Structured logging configuration."""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

# Context variables for request/job tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
job_stage_var: ContextVar[str | None] = ContextVar("job_stage", default=None)

# Sensitive fields to filter from logs
SENSITIVE_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "token",
        "secret",
        "password",
        "authorization",
        "auth",
        "credential",
        "key",
    }
)


def filter_sensitive(data: dict[str, Any]) -> dict[str, Any]:
    """Filter sensitive fields from a dictionary."""
    return {
        k: "[REDACTED]" if k.lower() in SENSITIVE_FIELDS else v for k, v in data.items()
    }


class StructuredFormatter(logging.Formatter):
    """JSON-like structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with request/job context."""
        request_id = request_id_var.get()
        job_id = job_id_var.get()
        job_stage = job_stage_var.get()

        # Build structured message
        parts = [f"[{record.levelname}]"]

        if request_id:
            parts.append(f"[req:{request_id[:8]}]")
        if job_id:
            parts.append(f"[job:{job_id[:8]}]")
        if job_stage:
            parts.append(f"[stage:{job_stage}]")

        parts.append(f"{record.name}: {record.getMessage()}")

        return " ".join(parts)


class JobDebugHandler(logging.Handler):
    """Mirror in-job log lines into the job status payload for the frontend."""

    def emit(self, record: logging.LogRecord) -> None:
        job_id = job_id_var.get()
        if not job_id:
            return

        try:
            from .services.redis_service import get_redis_service

            message = str(record.getMessage() or "").strip()
            if not message:
                return

            get_redis_service().append_debug_event(
                job_id,
                level=str(record.levelname or "INFO").lower(),
                message=message,
                source=record.name,
                stage=job_stage_var.get(),
                dedupe=True,
            )
        except Exception:
            # Never let debug mirroring break the main job path.
            return


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add structured handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(handler)

    debug_handler = JobDebugHandler()
    debug_handler.setLevel(logging.INFO)
    root_logger.addHandler(debug_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def set_request_id(request_id: str | None = None) -> str:
    """Set the request ID for the current context."""
    rid = request_id or generate_request_id()
    request_id_var.set(rid)
    return rid


def set_job_id(job_id: str | None) -> None:
    """Set the job ID for the current context."""
    job_id_var.set(job_id)


def set_job_stage(stage: str | None) -> None:
    """Set the job stage for the current context."""
    job_stage_var.set(stage)


def get_request_id() -> str | None:
    """Get the current request ID."""
    return request_id_var.get()


def get_job_id() -> str | None:
    """Get the current job ID."""
    return job_id_var.get()
