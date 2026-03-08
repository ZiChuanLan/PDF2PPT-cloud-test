"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings, parse_cors_allow_origins
from app.logging_config import (
    get_logger,
    set_request_id,
    setup_logging,
)
from app.models.error import AppException, ErrorCode, ErrorResponse
from app.routers import jobs_router, models_router
from app.services.job_cleanup import start_job_cleanup_daemon
from app.services.redis_service import get_redis_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Application starting up")
    cleanup_stop_event = None
    cleanup_thread = None
    try:
        cleanup_stop_event, cleanup_thread = start_job_cleanup_daemon(
            redis_service=get_redis_service()
        )
        yield
    finally:
        if cleanup_stop_event is not None:
            cleanup_stop_event.set()
        if cleanup_thread is not None and cleanup_thread.is_alive():
            cleanup_thread.join(timeout=5)
        logger.info("Application shutting down")


app = FastAPI(
    title="PDF to PPT API",
    description="Convert PDF documents to PowerPoint presentations",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
settings = get_settings()
cors_allow_origins = parse_cors_allow_origins(settings.cors_allow_origins)
allow_credentials = "*" not in cors_allow_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(jobs_router)
app.include_router(models_router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add request ID to each request."""
    request_id = request.headers.get("X-Request-ID") or set_request_id()
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle application-specific exceptions."""
    logger.warning(f"AppException: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response().model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    error = ErrorResponse(
        code=ErrorCode.INTERNAL_ERROR.value,
        message="An internal error occurred",
    )
    return JSONResponse(
        status_code=500,
        content=error.model_dump(),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/test-error")
async def test_error():
    """Test endpoint to verify error handling."""
    raise AppException(
        code=ErrorCode.PDF_ENCRYPTED,
        message="Test: PDF is password-protected",
        details={"test": True},
    )
