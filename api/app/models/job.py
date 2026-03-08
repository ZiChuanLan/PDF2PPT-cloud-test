# pyright: reportMissingImports=false, reportMissingTypeArgument=false

"""Job models for async PDF to PPT conversion."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job execution status."""

    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobStage(str, Enum):
    """Detailed job processing stages."""

    upload_received = "upload_received"
    queued = "queued"
    parsing = "parsing"
    ocr = "ocr"
    layout_assist = "layout_assist"
    pptx_generating = "pptx_generating"
    packaging = "packaging"
    cleanup = "cleanup"
    done = "done"


class LayoutMode(str, Enum):
    """Layout strategy selection."""

    fidelity = "fidelity"
    assist = "assist"


class JobDebugEvent(BaseModel):
    """Single debug event emitted while a job is running."""

    seq: int = Field(..., ge=1, description="Monotonic per-job event sequence")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp",
    )
    level: str = Field("info", description="Log level or event severity")
    message: str = Field(..., description="Event message")
    source: Optional[str] = Field(None, description="Logger or producer name")
    stage: Optional[str] = Field(None, description="Associated job stage code")
    progress: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Progress snapshot when the event was emitted",
    )


class Job(BaseModel):
    """Job metadata model."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    stage: JobStage = Field(..., description="Current processing stage")
    progress: int = Field(0, ge=0, le=100, description="Progress percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    expires_at: datetime = Field(..., description="Job expiration timestamp")
    message: Optional[str] = Field(None, description="Human-readable status message")
    error: Optional[dict[str, Any]] = Field(None, description="Error details if failed")
    layout_mode: LayoutMode = Field(
        LayoutMode.fidelity,
        description="Layout mode: fidelity (no AI) or assist (LLM-assisted)",
    )
    debug_events: list[JobDebugEvent] = Field(
        default_factory=list,
        description="Recent per-job debug events shown in the frontend",
    )


class JobCreateResponse(BaseModel):
    """Response model for job creation."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    expires_at: datetime = Field(..., description="Job expiration timestamp")


class LocalOcrCheckRequest(BaseModel):
    """Request model for local OCR runtime checks."""

    provider: Optional[str] = Field(
        "tesseract",
        description=(
            "Local OCR probe target "
            "(tesseract, paddle, tesseract_models, paddle_models)"
        ),
    )
    language: Optional[str] = Field(
        None,
        description=(
            "Requested OCR language hint. For tesseract use chi_sim+eng; "
            "for paddle use ch/en etc."
        ),
    )


class LocalOcrCheckResult(BaseModel):
    """Detailed local OCR environment check result."""

    provider: str
    requested_language: str
    requested_languages: list[str] = Field(default_factory=list)
    python_package_available: bool
    binary_available: bool
    version: Optional[str] = None
    available_languages: list[str] = Field(default_factory=list)
    missing_languages: list[str] = Field(default_factory=list)
    model_root_dir: Optional[str] = None
    required_models: list[str] = Field(default_factory=list)
    found_models: list[str] = Field(default_factory=list)
    missing_models: list[str] = Field(default_factory=list)
    model_files: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    ready: bool
    message: str


class LocalOcrCheckResponse(BaseModel):
    """Response model for local OCR runtime checks."""

    ok: bool = Field(..., description="Whether local OCR is ready")
    check: LocalOcrCheckResult


class AiOcrCheckRequest(BaseModel):
    """Request model for remote AI OCR capability checks."""

    provider: Optional[str] = Field(
        "auto",
        description=(
            "AI OCR vendor adapter (auto, openai, siliconflow, deepseek, ppio, novita)"
        ),
    )
    api_key: str = Field(..., description="API key for AI OCR provider")
    base_url: Optional[str] = Field(
        None, description="Optional OpenAI-compatible base URL"
    )
    model: str = Field(..., description="Target AI OCR model identifier")
    ocr_ai_chain_mode: Optional[str] = Field(
        "direct",
        description="AI OCR chain mode (direct, doc_parser, layout_block)",
    )
    ocr_ai_layout_model: Optional[str] = Field(
        "pp_doclayout_v3",
        description="Local layout model for layout_block chain",
    )
    ocr_paddle_vl_docparser_max_side_px: Optional[int] = Field(
        None,
        ge=0,
        le=6000,
        description=(
            "Optional max long-edge in pixels for PaddleOCR-VL doc_parser input images; "
            "0 disables downscale"
        ),
    )
    ocr_ai_block_concurrency: Optional[int] = Field(
        None,
        ge=1,
        le=8,
        description="Optional per-page block concurrency for layout_block OCR",
    )
    ocr_ai_requests_per_minute: Optional[int] = Field(
        None,
        ge=1,
        le=2000,
        description="Optional shared requests-per-minute cap for AI OCR requests",
    )
    ocr_ai_tokens_per_minute: Optional[int] = Field(
        None,
        ge=1,
        le=2_000_000,
        description="Optional shared tokens-per-minute cap for AI OCR requests",
    )
    ocr_ai_max_retries: Optional[int] = Field(
        None,
        ge=0,
        le=8,
        description="Optional retry count for retryable AI OCR chat/completions failures",
    )


class AiOcrCheckSampleItem(BaseModel):
    """Sample OCR item returned by capability check."""

    text: str
    bbox: list[float] = Field(default_factory=list)
    confidence: Optional[float] = None


class AiOcrCheckResult(BaseModel):
    """Detailed AI OCR capability check result."""

    provider: str
    model: str
    base_url: Optional[str] = None
    route_kind: Optional[str] = None
    elapsed_ms: int
    items_count: int
    valid_bbox_items: int
    ready: bool
    message: str
    error: Optional[str] = None
    sample_items: list[AiOcrCheckSampleItem] = Field(default_factory=list)


class AiOcrCheckResponse(BaseModel):
    """Response model for AI OCR capability checks."""

    ok: bool = Field(..., description="Whether the model can return OCR with bbox")
    check: AiOcrCheckResult


class JobStatusResponse(BaseModel):
    """Response model for job status query."""

    job_id: str
    status: JobStatus
    stage: JobStage
    progress: int
    created_at: datetime
    expires_at: datetime
    message: Optional[str] = None
    error: Optional[dict[str, Any]] = None
    debug_events: list[JobDebugEvent] = Field(default_factory=list)


class JobListItem(BaseModel):
    """Response item for job list query."""

    job_id: str
    status: JobStatus
    stage: JobStage
    progress: int
    created_at: datetime
    expires_at: datetime
    message: Optional[str] = None
    error: Optional[dict[str, Any]] = None
    # 1-based queue position when the job is still waiting in Redis queue.
    queue_position: Optional[int] = None
    # queued | running | waiting | done
    queue_state: Optional[str] = None


class JobListResponse(BaseModel):
    """Response model for job list query."""

    jobs: list[JobListItem]
    queue_size: int = 0
    returned: int = 0


class JobEvent(BaseModel):
    """SSE event model for job progress updates."""

    job_id: str
    status: JobStatus
    stage: JobStage
    progress: int
    message: Optional[str] = None
    error: Optional[dict[str, Any]] = None


class JobArtifactImage(BaseModel):
    """Single artifact image metadata."""

    page_index: int
    path: str
    url: str


class JobArtifactsResponse(BaseModel):
    """Artifact manifest used by frontend tracking/diff views."""

    job_id: str
    status: Optional[JobStatus] = None
    artifacts_retained: bool = False
    source_pdf_url: Optional[str] = None
    original_images: list[JobArtifactImage] = Field(default_factory=list)
    cleaned_images: list[JobArtifactImage] = Field(default_factory=list)
    final_preview_images: list[JobArtifactImage] = Field(default_factory=list)
    ocr_overlay_images: list[JobArtifactImage] = Field(default_factory=list)
    layout_before_images: list[JobArtifactImage] = Field(default_factory=list)
    layout_after_images: list[JobArtifactImage] = Field(default_factory=list)
    available_pages: list[int] = Field(default_factory=list)
