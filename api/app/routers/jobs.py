# pyright: reportMissingImports=false

"""Job API endpoints."""

import asyncio
import io
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import quote

import pymupdf
from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from rq import Queue
from rq.job import Job as RqJob

try:
    from rq.command import send_stop_job_command
except Exception:  # pragma: no cover - compatibility with older RQ builds
    send_stop_job_command = None

from ..config import get_settings
from ..job_options import validate_and_normalize_job_options
from ..job_paths import (
    ensure_job_dir as ensure_job_dir_via_paths,
    get_job_dir as get_job_dir_via_paths,
    resolve_artifact_file,
)
from ..logging_config import get_logger
from ..models.error import AppException, ErrorCode
from ..models.job import (
    AiOcrCheckRequest,
    AiOcrCheckResponse,
    AiOcrCheckResult,
    AiOcrCheckSampleItem,
    JobCreateResponse,
    JobEvent,
    JobArtifactImage,
    JobArtifactsResponse,
    LocalOcrCheckRequest,
    LocalOcrCheckResult,
    LocalOcrCheckResponse,
    JobListItem,
    JobListResponse,
    JobStage,
    JobStatus,
    JobStatusResponse,
)
from ..convert.ocr import (
    _coerce_bbox_xyxy,
    create_remote_ocr_client,
    probe_local_paddle_models,
    probe_local_paddleocr,
    probe_local_tesseract_models,
    probe_local_tesseract,
)
from ..services.redis_service import get_redis_service
from ..worker import get_redis_connection, process_pdf_job

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

_AI_OCR_PROBE_FONT_CACHE: dict[tuple[int, bool], tuple[Any, bool]] = {}
_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
_DEFAULT_UPLOAD_IMAGE_DPI = 144.0


def _normalize_upload_content_type(value: str | None) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def _classify_upload_kind(*, filename: str, content_type: str | None) -> str | None:
    suffix = Path(filename or "").suffix.lower()
    normalized_content_type = _normalize_upload_content_type(content_type)
    if normalized_content_type in _SUPPORTED_IMAGE_MIME_TYPES:
        return "image"
    if normalized_content_type == "application/pdf":
        return "pdf"
    if suffix in _SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    return None


def _resolve_upload_image_dpi(image: Image.Image) -> tuple[float, float]:
    raw_dpi = image.info.get("dpi")

    def _normalize_axis(value: Any) -> float | None:
        try:
            numeric = float(value)
        except Exception:
            return None
        if not (36.0 <= numeric <= 1200.0):
            return None
        return numeric

    if isinstance(raw_dpi, (tuple, list)):
        x = _normalize_axis(raw_dpi[0] if len(raw_dpi) >= 1 else None)
        y = _normalize_axis(raw_dpi[1] if len(raw_dpi) >= 2 else None)
    else:
        x = _normalize_axis(raw_dpi)
        y = x

    resolved_x = x or _DEFAULT_UPLOAD_IMAGE_DPI
    resolved_y = y or resolved_x or _DEFAULT_UPLOAD_IMAGE_DPI
    return resolved_x, resolved_y


def _flatten_upload_image(image: Image.Image) -> Image.Image:
    normalized = ImageOps.exif_transpose(image)
    has_alpha = "A" in normalized.getbands()
    has_palette_transparency = normalized.info.get("transparency") is not None
    if has_alpha or has_palette_transparency:
        rgba = normalized.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        rgba.close()
        if normalized is not image:
            normalized.close()
        return flattened

    converted = normalized.convert("RGB")
    if normalized is not image:
        normalized.close()
    return converted


def _write_upload_as_input_pdf(
    *,
    filename: str,
    content_type: str | None,
    content: bytes,
    output_path: Path,
) -> str:
    upload_kind = _classify_upload_kind(
        filename=filename,
        content_type=content_type,
    )
    if upload_kind == "pdf":
        output_path.write_bytes(content)
        return upload_kind
    if upload_kind != "image":
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Only PDF, PNG, JPG, JPEG, and WEBP files are supported",
            details={
                "filename": filename,
                "content_type": _normalize_upload_content_type(content_type),
            },
        )

    try:
        source_image = Image.open(io.BytesIO(content))
        source_image.load()
    except UnidentifiedImageError as e:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Uploaded image could not be decoded",
            details={"filename": filename, "error": str(e)},
        ) from e
    except Exception as e:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Failed to read uploaded image",
            details={"filename": filename, "error": str(e)},
        ) from e

    prepared_image: Image.Image | None = None
    pdf_doc: pymupdf.Document | None = None
    try:
        prepared_image = _flatten_upload_image(source_image)
        dpi_x, dpi_y = _resolve_upload_image_dpi(source_image)
        page_width_pt = max(1.0, float(prepared_image.width) * 72.0 / float(dpi_x))
        page_height_pt = max(1.0, float(prepared_image.height) * 72.0 / float(dpi_y))

        encoded = io.BytesIO()
        suffix = Path(filename or "").suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            prepared_image.save(encoded, format="JPEG", quality=95, subsampling=0)
        else:
            prepared_image.save(encoded, format="PNG")

        pdf_doc = pymupdf.open()
        page = pdf_doc.new_page(width=page_width_pt, height=page_height_pt)
        page.insert_image(page.rect, stream=encoded.getvalue())
        pdf_doc.save(str(output_path))
        return upload_kind
    finally:
        if pdf_doc is not None:
            pdf_doc.close()
        if prepared_image is not None:
            prepared_image.close()
        source_image.close()


def _sync_rq_cancel_state(*, job_id: str, status: JobStatus) -> None:
    """Mirror API-level cancellation into RQ so queued/running jobs unblock quickly."""
    redis_service = get_redis_service()
    if redis_service.is_memory_backend():
        return

    try:
        redis_conn = get_redis_connection()
    except Exception as e:
        logger.warning(
            "Failed to acquire Redis connection for job %s cancel: %s", job_id, e
        )
        return

    normalized_status = str(
        status.value if isinstance(status, JobStatus) else status
    ).strip().lower()
    try:
        if normalized_status == JobStatus.pending.value:
            rq_job = RqJob.fetch(job_id, connection=redis_conn)
            rq_job.cancel()
            logger.info("Cancelled queued RQ job %s", job_id)
            return

        if (
            normalized_status == JobStatus.processing.value
            and send_stop_job_command is not None
        ):
            send_stop_job_command(redis_conn, job_id)
            logger.info("Sent stop command to running RQ job %s", job_id)
            return

        if normalized_status == JobStatus.processing.value:
            logger.warning(
                "RQ stop command unavailable while cancelling running job %s",
                job_id,
            )
    except Exception as e:
        logger.warning("Failed to sync RQ cancel state for job %s: %s", job_id, e)


@router.post("/ocr/local/check", response_model=LocalOcrCheckResponse)
async def check_local_ocr(payload: LocalOcrCheckRequest):
    """Check whether local OCR runtime is available."""
    provider_id = (payload.provider or "tesseract").strip().lower()

    try:
        if provider_id in {"tesseract", "local", "local_tesseract"}:
            probe = probe_local_tesseract(language=payload.language)
        elif provider_id in {"paddle", "paddleocr", "paddle_ocr"}:
            probe = probe_local_paddleocr(language=payload.language)
        elif provider_id in {"tesseract_models", "tesseract-models", "tess_models"}:
            probe = probe_local_tesseract_models(language=payload.language)
        elif provider_id in {"paddle_models", "paddle-models", "paddleocr_models"}:
            probe = probe_local_paddle_models(language=payload.language)
        else:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="Unsupported local OCR provider",
                details={"provider": payload.provider},
                status_code=400,
            )

        ready = bool(probe.get("ready"))
        return LocalOcrCheckResponse(
            ok=ready,
            check=LocalOcrCheckResult.model_validate(probe),
        )
    except AppException:
        raise
    except Exception as e:
        logger.exception("Local OCR check failed (provider=%s): %s", provider_id, e)
        raise AppException(
            code=ErrorCode.INTERNAL_ERROR,
            message="Failed to check local OCR runtime",
            details={"error": str(e), "provider": provider_id},
            status_code=500,
        )


def _load_ai_ocr_probe_font(*, size_px: int, prefer_cjk: bool) -> tuple[Any, bool]:
    key = (int(max(8, size_px)), bool(prefer_cjk))
    cached = _AI_OCR_PROBE_FONT_CACHE.get(key)
    if cached is not None:
        return cached

    candidates: list[str] = []
    if prefer_cjk:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )

    for path in candidates:
        try:
            font = ImageFont.truetype(path, size=key[0])
            result = (font, False)
            _AI_OCR_PROBE_FONT_CACHE[key] = result
            return result
        except Exception:
            continue

    result = (ImageFont.load_default(), True)
    _AI_OCR_PROBE_FONT_CACHE[key] = result
    return result


def _create_ai_ocr_probe_image() -> Path:
    """Create a synthetic OCR probe image with large, low-risk text blocks."""
    fd, raw_path = tempfile.mkstemp(prefix="ai-ocr-probe-", suffix=".png")
    os.close(fd)
    out = Path(raw_path)

    image = Image.new("RGB", (1440, 960), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font, title_fallback = _load_ai_ocr_probe_font(size_px=56, prefer_cjk=False)
    body_font, body_fallback = _load_ai_ocr_probe_font(size_px=42, prefer_cjk=False)

    draw.rounded_rectangle(
        (48, 48, 1392, 912),
        radius=28,
        outline=(224, 228, 236),
        width=4,
        fill=(255, 255, 255),
    )
    draw.text((104, 116), "PPT OpenCode OCR Check", font=title_font, fill=(18, 18, 18))
    draw.text((104, 260), "Vision OCR Probe 2026", font=body_font, fill=(18, 18, 18))
    draw.text(
        (104, 400),
        "Invoice ID: A-2048-17   Total: 97.5",
        font=body_font,
        fill=(18, 18, 18),
    )
    draw.text(
        (104, 540),
        "Email: hello@example.com",
        font=body_font,
        fill=(18, 18, 18),
    )
    draw.text(
        (104, 680),
        "Status: Ready / bbox check",
        font=body_font,
        fill=(18, 18, 18),
    )

    if title_fallback or body_fallback:
        image = image.resize((2160, 1440), Image.Resampling.BICUBIC)

    image.save(out, format="PNG")
    return out


def _truncate_error(value: Exception | str, *, limit: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _run_ai_ocr_capability_check(
    *,
    provider: str | None,
    api_key: str,
    base_url: str | None,
    model: str,
    ocr_ai_chain_mode: str | None = None,
    ocr_ai_layout_model: str | None = None,
    ocr_ai_prompt_preset: str | None = None,
    ocr_ai_direct_prompt_override: str | None = None,
    ocr_ai_layout_block_prompt_override: str | None = None,
    ocr_ai_image_region_prompt_override: str | None = None,
    ocr_paddle_vl_docparser_max_side_px: int | None = None,
    ocr_ai_block_concurrency: int | None = None,
    ocr_ai_requests_per_minute: int | None = None,
    ocr_ai_tokens_per_minute: int | None = None,
    ocr_ai_max_retries: int | None = None,
) -> AiOcrCheckResponse:
    """Run AI OCR capability check and validate whether bbox items are returned."""
    start = time.perf_counter()
    image_path = _create_ai_ocr_probe_image()
    normalized_provider = (provider or "auto").strip() or "auto"
    normalized_base_url = (base_url or "").strip() or None
    normalized_model = model.strip()

    try:
        effective_block_concurrency = ocr_ai_block_concurrency
        if ocr_ai_chain_mode == "layout_block" and effective_block_concurrency is None:
            effective_block_concurrency = 1

        client = create_remote_ocr_client(
            requested_provider="aiocr",
            ai_api_key=api_key.strip(),
            ai_provider=normalized_provider,
            ai_base_url=normalized_base_url,
            ai_model=normalized_model,
            route_kind=ocr_ai_chain_mode,
            ai_layout_model=ocr_ai_layout_model,
            prompt_preset=ocr_ai_prompt_preset,
            direct_prompt_override=ocr_ai_direct_prompt_override,
            layout_block_prompt_override=ocr_ai_layout_block_prompt_override,
            image_region_prompt_override=ocr_ai_image_region_prompt_override,
            paddle_doc_max_side_px=ocr_paddle_vl_docparser_max_side_px,
            layout_block_max_concurrency=effective_block_concurrency,
            request_rpm_limit=ocr_ai_requests_per_minute,
            request_tpm_limit=ocr_ai_tokens_per_minute,
            request_max_retries=ocr_ai_max_retries,
        )
        raw_items: list[dict[str, Any]] = client.ocr_image(str(image_path))

        valid_bbox_items = 0
        sample_items: list[AiOcrCheckSampleItem] = []
        for item in raw_items or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            bbox = _coerce_bbox_xyxy(item.get("bbox"))
            if not text or not bbox:
                continue
            if float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
                continue
            valid_bbox_items += 1
            if len(sample_items) >= 3:
                continue
            confidence: float | None = None
            try:
                conf_raw = item.get("confidence")
                if conf_raw is not None:
                    confidence = float(conf_raw)
            except Exception:
                confidence = None
            sample_items.append(
                AiOcrCheckSampleItem(
                    text=text[:120],
                    bbox=[
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ],
                    confidence=confidence,
                )
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        ready = valid_bbox_items > 0
        message = (
            "模型可返回有效 bbox OCR 结果" if ready else "模型未返回有效 bbox OCR 结果"
        )
        check = AiOcrCheckResult(
            provider=normalized_provider,
            model=normalized_model,
            base_url=normalized_base_url,
            route_kind=getattr(client, "route_kind", None),
            elapsed_ms=elapsed_ms,
            items_count=len(raw_items or []),
            valid_bbox_items=valid_bbox_items,
            ready=ready,
            message=message,
            sample_items=sample_items,
        )
        return AiOcrCheckResponse(ok=ready, check=check)
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        check = AiOcrCheckResult(
            provider=normalized_provider,
            model=normalized_model,
            base_url=normalized_base_url,
            route_kind=None,
            elapsed_ms=elapsed_ms,
            items_count=0,
            valid_bbox_items=0,
            ready=False,
            message="模型调用失败",
            error=_truncate_error(e),
            sample_items=[],
        )
        return AiOcrCheckResponse(ok=False, check=check)
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/ocr/ai/check", response_model=AiOcrCheckResponse)
async def check_ai_ocr(payload: AiOcrCheckRequest):
    """Check whether the selected AI OCR model can return usable bbox items."""
    api_key = (payload.api_key or "").strip()
    model = (payload.model or "").strip()
    if not api_key:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="api_key is required",
            status_code=400,
        )
    if not model:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="model is required",
            status_code=400,
        )

    try:
        return await asyncio.to_thread(
            _run_ai_ocr_capability_check,
            provider=payload.provider,
            api_key=api_key,
            base_url=payload.base_url,
            model=model,
            ocr_ai_chain_mode=payload.ocr_ai_chain_mode,
            ocr_ai_layout_model=payload.ocr_ai_layout_model,
            ocr_ai_prompt_preset=payload.ocr_ai_prompt_preset,
            ocr_ai_direct_prompt_override=payload.ocr_ai_direct_prompt_override,
            ocr_ai_layout_block_prompt_override=payload.ocr_ai_layout_block_prompt_override,
            ocr_ai_image_region_prompt_override=payload.ocr_ai_image_region_prompt_override,
            ocr_paddle_vl_docparser_max_side_px=payload.ocr_paddle_vl_docparser_max_side_px,
            ocr_ai_block_concurrency=payload.ocr_ai_block_concurrency,
            ocr_ai_requests_per_minute=payload.ocr_ai_requests_per_minute,
            ocr_ai_tokens_per_minute=payload.ocr_ai_tokens_per_minute,
            ocr_ai_max_retries=payload.ocr_ai_max_retries,
        )
    except AppException:
        raise
    except Exception as e:
        logger.exception("AI OCR capability check failed: %s", e)
        raise AppException(
            code=ErrorCode.INTERNAL_ERROR,
            message="Failed to check AI OCR capability",
            details={"error": _truncate_error(e)},
            status_code=500,
        )


def get_job_dir(job_id: str) -> Path:
    """Get job directory path."""
    return get_job_dir_via_paths(job_id)


def ensure_job_dir(job_id: str) -> Path:
    """Create and return job directory."""
    return ensure_job_dir_via_paths(job_id)


def _safe_artifact_path(job_id: str, rel_path: str) -> Path:
    """Resolve an artifact path safely under the job directory."""
    return resolve_artifact_file(job_id, rel_path)


def _collect_page_images(
    *,
    job_dir: Path,
    subdir: str,
    regex: str,
    url_prefix: str,
) -> list[JobArtifactImage]:
    base_dir = job_dir / subdir
    if not base_dir.exists():
        return []
    matcher = re.compile(regex)
    images: list[JobArtifactImage] = []
    for path in sorted(base_dir.glob("*.png")):
        m = matcher.match(path.name)
        if not m:
            continue
        page_index = int(m.group(1))
        rel = str(path.relative_to(job_dir))
        images.append(
            JobArtifactImage(
                page_index=page_index,
                path=rel,
                url=f"{url_prefix}/file?path={quote(rel)}",
            )
        )
    return images


@router.get("", response_model=JobListResponse)
async def list_jobs(
    limit: int = Query(50, ge=1, le=200, description="Max jobs to return"),
):
    """List recent jobs with queue metadata for frontend history/queue panels."""
    redis_service = get_redis_service()
    jobs = redis_service.list_jobs(limit=limit)

    queue_job_ids: list[str] = []
    started_job_ids: set[str] = set()

    if not redis_service.is_memory_backend():
        try:
            redis_conn = get_redis_connection()
            raw_queue_ids = redis_conn.lrange("rq:queue:default", 0, -1) or []
            raw_started_ids = (
                redis_conn.zrange("rq:registry:started:default", 0, -1) or []
            )

            def _to_str(value: object) -> str:
                if isinstance(value, (bytes, bytearray)):
                    return value.decode("utf-8", errors="ignore")
                return str(value)

            queue_job_ids_raw = [_to_str(v) for v in raw_queue_ids if v is not None]
            queue_job_ids = []
            for queued_id in queue_job_ids_raw:
                queued_job = redis_service.get_job(queued_id)
                if queued_job is None:
                    queue_job_ids.append(queued_id)
                    continue
                if queued_job.status in {JobStatus.pending, JobStatus.processing}:
                    queue_job_ids.append(queued_id)
            started_job_ids = {_to_str(v) for v in raw_started_ids if v is not None}
        except Exception as e:
            logger.warning("Failed to load RQ queue metadata: %s", e)

    queue_pos_map = {job_id: idx + 1 for idx, job_id in enumerate(queue_job_ids)}

    items: list[JobListItem] = []
    for job in jobs:
        queue_position = queue_pos_map.get(job.job_id)
        if queue_position is not None:
            queue_state = "queued"
        elif job.status == JobStatus.processing:
            # If backend already marks the job as processing, treat it as running
            # even when RQ registry polling temporarily misses it.
            queue_state = "running"
        elif job.job_id in started_job_ids:
            queue_state = "running"
        elif job.status == JobStatus.pending:
            queue_state = "waiting"
        else:
            queue_state = "done"

        items.append(
            JobListItem(
                job_id=job.job_id,
                status=job.status,
                stage=job.stage,
                progress=job.progress,
                created_at=job.created_at,
                expires_at=job.expires_at,
                message=job.message,
                error=job.error,
                queue_position=queue_position,
                queue_state=queue_state,
            )
        )

    return JobListResponse(
        jobs=items,
        queue_size=len(queue_job_ids),
        returned=len(items),
    )


@router.post("", response_model=JobCreateResponse)
async def create_job(
    file: UploadFile = File(..., description="PDF or image file to convert"),
    enable_ocr: bool = Form(False, description="Enable OCR for scanned PDFs or images"),
    retain_process_artifacts: bool = Form(
        False,
        description=(
            "Keep process/debug artifacts under the job directory for tracking "
            "and visual comparison"
        ),
    ),
    remove_footer_notebooklm: bool = Form(
        False,
        description="Remove detected NotebookLM footer branding text from the output",
    ),
    text_erase_mode: str | None = Form(
        "fill", description="Text erase mode for scanned/mineru pages (smart, fill)"
    ),
    enable_layout_assist: bool = Form(
        False,
        description="Deprecated and ignored: AI layout assistance has been retired",
    ),
    layout_assist_apply_image_regions: bool = Form(
        False,
        description="Deprecated and ignored: layout-assist image region application has been retired",
    ),
    parse_provider: str = Form(
        "local",
        description=(
            "Parser provider (local, baidu_doc, mineru). Legacy `v2` is accepted for backward compatibility "
            "and maps to local+fullpage+AI OCR."
        ),
    ),
    provider: str = Form(
        "openai",
        description="LLM provider identifier (openai, claude, siliconflow, domestic)",
    ),
    api_key: str | None = Form(None, description="Optional API key for AI services"),
    baidu_doc_parse_type: str | None = Form(
        "paddle_vl",
        description="Optional Baidu parser variant when parse_provider=baidu_doc (general, paddle_vl)",
    ),
    base_url: str | None = Form(
        None, description="Optional OpenAI-compatible base URL"
    ),
    model: str | None = Form(
        None, description="Optional OpenAI-compatible model identifier"
    ),
    page_start: int | None = Form(
        None, description="Optional 1-based start page for conversion"
    ),
    page_end: int | None = Form(
        None, description="Optional 1-based end page for conversion"
    ),
    mineru_api_token: str | None = Form(
        None,
        description="Optional MinerU API token (required when parse_provider=mineru)",
    ),
    mineru_base_url: str | None = Form(
        None, description="Optional MinerU API base URL"
    ),
    mineru_model_version: str | None = Form(
        "vlm", description="MinerU model version (pipeline, vlm, MinerU-HTML)"
    ),
    mineru_enable_formula: bool | None = Form(
        True, description="Enable formula recognition in MinerU"
    ),
    mineru_enable_table: bool | None = Form(
        True, description="Enable table recognition in MinerU"
    ),
    mineru_language: str | None = Form(
        None, description="Optional MinerU language hint (e.g. ch, en)"
    ),
    mineru_is_ocr: bool | None = Form(
        None, description="Optional MinerU per-file OCR switch"
    ),
    mineru_hybrid_ocr: bool | None = Form(
        False,
        description="Deprecated and ignored: MinerU no longer layers local hybrid OCR alignment",
    ),
    ocr_provider: str | None = Form(
        "auto",
        description="OCR provider (auto, aiocr, baidu, tesseract, paddle, paddle_local); legacy ai/remote are accepted",
    ),
    ocr_baidu_app_id: str | None = Form(None, description="Optional Baidu OCR App ID"),
    ocr_baidu_api_key: str | None = Form(
        None, description="Optional Baidu OCR API key"
    ),
    ocr_baidu_secret_key: str | None = Form(
        None, description="Optional Baidu OCR secret key"
    ),
    ocr_tesseract_min_confidence: float | None = Form(
        None, description="Optional Tesseract min confidence (0-100)"
    ),
    ocr_tesseract_language: str | None = Form(
        None, description="Optional Tesseract language code (e.g. eng, chi_sim)"
    ),
    ocr_ai_api_key: str | None = Form(
        None, description="Optional AI OCR API key (OpenAI-compatible)"
    ),
    ocr_ai_provider: str | None = Form(
        "auto",
        description="Optional AI OCR vendor adapter (auto, openai, siliconflow, deepseek, ppio, novita)",
    ),
    ocr_ai_base_url: str | None = Form(
        None, description="Optional AI OCR base URL (OpenAI-compatible)"
    ),
    ocr_ai_model: str | None = Form(None, description="Optional AI OCR model name"),
    ocr_ai_chain_mode: str | None = Form(
        "direct",
        description="AI OCR chain mode (direct, doc_parser, layout_block)",
    ),
    ocr_ai_layout_model: str | None = Form(
        "pp_doclayout_v3",
        description="Local layout model for AI OCR layout_block chain",
    ),
    ocr_ai_prompt_preset: str | None = Form(
        "auto",
        description=(
            "Optional OCR prompt preset "
            "(auto, generic_vision, openai_vision, qwen_vl, glm_v, deepseek_ocr)"
        ),
    ),
    ocr_ai_direct_prompt_override: str | None = Form(
        None,
        description="Optional direct OCR prompt override",
    ),
    ocr_ai_layout_block_prompt_override: str | None = Form(
        None,
        description="Optional local layout block OCR prompt override",
    ),
    ocr_ai_image_region_prompt_override: str | None = Form(
        None,
        description="Optional image region detection prompt override",
    ),
    ocr_paddle_vl_docparser_max_side_px: int | None = Form(
        None,
        ge=0,
        le=6000,
        description=(
            "Optional max long-edge in pixels for PaddleOCR-VL doc_parser input images; "
            "0 disables downscale"
        ),
    ),
    ocr_ai_page_concurrency: int | None = Form(
        1,
        ge=1,
        le=8,
        description=(
            "Experimental multi-page AI OCR concurrency for direct/layout_block chains. "
            "1 keeps OCR page processing serial."
        ),
    ),
    ocr_ai_block_concurrency: int | None = Form(
        None,
        ge=1,
        le=8,
        description="Experimental per-page block concurrency override for layout_block OCR",
    ),
    ocr_ai_requests_per_minute: int | None = Form(
        None,
        ge=1,
        le=2000,
        description="Experimental shared requests-per-minute cap for AI OCR requests",
    ),
    ocr_ai_tokens_per_minute: int | None = Form(
        None,
        ge=1,
        le=2_000_000,
        description="Experimental shared tokens-per-minute cap for AI OCR requests",
    ),
    ocr_ai_max_retries: int | None = Form(
        0,
        ge=0,
        le=8,
        description="Experimental retry count for retryable AI OCR chat/completions failures",
    ),
    ocr_render_dpi: int | None = Form(
        None,
        ge=72,
        le=400,
        description="Optional OCR render DPI for scanned-page rasterization before OCR",
    ),
    ocr_geometry_mode: str | None = Form(
        "auto",
        description=(
            "Deprecated geometry hint for aiocr (auto, local_tesseract, direct_ai). "
            "Explicit aiocr now always stays on pure AI bbox at runtime."
        ),
    ),
    scanned_page_mode: str | None = Form(
        "segmented",
        description=(
            "Image placement mode in PPT generation (segmented, fullpage). "
            "Controls whether scanned pages and MinerU text pages keep images as "
            "editable blocks or leave them in the full-page background."
        ),
    ),
    ppt_generation_mode: str | None = Form(
        "standard",
        description=(
            "PPT generation mode (standard, fast, turbo). "
            "Fast and turbo prioritize speed over fidelity, with turbo being the most aggressive."
        ),
    ),
    image_bg_clear_expand_min_pt: float | None = Form(
        None,
        description="Optional min expansion (pt) when clearing background under image overlays",
    ),
    image_bg_clear_expand_max_pt: float | None = Form(
        None,
        description="Optional max expansion (pt) when clearing background under image overlays",
    ),
    image_bg_clear_expand_ratio: float | None = Form(
        None,
        description="Optional expansion ratio for image-overlay background clearing",
    ),
    scanned_image_region_min_area_ratio: float | None = Form(
        None,
        description="Optional min page-area ratio for scanned image region candidates",
    ),
    scanned_image_region_max_area_ratio: float | None = Form(
        None,
        description="Optional max page-area ratio for scanned image region candidates",
    ),
    scanned_image_region_max_aspect_ratio: float | None = Form(
        None,
        description="Optional max aspect ratio threshold for scanned image region candidates",
    ),
    ocr_ai_linebreak_assist: bool | None = Form(
        None,
        description=(
            "Optional AI OCR line-break post-process for OCR blocks (split coarse boxes into line-level boxes). "
            "When omitted (null), the backend may auto-enable this for some OCR providers/models."
        ),
    ),
    ocr_strict_mode: bool | None = Form(
        True,
        description=(
            "Strict OCR quality mode (default on): when enabled, disable implicit OCR fallbacks/downgrades and fail fast on OCR errors"
        ),
    ),
):
    """
    Create a new PDF/image to PPT conversion job.

    Uploads the file, normalizes image inputs into a single-page PDF, and queues it
    for processing.
    Returns immediately with a job_id for tracking progress.
    """
    settings = get_settings()
    redis_service = get_redis_service()
    normalized_options = validate_and_normalize_job_options(
        parse_provider=parse_provider,
        mineru_api_token=mineru_api_token,
        provider=provider,
        api_key=api_key,
        baidu_doc_parse_type=baidu_doc_parse_type,
        ocr_provider=ocr_provider,
        ocr_ai_provider=ocr_ai_provider,
        ocr_ai_api_key=ocr_ai_api_key,
        ocr_ai_model=ocr_ai_model,
        ocr_ai_chain_mode=ocr_ai_chain_mode,
        ocr_ai_layout_model=ocr_ai_layout_model,
        ocr_baidu_app_id=ocr_baidu_app_id,
        ocr_baidu_api_key=ocr_baidu_api_key,
        ocr_baidu_secret_key=ocr_baidu_secret_key,
        ocr_geometry_mode=ocr_geometry_mode,
        text_erase_mode=text_erase_mode,
        scanned_page_mode=scanned_page_mode,
        ppt_generation_mode=ppt_generation_mode,
        page_start=page_start,
        page_end=page_end,
    )
    enable_layout_assist = False
    layout_assist_apply_image_regions = False
    parse_provider_id = normalized_options.parse_provider

    if parse_provider_id == "v2":
        has_v2_key = (
            bool((api_key or "").strip())
            or bool((ocr_ai_api_key or "").strip())
            or bool((getattr(settings, "siliconflow_api_key", "") or "").strip())
            or bool((os.getenv("SILICONFLOW_API_KEY") or "").strip())
        )
        if not has_v2_key:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message=(
                    "api_key or ocr_ai_api_key is required when parse_provider=v2 "
                    "(or set SILICONFLOW_API_KEY env)"
                ),
            )

    filename = file.filename or ""
    normalized_content_type = _normalize_upload_content_type(file.content_type)
    upload_kind = _classify_upload_kind(
        filename=filename,
        content_type=normalized_content_type,
    )
    if upload_kind is None:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Only PDF, PNG, JPG, JPEG, and WEBP files are supported",
            details={
                "filename": file.filename,
                "content_type": normalized_content_type,
            },
        )

    # Generate job ID
    job_id = str(uuid.uuid4())
    job_dir: Path | None = None
    job_created = False

    try:
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.max_file_mb:
            raise AppException(
                code=ErrorCode.FILE_TOO_LARGE,
                message=f"File size exceeds {settings.max_file_mb}MB limit",
                details={"size_mb": file_size_mb, "limit_mb": settings.max_file_mb},
            )

        # Create job directory
        job_dir = ensure_job_dir(job_id)
        input_path = job_dir / "input.pdf"
        upload_kind = _write_upload_as_input_pdf(
            filename=filename,
            content_type=normalized_content_type,
            content=content,
            output_path=input_path,
        )

        # Create job in Redis
        job = redis_service.create_job(job_id)
        job_created = True

        # Persist queued state before starting worker execution so debug events
        # remain ordered even when a local in-process worker begins immediately.
        redis_service.update_job(
            job_id,
            status=JobStatus.pending,
            stage=JobStage.queued,
            message="Job queued for processing",
        )

        # Queue job for processing
        if redis_service.is_memory_backend():
            threading.Thread(
                target=process_pdf_job,
                kwargs={
                    "job_id": job_id,
                    "enable_ocr": enable_ocr,
                    "retain_process_artifacts": retain_process_artifacts,
                    "remove_footer_notebooklm": remove_footer_notebooklm,
                    "enable_layout_assist": enable_layout_assist,
                    "layout_assist_apply_image_regions": layout_assist_apply_image_regions,
                    "provider": normalized_options.provider,
                    "api_key": api_key,
                    "baidu_doc_parse_type": normalized_options.baidu_doc_parse_type,
                    "base_url": base_url,
                    "model": model,
                    "page_start": page_start,
                    "page_end": page_end,
                    "parse_provider": normalized_options.parse_provider,
                    "mineru_api_token": mineru_api_token,
                    "mineru_base_url": mineru_base_url,
                    "mineru_model_version": mineru_model_version,
                    "mineru_enable_formula": mineru_enable_formula,
                    "mineru_enable_table": mineru_enable_table,
                    "mineru_language": mineru_language,
                    "mineru_is_ocr": mineru_is_ocr,
                    "mineru_hybrid_ocr": mineru_hybrid_ocr,
                    "ocr_provider": normalized_options.ocr_provider,
                    "ocr_baidu_app_id": ocr_baidu_app_id,
                    "ocr_baidu_api_key": ocr_baidu_api_key,
                    "ocr_baidu_secret_key": ocr_baidu_secret_key,
                    "ocr_tesseract_min_confidence": ocr_tesseract_min_confidence,
                    "ocr_tesseract_language": ocr_tesseract_language,
                    "ocr_ai_api_key": ocr_ai_api_key,
                    "ocr_ai_provider": normalized_options.ocr_ai_provider,
                    "ocr_ai_base_url": ocr_ai_base_url,
                    "ocr_ai_model": ocr_ai_model,
                    "ocr_ai_chain_mode": normalized_options.ocr_ai_chain_mode,
                    "ocr_ai_layout_model": normalized_options.ocr_ai_layout_model,
                    "ocr_ai_prompt_preset": ocr_ai_prompt_preset,
                    "ocr_ai_direct_prompt_override": ocr_ai_direct_prompt_override,
                    "ocr_ai_layout_block_prompt_override": ocr_ai_layout_block_prompt_override,
                    "ocr_ai_image_region_prompt_override": ocr_ai_image_region_prompt_override,
                    "ocr_paddle_vl_docparser_max_side_px": ocr_paddle_vl_docparser_max_side_px,
                    "ocr_ai_page_concurrency": ocr_ai_page_concurrency,
                    "ocr_ai_block_concurrency": ocr_ai_block_concurrency,
                    "ocr_ai_requests_per_minute": ocr_ai_requests_per_minute,
                    "ocr_ai_tokens_per_minute": ocr_ai_tokens_per_minute,
                    "ocr_ai_max_retries": ocr_ai_max_retries,
                    "ocr_render_dpi": ocr_render_dpi,
                    "ocr_geometry_mode": normalized_options.ocr_geometry_mode,
                    "text_erase_mode": normalized_options.text_erase_mode,
                    "scanned_page_mode": normalized_options.scanned_page_mode,
                    "ppt_generation_mode": normalized_options.ppt_generation_mode,
                    "image_bg_clear_expand_min_pt": image_bg_clear_expand_min_pt,
                    "image_bg_clear_expand_max_pt": image_bg_clear_expand_max_pt,
                    "image_bg_clear_expand_ratio": image_bg_clear_expand_ratio,
                    "scanned_image_region_min_area_ratio": scanned_image_region_min_area_ratio,
                    "scanned_image_region_max_area_ratio": scanned_image_region_max_area_ratio,
                    "scanned_image_region_max_aspect_ratio": scanned_image_region_max_aspect_ratio,
                    "ocr_ai_linebreak_assist": ocr_ai_linebreak_assist,
                    "ocr_strict_mode": ocr_strict_mode,
                    "job_timeout": "1h",
                },
                daemon=True,
            ).start()
        else:
            redis_conn = get_redis_connection()
            queue = Queue(connection=redis_conn)
            queue.enqueue(
                "app.worker.process_pdf_job",
                # NOTE: rq.Queue.enqueue reserves the kwarg name `job_id` for the
                # RQ job identifier, so passing `job_id=...` does NOT forward it to
                # the function. We pass our conversion job_id as a positional arg,
                # and also set the RQ job id to match for easier debugging.
                job_id,
                enable_ocr=enable_ocr,
                retain_process_artifacts=retain_process_artifacts,
                remove_footer_notebooklm=remove_footer_notebooklm,
                enable_layout_assist=enable_layout_assist,
                layout_assist_apply_image_regions=layout_assist_apply_image_regions,
                provider=normalized_options.provider,
                api_key=api_key,
                baidu_doc_parse_type=normalized_options.baidu_doc_parse_type,
                base_url=base_url,
                model=model,
                page_start=page_start,
                page_end=page_end,
                parse_provider=normalized_options.parse_provider,
                mineru_api_token=mineru_api_token,
                mineru_base_url=mineru_base_url,
                mineru_model_version=mineru_model_version,
                mineru_enable_formula=mineru_enable_formula,
                mineru_enable_table=mineru_enable_table,
                mineru_language=mineru_language,
                mineru_is_ocr=mineru_is_ocr,
                mineru_hybrid_ocr=mineru_hybrid_ocr,
                ocr_provider=normalized_options.ocr_provider,
                ocr_baidu_app_id=ocr_baidu_app_id,
                ocr_baidu_api_key=ocr_baidu_api_key,
                ocr_baidu_secret_key=ocr_baidu_secret_key,
                ocr_tesseract_min_confidence=ocr_tesseract_min_confidence,
                ocr_tesseract_language=ocr_tesseract_language,
                ocr_ai_api_key=ocr_ai_api_key,
                ocr_ai_provider=normalized_options.ocr_ai_provider,
                ocr_ai_base_url=ocr_ai_base_url,
                ocr_ai_model=ocr_ai_model,
                ocr_ai_chain_mode=normalized_options.ocr_ai_chain_mode,
                ocr_ai_layout_model=normalized_options.ocr_ai_layout_model,
                ocr_ai_prompt_preset=ocr_ai_prompt_preset,
                ocr_ai_direct_prompt_override=ocr_ai_direct_prompt_override,
                ocr_ai_layout_block_prompt_override=ocr_ai_layout_block_prompt_override,
                ocr_ai_image_region_prompt_override=ocr_ai_image_region_prompt_override,
                ocr_paddle_vl_docparser_max_side_px=ocr_paddle_vl_docparser_max_side_px,
                ocr_ai_page_concurrency=ocr_ai_page_concurrency,
                ocr_ai_block_concurrency=ocr_ai_block_concurrency,
                ocr_ai_requests_per_minute=ocr_ai_requests_per_minute,
                ocr_ai_tokens_per_minute=ocr_ai_tokens_per_minute,
                ocr_ai_max_retries=ocr_ai_max_retries,
                ocr_render_dpi=ocr_render_dpi,
                ocr_geometry_mode=normalized_options.ocr_geometry_mode,
                text_erase_mode=normalized_options.text_erase_mode,
                scanned_page_mode=normalized_options.scanned_page_mode,
                ppt_generation_mode=normalized_options.ppt_generation_mode,
                image_bg_clear_expand_min_pt=image_bg_clear_expand_min_pt,
                image_bg_clear_expand_max_pt=image_bg_clear_expand_max_pt,
                image_bg_clear_expand_ratio=image_bg_clear_expand_ratio,
                scanned_image_region_min_area_ratio=scanned_image_region_min_area_ratio,
                scanned_image_region_max_area_ratio=scanned_image_region_max_area_ratio,
                scanned_image_region_max_aspect_ratio=scanned_image_region_max_aspect_ratio,
                ocr_ai_linebreak_assist=ocr_ai_linebreak_assist,
                ocr_strict_mode=ocr_strict_mode,
                job_id=job_id,
                job_timeout="1h",
                # Avoid leaking API keys in worker logs. RQ logs `job.description`
                # by default, which otherwise includes the full function call
                # with kwargs.
                description=f"process_pdf_job(job_id={job_id})",
            )

        logger.info("Job %s created and queued from %s upload", job_id, upload_kind)

        return JobCreateResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            expires_at=job.expires_at,
        )

    except AppException:
        if not job_created and job_dir is not None and job_dir.exists():
            shutil.rmtree(job_dir)
        raise
    except Exception as e:
        logger.exception(f"Failed to create job: {e}")
        if job_created:
            try:
                redis_service.delete_job(job_id)
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to rollback job metadata for %s: %s",
                    job_id,
                    cleanup_error,
                )
        if job_dir is not None and job_dir.exists():
            shutil.rmtree(job_dir)
        raise AppException(
            code=ErrorCode.INTERNAL_ERROR,
            message="Failed to create job",
            details={"error": str(e)},
            status_code=500,
        )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get current status of a job.

    Returns job metadata including status, stage, and progress.
    """
    redis_service = get_redis_service()
    job = redis_service.get_job(job_id)

    if not job:
        raise AppException(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job {job_id} not found",
            status_code=404,
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        created_at=job.created_at,
        expires_at=job.expires_at,
        message=job.message,
        error=job.error,
        debug_events=job.debug_events,
    )


async def job_event_generator(job_id: str) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for job progress.

    Polls Redis for job updates and yields SSE-formatted events.
    """
    redis_service = get_redis_service()

    # Check if job exists
    job = redis_service.get_job(job_id)
    if not job:
        event = JobEvent(
            job_id=job_id,
            status=JobStatus.failed,
            stage=JobStage.upload_received,
            progress=0,
            error={"code": ErrorCode.JOB_NOT_FOUND.value, "message": "Job not found"},
        )
        yield f"data: {event.model_dump_json()}\n\n"
        return

    last_status = None
    last_stage = None
    last_progress = None
    last_message = None

    while True:
        job = redis_service.get_job(job_id)

        if not job:
            # Job expired or deleted
            break

        # Send update if anything changed
        if (
            job.status != last_status
            or job.stage != last_stage
            or job.progress != last_progress
            or job.message != last_message
        ):
            event = JobEvent(
                job_id=job.job_id,
                status=job.status,
                stage=job.stage,
                progress=job.progress,
                message=job.message,
                error=job.error,
            )
            yield f"data: {event.model_dump_json()}\n\n"

            last_status = job.status
            last_stage = job.stage
            last_progress = job.progress
            last_message = job.message

        # Stop streaming if job is in terminal state
        if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
            break

        # Poll every 500ms
        await asyncio.sleep(0.5)


@router.get("/{job_id}/events")
async def stream_job_events(job_id: str):
    """
    Stream job progress events via Server-Sent Events (SSE).

    Clients can connect to this endpoint to receive real-time updates
    about job status, stage, and progress.
    """
    return StreamingResponse(
        job_event_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running job.

    Mirrors cancellation into Redis and RQ so queued/running jobs can release
    worker capacity promptly.
    """
    redis_service = get_redis_service()

    job = redis_service.get_job(job_id)
    if not job:
        raise AppException(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job {job_id} not found",
            status_code=404,
        )

    # Can only cancel pending or processing jobs
    if job.status not in [JobStatus.pending, JobStatus.processing]:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Cannot cancel job in {job.status} state",
            details={"status": job.status},
        )

    _sync_rq_cancel_state(job_id=job_id, status=job.status)

    # Set cancellation flag
    redis_service.set_cancel_flag(job_id)

    # Update job status
    redis_service.update_job(
        job_id,
        status=JobStatus.cancelled,
        stage=job.stage,
        progress=100,
        message="Job cancellation requested",
    )

    logger.info(f"Job {job_id} cancellation requested")

    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Cancellation requested",
    }


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete terminal job metadata and on-disk artifacts."""
    redis_service = get_redis_service()
    job = redis_service.get_job(job_id)
    job_dir = get_job_dir(job_id)

    if not job and not job_dir.exists():
        raise AppException(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job {job_id} not found",
            status_code=404,
        )

    if job and job.status in [JobStatus.pending, JobStatus.processing]:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Cannot delete an active job; cancel it first",
            details={"status": job.status},
            status_code=400,
        )

    artifacts_deleted = False
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir)
            artifacts_deleted = True
        except Exception as e:
            raise AppException(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to delete job artifacts",
                details={"job_id": job_id, "error": str(e)},
                status_code=500,
            ) from e

    redis_service.delete_job(job_id)
    logger.info(
        "Deleted job %s (had_metadata=%s, artifacts_deleted=%s)",
        job_id,
        bool(job),
        artifacts_deleted,
    )

    return {
        "job_id": job_id,
        "status": "deleted",
        "artifacts_deleted": artifacts_deleted,
    }


@router.get("/{job_id}/download")
async def download_result(job_id: str):
    """
    Download the converted PowerPoint file.

    Only available for completed jobs.
    """
    redis_service = get_redis_service()
    job = redis_service.get_job(job_id)
    if not job:
        raise AppException(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job {job_id} not found",
            status_code=404,
        )

    if job.status != JobStatus.completed:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Job is not completed (status: {job.status})",
            details={"status": job.status},
        )

    output_path = get_job_dir(job_id) / "output.pptx"
    if output_path.exists():
        return FileResponse(
            path=output_path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=f"converted_{job_id}.pptx",
        )

    # Job metadata exists and indicates completion, but output is missing.
    raise AppException(
        code=ErrorCode.INTERNAL_ERROR,
        message="Output file not found",
        status_code=500,
    )


@router.get("/{job_id}/artifacts", response_model=JobArtifactsResponse)
async def get_job_artifacts(job_id: str):
    """Return artifact image manifest for tracking/debug UI."""
    redis_service = get_redis_service()
    job = redis_service.get_job(job_id)
    job_dir = get_job_dir(job_id)
    if not job and not job_dir.exists():
        raise AppException(
            code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job {job_id} not found",
            status_code=404,
        )

    prefix = f"/api/v1/jobs/{job_id}/artifacts"
    artifacts_root = job_dir / "artifacts"
    source_pdf_rel = "input.pdf"
    source_pdf_path = job_dir / source_pdf_rel
    source_pdf_url = (
        f"{prefix}/file?path={quote(source_pdf_rel)}"
        if source_pdf_path.exists()
        else None
    )

    original_images = _collect_page_images(
        job_dir=job_dir,
        subdir="artifacts/page_renders",
        regex=r"^page-(\d{4})\.png$",
        url_prefix=prefix,
    )
    cleaned_images = _collect_page_images(
        job_dir=job_dir,
        subdir="artifacts/page_renders",
        regex=r"^page-(\d{4})\.(?:mineru\.)?clean\.png$",
        url_prefix=prefix,
    )
    final_preview_images = _collect_page_images(
        job_dir=job_dir,
        subdir="artifacts/final_preview",
        regex=r"^page-(\d{4})\.final\.png$",
        url_prefix=prefix,
    )
    ocr_overlay_images = _collect_page_images(
        job_dir=job_dir,
        subdir="artifacts/ocr",
        regex=r"^page-(\d{4})\.overlay\.png$",
        url_prefix=prefix,
    )
    layout_before_images = _collect_page_images(
        job_dir=job_dir,
        subdir="artifacts/layout_assist",
        regex=r"^page-(\d{4})\.before\.png$",
        url_prefix=prefix,
    )
    layout_after_images = _collect_page_images(
        job_dir=job_dir,
        subdir="artifacts/layout_assist",
        regex=r"^page-(\d{4})\.after\.png$",
        url_prefix=prefix,
    )

    all_pages = sorted(
        {
            *[img.page_index for img in original_images],
            *[img.page_index for img in cleaned_images],
            *[img.page_index for img in final_preview_images],
            *[img.page_index for img in ocr_overlay_images],
            *[img.page_index for img in layout_before_images],
            *[img.page_index for img in layout_after_images],
        }
    )

    return JobArtifactsResponse(
        job_id=job_id,
        status=job.status if job else None,
        artifacts_retained=artifacts_root.exists(),
        source_pdf_url=source_pdf_url,
        original_images=original_images,
        cleaned_images=cleaned_images,
        final_preview_images=final_preview_images,
        ocr_overlay_images=ocr_overlay_images,
        layout_before_images=layout_before_images,
        layout_after_images=layout_after_images,
        available_pages=all_pages,
    )


@router.get("/{job_id}/artifacts/file")
async def get_job_artifact_file(
    job_id: str, path: str = Query(..., description="Artifact path relative to job dir")
):
    """Read a single artifact file by relative path."""
    target = _safe_artifact_path(job_id, path)
    return FileResponse(path=target)
