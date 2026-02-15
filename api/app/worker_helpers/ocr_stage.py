from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pymupdf

from ..convert.ocr import ocr_image_to_elements
from ..logging_config import get_logger
from ..models.error import AppException, ErrorCode
from ..models.job import JobStage
from ..utils.concurrency import run_in_daemon_thread_with_timeout
from .debug import _build_ocr_effective_runtime_debug


logger = get_logger(__name__)


def _progress_in_span(
    done: int,
    total: int,
    *,
    start: int,
    end: int,
) -> int:
    if total <= 0:
        return int(end)
    ratio = max(0.0, min(1.0, float(done) / float(total)))
    return int(round(float(start) + (float(end - start) * ratio)))


def run_ocr_stage(
    *,
    ir: dict[str, Any],
    input_pdf: Path,
    job_path: Path,
    artifacts_dir: Path,
    settings: Any,
    ocr_manager: Any,
    text_refiner: Any | None,
    linebreak_refiner: Any | None,
    linebreak_assist_effective: bool | None,
    strict_no_fallback: bool,
    effective_ocr_provider: str,
    ocr_render_dpi: int,
    ocr_debug: dict[str, Any],
    set_processing_progress: Callable[[JobStage, int, str], None],
    abort_if_cancelled: Callable[..., None],
) -> None:
    ocr_dir = artifacts_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    ocr_debug.setdefault("pages", [])
    ocr_debug["runtime"] = _build_ocr_effective_runtime_debug(
        ocr_manager=ocr_manager,
        fallback_provider=ocr_debug.get("provider_effective"),
    )
    try:
        import shutil

        ocr_debug["which_tesseract"] = shutil.which("tesseract")
    except Exception as e:
        ocr_debug["which_tesseract"] = f"error: {e!s}"
    try:
        import pytesseract

        ocr_debug["pytesseract_cmd"] = getattr(
            getattr(pytesseract, "pytesseract", None),
            "tesseract_cmd",
            None,
        )
    except Exception as e:
        ocr_debug["pytesseract_cmd"] = f"error: {e!s}"

    doc = pymupdf.open(str(input_pdf))
    ocr_page_targets = sum(
        1
        for page in (ir.get("pages") or [])
        if isinstance(page, dict) and not page.get("has_text_layer")
    )
    ocr_page_processed = 0
    ocr_page_timeout = int(getattr(settings, "ocr_page_timeout_s", 300) or 300)
    ocr_total_timeout = int(getattr(settings, "ocr_total_timeout_s", 3600) or 3600)
    ocr_timeout_break_after = int(
        getattr(settings, "ocr_max_consecutive_timeouts", 2) or 2
    )
    ocr_timeout_break_after = max(1, ocr_timeout_break_after)
    ocr_consecutive_timeouts = 0
    ocr_stage_deadline = time.monotonic() + ocr_total_timeout
    try:
        for page in ir.get("pages") or []:
            abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")
            # --- overall OCR stage timeout ---
            if time.monotonic() >= ocr_stage_deadline:
                logger.warning(
                    "OCR stage timeout (%ss) exceeded – skipping remaining pages",
                    ocr_total_timeout,
                )
                break
            if not isinstance(page, dict):
                continue
            if page.get("has_text_layer"):
                ocr_debug["pages"].append(
                    {
                        "page_index": page.get("page_index"),
                        "skipped": "has_text_layer",
                    }
                )
                continue

            ocr_page_processed += 1
            set_processing_progress(
                JobStage.ocr,
                _progress_in_span(
                    ocr_page_processed - 1,
                    max(1, ocr_page_targets),
                    start=36,
                    end=68,
                ),
                f"OCR 识别中（第 {ocr_page_processed}/{max(1, ocr_page_targets)} 页）",
            )
            abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")

            page_index = int(page.get("page_index") or 0)
            page_w_pt = float(page.get("page_width_pt") or 0)
            page_h_pt = float(page.get("page_height_pt") or 0)
            if page_w_pt <= 0 or page_h_pt <= 0:
                ocr_debug["pages"].append(
                    {
                        "page_index": page_index,
                        "skipped": "invalid_dimensions",
                        "page_width_pt": page_w_pt,
                        "page_height_pt": page_h_pt,
                    }
                )
                continue

            try:
                pdf_page = doc.load_page(page_index)
                pix = pdf_page.get_pixmap(dpi=int(ocr_render_dpi), alpha=False)
            except Exception as e:
                logger.warning("Failed to render OCR page %s: %s", page_index, e)
                ocr_debug["pages"].append(
                    {
                        "page_index": page_index,
                        "error": f"render_failed: {e!s}",
                    }
                )
                continue

            image_path = ocr_dir / f"page-{page_index:04d}.png"
            try:
                pix.save(str(image_path))
            except Exception as e:
                logger.warning("Failed to save OCR image %s: %s", image_path, e)
                ocr_debug["pages"].append(
                    {
                        "page_index": page_index,
                        "error": f"image_save_failed: {e!s}",
                    }
                )
                continue

            fallback_reason: str | None = None

            try:
                abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")
                ocr_elements = run_in_daemon_thread_with_timeout(
                    lambda: ocr_image_to_elements(
                        str(image_path),
                        page_width_pt=page_w_pt,
                        page_height_pt=page_h_pt,
                        ocr_manager=ocr_manager,
                        text_refiner=text_refiner,
                        linebreak_refiner=linebreak_refiner,
                        linebreak_assist=linebreak_assist_effective,
                        strict_no_fallback=bool(strict_no_fallback),
                    ),
                    timeout_s=float(ocr_page_timeout),
                    label=f"worker:ocr_page:{page_index}",
                )
            except Exception as e:
                cause = getattr(e, "__cause__", None)
                details = f"{e!s}"
                if cause is not None:
                    details = f"{details}; cause={cause!s}"
                provider_choice = effective_ocr_provider
                logger.warning(
                    "OCR failed for page %s (provider=%s): %s",
                    page_index,
                    provider_choice,
                    details,
                )

                details_lower = details.lower()
                strict_now = bool(strict_no_fallback)
                is_timeout_error = isinstance(e, TimeoutError) or (
                    "timeout" in details_lower or "timed out" in details_lower
                )
                if is_timeout_error and not strict_now:
                    ocr_consecutive_timeouts += 1
                    page.setdefault("warnings", []).append(
                        "ocr_timeout_best_effort: "
                        f"provider={provider_choice}, page={page_index + 1}, "
                        f"consecutive={ocr_consecutive_timeouts}"
                    )
                    ocr_debug["pages"].append(
                        {
                            "page_index": page_index,
                            "warning": "ocr_timeout",
                            "provider": provider_choice,
                            "consecutive_timeouts": ocr_consecutive_timeouts,
                            "error": details,
                        }
                    )
                    if ocr_consecutive_timeouts >= ocr_timeout_break_after:
                        timeout_warning = (
                            "ocr_timeout_circuit_open: "
                            f"consecutive={ocr_consecutive_timeouts}, "
                            f"page_timeout_s={ocr_page_timeout}"
                        )
                        ir.setdefault("warnings", []).append(timeout_warning)
                        logger.warning(
                            "OCR timeout circuit open after %s consecutive timeout(s); "
                            "skipping remaining OCR pages",
                            ocr_consecutive_timeouts,
                        )
                        break
                    continue
                ocr_consecutive_timeouts = 0

                nonfatal_empty_ocr = any(
                    marker in details_lower
                    for marker in (
                        "ai ocr returned no items",
                        "ai ocr returned empty elements",
                        "ai ocr returned no parseable items",
                    )
                )

                # Strict policy: fail fast on OCR errors.
                # Non-strict mode is best-effort: keep the background
                # image-only page and continue conversion.
                if nonfatal_empty_ocr:
                    if strict_now:
                        raise AppException(
                            code=ErrorCode.OCR_FAILED,
                            message=(
                                f"{provider_choice.upper()} returned empty OCR result on page {page_index + 1}"
                            ),
                            details={
                                "page_index": page_index,
                                "provider": provider_choice,
                                "reason": details,
                            },
                        )
                    logger.warning(
                        "OCR returned empty result on page %s (provider=%s); keep background-only page",
                        page_index,
                        provider_choice,
                    )
                    page.setdefault("warnings", []).append(
                        f"ocr_empty_result: provider={provider_choice}, page={page_index + 1}"
                    )
                    ocr_debug["pages"].append(
                        {
                            "page_index": page_index,
                            "warning": "ocr_empty_result",
                            "provider": provider_choice,
                            "error": details,
                        }
                    )
                    continue

                if strict_now:
                    provider_label = provider_choice.upper()
                    raise AppException(
                        code=ErrorCode.OCR_FAILED,
                        message=f"{provider_label} failed on page {page_index + 1}: {details}",
                        details={
                            "page_index": page_index,
                            "provider": provider_choice,
                            "reason": details,
                        },
                    )

                ocr_debug["pages"].append(
                    {
                        "page_index": page_index,
                        "error": f"ocr_failed: {details}",
                    }
                )
                page.setdefault("warnings", []).append(
                    f"ocr_failed_best_effort: provider={provider_choice}, page={page_index + 1}"
                )
                continue

            ocr_consecutive_timeouts = 0
            used_provider = getattr(ocr_manager, "last_provider_name", None)
            fallback_reason = getattr(
                ocr_manager,
                "last_fallback_reason",
                fallback_reason,
            )

            # Strict policy: do not switch to local Tesseract geometry
            # unless the user explicitly selected `tesseract/local`.
            # Keep the original provider result as-is.

            # Debug/self-check: write an overlay image with OCR bboxes drawn
            # on top of the rendered page. This makes coordinate issues
            # immediately visible without opening PowerPoint.
            overlay_path: Path | None = None
            bbox_stats: dict[str, Any] = {}
            try:
                from PIL import Image, ImageDraw

                img = Image.open(image_path).convert("RGB")
                gray = img.convert("L")
                W, H = img.size
                draw = ImageDraw.Draw(img)

                stds: list[float] = []
                out_of_bounds = 0
                low_variance = 0
                low_std_threshold = 5.0

                sx = float(W) / float(page_w_pt) if page_w_pt else 1.0
                sy = float(H) / float(page_h_pt) if page_h_pt else 1.0

                for el in ocr_elements or []:
                    bbox_pt = el.get("bbox_pt")
                    if not isinstance(bbox_pt, list) or len(bbox_pt) != 4:
                        continue
                    try:
                        x0, y0, x1, y1 = (
                            float(bbox_pt[0]),
                            float(bbox_pt[1]),
                            float(bbox_pt[2]),
                            float(bbox_pt[3]),
                        )
                    except Exception:
                        continue

                    x0p = int(round(x0 * sx))
                    y0p = int(round(y0 * sy))
                    x1p = int(round(x1 * sx))
                    y1p = int(round(y1 * sy))

                    if x0p < 0 or y0p < 0 or x1p > W or y1p > H:
                        out_of_bounds += 1

                    # Clamp for drawing/stat sampling.
                    x0c = max(0, min(W - 1, x0p))
                    y0c = max(0, min(H - 1, y0p))
                    x1c = max(0, min(W, x1p))
                    y1c = max(0, min(H, y1p))
                    if x1c <= x0c or y1c <= y0c:
                        continue

                    draw.rectangle([x0c, y0c, x1c, y1c], outline=(255, 0, 0), width=2)

                    crop = gray.crop((x0c, y0c, x1c, y1c))
                    target_w = max(8, min(64, crop.width // 8))
                    target_h = max(8, min(64, crop.height // 8))
                    small = crop.resize((target_w, target_h))
                    pixels = list(small.getdata())
                    if not pixels:
                        continue
                    mean = sum(pixels) / len(pixels)
                    var = sum((p - mean) ** 2 for p in pixels) / len(pixels)
                    std = float(var**0.5)
                    stds.append(std)
                    if std <= low_std_threshold:
                        low_variance += 1

                overlay_path = ocr_dir / f"page-{page_index:04d}.overlay.png"
                img.save(overlay_path)

                bbox_stats = {
                    "out_of_bounds": out_of_bounds,
                    "low_variance": low_variance,
                    "low_std_threshold": low_std_threshold,
                    "median_std": (sorted(stds)[len(stds) // 2] if stds else None),
                }
            except Exception as e:
                bbox_stats = {"overlay_error": str(e)}
            if ocr_elements:
                page.setdefault("elements", []).extend(ocr_elements)
                page["ocr_used"] = True
                # Keep has_text_layer=False for scanned PDFs so the PPTX
                # generator can use the scanned-page strategy (background
                # render + masking + editable overlay text).
                ocr_debug["pages"].append(
                    {
                        "page_index": page_index,
                        "elements": len(ocr_elements),
                        "used_provider": used_provider,
                        "fallback_reason": fallback_reason,
                        "overlay_image": str(overlay_path) if overlay_path else None,
                        "bbox_stats": bbox_stats,
                    }
                )
            else:
                ocr_debug["pages"].append(
                    {
                        "page_index": page_index,
                        "elements": 0,
                        "used_provider": used_provider,
                        "fallback_reason": fallback_reason,
                        "overlay_image": str(overlay_path) if overlay_path else None,
                        "bbox_stats": bbox_stats,
                    }
                )
        set_processing_progress(
            JobStage.ocr,
            68,
            f"OCR 阶段完成（已处理 {ocr_page_processed}/{max(1, ocr_page_targets)} 页）",
        )
        abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")
    finally:
        doc.close()
        ocr_debug["runtime"] = _build_ocr_effective_runtime_debug(
            ocr_manager=ocr_manager,
            fallback_provider=ocr_debug.get("provider_effective"),
        )
        (ocr_dir / "ocr_debug.json").write_text(
            json.dumps(ocr_debug, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        # Persist IR after OCR for debugging.
        (job_path / "ir.ocr.json").write_text(
            json.dumps(ir, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
