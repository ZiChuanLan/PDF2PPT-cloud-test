from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def _summarize_ocr_page_runtime(
    *, page_entries: list[dict[str, Any]], ocr_manager: Any
) -> dict[str, Any]:
    provider_counts: dict[str, int] = {}
    fallback_reason_counts: dict[str, int] = {}
    pages_with_elements = 0
    pages_with_errors = 0
    pages_with_fallback = 0

    for entry in page_entries:
        if not isinstance(entry, dict):
            continue
        if "error" in entry or "warning" in entry:
            pages_with_errors += 1
        if int(entry.get("elements") or 0) > 0:
            pages_with_elements += 1

        used_provider = str(entry.get("used_provider") or "").strip()
        if used_provider:
            provider_counts[used_provider] = provider_counts.get(used_provider, 0) + 1

        fallback_reason = str(entry.get("fallback_reason") or "").strip()
        if fallback_reason:
            pages_with_fallback += 1
            fallback_reason_counts[fallback_reason] = (
                fallback_reason_counts.get(fallback_reason, 0) + 1
            )

    ai_provider_disabled = bool(getattr(ocr_manager, "ai_provider_disabled", False))
    ai_provider_disabled_reason = getattr(
        ocr_manager, "ai_provider_disabled_reason", None
    )

    return {
        "provider_counts": provider_counts,
        "distinct_provider_count": len(provider_counts),
        "pages_with_elements": pages_with_elements,
        "pages_with_errors": pages_with_errors,
        "fallback_pages": pages_with_fallback,
        "fallback_reason_counts": fallback_reason_counts,
        "ai_provider_disabled": ai_provider_disabled,
        "ai_provider_disabled_reason": ai_provider_disabled_reason,
    }


def _format_ocr_progress_message(
    *,
    ocr_page_processed: int,
    ocr_page_targets: int,
    pdf_page_index: int,
    source_page_count: int,
    overall_progress: int,
) -> str:
    ocr_total = max(1, int(ocr_page_targets))
    ocr_stage_percent = int(
        round((max(0, int(ocr_page_processed) - 1) / float(ocr_total)) * 100.0)
    )
    pdf_page_number = max(1, int(pdf_page_index) + 1)
    pdf_total = max(pdf_page_number, int(source_page_count or 0))
    return (
        "OCR 识别中（"
        f"OCR页 {int(ocr_page_processed)}/{ocr_total}，"
        f"PDF页 {pdf_page_number}/{pdf_total}，"
        f"OCR阶段 {ocr_stage_percent}%，"
        f"总进度 {int(overall_progress)}%"
        "）"
    )


def _format_parallel_ocr_progress_message(
    *,
    completed_pages: int,
    total_pages: int,
    running_pages: int,
    page_concurrency: int,
    latest_pdf_page_index: int | None,
    source_page_count: int,
    overall_progress: int,
) -> str:
    pdf_page_number = (
        max(1, int(latest_pdf_page_index) + 1)
        if latest_pdf_page_index is not None
        else None
    )
    pdf_total = max(
        int(source_page_count or 0),
        pdf_page_number or 1,
    )
    latest_page_text = (
        f"，最近 PDF页 {pdf_page_number}/{pdf_total}"
        if pdf_page_number is not None
        else ""
    )
    return (
        "OCR 识别中（"
        f"已完成 {int(completed_pages)}/{max(1, int(total_pages))} 页，"
        f"运行中 {max(0, int(running_pages))} 页，"
        f"页并发 {max(1, int(page_concurrency))}"
        f"{latest_page_text}，"
        f"总进度 {int(overall_progress)}%"
        "）"
    )


def _process_parallel_ai_ocr_page(
    *,
    page: dict[str, Any],
    input_pdf: Path,
    ocr_dir: Path,
    ocr_runtime_factory: Callable[[], Any],
    linebreak_assist_effective: bool | None,
    ocr_render_dpi: int,
    ocr_page_timeout: int,
    ocr_image_region_timeout: int,
    export_overlay_images: bool,
    abort_if_cancelled: Callable[..., None],
) -> dict[str, Any]:
    abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")
    page_index = int(page.get("page_index") or 0)
    page_w_pt = float(page.get("page_width_pt") or 0)
    page_h_pt = float(page.get("page_height_pt") or 0)
    if page_w_pt <= 0 or page_h_pt <= 0:
        return {
            "page_index": page_index,
            "page_warnings": [],
            "ir_warnings": [],
            "elements": [],
            "image_regions": [],
            "debug_entry": {
                "page_index": page_index,
                "skipped": "invalid_dimensions",
                "page_width_pt": page_w_pt,
                "page_height_pt": page_h_pt,
            },
        }

    runtime = ocr_runtime_factory()
    ocr_manager = getattr(runtime, "ocr_manager", None)
    if ocr_manager is None:
        raise RuntimeError("parallel OCR runtime has no ocr_manager")
    text_refiner = getattr(runtime, "text_refiner", None)
    linebreak_refiner = getattr(runtime, "linebreak_refiner", None)
    strict_no_fallback = bool(getattr(runtime, "strict_ocr_mode", True))
    effective_ocr_provider = str(
        getattr(runtime, "effective_ocr_provider", None) or "aiocr"
    )
    route_kind = getattr(ocr_manager, "route_kind", None)

    try:
        pdf_doc = pymupdf.open(str(input_pdf))
        try:
            abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")
            pdf_page = pdf_doc.load_page(page_index)
            pix = pdf_page.get_pixmap(dpi=int(ocr_render_dpi), alpha=False)
        finally:
            pdf_doc.close()
    except Exception as e:
        logger.warning("Failed to render parallel OCR page %s: %s", page_index, e)
        return {
            "page_index": page_index,
            "page_warnings": [],
            "ir_warnings": [],
            "elements": [],
            "image_regions": [],
            "debug_entry": {
                "page_index": page_index,
                "error": f"render_failed: {e!s}",
            },
        }

    image_path = ocr_dir / f"page-{page_index:04d}.png"
    try:
        pix.save(str(image_path))
    except Exception as e:
        logger.warning("Failed to save parallel OCR image %s: %s", image_path, e)
        return {
            "page_index": page_index,
            "page_warnings": [],
            "ir_warnings": [],
            "elements": [],
            "image_regions": [],
            "debug_entry": {
                "page_index": page_index,
                "error": f"image_save_failed: {e!s}",
            },
        }

    ocr_call_started = time.perf_counter()
    logger.info(
        "Starting parallel OCR page (pdf_page=%s, provider=%s, route=%s, timeout_s=%s, image=%s)",
        page_index + 1,
        effective_ocr_provider,
        route_kind,
        ocr_page_timeout,
        image_path.name,
    )

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
        logger.warning(
            "Parallel OCR failed for page %s (provider=%s): %s",
            page_index,
            effective_ocr_provider,
            details,
        )
        details_lower = details.lower()
        is_timeout_error = isinstance(e, TimeoutError) or (
            "timeout" in details_lower or "timed out" in details_lower
        )
        if is_timeout_error and not strict_no_fallback:
            return {
                "page_index": page_index,
                "page_warnings": [
                    f"ocr_timeout_best_effort: provider={effective_ocr_provider}, page={page_index + 1}, parallel=1"
                ],
                "ir_warnings": [],
                "elements": [],
                "image_regions": [],
                "debug_entry": {
                    "page_index": page_index,
                    "warning": "ocr_timeout",
                    "provider": effective_ocr_provider,
                    "error": details,
                },
            }

        nonfatal_empty_ocr = any(
            marker in details_lower
            for marker in (
                "ai ocr returned no items",
                "ai ocr returned empty elements",
                "ai ocr returned no parseable items",
            )
        )
        if nonfatal_empty_ocr:
            if strict_no_fallback:
                raise AppException(
                    code=ErrorCode.OCR_FAILED,
                    message=(
                        f"{effective_ocr_provider.upper()} returned empty OCR result on page {page_index + 1}"
                    ),
                    details={
                        "page_index": page_index,
                        "provider": effective_ocr_provider,
                        "reason": details,
                    },
                ) from e
            return {
                "page_index": page_index,
                "page_warnings": [
                    f"ocr_empty_result: provider={effective_ocr_provider}, page={page_index + 1}"
                ],
                "ir_warnings": [],
                "elements": [],
                "image_regions": [],
                "debug_entry": {
                    "page_index": page_index,
                    "warning": "ocr_empty_result",
                    "provider": effective_ocr_provider,
                    "error": details,
                },
            }

        if strict_no_fallback:
            raise AppException(
                code=ErrorCode.OCR_FAILED,
                message=(
                    f"{effective_ocr_provider.upper()} failed on page {page_index + 1}: {details}"
                ),
                details={
                    "page_index": page_index,
                    "provider": effective_ocr_provider,
                    "reason": details,
                },
            ) from e

        return {
            "page_index": page_index,
            "page_warnings": [
                f"ocr_failed_best_effort: provider={effective_ocr_provider}, page={page_index + 1}"
            ],
            "ir_warnings": [],
            "elements": [],
            "image_regions": [],
            "debug_entry": {
                "page_index": page_index,
                "error": f"ocr_failed: {details}",
            },
        }

    elapsed_ms = int(round(max(0.0, time.perf_counter() - ocr_call_started) * 1000.0))
    logger.info(
        "Finished parallel OCR page (pdf_page=%s, provider=%s, route=%s, elapsed_ms=%s, elements=%s)",
        page_index + 1,
        effective_ocr_provider,
        route_kind,
        elapsed_ms,
        len(ocr_elements or []),
    )

    used_provider = getattr(ocr_manager, "last_provider_name", None)
    fallback_reason = getattr(ocr_manager, "last_fallback_reason", None)
    quality_notes_raw = getattr(ocr_manager, "last_quality_notes", [])
    quality_notes = [
        str(note).strip()
        for note in (
            quality_notes_raw if isinstance(quality_notes_raw, list) else []
        )
        if str(note).strip()
    ]
    page_warnings = list(quality_notes)
    ir_warnings = [f"{note}:page={page_index + 1}" for note in quality_notes]

    detected_image_regions_pt: list[list[float]] = []
    image_region_error: str | None = None
    try:
        from PIL import Image

        with Image.open(image_path) as img_probe:
            image_width_px, image_height_px = img_probe.size

        detected_image_regions_px = run_in_daemon_thread_with_timeout(
            lambda: ocr_manager.detect_image_regions(str(image_path)),
            timeout_s=float(max(1, ocr_image_region_timeout)),
            label=f"worker:ocr_image_regions:{page_index}",
        )
        for bbox in detected_image_regions_px or []:
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                bbox_pt = ocr_manager.convert_bbox_to_pdf_coords(
                    bbox=bbox,
                    image_width=int(image_width_px),
                    image_height=int(image_height_px),
                    page_width_pt=page_w_pt,
                    page_height_pt=page_h_pt,
                )
            except Exception:
                continue
            detected_image_regions_pt.append(list(bbox_pt))
    except TimeoutError:
        image_region_error = (
            "image_region_detection_timeout:"
            f"{int(max(1, ocr_image_region_timeout))}s"
        )
    except Exception as e:
        image_region_error = str(e)

    overlay_path, bbox_stats = _maybe_export_ocr_overlay_image(
        enabled=export_overlay_images,
        image_path=image_path,
        ocr_dir=ocr_dir,
        page_index=page_index,
        page_w_pt=page_w_pt,
        page_h_pt=page_h_pt,
        ocr_elements=ocr_elements if isinstance(ocr_elements, list) else None,
    )

    return {
        "page_index": page_index,
        "page_warnings": page_warnings,
        "ir_warnings": ir_warnings,
        "elements": ocr_elements or [],
        "image_regions": detected_image_regions_pt,
        "debug_entry": {
            "page_index": page_index,
            "elements": len(ocr_elements or []),
            "image_regions": len(detected_image_regions_pt),
            "image_region_detection_error": image_region_error,
            "used_provider": used_provider,
            "fallback_reason": fallback_reason,
            "quality_notes": quality_notes,
            "overlay_image": str(overlay_path) if overlay_path else None,
            "bbox_stats": bbox_stats,
        },
    }


def _maybe_export_ocr_overlay_image(
    *,
    enabled: bool,
    image_path: Path,
    ocr_dir: Path,
    page_index: int,
    page_w_pt: float,
    page_h_pt: float,
    ocr_elements: list[dict[str, Any]] | None,
) -> tuple[Path | None, dict[str, Any]]:
    if not enabled:
        return None, {}

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

        return overlay_path, {
            "out_of_bounds": out_of_bounds,
            "low_variance": low_variance,
            "low_std_threshold": low_std_threshold,
            "median_std": (sorted(stds)[len(stds) // 2] if stds else None),
        }
    except Exception as e:
        return None, {"overlay_error": str(e)}


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
    export_overlay_images: bool,
    set_processing_progress: Callable[[JobStage, int, str], None],
    abort_if_cancelled: Callable[..., None],
    ocr_setup: Any | None = None,
    ocr_runtime_factory: Callable[[], Any] | None = None,
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
    ocr_image_region_timeout = int(
        getattr(settings, "ocr_image_region_timeout_s", 12) or 12
    )
    ocr_timeout_break_after = int(
        getattr(settings, "ocr_max_consecutive_timeouts", 2) or 2
    )
    ocr_timeout_break_after = max(1, ocr_timeout_break_after)
    ocr_consecutive_timeouts = 0
    ocr_stage_deadline = time.monotonic() + ocr_total_timeout
    source_page_count = int(
        ir.get("source_page_count")
        or ir.get("page_count")
        or len(ir.get("pages") or [])
        or 0
    )
    route_kind = str(getattr(ocr_manager, "route_kind", "") or "")
    page_concurrency = max(
        1,
        int(
            getattr(ocr_setup, "effective_ocr_ai_page_concurrency", 1)
            if ocr_setup is not None
            else 1
        ),
    )
    use_parallel_ai_ocr = (
        page_concurrency > 1
        and str(effective_ocr_provider or "").strip().lower() == "aiocr"
        and route_kind in {"remote_prompt_ocr", "local_layout_block_ocr"}
        and callable(ocr_runtime_factory)
    )
    try:
        if use_parallel_ai_ocr:
            ocr_pages = [
                page
                for page in (ir.get("pages") or [])
                if isinstance(page, dict) and not page.get("has_text_layer")
            ]
            completed_pages = 0
            latest_pdf_page_index: int | None = None
            running_initial = min(page_concurrency, len(ocr_pages))
            set_processing_progress(
                JobStage.ocr,
                36,
                _format_parallel_ocr_progress_message(
                    completed_pages=0,
                    total_pages=len(ocr_pages),
                    running_pages=running_initial,
                    page_concurrency=page_concurrency,
                    latest_pdf_page_index=None,
                    source_page_count=source_page_count,
                    overall_progress=36,
                ),
            )
            abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")

            page_iter = iter(ocr_pages)
            future_map: dict[Any, dict[str, Any]] = {}
            stop_submitting_new_pages = False
            with ThreadPoolExecutor(max_workers=page_concurrency) as executor:
                for _ in range(running_initial):
                    page = next(page_iter, None)
                    if page is None:
                        break
                    future = executor.submit(
                        _process_parallel_ai_ocr_page,
                        page=page,
                        input_pdf=input_pdf,
                        ocr_dir=ocr_dir,
                        ocr_runtime_factory=ocr_runtime_factory,
                        linebreak_assist_effective=linebreak_assist_effective,
                        ocr_render_dpi=int(ocr_render_dpi),
                        ocr_page_timeout=int(ocr_page_timeout),
                        ocr_image_region_timeout=int(ocr_image_region_timeout),
                        export_overlay_images=bool(export_overlay_images),
                        abort_if_cancelled=abort_if_cancelled,
                    )
                    future_map[future] = page

                while future_map:
                    done_futures, _ = wait(
                        set(future_map),
                        timeout=1.0,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done_futures:
                        if (
                            not stop_submitting_new_pages
                            and time.monotonic() >= ocr_stage_deadline
                        ):
                            stop_submitting_new_pages = True
                            logger.warning(
                                "Parallel OCR stage timeout (%ss) exceeded – stop scheduling new pages",
                                ocr_total_timeout,
                            )
                            ir.setdefault("warnings", []).append(
                                "ocr_parallel_total_timeout:"
                                f" total_timeout_s={ocr_total_timeout}"
                            )
                        abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")
                        continue
                    for future in done_futures:
                        page = future_map.pop(future)
                        result = future.result()
                        latest_pdf_page_index = int(result.get("page_index") or 0)
                        for note in result.get("page_warnings") or []:
                            if str(note).strip():
                                page.setdefault("warnings", []).append(str(note).strip())
                        for note in result.get("ir_warnings") or []:
                            if str(note).strip():
                                ir.setdefault("warnings", []).append(str(note).strip())
                        image_regions = result.get("image_regions") or []
                        if image_regions:
                            page["image_regions"] = image_regions
                        ocr_elements = result.get("elements") or []
                        if ocr_elements:
                            page.setdefault("elements", []).extend(ocr_elements)
                            page["ocr_used"] = True
                        debug_entry = result.get("debug_entry")
                        if isinstance(debug_entry, dict):
                            ocr_debug["pages"].append(debug_entry)

                        completed_pages += 1
                        if (
                            not stop_submitting_new_pages
                            and time.monotonic() >= ocr_stage_deadline
                        ):
                            stop_submitting_new_pages = True
                            logger.warning(
                                "Parallel OCR stage timeout (%ss) exceeded – stop scheduling new pages",
                                ocr_total_timeout,
                            )
                            ir.setdefault("warnings", []).append(
                                "ocr_parallel_total_timeout:"
                                f" total_timeout_s={ocr_total_timeout}"
                            )
                        next_page = None if stop_submitting_new_pages else next(page_iter, None)
                        if next_page is not None:
                            next_future = executor.submit(
                                _process_parallel_ai_ocr_page,
                                page=next_page,
                                input_pdf=input_pdf,
                                ocr_dir=ocr_dir,
                                ocr_runtime_factory=ocr_runtime_factory,
                                linebreak_assist_effective=linebreak_assist_effective,
                                ocr_render_dpi=int(ocr_render_dpi),
                                ocr_page_timeout=int(ocr_page_timeout),
                                ocr_image_region_timeout=int(ocr_image_region_timeout),
                                export_overlay_images=bool(export_overlay_images),
                                abort_if_cancelled=abort_if_cancelled,
                            )
                            future_map[next_future] = next_page

                        progress_value = _progress_in_span(
                            completed_pages,
                            max(1, len(ocr_pages)),
                            start=36,
                            end=68,
                        )
                        set_processing_progress(
                            JobStage.ocr,
                            progress_value,
                            _format_parallel_ocr_progress_message(
                                completed_pages=completed_pages,
                                total_pages=len(ocr_pages),
                                running_pages=len(future_map),
                                page_concurrency=page_concurrency,
                                latest_pdf_page_index=latest_pdf_page_index,
                                source_page_count=source_page_count,
                                overall_progress=progress_value,
                            ),
                        )
                        abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")

            set_processing_progress(
                JobStage.ocr,
                68,
                "OCR 阶段完成（并发模式）",
            )
            abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")
            return

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
            page_index = int(page.get("page_index") or 0)
            progress_value = _progress_in_span(
                ocr_page_processed - 1,
                max(1, ocr_page_targets),
                start=36,
                end=68,
            )
            set_processing_progress(
                JobStage.ocr,
                progress_value,
                _format_ocr_progress_message(
                    ocr_page_processed=ocr_page_processed,
                    ocr_page_targets=ocr_page_targets,
                    pdf_page_index=page_index,
                    source_page_count=source_page_count,
                    overall_progress=progress_value,
                ),
            )
            abort_if_cancelled(stage=JobStage.ocr, message="Job cancelled")

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
            ocr_call_started = time.perf_counter()
            route_kind = getattr(ocr_manager, "route_kind", None)
            logger.info(
                "Starting OCR page (ocr_page=%s/%s, pdf_page=%s/%s, provider=%s, route=%s, timeout_s=%s, image=%s)",
                ocr_page_processed,
                max(1, ocr_page_targets),
                page_index + 1,
                max(1, source_page_count),
                effective_ocr_provider,
                route_kind,
                ocr_page_timeout,
                image_path.name,
            )

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
            elapsed_ms = int(
                round(max(0.0, time.perf_counter() - ocr_call_started) * 1000.0)
            )
            logger.info(
                "Finished OCR page (ocr_page=%s/%s, pdf_page=%s/%s, provider=%s, route=%s, elapsed_ms=%s, elements=%s)",
                ocr_page_processed,
                max(1, ocr_page_targets),
                page_index + 1,
                max(1, source_page_count),
                effective_ocr_provider,
                route_kind,
                elapsed_ms,
                len(ocr_elements or []),
            )

            ocr_consecutive_timeouts = 0
            used_provider = getattr(ocr_manager, "last_provider_name", None)
            fallback_reason = getattr(
                ocr_manager,
                "last_fallback_reason",
                fallback_reason,
            )
            quality_notes_raw = getattr(ocr_manager, "last_quality_notes", [])
            quality_notes = [
                str(note).strip()
                for note in (
                    quality_notes_raw if isinstance(quality_notes_raw, list) else []
                )
                if str(note).strip()
            ]
            for note in quality_notes:
                page.setdefault("warnings", []).append(note)
                ir.setdefault("warnings", []).append(
                    f"{note}:page={page_index + 1}"
                )

            detected_image_regions_pt: list[list[float]] = []
            image_region_error: str | None = None
            try:
                from PIL import Image

                with Image.open(image_path) as img_probe:
                    image_width_px, image_height_px = img_probe.size

                detected_image_regions_px = run_in_daemon_thread_with_timeout(
                    lambda: ocr_manager.detect_image_regions(str(image_path)),
                    timeout_s=float(max(1, ocr_image_region_timeout)),
                    label=f"worker:ocr_image_regions:{page_index}",
                )
                for bbox in detected_image_regions_px or []:
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    try:
                        bbox_pt = ocr_manager.convert_bbox_to_pdf_coords(
                            bbox=bbox,
                            image_width=int(image_width_px),
                            image_height=int(image_height_px),
                            page_width_pt=page_w_pt,
                            page_height_pt=page_h_pt,
                        )
                    except Exception:
                        continue
                    detected_image_regions_pt.append(list(bbox_pt))
            except TimeoutError:
                image_region_error = (
                    "image_region_detection_timeout:"
                    f"{int(max(1, ocr_image_region_timeout))}s"
                )
            except Exception as e:
                image_region_error = str(e)

            if detected_image_regions_pt:
                page["image_regions"] = detected_image_regions_pt

            # Strict policy: do not switch to local Tesseract geometry
            # unless the user explicitly selected `tesseract/local`.
            # Keep the original provider result as-is.

            # Debug/self-check: write an overlay image with OCR bboxes drawn
            # on top of the rendered page. This makes coordinate issues
            # immediately visible without opening PowerPoint.
            overlay_path, bbox_stats = _maybe_export_ocr_overlay_image(
                enabled=export_overlay_images,
                image_path=image_path,
                ocr_dir=ocr_dir,
                page_index=page_index,
                page_w_pt=page_w_pt,
                page_h_pt=page_h_pt,
                ocr_elements=ocr_elements if isinstance(ocr_elements, list) else None,
            )
            if ocr_elements:
                page.setdefault("elements", []).extend(ocr_elements)
                page["ocr_used"] = True
                # Keep has_text_layer=False for scanned PDFs so the PPTX
                # generator can use the scanned-page strategy (background
                # render + masking + editable overlay text).
            ocr_debug["pages"].append(
                {
                    "page_index": page_index,
                    "elements": len(ocr_elements or []),
                    "image_regions": len(detected_image_regions_pt),
                    "image_region_detection_error": image_region_error,
                    "used_provider": used_provider,
                    "fallback_reason": fallback_reason,
                    "quality_notes": quality_notes,
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
        page_runtime_summary = _summarize_ocr_page_runtime(
            page_entries=ocr_debug.get("pages") if isinstance(ocr_debug.get("pages"), list) else [],
            ocr_manager=ocr_manager,
        )
        ocr_debug["page_runtime_summary"] = page_runtime_summary
        runtime_debug = _build_ocr_effective_runtime_debug(
            ocr_manager=ocr_manager,
            fallback_provider=ocr_debug.get("provider_effective"),
        )
        runtime_debug["page_summary"] = page_runtime_summary
        ocr_debug["runtime"] = runtime_debug

        if page_runtime_summary["distinct_provider_count"] > 1:
            providers = ",".join(sorted(page_runtime_summary["provider_counts"]))
            ir.setdefault("warnings", []).append(
                f"ocr_page_provider_switches: providers={providers}"
            )
        if page_runtime_summary["fallback_pages"] > 0:
            ir.setdefault("warnings", []).append(
                "ocr_page_fallbacks:"
                f" pages={page_runtime_summary['fallback_pages']}"
                f" reasons={json.dumps(page_runtime_summary['fallback_reason_counts'], ensure_ascii=True, sort_keys=True)}"
            )
        if page_runtime_summary["ai_provider_disabled"]:
            ir.setdefault("warnings", []).append(
                "ocr_ai_provider_disabled:"
                f" reason={page_runtime_summary['ai_provider_disabled_reason'] or 'unknown'}"
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
