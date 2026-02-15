from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pymupdf

from ..convert.ocr import create_ocr_manager, ocr_image_to_elements
from ..logging_config import get_logger
from .geometry_utils import (
    _bbox_center_distance_ratio,
    _bbox_overlap_ratio,
    _coerce_bbox_pt,
    _normalize_match_text,
)


logger = get_logger(__name__)


def _apply_mineru_hybrid_ocr_alignment(
    ir: dict[str, Any],
    *,
    source_pdf: Path,
    artifacts_dir: Path,
    ocr_render_dpi: int,
    ocr_provider: str | None,
    ocr_baidu_app_id: str | None,
    ocr_baidu_api_key: str | None,
    ocr_baidu_secret_key: str | None,
    ocr_tesseract_min_confidence: float | None,
    ocr_tesseract_language: str | None,
    ocr_ai_provider: str | None = None,
    ocr_strict_mode: bool = False,
) -> dict[str, Any]:
    pages = ir.get("pages")
    if not isinstance(pages, list) or not pages:
        return ir

    provider_id = (ocr_provider or "auto").strip().lower()
    if provider_id in {"paddle-local", "local_paddle"}:
        provider_id = "paddle_local"
    if provider_id in {"aiocr", "ai", "remote", "paddle"}:
        provider_id = "auto"
    if provider_id not in {
        "auto",
        "baidu",
        "tesseract",
        "local",
        "paddle",
        "paddle_local",
    }:
        provider_id = "auto"

    try:
        ocr_manager = create_ocr_manager(
            provider=provider_id,
            ai_provider=ocr_ai_provider,
            ai_api_key=None,
            ai_base_url=None,
            ai_model=None,
            baidu_app_id=ocr_baidu_app_id,
            baidu_api_key=ocr_baidu_api_key,
            baidu_secret_key=ocr_baidu_secret_key,
            tesseract_min_confidence=ocr_tesseract_min_confidence,
            tesseract_language=ocr_tesseract_language or "chi_sim+eng",
            strict_no_fallback=bool(ocr_strict_mode),
            allow_paddle_model_downgrade=not bool(ocr_strict_mode),
        )
    except Exception as e:
        ir.setdefault("warnings", []).append(f"mineru_hybrid_ocr_init_failed:{e!s}")
        logger.warning("MinerU hybrid OCR init failed: %s", e)
        return ir

    ocr_dir = artifacts_dir / "mineru_hybrid_ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    matched_total = 0
    mineru_text_total = 0
    pages_used = 0

    try:
        doc = pymupdf.open(str(source_pdf))
    except Exception as e:
        ir.setdefault("warnings", []).append(f"mineru_hybrid_ocr_open_pdf_failed:{e!s}")
        logger.warning("MinerU hybrid OCR failed to open source PDF: %s", e)
        return ir

    try:
        for page in pages:
            if not isinstance(page, dict):
                continue
            elements = page.get("elements")
            if not isinstance(elements, list) or not elements:
                continue

            page_index = int(page.get("page_index") or 0)
            page_w_pt = float(page.get("page_width_pt") or 0.0)
            page_h_pt = float(page.get("page_height_pt") or 0.0)
            if page_w_pt <= 0 or page_h_pt <= 0:
                continue

            mineru_text_with_bbox: list[
                tuple[int, tuple[float, float, float, float]]
            ] = []
            for idx, el in enumerate(elements):
                if not isinstance(el, dict):
                    continue
                if el.get("type") != "text":
                    continue
                if str(el.get("source") or "").strip().lower() != "mineru":
                    continue
                if not str(el.get("text") or "").strip():
                    continue
                bbox = _coerce_bbox_pt(el.get("bbox_pt"))
                if bbox is None:
                    continue
                mineru_text_with_bbox.append((idx, bbox))

            if not mineru_text_with_bbox:
                continue

            mineru_text_with_bbox.sort(
                key=lambda item: (
                    float(item[1][1]),
                    float(item[1][0]),
                )
            )
            mineru_text_indices = [item[0] for item in mineru_text_with_bbox]

            try:
                pdf_page = doc.load_page(page_index)
                pix = pdf_page.get_pixmap(dpi=int(max(72, ocr_render_dpi)), alpha=False)  # type: ignore[attr-defined]
                page_img_path = ocr_dir / f"page-{page_index:04d}.png"
                pix.save(str(page_img_path))
            except Exception as e:
                logger.warning(
                    "MinerU hybrid OCR render failed on page %s: %s", page_index, e
                )
                continue

            try:
                ocr_elements = ocr_image_to_elements(
                    str(page_img_path),
                    page_width_pt=page_w_pt,
                    page_height_pt=page_h_pt,
                    ocr_manager=ocr_manager,
                    text_refiner=None,
                    strict_no_fallback=bool(ocr_strict_mode),
                )
            except Exception as e:
                logger.warning("MinerU hybrid OCR failed on page %s: %s", page_index, e)
                continue

            ocr_lines = [
                el
                for el in ocr_elements
                if isinstance(el, dict)
                and el.get("type") == "text"
                and _coerce_bbox_pt(el.get("bbox_pt")) is not None
            ]
            if not ocr_lines:
                continue
            ocr_lines.sort(
                key=lambda el: (
                    float(el["bbox_pt"][1]),
                    float(el["bbox_pt"][0]),
                )
            )

            used_ocr_indices: set[int] = set()
            page_matched = 0

            for idx in mineru_text_indices:
                mineru_el = elements[idx]
                mineru_bbox = _coerce_bbox_pt(mineru_el.get("bbox_pt"))
                if mineru_bbox is None:
                    continue
                mineru_text = str(mineru_el.get("text") or "")
                mineru_norm_text = _normalize_match_text(mineru_text)

                best_idx: int | None = None
                best_score = -1.0
                best_overlap = 0.0
                best_center = 0.0
                best_text_ratio = 0.0

                for oidx, ocr_el in enumerate(ocr_lines):
                    if oidx in used_ocr_indices:
                        continue
                    ocr_bbox = _coerce_bbox_pt(ocr_el.get("bbox_pt"))
                    if ocr_bbox is None:
                        continue

                    overlap = _bbox_overlap_ratio(mineru_bbox, ocr_bbox)
                    dist_ratio = _bbox_center_distance_ratio(
                        mineru_bbox,
                        ocr_bbox,
                        page_w_pt=page_w_pt,
                        page_h_pt=page_h_pt,
                    )
                    center_score = max(0.0, 1.0 - min(1.0, dist_ratio * 2.5))

                    ocr_norm_text = _normalize_match_text(str(ocr_el.get("text") or ""))
                    if mineru_norm_text and ocr_norm_text:
                        text_ratio = SequenceMatcher(
                            None, mineru_norm_text, ocr_norm_text
                        ).ratio()
                    else:
                        text_ratio = 0.0

                    score = (
                        (0.55 * overlap) + (0.30 * text_ratio) + (0.15 * center_score)
                    )
                    if score > best_score:
                        best_idx = oidx
                        best_score = score
                        best_overlap = overlap
                        best_center = center_score
                        best_text_ratio = text_ratio

                if best_idx is None:
                    continue

                acceptable = (
                    (best_overlap >= 0.24 and best_center >= 0.40)
                    or (best_text_ratio >= 0.72 and best_center >= 0.35)
                    or (best_score >= 0.58)
                )
                if not acceptable:
                    continue

                matched_bbox = _coerce_bbox_pt(ocr_lines[best_idx].get("bbox_pt"))
                if matched_bbox is None:
                    continue
                mineru_el["bbox_pt"] = [
                    float(matched_bbox[0]),
                    float(matched_bbox[1]),
                    float(matched_bbox[2]),
                    float(matched_bbox[3]),
                ]
                used_ocr_indices.add(best_idx)
                page_matched += 1

            page.setdefault("warnings", []).append(
                f"mineru_hybrid_ocr_matched={page_matched}/{len(mineru_text_indices)}"
            )
            mineru_text_total += len(mineru_text_indices)
            matched_total += page_matched
            pages_used += 1
    finally:
        doc.close()

    ir.setdefault("warnings", []).append(
        f"mineru_hybrid_ocr=pages:{pages_used},matched:{matched_total}/{mineru_text_total}"
    )
    return ir
