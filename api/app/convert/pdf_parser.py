"""PDF parsing + IR generation (PyMuPDF).

This module extracts layout-relevant information from a PDF into a simple
intermediate representation (IR) for downstream OCR / layout assist / PPTX
rendering steps.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pymupdf

from app.models.error import AppException, ErrorCode


_EXT_TO_MIME: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "jpe": "image/jpeg",
    "jpx": "image/jpx",
    "jp2": "image/jp2",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "webp": "image/webp",
}


def _bbox_to_list(bbox: Any) -> list[float]:
    if bbox is None:
        return [0.0, 0.0, 0.0, 0.0]
    if isinstance(bbox, pymupdf.Rect):
        return [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
    # typical: (x0, y0, x1, y1)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    raise ValueError(f"Unsupported bbox type: {type(bbox)}")


def _color_int_to_hex(color: Any) -> str:
    if isinstance(color, int):
        return f"#{color & 0xFFFFFF:06x}"
    return "#000000"


def _iter_text_line_elements(page_dict: dict[str, Any]) -> Iterable[dict[str, Any]]:
    blocks = page_dict.get("blocks") or []
    for block in blocks:
        if block.get("type") != 0:
            continue

        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = "".join(
                (span.get("text") or "") if isinstance(span, dict) else ""
                for span in spans
            )
            if not text.strip():
                continue

            # Pick a representative span for style.
            # Prefer the span with the most non-whitespace characters.
            rep_span: dict[str, Any] = {}
            rep_score = -1
            for span in spans:
                if not isinstance(span, dict):
                    continue
                span_text = span.get("text") or ""
                score = len(span_text.strip())
                if score > rep_score:
                    rep_span = span
                    rep_score = score

            if not rep_span and spans and isinstance(spans[0], dict):
                rep_span = spans[0]

            flags = int(rep_span.get("flags") or 0)
            yield {
                "type": "text",
                "bbox_pt": _bbox_to_list(line.get("bbox") or block.get("bbox")),
                "text": text,
                "font_size_pt": float(rep_span.get("size") or 0.0),
                "font_name": str(rep_span.get("font") or ""),
                "color": _color_int_to_hex(rep_span.get("color")),
                "bold": bool(flags & pymupdf.TEXT_FONT_BOLD),
                "italic": bool(flags & pymupdf.TEXT_FONT_ITALIC),
                "source": "pdf",
            }


def _extract_image_elements(
    page_dict: dict[str, Any],
    *,
    page_index: int,
    artifacts_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    elements: list[dict[str, Any]] = []

    images_dir = artifacts_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    blocks = page_dict.get("blocks") or []
    img_blocks = [b for b in blocks if b.get("type") == 1]
    for img_index, block in enumerate(img_blocks):
        bbox = block.get("bbox")
        ext = str(block.get("ext") or "png").lower().lstrip(".")
        image_bytes = block.get("image")
        if not image_bytes:
            warnings.append(
                f"image_block_missing_bytes: page={page_index} idx={img_index}"
            )
            continue

        filename = f"page-{page_index:04d}-img-{img_index:04d}.{ext}"
        out_path = images_dir / filename
        try:
            out_path.write_bytes(image_bytes)
        except Exception as e:
            warnings.append(
                f"image_write_failed: page={page_index} idx={img_index} error={e!s}"
            )
            continue

        rel_path = str(out_path.relative_to(artifacts_dir))
        elements.append(
            {
                "type": "image",
                "bbox_pt": _bbox_to_list(bbox),
                "image_path": rel_path,
                "mime": _EXT_TO_MIME.get(ext, "application/octet-stream"),
            }
        )

    return elements, warnings


def _extract_table_elements(
    page: pymupdf.Page,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    elements: list[dict[str, Any]] = []

    try:
        finder = page.find_tables()  # type: ignore[attr-defined]
    except Exception as e:
        warnings.append(f"table_detection_failed: error={e!s}")
        return elements, warnings

    if not finder or not getattr(finder, "tables", None):
        return elements, warnings

    for table in finder.tables:
        try:
            rows = int(table.row_count)
            cols = int(table.col_count)
            extracted = table.extract()  # list[list[str|None]]
            cell_rects = list(getattr(table, "cells", []) or [])
        except Exception as e:
            warnings.append(f"table_extract_failed: error={e!s}")
            continue

        cells: list[dict[str, Any]] = []
        for r in range(rows):
            row_data = extracted[r] if r < len(extracted) else []
            for c in range(cols):
                idx = r * cols + c
                rect = cell_rects[idx] if idx < len(cell_rects) else None
                text = ""
                if c < len(row_data):
                    text = str(row_data[c] or "")
                cells.append(
                    {
                        "r": r,
                        "c": c,
                        "bbox_pt": _bbox_to_list(rect),
                        "text": text,
                    }
                )

        elements.append(
            {
                "type": "table",
                "bbox_pt": _bbox_to_list(getattr(table, "bbox", None)),
                "rows": rows,
                "cols": cols,
                "cells": cells,
                "source": "pdf",
            }
        )

    return elements, warnings


def parse_pdf_to_ir(
    pdf_path: str | Path,
    artifacts_dir: str | Path,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    """Parse a PDF into IR.

    Args:
        pdf_path: Input PDF path.
        artifacts_dir: Directory to write extracted artifacts (e.g. images).

    Returns:
        A JSON-serializable dict with `pages[]` containing per-page IR.

    Raises:
        AppException: For invalid / encrypted PDFs.
    """

    path = Path(pdf_path)
    if not path.exists():
        raise AppException(
            code=ErrorCode.INVALID_PDF,
            message="PDF file not found",
            details={"path": str(path)},
        )

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        raise AppException(
            code=ErrorCode.INVALID_PDF,
            message="Unable to open PDF file",
            details={"error": str(e)},
        )

    if doc.is_encrypted:
        doc.close()
        raise AppException(
            code=ErrorCode.PDF_ENCRYPTED,
            message="PDF is password-protected",
            details={"encrypted": True},
        )

    doc_warnings: list[str] = []
    pages: list[dict[str, Any]] = []
    total_pages = int(doc.page_count or 0)
    start = int(page_start or 1)
    end = int(page_end or total_pages)
    if start < 1 or end < 1 or start > end or end > total_pages:
        doc.close()
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid page range",
            details={
                "page_start": page_start,
                "page_end": page_end,
                "total_pages": total_pages,
            },
        )

    page_indices = range(start - 1, end)

    try:
        for page_index in page_indices:
            page = doc.load_page(page_index)
            page_warnings: list[str] = []

            # Detect text layer without forcing any reading-order assumptions.
            try:
                has_text_layer = bool((page.get_text("text") or "").strip())  # type: ignore[attr-defined]
            except Exception as e:
                has_text_layer = False
                page_warnings.append(f"text_layer_check_failed: error={e!s}")

            try:
                page_dict = page.get_text("dict")  # type: ignore[attr-defined]
            except Exception as e:
                pages.append(
                    {
                        "page_index": page_index,
                        "page_width_pt": float(page.rect.width),
                        "page_height_pt": float(page.rect.height),
                        "rotation": int(getattr(page, "rotation", 0) or 0),
                        "elements": [],
                        "warnings": page_warnings
                        + [f"page_text_extract_failed: error={e!s}"],
                        "has_text_layer": has_text_layer,
                    }
                )
                continue

            elements: list[dict[str, Any]] = []

            elements.extend(list(_iter_text_line_elements(page_dict)))

            image_elements, image_warnings = _extract_image_elements(
                page_dict,
                page_index=page_index,
                artifacts_dir=out_dir,
            )
            elements.extend(image_elements)
            page_warnings.extend(image_warnings)

            table_elements, table_warnings = _extract_table_elements(page)
            elements.extend(table_elements)
            page_warnings.extend(table_warnings)

            if not has_text_layer:
                page_warnings.append("no_text_layer")

            if not elements:
                page_warnings.append("no_elements_extracted")

            pages.append(
                {
                    "page_index": page_index,
                    "page_width_pt": float(page.rect.width),
                    "page_height_pt": float(page.rect.height),
                    "rotation": int(getattr(page, "rotation", 0) or 0),
                    "elements": elements,
                    "warnings": page_warnings,
                    "has_text_layer": has_text_layer,
                }
            )
    finally:
        doc.close()

    return {
        "source_pdf": str(path),
        "page_count": len(pages),
        "source_page_count": total_pages,
        "page_start": start,
        "page_end": end,
        "pages": pages,
        "warnings": doc_warnings,
    }
