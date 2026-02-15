"""Slide-level helpers used by the main PPTX generator."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .constants import SlideTransform, _EMU_PER_PT

def _set_slide_size_type(prs: Any, *, slide_w_emu: int, slide_h_emu: int) -> None:
    """Set sldSz@type to reduce size/aspect surprises in some viewers."""

    try:
        w = float(slide_w_emu)
        h = float(slide_h_emu)
        if w <= 0 or h <= 0:
            return
        ratio = w / h
    except Exception:
        return

    # Prefer "custom" unless we're close to a known widescreen/standard ratio.
    candidates: dict[str, float] = {
        "screen4x3": 4.0 / 3.0,
        "screen16x10": 16.0 / 10.0,
        "screen16x9": 16.0 / 9.0,
    }
    best_type = min(candidates, key=lambda k: abs(ratio - candidates[k]))
    if abs(ratio - candidates[best_type]) > 0.08:
        best_type = "custom"

    try:
        prs.part.presentation._element.sldSz.set("type", best_type)  # type: ignore[attr-defined]
    except Exception:
        pass


def _infer_font_size_pt(element: dict[str, Any], *, bbox_h_pt: float) -> float:
    size = float(element.get("font_size_pt") or 0.0)
    if size > 0.1:
        return size
    # OCR blocks may not have font info. Use bbox height as a rough proxy.
    source = str(element.get("source") or "")
    if source == "ocr":
        # For scanned pages we want to visually match the source slide. OCR line
        # boxes are usually tight, so we can start from the bbox height and let
        # PowerPoint shrink text if needed (TEXT_TO_FIT_SHAPE).
        multiplier = 0.85
        return max(4.5, min(96.0, bbox_h_pt * multiplier))
    multiplier = 0.8
    return max(8.0, min(48.0, bbox_h_pt * multiplier))


def _build_transform(
    *,
    page_width_pt: float,
    page_height_pt: float,
    slide_width_emu: int,
    slide_height_emu: int,
) -> SlideTransform:
    if page_width_pt <= 0 or page_height_pt <= 0:
        raise ValueError("Invalid page dimensions")

    content_w_emu = page_width_pt * _EMU_PER_PT
    content_h_emu = page_height_pt * _EMU_PER_PT

    # Fit page content into slide while preserving aspect ratio.
    scale = min(slide_width_emu / content_w_emu, slide_height_emu / content_h_emu)
    offset_x = (slide_width_emu - content_w_emu * scale) / 2.0
    offset_y = (slide_height_emu - content_h_emu * scale) / 2.0

    return SlideTransform(
        page_width_pt=page_width_pt,
        page_height_pt=page_height_pt,
        slide_width_emu=int(slide_width_emu),
        slide_height_emu=int(slide_height_emu),
        scale=float(scale),
        offset_x_emu=float(offset_x),
        offset_y_emu=float(offset_y),
    )

def _iter_page_elements(
    page: dict[str, Any], *, type_name: str
) -> Iterable[dict[str, Any]]:
    for el in page.get("elements") or []:
        if isinstance(el, dict) and el.get("type") == type_name:
            yield el
