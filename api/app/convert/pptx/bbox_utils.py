"""Bounding-box and path/coordinate utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..geometry import require_bbox_xyxy

from .constants import SlideTransform, _EMU_PER_PT
from .font_utils import _is_cjk_char

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _coerce_bbox_pt(bbox: Any) -> tuple[float, float, float, float]:
    return require_bbox_xyxy(bbox)


def _bbox_area_ratio_pt(
    bbox: Any, *, page_w_pt: float, page_h_pt: float
) -> float:
    """Return bbox/page area ratio in pt-space, or 0 for invalid inputs."""

    if page_w_pt <= 0 or page_h_pt <= 0:
        return 0.0
    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox)
    except Exception:
        return 0.0
    area = max(0.0, float(x1 - x0) * float(y1 - y0))
    page_area = max(1.0, float(page_w_pt) * float(page_h_pt))
    return float(area) / float(page_area)


def _is_near_full_page_bbox_pt(
    bbox: Any,
    *,
    page_w_pt: float,
    page_h_pt: float,
    min_area_ratio: float = 0.88,
    edge_tol_ratio: float = 0.015,
) -> bool:
    """Whether bbox is essentially a page-sized background image."""

    if page_w_pt <= 0 or page_h_pt <= 0:
        return False
    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox)
    except Exception:
        return False
    if x1 <= x0 or y1 <= y0:
        return False

    area_ratio = _bbox_area_ratio_pt(
        [x0, y0, x1, y1], page_w_pt=page_w_pt, page_h_pt=page_h_pt
    )
    tol_x = max(2.0, float(edge_tol_ratio) * float(page_w_pt))
    tol_y = max(2.0, float(edge_tol_ratio) * float(page_h_pt))
    touches_edges = (
        x0 <= tol_x
        and y0 <= tol_y
        and (float(page_w_pt) - x1) <= tol_x
        and (float(page_h_pt) - y1) <= tol_y
    )
    return bool(area_ratio >= float(min_area_ratio) and touches_edges)

def _bbox_pt_to_slide_emu(
    bbox_pt: Any, *, transform: SlideTransform
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)

    left = transform.offset_x_emu + x0 * _EMU_PER_PT * transform.scale
    # IR coordinates come from PyMuPDF (`page.get_text("dict")`) and OCR which both
    # use a top-left origin with Y increasing downward.
    top = transform.offset_y_emu + y0 * _EMU_PER_PT * transform.scale
    width = (x1 - x0) * _EMU_PER_PT * transform.scale
    height = (y1 - y0) * _EMU_PER_PT * transform.scale

    # PowerPoint expects integer EMUs.
    return (
        int(round(left)),
        int(round(top)),
        max(0, int(round(width))),
        max(0, int(round(height))),
    )

def _bbox_intersection_area_pt(a: list[float], b: list[float]) -> float:
    try:
        ax0, ay0, ax1, ay1 = _coerce_bbox_pt(a)
        bx0, by0, bx1, by1 = _coerce_bbox_pt(b)
    except Exception:
        return 0.0
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


def _bbox_iou_pt(a: list[float], b: list[float]) -> float:
    inter = _bbox_intersection_area_pt(a, b)
    if inter <= 0.0:
        return 0.0
    try:
        ax0, ay0, ax1, ay1 = _coerce_bbox_pt(a)
        bx0, by0, bx1, by1 = _coerce_bbox_pt(b)
    except Exception:
        return 0.0
    a_area = max(1.0, float((ax1 - ax0) * (ay1 - ay0)))
    b_area = max(1.0, float((bx1 - bx0) * (by1 - by0)))
    return float(inter) / max(1.0, float(a_area + b_area - inter))

def _compute_text_erase_padding_pt(
    *,
    bbox_h_pt: float,
    text_erase_mode: str,
) -> tuple[float, float]:
    """Compute erase padding (in pt) for OCR text cleanup.

    Use a shared strategy for scanned OCR and MinerU text cleanup so the
    rendered erase behavior stays consistent across parse backends.
    """

    h_pt = max(1.0, float(bbox_h_pt))
    mode = str(text_erase_mode or "smart").strip().lower()

    if mode == "fill":
        # Fill mode should stay local, but still cover anti-aliased glyph halos.
        # Slightly stronger padding reduces residual ghosting on OCR-heavy slides.
        pad_x_pt = max(1.3, min(6.6, 0.30 * h_pt))
        pad_y_pt = max(1.0, min(4.4, 0.23 * h_pt))
    else:
        # Smart mode supports wider context because replacement is pixel-adaptive.
        pad_x_pt = max(1.0, min(8.0, 0.35 * h_pt))
        pad_y_pt = max(0.8, min(4.0, 0.20 * h_pt))

    return (float(pad_x_pt), float(pad_y_pt))

def _normalize_text_for_bbox_dedupe(text: str) -> str:
    return "".join(
        ch.lower()
        for ch in str(text or "")
        if ch.isalnum() or _is_cjk_char(ch)
    )


def _texts_similar_for_bbox_dedupe(a: str, b: str) -> bool:
    na = _normalize_text_for_bbox_dedupe(a)
    nb = _normalize_text_for_bbox_dedupe(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        short = min(len(na), len(nb))
        long = max(len(na), len(nb))
        return short >= 3 and (float(short) / float(long)) >= 0.66
    return False
