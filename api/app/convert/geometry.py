"""Shared geometry helpers for bbox normalization and coordinate mapping."""

from __future__ import annotations

import math
from typing import Any


def coerce_bbox_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    """Normalize a bbox-like value into `(x0, y0, x1, y1)` or return `None`."""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def require_bbox_xyxy(value: Any) -> tuple[float, float, float, float]:
    """Strict variant of :func:`coerce_bbox_xyxy` that raises on invalid input."""

    box = coerce_bbox_xyxy(value)
    if box is None:
        raise ValueError(f"Invalid bbox: {value!r}")
    return box


def bbox_pt_to_px(
    bbox: Any,
    *,
    page_w_pt: float,
    page_h_pt: float,
    img_w_px: int,
    img_h_px: int,
) -> tuple[int, int, int, int] | None:
    """Map bbox from PDF-point/page space to pixel/image space."""

    box = coerce_bbox_xyxy(bbox)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if page_w_pt <= 0 or page_h_pt <= 0 or img_w_px <= 0 or img_h_px <= 0:
        return None

    sx = float(img_w_px) / float(page_w_pt)
    sy = float(img_h_px) / float(page_h_pt)
    x0p = max(0, min(int(round(x0 * sx)), int(img_w_px - 1)))
    y0p = max(0, min(int(round(y0 * sy)), int(img_h_px - 1)))
    x1p = max(0, min(int(round(x1 * sx)), int(img_w_px)))
    y1p = max(0, min(int(round(y1 * sy)), int(img_h_px)))
    if x1p <= x0p or y1p <= y0p:
        return None
    return (x0p, y0p, x1p, y1p)


def bbox_px_to_pt(
    bbox: Any,
    *,
    img_w_px: int,
    img_h_px: int,
    page_w_pt: float,
    page_h_pt: float,
) -> list[float] | None:
    """Map bbox from pixel/image space to PDF-point/page space."""

    box = coerce_bbox_xyxy(bbox)
    if box is None:
        return None
    if img_w_px <= 0 or img_h_px <= 0 or page_w_pt <= 0 or page_h_pt <= 0:
        return None

    x0, y0, x1, y1 = box
    x0 = max(0.0, min(x0, float(img_w_px - 1)))
    y0 = max(0.0, min(y0, float(img_h_px - 1)))
    x1 = max(0.0, min(x1, float(img_w_px)))
    y1 = max(0.0, min(y1, float(img_h_px)))
    if x1 <= x0 or y1 <= y0:
        return None

    sx = float(page_w_pt) / float(img_w_px)
    sy = float(page_h_pt) / float(img_h_px)
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
