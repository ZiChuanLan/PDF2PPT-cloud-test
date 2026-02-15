from __future__ import annotations

import re
from typing import Any

from ..convert.geometry import bbox_pt_to_px as _bbox_pt_to_px_shared
from ..convert.geometry import coerce_bbox_xyxy


def _bbox_pt_to_px(
    bbox: Any,
    *,
    page_w_pt: float,
    page_h_pt: float,
    img_w_px: int,
    img_h_px: int,
) -> tuple[int, int, int, int] | None:
    return _bbox_pt_to_px_shared(
        bbox,
        page_w_pt=page_w_pt,
        page_h_pt=page_h_pt,
        img_w_px=img_w_px,
        img_h_px=img_h_px,
    )


def _coerce_bbox_pt(value: Any) -> tuple[float, float, float, float] | None:
    return coerce_bbox_xyxy(value)


def _normalize_match_text(text: str) -> str:
    cleaned = re.sub(r"\s+", "", str(text or "").lower())
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", cleaned)
    return cleaned


def _bbox_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    left_area = max(1.0, float((lx1 - lx0) * (ly1 - ly0)))
    right_area = max(1.0, float((rx1 - rx0) * (ry1 - ry0)))
    return float(inter) / float(min(left_area, right_area))


def _bbox_center_distance_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    page_w_pt: float,
    page_h_pt: float,
) -> float:
    lcx = (left[0] + left[2]) / 2.0
    lcy = (left[1] + left[3]) / 2.0
    rcx = (right[0] + right[2]) / 2.0
    rcy = (right[1] + right[3]) / 2.0
    dist = float(((lcx - rcx) ** 2 + (lcy - rcy) ** 2) ** 0.5)
    diag = max(1.0, float((page_w_pt**2 + page_h_pt**2) ** 0.5))
    return dist / diag
