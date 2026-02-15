"""OCR utility helpers."""

from typing import Any

from ..geometry import coerce_bbox_xyxy as geometry_coerce_bbox_xyxy
from .base import _clean_str


def _looks_like_structural_gibberish(text: str) -> bool:
    """Detect malformed gateway outputs made mostly of JSON delimiters.

    Some OpenAI-compatible OCR gateways occasionally return long streams like
    `}}]}]}...` with almost no textual content. These payloads are not valid
    OCR results and retrying with smaller limits rarely helps.
    """

    content = str(text or "").strip()
    if len(content) < 64:
        return False

    compact = "".join(ch for ch in content if not ch.isspace())
    if len(compact) < 64:
        return False

    structural = sum(1 for ch in compact if ch in "{}[],:")
    alnum = sum(1 for ch in compact if ch.isalnum())
    structural_ratio = float(structural) / float(max(1, len(compact)))
    alnum_ratio = float(alnum) / float(max(1, len(compact)))

    if structural_ratio >= 0.88 and alnum_ratio <= 0.06:
        return True

    if "}}]}" in compact or "}]}]" in compact:
        repeat_hits = compact.count("}}]}") + compact.count("}]}]")
        if repeat_hits >= max(6, len(compact) // 80):
            return True

    return False


def _coerce_bbox_xyxy(raw_bbox: Any) -> list[float] | None:
    base_box = geometry_coerce_bbox_xyxy(raw_bbox)
    if base_box is not None:
        return [
            float(base_box[0]),
            float(base_box[1]),
            float(base_box[2]),
            float(base_box[3]),
        ]

    if raw_bbox is None:
        return None

    # Numpy arrays (and some other tensor-like objects) show up in OCR SDK
    # outputs (e.g. PaddleX / PaddleOCR 3.x). Convert them to Python lists so
    # downstream logic can treat them uniformly.
    if hasattr(raw_bbox, "tolist"):
        try:
            raw_bbox = raw_bbox.tolist()
        except Exception:
            pass

    if isinstance(raw_bbox, dict):
        if all(k in raw_bbox for k in ("left", "top", "width", "height")):
            try:
                x0 = float(raw_bbox.get("left") or 0)
                y0 = float(raw_bbox.get("top") or 0)
                width = float(raw_bbox.get("width") or 0)
                height = float(raw_bbox.get("height") or 0)
                return [x0, y0, x0 + width, y0 + height]
            except Exception:
                return None
        for keys in (("x0", "y0", "x1", "y1"), ("xmin", "ymin", "xmax", "ymax")):
            if not all(k in raw_bbox for k in keys):
                continue
            try:
                x0_raw = raw_bbox.get(keys[0])
                y0_raw = raw_bbox.get(keys[1])
                x1_raw = raw_bbox.get(keys[2])
                y1_raw = raw_bbox.get(keys[3])
                if x0_raw is None or y0_raw is None or x1_raw is None or y1_raw is None:
                    return None
                x0 = float(x0_raw)
                y0 = float(y0_raw)
                x1 = float(x1_raw)
                y1 = float(y1_raw)
                return [x0, y0, x1, y1]
            except Exception:
                return None
        return None

    if isinstance(raw_bbox, tuple):
        raw_bbox = list(raw_bbox)
    if not isinstance(raw_bbox, list):
        return None

    if len(raw_bbox) == 4 and all(isinstance(v, (int, float)) for v in raw_bbox):
        return [
            float(raw_bbox[0]),
            float(raw_bbox[1]),
            float(raw_bbox[2]),
            float(raw_bbox[3]),
        ]

    if raw_bbox and all(isinstance(v, dict) for v in raw_bbox):
        xs: list[float] = []
        ys: list[float] = []
        for point in raw_bbox:
            try:
                x = point.get("x")
                y = point.get("y")
                if x is None:
                    x = point.get("left")
                if y is None:
                    y = point.get("top")
                xs.append(float(x))
                ys.append(float(y))
            except Exception:
                return None
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]

    if raw_bbox and all(isinstance(v, list) and len(v) >= 2 for v in raw_bbox):
        xs: list[float] = []
        ys: list[float] = []
        for point in raw_bbox:
            try:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
            except Exception:
                return None
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]

    if (
        len(raw_bbox) >= 8
        and len(raw_bbox) % 2 == 0
        and all(isinstance(v, (int, float)) for v in raw_bbox)
    ):
        xs = [float(raw_bbox[i]) for i in range(0, len(raw_bbox), 2)]
        ys = [float(raw_bbox[i]) for i in range(1, len(raw_bbox), 2)]
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]

    return None


def _is_paddleocr_vl_model(model_name: str | None) -> bool:
    cleaned = _clean_str(model_name)
    if not cleaned:
        return False
    return "paddleocr-vl" in cleaned.lower()
