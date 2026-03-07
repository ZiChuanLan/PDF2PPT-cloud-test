"""Pure helpers for OCR result parsing and Paddle doc-parser outputs."""

from __future__ import annotations

import json
import math
import os
from typing import Any

from .base import _clean_str, _strip_loc_tokens
from .deepseek_parser import _extract_deepseek_grounding_regions
from .json_extraction import _extract_message_text
from .utils import _coerce_bbox_xyxy

_IMAGE_REGION_LABEL_TOKENS = {
    "image",
    "figure",
    "chart",
    "diagram",
    "illustration",
    "graphic",
    "picture",
    "photo",
    "screenshot",
    "logo",
    "icon",
    "seal",
}

_NON_IMAGE_REGION_LABEL_TOKENS = {
    "text",
    "table",
    "formula",
    "caption",
    "title",
    "paragraph",
    "list",
    "header",
    "footer",
}

_PADDLE_DOC_VLM_PIXEL_FACTOR = 28 * 28
_PADDLE_DOC_VLM_MIN_PIXELS = _PADDLE_DOC_VLM_PIXEL_FACTOR * 130
_PADDLE_DOC_VLM_DEFAULT_MAX_PIXELS = _PADDLE_DOC_VLM_PIXEL_FACTOR * 1280
_PADDLE_DOC_VLM_BASE_MAX_SIDE_PX = 2200


def _quantize_paddle_doc_pixels(value: int) -> int:
    normalized = max(int(value), _PADDLE_DOC_VLM_MIN_PIXELS)
    return max(
        _PADDLE_DOC_VLM_MIN_PIXELS,
        int(normalized // _PADDLE_DOC_VLM_PIXEL_FACTOR) * _PADDLE_DOC_VLM_PIXEL_FACTOR,
    )


def _derive_paddle_doc_predict_max_pixels(
    *,
    max_side_px: int,
    did_downscale: bool,
) -> int | None:
    raw_override = _clean_str(os.getenv("OCR_PADDLE_VL_DOCPARSER_MAX_PIXELS"))
    if raw_override is not None:
        try:
            parsed_override = int(raw_override)
        except Exception:
            parsed_override = 0
        if parsed_override > 0:
            return _quantize_paddle_doc_pixels(parsed_override)
        return None

    if not did_downscale:
        return None

    normalized_max_side = max(0, int(max_side_px))
    if normalized_max_side <= 0:
        return None
    if normalized_max_side >= _PADDLE_DOC_VLM_BASE_MAX_SIDE_PX:
        return None

    ratio = float(normalized_max_side) / float(_PADDLE_DOC_VLM_BASE_MAX_SIDE_PX)
    scaled_max_pixels = int(
        math.floor(
            (_PADDLE_DOC_VLM_DEFAULT_MAX_PIXELS * ratio * ratio)
            / _PADDLE_DOC_VLM_PIXEL_FACTOR
        )
    ) * _PADDLE_DOC_VLM_PIXEL_FACTOR
    return max(_PADDLE_DOC_VLM_MIN_PIXELS, scaled_max_pixels)


def _normalize_bbox_px(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
    except Exception:
        return None
    if math.isnan(x0) or math.isnan(y0) or math.isnan(x1) or math.isnan(y1):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _normalize_layout_label(value: Any) -> str:
    cleaned = _clean_str(str(value or ""))
    if not cleaned:
        return ""
    return (
        str(cleaned)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _is_image_like_layout_label(value: Any) -> bool:
    label = _normalize_layout_label(value)
    if not label:
        return False
    if any(token in label for token in _NON_IMAGE_REGION_LABEL_TOKENS):
        return False
    return any(token in label for token in _IMAGE_REGION_LABEL_TOKENS)


def _extract_image_regions_from_json_payload(value: Any) -> list[list[float]] | None:
    if isinstance(value, list):
        out: list[list[float]] = []
        for item in value:
            bbox_raw = item.get("bbox") if isinstance(item, dict) else item
            bbox = _coerce_bbox_xyxy(bbox_raw)
            if bbox is None:
                continue
            out.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
        return out or None

    if not isinstance(value, dict):
        return None

    for key in (
        "image_regions",
        "regions",
        "items",
        "result",
        "results",
        "data",
        "boxes",
    ):
        if key not in value:
            continue
        extracted = _extract_image_regions_from_json_payload(value.get(key))
        if extracted:
            return extracted

    bbox = _coerce_bbox_xyxy(value.get("bbox"))
    if bbox is not None:
        return [[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]]
    return None


def _extract_image_regions_json(text: Any) -> list[list[float]] | None:
    content = _extract_message_text(text)
    if not content:
        return None

    candidates: list[str] = [content]
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()
        if fenced:
            candidates.append(fenced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if parsed is not None:
            extracted = _extract_image_regions_from_json_payload(parsed)
            if extracted:
                return extracted

        start_idx = candidate.find("[")
        end_idx = candidate.rfind("]")
        if start_idx >= 0 and end_idx > start_idx:
            clipped = candidate[start_idx : end_idx + 1]
            try:
                parsed = json.loads(clipped)
            except Exception:
                parsed = None
            if parsed is not None:
                extracted = _extract_image_regions_from_json_payload(parsed)
                if extracted:
                    return extracted

    return None


def _extract_deepseek_image_regions(text: Any) -> list[list[float]] | None:
    regions = _extract_deepseek_grounding_regions(text)
    if not regions:
        return None

    out: list[list[float]] = []
    for item in regions:
        if not isinstance(item, dict):
            continue
        if not _is_image_like_layout_label(item.get("label")):
            continue
        bbox = _coerce_bbox_xyxy(item.get("bbox"))
        if bbox is None:
            continue
        out.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

    return out or None


def _extract_paddle_doc_parser_output(
    output: Any,
) -> tuple[list[dict[str, Any]], list[list[float]], list[dict[str, Any]]]:
    raw_elements: list[dict[str, Any]] = []
    image_regions: list[list[float]] = []
    layout_blocks: list[dict[str, Any]] = []

    def _extract_parsing_blocks(result_obj: Any) -> list[Any]:
        payload_candidates: list[Any] = []

        json_payload = getattr(result_obj, "json", None)
        if callable(json_payload):
            try:
                json_payload = json_payload()
            except Exception:
                json_payload = None
        if json_payload is not None:
            payload_candidates.append(json_payload)

        to_dict_payload = getattr(result_obj, "to_dict", None)
        if callable(to_dict_payload):
            try:
                payload_candidates.append(to_dict_payload())
            except Exception:
                pass

        payload_candidates.append(result_obj)

        for payload in payload_candidates:
            if not isinstance(payload, dict):
                continue
            root = payload.get("res") if isinstance(payload.get("res"), dict) else payload
            if not isinstance(root, dict):
                continue
            blocks = root.get("parsing_res_list")
            if isinstance(blocks, list):
                return blocks
        return []

    def _extract_block_fields(block: Any) -> tuple[str, Any, Any, str]:
        if isinstance(block, dict):
            text = _strip_loc_tokens(
                block.get("block_content")
                or block.get("content")
                or block.get("text")
                or ""
            )
            bbox_raw: Any = None
            for key in ("block_bbox", "bbox", "box", "b"):
                if key in block and block[key] is not None:
                    bbox_raw = block[key]
                    break

            confidence_raw: Any = None
            for key in ("confidence", "score", "prob"):
                if key in block and block[key] is not None:
                    confidence_raw = block[key]
                    break

            label = (
                block.get("block_label")
                or block.get("label")
                or block.get("block_type")
                or block.get("type")
                or block.get("category")
                or block.get("cls")
                or ""
            )
            return text, bbox_raw, confidence_raw, str(label or "")

        text = _strip_loc_tokens(
            getattr(block, "block_content", None)
            or getattr(block, "content", None)
            or getattr(block, "text", None)
            or ""
        )
        bbox_raw: Any = None
        for attr in ("block_bbox", "bbox", "box", "b"):
            value = getattr(block, attr, None)
            if value is not None:
                bbox_raw = value
                break

        confidence_raw: Any = None
        for attr in ("confidence", "score", "prob"):
            value = getattr(block, attr, None)
            if value is not None:
                confidence_raw = value
                break

        label = (
            getattr(block, "block_label", None)
            or getattr(block, "label", None)
            or getattr(block, "block_type", None)
            or getattr(block, "type", None)
            or getattr(block, "category", None)
            or getattr(block, "cls", None)
            or ""
        )
        return text, bbox_raw, confidence_raw, str(label or "")

    if isinstance(output, dict):
        results_iter = [output]
    elif isinstance(output, list):
        results_iter = output
    elif isinstance(output, tuple):
        results_iter = list(output)
    else:
        try:
            results_iter = list(output)
        except Exception:
            results_iter = [output]

    for result in results_iter:
        blocks = _extract_parsing_blocks(result)
        for block in blocks:
            text, bbox_raw, confidence_raw, label = _extract_block_fields(block)
            bbox = _coerce_bbox_xyxy(bbox_raw)
            if not bbox:
                continue

            bbox_xyxy = [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            ]
            label_normalized = _normalize_layout_label(label)
            layout_blocks.append(
                {
                    "label": label_normalized,
                    "bbox": list(bbox_xyxy),
                    "text": text,
                }
            )

            if _is_image_like_layout_label(label):
                image_regions.append(list(bbox_xyxy))

            if not text:
                continue

            try:
                confidence = (
                    float(confidence_raw) if confidence_raw is not None else 0.9
                )
            except Exception:
                confidence = 0.9
            if confidence > 1.0:
                confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
            confidence = max(0.0, min(confidence, 1.0))

            raw_elements.append(
                {
                    "text": text,
                    "bbox": list(bbox_xyxy),
                    "confidence": confidence,
                }
            )

    return raw_elements, image_regions, layout_blocks


def _scale_bbox_xyxy(bbox: list[float], *, scale_x: float, scale_y: float) -> list[float]:
    if len(bbox) != 4:
        return list(bbox)
    return [
        float(bbox[0]) * float(scale_x),
        float(bbox[1]) * float(scale_y),
        float(bbox[2]) * float(scale_x),
        float(bbox[3]) * float(scale_y),
    ]


def _scale_paddle_doc_parser_output(
    raw_elements: list[dict[str, Any]],
    image_regions: list[list[float]],
    layout_blocks: list[dict[str, Any]],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[list[dict[str, Any]], list[list[float]], list[dict[str, Any]]]:
    if abs(float(scale_x) - 1.0) < 1e-6 and abs(float(scale_y) - 1.0) < 1e-6:
        return raw_elements, image_regions, layout_blocks

    scaled_elements: list[dict[str, Any]] = []
    for item in raw_elements:
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            scaled_item = dict(item)
            scaled_item["bbox"] = _scale_bbox_xyxy(
                [float(v) for v in bbox],
                scale_x=scale_x,
                scale_y=scale_y,
            )
            scaled_elements.append(scaled_item)
        else:
            scaled_elements.append(dict(item))

    scaled_regions = [
        _scale_bbox_xyxy([float(v) for v in bbox], scale_x=scale_x, scale_y=scale_y)
        for bbox in image_regions
        if isinstance(bbox, list) and len(bbox) == 4
    ]

    scaled_layout_blocks: list[dict[str, Any]] = []
    for block in layout_blocks:
        bbox = block.get("bbox")
        scaled_block = dict(block)
        if isinstance(bbox, list) and len(bbox) == 4:
            scaled_block["bbox"] = _scale_bbox_xyxy(
                [float(v) for v in bbox],
                scale_x=scale_x,
                scale_y=scale_y,
            )
        scaled_layout_blocks.append(scaled_block)

    return scaled_elements, scaled_regions, scaled_layout_blocks


__all__ = [
    "_derive_paddle_doc_predict_max_pixels",
    "_extract_deepseek_image_regions",
    "_extract_image_regions_json",
    "_extract_paddle_doc_parser_output",
    "_normalize_bbox_px",
    "_scale_paddle_doc_parser_output",
]
