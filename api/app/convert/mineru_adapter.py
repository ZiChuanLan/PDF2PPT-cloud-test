# pyright: reportMissingImports=false

"""MinerU API integration helpers."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, cast

import httpx
import pymupdf

from app.models.error import AppException, ErrorCode
from app.utils.text import clean_str as _clean_str


_DEFAULT_BASE_URL = "https://mineru.net"
_DEFAULT_PAGE_WIDTH_PT = 1000.0
_DEFAULT_PAGE_HEIGHT_PT = 1000.0
_TERMINAL_STATES = {"done", "failed"}
_ACTIVE_STATES = {"waiting-file", "pending", "running", "converting"}
_IMAGE_KIND_TOKENS = (
    "image",
    "img",
    "figure",
    "picture",
    "photo",
    "chart",
    "graphic",
    "illustration",
)


def _normalize_hex_color(value: Any) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            r = max(0, min(255, int(value[0])))
            g = max(0, min(255, int(value[1])))
            b = max(0, min(255, int(value[2])))
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return None

    cleaned = _clean_str(value if isinstance(value, str) else None)
    if not cleaned:
        return None
    normalized = cleaned.strip()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    elif normalized.lower().startswith("0x"):
        normalized = normalized[2:]

    if len(normalized) == 8:
        # Keep RGB and drop alpha for AARRGGBB-like values.
        normalized = normalized[-6:]
    if len(normalized) == 3:
        normalized = "".join(ch * 2 for ch in normalized)
    if len(normalized) != 6:
        return None
    try:
        int(normalized, 16)
    except Exception:
        return None
    return f"#{normalized.lower()}"


def _extract_style_value(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item.get(key)
    style = item.get("style")
    if isinstance(style, dict):
        for key in keys:
            if key in style:
                return style.get(key)
    return None


def _extract_positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            num = float(value)
        except Exception:
            return None
        return num if num > 0 else None
    if isinstance(value, str):
        cleaned = value.strip().lower().replace("pt", "").strip()
        if not cleaned:
            return None
        try:
            num = float(cleaned)
        except Exception:
            return None
        return num if num > 0 else None
    return None


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return None
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    if lowered in {"bold", "semibold"}:
        return True
    if lowered in {"normal", "regular"}:
        return False
    if lowered.isdigit():
        try:
            return int(lowered) >= 600
        except Exception:
            return None
    return None


def _extract_text_style(item: dict[str, Any]) -> dict[str, Any]:
    style: dict[str, Any] = {}

    color = _normalize_hex_color(
        _extract_style_value(item, "color", "text_color", "font_color", "fg_color")
    )
    if color:
        style["color"] = color

    font_size = _extract_positive_float(
        _extract_style_value(item, "font_size_pt", "font_size", "size", "text_size")
    )
    if font_size is not None and 1.0 <= font_size <= 240.0:
        style["font_size_pt"] = float(font_size)

    font_name = _clean_str(
        _extract_style_value(item, "font_name", "font", "font_family", "family")
    )
    if font_name:
        style["font_name"] = font_name

    bold = _coerce_optional_bool(
        _extract_style_value(
            item, "bold", "is_bold", "font_bold", "font_weight", "weight"
        )
    )
    if bold is not None:
        style["bold"] = bool(bold)

    italic = _coerce_optional_bool(
        _extract_style_value(item, "italic", "is_italic", "font_italic", "slant")
    )
    if italic is not None:
        style["italic"] = bool(italic)

    return style


def _normalize_mineru_token(value: str | None) -> str | None:
    cleaned = _clean_str(value)
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered.startswith("bearer "):
        token = cleaned[7:].strip()
        return token if token else None
    return cleaned


def _parse_page_ranges(page_start: int | None, page_end: int | None) -> str | None:
    if page_start is None and page_end is None:
        return None
    if page_start is None or page_end is None:
        return None
    start = int(page_start)
    end = int(page_end)
    if start <= 0 or end <= 0 or start > end:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid page range",
            details={"page_start": page_start, "page_end": page_end},
        )
    return f"{start}-{end}"


def _find_json_file(
    root: Path,
    *,
    exact_name: str,
    suffix_name: str,
    contain_name: str | None = None,
) -> Path | None:
    exact_name_lower = exact_name.lower()
    suffix_name_lower = suffix_name.lower()
    contain_name_lower = contain_name.lower() if contain_name else None

    candidates: list[Path] = []
    for path in root.rglob("*.json"):
        name_lower = path.name.lower()
        if name_lower == exact_name_lower or name_lower.endswith(suffix_name_lower):
            candidates.append(path)
            continue
        if contain_name_lower and contain_name_lower in name_lower:
            candidates.append(path)
    if not candidates:
        return None

    def _candidate_sort_key(path: Path) -> tuple[int, int, str]:
        name_lower = path.name.lower()
        if name_lower == exact_name_lower:
            rank = 0
        elif name_lower.endswith(suffix_name_lower):
            rank = 1
        elif contain_name_lower and contain_name_lower in name_lower:
            rank = 2
        else:
            rank = 3
        return (rank, len(str(path)), str(path))

    candidates.sort(key=_candidate_sort_key)
    return candidates[0]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="Failed to parse MinerU output JSON",
            details={"path": str(path), "error": str(e)},
            status_code=500,
        )


def _extract_page_sizes(middle_payload: Any) -> dict[int, tuple[float, float]]:
    page_sizes: dict[int, tuple[float, float]] = {}
    pdf_info: Any = None

    if isinstance(middle_payload, dict):
        pdf_info = middle_payload.get("pdf_info")
        if pdf_info is None and isinstance(middle_payload.get("data"), dict):
            pdf_info = middle_payload["data"].get("pdf_info")

    if not isinstance(pdf_info, list):
        return page_sizes

    for fallback_idx, page in enumerate(pdf_info):
        if not isinstance(page, dict):
            continue
        idx_raw = page.get("page_idx")
        try:
            page_idx = int(idx_raw) if idx_raw is not None else int(fallback_idx)
        except Exception:
            page_idx = int(fallback_idx)

        size = page.get("page_size")
        if not isinstance(size, list) or len(size) != 2:
            continue
        try:
            page_w = float(size[0])
            page_h = float(size[1])
        except Exception:
            continue
        if page_w <= 0 or page_h <= 0:
            continue
        page_sizes[page_idx] = (page_w, page_h)

    return page_sizes


def _extract_pdf_page_sizes(pdf_path: Path) -> dict[int, tuple[float, float]]:
    page_sizes: dict[int, tuple[float, float]] = {}
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return page_sizes
    try:
        for idx in range(int(doc.page_count or 0)):
            page = doc.load_page(idx)
            page_sizes[idx] = (float(page.rect.width), float(page.rect.height))
    except Exception:
        return {}
    finally:
        doc.close()
    return page_sizes


def _with_inferred_page_idx(item: dict[str, Any], *, page_idx: int) -> dict[str, Any]:
    for key in ("page_idx", "page_index", "page"):
        if key in item:
            return item
    copied = dict(item)
    copied["page_idx"] = int(page_idx)
    return copied


def _extract_items_from_sequence(sequence: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, entry in enumerate(sequence):
        if isinstance(entry, dict):
            items.append(entry)
            continue
        if isinstance(entry, list):
            # content_list_v2 may be list[list[dict]], where each outer index
            # is a page and inner items don't carry page_idx explicitly.
            for nested in entry:
                if isinstance(nested, dict):
                    items.append(_with_inferred_page_idx(nested, page_idx=idx))
    return items


def _extract_content_items(content_payload: Any) -> list[dict[str, Any]]:
    if isinstance(content_payload, list):
        direct = _extract_items_from_sequence(content_payload)
        if direct:
            return direct

    if not isinstance(content_payload, dict):
        return []

    direct = content_payload.get("content_list")
    if isinstance(direct, list):
        direct_items = _extract_items_from_sequence(direct)
        if direct_items:
            return direct_items

    nested_data = content_payload.get("data")
    if isinstance(nested_data, dict):
        nested = nested_data.get("content_list")
        if isinstance(nested, list):
            nested_items = _extract_items_from_sequence(nested)
            if nested_items:
                return nested_items

    fallback_items: list[dict[str, Any]] = []
    for value in content_payload.values():
        if isinstance(value, list):
            fallback_items.extend(_extract_items_from_sequence(value))
    return fallback_items


def _is_layout_formula_span(span_type: str) -> bool:
    lowered = str(span_type or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in ("equation", "formula", "latex", "math"))


def _is_layout_image_span(span_type: str) -> bool:
    lowered = str(span_type or "").strip().lower()
    if not lowered:
        return False
    return lowered in {"image", "img", "figure", "picture", "photo"}


def _normalize_layout_image_path(path_value: Any) -> str | None:
    cleaned = _clean_str(path_value if isinstance(path_value, str) else None)
    if not cleaned:
        return None
    if "/" not in cleaned and "\\" not in cleaned:
        return f"images/{cleaned}"
    return cleaned


def _extract_layout_line_items(
    *,
    lines: Any,
    block_type: str,
    block_bbox: Any,
    page_idx: int,
) -> list[dict[str, Any]]:
    out_items: list[dict[str, Any]] = []
    if not isinstance(lines, list):
        return out_items

    fallback_bbox = (
        list(block_bbox)
        if isinstance(block_bbox, list) and len(block_bbox) == 4
        else None
    )

    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list) or not spans:
            continue

        line_bbox_raw = line.get("bbox")
        line_bbox = (
            list(line_bbox_raw)
            if isinstance(line_bbox_raw, list) and len(line_bbox_raw) == 4
            else fallback_bbox
        )

        text_parts: list[str] = []
        has_formula_span = False
        line_style = _extract_text_style(line)

        for span in spans:
            if not isinstance(span, dict):
                continue
            span_type = str(span.get("type") or "").strip().lower()
            span_bbox_raw = span.get("bbox")
            span_bbox = (
                list(span_bbox_raw)
                if isinstance(span_bbox_raw, list) and len(span_bbox_raw) == 4
                else line_bbox
            )

            if _is_layout_image_span(span_type):
                normalized_path = _normalize_layout_image_path(
                    span.get("image_path") or span.get("path")
                )
                if normalized_path and span_bbox is not None:
                    out_items.append(
                        {
                            "type": "image",
                            "bbox": list(span_bbox),
                            "page_idx": int(page_idx),
                            "img_path": normalized_path,
                            "bbox_mode": "absolute",
                        }
                    )
                continue

            if span_type == "text" or _is_layout_formula_span(span_type):
                content = span.get("content")
                if isinstance(content, str) and content.strip():
                    text_parts.append(content.strip())
                    if _is_layout_formula_span(span_type):
                        has_formula_span = True
                    span_style = _extract_text_style(span)
                    for key, value in span_style.items():
                        line_style.setdefault(key, value)

        if text_parts and line_bbox is not None:
            line_kind = block_type or "text"
            if has_formula_span and line_kind in {"text", "paragraph", "list"}:
                # Keep equation-bearing lines out of heading/list heuristics.
                line_kind = "equation"
            line_item: dict[str, Any] = {
                "type": line_kind,
                "bbox": list(line_bbox),
                "page_idx": int(page_idx),
                "text": "".join(text_parts),
                "bbox_mode": "absolute",
            }
            if line_style:
                line_item.update(line_style)
            out_items.append(line_item)

    return out_items


def _collect_layout_block_items(
    block: Any,
    *,
    page_idx: int,
    out_items: list[dict[str, Any]],
) -> None:
    if not isinstance(block, dict):
        return

    block_type = str(block.get("type") or "").strip().lower()
    bbox = block.get("bbox")
    nested_blocks = block.get("blocks")

    # List blocks in layout.json usually contain per-item child blocks with more
    # accurate bboxes; prefer children over a single merged parent box.
    if block_type == "list" and isinstance(nested_blocks, list) and nested_blocks:
        for nested in nested_blocks:
            _collect_layout_block_items(nested, page_idx=page_idx, out_items=out_items)
        return

    line_items = _extract_layout_line_items(
        lines=block.get("lines"),
        block_type=block_type,
        block_bbox=bbox,
        page_idx=page_idx,
    )
    if line_items:
        out_items.extend(line_items)
        return

    # Fallback for unexpected layout variants where content is attached directly
    # to block-level fields without `lines/spans`.
    direct_image_path = _normalize_layout_image_path(
        block.get("image_path") or block.get("img_path") or block.get("path")
    )
    if direct_image_path and isinstance(bbox, list) and len(bbox) == 4:
        out_items.append(
            {
                "type": "image",
                "bbox": list(bbox),
                "page_idx": int(page_idx),
                "img_path": direct_image_path,
                "bbox_mode": "absolute",
            }
        )
        return

    if isinstance(bbox, list) and len(bbox) == 4:
        direct_text = _extract_text(block)
        if direct_text:
            direct_item: dict[str, Any] = {
                "type": block_type or "text",
                "bbox": list(bbox),
                "page_idx": int(page_idx),
                "text": direct_text,
                "bbox_mode": "absolute",
            }
            block_style = _extract_text_style(block)
            if block_style:
                direct_item.update(block_style)
            out_items.append(direct_item)
            return

    if isinstance(nested_blocks, list):
        for nested in nested_blocks:
            _collect_layout_block_items(nested, page_idx=page_idx, out_items=out_items)


def _extract_content_items_from_layout(layout_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(layout_payload, dict):
        return []

    pdf_info = layout_payload.get("pdf_info")
    if pdf_info is None and isinstance(layout_payload.get("data"), dict):
        pdf_info = layout_payload["data"].get("pdf_info")
    if not isinstance(pdf_info, list):
        return []

    items: list[dict[str, Any]] = []
    for fallback_page_idx, page in enumerate(pdf_info):
        if not isinstance(page, dict):
            continue
        raw_page_idx = page.get("page_idx")
        try:
            page_idx = (
                int(raw_page_idx)
                if raw_page_idx is not None
                else int(fallback_page_idx)
            )
        except Exception:
            page_idx = int(fallback_page_idx)
        para_blocks = page.get("para_blocks")
        if not isinstance(para_blocks, list):
            continue
        for block in para_blocks:
            _collect_layout_block_items(block, page_idx=page_idx, out_items=items)

    return items


def _extract_page_idx(item: dict[str, Any], *, fallback: int = 0) -> int:
    for key in ("page_idx", "page_index", "page"):
        if key in item:
            try:
                return int(item[key])
            except Exception:
                continue
    return int(fallback)


def _collect_text_fragments(value: Any, fragments: list[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            fragments.append(text)
        return

    if isinstance(value, list):
        for row in value:
            _collect_text_fragments(row, fragments)
        return

    if not isinstance(value, dict):
        return

    for key, nested in value.items():
        key_lower = str(key).strip().lower()
        if key_lower in {
            "bbox",
            "poly",
            "page_idx",
            "page_index",
            "page",
            "id",
            "index",
            "level",
            "text_level",
            "type",
            "sub_type",
            "list_type",
            "item_type",
            "img_path",
            "image_path",
            "image_source",
        }:
            continue
        _collect_text_fragments(nested, fragments)


def _join_text_fragments(fragments: list[str]) -> str:
    seen: set[str] = set()
    deduped: list[str] = []
    for part in fragments:
        normalized = part.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return "\n".join(deduped)


def _extract_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "latex"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    content = item.get("content")
    if isinstance(content, (list, dict)):
        parts: list[str] = []
        _collect_text_fragments(content, parts)
        merged = _join_text_fragments(parts)
        if merged:
            return merged

    list_items = item.get("list_items")
    if isinstance(list_items, list):
        parts: list[str] = []
        _collect_text_fragments(list_items, parts)
        merged = _join_text_fragments(parts)
        if merged:
            return merged

    list_items = item.get("list_item_infos")
    if isinstance(list_items, list):
        parts: list[str] = []
        for row in list_items:
            if not isinstance(row, dict):
                continue
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return _join_text_fragments(parts)

    table_body = item.get("table_body")
    if isinstance(table_body, str) and table_body.strip():
        return table_body.strip()

    return ""


def _extract_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            x0 = float(bbox[0])
            y0 = float(bbox[1])
            x1 = float(bbox[2])
            y1 = float(bbox[3])
            return (x0, y0, x1, y1)
        except Exception:
            return None

    poly = item.get("poly")
    if isinstance(poly, list) and len(poly) >= 8:
        coords: list[float] = []
        for value in poly:
            try:
                coords.append(float(value))
            except Exception:
                return None
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    return None


def _extract_item_kind(item: dict[str, Any]) -> str:
    for key in (
        "type",
        "category_type",
        "block_type",
        "kind",
        "tag",
        "content_type",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _is_image_like_kind(kind: str) -> bool:
    if not kind:
        return False
    lowered = kind.lower()
    return any(token in lowered for token in _IMAGE_KIND_TOKENS)


def _extract_image_rel_path(item: dict[str, Any]) -> str | None:
    for key in ("img_path", "image_path", "path"):
        value = item.get(key)
        cleaned = _clean_str(value if isinstance(value, str) else None)
        if cleaned:
            return cleaned

    content = item.get("content")
    if isinstance(content, dict):
        image_source = content.get("image_source")
        if isinstance(image_source, dict):
            for key in ("path", "img_path", "image_path", "url"):
                value = image_source.get(key)
                cleaned = _clean_str(value if isinstance(value, str) else None)
                if cleaned:
                    return cleaned
        if isinstance(image_source, str):
            cleaned = _clean_str(image_source)
            if cleaned:
                return cleaned

    return None


def _crop_pdf_region_png(
    *,
    doc: pymupdf.Document,
    page_index: int,
    bbox_pt: list[float],
    out_path: Path,
    zoom: float = 2.0,
) -> bool:
    if page_index < 0 or page_index >= int(doc.page_count or 0):
        return False

    try:
        x0, y0, x1, y1 = (
            float(bbox_pt[0]),
            float(bbox_pt[1]),
            float(bbox_pt[2]),
            float(bbox_pt[3]),
        )
    except Exception:
        return False
    if x1 <= x0 or y1 <= y0:
        return False

    page = doc.load_page(page_index)
    clip = pymupdf.Rect(x0, y0, x1, y1) & page.rect
    if clip.is_empty or clip.width <= 1.0 or clip.height <= 1.0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(  # type: ignore[attr-defined]
        matrix=pymupdf.Matrix(float(zoom), float(zoom)),
        clip=clip,
        alpha=False,
    )
    pix.save(str(out_path))
    return True


def _bbox_to_page_pt(
    bbox: tuple[float, float, float, float],
    *,
    page_width_pt: float,
    page_height_pt: float,
    assume_normalized: bool | None = None,
) -> list[float] | None:
    x0, y0, x1, y1 = bbox

    should_normalize = False
    if assume_normalized is True:
        should_normalize = True
    elif assume_normalized is None:
        # content_list commonly uses 0-1000 normalized coordinates.
        should_normalize = max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1100.0

    if should_normalize:
        x0 = (x0 / 1000.0) * page_width_pt
        x1 = (x1 / 1000.0) * page_width_pt
        y0 = (y0 / 1000.0) * page_height_pt
        y1 = (y1 / 1000.0) * page_height_pt

    left = max(0.0, min(float(x0), float(x1)))
    right = min(float(page_width_pt), max(float(x0), float(x1)))
    top = max(0.0, min(float(y0), float(y1)))
    bottom = min(float(page_height_pt), max(float(y0), float(y1)))

    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _build_ir_from_mineru_outputs(
    *,
    source_pdf: Path,
    content_items: list[dict[str, Any]],
    page_sizes: dict[int, tuple[float, float]],
    page_start: int | None = None,
    page_end: int | None = None,
    image_output_dir: Path | None = None,
    image_path_prefix: str | None = None,
    mineru_result_dir: Path | None = None,
    mineru_result_path_prefix: str | None = None,
) -> dict[str, Any]:
    item_page_pairs: list[tuple[dict[str, Any], int]] = [
        (item, _extract_page_idx(item, fallback=idx))
        for idx, item in enumerate(content_items)
    ]
    raw_indices = [page_idx for _, page_idx in item_page_pairs]

    page_index_shift = 0
    if raw_indices:
        target_start: int | None = None
        target_end: int | None = None
        if page_start is not None and page_end is not None:
            target_start = int(page_start) - 1
            target_end = int(page_end) - 1
        page_size_keys = set(page_sizes.keys())

        candidate_shifts: set[int] = {0, -1, 1}
        if target_start is not None:
            # MinerU may re-index selected page ranges to start from 0 (or 1).
            candidate_shifts.add(int(target_start))
            candidate_shifts.add(int(target_start - 1))

        best_key: tuple[int, int, int, int, int] | None = None
        best_shift = 0
        for shift in sorted(candidate_shifts):
            adjusted = [int(idx + shift) for idx in raw_indices]
            in_range_hits = (
                sum(
                    1
                    for idx in adjusted
                    if target_start is not None
                    and target_end is not None
                    and target_start <= idx <= target_end
                )
                if target_start is not None and target_end is not None
                else 0
            )
            page_size_hits = (
                sum(1 for idx in adjusted if idx in page_size_keys)
                if page_size_keys
                else 0
            )
            non_negative_hits = sum(1 for idx in adjusted if idx >= 0)
            prefer_zero = 1 if shift == 0 else 0
            key = (
                int(in_range_hits),
                int(page_size_hits),
                int(non_negative_hits),
                int(prefer_zero),
                int(-abs(int(shift))),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_shift = int(shift)

        page_index_shift = int(best_shift)

    items_by_page: dict[int, list[dict[str, Any]]] = {}
    for item, raw_page_idx in item_page_pairs:
        page_idx = int(raw_page_idx + page_index_shift)
        if page_idx < 0:
            continue
        items_by_page.setdefault(page_idx, []).append(item)

    ordered_indices = sorted(items_by_page.keys())
    if not ordered_indices:
        if page_start is not None and page_end is not None:
            start_idx = int(page_start) - 1
            end_idx = int(page_end) - 1
            ordered_indices = [idx for idx in range(start_idx, end_idx + 1) if idx >= 0]
        elif page_sizes:
            ordered_indices = sorted(page_sizes.keys())
        else:
            ordered_indices = [0]

    if page_start is not None and page_end is not None:
        start_idx = int(page_start) - 1
        end_idx = int(page_end) - 1
        ordered_indices = [
            idx for idx in ordered_indices if start_idx <= idx <= end_idx
        ]
        if not ordered_indices:
            ordered_indices = [idx for idx in range(start_idx, end_idx + 1) if idx >= 0]

    if not ordered_indices:
        if page_sizes:
            ordered_indices = sorted(page_sizes.keys())
        else:
            ordered_indices = [0]

    pages: list[dict[str, Any]] = []
    ir_warnings: list[str] = []
    image_prefix = _clean_str(image_path_prefix) or "images"
    result_prefix = _clean_str(mineru_result_path_prefix) or ""

    pdf_doc: pymupdf.Document | None = None
    if image_output_dir is not None and source_pdf.exists():
        try:
            pdf_doc = pymupdf.open(str(source_pdf))
        except Exception:
            pdf_doc = None

    image_counter = 0

    try:
        for page_idx in ordered_indices:
            page_w, page_h = page_sizes.get(
                page_idx, (_DEFAULT_PAGE_WIDTH_PT, _DEFAULT_PAGE_HEIGHT_PT)
            )
            page_items = items_by_page.get(page_idx, [])

            elements: list[dict[str, Any]] = []
            dropped_items = 0
            for item in page_items:
                kind = _extract_item_kind(item)
                bbox = _extract_bbox(item)
                if bbox is None:
                    dropped_items += 1
                    continue
                bbox_mode = str(item.get("bbox_mode") or "").strip().lower()
                assume_normalized: bool | None = None
                if bbox_mode == "absolute":
                    assume_normalized = False
                elif bbox_mode == "normalized":
                    assume_normalized = True
                bbox_pt = _bbox_to_page_pt(
                    bbox,
                    page_width_pt=float(page_w),
                    page_height_pt=float(page_h),
                    assume_normalized=assume_normalized,
                )
                if bbox_pt is None:
                    dropped_items += 1
                    continue

                text = _extract_text(item)
                if text and not _is_image_like_kind(kind):
                    text_element: dict[str, Any] = {
                        "type": "text",
                        "bbox_pt": bbox_pt,
                        "text": text,
                        "source": "mineru",
                        "mineru_block_type": kind,
                    }
                    text_style = _extract_text_style(item)
                    if text_style:
                        text_element.update(text_style)
                    text_level_raw = item.get("text_level")
                    if text_level_raw is not None:
                        try:
                            text_level = int(text_level_raw)
                            if text_level > 0:
                                text_element["mineru_text_level"] = text_level
                        except Exception:
                            pass
                    elements.append(text_element)
                    continue

                if _is_image_like_kind(kind):
                    rel_image_path = _extract_image_rel_path(item)
                    if (
                        rel_image_path
                        and mineru_result_dir is not None
                        and result_prefix
                    ):
                        image_added = False
                        candidate_paths: list[Path] = []
                        rel_path_obj = Path(rel_image_path)
                        if (
                            not rel_path_obj.is_absolute()
                            and ".." not in rel_path_obj.parts
                        ):
                            candidate_paths.append(rel_path_obj)
                            if len(rel_path_obj.parts) <= 1:
                                candidate_paths.append(
                                    Path("images") / rel_path_obj.name
                                )

                        if rel_image_path.startswith(("http://", "https://")):
                            file_name = rel_image_path.rsplit("/", 1)[-1].strip()
                            if file_name:
                                candidate_paths.append(Path("images") / file_name)

                        seen_candidate: set[str] = set()
                        for candidate_rel in candidate_paths:
                            key = candidate_rel.as_posix()
                            if key in seen_candidate:
                                continue
                            seen_candidate.add(key)
                            resolved = (mineru_result_dir / candidate_rel).resolve()
                            try:
                                resolved.relative_to(mineru_result_dir.resolve())
                                within_root = True
                            except Exception:
                                within_root = False
                            if not within_root:
                                continue
                            if not (resolved.exists() and resolved.is_file()):
                                continue
                            normalized_rel = candidate_rel.as_posix().lstrip("./")
                            elements.append(
                                {
                                    "type": "image",
                                    "bbox_pt": bbox_pt,
                                    "image_path": f"{result_prefix}/{normalized_rel}",
                                    "source": "mineru",
                                }
                            )
                            image_added = True
                            break
                        if image_added:
                            continue

                if (
                    _is_image_like_kind(kind)
                    and image_output_dir is not None
                    and pdf_doc is not None
                ):
                    image_counter += 1
                    image_name = (
                        f"page-{int(page_idx):04d}-img-{int(image_counter):04d}.png"
                    )
                    image_abs_path = image_output_dir / image_name
                    try:
                        saved = _crop_pdf_region_png(
                            doc=pdf_doc,
                            page_index=int(page_idx),
                            bbox_pt=bbox_pt,
                            out_path=image_abs_path,
                            zoom=2.0,
                        )
                    except Exception:
                        saved = False
                    if saved:
                        elements.append(
                            {
                                "type": "image",
                                "bbox_pt": bbox_pt,
                                "image_path": f"{image_prefix}/{image_name}",
                                "mime": "image/png",
                                "source": "mineru",
                            }
                        )

            elements.sort(key=lambda item: (item["bbox_pt"][1], item["bbox_pt"][0]))
            page_warnings: list[str] = []
            if dropped_items:
                page_warnings.append(f"mineru_items_dropped={dropped_items}")
            if not elements:
                page_warnings.append("mineru_no_elements")
            has_text_like_elements = any(
                str(el.get("type") or "").strip().lower() in {"text", "table"}
                for el in elements
            )

            pages.append(
                {
                    "page_index": int(page_idx),
                    "page_width_pt": float(page_w),
                    "page_height_pt": float(page_h),
                    "rotation": 0,
                    "elements": elements,
                    "warnings": page_warnings,
                    # Use direct placement path when MinerU blocks are available.
                    # If a page has no elements, fallback to scanned-page render path
                    # so the slide is never blank.
                    "has_text_layer": bool(has_text_like_elements),
                    "ocr_used": any(el.get("type") == "text" for el in elements),
                }
            )
    finally:
        if pdf_doc is not None:
            pdf_doc.close()

    pages.sort(key=lambda page: int(page.get("page_index") or 0))
    if not pages:
        ir_warnings.append("mineru_no_pages")
    if page_index_shift:
        ir_warnings.append(f"mineru_page_index_shift={page_index_shift}")

    source_page_count = len(page_sizes) if page_sizes else len(pages)
    selected_start = (
        int(page_start)
        if page_start is not None
        else (pages[0]["page_index"] + 1 if pages else 1)
    )
    selected_end = (
        int(page_end)
        if page_end is not None
        else (pages[-1]["page_index"] + 1 if pages else selected_start)
    )

    return {
        "source_pdf": str(source_pdf),
        "page_count": len(pages),
        "source_page_count": max(0, int(source_page_count)),
        "page_start": selected_start,
        "page_end": selected_end,
        "pages": pages,
        "warnings": ir_warnings,
    }


def _estimate_content_items_quality(items: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (usable_count, score) for candidate content_list payloads."""
    usable_count = 0
    score = 0
    for item in items:
        bbox = _extract_bbox(item)
        if bbox is None:
            continue
        usable_count += 1
        kind = _extract_item_kind(item)
        text = _extract_text(item)
        if text and not _is_image_like_kind(kind):
            score += 4
            # Prefer finer-grained line-like text items for accurate placement.
            text_len = len(text.replace("\n", "").strip())
            bbox_h = abs(float(bbox[3]) - float(bbox[1]))
            if text_len <= 100:
                score += 1
            elif text_len >= 500:
                score -= 2
            elif text_len >= 240:
                score -= 1
            if bbox_h <= 30.0:
                score += 2
            elif bbox_h <= 55.0:
                score += 1
            elif bbox_h >= 110.0:
                score -= 2
            elif bbox_h >= 75.0:
                score -= 1
            continue
        if _is_image_like_kind(kind):
            if _extract_image_rel_path(item):
                score += 3
            else:
                score += 2
            continue
        score += 1
    return (usable_count, score)


def _estimate_text_bbox_stats(items: list[dict[str, Any]]) -> dict[str, float] | None:
    heights: list[float] = []
    for item in items:
        kind = _extract_item_kind(item)
        if _is_image_like_kind(kind):
            continue
        text = _extract_text(item)
        if not text:
            continue
        bbox = _extract_bbox(item)
        if bbox is None:
            continue
        h = abs(float(bbox[3]) - float(bbox[1]))
        if h > 0:
            heights.append(h)

    if not heights:
        return None

    heights.sort()
    mid = len(heights) // 2
    p90_idx = int(round((len(heights) - 1) * 0.9))
    p90_idx = max(0, min(p90_idx, len(heights) - 1))
    return {
        "count": float(len(heights)),
        "median_h": float(heights[mid]),
        "p90_h": float(heights[p90_idx]),
    }


def _should_prefer_layout_candidate(
    *,
    content_items: list[dict[str, Any]],
    content_score: tuple[int, int],
    layout_items: list[dict[str, Any]],
    layout_score: tuple[int, int],
) -> bool:
    # Strong winner by existing scoring.
    if layout_score > content_score:
        return True

    # Near-tie fallback: prefer layout when its text boxes are materially tighter.
    score_gap = int(content_score[1]) - int(layout_score[1])
    usable_gap = int(content_score[0]) - int(layout_score[0])
    if score_gap > 8 or usable_gap > 2:
        return False

    content_stats = _estimate_text_bbox_stats(content_items)
    layout_stats = _estimate_text_bbox_stats(layout_items)
    if content_stats is None or layout_stats is None:
        return False

    content_median = max(1.0, float(content_stats["median_h"]))
    layout_median = max(1.0, float(layout_stats["median_h"]))
    content_p90 = max(1.0, float(content_stats["p90_h"]))
    layout_p90 = max(1.0, float(layout_stats["p90_h"]))

    return layout_median <= (0.82 * content_median) and layout_p90 <= (
        0.88 * content_p90
    )


class MineruClient:
    """Minimal MinerU client for file-upload batch parsing."""

    def __init__(
        self,
        *,
        token: str,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        token_cleaned = _normalize_mineru_token(token)
        if not token_cleaned:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="MinerU token is required",
            )
        self.base_url = _clean_str(base_url) or _DEFAULT_BASE_URL
        self._headers = {
            "Authorization": f"Bearer {token_cleaned}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        self._timeout = float(timeout_seconds)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        try:
            response = httpx.request(
                method,
                url,
                headers=self._headers,
                json=json_body,
                timeout=self._timeout,
            )
        except Exception as e:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="MinerU request failed",
                details={"path": path, "error": str(e)},
                status_code=502,
            )

        try:
            payload = response.json()
        except Exception as e:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="MinerU returned invalid JSON",
                details={
                    "path": path,
                    "status_code": response.status_code,
                    "error": str(e),
                },
                status_code=502,
            )

        if response.status_code >= 400:
            msg_code = (
                str(payload.get("msgCode") or payload.get("code") or "").strip().upper()
            )
            if msg_code in {"A0202", "A0211"} or response.status_code == 401:
                raise AppException(
                    code=ErrorCode.CONVERSION_FAILED,
                    message="MinerU token invalid or expired",
                    details={
                        "path": path,
                        "status_code": response.status_code,
                        "response": payload,
                    },
                    status_code=502,
                )
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="MinerU request rejected",
                details={
                    "path": path,
                    "status_code": response.status_code,
                    "response": payload,
                },
                status_code=502,
            )

        code = payload.get("code")
        if code not in (0, "0"):
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="MinerU API returned an error",
                details={"path": path, "response": payload},
                status_code=502,
            )
        return payload

    def create_upload_batch(
        self,
        *,
        file_name: str,
        data_id: str | None = None,
        model_version: str | None = None,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        is_ocr: bool | None = None,
        page_ranges: str | None = None,
    ) -> tuple[str, str]:
        file_item: dict[str, Any] = {"name": file_name}
        if _clean_str(data_id):
            file_item["data_id"] = _clean_str(data_id)
        if is_ocr is not None:
            file_item["is_ocr"] = bool(is_ocr)
        if _clean_str(page_ranges):
            file_item["page_ranges"] = _clean_str(page_ranges)

        body: dict[str, Any] = {
            "files": [file_item],
            "model_version": _clean_str(model_version) or "vlm",
        }
        if enable_formula is not None:
            body["enable_formula"] = bool(enable_formula)
        if enable_table is not None:
            body["enable_table"] = bool(enable_table)
        if _clean_str(language):
            body["language"] = _clean_str(language)

        payload = self._request_json("POST", "/api/v4/file-urls/batch", json_body=body)
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="MinerU returned invalid upload batch payload",
                details={"response": payload},
                status_code=502,
            )

        batch_id = _clean_str(str(data.get("batch_id") or ""))
        file_urls = data.get("file_urls") or data.get("files") or []
        upload_url = ""
        if isinstance(file_urls, list) and file_urls:
            upload_url = str(file_urls[0] or "").strip()

        if not batch_id or not upload_url:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="MinerU did not return upload URL",
                details={"response": payload},
                status_code=502,
            )

        return (batch_id, upload_url)

    def upload_file(self, *, upload_url: str, file_path: Path) -> None:
        try:
            with file_path.open("rb") as f:
                response = httpx.put(
                    upload_url,
                    data=cast(Any, f),
                    timeout=max(self._timeout, 120.0),
                )
        except Exception as e:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="Failed to upload file to MinerU",
                details={"error": str(e)},
                status_code=502,
            )

        if response.status_code >= 400:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="MinerU upload URL rejected file",
                details={"status_code": response.status_code},
                status_code=502,
            )

    def poll_batch_result(
        self,
        *,
        batch_id: str,
        poll_interval_s: float = 2.0,
        timeout_s: float = 1200.0,
        cancel_check: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Poll MinerU for batch results.

        Parameters
        ----------
        cancel_check:
            Optional callable invoked every poll iteration.  If it raises an
            exception (e.g. ``JobCancelledError``) the polling loop is aborted
            immediately, preventing the 20-minute blocking window.
        """
        deadline = time.monotonic() + float(timeout_s)

        while True:
            # Allow the caller to abort the poll early (e.g. job cancelled).
            if cancel_check is not None:
                cancel_check()

            payload = self._request_json(
                "GET", f"/api/v4/extract-results/batch/{batch_id}"
            )
            data = payload.get("data") or {}
            extract_result = (
                data.get("extract_result") if isinstance(data, dict) else None
            )

            first_item: dict[str, Any] | None = None
            if isinstance(extract_result, list) and extract_result:
                if isinstance(extract_result[0], dict):
                    first_item = extract_result[0]

            if first_item is None:
                if time.monotonic() >= deadline:
                    raise AppException(
                        code=ErrorCode.CONVERSION_FAILED,
                        message="Timed out waiting for MinerU batch result",
                        details={"batch_id": batch_id},
                        status_code=504,
                    )
                time.sleep(max(0.2, float(poll_interval_s)))
                continue

            state = str(first_item.get("state") or "").strip().lower()
            if state in _TERMINAL_STATES:
                return first_item

            if state not in _ACTIVE_STATES and state:
                # Unknown state; continue polling but include context in timeout.
                pass

            if time.monotonic() >= deadline:
                raise AppException(
                    code=ErrorCode.CONVERSION_FAILED,
                    message="Timed out waiting for MinerU parsing to finish",
                    details={"batch_id": batch_id, "last_state": state},
                    status_code=504,
                )

            time.sleep(max(0.2, float(poll_interval_s)))

    def download_result_zip(self, *, zip_url: str, output_zip: Path) -> None:
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.stream(
                "GET", zip_url, timeout=max(self._timeout, 120.0)
            ) as response:
                response.raise_for_status()
                with output_zip.open("wb") as f:
                    for chunk in response.iter_bytes():
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="Failed to download MinerU result archive",
                details={"error": str(e)},
                status_code=502,
            )


def parse_pdf_to_ir_with_mineru(
    pdf_path: str | Path,
    artifacts_dir: str | Path,
    *,
    token: str | None,
    base_url: str | None = None,
    model_version: str | None = None,
    enable_formula: bool | None = None,
    enable_table: bool | None = None,
    language: str | None = None,
    is_ocr: bool | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    data_id: str | None = None,
    poll_interval_s: float = 2.0,
    poll_timeout_s: float = 1200.0,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    path = Path(pdf_path)
    if not path.exists():
        raise AppException(
            code=ErrorCode.INVALID_PDF,
            message="PDF file not found",
            details={"path": str(path)},
        )

    token_cleaned = _normalize_mineru_token(token)
    if not token_cleaned:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="MinerU token is required when parse_provider=mineru",
        )

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_ranges = _parse_page_ranges(page_start=page_start, page_end=page_end)

    client = MineruClient(token=token_cleaned, base_url=base_url)
    batch_id, upload_url = client.create_upload_batch(
        file_name=path.name,
        data_id=data_id,
        model_version=model_version,
        enable_formula=enable_formula,
        enable_table=enable_table,
        language=language,
        is_ocr=is_ocr,
        page_ranges=page_ranges,
    )

    (out_dir / "create_batch.json").write_text(
        json.dumps(
            {"batch_id": batch_id, "upload_url": upload_url},
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    client.upload_file(upload_url=upload_url, file_path=path)
    result = client.poll_batch_result(
        batch_id=batch_id,
        poll_interval_s=poll_interval_s,
        timeout_s=poll_timeout_s,
        cancel_check=cancel_check,
    )
    (out_dir / "batch_result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    state = str(result.get("state") or "").strip().lower()
    if state != "done":
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="MinerU parsing failed",
            details={
                "state": result.get("state"),
                "err_msg": result.get("err_msg"),
                "batch_id": batch_id,
            },
            status_code=502,
        )

    zip_url = _clean_str(result.get("full_zip_url"))
    if not zip_url:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="MinerU did not return result archive URL",
            details={"batch_id": batch_id, "result": result},
            status_code=502,
        )

    archive_path = out_dir / "result.zip"
    extracted_dir = out_dir / "result"
    client.download_result_zip(zip_url=zip_url, output_zip=archive_path)

    extracted_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extracted_dir)
    except Exception as e:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="Failed to extract MinerU result archive",
            details={"path": str(archive_path), "error": str(e)},
            status_code=500,
        )

    content_candidates = [
        path
        for path in extracted_dir.rglob("*.json")
        if "content_list" in path.name.lower()
    ]
    if not content_candidates:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="MinerU output missing content_list JSON",
            details={"path": str(extracted_dir)},
            status_code=500,
        )

    def _content_candidate_sort_key(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        if name == "content_list.json":
            rank = 0
        elif name.endswith("_content_list.json"):
            rank = 1
        elif "content_list_v2" in name:
            rank = 2
        else:
            rank = 3
        return (rank, len(str(path)), str(path))

    content_candidates.sort(key=_content_candidate_sort_key)

    content_json: Path | None = None
    content_items: list[dict[str, Any]] = []
    best_score: tuple[int, int] = (-1, -1)
    for candidate in content_candidates:
        items = _extract_content_items(_load_json(candidate))
        candidate_score = _estimate_content_items_quality(items)
        if candidate_score > best_score:
            best_score = candidate_score
            content_json = candidate
            content_items = items

    if content_json is None:
        content_json = content_candidates[0]
        content_items = _extract_content_items(_load_json(content_json))
    selected_source = (
        f"content:{content_json.name}"
        if content_json is not None
        else "content:unknown"
    )

    layout_json = _find_json_file(
        extracted_dir,
        exact_name="layout.json",
        suffix_name="_layout.json",
        contain_name="layout",
    )
    layout_score: tuple[int, int] = (-1, -1)
    if layout_json is not None:
        layout_items = _extract_content_items_from_layout(_load_json(layout_json))
        layout_score = _estimate_content_items_quality(layout_items)
        if _should_prefer_layout_candidate(
            content_items=content_items,
            content_score=best_score,
            layout_items=layout_items,
            layout_score=layout_score,
        ):
            content_items = layout_items
            best_score = layout_score
            selected_source = f"layout:{layout_json.name}"

    if not content_items:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="MinerU result JSON has no parseable items",
            details={
                "content_json": str(content_json) if content_json is not None else None,
                "layout_json": str(layout_json) if layout_json is not None else None,
            },
            status_code=500,
        )

    middle_json = _find_json_file(
        extracted_dir,
        exact_name="middle.json",
        suffix_name="_middle.json",
        contain_name="middle",
    )
    page_sizes: dict[int, tuple[float, float]] = {}
    if middle_json is not None:
        middle_payload = _load_json(middle_json)
        page_sizes = _extract_page_sizes(middle_payload)
    pdf_page_sizes = _extract_pdf_page_sizes(path)
    if pdf_page_sizes:
        merged_page_sizes = dict(page_sizes)
        merged_page_sizes.update(pdf_page_sizes)
        page_sizes = merged_page_sizes

    ir = _build_ir_from_mineru_outputs(
        source_pdf=path,
        content_items=content_items,
        page_sizes=page_sizes,
        page_start=page_start,
        page_end=page_end,
        image_output_dir=out_dir / "images",
        image_path_prefix=f"{out_dir.name}/images",
        mineru_result_dir=extracted_dir,
        mineru_result_path_prefix=f"{out_dir.name}/result",
    )
    ir["warnings"] = list(ir.get("warnings") or [])
    ir["warnings"].append(f"mineru_batch_id={batch_id}")
    ir["warnings"].append(f"mineru_content_json={content_json.name}")
    if layout_json is not None:
        ir["warnings"].append(
            f"mineru_layout_quality=usable:{layout_score[0]},score:{layout_score[1]}"
        )
    ir["warnings"].append(f"mineru_selected_source={selected_source}")
    ir["warnings"].append(
        f"mineru_content_quality=usable:{best_score[0]},score:{best_score[1]}"
    )
    return ir
