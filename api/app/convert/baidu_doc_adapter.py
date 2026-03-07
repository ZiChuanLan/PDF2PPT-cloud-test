"""Baidu document parser integration helpers."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pymupdf

from app.models.error import AppException, ErrorCode
from app.utils.text import clean_str as _clean_str

from .mineru_adapter import _build_ir_from_mineru_outputs, _extract_pdf_page_sizes


_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
_BAIDU_DOC_ENDPOINTS = {
    "general": {
        "task_url": "https://aip.baidubce.com/rest/2.0/brain/online/v2/parser/task",
        "result_url": "https://aip.baidubce.com/rest/2.0/brain/online/v2/parser/task/query",
    },
    "paddle_vl": {
        "task_url": "https://aip.baidubce.com/rest/2.0/brain/online/v2/paddle-vl-parser/task",
        "result_url": "https://aip.baidubce.com/rest/2.0/brain/online/v2/paddle-vl-parser/task/query",
    },
}
_SUCCESS_STATUSES = {"success", "succeeded", "done", "finished", "completed"}
_FAILED_STATUSES = {"failed", "error", "cancelled", "canceled", "timeout"}
_ACTIVE_STATUSES = {"created", "queued", "pending", "processing", "running"}
_IMAGE_KIND_TOKENS = (
    "image",
    "img",
    "figure",
    "picture",
    "photo",
    "chart",
    "graphic",
    "illustration",
    "screenshot",
    "logo",
    "seal",
)
_SKIP_CHILD_KEYS = {
    "bbox",
    "box",
    "rect",
    "rectangle",
    "polygon",
    "poly",
    "points",
    "position",
    "location",
    "coordinate",
    "coordinates",
    "text",
    "content",
    "value",
    "words",
    "markdown",
    "html",
    "label",
    "type",
    "kind",
    "category",
    "block_type",
    "layout_type",
    "name",
    "cls",
    "score",
    "confidence",
    "prob",
    "page_idx",
    "page_index",
    "page",
    "page_no",
    "page_num",
    "page_id",
    "index",
    "id",
    "file_url",
    "result_url",
    "image_path",
    "image_url",
    "url",
    "w",
    "width",
    "h",
    "height",
    "page_width",
    "page_height",
    "page_w",
    "page_h",
    "page_size",
    "size",
}


def _raise_if_baidu_error(payload: Any, *, context: str) -> None:
    if not isinstance(payload, dict):
        return
    error_code = payload.get("error_code")
    if error_code in (None, "", 0, "0"):
        return
    error_msg = payload.get("error_msg")
    message = f"Baidu document parser {context} failed"
    if isinstance(error_msg, str) and error_msg.strip():
        message = f"{message}: {error_msg.strip()}"
    raise AppException(
        code=ErrorCode.CONVERSION_FAILED,
        message=message,
        details={
            "error_code": error_code,
            "error_msg": error_msg,
            "payload": payload,
        },
        status_code=502,
    )


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower().replace("pt", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _extract_page_size(entry: Any) -> tuple[float, float] | None:
    if not isinstance(entry, dict):
        return None

    page_size = entry.get("page_size")
    if isinstance(page_size, (list, tuple)) and len(page_size) >= 2:
        width = _coerce_float(page_size[0])
        height = _coerce_float(page_size[1])
        if width and height and width > 0 and height > 0:
            return (float(width), float(height))

    size = entry.get("size")
    if isinstance(size, dict):
        width = _coerce_float(size.get("width") or size.get("w"))
        height = _coerce_float(size.get("height") or size.get("h"))
        if width and height and width > 0 and height > 0:
            return (float(width), float(height))

    has_page_size_context = any(
        key in entry
        for key in (
            "page_idx",
            "page_index",
            "page",
            "page_no",
            "page_num",
            "page_id",
            "page_width",
            "page_height",
            "page_w",
            "page_h",
            "page_size",
            "size",
            "blocks",
            "items",
            "elements",
            "layouts",
            "paragraphs",
            "regions",
        )
    )
    if not has_page_size_context:
        return None

    width = _coerce_float(entry.get("page_width") or entry.get("page_w"))
    height = _coerce_float(entry.get("page_height") or entry.get("page_h"))
    if width and height and width > 0 and height > 0:
        return (float(width), float(height))

    width = _coerce_float(entry.get("width"))
    height = _coerce_float(entry.get("height"))
    if width and height and width > 0 and height > 0:
        return (float(width), float(height))
    return None


def _extract_page_idx(entry: Any, *, fallback: int | None = None) -> int | None:
    if isinstance(entry, dict):
        for key in ("page_idx", "page_index", "page", "page_no", "page_num", "page_id"):
            if key not in entry:
                continue
            value = entry.get(key)
            try:
                page_idx = int(value)
            except Exception:
                continue
            if key in {"page_no", "page_num"} and page_idx > 0:
                return page_idx - 1
            return page_idx
    return fallback


def _extract_bbox_candidate(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        for key in (
            "bbox",
            "box",
            "rect",
            "rectangle",
            "polygon",
            "poly",
            "points",
            "position",
            "location",
            "coordinates",
            "coordinate",
        ):
            nested = value.get(key)
            if nested is not None:
                bbox = _extract_bbox_candidate(nested)
                if bbox is not None:
                    return bbox

        left = _coerce_float(value.get("left") or value.get("x0") or value.get("x"))
        top = _coerce_float(value.get("top") or value.get("y0") or value.get("y"))
        right = _coerce_float(value.get("right") or value.get("x1"))
        bottom = _coerce_float(value.get("bottom") or value.get("y1"))
        width = _coerce_float(value.get("width") or value.get("w"))
        height = _coerce_float(value.get("height") or value.get("h"))
        if (
            left is not None
            and top is not None
            and right is not None
            and bottom is not None
        ):
            return (float(left), float(top), float(right), float(bottom))
        if (
            left is not None
            and top is not None
            and width is not None
            and height is not None
        ):
            return (
                float(left),
                float(top),
                float(left + width),
                float(top + height),
            )
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 4:
            coords = [_coerce_float(item) for item in value]
            if all(item is not None for item in coords):
                x0, y0, x1, y1 = (float(item) for item in coords if item is not None)
                if x1 <= x0 or y1 <= y0:
                    return (x0, y0, x0 + x1, y0 + y1)
                return (x0, y0, x1, y1)
        if len(value) >= 8:
            coords = [_coerce_float(item) for item in value]
            if all(item is not None for item in coords):
                xs = [float(item) for idx, item in enumerate(coords) if idx % 2 == 0 and item is not None]
                ys = [float(item) for idx, item in enumerate(coords) if idx % 2 == 1 and item is not None]
                if xs and ys:
                    return (min(xs), min(ys), max(xs), max(ys))
        if value and all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in value):
            xs: list[float] = []
            ys: list[float] = []
            for item in value:
                x = _coerce_float(item[0])
                y = _coerce_float(item[1])
                if x is None or y is None:
                    return None
                xs.append(float(x))
                ys.append(float(y))
            if xs and ys:
                return (min(xs), min(ys), max(xs), max(ys))
    return None


def _extract_text(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, list):
        parts = [_extract_text(item) for item in entry]
        return "\n".join([part for part in parts if part]).strip()
    if not isinstance(entry, dict):
        return ""

    for key in ("text", "block_content", "content", "words", "value", "markdown", "title"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("lines", "paragraphs", "cells", "spans", "texts", "children"):
        value = entry.get(key)
        if isinstance(value, list):
            parts = [_extract_text(item) for item in value]
            merged = "\n".join([part for part in parts if part]).strip()
            if merged:
                return merged
    return ""


def _extract_kind(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in (
        "type",
        "kind",
        "category",
        "block_type",
        "layout_type",
        "label",
        "name",
        "cls",
        "role",
    ):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _extract_image_path(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    for key in ("image_path", "image_url", "file_url", "url"):
        value = entry.get(key)
        cleaned = _clean_str(value if isinstance(value, str) else None)
        if cleaned:
            return cleaned
    return None


def _is_image_like_kind(kind: str) -> bool:
    if not kind:
        return False
    lowered = kind.lower()
    return any(token in lowered for token in _IMAGE_KIND_TOKENS)


def _normalize_bbox_to_pdf_pt(
    bbox: tuple[float, float, float, float],
    *,
    payload_page_size: tuple[float, float] | None,
    pdf_page_size: tuple[float, float] | None,
) -> list[float] | None:
    if pdf_page_size is None:
        return None
    pdf_w, pdf_h = pdf_page_size
    if pdf_w <= 0 or pdf_h <= 0:
        return None

    x0, y0, x1, y1 = bbox
    left = min(float(x0), float(x1))
    right = max(float(x0), float(x1))
    top = min(float(y0), float(y1))
    bottom = max(float(y0), float(y1))
    if right <= left or bottom <= top:
        return None

    max_x = max(abs(left), abs(right))
    max_y = max(abs(top), abs(bottom))

    if max(max_x, max_y) <= 1.2:
        left *= pdf_w
        right *= pdf_w
        top *= pdf_h
        bottom *= pdf_h
    else:
        payload_w = float(payload_page_size[0]) if payload_page_size else 0.0
        payload_h = float(payload_page_size[1]) if payload_page_size else 0.0
        if payload_w > 0 and payload_h > 0:
            looks_like_1000_grid = payload_w <= 1100.0 and payload_h <= 1100.0
            if looks_like_1000_grid and max(max_x, max_y) <= 1100.0:
                left = left / payload_w * pdf_w
                right = right / payload_w * pdf_w
                top = top / payload_h * pdf_h
                bottom = bottom / payload_h * pdf_h
            elif right <= payload_w * 1.05 and bottom <= payload_h * 1.05:
                left = left / payload_w * pdf_w
                right = right / payload_w * pdf_w
                top = top / payload_h * pdf_h
                bottom = bottom / payload_h * pdf_h
        elif right <= pdf_w * 1.05 and bottom <= pdf_h * 1.05:
            pass
        elif max(max_x, max_y) <= 1100.0:
            left = left / 1000.0 * pdf_w
            right = right / 1000.0 * pdf_w
            top = top / 1000.0 * pdf_h
            bottom = bottom / 1000.0 * pdf_h
        else:
            scale_x = pdf_w / max(1.0, right)
            scale_y = pdf_h / max(1.0, bottom)
            scale = min(scale_x, scale_y)
            left *= scale
            right *= scale
            top *= scale
            bottom *= scale

    left = max(0.0, min(float(pdf_w), float(left)))
    right = max(0.0, min(float(pdf_w), float(right)))
    top = max(0.0, min(float(pdf_h), float(top)))
    bottom = max(0.0, min(float(pdf_h), float(bottom)))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _collect_page_payload(
    node: Any,
    *,
    fallback_page_idx: int | None = None,
    out_pages: dict[int, tuple[float, float]],
) -> None:
    if isinstance(node, list):
        for idx, item in enumerate(node):
            _collect_page_payload(item, fallback_page_idx=idx, out_pages=out_pages)
        return
    if not isinstance(node, dict):
        return

    page_idx = _extract_page_idx(node, fallback=fallback_page_idx)
    page_size = _extract_page_size(node)
    if page_idx is not None and page_size is not None:
        out_pages[int(page_idx)] = page_size

    for value in node.values():
        if isinstance(value, (dict, list)):
            _collect_page_payload(value, fallback_page_idx=page_idx, out_pages=out_pages)


def _build_content_item(
    entry: dict[str, Any],
    *,
    page_idx: int | None,
    payload_page_sizes: dict[int, tuple[float, float]],
    pdf_page_sizes: dict[int, tuple[float, float]],
) -> dict[str, Any] | None:
    if page_idx is None:
        return None

    bbox = _extract_bbox_candidate(entry)
    if bbox is None:
        return None

    pdf_page_size = pdf_page_sizes.get(int(page_idx))
    if pdf_page_size is None:
        return None
    bbox_pt = _normalize_bbox_to_pdf_pt(
        bbox,
        payload_page_size=payload_page_sizes.get(int(page_idx)),
        pdf_page_size=pdf_page_size,
    )
    if bbox_pt is None:
        return None

    kind = _extract_kind(entry)
    text = _extract_text(entry)
    rel_image_path = _extract_image_path(entry)
    if not text and not kind and not rel_image_path:
        return None

    item: dict[str, Any] = {
        "page_idx": int(page_idx),
        "bbox": bbox_pt,
        "bbox_mode": "absolute",
        "type": kind or ("image" if rel_image_path else "text"),
    }
    if text:
        item["text"] = text
    if rel_image_path:
        item["image_path"] = rel_image_path
    text_level = entry.get("text_level") or entry.get("level")
    if text_level is not None:
        item["text_level"] = text_level
    return item


def _collect_content_items(
    node: Any,
    *,
    page_idx: int | None,
    payload_page_sizes: dict[int, tuple[float, float]],
    pdf_page_sizes: dict[int, tuple[float, float]],
    out_items: list[dict[str, Any]],
    seen: set[tuple[int, str, str, str]],
) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_content_items(
                item,
                page_idx=page_idx,
                payload_page_sizes=payload_page_sizes,
                pdf_page_sizes=pdf_page_sizes,
                out_items=out_items,
                seen=seen,
            )
        return

    parsed = _maybe_parse_json(node)
    if parsed is not None and parsed is not node:
        _collect_content_items(
            parsed,
            page_idx=page_idx,
            payload_page_sizes=payload_page_sizes,
            pdf_page_sizes=pdf_page_sizes,
            out_items=out_items,
            seen=seen,
        )
        return

    if not isinstance(node, dict):
        return

    current_page_idx = _extract_page_idx(node, fallback=page_idx)
    item = _build_content_item(
        node,
        page_idx=current_page_idx,
        payload_page_sizes=payload_page_sizes,
        pdf_page_sizes=pdf_page_sizes,
    )
    if item is not None:
        key = (
            int(item["page_idx"]),
            json.dumps(item.get("bbox"), ensure_ascii=True),
            str(item.get("text") or ""),
            str(item.get("type") or ""),
        )
        if key not in seen:
            seen.add(key)
            out_items.append(item)

    for key, value in node.items():
        if key in _SKIP_CHILD_KEYS:
            continue
        if not isinstance(value, (dict, list, str)):
            continue
        _collect_content_items(
            value,
            page_idx=current_page_idx,
            payload_page_sizes=payload_page_sizes,
            pdf_page_sizes=pdf_page_sizes,
            out_items=out_items,
            seen=seen,
        )


def _extract_result_container(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in ("result", "data", "res"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _extract_status(payload: Any) -> str:
    container = _extract_result_container(payload)
    for source in (container, payload if isinstance(payload, dict) else {}):
        for key in ("status", "task_status", "state"):
            value = source.get(key)
            cleaned = _clean_str(value if isinstance(value, str) else None)
            if cleaned:
                return cleaned.lower()
    return ""


def _extract_task_id(payload: Any) -> str | None:
    container = _extract_result_container(payload)
    for source in (container, payload if isinstance(payload, dict) else {}):
        value = source.get("task_id")
        cleaned = _clean_str(value if isinstance(value, str) else None)
        if cleaned:
            return cleaned
    return None


def _find_result_url(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in (
            "parse_result_url",
            "result_url",
            "result_file_url",
            "result_json_url",
            "json_result_url",
            "file_url",
            "url",
        ):
            value = payload.get(key)
            cleaned = _clean_str(value if isinstance(value, str) else None)
            if cleaned and cleaned.startswith(("http://", "https://")):
                return cleaned
        for value in payload.values():
            found = _find_result_url(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_result_url(item)
            if found:
                return found
    return None


def _looks_like_parse_result_payload(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    if not isinstance(value, dict):
        return False
    if any(key in value for key in ("pages", "layouts", "tables", "images", "cells", "matrix")):
        return True
    return False


def _resolve_result_payload(payload: Any, *, client: httpx.Client) -> tuple[Any, str]:
    container = _extract_result_container(payload)
    candidate_values: list[tuple[Any, str]] = []
    for key in ("result_data", "json_result", "data", "res"):
        if key in container:
            candidate_values.append((container.get(key), f"inline:{key}"))
        if isinstance(payload, dict) and key in payload:
            candidate_values.append((payload.get(key), f"inline:{key}"))

    for value, label in candidate_values:
        parsed = _maybe_parse_json(value)
        if parsed is not None:
            return parsed, label
        if _looks_like_parse_result_payload(value):
            return value, label

    result_url = _find_result_url(container) or _find_result_url(payload)
    if result_url:
        response = client.get(result_url)
        response.raise_for_status()
        try:
            return response.json(), "remote:json_url"
        except Exception:
            parsed = _maybe_parse_json(response.text)
            if parsed is not None:
                return parsed, "remote:text_url"
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="Baidu document parser result URL did not return JSON",
                details={"result_url": result_url},
                status_code=502,
            )

    for value, label in ((container, "inline:result"), (payload, "inline:payload")):
        parsed = _maybe_parse_json(value)
        if parsed is not None:
            return parsed, label
        if _looks_like_parse_result_payload(value):
            return value, label

    raise AppException(
        code=ErrorCode.CONVERSION_FAILED,
        message="Baidu document parser returned no usable result payload",
        details={"payload": payload},
        status_code=502,
    )


class BaiduDocParserClient:
    def __init__(
        self,
        *,
        api_key: str,
        secret_key: str,
        parse_type: str = "paddle_vl",
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = _clean_str(api_key)
        self.secret_key = _clean_str(secret_key)
        self.parse_type = _clean_str(parse_type) or "paddle_vl"
        if not self.api_key or not self.secret_key:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="Baidu document parser requires api_key / secret_key",
                status_code=400,
            )
        endpoints = _BAIDU_DOC_ENDPOINTS.get(self.parse_type)
        if endpoints is None:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="Unsupported Baidu document parser type",
                details={"parse_type": parse_type},
                status_code=400,
            )
        self.task_url = endpoints["task_url"]
        self.result_url = endpoints["result_url"]
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=True,
        )
        self._access_token: str | None = None

    def close(self) -> None:
        self.client.close()

    def get_access_token(self) -> str:
        if self._access_token:
            return self._access_token
        response = self.client.post(
            _TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
        )
        response.raise_for_status()
        payload = response.json()
        _raise_if_baidu_error(payload, context="token request")
        token = _clean_str(payload.get("access_token") if isinstance(payload, dict) else None)
        if not token:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="Failed to obtain Baidu access token",
                details={"payload": payload},
                status_code=502,
            )
        self._access_token = token
        return token

    def submit_task(self, pdf_path: Path) -> dict[str, Any]:
        payload = {
            "file_data": base64.b64encode(pdf_path.read_bytes()).decode("ascii"),
            "file_name": pdf_path.name or "input.pdf",
        }
        response = self.client.post(
            self.task_url,
            params={"access_token": self.get_access_token()},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=payload,
        )
        response.raise_for_status()
        data = response.json()
        _raise_if_baidu_error(data, context="task submission")
        if _extract_task_id(data) is None:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="Baidu document parser did not return task_id",
                details={"payload": data},
                status_code=502,
            )
        return data

    def get_result(self, task_id: str) -> dict[str, Any]:
        response = self.client.post(
            self.result_url,
            params={"access_token": self.get_access_token()},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"task_id": task_id},
        )
        response.raise_for_status()
        payload = response.json()
        _raise_if_baidu_error(payload, context="result polling")
        return payload


def _create_selected_pdf(
    source_pdf: Path,
    out_pdf: Path,
    *,
    page_start: int | None,
    page_end: int | None,
) -> Path:
    if page_start is None or page_end is None:
        return source_pdf

    src = pymupdf.open(str(source_pdf))
    out = pymupdf.open()
    try:
        out.insert_pdf(src, from_page=int(page_start) - 1, to_page=int(page_end) - 1)
        out.save(str(out_pdf))
    except Exception as e:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="Failed to build selected-page PDF for Baidu parser",
            details={"error": str(e), "page_start": page_start, "page_end": page_end},
            status_code=500,
        )
    finally:
        out.close()
        src.close()
    return out_pdf


def parse_pdf_to_ir_with_baidu_doc(
    pdf_path: str | Path,
    artifacts_dir: str | Path,
    *,
    api_key: str | None,
    secret_key: str | None,
    parse_type: str = "paddle_vl",
    page_start: int | None = None,
    page_end: int | None = None,
    poll_interval_s: float = 2.0,
    poll_timeout_s: float = 900.0,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    path = Path(pdf_path)
    if not path.exists():
        raise AppException(
            code=ErrorCode.INVALID_PDF,
            message="PDF file not found",
            details={"path": str(path)},
        )

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    upload_pdf = _create_selected_pdf(
        path,
        out_dir / "selected-pages.pdf",
        page_start=page_start,
        page_end=page_end,
    )
    upload_page_sizes = _extract_pdf_page_sizes(upload_pdf)
    original_page_sizes = _extract_pdf_page_sizes(path)

    def _step_check() -> None:
        if cancel_check is not None:
            cancel_check()

    client: BaiduDocParserClient | None = None

    try:
        client = BaiduDocParserClient(
            api_key=_clean_str(api_key) or "",
            secret_key=_clean_str(secret_key) or "",
            parse_type=parse_type,
        )
        _step_check()
        submit_payload = client.submit_task(upload_pdf)
        (out_dir / "submit_task.json").write_text(
            json.dumps(submit_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        task_id = _extract_task_id(submit_payload)
        if not task_id:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="Baidu document parser missing task_id",
                details={"payload": submit_payload},
                status_code=502,
            )

        started_at = time.monotonic()
        latest_payload: dict[str, Any] = submit_payload
        latest_status = _extract_status(submit_payload)
        while True:
            _step_check()
            latest_status = _extract_status(latest_payload)
            if latest_status in _SUCCESS_STATUSES | _FAILED_STATUSES:
                break
            if time.monotonic() - started_at >= float(poll_timeout_s):
                raise AppException(
                    code=ErrorCode.CONVERSION_FAILED,
                    message="Baidu document parser timed out",
                    details={"task_id": task_id, "status": latest_status},
                    status_code=504,
                )
            time.sleep(max(0.2, float(poll_interval_s)))
            latest_payload = client.get_result(task_id)

        (out_dir / "task_result.json").write_text(
            json.dumps(latest_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        if latest_status in _FAILED_STATUSES:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="Baidu document parser failed",
                details={"task_id": task_id, "status": latest_status, "payload": latest_payload},
                status_code=502,
            )

        result_payload, result_source = _resolve_result_payload(latest_payload, client=client.client)
        (out_dir / "result_payload.json").write_text(
            json.dumps(result_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        payload_page_sizes: dict[int, tuple[float, float]] = {}
        _collect_page_payload(result_payload, out_pages=payload_page_sizes)
        content_items: list[dict[str, Any]] = []
        seen_items: set[tuple[int, str, str, str]] = set()
        _collect_content_items(
            result_payload,
            page_idx=None,
            payload_page_sizes=payload_page_sizes,
            pdf_page_sizes=upload_page_sizes,
            out_items=content_items,
            seen=seen_items,
        )
        if not content_items:
            raise AppException(
                code=ErrorCode.CONVERSION_FAILED,
                message="Baidu document parser returned no parseable layout items",
                details={"task_id": task_id, "status": latest_status},
                status_code=502,
            )

        ir = _build_ir_from_mineru_outputs(
            source_pdf=path,
            content_items=content_items,
            page_sizes=original_page_sizes or upload_page_sizes,
            page_start=page_start,
            page_end=page_end,
            image_output_dir=out_dir / "images",
            image_path_prefix=f"{out_dir.name}/images",
            layout_source="baidu_doc",
            warning_prefix="baidu_doc",
        )
        ir["warnings"] = list(ir.get("warnings") or [])
        ir["warnings"].append(f"baidu_doc_task_id={task_id}")
        ir["warnings"].append(f"baidu_doc_parse_type={parse_type}")
        ir["warnings"].append(f"baidu_doc_status={latest_status or 'success'}")
        ir["warnings"].append(f"baidu_doc_result_source={result_source}")
        return ir
    except httpx.HTTPError as e:
        response = getattr(e, "response", None)
        detail = {
            "error": str(e),
            "status_code": getattr(response, "status_code", None),
        }
        if response is not None:
            try:
                detail["response"] = response.json()
            except Exception:
                detail["response_text"] = response.text
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="Baidu document parser request failed",
            details=detail,
            status_code=502,
        ) from e
    finally:
        if client is not None:
            client.close()
