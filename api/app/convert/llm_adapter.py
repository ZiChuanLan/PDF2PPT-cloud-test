# pyright: reportMissingImports=false, reportMissingTypeArgument=false

"""LLM-powered layout assist (optional).

This module provides a small, provider-pluggable interface for *narrow* layout
assists:
- reading order suggestions (useful for multi-column pages)
- table grid inference (useful when OCR returns words without table structure)

All usage must be best-effort: time-bounded (30s/page) and failures must not
break the baseline conversion.
"""

from __future__ import annotations

import base64
import copy
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from .geometry import bbox_px_to_pt
from ..logging_config import get_logger


logger = get_logger(__name__)

_PAGE_TIMEOUT_S = 30.0
_DEFAULT_RENDER_DPI = 150


def _allow_layout_assist_image_regions() -> bool:
    value = str(os.getenv("LAYOUT_ASSIST_ENABLE_IMAGE_REGIONS") or "").strip().lower()
    if not value:
        return False
    return value in {"1", "true", "yes", "on"}


def _with_ir_warning(ir: dict[str, Any], warning: str) -> dict[str, Any]:
    out = copy.deepcopy(ir)
    warnings = out.get("warnings")
    warning_list = warnings if isinstance(warnings, list) else []
    warning_list.append(str(warning))
    out["warnings"] = warning_list
    return out


class LlmProvider(ABC):
    @abstractmethod
    def analyze_layout(
        self, page_image_bytes: bytes, elements: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze page layout and return suggestions.

        Returns:
            {
              "reading_order": [element_indices],
              "table_grids": [{"bbox": [...], "rows": int, "cols": int}],
              "image_regions": [{"bbox": [x0,y0,x1,y1]}]  # in image pixel coordinates
            }
        """


def _guess_media_type(image_bytes: bytes) -> str:
    # Magic headers for common formats.
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "application/octet-stream"


def _data_uri(image_bytes: bytes) -> str:
    media_type = _guess_media_type(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model text."""

    if not text:
        return None
    # Fast path: clean JSON.
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Common: fenced code block.
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove first/last fence lines if present
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

    # Last resort: substring between first '{' and last '}'.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _validate_reading_order(order: Any, *, n: int) -> list[int] | None:
    if not isinstance(order, list):
        return None
    out: list[int] = []
    for v in order:
        if isinstance(v, bool):
            return None
        if not isinstance(v, int):
            return None
        if v < 0 or v >= n:
            return None
        out.append(v)
    if len(set(out)) != len(out):
        return None
    return out


def _validate_table_grids(grids: Any) -> list[dict[str, Any]] | None:
    if not isinstance(grids, list):
        return None
    out: list[dict[str, Any]] = []
    for g in grids:
        if not isinstance(g, dict):
            continue
        bbox = g.get("bbox")
        rows = g.get("rows")
        cols = g.get("cols")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(x, (int, float)) for x in bbox)
        ):
            continue
        if not isinstance(rows, int) or not isinstance(cols, int):
            continue
        if rows <= 0 or cols <= 0:
            continue
        out.append(
            {
                "bbox": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ],
                "rows": int(rows),
                "cols": int(cols),
            }
        )
    return out


def _validate_image_regions_px(
    regions: Any, *, width_px: int, height_px: int, max_regions: int = 12
) -> list[list[float]] | None:
    """Validate image regions returned by the model.

    The model returns bbox in image pixel coordinates (top-left origin).
    """

    if not isinstance(regions, list):
        return None

    W = int(width_px)
    H = int(height_px)
    if W <= 0 or H <= 0:
        return None

    out: list[list[float]] = []
    max_regions = max(1, int(max_regions))
    page_area = float(W * H)
    for item in regions:
        bbox = None
        if isinstance(item, dict):
            bbox = item.get("bbox")
        elif isinstance(item, (list, tuple)):
            bbox = item
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(x, (int, float)) for x in bbox)
        ):
            continue
        x0, y0, x1, y1 = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
        x0, x1 = (min(x0, x1), max(x0, x1))
        y0, y1 = (min(y0, y1), max(y0, y1))

        # Clamp to image bounds.
        x0 = max(0.0, min(x0, float(W - 1)))
        y0 = max(0.0, min(y0, float(H - 1)))
        x1 = max(0.0, min(x1, float(W)))
        y1 = max(0.0, min(y1, float(H)))
        if x1 <= x0 or y1 <= y0:
            continue

        w = x1 - x0
        h = y1 - y0
        if w < 24.0 or h < 24.0:
            continue

        area = w * h
        page_ratio = area / page_area if page_area > 0 else 0.0
        # Skip regions that are almost the whole slide (usually background).
        if page_ratio > 0.85:
            continue

        # Avoid card-size panel regions that usually represent text containers
        # rather than real image assets on slide-like pages.  Use relaxed
        # thresholds so that large screenshots / charts are preserved.
        if page_ratio >= 0.25 and max(w, h) >= 0.70 * float(max(W, H)):
            continue

        out.append([x0, y0, x1, y1])
        if len(out) >= max_regions:
            break

    return out


def _image_regions_px_to_pt(
    regions_px: list[list[float]],
    *,
    image_width_px: int,
    image_height_px: int,
    page_width_pt: float,
    page_height_pt: float,
) -> list[list[float]]:
    """Convert validated image-region bboxes from pixel space to page points."""

    out: list[list[float]] = []
    for bbox in regions_px:
        converted = bbox_px_to_pt(
            bbox,
            img_w_px=int(image_width_px),
            img_h_px=int(image_height_px),
            page_w_pt=float(page_width_pt),
            page_h_pt=float(page_height_pt),
        )
        if converted is None:
            continue
        out.append(converted)
    return out


def _build_prompt(
    *,
    page_w_pt: float | None,
    page_h_pt: float | None,
    image_width_px: int | None,
    image_height_px: int | None,
    image_dpi: int | None,
    elements: list[dict[str, Any]],
) -> str:
    w = float(page_w_pt) if isinstance(page_w_pt, (int, float)) else None
    h = float(page_h_pt) if isinstance(page_h_pt, (int, float)) else None

    # Keep element payload compact but index-addressable.
    simplified: list[dict[str, Any]] = []
    for i, el in enumerate(elements):
        if not isinstance(el, dict):
            continue
        bbox = el.get("bbox_pt")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_pt = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        else:
            bbox_pt = None

        item: dict[str, Any] = {
            "i": i,
            "type": str(el.get("type") or ""),
            "bbox_pt": bbox_pt,
        }
        if item["type"] == "text":
            txt = el.get("text")
            if isinstance(txt, str) and txt:
                item["text_preview"] = (txt[:80] + "...") if len(txt) > 80 else txt
        if item["type"] == "table":
            # If a table already exists, the model should usually not invent one.
            item["rows"] = el.get("rows")
            item["cols"] = el.get("cols")
        simplified.append(item)

    payload = {
        "page_width_pt": w,
        "page_height_pt": h,
        "image_width_px": int(image_width_px)
        if isinstance(image_width_px, int)
        else None,
        "image_height_px": int(image_height_px)
        if isinstance(image_height_px, int)
        else None,
        "image_dpi": int(image_dpi) if isinstance(image_dpi, int) else None,
        "elements": simplified,
    }

    return (
        "You are a PDF layout analysis assistant. Do NOT rewrite, summarize, or translate content. "
        "Only analyze layout.\n\n"
        "Task: Return a single JSON object with keys:\n"
        "- reading_order: a list of element indices (integers) in human reading order. Include each index at most once.\n"
        "- table_grids: a list of inferred table grids: {bbox:[x0,y0,x1,y1], rows:int, cols:int} in PDF point coordinates.\n"
        "- image_regions: a list of non-text image regions on the page: {bbox:[x0,y0,x1,y1]} in IMAGE PIXELS of the attached image "
        "(origin top-left). Include photos/screenshots/diagrams. EXCLUDE tiny icons and EXCLUDE the full-page background.\n"
        "Guidance for image_regions:\n"
        "- Return 1-12 regions when the page contains obvious pictures/diagrams/screenshots.\n"
        "- Prefer a few larger regions over many tiny ones.\n"
        "- Do NOT return the full-page background.\n"
        "- If a region contains both an image and some embedded text (e.g. a screenshot), still include it.\n"
        "If you are unsure, use an empty list for table_grids and image_regions and use the natural top-to-bottom then left-to-right order for reading_order.\n\n"
        "Input JSON:\n" + json.dumps(payload, ensure_ascii=True) + "\n\n"
        "Output ONLY the JSON object."
    )


def _render_page_png_bytes(
    source_pdf: str | Path,
    *,
    page_index: int,
    dpi: int = _DEFAULT_RENDER_DPI,
) -> bytes:
    """Render a PDF page to PNG bytes (best-effort)."""

    import importlib

    pymupdf = importlib.import_module("pymupdf")
    doc = pymupdf.open(str(source_pdf))
    try:
        page = doc.load_page(int(page_index))
        cs_rgb = getattr(pymupdf, "csRGB", None)
        try:
            if cs_rgb is not None:
                pix = page.get_pixmap(dpi=int(dpi), colorspace=cs_rgb, alpha=False)
            else:
                pix = page.get_pixmap(dpi=int(dpi), alpha=False)
        except TypeError:
            pix = page.get_pixmap(dpi=int(dpi))
        try:
            return pix.tobytes("png")
        except Exception:
            # Fallback: save to a temp file next to the PDF and read.
            tmp = Path(source_pdf).with_suffix(f".page{int(page_index):04d}.tmp.png")
            pix.save(str(tmp))
            data = tmp.read_bytes()
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return data
    finally:
        doc.close()


def _is_multi_column_ambiguous(page: dict[str, Any]) -> bool:
    elements = page.get("elements") or []
    if not isinstance(elements, list):
        return False

    w = page.get("page_width_pt")
    h = page.get("page_height_pt")
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
        return False
    w = float(w)
    h = float(h)
    if w <= 0 or h <= 0:
        return False

    text_boxes: list[list[float]] = []
    for el in elements:
        if not isinstance(el, dict) or el.get("type") != "text":
            continue
        bbox = el.get("bbox_pt")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        text_boxes.append([x0, y0, x1, y1])

    if len(text_boxes) < 8:
        return False

    left: list[list[float]] = []
    right: list[list[float]] = []
    for x0, y0, x1, y1 in text_boxes:
        xc = (x0 + x1) / 2.0
        if xc < 0.45 * w:
            left.append([x0, y0, x1, y1])
        elif xc > 0.55 * w:
            right.append([x0, y0, x1, y1])

    if len(left) < 3 or len(right) < 3:
        return False

    left_y0 = min(b[1] for b in left)
    left_y1 = max(b[3] for b in left)
    right_y0 = min(b[1] for b in right)
    right_y1 = max(b[3] for b in right)
    overlap = min(left_y1, right_y1) - max(left_y0, right_y0)
    return overlap > 0.10 * h


def _needs_table_inference(page: dict[str, Any]) -> bool:
    elements = page.get("elements") or []
    if not isinstance(elements, list):
        return False
    if any(isinstance(el, dict) and el.get("type") == "table" for el in elements):
        return False

    # Heuristic: scanned/OCR-heavy pages with many text boxes are candidates.
    has_text_layer = page.get("has_text_layer")
    scanned_like = (has_text_layer is False) or (has_text_layer is None)

    ocr_text = 0
    for el in elements:
        if not isinstance(el, dict):
            continue
        if el.get("type") != "text":
            continue
        if str(el.get("source") or "") == "ocr":
            ocr_text += 1
    if scanned_like and ocr_text >= 12:
        return True
    if (
        scanned_like
        and sum(
            1 for el in elements if isinstance(el, dict) and el.get("type") == "text"
        )
        >= 30
    ):
        return True
    return False


class OpenAiProvider(LlmProvider):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
    ):
        import openai

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)
        self.model = model or "gpt-4o-mini"
        self.use_responses = base_url is None

    def analyze_layout(
        self, page_image_bytes: bytes, elements: list[dict[str, Any]]
    ) -> dict[str, Any]:
        page_w_pt: float | None = None
        page_h_pt: float | None = None
        for el in elements:
            if not isinstance(el, dict):
                continue
            w = el.get("_page_width_pt")
            h = el.get("_page_height_pt")
            if page_w_pt is None and isinstance(w, (int, float)):
                page_w_pt = float(w)
            if page_h_pt is None and isinstance(h, (int, float)):
                page_h_pt = float(h)
            if page_w_pt is not None and page_h_pt is not None:
                break

        image_w_px: int | None = None
        image_h_px: int | None = None
        try:
            import io
            from PIL import Image

            im = Image.open(io.BytesIO(page_image_bytes))
            image_w_px, image_h_px = im.size
        except Exception:
            image_w_px, image_h_px = None, None

        prompt = _build_prompt(
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            image_width_px=image_w_px,
            image_height_px=image_h_px,
            image_dpi=_DEFAULT_RENDER_DPI,
            elements=[dict(e) for e in elements if isinstance(e, dict)],
        )
        image_uri = _data_uri(page_image_bytes)

        if self.use_responses:
            # Prefer Responses API for OpenAI. Keep parameters conservative to avoid
            # compatibility issues across SDK versions.
            response = self.client.with_options(
                timeout=_PAGE_TIMEOUT_S
            ).responses.create(
                model=self.model,
                input=[  # type: ignore[arg-type]
                    {  # pyright: ignore[reportArgumentType]
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_uri},
                        ],
                    }
                ],
            )

            parsed = _extract_json_object(getattr(response, "output_text", "") or "")
            return parsed or {
                "reading_order": [],
                "table_grids": [],
                "image_regions": [],
            }

        # OpenAI-compatible fallback: use chat.completions for broader support.
        completion = self.client.with_options(
            timeout=_PAGE_TIMEOUT_S
        ).chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=1024,
            messages=[  # type: ignore[arg-type]
                {
                    "role": "system",
                    "content": "You are a layout analysis assistant. Output JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                },
            ],
        )

        content = (
            completion.choices[0].message.content
            if getattr(completion, "choices", None)
            else ""
        )
        parsed = _extract_json_object(content or "")
        return parsed or {"reading_order": [], "table_grids": [], "image_regions": []}


class AnthropicProvider(LlmProvider):
    def __init__(self, api_key: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze_layout(
        self, page_image_bytes: bytes, elements: list[dict[str, Any]]
    ) -> dict[str, Any]:
        media_type = _guess_media_type(page_image_bytes)
        page_w_pt: float | None = None
        page_h_pt: float | None = None
        for el in elements:
            if not isinstance(el, dict):
                continue
            w = el.get("_page_width_pt")
            h = el.get("_page_height_pt")
            if page_w_pt is None and isinstance(w, (int, float)):
                page_w_pt = float(w)
            if page_h_pt is None and isinstance(h, (int, float)):
                page_h_pt = float(h)
            if page_w_pt is not None and page_h_pt is not None:
                break

        image_w_px: int | None = None
        image_h_px: int | None = None
        try:
            import io
            from PIL import Image

            im = Image.open(io.BytesIO(page_image_bytes))
            image_w_px, image_h_px = im.size
        except Exception:
            image_w_px, image_h_px = None, None

        prompt = _build_prompt(
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            image_width_px=image_w_px,
            image_height_px=image_h_px,
            image_dpi=_DEFAULT_RENDER_DPI,
            elements=[dict(e) for e in elements if isinstance(e, dict)],
        )
        b64 = base64.b64encode(page_image_bytes).decode("utf-8")

        msg = self.client.with_options(timeout=_PAGE_TIMEOUT_S).messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # pyright: ignore[reportArgumentType]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # Anthropic returns a list of content blocks.
        text_parts: list[str] = []
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_parts.append(str(block.text))
        parsed = _extract_json_object("\n".join(text_parts))
        return parsed or {"reading_order": [], "table_grids": [], "image_regions": []}


class LlmLayoutService:
    def __init__(self, provider: Optional[LlmProvider] = None):
        self.provider = provider

    def enhance_ir(
        self,
        ir: dict[str, Any],
        layout_mode: str = "fidelity",
        *,
        force_ai: bool = False,
        allow_image_regions: bool | None = None,
    ) -> dict[str, Any]:
        """Enhance IR with AI layout analysis.

        layout_mode: 'fidelity' (no AI) or 'assist' (use AI)
        """

        if layout_mode == "fidelity" or not self.provider:
            return ir  # No AI, use heuristics

        try:
            pages = ir.get("pages")
            if not isinstance(pages, list):
                return ir
            source_pdf = ir.get("source_pdf")
            if not source_pdf:
                return ir

            out: dict[str, Any] = copy.deepcopy(ir)
            out_pages = out.get("pages")
            if not isinstance(out_pages, list):
                return ir

            pages_considered = 0
            pages_changed = 0
            image_regions_enabled = (
                bool(allow_image_regions)
                if allow_image_regions is not None
                else _allow_layout_assist_image_regions()
            )
            suggested_image_region_pages = 0
            applied_image_region_pages = 0
            preserved_existing_image_region_pages = 0

            for page in out_pages:
                if not isinstance(page, dict):
                    continue
                elements = page.get("elements")
                if not isinstance(elements, list) or not elements:
                    continue

                if not force_ai:
                    needs_ai = _is_multi_column_ambiguous(
                        page
                    ) or _needs_table_inference(page)
                    if not needs_ai:
                        continue

                page_index = int(page.get("page_index") or 0)
                img_bytes = _render_page_png_bytes(source_pdf, page_index=page_index)
                img_w_px: int | None = None
                img_h_px: int | None = None
                try:
                    import io
                    from PIL import Image

                    im = Image.open(io.BytesIO(img_bytes))
                    img_w_px, img_h_px = im.size
                except Exception:
                    img_w_px, img_h_px = None, None

                llm_elements: list[dict[str, Any]] = []
                for el in elements:
                    if not isinstance(el, dict):
                        continue
                    item = dict(el)
                    item["_page_width_pt"] = page.get("page_width_pt")
                    item["_page_height_pt"] = page.get("page_height_pt")
                    llm_elements.append(item)

                suggestion = self.provider.analyze_layout(img_bytes, llm_elements)  # type: ignore[arg-type]
                if not isinstance(suggestion, dict):
                    continue

                pages_considered += 1
                page_changed = False

                ro = _validate_reading_order(
                    suggestion.get("reading_order"), n=len(elements)
                )
                tg = _validate_table_grids(suggestion.get("table_grids"))
                if ro == [] and not page.get("reading_order"):
                    ro = None
                if tg == [] and not page.get("table_grids"):
                    tg = None
                ir_image_regions: list[list[float]] | None = None
                if img_w_px and img_h_px:
                    regions_px = _validate_image_regions_px(
                        suggestion.get("image_regions"),
                        width_px=int(img_w_px),
                        height_px=int(img_h_px),
                    )
                    if regions_px is not None:
                        page_w_pt = page.get("page_width_pt")
                        page_h_pt = page.get("page_height_pt")
                        if isinstance(page_w_pt, (int, float)) and isinstance(
                            page_h_pt, (int, float)
                        ):
                            ir_image_regions = _image_regions_px_to_pt(
                                regions_px,
                                image_width_px=int(img_w_px),
                                image_height_px=int(img_h_px),
                                page_width_pt=float(page_w_pt),
                                page_height_pt=float(page_h_pt),
                            )

                if ro is not None:
                    if page.get("reading_order") != ro:
                        page_changed = True
                    page["reading_order"] = ro
                if tg is not None:
                    if page.get("table_grids") != tg:
                        page_changed = True
                    page["table_grids"] = tg
                if ir_image_regions is not None:
                    suggested_image_region_pages += 1
                    if image_regions_enabled:
                        existing_image_regions = page.get("image_regions")
                        has_existing_image_regions = (
                            isinstance(existing_image_regions, list)
                            and len(existing_image_regions) > 0
                        )
                        # Treat an empty layout-assist suggestion as "no opinion"
                        # when OCR has already provided image regions. Otherwise
                        # layout assist can wipe authoritative OCR/Paddle boxes
                        # and force PPT generation back onto heuristic crops.
                        if (not ir_image_regions) and has_existing_image_regions:
                            preserved_existing_image_region_pages += 1
                        else:
                            if page.get("image_regions") != ir_image_regions:
                                page_changed = True
                            page["image_regions"] = ir_image_regions
                            applied_image_region_pages += 1
                if page_changed:
                    pages_changed += 1

            out.setdefault("warnings", []).append(
                f"layout_assist_pages={pages_considered},changed={pages_changed}"
            )
            if suggested_image_region_pages > 0:
                if image_regions_enabled:
                    out.setdefault("warnings", []).append(
                        f"layout_assist_image_regions=applied:{applied_image_region_pages}/{suggested_image_region_pages}"
                    )
                    if preserved_existing_image_region_pages > 0:
                        out.setdefault("warnings", []).append(
                            "layout_assist_image_regions_preserved_existing="
                            f"{preserved_existing_image_region_pages}"
                        )
                else:
                    out.setdefault("warnings", []).append(
                        f"layout_assist_image_regions=ignored:{suggested_image_region_pages}"
                    )

            return out
        except Exception as e:
            # Graceful degradation
            logger.warning(f"LLM layout assist failed; falling back to fidelity: {e!s}")
            return _with_ir_warning(ir, f"layout_assist_failed:{e!s}")
