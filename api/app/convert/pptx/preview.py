"""Preview/export helpers for final slide image snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..geometry import bbox_pt_to_px as _bbox_pt_to_px_shared

from .bbox_utils import _as_path, _coerce_bbox_pt, _is_near_full_page_bbox_pt
from .color_utils import _hex_to_rgb
from .constants import _PTS_PER_INCH
from .font_utils import (
    _contains_cjk,
    _fit_font_size_pt,
    _fit_ocr_text_style,
    _wrap_text_to_width,
)
from .slide_builder import _infer_font_size_pt, _iter_page_elements
from .scanned_page import _estimate_baseline_ocr_line_height_pt, _render_pdf_page_png

_PREVIEW_FONT_CACHE: dict[tuple[int, bool], Any] = {}

def _load_preview_font(*, size_px: int, prefer_cjk: bool) -> Any:
    from PIL import ImageFont

    key = (int(max(6, size_px)), bool(prefer_cjk))
    cached = _PREVIEW_FONT_CACHE.get(key)
    if cached is not None:
        return cached

    candidates: list[str] = []
    if prefer_cjk:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )

    for path in candidates:
        try:
            font = ImageFont.truetype(path, size=key[0])
            _PREVIEW_FONT_CACHE[key] = font
            return font
        except Exception:
            continue

    font = ImageFont.load_default()
    _PREVIEW_FONT_CACHE[key] = font
    return font


def _resolve_preview_image_path(
    *,
    image_path: Any,
    artifacts_dir: Path,
) -> Path | None:
    raw = str(image_path or "").strip()
    if not raw:
        return None
    if raw.startswith("http://") or raw.startswith("https://"):
        return None
    try:
        img_path = _as_path(raw)
    except Exception:
        return None
    if not img_path.is_absolute():
        candidate = artifacts_dir / img_path
        if candidate.exists():
            img_path = candidate
    if not img_path.exists() or not img_path.is_file():
        return None
    return img_path


def _bbox_pt_to_preview_px(
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


def _export_final_preview_page_image(
    *,
    page: dict[str, Any],
    page_index: int,
    page_w_pt: float,
    page_h_pt: float,
    source_pdf: Path,
    artifacts_dir: Path,
    dpi: int,
    scanned_image_region_crops: list[tuple[list[float], Path]] | None = None,
) -> None:
    """Export a visual approximation of the converted slide for tracking UI."""

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    preview_dir = artifacts_dir / "final_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    base_candidates = [
        artifacts_dir / "page_renders" / f"page-{page_index:04d}.mineru.clean.png",
        artifacts_dir / "page_renders" / f"page-{page_index:04d}.clean.png",
        artifacts_dir / "page_renders" / f"page-{page_index:04d}.png",
        artifacts_dir / "page_renders" / f"page-{page_index:04d}.mineru.png",
    ]
    base_path = next((p for p in base_candidates if p.exists()), None)

    if base_path is None and source_pdf.exists():
        try:
            base_path = preview_dir / f"page-{page_index:04d}.base.png"
            _render_pdf_page_png(
                source_pdf,
                page_index=page_index,
                dpi=int(dpi),
                out_path=base_path,
            )
        except Exception:
            base_path = None

    if base_path is None:
        return

    try:
        img = Image.open(base_path).convert("RGB")
    except Exception:
        return

    img_w, img_h = img.size
    if img_w <= 0 or img_h <= 0:
        return

    draw = ImageDraw.Draw(img)

    scanned_crop_rects: list[tuple[int, int, int, int]] = []
    for crop_info in scanned_image_region_crops or []:
        if not isinstance(crop_info, tuple) or len(crop_info) != 2:
            continue
        bbox_pt, _ = crop_info
        rect = _bbox_pt_to_preview_px(
            bbox_pt,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            img_w_px=img_w,
            img_h_px=img_h,
        )
        if rect is None:
            continue
        scanned_crop_rects.append(rect)

    is_scanned_page = not bool(page.get("has_text_layer"))
    baseline_ocr_h_pt: float | None = None
    if is_scanned_page:
        try:
            ocr_text_elements = [
                el
                for el in _iter_page_elements(page, type_name="text")
                if str(el.get("source") or "").strip().lower() == "ocr"
            ]
            if ocr_text_elements:
                baseline_ocr_h_pt = _estimate_baseline_ocr_line_height_pt(
                    ocr_text_elements=ocr_text_elements,
                    page_w_pt=float(page_w_pt),
                )
        except Exception:
            baseline_ocr_h_pt = None

    for el in _iter_page_elements(page, type_name="image"):
        # For scanned-page OCR IR, a full-page image element is often just the
        # original rendered background. Re-pasting it on top of a cleaned base
        # preview causes apparent "double text"/ghosting.
        if _is_near_full_page_bbox_pt(
            el.get("bbox_pt"), page_w_pt=page_w_pt, page_h_pt=page_h_pt
        ):
            continue

        rect = _bbox_pt_to_preview_px(
            el.get("bbox_pt"),
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            img_w_px=img_w,
            img_h_px=img_h,
        )
        img_path = _resolve_preview_image_path(
            image_path=el.get("image_path"), artifacts_dir=artifacts_dir
        )
        if rect is None or img_path is None:
            continue
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue
        try:
            patch = Image.open(img_path).convert("RGB").resize(
                (max(1, x1 - x0), max(1, y1 - y0))
            )
            img.paste(patch, (x0, y0))
        except Exception:
            continue

    # Scanned-page overlay crops are not always present in IR `image` elements.
    # Include them explicitly in preview so users can visually verify image
    # recovery/parity with PPT composition.
    for crop_info in scanned_image_region_crops or []:
        if not isinstance(crop_info, tuple) or len(crop_info) != 2:
            continue
        bbox_pt, crop_path = crop_info
        rect = _bbox_pt_to_preview_px(
            bbox_pt,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            img_w_px=img_w,
            img_h_px=img_h,
        )
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue
        try:
            crop = Image.open(crop_path).convert("RGBA").resize(
                (max(1, x1 - x0), max(1, y1 - y0))
            )
            img.paste(crop.convert("RGB"), (x0, y0), crop)
        except Exception:
            continue

    for el in _iter_page_elements(page, type_name="text"):
        rect = _bbox_pt_to_preview_px(
            el.get("bbox_pt"),
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            img_w_px=img_w,
            img_h_px=img_h,
        )
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue

        raw_text = str(el.get("text") or "")
        source_id = str(el.get("source") or "").strip().lower()
        is_scanned_ocr = bool(is_scanned_page and source_id == "ocr")
        is_layout_text = source_id in {"mineru"} or is_scanned_ocr

        # When we overlay a scanned crop (screenshot/diagram), we also suppress
        # OCR text inside that crop in the PPT composition. Mirror that here so
        # the preview doesn't show confusing "extra" text on top of images.
        if scanned_crop_rects and source_id == "ocr":
            t_area = max(1.0, float((x1 - x0) * (y1 - y0)))
            tcx = (float(x0) + float(x1)) / 2.0
            tcy = (float(y0) + float(y1)) / 2.0
            suppressed = False
            for ix0, iy0, ix1, iy1 in scanned_crop_rects:
                inter_x0 = max(int(x0), int(ix0))
                inter_y0 = max(int(y0), int(iy0))
                inter_x1 = min(int(x1), int(ix1))
                inter_y1 = min(int(y1), int(iy1))
                if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                    continue
                inter = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
                overlap_ratio = inter / t_area
                center_inside = (
                    tcx >= float(ix0)
                    and tcx <= float(ix1)
                    and tcy >= float(iy0)
                    and tcy <= float(iy1)
                )
                if overlap_ratio >= 0.72 or (center_inside and overlap_ratio >= 0.25):
                    suppressed = True
                    break
            if suppressed:
                continue

        if is_layout_text:
            text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
            text = "\n".join([line.strip() for line in text.split("\n") if line.strip()]).strip()
        else:
            text = raw_text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
        if not text:
            continue

        if is_scanned_ocr and baseline_ocr_h_pt is not None:
            try:
                x0_pt, y0_pt, x1_pt, y1_pt = _coerce_bbox_pt(el.get("bbox_pt"))
            except Exception:
                continue
            bbox_w_pt = max(1.0, float(x1_pt - x0_pt))
            bbox_h_pt = max(1.0, float(y1_pt - y0_pt))
            is_heading = (
                y0_pt <= 0.22 * float(page_h_pt)
                and bbox_h_pt >= 1.6 * float(baseline_ocr_h_pt)
                and len(text) <= 40
            )
            text_to_render, font_size_pt, _ = _fit_ocr_text_style(
                text=text,
                bbox_w_pt=bbox_w_pt,
                bbox_h_pt=bbox_h_pt,
                baseline_ocr_h_pt=float(baseline_ocr_h_pt),
                is_heading=bool(is_heading),
            )
        else:
            bbox_w_pt = max(1.0, float(x1 - x0) * _PTS_PER_INCH / float(dpi))
            bbox_h_pt = max(1.0, float(y1 - y0) * _PTS_PER_INCH / float(dpi))
            if is_layout_text:
                font_size_pt = _fit_font_size_pt(
                    text,
                    bbox_w_pt=bbox_w_pt,
                    bbox_h_pt=bbox_h_pt,
                    wrap=True,
                    min_pt=6.0,
                    max_pt=min(24.0, max(9.0, 0.60 * bbox_h_pt)),
                    width_fit_ratio=0.98,
                    height_fit_ratio=0.95,
                )
                text_to_render = _wrap_text_to_width(
                    text,
                    max_width_pt=max(1.0, 0.98 * bbox_w_pt),
                    font_size_pt=float(font_size_pt),
                )
                text_to_render = text_to_render if text_to_render.strip() else text
            else:
                font_size_pt = _infer_font_size_pt(el, bbox_h_pt=bbox_h_pt)
                text_to_render = text

        size_px = int(round(max(7.0, float(font_size_pt)) * float(dpi) / _PTS_PER_INCH))
        font = _load_preview_font(size_px=size_px, prefer_cjk=_contains_cjk(text_to_render))
        rgb = _hex_to_rgb(el.get("color")) or (0, 0, 0)
        # For scanned OCR we already insert explicit line breaks when needed.
        # Additional PIL line spacing can drift from PPT rendering.
        spacing = 0 if is_scanned_ocr else max(1, int(round(0.18 * float(size_px))))

        try:
            # Draw into a clipped patch so preview text cannot overflow across
            # neighboring cards/areas, matching PPT textbox boundaries better.
            box_w = max(1, int(x1 - x0))
            box_h = max(1, int(y1 - y0))
            patch = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
            patch_draw = ImageDraw.Draw(patch)
            patch_draw.multiline_text(
                (0, 0),
                text_to_render,
                fill=(int(rgb[0]), int(rgb[1]), int(rgb[2]), 255),
                font=font,
                spacing=spacing,
            )
            img.paste(patch.convert("RGB"), (int(x0), int(y0)), patch)
        except Exception:
            continue

    out_path = preview_dir / f"page-{page_index:04d}.final.png"
    try:
        img.save(out_path)
    except Exception:
        return
