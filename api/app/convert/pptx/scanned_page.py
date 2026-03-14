"""Scanned-page rendering and image-region processing helpers."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...models.error import AppException, ErrorCode

from .bbox_utils import (
    _bbox_iou_pt,
    _bbox_intersection_area_pt,
    _coerce_bbox_pt,
    _compute_text_erase_padding_pt,
    _ensure_parent_dir,
    _texts_similar_for_bbox_dedupe,
)
from .color_utils import _rgb_luma, _rgb_sq_distance
from .constants import _PTS_PER_INCH
from .font_utils import (
    _compact_text_length,
    _contains_cjk,
    _fit_ocr_text_style,
    _is_inline_short_token,
    _normalize_ocr_text_for_render,
    _prefer_wrap_for_ocr_text,
)
from .slide_builder import _iter_page_elements


def _pixel_to_int(pixel: Any) -> int:
    if isinstance(pixel, tuple):
        if not pixel:
            return 0
        pixel = pixel[0]
    if pixel is None:
        return 0
    try:
        return int(pixel)
    except Exception:
        return 0


def _pixel_to_rgb_triplet(pixel: Any) -> tuple[int, int, int] | None:
    if isinstance(pixel, tuple):
        if len(pixel) >= 3:
            c0, c1, c2 = pixel[0], pixel[1], pixel[2]
        elif len(pixel) == 1:
            c0 = c1 = c2 = pixel[0]
        else:
            return None
    else:
        c0 = c1 = c2 = pixel

    if c0 is None or c1 is None or c2 is None:
        return None

    try:
        return (int(c0), int(c1), int(c2))
    except Exception:
        return None


def _apply_max_filter_l(image: Any, *, size: int) -> Any:
    """Apply an L-mode max filter, preferring a faster NumPy implementation."""

    try:
        size_id = int(size)
    except Exception:
        return image

    if size_id <= 1:
        return image
    if size_id % 2 == 0:
        size_id += 1

    try:
        import numpy as np  # type: ignore
        from PIL import Image

        if getattr(image, "mode", None) == "L":
            arr = np.asarray(image, dtype=np.uint8)
            if arr.ndim == 2 and arr.size > 0:
                radius = size_id // 2
                pad = np.pad(arr, radius, mode="edge")
                out = pad[: arr.shape[0], : arr.shape[1]].copy()
                for dy in range(size_id):
                    row_start = dy
                    row_end = dy + arr.shape[0]
                    for dx in range(size_id):
                        if dy == 0 and dx == 0:
                            continue
                        col_start = dx
                        col_end = dx + arr.shape[1]
                        np.maximum(
                            out,
                            pad[row_start:row_end, col_start:col_end],
                            out=out,
                        )
                return Image.fromarray(out, mode="L")
    except Exception:
        pass

    try:
        from PIL import ImageFilter

        return image.filter(ImageFilter.MaxFilter(size_id))
    except Exception:
        return image


def _render_pdf_page_png(
    pdf_path: Path,
    *,
    page_index: int,
    dpi: int,
    out_path: Path,
) -> Any:
    try:
        pymupdf = importlib.import_module("pymupdf")
    except Exception as e:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="PyMuPDF (pymupdf) is required for scanned-page rendering",
            details={"error": str(e)},
        )
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="Unable to open source PDF for scanned-page rendering",
            details={"path": str(pdf_path), "error": str(e)},
        )

    try:
        page = doc.load_page(int(page_index))
        _ensure_parent_dir(out_path)
        cs_rgb = getattr(pymupdf, "csRGB", None)
        try:
            if cs_rgb is not None:
                pix = page.get_pixmap(dpi=int(dpi), colorspace=cs_rgb, alpha=False)
            else:
                pix = page.get_pixmap(dpi=int(dpi), alpha=False)
        except TypeError:
            # Older/newer PyMuPDF versions may not accept colorspace/alpha arguments.
            pix = page.get_pixmap(dpi=int(dpi))
        pix.save(str(out_path))
        return pix
    except AppException:
        raise
    except Exception as e:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="Failed to render scanned PDF page to image",
            details={"path": str(pdf_path), "page_index": page_index, "error": str(e)},
        )
    finally:
        doc.close()


def _detect_image_regions_from_render(
    render_path: Path,
    *,
    page_width_pt: float,
    page_height_pt: float,
    dpi: int,
    ocr_text_elements: list[dict[str, Any]] | None = None,
    max_regions: int = 12,
    merge_gap_scale: float = 0.06,
) -> list[list[float]]:
    """Heuristically detect non-text image regions on a scanned page.

    This is a best-effort fallback when AI layout assist is disabled/unavailable.
    It tries to find "busy" visual regions (diagrams, screenshots, photos) by:
    - masking out OCR text boxes on the rendered page image
    - edge-detecting the remaining content
    - connected-component grouping of edge pixels

    Returns bboxes in *PDF point* coordinates using the IR convention (top-left
    origin, y increasing downward).
    """

    try:
        from PIL import Image, ImageDraw, ImageFilter
    except Exception:
        return []

    try:
        img = Image.open(render_path).convert("RGB")
    except Exception:
        return []

    W, H = img.size
    if W <= 0 or H <= 0:
        return []

    scale = float(dpi) / _PTS_PER_INCH  # px per pt

    # 1) Build a text mask to reduce edges caused by glyph strokes.
    #
    # NOTE: OCR engines sometimes output spurious bboxes inside icons/photos
    # (e.g. "Br" inside a logo). If we mask those boxes we may erase the very
    # edges we need to detect the image region. We therefore ignore OCR boxes
    # that look abnormally tall/small relative to a baseline line height.
    mask = Image.new("L", (W, H), 0)
    # Keep the concrete pixel rectangles we masked so we can also "paint out"
    # text in the RGB image using a *local* background color. This avoids
    # creating hard-edged rectangles when the slide background is non-uniform
    # (gradients / cards), which otherwise confuses the edge detector.
    masked_regions_px: list[
        tuple[tuple[int, int, int, int], list[tuple[int, int]] | None]
    ] = []
    if ocr_text_elements:
        baseline_h_pt = _estimate_baseline_ocr_line_height_pt(
            ocr_text_elements=ocr_text_elements,
            page_w_pt=float(page_width_pt),
        )

        draw = ImageDraw.Draw(mask)
        for el in ocr_text_elements:
            bbox_pt = el.get("bbox_pt")
            try:
                x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
            except Exception:
                continue

            w_pt = max(1.0, float(x1 - x0))
            h_pt = max(1.0, float(y1 - y0))
            if h_pt < (0.35 * baseline_h_pt):
                continue
            width_ratio = w_pt / max(1.0, float(page_width_pt))
            # Better OCR models often detect lots of small UI lines inside
            # screenshots/diagrams. Masking those bboxes erases the edges we
            # need to detect the screenshot region, causing the region to be
            # split into multiple fragments. Skip narrow+short boxes so we
            # mostly mask real slide text (wider and/or taller).
            if width_ratio < 0.18 and h_pt < (0.78 * baseline_h_pt):
                continue
            if h_pt > (2.8 * baseline_h_pt):
                # Keep wide headings (real text), but ignore tall+narrow boxes
                # which are often false positives inside icons/photos.
                if w_pt < (3.2 * h_pt):
                    continue
            # Expand a bit to cover anti-aliased edges around characters.
            pad_pt = max(1.0, min(5.0, 0.14 * h_pt))
            x0p = int(round((x0 - pad_pt) * scale))
            y0p = int(round((y0 - pad_pt) * scale))
            x1p = int(round((x1 + pad_pt) * scale))
            y1p = int(round((y1 + pad_pt) * scale))

            x0p = max(0, min(W - 1, x0p))
            y0p = max(0, min(H - 1, y0p))
            x1p = max(0, min(W, x1p))
            y1p = max(0, min(H, y1p))
            if x1p <= x0p or y1p <= y0p:
                continue

            polygon_px = _element_polygon_points_px(
                el,
                page_height_pt=page_height_pt,
                dpi=int(dpi),
                width_px=W,
                height_px=H,
            )
            if polygon_px is not None:
                draw.polygon(polygon_px, fill=255)
            else:
                draw.rectangle([x0p, y0p, x1p, y1p], fill=255)
            masked_regions_px.append(((x0p, y0p, x1p, y1p), polygon_px))

        # Dilate the mask a bit to cover edge halos.
        mask = _apply_max_filter_l(mask, size=5)

    def _median_rgb(samples: list[tuple[int, int, int]]) -> tuple[int, int, int]:
        if not samples:
            return (255, 255, 255)
        rs = sorted(int(s[0]) for s in samples)
        gs = sorted(int(s[1]) for s in samples)
        bs = sorted(int(s[2]) for s in samples)
        mid = len(rs) // 2
        return (rs[mid], gs[mid], bs[mid])

    def _sample_local_bg_rgb(
        source: Image.Image, *, x0: int, y0: int, x1: int, y1: int
    ) -> tuple[int, int, int]:
        # Sample just outside the bbox so we don't hit glyph pixels.
        pad = 4
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        pts = [
            (x0 - pad, y0 - pad),
            (x1 + pad, y0 - pad),
            (x0 - pad, y1 + pad),
            (x1 + pad, y1 + pad),
            (x0 - pad, cy),
            (x1 + pad, cy),
            (cx, y0 - pad),
            (cx, y1 + pad),
        ]
        cols: list[tuple[int, int, int]] = []
        for px, py in pts:
            px = max(0, min(int(px), int(W - 1)))
            py = max(0, min(int(py), int(H - 1)))
            try:
                rgb = _pixel_to_rgb_triplet(source.getpixel((px, py)))
                if rgb is None:
                    continue
                cols.append(rgb)
            except Exception:
                continue
        return _median_rgb(cols)

    if masked_regions_px and mask.getbbox():
        try:
            masked_img = img.copy()
            draw_masked = ImageDraw.Draw(masked_img)
            for (x0p, y0p, x1p, y1p), polygon_px in masked_regions_px:
                bg = _sample_local_bg_rgb(img, x0=x0p, y0=y0p, x1=x1p, y1=y1p)
                if polygon_px is not None:
                    draw_masked.polygon(polygon_px, fill=bg)
                else:
                    draw_masked.rectangle([x0p, y0p, x1p, y1p], fill=bg)
            # A tiny blur helps hide hard boundaries of painted regions.
            try:
                masked_img = masked_img.filter(ImageFilter.BoxBlur(0.6))
            except Exception:
                pass
            img = masked_img
        except Exception:
            # If this fails for any reason, fall back to using the original image.
            pass

    # 2) Edge-detect + threshold.
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    # Threshold chosen empirically for rendered PDF pages (antialiasing present).
    # Slightly lower improves recall for screenshots with soft drop-shadows.
    threshold = 32
    bw = edges.point(lambda p: 255 if p > threshold else 0, "L")  # type: ignore[reportOperatorIssue]
    # Thicken edges to connect disjoint strokes belonging to the same image.
    bw = _apply_max_filter_l(bw, size=5)

    # 3) Connected components on a downsampled binary image.
    factor = 8 if max(W, H) >= 3000 else (6 if max(W, H) >= 1600 else 4)
    SW = max(1, W // factor)
    SH = max(1, H // factor)
    small = bw.resize((SW, SH), Image.Resampling.NEAREST)  # type: ignore[reportAttributeAccessIssue]
    px = small.load()
    if px is None:
        return []

    visited: list[bytearray] = [bytearray(SW) for _ in range(SH)]
    comps: list[
        tuple[int, float, tuple[int, int, int, int]]
    ] = []  # (area, density, bbox)
    page_area = float(SW * SH)

    for y in range(SH):
        row = visited[y]
        for x in range(SW):
            if row[x]:
                continue
            pxy_v = _pixel_to_int(px[x, y])
            if pxy_v == 0:
                continue
            # BFS over 4-neighborhood.
            q: list[tuple[int, int]] = [(x, y)]
            row[x] = 1
            minx = maxx = x
            miny = maxy = y
            count = 0
            while q:
                cx, cy = q.pop()
                count += 1
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                nx = cx - 1
                if nx >= 0 and not visited[cy][nx]:
                    pn_v = _pixel_to_int(px[nx, cy])
                    if pn_v != 0:
                        visited[cy][nx] = 1
                        q.append((nx, cy))
                nx = cx + 1
                if nx < SW and not visited[cy][nx]:
                    pn_v = _pixel_to_int(px[nx, cy])
                    if pn_v != 0:
                        visited[cy][nx] = 1
                        q.append((nx, cy))
                ny = cy - 1
                if ny >= 0 and not visited[ny][cx]:
                    pn_v = _pixel_to_int(px[cx, ny])
                    if pn_v != 0:
                        visited[ny][cx] = 1
                        q.append((cx, ny))
                ny = cy + 1
                if ny < SH and not visited[ny][cx]:
                    pn_v = _pixel_to_int(px[cx, ny])
                    if pn_v != 0:
                        visited[ny][cx] = 1
                        q.append((cx, ny))

            w = maxx - minx + 1
            h = maxy - miny + 1
            area = int(w * h)
            if area <= 0:
                continue
            density = float(count) / float(area)
            # Store bbox in small-image coords as [x0,y0,x1,y1) (exclusive max).
            comps.append((area, density, (minx, miny, maxx + 1, maxy + 1)))

    # Filter candidates.
    min_area = max(80, int(0.0012 * page_area))
    candidates: list[tuple[int, float, tuple[int, int, int, int]]] = []
    for area, density, (x0, y0, x1, y1) in comps:
        if area < min_area:
            continue
        if page_area > 0 and (float(area) / page_area) > 0.60:
            continue
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            continue
        # Discard extremely thin components (likely borders/lines).
        if (w >= 12 and h <= 2) or (h >= 12 and w <= 2):
            continue
        if w > 16 * h or h > 16 * w:
            continue
        # Screenshots often have scattered edges (low density) but are still
        # visually important. Lowering this cutoff improves recall; additional
        # size/shape filtering happens later.
        if density < 0.04:
            continue
        candidates.append((area, density, (x0, y0, x1, y1)))

    # Prefer larger regions.
    candidates.sort(key=lambda t: t[0], reverse=True)

    def _merge_boxes(
        boxes: list[tuple[int, int, int, int]],
        *,
        iou_thresh: float = 0.18,
        gap: int = 6,
    ) -> list[tuple[int, int, int, int]]:
        merged: list[tuple[int, int, int, int]] = []
        for b in boxes:
            bx0, by0, bx1, by1 = b
            did_merge = False
            for i, a in enumerate(merged):
                ax0, ay0, ax1, ay1 = a
                # Expand by a small gap so near-touching regions merge.
                ax0g, ay0g, ax1g, ay1g = ax0 - gap, ay0 - gap, ax1 + gap, ay1 + gap
                inter_x0 = max(ax0g, bx0)
                inter_y0 = max(ay0g, by0)
                inter_x1 = min(ax1g, bx1)
                inter_y1 = min(ay1g, by1)
                if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                    continue
                inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
                area_b = max(1, (bx1 - bx0) * (by1 - by0))
                union = area_a + area_b - inter
                iou = float(inter) / float(max(1, union))
                if iou >= iou_thresh or inter >= 0.45 * float(min(area_a, area_b)):
                    merged[i] = (
                        min(ax0, bx0),
                        min(ay0, by0),
                        max(ax1, bx1),
                        max(ay1, by1),
                    )
                    did_merge = True
                    break
            if not did_merge:
                merged.append((bx0, by0, bx1, by1))
        return merged

    # Convert candidate boxes from small coords to pt coords.
    boxes_small = [bbox for _, _, bbox in candidates[: max_regions * 3]]
    # Screenshots often yield multiple disjoint edge components (text blocks,
    # icons, UI chrome) that don't strictly overlap. Use a larger merge gap on
    # the downsampled grid so we can recover a single screenshot bbox.
    merge_gap_scale = float(merge_gap_scale)
    merge_gap_scale = max(0.02, min(0.25, merge_gap_scale))
    merge_gap = max(6, int(round(merge_gap_scale * float(min(SW, SH)))))
    boxes_small = _merge_boxes(boxes_small, gap=merge_gap)

    regions_pt: list[list[float]] = []
    for x0, y0, x1, y1 in boxes_small[:max_regions]:
        # Convert to pixel coordinates on full-size render.
        px0 = int(x0 * factor)
        py0 = int(y0 * factor)
        px1 = int(min(W, x1 * factor))
        py1 = int(min(H, y1 * factor))
        if px1 <= px0 or py1 <= py0:
            continue

        # Pad slightly to include soft shadows/anti-aliasing.
        pad = int(round(0.03 * float(min(px1 - px0, py1 - py0))))
        pad = max(3, min(24, pad))
        px0 = max(0, px0 - pad)
        py0 = max(0, py0 - pad)
        px1 = min(W, px1 + pad)
        py1 = min(H, py1 + pad)

        x0_pt = float(px0) / scale
        y0_pt = float(py0) / scale
        x1_pt = float(px1) / scale
        y1_pt = float(py1) / scale

        # Clamp to page bounds in pt.
        x0_pt = max(0.0, min(float(page_width_pt), x0_pt))
        y0_pt = max(0.0, min(float(page_height_pt), y0_pt))
        x1_pt = max(0.0, min(float(page_width_pt), x1_pt))
        y1_pt = max(0.0, min(float(page_height_pt), y1_pt))
        if x1_pt <= x0_pt or y1_pt <= y0_pt:
            continue

        # Skip near-full-page regions (usually background).
        area_pt = (x1_pt - x0_pt) * (y1_pt - y0_pt)
        if area_pt / max(1.0, float(page_width_pt) * float(page_height_pt)) > 0.80:
            continue

        regions_pt.append([x0_pt, y0_pt, x1_pt, y1_pt])

    # De-duplicate nearly identical bboxes.
    uniq: list[list[float]] = []
    for bb in regions_pt:
        x0, y0, x1, y1 = bb
        keep = True
        for ub in uniq:
            ux0, uy0, ux1, uy1 = ub
            if (
                abs(x0 - ux0) <= 2.0
                and abs(y0 - uy0) <= 2.0
                and abs(x1 - ux1) <= 2.0
                and abs(y1 - uy1) <= 2.0
            ):
                keep = False
                break
        if keep:
            uniq.append(bb)
    return uniq[:max_regions]


def _analyze_shape_crop(crop_path: Path) -> dict[str, Any]:
    """Return best-effort "image-likeness" stats for a rendered crop.

    This is a lightweight *visual* heuristic that helps answer:
    - does this crop look like a real screenshot/diagram/icon?
    - is it likely a text-only panel/strip?

    It intentionally avoids any extra model calls (VLM/LLM) so it stays cheap
    and works offline. The output is used as an internal quality signal for
    merging fragmented image regions.
    """

    try:
        from PIL import Image, ImageFilter
    except Exception:
        return {"confirmed": False, "score": 0.0}

    try:
        img = Image.open(crop_path).convert("L")
    except Exception:
        return {"confirmed": False, "score": 0.0}

    w, h = img.size
    if w < 18 or h < 18:
        return {"confirmed": False, "score": 0.0, "w": int(w), "h": int(h)}

    # Normalize size for stable thresholds.
    max_side = max(w, h)
    if max_side > 320:
        scale = 320.0 / float(max_side)
        w2 = max(16, int(round(float(w) * scale)))
        h2 = max(16, int(round(float(h) * scale)))
        img = img.resize((w2, h2))
        w, h = img.size

    edges = img.filter(ImageFilter.FIND_EDGES)
    bw = edges.point(lambda p: 255 if p > 34 else 0, "L")  # type: ignore[reportOperatorIssue]
    pix = bw.load()

    if pix is None or w <= 0 or h <= 0:
        return {"confirmed": False, "score": 0.0, "w": int(w), "h": int(h)}

    band = max(2, min(7, int(round(0.03 * float(min(w, h))))))

    def _edge_ratio_rect(x0: int, y0: int, x1: int, y1: int) -> float:
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        total = max(1, (x1 - x0) * (y1 - y0))
        on = 0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                pxy_v = _pixel_to_int(pix[xx, yy])
                if pxy_v > 0:
                    on += 1
        return float(on) / float(total)

    top_r = _edge_ratio_rect(0, 0, w, band)
    bottom_r = _edge_ratio_rect(0, h - band, w, h)
    left_r = _edge_ratio_rect(0, 0, band, h)
    right_r = _edge_ratio_rect(w - band, 0, w, h)

    border_side_hits = sum(1 for r in (top_r, bottom_r, left_r, right_r) if r >= 0.06)

    inset = max(2 * band, int(round(0.10 * float(min(w, h)))))
    interior_r = _edge_ratio_rect(inset, inset, w - inset, h - inset)

    has_h_pair = top_r >= 0.07 and bottom_r >= 0.07
    has_v_pair = left_r >= 0.07 and right_r >= 0.07
    has_frame = has_h_pair or has_v_pair

    aspect = max(float(w) / max(1.0, float(h)), float(h) / max(1.0, float(w)))
    icon_like = aspect <= 1.8 and interior_r >= 0.075 and (w * h) >= 1200
    screenshot_like = (
        (w * h) >= 8500
        and aspect <= 3.8
        and interior_r >= 0.032
        and border_side_hits >= 1
    )

    confirmed = False
    if has_frame and border_side_hits >= 2 and interior_r >= 0.010:
        confirmed = True
    elif screenshot_like:
        confirmed = True
    elif icon_like and border_side_hits >= 1:
        confirmed = True

    # Soft score: in [0..1], higher means "more likely a real image crop".
    border_avg = (top_r + bottom_r + left_r + right_r) / 4.0
    border_strength = min(1.0, float(border_avg) / 0.10)
    interior_strength = min(1.0, float(interior_r) / 0.06)
    score = 0.55 * interior_strength + 0.35 * border_strength
    if has_frame:
        score += 0.08
    if screenshot_like:
        score += 0.08
    if icon_like:
        score += 0.05
    score = max(0.0, min(1.0, float(score)))

    return {
        "confirmed": bool(confirmed),
        "score": float(score),
        "w": int(w),
        "h": int(h),
        "aspect": float(aspect),
        "border_side_hits": int(border_side_hits),
        "top_r": float(top_r),
        "bottom_r": float(bottom_r),
        "left_r": float(left_r),
        "right_r": float(right_r),
        "interior_r": float(interior_r),
        "has_frame": bool(has_frame),
        "icon_like": bool(icon_like),
        "screenshot_like": bool(screenshot_like),
    }


def _is_shape_confirmed_crop(crop_path: Path) -> bool:
    """Best-effort check whether a crop looks like a real image/diagram region.

    We treat regions with clear rectangular edges and non-trivial interior
    structure as "confirmed image". This helps suppress OCR edits *inside*
    screenshots/diagrams while avoiding false positives on plain text blocks.
    """

    try:
        return bool(_analyze_shape_crop(crop_path).get("confirmed"))
    except Exception:
        return False


def _sample_pixmap_rgb(
    pix: Any,
    *,
    x_px: int,
    y_px: int,
) -> tuple[int, int, int]:
    x = max(0, min(int(x_px), int(pix.width) - 1))
    y = max(0, min(int(y_px), int(pix.height) - 1))

    n = int(getattr(pix, "n", 0) or 0)
    if n <= 0:
        return (255, 255, 255)
    samples = pix.samples
    idx = (y * int(pix.width) + x) * n
    if idx + 1 >= len(samples):
        return (255, 255, 255)

    if n == 1:
        v = samples[idx]
        return (v, v, v)
    if n >= 3 and idx + 2 < len(samples):
        return (samples[idx], samples[idx + 1], samples[idx + 2])
    v = samples[idx]
    return (v, v, v)


_PIX_RGB_ARRAY_CACHE: dict[int, tuple[int, int, int, Any]] = {}


def _pix_to_rgb_array(pix: Any) -> Any | None:
    """Return cached HxWx3 uint8 array for a PyMuPDF pixmap."""

    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    try:
        w = int(getattr(pix, "width", 0) or 0)
        h = int(getattr(pix, "height", 0) or 0)
        n = int(getattr(pix, "n", 0) or 0)
    except Exception:
        return None

    if w <= 0 or h <= 0 or n <= 0:
        return None

    cache_key = id(pix)
    cached = _PIX_RGB_ARRAY_CACHE.get(cache_key)
    if cached is not None:
        cw, ch, cn, carr = cached
        if cw == w and ch == h and cn == n:
            return carr

    try:
        raw = np.frombuffer(pix.samples, dtype=np.uint8)
        expected = int(w) * int(h) * int(n)
        if raw.size < expected:
            return None
        arr = raw[:expected].reshape((h, w, n))
        if n == 1:
            rgb = np.repeat(arr[:, :, :1], 3, axis=2)
        else:
            rgb = arr[:, :, :3]
        _PIX_RGB_ARRAY_CACHE[cache_key] = (w, h, n, rgb)
        if len(_PIX_RGB_ARRAY_CACHE) > 24:
            _PIX_RGB_ARRAY_CACHE.clear()
        return rgb
    except Exception:
        return None


def _sample_bbox_background_rgb(
    pix: Any,
    *,
    bbox_pt: Any,
    page_height_pt: float,
    dpi: int,
) -> tuple[int, int, int]:
    """Best-effort background color sampling for a text bbox.

    Sampling the bbox center can hit foreground glyph pixels (dark text / white
    text), producing obvious masking artifacts. Instead sample just outside the
    bbox and average.
    """

    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
    except Exception:
        return (255, 255, 255)

    h = max(1.0, y1 - y0)
    pad_pt = max(1.0, min(3.0, 0.1 * h))

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    sample_pts = [
        (x0 - pad_pt, y0 - pad_pt),
        (x1 + pad_pt, y0 - pad_pt),
        (x0 - pad_pt, y1 + pad_pt),
        (x1 + pad_pt, y1 + pad_pt),
        (x0 - pad_pt, cy),
        (x1 + pad_pt, cy),
        (cx, y0 - pad_pt),
        (cx, y1 + pad_pt),
    ]

    colors: list[tuple[int, int, int]] = []
    for px_pt, py_pt in sample_pts:
        px, py = _pdf_pt_to_pix_px(
            float(px_pt),
            float(py_pt),
            page_height_pt=page_height_pt,
            dpi=int(dpi),
        )
        colors.append(_sample_pixmap_rgb(pix, x_px=px, y_px=py))

    if not colors:
        return (255, 255, 255)
    # Median is more robust than mean when one of the sample points hits a glyph
    # stroke or a nearby colorful element.
    rs = sorted(c[0] for c in colors)
    gs = sorted(c[1] for c in colors)
    bs = sorted(c[2] for c in colors)
    mid = len(rs) // 2
    r = int(rs[mid])
    g = int(gs[mid])
    b = int(bs[mid])
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _sample_bbox_text_rgb(
    pix: Any,
    *,
    bbox_pt: Any,
    page_height_pt: float,
    dpi: int,
    bg_rgb: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    """Estimate text color inside a bbox by selecting high-contrast pixels."""

    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
    except Exception:
        return None

    x0p, y0p = _pdf_pt_to_pix_px(
        float(x0), float(y0), page_height_pt=page_height_pt, dpi=int(dpi)
    )
    x1p, y1p = _pdf_pt_to_pix_px(
        float(x1), float(y1), page_height_pt=page_height_pt, dpi=int(dpi)
    )
    left = max(0, min(int(x0p), int(x1p)))
    right = max(0, max(int(x0p), int(x1p)))
    top = max(0, min(int(y0p), int(y1p)))
    bottom = max(0, max(int(y0p), int(y1p)))

    width = max(0, right - left)
    height = max(0, bottom - top)
    if width < 2 or height < 2:
        return None

    max_samples = 1600
    area = max(1, width * height)
    step = max(1, int(round((float(area) / float(max_samples)) ** 0.5)))

    rgb_arr = _pix_to_rgb_array(pix)
    if rgb_arr is not None:
        try:
            import numpy as np  # type: ignore

            ys = np.arange(top, bottom, step, dtype=np.int32)
            xs = np.arange(left, right, step, dtype=np.int32)
            if ys.size >= 1 and xs.size >= 1:
                yy, xx = np.meshgrid(ys, xs, indexing="ij")
                sampled = rgb_arr[yy, xx]
                sampled_flat = sampled.reshape(-1, 3).astype(np.float32, copy=False)

                bg_luma = float(_rgb_luma(bg_rgb))
                luma = (
                    0.299 * sampled_flat[:, 0]
                    + 0.587 * sampled_flat[:, 1]
                    + 0.114 * sampled_flat[:, 2]
                )
                contrast = np.abs(luma - float(bg_luma))
                keep = contrast >= 14.0
                if int(np.count_nonzero(keep)) >= 6:
                    kept_rgb = sampled_flat[keep]
                    kept_contrast = contrast[keep]
                    top_k = max(6, int(round(0.25 * kept_rgb.shape[0])))
                    if kept_rgb.shape[0] > top_k:
                        idx = np.argpartition(kept_contrast, -top_k)[-top_k:]
                        selected = kept_rgb[idx]
                    else:
                        selected = kept_rgb

                    med = np.median(selected, axis=0)
                    estimated = (
                        int(max(0, min(255, round(float(med[0]))))),
                        int(max(0, min(255, round(float(med[1]))))),
                        int(max(0, min(255, round(float(med[2]))))),
                    )
                    if _rgb_sq_distance(estimated, bg_rgb) < (24 * 24):
                        return None
                    return estimated
        except Exception:
            pass

    bg_luma = _rgb_luma(bg_rgb)
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for yp in range(top, bottom, step):
        for xp in range(left, right, step):
            rgb = _sample_pixmap_rgb(pix, x_px=int(xp), y_px=int(yp))
            luma = _rgb_luma(rgb)
            contrast = abs(float(luma) - float(bg_luma))
            if contrast >= 14.0:
                candidates.append((contrast, rgb))

    if len(candidates) < 6:
        return None

    candidates.sort(key=lambda row: row[0], reverse=True)
    top_k = max(6, int(round(0.25 * len(candidates))))
    selected = [rgb for _, rgb in candidates[:top_k]]
    rs = sorted(int(c[0]) for c in selected)
    gs = sorted(int(c[1]) for c in selected)
    bs = sorted(int(c[2]) for c in selected)
    mid = len(rs) // 2
    estimated = (int(rs[mid]), int(gs[mid]), int(bs[mid]))
    if _rgb_sq_distance(estimated, bg_rgb) < (24 * 24):
        return None
    return estimated


def _estimate_bbox_ink_line_count(
    pix: Any,
    *,
    bbox_pt: Any,
    page_height_pt: float,
    dpi: int,
    max_lines: int = 3,
) -> int | None:
    """Estimate visible text line count in a bbox from source-page pixels.

    This is a lightweight visual signal used by OCR text rendering to choose
    single-line vs wrapped layout when AI/heuristic split metadata is absent.
    """

    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
    except Exception:
        return None

    x0p, y0p = _pdf_pt_to_pix_px(
        float(x0), float(y0), page_height_pt=page_height_pt, dpi=int(dpi)
    )
    x1p, y1p = _pdf_pt_to_pix_px(
        float(x1), float(y1), page_height_pt=page_height_pt, dpi=int(dpi)
    )
    left = max(0, min(int(x0p), int(x1p)))
    right = max(0, max(int(x0p), int(x1p)))
    top = max(0, min(int(y0p), int(y1p)))
    bottom = max(0, max(int(y0p), int(y1p)))

    width = max(0, right - left)
    height = max(0, bottom - top)
    if width < 8 or height < 8:
        return 1

    rgb_arr = _pix_to_rgb_array(pix)
    if rgb_arr is None:
        return None

    try:
        patch = rgb_arr[top:bottom, left:right]
        if patch.size <= 0:
            return 1

        gray = (
            0.299 * patch[:, :, 0].astype(np.float32)
            + 0.587 * patch[:, :, 1].astype(np.float32)
            + 0.114 * patch[:, :, 2].astype(np.float32)
        )

        bg = float(np.percentile(gray, 92.0))
        threshold = max(0.0, bg - 18.0)
        ink = gray < threshold
        if float(np.mean(ink)) < 0.004:
            return 1

        row_density = np.mean(ink, axis=1)
        if row_density.size < 3:
            return 1

        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        row_density = np.convolve(row_density, kernel, mode="same")

        active_th = float(np.percentile(row_density, 72.0)) * 0.58
        active_th = max(0.018, min(0.22, active_th))
        active = row_density >= active_th

        runs = 0
        run_len = 0
        min_run = max(2, int(round(0.015 * float(height))))
        for flag in active.tolist():
            if flag:
                run_len += 1
            else:
                if run_len >= min_run:
                    runs += 1
                run_len = 0
        if run_len >= min_run:
            runs += 1

        if runs <= 0:
            return 1
        return max(1, min(int(max_lines), int(runs)))
    except Exception:
        return None


def _pdf_pt_to_pix_px(
    x_pt: float,
    y_pt: float,
    *,
    page_height_pt: float,
    dpi: int,
) -> tuple[int, int]:
    # IR coordinates and rendered pixmaps both use a top-left origin.
    x_px = x_pt * dpi / _PTS_PER_INCH
    y_px = y_pt * dpi / _PTS_PER_INCH
    return (int(round(x_px)), int(round(y_px)))


def _coerce_polygon_points_pt(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, (list, tuple)):
        return None
    points: list[tuple[float, float]] = []
    for raw_point in value:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
            return None
        try:
            x = float(raw_point[0])
            y = float(raw_point[1])
        except Exception:
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append((x, y))
    if len(points) < 3:
        return None
    return points


def _polygon_points_pt_to_px(
    value: Any,
    *,
    page_height_pt: float,
    dpi: int,
    width_px: int,
    height_px: int,
) -> list[tuple[int, int]] | None:
    points_pt = _coerce_polygon_points_pt(value)
    if points_pt is None or width_px <= 0 or height_px <= 0:
        return None

    converted: list[tuple[int, int]] = []
    for x_pt, y_pt in points_pt:
        x_px, y_px = _pdf_pt_to_pix_px(
            x_pt,
            y_pt,
            page_height_pt=page_height_pt,
            dpi=int(dpi),
        )
        coord = (
            max(0, min(int(width_px - 1), int(x_px))),
            max(0, min(int(height_px - 1), int(y_px))),
        )
        if converted and coord == converted[-1]:
            continue
        converted.append(coord)
    if len({point for point in converted}) < 3:
        return None
    return converted


def _element_polygon_points_px(
    element: dict[str, Any],
    *,
    page_height_pt: float,
    dpi: int,
    width_px: int,
    height_px: int,
) -> list[tuple[int, int]] | None:
    if bool(element.get("ocr_linebreak_assisted")):
        return None
    if str(element.get("ocr_layout_geometry_kind") or "").strip().lower() != "polygon":
        return None
    return _polygon_points_pt_to_px(
        element.get("ocr_layout_geometry_points_pt"),
        page_height_pt=page_height_pt,
        dpi=int(dpi),
        width_px=width_px,
        height_px=height_px,
    )


def _coerce_image_region_entry_pt(value: Any) -> dict[str, Any] | None:
    raw_bbox = value.get("bbox_pt") if isinstance(value, dict) else value
    if raw_bbox is None and isinstance(value, dict):
        raw_bbox = value.get("bbox")
    try:
        bbox_pt = [float(v) for v in _coerce_bbox_pt(raw_bbox)]
    except Exception:
        return None

    out: dict[str, Any] = {"bbox_pt": bbox_pt}
    if not isinstance(value, dict):
        return out

    for key in ("label", "score", "order", "geometry_source"):
        if value.get(key) is not None:
            out[key] = value.get(key)

    geometry_kind = str(value.get("geometry_kind") or "").strip().lower()
    raw_points = value.get("geometry_points_pt")
    if raw_points is None:
        raw_points = value.get("geometry_points")
    points = _coerce_polygon_points_pt(raw_points)
    if points is not None:
        out["geometry_kind"] = "polygon"
        out["geometry_points_pt"] = [[float(x), float(y)] for x, y in points]
    elif geometry_kind:
        out["geometry_kind"] = geometry_kind

    return out


def _erase_regions_in_render_image(
    render_path: Path,
    *,
    out_path: Path,
    erase_bboxes_pt: list[list[float]],
    erase_polygons_pt: list[list[list[float]] | None] | None = None,
    protect_bboxes_pt: list[list[float]] | None = None,
    page_height_pt: float,
    dpi: int,
    text_erase_mode: str = "fill",
) -> Path:
    """Erase bboxes directly in the rendered background image.

    This avoids PPT rectangle masks (which can look like color blocks) and
    produces a cleaner editable overlay: erase first, then place text boxes.
    """

    if not erase_bboxes_pt:
        return render_path

    try:
        from PIL import Image, ImageChops, ImageDraw, ImageFilter
    except Exception:
        return render_path

    try:
        img = Image.open(render_path).convert("RGB")
    except Exception:
        return render_path

    W, H = img.size
    if W <= 0 or H <= 0:
        return render_path

    def _bbox_pt_to_rect_px(
        bb: list[float], *, pad: int = 0
    ) -> tuple[int, int, int, int] | None:
        try:
            x0, y0, x1, y1 = _coerce_bbox_pt(bb)
        except Exception:
            return None
        x0p, y0p = _pdf_pt_to_pix_px(
            x0, y0, page_height_pt=page_height_pt, dpi=int(dpi)
        )
        x1p, y1p = _pdf_pt_to_pix_px(
            x1, y1, page_height_pt=page_height_pt, dpi=int(dpi)
        )
        x0p = max(0, min(int(W - 1), int(x0p) - int(pad)))
        y0p = max(0, min(int(H - 1), int(y0p) - int(pad)))
        x1p = max(0, min(int(W), int(x1p) + int(pad)))
        y1p = max(0, min(int(H), int(y1p) + int(pad)))
        if x1p <= x0p or y1p <= y0p:
            return None
        return (x0p, y0p, x1p, y1p)

    rects: list[tuple[int, int, int, int]] = []
    core_rects: list[tuple[int, int, int, int]] = []
    polygon_masks_px: list[list[tuple[int, int]] | None] = []
    raw_polygons = erase_polygons_pt or []
    for index, bb in enumerate(erase_bboxes_pt):
        core = _bbox_pt_to_rect_px(bb, pad=0)
        if core is None:
            continue
        expanded = _bbox_pt_to_rect_px(bb, pad=1)
        if expanded is None:
            expanded = core
        rects.append(expanded)
        core_rects.append(core)
        polygon_masks_px.append(
            _polygon_points_pt_to_px(
                raw_polygons[index] if index < len(raw_polygons) else None,
                page_height_pt=page_height_pt,
                dpi=int(dpi),
                width_px=W,
                height_px=H,
            )
        )

    if not rects:
        return render_path

    protect_rects: list[tuple[int, int, int, int]] = []
    for bb in protect_bboxes_pt or []:
        rect = _bbox_pt_to_rect_px(bb, pad=2)
        if rect is not None:
            protect_rects.append(rect)

    erase_mode = str(text_erase_mode or "smart").strip().lower()
    if erase_mode not in {"smart", "fill"}:
        erase_mode = "smart"

    if erase_mode == "fill":
        dilate_size = 5 if max(W, H) >= 1600 else 3
        dilate_pad = max(1, int(dilate_size // 2) + 1)

        def _point_in_protect(x: int, y: int) -> bool:
            for px0, py0, px1, py1 in protect_rects:
                if px0 <= x < px1 and py0 <= y < py1:
                    return True
            return False

        def _median_color(values: list[tuple[int, int, int]]) -> tuple[int, int, int]:
            if not values:
                return (255, 255, 255)
            rs = sorted(v[0] for v in values)
            gs = sorted(v[1] for v in values)
            bs = sorted(v[2] for v in values)
            mid = len(values) // 2
            return (int(rs[mid]), int(gs[mid]), int(bs[mid]))

        def _estimate_fill_color(
            x0: int, y0: int, x1: int, y1: int
        ) -> tuple[int, int, int]:
            h = max(1, int(y1 - y0))
            w = max(1, int(x1 - x0))
            # Keep sampling close to the text bbox; too-large pads can pull colors
            # from unrelated nearby cards/charts and create obvious fill blocks.
            pad = max(1, min(8, int(round(0.28 * float(h)))))

            sample_points: list[tuple[int, int]] = []
            x_fracs = [0.15, 0.35, 0.50, 0.65, 0.85]
            y_fracs = [0.15, 0.35, 0.50, 0.65, 0.85]
            for frac in x_fracs:
                px = int(round(x0 + frac * float(w)))
                sample_points.append((px, y0 - pad))
                sample_points.append((px, y1 + pad))
            for frac in y_fracs:
                py = int(round(y0 + frac * float(h)))
                sample_points.append((x0 - pad, py))
                sample_points.append((x1 + pad, py))

            sample_points.extend(
                [
                    (x0 - pad, y0 - pad),
                    (x1 + pad, y0 - pad),
                    (x0 - pad, y1 + pad),
                    (x1 + pad, y1 + pad),
                ]
            )

            values: list[tuple[int, int, int]] = []
            for sx, sy in sample_points:
                cx = max(0, min(W - 1, int(sx)))
                cy = max(0, min(H - 1, int(sy)))
                if _point_in_protect(cx, cy):
                    continue
                rgb = _pixel_to_rgb_triplet(img.getpixel((cx, cy)))
                if rgb is not None:
                    values.append(rgb)

            if not values:
                rgb = _pixel_to_rgb_triplet(
                    img.getpixel((max(0, min(W - 1, x0)), max(0, min(H - 1, y0))))
                )
                if rgb is not None:
                    values.append(rgb)
            return _median_color(values)

        try:
            fill_img = img.copy()
            protect_mask_img = Image.new("L", (W, H), 0)
            if protect_rects:
                protect_draw = ImageDraw.Draw(protect_mask_img)
                for x0, y0, x1, y1 in protect_rects:
                    protect_draw.rectangle(
                        [x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], fill=255
                    )

            for (x0, y0, x1, y1), polygon_px in zip(rects, polygon_masks_px):
                color = _estimate_fill_color(x0, y0, x1, y1)

                patch_x0 = int(x0)
                patch_y0 = int(y0)
                patch_x1 = int(x1)
                patch_y1 = int(y1)
                if polygon_px is not None:
                    xs = [int(pt[0]) for pt in polygon_px]
                    ys = [int(pt[1]) for pt in polygon_px]
                    if xs and ys:
                        patch_x0 = min(patch_x0, min(xs))
                        patch_y0 = min(patch_y0, min(ys))
                        patch_x1 = max(patch_x1, max(xs) + 1)
                        patch_y1 = max(patch_y1, max(ys) + 1)

                patch_x0 = max(0, patch_x0 - dilate_pad)
                patch_y0 = max(0, patch_y0 - dilate_pad)
                patch_x1 = min(W, patch_x1 + dilate_pad)
                patch_y1 = min(H, patch_y1 + dilate_pad)
                if patch_x1 <= patch_x0 or patch_y1 <= patch_y0:
                    continue

                rect_mask = Image.new(
                    "L",
                    (int(patch_x1 - patch_x0), int(patch_y1 - patch_y0)),
                    0,
                )
                rect_draw = ImageDraw.Draw(rect_mask)
                if polygon_px is not None:
                    local_polygon_px = [
                        (int(px - patch_x0), int(py - patch_y0))
                        for px, py in polygon_px
                    ]
                    if len({pt for pt in local_polygon_px}) >= 3:
                        rect_draw.polygon(local_polygon_px, fill=255)
                    else:
                        rect_draw.rectangle(
                            [
                                int(x0 - patch_x0),
                                int(y0 - patch_y0),
                                max(int(x0 - patch_x0), int(x1 - patch_x0 - 1)),
                                max(int(y0 - patch_y0), int(y1 - patch_y0 - 1)),
                            ],
                            fill=255,
                        )
                else:
                    rect_draw.rectangle(
                        [
                            int(x0 - patch_x0),
                            int(y0 - patch_y0),
                            max(int(x0 - patch_x0), int(x1 - patch_x0 - 1)),
                            max(int(y0 - patch_y0), int(y1 - patch_y0 - 1)),
                        ],
                        fill=255,
                    )
                try:
                    # Expand mask by ~1px to cover anti-aliased text edges.
                    rect_mask = rect_mask.filter(ImageFilter.MaxFilter(dilate_size))
                except Exception:
                    pass
                if protect_rects:
                    rect_mask = ImageChops.subtract(
                        rect_mask,
                        protect_mask_img.crop(
                            (patch_x0, patch_y0, patch_x1, patch_y1)
                        ),
                    )
                    if rect_mask.getbbox() is None:
                        continue
                fill_patch = Image.new(
                    "RGB",
                    (int(patch_x1 - patch_x0), int(patch_y1 - patch_y0)),
                    color,
                )
                fill_img.paste(fill_patch, (patch_x0, patch_y0), rect_mask)

            _ensure_parent_dir(out_path)
            fill_img.save(out_path)
            return out_path
        except Exception:
            return render_path

    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    if np is None:
        # Degraded mode when numpy is unavailable (e.g. partially installed
        # local environments). We still erase text using PIL masks so we don't
        # fall back to "no erase" overlap behavior.
        blur_radius = 2.2 if max(W, H) >= 1600 else 1.6
        strong_blur_radius = min(34.0, max(18.0, 7.5 * float(blur_radius)))
        try:
            bg_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            bg_strong_img = img.filter(
                ImageFilter.GaussianBlur(radius=strong_blur_radius)
            )
        except Exception:
            bg_img = img.copy()
            bg_strong_img = img.copy()

        remove_mask = Image.new("L", (W, H), 0)
        fallback_mask = Image.new("L", (W, H), 0)
        protect_mask_img = Image.new("L", (W, H), 0)
        draw_remove = ImageDraw.Draw(remove_mask)
        draw_fallback = ImageDraw.Draw(fallback_mask)
        draw_protect = ImageDraw.Draw(protect_mask_img)

        for (x0, y0, x1, y1), polygon_px in zip(rects, polygon_masks_px):
            if polygon_px is not None:
                draw_remove.polygon(polygon_px, fill=255)
            else:
                draw_remove.rectangle(
                    [x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], fill=255
                )
        for x0, y0, x1, y1 in core_rects:
            draw_fallback.rectangle(
                [x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], fill=255
            )
        for x0, y0, x1, y1 in protect_rects:
            draw_protect.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], fill=255)

        try:
            dilate_size = 5 if max(W, H) >= 1600 else 3
            remove_mask = remove_mask.filter(ImageFilter.MaxFilter(dilate_size))
            fallback_mask = fallback_mask.filter(ImageFilter.MaxFilter(dilate_size))
        except Exception:
            pass

        if protect_rects:
            try:
                remove_mask = ImageChops.subtract(remove_mask, protect_mask_img)
                fallback_mask = ImageChops.subtract(fallback_mask, protect_mask_img)
            except Exception:
                pass

        out_img = Image.composite(bg_img, img, remove_mask)
        out_img = Image.composite(bg_strong_img, out_img, fallback_mask)
        try:
            _ensure_parent_dir(out_path)
            out_img.save(out_path)
            return out_path
        except Exception:
            return render_path

    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return render_path

    # Smooth background estimate used for pixel-level replacement. This avoids
    # rectangle color blocks and is visually closer to "text removed".
    try:
        blur_radius = 2.2 if max(W, H) >= 1600 else 1.6
        bg_arr = np.array(
            img.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.uint8
        )
        strong_blur_radius = min(34.0, max(18.0, 7.5 * float(blur_radius)))
        bg_strong_arr = np.array(
            img.filter(ImageFilter.GaussianBlur(radius=strong_blur_radius)),
            dtype=np.uint8,
        )
    except Exception:
        bg_arr = arr.copy()
        bg_strong_arr = arr.copy()

    # Luma map from the *original* render; detection should not depend on
    # sequential edits to nearby boxes.
    gray = (
        0.299 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.114 * arr[:, :, 2].astype(np.float32)
    )

    protect_mask = np.zeros((H, W), dtype=bool)
    for x0p, y0p, x1p, y1p in protect_rects:
        protect_mask[y0p:y1p, x0p:x1p] = True

    out = arr.copy()
    rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    remove_mask = np.zeros((H, W), dtype=bool)
    fallback_mask = np.zeros((H, W), dtype=bool)
    remove_color_mask = np.zeros((H, W), dtype=bool)
    remove_color_map = np.zeros((H, W, 3), dtype=np.uint8)

    def _dilate_mask(mask: Any, radius: int = 1) -> Any:
        if radius <= 0:
            return mask
        hh, ww = mask.shape
        pad = int(radius)
        src = np.pad(
            mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=False
        )
        dil = np.zeros_like(mask, dtype=bool)
        for dy in range(0, 2 * pad + 1):
            y_slice = slice(dy, dy + hh)
            for dx in range(0, 2 * pad + 1):
                dil |= src[y_slice, dx : dx + ww]
        return dil

    def _median_ring_rgb(
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> tuple[int, int, int]:
        if x1 <= x0 or y1 <= y0:
            return (255, 255, 255)

        h = max(1, int(y1 - y0))
        pad = max(2, min(12, int(round(0.45 * float(h)))))
        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(W, x1 + pad)
        ry1 = min(H, y1 + pad)
        if rx1 <= rx0 or ry1 <= ry0:
            return (255, 255, 255)

        ring = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
        ix0 = max(0, x0 - rx0)
        iy0 = max(0, y0 - ry0)
        ix1 = min(ring.shape[1], x1 - rx0)
        iy1 = min(ring.shape[0], y1 - ry0)
        ring[iy0:iy1, ix0:ix1] = False

        sub_protect = protect_mask[ry0:ry1, rx0:rx1]
        if sub_protect.any():
            ring &= ~sub_protect

        ring_pixels = arr[ry0:ry1, rx0:rx1][ring]
        if ring_pixels.size <= 0:
            # Fallback: median of the local strong-blur patch.
            sub_blur = bg_strong_arr[y0:y1, x0:x1]
            if sub_blur.size <= 0:
                return (255, 255, 255)
            med = np.median(sub_blur.reshape(-1, 3), axis=0)
        else:
            med = np.median(ring_pixels.reshape(-1, 3), axis=0)

        return (
            int(max(0, min(255, round(float(med[0]))))),
            int(max(0, min(255, round(float(med[1]))))),
            int(max(0, min(255, round(float(med[2]))))),
        )

    for x0, y0, x1, y1 in rects:
        w = max(1, int(x1 - x0))
        h = max(1, int(y1 - y0))

        # Expand mostly in X so we can remove missed glyph tails in the same line
        # without crossing to unrelated rows.
        grow_x = max(2, min(18, int(round(0.55 * float(h)))))
        grow_y = max(1, min(4, int(round(0.18 * float(h)))))
        ex0 = max(0, x0 - grow_x)
        ey0 = max(0, y0 - grow_y)
        ex1 = min(W, x1 + grow_x)
        ey1 = min(H, y1 + grow_y)
        if ex1 <= ex0 or ey1 <= ey0:
            continue

        sub_gray = gray[ey0:ey1, ex0:ex1]
        sub_protect = protect_mask[ey0:ey1, ex0:ex1]

        ix0 = max(0, x0 - ex0)
        iy0 = max(0, y0 - ey0)
        ix1 = min(sub_gray.shape[1], x1 - ex0)
        iy1 = min(sub_gray.shape[0], y1 - ey0)
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        # Local background estimate from the ring around the core box.
        ring_mask = np.ones_like(sub_gray, dtype=bool)
        ring_mask[iy0:iy1, ix0:ix1] = False
        if sub_protect.any():
            ring_mask &= ~sub_protect
        ring_vals = sub_gray[ring_mask]
        local_bg = (
            float(np.median(ring_vals))
            if ring_vals.size > 0
            else float(np.median(sub_gray))
        )

        # Text-like pixel mask (dark on light, and bright on dark backgrounds).
        delta_dark = max(8.0, min(26.0, 0.14 * local_bg + 4.0))
        delta_bright = max(9.0, min(30.0, 0.13 * (255.0 - local_bg) + 6.0))
        dark_mask = sub_gray <= (local_bg - delta_dark)
        if local_bg < 125.0:
            bright_mask = sub_gray >= (local_bg + delta_bright)
            text_like = dark_mask | bright_mask
        else:
            text_like = dark_mask

        # Constrain to a near-line band to avoid touching unrelated content.
        band_pad = max(0, min(2, int(round(0.10 * float(h)))))
        by0 = max(0, iy0 - band_pad)
        by1 = min(sub_gray.shape[0], iy1 + band_pad)
        band_mask = np.zeros_like(sub_gray, dtype=bool)
        band_mask[by0:by1, :] = True

        m = text_like & band_mask & (~sub_protect)

        # If the mask is unexpectedly huge, tighten threshold for stability.
        band_area = max(1, int(np.count_nonzero(band_mask & (~sub_protect))))
        if (int(np.count_nonzero(m)) / float(band_area)) > 0.42:
            stricter = max(10.0, delta_dark + 5.0)
            m = (sub_gray <= (local_bg - stricter)) & band_mask & (~sub_protect)

        # Include anti-aliased fringes while staying around text-like pixels.
        near_delta = max(4.0, min(14.0, 0.55 * delta_dark))
        near_mask = (np.abs(sub_gray - local_bg) >= near_delta) & band_mask
        m = (m | (_dilate_mask(m, radius=1) & near_mask)) & (~sub_protect)

        core = np.zeros_like(sub_gray, dtype=bool)
        core[iy0:iy1, ix0:ix1] = True
        core &= ~sub_protect
        core_pixels = int(np.count_nonzero(core))
        masked_pixels = int(np.count_nonzero(m))

        # If pixel mask misses too much of the OCR bbox, force a fallback wipe on
        # the bbox core. This prevents the "no erase -> text overlap" failure mode.
        need_fallback = False
        if masked_pixels <= 0:
            need_fallback = True
        elif core_pixels > 0 and (masked_pixels / float(core_pixels)) < 0.08:
            need_fallback = True

        if masked_pixels > 0:
            remove_mask[ey0:ey1, ex0:ex1] |= m
            fr, fg, fb = _median_ring_rgb(x0, y0, x1, y1)
            sub_color_map = remove_color_map[ey0:ey1, ex0:ex1]
            sub_color_map[m] = (fr, fg, fb)
            remove_color_map[ey0:ey1, ex0:ex1] = sub_color_map
            remove_color_mask[ey0:ey1, ex0:ex1] |= m
        if need_fallback and core_pixels > 0:
            fallback_mask[ey0:ey1, ex0:ex1] |= core

    if np.any(remove_mask) or np.any(fallback_mask):
        # A tiny dilation better covers anti-aliased fringes.
        remove_mask = _dilate_mask(remove_mask, radius=1)
        fallback_mask = _dilate_mask(fallback_mask, radius=1)
        remove_mask &= ~protect_mask
        fallback_mask &= ~protect_mask
        fallback_only = fallback_mask & (~remove_mask)

        # Prefer local ring-color replacement to avoid long blur streaks.
        fill_remove_mask = remove_mask & remove_color_mask
        if np.any(fill_remove_mask):
            out[fill_remove_mask] = remove_color_map[fill_remove_mask]

        residual_remove_mask = remove_mask & (~remove_color_mask)
        if np.any(residual_remove_mask):
            out[residual_remove_mask] = bg_arr[residual_remove_mask]

        if np.any(fallback_only):
            out[fallback_only] = bg_arr[fallback_only]

    # Second pass: if a core OCR box changed too little, force a stronger wipe.
    # This specifically addresses "old text still visible under new text".
    unresolved_mask = np.zeros((H, W), dtype=bool)
    try:
        diff_changed = (
            np.abs(out.astype(np.int16) - arr.astype(np.int16)).sum(axis=2) >= 8
        )
        for x0, y0, x1, y1 in core_rects:
            sub_protect = protect_mask[y0:y1, x0:x1]
            eligible = ~sub_protect
            eligible_px = int(np.count_nonzero(eligible))
            if eligible_px <= 0:
                continue
            changed_px = int(np.count_nonzero(diff_changed[y0:y1, x0:x1] & eligible))
            # Require a fairly high changed ratio in the OCR core box. If too
            # little changed, we still risk visible old glyphs under new text.
            if (changed_px / float(eligible_px)) < 0.72:
                unresolved_mask[y0:y1, x0:x1] |= eligible
    except Exception:
        unresolved_mask = np.zeros((H, W), dtype=bool)

    if np.any(unresolved_mask):
        unresolved_mask = _dilate_mask(unresolved_mask, radius=1)
        unresolved_mask &= ~protect_mask
        out[unresolved_mask] = bg_arr[unresolved_mask]

    # Local final touch: only repaint OCR core pixels that still look like
    # high-contrast glyph strokes, and keep image-protected areas untouched.
    if core_rects:
        final_force_mask = np.zeros((H, W), dtype=bool)
        for x0, y0, x1, y1 in core_rects:
            if x1 <= x0 or y1 <= y0:
                continue
            sub_protect = protect_mask[y0:y1, x0:x1]
            if sub_protect.shape[0] <= 0 or sub_protect.shape[1] <= 0:
                continue

            sub_gray = gray[y0:y1, x0:x1]
            sub_out = out[y0:y1, x0:x1]
            out_luma = (
                0.299 * sub_out[:, :, 0].astype(np.float32)
                + 0.587 * sub_out[:, :, 1].astype(np.float32)
                + 0.114 * sub_out[:, :, 2].astype(np.float32)
            )
            residual = np.abs(sub_gray - out_luma)
            # Repaint only pixels that still differ notably from the surrounding
            # smoothed background estimate (likely residual old glyph strokes).
            local_mask = (residual >= 10.0) & (~sub_protect)
            if not np.any(local_mask):
                continue
            sub_force = final_force_mask[y0:y1, x0:x1]
            sub_force |= local_mask
            final_force_mask[y0:y1, x0:x1] = sub_force

        if np.any(final_force_mask):
            final_force_mask = _dilate_mask(final_force_mask, radius=1)
            final_force_mask &= ~protect_mask
            out[final_force_mask] = bg_arr[final_force_mask]

    # Avoid an additional whole-core repaint pass here. It can create
    # visible blur bands on light templates when OCR boxes are slightly wide.

    try:
        out_img = Image.fromarray(out.astype(np.uint8), mode="RGB")
        _ensure_parent_dir(out_path)
        out_img.save(out_path)
        return out_path
    except Exception:
        return render_path


@dataclass
class _ScannedImageRegionInfo:
    bbox_pt: list[float]
    suppress_bbox_pt: list[float]
    crop_path: Path
    shape_confirmed: bool
    ai_hint: bool = False
    background_removed: bool = False
    geometry_kind: str | None = None
    geometry_points_pt: list[list[float]] | None = None


def _geometry_points_signature(
    points: list[list[float]] | None,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(points, list):
        return ()
    signature: list[tuple[int, int]] = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            return ()
        try:
            signature.append(
                (
                    int(round(float(point[0]) * 10.0)),
                    int(round(float(point[1]) * 10.0)),
                )
            )
        except Exception:
            return ()
    return tuple(signature)


def _save_scanned_image_region_crop(
    *,
    img: Any,
    bbox_pt: list[float],
    crop_out_path: Path,
    page_h_pt: float,
    scanned_render_dpi: int,
    geometry_points_pt: list[list[float]] | None = None,
) -> bool:
    try:
        from PIL import ImageDraw, ImageFilter
    except Exception:
        return False

    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
    except Exception:
        return False

    x0p, y0p = _pdf_pt_to_pix_px(
        x0,
        y0,
        page_height_pt=page_h_pt,
        dpi=int(scanned_render_dpi),
    )
    x1p, y1p = _pdf_pt_to_pix_px(
        x1,
        y1,
        page_height_pt=page_h_pt,
        dpi=int(scanned_render_dpi),
    )
    x0p = max(0, min(int(img.width - 1), int(x0p)))
    y0p = max(0, min(int(img.height - 1), int(y0p)))
    x1p = max(0, min(int(img.width), int(x1p)))
    y1p = max(0, min(int(img.height), int(y1p)))
    if x1p <= x0p or y1p <= y0p:
        return False

    crop = img.crop((x0p, y0p, x1p, y1p))
    if geometry_points_pt:
        polygon_px = _polygon_points_pt_to_px(
            geometry_points_pt,
            page_height_pt=page_h_pt,
            dpi=int(scanned_render_dpi),
            width_px=int(img.width),
            height_px=int(img.height),
        )
        if polygon_px is not None:
            local_points: list[tuple[int, int]] = []
            for px, py in polygon_px:
                coord = (
                    max(0, min(int(crop.width - 1), int(px - x0p))),
                    max(0, min(int(crop.height - 1), int(py - y0p))),
                )
                if local_points and coord == local_points[-1]:
                    continue
                local_points.append(coord)
            if len({point for point in local_points}) >= 3:
                crop = crop.convert("RGBA")
                mask = crop.getchannel("A")
                mask.paste(0, (0, 0, crop.width, crop.height))
                ImageDraw.Draw(mask).polygon(local_points, fill=255)
                try:
                    mask = mask.filter(ImageFilter.MaxFilter(3))
                except Exception:
                    pass
                crop.putalpha(mask)

    try:
        _ensure_parent_dir(crop_out_path)
        crop.save(crop_out_path)
        return True
    except Exception:
        return False


def _build_scanned_image_region_suppress_bbox(
    bbox_pt: list[float],
    *,
    page_w_pt: float,
    page_h_pt: float,
    shape_confirmed: bool,
) -> list[float]:
    x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
    w_pt = float(x1 - x0)
    h_pt = float(y1 - y0)
    if shape_confirmed:
        pad_x = max(1.5, min(8.0, 0.05 * w_pt))
        pad_y = max(1.0, min(6.0, 0.07 * h_pt))
    else:
        pad_x = max(0.8, min(3.5, 0.02 * w_pt))
        pad_y = max(0.8, min(3.0, 0.03 * h_pt))
    return [
        max(0.0, float(x0) - pad_x),
        max(0.0, float(y0) - pad_y),
        min(float(page_w_pt), float(x1) + pad_x),
        min(float(page_h_pt), float(y1) + pad_y),
    ]


def _tighten_scanned_image_region_bbox_by_visual_bounds(
    *,
    img: Any,
    bbox_pt: list[float],
    page_w_pt: float,
    page_h_pt: float,
    scanned_render_dpi: int,
    shape_confirmed: bool,
    ocr_text_elements: list[dict[str, Any]] | None = None,
) -> list[float] | None:
    """Best-effort tighten for image crops with excessive blank margins."""

    try:
        import numpy as np
        from PIL import ImageFilter
    except Exception:
        return None

    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
    except Exception:
        return None

    if float(x1 - x0) <= 0.0 or float(y1 - y0) <= 0.0:
        return None

    x0p, y0p = _pdf_pt_to_pix_px(
        x0,
        y0,
        page_height_pt=page_h_pt,
        dpi=int(scanned_render_dpi),
    )
    x1p, y1p = _pdf_pt_to_pix_px(
        x1,
        y1,
        page_height_pt=page_h_pt,
        dpi=int(scanned_render_dpi),
    )
    x0p = max(0, min(int(img.width - 1), int(x0p)))
    y0p = max(0, min(int(img.height - 1), int(y0p)))
    x1p = max(0, min(int(img.width), int(x1p)))
    y1p = max(0, min(int(img.height), int(y1p)))
    if x1p <= x0p or y1p <= y0p:
        return None

    try:
        crop = img.crop((x0p, y0p, x1p, y1p)).convert("L")
    except Exception:
        return None

    w = int(crop.width)
    h = int(crop.height)
    if w < 40 or h < 40:
        return None

    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim != 2 or arr.size <= 0:
        return None

    ring = max(3, min(14, int(round(0.05 * float(min(w, h))))))
    border_vals = np.concatenate(
        [
            arr[:ring, :].reshape(-1),
            arr[max(0, h - ring) :, :].reshape(-1),
            arr[:, :ring].reshape(-1),
            arr[:, max(0, w - ring) :].reshape(-1),
        ]
    )
    if border_vals.size <= 0:
        return None

    bg = float(np.median(border_vals))
    diff = np.abs(arr.astype(np.int16) - int(round(bg)))

    edges_img = crop.filter(ImageFilter.FIND_EDGES)
    edges = np.asarray(edges_img, dtype=np.uint8)
    if edges.shape != arr.shape:
        return None

    diff_thresh = 18.0 if bg >= 150.0 else 22.0
    edge_thresh = 24 if shape_confirmed else 28
    mask = (edges >= edge_thresh) | (diff >= diff_thresh)

    if ocr_text_elements:
        overlap_boxes: list[tuple[int, int, int, int]] = []
        overlap_cov = 0.0
        crop_area = max(1.0, float(x1 - x0) * float(y1 - y0))
        for el in ocr_text_elements:
            bbox_el = el.get("bbox_pt") if isinstance(el, dict) else None
            if not isinstance(bbox_el, list) or len(bbox_el) != 4:
                continue
            try:
                tx0, ty0, tx1, ty1 = _coerce_bbox_pt(bbox_el)
            except Exception:
                continue
            ix0 = max(float(x0), float(tx0))
            iy0 = max(float(y0), float(ty0))
            ix1 = min(float(x1), float(tx1))
            iy1 = min(float(y1), float(ty1))
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            overlap_cov += float((ix1 - ix0) * (iy1 - iy0)) / crop_area
            ex0 = max(
                0,
                int(
                    round(
                        (float(tx0) * float(scanned_render_dpi) / _PTS_PER_INCH)
                        - x0p
                    )
                ),
            )
            ey0 = max(
                0,
                int(
                    round(
                        (float(ty0) * float(scanned_render_dpi) / _PTS_PER_INCH)
                        - y0p
                    )
                ),
            )
            ex1 = min(
                w,
                int(
                    round(
                        (float(tx1) * float(scanned_render_dpi) / _PTS_PER_INCH)
                        - x0p
                    )
                ),
            )
            ey1 = min(
                h,
                int(
                    round(
                        (float(ty1) * float(scanned_render_dpi) / _PTS_PER_INCH)
                        - y0p
                    )
                ),
            )
            if ex1 <= ex0 or ey1 <= ey0:
                continue
            overlap_boxes.append((ex0, ey0, ex1, ey1))

        if overlap_boxes and len(overlap_boxes) <= 2 and overlap_cov >= 0.10:
            text_x0 = min(box[0] for box in overlap_boxes)
            text_y0 = min(box[1] for box in overlap_boxes)
            text_x1 = max(box[2] for box in overlap_boxes)
            text_y1 = max(box[3] for box in overlap_boxes)
            text_w = max(1, text_x1 - text_x0)
            text_h = max(1, text_y1 - text_y0)

            if (
                text_y0 >= int(round(0.48 * float(h)))
                and text_w >= int(round(0.28 * float(w)))
                and text_h <= int(round(0.42 * float(h)))
                and text_y0 >= max(18, int(round(0.22 * float(h))))
            ):
                clip_pad = max(4, min(14, int(round(0.06 * float(h)))))
                ny1p = max(y0p + 8, y0p + text_y0 - clip_pad)
                if ny1p > y0p + int(round(0.22 * float(h))):
                    scale = float(_PTS_PER_INCH) / float(max(1, int(scanned_render_dpi)))
                    clipped = [
                        float(x0p) * scale,
                        float(y0p) * scale,
                        float(x1p) * scale,
                        float(ny1p) * scale,
                    ]
                    clipped = [float(v) for v in _coerce_bbox_pt(clipped)]
                    if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                        return clipped

            for ex0, ey0, ex1, ey1 in overlap_boxes:
                pad_x = max(4, min(18, int(round(0.12 * float(ex1 - ex0)))))
                pad_y = max(4, min(20, int(round(0.28 * float(ey1 - ey0)))))
                ex0 = max(0, ex0 - pad_x)
                ey0 = max(0, ey0 - pad_y)
                ex1 = min(w, ex1 + pad_x)
                ey1 = min(h, ey1 + pad_y)
                mask[ey0:ey1, ex0:ex1] = False

    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    if row_counts.size > 0 and col_counts.size > 0:
        row_peak = int(row_counts.max()) if row_counts.size else 0
        col_peak = int(col_counts.max()) if col_counts.size else 0
        row_thresh = max(2, int(round(0.10 * float(row_peak)))) if row_peak > 0 else 2
        col_thresh = max(2, int(round(0.10 * float(col_peak)))) if col_peak > 0 else 2
        ys = np.where(row_counts >= row_thresh)[0]
        xs = np.where(col_counts >= col_thresh)[0]
    else:
        ys = np.empty((0,), dtype=np.int32)
        xs = np.empty((0,), dtype=np.int32)

    if xs.size < 24 or ys.size < 24:
        ys, xs = np.where(mask)
    if xs.size < 24 or ys.size < 24:
        return None

    cx0 = int(xs.min())
    cy0 = int(ys.min())
    cx1 = int(xs.max()) + 1
    cy1 = int(ys.max()) + 1

    left_margin = cx0
    top_margin = cy0
    right_margin = max(0, w - cx1)
    bottom_margin = max(0, h - cy1)
    trim_x = float(left_margin + right_margin) / max(1.0, float(w))
    trim_y = float(top_margin + bottom_margin) / max(1.0, float(h))
    side_trim_px = max(left_margin, top_margin, right_margin, bottom_margin)
    if side_trim_px < max(10, int(round(0.06 * float(min(w, h))))) and (
        trim_x < 0.10 and trim_y < 0.10
    ):
        return None

    pad = max(4, min(18, int(round(0.03 * float(min(w, h))))))
    nx0p = max(x0p, x0p + cx0 - pad)
    ny0p = max(y0p, y0p + cy0 - pad)
    nx1p = min(x1p, x0p + cx1 + pad)
    ny1p = min(y1p, y0p + cy1 + pad)
    if nx1p <= nx0p or ny1p <= ny0p:
        return None

    orig_area = max(1.0, float(x1p - x0p) * float(y1p - y0p))
    new_area = max(1.0, float(nx1p - nx0p) * float(ny1p - ny0p))
    shrink_ratio = float(new_area) / float(orig_area)
    if shrink_ratio <= 0.18:
        return None

    scale = float(_PTS_PER_INCH) / float(max(1, int(scanned_render_dpi)))
    tightened = [
        float(nx0p) * scale,
        float(ny0p) * scale,
        float(nx1p) * scale,
        float(ny1p) * scale,
    ]
    tightened = [float(v) for v in _coerce_bbox_pt(tightened)]
    if tightened[2] <= tightened[0] or tightened[3] <= tightened[1]:
        return None
    return tightened


def _tighten_scanned_image_region_infos(
    *,
    infos: list[_ScannedImageRegionInfo],
    img: Any,
    page_w_pt: float,
    page_h_pt: float,
    scanned_render_dpi: int,
    ocr_text_elements: list[dict[str, Any]] | None = None,
) -> list[_ScannedImageRegionInfo]:
    if not infos:
        return infos

    out: list[_ScannedImageRegionInfo] = []
    for info in infos:
        if info.geometry_points_pt:
            out.append(info)
            continue
        tightened_bbox = _tighten_scanned_image_region_bbox_by_visual_bounds(
            img=img,
            bbox_pt=info.bbox_pt,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            scanned_render_dpi=scanned_render_dpi,
            shape_confirmed=bool(info.shape_confirmed),
            ocr_text_elements=ocr_text_elements,
        )
        if tightened_bbox is None:
            out.append(info)
            continue

        try:
            x0, y0, x1, y1 = _coerce_bbox_pt(tightened_bbox)
            x0p, y0p = _pdf_pt_to_pix_px(
                x0,
                y0,
                page_height_pt=page_h_pt,
                dpi=int(scanned_render_dpi),
            )
            x1p, y1p = _pdf_pt_to_pix_px(
                x1,
                y1,
                page_height_pt=page_h_pt,
                dpi=int(scanned_render_dpi),
            )
            x0p = max(0, min(int(img.width - 1), int(x0p)))
            y0p = max(0, min(int(img.height - 1), int(y0p)))
            x1p = max(0, min(int(img.width), int(x1p)))
            y1p = max(0, min(int(img.height), int(y1p)))
            if x1p <= x0p or y1p <= y0p:
                out.append(info)
                continue

            crop = img.crop((x0p, y0p, x1p, y1p))
            _ensure_parent_dir(info.crop_path)
            crop.save(info.crop_path)
        except Exception:
            out.append(info)
            continue

        suppress_bbox = _build_scanned_image_region_suppress_bbox(
            tightened_bbox,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            shape_confirmed=bool(info.shape_confirmed),
        )
        out.append(
            _ScannedImageRegionInfo(
                bbox_pt=[float(v) for v in _coerce_bbox_pt(tightened_bbox)],
                suppress_bbox_pt=[float(v) for v in _coerce_bbox_pt(suppress_bbox)],
                crop_path=info.crop_path,
                shape_confirmed=bool(info.shape_confirmed),
                ai_hint=bool(info.ai_hint),
                background_removed=bool(info.background_removed),
                geometry_kind=info.geometry_kind,
                geometry_points_pt=info.geometry_points_pt,
            )
        )
    return out


def _estimate_baseline_ocr_line_height_pt(
    *,
    ocr_text_elements: list[dict[str, Any]],
    page_w_pt: float,
) -> float:
    """Estimate a "typical" OCR line height (pt) on scanned pages.

    Many scanned-slide OCR engines also detect lots of tiny UI text inside
    screenshots/diagrams. Using a raw median/low-quantile can be skewed toward
    those tiny boxes, which then breaks downstream heuristics (wrap decision,
    image-region detection, dedupe thresholds).

    We therefore:
    - filter invalid/extreme boxes
    - focus on the *widest* OCR lines (more likely slide body text)
    - compute a width-weighted upper-median (slightly biased toward larger text)
    """

    samples: list[tuple[float, float]] = []  # (height_pt, width_ratio)
    width_pt = max(1.0, float(page_w_pt))

    for el in ocr_text_elements:
        if not isinstance(el, dict):
            continue
        bbox_pt = el.get("bbox_pt")
        if not isinstance(bbox_pt, list) or len(bbox_pt) != 4:
            continue
        try:
            x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
        except Exception:
            continue
        w = max(0.0, float(x1 - x0))
        h = max(0.0, float(y1 - y0))
        if w <= 0.0 or h <= 0.0:
            continue
        # Filter extreme outliers.
        if h < 4.5 or h > 96.0:
            continue
        width_ratio = w / width_pt
        samples.append((float(h), float(width_ratio)))

    if not samples:
        return 12.0

    # Use only the widest OCR lines to avoid being skewed by many tiny UI
    # elements inside screenshots. For small sample sizes keep all.
    samples.sort(key=lambda t: float(t[1]), reverse=True)
    if len(samples) > 24:
        k = max(12, int(round(0.25 * float(len(samples)))))
        k = max(12, min(int(k), len(samples)))
        samples = samples[:k]

    # Compute a width-weighted quantile on heights. Squaring width_ratio makes
    # narrow UI lines contribute much less even when they are numerous.
    weighted: list[tuple[float, float]] = []
    for h, width_ratio in samples:
        wr = max(0.0, min(1.0, float(width_ratio)))
        weight = max(1e-4, float(wr) * float(wr))
        weighted.append((float(h), float(weight)))

    weighted.sort(key=lambda t: float(t[0]))
    total_w = sum(float(w) for _, w in weighted) or 1.0
    target = 0.60 * total_w
    acc = 0.0
    baseline = float(weighted[len(weighted) // 2][0])
    for h, w in weighted:
        acc += float(w)
        if acc >= target:
            baseline = float(h)
            break

    return max(6.0, min(48.0, float(baseline)))


def _try_make_crop_background_transparent(crop_path: Path) -> bool:
    """Best-effort background removal for icon-like crops.

    We estimate the dominant border color, then flood-fill similar colors from
    image edges as background and convert them to transparent alpha.
    """

    try:
        from PIL import Image, ImageFilter
    except Exception:
        return False

    try:
        img = Image.open(crop_path).convert("RGBA")
    except Exception:
        return False

    w, h = img.size
    if w < 18 or h < 18:
        return False

    band = max(1, min(6, int(round(0.045 * float(min(w, h))))))
    pix = img.load()
    if pix is None:
        return False

    border_rgb: list[tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if x < band or x >= (w - band) or y < band or y >= (h - band):
                rgb = _pixel_to_rgb_triplet(pix[x, y])
                if rgb is None:
                    continue
                border_rgb.append(rgb)
    if len(border_rgb) < 12:
        return False

    def _median(vals: list[int]) -> int:
        if not vals:
            return 0
        s = sorted(vals)
        return int(s[len(s) // 2])

    med_r = _median([c[0] for c in border_rgb])
    med_g = _median([c[1] for c in border_rgb])
    med_b = _median([c[2] for c in border_rgb])

    def _dist_l1(rgb: tuple[int, int, int]) -> int:
        return (
            abs(int(rgb[0]) - med_r)
            + abs(int(rgb[1]) - med_g)
            + abs(int(rgb[2]) - med_b)
        )

    border_d = sorted(_dist_l1(c) for c in border_rgb)
    p90_idx = max(0, min(len(border_d) - 1, int(round(0.90 * (len(border_d) - 1)))))
    p90 = int(border_d[p90_idx])
    # Adaptive threshold; avoid aggressive removal on textured/screenshot-like crops.
    dist_thresh = max(14, min(72, int(round(1.35 * float(p90) + 8.0))))

    bg_candidate = [[False] * w for _ in range(h)]
    for y in range(h):
        row = bg_candidate[y]
        for x in range(w):
            rgb = _pixel_to_rgb_triplet(pix[x, y])
            if rgb is None:
                continue
            r, g, b = rgb
            row[x] = _dist_l1((int(r), int(g), int(b))) <= dist_thresh

    from collections import deque

    bg_mask = [[False] * w for _ in range(h)]
    q: deque[tuple[int, int]] = deque()

    def _enqueue_if_bg(x: int, y: int) -> None:
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        if bg_mask[y][x] or (not bg_candidate[y][x]):
            return
        bg_mask[y][x] = True
        q.append((x, y))

    for x in range(w):
        _enqueue_if_bg(x, 0)
        _enqueue_if_bg(x, h - 1)
    for y in range(h):
        _enqueue_if_bg(0, y)
        _enqueue_if_bg(w - 1, y)

    while q:
        x, y = q.popleft()
        _enqueue_if_bg(x - 1, y)
        _enqueue_if_bg(x + 1, y)
        _enqueue_if_bg(x, y - 1)
        _enqueue_if_bg(x, y + 1)

    total = max(1, w * h)
    bg_pixels = sum(1 for y in range(h) for x in range(w) if bg_mask[y][x])
    bg_ratio = float(bg_pixels) / float(total)
    # Too little/too much background means this likely isn't an icon foreground crop.
    if bg_ratio < 0.15 or bg_ratio > 0.93:
        return False

    alpha_bytes = bytearray(total)
    idx = 0
    for y in range(h):
        for x in range(w):
            alpha_bytes[idx] = 0 if bg_mask[y][x] else 255
            idx += 1

    alpha = Image.frombytes("L", (w, h), bytes(alpha_bytes)).filter(
        ImageFilter.GaussianBlur(radius=0.7)
    )
    img.putalpha(alpha)

    try:
        img.save(crop_path)
    except Exception:
        return False
    return True


def _clear_regions_for_transparent_crops(
    *,
    cleaned_render_path: Path,
    out_path: Path,
    regions_pt: list[list[float]],
    regions_polygons_pt: list[list[list[float]] | None] | None = None,
    pix: Any,
    page_height_pt: float,
    dpi: int,
    clear_expand_min_pt: float = 0.35,
    clear_expand_max_pt: float = 1.5,
    clear_expand_ratio: float = 0.012,
) -> Path:
    if not regions_pt:
        return cleaned_render_path

    try:
        min_expand_pt = float(clear_expand_min_pt)
    except Exception:
        min_expand_pt = 0.35
    try:
        max_expand_pt = float(clear_expand_max_pt)
    except Exception:
        max_expand_pt = 1.5
    try:
        expand_ratio = float(clear_expand_ratio)
    except Exception:
        expand_ratio = 0.012

    min_expand_pt = max(0.0, min(6.0, min_expand_pt))
    max_expand_pt = max(0.0, min(8.0, max_expand_pt))
    if max_expand_pt < min_expand_pt:
        max_expand_pt = min_expand_pt
    expand_ratio = max(0.0, min(0.12, expand_ratio))

    try:
        from PIL import Image, ImageDraw, ImageFilter
    except Exception:
        return cleaned_render_path

    try:
        img = Image.open(cleaned_render_path).convert("RGB")
    except Exception:
        return cleaned_render_path

    fill_img = img.copy()
    raw_polygons = regions_polygons_pt or []
    dilate_size = 5 if max(img.width, img.height) >= 1600 else 3
    for index, bb in enumerate(regions_pt):
        try:
            x0, y0, x1, y1 = _coerce_bbox_pt(bb)
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue

        # Use local surrounding color so erase remains visually consistent.
        fill_rgb = _sample_bbox_background_rgb(
            pix,
            bbox_pt=[x0, y0, x1, y1],
            page_height_pt=page_height_pt,
            dpi=int(dpi),
        )
        # Slight outward expansion + floor/ceil projection reduces edge halos
        # from bbox rounding and anti-aliased crop borders.
        w_pt = max(1.0, float(x1 - x0))
        h_pt = max(1.0, float(y1 - y0))
        expand_pt = max(
            float(min_expand_pt),
            min(float(max_expand_pt), float(expand_ratio) * min(w_pt, h_pt)),
        )
        x0e = float(x0) - float(expand_pt)
        y0e = float(y0) - float(expand_pt)
        x1e = float(x1) + float(expand_pt)
        y1e = float(y1) + float(expand_pt)

        scale = float(dpi) / float(_PTS_PER_INCH)
        x0p = int(math.floor(x0e * scale))
        y0p = int(math.floor(y0e * scale))
        x1p = int(math.ceil(x1e * scale))
        y1p = int(math.ceil(y1e * scale))
        x0p = max(0, min(int(img.width - 1), int(x0p)))
        y0p = max(0, min(int(img.height - 1), int(y0p)))
        x1p = max(0, min(int(img.width), int(x1p)))
        y1p = max(0, min(int(img.height), int(y1p)))
        if x1p <= x0p or y1p <= y0p:
            continue

        region_mask = Image.new("L", (img.width, img.height), 0)
        region_draw = ImageDraw.Draw(region_mask)
        polygon_px = _polygon_points_pt_to_px(
            raw_polygons[index] if index < len(raw_polygons) else None,
            page_height_pt=page_height_pt,
            dpi=int(dpi),
            width_px=int(img.width),
            height_px=int(img.height),
        )
        if polygon_px is not None:
            region_draw.polygon(polygon_px, fill=255)
        else:
            region_draw.rectangle(
                [x0p, y0p, max(x0p, x1p - 1), max(y0p, y1p - 1)],
                fill=255,
            )
        try:
            region_mask = region_mask.filter(ImageFilter.MaxFilter(dilate_size))
        except Exception:
            pass
        fill_img.paste(fill_rgb, (0, 0, img.width, img.height), region_mask)

    try:
        _ensure_parent_dir(out_path)
        fill_img.save(out_path)
        return out_path
    except Exception:
        return cleaned_render_path


def _dedupe_scanned_ocr_text_elements(
    *,
    ocr_text_elements: list[dict[str, Any]],
    baseline_ocr_h_pt: float,
) -> list[dict[str, Any]]:
    """Drop near-duplicate OCR text bboxes on scanned pages.

    Some OCR backends (or their post-processors) can output the same visual line
    twice with tiny bbox jitter. In PPT output this shows up as "double text"
    with a slight offset. We keep the most confident/tight bbox per line.
    """

    if len(ocr_text_elements) <= 1:
        return list(ocr_text_elements)

    candidates: list[dict[str, Any]] = []
    for el in ocr_text_elements:
        if not isinstance(el, dict):
            continue
        bbox_pt = el.get("bbox_pt")
        if not isinstance(bbox_pt, list) or len(bbox_pt) != 4:
            continue
        text = str(el.get("text") or "").strip()
        if not text:
            continue
        try:
            x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        area = float((x1 - x0) * (y1 - y0))
        conf = float(el.get("confidence") or 0.0)
        candidates.append(
            {
                **el,
                "bbox_pt": [float(x0), float(y0), float(x1), float(y1)],
                "_bbox": [float(x0), float(y0), float(x1), float(y1)],
                "_area": float(area),
                "_conf": float(conf),
                "_text": text,
            }
        )

    if len(candidates) <= 1:
        return [dict(el) for el in ocr_text_elements if isinstance(el, dict)]

    # Prefer higher confidence, then smaller/tighter bboxes.
    candidates.sort(
        key=lambda it: (-float(it.get("_conf") or 0.0), float(it.get("_area") or 0.0))
    )

    baseline = max(4.0, float(baseline_ocr_h_pt))
    kept: list[dict[str, Any]] = []
    for cur in candidates:
        cur_bbox = cur.get("_bbox")
        if not isinstance(cur_bbox, list) or len(cur_bbox) != 4:
            continue
        cur_area = float(cur.get("_area") or 1.0)
        cur_text = str(cur.get("_text") or "")
        cur_cy = float(cur_bbox[1] + cur_bbox[3]) / 2.0

        duplicate = False
        for prev in kept:
            prev_bbox = prev.get("_bbox")
            if not isinstance(prev_bbox, list) or len(prev_bbox) != 4:
                continue
            prev_area = float(prev.get("_area") or 1.0)
            prev_cy = float(prev_bbox[1] + prev_bbox[3]) / 2.0
            inter = _bbox_intersection_area_pt(cur_bbox, prev_bbox)
            if inter <= 0.0:
                continue

            overlap_small = float(inter) / max(1.0, float(min(cur_area, prev_area)))
            iou = _bbox_iou_pt(cur_bbox, prev_bbox)
            dy = abs(float(cur_cy) - float(prev_cy))

            # Strong geometry duplicates (same line, jitter).
            if overlap_small >= 0.965 and iou >= 0.85:
                duplicate = True
                break

            # Same text, reasonably overlapping bboxes.
            if overlap_small >= 0.86 and _texts_similar_for_bbox_dedupe(
                cur_text, str(prev.get("_text") or "")
            ):
                duplicate = True
                break

            # Some AI OCR engines (notably DeepSeek grounding outputs on gateways)
            # can emit the *same* visual line twice with a moderate bbox jitter
            # (overlap ~0.70-0.85). Use a vertical-center guard so we don't
            # accidentally delete distinct adjacent lines.
            if dy <= (0.55 * baseline) and _texts_similar_for_bbox_dedupe(
                cur_text, str(prev.get("_text") or "")
            ):
                if overlap_small >= 0.70 or iou >= 0.55:
                    duplicate = True
                    break

            # Defensive: when two boxes are nearly on the same baseline and
            # overlap heavily, keep only one even if text differs (malformed OCR
            # can map different strings onto the same ink).
            if dy <= (0.35 * baseline) and (overlap_small >= 0.80 or iou >= 0.60):
                duplicate = True
                break

            # When line height is near baseline, even slightly smaller overlaps
            # are suspicious duplicates (word vs line boxing). Keep only one.
            try:
                _, y0, _, y1 = _coerce_bbox_pt(cur_bbox)
                cur_h = float(y1 - y0)
            except Exception:
                cur_h = baseline
            if (
                cur_h <= (1.35 * baseline)
                and overlap_small >= 0.78
                and _texts_similar_for_bbox_dedupe(
                    cur_text, str(prev.get("_text") or "")
                )
            ):
                duplicate = True
                break

        if duplicate:
            continue
        kept.append(cur)

    def _reading_key(it: dict[str, Any]) -> tuple[float, float]:
        bb = it.get("_bbox")
        if not isinstance(bb, list) or len(bb) != 4:
            return (0.0, 0.0)
        x0, y0, x1, y1 = bb
        return ((float(y0) + float(y1)) / 2.0, float(x0))

    kept.sort(key=_reading_key)
    out: list[dict[str, Any]] = []
    for it in kept:
        cp = dict(it)
        cp.pop("_bbox", None)
        cp.pop("_area", None)
        cp.pop("_conf", None)
        cp.pop("_text", None)
        out.append(cp)
    return out


def _merge_neighbor_boxes_pt(
    boxes: list[list[float]],
    *,
    page_w_pt: float,
    page_h_pt: float,
    text_coverage_ratio_fn: Callable[[list[float]], tuple[float, int]],
) -> list[list[float]]:
    if len(boxes) <= 1:
        return [list(_coerce_bbox_pt(bb)) for bb in boxes if isinstance(bb, list)]

    merged = [list(_coerce_bbox_pt(bb)) for bb in boxes if isinstance(bb, list)]
    if len(merged) <= 1:
        return merged

    gap_x_pt = max(16.0, 0.04 * float(page_w_pt))
    gap_y_pt = max(12.0, 0.03 * float(page_h_pt))
    for _ in range(2):
        out: list[list[float]] = []
        for bb in merged:
            x0, y0, x1, y1 = _coerce_bbox_pt(bb)
            did_merge = False
            for i, ub in enumerate(out):
                ux0, uy0, ux1, uy1 = _coerce_bbox_pt(ub)
                # Two merge modes:
                # - horizontal adjacency (left/right fragments): require strong Y overlap
                # - vertical adjacency (top/bottom fragments): require strong X overlap
                y_overlap = float(min(y1, uy1) - max(y0, uy0))
                x_overlap = float(min(x1, ux1) - max(x0, ux0))
                min_h = max(1.0, float(min(y1 - y0, uy1 - uy0)))
                min_w = max(1.0, float(min(x1 - x0, ux1 - ux0)))

                # Horizontal merge (existing behavior).
                horizontal_ok = False
                if y_overlap > 0.0 and y_overlap >= (0.62 * min_h):
                    if x0 > ux1:
                        x_gap = float(x0 - ux1)
                    elif ux0 > x1:
                        x_gap = float(ux0 - x1)
                    else:
                        x_gap = 0.0
                    horizontal_ok = x_gap <= gap_x_pt

                # Vertical merge (new): needed for screenshots that are detected as
                # separate top/bottom strips when OCR masking removes some edges.
                vertical_ok = False
                if x_overlap > 0.0 and x_overlap >= (0.62 * min_w):
                    if y0 > uy1:
                        y_gap = float(y0 - uy1)
                    elif uy0 > y1:
                        y_gap = float(uy0 - y1)
                    else:
                        y_gap = 0.0
                    vertical_ok = y_gap <= gap_y_pt

                if not (horizontal_ok or vertical_ok):
                    continue

                candidate = [
                    min(x0, ux0),
                    min(y0, uy0),
                    max(x1, ux1),
                    max(y1, uy1),
                ]
                cw = float(candidate[2] - candidate[0])
                ch = float(candidate[3] - candidate[1])
                page_area = max(1.0, float(page_w_pt) * float(page_h_pt))
                area_ratio = max(0.0, cw * ch) / page_area
                width_ratio = cw / max(1.0, float(page_w_pt))
                cov, n = text_coverage_ratio_fn(candidate)

                # Avoid cross-card and mixed text+image mega merges that swallow paragraphs.
                if (
                    (width_ratio >= 0.56 and (n >= 2 or cov >= 0.08))
                    or (area_ratio >= 0.16 and (n >= 3 or cov >= 0.12))
                    or (width_ratio >= 0.34 and n >= 2 and cov >= 0.05)
                    or (width_ratio >= 0.26 and n >= 3 and cov >= 0.04)
                    or (area_ratio >= 0.08 and n >= 2 and cov >= 0.07)
                ):
                    continue

                out[i] = candidate
                did_merge = True
                break

            if not did_merge:
                out.append([x0, y0, x1, y1])
        merged = out

    return merged


def _collect_scanned_image_region_candidates(
    *,
    page: dict[str, Any],
) -> list[list[float]]:
    """Return only explicit image-region hints from OCR/layout analysis.

    The scanned-page PPT path no longer falls back to local render-based
    heuristics when `image_regions` is missing. If upstream analysis does not
    provide image regions, scanned pages are kept as background + text only.
    """
    regions_pt_from_ai: list[list[float]] = []
    regions = page.get("image_regions")
    if isinstance(regions, list) and regions:
        for raw_region in regions:
            region_info = _coerce_image_region_entry_pt(raw_region)
            if region_info is None:
                continue
            try:
                x0, y0, x1, y1 = _coerce_bbox_pt(region_info.get("bbox_pt"))
            except Exception:
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            regions_pt_from_ai.append([float(x0), float(y0), float(x1), float(y1)])
    return regions_pt_from_ai


def _is_card_like_region(
    bbox: list[float],
    *,
    page_w_pt: float,
    page_h_pt: float,
    baseline_ocr_h_pt: float,
    ocr_text_elements: list[dict[str, Any]],
) -> bool:
    """Detect card-like mixed content region on scanned slides.

    These regions usually contain: an icon + title + paragraph + embedded figure,
    and should *not* be treated as a single image crop.
    """

    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox)
    except Exception:
        return False
    w = float(x1 - x0)
    h = float(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return False

    page_area = max(1.0, float(page_w_pt) * float(page_h_pt))
    area_ratio = (w * h) / page_area
    width_ratio = w / max(1.0, float(page_w_pt))
    height_ratio = h / max(1.0, float(page_h_pt))

    # Cards are usually medium-to-large panels.
    if area_ratio < 0.10:
        return False
    if width_ratio < 0.22 or height_ratio < 0.18:
        return False
    if width_ratio > 0.78 or height_ratio > 0.78:
        return False

    line_h_threshold = max(4.0, 0.60 * float(baseline_ocr_h_pt))
    text_lines = 0
    cjk_lines = 0
    area_overlap = 0.0

    for tel in ocr_text_elements:
        bbox_pt = tel.get("bbox_pt")
        if not isinstance(bbox_pt, list) or len(bbox_pt) != 4:
            continue
        try:
            tx0, ty0, tx1, ty1 = _coerce_bbox_pt(bbox_pt)
        except Exception:
            continue

        cx = (tx0 + tx1) / 2.0
        cy = (ty0 + ty1) / 2.0
        if cx < x0 or cx > x1 or cy < y0 or cy > y1:
            continue

        text_value = str(tel.get("text") or "")
        if _is_inline_short_token(text_value):
            continue

        tw = max(1.0, float(tx1 - tx0))
        th = max(1.0, float(ty1 - ty0))
        if th < line_h_threshold:
            continue

        text_lines += 1
        if _contains_cjk(text_value):
            cjk_lines += 1
        area_overlap += _bbox_intersection_area_pt(
            [x0, y0, x1, y1], [tx0, ty0, tx1, ty1]
        )

    cov = min(1.0, area_overlap / max(1.0, w * h))
    if text_lines >= 4:
        return True
    if cjk_lines >= 2 and text_lines >= 3:
        return True
    if text_lines >= 2 and cov >= 0.05 and area_ratio >= 0.14:
        return True
    return False


def _is_small_text_fragment_region(
    bbox: list[float],
    *,
    page_w_pt: float,
    page_h_pt: float,
    baseline_ocr_h_pt: float,
    ocr_text_elements: list[dict[str, Any]],
) -> bool:
    """Return whether a small candidate region is likely a text fragment.

    Render-based region detection can sometimes isolate part of a title/line as
    an "image crop". Those false positives are usually:
    - small relative to the page
    - strongly aligned with a nearby OCR text line
    - largely explained by one longer OCR bbox on the same baseline
    """

    if not ocr_text_elements:
        return False

    try:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox)
    except Exception:
        return False

    w = max(1.0, float(x1 - x0))
    h = max(1.0, float(y1 - y0))
    area = max(1.0, float(w * h))
    page_area = max(1.0, float(page_w_pt) * float(page_h_pt))
    area_ratio = float(area) / float(page_area)
    if area_ratio > 0.015:
        return False

    line_h_threshold = max(4.0, 0.55 * float(baseline_ocr_h_pt))
    for tel in ocr_text_elements:
        bbox_pt = tel.get("bbox_pt")
        if not isinstance(bbox_pt, list) or len(bbox_pt) != 4:
            continue

        text_value = str(tel.get("text") or "")
        compact_len = _compact_text_length(text_value)
        if compact_len < 4:
            continue

        try:
            tx0, ty0, tx1, ty1 = _coerce_bbox_pt(bbox_pt)
        except Exception:
            continue

        tw = max(1.0, float(tx1 - tx0))
        th = max(1.0, float(ty1 - ty0))
        if th < line_h_threshold:
            continue

        inter = _bbox_intersection_area_pt(
            [x0, y0, x1, y1],
            [tx0, ty0, tx1, ty1],
        )
        if inter > 0.0 and (float(inter) / float(area)) >= 0.72:
            return True

        y_overlap = float(min(y1, ty1) - max(y0, ty0))
        if y_overlap <= 0.0:
            continue
        min_h = max(1.0, float(min(h, th)))
        y_overlap_ratio = float(y_overlap) / float(min_h)
        if y_overlap_ratio < 0.68:
            continue

        x_overlap = max(0.0, float(min(x1, tx1) - max(x0, tx0)))
        min_w = max(1.0, float(min(w, tw)))
        x_overlap_ratio = float(x_overlap) / float(min_w)

        if x_overlap_ratio < 0.10:
            continue

        if tw >= (1.10 * float(w)) or (float(inter) / float(area)) >= 0.55:
            return True

    return False


def _save_scanned_regions_debug_overlay(
    *,
    render_path: Path,
    regions_pt: list[list[float]],
    artifacts_dir: Path,
    page_index: int,
    page_h_pt: float,
    scanned_render_dpi: int,
) -> None:
    if not regions_pt:
        return
    try:
        import json

        dbg_dir = artifacts_dir / "image_regions"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        try:
            json_path = dbg_dir / f"page-{page_index:04d}.regions.json"
            payload = {
                "page_index": int(page_index),
                "regions_pt": [list(_coerce_bbox_pt(bb)) for bb in regions_pt],
            }
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    except Exception:
        pass


def _try_merge_fragmented_scanned_image_regions(
    *,
    infos: list[_ScannedImageRegionInfo],
    img: Any,
    crops_dir: Path,
    page_index: int,
    page_w_pt: float,
    page_h_pt: float,
    scanned_render_dpi: int,
    baseline_ocr_h_pt: float,
    ocr_text_elements: list[dict[str, Any]],
    text_coverage_ratio_fn: Callable[[list[float]], tuple[float, int]],
) -> list[_ScannedImageRegionInfo]:
    """Try to merge split screenshot/diagram regions on scanned pages.

    When OCR masking removes key edges, the render-based detector may find
    multiple fragments (top strip + bottom strip, etc.). Before we commit the
    crops into the PPT we do a small greedy merge pass:
    - only merge close/adjacent regions
    - require the *union crop* to look more like a real image (shape-confirmed)
    - keep conservative text-coverage guards to avoid swallowing paragraphs

    This is intentionally heuristic; it's meant to improve broad reliability
    across OCR models (including ones that output lots of tiny UI text boxes).
    """

    if len(infos) <= 1:
        return infos
    if page_w_pt <= 0 or page_h_pt <= 0:
        return infos

    page_area = max(1.0, float(page_w_pt) * float(page_h_pt))
    merge_counter = 0

    def _bbox_area(bb: list[float]) -> float:
        try:
            x0, y0, x1, y1 = _coerce_bbox_pt(bb)
        except Exception:
            return 0.0
        return max(0.0, float(x1 - x0) * float(y1 - y0))

    def _bbox_union(a: list[float], b: list[float]) -> list[float]:
        ax0, ay0, ax1, ay1 = _coerce_bbox_pt(a)
        bx0, by0, bx1, by1 = _coerce_bbox_pt(b)
        return [
            float(min(ax0, bx0)),
            float(min(ay0, by0)),
            float(max(ax1, bx1)),
            float(max(ay1, by1)),
        ]

    def _gap_and_overlap_ratios(
        a: list[float], b: list[float]
    ) -> tuple[float, float, float, float]:
        ax0, ay0, ax1, ay1 = _coerce_bbox_pt(a)
        bx0, by0, bx1, by1 = _coerce_bbox_pt(b)

        x_overlap = float(min(ax1, bx1) - max(ax0, bx0))
        y_overlap = float(min(ay1, by1) - max(ay0, by0))
        min_w = max(1.0, float(min(ax1 - ax0, bx1 - bx0)))
        min_h = max(1.0, float(min(ay1 - ay0, by1 - by0)))
        x_overlap_ratio = (x_overlap / min_w) if x_overlap > 0.0 else 0.0
        y_overlap_ratio = (y_overlap / min_h) if y_overlap > 0.0 else 0.0

        x_gap = 0.0
        if ax0 > bx1:
            x_gap = float(ax0 - bx1)
        elif bx0 > ax1:
            x_gap = float(bx0 - ax1)

        y_gap = 0.0
        if ay0 > by1:
            y_gap = float(ay0 - by1)
        elif by0 > ay1:
            y_gap = float(by0 - ay1)

        return (x_gap, y_gap, x_overlap_ratio, y_overlap_ratio)

    def _crop_bbox_to_path(bbox_pt: list[float], out_path: Path) -> bool:
        try:
            from PIL import Image
        except Exception:
            return False

        try:
            x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
        except Exception:
            return False

        x0p, y0p = _pdf_pt_to_pix_px(
            x0,
            y0,
            page_height_pt=page_h_pt,
            dpi=int(scanned_render_dpi),
        )
        x1p, y1p = _pdf_pt_to_pix_px(
            x1,
            y1,
            page_height_pt=page_h_pt,
            dpi=int(scanned_render_dpi),
        )
        x0p = max(0, min(int(img.width - 1), int(x0p)))
        y0p = max(0, min(int(img.height - 1), int(y0p)))
        x1p = max(0, min(int(img.width), int(x1p)))
        y1p = max(0, min(int(img.height), int(y1p)))
        if x1p <= x0p or y1p <= y0p:
            return False

        try:
            crop = img.crop((x0p, y0p, x1p, y1p))
            _ensure_parent_dir(out_path)
            crop.save(out_path)
            return True
        except Exception:
            return False

    def _build_union_info(
        bbox_pt: list[float],
        *,
        crop_path: Path,
        shape_confirmed: bool,
        ai_hint: bool,
    ) -> _ScannedImageRegionInfo:
        x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
        suppress_bbox = _build_scanned_image_region_suppress_bbox(
            [float(x0), float(y0), float(x1), float(y1)],
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            shape_confirmed=bool(shape_confirmed),
        )
        return _ScannedImageRegionInfo(
            bbox_pt=[float(x0), float(y0), float(x1), float(y1)],
            suppress_bbox_pt=[float(v) for v in _coerce_bbox_pt(suppress_bbox)],
            crop_path=crop_path,
            shape_confirmed=bool(shape_confirmed),
            ai_hint=bool(ai_hint),
            background_removed=False,
            geometry_kind=None,
            geometry_points_pt=None,
        )

    merged = list(infos)
    for _ in range(3):
        changed = False
        merged.sort(key=lambda info: _bbox_area(info.bbox_pt), reverse=True)

        for i in range(len(merged)):
            a = merged[i]
            a_area = _bbox_area(a.bbox_pt)
            if a_area <= 0.0:
                continue
            for j in range(i + 1, len(merged)):
                b = merged[j]
                b_area = _bbox_area(b.bbox_pt)
                if b_area <= 0.0:
                    continue
                same_ai_hint_polygon = False
                if (
                    bool(getattr(a, "ai_hint", False))
                    and bool(getattr(b, "ai_hint", False))
                    and str(getattr(a, "geometry_kind", "") or "").strip().lower()
                    == "polygon"
                    and str(getattr(b, "geometry_kind", "") or "").strip().lower()
                    == "polygon"
                ):
                    same_ai_hint_polygon = _geometry_points_signature(
                        getattr(a, "geometry_points_pt", None)
                    ) == _geometry_points_signature(
                        getattr(b, "geometry_points_pt", None)
                    ) and bool(getattr(a, "geometry_points_pt", None))
                # Don't merge transparent/icon crops; those should remain editable.
                if (
                    a.background_removed
                    or b.background_removed
                ):
                    continue
                if (a.geometry_points_pt or b.geometry_points_pt) and (
                    not same_ai_hint_polygon
                ):
                    continue

                ax0, ay0, ax1, ay1 = _coerce_bbox_pt(a.bbox_pt)
                bx0, by0, bx1, by1 = _coerce_bbox_pt(b.bbox_pt)
                aw = max(1.0, float(ax1 - ax0))
                ah = max(1.0, float(ay1 - ay0))
                bw = max(1.0, float(bx1 - bx0))
                bh = max(1.0, float(by1 - by0))

                x_gap, y_gap, x_ov, y_ov = _gap_and_overlap_ratios(a.bbox_pt, b.bbox_pt)
                # Only consider adjacent fragments.
                gap_x_limit = max(6.0, min(0.04 * float(page_w_pt), 40.0))
                gap_y_limit = max(6.0, min(0.03 * float(page_h_pt), 32.0))

                horizontal_adjacent = y_ov >= 0.70 and x_gap <= gap_x_limit
                vertical_adjacent = x_ov >= 0.70 and y_gap <= gap_y_limit
                if not (horizontal_adjacent or vertical_adjacent):
                    continue

                # Alignment check: fragmented screenshot detections typically share
                # the same left/right (or top/bottom) edges. Avoid merging two
                # unrelated nearby images in a grid.
                tol_x = max(8.0, min(0.05 * float(page_w_pt), 48.0))
                tol_y = max(8.0, min(0.05 * float(page_h_pt), 48.0))
                width_sim = abs(aw - bw) <= (0.25 * max(aw, bw))
                height_sim = abs(ah - bh) <= (0.25 * max(ah, bh))

                aligned = False
                if same_ai_hint_polygon:
                    aligned = bool(horizontal_adjacent or vertical_adjacent)
                elif vertical_adjacent:
                    aligned = (abs(ax0 - bx0) <= tol_x and abs(ax1 - bx1) <= tol_x) or (
                        width_sim and x_ov >= 0.85
                    )
                elif horizontal_adjacent:
                    aligned = (abs(ay0 - by0) <= tol_y and abs(ay1 - by1) <= tol_y) or (
                        height_sim and y_ov >= 0.85
                    )
                if not aligned:
                    continue

                union_bbox = _bbox_union(a.bbox_pt, b.bbox_pt)
                union_area = _bbox_area(union_bbox)
                if union_area <= 0.0:
                    continue

                # Avoid massive unions (two separate images far apart).
                if union_area > (1.45 * float(a_area + b_area)):
                    continue

                union_area_ratio = float(union_area) / float(page_area)
                if union_area_ratio < 0.020:
                    continue
                if union_area_ratio > 0.72:
                    continue

                cov, n = text_coverage_ratio_fn(union_bbox)
                # Keep conservative: merged screenshot regions should not be text-heavy.
                if cov >= 0.18 or n >= 16:
                    continue

                if _is_card_like_region(
                    union_bbox,
                    page_w_pt=page_w_pt,
                    page_h_pt=page_h_pt,
                    baseline_ocr_h_pt=float(baseline_ocr_h_pt),
                    ocr_text_elements=ocr_text_elements,
                ):
                    continue

                merge_counter += 1
                union_crop_path = (
                    crops_dir
                    / f"page-{page_index:04d}-crop-merge-{merge_counter:02d}.png"
                )
                if not _crop_bbox_to_path(union_bbox, union_crop_path):
                    continue

                union_stats = _analyze_shape_crop(union_crop_path)
                if not union_stats.get("confirmed"):
                    continue

                union_info = _build_union_info(
                    union_bbox,
                    crop_path=union_crop_path,
                    shape_confirmed=bool(union_stats.get("confirmed")),
                    ai_hint=bool(
                        getattr(a, "ai_hint", False) or getattr(b, "ai_hint", False)
                    ),
                )

                # Replace (a,b) with their union.
                keep: list[_ScannedImageRegionInfo] = []
                for k, info in enumerate(merged):
                    if k in (i, j):
                        continue
                    keep.append(info)
                keep.append(union_info)
                merged = keep
                changed = True
                break

            if changed:
                break

        if not changed:
            break

    return merged


def _build_scanned_image_region_infos(
    *,
    page: dict[str, Any],
    render_path: Path,
    artifacts_dir: Path,
    page_index: int,
    page_w_pt: float,
    page_h_pt: float,
    scanned_render_dpi: int,
    baseline_ocr_h_pt: float,
    ocr_text_elements: list[dict[str, Any]],
    has_full_page_bg_image: bool,
    text_coverage_ratio_fn: Callable[[list[float]], tuple[float, int]],
    text_inside_counts_fn: Callable[[list[float]], tuple[int, int]],
    min_area_ratio: float = 0.0025,
    max_area_ratio: float = 0.72,
    max_aspect_ratio: float = 4.8,
) -> list[_ScannedImageRegionInfo]:
    try:
        min_area_ratio_id = float(min_area_ratio)
    except Exception:
        min_area_ratio_id = 0.0025
    try:
        max_area_ratio_id = float(max_area_ratio)
    except Exception:
        max_area_ratio_id = 0.72
    try:
        max_aspect_ratio_id = float(max_aspect_ratio)
    except Exception:
        max_aspect_ratio_id = 4.8

    min_area_ratio_id = max(0.0, min(0.35, min_area_ratio_id))
    max_area_ratio_id = max(0.05, min(1.0, max_area_ratio_id))
    if max_area_ratio_id <= min_area_ratio_id:
        max_area_ratio_id = min(1.0, min_area_ratio_id + 0.05)
    max_aspect_ratio_id = max(1.2, min(30.0, max_aspect_ratio_id))

    regions_pt = _collect_scanned_image_region_candidates(
        page=page,
    )
    _save_scanned_regions_debug_overlay(
        render_path=render_path,
        regions_pt=regions_pt,
        artifacts_dir=artifacts_dir,
        page_index=page_index,
        page_h_pt=page_h_pt,
        scanned_render_dpi=scanned_render_dpi,
    )
    if not regions_pt:
        return []

    try:
        from PIL import Image
    except Exception:
        return []

    try:
        img = Image.open(render_path).convert("RGB")
    except Exception:
        return []

    crops_dir = artifacts_dir / "image_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    page_area = max(1.0, float(page_w_pt) * float(page_h_pt))
    ai_hint_regions_pt: list[dict[str, Any]] = []
    raw_ai_regions = page.get("image_regions")
    if isinstance(raw_ai_regions, list):
        for raw_region in raw_ai_regions:
            region_info = _coerce_image_region_entry_pt(raw_region)
            if region_info is None:
                continue
            try:
                hx0, hy0, hx1, hy1 = _coerce_bbox_pt(region_info.get("bbox_pt"))
            except Exception:
                continue
            if hx1 <= hx0 or hy1 <= hy0:
                continue
            ai_hint_regions_pt.append(
                {
                    **region_info,
                    "bbox_pt": [float(hx0), float(hy0), float(hx1), float(hy1)],
                }
            )

    def _match_ai_hint_candidate(cand_bbox: list[float]) -> dict[str, Any] | None:
        c_area = max(
            1.0,
            float(
                max(
                    0.0,
                    float(cand_bbox[2] - cand_bbox[0])
                    * float(cand_bbox[3] - cand_bbox[1]),
                )
            ),
        )
        best_hint: dict[str, Any] | None = None
        best_score = 0.0
        for hint in ai_hint_regions_pt:
            hint_bbox = hint.get("bbox_pt")
            if not isinstance(hint_bbox, list) or len(hint_bbox) != 4:
                continue
            h_area = max(
                1.0,
                float(
                    max(
                        0.0,
                        float(hint_bbox[2] - hint_bbox[0])
                        * float(hint_bbox[3] - hint_bbox[1]),
                    )
                ),
            )
            inter = _bbox_intersection_area_pt(cand_bbox, hint_bbox)
            if inter <= 0.0:
                continue
            iou = _bbox_iou_pt(cand_bbox, hint_bbox)
            cover_c = inter / c_area
            cover_h = inter / h_area
            if iou < 0.52 and cover_c < 0.72 and cover_h < 0.72:
                continue
            score = max(float(iou), float(cover_c), float(cover_h))
            if best_hint is None or score > best_score:
                best_hint = hint
                best_score = score
        return best_hint

    infos: list[_ScannedImageRegionInfo] = []

    for ri, bbox in enumerate(regions_pt):
        if len(infos) >= 12:
            break

        try:
            x0, y0, x1, y1 = _coerce_bbox_pt(bbox)
        except Exception:
            continue
        w_pt = float(x1 - x0)
        h_pt = float(y1 - y0)
        if w_pt <= 0.0 or h_pt <= 0.0:
            continue

        cand_bbox = [float(x0), float(y0), float(x1), float(y1)]
        matched_ai_hint = _match_ai_hint_candidate(cand_bbox)
        is_ai_hint = matched_ai_hint is not None
        area_pt = max(0.0, w_pt * h_pt)
        area_ratio = area_pt / page_area
        if not is_ai_hint:
            if area_ratio < min_area_ratio_id or area_ratio > max_area_ratio_id:
                continue
        min_dim_threshold_pt = 1.0 if is_ai_hint else 12.0
        if min(w_pt, h_pt) < min_dim_threshold_pt:
            continue

        aspect = max(w_pt / max(1.0, h_pt), h_pt / max(1.0, w_pt))
        if (not is_ai_hint) and aspect >= max_aspect_ratio_id and area_ratio < 0.08:
            continue

        min_dim_pt = max(18.0, 1.8 * float(baseline_ocr_h_pt))
        min_dim_pt = min(72.0, float(min_dim_pt))
        min_area_pt = 0.65 * float(min_dim_pt) * float(min_dim_pt)
        if (not is_ai_hint) and area_pt < min_area_pt:
            continue

        cov, n = text_coverage_ratio_fn([x0, y0, x1, y1])
        n_inside, n_cjk_inside = text_inside_counts_fn([x0, y0, x1, y1])

        if (not is_ai_hint) and _is_card_like_region(
            [x0, y0, x1, y1],
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            baseline_ocr_h_pt=float(baseline_ocr_h_pt),
            ocr_text_elements=ocr_text_elements,
        ):
            continue

        if (not is_ai_hint) and _is_small_text_fragment_region(
            [x0, y0, x1, y1],
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            baseline_ocr_h_pt=float(baseline_ocr_h_pt),
            ocr_text_elements=ocr_text_elements,
        ):
            continue

        large_line_inside = 0
        wide_large_line_inside = 0
        large_line_overlap = 0.0
        large_line_h_threshold = max(4.0, 0.72 * float(baseline_ocr_h_pt))
        for tel in ocr_text_elements:
            bbox_pt = tel.get("bbox_pt")
            if not isinstance(bbox_pt, list) or len(bbox_pt) != 4:
                continue
            try:
                tx0, ty0, tx1, ty1 = _coerce_bbox_pt(bbox_pt)
            except Exception:
                continue
            tcx = (tx0 + tx1) / 2.0
            tcy = (ty0 + ty1) / 2.0
            if tcx < x0 or tcx > x1 or tcy < y0 or tcy > y1:
                continue

            tw = max(1.0, float(tx1 - tx0))
            th = max(1.0, float(ty1 - ty0))
            text_value = str(tel.get("text") or "")
            compact_len = _compact_text_length(text_value)
            if compact_len < 4:
                continue
            if th < large_line_h_threshold:
                continue
            if _is_inline_short_token(text_value):
                continue

            large_line_inside += 1
            if tw >= 0.22 * w_pt:
                wide_large_line_inside += 1
            large_line_overlap += _bbox_intersection_area_pt(
                [tx0, ty0, tx1, ty1], [x0, y0, x1, y1]
            )

        large_line_cov = min(1.0, large_line_overlap / max(1.0, area_pt))

        if not is_ai_hint:
            # Strong text-block candidates should not become image crops.
            if (
                (n >= 4 and cov >= 0.10)
                or (n >= 3 and cov >= 0.16)
                or (n >= 2 and cov >= 0.24)
            ):
                continue
            # Single OCR block with high coverage is often a false-positive card crop
            # (e.g. CJK title/body region) rather than a real screenshot/diagram.
            if n_inside >= 1 and cov >= 0.42 and area_ratio >= 0.012:
                continue
            if n_cjk_inside >= 1 and cov >= 0.30 and area_ratio >= 0.020:
                continue
            # Large text overlap can indicate mixed text panels, but screenshots may
            # legitimately contain some embedded text. Keep this gate conservative.
            if (
                area_ratio >= 0.020
                and large_line_inside >= 2
                and large_line_cov >= 0.10
            ):
                continue
            if large_line_inside >= 4 and (cov >= 0.08 or large_line_cov >= 0.10):
                continue
            if (
                wide_large_line_inside >= 2
                and large_line_cov >= 0.08
                and area_ratio >= 0.030
            ):
                continue

        geometry_kind = None
        geometry_points_pt = None
        if matched_ai_hint is not None:
            raw_geometry_kind = str(
                matched_ai_hint.get("geometry_kind") or ""
            ).strip().lower()
            raw_geometry_points = matched_ai_hint.get("geometry_points_pt")
            if raw_geometry_kind == "polygon" and isinstance(
                raw_geometry_points, list
            ):
                geometry_kind = "polygon"
                geometry_points_pt = [
                    [float(point[0]), float(point[1])]
                    for point in raw_geometry_points
                    if isinstance(point, list) and len(point) >= 2
                ] or None

        crop_out_path = crops_dir / f"page-{page_index:04d}-crop-{ri:02d}.png"
        if not _save_scanned_image_region_crop(
            img=img,
            bbox_pt=cand_bbox,
            crop_out_path=crop_out_path,
            page_h_pt=page_h_pt,
            scanned_render_dpi=int(scanned_render_dpi),
            geometry_points_pt=geometry_points_pt,
        ):
            continue
        shape_confirmed = _is_shape_confirmed_crop(crop_out_path)
        background_removed = False

        # Keep small icon-like regions as independent picture crops so users can
        # edit/move them in PPT (instead of baking them into a single background).

        # For compact/icon-like crops, remove flat background to avoid card-color
        # patches when re-pasting into PPT.
        if shape_confirmed and geometry_points_pt is None:
            try:
                # Background removal is intended for small icons/logos. Applying
                # it to screenshots can incorrectly make large white areas
                # transparent, causing "see-through" artifacts.
                if (
                    area_ratio <= 0.020
                    and aspect <= 2.4
                    and cov <= 0.06
                    and n_inside <= 1
                    and max(w_pt, h_pt) <= (7.0 * float(baseline_ocr_h_pt))
                ):
                    background_removed = _try_make_crop_background_transparent(
                        crop_out_path
                    )
            except Exception:
                background_removed = False

        cjk_text_heavy = (
            n_cjk_inside >= 2 and n_inside >= 3 and cov >= 0.08 and area_ratio >= 0.03
        )
        if shape_confirmed:
            if (not is_ai_hint) and area_ratio >= 0.40 and (
                cov >= 0.20 or n_inside >= 10
            ):
                continue
            if (not is_ai_hint) and cjk_text_heavy and area_ratio >= 0.07:
                continue
            if (not is_ai_hint) and (
                area_ratio >= 0.030
                and large_line_inside >= 3
                and large_line_cov >= 0.10
            ):
                continue
            if (not is_ai_hint) and large_line_inside >= 5 and (
                cov >= 0.08 or large_line_cov >= 0.10
            ):
                continue
            if (not is_ai_hint) and (
                wide_large_line_inside >= 2
                and large_line_cov >= 0.08
                and area_ratio >= 0.030
            ):
                continue
        else:
            if not is_ai_hint:
                if cov >= 0.16 or n_inside >= 5 or large_line_inside >= 3:
                    continue
                if area_ratio >= 0.24:
                    continue
                if cjk_text_heavy and area_ratio >= 0.06:
                    continue

        cand_area = max(1.0, area_pt)
        duplicated = False
        for info in infos:
            inter = _bbox_intersection_area_pt(cand_bbox, info.bbox_pt)
            if inter <= 0.0:
                continue
            if _bbox_iou_pt(cand_bbox, info.bbox_pt) >= 0.66:
                duplicated = True
                break
            ex0, ey0, ex1, ey1 = _coerce_bbox_pt(info.bbox_pt)
            ex_area = max(1.0, float((ex1 - ex0) * (ey1 - ey0)))
            if (inter / cand_area) >= 0.88:
                duplicated = True
                break
            if (
                (inter / ex_area) >= 0.88
                and (cand_area / ex_area) >= 1.6
                and cov >= 0.08
            ):
                duplicated = True
                break
        if duplicated:
            continue

        suppress_bbox = _build_scanned_image_region_suppress_bbox(
            cand_bbox,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            shape_confirmed=bool(shape_confirmed),
        )
        infos.append(
            _ScannedImageRegionInfo(
                bbox_pt=cand_bbox,
                suppress_bbox_pt=[float(v) for v in _coerce_bbox_pt(suppress_bbox)],
                crop_path=crop_out_path,
                shape_confirmed=bool(shape_confirmed),
                ai_hint=bool(is_ai_hint),
                background_removed=bool(background_removed),
                geometry_kind=geometry_kind,
                geometry_points_pt=geometry_points_pt,
            )
        )

    infos = _try_merge_fragmented_scanned_image_regions(
        infos=infos,
        img=img,
        crops_dir=crops_dir,
        page_index=page_index,
        page_w_pt=page_w_pt,
        page_h_pt=page_h_pt,
        scanned_render_dpi=scanned_render_dpi,
        baseline_ocr_h_pt=baseline_ocr_h_pt,
        ocr_text_elements=ocr_text_elements,
        text_coverage_ratio_fn=text_coverage_ratio_fn,
    )
    infos = _tighten_scanned_image_region_infos(
        infos=infos,
        img=img,
        page_w_pt=page_w_pt,
        page_h_pt=page_h_pt,
        scanned_render_dpi=scanned_render_dpi,
        ocr_text_elements=ocr_text_elements,
    )

    # Debug/self-check: persist final crop bboxes used for PPT composition.
    try:
        import json

        dbg_dir = artifacts_dir / "image_regions"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        json_path = dbg_dir / f"page-{page_index:04d}.crops.json"
        payload = {
            "page_index": int(page_index),
            "crops": [
                {
                    "bbox_pt": list(_coerce_bbox_pt(info.bbox_pt)),
                    "suppress_bbox_pt": list(_coerce_bbox_pt(info.suppress_bbox_pt)),
                    "crop_path": str(info.crop_path),
                    "shape_confirmed": bool(info.shape_confirmed),
                    "ai_hint": bool(info.ai_hint),
                    "background_removed": bool(info.background_removed),
                    "geometry_kind": info.geometry_kind,
                    "geometry_points_pt": info.geometry_points_pt,
                }
                for info in infos
            ],
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return infos


def _filter_scanned_ocr_text_elements(
    *,
    ocr_text_elements: list[dict[str, Any]],
    image_region_infos: list[_ScannedImageRegionInfo],
    baseline_ocr_h_pt: float,
) -> list[dict[str, Any]]:
    if not ocr_text_elements or not image_region_infos:
        return list(ocr_text_elements)

    filtered: list[dict[str, Any]] = []
    for el in ocr_text_elements:
        bb = el.get("bbox_pt") if isinstance(el, dict) else None
        if not isinstance(bb, list) or len(bb) != 4:
            continue
        try:
            tx0, ty0, tx1, ty1 = _coerce_bbox_pt(bb)
        except Exception:
            continue
        tw = float(tx1 - tx0)
        th = float(ty1 - ty0)
        if tw <= 0.0 or th <= 0.0:
            continue

        text_value = str(el.get("text") or "").strip()
        compact_len = _compact_text_length(text_value)
        is_cjk_line = _contains_cjk(text_value)
        keep_as_text_preferred = (
            is_cjk_line and compact_len >= 4 and th >= (0.65 * float(baseline_ocr_h_pt))
        )

        t_area = max(1.0, tw * th)
        tcx = (tx0 + tx1) / 2.0
        tcy = (ty0 + ty1) / 2.0
        inside_image = False
        for info in image_region_infos:
            ai_hint = bool(getattr(info, "ai_hint", False))
            region_bbox = info.bbox_pt if ai_hint else info.suppress_bbox_pt
            try:
                ix0, iy0, ix1, iy1 = _coerce_bbox_pt(region_bbox)
            except Exception:
                continue

            inter = _bbox_intersection_area_pt(
                [tx0, ty0, tx1, ty1], [ix0, iy0, ix1, iy1]
            )
            if inter <= 0.0:
                continue
            overlap_ratio = float(inter) / t_area
            center_inside = tcx >= ix0 and tcx <= ix1 and tcy >= iy0 and tcy <= iy1

            if ai_hint:
                # Explicit AI regions already define the crop geometry. Be
                # conservative about removing editable text: only suppress OCR
                # boxes that are almost entirely covered by the model-provided
                # image region, and skip the looser local overlap heuristics.
                if overlap_ratio >= 0.86:
                    inside_image = True
                    break
                continue

            if keep_as_text_preferred and not info.shape_confirmed:
                # Prefer keeping CJK/body text editable when the "image region"
                # is ambiguous (not shape-confirmed). Only suppress tiny labels
                # that are almost fully inside an image region.
                if center_inside and compact_len <= 3 and overlap_ratio >= 0.97:
                    inside_image = True
                    break
                continue

            if overlap_ratio >= 0.72:
                inside_image = True
                break
            if info.shape_confirmed and center_inside and overlap_ratio >= 0.25:
                inside_image = True
                break
            if (not info.shape_confirmed) and center_inside and overlap_ratio >= 0.82:
                inside_image = True
                break
            if center_inside and compact_len <= 3 and overlap_ratio >= 0.22:
                inside_image = True
                break

        if not inside_image:
            filtered.append(el)

    return filtered
