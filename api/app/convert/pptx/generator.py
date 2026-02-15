"""Main entrypoint for generating PPTX from IR."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...models.error import AppException, ErrorCode

from .bbox_utils import (
    _as_path,
    _bbox_pt_to_slide_emu,
    _coerce_bbox_pt,
    _compute_text_erase_padding_pt,
    _ensure_parent_dir,
    _is_near_full_page_bbox_pt,
)
from .color_utils import (
    _hex_to_rgb,
    _pick_contrasting_text_rgb,
    _rgb_sq_distance,
)
from .constants import _EMU_PER_INCH, _EMU_PER_PT
from .font_utils import (
    _compact_text_length,
    _contains_cjk,
    _fit_font_size_pt,
    _fit_ocr_text_style,
    _is_inline_short_token,
    _map_font_name,
    _measure_text_lines,
    _normalize_ocr_text_for_render,
    _prefer_wrap_for_ocr_text,
    _wrap_text_to_width,
)
from .preview import _export_final_preview_page_image
from .scanned_page import (
    _build_scanned_image_region_infos,
    _clear_regions_for_transparent_crops,
    _dedupe_scanned_ocr_text_elements,
    _erase_regions_in_render_image,
    _estimate_baseline_ocr_line_height_pt,
    _filter_scanned_ocr_text_elements,
    _render_pdf_page_png,
    _sample_bbox_background_rgb,
    _sample_bbox_text_rgb,
)
from .slide_builder import (
    _build_transform,
    _infer_font_size_pt,
    _iter_page_elements,
    _set_slide_size_type,
)


def generate_pptx_from_ir(
    ir: dict[str, Any],
    output_pptx_path: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    force_16x9: bool = False,
    scanned_render_dpi: int = 200,
    scanned_page_mode: str = "fullpage",
    text_erase_mode: str = "fill",
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Generate a PPTX from the provided IR.

    Args:
        ir: The intermediate representation dict.
        output_pptx_path: Where to write the PPTX file.
        artifacts_dir: Directory for any intermediate artifacts (e.g. scanned page renders).
        force_16x9: If True, use a 16:9 slide size and letterbox PDF content.
        scanned_render_dpi: DPI used when rendering scanned pages to images.
        text_erase_mode: Erase strategy for background cleanup (smart, fill).
        progress_callback: Optional callback(done_pages, total_pages), called
            after each IR page is written.

    Returns:
        The output PPTX path.
    """

    try:
        pptx = importlib.import_module("pptx")
        Presentation = getattr(pptx, "Presentation")

        RGBColor = getattr(importlib.import_module("pptx.dml.color"), "RGBColor")
        text_enums = importlib.import_module("pptx.enum.text")
        MSO_AUTO_SIZE = getattr(text_enums, "MSO_AUTO_SIZE")
        MSO_ANCHOR = getattr(text_enums, "MSO_ANCHOR")
        PP_ALIGN = getattr(text_enums, "PP_ALIGN")
        util = importlib.import_module("pptx.util")
        Emu = getattr(util, "Emu")
        Pt = getattr(util, "Pt")
    except Exception as e:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="python-pptx is required to generate PPTX output",
            details={"error": str(e)},
        )

    pages = ir.get("pages")
    if not isinstance(pages, list) or not pages:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="IR is missing pages[]",
        )

    first_page = pages[0] if isinstance(pages[0], dict) else None
    if not first_page:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="IR pages[0] is invalid",
        )

    text_erase_mode_id = str(text_erase_mode or "fill").strip().lower()
    if text_erase_mode_id not in {"smart", "fill"}:
        text_erase_mode_id = "fill"

    scanned_page_mode_id = str(scanned_page_mode or "segmented").strip().lower()
    if scanned_page_mode_id in {"chunk", "chunked", "split", "blocks"}:
        scanned_page_mode_id = "segmented"
    if scanned_page_mode_id in {"page", "full", "full_page"}:
        scanned_page_mode_id = "fullpage"
    if scanned_page_mode_id not in {"segmented", "fullpage"}:
        scanned_page_mode_id = "segmented"

    try:
        first_w_pt = float(first_page.get("page_width_pt") or 0.0)
        first_h_pt = float(first_page.get("page_height_pt") or 0.0)
    except Exception as e:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="IR page dimensions are invalid",
            details={"error": str(e)},
        )

    if first_w_pt <= 0 or first_h_pt <= 0:
        raise AppException(
            code=ErrorCode.CONVERSION_FAILED,
            message="IR page dimensions are missing",
            details={"page_width_pt": first_w_pt, "page_height_pt": first_h_pt},
        )

    out_path = _as_path(output_pptx_path)
    _ensure_parent_dir(out_path)

    artifacts = (
        _as_path(artifacts_dir)
        if artifacts_dir is not None
        else (out_path.parent / "artifacts")
    )
    artifacts.mkdir(parents=True, exist_ok=True)

    prs = Presentation()

    if force_16x9:
        # 13.333" x 7.5" is the common widescreen (16:9) size.
        slide_w_emu = int(round(13.333 * _EMU_PER_INCH))
        slide_h_emu = int(round(7.5 * _EMU_PER_INCH))
    else:
        # Default: 1:1 mapping with PDF points (8.5x11 for letter, etc.).
        slide_w_emu = int(round(first_w_pt * _EMU_PER_PT))
        slide_h_emu = int(round(first_h_pt * _EMU_PER_PT))

    prs.slide_width = Emu(slide_w_emu)
    prs.slide_height = Emu(slide_h_emu)
    _set_slide_size_type(prs, slide_w_emu=slide_w_emu, slide_h_emu=slide_h_emu)

    blank_layout = prs.slide_layouts[6]
    source_pdf = _as_path(str(ir.get("source_pdf") or ""))
    total_pages = sum(1 for page in pages if isinstance(page, dict))
    done_pages = 0

    for page in pages:
        if not isinstance(page, dict):
            continue
        page_index = int(page.get("page_index") or 0)
        page_w_pt = float(page.get("page_width_pt") or first_w_pt)
        page_h_pt = float(page.get("page_height_pt") or first_h_pt)

        transform = _build_transform(
            page_width_pt=page_w_pt,
            page_height_pt=page_h_pt,
            slide_width_emu=slide_w_emu,
            slide_height_emu=slide_h_emu,
        )

        slide = prs.slides.add_slide(blank_layout)
        has_text_layer = bool(page.get("has_text_layer"))
        page_elements = [
            el for el in (page.get("elements") or []) if isinstance(el, dict)
        ]
        has_mineru_elements = any(
            str(el.get("source") or "").strip().lower() == "mineru"
            for el in page_elements
        )

        if not has_text_layer:
            overlay_scanned_image_crops = scanned_page_mode_id != "fullpage"
            # Scanned page strategy: render page image, erase OCR/image areas
            # in the render, then overlay cropped images + editable text.
            render_path = artifacts / "page_renders" / f"page-{page_index:04d}.png"
            pix = _render_pdf_page_png(
                source_pdf,
                page_index=page_index,
                dpi=int(scanned_render_dpi),
                out_path=render_path,
            )

            bg_left = int(round(transform.offset_x_emu))
            bg_top = int(round(transform.offset_y_emu))
            bg_w = int(round(page_w_pt * _EMU_PER_PT * transform.scale))
            bg_h = int(round(page_h_pt * _EMU_PER_PT * transform.scale))

            # Collect OCR text blocks + stats for heuristics.
            ocr_text_elements = [
                el
                for el in _iter_page_elements(page, type_name="text")
                if str(el.get("source") or "") == "ocr"
            ]
            baseline_ocr_h_pt = _estimate_baseline_ocr_line_height_pt(
                ocr_text_elements=ocr_text_elements,
                page_w_pt=float(page_w_pt),
            )

            def _text_coverage_ratio(bb: list[float]) -> tuple[float, int]:
                """Return (overlap_area_ratio, ocr_items_inside_count) for a bbox.

                Used to reject image-region candidates that are actually paragraph
                text blocks or card backgrounds. Coverage is computed against OCR
                text boxes in PDF point coordinates.
                """

                if not ocr_text_elements:
                    return (0.0, 0)
                try:
                    x0, y0, x1, y1 = _coerce_bbox_pt(bb)
                except Exception:
                    return (0.0, 0)
                area = float(max(1.0, (x1 - x0) * (y1 - y0)))
                # Expand OCR bboxes a bit to account for line spacing gaps,
                # which otherwise underestimates text coverage.
                pad = max(1.0, min(6.0, 0.18 * float(baseline_ocr_h_pt)))
                overlap = 0.0
                count = 0
                for tel in ocr_text_elements:
                    bbox_pt = tel.get("bbox_pt")
                    if not isinstance(bbox_pt, list) or len(bbox_pt) != 4:
                        continue
                    try:
                        tx0, ty0, tx1, ty1 = _coerce_bbox_pt(bbox_pt)
                    except Exception:
                        continue
                    text_value = str(tel.get("text") or "")
                    if _is_inline_short_token(text_value):
                        continue

                    text_area = max(1.0, float((tx1 - tx0) * (ty1 - ty0)))
                    cx = (tx0 + tx1) / 2.0
                    cy = (ty0 + ty1) / 2.0

                    tx0 -= pad
                    ty0 -= pad
                    tx1 += pad
                    ty1 += pad
                    ix0 = max(x0, tx0)
                    iy0 = max(y0, ty0)
                    ix1 = min(x1, tx1)
                    iy1 = min(y1, ty1)
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    inter = float((ix1 - ix0) * (iy1 - iy0))
                    overlap += inter

                    center_inside = cx >= x0 and cx <= x1 and cy >= y0 and cy <= y1
                    if center_inside or (inter / text_area) >= 0.18:
                        count += 1
                overlap = min(overlap, area)
                return (float(overlap) / area, int(count))

            def _text_inside_counts(bb: list[float]) -> tuple[int, int]:
                """Return (items_inside_count, cjk_items_inside_count) for a bbox.

                This complements area-based coverage with linguistic hints so we can
                reject large mixed regions that accidentally swallow CJK body text.
                """

                if not ocr_text_elements:
                    return (0, 0)
                try:
                    x0, y0, x1, y1 = _coerce_bbox_pt(bb)
                except Exception:
                    return (0, 0)
                inside = 0
                cjk_inside = 0
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

                    inside += 1
                    if _contains_cjk(text_value):
                        cjk_inside += 1
                return (int(inside), int(cjk_inside))

            has_full_page_bg_image = any(
                _is_near_full_page_bbox_pt(
                    el.get("bbox_pt"), page_w_pt=page_w_pt, page_h_pt=page_h_pt
                )
                for el in _iter_page_elements(page, type_name="image")
            )

            image_region_infos = _build_scanned_image_region_infos(
                page=page,
                render_path=render_path,
                artifacts_dir=artifacts,
                page_index=page_index,
                page_w_pt=page_w_pt,
                page_h_pt=page_h_pt,
                scanned_render_dpi=int(scanned_render_dpi),
                baseline_ocr_h_pt=float(baseline_ocr_h_pt),
                ocr_text_elements=ocr_text_elements,
                has_full_page_bg_image=has_full_page_bg_image,
                text_coverage_ratio_fn=_text_coverage_ratio,
                text_inside_counts_fn=_text_inside_counts,
            )
            ocr_text_elements = _filter_scanned_ocr_text_elements(
                ocr_text_elements=ocr_text_elements,
                image_region_infos=image_region_infos,
                baseline_ocr_h_pt=float(baseline_ocr_h_pt),
            )
            ocr_text_elements = _dedupe_scanned_ocr_text_elements(
                ocr_text_elements=ocr_text_elements,
                baseline_ocr_h_pt=float(baseline_ocr_h_pt),
            )

            # Build editable text items for scanned-page overlay. We first erase
            # OCR text in the rendered background image, then place cropped images
            # and editable text above it.

            text_erase_bboxes_pt: list[list[float]] = []
            text_items: list[
                tuple[dict[str, Any], list[float], str, tuple[int, int, int]]
            ] = []
            is_fill_mode = text_erase_mode_id == "fill"

            for el in ocr_text_elements:
                bbox_pt = el.get("bbox_pt")
                try:
                    x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
                except Exception:
                    continue

                raw_text = str(el.get("text") or "")
                text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
                text = "\n".join(
                    [line.strip() for line in text.split("\n") if line.strip()]
                ).strip()
                if not text:
                    continue

                bbox_w_pt = max(1.0, x1 - x0)
                bbox_h_pt = max(1.0, y1 - y0)

                # Sample the local background for masking.
                bg_rgb = _sample_bbox_background_rgb(
                    pix,
                    bbox_pt=[x0, y0, x1, y1],
                    page_height_pt=page_h_pt,
                    dpi=int(scanned_render_dpi),
                )

                # Expand erase region to remove anti-aliased glyph halos.
                # Use anisotropic padding so line tails don't leave gray remnants.
                pad_x_pt, pad_y_pt = _compute_text_erase_padding_pt(
                    bbox_h_pt=bbox_h_pt,
                    text_erase_mode=text_erase_mode_id,
                )
                text_erase_bboxes_pt.append(
                    [x0 - pad_x_pt, y0 - pad_y_pt, x1 + pad_x_pt, y1 + pad_y_pt]
                )

                text_items.append((el, [x0, y0, x1, y1], text, bg_rgb))

            def _merge_text_erase_bboxes(
                boxes: list[list[float]], *, gap_pt: float
            ) -> list[list[float]]:
                """Merge nearby same-line erase boxes to improve wipe completeness."""

                merged = [
                    list(_coerce_bbox_pt(bb))
                    for bb in boxes
                    if isinstance(bb, list) and len(bb) == 4
                ]
                if len(merged) <= 1:
                    return merged

                gap_pt = max(0.0, float(gap_pt))
                changed = True
                while changed:
                    changed = False
                    merged.sort(key=lambda b: (b[1], b[0]))
                    out: list[list[float]] = []
                    for bb in merged:
                        x0, y0, x1, y1 = _coerce_bbox_pt(bb)
                        did_merge = False
                        for i, ub in enumerate(out):
                            ux0, uy0, ux1, uy1 = _coerce_bbox_pt(ub)
                            y_overlap = min(y1, uy1) - max(y0, uy0)
                            min_h = max(1.0, min(y1 - y0, uy1 - uy0))
                            if y_overlap < (0.40 * min_h):
                                continue
                            if x0 > ux1:
                                x_gap = float(x0 - ux1)
                            elif ux0 > x1:
                                x_gap = float(ux0 - x1)
                            else:
                                x_gap = 0.0
                            if x_gap > gap_pt:
                                continue
                            out[i] = [
                                min(x0, ux0),
                                min(y0, uy0),
                                max(x1, ux1),
                                max(y1, uy1),
                            ]
                            did_merge = True
                            changed = True
                            break
                        if not did_merge:
                            out.append([x0, y0, x1, y1])
                    merged = out

                return merged

            if is_fill_mode:
                # For fill mode, prefer local boxes so color fill remains local,
                # but if OCR output is extremely fragmented (word-level boxes)
                # also add a mild merge pass to avoid "text ghosts" between
                # adjacent bboxes.
                erase_bboxes_for_background = list(text_erase_bboxes_pt)
                # AI OCR outputs can also leave small gaps between line boxes,
                # causing visible "double text" (residual background glyphs
                # + editable overlay). Apply a conservative merge for medium-
                # sized pages as well, but keep geometry guards so we don't
                # wipe across columns.
                if len(text_erase_bboxes_pt) >= 60:
                    merged_fill_bboxes_pt = _merge_text_erase_bboxes(
                        text_erase_bboxes_pt,
                        gap_pt=max(1.5, 0.42 * float(baseline_ocr_h_pt)),
                    )
                    for bb in merged_fill_bboxes_pt:
                        try:
                            mx0, my0, mx1, my1 = _coerce_bbox_pt(bb)
                        except Exception:
                            continue
                        if (mx1 - mx0) >= 0.92 * float(page_w_pt):
                            continue
                        if (my1 - my0) >= 3.8 * float(baseline_ocr_h_pt):
                            continue
                        erase_bboxes_for_background.append([mx0, my0, mx1, my1])
            else:
                merged_text_erase_bboxes_pt = _merge_text_erase_bboxes(
                    text_erase_bboxes_pt,
                    gap_pt=max(2.0, 0.75 * float(baseline_ocr_h_pt)),
                )
                # Keep both merged and original line boxes. Merged boxes improve wipe
                # continuity, while raw line boxes help detect and force-clean local
                # leftovers that can still cause visible text overlap.
                erase_bboxes_for_background = list(merged_text_erase_bboxes_pt) + list(
                    text_erase_bboxes_pt
                )

            # 1) Background image (after erase)
            protect_bboxes_for_erase: list[list[float]] = []
            for info in image_region_infos:
                if not info.shape_confirmed:
                    continue
                try:
                    ix0, iy0, ix1, iy1 = _coerce_bbox_pt(info.bbox_pt)
                except Exception:
                    continue
                iw = float(ix1 - ix0)
                ih = float(iy1 - iy0)
                area_ratio = max(0.0, iw * ih) / max(
                    1.0, float(page_w_pt) * float(page_h_pt)
                )
                # Only protect clearly image-dominant regions. Small icon-like crops
                # often overlap heading/text bboxes and can block text cleanup.
                if area_ratio < 0.030:
                    continue
                protect_bboxes_for_erase.append([ix0, iy0, ix1, iy1])

            cleaned_render_path = _erase_regions_in_render_image(
                render_path,
                out_path=artifacts
                / "page_renders"
                / f"page-{page_index:04d}.clean.png",
                # Erase OCR text only. Do NOT erase image regions in the background:
                # image crops are overlaid later, and region erase can introduce
                # large inpaint artifacts on complex templates.
                erase_bboxes_pt=erase_bboxes_for_background,
                # Never modify pixels inside confirmed *large* image crops.
                protect_bboxes_pt=protect_bboxes_for_erase,
                page_height_pt=page_h_pt,
                dpi=int(scanned_render_dpi),
                text_erase_mode=text_erase_mode_id,
            )

            if overlay_scanned_image_crops:
                # If icon-like crops use transparent background, clear their original
                # background area from the base render first to avoid double-background.
                transparent_bg_regions = [
                    info.bbox_pt
                    for info in image_region_infos
                    if info.background_removed
                ]
                if transparent_bg_regions:
                    cleaned_render_path = _clear_regions_for_transparent_crops(
                        cleaned_render_path=cleaned_render_path,
                        out_path=artifacts
                        / "page_renders"
                        / f"page-{page_index:04d}.clean.icons-bg-cleared.png",
                        regions_pt=transparent_bg_regions,
                        pix=pix,
                        page_height_pt=page_h_pt,
                        dpi=int(scanned_render_dpi),
                    )

            slide.shapes.add_picture(
                str(cleaned_render_path),
                Emu(bg_left),
                Emu(bg_top),
                Emu(bg_w),
                Emu(bg_h),
            )

            # 2) Cropped images
            if overlay_scanned_image_crops:
                for info in image_region_infos:
                    try:
                        left, top, width, height = _bbox_pt_to_slide_emu(
                            info.bbox_pt, transform=transform
                        )
                    except Exception:
                        continue
                    if width <= 0 or height <= 0:
                        continue
                    slide.shapes.add_picture(
                        str(info.crop_path),
                        Emu(left),
                        Emu(top),
                        Emu(width),
                        Emu(height),
                    )

            # 3) Editable text boxes.
            for el, bbox_pt, text, (r, g, b) in text_items:
                try:
                    left, top, width, height = _bbox_pt_to_slide_emu(
                        bbox_pt, transform=transform
                    )
                except Exception:
                    continue
                if width <= 0 or height <= 0:
                    continue

                x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
                bbox_w_pt = max(1.0, x1 - x0)
                bbox_h_pt = max(1.0, y1 - y0)

                # Heading detection: big text near the top.
                is_heading = (
                    y0 <= 0.22 * float(page_h_pt)
                    and bbox_h_pt >= 1.6 * float(baseline_ocr_h_pt)
                    and len(text) <= 40
                )

                # We'll nudge OCR text boxes slightly upward and extend their
                # height by a tiny amount (see below). Feed that slack into the
                # font-fitting step so we don't pick an overly small font.
                fit_bbox_h_pt = float(bbox_h_pt) + float(
                    min(1.2, 0.06 * float(bbox_h_pt))
                )

                # IMPORTANT: decide wrapping based on the *original* OCR bbox.
                # The extra fit slack is only for font fitting; using it for the
                # wrap decision can cause spurious breaks (e.g. putting "：" on
                # its own line) on slightly padded single-line headings.
                wrap_hint = _prefer_wrap_for_ocr_text(
                    text=text,
                    bbox_w_pt=bbox_w_pt,
                    bbox_h_pt=bbox_h_pt,
                    baseline_ocr_h_pt=float(baseline_ocr_h_pt),
                )

                text_to_render, font_size_pt, wrap = _fit_ocr_text_style(
                    text=text,
                    bbox_w_pt=bbox_w_pt,
                    bbox_h_pt=fit_bbox_h_pt,
                    baseline_ocr_h_pt=float(baseline_ocr_h_pt),
                    is_heading=bool(is_heading),
                    wrap_override=wrap_hint,
                )
                if not text_to_render.strip():
                    continue

                # OCR text in Office/WPS usually appears slightly lower than image-rendered
                # glyphs due to font ascent metrics. Nudge up a tiny amount.
                nudge_up_pt = min(
                    2.2,
                    max(
                        0.6,
                        0.08 * float(bbox_h_pt),
                        0.10 * float(font_size_pt),
                    ),
                )
                nudge_emu = int(
                    round(float(nudge_up_pt) * _EMU_PER_PT * transform.scale)
                )
                textbox_top = max(0, int(top) - nudge_emu)
                textbox_height = int(height) + nudge_emu

                # Add right-side tolerance to avoid last-char unexpected wraps
                # caused by viewer/font metric differences.
                if wrap:
                    nudge_right_pt = min(3.2, max(1.2, 0.07 * float(bbox_h_pt)))
                else:
                    # Single-line text is where Office/WPS most often re-wraps one
                    # trailing character. Keep a larger right guard in this case.
                    nudge_right_pt = min(
                        8.0,
                        max(
                            3.0,
                            0.16 * float(bbox_h_pt),
                            0.50 * float(font_size_pt),
                        ),
                    )
                nudge_right_emu = int(
                    round(float(nudge_right_pt) * _EMU_PER_PT * transform.scale)
                )
                textbox_left = int(left)
                textbox_width = int(width) + nudge_right_emu
                max_box_w = max(1, int(slide_w_emu) - textbox_left)
                textbox_width = max(1, min(textbox_width, max_box_w))

                tx = slide.shapes.add_textbox(
                    Emu(textbox_left),
                    Emu(textbox_top),
                    Emu(textbox_width),
                    Emu(textbox_height),
                )
                tx.fill.background()
                tx.line.fill.background()
                tf = tx.text_frame
                # Keep top alignment so OCR bboxes map visually (WPS/Office differ
                # on default vertical anchoring).
                try:
                    tf.vertical_anchor = MSO_ANCHOR.TOP
                except Exception:
                    pass
                # We insert explicit line breaks when wrapping is needed. Keeping
                # `word_wrap=True` lets Office/WPS reflow text differently across
                # platforms/fonts (often moving a trailing punctuation like "："
                # onto its own line). Disable auto wrapping for more stable
                # scan-to-PPT visual fidelity.
                tf.word_wrap = False
                # Disable viewer auto-size to reduce text-box drift between Office/WPS.
                tf.auto_size = MSO_AUTO_SIZE.NONE
                tf.margin_left = 0
                tf.margin_right = 0
                tf.margin_top = 0
                tf.margin_bottom = 0
                tf.text = text_to_render

                for p in tf.paragraphs:
                    try:
                        if is_heading:
                            p.alignment = PP_ALIGN.CENTER
                    except Exception:
                        pass
                    # Reduce unexpected spacing differences across viewers.
                    try:
                        p.space_before = Pt(0)
                        p.space_after = Pt(0)
                    except Exception:
                        pass
                    try:
                        p.line_spacing = 1.0
                    except Exception:
                        pass

                    for run in p.runs:
                        font = run.font
                        font.size = Pt(float(font_size_pt))
                        if _contains_cjk(text):
                            font.name = "Microsoft YaHei"
                        else:
                            font.name = _map_font_name(el.get("font_name")) or "Arial"
                        font.bold = (
                            True
                            if is_heading
                            else (bool(el.get("bold")) if "bold" in el else None)
                        )
                        font.italic = bool(el.get("italic")) if "italic" in el else None

                        rgb = _hex_to_rgb(el.get("color"))
                        if rgb is None:
                            rgb = (
                                (0, 0, 0)
                                if (0.2126 * r + 0.7152 * g + 0.0722 * b) >= 128
                                else (255, 255, 255)
                            )
                        else:
                            dr = int(rgb[0]) - int(r)
                            dg = int(rgb[1]) - int(g)
                            db = int(rgb[2]) - int(b)
                            if (dr * dr + dg * dg + db * db) < (35 * 35):
                                rgb = (
                                    (0, 0, 0)
                                    if (0.2126 * r + 0.7152 * g + 0.0722 * b) >= 128
                                    else (255, 255, 255)
                                )
                        font.color.rgb = RGBColor(*rgb)

            _export_final_preview_page_image(
                page=page,
                page_index=page_index,
                page_w_pt=page_w_pt,
                page_h_pt=page_h_pt,
                source_pdf=source_pdf,
                artifacts_dir=artifacts,
                dpi=int(scanned_render_dpi),
                scanned_image_region_crops=[
                    (list(info.bbox_pt), info.crop_path) for info in image_region_infos
                ]
                if overlay_scanned_image_crops
                else [],
            )
            continue

        # Text-based page: place elements directly.
        mineru_background_placed = False
        mineru_render_pix: Any | None = None
        if has_mineru_elements and source_pdf.exists():
            try:
                render_path = (
                    artifacts / "page_renders" / f"page-{page_index:04d}.mineru.png"
                )
                mineru_render_pix = _render_pdf_page_png(
                    source_pdf,
                    page_index=page_index,
                    dpi=int(scanned_render_dpi),
                    out_path=render_path,
                )

                text_erase_bboxes_pt: list[list[float]] = []
                protect_bboxes_pt: list[list[float]] = []

                for el in _iter_page_elements(page, type_name="text"):
                    if str(el.get("source") or "").strip().lower() != "mineru":
                        continue
                    try:
                        x0, y0, x1, y1 = _coerce_bbox_pt(el.get("bbox_pt"))
                    except Exception:
                        continue
                    bbox_h_pt = max(1.0, y1 - y0)
                    pad_x_pt, pad_y_pt = _compute_text_erase_padding_pt(
                        bbox_h_pt=bbox_h_pt,
                        text_erase_mode=text_erase_mode_id,
                    )
                    text_erase_bboxes_pt.append(
                        [x0 - pad_x_pt, y0 - pad_y_pt, x1 + pad_x_pt, y1 + pad_y_pt]
                    )

                for el in _iter_page_elements(page, type_name="image"):
                    if str(el.get("source") or "").strip().lower() != "mineru":
                        continue
                    try:
                        ix0, iy0, ix1, iy1 = _coerce_bbox_pt(el.get("bbox_pt"))
                    except Exception:
                        continue
                    protect_bboxes_pt.append([ix0, iy0, ix1, iy1])

                cleaned_render_path = _erase_regions_in_render_image(
                    render_path,
                    out_path=artifacts
                    / "page_renders"
                    / f"page-{page_index:04d}.mineru.clean.png",
                    erase_bboxes_pt=text_erase_bboxes_pt,
                    protect_bboxes_pt=protect_bboxes_pt,
                    page_height_pt=page_h_pt,
                    dpi=int(scanned_render_dpi),
                    text_erase_mode=text_erase_mode_id,
                )

                bg_left = int(round(transform.offset_x_emu))
                bg_top = int(round(transform.offset_y_emu))
                bg_w = int(round(page_w_pt * _EMU_PER_PT * transform.scale))
                bg_h = int(round(page_h_pt * _EMU_PER_PT * transform.scale))
                slide.shapes.add_picture(
                    str(cleaned_render_path),
                    Emu(bg_left),
                    Emu(bg_top),
                    Emu(bg_w),
                    Emu(bg_h),
                )
                mineru_background_placed = True
            except Exception:
                mineru_background_placed = False
                mineru_render_pix = None

        for el in _iter_page_elements(page, type_name="image"):
            bbox_pt = el.get("bbox_pt")
            image_path = el.get("image_path")
            if not image_path:
                continue
            img_path = _as_path(str(image_path))
            if not img_path.is_absolute():
                candidate = artifacts / img_path
                if candidate.exists():
                    img_path = candidate
            if not img_path.exists():
                continue
            try:
                left, top, width, height = _bbox_pt_to_slide_emu(
                    bbox_pt, transform=transform
                )
            except Exception:
                continue
            slide.shapes.add_picture(
                str(img_path), Emu(left), Emu(top), Emu(width), Emu(height)
            )

        for el in _iter_page_elements(page, type_name="table"):
            bbox_pt = el.get("bbox_pt")
            try:
                rows = int(el.get("rows") or 0)
                cols = int(el.get("cols") or 0)
            except Exception:
                rows, cols = 0, 0
            if rows <= 0 or cols <= 0:
                continue
            try:
                left, top, width, height = _bbox_pt_to_slide_emu(
                    bbox_pt, transform=transform
                )
            except Exception:
                continue

            table_shape = slide.shapes.add_table(
                rows, cols, Emu(left), Emu(top), Emu(width), Emu(height)
            )
            table = table_shape.table

            # Best-effort column/row sizing from cell bboxes if available.
            cells = el.get("cells") or []
            if isinstance(cells, list) and cells:
                col_widths_pt = [0.0 for _ in range(cols)]
                row_heights_pt = [0.0 for _ in range(rows)]
                for cell in cells:
                    if not isinstance(cell, dict):
                        continue
                    r = int(cell.get("r") or 0)
                    c = int(cell.get("c") or 0)
                    if r < 0 or r >= rows or c < 0 or c >= cols:
                        continue
                    try:
                        x0, y0, x1, y1 = _coerce_bbox_pt(cell.get("bbox_pt"))
                    except Exception:
                        continue
                    col_widths_pt[c] = max(col_widths_pt[c], x1 - x0)
                    row_heights_pt[r] = max(row_heights_pt[r], y1 - y0)

                # Fall back to uniform sizing when bbox data is missing/degenerate.
                if sum(col_widths_pt) <= 0:
                    col_widths_pt = [page_w_pt / cols for _ in range(cols)]
                if sum(row_heights_pt) <= 0:
                    row_heights_pt = [page_h_pt / rows for _ in range(rows)]

                col_widths_emu = [
                    int(round(w * _EMU_PER_PT * transform.scale)) for w in col_widths_pt
                ]
                row_heights_emu = [
                    int(round(h * _EMU_PER_PT * transform.scale))
                    for h in row_heights_pt
                ]

                # Adjust last row/col to account for rounding so totals match table bbox.
                if col_widths_emu:
                    col_widths_emu[-1] += int(width - sum(col_widths_emu))
                if row_heights_emu:
                    row_heights_emu[-1] += int(height - sum(row_heights_emu))

                for c, w in enumerate(col_widths_emu):
                    table.columns[c].width = Emu(max(0, w))
                for r, h in enumerate(row_heights_emu):
                    table.rows[r].height = Emu(max(0, h))

                for cell in cells:
                    if not isinstance(cell, dict):
                        continue
                    r = int(cell.get("r") or 0)
                    c = int(cell.get("c") or 0)
                    if r < 0 or r >= rows or c < 0 or c >= cols:
                        continue
                    text = str(cell.get("text") or "")
                    table.cell(r, c).text = text
            else:
                # No structured cells; leave empty for now.
                pass

        for el in _iter_page_elements(page, type_name="text"):
            bbox_pt = el.get("bbox_pt")
            try:
                left, top, width, height = _bbox_pt_to_slide_emu(
                    bbox_pt, transform=transform
                )
            except Exception:
                continue

            x0, y0, x1, y1 = _coerce_bbox_pt(bbox_pt)
            source_id = str(el.get("source") or "").strip().lower()
            is_mineru_text = source_id == "mineru"
            is_ocr_text = source_id == "ocr"
            raw_text = str(el.get("text") or "")
            text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
            if is_mineru_text:
                text = "\n".join(
                    [line.strip() for line in text.split("\n") if line.strip()]
                ).strip()
            elif is_ocr_text:
                text = _normalize_ocr_text_for_render(text)
            else:
                text = text.replace("\n", " ").strip()
            if not text:
                continue

            tx = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
            tf = tx.text_frame
            bbox_w_pt = max(1.0, x1 - x0)
            bbox_h_pt = max(1.0, y1 - y0)
            text_to_render = text
            sampled_bg_rgb: tuple[int, int, int] | None = None
            sampled_text_rgb: tuple[int, int, int] | None = None
            if is_mineru_text:
                mineru_block_type = (
                    str(el.get("mineru_block_type") or "").strip().lower()
                )
                is_bullet_like = text.lstrip().startswith(("-", "•", "·", "●"))
                plain_len = len(text.replace("\n", ""))
                text_level_raw = el.get("mineru_text_level")
                text_level: int | None = None
                if text_level_raw is not None:
                    try:
                        text_level = int(text_level_raw)
                    except Exception:
                        text_level = None
                is_heading = bool(
                    mineru_block_type in {"title", "heading", "header", "h1", "h2"}
                ) or (
                    (text_level is not None and text_level <= 2 and plain_len <= 60)
                    or (
                        y0 <= 0.22 * float(page_h_pt)
                        and bbox_h_pt >= 18.0
                        and plain_len <= 56
                        and not is_bullet_like
                    )
                )
                # Distinguish page hero title vs card/section titles by geometry.
                is_primary_heading = bool(
                    is_heading
                    and y0 <= 0.16 * float(page_h_pt)
                    and bbox_w_pt >= 0.34 * float(page_w_pt)
                )

                # For non-heading text, always fit with wrapping. For heading text,
                # allow wrapping when it materially improves usable size.
                wrap_for_fit = bool(not is_heading)
                max_body_pt = min(
                    96.0 if is_primary_heading else 72.0,
                    max(7.0, (0.98 if is_heading else 0.94) * float(bbox_h_pt)),
                )
                min_body_pt = 6.0
                prefit_font_size_pt: float | None = None

                if is_heading and (not is_primary_heading) and plain_len >= 14:
                    single_line_pt = _fit_font_size_pt(
                        text,
                        bbox_w_pt=bbox_w_pt,
                        bbox_h_pt=bbox_h_pt,
                        wrap=False,
                        min_pt=min_body_pt,
                        max_pt=max_body_pt,
                        width_fit_ratio=1.00,
                        height_fit_ratio=0.995,
                    )
                    wrapped_pt = _fit_font_size_pt(
                        text,
                        bbox_w_pt=bbox_w_pt,
                        bbox_h_pt=bbox_h_pt,
                        wrap=True,
                        min_pt=min_body_pt,
                        max_pt=max_body_pt,
                        width_fit_ratio=1.01,
                        height_fit_ratio=0.96,
                    )
                    wrapped_lines, _ = _measure_text_lines(
                        text,
                        max_width_pt=max(1.0, 0.99 * bbox_w_pt),
                        font_size_pt=float(wrapped_pt),
                        wrap=True,
                    )
                    if (
                        wrapped_lines >= 2
                        and wrapped_lines <= 3
                        and wrapped_pt
                        >= max(single_line_pt + 1.2, 1.18 * single_line_pt)
                    ):
                        wrap_for_fit = True
                        prefit_font_size_pt = float(wrapped_pt)
                    else:
                        wrap_for_fit = False
                        prefit_font_size_pt = float(single_line_pt)

                if prefit_font_size_pt is not None:
                    font_size_pt = float(prefit_font_size_pt)
                else:
                    font_size_pt = _fit_font_size_pt(
                        text,
                        bbox_w_pt=bbox_w_pt,
                        bbox_h_pt=bbox_h_pt,
                        wrap=wrap_for_fit,
                        min_pt=min_body_pt,
                        max_pt=max_body_pt,
                        width_fit_ratio=1.01 if wrap_for_fit else 1.00,
                        height_fit_ratio=0.96 if wrap_for_fit else 0.995,
                    )
                if wrap_for_fit:
                    # Lock explicit line breaks so Office/WPS viewers don't
                    # reflow CJK text differently and cause overlap/shift.
                    for _ in range(12):
                        candidate_text = _wrap_text_to_width(
                            text,
                            max_width_pt=max(1.0, 1.01 * bbox_w_pt),
                            font_size_pt=float(font_size_pt),
                        )
                        candidate_lines = [
                            line for line in candidate_text.splitlines() if line.strip()
                        ]
                        if not candidate_lines:
                            candidate_lines = [text]
                            candidate_text = text
                        line_height = 1.18 if _contains_cjk(text) else 1.15
                        total_h = (
                            float(len(candidate_lines))
                            * float(font_size_pt)
                            * line_height
                        )
                        if total_h <= (0.985 * bbox_h_pt):
                            text_to_render = candidate_text
                            break
                        font_size_pt = max(
                            float(min_body_pt), float(font_size_pt) - 0.35
                        )
                    else:
                        text_to_render = candidate_text if candidate_text else text
                wrap = bool(wrap_for_fit)
            elif is_ocr_text:
                compact_len = _compact_text_length(text)
                is_heading = bool(
                    y0 <= 0.20 * float(page_h_pt)
                    and bbox_h_pt >= 1.45 * float(baseline_ocr_h_pt)
                    and compact_len <= 56
                )
                text_to_render, font_size_pt, wrap = _fit_ocr_text_style(
                    text=text,
                    bbox_w_pt=bbox_w_pt,
                    bbox_h_pt=bbox_h_pt,
                    baseline_ocr_h_pt=float(baseline_ocr_h_pt),
                    is_heading=is_heading,
                )

                if (
                    has_text_layer
                    and source_pdf.exists()
                    and (mineru_render_pix is not None)
                ):
                    try:
                        sampled_bg_rgb = _sample_bbox_background_rgb(
                            mineru_render_pix,
                            bbox_pt=bbox_pt,
                            page_height_pt=page_h_pt,
                            dpi=int(scanned_render_dpi),
                        )
                        sampled_text_rgb = _sample_bbox_text_rgb(
                            mineru_render_pix,
                            bbox_pt=bbox_pt,
                            page_height_pt=page_h_pt,
                            dpi=int(scanned_render_dpi),
                            bg_rgb=sampled_bg_rgb,
                        )
                    except Exception:
                        sampled_bg_rgb = None
                        sampled_text_rgb = None
            else:
                wrap = False
                font_size_pt = _infer_font_size_pt(el, bbox_h_pt=bbox_h_pt)
                is_heading = False

            tf.word_wrap = bool(wrap)
            if is_mineru_text:
                tf.auto_size = MSO_AUTO_SIZE.NONE
                try:
                    tf.vertical_anchor = MSO_ANCHOR.TOP
                except Exception:
                    pass
            else:
                tf.auto_size = MSO_AUTO_SIZE.NONE
            tf.margin_left = 0
            tf.margin_right = 0
            tf.margin_top = 0
            tf.margin_bottom = 0
            tf.text = text_to_render

            if is_mineru_text:
                for p in tf.paragraphs:
                    try:
                        if is_primary_heading:
                            p.alignment = PP_ALIGN.CENTER
                    except Exception:
                        pass
                    try:
                        p.line_spacing = 1.0
                        p.space_before = Pt(0)
                        p.space_after = Pt(0)
                    except Exception:
                        pass

            mapped_font = _map_font_name(el.get("font_name"))
            rgb = _hex_to_rgb(el.get("color"))
            if is_mineru_text and mineru_render_pix is not None:
                try:
                    sampled_bg_rgb = _sample_bbox_background_rgb(
                        mineru_render_pix,
                        bbox_pt=bbox_pt,
                        page_height_pt=page_h_pt,
                        dpi=int(scanned_render_dpi),
                    )
                    sampled_text_rgb = _sample_bbox_text_rgb(
                        mineru_render_pix,
                        bbox_pt=bbox_pt,
                        page_height_pt=page_h_pt,
                        dpi=int(scanned_render_dpi),
                        bg_rgb=sampled_bg_rgb,
                    )
                except Exception:
                    sampled_bg_rgb = None
                    sampled_text_rgb = None
            elif (
                is_ocr_text
                and sampled_bg_rgb is None
                and has_text_layer
                and source_pdf.exists()
            ):
                # Best-effort local color sampling for OCR elements on text-layer pages.
                # Reuse mineru render when available (already aligned to source PDF).
                if mineru_render_pix is not None:
                    try:
                        sampled_bg_rgb = _sample_bbox_background_rgb(
                            mineru_render_pix,
                            bbox_pt=bbox_pt,
                            page_height_pt=page_h_pt,
                            dpi=int(scanned_render_dpi),
                        )
                        sampled_text_rgb = _sample_bbox_text_rgb(
                            mineru_render_pix,
                            bbox_pt=bbox_pt,
                            page_height_pt=page_h_pt,
                            dpi=int(scanned_render_dpi),
                            bg_rgb=sampled_bg_rgb,
                        )
                    except Exception:
                        sampled_bg_rgb = None
                        sampled_text_rgb = None
            if rgb is None and sampled_bg_rgb is not None:
                if sampled_text_rgb is not None:
                    rgb = sampled_text_rgb
                else:
                    rgb = _pick_contrasting_text_rgb(sampled_bg_rgb)
            elif rgb is not None and sampled_bg_rgb is not None:
                # If upstream color is too close to local background, prioritize
                # readability in the exported PPT.
                if _rgb_sq_distance(rgb, sampled_bg_rgb) < (32 * 32):
                    if sampled_text_rgb is not None:
                        rgb = sampled_text_rgb
                    else:
                        rgb = _pick_contrasting_text_rgb(sampled_bg_rgb)
            applied = False
            for p in tf.paragraphs:
                for run in p.runs:
                    font = run.font
                    font.size = Pt(float(font_size_pt))
                    if mapped_font:
                        font.name = mapped_font
                    elif is_mineru_text:
                        font.name = (
                            "Microsoft YaHei"
                            if _contains_cjk(text_to_render)
                            else "Arial"
                        )
                    elif is_ocr_text:
                        font.name = (
                            "Microsoft YaHei"
                            if _contains_cjk(text_to_render)
                            else "Arial"
                        )
                    font.bold = bool(el.get("bold")) if "bold" in el else None
                    font.italic = bool(el.get("italic")) if "italic" in el else None
                    if rgb:
                        font.color.rgb = RGBColor(*rgb)
                    applied = True

            if not applied:
                continue

        _export_final_preview_page_image(
            page=page,
            page_index=page_index,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
            source_pdf=source_pdf,
            artifacts_dir=artifacts,
            dpi=int(scanned_render_dpi),
        )
        done_pages += 1
        if progress_callback:
            try:
                progress_callback(done_pages, max(1, total_pages))
            except Exception:
                pass

    prs.save(str(out_path))
    return out_path
