from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pymupdf

from .geometry_utils import _bbox_pt_to_px
from .layout import _layout_page_signature, _to_page_map


def _build_ocr_effective_runtime_debug(
    *,
    ocr_manager: Any,
    fallback_provider: str | None,
) -> dict[str, Any]:
    debug: dict[str, Any] = {
        "configured_provider": getattr(ocr_manager, "provider_id", None) or fallback_provider or "unknown",
        "runtime_provider": fallback_provider or "unknown",
        "provider_chain": [],
        "paddle_doc_parser": None,
    }

    try:
        providers = getattr(ocr_manager, "providers", None)
        if isinstance(providers, list):
            debug["provider_chain"] = [type(provider).__name__ for provider in providers]
    except Exception:
        pass

    try:
        primary_provider = getattr(ocr_manager, "primary_provider", None)
    except Exception:
        primary_provider = None

    if primary_provider is None:
        try:
            debug["ai_provider_disabled"] = bool(
                getattr(ocr_manager, "ai_provider_disabled", False)
            )
            debug["ai_provider_disabled_reason"] = getattr(
                ocr_manager, "ai_provider_disabled_reason", None
            )
        except Exception:
            pass
        return debug

    runtime_provider_name = type(primary_provider).__name__
    if runtime_provider_name:
        debug["runtime_provider"] = runtime_provider_name

    provider_id = str(getattr(primary_provider, "provider_id", "") or "").lower()
    model = str(getattr(primary_provider, "model", "") or "").strip()
    is_paddle_vl = "paddleocr-vl" in model.lower()
    if provider_id == "paddle" or is_paddle_vl:
        debug["paddle_doc_parser"] = {
            "provider": provider_id or None,
            "requested_model": model or None,
            "effective_model": getattr(
                primary_provider,
                "_paddle_doc_effective_model",
                None,
            ),
            "pipeline_version": getattr(
                primary_provider,
                "_paddle_doc_pipeline_version",
                None,
            ),
            "server_url": getattr(
                primary_provider,
                "_paddle_doc_server_url",
                None,
            ),
            "backend": getattr(
                primary_provider,
                "_paddle_doc_backend",
                None,
            ),
            "last_predict": getattr(
                primary_provider,
                "_paddle_doc_last_predict_debug",
                None,
            ),
            "recent_predicts": getattr(
                primary_provider,
                "_paddle_doc_recent_predict_debug",
                None,
            ),
        }

    try:
        debug["ai_provider_disabled"] = bool(
            getattr(ocr_manager, "ai_provider_disabled", False)
        )
        debug["ai_provider_disabled_reason"] = getattr(
            ocr_manager, "ai_provider_disabled_reason", None
        )
        debug["last_provider_error"] = getattr(
            ocr_manager, "last_provider_error", None
        )
        quality_notes = getattr(ocr_manager, "last_quality_notes", [])
        debug["last_quality_notes"] = (
            list(quality_notes) if isinstance(quality_notes, list) else []
        )
    except Exception:
        pass

    return debug


def _draw_layout_assist_overlay(
    *,
    base_img: Any,
    page: dict[str, Any] | None,
    page_w_pt: float,
    page_h_pt: float,
) -> tuple[Any, dict[str, int]]:
    from PIL import ImageDraw

    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    stats = {
        "text": 0,
        "image": 0,
        "table": 0,
        "table_grids": 0,
        "image_regions": 0,
    }

    if isinstance(page, dict):
        elements = page.get("elements")
        if isinstance(elements, list):
            for el in elements:
                if not isinstance(el, dict):
                    continue
                el_type = str(el.get("type") or "").strip().lower()
                color = (
                    (220, 53, 69)
                    if el_type == "text"
                    else (52, 152, 219)
                    if el_type == "image"
                    else (46, 204, 113)
                    if el_type == "table"
                    else (120, 120, 120)
                )
                rect = _bbox_pt_to_px(
                    el.get("bbox_pt"),
                    page_w_pt=page_w_pt,
                    page_h_pt=page_h_pt,
                    img_w_px=img_w,
                    img_h_px=img_h,
                )
                if rect is None:
                    continue
                draw.rectangle(rect, outline=color, width=2)
                if el_type in stats:
                    stats[el_type] += 1

        for grid in page.get("table_grids") or []:
            rect = _bbox_pt_to_px(
                grid.get("bbox") if isinstance(grid, dict) else None,
                page_w_pt=page_w_pt,
                page_h_pt=page_h_pt,
                img_w_px=img_w,
                img_h_px=img_h,
            )
            if rect is None:
                continue
            draw.rectangle(rect, outline=(128, 0, 255), width=2)
            stats["table_grids"] += 1

        for region in page.get("image_regions") or []:
            rect = _bbox_pt_to_px(
                region,
                page_w_pt=page_w_pt,
                page_h_pt=page_h_pt,
                img_w_px=img_w,
                img_h_px=img_h,
            )
            if rect is None:
                continue
            draw.rectangle(rect, outline=(255, 140, 0), width=2)
            stats["image_regions"] += 1

    return img, stats


def _export_layout_assist_debug_images(
    *,
    source_pdf: Path,
    before_ir: dict[str, Any],
    after_ir: dict[str, Any],
    out_dir: Path,
    render_dpi: int = 144,
    assist_status: str | None = None,
    assist_error: str | None = None,
) -> dict[str, Any]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    before_pages = _to_page_map(before_ir)
    after_pages = _to_page_map(after_ir)
    page_indices = sorted(set(before_pages.keys()) | set(after_pages.keys()))
    manifest: dict[str, Any] = {
        "render_dpi": int(render_dpi),
        "assist_status": str(assist_status or "unknown"),
        "assist_error": str(assist_error or ""),
        "pages": [],
    }
    if not page_indices:
        manifest["summary"] = {"pages_exported": 0, "pages_changed": 0}
        (out_dir / "layout_assist_debug.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"pages_exported": 0, "pages_changed": 0}

    doc = pymupdf.open(str(source_pdf))
    pages_changed = 0
    try:
        for page_index in page_indices:
            if page_index < 0 or page_index >= int(doc.page_count or 0):
                continue
            pdf_page = doc.load_page(page_index)
            pix = pdf_page.get_pixmap(dpi=int(max(72, render_dpi)), alpha=False)  # type: ignore[attr-defined]
            base_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

            before_page = before_pages.get(page_index)
            after_page = after_pages.get(page_index)
            page_changed = _layout_page_signature(
                before_page
            ) != _layout_page_signature(after_page)
            if page_changed:
                pages_changed += 1
            page_w_pt = float(
                (
                    (after_page or {}).get("page_width_pt")
                    or (before_page or {}).get("page_width_pt")
                    or float(pdf_page.rect.width)
                )
                or float(pdf_page.rect.width)
            )
            page_h_pt = float(
                (
                    (after_page or {}).get("page_height_pt")
                    or (before_page or {}).get("page_height_pt")
                    or float(pdf_page.rect.height)
                )
                or float(pdf_page.rect.height)
            )

            before_img, before_stats = _draw_layout_assist_overlay(
                base_img=base_img,
                page=before_page,
                page_w_pt=page_w_pt,
                page_h_pt=page_h_pt,
            )
            after_img, after_stats = _draw_layout_assist_overlay(
                base_img=base_img,
                page=after_page,
                page_w_pt=page_w_pt,
                page_h_pt=page_h_pt,
            )

            before_name = f"page-{page_index:04d}.before.png"
            after_name = f"page-{page_index:04d}.after.png"
            before_path = out_dir / before_name
            after_path = out_dir / after_name
            before_img.save(before_path)
            after_img.save(after_path)

            manifest["pages"].append(
                {
                    "page_index": int(page_index),
                    "before_image": str(before_path),
                    "after_image": str(after_path),
                    "before_stats": before_stats,
                    "after_stats": after_stats,
                    "changed": bool(page_changed),
                }
            )
    finally:
        doc.close()

    manifest["summary"] = {
        "pages_exported": len(manifest.get("pages") or []),
        "pages_changed": int(pages_changed),
    }
    (out_dir / "layout_assist_debug.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "pages_exported": len(manifest.get("pages") or []),
        "pages_changed": int(pages_changed),
    }
