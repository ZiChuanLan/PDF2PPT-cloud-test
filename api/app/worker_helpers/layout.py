from __future__ import annotations

from typing import Any


def _normalize_optional_list(value: Any) -> Any:
    if value is None:
        return []
    return value


def _apply_ai_tables(ir: dict[str, Any]) -> dict[str, Any]:
    pages = ir.get("pages")
    if not isinstance(pages, list):
        return ir

    for page in pages:
        if not isinstance(page, dict):
            continue
        grids = page.get("table_grids")
        if not isinstance(grids, list) or not grids:
            continue

        elements = page.get("elements")
        if not isinstance(elements, list):
            continue

        text_elements: list[dict[str, Any]] = []
        other_elements: list[dict[str, Any]] = []
        for el in elements:
            if not isinstance(el, dict):
                continue
            if el.get("type") == "text" and str(el.get("source") or "") == "ocr":
                text_elements.append(el)
            else:
                other_elements.append(el)

        used_text_indices: set[int] = set()
        new_tables: list[dict[str, Any]] = []

        for grid in grids:
            if not isinstance(grid, dict):
                continue
            bbox = grid.get("bbox")
            try:
                rows = int(grid.get("rows") or 0)
                cols = int(grid.get("cols") or 0)
            except Exception:
                rows, cols = 0, 0
            if not isinstance(bbox, list) or len(bbox) != 4 or rows <= 0 or cols <= 0:
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

            cell_w = (x1 - x0) / cols
            cell_h = (y1 - y0) / rows
            if cell_w <= 0 or cell_h <= 0:
                continue

            cell_texts: list[list[str]] = [[] for _ in range(rows * cols)]
            for idx, el in enumerate(text_elements):
                bbox_pt = el.get("bbox_pt")
                if not isinstance(bbox_pt, list) or len(bbox_pt) != 4:
                    continue
                try:
                    cx = (float(bbox_pt[0]) + float(bbox_pt[2])) / 2.0
                    cy = (float(bbox_pt[1]) + float(bbox_pt[3])) / 2.0
                except Exception:
                    continue
                if cx < x0 or cx > x1 or cy < y0 or cy > y1:
                    continue
                col = int((cx - x0) / cell_w)
                row = int((cy - y0) / cell_h)
                col = min(max(col, 0), cols - 1)
                row = min(max(row, 0), rows - 1)
                cell_texts[row * cols + col].append(str(el.get("text") or "").strip())
                used_text_indices.add(idx)

            cells: list[dict[str, Any]] = []
            for r in range(rows):
                for c in range(cols):
                    cell_bbox = [
                        x0 + c * cell_w,
                        y0 + r * cell_h,
                        x0 + (c + 1) * cell_w,
                        y0 + (r + 1) * cell_h,
                    ]
                    text = " ".join(t for t in cell_texts[r * cols + c] if t)
                    cells.append(
                        {
                            "r": r,
                            "c": c,
                            "bbox_pt": cell_bbox,
                            "text": text,
                        }
                    )

            new_tables.append(
                {
                    "type": "table",
                    "bbox_pt": [x0, y0, x1, y1],
                    "rows": rows,
                    "cols": cols,
                    "cells": cells,
                    "source": "ai",
                }
            )

        if new_tables:
            remaining_text = [
                el
                for idx, el in enumerate(text_elements)
                if idx not in used_text_indices
            ]
            page["elements"] = other_elements + remaining_text + new_tables

    return ir


def _to_page_map(ir: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages = ir.get("pages")
    if not isinstance(pages, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            page_index = int(page.get("page_index") or 0)
        except Exception:
            continue
        out[page_index] = page
    return out


def _layout_page_signature(page: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(page, dict):
        return {}
    return {
        "elements": page.get("elements"),
        "reading_order": _normalize_optional_list(page.get("reading_order")),
        "table_grids": _normalize_optional_list(page.get("table_grids")),
        "image_regions": _normalize_optional_list(page.get("image_regions")),
    }


def _count_layout_assist_page_changes(
    before_ir: dict[str, Any], after_ir: dict[str, Any]
) -> tuple[int, int]:
    before_pages = _to_page_map(before_ir)
    after_pages = _to_page_map(after_ir)
    page_indices = sorted(set(before_pages.keys()) | set(after_pages.keys()))
    pages_total = 0
    pages_changed = 0
    for page_index in page_indices:
        before_page = before_pages.get(page_index)
        after_page = after_pages.get(page_index)
        if not isinstance(before_page, dict) and not isinstance(after_page, dict):
            continue
        pages_total += 1
        if _layout_page_signature(before_page) != _layout_page_signature(after_page):
            pages_changed += 1
    return pages_changed, pages_total


def _extract_warning_suffix(warnings: list[Any] | None, *, prefix: str) -> str | None:
    if not isinstance(warnings, list):
        return None
    for item in warnings:
        if not isinstance(item, str):
            continue
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None
