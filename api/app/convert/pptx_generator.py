"""Backward-compatible re-export shim."""

from app.convert.pptx.generator import generate_pptx_from_ir
from app.convert.pptx.font_utils import _token_width_pt, _wrap_paragraph_to_lines
from app.convert.pptx.scanned_page import _estimate_baseline_ocr_line_height_pt

__all__ = [
    "generate_pptx_from_ir",
    "_estimate_baseline_ocr_line_height_pt",
    "_token_width_pt",
    "_wrap_paragraph_to_lines",
]
