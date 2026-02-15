"""Constants and core transform dataclass for PPTX generation."""

from __future__ import annotations

from dataclasses import dataclass

# PowerPoint uses EMUs (English Metric Units): 914400 EMU = 1 inch.

_EMU_PER_INCH = 914_400
_PTS_PER_INCH = 72.0
_EMU_PER_PT = _EMU_PER_INCH / _PTS_PER_INCH  # 12700.0

@dataclass(frozen=True)
class SlideTransform:
    """Coordinate transform from a PDF page (pt, bottom-left origin) to PPT slide EMUs."""

    page_width_pt: float
    page_height_pt: float
    slide_width_emu: int
    slide_height_emu: int
    scale: float
    offset_x_emu: float
    offset_y_emu: float
