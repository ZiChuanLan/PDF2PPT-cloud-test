from __future__ import annotations

from app.convert.llm_adapter import _image_regions_px_to_pt


def test_image_regions_px_to_pt_uses_actual_render_dimensions() -> None:
    # Regression: conversion must use real image dimensions rather than a fixed
    # DPI scale constant, otherwise crops shift when render DPI differs.
    out = _image_regions_px_to_pt(
        [[180.0, 90.0, 900.0, 450.0]],
        image_width_px=1800,
        image_height_px=900,
        page_width_pt=720.0,
        page_height_pt=360.0,
    )

    assert len(out) == 1
    x0, y0, x1, y1 = out[0]
    assert abs(x0 - 72.0) <= 1e-6
    assert abs(y0 - 36.0) <= 1e-6
    assert abs(x1 - 360.0) <= 1e-6
    assert abs(y1 - 180.0) <= 1e-6


def test_image_regions_px_to_pt_drops_invalid_regions() -> None:
    out = _image_regions_px_to_pt(
        [[10.0, 10.0, 10.0, 20.0], [30.0, 20.0, 60.0, 70.0]],
        image_width_px=100,
        image_height_px=100,
        page_width_pt=200.0,
        page_height_pt=200.0,
    )

    assert out == [[60.0, 40.0, 120.0, 140.0]]
