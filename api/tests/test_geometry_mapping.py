from __future__ import annotations

from app.convert.geometry import bbox_pt_to_px, bbox_px_to_pt, coerce_bbox_xyxy


def test_bbox_pt_px_roundtrip_is_stable() -> None:
    bbox_pt = [10.0, 20.0, 110.0, 70.0]

    rect_px = bbox_pt_to_px(
        bbox_pt,
        page_w_pt=200.0,
        page_h_pt=100.0,
        img_w_px=1000,
        img_h_px=500,
    )
    assert rect_px == (50, 100, 550, 350)

    roundtrip = bbox_px_to_pt(
        rect_px,
        img_w_px=1000,
        img_h_px=500,
        page_w_pt=200.0,
        page_h_pt=100.0,
    )
    assert roundtrip is not None
    x0, y0, x1, y1 = roundtrip
    assert abs(x0 - 10.0) <= 1e-6
    assert abs(y0 - 20.0) <= 1e-6
    assert abs(x1 - 110.0) <= 1e-6
    assert abs(y1 - 70.0) <= 1e-6


def test_geometry_helpers_reject_invalid_boxes() -> None:
    assert coerce_bbox_xyxy([1, 2, 3]) is None
    assert coerce_bbox_xyxy([1, 2, 3, float("nan")]) is None
    assert (
        bbox_pt_to_px(
            [10, 10, 10, 20],
            page_w_pt=200.0,
            page_h_pt=100.0,
            img_w_px=1000,
            img_h_px=500,
        )
        is None
    )
