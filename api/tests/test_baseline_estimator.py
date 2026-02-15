from __future__ import annotations

from app.convert.pptx_generator import _estimate_baseline_ocr_line_height_pt


def test_estimate_baseline_prefers_wide_body_text_over_tiny_ui_text() -> None:
    # Simulate a scanned slide where an embedded screenshot produces many tiny
    # OCR boxes (e.g. browser UI), while the real slide body text boxes are
    # wider and taller.
    page_w_pt = 1000.0

    elements: list[dict[str, object]] = []

    # Tiny UI text inside a screenshot (narrow + short).
    for i in range(14):
        elements.append(
            {
                "bbox_pt": [50.0, 40.0 + i * 10.0, 110.0, 46.0 + i * 10.0],
                "text": f"ui-{i}",
            }
        )

    # Slide body text lines (wider + taller).
    for i in range(5):
        elements.append(
            {
                "bbox_pt": [80.0, 300.0 + i * 30.0, 560.0, 320.0 + i * 30.0],
                "text": f"body-{i}",
            }
        )

    baseline = _estimate_baseline_ocr_line_height_pt(
        ocr_text_elements=[dict(e) for e in elements],
        page_w_pt=page_w_pt,
    )

    # Expect something close to the body line height (~20pt), not the tiny UI
    # height (~6pt).
    assert baseline >= 16.0
