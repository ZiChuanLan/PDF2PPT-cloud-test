from __future__ import annotations

from app.convert.ocr import _dedupe_overlapping_ocr_items


def test_dedupe_removes_shifted_duplicate_text_in_single_provider_mode() -> None:
    # Two identical lines, shifted down by 3px (overlap_small ~= 0.85).
    items = [
        {"text": "Hello world", "bbox": [10, 10, 210, 30], "confidence": 0.72},
        {"text": "Hello world", "bbox": [10, 13, 210, 33], "confidence": 0.72},
    ]

    out = _dedupe_overlapping_ocr_items(items)
    assert len(out) == 1
    assert out[0]["text"] == "Hello world"
