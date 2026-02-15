from __future__ import annotations

from PIL import Image, ImageDraw

from app.convert.ocr import AiOcrClient


def test_aiocr_bbox_normalization_handles_aspect_resized_coordinate_spaces() -> None:
    """Regression: some VLM OCR models emit bboxes in resized input pixel space.

    Example: the gateway resizes a 200x100 image to 1000x500 before inference,
    then returns bboxes in that resized grid. If we incorrectly assume a single
    uniform base (e.g. 1000 for both axes), Y coordinates get squashed by 2x,
    causing visible vertical offsets (and incomplete background erasure).
    """

    img_w, img_h = 200, 100
    image = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Draw two synthetic "text lines": sparse glyph-like blocks inside the
    # intended bboxes. We avoid solid rectangles because the bbox scoring
    # heuristic uses pixel variance to distinguish real text regions.
    def _draw_fake_glyphs(y_top: int) -> None:
        for i in range(10):
            x0 = 24 + i * 14
            draw.rectangle([x0, y_top, x0 + 7, y_top + 9], fill=(0, 0, 0))

    _draw_fake_glyphs(11)
    _draw_fake_glyphs(41)

    # Coordinates expressed in a 1000x500 resized space.
    raw_items = [
        {"text": "line1", "bbox": [100, 50, 900, 100], "confidence": 0.9},
        {"text": "line2", "bbox": [100, 200, 900, 250], "confidence": 0.9},
    ]

    # Dummy client: we only use internal normalization helpers (no network).
    client = AiOcrClient(
        api_key="dummy", provider="openai", base_url=None, model="gpt-4o-mini"
    )

    normalized, debug = client._normalize_items_to_pixels(raw_items, image=image)
    assert normalized

    chosen = debug.get("chosen_details") or {}
    sx = float(chosen.get("sx") or 0.0)
    sy = float(chosen.get("sy") or 0.0)
    assert sx > 0.0
    assert sy > 0.0

    # The correct transform should preserve aspect ratio (sx ~= sy),
    # not squash Y (sy ~= sx/2).
    assert (sy / sx) >= 0.75

    by_text = {str(it.get("text") or ""): it for it in normalized}
    assert "line1" in by_text
    assert "line2" in by_text

    x0, y0, x1, y1 = [float(v) for v in by_text["line1"]["bbox"]]
    assert abs(x0 - 20.0) <= 6.0
    assert abs(y0 - 10.0) <= 6.0
    assert abs(x1 - 180.0) <= 6.0
    assert abs(y1 - 20.0) <= 6.0
