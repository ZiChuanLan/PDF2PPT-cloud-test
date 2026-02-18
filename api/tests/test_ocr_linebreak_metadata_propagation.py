from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.convert.ocr import local_providers as ocr_module


class _DummyOcrManager:
    provider_id = "aiocr"
    last_provider_name = "AiOcrClient"

    def ocr_image_lines(
        self, _image_path: str, *, image_width: int, image_height: int
    ) -> list[dict]:
        assert image_width > 0 and image_height > 0
        return [
            {
                "text": "Hello OCR",
                "bbox": [20.0, 18.0, 120.0, 42.0],
                "confidence": 0.9,
                "provider": "aiocr",
                "model": "gpt-4o-mini",
                "linebreak_assisted": True,
                "linebreak_assist_source": "ai",
            }
        ]

    def convert_bbox_to_pdf_coords(
        self,
        *,
        bbox: list[float],
        image_width: int,
        image_height: int,
        page_width_pt: float,
        page_height_pt: float,
    ) -> list[float]:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        sx = float(page_width_pt) / float(max(1, image_width))
        sy = float(page_height_pt) / float(max(1, image_height))
        return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]


def test_ocr_linebreak_metadata_propagates_to_elements(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 120), (255, 255, 255)).save(image_path)

    out = ocr_module.ocr_image_to_elements(
        str(image_path),
        page_width_pt=600.0,
        page_height_pt=360.0,
        ocr_manager=_DummyOcrManager(),
        text_refiner=None,
        linebreak_refiner=None,
        strict_no_fallback=True,
        linebreak_assist=None,
    )

    assert len(out) == 1
    row = out[0]
    assert row.get("source") == "ocr"
    assert row.get("ocr_linebreak_assisted") is True
    assert row.get("ocr_linebreak_assist_source") == "ai"

