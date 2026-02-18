from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from app.convert.ocr.ai_client import AiOcrTextRefiner


def _write_multiline_mock_image(path: Path) -> None:
    image = Image.new("RGB", (220, 140), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Simulate two text lines with clear left/right margins inside a coarse bbox.
    draw.rectangle((62, 36, 156, 44), fill=(20, 20, 20))
    draw.rectangle((70, 70, 150, 78), fill=(20, 20, 20))
    # Extra line elsewhere to keep median line height realistic.
    draw.rectangle((30, 110, 95, 118), fill=(20, 20, 20))
    image.save(path)


def test_ai_linebreak_assist_tightens_x_and_marks_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    class _DummyOpenAI:
        def __init__(self, **_: object):
            self.chat = SimpleNamespace(completions=self)

        def with_options(self, **_: object):
            return self

        def create(self, **_: object):
            # Split only the first candidate item.
            payload = '[{"i":0,"lines":["第一行测试文本","第二行测试文本"]}]'
            choice = SimpleNamespace(
                message=SimpleNamespace(content=payload),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice])

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)

    image_path = tmp_path / "multiline.png"
    _write_multiline_mock_image(image_path)

    refiner = AiOcrTextRefiner(
        api_key="dummy",
        provider="openai",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
    )
    out = refiner.assist_line_breaks(
        str(image_path),
        items=[
            {
                "text": "第一行测试文本第二行测试文本",
                "bbox": [20.0, 24.0, 180.0, 96.0],
                "confidence": 0.88,
            },
            {
                "text": "footer",
                "bbox": [24.0, 104.0, 110.0, 124.0],
                "confidence": 0.92,
            },
        ],
    )

    assert len(out) == 3

    split_rows = [row for row in out if row.get("linebreak_assisted")]
    assert len(split_rows) == 2
    assert all(row.get("linebreak_assist_source") == "ai" for row in split_rows)

    # Coarse source bbox is x=[20,180]. Tightened rows should shrink inward.
    assert all(float(row["bbox"][0]) > 30.0 for row in split_rows)
    assert all(float(row["bbox"][2]) < 176.0 for row in split_rows)

