from __future__ import annotations

from pathlib import Path

from app.convert.pptx import generator as generator_module


def test_generator_uses_wrap_override_for_linebreak_assisted_ocr(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[bool | None] = []

    def _fake_fit_ocr_text_style(
        *,
        text: str,
        bbox_w_pt: float,
        bbox_h_pt: float,
        baseline_ocr_h_pt: float,
        is_heading: bool,
        wrap_override: bool | None = None,
    ) -> tuple[str, float, bool]:
        del bbox_w_pt, bbox_h_pt, baseline_ocr_h_pt, is_heading
        calls.append(wrap_override)
        # Return wrap=True only when override is not provided.
        return (text, 14.0, False if wrap_override is False else True)

    monkeypatch.setattr(generator_module, "_fit_ocr_text_style", _fake_fit_ocr_text_style)

    ir = {
        "source_pdf_path": str(tmp_path / "missing.pdf"),
        "pages": [
            {
                "page_index": 0,
                "page_width_pt": 720.0,
                "page_height_pt": 405.0,
                "has_text_layer": True,
                "elements": [
                    {
                        "type": "text",
                        "source": "ocr",
                        "bbox_pt": [80.0, 90.0, 220.0, 120.0],
                        "text": "这是行拆分后的单行文本",
                        "ocr_linebreak_assisted": True,
                    }
                ],
            }
        ],
    }

    out_path = tmp_path / "out.pptx"
    generator_module.generate_pptx_from_ir(
        ir,
        out_path,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert out_path.exists()
    assert calls == [False]

