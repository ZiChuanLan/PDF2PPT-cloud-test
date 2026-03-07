from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert.pptx import generator


def test_visual_wrap_probe_skips_clear_single_line_label() -> None:
    should_probe = generator._should_probe_visual_wrap_for_ocr_text(
        text="Fire return intervals",
        bbox_w_pt=220.0,
        bbox_h_pt=18.0,
        baseline_ocr_h_pt=18.0,
        is_heading=False,
        wrap_hint=False,
        ocr_linebreak_assisted=False,
    )

    assert should_probe is False


def test_visual_wrap_probe_keeps_ambiguous_long_ocr_box() -> None:
    should_probe = generator._should_probe_visual_wrap_for_ocr_text(
        text=(
            "It increases the rate that nutrients become available to plants by "
            "rapidly releasing them from forest litter."
        ),
        bbox_w_pt=620.0,
        bbox_h_pt=24.0,
        baseline_ocr_h_pt=18.0,
        is_heading=False,
        wrap_hint=False,
        ocr_linebreak_assisted=False,
    )

    assert should_probe is True


def test_local_color_sampling_skips_when_upstream_color_exists() -> None:
    should_sample = generator._should_sample_local_text_colors(
        source_id="ocr",
        element_color="#1a1a1a",
    )

    assert should_sample is False


def test_page_sampling_render_skips_when_no_element_needs_probe_or_local_color() -> None:
    needs_render = generator._page_needs_ocr_sampling_render(
        page_elements=[
            {
                "source": "ocr",
                "text": "Fire return intervals",
                "bbox_pt": [20.0, 20.0, 240.0, 38.0],
                "color": "#111111",
                "ocr_linebreak_assisted": False,
            }
        ],
        page_h_pt=540.0,
        baseline_ocr_h_pt=18.0,
    )

    assert needs_render is False


def test_page_sampling_render_keeps_render_for_ambiguous_long_box() -> None:
    needs_render = generator._page_needs_ocr_sampling_render(
        page_elements=[
            {
                "source": "ocr",
                "text": (
                    "It increases the rate that nutrients become available to plants by "
                    "rapidly releasing them from forest litter."
                ),
                "bbox_pt": [20.0, 40.0, 640.0, 64.0],
                "color": "#111111",
                "ocr_linebreak_assisted": False,
            }
        ],
        page_h_pt=540.0,
        baseline_ocr_h_pt=18.0,
    )

    assert needs_render is True


def test_generate_pptx_skips_final_preview_export_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    preview_calls: list[int] = []

    monkeypatch.setattr(
        generator,
        "_export_final_preview_page_image",
        lambda **_kwargs: preview_calls.append(1),
    )

    out_path = generator.generate_pptx_from_ir(
        {
            "pages": [
                {
                    "page_index": 0,
                    "page_width_pt": 720.0,
                    "page_height_pt": 540.0,
                    "has_text_layer": True,
                    "elements": [],
                }
            ]
        },
        tmp_path / "out.pptx",
        artifacts_dir=tmp_path / "artifacts",
        export_final_preview_images=False,
    )

    assert out_path.exists()
    assert preview_calls == []


def test_generate_pptx_keeps_final_preview_export_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    preview_calls: list[int] = []

    monkeypatch.setattr(
        generator,
        "_export_final_preview_page_image",
        lambda **_kwargs: preview_calls.append(1),
    )

    generator.generate_pptx_from_ir(
        {
            "pages": [
                {
                    "page_index": 0,
                    "page_width_pt": 720.0,
                    "page_height_pt": 540.0,
                    "has_text_layer": True,
                    "elements": [],
                }
            ]
        },
        tmp_path / "out-enabled.pptx",
        artifacts_dir=tmp_path / "artifacts-enabled",
        export_final_preview_images=True,
    )

    assert preview_calls == [1]
