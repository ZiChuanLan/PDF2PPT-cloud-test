from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_font_utils_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "convert"
        / "pptx"
        / "font_utils.py"
    )
    spec = importlib.util.spec_from_file_location("test_font_utils_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


font_utils = _load_font_utils_module()


def test_visual_single_line_veto_kept_for_short_text():
    override = font_utils._resolve_visual_wrap_override_for_ocr_text(
        visual_line_count=1,
        compact_len=14,
        bbox_h_pt=11.0,
        baseline_ocr_h_pt=16.0,
        is_heading=False,
    )

    assert override is False


def test_visual_single_line_veto_relaxed_for_long_paragraph_text():
    override = font_utils._resolve_visual_wrap_override_for_ocr_text(
        visual_line_count=1,
        compact_len=96,
        bbox_h_pt=18.0,
        baseline_ocr_h_pt=16.0,
        is_heading=False,
    )

    assert override is None


def test_long_ocr_paragraph_can_recover_wrapping_when_single_line_signal_is_weak():
    text = (
        "Yellowstone is more than a geologic marvel. It is a living laboratory "
        "and a cultural touchstone, preserved for the benefit and enjoyment of "
        "the people."
    )
    compact_len = font_utils._compact_text_length(text)
    weak_single_line_override = font_utils._resolve_visual_wrap_override_for_ocr_text(
        visual_line_count=1,
        compact_len=compact_len,
        bbox_h_pt=28.0,
        baseline_ocr_h_pt=16.0,
        is_heading=False,
    )

    recovered = font_utils._fit_ocr_text_style(
        text=text,
        bbox_w_pt=360.0,
        bbox_h_pt=28.0,
        baseline_ocr_h_pt=16.0,
        is_heading=False,
        wrap_override=weak_single_line_override,
    )

    assert weak_single_line_override is None
    assert recovered[2] is True
    assert "\n" in recovered[0]


def test_measurement_driven_fit_wraps_long_body_text_without_visual_override():
    text = (
        "Yellowstone is more than a geologic marvel. It is a living laboratory "
        "and a cultural touchstone, preserved for the benefit and enjoyment of "
        "the people."
    )

    recovered = font_utils._fit_ocr_text_style(
        text=text,
        bbox_w_pt=360.0,
        bbox_h_pt=28.0,
        baseline_ocr_h_pt=16.0,
        is_heading=False,
    )

    assert recovered[2] is True
    assert recovered[1] >= 8.0
    assert "\n" in recovered[0]


def test_forced_single_line_veto_is_relaxed_for_very_long_ocr_paragraphs():
    text = (
        "It increases the rate that nutrients become available to plants by "
        "rapidly releasing them from forest litter. Serotinous cones are "
        "sealed by resin and require the heat of a fire to open and release "
        "their seeds."
    )

    recovered = font_utils._fit_ocr_text_style(
        text=text,
        bbox_w_pt=620.0,
        bbox_h_pt=24.0,
        baseline_ocr_h_pt=18.0,
        is_heading=False,
        wrap_override=False,
    )

    assert recovered[2] is True
    assert recovered[1] > 8.0
    assert "\n" in recovered[0]


def test_forced_single_line_veto_still_holds_for_short_label_text():
    recovered = font_utils._fit_ocr_text_style(
        text="Fire return intervals",
        bbox_w_pt=220.0,
        bbox_h_pt=22.0,
        baseline_ocr_h_pt=18.0,
        is_heading=False,
        wrap_override=False,
    )

    assert recovered[2] is False
    assert "\n" not in recovered[0]


def test_single_line_ocr_text_expands_to_fill_tall_wide_bbox():
    recovered = font_utils._fit_ocr_text_style(
        text="释放未来·构建你的第一个AI智能体",
        bbox_w_pt=1168.46,
        bbox_h_pt=80.4,
        baseline_ocr_h_pt=63.0,
        is_heading=False,
    )

    assert recovered[2] is False
    assert "\n" not in recovered[0]
    assert recovered[1] > 60.0
