from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert.ocr.prompts import (
    build_ai_ocr_direct_prompt,
    build_ai_ocr_image_region_prompt,
    build_ai_ocr_layout_block_prompt,
    render_ai_ocr_prompt_template,
    resolve_ai_ocr_prompt_preset,
)


def test_resolve_ai_ocr_prompt_preset_infers_known_model_families() -> None:
    assert (
        resolve_ai_ocr_prompt_preset(
            preset="auto",
            model_name="Qwen/Qwen2.5-VL-72B-Instruct",
            provider_id="openai",
        )
        == "qwen_vl"
    )
    assert (
        resolve_ai_ocr_prompt_preset(
            preset="auto",
            model_name="zai-org/GLM-4.6V",
            provider_id="openai",
        )
        == "glm_v"
    )
    assert (
        resolve_ai_ocr_prompt_preset(
            preset="auto",
            model_name="deepseek-ai/DeepSeek-OCR",
            provider_id="siliconflow",
        )
        == "deepseek_ocr"
    )


def test_render_ai_ocr_prompt_template_substitutes_known_variables_only() -> None:
    rendered = render_ai_ocr_prompt_template(
        "Size {{image_width}}x{{image_height}} keep {{missing}}",
        values={"image_width": 320, "image_height": 240},
    )
    assert rendered == "Size 320x240 keep {{missing}}"


def test_prompt_builders_use_custom_overrides() -> None:
    direct = build_ai_ocr_direct_prompt(
        preset="openai_vision",
        image_width=1024,
        image_height=768,
        item_limit=32,
        override="OCR {{image_width}}x{{image_height}} <= {{item_limit}}",
    )
    layout = build_ai_ocr_layout_block_prompt(
        preset="qwen_vl",
        label="title",
        crop_width=300,
        crop_height=80,
        override="Block {{block_label}} {{crop_width}}x{{crop_height}}",
    )
    image_region = build_ai_ocr_image_region_prompt(
        preset="generic_vision",
        image_width=1280,
        image_height=720,
        override="Detect {{image_width}}x{{image_height}}",
    )

    assert direct == "OCR 1024x768 <= 32"
    assert layout == "Block title 300x80"
    assert image_region == "Detect 1280x720"


def test_deepseek_layout_block_prompt_uses_plain_text_instead_of_grounding() -> None:
    prompt = build_ai_ocr_layout_block_prompt(
        preset="deepseek_ocr",
        label="paragraph_title",
        crop_width=640,
        crop_height=120,
    )

    assert "<|grounding|>" not in prompt
    assert "<|ref|>" not in prompt
    assert "<|det|>" not in prompt
    assert "plain text only" in prompt.lower()
    assert "paragraph_title" in prompt
