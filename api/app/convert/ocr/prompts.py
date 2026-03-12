"""AI OCR prompt preset resolution and template rendering."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .base import _clean_str

VALID_AI_OCR_PROMPT_PRESETS = frozenset(
    {
        "auto",
        "generic_vision",
        "openai_vision",
        "qwen_vl",
        "glm_v",
        "deepseek_ocr",
    }
)

_PROMPT_TEMPLATE_VAR_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


@dataclass(frozen=True)
class AiOcrPromptSettings:
    preset: str
    direct_prompt: str | None = None
    layout_block_prompt: str | None = None
    image_region_prompt: str | None = None


def normalize_ai_ocr_prompt_preset(value: str | None) -> str:
    cleaned = (_clean_str(value) or "auto").strip().lower()
    return cleaned if cleaned in VALID_AI_OCR_PROMPT_PRESETS else "auto"


def normalize_ai_ocr_prompt_override(
    value: str | None,
    *,
    max_chars: int = 6000,
) -> str | None:
    cleaned = _clean_str(value)
    if not cleaned:
        return None
    compact = cleaned[: max(0, int(max_chars))].strip()
    return compact or None


def infer_ai_ocr_prompt_preset(
    *,
    model_name: str | None,
    provider_id: str | None = None,
) -> str:
    lowered = str(model_name or "").strip().lower()
    normalized = lowered.replace("/", "-").replace("_", "-")
    normalized = re.sub(r"[^a-z0-9.+-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")

    if "deepseek-ocr" in normalized or "deepseekocr" in normalized:
        return "deepseek_ocr"
    if "qwen-vl" in normalized or "qwen2-vl" in normalized or "qwen2.5-vl" in normalized:
        return "qwen_vl"
    if "qwen3-vl" in normalized or normalized.startswith("qvq-"):
        return "qwen_vl"
    if re.search(r"\bglm-\d+(?:\.\d+)?v(?:[-.].*)?$", normalized):
        return "glm_v"
    if normalized.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3")):
        return "openai_vision"
    provider_cleaned = (_clean_str(provider_id) or "").lower()
    if provider_cleaned == "openai":
        return "openai_vision"
    return "generic_vision"


def resolve_ai_ocr_prompt_preset(
    *,
    preset: str | None,
    model_name: str | None,
    provider_id: str | None = None,
) -> str:
    normalized = normalize_ai_ocr_prompt_preset(preset)
    if normalized != "auto":
        return normalized
    return infer_ai_ocr_prompt_preset(model_name=model_name, provider_id=provider_id)


def render_ai_ocr_prompt_template(
    template: str,
    *,
    values: dict[str, Any] | None = None,
) -> str:
    payload = dict(values or {})
    raw = str(template or "")

    def _replace(match: re.Match[str]) -> str:
        key = str(match.group(1) or "").strip()
        if not key:
            return match.group(0)
        if key not in payload:
            return match.group(0)
        return str(payload.get(key) or "")

    rendered = _PROMPT_TEMPLATE_VAR_PATTERN.sub(_replace, raw)
    return rendered.strip()


def build_ai_ocr_direct_prompt(
    *,
    preset: str,
    image_width: int,
    image_height: int,
    item_limit: int,
    override: str | None = None,
) -> str:
    values = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "item_limit": int(item_limit),
    }
    custom = normalize_ai_ocr_prompt_override(override)
    if custom:
        return render_ai_ocr_prompt_template(custom, values=values)

    if preset == "deepseek_ocr":
        return "<image>\n<|grounding|>OCR this image."
    if preset == "qwen_vl":
        return (
            "OCR the document image. Return ONLY a minified JSON array. "
            "Use one item per visual text line with tight bbox in pixels. "
            "Schema: {\"text\":\"...\",\"bbox\":[x0,y0,x1,y1]}. "
            "Image size: {{image_width}}x{{image_height}} px. "
            "Keep output <= {{item_limit}} items. "
            "No markdown or explanation."
        ).replace("{{image_width}}", str(int(image_width))).replace(
            "{{image_height}}", str(int(image_height))
        ).replace("{{item_limit}}", str(int(item_limit)))
    if preset == "glm_v":
        return (
            "Read all visible document text and return ONLY a compact JSON array. "
            "Each item must be one line-level bbox in pixel coordinates: "
            "{\"text\":\"...\",\"bbox\":[x0,y0,x1,y1]}. "
            f"Image size: {int(image_width)}x{int(image_height)} px. "
            f"Keep output <= {int(item_limit)} items. "
            "Do not output markdown or commentary."
        )
    if preset == "openai_vision":
        return (
            f"OCR task. Return ONLY minified JSON array, no markdown. "
            f"Image size: {int(image_width)}x{int(image_height)} px. "
            "Each item must be one visual text line with tight bbox. "
            "Do not output duplicate lines/boxes; skip pure punctuation/noise. "
            'Preferred item schema: {"t":"text","b":[x0,y0,x1,y1],"c":0.0-1.0}. '
            'Also accepted: {"text":...,"bbox":...}. '
            "Coordinates are pixel values, origin top-left. "
            f"Keep output <= {int(item_limit)} items. "
            "If dense page, merge words into line-level entries. "
            "Stop immediately after JSON closes."
        )
    return (
        f"OCR task. Return ONLY minified JSON array, no markdown. "
        f"Image size: {int(image_width)}x{int(image_height)} px. "
        "Each item must be one visual text line with tight bbox. "
        "Do not output duplicate lines/boxes; skip pure punctuation/noise. "
        'Preferred item schema: {"t":"text","b":[x0,y0,x1,y1],"c":0.0-1.0}. '
        'Also accepted: {"text":...,"bbox":...}. '
        "Coordinates are pixel values, origin top-left. "
        f"Keep output <= {int(item_limit)} items. "
        "If dense page, merge words into line-level entries. "
        "Stop immediately after JSON closes."
    )


def build_ai_ocr_layout_block_prompt(
    *,
    preset: str,
    label: str,
    crop_width: int,
    crop_height: int,
    override: str | None = None,
) -> str:
    values = {
        "block_label": str(label or "text"),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
    }
    custom = normalize_ai_ocr_prompt_override(override)
    if custom:
        return render_ai_ocr_prompt_template(custom, values=values)

    if preset == "deepseek_ocr":
        return (
            "Read the cropped document text block and return plain text only. "
            f"Block label: {values['block_label']}. "
            f"Crop size: {values['crop_width']}x{values['crop_height']} px. "
            "Preserve obvious line breaks. Do not output grounding tags, JSON, markdown, or explanations. "
            "Return empty string if unreadable."
        )
    if preset == "qwen_vl":
        return (
            "Read the cropped document text block and return plain text only. "
            f"Block label: {values['block_label']}. "
            f"Crop size: {values['crop_width']}x{values['crop_height']} px. "
            "Preserve obvious line breaks. No JSON, markdown, or explanations. "
            "Return empty string if unreadable."
        )
    if preset == "glm_v":
        return (
            "Recognize all visible text inside this cropped document region and return plain text only. "
            f"Block label: {values['block_label']}. "
            f"Crop size: {values['crop_width']}x{values['crop_height']} px. "
            "Keep line breaks when obvious. Do not output JSON or commentary."
        )
    return (
        "Read all visible text in this cropped document block and return plain text only. "
        f"Block label: {values['block_label']}. Crop size: {values['crop_width']}x{values['crop_height']} px. "
        "Preserve obvious line breaks. Do not return JSON, markdown, or explanations. "
        "If the crop contains no readable text, return an empty string."
    )


def build_ai_ocr_image_region_prompt(
    *,
    preset: str,
    image_width: int,
    image_height: int,
    override: str | None = None,
) -> str:
    values = {
        "image_width": int(image_width),
        "image_height": int(image_height),
    }
    custom = normalize_ai_ocr_prompt_override(override)
    if custom:
        return render_ai_ocr_prompt_template(custom, values=values)

    if preset == "deepseek_ocr":
        return (
            "<image>\n<|grounding|>Locate <|ref|>screenshots, charts, diagrams, "
            "illustrations, photos, logos, and icons<|/ref|> in the image."
        )
    if preset == "qwen_vl":
        return (
            "Detect standalone non-text visual regions on this page and return ONLY a minified JSON array. "
            f"Image size: {values['image_width']}x{values['image_height']} px. "
            'Each item must be {"bbox":[x0,y0,x1,y1]}. '
            "Include screenshots, charts, diagrams, illustrations, photos, logos, and icons. "
            "Exclude surrounding text and full-page background."
        )
    return (
        "Locate standalone non-text visual regions on this page. "
        "Return ONLY minified JSON array, no markdown. "
        f"Image size: {values['image_width']}x{values['image_height']} px. "
        'Each item must be {"bbox":[x0,y0,x1,y1]}. '
        "Regions include screenshots, charts, diagrams, illustrations, photos, logos, and icons. "
        "Use tight pixel bbox around the visual asset only. "
        "Exclude surrounding text, card backgrounds, tables, formulas, and the full-page background. "
        "Keep output <= 12 items."
    )
