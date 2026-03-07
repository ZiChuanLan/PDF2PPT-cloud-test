"""DeepSeek OCR grounding-tag parser helpers."""

import html
import re
from typing import Any

from .base import _clean_str
from .json_extraction import _extract_message_text

_DEEPSEEK_DET_BOX_PATTERN = re.compile(
    r"<\|det\|>\s*\[\[\s*"
    r"(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*"
    r"\]\]\s*<\|/det\|>",
    re.IGNORECASE,
)

_DEEPSEEK_REF_THEN_DET_PATTERN = re.compile(
    r"<\|ref\|>(?P<text>.*?)<\|/ref\|>\s*"
    r"<\|det\|>\s*\[\[\s*"
    r"(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*"
    r"\]\]\s*<\|/det\|>",
    re.IGNORECASE | re.DOTALL,
)

_DEEPSEEK_DET_THEN_REF_PATTERN = re.compile(
    r"<\|det\|>\s*\[\[\s*"
    r"(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*"
    r"\]\]\s*<\|/det\|>\s*"
    r"<\|ref\|>(?P<text>.*?)<\|/ref\|>",
    re.IGNORECASE | re.DOTALL,
)

_DEEPSEEK_REF_DET_INLINE_TEXT_PATTERN = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*"
    r"<\|det\|>\s*\[\[\s*"
    r"(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*"
    r"\]\]\s*<\|/det\|>\s*"
    r"(?P<text>.*?)(?=(?:<\|ref\|>|$))",
    re.IGNORECASE | re.DOTALL,
)

_DEEPSEEK_TAG_TOKEN_PATTERN = re.compile(
    r"(?P<ref><\|ref\|>(?P<ref_text>.*?)<\|/ref\|>)"
    r"|(?P<det><\|det\|>\s*\[\[\s*"
    r"(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*"
    r"\]\]\s*<\|/det\|>)",
    re.IGNORECASE | re.DOTALL,
)

_DEEPSEEK_DET_INLINE_TEXT_PATTERN = re.compile(
    r"<\|det\|>\s*\[\[\s*"
    r"(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*"
    r"\]\]\s*<\|/det\|>\s*"
    r"(?P<text>.*?)(?=(?:<\|det\|>|$))",
    re.IGNORECASE | re.DOTALL,
)

_DEEPSEEK_PLAIN_BOX_INLINE_PATTERN = re.compile(
    r"\[\[?\s*"
    r"(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*"
    r"\]\]?\s*"
    r"(?P<text>[^\n\r]+)",
    re.IGNORECASE,
)

_DEEPSEEK_GENERIC_REF_LABELS = {
    "text",
    "image",
    "figure",
    "icon",
    "diagram",
    "chart",
    "logo",
    "subtitle",
    "sub_title",
    "equation",
    "formula",
    "table",
    "title",
    "caption",
    "footnote",
    "header",
    "footer",
}

_OCR_PROMPT_ECHO_PREFIXES = (
    "ocr task",
    "image size",
    "return line-level ocr",
    "return line level ocr",
    "preferred format",
    "json array is also accepted",
    "each item must be one visual text line",
    "coordinates are pixel values",
    "stop immediately after json closes",
)


def _is_deepseek_ocr_model(model_name: str | None) -> bool:
    cleaned = (_clean_str(model_name) or "").lower()
    if not cleaned:
        return False
    return "deepseek-ocr" in cleaned or "deepseekocr" in cleaned


def _looks_like_ocr_prompt_echo_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized:
        return False

    if normalized in _DEEPSEEK_GENERIC_REF_LABELS:
        return True

    if re.fullmatch(r"region[_\s-]?\d{1,4}", normalized):
        return True

    if normalized.startswith("image size:") and "px" in normalized:
        return True

    if any(normalized.startswith(prefix) for prefix in _OCR_PROMPT_ECHO_PREFIXES):
        return True

    if "return line-level ocr" in normalized and "bbox" in normalized:
        return True

    return False


def _clean_deepseek_ref_text(raw_text: str) -> str:
    text = str(raw_text or "")
    if not text:
        return ""
    # Decode HTML-escaped tags (e.g. &lt;|ref|&gt;).
    for _ in range(2):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = re.sub(r"<\|/?[a-zA-Z0-9_]+\|>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_deepseek_tagged_items(
    text: Any, *, max_items: int = 1500
) -> list[dict] | None:
    content = _extract_message_text(text)
    if not content:
        return None

    normalized = content
    for _ in range(2):
        decoded = html.unescape(normalized)
        if decoded == normalized:
            break
        normalized = decoded

    out: list[dict] = []
    seen: set[tuple[str, float, float, float, float]] = set()

    def _append_item(raw_text: str, x0: str, y0: str, x1: str, y1: str) -> None:
        if len(out) >= max_items:
            return
        text_cleaned = _clean_deepseek_ref_text(raw_text)
        if not text_cleaned:
            return
        if _looks_like_ocr_prompt_echo_text(text_cleaned):
            return
        try:
            fx0 = float(x0)
            fy0 = float(y0)
            fx1 = float(x1)
            fy1 = float(y1)
        except Exception:
            return
        key = (text_cleaned, fx0, fy0, fx1, fy1)
        if key in seen:
            return
        seen.add(key)
        out.append(
            {
                "text": text_cleaned,
                "bbox": [fx0, fy0, fx1, fy1],
                "confidence": 0.72,
            }
        )

    has_ref_tags = "<|ref|>" in normalized.lower()
    has_det_tags = "<|det|>" in normalized.lower()

    def _clean_inline_text(chunk: str) -> str:
        # Inline text after a det tag can contain newlines or extra whitespace.
        cleaned = _clean_deepseek_ref_text(chunk)
        if not cleaned:
            return ""
        # Avoid swallowing the next tag if the gateway concatenated outputs.
        cleaned = cleaned.split("<|ref|>", 1)[0].split("<|det|>", 1)[0].strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # Parse DeepSeek grounding tags via a non-overlapping token stream. Regex
    # patterns like det-then-ref can accidentally pair a <|det|> from one item
    # with the <|ref|> of the *next* item (because tags are adjacent), creating
    # a shifted "two texts per bbox" ladder. Tokenize and pair sequentially
    # instead.
    if has_det_tags and has_ref_tags:
        tokens: list[dict[str, Any]] = []
        for match in _DEEPSEEK_TAG_TOKEN_PATTERN.finditer(normalized):
            if match.group("ref") is not None:
                tokens.append(
                    {
                        "type": "ref",
                        "text": str(match.group("ref_text") or ""),
                        "start": int(match.start()),
                        "end": int(match.end()),
                    }
                )
                continue
            if match.group("det") is not None:
                tokens.append(
                    {
                        "type": "det",
                        "bbox": (
                            str(match.group("x0") or ""),
                            str(match.group("y0") or ""),
                            str(match.group("x1") or ""),
                            str(match.group("y1") or ""),
                        ),
                        "start": int(match.start()),
                        "end": int(match.end()),
                    }
                )

        i = 0
        while (i + 1) < len(tokens) and len(out) < max_items:
            a = tokens[i]
            b = tokens[i + 1]
            a_type = str(a.get("type") or "")
            b_type = str(b.get("type") or "")

            if a_type == "ref" and b_type == "det":
                ref_text_raw = str(a.get("text") or "")
                x0, y0, x1, y1 = a.get("bbox") or ("", "", "", "")
                if isinstance(b.get("bbox"), tuple) and len(b["bbox"]) == 4:
                    x0, y0, x1, y1 = b["bbox"]

                next_start = (
                    int(tokens[i + 2]["start"])
                    if (i + 2) < len(tokens)
                    else len(normalized)
                )
                inline_raw = normalized[int(b.get("end") or 0) : next_start]
                inline_clean = _clean_inline_text(inline_raw)
                ref_clean = _clean_deepseek_ref_text(ref_text_raw)

                chosen_text = ref_text_raw
                if inline_clean and (
                    (not ref_clean)
                    or (ref_clean.lower() in _DEEPSEEK_GENERIC_REF_LABELS)
                    or _looks_like_ocr_prompt_echo_text(ref_clean)
                ):
                    chosen_text = inline_clean

                _append_item(chosen_text, x0, y0, x1, y1)
                i += 2
                continue

            if a_type == "det" and b_type == "ref":
                ref_text_raw = str(b.get("text") or "")
                x0, y0, x1, y1 = a.get("bbox") or ("", "", "", "")
                if isinstance(a.get("bbox"), tuple) and len(a["bbox"]) == 4:
                    x0, y0, x1, y1 = a["bbox"]

                next_start = (
                    int(tokens[i + 2]["start"])
                    if (i + 2) < len(tokens)
                    else len(normalized)
                )
                inline_raw = normalized[int(b.get("end") or 0) : next_start]
                inline_clean = _clean_inline_text(inline_raw)
                ref_clean = _clean_deepseek_ref_text(ref_text_raw)

                chosen_text = ref_text_raw
                if inline_clean and (
                    (not ref_clean)
                    or (ref_clean.lower() in _DEEPSEEK_GENERIC_REF_LABELS)
                    or _looks_like_ocr_prompt_echo_text(ref_clean)
                ):
                    chosen_text = inline_clean

                _append_item(chosen_text, x0, y0, x1, y1)
                i += 2
                continue

            i += 1

    # Backward compatibility: some gateways return tagged grounding but without
    # both ref+det tokens being parsable by the tokenizer above. Fall back to
    # simple ref-then-det pairing only (avoid det-then-ref which can cross-pair).
    #
    # IMPORTANT: only run this when token pairing produced no items; otherwise
    # we risk adding a second text candidate for the same bbox (e.g. generic
    # ref labels vs inline text), which reintroduces duplicated/shifted lines.
    if not out and has_ref_tags:
        for match in _DEEPSEEK_REF_THEN_DET_PATTERN.finditer(normalized):
            _append_item(
                str(match.group("text") or ""),
                str(match.group("x0") or ""),
                str(match.group("y0") or ""),
                str(match.group("x1") or ""),
                str(match.group("y1") or ""),
            )
            if len(out) >= max_items:
                break

    # Det-only inline text formats are used as fallback only when ref tags are
    # absent. This prevents shifted duplicate lines in DeepSeek grounding output.
    if len(out) < max_items and not has_ref_tags:
        for match in _DEEPSEEK_DET_INLINE_TEXT_PATTERN.finditer(normalized):
            _append_item(
                str(match.group("text") or ""),
                str(match.group("x0") or ""),
                str(match.group("y0") or ""),
                str(match.group("x1") or ""),
                str(match.group("y1") or ""),
            )
            if len(out) >= max_items:
                break

    if len(out) < max_items and not has_ref_tags:
        for match in _DEEPSEEK_PLAIN_BOX_INLINE_PATTERN.finditer(normalized):
            _append_item(
                str(match.group("text") or ""),
                str(match.group("x0") or ""),
                str(match.group("y0") or ""),
                str(match.group("x1") or ""),
                str(match.group("y1") or ""),
            )
            if len(out) >= max_items:
                break

    if out:
        return out

    # Fallback: when only <|det|> boxes are present (no usable text), still return
    # coarse placeholders so caller can decide whether to keep/fallback.
    det_only: list[dict] = []
    for idx, match in enumerate(
        _DEEPSEEK_DET_BOX_PATTERN.finditer(normalized), start=1
    ):
        if len(det_only) >= max_items:
            break
        try:
            fx0 = float(match.group("x0"))
            fy0 = float(match.group("y0"))
            fx1 = float(match.group("x1"))
            fy1 = float(match.group("y1"))
        except Exception:
            continue
        det_only.append(
            {
                "text": f"region_{idx}",
                "bbox": [fx0, fy0, fx1, fy1],
                "confidence": 0.45,
            }
        )

    return det_only or None


def _extract_deepseek_grounding_regions(
    text: Any, *, max_items: int = 256
) -> list[dict] | None:
    content = _extract_message_text(text)
    if not content:
        return None

    normalized = content
    for _ in range(2):
        decoded = html.unescape(normalized)
        if decoded == normalized:
            break
        normalized = decoded

    out: list[dict] = []
    seen: set[tuple[str, float, float, float, float]] = set()

    def _append_region(label_raw: str, x0: str, y0: str, x1: str, y1: str) -> None:
        if len(out) >= max_items:
            return
        label = _clean_deepseek_ref_text(label_raw)
        if not label:
            return
        try:
            fx0 = float(x0)
            fy0 = float(y0)
            fx1 = float(x1)
            fy1 = float(y1)
        except Exception:
            return
        key = (label, fx0, fy0, fx1, fy1)
        if key in seen:
            return
        seen.add(key)
        out.append(
            {
                "label": label,
                "bbox": [fx0, fy0, fx1, fy1],
            }
        )

    tokens: list[dict[str, Any]] = []
    for match in _DEEPSEEK_TAG_TOKEN_PATTERN.finditer(normalized):
        if match.group("ref") is not None:
            tokens.append(
                {
                    "type": "ref",
                    "text": str(match.group("ref_text") or ""),
                }
            )
            continue
        if match.group("det") is not None:
            tokens.append(
                {
                    "type": "det",
                    "bbox": (
                        str(match.group("x0") or ""),
                        str(match.group("y0") or ""),
                        str(match.group("x1") or ""),
                        str(match.group("y1") or ""),
                    ),
                }
            )

    i = 0
    while (i + 1) < len(tokens) and len(out) < max_items:
        a = tokens[i]
        b = tokens[i + 1]
        a_type = str(a.get("type") or "")
        b_type = str(b.get("type") or "")

        if a_type == "ref" and b_type == "det":
            x0, y0, x1, y1 = b.get("bbox") or ("", "", "", "")
            _append_region(str(a.get("text") or ""), x0, y0, x1, y1)
            i += 2
            continue

        if a_type == "det" and b_type == "ref":
            x0, y0, x1, y1 = a.get("bbox") or ("", "", "", "")
            _append_region(str(b.get("text") or ""), x0, y0, x1, y1)
            i += 2
            continue

        i += 1

    if not out:
        for match in _DEEPSEEK_REF_THEN_DET_PATTERN.finditer(normalized):
            _append_region(
                str(match.group("text") or ""),
                str(match.group("x0") or ""),
                str(match.group("y0") or ""),
                str(match.group("x1") or ""),
                str(match.group("y1") or ""),
            )
            if len(out) >= max_items:
                break

    return out or None
