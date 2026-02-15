"""JSON extraction and recovery helpers for OCR payloads."""

import ast
import json
from typing import Any

def _extract_items_from_json_payload(
    value: Any, *, _depth: int = 0
) -> list[dict] | None:
    if _depth > 4:
        return None

    if isinstance(value, list):
        rows = [item for item in value if isinstance(item, dict)]
        return rows or None

    if not isinstance(value, dict):
        return None

    if any(
        key in value
        for key in (
            "bbox",
            "box",
            "bounding_box",
            "location",
            "rect",
            "points",
            "polygon",
            "position",
            "coordinates",
            "quad",
            "b",
            "bbox_2d",
        )
    ) and any(
        key in value
        for key in (
            "text",
            "words",
            "content",
            "transcription",
            "value",
            "label",
            "t",
        )
    ):
        return [value]

    preferred_keys = (
        "items",
        "result",
        "results",
        "data",
        "lines",
        "blocks",
        "text_blocks",
        "ocr",
        "output",
    )
    for key in preferred_keys:
        if key not in value:
            continue
        extracted = _extract_items_from_json_payload(value.get(key), _depth=_depth + 1)
        if extracted:
            return extracted

    for candidate in value.values():
        extracted = _extract_items_from_json_payload(candidate, _depth=_depth + 1)
        if extracted:
            return extracted
    return None


def _extract_partial_json_array_items(
    text: str, *, max_items: int = 1000
) -> list[dict]:
    """Best-effort parse of a possibly truncated JSON array payload."""

    if not text:
        return []

    decoder = json.JSONDecoder()
    n = len(text)
    bracket_positions: list[int] = []
    pos = -1
    while len(bracket_positions) < 16:
        pos = text.find("[", pos + 1)
        if pos < 0:
            break
        bracket_positions.append(pos)

    best: list[dict] = []

    for start in bracket_positions:
        i = start + 1
        out: list[dict] = []

        while i < n and len(out) < max_items:
            while i < n and text[i] in " \t\r\n,":
                i += 1
            if i >= n or text[i] == "]":
                break

            try:
                value, next_i = decoder.raw_decode(text, i)
            except Exception:
                # Most likely truncated tail; keep successfully decoded prefix.
                break

            extracted = _extract_items_from_json_payload(value)
            if extracted:
                for row in extracted:
                    out.append(row)
                    if len(out) >= max_items:
                        break
            elif isinstance(value, dict):
                out.append(value)

            if next_i <= i:
                i += 1
            else:
                i = next_i

        if len(out) > len(best):
            best = out
            if len(best) >= max_items:
                break

    return best


def _parse_relaxed_json(candidate: str) -> Any | None:
    try:
        return json.loads(candidate)
    except Exception:
        pass

    try:
        return ast.literal_eval(candidate)
    except Exception:
        return None


def _extract_balanced_object_snippets(
    text: str, *, max_items: int = 1000
) -> list[dict]:
    if not text:
        return []

    n = len(text)
    i = 0
    depth = 0
    start_idx = -1
    in_string = False
    escaped = False
    out: list[dict] = []

    while i < n and len(out) < max_items:
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
            i += 1
            continue

        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx >= 0:
                snippet = text[start_idx : i + 1]
                parsed = _parse_relaxed_json(snippet)
                extracted = _extract_items_from_json_payload(parsed)
                if extracted:
                    for row in extracted:
                        out.append(row)
                        if len(out) >= max_items:
                            break
                elif isinstance(parsed, dict):
                    out.append(parsed)
                start_idx = -1
            i += 1
            continue

        i += 1

    return out[:max_items]


def _extract_partial_json_object_list(
    text: str, *, max_items: int = 1000
) -> list[dict]:
    if not text:
        return []

    items = _extract_partial_json_array_items(text, max_items=max_items)
    if items:
        return items[:max_items]

    decoder = json.JSONDecoder()
    n = len(text)
    i = 0
    out: list[dict] = []

    while i < n and len(out) < max_items:
        start = text.find("{", i)
        if start < 0:
            break

        try:
            parsed_obj, next_i = decoder.raw_decode(text, start)
        except Exception:
            i = start + 1
            continue

        extracted = _extract_items_from_json_payload(parsed_obj)
        if extracted:
            for row in extracted:
                out.append(row)
                if len(out) >= max_items:
                    break
        elif isinstance(parsed_obj, dict):
            out.append(parsed_obj)

        if next_i <= start:
            i = start + 1
        else:
            i = next_i

    if out:
        return out[:max_items]

    relaxed = _extract_balanced_object_snippets(text, max_items=max_items)
    if relaxed:
        return relaxed[:max_items]

    return []


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                cleaned = part.strip()
                if cleaned:
                    chunks.append(cleaned)
                continue
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if not isinstance(text_value, str):
                text_value = part.get("content")
            if isinstance(text_value, str):
                cleaned = text_value.strip()
                if cleaned:
                    chunks.append(cleaned)
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content)


def _extract_json_list(text: Any) -> list[dict] | None:
    content = _extract_message_text(text)
    if not content:
        return None

    candidates: list[str] = [content]
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()
        if fenced:
            candidates.append(fenced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            extracted = _extract_items_from_json_payload(parsed)
            if extracted:
                return extracted
        except Exception:
            pass

        start_idx = candidate.find("[")
        end_idx = candidate.rfind("]")
        if start_idx >= 0 and end_idx > start_idx:
            clipped = candidate[start_idx : end_idx + 1]
            try:
                parsed = json.loads(clipped)
                extracted = _extract_items_from_json_payload(parsed)
                if extracted:
                    return extracted
            except Exception:
                pass

        partial = _extract_partial_json_object_list(candidate)
        if partial:
            return partial

    return None
