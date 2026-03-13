"""Font mapping, text measurement, and OCR text fit utilities."""

from __future__ import annotations

import math
from typing import Any


def _map_font_name(name: str | None) -> str | None:
    if not name:
        return None
    n = str(name).strip()
    if not n:
        return None
    # Best-effort font mapping. PowerPoint will substitute missing fonts, but mapping
    # common PDF fonts to safe fonts helps consistency across platforms.
    mapping = {
        "Helvetica": "Arial",
        "Times-Roman": "Times New Roman",
        "Courier": "Courier New",
    }
    return mapping.get(n, n)


def _contains_cjk(text: str) -> bool:
    for ch in text or "":
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
            or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
            or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
        ):
            return True
    return False


def _is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
        or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
    )


def _char_width_factor(ch: str) -> float:
    """Very rough glyph width estimate relative to font size.

    We use this to pick a conservative font size for OCR text boxes without
    relying on Office-specific rendering APIs.
    """

    if not ch:
        return 0.0
    if ch.isspace():
        return 0.33
    if _is_cjk_char(ch):
        return 1.0
    # ASCII-ish heuristics.
    if "0" <= ch <= "9":
        return 0.58
    if "A" <= ch <= "Z":
        return 0.70
    if "a" <= ch <= "z":
        return 0.56
    # punctuation / symbols
    return 0.38


_MEASURE_FONT_CACHE: dict[tuple[int, bool], Any] = {}


def _try_load_measure_font(*, size_px: int, prefer_cjk: bool) -> Any | None:
    """Load a reasonably representative font for measuring text width.

    The PPT generator runs on Linux, while the resulting PPTX is viewed in
    Office/WPS on different OSes. We only need *approximate* metrics to decide
    font size and line breaks. If no suitable font is available, callers should
    fall back to heuristic width factors.
    """

    try:
        from PIL import ImageFont
    except Exception:
        return None

    key = (int(max(6, size_px)), bool(prefer_cjk))
    if key in _MEASURE_FONT_CACHE:
        return _MEASURE_FONT_CACHE[key]

    candidates: list[str] = []
    if prefer_cjk:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]
        )

    # Latin fallbacks (Arial-like) for width estimation.
    candidates.extend(
        [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )

    for path in candidates:
        try:
            font = ImageFont.truetype(path, size=key[0])
            _MEASURE_FONT_CACHE[key] = font
            return font
        except Exception:
            continue

    _MEASURE_FONT_CACHE[key] = None
    return None


def _measure_text_width_pt(
    text: str,
    *,
    font_size_pt: float,
    prefer_cjk: bool,
) -> float:
    """Best-effort text width in the same 'pt-like' space used by bbox_w_pt.

    We treat the font size (pt) as a pixel size at 72 DPI. That keeps ratios
    consistent and is sufficient for line-break/fit heuristics.
    """

    if not text:
        return 0.0

    font_size_pt = max(1.0, float(font_size_pt))
    font = _try_load_measure_font(
        size_px=int(round(font_size_pt)),
        prefer_cjk=prefer_cjk,
    )
    if font is None:
        return sum(_char_width_factor(ch) for ch in text) * font_size_pt

    # Pillow >=8 exposes getlength() for accurate advance-width measurement.
    try:
        width = float(font.getlength(text))  # type: ignore[attr-defined]
        if math.isfinite(width) and width > 0.0:
            return width
    except Exception:
        pass

    try:
        bbox = font.getbbox(text)
        width = float(bbox[2] - bbox[0])
        if math.isfinite(width) and width > 0.0:
            return width
    except Exception:
        pass

    return sum(_char_width_factor(ch) for ch in text) * font_size_pt


def _measure_text_lines(
    text: str,
    *,
    max_width_pt: float,
    font_size_pt: float,
    wrap: bool,
) -> tuple[int, float]:
    """Return (line_count, max_line_width_pt) for a text string."""

    if not text:
        return (0, 0.0)

    max_width_pt = max(1.0, float(max_width_pt))
    font_size_pt = max(1.0, float(font_size_pt))

    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paragraphs = [p for p in paragraphs if p.strip()]
    if not paragraphs:
        return (0, 0.0)

    total_lines = 0
    max_line_w = 0.0

    for para in paragraphs:
        prefer_cjk = _contains_cjk(para)
        if not wrap:
            w = _measure_text_width_pt(
                para,
                font_size_pt=font_size_pt,
                prefer_cjk=prefer_cjk,
            )
            total_lines += 1
            max_line_w = max(max_line_w, w)
            continue

        wrapped = _wrap_paragraph_to_lines(
            para, max_width_pt=max_width_pt, font_size_pt=font_size_pt
        )
        if not wrapped:
            wrapped = [para]
        total_lines += len(wrapped)
        for line in wrapped:
            line_w = _measure_text_width_pt(
                line,
                font_size_pt=font_size_pt,
                prefer_cjk=prefer_cjk,
            )
            max_line_w = max(max_line_w, float(line_w))

    return (total_lines, float(max_line_w))


def _tokenize_for_wrap(para: str) -> list[str]:
    if not para:
        return []

    if (not _contains_cjk(para)) and (" " in para):
        tokens: list[str] = []
        parts = [p for p in para.split(" ") if p != ""]
        for i, part in enumerate(parts):
            if i > 0:
                tokens.append(" ")
            tokens.append(part)
        return tokens

    # Mixed CJK/ASCII text needs token-level grouping for contiguous ASCII runs
    # (e.g. "API", "HTTP", "A/B") so line-wrap does not split them into
    # awkward fragments like "AP" + "I".
    def _is_ascii_word_char(ch: str) -> bool:
        return bool(ch) and ch.isascii() and (ch.isalnum() or ch in "_-./:+#%&@")

    out: list[str] = []
    i = 0
    n = len(para)
    while i < n:
        ch = para[i]
        if ch.isspace():
            if not out or out[-1] != " ":
                out.append(" ")
            i += 1
            continue
        if _is_ascii_word_char(ch):
            j = i + 1
            while j < n and _is_ascii_word_char(para[j]):
                j += 1
            out.append(para[i:j])
            i = j
            continue
        out.append(ch)
        i += 1

    return out


def _token_width_pt(token: str, *, font_size_pt: float, prefer_cjk: bool) -> float:
    return _measure_text_width_pt(
        token,
        font_size_pt=float(font_size_pt),
        prefer_cjk=bool(prefer_cjk),
    )


def _wrap_paragraph_to_lines(
    para: str, *, max_width_pt: float, font_size_pt: float
) -> list[str]:
    max_width_pt = max(1.0, float(max_width_pt))
    font_size_pt = max(1.0, float(font_size_pt))
    if not para:
        return [""]

    tokens = _tokenize_for_wrap(para)
    prefer_cjk = _contains_cjk(para)
    lines: list[str] = []
    current_tokens: list[str] = []
    current_width = 0.0

    def _flush_current() -> None:
        nonlocal current_tokens, current_width
        if not current_tokens:
            return
        line = "".join(current_tokens).rstrip()
        if line:
            lines.append(line)
        current_tokens = []
        current_width = 0.0

    for token in tokens:
        token_w = _token_width_pt(
            token, font_size_pt=font_size_pt, prefer_cjk=prefer_cjk
        )
        if token == " " and not current_tokens:
            continue

        if token_w <= max_width_pt:
            if current_width <= 0.0:
                current_tokens = [token]
                current_width = token_w
                continue
            if current_width + token_w <= max_width_pt:
                current_tokens.append(token)
                current_width += token_w
                continue
            _flush_current()
            if token != " ":
                current_tokens = [token]
                current_width = token_w
            continue

        # Token itself is wider than one line; split by character.
        for ch in token:
            ch_w = _measure_text_width_pt(
                ch,
                font_size_pt=font_size_pt,
                prefer_cjk=prefer_cjk,
            )
            if current_width <= 0.0:
                current_tokens = [ch]
                current_width = ch_w
                continue
            if current_width + ch_w <= max_width_pt:
                current_tokens.append(ch)
                current_width += ch_w
                continue
            _flush_current()
            current_tokens = [ch]
            current_width = ch_w

    _flush_current()
    if not lines:
        return [para]

    # Punctuation guards: avoid line breaks that leave closing punctuation at
    # the beginning of a line (e.g. "：") or opening punctuation at the end of
    # a line (e.g. "（"). This improves visual fidelity for CJK headings.
    NO_BREAK_BEFORE = set(",.;:!?)]}、，。！？：；）】」』》〉%‰°")
    NO_BREAK_AFTER = set("([{（《【「『“‘")

    out = [str(seg or "") for seg in lines]
    for _ in range(3):
        changed = False
        for i in range(1, len(out)):
            prev = out[i - 1]
            cur = out[i]
            if not prev or not cur:
                continue

            while cur and cur[0] in NO_BREAK_BEFORE and prev:
                prev = prev + cur[0]
                cur = cur[1:].lstrip()
                changed = True
                if not cur:
                    break

            while prev and prev[-1] in NO_BREAK_AFTER and cur:
                cur = prev[-1] + cur
                prev = prev[:-1].rstrip()
                changed = True
                if not prev:
                    break

            out[i - 1] = prev
            out[i] = cur

        if not changed:
            break

    out = [seg for seg in (s.strip() for s in out) if seg]
    return out if out else [para]


def _wrap_text_to_width(text: str, *, max_width_pt: float, font_size_pt: float) -> str:
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    wrapped_lines: list[str] = []
    for para in paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue
        wrapped_lines.extend(
            _wrap_paragraph_to_lines(
                cleaned, max_width_pt=max_width_pt, font_size_pt=font_size_pt
            )
        )
    return "\n".join([line for line in wrapped_lines if line.strip()])


def _fit_font_size_pt(
    text: str,
    *,
    bbox_w_pt: float,
    bbox_h_pt: float,
    wrap: bool,
    min_pt: float = 6.0,
    max_pt: float = 48.0,
    width_fit_ratio: float = 0.98,
    height_fit_ratio: float = 0.98,
) -> float:
    """Pick a conservative font size for OCR text in a fixed bbox."""

    text = str(text or "").strip()
    if not text:
        return float(min_pt)

    bbox_w_pt = max(1.0, float(bbox_w_pt))
    bbox_h_pt = max(1.0, float(bbox_h_pt))

    # A rough line-height multiplier; PowerPoint text metrics vary by font, but
    # we want to avoid overflow in common viewers (Office/WPS/Google Slides).
    line_height = 1.18 if _contains_cjk(text) else 1.15

    lo = max(1.0, float(min_pt))
    hi = min(float(max_pt), float(bbox_h_pt))
    width_ratio = max(0.85, min(1.20, float(width_fit_ratio)))
    height_ratio = max(0.85, min(1.20, float(height_fit_ratio)))

    # For wrapped text, layout fit is not monotonic (line breaks jump between
    # candidate sizes), so binary search can get trapped in tiny fonts.
    if wrap:
        step = 0.2
        size = hi
        while size >= lo:
            lines, max_line_w = _measure_text_lines(
                text, max_width_pt=bbox_w_pt, font_size_pt=size, wrap=wrap
            )
            lines = max(1, int(lines))
            total_h = float(lines) * float(size) * float(line_height)
            width_ok = max_line_w <= (bbox_w_pt * width_ratio)
            height_ok = total_h <= (bbox_h_pt * height_ratio)
            if width_ok and height_ok:
                return max(float(min_pt), min(float(max_pt), round(float(size), 1)))
            size -= step
        return max(float(min_pt), min(float(max_pt), round(float(lo), 1)))

    best = lo
    # Non-wrap is close to monotonic; binary search is fine.
    for _ in range(14):
        mid = (lo + hi) / 2.0
        lines, max_line_w = _measure_text_lines(
            text, max_width_pt=bbox_w_pt, font_size_pt=mid, wrap=wrap
        )
        lines = max(1, int(lines))
        total_h = float(lines) * float(mid) * float(line_height)

        width_ok = max_line_w <= (bbox_w_pt * width_ratio)
        height_ok = total_h <= (bbox_h_pt * height_ratio)

        if width_ok and height_ok:
            best = mid
            lo = mid
        else:
            hi = mid

    return max(float(min_pt), min(float(max_pt), round(float(best), 1)))


def _compact_text_length(text: str) -> int:
    return len("".join(ch for ch in str(text or "") if not ch.isspace()))


def _is_inline_short_token(text: str) -> bool:
    """Heuristic: short parenthetical/label-like token, often not body text."""

    raw = str(text or "").strip()
    if not raw:
        return False
    compact_len = _compact_text_length(raw)
    if compact_len <= 3:
        return True
    if compact_len <= 12 and ("(" in raw or ")" in raw or "/" in raw):
        return True
    alpha = sum(1 for ch in raw if ch.isalpha())
    cjk = sum(1 for ch in raw if "一" <= ch <= "鿿")
    digit = sum(1 for ch in raw if ch.isdigit())
    punct = sum(1 for ch in raw if not ch.isalnum() and not ch.isspace())
    if compact_len <= 6 and alpha >= 2 and cjk == 0 and punct <= 2:
        return True
    if compact_len <= 6 and digit >= 2 and cjk == 0:
        return True
    return False


def _normalize_ocr_text_for_render(text: str) -> str:
    """Normalize OCR text while preserving meaningful line structure."""

    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines)


def _split_heading_text_after_colon(text: str) -> str:
    """Split heading text at the first colon when it has a meaningful tail.

    This helps bilingual heading patterns like:
    - "能力（The Abilities）：插件/工具（Plugins/Tools）"
    - "记忆 (The Memory) : RAG & Embedding"
    so we can keep a stable, explicit line break and avoid viewer-specific
    trailing-character wraps.
    """

    normalized = _normalize_ocr_text_for_render(text)
    if not normalized or "\n" in normalized:
        return normalized

    def _has_ascii_alpha(s: str) -> bool:
        return any(ch.isascii() and ch.isalpha() for ch in (s or ""))

    for sep in ("：", ":"):
        split_at = normalized.find(sep)
        if split_at < 2 or split_at >= (len(normalized) - 2):
            continue
        left_part = normalized[: split_at + 1].strip()
        right_part = normalized[split_at + 1 :].strip()
        if not left_part or not right_part:
            continue
        if _compact_text_length(right_part) < 2:
            continue
        left_has_paren = ("(" in left_part and ")" in left_part) or (
            "（" in left_part and "）" in left_part
        )
        right_has_paren = ("(" in right_part and ")" in right_part) or (
            "（" in right_part and "）" in right_part
        )
        right_has_struct_tail = any(
            token in right_part for token in ("/", "&", "+", "、")
        )
        has_bilingual_signal = _has_ascii_alpha(left_part) or _has_ascii_alpha(
            right_part
        )
        if not left_has_paren:
            continue
        if not (right_has_paren or right_has_struct_tail or has_bilingual_signal):
            continue
        return f"{left_part}\n{right_part}"

    return normalized


def _fit_mineru_text_style(
    *,
    text: str,
    bbox_w_pt: float,
    bbox_h_pt: float,
    page_w_pt: float,
    page_h_pt: float,
    y0_pt: float,
    mineru_block_type: str | None,
    mineru_text_level: int | None,
) -> tuple[str, float, bool, bool, bool]:
    """Return (text_to_render, font_size_pt, wrap, is_heading, is_primary_heading)."""

    normalized = _normalize_ocr_text_for_render(text)
    if not normalized:
        return ("", 6.0, False, False, False)

    bbox_w_pt = max(1.0, float(bbox_w_pt))
    bbox_h_pt = max(1.0, float(bbox_h_pt))
    page_w_pt = max(1.0, float(page_w_pt))
    page_h_pt = max(1.0, float(page_h_pt))
    y0_pt = max(0.0, float(y0_pt))

    text_to_fit = normalized
    is_bullet_like = text_to_fit.lstrip().startswith(("-", "•", "·", "●"))
    plain_len = len(text_to_fit.replace("\n", ""))

    block_type = str(mineru_block_type or "").strip().lower()
    text_level: int | None = None
    if mineru_text_level is not None:
        try:
            text_level = int(mineru_text_level)
        except Exception:
            text_level = None

    semantic_heading_tokens = {
        "title",
        "heading",
        "header",
        "h1",
        "h2",
        "subtitle",
        "subheading",
        "section_title",
        "paragraph_title",
        "title_1",
        "title_2",
        "title_3",
    }
    is_semantic_heading = bool(
        block_type in semantic_heading_tokens
        or any(
            token in block_type for token in ("title", "heading", "header", "subtitle")
        )
    )

    is_heading = bool(is_semantic_heading) or (
        (text_level is not None and text_level <= 2 and plain_len <= 60)
        or (
            y0_pt <= 0.22 * page_h_pt
            and bbox_h_pt >= 18.0
            and plain_len <= 56
            and (not is_bullet_like)
        )
    )
    is_primary_heading = bool(
        is_heading and y0_pt <= 0.16 * page_h_pt and bbox_w_pt >= 0.34 * page_w_pt
    )

    source_heading_text = text_to_fit
    if is_heading:
        text_to_fit = " ".join(
            [
                line.strip()
                for line in str(text_to_fit or "").split("\n")
                if line.strip()
            ]
        ).strip()

    wrap_for_fit = bool(not is_heading)
    max_body_pt = min(
        96.0 if is_primary_heading else 72.0,
        max(7.0, (0.98 if is_heading else 0.94) * float(bbox_h_pt)),
    )
    min_body_pt = 6.0
    line_height = 1.18 if _contains_cjk(text_to_fit or normalized) else 1.15
    prefit_font_size_pt: float | None = None

    if is_heading:
        wrap_for_fit = False
        prefit_font_size_pt = _fit_font_size_pt(
            text_to_fit,
            bbox_w_pt=bbox_w_pt,
            bbox_h_pt=bbox_h_pt,
            wrap=False,
            min_pt=min_body_pt,
            max_pt=max_body_pt,
            width_fit_ratio=1.00,
            height_fit_ratio=0.995,
        )

    if prefit_font_size_pt is not None:
        font_size_pt = float(prefit_font_size_pt)
    else:
        font_size_pt = _fit_font_size_pt(
            text_to_fit,
            bbox_w_pt=bbox_w_pt,
            bbox_h_pt=bbox_h_pt,
            wrap=wrap_for_fit,
            min_pt=min_body_pt,
            max_pt=max_body_pt,
            width_fit_ratio=1.03 if wrap_for_fit else 1.00,
            height_fit_ratio=0.96 if wrap_for_fit else 0.995,
        )

    text_to_render = text_to_fit
    if is_heading:
        single_line_font_size_pt = float(font_size_pt)
        single_line_fill_ratio = (
            float(single_line_font_size_pt) * float(line_height)
        ) / max(1.0, float(bbox_h_pt))
        multiline_candidates: list[str] = []
        seen_multiline_candidates: set[str] = set()
        for candidate in (
            _normalize_ocr_text_for_render(source_heading_text),
            _split_heading_text_after_colon(text_to_fit),
        ):
            cleaned_candidate = _normalize_ocr_text_for_render(candidate)
            if (
                not cleaned_candidate
                or "\n" not in cleaned_candidate
                or cleaned_candidate == text_to_fit
                or cleaned_candidate in seen_multiline_candidates
            ):
                continue
            seen_multiline_candidates.add(cleaned_candidate)
            multiline_candidates.append(cleaned_candidate)

        best_multiline_text: str | None = None
        best_multiline_font_pt = float(single_line_font_size_pt)
        best_multiline_fill_ratio = float(single_line_fill_ratio)

        for candidate_text in multiline_candidates:
            candidate_lines = [
                line for line in candidate_text.splitlines() if line.strip()
            ]
            if len(candidate_lines) < 2:
                continue
            candidate_font_pt = _fit_font_size_pt(
                candidate_text,
                bbox_w_pt=bbox_w_pt,
                bbox_h_pt=bbox_h_pt,
                wrap=True,
                min_pt=min_body_pt,
                max_pt=max_body_pt,
                width_fit_ratio=1.02,
                height_fit_ratio=0.985,
            )
            candidate_fill_ratio = (
                float(len(candidate_lines))
                * float(candidate_font_pt)
                * float(line_height)
            ) / max(1.0, float(bbox_h_pt))
            if candidate_fill_ratio > 0.995:
                continue
            if candidate_font_pt <= (single_line_font_size_pt + 0.6):
                continue
            if candidate_fill_ratio <= (single_line_fill_ratio + 0.18):
                continue
            if candidate_font_pt <= best_multiline_font_pt:
                continue
            best_multiline_text = candidate_text
            best_multiline_font_pt = float(candidate_font_pt)
            best_multiline_fill_ratio = float(candidate_fill_ratio)

        if best_multiline_text is not None and (
            best_multiline_font_pt >= (single_line_font_size_pt * 1.12)
            or best_multiline_fill_ratio >= (single_line_fill_ratio + 0.24)
        ):
            text_to_render = best_multiline_text
            font_size_pt = float(best_multiline_font_pt)
            wrap_for_fit = True

    if wrap_for_fit and not (is_heading and "\n" in text_to_render):
        candidate_text = text_to_fit
        for _ in range(12):
            wrap_width_pt = max(
                1.0,
                float(bbox_w_pt)
                + max(
                    2.4,
                    0.20 * float(bbox_h_pt),
                    0.42 * float(font_size_pt),
                ),
            )
            candidate_text = _wrap_text_to_width(
                text_to_fit,
                max_width_pt=wrap_width_pt,
                font_size_pt=float(font_size_pt),
            )
            candidate_lines = [
                line for line in candidate_text.splitlines() if line.strip()
            ]
            if not candidate_lines:
                candidate_lines = [text_to_fit]
                candidate_text = text_to_fit
            line_height = 1.18 if _contains_cjk(text_to_fit) else 1.15
            total_h = float(len(candidate_lines)) * float(font_size_pt) * line_height
            if total_h <= (0.985 * bbox_h_pt):
                text_to_render = candidate_text
                break
            font_size_pt = max(float(min_body_pt), float(font_size_pt) - 0.35)
        else:
            text_to_render = candidate_text if candidate_text else text_to_fit

    return (
        text_to_render,
        float(font_size_pt),
        bool(wrap_for_fit),
        bool(is_heading),
        bool(is_primary_heading),
    )


def _prefer_wrap_for_ocr_text(
    *,
    text: str,
    bbox_w_pt: float,
    bbox_h_pt: float,
    baseline_ocr_h_pt: float,
) -> bool:
    """Heuristic wrap decision for scanned OCR text.

    The goal is to be robust across pages/models instead of relying on a fixed
    threshold: estimate both width and likely line count from geometry/text.
    """

    compact_len = _compact_text_length(text)
    if compact_len <= 0:
        return False
    if "\n" in text:
        return True

    w = max(1.0, float(bbox_w_pt))
    h = max(1.0, float(bbox_h_pt))
    baseline = max(4.0, float(baseline_ocr_h_pt))

    # Very line-like boxes generally should not wrap.
    #
    # Many OCR engines return slightly padded bboxes (h ~ 1.2-1.4x baseline)
    # even for single visual lines. Treat those as single-line and prefer
    # shrinking font size (non-wrap) over inserting synthetic line breaks,
    # which often causes "wrong wrap" reports from users.
    # OCR bbox heights are often padded (especially for CJK + punctuation).
    # If we are not clearly multi-line, prefer *no wrap* and rely on font-size
    # fitting + right-side slack to avoid spurious line breaks like
    # "标题（Title）\n：".
    # NOTE: Some OCR backends (notably PaddleOCR-VL doc_parser) can emit
    # paragraph-like bboxes that are only ~1.4-1.6x the typical line height.
    # Treating those as "single line" makes the font fitter shrink text into
    # illegible tiny sizes. Keep the single-line guard stricter so we still
    # wrap these moderate-height blocks.
    if h <= max(1.45 * baseline, 10.5) and compact_len <= 120:
        return False

    width_pressure = float(compact_len) / max(1.0, w)
    # Height-based line estimation: use a slightly larger divisor to avoid
    # misclassifying single-line headers as 2-line blocks.
    est_lines_by_height = max(1, int(round(h / max(8.0, 1.10 * baseline))))

    if est_lines_by_height >= 2:
        return True

    # Only use width-pressure-based wrapping when the bbox is not line-like.
    # For near-single-line boxes, it's more robust to keep one line and let
    # font fitting shrink the size a bit, rather than forcing a wrap that may
    # not match the original slide.
    if h >= (1.35 * baseline):
        if _contains_cjk(text):
            if compact_len >= 18 and width_pressure >= 0.090:
                return True
            if compact_len >= 28 and width_pressure >= 0.075:
                return True
        else:
            if compact_len >= 22 and width_pressure >= 0.080:
                return True
            if compact_len >= 36 and width_pressure >= 0.065:
                return True

    return False


def _resolve_visual_wrap_override_for_ocr_text(
    *,
    visual_line_count: int | None,
    compact_len: int,
    bbox_h_pt: float,
    baseline_ocr_h_pt: float,
    is_heading: bool,
) -> bool | None:
    """Resolve whether source-pixel line counting should override OCR wrap.

    A single visually-detected text row is a strong negative signal for short
    labels/headings, but only a weak negative signal for long OCR paragraphs.
    Some OCR backends return long paragraph text in shallow bboxes; forcing
    `wrap=False` in those cases prevents the fitter from recovering with a
    wrapped layout and results in tiny illegible text.
    """

    if not isinstance(visual_line_count, int) or visual_line_count < 1:
        return None

    if visual_line_count >= 2:
        return True if int(compact_len) >= 12 else None

    if is_heading:
        return False

    compact_len = max(0, int(compact_len))
    bbox_h_pt = max(0.0, float(bbox_h_pt))
    baseline = max(4.0, float(baseline_ocr_h_pt))

    if compact_len >= 28:
        return None
    if compact_len >= 18 and bbox_h_pt >= max(1.10 * baseline, 10.0):
        return None
    return False


def _fit_ocr_text_style(
    *,
    text: str,
    bbox_w_pt: float,
    bbox_h_pt: float,
    baseline_ocr_h_pt: float,
    is_heading: bool,
    wrap_override: bool | None = None,
) -> tuple[str, float, bool]:
    """Return (text_to_render, font_size_pt, wrap) for OCR text boxes.

    Compare single-line and wrapped layouts using actual measurement, then pick
    the more readable fit. Keep only a small amount of policy around headings
    and explicit line structure.
    """

    normalized = _normalize_ocr_text_for_render(text)
    if not normalized:
        return ("", 6.0, False)

    compact_len = _compact_text_length(normalized)
    line_height = 1.18 if _contains_cjk(normalized) else 1.15

    bbox_w_pt = max(1.0, float(bbox_w_pt))
    bbox_h_pt = max(1.0, float(bbox_h_pt))
    baseline_ocr_h_pt = max(4.0, float(baseline_ocr_h_pt))

    min_pt = max(5.0, min(8.0, 0.52 * float(baseline_ocr_h_pt)))
    max_pt = min(
        84.0 if is_heading else 54.0,
        max(7.0, float(bbox_h_pt) * (0.98 if is_heading else 0.94)),
    )

    def _fit_single_candidate(
        *,
        max_pt_override: float | None = None,
        height_fit_ratio: float = 0.995,
    ) -> tuple[str, float, int, float]:
        resolved_max_pt = (
            float(max_pt)
            if max_pt_override is None
            else max(float(min_pt), float(max_pt_override))
        )
        font_size_pt = _fit_font_size_pt(
            normalized,
            bbox_w_pt=float(bbox_w_pt),
            bbox_h_pt=float(bbox_h_pt),
            wrap=False,
            min_pt=float(min_pt),
            max_pt=float(resolved_max_pt),
            width_fit_ratio=1.00,
            height_fit_ratio=float(height_fit_ratio),
        )
        fill_ratio = (float(font_size_pt) * float(line_height)) / max(
            1.0, float(bbox_h_pt)
        )
        return (normalized, float(font_size_pt), 1, float(fill_ratio))

    def _fit_wrapped_text(
        *,
        seed_font_pt: float,
        min_pt: float,
        bbox_w_pt: float,
        bbox_h_pt: float,
    ) -> tuple[str, float, int, float]:
        wrapped_font_pt = float(seed_font_pt)
        wrapped_text = normalized
        wrapped_lines_count = 1
        for _ in range(14):
            candidate_text = _wrap_text_to_width(
                normalized,
                max_width_pt=max(1.0, 1.01 * float(bbox_w_pt)),
                font_size_pt=float(wrapped_font_pt),
            )
            candidate_lines = [
                line for line in candidate_text.splitlines() if line.strip()
            ]
            if not candidate_lines:
                candidate_lines = [normalized]
                candidate_text = normalized
            total_h = (
                float(len(candidate_lines))
                * float(wrapped_font_pt)
                * float(line_height)
            )
            wrapped_text = candidate_text
            wrapped_lines_count = len(candidate_lines)
            if total_h <= (0.985 * float(bbox_h_pt)):
                break
            wrapped_font_pt = max(float(min_pt), float(wrapped_font_pt) - 0.32)

        fill_ratio = (
            float(wrapped_lines_count) * float(wrapped_font_pt) * float(line_height)
        ) / max(1.0, float(bbox_h_pt))
        return (
            wrapped_text,
            float(wrapped_font_pt),
            int(wrapped_lines_count),
            float(fill_ratio),
        )

    explicit_multiline = "\n" in normalized
    single_text, single_font_pt, _single_lines, single_fill = _fit_single_candidate()

    if is_heading and not explicit_multiline:
        return (single_text, float(single_font_pt), False)

    wrapped_seed = _fit_font_size_pt(
        normalized,
        bbox_w_pt=max(1.0, 1.01 * float(bbox_w_pt)),
        bbox_h_pt=float(bbox_h_pt),
        wrap=True,
        min_pt=float(min_pt),
        max_pt=float(max_pt),
        width_fit_ratio=1.02,
        height_fit_ratio=0.95,
    )
    wrapped_text, wrapped_font_pt, wrapped_lines, wrapped_fill = _fit_wrapped_text(
        seed_font_pt=float(wrapped_seed),
        min_pt=float(min_pt),
        bbox_w_pt=float(bbox_w_pt),
        bbox_h_pt=float(bbox_h_pt),
    )

    if explicit_multiline and wrapped_lines >= 2:
        return (wrapped_text, float(wrapped_font_pt), True)

    min_readable_pt = max(7.0, min(11.0, 0.62 * float(baseline_ocr_h_pt)))
    choose_wrap = False

    if wrap_override is True:
        choose_wrap = wrapped_lines >= 2
    elif wrap_override is False:
        choose_wrap = (
            (not is_heading)
            and wrapped_lines >= 2
            and (
                compact_len >= 72
                or float(single_font_pt) < float(min_readable_pt)
                or float(wrapped_font_pt) >= (float(single_font_pt) * 1.12)
            )
        )
    elif wrapped_lines >= 2:
        single_is_tiny = float(single_font_pt) < float(min_readable_pt)
        wrapped_is_clearly_better = float(wrapped_font_pt) >= (
            float(single_font_pt) * 1.10
        )
        wrapped_fills_box_better = float(wrapped_fill) >= (
            float(single_fill) + 0.16
        ) and float(wrapped_font_pt) >= (float(single_font_pt) * 1.02)
        choose_wrap = bool(
            single_is_tiny or wrapped_is_clearly_better or wrapped_fills_box_better
        )

    if choose_wrap and wrapped_lines >= 2:
        return (wrapped_text, float(wrapped_font_pt), True)

    # If we are staying single-line, do one more pass that tries to fill the OCR
    # box geometrically instead of stopping at the generic body-text cap. This
    # helps large OCR title boxes that were not classified as headings.
    if not explicit_multiline:
        allow_single_line_fill_expand = bool(
            is_heading
            or wrap_override is False
            or compact_len <= 56
            or bbox_h_pt >= (1.12 * float(baseline_ocr_h_pt))
        )
        if allow_single_line_fill_expand:
            expanded_max_pt = min(96.0, max(float(max_pt), 0.995 * float(bbox_h_pt)))
            if expanded_max_pt > (float(single_font_pt) + 0.8):
                _, expanded_font_pt, _expanded_lines, expanded_fill = (
                    _fit_single_candidate(
                        max_pt_override=float(expanded_max_pt),
                        height_fit_ratio=0.999,
                    )
                )
                if (
                    expanded_font_pt >= max(
                        float(single_font_pt) + 0.8,
                        float(single_font_pt) * 1.08,
                    )
                    and expanded_fill >= min(0.995, float(single_fill) + 0.10)
                ):
                    return (single_text, float(expanded_font_pt), False)
    return (single_text, float(single_font_pt), False)
