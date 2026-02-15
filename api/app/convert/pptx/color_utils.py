"""Color utility helpers for PPTX generation."""

from __future__ import annotations

def _hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    if not color:
        return None
    s = str(color).strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


def _rgb_luma(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return (0.2126 * float(r)) + (0.7152 * float(g)) + (0.0722 * float(b))


def _rgb_sq_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    dr = int(a[0]) - int(b[0])
    dg = int(a[1]) - int(b[1])
    db = int(a[2]) - int(b[2])
    return (dr * dr) + (dg * dg) + (db * db)


def _pick_contrasting_text_rgb(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (0, 0, 0) if _rgb_luma(bg_rgb) >= 130.0 else (255, 255, 255)
