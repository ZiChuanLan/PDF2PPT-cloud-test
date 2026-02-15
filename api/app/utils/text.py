"""Shared text normalization helpers."""

from __future__ import annotations

import re


def clean_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned if cleaned else None
