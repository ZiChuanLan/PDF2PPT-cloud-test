"""Shared OCR base types, constants, and helpers."""

import os
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse

from app.utils.text import clean_str

from ...utils.concurrency import run_in_daemon_thread_with_timeout

logger = logging.getLogger(__name__)

# PaddleX (used by PaddleOCR 3.x / PaddleOCR-VL pipelines) performs a network
# connectivity check to model hosters on import. In some environments this can
# take a *very* long time or even hang, which in turn makes OCR jobs appear
# "stuck" during initialization. Prefer skipping this check by default.
#
# Users can override by explicitly setting this env var before launching.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_ACRONYM_ALLOWLIST = {
    "AI",
    "API",
    "LLM",
    "RAG",
    "NLP",
    "OCR",
    "PDF",
    "PPT",
    "PPTX",
    "GPT",
    "CPU",
    "GPU",
    "HTTP",
    "HTTPS",
    "JSON",
    "SQL",
    "UI",
    "UX",
    "SDK",
    "IDE",
    "ETL",
}


_VALID_AI_OCR_PROVIDERS = {
    "auto",
    "openai",
    "siliconflow",
    "ppio",
    "novita",
    "deepseek",
}

_AI_OCR_PROVIDER_ALIASES = {
    "": "auto",
    "auto": "auto",
    "openai": "openai",
    "openai_compatible": "openai",
    "openai-compatible": "openai",
    "siliconflow": "siliconflow",
    "silicon_flow": "siliconflow",
    "sf": "siliconflow",
    "ppio": "ppio",
    "ppinfra": "ppio",
    "novita": "novita",
    "deepseek": "deepseek",
    "deep_seek": "deepseek",
}

_PADDLE_OCR_VL_MODEL_V1 = "PaddlePaddle/PaddleOCR-VL"
_PADDLE_OCR_VL_MODEL_V15 = "PaddlePaddle/PaddleOCR-VL-1.5"
_DEFAULT_PADDLE_OCR_VL_MODEL = _PADDLE_OCR_VL_MODEL_V1
_DEFAULT_PADDLE_DOC_BACKEND = "vllm-server"
_VALID_PADDLE_DOC_BACKENDS = {"vllm-server", "sglang-server"}


def _clean_str(value: str | None) -> str | None:
    return clean_str(value)

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        value = float(str(raw).strip())
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(value)


def _run_in_daemon_thread_with_timeout(
    func: Any,
    *,
    timeout_s: float,
    label: str,
) -> Any:
    return run_in_daemon_thread_with_timeout(
        func,
        timeout_s=timeout_s,
        label=label,
    )

def _normalize_tesseract_language(language: str | None) -> str:
    cleaned = _clean_str(language)
    return cleaned or "chi_sim+eng"


def _split_tesseract_languages(language: str | None) -> list[str]:
    normalized = _normalize_tesseract_language(language)
    out: list[str] = []
    for token in normalized.split("+"):
        cleaned = token.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out
def _normalize_paddle_language(language: str | None) -> str:
    cleaned = _clean_str(language)
    if not cleaned:
        return "ch"
    lowered = cleaned.lower()
    alias_map = {
        "zh": "ch",
        "zh-cn": "ch",
        "cn": "ch",
        "chinese": "ch",
        "en-us": "en",
        "english": "en",
    }
    return alias_map.get(lowered, lowered)

class OcrProvider(ABC):
    """Abstract base class for OCR providers."""

    @abstractmethod
    def ocr_image(self, image_path: str) -> List[Dict]:
        """
        Perform OCR on an image.

        Args:
            image_path: Path to the image file

        Returns:
            List of text elements with format:
            [
              {
                "text": "string",
                "bbox": [x0, y0, x1, y1],  # in image coordinates
                "confidence": 0.95
              }
            ]
        """
        pass
def _normalize_paddle_doc_backend(value: str | None) -> str:
    cleaned = (_clean_str(value) or "").lower()
    if cleaned in _VALID_PADDLE_DOC_BACKENDS:
        return cleaned
    return _DEFAULT_PADDLE_DOC_BACKEND


def _normalize_paddle_doc_server_url(
    value: str | None,
    *,
    provider_id: str | None,
) -> str | None:
    cleaned = _clean_str(value)
    if not cleaned:
        return None

    trimmed = cleaned.rstrip("/")
    try:
        parsed = urlparse(trimmed)
    except Exception:
        return trimmed

    if not parsed.scheme or not parsed.netloc:
        return trimmed

    host = (parsed.netloc or "").lower()
    normalized_provider = (_clean_str(provider_id) or "").lower()
    forced_path: str | None = None
    if normalized_provider == "siliconflow" or "siliconflow" in host:
        forced_path = "/v1"
    elif normalized_provider == "novita" or "novita.ai" in host:
        forced_path = "/openai"
    elif (
        normalized_provider == "ppio"
        or "ppio.com" in host
        or "ppinfra.com" in host
    ):
        forced_path = "/openai"
    elif normalized_provider == "deepseek" or "deepseek.com" in host:
        forced_path = "/v1"

    normalized_path = (parsed.path or "").rstrip("/")
    if forced_path and not normalized_path:
        normalized_path = forced_path

    normalized = urlunparse(
        parsed._replace(path=normalized_path or parsed.path)
    ).rstrip("/")
    return normalized


def _resolve_paddle_doc_model_and_pipeline(
    *,
    model: str | None,
    provider_id: str | None,
    allow_model_downgrade: bool | None = None,
) -> tuple[str, str | None]:
    effective_model = _clean_str(model) or _DEFAULT_PADDLE_OCR_VL_MODEL
    pipeline_version = _clean_str(os.getenv("OCR_PADDLE_VL_PIPELINE_VERSION"))
    normalized_provider = (_clean_str(provider_id) or "").lower()
    can_downgrade = bool(allow_model_downgrade)

    model_lower = effective_model.lower()

    if "paddleocr-vl-1.5" in model_lower and normalized_provider in {"novita", "ppio"}:
        if can_downgrade:
            logger.warning(
                "%s does not expose PaddleOCR-VL-1.5 on doc_parser channel; downgrading to v1 model",
                normalized_provider,
            )
            effective_model = "paddlepaddle/paddleocr-vl"
            model_lower = effective_model.lower()
            if not pipeline_version:
                pipeline_version = "v1"
        else:
            raise RuntimeError(
                f"{normalized_provider} currently supports only PaddleOCR-VL v1 on doc_parser channel; "
                f"requested model={effective_model}. "
                "Switch to paddlepaddle/paddleocr-vl or enable OCR_PADDLE_ALLOW_MODEL_DOWNGRADE=1."
            )

    if "paddleocr-vl" in model_lower and normalized_provider in {"novita", "ppio"}:
        effective_model = "paddlepaddle/paddleocr-vl"
        model_lower = effective_model.lower()

    if (
        model_lower in {_PADDLE_OCR_VL_MODEL_V1.lower(), "paddlepaddle/paddleocr-vl"}
        and not pipeline_version
    ):
        pipeline_version = "v1"
    if (
        model_lower
        in {_PADDLE_OCR_VL_MODEL_V15.lower(), "paddlepaddle/paddleocr-vl-1.5"}
        and not pipeline_version
    ):
        pipeline_version = "v1.5"

    return effective_model, pipeline_version


def _is_probably_model_unsupported_error(error: Exception) -> bool:
    text = str(error or "").lower()
    signals = (
        "invalid model",
        "model not found",
        "unsupported model",
        "does not support",
        "not support",
        "unknown model",
        "model not exist",
        "404",
    )
    return any(sig in text for sig in signals)

_LOC_TOKEN_PATTERN = re.compile(r"<\|LOC_\d+\|>")


def _strip_loc_tokens(text: str) -> str:
    """Remove PaddleOCR-VL location markers from plain text fields."""

    cleaned = _LOC_TOKEN_PATTERN.sub("", str(text or ""))
    return cleaned.strip()
