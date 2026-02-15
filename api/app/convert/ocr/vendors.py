"""AI OCR vendor profiles and adapters."""

from abc import ABC
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .base import _AI_OCR_PROVIDER_ALIASES, _VALID_AI_OCR_PROVIDERS, _clean_str
from .deepseek_parser import _is_deepseek_ocr_model


@dataclass(frozen=True)
class AiOcrVendorProfile:
    provider_id: str
    default_base_url: str | None
    default_model: str
    max_tokens_ocr: int
    max_tokens_refiner: int
    supports_remote_paddle_doc_parser: bool = False


_AI_OCR_VENDOR_PROFILES: dict[str, AiOcrVendorProfile] = {
    "openai": AiOcrVendorProfile(
        provider_id="openai",
        default_base_url=None,
        default_model="gpt-4o-mini",
        max_tokens_ocr=8192,
        max_tokens_refiner=4096,
    ),
    "siliconflow": AiOcrVendorProfile(
        provider_id="siliconflow",
        default_base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen2.5-VL-72B-Instruct",
        max_tokens_ocr=4096,
        max_tokens_refiner=2048,
        supports_remote_paddle_doc_parser=True,
    ),
    "ppio": AiOcrVendorProfile(
        provider_id="ppio",
        default_base_url="https://api.ppio.com/openai",
        default_model="qwen/qwen2.5-vl-72b-instruct",
        max_tokens_ocr=4096,
        max_tokens_refiner=3072,
    ),
    "novita": AiOcrVendorProfile(
        provider_id="novita",
        default_base_url="https://api.novita.ai/openai",
        default_model="qwen/qwen2.5-vl-72b-instruct",
        max_tokens_ocr=4096,
        max_tokens_refiner=3072,
        supports_remote_paddle_doc_parser=True,
    ),
    "deepseek": AiOcrVendorProfile(
        provider_id="deepseek",
        default_base_url="https://api.deepseek.com/v1",
        default_model="deepseek-ai/DeepSeek-OCR",
        max_tokens_ocr=4096,
        max_tokens_refiner=2048,
    ),
}


def _normalize_ai_ocr_model_name(
    model_name: str | None,
    *,
    provider_id: str | None,
) -> str | None:
    cleaned = _clean_str(model_name)
    if not cleaned:
        return cleaned

    lowered = cleaned.lower()

    normalized_provider = (_clean_str(provider_id) or "").lower()

    # OCR gateways often alias model ids with a Pro/ prefix.
    if lowered.startswith("pro/deepseek-ai/deepseek-ocr"):
        return "deepseek-ai/DeepSeek-OCR"

    if lowered == "deepseek-ai/deepseek-ocr":
        return "deepseek-ai/DeepSeek-OCR"

    if "paddleocr-vl-1.5" in lowered:
        if normalized_provider in {"novita", "ppio"}:
            return "paddlepaddle/paddleocr-vl-1.5"
        return "PaddlePaddle/PaddleOCR-VL-1.5"

    if "paddleocr-vl" in lowered:
        if normalized_provider in {"novita", "ppio"}:
            return "paddlepaddle/paddleocr-vl"
        return "PaddlePaddle/PaddleOCR-VL"

    return cleaned


def _should_send_image_first_for_ai_ocr(
    *,
    provider_id: str | None,
    model_name: str | None,
) -> bool:
    # DeepSeek-OCR on OpenAI-compatible gateways (including SiliconFlow)
    # is much more stable when image appears before text in user content.
    return _is_deepseek_ocr_model(model_name)


def _normalize_ai_ocr_provider(value: str | None) -> str:
    cleaned = (_clean_str(value) or "").lower()
    provider_id = _AI_OCR_PROVIDER_ALIASES.get(cleaned, cleaned)
    if provider_id not in _VALID_AI_OCR_PROVIDERS:
        return "auto"
    return provider_id


def _infer_ai_ocr_provider_from_base_url(base_url: str | None) -> str:
    cleaned = _clean_str(base_url)
    if not cleaned:
        return "openai"
    try:
        host = (urlparse(cleaned).netloc or "").lower()
    except Exception:
        host = ""
    if not host:
        return "openai"
    if "siliconflow" in host:
        return "siliconflow"
    if "ppio.com" in host or "ppinfra.com" in host:
        return "ppio"
    if "novita.ai" in host:
        return "novita"
    if "deepseek.com" in host:
        return "deepseek"
    if "openai.com" in host:
        return "openai"
    return "openai"


def _is_paddleocr_vl_model_name(model_name: str | None) -> bool:
    cleaned = _clean_str(model_name)
    if not cleaned:
        return False
    return "paddleocr-vl" in cleaned.lower()


def _is_local_or_private_base_url(base_url: str | None) -> bool:
    cleaned = _clean_str(base_url)
    if not cleaned:
        return False
    try:
        host = (urlparse(cleaned).hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    if host.endswith(".local"):
        return True
    if host.startswith("10.") or host.startswith("192.168."):
        return True
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) >= 2:
            try:
                second = int(parts[1])
            except Exception:
                second = -1
            if 16 <= second <= 31:
                return True
    return False


def _resolve_ai_ocr_profile(
    *, provider: str | None, base_url: str | None
) -> tuple[str, AiOcrVendorProfile]:
    provider_id = _normalize_ai_ocr_provider(provider)
    if provider_id == "auto":
        provider_id = _infer_ai_ocr_provider_from_base_url(base_url)
    profile = _AI_OCR_VENDOR_PROFILES.get(
        provider_id,
        _AI_OCR_VENDOR_PROFILES["openai"],
    )
    return provider_id, profile


class AiOcrVendorAdapter(ABC):
    """Vendor adapter for OpenAI-compatible OCR gateways."""

    def __init__(self, *, profile: AiOcrVendorProfile):
        self.profile = profile

    @property
    def provider_id(self) -> str:
        return self.profile.provider_id

    def resolve_base_url(self, base_url: str | None) -> str | None:
        return _clean_str(base_url) or self.profile.default_base_url

    def resolve_model(self, model: str | None) -> str:
        return _clean_str(model) or self.profile.default_model

    def clamp_max_tokens(self, requested: int, *, kind: str) -> int:
        if kind == "refiner":
            limit = int(self.profile.max_tokens_refiner)
        else:
            limit = int(self.profile.max_tokens_ocr)
        req = max(256, int(requested))
        return max(256, min(req, max(256, limit)))

    def build_user_content(
        self,
        *,
        prompt: str,
        image_data_uri: str,
        image_first: bool = False,
    ) -> list[dict[str, Any]]:
        text_part = {"type": "text", "text": prompt}
        image_part = {"type": "image_url", "image_url": {"url": image_data_uri}}
        if image_first:
            return [image_part, text_part]
        return [text_part, image_part]

    def supports_remote_paddle_doc_parser(self, *, base_url: str | None) -> bool:
        _ = base_url
        return bool(self.profile.supports_remote_paddle_doc_parser)

    def should_use_paddle_doc_parser(
        self,
        *,
        base_url: str | None,
        model_name: str | None,
    ) -> bool:
        if not _is_paddleocr_vl_model_name(model_name):
            return False
        if not _clean_str(base_url):
            return True
        return self.supports_remote_paddle_doc_parser(base_url=base_url)


class OpenAiAiOcrAdapter(AiOcrVendorAdapter):
    def supports_remote_paddle_doc_parser(self, *, base_url: str | None) -> bool:
        # Generic OpenAI-compatible provider can still be a self-hosted vLLM/
        # sglang endpoint that supports PaddleOCRVL doc_parser protocol.
        return _is_local_or_private_base_url(base_url)


class SiliconFlowAiOcrAdapter(AiOcrVendorAdapter):
    pass


class PpioAiOcrAdapter(AiOcrVendorAdapter):
    pass


class NovitaAiOcrAdapter(AiOcrVendorAdapter):
    pass


class DeepSeekAiOcrAdapter(AiOcrVendorAdapter):
    pass


_AI_OCR_VENDOR_ADAPTERS: dict[str, type[AiOcrVendorAdapter]] = {
    "openai": OpenAiAiOcrAdapter,
    "siliconflow": SiliconFlowAiOcrAdapter,
    "ppio": PpioAiOcrAdapter,
    "novita": NovitaAiOcrAdapter,
    "deepseek": DeepSeekAiOcrAdapter,
}


def _create_ai_ocr_vendor_adapter(
    *, provider: str | None, base_url: str | None
) -> AiOcrVendorAdapter:
    _, profile = _resolve_ai_ocr_profile(provider=provider, base_url=base_url)
    adapter_cls = _AI_OCR_VENDOR_ADAPTERS.get(profile.provider_id, OpenAiAiOcrAdapter)
    return adapter_cls(profile=profile)
