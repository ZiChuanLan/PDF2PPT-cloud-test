from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert.ocr import local_providers
from app.convert.ocr.routing import (
    ROUTE_KIND_HYBRID_AUTO,
    ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    ROUTE_KIND_REMOTE_DOC_PARSER,
    ROUTE_KIND_REMOTE_PROMPT_OCR,
    normalize_ocr_route_kind,
)


def test_normalize_ocr_route_kind_maps_aliases() -> None:
    assert normalize_ocr_route_kind("doc_parser") == ROUTE_KIND_REMOTE_DOC_PARSER
    assert normalize_ocr_route_kind("prompt") == ROUTE_KIND_REMOTE_PROMPT_OCR
    assert normalize_ocr_route_kind("auto", default=ROUTE_KIND_HYBRID_AUTO) == (
        ROUTE_KIND_HYBRID_AUTO
    )
    assert normalize_ocr_route_kind("unknown-value") == "unknown"


def test_resolve_remote_aiocr_spec_defaults_to_prompt_route() -> None:
    spec = local_providers.resolve_remote_ocr_client_spec(
        provider_id="aiocr",
        ai_provider="openai",
        ai_base_url="https://example.com/v1",
        ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
        route_kind=None,
    )

    assert spec.requested_provider == "aiocr"
    assert spec.route_kind == ROUTE_KIND_REMOTE_PROMPT_OCR
    assert spec.ai_provider == "openai"
    assert spec.ai_model == "Qwen/Qwen2.5-VL-72B-Instruct"


def test_resolve_remote_aiocr_doc_parser_spec_prefers_siliconflow() -> None:
    spec = local_providers.resolve_remote_ocr_client_spec(
        provider_id="aiocr",
        ai_provider="auto",
        ai_base_url="",
        ai_model="PaddlePaddle/PaddleOCR-VL-1.5",
        route_kind=ROUTE_KIND_REMOTE_DOC_PARSER,
    )

    assert spec.requested_provider == "aiocr"
    assert spec.route_kind == ROUTE_KIND_REMOTE_DOC_PARSER
    assert spec.ai_provider == "siliconflow"
    assert spec.ai_model == "PaddlePaddle/PaddleOCR-VL-1.5"


def test_resolve_remote_aiocr_layout_block_spec_preserves_route() -> None:
    spec = local_providers.resolve_remote_ocr_client_spec(
        provider_id="aiocr",
        ai_provider="openai",
        ai_base_url="https://example.com/v1",
        ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    assert spec.requested_provider == "aiocr"
    assert spec.route_kind == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR
    assert spec.ai_provider == "openai"
    assert spec.ai_model == "Qwen/Qwen2.5-VL-72B-Instruct"


def test_resolve_explicit_paddle_spec_defaults_model_and_doc_parser() -> None:
    spec = local_providers.resolve_remote_ocr_client_spec(
        provider_id="paddle",
        ai_provider="auto",
        ai_base_url="",
        ai_model="",
        route_kind=None,
    )

    assert spec.requested_provider == "paddle"
    assert spec.route_kind == ROUTE_KIND_REMOTE_DOC_PARSER
    assert spec.ai_provider == "siliconflow"
    assert spec.ai_model == local_providers._DEFAULT_PADDLE_OCR_VL_MODEL


def test_strict_auto_manager_tracks_effective_remote_route(monkeypatch) -> None:
    created: list[dict[str, str | None]] = []

    class _FakeAiOcrClient:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str | None = None,
            model: str | None = None,
            provider: str | None = None,
            layout_model: str | None = None,
            paddle_doc_max_side_px: int | None = None,
            layout_block_max_concurrency: int | None = None,
            request_rpm_limit: int | None = None,
            request_tpm_limit: int | None = None,
            request_max_retries: int | None = None,
            route_kind: str | None = None,
        ) -> None:
            _ = api_key
            _ = paddle_doc_max_side_px
            _ = layout_model
            _ = layout_block_max_concurrency
            _ = request_rpm_limit
            _ = request_tpm_limit
            _ = request_max_retries
            self.base_url = base_url
            self.model = model
            self.provider_id = provider or "auto"
            self.route_kind = route_kind or "unknown"
            self.allow_model_downgrade = False
            created.append(
                {
                    "base_url": base_url,
                    "model": model,
                    "provider": provider,
                    "route_kind": route_kind,
                }
            )

    monkeypatch.setattr(local_providers, "AiOcrClient", _FakeAiOcrClient)

    manager = local_providers.create_ocr_manager(
        provider="auto",
        route_kind=ROUTE_KIND_HYBRID_AUTO,
        ai_provider="openai",
        ai_api_key="key",
        ai_base_url="https://example.com/v1",
        ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
        strict_no_fallback=True,
    )

    assert manager.route_kind == ROUTE_KIND_REMOTE_PROMPT_OCR
    assert manager.ai_provider is manager.primary_provider
    assert created == [
        {
            "base_url": "https://example.com/v1",
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "provider": "openai",
            "route_kind": ROUTE_KIND_REMOTE_PROMPT_OCR,
        }
    ]
