from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert.ocr import local_providers
from app.convert.ocr.base import _DEFAULT_PADDLE_OCR_VL_MODEL
from app.convert.ocr.routing import (
    ROUTE_KIND_HYBRID_AUTO,
    ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    ROUTE_KIND_REMOTE_DOC_PARSER,
    ROUTE_KIND_REMOTE_PROMPT_OCR,
)


class DummyAiOcrClient:
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
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider_id = provider or "auto"
        self.layout_model = layout_model
        self.paddle_doc_max_side_px = paddle_doc_max_side_px
        self.layout_block_max_concurrency = layout_block_max_concurrency
        self.request_rpm_limit = request_rpm_limit
        self.request_tpm_limit = request_tpm_limit
        self.request_max_retries = request_max_retries
        self.route_kind = route_kind or "unknown"
        self.allow_model_downgrade = False

    def ocr_image(self, image_path: str):
        return []


def test_create_ocr_manager_passes_prompt_route_for_aiocr(monkeypatch) -> None:
    monkeypatch.setattr(local_providers, "AiOcrClient", DummyAiOcrClient)

    manager = local_providers.create_ocr_manager(
        provider="aiocr",
        route_kind=ROUTE_KIND_REMOTE_PROMPT_OCR,
        ai_provider="openai",
        ai_api_key="test-key",
        ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
        strict_no_fallback=True,
    )

    assert isinstance(manager.ai_provider, DummyAiOcrClient)
    assert manager.route_kind == ROUTE_KIND_REMOTE_PROMPT_OCR
    assert manager.ai_provider.route_kind == ROUTE_KIND_REMOTE_PROMPT_OCR
    assert manager.ai_provider.model == "Qwen/Qwen2.5-VL-72B-Instruct"


def test_create_ocr_manager_forces_doc_parser_route_for_explicit_paddle(
    monkeypatch,
) -> None:
    monkeypatch.setattr(local_providers, "AiOcrClient", DummyAiOcrClient)

    manager = local_providers.create_ocr_manager(
        provider="paddle",
        route_kind=ROUTE_KIND_REMOTE_PROMPT_OCR,
        ai_provider="auto",
        ai_api_key="test-key",
        ai_model=None,
        strict_no_fallback=True,
    )

    assert isinstance(manager.paddle_provider, DummyAiOcrClient)
    assert manager.route_kind == ROUTE_KIND_REMOTE_DOC_PARSER
    assert manager.paddle_provider.route_kind == ROUTE_KIND_REMOTE_DOC_PARSER
    assert manager.paddle_provider.model == _DEFAULT_PADDLE_OCR_VL_MODEL
    assert manager.paddle_provider.provider_id == "siliconflow"


def test_strict_auto_mode_can_compose_remote_doc_parser(monkeypatch) -> None:
    monkeypatch.setattr(local_providers, "AiOcrClient", DummyAiOcrClient)

    manager = local_providers.create_ocr_manager(
        provider="auto",
        ai_provider="auto",
        ai_api_key="test-key",
        ai_model="PaddlePaddle/PaddleOCR-VL-1.5",
        strict_no_fallback=True,
    )

    assert manager.route_kind == ROUTE_KIND_REMOTE_DOC_PARSER
    assert isinstance(manager.ai_provider, DummyAiOcrClient)
    assert manager.ai_provider.route_kind == ROUTE_KIND_REMOTE_DOC_PARSER


def test_create_ocr_manager_passes_layout_model_for_layout_block_route(
    monkeypatch,
) -> None:
    monkeypatch.setattr(local_providers, "AiOcrClient", DummyAiOcrClient)

    manager = local_providers.create_ocr_manager(
        provider="aiocr",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
        ai_provider="openai",
        ai_api_key="test-key",
        ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
        ai_layout_model="pp_doclayout_v3",
        strict_no_fallback=True,
    )

    assert isinstance(manager.ai_provider, DummyAiOcrClient)
    assert manager.route_kind == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR
    assert manager.ai_provider.route_kind == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR
    assert manager.ai_provider.layout_model == "pp_doclayout_v3"


def test_tesseract_client_initializes_with_language_normalization_helpers(
    monkeypatch,
) -> None:
    fake_pytesseract = SimpleNamespace(Output=SimpleNamespace(DICT="DICT"))
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    monkeypatch.setattr(
        local_providers,
        "probe_local_tesseract",
        lambda **_: {
            "binary_available": True,
            "missing_languages": [],
            "available_languages": ["chi_sim", "eng"],
            "version": "5.0.0",
        },
    )

    client = local_providers.TesseractOcrClient(language="eng")

    assert client.language == "eng"
