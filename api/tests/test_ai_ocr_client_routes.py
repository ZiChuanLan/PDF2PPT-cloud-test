from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert.ocr import ai_client as ai_client_module
from app.convert.ocr.routing import (
    ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    ROUTE_KIND_REMOTE_DOC_PARSER,
    ROUTE_KIND_REMOTE_PROMPT_OCR,
)


class _DummyOpenAIClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _DummyVendorAdapter:
    provider_id = "siliconflow"

    def resolve_base_url(self, base_url: str | None) -> str | None:
        return base_url or "https://api.siliconflow.cn/v1"

    def resolve_model(self, model: str | None) -> str | None:
        return model

    def should_use_paddle_doc_parser(
        self,
        *,
        base_url: str | None,
        model_name: str | None,
    ) -> bool:
        return True


def _patch_openai_and_adapter(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_DummyOpenAIClient))
    monkeypatch.setattr(
        ai_client_module,
        "_create_ai_ocr_vendor_adapter",
        lambda provider, base_url: _DummyVendorAdapter(),
    )


def test_ai_ocr_client_prompt_route_disables_doc_parser(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)
    monkeypatch.setenv("OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK", "1")

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
        provider="siliconflow",
        route_kind=ROUTE_KIND_REMOTE_PROMPT_OCR,
    )

    assert client.route_kind == ROUTE_KIND_REMOTE_PROMPT_OCR
    assert client._should_use_paddle_doc_parser() is False


def test_ai_ocr_client_prompt_route_error_explains_chain_mode(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        ai_client_module.AiOcrClient(
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            model="PaddlePaddle/PaddleOCR-VL-1.5",
            provider="siliconflow",
            route_kind=ROUTE_KIND_REMOTE_PROMPT_OCR,
        )

    message = str(exc_info.value)
    assert "current chain mode is direct/prompt, not doc_parser" in message
    assert "Choose `内置文档解析（PaddleOCR-VL）` / `doc_parser`" in message


def test_ai_ocr_client_doc_parser_route_can_be_forced(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace())

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
        provider="siliconflow",
        route_kind=ROUTE_KIND_REMOTE_DOC_PARSER,
    )

    assert client.route_kind == ROUTE_KIND_REMOTE_DOC_PARSER
    assert client._should_use_paddle_doc_parser() is True


def test_ai_ocr_client_route_kind_refreshes_after_doc_parser_disable(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace())

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
        provider="siliconflow",
        route_kind=ROUTE_KIND_REMOTE_DOC_PARSER,
    )
    client._paddle_doc_parser_disabled = True

    assert client._refresh_route_kind() == ROUTE_KIND_REMOTE_PROMPT_OCR


def test_ai_ocr_client_layout_block_route_is_preserved(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        provider="openai",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    assert client.route_kind == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR
    assert client.layout_model == "pp_doclayout_v3"
    assert client._refresh_route_kind() == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR


def test_ai_ocr_client_layout_block_route_allows_paddleocr_vl(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
        provider="siliconflow",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    assert client.route_kind == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR
    assert client._refresh_route_kind() == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR


def test_paddle_doc_singleflight_lock_released_after_timeout(
    monkeypatch, tmp_path
) -> None:
    _patch_openai_and_adapter(monkeypatch)
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace())
    monkeypatch.setenv("OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT_WAIT_S", "0.2")

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
        provider="siliconflow",
        route_kind=ROUTE_KIND_REMOTE_DOC_PARSER,
    )

    started = threading.Event()
    release = threading.Event()

    def _long_running_predict() -> str:
        started.set()
        release.wait(timeout=2.0)
        return "long"

    with pytest.raises(TimeoutError):
        client._run_paddle_doc_predict_with_timeout(
            _long_running_predict,
            timeout_s=0.1,
            label="paddleocr-vl:test-lock",
        )

    assert started.wait(timeout=0.5)
    try:
        assert (
            client._run_paddle_doc_predict_with_timeout(
                lambda: "ok",
                timeout_s=0.5,
                label="paddleocr-vl:test-lock",
            )
            == "ok"
        )
    finally:
        release.set()


def test_local_layout_block_default_concurrency_reduced_for_siliconflow_qwen3(
    monkeypatch,
) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3-VL-235B-A22B-Instruct",
        provider="siliconflow",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    assert (
        client._resolve_local_layout_block_max_workers(
            effective_model="Qwen/Qwen3-VL-235B-A22B-Instruct"
        )
        == 2
    )


def test_deepseek_request_timeout_uses_longer_default_for_siliconflow(
    monkeypatch,
) -> None:
    _patch_openai_and_adapter(monkeypatch)
    monkeypatch.delenv("OCR_AI_REQUEST_TIMEOUT_S", raising=False)
    monkeypatch.delenv("OCR_AI_REQUEST_TIMEOUT_S_DEEPSEEK_OCR", raising=False)
    monkeypatch.delenv(
        "OCR_AI_REQUEST_TIMEOUT_S_DEEPSEEK_OCR_SILICONFLOW",
        raising=False,
    )

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-OCR",
        provider="siliconflow",
    )

    assert (
        client._resolve_model_request_timeout_s(
            model_name="deepseek-ai/DeepSeek-OCR"
        )
        == 90.0
    )


def test_deepseek_request_timeout_prefers_siliconflow_override_over_generic(
    monkeypatch,
) -> None:
    _patch_openai_and_adapter(monkeypatch)
    monkeypatch.setenv("OCR_AI_REQUEST_TIMEOUT_S", "25")
    monkeypatch.setenv("OCR_AI_REQUEST_TIMEOUT_S_DEEPSEEK_OCR", "60")
    monkeypatch.setenv("OCR_AI_REQUEST_TIMEOUT_S_DEEPSEEK_OCR_SILICONFLOW", "75")

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-OCR",
        provider="siliconflow",
    )

    assert (
        client._resolve_model_request_timeout_s(
            model_name="deepseek-ai/DeepSeek-OCR"
        )
        == 75.0
    )
