from __future__ import annotations

import pytest

from app.convert.ocr.base import (
    _normalize_paddle_doc_server_url,
    _resolve_paddle_doc_model_and_pipeline,
)
from app.convert.ocr.vendors import _create_ai_ocr_vendor_adapter


def test_normalize_paddle_doc_server_url_by_provider() -> None:
    assert (
        _normalize_paddle_doc_server_url(
            "https://api.siliconflow.cn",
            provider_id="siliconflow",
        )
        == "https://api.siliconflow.cn/v1"
    )
    assert (
        _normalize_paddle_doc_server_url(
            "https://api.novita.ai",
            provider_id="novita",
        )
        == "https://api.novita.ai/openai"
    )
    assert (
        _normalize_paddle_doc_server_url(
            "https://api.ppio.com",
            provider_id="ppio",
        )
        == "https://api.ppio.com/openai"
    )


def test_vendor_adapter_paddle_doc_parser_channel_matrix() -> None:
    sf = _create_ai_ocr_vendor_adapter(
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
    )
    assert sf.should_use_paddle_doc_parser(
        base_url="https://api.siliconflow.cn/v1",
        model_name="PaddlePaddle/PaddleOCR-VL",
    )

    ppio = _create_ai_ocr_vendor_adapter(
        provider="ppio",
        base_url="https://api.ppio.com/openai",
    )
    assert not ppio.should_use_paddle_doc_parser(
        base_url="https://api.ppio.com/openai",
        model_name="PaddlePaddle/PaddleOCR-VL",
    )

    openai_local = _create_ai_ocr_vendor_adapter(
        provider="openai",
        base_url="http://127.0.0.1:8000/v1",
    )
    assert openai_local.should_use_paddle_doc_parser(
        base_url="http://127.0.0.1:8000/v1",
        model_name="PaddlePaddle/PaddleOCR-VL",
    )


def test_novita_paddle_v15_requires_explicit_downgrade(monkeypatch) -> None:
    monkeypatch.delenv("OCR_PADDLE_VL_PIPELINE_VERSION", raising=False)

    with pytest.raises(RuntimeError, match="only PaddleOCR-VL v1"):
        _resolve_paddle_doc_model_and_pipeline(
            model="PaddlePaddle/PaddleOCR-VL-1.5",
            provider_id="novita",
            allow_model_downgrade=False,
        )


def test_novita_paddle_v15_can_downgrade_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv("OCR_PADDLE_VL_PIPELINE_VERSION", raising=False)

    effective_model, pipeline_version = _resolve_paddle_doc_model_and_pipeline(
        model="PaddlePaddle/PaddleOCR-VL-1.5",
        provider_id="novita",
        allow_model_downgrade=True,
    )
    assert effective_model == "paddlepaddle/paddleocr-vl"
    assert pipeline_version == "v1"


def test_siliconflow_paddle_v15_defaults_to_v15_pipeline(monkeypatch) -> None:
    monkeypatch.delenv("OCR_PADDLE_VL_PIPELINE_VERSION", raising=False)

    effective_model, pipeline_version = _resolve_paddle_doc_model_and_pipeline(
        model="PaddlePaddle/PaddleOCR-VL-1.5",
        provider_id="siliconflow",
        allow_model_downgrade=False,
    )
    assert effective_model == "PaddlePaddle/PaddleOCR-VL-1.5"
    assert pipeline_version == "v1.5"
