from __future__ import annotations

import sys
from pathlib import Path

import pytest


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.job_options import validate_and_normalize_job_options
from app.models.error import AppException


def _base_kwargs() -> dict:
    return {
        "provider": "openai",
        "api_key": None,
        "baidu_doc_parse_type": "paddle_vl",
        "ocr_ai_provider": "auto",
        "ocr_ai_api_key": None,
        "ocr_ai_model": None,
        "ocr_ai_chain_mode": "direct",
        "ocr_ai_layout_model": "pp_doclayout_v3",
        "ocr_baidu_app_id": None,
        "ocr_baidu_api_key": "baidu-key",
        "ocr_baidu_secret_key": "baidu-secret",
        "ocr_geometry_mode": "auto",
        "text_erase_mode": "fill",
        "scanned_page_mode": "segmented",
        "page_start": None,
        "page_end": None,
        "mineru_api_token": "mineru-token",
    }


def test_baidu_doc_normalizes_legacy_baidu_ocr_provider_alias() -> None:
    normalized = validate_and_normalize_job_options(
        parse_provider="baidu_doc",
        ocr_provider="baidu",
        **_base_kwargs(),
    )

    assert normalized.parse_provider == "baidu_doc"
    assert normalized.ocr_provider == "auto"


def test_baidu_doc_rejects_unrelated_explicit_ocr_provider() -> None:
    with pytest.raises(AppException) as exc_info:
        validate_and_normalize_job_options(
            parse_provider="baidu_doc",
            ocr_provider="tesseract",
            **_base_kwargs(),
        )

    assert exc_info.value.message == "Baidu document parser does not use OCR provider selection"


def test_mineru_rejects_explicit_ocr_provider_selection() -> None:
    with pytest.raises(AppException) as exc_info:
        validate_and_normalize_job_options(
            parse_provider="mineru",
            ocr_provider="paddle_local",
            **_base_kwargs(),
        )

    assert exc_info.value.message == "MinerU parse does not use OCR provider selection"


def test_local_aiocr_layout_block_chain_is_normalized() -> None:
    kwargs = _base_kwargs()
    kwargs.update(
        {
            "ocr_ai_api_key": "ocr-key",
            "ocr_ai_model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "ocr_ai_chain_mode": "layout_block",
            "ocr_ai_layout_model": "pp_doclayout_v3",
        }
    )
    normalized = validate_and_normalize_job_options(
        parse_provider="local",
        ocr_provider="aiocr",
        **kwargs,
    )

    assert normalized.ocr_provider == "aiocr"
    assert normalized.ocr_ai_chain_mode == "layout_block"
    assert normalized.ocr_ai_layout_model == "pp_doclayout_v3"
