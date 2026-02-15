from __future__ import annotations

import asyncio

import app.routers.jobs as jobs_router
from app.models.error import AppException
from app.models.job import AiOcrCheckRequest, AiOcrCheckResponse, AiOcrCheckResult


def test_ai_ocr_capability_check_success(monkeypatch) -> None:
    class _DummyAiClient:
        def __init__(self, **_: object):
            pass

        def ocr_image(self, image_path: str):
            assert image_path.endswith(".png")
            return [
                {"text": "hello", "bbox": [1, 2, 120, 32], "confidence": 0.9},
                {"text": "world", "bbox": [2, 40, 122, 70], "confidence": 0.8},
            ]

    monkeypatch.setattr(jobs_router, "AiOcrClient", _DummyAiClient)

    result = jobs_router._run_ai_ocr_capability_check(
        provider="siliconflow",
        api_key="dummy",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-OCR",
    )

    assert result.ok is True
    assert result.check.ready is True
    assert result.check.valid_bbox_items == 2
    assert result.check.items_count == 2
    assert not result.check.error
    assert len(result.check.sample_items) >= 1


def test_ai_ocr_capability_check_no_bbox_items(monkeypatch) -> None:
    class _DummyAiClient:
        def __init__(self, **_: object):
            pass

        def ocr_image(self, image_path: str):
            assert image_path.endswith(".png")
            return [
                {"text": "plain text only"},
                {"text": "invalid", "bbox": [10, 10, 10, 10]},
            ]

    monkeypatch.setattr(jobs_router, "AiOcrClient", _DummyAiClient)

    result = jobs_router._run_ai_ocr_capability_check(
        provider="openai",
        api_key="dummy",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
    )

    assert result.ok is False
    assert result.check.ready is False
    assert result.check.valid_bbox_items == 0
    assert result.check.message == "模型未返回有效 bbox OCR 结果"


def test_ai_ocr_capability_check_runtime_error(monkeypatch) -> None:
    class _DummyAiClient:
        def __init__(self, **_: object):
            pass

        def ocr_image(self, image_path: str):
            assert image_path.endswith(".png")
            raise RuntimeError("AI OCR returned no items")

    monkeypatch.setattr(jobs_router, "AiOcrClient", _DummyAiClient)

    result = jobs_router._run_ai_ocr_capability_check(
        provider="siliconflow",
        api_key="dummy",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
    )

    assert result.ok is False
    assert result.check.ready is False
    assert result.check.error and "no items" in result.check.error.lower()
    assert result.check.message == "模型调用失败"


def test_check_ai_ocr_requires_model() -> None:
    payload = AiOcrCheckRequest(
        provider="siliconflow",
        api_key="dummy",
        base_url="https://api.siliconflow.cn/v1",
        model="",
    )
    try:
        asyncio.run(jobs_router.check_ai_ocr(payload))
        assert False, "expected AppException"
    except AppException as exc:
        assert exc.code == "validation_error"
        assert "model is required" in exc.message


def test_check_ai_ocr_endpoint_wrapper(monkeypatch) -> None:
    expected = AiOcrCheckResponse(
        ok=True,
        check=AiOcrCheckResult(
            provider="siliconflow",
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            base_url="https://api.siliconflow.cn/v1",
            elapsed_ms=1234,
            items_count=3,
            valid_bbox_items=3,
            ready=True,
            message="模型可返回有效 bbox OCR 结果",
            error=None,
            sample_items=[],
        ),
    )

    def _fake_run(**_: object) -> AiOcrCheckResponse:
        return expected

    monkeypatch.setattr(jobs_router, "_run_ai_ocr_capability_check", _fake_run)
    payload = AiOcrCheckRequest(
        provider="siliconflow",
        api_key="dummy",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
    )
    out = asyncio.run(jobs_router.check_ai_ocr(payload))
    assert out.ok is True
    assert out.check.valid_bbox_items == 3
