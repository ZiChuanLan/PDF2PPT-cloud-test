from __future__ import annotations

import pytest

import app.convert.ocr.local_providers as local_providers
from app.convert.ocr.base import OcrProvider


def test_aiocr_non_strict_can_fallback_to_local_paddle(monkeypatch) -> None:
    class _FailingAiClient(OcrProvider):
        def __init__(self, **_: object):
            self.provider_id = "siliconflow"
            self.model = "PaddlePaddle/PaddleOCR-VL-1.5"
            self.base_url = "https://api.siliconflow.cn/v1"
            self.allow_model_downgrade = False

        def ocr_image(self, image_path: str):
            raise RuntimeError("AI OCR returned no items")

    class _MissingTesseract(OcrProvider):
        def __init__(self, **_: object):
            raise RuntimeError("tesseract missing")

        def ocr_image(self, image_path: str):
            return []

    class _LocalPaddleStub(OcrProvider):
        def __init__(self, **_: object):
            pass

        def ocr_image(self, image_path: str):
            return [{"text": "ok", "bbox": [1, 1, 10, 10], "confidence": 0.9}]

    monkeypatch.setattr(local_providers, "AiOcrClient", _FailingAiClient)
    monkeypatch.setattr(local_providers, "TesseractOcrClient", _MissingTesseract)
    monkeypatch.setattr(local_providers, "PaddleOcrClient", _LocalPaddleStub)
    monkeypatch.setenv("OCR_ALLOW_EXPLICIT_AI_FALLBACK", "1")

    manager = local_providers.OcrManager(
        provider="aiocr",
        ai_provider="siliconflow",
        ai_api_key="dummy-key",
        ai_base_url="https://api.siliconflow.cn/v1",
        ai_model="PaddlePaddle/PaddleOCR-VL-1.5",
        strict_no_fallback=False,
    )

    items = manager.ocr_image("dummy.png")
    assert len(items) == 1
    assert items[0]["text"] == "ok"
    assert manager.last_provider_name == "LazyPaddleOcrClient"


def test_aiocr_strict_mode_disables_local_paddle_fallback(monkeypatch) -> None:
    class _FailingAiClient(OcrProvider):
        def __init__(self, **_: object):
            self.provider_id = "siliconflow"
            self.model = "PaddlePaddle/PaddleOCR-VL-1.5"
            self.base_url = "https://api.siliconflow.cn/v1"
            self.allow_model_downgrade = False

        def ocr_image(self, image_path: str):
            raise RuntimeError("AI OCR returned no items")

    monkeypatch.setattr(local_providers, "AiOcrClient", _FailingAiClient)

    manager = local_providers.OcrManager(
        provider="aiocr",
        ai_provider="siliconflow",
        ai_api_key="dummy-key",
        ai_base_url="https://api.siliconflow.cn/v1",
        ai_model="PaddlePaddle/PaddleOCR-VL-1.5",
        strict_no_fallback=True,
    )

    with pytest.raises(RuntimeError, match="All OCR providers failed"):
        manager.ocr_image("dummy.png")


def test_aiocr_non_strict_defaults_to_fail_fast_without_hidden_fallback(
    monkeypatch,
) -> None:
    class _FailingAiClient(OcrProvider):
        def __init__(self, **_: object):
            self.provider_id = "siliconflow"
            self.model = "deepseek-ai/DeepSeek-OCR"
            self.base_url = "https://api.siliconflow.cn/v1"
            self.allow_model_downgrade = False

        def ocr_image(self, image_path: str):
            raise RuntimeError("AI OCR returned no items")

    class _LocalPaddleStub(OcrProvider):
        def __init__(self, **_: object):
            pass

        def ocr_image(self, image_path: str):
            return [{"text": "ok", "bbox": [1, 1, 10, 10], "confidence": 0.9}]

    monkeypatch.setattr(local_providers, "AiOcrClient", _FailingAiClient)
    monkeypatch.setattr(local_providers, "PaddleOcrClient", _LocalPaddleStub)
    monkeypatch.delenv("OCR_ALLOW_EXPLICIT_AI_FALLBACK", raising=False)

    manager = local_providers.OcrManager(
        provider="aiocr",
        ai_provider="siliconflow",
        ai_api_key="dummy-key",
        ai_base_url="https://api.siliconflow.cn/v1",
        ai_model="deepseek-ai/DeepSeek-OCR",
        strict_no_fallback=False,
    )

    with pytest.raises(RuntimeError, match="All OCR providers failed"):
        manager.ocr_image("dummy.png")


def test_aiocr_runtime_failure_disables_ai_for_following_pages(monkeypatch) -> None:
    calls = {"ai": 0, "local": 0}

    class _FailingAiClient(OcrProvider):
        def __init__(self, **_: object):
            self.provider_id = "siliconflow"
            self.model = "deepseek-ai/DeepSeek-OCR"
            self.base_url = "https://api.siliconflow.cn/v1"
            self.allow_model_downgrade = False

        def ocr_image(self, image_path: str):
            calls["ai"] += 1
            raise RuntimeError("AI OCR returned no items")

    class _MissingTesseract(OcrProvider):
        def __init__(self, **_: object):
            raise RuntimeError("tesseract missing")

        def ocr_image(self, image_path: str):
            return []

    class _LocalPaddleStub(OcrProvider):
        def __init__(self, **_: object):
            pass

        def ocr_image(self, image_path: str):
            calls["local"] += 1
            return [{"text": "ok", "bbox": [1, 1, 10, 10], "confidence": 0.9}]

    monkeypatch.setattr(local_providers, "AiOcrClient", _FailingAiClient)
    monkeypatch.setattr(local_providers, "TesseractOcrClient", _MissingTesseract)
    monkeypatch.setattr(local_providers, "PaddleOcrClient", _LocalPaddleStub)
    monkeypatch.setenv("OCR_ALLOW_EXPLICIT_AI_FALLBACK", "1")

    manager = local_providers.OcrManager(
        provider="aiocr",
        ai_provider="siliconflow",
        ai_api_key="dummy-key",
        ai_base_url="https://api.siliconflow.cn/v1",
        ai_model="deepseek-ai/DeepSeek-OCR",
        strict_no_fallback=False,
    )

    out1 = manager.ocr_image("dummy-1.png")
    out2 = manager.ocr_image("dummy-2.png")

    assert len(out1) == 1
    assert len(out2) == 1
    # After the first runtime failure, AI provider should be skipped for
    # remaining pages in the same job to avoid repeated slow failures.
    assert calls["ai"] == 1
    assert calls["local"] == 2
    assert manager.ai_provider_disabled is True
    assert manager.last_provider_name == "LazyPaddleOcrClient"
