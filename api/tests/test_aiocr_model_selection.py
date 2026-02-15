from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.convert.ocr import AiOcrClient


def _write_test_image(path: Path) -> None:
    image = Image.new("RGB", (160, 96), (255, 255, 255))
    image.save(path)


def test_paddle_model_keeps_requested_model_before_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class _DummyOpenAI:
        def __init__(self, **_: object):
            self.chat = SimpleNamespace(completions=self)

        def with_options(self, **_: object):
            return self

        def create(self, **kwargs: object):
            model = str(kwargs.get("model") or "")
            calls.append(model)
            payload = '[{"text":"hello","bbox":[10,12,88,34],"confidence":0.9}]'
            choice = SimpleNamespace(
                message=SimpleNamespace(content=payload),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice])

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.setenv("OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK", "1")
    monkeypatch.setenv("OCR_PADDLE_VL_USE_DOCPARSER", "0")
    monkeypatch.setenv("OCR_PADDLE_PROMPT_FALLBACK_MODEL", "deepseek-ai/DeepSeek-OCR")

    image_path = tmp_path / "img.png"
    _write_test_image(image_path)

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    out = client.ocr_image(str(image_path))

    assert len(out) == 1
    assert calls == ["PaddlePaddle/PaddleOCR-VL-1.5"]


def test_paddle_model_can_try_configured_fallback_after_empty_result(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class _DummyOpenAI:
        def __init__(self, **_: object):
            self.chat = SimpleNamespace(completions=self)

        def with_options(self, **_: object):
            return self

        def create(self, **kwargs: object):
            model = str(kwargs.get("model") or "")
            calls.append(model)
            if "DeepSeek-OCR" in model:
                payload = "<|ref|>line<|/ref|><|det|>[[1,2,30,12]]<|/det|>"
            else:
                payload = ""
            choice = SimpleNamespace(
                message=SimpleNamespace(content=payload),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice])

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.setenv("OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK", "1")
    monkeypatch.setenv("OCR_PADDLE_VL_USE_DOCPARSER", "0")
    monkeypatch.setenv("OCR_PADDLE_PROMPT_FALLBACK_MODEL", "deepseek-ai/DeepSeek-OCR")

    image_path = tmp_path / "img.png"
    _write_test_image(image_path)

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    out = client.ocr_image(str(image_path))

    assert len(out) == 1
    assert out[0]["text"] == "line"
    assert calls[0] == "PaddlePaddle/PaddleOCR-VL-1.5"
    assert calls[-1] == "deepseek-ai/DeepSeek-OCR"


def test_deepseek_empty_response_stops_after_small_number_of_attempts(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class _DummyOpenAI:
        def __init__(self, **_: object):
            self.chat = SimpleNamespace(completions=self)

        def with_options(self, **_: object):
            return self

        def create(self, **kwargs: object):
            model = str(kwargs.get("model") or "")
            calls.append(model)
            choice = SimpleNamespace(
                message=SimpleNamespace(content=""),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice])

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)

    image_path = tmp_path / "img.png"
    _write_test_image(image_path)

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-OCR",
    )

    with pytest.raises(RuntimeError, match="AI OCR returned no items"):
        client.ocr_image(str(image_path))

    # Avoid burning page timeout on repeated empty responses.
    assert len(calls) <= 2


def test_deepseek_prompt_uses_official_grounding_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    seen_text_prompts: list[str] = []

    class _DummyOpenAI:
        def __init__(self, **_: object):
            self.chat = SimpleNamespace(completions=self)

        def with_options(self, **_: object):
            return self

        def create(self, **kwargs: object):
            messages = kwargs.get("messages") or []
            assert isinstance(messages, list) and len(messages) >= 1
            user_message = messages[0]
            content = user_message.get("content") if isinstance(user_message, dict) else []
            assert isinstance(content, list)
            text_parts = [
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and str(part.get("type") or "") == "text"
            ]
            assert text_parts, "expected text part in deepseek user content"
            seen_text_prompts.extend(text_parts)
            choice = SimpleNamespace(
                message=SimpleNamespace(
                    content="<|ref|>line<|/ref|><|det|>[[1,2,30,12]]<|/det|>"
                ),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice])

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)

    image_path = tmp_path / "img.png"
    _write_test_image(image_path)

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-OCR",
    )
    out = client.ocr_image(str(image_path))

    assert len(out) == 1
    assert seen_text_prompts
    assert seen_text_prompts[0].startswith("<image>\n<|grounding|>OCR this image.")


def test_qwen_timeout_override_env_applies(monkeypatch) -> None:
    class _DummyOpenAI:
        def __init__(self, **_: object):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.setenv("OCR_AI_REQUEST_TIMEOUT_S", "25")
    monkeypatch.setenv("OCR_AI_REQUEST_TIMEOUT_S_QWEN", "90")

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
    )
    timeout_s = client._resolve_model_request_timeout_s(
        model_name="Qwen/Qwen2.5-VL-72B-Instruct"
    )
    assert timeout_s >= 90.0


def test_paddle_v15_doc_parser_timeout_default_is_higher(monkeypatch) -> None:
    class _DummyOpenAI:
        def __init__(self, **_: object):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.setenv("OCR_PADDLE_VL_USE_DOCPARSER", "1")
    monkeypatch.setenv("OCR_PADDLE_VL_DOCPARSER_PREDICT_TIMEOUT_S", "120")
    monkeypatch.delenv(
        "OCR_PADDLE_VL_DOCPARSER_PREDICT_TIMEOUT_S_V15",
        raising=False,
    )

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    timeout_s = client._resolve_paddle_doc_predict_timeout_s()
    assert timeout_s >= 180.0


def test_paddle_model_on_unsupported_provider_fails_fast_when_prompt_fallback_disabled(
    monkeypatch,
) -> None:
    class _DummyOpenAI:
        def __init__(self, **_: object):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.delenv("OCR_PADDLE_VL_USE_DOCPARSER", raising=False)
    monkeypatch.setenv("OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK", "0")

    with pytest.raises(ValueError, match="dedicated structured OCR channel"):
        AiOcrClient(
            api_key="dummy",
            provider="ppio",
            base_url="https://api.ppio.com/openai",
            model="PaddlePaddle/PaddleOCR-VL",
        )


def test_paddle_doc_parser_failure_does_not_secretly_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {"chat": 0}

    class _DummyOpenAI:
        def __init__(self, **_: object):
            self.chat = SimpleNamespace(completions=self)

        def with_options(self, **_: object):
            return self

        def create(self, **_: object):
            calls["chat"] += 1
            choice = SimpleNamespace(
                message=SimpleNamespace(content="[]"),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice])

    def _fail_doc_parser(self: AiOcrClient, image_path: str):
        raise RuntimeError("doc parser timeout")

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace())
    monkeypatch.setattr(AiOcrClient, "_ocr_image_with_paddle_doc_parser", _fail_doc_parser)
    monkeypatch.delenv("OCR_PADDLE_VL_USE_DOCPARSER", raising=False)
    monkeypatch.setenv("OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK", "0")

    image_path = tmp_path / "img.png"
    _write_test_image(image_path)

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL",
    )
    with pytest.raises(RuntimeError, match="dedicated channel failed"):
        client.ocr_image(str(image_path))

    assert calls["chat"] == 0


def test_paddle_doc_parser_v15_uses_conservative_default_max_concurrency(
    monkeypatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class _DummyOpenAI:
        def __init__(self, **_: object):
            pass

    class _DummyPaddleOCRVL:
        def __init__(self, **kwargs: object):
            captured_kwargs.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCRVL=_DummyPaddleOCRVL),
    )
    monkeypatch.setenv("OCR_PADDLE_VL_USE_DOCPARSER", "1")
    monkeypatch.delenv("OCR_PADDLE_VL_DOCPARSER_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("OCR_PADDLE_VL_DOCPARSER_USE_QUEUES", raising=False)

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    client._get_paddle_doc_parser()

    assert captured_kwargs.get("vl_rec_max_concurrency") == 1


def test_paddle_doc_parser_tuning_env_is_forwarded_to_paddleocrvl(
    monkeypatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class _DummyOpenAI:
        def __init__(self, **_: object):
            pass

    class _DummyPaddleOCRVL:
        def __init__(self, **kwargs: object):
            captured_kwargs.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCRVL=_DummyPaddleOCRVL),
    )
    monkeypatch.setenv("OCR_PADDLE_VL_USE_DOCPARSER", "1")
    monkeypatch.setenv("OCR_PADDLE_VL_DOCPARSER_MAX_CONCURRENCY", "16")
    monkeypatch.setenv("OCR_PADDLE_VL_DOCPARSER_USE_QUEUES", "0")

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    client._get_paddle_doc_parser()

    assert captured_kwargs.get("vl_rec_max_concurrency") == 16
    assert captured_kwargs.get("use_queues") is False


def test_paddle_v15_timeout_retries_once_with_same_model(
    monkeypatch, tmp_path: Path
) -> None:
    labels: list[str] = []

    class _DummyOpenAI:
        def __init__(self, **_: object):
            pass

    class _DummyParser:
        def predict(self, *_: object, **__: object):
            return [
                {
                    "parsing_res_list": [
                        {
                            "block_label": "text",
                            "block_content": "retry-ok",
                            "block_bbox": [1, 2, 80, 18],
                        }
                    ]
                }
            ]

    def _fake_run_with_retry(
        func,
        *,
        timeout_s: float,
        label: str,
    ):
        labels.append(label)
        if label == "paddleocr-vl:predict":
            raise TimeoutError("first timeout")
        return func()

    import app.convert.ocr.ai_client as ai_client_module
    import openai

    monkeypatch.setattr(openai, "OpenAI", _DummyOpenAI)
    monkeypatch.setattr(
        ai_client_module,
        "_run_in_daemon_thread_with_timeout",
        _fake_run_with_retry,
    )
    monkeypatch.setattr(
        AiOcrClient,
        "_get_paddle_doc_parser",
        lambda self: _DummyParser(),
    )
    monkeypatch.setenv("OCR_PADDLE_VL_DOCPARSER_RETRY_ON_TIMEOUT", "1")
    monkeypatch.setenv("OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK", "0")
    monkeypatch.setenv("OCR_PADDLE_ALLOW_MODEL_DOWNGRADE", "0")
    monkeypatch.setenv("OCR_PADDLE_VL_USE_DOCPARSER", "1")

    image_path = tmp_path / "img.png"
    _write_test_image(image_path)

    client = AiOcrClient(
        api_key="dummy",
        provider="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    out = client._ocr_image_with_paddle_doc_parser(str(image_path))

    assert out and out[0]["text"] == "retry-ok"
    assert labels == ["paddleocr-vl:predict", "paddleocr-vl:predict:retry"]
