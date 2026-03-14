from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.worker_helpers import ocr_runtime


def test_route_plan_marks_explicit_paddle_as_remote_doc_parser() -> None:
    plan = ocr_runtime._build_ocr_route_plan(
        requested_ocr_provider="paddle",
        effective_ai_model="PaddlePaddle/PaddleOCR-VL-1.5",
    )

    assert plan.route_kind == "remote_doc_parser"
    assert plan.is_paddle_vl_model is True
    assert plan.allow_text_refiner is False
    assert plan.allow_linebreak_refiner is True


def test_route_plan_marks_generic_aiocr_as_remote_prompt_ocr() -> None:
    plan = ocr_runtime._build_ocr_route_plan(
        requested_ocr_provider="aiocr",
        effective_ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
    )

    assert plan.route_kind == "remote_prompt_ocr"
    assert plan.is_paddle_vl_model is False
    assert plan.allow_text_refiner is False
    assert plan.auto_enable_linebreak is True


def test_route_plan_marks_layout_block_aiocr_as_local_layout_block_ocr() -> None:
    plan = ocr_runtime._build_ocr_route_plan(
        requested_ocr_provider="aiocr",
        effective_ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
        ai_chain_mode="layout_block",
    )

    assert plan.route_kind == "local_layout_block_ocr"
    assert plan.runtime_provider == "aiocr"
    assert plan.allow_text_refiner is False
    assert plan.allow_linebreak_refiner is True


def test_route_plan_blocks_main_ai_reuse_for_explicit_baidu() -> None:
    plan = ocr_runtime._build_ocr_route_plan(
        requested_ocr_provider="baidu",
        effective_ai_model=None,
    )

    assert plan.route_kind == "machine_ocr"
    assert plan.allow_main_ai_reuse is False
    assert plan.force_disable_linebreak is True


def test_debug_payload_includes_route_metadata() -> None:
    route_plan = ocr_runtime._build_ocr_route_plan(
        requested_ocr_provider="aiocr",
        effective_ai_model="PaddlePaddle/PaddleOCR-VL-1.5",
        ai_chain_mode="doc_parser",
    )
    setup = ocr_runtime.OcrRuntimeSetup(
        requested_ocr_provider="aiocr",
        ocr_manager=None,
        text_refiner=None,
        linebreak_refiner=None,
        effective_ocr_provider="aiocr",
        effective_ocr_ai_provider="siliconflow",
        effective_ocr_ai_api_key="key",
        effective_ocr_ai_base_url="https://api.siliconflow.cn/v1",
        effective_ocr_ai_model="PaddlePaddle/PaddleOCR-VL-1.5",
        effective_ocr_ai_chain_mode="doc_parser",
        effective_ocr_ai_layout_model="pp_doclayout_v3",
        effective_paddle_doc_max_side_px=2200,
        effective_ocr_ai_page_concurrency=1,
        effective_ocr_ai_block_concurrency=None,
        effective_ocr_ai_requests_per_minute=None,
        effective_ocr_ai_tokens_per_minute=None,
        effective_ocr_ai_max_retries=0,
        effective_tesseract_language="chi_sim+eng",
        effective_tesseract_min_conf=None,
        strict_ocr_mode=True,
        linebreak_enabled=True,
        auto_linebreak_enabled=True,
        linebreak_mode="ai_refiner",
        linebreak_unavailable_reason=None,
        ocr_ai_config_source="dedicated",
        ocr_ai_api_key_source="ocr",
        ocr_ai_base_url_source="ocr",
        ocr_ai_model_source="ocr",
        ocr_geometry_provider="aiocr",
        ocr_geometry_strategy="direct",
        ocr_geometry_mode_requested="auto",
        ocr_geometry_mode_effective="direct_ai",
        route_plan=route_plan,
        setup_warning=None,
        setup_notes=(),
    )

    payload = ocr_runtime.build_ocr_debug_payload(
        provider_requested="aiocr",
        ocr_render_dpi=200,
        scanned_render_dpi=200,
        ocr_ai_linebreak_assist=None,
        setup=setup,
    )

    assert payload["ocr_route"]["kind"] == "remote_doc_parser"
    assert payload["ocr_route"]["chain_mode"] == "doc_parser"
    assert payload["ocr_route"]["layout_model"] == "pp_doclayout_v3"
    assert payload["ocr_route"]["runtime_provider"] == "aiocr"
    assert payload["ocr_route"]["allow_linebreak_refiner"] is True
    assert payload["ai_ocr"]["page_concurrency"] == 1
    assert payload["ai_ocr"]["max_retries"] == 0


def test_setup_runtime_keeps_text_refiner_off_for_explicit_aiocr(
    monkeypatch,
) -> None:
    created_refiners: list[dict[str, str | None]] = []

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
            request_rpm_limit: int | None = None,
            request_tpm_limit: int | None = None,
            request_max_retries: int | None = None,
        ) -> None:
            created_refiners.append(
                {
                    "api_key": api_key,
                    "provider": provider,
                    "base_url": base_url,
                    "model": model,
                    "request_rpm_limit": request_rpm_limit,
                    "request_tpm_limit": request_tpm_limit,
                    "request_max_retries": request_max_retries,
                }
            )

    monkeypatch.setattr(ocr_runtime, "create_ocr_manager", lambda **_: object())
    monkeypatch.setattr(ocr_runtime, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime.setup_ocr_runtime(
        provider=None,
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="ocr-key",
        ocr_ai_provider="siliconflow",
        ocr_ai_base_url="https://api.siliconflow.cn/v1",
        ocr_ai_model="deepseek-ai/DeepSeek-OCR",
        ocr_ai_linebreak_assist=True,
        ocr_strict_mode=True,
    )

    assert setup.text_refiner is None
    assert setup.linebreak_refiner is not None
    assert len(created_refiners) == 1


def test_setup_runtime_disables_auto_linebreak_in_strict_mode(
    monkeypatch,
) -> None:
    created_refiners: list[dict[str, str | None]] = []

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
            request_rpm_limit: int | None = None,
            request_tpm_limit: int | None = None,
            request_max_retries: int | None = None,
        ) -> None:
            created_refiners.append(
                {
                    "api_key": api_key,
                    "provider": provider,
                    "base_url": base_url,
                    "model": model,
                    "request_rpm_limit": request_rpm_limit,
                    "request_tpm_limit": request_tpm_limit,
                    "request_max_retries": request_max_retries,
                }
            )

    monkeypatch.setattr(ocr_runtime, "create_ocr_manager", lambda **_: object())
    monkeypatch.setattr(ocr_runtime, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime.setup_ocr_runtime(
        provider=None,
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="ocr-key",
        ocr_ai_provider="siliconflow",
        ocr_ai_base_url="https://api.siliconflow.cn/v1",
        ocr_ai_model="deepseek-ai/DeepSeek-OCR",
        ocr_ai_linebreak_assist=None,
        ocr_strict_mode=True,
    )

    assert setup.linebreak_enabled is False
    assert setup.auto_linebreak_enabled is False
    assert setup.linebreak_refiner is None
    assert "ocr_ai_linebreak_auto_disabled_in_strict_mode" in setup.setup_notes
    assert len(created_refiners) == 0


def test_setup_runtime_keeps_auto_linebreak_when_not_strict(
    monkeypatch,
) -> None:
    created_refiners: list[dict[str, str | None]] = []

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
            request_rpm_limit: int | None = None,
            request_tpm_limit: int | None = None,
            request_max_retries: int | None = None,
        ) -> None:
            created_refiners.append(
                {
                    "api_key": api_key,
                    "provider": provider,
                    "base_url": base_url,
                    "model": model,
                    "request_rpm_limit": request_rpm_limit,
                    "request_tpm_limit": request_tpm_limit,
                    "request_max_retries": request_max_retries,
                }
            )

    monkeypatch.setattr(ocr_runtime, "create_ocr_manager", lambda **_: object())
    monkeypatch.setattr(ocr_runtime, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime.setup_ocr_runtime(
        provider=None,
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="ocr-key",
        ocr_ai_provider="siliconflow",
        ocr_ai_base_url="https://api.siliconflow.cn/v1",
        ocr_ai_model="deepseek-ai/DeepSeek-OCR",
        ocr_ai_linebreak_assist=None,
        ocr_strict_mode=False,
    )

    assert setup.linebreak_enabled is True
    assert setup.auto_linebreak_enabled is True
    assert setup.linebreak_refiner is not None
    assert len(created_refiners) == 1


def test_setup_runtime_forwards_experimental_ai_ocr_controls(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_create_ocr_manager(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ocr_runtime, "create_ocr_manager", _fake_create_ocr_manager)
    monkeypatch.setattr(ocr_runtime, "AiOcrTextRefiner", lambda **_: object())

    setup = ocr_runtime.setup_ocr_runtime(
        provider=None,
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="ocr-key",
        ocr_ai_provider="openai",
        ocr_ai_base_url="https://example.com/v1",
        ocr_ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
        ocr_ai_chain_mode="layout_block",
        ocr_ai_layout_model="pp_doclayout_v3",
        ocr_ai_page_concurrency=3,
        ocr_ai_block_concurrency=2,
        ocr_ai_requests_per_minute=90,
        ocr_ai_tokens_per_minute=180000,
        ocr_ai_max_retries=2,
        ocr_ai_linebreak_assist=False,
        ocr_strict_mode=True,
    )

    assert setup.effective_ocr_ai_page_concurrency == 3
    assert setup.effective_ocr_ai_block_concurrency == 2
    assert setup.effective_ocr_ai_requests_per_minute == 90
    assert setup.effective_ocr_ai_tokens_per_minute == 180000
    assert setup.effective_ocr_ai_max_retries == 2
    assert captured["layout_block_max_concurrency"] == 2
    assert captured["request_rpm_limit"] == 90
    assert captured["request_tpm_limit"] == 180000
    assert captured["request_max_retries"] == 2


def test_setup_runtime_auto_aligns_layout_block_concurrency_with_page_concurrency(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_create_ocr_manager(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ocr_runtime, "create_ocr_manager", _fake_create_ocr_manager)
    monkeypatch.setattr(ocr_runtime, "AiOcrTextRefiner", lambda **_: object())

    setup = ocr_runtime.setup_ocr_runtime(
        provider=None,
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="ocr-key",
        ocr_ai_provider="openai",
        ocr_ai_base_url="https://example.com/v1",
        ocr_ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
        ocr_ai_chain_mode="layout_block",
        ocr_ai_layout_model="pp_doclayout_v3",
        ocr_ai_page_concurrency=2,
        ocr_ai_block_concurrency=None,
        ocr_ai_requests_per_minute=None,
        ocr_ai_tokens_per_minute=None,
        ocr_ai_max_retries=0,
        ocr_ai_linebreak_assist=False,
        ocr_strict_mode=True,
    )

    assert setup.effective_ocr_ai_page_concurrency == 2
    assert setup.effective_ocr_ai_block_concurrency == 2
    assert captured["layout_block_max_concurrency"] == 2


def test_local_runtime_does_not_auto_enable_text_refiner_from_main_config(
    monkeypatch,
) -> None:
    created_refiners: list[dict[str, str | None]] = []

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
            request_rpm_limit: int | None = None,
            request_tpm_limit: int | None = None,
            request_max_retries: int | None = None,
        ) -> None:
            created_refiners.append(
                {
                    "api_key": api_key,
                    "provider": provider,
                    "base_url": base_url,
                    "model": model,
                    "request_rpm_limit": request_rpm_limit,
                    "request_tpm_limit": request_tpm_limit,
                    "request_max_retries": request_max_retries,
                }
            )

    monkeypatch.setattr(ocr_runtime, "create_ocr_manager", lambda **_: object())
    monkeypatch.setattr(ocr_runtime, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime.setup_ocr_runtime(
        provider="openai",
        api_key="main-key",
        base_url="https://example.com/v1",
        model="gpt-4.1-mini",
        ocr_provider="tesseract",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key=None,
        ocr_ai_provider="auto",
        ocr_ai_base_url=None,
        ocr_ai_model=None,
        ocr_ai_linebreak_assist=None,
        ocr_strict_mode=True,
    )

    assert setup.ocr_ai_config_source == "main_fallback"
    assert setup.text_refiner is None
    assert setup.linebreak_refiner is None
    assert len(created_refiners) == 0


def test_local_runtime_enables_text_refiner_when_postprocess_explicitly_on(
    monkeypatch,
) -> None:
    created_refiners: list[dict[str, str | None]] = []

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
            request_rpm_limit: int | None = None,
            request_tpm_limit: int | None = None,
            request_max_retries: int | None = None,
        ) -> None:
            created_refiners.append(
                {
                    "api_key": api_key,
                    "provider": provider,
                    "base_url": base_url,
                    "model": model,
                    "request_rpm_limit": request_rpm_limit,
                    "request_tpm_limit": request_tpm_limit,
                    "request_max_retries": request_max_retries,
                }
            )

    monkeypatch.setattr(ocr_runtime, "create_ocr_manager", lambda **_: object())
    monkeypatch.setattr(ocr_runtime, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime.setup_ocr_runtime(
        provider="openai",
        api_key="main-key",
        base_url="https://example.com/v1",
        model="gpt-4.1-mini",
        ocr_provider="tesseract",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key=None,
        ocr_ai_provider="auto",
        ocr_ai_base_url=None,
        ocr_ai_model=None,
        ocr_ai_linebreak_assist=True,
        ocr_strict_mode=True,
    )

    assert setup.ocr_ai_config_source == "main_fallback"
    assert setup.text_refiner is not None
    assert setup.linebreak_refiner is not None
    assert len(created_refiners) == 1
