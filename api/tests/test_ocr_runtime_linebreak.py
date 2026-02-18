from __future__ import annotations

from app.worker_helpers import ocr_runtime as ocr_runtime_module


def test_auto_linebreak_enabled_for_explicit_aiocr_non_paddle(monkeypatch) -> None:
    class _DummyManager:
        pass

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
        ) -> None:
            self.api_key = api_key
            self.provider = provider
            self.base_url = base_url
            self.model = model

    monkeypatch.setattr(
        ocr_runtime_module,
        "create_ocr_manager",
        lambda **_kwargs: _DummyManager(),
    )
    monkeypatch.setattr(ocr_runtime_module, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime_module.setup_ocr_runtime(
        provider="openai",
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="test-key",
        ocr_ai_provider="openai",
        ocr_ai_base_url="https://example.test/v1",
        ocr_ai_model="gpt-4o-mini",
        ocr_ai_linebreak_assist=None,
        ocr_strict_mode=True,
    )

    assert setup.linebreak_enabled is True
    assert setup.auto_linebreak_enabled is True
    assert setup.linebreak_refiner is not None


def test_explicit_off_disables_auto_linebreak_for_aiocr(monkeypatch) -> None:
    class _DummyManager:
        pass

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
        ) -> None:
            self.api_key = api_key
            self.provider = provider
            self.base_url = base_url
            self.model = model

    monkeypatch.setattr(
        ocr_runtime_module,
        "create_ocr_manager",
        lambda **_kwargs: _DummyManager(),
    )
    monkeypatch.setattr(ocr_runtime_module, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime_module.setup_ocr_runtime(
        provider="openai",
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="test-key",
        ocr_ai_provider="openai",
        ocr_ai_base_url="https://example.test/v1",
        ocr_ai_model="gpt-4o-mini",
        ocr_ai_linebreak_assist=False,
        ocr_strict_mode=True,
    )

    assert setup.linebreak_enabled is False
    assert setup.auto_linebreak_enabled is False
    assert setup.linebreak_refiner is None


def test_aiocr_generic_vl_uses_tesseract_geometry_runtime(monkeypatch) -> None:
    captured_provider: dict[str, str] = {}

    class _DummyManager:
        pass

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
        ) -> None:
            self.api_key = api_key
            self.provider = provider
            self.base_url = base_url
            self.model = model

    def _fake_create_ocr_manager(**kwargs):
        captured_provider["value"] = str(kwargs.get("provider") or "")
        return _DummyManager()

    monkeypatch.setattr(ocr_runtime_module, "create_ocr_manager", _fake_create_ocr_manager)
    monkeypatch.setattr(ocr_runtime_module, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime_module.setup_ocr_runtime(
        provider="openai",
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="test-key",
        ocr_ai_provider="openai",
        ocr_ai_base_url="https://example.test/v1",
        ocr_ai_model="gpt-4o-mini",
        ocr_ai_linebreak_assist=None,
        ocr_strict_mode=True,
    )

    assert captured_provider["value"] == "tesseract"
    assert setup.effective_ocr_provider == "tesseract"
    assert setup.ocr_geometry_strategy == "local_tesseract_geometry_with_ai_refine"
    assert setup.text_refiner is not None


def test_aiocr_dedicated_ocr_model_keeps_aiocr_geometry(monkeypatch) -> None:
    captured_provider: dict[str, str] = {}

    class _DummyManager:
        pass

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
        ) -> None:
            self.api_key = api_key
            self.provider = provider
            self.base_url = base_url
            self.model = model

    def _fake_create_ocr_manager(**kwargs):
        captured_provider["value"] = str(kwargs.get("provider") or "")
        return _DummyManager()

    monkeypatch.setattr(ocr_runtime_module, "create_ocr_manager", _fake_create_ocr_manager)
    monkeypatch.setattr(ocr_runtime_module, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime_module.setup_ocr_runtime(
        provider="openai",
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="test-key",
        ocr_ai_provider="siliconflow",
        ocr_ai_base_url="https://api.siliconflow.cn/v1",
        ocr_ai_model="Pro/deepseek-ai/deepseek-ocr",
        ocr_ai_linebreak_assist=None,
        ocr_strict_mode=True,
    )

    assert captured_provider["value"] == "aiocr"
    assert setup.effective_ocr_provider == "aiocr"
    assert setup.ocr_geometry_strategy == "direct"


def test_aiocr_geometry_mode_forced_direct_uses_ai_geometry(monkeypatch) -> None:
    captured_provider: dict[str, str] = {}

    class _DummyManager:
        pass

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
        ) -> None:
            self.api_key = api_key
            self.provider = provider
            self.base_url = base_url
            self.model = model

    def _fake_create_ocr_manager(**kwargs):
        captured_provider["value"] = str(kwargs.get("provider") or "")
        return _DummyManager()

    monkeypatch.setattr(ocr_runtime_module, "create_ocr_manager", _fake_create_ocr_manager)
    monkeypatch.setattr(ocr_runtime_module, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime_module.setup_ocr_runtime(
        provider="openai",
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="test-key",
        ocr_ai_provider="openai",
        ocr_ai_base_url="https://example.test/v1",
        ocr_ai_model="gpt-4o-mini",
        ocr_geometry_mode="direct_ai",
        ocr_ai_linebreak_assist=None,
        ocr_strict_mode=True,
    )

    debug = ocr_runtime_module.build_ocr_debug_payload(
        provider_requested="aiocr",
        ocr_render_dpi=200,
        scanned_render_dpi=200,
        ocr_ai_linebreak_assist=None,
        setup=setup,
    )

    assert captured_provider["value"] == "aiocr"
    assert setup.effective_ocr_provider == "aiocr"
    assert setup.ocr_geometry_strategy == "forced_direct_ai_geometry"
    assert debug["ocr_geometry"]["mode_requested"] == "direct_ai"
    assert debug["ocr_geometry"]["mode_effective"] == "direct_ai"


def test_aiocr_geometry_mode_forced_local_uses_tesseract_geometry(monkeypatch) -> None:
    captured_provider: dict[str, str] = {}

    class _DummyManager:
        pass

    class _DummyRefiner:
        def __init__(
            self,
            *,
            api_key: str,
            provider: str | None = None,
            base_url: str | None = None,
            model: str | None = None,
        ) -> None:
            self.api_key = api_key
            self.provider = provider
            self.base_url = base_url
            self.model = model

    def _fake_create_ocr_manager(**kwargs):
        captured_provider["value"] = str(kwargs.get("provider") or "")
        return _DummyManager()

    monkeypatch.setattr(ocr_runtime_module, "create_ocr_manager", _fake_create_ocr_manager)
    monkeypatch.setattr(ocr_runtime_module, "AiOcrTextRefiner", _DummyRefiner)

    setup = ocr_runtime_module.setup_ocr_runtime(
        provider="openai",
        api_key=None,
        base_url=None,
        model=None,
        ocr_provider="aiocr",
        ocr_baidu_app_id=None,
        ocr_baidu_api_key=None,
        ocr_baidu_secret_key=None,
        ocr_tesseract_min_confidence=None,
        ocr_tesseract_language=None,
        ocr_ai_api_key="test-key",
        ocr_ai_provider="siliconflow",
        ocr_ai_base_url="https://api.siliconflow.cn/v1",
        ocr_ai_model="Pro/deepseek-ai/deepseek-ocr",
        ocr_geometry_mode="local_tesseract",
        ocr_ai_linebreak_assist=None,
        ocr_strict_mode=True,
    )

    debug = ocr_runtime_module.build_ocr_debug_payload(
        provider_requested="aiocr",
        ocr_render_dpi=200,
        scanned_render_dpi=200,
        ocr_ai_linebreak_assist=None,
        setup=setup,
    )

    assert captured_provider["value"] == "tesseract"
    assert setup.effective_ocr_provider == "tesseract"
    assert (
        setup.ocr_geometry_strategy
        == "forced_local_tesseract_geometry_with_ai_refine"
    )
    assert debug["ocr_geometry"]["mode_requested"] == "local_tesseract"
    assert debug["ocr_geometry"]["mode_effective"] == "local_tesseract"
