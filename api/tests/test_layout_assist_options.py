from __future__ import annotations

from pathlib import Path

from app import worker as worker_module
from app.convert.pptx import scanned_page as scanned_page_module
from app.worker_helpers.layout_assist_stage import run_layout_assist_stage


def test_select_layout_assist_provider_prefers_main_claude_key(monkeypatch) -> None:
    class _DummyOpenAiProvider:
        def __init__(
            self,
            api_key: str,
            *,
            base_url: str | None = None,
            model: str | None = None,
        ) -> None:
            self.kind = "openai"
            self.api_key = api_key
            self.base_url = base_url
            self.model = model

    class _DummyAnthropicProvider:
        def __init__(self, api_key: str) -> None:
            self.kind = "claude"
            self.api_key = api_key

    monkeypatch.setattr(worker_module, "OpenAiProvider", _DummyOpenAiProvider)
    monkeypatch.setattr(worker_module, "AnthropicProvider", _DummyAnthropicProvider)

    provider = worker_module._select_layout_assist_provider(
        provider="claude",
        api_key="main-claude-key",
        base_url=None,
        model=None,
        ocr_ai_api_key="ocr-key",
        ocr_ai_base_url="https://api.siliconflow.cn/v1",
        ocr_ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
    )

    assert isinstance(provider, _DummyAnthropicProvider)
    assert provider.api_key == "main-claude-key"


def test_select_layout_assist_provider_falls_back_to_ocr_key(monkeypatch) -> None:
    class _DummyOpenAiProvider:
        def __init__(
            self,
            api_key: str,
            *,
            base_url: str | None = None,
            model: str | None = None,
        ) -> None:
            self.kind = "openai"
            self.api_key = api_key
            self.base_url = base_url
            self.model = model

    class _DummyAnthropicProvider:
        def __init__(self, api_key: str) -> None:
            self.kind = "claude"
            self.api_key = api_key

    monkeypatch.setattr(worker_module, "OpenAiProvider", _DummyOpenAiProvider)
    monkeypatch.setattr(worker_module, "AnthropicProvider", _DummyAnthropicProvider)

    provider = worker_module._select_layout_assist_provider(
        provider="openai",
        api_key=None,
        base_url=None,
        model=None,
        ocr_ai_api_key="ocr-fallback-key",
        ocr_ai_base_url="https://api.siliconflow.cn/v1",
        ocr_ai_model="Qwen/Qwen2.5-VL-72B-Instruct",
    )

    assert isinstance(provider, _DummyOpenAiProvider)
    assert provider.api_key == "ocr-fallback-key"
    assert provider.base_url == "https://api.siliconflow.cn/v1"
    assert provider.model == "Qwen/Qwen2.5-VL-72B-Instruct"


def test_layout_assist_stage_records_skip_warning_when_provider_missing(tmp_path) -> None:
    ir = {
        "source_pdf": "dummy.pdf",
        "pages": [
            {
                "page_index": 0,
                "elements": [{"type": "text", "text": "hello"}],
            }
        ],
    }

    result = run_layout_assist_stage(
        ir=ir,
        job_id="job-test",
        enable_layout_assist=True,
        layout_assist_apply_image_regions=True,
        input_pdf=tmp_path / "input.pdf",
        job_path=tmp_path / "job",
        artifacts_dir=tmp_path / "artifacts",
        scanned_render_dpi=150,
        select_provider=lambda: None,
        set_processing_progress=lambda *_args, **_kwargs: None,
        abort_if_cancelled=lambda **_kwargs: None,
    )

    warnings = result.ir.get("warnings")
    assert isinstance(warnings, list)
    assert "layout_assist_status=skipped_missing_provider" in warnings
    assert "layout_assist_pages_changed=0/0" in warnings


def test_collect_scanned_regions_keeps_ai_regions_with_ocr_text(monkeypatch) -> None:
    monkeypatch.setattr(
        scanned_page_module,
        "_detect_image_regions_from_render",
        lambda *_args, **_kwargs: [],
    )

    regions = scanned_page_module._collect_scanned_image_region_candidates(
        page={"image_regions": [[100.0, 100.0, 300.0, 260.0]]},
        render_path=Path("/tmp/not-used.png"),
        page_w_pt=1000.0,
        page_h_pt=1000.0,
        scanned_render_dpi=150,
        ocr_text_elements=[{"bbox_pt": [120.0, 120.0, 180.0, 140.0], "text": "hello"}],
        has_full_page_bg_image=False,
        text_coverage_ratio_fn=lambda _bbox: (0.0, 0),
    )

    assert regions
    assert regions[0][0] <= 100.0
    assert regions[0][1] <= 100.0
    assert regions[0][2] >= 300.0
    assert regions[0][3] >= 260.0
