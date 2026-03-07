from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.routers import jobs
from app.convert.ocr.routing import (
    ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    ROUTE_KIND_REMOTE_DOC_PARSER,
)


class _FakeRemoteClient:
    def __init__(self, route_kind: str) -> None:
        self.route_kind = route_kind

    def ocr_image(self, image_path: str):
        _ = image_path
        return [
            {
                "text": "标题",
                "bbox": [10, 10, 120, 40],
                "confidence": 0.98,
            }
        ]


def test_ai_ocr_capability_check_reports_effective_route(monkeypatch, tmp_path) -> None:
    probe_image = tmp_path / "probe.png"
    Image.new("RGB", (256, 96), "white").save(probe_image)

    captured: dict[str, str | None] = {}

    def _fake_create_remote_ocr_client(**kwargs):
        captured.update(kwargs)
        return _FakeRemoteClient(ROUTE_KIND_REMOTE_DOC_PARSER)

    monkeypatch.setattr(jobs, "_create_ai_ocr_probe_image", lambda: probe_image)
    monkeypatch.setattr(jobs, "create_remote_ocr_client", _fake_create_remote_ocr_client)

    response = jobs._run_ai_ocr_capability_check(
        provider="siliconflow",
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="PaddlePaddle/PaddleOCR-VL-1.5",
    )

    assert response.ok is True
    assert response.check.route_kind == ROUTE_KIND_REMOTE_DOC_PARSER
    assert captured == {
        "requested_provider": "aiocr",
        "ai_api_key": "test-key",
        "ai_provider": "siliconflow",
        "ai_base_url": "https://api.siliconflow.cn/v1",
        "ai_model": "PaddlePaddle/PaddleOCR-VL-1.5",
        "route_kind": None,
        "ai_layout_model": None,
        "paddle_doc_max_side_px": None,
        "layout_block_max_concurrency": None,
        "request_rpm_limit": None,
        "request_tpm_limit": None,
        "request_max_retries": None,
    }


def test_ai_ocr_capability_check_passes_layout_block_route(monkeypatch, tmp_path) -> None:
    probe_image = tmp_path / "probe.png"
    Image.new("RGB", (256, 96), "white").save(probe_image)

    captured: dict[str, str | None] = {}

    def _fake_create_remote_ocr_client(**kwargs):
        captured.update(kwargs)
        return _FakeRemoteClient(ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR)

    monkeypatch.setattr(jobs, "_create_ai_ocr_probe_image", lambda: probe_image)
    monkeypatch.setattr(jobs, "create_remote_ocr_client", _fake_create_remote_ocr_client)

    response = jobs._run_ai_ocr_capability_check(
        provider="openai",
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        ocr_ai_chain_mode="layout_block",
        ocr_ai_layout_model="pp_doclayout_v3",
    )

    assert response.ok is True
    assert response.check.route_kind == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR
    assert captured == {
        "requested_provider": "aiocr",
        "ai_api_key": "test-key",
        "ai_provider": "openai",
        "ai_base_url": "https://example.com/v1",
        "ai_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "route_kind": "layout_block",
        "ai_layout_model": "pp_doclayout_v3",
        "paddle_doc_max_side_px": None,
        "layout_block_max_concurrency": None,
        "request_rpm_limit": None,
        "request_tpm_limit": None,
        "request_max_retries": None,
    }


def test_ai_ocr_capability_check_forwards_experimental_request_controls(
    monkeypatch, tmp_path
) -> None:
    probe_image = tmp_path / "probe.png"
    Image.new("RGB", (256, 96), "white").save(probe_image)

    captured: dict[str, str | int | None] = {}

    def _fake_create_remote_ocr_client(**kwargs):
        captured.update(kwargs)
        return _FakeRemoteClient(ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR)

    monkeypatch.setattr(jobs, "_create_ai_ocr_probe_image", lambda: probe_image)
    monkeypatch.setattr(jobs, "create_remote_ocr_client", _fake_create_remote_ocr_client)

    response = jobs._run_ai_ocr_capability_check(
        provider="openai",
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        ocr_ai_chain_mode="layout_block",
        ocr_ai_layout_model="pp_doclayout_v3",
        ocr_ai_block_concurrency=2,
        ocr_ai_requests_per_minute=40,
        ocr_ai_tokens_per_minute=120000,
        ocr_ai_max_retries=3,
    )

    assert response.ok is True
    assert captured["layout_block_max_concurrency"] == 2
    assert captured["request_rpm_limit"] == 40
    assert captured["request_tpm_limit"] == 120000
    assert captured["request_max_retries"] == 3
