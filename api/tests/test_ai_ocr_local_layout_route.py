from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

from PIL import Image


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert.ocr import ai_client as ai_client_module
from app.convert.ocr.routing import ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR


class _DummyOpenAIClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _DummyVendorAdapter:
    provider_id = "openai"

    def resolve_base_url(self, base_url: str | None) -> str | None:
        return base_url

    def resolve_model(self, model: str | None) -> str | None:
        return model

    def should_use_paddle_doc_parser(
        self,
        *,
        base_url: str | None,
        model_name: str | None,
    ) -> bool:
        _ = base_url
        _ = model_name
        return False

    def build_user_content(
        self,
        *,
        prompt: str,
        image_data_uri: str,
        image_first: bool,
    ):
        _ = image_data_uri
        _ = image_first
        return prompt

    def clamp_max_tokens(self, value: int, *, kind: str) -> int:
        _ = kind
        return value


def _patch_openai_and_adapter(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=_DummyOpenAIClient),
    )
    monkeypatch.setattr(
        ai_client_module,
        "_create_ai_ocr_vendor_adapter",
        lambda provider, base_url: _DummyVendorAdapter(),
    )


def test_extract_local_layout_blocks_marks_image_regions(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        provider="openai",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    layout_blocks, image_regions = client._extract_local_layout_blocks(
        {
            "res": {
                "boxes": [
                    {
                        "label": "image",
                        "score": 0.88,
                        "coordinate": [140, 20, 240, 120],
                    },
                    {
                        "label": "text",
                        "score": 0.93,
                        "order": 1,
                        "coordinate": [10, 10, 110, 40],
                    },
                ]
            }
        }
    )

    assert [block["label"] for block in layout_blocks] == ["text", "image"]
    assert layout_blocks[0]["bbox"] == [10.0, 10.0, 110.0, 40.0]
    assert image_regions == [[140.0, 20.0, 240.0, 120.0]]


def test_layout_block_route_ocr_image_uses_local_layout_blocks(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        provider="openai",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    monkeypatch.setattr(
        client,
        "_run_local_layout_analysis",
        lambda image_path: (
            [
                {
                    "label": "title",
                    "bbox": [10.0, 10.0, 110.0, 40.0],
                    "score": 0.93,
                    "order": 1,
                    "text": "",
                },
                {
                    "label": "image",
                    "bbox": [140.0, 20.0, 240.0, 120.0],
                    "score": 0.88,
                    "order": None,
                    "text": "",
                },
            ],
            [[140.0, 20.0, 240.0, 120.0]],
        ),
    )
    monkeypatch.setattr(
        client,
        "_ocr_local_layout_block_crop",
        lambda **kwargs: "Recognized Title",
    )

    image_path = tmp_path / "page.png"
    Image.new("RGB", (300, 180), "white").save(image_path)

    items = client.ocr_image(str(image_path))

    assert client.route_kind == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR
    assert items == [
        {
            "text": "Recognized Title",
            "bbox": [10.0, 10.0, 110.0, 40.0],
            "confidence": 0.93,
            "provider": "openai",
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "ocr_layout_label": "title",
        }
    ]
    assert client.last_image_regions_px == [[140.0, 20.0, 240.0, 120.0]]
    assert client.last_layout_blocks[0]["text"] == "Recognized Title"
    assert client.detect_image_regions(str(image_path)) == [
        [140.0, 20.0, 240.0, 120.0]
    ]


def test_layout_block_route_upscales_tiny_qwen3_crops(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen3-VL-235B-A22B-Instruct",
        provider="openai",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    monkeypatch.setattr(
        client,
        "_run_local_layout_analysis",
        lambda image_path: (
            [
                {
                    "label": "text",
                    "bbox": [10.0, 10.0, 141.0, 22.0],
                    "score": 0.91,
                    "order": 0,
                    "text": "",
                }
            ],
            [],
        ),
    )

    captured: dict[str, int] = {}

    def _fake_ocr_local_layout_block_crop(**kwargs):
        captured["crop_width"] = int(kwargs["crop_width"])
        captured["crop_height"] = int(kwargs["crop_height"])
        return "Tiny strip text"

    monkeypatch.setattr(
        client,
        "_ocr_local_layout_block_crop",
        _fake_ocr_local_layout_block_crop,
    )

    image_path = tmp_path / "tiny-strip.png"
    Image.new("RGB", (300, 180), "white").save(image_path)

    items = client.ocr_image(str(image_path))

    assert captured["crop_height"] >= 32
    assert captured["crop_width"] > 139
    assert items == [
        {
            "text": "Tiny strip text",
            "bbox": [10.0, 10.0, 141.0, 22.0],
            "confidence": 0.91,
            "provider": "openai",
            "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "ocr_layout_label": "text",
        }
    ]


def test_local_layout_analysis_serializes_shared_model_predict(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_openai_and_adapter(monkeypatch)

    class _FakeLayoutModel:
        def __init__(self) -> None:
            self._active = 0
            self._active_lock = threading.Lock()
            self.max_active = 0
            self._shared_output = {"res": {"boxes": []}}

        def predict(self, input=None, **kwargs):
            image_path = str(input or kwargs.get("input") or "")
            with self._active_lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            try:
                time.sleep(0.02)
                page_name = Path(image_path).name
                if page_name == "page-a.png":
                    self._shared_output["res"]["boxes"] = [
                        {
                            "label": "text",
                            "order": 0,
                            "coordinate": [10, 10, 110, 40],
                        }
                    ]
                else:
                    self._shared_output["res"]["boxes"] = [
                        {
                            "label": "text",
                            "order": 0,
                            "coordinate": [200, 200, 320, 260],
                        }
                    ]
                return self._shared_output
            finally:
                with self._active_lock:
                    self._active -= 1

    fake_model = _FakeLayoutModel()
    create_model_calls: list[str] = []

    monkeypatch.setitem(
        sys.modules,
        "paddlex",
        types.SimpleNamespace(
            create_model=lambda model_name: create_model_calls.append(model_name)
            or fake_model
        ),
    )
    monkeypatch.setattr(ai_client_module.AiOcrClient, "_local_layout_model", None)
    monkeypatch.setattr(ai_client_module.AiOcrClient, "_local_layout_model_name", None)
    original_extract = ai_client_module.AiOcrClient._extract_local_layout_blocks

    def _slow_extract(self, output):
        time.sleep(0.05)
        return original_extract(self, output)

    monkeypatch.setattr(
        ai_client_module.AiOcrClient,
        "_extract_local_layout_blocks",
        _slow_extract,
    )

    client_a = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        provider="openai",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )
    client_b = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        provider="openai",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    image_a = tmp_path / "page-a.png"
    image_b = tmp_path / "page-b.png"
    Image.new("RGB", (100, 100), "white").save(image_a)
    Image.new("RGB", (100, 100), "white").save(image_b)

    start_event = threading.Event()
    results: dict[str, tuple[list[dict[str, object]], list[list[float]]]] = {}

    def _run(client, image_path: Path, key: str) -> None:
        start_event.wait(timeout=1.0)
        results[key] = client._run_local_layout_analysis(str(image_path))

    thread_a = threading.Thread(target=_run, args=(client_a, image_a, "a"))
    thread_b = threading.Thread(target=_run, args=(client_b, image_b, "b"))
    thread_a.start()
    thread_b.start()
    start_event.set()
    thread_a.join(timeout=2.0)
    thread_b.join(timeout=2.0)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert create_model_calls == ["PP-DocLayoutV3"]
    assert fake_model.max_active == 1
    assert results["a"][0][0]["bbox"] == [10.0, 10.0, 110.0, 40.0]
    assert results["b"][0][0]["bbox"] == [200.0, 200.0, 320.0, 260.0]
