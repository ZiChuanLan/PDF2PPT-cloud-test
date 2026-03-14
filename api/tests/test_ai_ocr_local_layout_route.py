from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert.ocr import ai_client as ai_client_module
from app.convert.ocr import local_providers
from app.convert.ocr.routing import ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR


class _DummyOpenAIClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _DummyVendorAdapter:
    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id

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
        lambda provider, base_url: _DummyVendorAdapter(str(provider or "openai")),
    )


def _assert_layout_block_item(
    item: dict[str, object],
    *,
    text: str,
    bbox: list[float],
    confidence: float,
    provider: str,
    model: str,
    label: str,
) -> None:
    assert item["text"] == text
    assert item["bbox"] == bbox
    assert item["confidence"] == confidence
    assert item["provider"] == provider
    assert item["model"] == model
    assert item["ocr_layout_label"] == label
    assert item["ocr_layout_geometry_source"] is None
    assert item["ocr_layout_geometry_kind"] is None
    assert item["ocr_layout_geometry_points"] is None


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


def test_extract_local_layout_blocks_preserves_polygon_geometry(monkeypatch) -> None:
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
                        "label": "figure",
                        "score": 0.91,
                        "order": 2,
                        "polygon_points": [[20, 20], [100, 24], [92, 84], [26, 78]],
                    }
                ]
            }
        }
    )

    assert image_regions == [
        {
            "bbox": [20.0, 20.0, 100.0, 84.0],
            "label": "figure",
            "score": 0.91,
            "order": 2,
            "geometry_source": "polygon_points",
            "geometry_kind": "polygon",
            "geometry_points": [
                [20.0, 20.0],
                [100.0, 24.0],
                [92.0, 84.0],
                [26.0, 78.0],
            ],
        }
    ]
    assert layout_blocks[0]["geometry_source"] == "polygon_points"
    assert layout_blocks[0]["geometry_kind"] == "polygon"
    assert layout_blocks[0]["geometry_points"] == [
        [20.0, 20.0],
        [100.0, 24.0],
        [92.0, 84.0],
        [26.0, 78.0],
    ]
    layout_debug = client.last_layout_analysis_debug
    assert layout_debug is not None
    assert layout_debug["raw_boxes"][0]["geometry_kind"] == "polygon"


def test_crop_layout_block_masks_polygon_geometry(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        provider="openai",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    bbox = [20.0, 20.0, 100.0, 100.0]
    draw.rectangle(bbox, fill=(255, 0, 0))
    polygon = [[60.0, 20.0], [100.0, 60.0], [60.0, 100.0], [20.0, 60.0]]
    draw.polygon([(x, y) for x, y in polygon], fill=(0, 0, 0))

    crop = client._crop_layout_block(image=image, bbox=bbox, geometry_points=polygon)

    assert crop is not None
    pad_x = min(24, max(2, int(round((bbox[2] - bbox[0]) * 0.03))))
    pad_y = min(24, max(2, int(round((bbox[3] - bbox[1]) * 0.18))))
    assert crop.getpixel((pad_x + 4, pad_y + 4)) == (255, 255, 255)
    assert crop.getpixel((pad_x + 40, pad_y + 40)) == (0, 0, 0)


def test_tighten_layout_block_bbox_by_visual_bounds_trims_loose_text_block(
    monkeypatch,
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

    image = Image.new("RGB", (620, 180), "white")
    draw = ImageDraw.Draw(image)
    loose_bbox = [40.0, 18.0, 580.0, 92.0]
    text_segments = [
        [170.0, 40.0, 248.0, 52.0],
        [270.0, 40.0, 352.0, 52.0],
        [372.0, 40.0, 454.0, 52.0],
    ]
    for segment in text_segments:
        draw.rectangle(segment, fill=(0, 0, 0))

    tightened = client._tighten_layout_block_bbox_by_visual_bounds(
        image=image,
        bbox=loose_bbox,
    )

    assert tightened is not None
    assert tightened[0] >= 150.0
    assert tightened[1] >= 28.0
    assert tightened[2] <= 480.0
    assert tightened[3] <= 66.0


def test_ocr_image_to_elements_keeps_layout_geometry_metadata(monkeypatch) -> None:
    manager = cast(
        local_providers.OcrManager,
        types.SimpleNamespace(
            ocr_image_lines=lambda _image_path, **_kwargs: [
                {
                    "bbox": [10.0, 8.0, 70.0, 28.0],
                    "text": "Yellowstone",
                    "confidence": 0.93,
                    "provider": "aiocr",
                    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "linebreak_assisted": True,
                    "linebreak_assist_source": "ai",
                    "ocr_layout_geometry_source": "polygon_points",
                    "ocr_layout_geometry_kind": "polygon",
                    "ocr_layout_geometry_points": [
                        [10.0, 8.0],
                        [70.0, 8.0],
                        [70.0, 28.0],
                        [10.0, 28.0],
                    ],
                }
            ],
            convert_bbox_to_pdf_coords=lambda **kwargs: kwargs["bbox"],
            last_provider_name="AiOcrClient",
            provider_id="aiocr",
        ),
    )

    monkeypatch.setattr(
        local_providers, "_dedupe_overlapping_ocr_items", lambda items: items
    )
    monkeypatch.setattr(
        local_providers, "_filter_contextual_noise_items", lambda items, **kwargs: items
    )

    image_path = Path("/tmp/local-layout-geometry.png")
    Image.new("RGB", (100, 60), "white").save(image_path)

    elements = local_providers.ocr_image_to_elements(
        image_path=str(image_path),
        ocr_manager=manager,
        page_width_pt=100.0,
        page_height_pt=60.0,
    )

    assert len(elements) == 1
    assert elements[0]["ocr_layout_geometry_source"] == "polygon_points"
    assert elements[0]["ocr_layout_geometry_kind"] == "polygon"
    assert elements[0]["ocr_layout_geometry_points_pt"] is None


def test_ocr_image_to_elements_converts_layout_geometry_points_to_pdf_coords(monkeypatch) -> None:
    manager = cast(
        local_providers.OcrManager,
        types.SimpleNamespace(
            ocr_image_lines=lambda _image_path, **_kwargs: [
                {
                    "bbox": [10.0, 8.0, 70.0, 28.0],
                    "text": "Yellowstone",
                    "confidence": 0.93,
                    "provider": "aiocr",
                    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "ocr_layout_geometry_source": "polygon_points",
                    "ocr_layout_geometry_kind": "polygon",
                    "ocr_layout_geometry_points": [
                        [10.0, 8.0],
                        [70.0, 8.0],
                        [70.0, 28.0],
                        [10.0, 28.0],
                    ],
                }
            ],
            convert_bbox_to_pdf_coords=lambda **kwargs: kwargs["bbox"],
            last_provider_name="AiOcrClient",
            provider_id="aiocr",
        ),
    )

    monkeypatch.setattr(
        local_providers, "_dedupe_overlapping_ocr_items", lambda items: items
    )
    monkeypatch.setattr(
        local_providers, "_filter_contextual_noise_items", lambda items, **kwargs: items
    )

    image_path = Path("/tmp/local-layout-geometry-pt.png")
    Image.new("RGB", (100, 60), "white").save(image_path)

    elements = local_providers.ocr_image_to_elements(
        image_path=str(image_path),
        ocr_manager=manager,
        page_width_pt=100.0,
        page_height_pt=60.0,
    )

    assert len(elements) == 1
    assert elements[0]["ocr_layout_geometry_points_pt"] == [
        [10.0, 8.0],
        [70.0, 8.0],
        [70.0, 28.0],
        [10.0, 28.0],
    ]


def test_sample_text_color_prefers_dark_ink_for_large_paragraph_bbox() -> None:
    bg_rgb = (244, 241, 234)
    ink_rgb = (12, 12, 10)

    image = Image.new("RGB", (420, 220), bg_rgb)
    draw = ImageDraw.Draw(image)
    bbox = [20.0, 20.0, 380.0, 180.0]

    # Simulate several thin paragraph lines placed away from the old sparse
    # 5x3 inner-grid sample rows so the regression exercises dense ink sampling.
    line_specs = [
        (40, 48, 36, 364),
        (84, 92, 28, 370),
        (128, 136, 42, 352),
    ]
    for y0, y1, x0, x1 in line_specs:
        draw.rectangle([x0, y0, x1, y1], fill=ink_rgb)
        # Break the strips into text-like segments instead of solid bars.
        draw.rectangle([x0 + 48, y0, x0 + 62, y1], fill=bg_rgb)
        draw.rectangle([x0 + 120, y0, x0 + 138, y1], fill=bg_rgb)
        draw.rectangle([x0 + 210, y0, x0 + 224, y1], fill=bg_rgb)

    sampled_hex = local_providers._sample_text_color(image, bbox)
    r = int(sampled_hex[1:3], 16)
    g = int(sampled_hex[3:5], 16)
    b = int(sampled_hex[5:7], 16)
    sampled_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    assert sampled_luma < 96.0


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
    assert len(items) == 1
    _assert_layout_block_item(
        items[0],
        text="Recognized Title",
        bbox=[10.0, 10.0, 110.0, 40.0],
        confidence=0.93,
        provider="openai",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        label="title",
    )
    assert client.last_image_regions_px == [[140.0, 20.0, 240.0, 120.0]]
    assert client.last_layout_blocks[0]["text"] == "Recognized Title"
    assert client.detect_image_regions(str(image_path)) == [[140.0, 20.0, 240.0, 120.0]]


def test_layout_block_route_uses_tightened_bbox_for_ocr_and_output(
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

    loose_bbox = [40.0, 18.0, 580.0, 92.0]
    monkeypatch.setattr(
        client,
        "_run_local_layout_analysis",
        lambda image_path: (
            [
                {
                    "label": "text",
                    "bbox": list(loose_bbox),
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
        return "Tightened text"

    monkeypatch.setattr(
        client,
        "_ocr_local_layout_block_crop",
        _fake_ocr_local_layout_block_crop,
    )

    image_path = tmp_path / "tightened-layout.png"
    image = Image.new("RGB", (620, 180), "white")
    draw = ImageDraw.Draw(image)
    for segment in (
        [170.0, 40.0, 248.0, 52.0],
        [270.0, 40.0, 352.0, 52.0],
        [372.0, 40.0, 454.0, 52.0],
    ):
        draw.rectangle(segment, fill=(0, 0, 0))
    image.save(image_path)

    items = client.ocr_image(str(image_path))

    assert len(items) == 1
    assert items[0]["text"] == "Tightened text"
    assert items[0]["bbox"][0] >= 150.0
    assert items[0]["bbox"][1] >= 28.0
    assert items[0]["bbox"][2] <= 480.0
    assert items[0]["bbox"][3] <= 66.0
    assert captured["crop_width"] < 420
    assert captured["crop_height"] < 64
    assert client.last_layout_blocks[0]["ocr_bbox_tightened"] is True
    assert client.last_layout_blocks[0]["bbox_original"] == loose_bbox


def test_layout_block_route_bypasses_wide_flat_blocks_to_direct_page_ocr(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-OCR",
        provider="siliconflow",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    monkeypatch.setattr(
        client,
        "_run_local_layout_analysis",
        lambda image_path: (
            [
                {
                    "label": "paragraph_title",
                    "bbox": [40.0, 60.0, 560.0, 90.0],
                    "score": 0.88,
                    "order": 0,
                    "text": "",
                },
                {
                    "label": "text",
                    "bbox": [110.0, 120.0, 500.0, 140.0],
                    "score": 0.85,
                    "order": 1,
                    "text": "",
                },
            ],
            [],
        ),
    )

    def _unexpected_layout_block(**kwargs):
        raise AssertionError("local layout block OCR should have been bypassed")

    monkeypatch.setattr(
        client,
        "_ocr_image_with_local_layout_blocks",
        _unexpected_layout_block,
    )
    monkeypatch.setattr(
        client,
        "_chat_completion",
        lambda **kwargs: types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            "<|ref|>释放未来·构建你的第一个AI智能体<|/ref|>"
                            "<|det|>[[170,60,454,90]]<|/det|>"
                            "<|ref|>HiAgent智能体开发平台大学生指南<|/ref|>"
                            "<|det|>[[120,120,500,140]]<|/det|>"
                        )
                    ),
                    finish_reason="stop",
                )
            ]
        ),
    )

    image_path = tmp_path / "wide-flat-bypass.png"
    Image.new("RGB", (620, 300), "white").save(image_path)

    items = client.ocr_image(str(image_path))

    assert len(items) == 2
    assert items[0]["text"] == "释放未来·构建你的第一个AI智能体"
    assert items[1]["text"] == "HiAgent智能体开发平台大学生指南"
    assert client.last_layout_analysis_debug["layout_block_bypass_reason"] == (
        "wide_flat_layout_blocks"
    )


def test_layout_block_route_keeps_non_deepseek_models_on_block_ocr(
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

    def _unexpected_bypass_check(**kwargs):
        raise AssertionError("non-DeepSeek layout_block OCR should not evaluate bypass")

    monkeypatch.setattr(
        client,
        "_should_bypass_local_layout_block_ocr",
        _unexpected_bypass_check,
    )
    monkeypatch.setattr(
        client,
        "_ocr_image_with_local_layout_blocks",
        lambda image_path, image: [
            {
                "text": "Block OCR text",
                "bbox": [40.0, 60.0, 560.0, 90.0],
                "confidence": 0.91,
                "provider": "openai",
                "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                "ocr_layout_label": "paragraph_title",
                "ocr_layout_geometry_source": None,
                "ocr_layout_geometry_kind": None,
                "ocr_layout_geometry_points": None,
            }
        ],
    )

    image_path = tmp_path / "non-deepseek-layout.png"
    Image.new("RGB", (620, 300), "white").save(image_path)

    items = client.ocr_image(str(image_path))

    assert len(items) == 1
    assert items[0]["text"] == "Block OCR text"
    assert client.route_kind == ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR


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
    assert len(items) == 1
    _assert_layout_block_item(
        items[0],
        text="Tiny strip text",
        bbox=[10.0, 10.0, 141.0, 22.0],
        confidence=0.91,
        provider="openai",
        model="Qwen/Qwen3-VL-235B-A22B-Instruct",
        label="text",
    )


def test_deepseek_layout_block_route_prefers_direct_page_ocr_and_keeps_image_regions(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-OCR",
        provider="siliconflow",
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
                    "bbox": [10.0, 10.0, 146.0, 44.0],
                    "score": 0.9,
                    "order": 0,
                    "text": "",
                }
            ],
            [
                {
                    "bbox": [170.0, 36.0, 286.0, 156.0],
                    "label": "image",
                    "score": 0.83,
                }
            ],
        ),
    )

    monkeypatch.setattr(
        client,
        "_ocr_image_with_local_layout_blocks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DeepSeek route should not use local layout block OCR")
        ),
    )
    monkeypatch.setattr(
        client,
        "_chat_completion",
        lambda **kwargs: types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            "<|ref|>DeepSeek direct page text<|/ref|>"
                            "<|det|>[[12,12,180,42]]<|/det|>"
                        )
                    ),
                    finish_reason="stop",
                )
            ]
        ),
    )

    image_path = tmp_path / "deepseek-direct-only.png"
    Image.new("RGB", (300, 180), "white").save(image_path)

    items = client.ocr_image(str(image_path))

    assert len(items) == 1
    assert items[0]["text"] == "DeepSeek direct page text"
    assert client.last_layout_analysis_debug["layout_block_bypass_reason"] == (
        "deepseek_model_prefers_direct_page_ocr"
    )
    assert client.last_image_regions_px == [
        {
            "bbox": [170.0, 36.0, 286.0, 156.0],
            "label": "image",
            "score": 0.83,
        }
    ]


def test_layout_block_route_skips_footer_blocks(
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
                    "label": "text",
                    "bbox": [10.0, 10.0, 146.0, 22.0],
                    "score": 0.9,
                    "order": 0,
                    "text": "",
                },
                {
                    "label": "footer",
                    "bbox": [10.0, 150.0, 140.0, 170.0],
                    "score": 0.8,
                    "order": 1,
                    "text": "",
                },
            ],
            [],
        ),
    )

    seen_labels: list[str] = []

    def _fake_ocr_local_layout_block_crop(**kwargs):
        seen_labels.append(str(kwargs["label"]))
        return "Body text"

    monkeypatch.setattr(
        client,
        "_ocr_local_layout_block_crop",
        _fake_ocr_local_layout_block_crop,
    )

    image_path = tmp_path / "footer-skip.png"
    Image.new("RGB", (300, 180), "white").save(image_path)

    items = client.ocr_image(str(image_path))

    assert seen_labels == ["text"]
    assert len(items) == 1
    assert client.last_layout_blocks[1]["ocr_skipped"] is True
    assert client.last_layout_blocks[1]["ocr_skip_reason"] == "low_value_layout_label"


def test_deepseek_layout_block_crop_extracts_text_from_grounding_tags(
    monkeypatch,
) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-OCR",
        provider="siliconflow",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    monkeypatch.setattr(
        client,
        "_chat_completion",
        lambda **kwargs: types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            "<|ref|>Invoice ID: A-2048-17<|/ref|>"
                            "<|det|>[[12,34,240,92]]<|/det|>"
                        )
                    )
                )
            ]
        ),
    )

    text = client._ocr_local_layout_block_crop(
        data_uri="data:image/png;base64,AAAA",
        label="text",
        crop_width=320,
        crop_height=64,
        effective_model="deepseek-ai/DeepSeek-OCR",
    )

    assert text == "Invoice ID: A-2048-17"


def test_clean_plain_text_ocr_output_strips_special_box_tokens(
    monkeypatch,
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

    text = client._clean_plain_text_ocr_output(
        "&lt;|begin_of_box|&gt;Invoice ID: A-2048-17&lt;|end_of_box|&gt;\n"
        "<|begin_of_text|>Status: Ready<|end_of_text|>\n"
        "[[12,34,240,92]]"
    )

    assert text == "Invoice ID: A-2048-17\nStatus: Ready"


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
            create_model=lambda model_name: (
                create_model_calls.append(model_name) or fake_model
            )
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


def test_layout_block_crop_retries_timeout_once_for_qwen(monkeypatch) -> None:
    _patch_openai_and_adapter(monkeypatch)

    client = ai_client_module.AiOcrClient(
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        provider="siliconflow",
        layout_model="pp_doclayout_v3",
        route_kind=ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    )

    calls: list[dict[str, object]] = []

    def _fake_chat_completion(**kwargs):
        calls.append(
            {
                "timeout_s": kwargs.get("timeout_s"),
                "request_label": kwargs.get("request_label"),
            }
        )
        if len(calls) == 1:
            raise TimeoutError("Request timed out.")
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="Recovered layout block text")
                )
            ]
        )

    monkeypatch.setattr(client, "_chat_completion", _fake_chat_completion)

    text = client._ocr_local_layout_block_crop(
        data_uri="data:image/png;base64,AAAA",
        label="text",
        crop_width=1247,
        crop_height=221,
        effective_model="Qwen/Qwen2.5-VL-72B-Instruct",
    )

    assert text == "Recovered layout block text"
    assert len(calls) == 2
    assert calls[0]["request_label"] == "layout_block_crop"
    assert calls[1]["request_label"] == "layout_block_crop_retry"
    first_timeout = cast(float, calls[0]["timeout_s"])
    second_timeout = cast(float, calls[1]["timeout_s"])
    assert first_timeout >= 40.0
    assert second_timeout > first_timeout
