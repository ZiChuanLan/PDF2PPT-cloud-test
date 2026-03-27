# pyright: reportMissingImports=false

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert.pptx import scanned_page


class _PixStub:
    def __init__(self, image: Image.Image) -> None:
        rgb = image.convert("RGB")
        self.width, self.height = rgb.size
        self.n = 3
        self.samples = rgb.tobytes()


def test_erase_regions_fill_uses_polygon_mask_when_available(tmp_path) -> None:
    render_path = tmp_path / "render.png"
    out_path = tmp_path / "render.clean.png"

    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 100, 100], fill=(180, 180, 180))
    draw.polygon([(60, 20), (100, 60), (60, 100), (20, 60)], fill=(0, 0, 0))
    image.save(render_path)

    result_path = scanned_page._erase_regions_in_render_image(
        render_path,
        out_path=out_path,
        erase_bboxes_pt=[[20.0, 20.0, 100.0, 100.0]],
        erase_polygons_pt=[
            [[60.0, 20.0], [100.0, 60.0], [60.0, 100.0], [20.0, 60.0]]
        ],
        page_height_pt=120.0,
        dpi=72,
        text_erase_mode="fill",
    )

    assert result_path == out_path
    cleaned = Image.open(result_path).convert("RGB")
    center = cleaned.getpixel((60, 60))
    corner = cleaned.getpixel((24, 24))

    assert min(center) >= 230
    assert 150 <= corner[0] <= 210
    assert abs(int(corner[0]) - int(corner[1])) <= 5
    assert abs(int(corner[1]) - int(corner[2])) <= 5


def test_erase_regions_fill_keeps_protected_pixels(tmp_path) -> None:
    render_path = tmp_path / "render.protect.png"
    out_path = tmp_path / "render.protect.clean.png"

    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 100, 100], fill=(180, 180, 180))
    draw.rectangle([20, 48, 100, 72], fill=(0, 0, 0))
    image.save(render_path)

    result_path = scanned_page._erase_regions_in_render_image(
        render_path,
        out_path=out_path,
        erase_bboxes_pt=[[20.0, 48.0, 100.0, 72.0]],
        protect_bboxes_pt=[[60.0, 20.0, 100.0, 100.0]],
        page_height_pt=120.0,
        dpi=72,
        text_erase_mode="fill",
    )

    assert result_path == out_path
    cleaned = Image.open(result_path).convert("RGB")
    erased_left = cleaned.getpixel((36, 60))
    protected_right = cleaned.getpixel((84, 60))

    assert min(erased_left) >= 150
    assert max(protected_right) <= 40


def test_clear_regions_for_transparent_crops_uses_polygon_mask_when_available(
    tmp_path,
) -> None:
    render_path = tmp_path / "render.png"
    out_path = tmp_path / "render.clear.png"

    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 100, 100], fill=(180, 180, 180))
    draw.polygon([(60, 20), (100, 60), (60, 100), (20, 60)], fill=(0, 0, 0))
    image.save(render_path)

    result_path = scanned_page._clear_regions_for_transparent_crops(
        cleaned_render_path=render_path,
        out_path=out_path,
        regions_pt=[[20.0, 20.0, 100.0, 100.0]],
        regions_polygons_pt=[
            [[60.0, 20.0], [100.0, 60.0], [60.0, 100.0], [20.0, 60.0]]
        ],
        pix=_PixStub(image),
        page_height_pt=120.0,
        dpi=72,
    )

    assert result_path == out_path
    cleared = Image.open(result_path).convert("RGB")
    center = cleared.getpixel((60, 60))
    corner = cleared.getpixel((24, 24))

    assert min(center) >= 230
    assert 150 <= corner[0] <= 210
    assert abs(int(corner[0]) - int(corner[1])) <= 5
    assert abs(int(corner[1]) - int(corner[2])) <= 5


def test_build_scanned_image_region_infos_preserves_polygon_masked_crop(
    tmp_path,
) -> None:
    render_path = tmp_path / "render.png"
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 100, 100], fill=(180, 180, 180))
    draw.polygon([(60, 20), (100, 60), (60, 100), (20, 60)], fill=(0, 0, 0))
    image.save(render_path)

    infos = scanned_page._build_scanned_image_region_infos(
        page={
            "image_regions": [
                {
                    "bbox_pt": [20.0, 20.0, 100.0, 100.0],
                    "geometry_kind": "polygon",
                    "geometry_points_pt": [
                        [60.0, 20.0],
                        [100.0, 60.0],
                        [60.0, 100.0],
                        [20.0, 60.0],
                    ],
                }
            ]
        },
        render_path=render_path,
        artifacts_dir=tmp_path / "artifacts",
        page_index=0,
        page_w_pt=120.0,
        page_h_pt=120.0,
        scanned_render_dpi=72,
        baseline_ocr_h_pt=12.0,
        ocr_text_elements=[],
        has_full_page_bg_image=False,
        text_coverage_ratio_fn=lambda _bbox: (0.0, 0),
        text_inside_counts_fn=lambda _bbox: (0, 0),
    )

    assert len(infos) == 1
    assert infos[0].geometry_kind == "polygon"
    assert infos[0].geometry_points_pt == [
        [60.0, 20.0],
        [100.0, 60.0],
        [60.0, 100.0],
        [20.0, 60.0],
    ]

    crop = Image.open(infos[0].crop_path).convert("RGBA")
    assert crop.getpixel((4, 4))[3] == 0
    assert crop.getpixel((40, 40))[3] == 255


def test_apply_text_cutouts_to_scanned_ai_hint_polygon_crop(tmp_path) -> None:
    render_path = tmp_path / "render.text-cutout.png"
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 10, 110, 110], fill=(180, 180, 180))
    draw.polygon(
        [(70, 10), (110, 10), (110, 110), (50, 110), (60, 60)],
        fill=(0, 0, 0),
    )
    image.save(render_path)

    info = scanned_page._ScannedImageRegionInfo(
        bbox_pt=[50.0, 10.0, 110.0, 110.0],
        suppress_bbox_pt=[48.0, 8.0, 112.0, 112.0],
        crop_path=tmp_path / "artifacts" / "crop.png",
        shape_confirmed=True,
        ai_hint=True,
        geometry_kind="polygon",
        geometry_points_pt=[
            [70.0, 10.0],
            [110.0, 10.0],
            [110.0, 110.0],
            [50.0, 110.0],
            [60.0, 60.0],
        ],
    )
    scanned_page._save_scanned_image_region_crop(
        img=image,
        bbox_pt=info.bbox_pt,
        crop_out_path=info.crop_path,
        page_h_pt=120.0,
        scanned_render_dpi=72,
        geometry_points_pt=info.geometry_points_pt,
    )

    kept = scanned_page._filter_scanned_ocr_text_elements(
        ocr_text_elements=[
            {
                "text": "Planning",
                "bbox_pt": [20.0, 40.0, 80.0, 70.0],
                "ocr_layout_geometry_kind": "polygon",
                "ocr_layout_geometry_points_pt": [
                    [20.0, 40.0],
                    [80.0, 40.0],
                    [80.0, 70.0],
                    [20.0, 70.0],
                ],
            }
        ],
        image_region_infos=[info],
        baseline_ocr_h_pt=12.0,
    )

    scanned_page._apply_text_cutouts_to_scanned_image_region_crops(
        infos=[info],
        render_path=render_path,
        page_h_pt=120.0,
        scanned_render_dpi=72,
        ocr_text_elements=kept,
    )

    crop = Image.open(info.crop_path).convert("RGBA")
    assert crop.getpixel((22, 40))[3] == 0
    assert crop.getpixel((50, 40))[3] == 255


def test_build_scanned_image_region_infos_trusts_explicit_image_regions(
    tmp_path,
) -> None:
    render_path = tmp_path / "render.trust-ai.png"
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 110, 110], fill=(180, 180, 180))
    draw.rectangle([24, 28, 96, 88], fill=(40, 40, 40))
    image.save(render_path)

    infos = scanned_page._build_scanned_image_region_infos(
        page={
            "image_regions": [
                {
                    "bbox_pt": [10.0, 10.0, 110.0, 110.0],
                }
            ]
        },
        render_path=render_path,
        artifacts_dir=tmp_path / "artifacts",
        page_index=0,
        page_w_pt=120.0,
        page_h_pt=120.0,
        scanned_render_dpi=72,
        baseline_ocr_h_pt=12.0,
        ocr_text_elements=[],
        has_full_page_bg_image=False,
        text_coverage_ratio_fn=lambda _bbox: (0.80, 10),
        text_inside_counts_fn=lambda _bbox: (10, 10),
    )

    assert len(infos) == 1
    assert infos[0].bbox_pt == [10.0, 10.0, 110.0, 110.0]
    assert infos[0].ai_hint is True
    assert infos[0].crop_path.exists()


def test_build_scanned_image_region_infos_skips_without_explicit_image_regions(
    tmp_path, monkeypatch
) -> None:
    render_path = tmp_path / "render.no-fallback.png"
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 110, 110], fill=(180, 180, 180))
    draw.rectangle([24, 28, 96, 88], fill=(40, 40, 40))
    image.save(render_path)

    monkeypatch.setattr(
        scanned_page,
        "_detect_image_regions_from_render",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("render-based fallback should not run")
        ),
    )

    infos = scanned_page._build_scanned_image_region_infos(
        page={},
        render_path=render_path,
        artifacts_dir=tmp_path / "artifacts",
        page_index=0,
        page_w_pt=120.0,
        page_h_pt=120.0,
        scanned_render_dpi=72,
        baseline_ocr_h_pt=12.0,
        ocr_text_elements=[],
        has_full_page_bg_image=False,
        text_coverage_ratio_fn=lambda _bbox: (0.0, 0),
        text_inside_counts_fn=lambda _bbox: (0, 0),
    )

    assert infos == []


def test_merge_fragmented_scanned_regions_merges_same_ai_hint_polygon_fragments(
    tmp_path, monkeypatch
) -> None:
    img = Image.new("RGB", (200, 120), "white")
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    polygon = [[10.0, 20.0], [170.0, 20.0], [170.0, 96.0], [10.0, 96.0]]
    infos = [
        scanned_page._ScannedImageRegionInfo(
            bbox_pt=[50.0, 20.0, 150.0, 60.0],
            suppress_bbox_pt=[48.0, 18.0, 152.0, 62.0],
            crop_path=crops_dir / "a.png",
            shape_confirmed=True,
            ai_hint=True,
            geometry_kind="polygon",
            geometry_points_pt=polygon,
        ),
        scanned_page._ScannedImageRegionInfo(
            bbox_pt=[50.0, 55.0, 110.0, 95.0],
            suppress_bbox_pt=[48.0, 53.0, 112.0, 97.0],
            crop_path=crops_dir / "b.png",
            shape_confirmed=True,
            ai_hint=True,
            geometry_kind="polygon",
            geometry_points_pt=polygon,
        ),
    ]

    monkeypatch.setattr(
        scanned_page,
        "_analyze_shape_crop",
        lambda _path: {"confirmed": True},
    )

    merged = scanned_page._try_merge_fragmented_scanned_image_regions(
        infos=infos,
        img=img,
        crops_dir=crops_dir,
        page_index=0,
        page_w_pt=200.0,
        page_h_pt=120.0,
        scanned_render_dpi=72,
        baseline_ocr_h_pt=12.0,
        ocr_text_elements=[],
        text_coverage_ratio_fn=lambda _bbox: (0.0, 0),
    )

    assert len(merged) == 1
    assert merged[0].bbox_pt == [50.0, 20.0, 150.0, 95.0]
    assert merged[0].ai_hint is True
    assert merged[0].geometry_kind is None
    assert merged[0].geometry_points_pt is None


def test_filter_scanned_ocr_text_elements_keeps_partial_overlap_for_ai_hint() -> None:
    info = scanned_page._ScannedImageRegionInfo(
        bbox_pt=[10.0, 10.0, 70.0, 70.0],
        suppress_bbox_pt=[0.0, 0.0, 80.0, 80.0],
        crop_path=Path("/tmp/ai-hint.png"),
        shape_confirmed=True,
        ai_hint=True,
    )

    kept = scanned_page._filter_scanned_ocr_text_elements(
        ocr_text_elements=[
            {
                "text": "Planning",
                "bbox_pt": [20.0, 20.0, 90.0, 60.0],
            }
        ],
        image_region_infos=[info],
        baseline_ocr_h_pt=12.0,
    )

    assert len(kept) == 1
    assert kept[0]["text"] == "Planning"


def test_filter_scanned_ocr_text_elements_suppresses_mostly_inside_ai_hint() -> None:
    info = scanned_page._ScannedImageRegionInfo(
        bbox_pt=[10.0, 10.0, 70.0, 70.0],
        suppress_bbox_pt=[0.0, 0.0, 80.0, 80.0],
        crop_path=Path("/tmp/ai-hint.png"),
        shape_confirmed=True,
        ai_hint=True,
    )

    kept = scanned_page._filter_scanned_ocr_text_elements(
        ocr_text_elements=[
            {
                "text": "Planning",
                "bbox_pt": [18.0, 18.0, 62.0, 58.0],
            }
        ],
        image_region_infos=[info],
        baseline_ocr_h_pt=12.0,
    )

    assert kept == []


def test_filter_scanned_ocr_text_elements_keeps_non_ai_shape_confirmed_suppression() -> None:
    info = scanned_page._ScannedImageRegionInfo(
        bbox_pt=[10.0, 10.0, 70.0, 70.0],
        suppress_bbox_pt=[0.0, 0.0, 80.0, 80.0],
        crop_path=Path("/tmp/non-ai.png"),
        shape_confirmed=True,
        ai_hint=False,
    )

    kept = scanned_page._filter_scanned_ocr_text_elements(
        ocr_text_elements=[
            {
                "text": "Planning",
                "bbox_pt": [20.0, 20.0, 90.0, 60.0],
            }
        ],
        image_region_infos=[info],
        baseline_ocr_h_pt=12.0,
    )

    assert kept == []


def test_small_text_fragment_region_rejects_small_same_line_crop() -> None:
    assert scanned_page._is_small_text_fragment_region(
        [66.24, 296.64, 190.08, 397.44],
        page_w_pt=1366.08,
        page_h_pt=768.0,
        baseline_ocr_h_pt=34.55,
        ocr_text_elements=[
            {
                "text": "模型(思想引擎)",
                "bbox_pt": [168.81, 368.52, 319.97, 401.63],
            }
        ],
    )


def test_small_text_fragment_region_keeps_large_diagram_region() -> None:
    assert not scanned_page._is_small_text_fragment_region(
        [581.76, 129.6, 1330.56, 705.6],
        page_w_pt=1366.08,
        page_h_pt=768.0,
        baseline_ocr_h_pt=34.55,
        ocr_text_elements=[
            {
                "text": "短期记忆(上下文)",
                "bbox_pt": [788.96, 151.15, 919.97, 166.99],
            },
            {
                "text": "Planning",
                "bbox_pt": [1030.0, 330.0, 1120.0, 390.0],
            },
        ],
    )


def test_apply_max_filter_l_matches_pil_max_filter() -> None:
    image = Image.new("L", (9, 9), 0)
    for x, y, value in (
        (0, 0, 12),
        (2, 2, 160),
        (4, 1, 200),
        (6, 6, 80),
        (8, 8, 255),
    ):
        image.putpixel((x, y), value)

    expected = image.filter(ImageFilter.MaxFilter(5))
    actual = scanned_page._apply_max_filter_l(image, size=5)

    assert list(actual.getdata()) == list(expected.getdata())


def test_save_scanned_regions_debug_overlay_writes_regions_json_only(tmp_path) -> None:
    render_path = tmp_path / "render.png"
    Image.new("RGB", (120, 120), "white").save(render_path)

    scanned_page._save_scanned_regions_debug_overlay(
        render_path=render_path,
        regions_pt=[[20.0, 20.0, 100.0, 100.0]],
        artifacts_dir=tmp_path / "artifacts",
        page_index=0,
        page_h_pt=120.0,
        scanned_render_dpi=72,
    )

    json_path = tmp_path / "artifacts" / "image_regions" / "page-0000.regions.json"
    png_path = tmp_path / "artifacts" / "image_regions" / "page-0000.regions.png"

    assert json_path.exists()
    assert not png_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["page_index"] == 0
    assert payload["regions_pt"] == [[20.0, 20.0, 100.0, 100.0]]
