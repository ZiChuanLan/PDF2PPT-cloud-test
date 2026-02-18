from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.convert.pptx.preview import _export_final_preview_page_image
from app.convert.pptx.scanned_page import _clear_regions_for_transparent_crops


class _SolidPixmap:
    def __init__(self, *, width: int, height: int, rgb: tuple[int, int, int]) -> None:
        self.width = int(width)
        self.height = int(height)
        self.n = 3
        self.samples = bytes(
            [int(rgb[0]) & 255, int(rgb[1]) & 255, int(rgb[2]) & 255]
        ) * (self.width * self.height)


def _write_solid(path: Path, rgb: tuple[int, int, int], *, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path)


def test_preview_prefers_images_bg_cleared_render(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    page_renders = artifacts / "page_renders"
    final_preview = artifacts / "final_preview"

    _write_solid(
        page_renders / "page-0000.clean.png",
        (20, 30, 40),
        size=(24, 24),
    )
    _write_solid(
        page_renders / "page-0000.clean.images-bg-cleared.png",
        (180, 190, 200),
        size=(24, 24),
    )

    _export_final_preview_page_image(
        page={"page_index": 0, "has_text_layer": False, "elements": []},
        page_index=0,
        page_w_pt=24.0,
        page_h_pt=24.0,
        source_pdf=artifacts / "missing.pdf",
        artifacts_dir=artifacts,
        dpi=72,
        scanned_image_region_crops=None,
    )

    out = final_preview / "page-0000.final.png"
    assert out.exists()
    color = Image.open(out).convert("RGB").getpixel((0, 0))
    assert color == (180, 190, 200)


def test_clear_regions_expands_edges_to_cover_halo(tmp_path: Path) -> None:
    root = tmp_path / "clear_regions_expand"
    cleaned = root / "clean.png"
    out = root / "out.png"

    _write_solid(cleaned, (0, 0, 0), size=(20, 20))
    pix = _SolidPixmap(width=20, height=20, rgb=(255, 255, 255))

    result = _clear_regions_for_transparent_crops(
        cleaned_render_path=cleaned,
        out_path=out,
        regions_pt=[[8.0, 8.0, 12.0, 12.0]],
        pix=pix,
        page_height_pt=20.0,
        dpi=72,
    )
    assert result == out
    rendered = Image.open(out).convert("RGB")
    assert rendered.getpixel((10, 10)) == (255, 255, 255)
    assert rendered.getpixel((7, 10)) == (255, 255, 255)
    assert rendered.getpixel((6, 10)) == (0, 0, 0)
