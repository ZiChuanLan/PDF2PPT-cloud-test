from __future__ import annotations

from app.convert.ocr import _extract_deepseek_tagged_items


def _as_xyxy(item: dict) -> tuple[float, float, float, float]:
    bbox = item.get("bbox")
    assert isinstance(bbox, list) and len(bbox) == 4
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def test_deepseek_ref_det_alternating_pairs_without_cross_pairing() -> None:
    content = (
        "<|ref|>A<|/ref|><|det|>[[10,20,30,40]]<|/det|>\n"
        "<|ref|>B<|/ref|><|det|>[[50,60,70,80]]<|/det|>\n"
    )
    items = _extract_deepseek_tagged_items(content)
    assert items is not None
    assert len(items) == 2

    assert items[0]["text"] == "A"
    assert _as_xyxy(items[0]) == (10.0, 20.0, 30.0, 40.0)

    assert items[1]["text"] == "B"
    assert _as_xyxy(items[1]) == (50.0, 60.0, 70.0, 80.0)

    # Regression: avoid producing ("B" @ first bbox) via det-then-ref cross match.
    assert not any(
        it.get("text") == "B" and _as_xyxy(it) == (10.0, 20.0, 30.0, 40.0)
        for it in items
    )


def test_deepseek_det_ref_alternating_pairs_without_cross_pairing() -> None:
    content = (
        "<|det|>[[10,20,30,40]]<|/det|><|ref|>A<|/ref|>\n"
        "<|det|>[[50,60,70,80]]<|/det|><|ref|>B<|/ref|>\n"
    )
    items = _extract_deepseek_tagged_items(content)
    assert items is not None
    assert len(items) == 2

    assert items[0]["text"] == "A"
    assert _as_xyxy(items[0]) == (10.0, 20.0, 30.0, 40.0)

    assert items[1]["text"] == "B"
    assert _as_xyxy(items[1]) == (50.0, 60.0, 70.0, 80.0)


def test_deepseek_prefers_inline_text_when_ref_is_generic_label() -> None:
    content = "<|ref|>text<|/ref|><|det|>[[1,2,3,4]]<|/det|>Hello world\n"
    items = _extract_deepseek_tagged_items(content)
    assert items is not None
    assert len(items) == 1
    assert items[0]["text"] == "Hello world"
    assert _as_xyxy(items[0]) == (1.0, 2.0, 3.0, 4.0)


def test_deepseek_supports_html_escaped_tags() -> None:
    content = "&lt;|ref|&gt;A&lt;|/ref|&gt;&lt;|det|&gt;[[10,20,30,40]]&lt;|/det|&gt;\n"
    items = _extract_deepseek_tagged_items(content)
    assert items is not None
    assert len(items) == 1
    assert items[0]["text"] == "A"
    assert _as_xyxy(items[0]) == (10.0, 20.0, 30.0, 40.0)
