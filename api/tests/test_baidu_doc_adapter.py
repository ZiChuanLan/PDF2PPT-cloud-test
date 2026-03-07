from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.convert import baidu_doc_adapter as adapter


def test_collect_page_payload_does_not_reindex_nested_page_sizes() -> None:
    payload = {
        "pages": [
            {
                "page_idx": 0,
                "width": 1000,
                "height": 1000,
                "items": [
                    {"page_idx": 0, "bbox": [100, 100, 200, 200], "text": "page-0"}
                ],
            },
            {
                "page_idx": 1,
                "width": 1000,
                "height": 1000,
                "items": [
                    {
                        "size": {"width": 100, "height": 50},
                        "elements": [
                            {
                                "page_idx": 1,
                                "bbox": [150, 200, 250, 300],
                                "text": "nested-page-1",
                            }
                        ],
                    }
                ],
            },
        ]
    }

    payload_page_sizes: dict[int, tuple[float, float]] = {}
    adapter._collect_page_payload(payload, out_pages=payload_page_sizes)

    assert payload_page_sizes == {
        0: (1000.0, 1000.0),
        1: (1000.0, 1000.0),
    }


def test_collect_content_items_keeps_bbox_scaling_stable_across_pages() -> None:
    payload = {
        "pages": [
            {
                "page_idx": 0,
                "width": 1000,
                "height": 1000,
                "items": [
                    {"page_idx": 0, "bbox": [100, 100, 200, 200], "text": "page-0"}
                ],
            },
            {
                "page_idx": 1,
                "width": 1000,
                "height": 1000,
                "items": [
                    {
                        "size": {"width": 100, "height": 50},
                        "elements": [
                            {
                                "page_idx": 1,
                                "bbox": [250, 250, 350, 350],
                                "text": "nested-page-1",
                            }
                        ],
                    }
                ],
            },
        ]
    }

    payload_page_sizes: dict[int, tuple[float, float]] = {}
    adapter._collect_page_payload(payload, out_pages=payload_page_sizes)

    content_items: list[dict[str, object]] = []
    adapter._collect_content_items(
        payload,
        page_idx=None,
        payload_page_sizes=payload_page_sizes,
        pdf_page_sizes={0: (600.0, 800.0), 1: (600.0, 800.0)},
        out_items=content_items,
        seen=set(),
    )

    by_text = {str(item.get("text")): item for item in content_items}
    assert by_text["page-0"]["bbox"] == [60.0, 80.0, 120.0, 160.0]
    assert by_text["nested-page-1"]["bbox"] == [150.0, 200.0, 210.0, 280.0]
