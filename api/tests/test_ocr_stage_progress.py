from __future__ import annotations

import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.worker_helpers.ocr_stage import (
    _format_ocr_progress_message,
    _format_parallel_ocr_progress_message,
)


def test_format_ocr_progress_message_includes_pdf_page_and_percent() -> None:
    message = _format_ocr_progress_message(
        ocr_page_processed=1,
        ocr_page_targets=13,
        pdf_page_index=2,
        source_page_count=20,
        overall_progress=36,
    )

    assert "OCR页 1/13" in message
    assert "PDF页 3/20" in message
    assert "OCR阶段 0%" in message
    assert "总进度 36%" in message


def test_format_parallel_ocr_progress_message_includes_running_and_latest_page() -> None:
    message = _format_parallel_ocr_progress_message(
        completed_pages=3,
        total_pages=13,
        running_pages=2,
        page_concurrency=3,
        latest_pdf_page_index=4,
        source_page_count=20,
        overall_progress=44,
    )

    assert "已完成 3/13 页" in message
    assert "运行中 2 页" in message
    assert "页并发 3" in message
    assert "最近 PDF页 5/20" in message
    assert "总进度 44%" in message
