from __future__ import annotations

from app.convert.ocr import _is_multiline_candidate_for_linebreak_assist


def test_linebreak_candidate_flags_paragraph_like_bbox() -> None:
    # Simulate a scanned slide rendered around 200 DPI (similar to the project
    # benchmark PDF). Typical body line heights are ~170px, while 2-line blocks
    # are ~260px.
    assert _is_multiline_candidate_for_linebreak_assist(
        # Use a longer body-like bullet so it won't be mistaken as a short banner.
        text="- 工具（Tools）：连接外部世界的能力，如调用搜索引擎、代码解释器、日历、邮件、数据库等…",
        bbox=(100.0, 200.0, 3300.0, 460.0),  # h=260px
        image_width=3823,
        image_height=2134,
        median_line_height=170.0,
    )


def test_linebreak_candidate_ignores_small_single_line_bbox() -> None:
    assert not _is_multiline_candidate_for_linebreak_assist(
        text="欢迎来到智能体时代：超越聊天机器人",
        bbox=(200.0, 120.0, 2400.0, 230.0),  # h=110px
        image_width=3823,
        image_height=2134,
        median_line_height=170.0,
    )


def test_linebreak_candidate_ignores_wide_banner_titles() -> None:
    # Wide titles with small height/width ratio should not be over-split.
    assert not _is_multiline_candidate_for_linebreak_assist(
        text="大语言模型 (LLM)",
        # h=270px (>=1.45*median), but still banner-like by aspect ratio.
        bbox=(100.0, 80.0, 3500.0, 350.0),
        image_width=3823,
        image_height=2134,
        median_line_height=170.0,
    )
