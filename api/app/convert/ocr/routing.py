from __future__ import annotations

from dataclasses import dataclass

ROUTE_KIND_MACHINE_OCR = "machine_ocr"
ROUTE_KIND_REMOTE_DOC_PARSER = "remote_doc_parser"
ROUTE_KIND_REMOTE_PROMPT_OCR = "remote_prompt_ocr"
ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR = "local_layout_block_ocr"
ROUTE_KIND_HYBRID_AUTO = "hybrid_auto"
ROUTE_KIND_UNKNOWN = "unknown"

VALID_ROUTE_KINDS = {
    ROUTE_KIND_MACHINE_OCR,
    ROUTE_KIND_REMOTE_DOC_PARSER,
    ROUTE_KIND_REMOTE_PROMPT_OCR,
    ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    ROUTE_KIND_HYBRID_AUTO,
    ROUTE_KIND_UNKNOWN,
}


@dataclass(frozen=True)
class OcrRoutePlan:
    requested_provider: str
    runtime_provider: str
    route_kind: str
    is_paddle_vl_model: bool
    allow_main_ai_reuse: bool
    allow_text_refiner: bool
    allow_linebreak_refiner: bool
    auto_enable_linebreak: bool
    force_disable_linebreak: bool


def should_allow_main_ai_reuse(requested_ocr_provider: str) -> bool:
    return requested_ocr_provider not in {"aiocr", "paddle", "baidu"}


def normalize_ocr_route_kind(
    value: str | None,
    *,
    default: str = ROUTE_KIND_UNKNOWN,
) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    aliases = {
        "doc_parser": ROUTE_KIND_REMOTE_DOC_PARSER,
        "docparser": ROUTE_KIND_REMOTE_DOC_PARSER,
        "doc-parse": ROUTE_KIND_REMOTE_DOC_PARSER,
        "structured": ROUTE_KIND_REMOTE_DOC_PARSER,
        "structured_doc_parse": ROUTE_KIND_REMOTE_DOC_PARSER,
        "prompt": ROUTE_KIND_REMOTE_PROMPT_OCR,
        "prompt_ocr": ROUTE_KIND_REMOTE_PROMPT_OCR,
        "vision_prompt": ROUTE_KIND_REMOTE_PROMPT_OCR,
        "remote_ai": ROUTE_KIND_REMOTE_PROMPT_OCR,
        "layout_block": ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
        "block_ocr": ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
        "local_layout": ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
        "local_layout_block_ocr": ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
        "machine": ROUTE_KIND_MACHINE_OCR,
        "local_machine_ocr": ROUTE_KIND_MACHINE_OCR,
        "hybrid": ROUTE_KIND_HYBRID_AUTO,
        "auto": default,
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in VALID_ROUTE_KINDS else default


def build_ocr_route_plan(
    *,
    requested_ocr_provider: str,
    effective_ai_model: str | None,
    ai_chain_mode: str | None = None,
) -> OcrRoutePlan:
    is_paddle_vl_model = "paddleocr-vl" in (
        str(effective_ai_model or "").strip().lower()
    )
    normalized_ai_chain_mode = str(ai_chain_mode or "").strip().lower() or "direct"

    if requested_ocr_provider == "baidu":
        return OcrRoutePlan(
            requested_provider=requested_ocr_provider,
            runtime_provider="baidu",
            route_kind=ROUTE_KIND_MACHINE_OCR,
            is_paddle_vl_model=False,
            allow_main_ai_reuse=False,
            allow_text_refiner=False,
            allow_linebreak_refiner=False,
            auto_enable_linebreak=False,
            force_disable_linebreak=True,
        )

    if requested_ocr_provider == "paddle_local":
        return OcrRoutePlan(
            requested_provider=requested_ocr_provider,
            runtime_provider="paddle_local",
            route_kind=ROUTE_KIND_MACHINE_OCR,
            is_paddle_vl_model=False,
            allow_main_ai_reuse=True,
            allow_text_refiner=True,
            allow_linebreak_refiner=True,
            auto_enable_linebreak=False,
            force_disable_linebreak=False,
        )

    if requested_ocr_provider in {"local", "tesseract"}:
        return OcrRoutePlan(
            requested_provider=requested_ocr_provider,
            runtime_provider=requested_ocr_provider,
            route_kind=ROUTE_KIND_MACHINE_OCR,
            is_paddle_vl_model=False,
            allow_main_ai_reuse=True,
            allow_text_refiner=True,
            allow_linebreak_refiner=True,
            auto_enable_linebreak=False,
            force_disable_linebreak=False,
        )

    if requested_ocr_provider == "paddle":
        return OcrRoutePlan(
            requested_provider=requested_ocr_provider,
            runtime_provider="paddle",
            route_kind=ROUTE_KIND_REMOTE_DOC_PARSER,
            is_paddle_vl_model=True,
            allow_main_ai_reuse=False,
            allow_text_refiner=False,
            allow_linebreak_refiner=True,
            auto_enable_linebreak=True,
            force_disable_linebreak=False,
        )

    if requested_ocr_provider == "aiocr":
        route_kind = ROUTE_KIND_REMOTE_PROMPT_OCR
        if normalized_ai_chain_mode == "doc_parser":
            route_kind = ROUTE_KIND_REMOTE_DOC_PARSER
        elif normalized_ai_chain_mode == "layout_block":
            route_kind = ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR
        return OcrRoutePlan(
            requested_provider=requested_ocr_provider,
            runtime_provider="aiocr",
            route_kind=route_kind,
            is_paddle_vl_model=is_paddle_vl_model,
            allow_main_ai_reuse=False,
            # Explicit AI OCR already gets its text directly from the remote OCR
            # model. Keep the optional post-process focused on line splitting
            # instead of running a second "text refinement" pass on the same
            # chain.
            allow_text_refiner=False,
            allow_linebreak_refiner=True,
            auto_enable_linebreak=True,
            force_disable_linebreak=False,
        )

    if requested_ocr_provider == "auto":
        return OcrRoutePlan(
            requested_provider=requested_ocr_provider,
            runtime_provider="auto",
            route_kind=ROUTE_KIND_HYBRID_AUTO,
            is_paddle_vl_model=is_paddle_vl_model,
            allow_main_ai_reuse=True,
            allow_text_refiner=True,
            allow_linebreak_refiner=True,
            auto_enable_linebreak=is_paddle_vl_model,
            force_disable_linebreak=False,
        )

    return OcrRoutePlan(
        requested_provider=requested_ocr_provider,
        runtime_provider=requested_ocr_provider,
        route_kind=ROUTE_KIND_UNKNOWN,
        is_paddle_vl_model=is_paddle_vl_model,
        allow_main_ai_reuse=should_allow_main_ai_reuse(requested_ocr_provider),
        allow_text_refiner=False,
        allow_linebreak_refiner=False,
        auto_enable_linebreak=False,
        force_disable_linebreak=False,
    )


__all__ = [
    "ROUTE_KIND_HYBRID_AUTO",
    "ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR",
    "ROUTE_KIND_MACHINE_OCR",
    "ROUTE_KIND_REMOTE_DOC_PARSER",
    "ROUTE_KIND_REMOTE_PROMPT_OCR",
    "ROUTE_KIND_UNKNOWN",
    "VALID_ROUTE_KINDS",
    "OcrRoutePlan",
    "build_ocr_route_plan",
    "normalize_ocr_route_kind",
    "should_allow_main_ai_reuse",
]
