from __future__ import annotations

from dataclasses import dataclass

from .models.error import AppException, ErrorCode
from .utils.text import clean_str


VALID_PARSE_PROVIDERS = {"local", "mineru", "baidu_doc", "v2"}
VALID_OCR_PROVIDERS = {"auto", "aiocr", "baidu", "tesseract", "paddle", "paddle_local"}
VALID_LAYOUT_PROVIDERS = {"openai", "claude"}
VALID_BAIDU_DOC_PARSE_TYPES = {"general", "paddle_vl"}
VALID_OCR_AI_PROVIDERS = {
    "auto",
    "openai",
    "siliconflow",
    "deepseek",
    "ppio",
    "novita",
}
VALID_OCR_AI_CHAIN_MODES = {"direct", "doc_parser", "layout_block"}
VALID_OCR_AI_LAYOUT_MODELS = {"pp_doclayout_v3"}
VALID_OCR_GEOMETRY_MODES = {"auto", "local_tesseract", "direct_ai"}
VALID_TEXT_ERASE_MODES = {"smart", "fill"}
VALID_SCANNED_PAGE_MODES = {"segmented", "fullpage"}
VALID_PPT_GENERATION_MODES = {"standard", "fast", "turbo"}
OCR_GEOMETRY_ALIASES = {
    "auto",
    "direct",
    "direct_ai",
    "ai",
    "requested",
    "local",
    "local_tesseract",
    "tesseract",
    "hybrid_local",
}
SCANNED_PAGE_MODE_ALIASES = VALID_SCANNED_PAGE_MODES | {
    "chunk",
    "chunked",
    "split",
    "blocks",
    "page",
    "full",
    "full_page",
}
LAYOUT_PROVIDER_ALIASES = VALID_LAYOUT_PROVIDERS | {
    "domestic",
    "siliconflow",
    "openai-compatible",
    "openai_compatible",
    "anthropic",
}
BAIDU_DOC_PARSE_TYPE_ALIASES = VALID_BAIDU_DOC_PARSE_TYPES | {
    "default",
    "normal",
    "ordinary",
    "common",
    "paddle-vl",
    "paddleocr-vl",
    "paddleocr_vl",
    "vl",
}
OCR_AI_CHAIN_MODE_ALIASES = VALID_OCR_AI_CHAIN_MODES | {
    "prompt",
    "prompt_ocr",
    "model_direct",
    "direct_ai",
    "docparser",
    "doc-parse",
    "structured",
    "layout",
    "layout_block_ocr",
    "local_layout",
}
OCR_AI_LAYOUT_MODEL_ALIASES = VALID_OCR_AI_LAYOUT_MODELS | {
    "pp-doclayoutv3",
    "pp_doclayoutv3",
    "pp_doclayout",
    "pp-doclayout-v3",
}
PPT_GENERATION_MODE_ALIASES = VALID_PPT_GENERATION_MODES | {
    "default",
    "normal",
    "balanced",
    "quality",
    "speed",
    "speed_first",
    "speed-first",
    "fast_experimental",
    "experimental_fast",
    "ultra",
    "extreme",
    "turbo_fast",
    "turbo-fast",
}


@dataclass(frozen=True)
class NormalizedJobOptions:
    parse_provider: str
    provider: str
    baidu_doc_parse_type: str
    ocr_provider: str
    ocr_ai_provider: str
    ocr_ai_chain_mode: str
    ocr_ai_layout_model: str
    ocr_geometry_mode: str
    text_erase_mode: str
    scanned_page_mode: str
    ppt_generation_mode: str


def _unwrap_form_default(value):
    if hasattr(value, "default"):
        return getattr(value, "default")
    return value


def normalize_parse_provider(value: str | None) -> str:
    return (clean_str(_unwrap_form_default(value)) or "local").lower()


def normalize_requested_ocr_provider(value: str | None) -> str:
    provider_id = (clean_str(_unwrap_form_default(value)) or "auto").lower()
    if provider_id in {"remote", "ai"}:
        return "aiocr"
    if provider_id in {"paddle-local", "local_paddle"}:
        return "paddle_local"
    return provider_id


def normalize_layout_provider(value: str | None) -> str:
    provider_id = (clean_str(_unwrap_form_default(value)) or "openai").lower()
    if provider_id in {"domestic", "siliconflow", "openai-compatible", "openai_compatible"}:
        return "openai"
    if provider_id == "anthropic":
        return "claude"
    return provider_id


def normalize_baidu_doc_parse_type(value: str | None) -> str:
    parse_type = (clean_str(_unwrap_form_default(value)) or "paddle_vl").lower()
    if parse_type in {"default", "normal", "ordinary", "common"}:
        return "general"
    if parse_type in {"paddle-vl", "paddleocr-vl", "paddleocr_vl", "vl"}:
        return "paddle_vl"
    return parse_type if parse_type in VALID_BAIDU_DOC_PARSE_TYPES else "paddle_vl"


def normalize_ai_ocr_provider(value: str | None) -> str:
    provider_id = (clean_str(_unwrap_form_default(value)) or "auto").lower()
    if provider_id in {"openai_compatible", "openai-compatible"}:
        return "openai"
    return provider_id


def normalize_ai_ocr_chain_mode(value: str | None) -> str:
    mode = (clean_str(_unwrap_form_default(value)) or "direct").lower()
    if mode in {"prompt", "prompt_ocr", "model_direct", "direct_ai"}:
        return "direct"
    if mode in {"docparser", "doc-parse", "structured"}:
        return "doc_parser"
    if mode in {"layout", "layout_block_ocr", "local_layout"}:
        return "layout_block"
    return mode if mode in VALID_OCR_AI_CHAIN_MODES else "direct"


def normalize_ai_ocr_layout_model(value: str | None) -> str:
    model = (clean_str(_unwrap_form_default(value)) or "pp_doclayout_v3").lower()
    if model in {"pp-doclayoutv3", "pp_doclayoutv3", "pp_doclayout", "pp-doclayout-v3"}:
        return "pp_doclayout_v3"
    return model if model in VALID_OCR_AI_LAYOUT_MODELS else "pp_doclayout_v3"


def normalize_ocr_geometry_mode(value: str | None) -> str:
    mode = (clean_str(_unwrap_form_default(value)) or "auto").lower()
    if mode in {"direct", "direct_ai", "ai", "requested"}:
        return "direct_ai"
    if mode in {"local", "local_tesseract", "tesseract", "hybrid_local"}:
        return "local_tesseract"
    return mode if mode in VALID_OCR_GEOMETRY_MODES else "auto"


def normalize_text_erase_mode(value: str | None) -> str:
    mode = (clean_str(_unwrap_form_default(value)) or "fill").lower()
    return mode if mode in VALID_TEXT_ERASE_MODES else "fill"


def normalize_scanned_page_mode(value: str | None) -> str:
    mode = (clean_str(_unwrap_form_default(value)) or "segmented").lower()
    if mode in {"chunk", "chunked", "split", "blocks"}:
        return "segmented"
    if mode in {"page", "full", "full_page"}:
        return "fullpage"
    return mode if mode in VALID_SCANNED_PAGE_MODES else "segmented"


def normalize_ppt_generation_mode(value: str | None) -> str:
    mode = (clean_str(_unwrap_form_default(value)) or "standard").lower()
    if mode in {"default", "normal", "balanced", "quality"}:
        return "standard"
    if mode in {"speed", "speed_first", "speed-first", "fast_experimental", "experimental_fast"}:
        return "fast"
    if mode in {"ultra", "extreme", "turbo_fast", "turbo-fast"}:
        return "turbo"
    return mode if mode in VALID_PPT_GENERATION_MODES else "standard"


def validate_page_range(*, page_start: int | None, page_end: int | None) -> None:
    page_start = _unwrap_form_default(page_start)
    page_end = _unwrap_form_default(page_end)
    if (page_start is None) != (page_end is None):
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="page_start and page_end must be provided together",
            status_code=400,
        )
    if page_start is not None and page_start <= 0:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="page_start must be >= 1",
            status_code=400,
        )
    if page_end is not None and page_end <= 0:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="page_end must be >= 1",
            status_code=400,
        )
    if page_start is not None and page_end is not None and page_start > page_end:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="page_start cannot be greater than page_end",
            status_code=400,
        )


def validate_and_normalize_job_options(
    *,
    parse_provider: str | None,
    mineru_api_token: str | None,
    provider: str | None = None,
    api_key: str | None = None,
    baidu_doc_parse_type: str | None = None,
    ocr_provider: str | None,
    ocr_ai_provider: str | None,
    ocr_ai_api_key: str | None,
    ocr_ai_model: str | None,
    ocr_ai_chain_mode: str | None,
    ocr_ai_layout_model: str | None,
    ocr_baidu_app_id: str | None = None,
    ocr_baidu_api_key: str | None = None,
    ocr_baidu_secret_key: str | None = None,
    ocr_geometry_mode: str | None,
    text_erase_mode: str | None,
    scanned_page_mode: str | None,
    ppt_generation_mode: str | None,
    page_start: int | None,
    page_end: int | None,
) -> NormalizedJobOptions:
    api_key = _unwrap_form_default(api_key)
    provider = _unwrap_form_default(provider)
    baidu_doc_parse_type = _unwrap_form_default(baidu_doc_parse_type)
    _ = api_key
    mineru_api_token = _unwrap_form_default(mineru_api_token)
    ocr_ai_api_key = _unwrap_form_default(ocr_ai_api_key)
    ocr_ai_model = _unwrap_form_default(ocr_ai_model)
    ocr_ai_chain_mode = _unwrap_form_default(ocr_ai_chain_mode)
    ocr_ai_layout_model = _unwrap_form_default(ocr_ai_layout_model)
    ocr_baidu_app_id = _unwrap_form_default(ocr_baidu_app_id)
    ocr_baidu_api_key = _unwrap_form_default(ocr_baidu_api_key)
    ocr_baidu_secret_key = _unwrap_form_default(ocr_baidu_secret_key)
    ocr_geometry_mode = _unwrap_form_default(ocr_geometry_mode)
    text_erase_mode = _unwrap_form_default(text_erase_mode)
    scanned_page_mode = _unwrap_form_default(scanned_page_mode)
    ppt_generation_mode = _unwrap_form_default(ppt_generation_mode)
    page_start = _unwrap_form_default(page_start)
    page_end = _unwrap_form_default(page_end)

    normalized_parse_provider = normalize_parse_provider(parse_provider)
    if normalized_parse_provider not in VALID_PARSE_PROVIDERS:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported parse provider",
            details={"parse_provider": parse_provider},
            status_code=400,
        )
    normalized_provider = normalize_layout_provider(provider)
    raw_provider = (clean_str(provider) or "").lower()
    if raw_provider and raw_provider not in LAYOUT_PROVIDER_ALIASES:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported layout assist provider",
            details={"provider": provider},
            status_code=400,
        )

    validate_page_range(page_start=page_start, page_end=page_end)

    if normalized_parse_provider == "mineru" and not (mineru_api_token or "").strip():
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="mineru_api_token is required when parse_provider=mineru",
            status_code=400,
        )

    normalized_baidu_doc_parse_type = normalize_baidu_doc_parse_type(baidu_doc_parse_type)
    raw_baidu_doc_parse_type = (clean_str(baidu_doc_parse_type) or "").lower()
    if (
        normalized_parse_provider == "baidu_doc"
        and raw_baidu_doc_parse_type
        and raw_baidu_doc_parse_type not in BAIDU_DOC_PARSE_TYPE_ALIASES
    ):
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported Baidu document parser type",
            details={"baidu_doc_parse_type": baidu_doc_parse_type},
            status_code=400,
        )

    normalized_ocr_provider = normalize_requested_ocr_provider(ocr_provider)
    if normalized_ocr_provider not in VALID_OCR_PROVIDERS:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported OCR provider",
            details={"ocr_provider": ocr_provider},
            status_code=400,
        )
    # Legacy compatibility: some older callers redundantly paired the new
    # structured Baidu parser with `ocr_provider=baidu`. The canonical Baidu
    # document chain no longer consumes OCR-provider routing, so normalize the
    # redundant alias away before later validation.
    if normalized_parse_provider == "baidu_doc" and normalized_ocr_provider == "baidu":
        normalized_ocr_provider = "auto"

    normalized_ocr_ai_provider = normalize_ai_ocr_provider(ocr_ai_provider)
    if normalized_ocr_ai_provider not in VALID_OCR_AI_PROVIDERS:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported OCR AI provider",
            details={"ocr_ai_provider": ocr_ai_provider},
            status_code=400,
        )

    normalized_ocr_ai_chain_mode = normalize_ai_ocr_chain_mode(ocr_ai_chain_mode)
    raw_ocr_ai_chain_mode = (clean_str(ocr_ai_chain_mode) or "").lower()
    if raw_ocr_ai_chain_mode and raw_ocr_ai_chain_mode not in OCR_AI_CHAIN_MODE_ALIASES:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported OCR AI chain mode",
            details={"ocr_ai_chain_mode": ocr_ai_chain_mode},
            status_code=400,
        )

    normalized_ocr_ai_layout_model = normalize_ai_ocr_layout_model(ocr_ai_layout_model)
    raw_ocr_ai_layout_model = (clean_str(ocr_ai_layout_model) or "").lower()
    if (
        raw_ocr_ai_layout_model
        and raw_ocr_ai_layout_model not in OCR_AI_LAYOUT_MODEL_ALIASES
    ):
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported OCR AI layout model",
            details={"ocr_ai_layout_model": ocr_ai_layout_model},
            status_code=400,
        )

    normalized_geometry_mode = normalize_ocr_geometry_mode(ocr_geometry_mode)
    raw_geometry_mode = (clean_str(ocr_geometry_mode) or "").lower()
    if raw_geometry_mode and raw_geometry_mode not in OCR_GEOMETRY_ALIASES:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported OCR geometry mode",
            details={"ocr_geometry_mode": ocr_geometry_mode},
            status_code=400,
        )

    raw_text_erase_mode = (clean_str(text_erase_mode) or "").lower()
    if raw_text_erase_mode and raw_text_erase_mode not in VALID_TEXT_ERASE_MODES:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported text erase mode",
            details={"text_erase_mode": text_erase_mode},
            status_code=400,
        )

    raw_scanned_page_mode = (clean_str(scanned_page_mode) or "").lower()
    if raw_scanned_page_mode and raw_scanned_page_mode not in SCANNED_PAGE_MODE_ALIASES:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported scanned page mode",
            details={"scanned_page_mode": scanned_page_mode},
            status_code=400,
        )

    normalized_ppt_generation_mode = normalize_ppt_generation_mode(ppt_generation_mode)
    raw_ppt_generation_mode = (clean_str(ppt_generation_mode) or "").lower()
    if raw_ppt_generation_mode and raw_ppt_generation_mode not in PPT_GENERATION_MODE_ALIASES:
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Unsupported PPT generation mode",
            details={"ppt_generation_mode": ppt_generation_mode},
            status_code=400,
        )

    if normalized_geometry_mode != "auto" and normalized_ocr_provider != "aiocr":
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="ocr_geometry_mode is only supported when ocr_provider=aiocr",
            details={
                "ocr_provider": normalized_ocr_provider,
                "ocr_geometry_mode": normalized_geometry_mode,
            },
            status_code=400,
        )

    if normalized_parse_provider == "mineru" and normalized_ocr_provider != "auto":
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="MinerU parse does not use OCR provider selection",
            details={
                "parse_provider": normalized_parse_provider,
                "ocr_provider": normalized_ocr_provider,
            },
            status_code=400,
        )

    if normalized_parse_provider == "baidu_doc" and normalized_ocr_provider != "auto":
        raise AppException(
            code=ErrorCode.VALIDATION_ERROR,
            message="Baidu document parser does not use OCR provider selection",
            details={
                "parse_provider": normalized_parse_provider,
                "ocr_provider": normalized_ocr_provider,
            },
            status_code=400,
        )

    if normalized_parse_provider == "local" and normalized_ocr_provider in {"aiocr", "paddle"}:
        if not (ocr_ai_api_key or "").strip():
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="ocr_ai_api_key is required for explicit AI OCR providers",
                details={"ocr_provider": normalized_ocr_provider},
                status_code=400,
            )

        if (
            normalized_ocr_provider == "aiocr"
            and normalized_ocr_ai_chain_mode == "doc_parser"
            and "paddleocr-vl" not in str(ocr_ai_model or "").strip().lower()
        ):
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="doc_parser chain requires a PaddleOCR-VL model",
                details={
                    "ocr_provider": normalized_ocr_provider,
                    "ocr_ai_chain_mode": normalized_ocr_ai_chain_mode,
                    "ocr_ai_model": ocr_ai_model,
                },
                status_code=400,
            )
        if (
            normalized_ocr_provider == "aiocr"
            and normalized_ocr_ai_chain_mode == "direct"
            and "paddleocr-vl" in str(ocr_ai_model or "").strip().lower()
        ):
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="direct chain does not support PaddleOCR-VL models",
                details={
                    "ocr_provider": normalized_ocr_provider,
                    "ocr_ai_chain_mode": normalized_ocr_ai_chain_mode,
                    "ocr_ai_model": ocr_ai_model,
                },
                status_code=400,
            )
        if not (ocr_ai_model or "").strip():
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="ocr_ai_model is required for explicit AI OCR providers",
                details={"ocr_provider": normalized_ocr_provider},
                status_code=400,
            )

    if normalized_ocr_provider == "baidu":
        has_baidu_credentials = all(
            bool((value or "").strip())
            for value in (ocr_baidu_api_key, ocr_baidu_secret_key)
        )
        if not has_baidu_credentials:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="Baidu OCR requires api_key / secret_key",
                details={"ocr_provider": normalized_ocr_provider},
                status_code=400,
            )

    if normalized_parse_provider == "baidu_doc":
        has_baidu_doc_credentials = all(
            bool((value or "").strip())
            for value in (ocr_baidu_api_key, ocr_baidu_secret_key)
        )
        if not has_baidu_doc_credentials:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="Baidu document parser requires api_key / secret_key",
                details={"parse_provider": normalized_parse_provider},
                status_code=400,
            )

    return NormalizedJobOptions(
        parse_provider=normalized_parse_provider,
        provider=normalized_provider,
        baidu_doc_parse_type=normalized_baidu_doc_parse_type,
        ocr_provider=normalized_ocr_provider,
        ocr_ai_provider=normalized_ocr_ai_provider,
        ocr_ai_chain_mode=normalized_ocr_ai_chain_mode,
        ocr_ai_layout_model=normalized_ocr_ai_layout_model,
        ocr_geometry_mode=normalized_geometry_mode,
        text_erase_mode=normalize_text_erase_mode(text_erase_mode),
        scanned_page_mode=normalize_scanned_page_mode(scanned_page_mode),
        ppt_generation_mode=normalized_ppt_generation_mode,
    )
