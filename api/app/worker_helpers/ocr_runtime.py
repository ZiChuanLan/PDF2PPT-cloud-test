from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..job_options import (
    normalize_ai_ocr_provider,
    normalize_ocr_geometry_mode,
    normalize_requested_ocr_provider,
)
from ..convert.ocr import AiOcrTextRefiner, create_ocr_manager
from ..logging_config import get_logger
from ..models.error import AppException, ErrorCode
from ..utils.text import clean_str


logger = get_logger(__name__)


@dataclass(frozen=True)
class OcrRuntimeSetup:
    requested_ocr_provider: str
    ocr_manager: Any | None
    text_refiner: AiOcrTextRefiner | None
    linebreak_refiner: AiOcrTextRefiner | None
    effective_ocr_provider: str
    effective_ocr_ai_provider: str
    effective_ocr_ai_api_key: str | None
    effective_ocr_ai_base_url: str | None
    effective_ocr_ai_model: str | None
    effective_paddle_doc_max_side_px: int | None
    effective_tesseract_language: str
    effective_tesseract_min_conf: float | None
    strict_ocr_mode: bool
    linebreak_enabled: bool
    auto_linebreak_enabled: bool
    linebreak_mode: str
    linebreak_unavailable_reason: str | None
    ocr_ai_config_source: str
    ocr_ai_api_key_source: str
    ocr_ai_base_url_source: str
    ocr_ai_model_source: str
    ocr_geometry_provider: str | None = None
    ocr_geometry_strategy: str | None = None
    ocr_geometry_mode_requested: str | None = None
    ocr_geometry_mode_effective: str | None = None
    setup_warning: str | None = None
    setup_notes: tuple[str, ...] = ()

def setup_ocr_runtime(
    *,
    provider: str | None,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    ocr_provider: str | None,
    ocr_baidu_app_id: str | None,
    ocr_baidu_api_key: str | None,
    ocr_baidu_secret_key: str | None,
    ocr_tesseract_min_confidence: float | None,
    ocr_tesseract_language: str | None,
    ocr_ai_api_key: str | None,
    ocr_ai_provider: str | None,
    ocr_ai_base_url: str | None,
    ocr_ai_model: str | None,
    ocr_paddle_vl_docparser_max_side_px: int | None = None,
    ocr_geometry_mode: str | None = None,
    ocr_ai_linebreak_assist: bool | None = None,
    ocr_strict_mode: bool | None = True,
) -> OcrRuntimeSetup:
    requested_ocr_provider = normalize_requested_ocr_provider(ocr_provider)
    allow_main_ai_reuse = requested_ocr_provider not in {"aiocr", "paddle", "baidu"}

    # If the user didn't configure separate AI OCR credentials, reuse
    # the layout-assist OpenAI-compatible settings (when available).
    effective_ocr_ai_api_key = clean_str(ocr_ai_api_key)
    effective_ocr_ai_provider = normalize_ai_ocr_provider(ocr_ai_provider)
    effective_ocr_ai_base_url = clean_str(ocr_ai_base_url)
    effective_ocr_ai_model = clean_str(ocr_ai_model)
    ocr_ai_api_key_source = "ocr" if effective_ocr_ai_api_key else "none"
    ocr_ai_base_url_source = "ocr" if effective_ocr_ai_base_url else "none"
    ocr_ai_model_source = "ocr" if effective_ocr_ai_model else "none"
    ocr_ai_config_source = "dedicated" if effective_ocr_ai_api_key else "none"
    setup_notes: list[str] = []
    provider_id = (clean_str(provider) or "openai").lower()
    if (
        allow_main_ai_reuse
        and not effective_ocr_ai_api_key
        and provider_id != "claude"
        and clean_str(api_key)
    ):
        effective_ocr_ai_api_key = clean_str(api_key)
        effective_ocr_ai_base_url = clean_str(base_url)
        effective_ocr_ai_model = clean_str(model)
        ocr_ai_config_source = "main_fallback"
        ocr_ai_api_key_source = "main"
        ocr_ai_base_url_source = "main" if effective_ocr_ai_base_url else "none"
        ocr_ai_model_source = "main" if effective_ocr_ai_model else "none"
        setup_notes.append("ocr_ai_config_reused_from_main")
    elif not allow_main_ai_reuse and clean_str(api_key):
        setup_notes.append("ocr_ai_config_main_reuse_blocked_for_explicit_provider")

    effective_ocr_provider = requested_ocr_provider
    strict_ocr_mode = bool(True if ocr_strict_mode is None else ocr_strict_mode)

    effective_tesseract_language = clean_str(ocr_tesseract_language) or "chi_sim+eng"
    effective_tesseract_min_conf: float | None = None
    if ocr_tesseract_min_confidence is not None:
        try:
            effective_tesseract_min_conf = float(ocr_tesseract_min_confidence)
        except Exception:
            effective_tesseract_min_conf = None

    # High recall helps text erase completeness; noise can be cleaned
    # downstream. Keep conservative auto-overrides in machine/hybrid
    # modes so users don't accidentally run with eng-only + high conf.
    if effective_ocr_provider in {"auto", "tesseract", "local"} and not strict_ocr_mode:
        if effective_tesseract_language.strip().lower() == "eng":
            effective_tesseract_language = "chi_sim+eng"
        if effective_tesseract_min_conf is None:
            effective_tesseract_min_conf = 35.0
        else:
            effective_tesseract_min_conf = min(float(effective_tesseract_min_conf), 35.0)
    effective_paddle_doc_max_side_px: int | None = None
    if ocr_paddle_vl_docparser_max_side_px is not None:
        try:
            effective_paddle_doc_max_side_px = int(ocr_paddle_vl_docparser_max_side_px)
        except Exception:
            effective_paddle_doc_max_side_px = None
        if effective_paddle_doc_max_side_px is not None:
            effective_paddle_doc_max_side_px = max(
                0, min(6000, int(effective_paddle_doc_max_side_px))
            )
            setup_notes.append(
                "ocr_paddle_vl_docparser_max_side_px="
                f"{effective_paddle_doc_max_side_px}"
            )

    text_refiner: AiOcrTextRefiner | None = None
    linebreak_refiner: AiOcrTextRefiner | None = None
    linebreak_enabled = False
    auto_linebreak_enabled = False
    linebreak_mode = "off"
    linebreak_unavailable_reason: str | None = None
    requested_geometry_mode = normalize_ocr_geometry_mode(ocr_geometry_mode)
    runtime_ocr_provider = effective_ocr_provider
    geometry_strategy = "direct"
    geometry_mode_effective = "n/a"

    if requested_ocr_provider == "aiocr":
        if not effective_ocr_ai_api_key:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="AI OCR requires api_key",
                details={"ocr_provider": requested_ocr_provider},
            )
        if not effective_ocr_ai_model:
            raise AppException(
                code=ErrorCode.VALIDATION_ERROR,
                message="AI OCR requires model",
                details={"ocr_provider": requested_ocr_provider},
            )

    # Explicit AI OCR now always stays on the pure AI OCR path. We keep
    # `ocr_geometry_mode` only for backward-compatible request parsing/debug,
    # but no longer switch runtime geometry to local Tesseract.
    if requested_ocr_provider == "aiocr":
        geometry_mode_effective = "direct_ai"
        if requested_geometry_mode == "local_tesseract":
            setup_notes.append(
                "ocr_geometry_local_tesseract_ignored_for_explicit_aiocr:"
                f" model={effective_ocr_ai_model or '<unset>'}"
            )
        elif requested_geometry_mode == "direct_ai":
            geometry_mode_effective = "direct_ai"
            geometry_strategy = "forced_direct_ai_geometry"
            setup_notes.append(
                "ocr_geometry_forced_direct_ai:"
                f" model={effective_ocr_ai_model or '<unset>'}"
            )
        else:
            geometry_mode_effective = "direct_ai"
            setup_notes.append(
                "ocr_geometry_auto_direct_ai:"
                f" model={effective_ocr_ai_model or '<unset>'}"
            )

    try:
        ocr_manager = create_ocr_manager(
            provider=runtime_ocr_provider,
            ai_provider=effective_ocr_ai_provider,
            ai_api_key=effective_ocr_ai_api_key,
            ai_base_url=effective_ocr_ai_base_url,
            ai_model=effective_ocr_ai_model,
            paddle_doc_max_side_px=effective_paddle_doc_max_side_px,
            baidu_app_id=ocr_baidu_app_id,
            baidu_api_key=ocr_baidu_api_key,
            baidu_secret_key=ocr_baidu_secret_key,
            tesseract_min_confidence=effective_tesseract_min_conf,
            tesseract_language=effective_tesseract_language,
            strict_no_fallback=strict_ocr_mode,
            allow_paddle_model_downgrade=not strict_ocr_mode,
        )

        # Optional OCR post-process: refine OCR texts or split coarse boxes
        # into line-level boxes after the primary OCR provider is selected.
        try:
            provider_choice = requested_ocr_provider
            is_paddle_vl_model = "paddleocr-vl" in (
                str(effective_ocr_ai_model or "").strip().lower()
            )
            text_refiner_allowed = provider_choice in {
                "auto",
                "aiocr",
                "tesseract",
                "local",
                "paddle_local",
            }
            linebreak_refiner_allowed = provider_choice in {
                "auto",
                "aiocr",
                "tesseract",
                "local",
                "paddle",
                "paddle_local",
            }
            linebreak_requested = ocr_ai_linebreak_assist
            linebreak_enabled = bool(linebreak_requested)
            auto_linebreak_enabled = False

            if provider_choice == "baidu":
                if linebreak_requested is not None:
                    setup_notes.append(
                        "ocr_ai_linebreak_assist_ignored_for_explicit_baidu"
                    )
                linebreak_enabled = False

            # PaddleOCR-VL doc_parser often returns paragraph-like bboxes.
            # Auto-enable line-break assist (when user didn't specify)
            # so downstream PPT rendering doesn't have to guess wraps.
            if linebreak_requested is None and (
                provider_choice == "paddle"
                or (provider_choice in {"aiocr", "auto"} and is_paddle_vl_model)
            ):
                linebreak_enabled = True
                auto_linebreak_enabled = True
            # For explicit AI OCR with non-Paddle models (for example GPT/Gemini/
            # Qwen-VL style endpoints), line-level geometry can still be coarse on
            # dense scanned pages. Auto-enable visual line-break assist so the
            # model can split paragraph-like OCR boxes into line-level boxes.
            elif (
                linebreak_requested is None
                and provider_choice == "aiocr"
                and (not is_paddle_vl_model)
                and bool(effective_ocr_ai_api_key)
            ):
                linebreak_enabled = True
                auto_linebreak_enabled = True

            needs_refiner = text_refiner_allowed or (
                linebreak_enabled and linebreak_refiner_allowed
            )
            if needs_refiner and effective_ocr_ai_api_key:
                shared_refiner = AiOcrTextRefiner(
                    api_key=effective_ocr_ai_api_key,
                    provider=effective_ocr_ai_provider,
                    base_url=effective_ocr_ai_base_url,
                    model=effective_ocr_ai_model,
                )

                # If the user enabled line-break assist, we can also reuse the
                # same vision model to refine OCR texts (keeping machine bboxes)
                # and improve transcription quality in non-AI providers.
                text_refine_enabled = bool(text_refiner_allowed) or bool(linebreak_enabled)
                if text_refine_enabled and not is_paddle_vl_model:
                    text_refiner = shared_refiner

                if linebreak_enabled and linebreak_refiner_allowed:
                    linebreak_refiner = shared_refiner
            elif linebreak_enabled and not effective_ocr_ai_api_key:
                linebreak_unavailable_reason = "missing_ai_ocr_key"
        except Exception as e:
            logger.warning("AI OCR text refiner setup failed: %s", e)
            text_refiner = None
            linebreak_refiner = None
            if linebreak_enabled and not linebreak_unavailable_reason:
                linebreak_unavailable_reason = f"refiner_setup_failed:{e!s}"

        if linebreak_enabled:
            linebreak_mode = "ai_refiner" if linebreak_refiner is not None else "heuristic"

        return OcrRuntimeSetup(
            requested_ocr_provider=requested_ocr_provider,
            ocr_manager=ocr_manager,
            text_refiner=text_refiner,
            linebreak_refiner=linebreak_refiner,
            effective_ocr_provider=runtime_ocr_provider,
            effective_ocr_ai_provider=effective_ocr_ai_provider,
            effective_ocr_ai_api_key=effective_ocr_ai_api_key,
            effective_ocr_ai_base_url=effective_ocr_ai_base_url,
            effective_ocr_ai_model=effective_ocr_ai_model,
            effective_paddle_doc_max_side_px=effective_paddle_doc_max_side_px,
            effective_tesseract_language=effective_tesseract_language,
            effective_tesseract_min_conf=effective_tesseract_min_conf,
            strict_ocr_mode=strict_ocr_mode,
            linebreak_enabled=linebreak_enabled,
            auto_linebreak_enabled=auto_linebreak_enabled,
            linebreak_mode=linebreak_mode,
            linebreak_unavailable_reason=linebreak_unavailable_reason,
            ocr_ai_config_source=ocr_ai_config_source,
            ocr_ai_api_key_source=ocr_ai_api_key_source,
            ocr_ai_base_url_source=ocr_ai_base_url_source,
            ocr_ai_model_source=ocr_ai_model_source,
            ocr_geometry_provider=runtime_ocr_provider,
            ocr_geometry_strategy=geometry_strategy,
            ocr_geometry_mode_requested=requested_geometry_mode,
            ocr_geometry_mode_effective=geometry_mode_effective,
            setup_warning=None,
            setup_notes=tuple(setup_notes),
        )
    except Exception as e:
        # In strict mode, fail loudly. In non-strict mode we degrade
        # gracefully to image-only output if OCR cannot be set up.
        if strict_ocr_mode:
            raise AppException(
                code=ErrorCode.OCR_FAILED,
                message=f"OCR setup failed: {e!s}",
                details={"error": str(e)},
            ) from e

        return OcrRuntimeSetup(
            requested_ocr_provider=requested_ocr_provider,
            ocr_manager=None,
            text_refiner=None,
            linebreak_refiner=None,
            effective_ocr_provider=runtime_ocr_provider,
            effective_ocr_ai_provider=effective_ocr_ai_provider,
            effective_ocr_ai_api_key=effective_ocr_ai_api_key,
            effective_ocr_ai_base_url=effective_ocr_ai_base_url,
            effective_ocr_ai_model=effective_ocr_ai_model,
            effective_paddle_doc_max_side_px=effective_paddle_doc_max_side_px,
            effective_tesseract_language=effective_tesseract_language,
            effective_tesseract_min_conf=effective_tesseract_min_conf,
            strict_ocr_mode=strict_ocr_mode,
            linebreak_enabled=linebreak_enabled,
            auto_linebreak_enabled=auto_linebreak_enabled,
            linebreak_mode=linebreak_mode,
            linebreak_unavailable_reason=linebreak_unavailable_reason,
            ocr_ai_config_source=ocr_ai_config_source,
            ocr_ai_api_key_source=ocr_ai_api_key_source,
            ocr_ai_base_url_source=ocr_ai_base_url_source,
            ocr_ai_model_source=ocr_ai_model_source,
            ocr_geometry_provider=runtime_ocr_provider,
            ocr_geometry_strategy=geometry_strategy,
            ocr_geometry_mode_requested=requested_geometry_mode,
            ocr_geometry_mode_effective=geometry_mode_effective,
            setup_warning=f"{e!s}",
            setup_notes=tuple(setup_notes),
        )


def build_ocr_debug_payload(
    *,
    provider_requested: str | None,
    ocr_render_dpi: int,
    scanned_render_dpi: int,
    ocr_ai_linebreak_assist: bool | None,
    setup: OcrRuntimeSetup,
) -> dict[str, Any]:
    return {
        "provider_requested": provider_requested or "auto",
        "provider_requested_normalized": setup.requested_ocr_provider,
        "provider_selected": setup.requested_ocr_provider,
        "provider_runtime": setup.effective_ocr_provider,
        # Backward-compatible alias. Historically this field actually reflected
        # the runtime OCR/geometry provider, which was easy to misread as the
        # user-selected OCR provider.
        "provider_effective": setup.effective_ocr_provider,
        "ocr_geometry": {
            "provider": setup.ocr_geometry_provider or setup.effective_ocr_provider,
            "strategy": setup.ocr_geometry_strategy or "direct",
            "mode_requested": setup.ocr_geometry_mode_requested or "auto",
            "mode_effective": setup.ocr_geometry_mode_effective or "n/a",
        },
        "tesseract_language": setup.effective_tesseract_language,
        "tesseract_min_confidence": setup.effective_tesseract_min_conf,
        "ocr_render_dpi": int(ocr_render_dpi),
        "scanned_render_dpi": int(scanned_render_dpi),
        "ai_ocr": {
            "provider": setup.effective_ocr_ai_provider,
            "base_url": setup.effective_ocr_ai_base_url,
            "model": setup.effective_ocr_ai_model,
            "paddle_doc_max_side_px": setup.effective_paddle_doc_max_side_px,
            "config_source": setup.ocr_ai_config_source,
            "sources": {
                "api_key": setup.ocr_ai_api_key_source,
                "base_url": setup.ocr_ai_base_url_source,
                "model": setup.ocr_ai_model_source,
            },
        },
        "ai_text_refiner": {
            "enabled": bool(setup.text_refiner),
            "provider": setup.effective_ocr_ai_provider,
            "base_url": setup.effective_ocr_ai_base_url,
            "model": setup.effective_ocr_ai_model,
        },
        "ai_linebreak_refiner": {
            "enabled": bool(setup.linebreak_refiner),
            # Keep raw request so QA can tell None vs explicit false.
            "requested": ocr_ai_linebreak_assist,
            "auto_enabled": bool(setup.auto_linebreak_enabled),
            "effective": bool(setup.linebreak_enabled),
            "mode": setup.linebreak_mode,
            "unavailable_reason": setup.linebreak_unavailable_reason,
            "provider": setup.effective_ocr_ai_provider,
            "base_url": setup.effective_ocr_ai_base_url,
            "model": setup.effective_ocr_ai_model,
        },
        "runtime_decisions": {
            "requested_provider": setup.requested_ocr_provider,
            "selected_provider": setup.requested_ocr_provider,
            "runtime_provider": setup.effective_ocr_provider,
            "effective_provider": setup.effective_ocr_provider,
            "geometry_provider": setup.ocr_geometry_provider or setup.effective_ocr_provider,
            "geometry_strategy": setup.ocr_geometry_strategy or "direct",
            "ocr_ai_config_source": setup.ocr_ai_config_source,
            "notes": list(setup.setup_notes),
        },
        "setup_notes": list(setup.setup_notes),
        "ocr_strict_mode": bool(setup.strict_ocr_mode),
    }
