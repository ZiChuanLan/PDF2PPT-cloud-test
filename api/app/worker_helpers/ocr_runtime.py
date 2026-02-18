from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ..convert.ocr import AiOcrTextRefiner, create_ocr_manager
from ..logging_config import get_logger
from ..models.error import AppException, ErrorCode
from ..utils.text import clean_str


logger = get_logger(__name__)


@dataclass(frozen=True)
class OcrRuntimeSetup:
    ocr_manager: Any | None
    text_refiner: AiOcrTextRefiner | None
    linebreak_refiner: AiOcrTextRefiner | None
    effective_ocr_provider: str
    effective_ocr_ai_provider: str
    effective_ocr_ai_api_key: str | None
    effective_ocr_ai_base_url: str | None
    effective_ocr_ai_model: str | None
    effective_tesseract_language: str
    effective_tesseract_min_conf: float | None
    strict_ocr_mode: bool
    linebreak_enabled: bool
    auto_linebreak_enabled: bool
    ocr_geometry_provider: str | None = None
    ocr_geometry_strategy: str | None = None
    ocr_geometry_mode_requested: str | None = None
    ocr_geometry_mode_effective: str | None = None
    setup_warning: str | None = None


def _normalize_requested_ocr_provider(value: str | None) -> str:
    provider_id = (clean_str(value) or "auto").lower()
    if provider_id in {"remote", "ai"}:
        return "aiocr"
    if provider_id in {"paddle-local", "local_paddle"}:
        return "paddle_local"
    return provider_id


def _normalize_ocr_geometry_mode(value: str | None) -> str:
    mode = (clean_str(value) or "auto").lower()
    if mode in {"direct", "direct_ai", "ai", "requested"}:
        return "direct_ai"
    if mode in {"local", "local_tesseract", "tesseract", "hybrid_local"}:
        return "local_tesseract"
    return "auto"


def _is_bbox_reliable_ai_ocr_model(model: str | None) -> bool:
    lowered = str(model or "").strip().lower()
    if not lowered:
        return False
    if "paddleocr-vl" in lowered:
        return True
    if "deepseek-ocr" in lowered:
        return True
    # Keep this conservative: only treat clearly OCR-specialized names as
    # bbox-reliable to avoid regressing generic VL models.
    if re.search(r"(?:^|[\\/_\\-])ocr(?:$|[\\/_\\-])", lowered):
        return True
    return False


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
    ocr_geometry_mode: str | None = None,
    ocr_ai_linebreak_assist: bool | None = None,
    ocr_strict_mode: bool | None = None,
) -> OcrRuntimeSetup:
    # If the user didn't configure separate AI OCR credentials, reuse
    # the layout-assist OpenAI-compatible settings (when available).
    effective_ocr_ai_api_key = clean_str(ocr_ai_api_key)
    effective_ocr_ai_provider = (clean_str(ocr_ai_provider) or "auto").lower()
    if effective_ocr_ai_provider in {"openai_compatible", "openai-compatible"}:
        effective_ocr_ai_provider = "openai"
    effective_ocr_ai_base_url = clean_str(ocr_ai_base_url)
    effective_ocr_ai_model = clean_str(ocr_ai_model)
    provider_id = (clean_str(provider) or "openai").lower()
    if (
        not effective_ocr_ai_api_key
        and provider_id != "claude"
        and clean_str(api_key)
    ):
        effective_ocr_ai_api_key = clean_str(api_key)
        effective_ocr_ai_base_url = clean_str(base_url)
        effective_ocr_ai_model = clean_str(model)

    effective_ocr_provider = _normalize_requested_ocr_provider(ocr_provider)
    requested_ocr_provider = effective_ocr_provider
    strict_ocr_mode = bool(False if ocr_strict_mode is None else ocr_strict_mode)
    if effective_ocr_provider in {"aiocr", "paddle"} and not strict_ocr_mode:
        strict_ocr_mode = True
        logger.info(
            "Force strict OCR mode for explicit provider=%s (no hidden fallback, fail-fast on OCR errors)",
            effective_ocr_provider,
        )

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

    text_refiner: AiOcrTextRefiner | None = None
    linebreak_refiner: AiOcrTextRefiner | None = None
    linebreak_enabled = False
    auto_linebreak_enabled = False
    requested_geometry_mode = _normalize_ocr_geometry_mode(ocr_geometry_mode)
    runtime_ocr_provider = effective_ocr_provider
    geometry_strategy = "direct"
    geometry_mode_effective = "n/a"

    # Practical strategy:
    # For generic VL models (Qwen/GPT/Gemini-style) selected as `aiocr`, bbox
    # geometry is often coarse/unstable. Keep AI for recognition/refinement,
    # but let a local bbox-accurate OCR engine provide geometry.
    if requested_ocr_provider == "aiocr":
        geometry_mode_effective = "direct_ai"
        has_ai_key = bool(effective_ocr_ai_api_key)
        is_bbox_reliable = _is_bbox_reliable_ai_ocr_model(effective_ocr_ai_model)

        if requested_geometry_mode == "local_tesseract":
            runtime_ocr_provider = "tesseract"
            geometry_mode_effective = "local_tesseract"
            geometry_strategy = (
                "forced_local_tesseract_geometry_with_ai_refine"
                if has_ai_key
                else "forced_local_tesseract_geometry"
            )
            logger.info(
                "Forced OCR geometry mode=%s: provider=%s -> runtime=%s (model=%s)",
                requested_geometry_mode,
                requested_ocr_provider,
                runtime_ocr_provider,
                effective_ocr_ai_model or "<unset>",
            )
        elif requested_geometry_mode == "direct_ai":
            geometry_mode_effective = "direct_ai"
            geometry_strategy = "forced_direct_ai_geometry"
        elif has_ai_key and (not is_bbox_reliable):
            runtime_ocr_provider = "tesseract"
            geometry_mode_effective = "local_tesseract"
            geometry_strategy = "local_tesseract_geometry_with_ai_refine"
            logger.info(
                "AI OCR geometry strategy enabled: provider=%s -> runtime=%s (model=%s)",
                requested_ocr_provider,
                runtime_ocr_provider,
                effective_ocr_ai_model or "<unset>",
            )
        else:
            geometry_mode_effective = "direct_ai"

    try:
        try:
            ocr_manager = create_ocr_manager(
                provider=runtime_ocr_provider,
                ai_provider=effective_ocr_ai_provider,
                ai_api_key=effective_ocr_ai_api_key,
                ai_base_url=effective_ocr_ai_base_url,
                ai_model=effective_ocr_ai_model,
                baidu_app_id=ocr_baidu_app_id,
                baidu_api_key=ocr_baidu_api_key,
                baidu_secret_key=ocr_baidu_secret_key,
                tesseract_min_confidence=effective_tesseract_min_conf,
                tesseract_language=effective_tesseract_language,
                strict_no_fallback=strict_ocr_mode,
                allow_paddle_model_downgrade=not strict_ocr_mode,
            )
        except Exception as e:
            if runtime_ocr_provider != requested_ocr_provider:
                logger.warning(
                    "OCR geometry fallback provider failed (%s), retrying requested provider=%s: %s",
                    runtime_ocr_provider,
                    requested_ocr_provider,
                    e,
                )
                runtime_ocr_provider = requested_ocr_provider
                geometry_strategy = "direct"
                geometry_mode_effective = (
                    "direct_ai" if requested_ocr_provider == "aiocr" else "n/a"
                )
                ocr_manager = create_ocr_manager(
                    provider=runtime_ocr_provider,
                    ai_provider=effective_ocr_ai_provider,
                    ai_api_key=effective_ocr_ai_api_key,
                    ai_base_url=effective_ocr_ai_base_url,
                    ai_model=effective_ocr_ai_model,
                    baidu_app_id=ocr_baidu_app_id,
                    baidu_api_key=ocr_baidu_api_key,
                    baidu_secret_key=ocr_baidu_secret_key,
                    tesseract_min_confidence=effective_tesseract_min_conf,
                    tesseract_language=effective_tesseract_language,
                    strict_no_fallback=strict_ocr_mode,
                    allow_paddle_model_downgrade=not strict_ocr_mode,
                )
            else:
                raise

        # Optional: refine line texts using an AI vision model while keeping
        # bbox geometry from a bbox-accurate OCR engine (e.g. Tesseract).
        try:
            provider_choice = requested_ocr_provider
            is_paddle_vl_model = "paddleocr-vl" in (
                str(effective_ocr_ai_model or "").strip().lower()
            )
            text_refiner_allowed = provider_choice in {"auto", "aiocr"}
            linebreak_refiner_allowed = provider_choice in {
                "auto",
                "aiocr",
                "tesseract",
                "local",
                "baidu",
                "paddle",
                "paddle_local",
            }
            linebreak_requested = ocr_ai_linebreak_assist
            linebreak_enabled = bool(linebreak_requested)
            auto_linebreak_enabled = False

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
        except Exception as e:
            logger.warning("AI OCR text refiner setup failed: %s", e)
            text_refiner = None
            linebreak_refiner = None

        return OcrRuntimeSetup(
            ocr_manager=ocr_manager,
            text_refiner=text_refiner,
            linebreak_refiner=linebreak_refiner,
            effective_ocr_provider=runtime_ocr_provider,
            effective_ocr_ai_provider=effective_ocr_ai_provider,
            effective_ocr_ai_api_key=effective_ocr_ai_api_key,
            effective_ocr_ai_base_url=effective_ocr_ai_base_url,
            effective_ocr_ai_model=effective_ocr_ai_model,
            effective_tesseract_language=effective_tesseract_language,
            effective_tesseract_min_conf=effective_tesseract_min_conf,
            strict_ocr_mode=strict_ocr_mode,
            linebreak_enabled=linebreak_enabled,
            auto_linebreak_enabled=auto_linebreak_enabled,
            ocr_geometry_provider=runtime_ocr_provider,
            ocr_geometry_strategy=geometry_strategy,
            ocr_geometry_mode_requested=requested_geometry_mode,
            ocr_geometry_mode_effective=geometry_mode_effective,
            setup_warning=None,
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
            ocr_manager=None,
            text_refiner=None,
            linebreak_refiner=None,
            effective_ocr_provider=runtime_ocr_provider,
            effective_ocr_ai_provider=effective_ocr_ai_provider,
            effective_ocr_ai_api_key=effective_ocr_ai_api_key,
            effective_ocr_ai_base_url=effective_ocr_ai_base_url,
            effective_ocr_ai_model=effective_ocr_ai_model,
            effective_tesseract_language=effective_tesseract_language,
            effective_tesseract_min_conf=effective_tesseract_min_conf,
            strict_ocr_mode=strict_ocr_mode,
            linebreak_enabled=linebreak_enabled,
            auto_linebreak_enabled=auto_linebreak_enabled,
            ocr_geometry_provider=runtime_ocr_provider,
            ocr_geometry_strategy=geometry_strategy,
            ocr_geometry_mode_requested=requested_geometry_mode,
            ocr_geometry_mode_effective=geometry_mode_effective,
            setup_warning=f"{e!s}",
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
            "provider": setup.effective_ocr_ai_provider,
            "base_url": setup.effective_ocr_ai_base_url,
            "model": setup.effective_ocr_ai_model,
        },
        "ocr_strict_mode": bool(setup.strict_ocr_mode),
    }
