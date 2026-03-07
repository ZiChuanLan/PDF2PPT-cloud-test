"""Local OCR providers, manager orchestration, and conversion helpers."""

from dataclasses import dataclass
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .ai_client import (
    AiOcrClient,
    AiOcrTextRefiner,
    _is_multiline_candidate_for_linebreak_assist,
)
from .base import (
    _ACRONYM_ALLOWLIST,
    _DEFAULT_PADDLE_OCR_VL_MODEL,
    _clean_str,
    _normalize_paddle_language,
    _normalize_tesseract_language,
    _split_tesseract_languages,
    OcrProvider,
)
from .routing import (
    ROUTE_KIND_HYBRID_AUTO,
    ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
    ROUTE_KIND_REMOTE_DOC_PARSER,
    ROUTE_KIND_REMOTE_PROMPT_OCR,
    normalize_ocr_route_kind,
)
from .runtime_probe import (
    probe_local_paddle_models,
    probe_local_paddleocr,
    probe_local_tesseract,
    probe_local_tesseract_models,
)
from .utils import _coerce_bbox_xyxy, _is_paddleocr_vl_model
from .vendors import _normalize_ai_ocr_provider
from .deepseek_parser import _looks_like_ocr_prompt_echo_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteOcrClientSpec:
    requested_provider: str
    route_kind: str
    ai_provider: str | None
    ai_model: str | None


def resolve_remote_ocr_client_spec(
    *,
    provider_id: str,
    ai_provider: str | None,
    ai_base_url: str | None,
    ai_model: str | None,
    route_kind: str | None,
) -> RemoteOcrClientSpec:
    normalized_route_kind = normalize_ocr_route_kind(route_kind)
    resolved_model = _clean_str(ai_model) or None

    if provider_id == "paddle":
        resolved_model = resolved_model or _DEFAULT_PADDLE_OCR_VL_MODEL
        if not _is_paddleocr_vl_model(resolved_model):
            raise ValueError(
                "Paddle OCR provider requires a PaddleOCR-VL model (for example PaddlePaddle/PaddleOCR-VL or PaddlePaddle/PaddleOCR-VL-1.5)"
            )
        normalized_route_kind = ROUTE_KIND_REMOTE_DOC_PARSER
    elif provider_id == "aiocr":
        if normalized_route_kind not in {
            ROUTE_KIND_REMOTE_PROMPT_OCR,
            ROUTE_KIND_REMOTE_DOC_PARSER,
            ROUTE_KIND_LOCAL_LAYOUT_BLOCK_OCR,
        }:
            normalized_route_kind = (
                ROUTE_KIND_REMOTE_DOC_PARSER
                if _is_paddleocr_vl_model(resolved_model)
                else ROUTE_KIND_REMOTE_PROMPT_OCR
            )
    else:
        raise ValueError(f"Unsupported remote OCR provider: {provider_id}")

    resolved_ai_provider = ai_provider
    if _is_paddleocr_vl_model(resolved_model):
        normalized_vendor = _normalize_ai_ocr_provider(ai_provider)
        if normalized_vendor == "auto" and not _clean_str(ai_base_url):
            resolved_ai_provider = "siliconflow"

    return RemoteOcrClientSpec(
        requested_provider=provider_id,
        route_kind=normalized_route_kind,
        ai_provider=resolved_ai_provider,
        ai_model=resolved_model,
    )


def create_remote_ocr_client(
    *,
    requested_provider: str,
    route_kind: str | None = None,
    ai_provider: str | None = None,
    ai_api_key: str,
    ai_base_url: str | None = None,
    ai_model: str | None = None,
    ai_layout_model: str | None = None,
    paddle_doc_max_side_px: int | None = None,
    layout_block_max_concurrency: int | None = None,
    request_rpm_limit: int | None = None,
    request_tpm_limit: int | None = None,
    request_max_retries: int | None = None,
    allow_paddle_model_downgrade: bool = False,
) -> AiOcrClient:
    spec = resolve_remote_ocr_client_spec(
        provider_id=requested_provider,
        ai_provider=ai_provider,
        ai_base_url=ai_base_url,
        ai_model=ai_model,
        route_kind=route_kind,
    )
    return _build_remote_ocr_client_from_spec(
        spec=spec,
        ai_api_key=ai_api_key,
        ai_base_url=ai_base_url,
        ai_layout_model=ai_layout_model,
        paddle_doc_max_side_px=paddle_doc_max_side_px,
        layout_block_max_concurrency=layout_block_max_concurrency,
        request_rpm_limit=request_rpm_limit,
        request_tpm_limit=request_tpm_limit,
        request_max_retries=request_max_retries,
        allow_paddle_model_downgrade=allow_paddle_model_downgrade,
    )


def _build_remote_ocr_client_from_spec(
    *,
    spec: RemoteOcrClientSpec,
    ai_api_key: str,
    ai_base_url: str | None,
    ai_layout_model: str | None,
    paddle_doc_max_side_px: int | None,
    layout_block_max_concurrency: int | None,
    request_rpm_limit: int | None,
    request_tpm_limit: int | None,
    request_max_retries: int | None,
    allow_paddle_model_downgrade: bool,
) -> AiOcrClient:
    client = AiOcrClient(
        api_key=ai_api_key,
        base_url=ai_base_url,
        model=spec.ai_model,
        provider=spec.ai_provider,
        layout_model=ai_layout_model,
        paddle_doc_max_side_px=paddle_doc_max_side_px,
        layout_block_max_concurrency=layout_block_max_concurrency,
        request_rpm_limit=request_rpm_limit,
        request_tpm_limit=request_tpm_limit,
        request_max_retries=request_max_retries,
        route_kind=spec.route_kind,
    )
    client.allow_model_downgrade = bool(allow_paddle_model_downgrade)
    return client


_resolve_remote_ocr_client_spec = resolve_remote_ocr_client_spec
_build_remote_ocr_client = create_remote_ocr_client


class BaiduOcrClient(OcrProvider):
    """Baidu OCR client implementation."""

    def __init__(
        self,
        app_id: str | None = None,
        api_key: str | None = None,
        secret_key: str | None = None,
    ):
        """Initialize Baidu OCR client with credentials from parameters or env."""
        self.app_id = (app_id or os.getenv("BAIDU_OCR_APP_ID") or "").strip()
        self.api_key = (api_key or os.getenv("BAIDU_OCR_API_KEY") or "").strip()
        self.secret_key = (
            secret_key or os.getenv("BAIDU_OCR_SECRET_KEY") or ""
        ).strip()

        if not all([self.api_key, self.secret_key]):
            raise ValueError(
                "Baidu OCR credentials not found. "
                "Set BAIDU_OCR_API_KEY and BAIDU_OCR_SECRET_KEY"
            )

        try:
            from aip import AipOcr

            # The legacy `baidu-aip` SDK constructor still accepts appId, but its
            # token flow authenticates with apiKey/secretKey only. Keep App ID as
            # an optional compatibility field instead of a hard requirement.
            self.client = AipOcr(self.app_id, self.api_key, self.secret_key)
            logger.info("Baidu OCR client initialized successfully")
        except ImportError:
            raise ImportError(
                "baidu-aip package not installed. Install with: pip install baidu-aip"
            )

    def ocr_image(self, image_path: str) -> List[Dict]:
        """
        Perform OCR using Baidu API.

        Args:
            image_path: Path to the image file

        Returns:
            List of text elements with bbox and confidence
        """
        try:
            # Read image as binary
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Request direction + probability when supported. These options
            # improve robustness on scan-heavy slide decks and keep the output
            # stable across Baidu OCR endpoints/SDK versions.
            options = {
                "detect_direction": "true",
                "probability": "true",
                # Prefer bilingual recognition for typical CN/EN decks.
                "language_type": "CHN_ENG",
            }

            # Prefer high-accuracy endpoint *with location* so we can place
            # editable text boxes precisely. SDK method names vary slightly
            # across versions, so we probe a few.
            call_candidates: list[tuple[str, Any]] = []
            if hasattr(self.client, "accurate"):
                call_candidates.append(("accurate", getattr(self.client, "accurate")))
            if hasattr(self.client, "general"):
                call_candidates.append(("general", getattr(self.client, "general")))
            if hasattr(self.client, "basicAccurate"):
                # Some SDKs expose this name; it typically maps to accurate_basic.
                call_candidates.append(
                    ("basicAccurate", getattr(self.client, "basicAccurate"))
                )
            if hasattr(self.client, "basicGeneral"):
                call_candidates.append(
                    ("basicGeneral", getattr(self.client, "basicGeneral"))
                )

            if not call_candidates:
                raise RuntimeError("Baidu OCR SDK has no callable OCR methods")

            last_error: Exception | None = None
            result: dict | None = None
            used_method = None
            for name, fn in call_candidates:
                try:
                    used_method = name
                    # Keep options minimal; callers may still get direction info
                    # by enabling detect_direction via Baidu console if needed.
                    try:
                        result = fn(image_data, options)
                    except TypeError:
                        # Some SDK versions/endpoints may not accept an options
                        # arg (or may have a different signature).
                        result = fn(image_data)
                    if isinstance(result, dict) and "error_code" not in result:
                        break
                except Exception as e:
                    last_error = e
                    result = None
                    continue

            if not isinstance(result, dict):
                raise RuntimeError("Baidu OCR returned no result") from last_error

            # Check for errors
            if "error_code" in result:
                error_msg = result.get("error_msg", "Unknown error")
                logger.error("Baidu OCR API error (%s): %s", used_method, error_msg)
                raise RuntimeError(f"Baidu OCR failed: {error_msg}")

            # Parse results
            img_w = 0.0
            img_h = 0.0
            try:
                with Image.open(image_path) as _im:
                    img_w = float(_im.width)
                    img_h = float(_im.height)
            except Exception:
                img_w = 0.0
                img_h = 0.0

            elements: list[dict] = []
            words_result = result.get("words_result", [])
            if not isinstance(words_result, list):
                words_result = []

            for item in words_result:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("words") or "").strip()
                location = item.get("location") or {}
                if not text or not isinstance(location, dict):
                    continue

                # Baidu returns: {left, top, width, height} in pixels
                try:
                    x0 = float(location.get("left", 0) or 0)
                    y0 = float(location.get("top", 0) or 0)
                    w = float(location.get("width", 0) or 0)
                    h = float(location.get("height", 0) or 0)
                except Exception:
                    continue
                if w <= 0 or h <= 0:
                    continue

                # Defensive pruning for occasional coarse/paragraph-level boxes.
                # Such boxes are harmful in slide conversion because they can wipe
                # image regions and create duplicate/stacked text overlays.
                compact = "".join(ch for ch in text if not ch.isspace())
                if img_w > 0 and img_h > 0:
                    area_ratio = float(w * h) / float(max(1.0, img_w * img_h))
                    width_ratio = float(w) / float(max(1.0, img_w))
                    height_ratio = float(h) / float(max(1.0, img_h))
                    if area_ratio >= 0.16:
                        continue
                    if (
                        width_ratio >= 0.85
                        and height_ratio >= 0.08
                        and len(compact) <= 24
                    ):
                        continue
                    if (
                        area_ratio >= 0.06
                        and len(compact) <= 6
                        and height_ratio >= 0.06
                    ):
                        continue

                elements.append(
                    {
                        "text": text,
                        "bbox": [x0, y0, x0 + w, y0 + h],
                        # Baidu does not reliably return confidences across
                        # endpoints; keep a high default so downstream can treat
                        # it as a strong signal.
                        "confidence": 0.95,
                    }
                )

            logger.info(
                "Baidu OCR extracted %s text elements from %s (method=%s)",
                len(elements),
                image_path,
                used_method,
            )
            return elements

        except Exception as e:
            logger.error("Baidu OCR failed on %s: %s", image_path, e)
            raise


class TesseractOcrClient(OcrProvider):
    """Tesseract OCR client implementation."""

    def __init__(self, min_confidence: float = 50.0, language: str = "chi_sim+eng"):
        """
        Initialize Tesseract OCR client.

        Args:
            min_confidence: Minimum confidence threshold (0-100)
        """
        self.min_confidence = min_confidence
        # Prefer a bilingual default for typical scanned PDFs. This project is
        # mostly used on Chinese+English slide decks.
        self.language = _normalize_tesseract_language(language)

        try:
            import pytesseract
            from pytesseract import Output

            self.pytesseract = pytesseract
            self.Output = Output
            probe = probe_local_tesseract(language=self.language)
            if not bool(probe.get("binary_available")):
                raise RuntimeError(
                    "Tesseract executable is not available. "
                    "Install system package: tesseract-ocr"
                )

            missing_languages = [
                str(item).strip()
                for item in (probe.get("missing_languages") or [])
                if str(item).strip()
            ]
            if missing_languages:
                requested_languages = _split_tesseract_languages(self.language)
                available_languages = [
                    str(item).strip()
                    for item in (probe.get("available_languages") or [])
                    if str(item).strip()
                ]
                available_set = {lang.lower() for lang in available_languages}
                fallback_languages = [
                    lang
                    for lang in requested_languages
                    if lang.lower() in available_set
                ]

                if fallback_languages:
                    fallback = "+".join(fallback_languages)
                    logger.warning(
                        "Tesseract requested lang '%s' is partially missing. "
                        "Fallback to '%s'. Missing=%s",
                        self.language,
                        fallback,
                        ",".join(missing_languages),
                    )
                    self.language = fallback
                else:
                    raise RuntimeError(
                        "Tesseract language pack(s) not available: "
                        f"{', '.join(missing_languages)}"
                    )

            logger.info(
                "Tesseract OCR client initialized successfully (lang=%s, version=%s)",
                self.language,
                str(probe.get("version") or "unknown"),
            )
        except ImportError:
            raise ImportError(
                "pytesseract package not installed. "
                "Install with: pip install pytesseract"
            )

    def _extract_elements_from_data(
        self, data: dict, *, min_conf: float
    ) -> tuple[list[dict], dict]:
        elements: list[dict] = []
        n_boxes = len(data.get("text") or [])
        line_keys: set[tuple[int, int, int]] = set()
        conf_sum = 0.0
        conf_n = 0

        for i in range(n_boxes):
            # Tesseract returns conf as string numbers; it can also be "-1".
            try:
                conf = float((data.get("conf") or ["-1"] * n_boxes)[i])
            except Exception:
                conf = -1.0
            text = str((data.get("text") or [""] * n_boxes)[i] or "").strip()

            if conf < float(min_conf) or not text:
                continue

            try:
                x = int((data.get("left") or [0] * n_boxes)[i])
                y = int((data.get("top") or [0] * n_boxes)[i])
                w = int((data.get("width") or [0] * n_boxes)[i])
                h = int((data.get("height") or [0] * n_boxes)[i])
            except Exception:
                continue

            block_num = (data.get("block_num") or [None] * n_boxes)[i]
            par_num = (data.get("par_num") or [None] * n_boxes)[i]
            line_num = (data.get("line_num") or [None] * n_boxes)[i]
            word_num = (data.get("word_num") or [None] * n_boxes)[i]

            try:
                lk = (int(block_num or 0), int(par_num or 0), int(line_num or 0))
                line_keys.add(lk)
            except Exception:
                pass

            elements.append(
                {
                    "text": text,
                    "bbox": [x, y, x + w, y + h],
                    "confidence": conf / 100.0,  # Normalize to 0-1
                    # Preserve Tesseract's structural hints so we can merge
                    # words into line-level boxes more accurately.
                    "block_num": block_num,
                    "par_num": par_num,
                    "line_num": line_num,
                    "word_num": word_num,
                }
            )
            conf_sum += conf
            conf_n += 1

        avg_conf = (conf_sum / conf_n) if conf_n else 0.0
        stats = {
            "words": len(elements),
            "lines": len(line_keys),
            "avg_conf": avg_conf,
        }
        return elements, stats

    def ocr_image(self, image_path: str) -> List[Dict]:
        """
        Perform OCR using Tesseract.

        Args:
            image_path: Path to the image file

        Returns:
            List of text elements with bbox and confidence
        """
        try:
            # Open image
            image = Image.open(image_path).convert("RGB")

            # Slides / scanned pages often have multiple isolated text boxes.
            # Sparse-text mode (PSM 11) typically yields higher recall, but some
            # documents behave better with other modes. We start with PSM 11 and
            # only try extra modes when the first pass looks suspiciously low.
            psm_candidates: list[int] = [11]

            # Try the configured language first, but in real-world usage users
            # sometimes set lang=eng while the PDF contains Chinese. In that case
            # we automatically try a bilingual fallback and pick the better run.
            lang_candidates: list[str] = []
            primary_lang = (self.language or "").strip()
            if primary_lang:
                lang_candidates.append(primary_lang)
            fallback_lang = "chi_sim+eng"
            if fallback_lang not in lang_candidates:
                lang_candidates.append(fallback_lang)

            best_elements: list[dict] = []
            best_stats: dict = {"words": 0, "lines": 0, "avg_conf": 0.0}
            best_lang: str | None = None
            best_psm: int | None = None
            last_error: Exception | None = None

            def _score(stats: dict) -> int:
                return (int(stats.get("lines") or 0) * 10) + int(
                    stats.get("words") or 0
                )

            def _run(
                lang: str, psm: int, *, min_conf: float
            ) -> tuple[list[dict] | None, dict | None]:
                nonlocal last_error
                try:
                    data = self.pytesseract.image_to_data(
                        image,
                        output_type=self.Output.DICT,
                        lang=lang,
                        config=f"--psm {int(psm)}",
                    )
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Tesseract OCR run failed (lang=%s, psm=%s): %s", lang, psm, e
                    )
                    return (None, None)

                elems, stats = self._extract_elements_from_data(
                    data, min_conf=float(min_conf)
                )
                return (elems, stats)

            min_conf_primary = float(self.min_confidence)
            used_min_conf = float(min_conf_primary)

            # First pass: PSM 11.
            for lang in lang_candidates:
                elems, stats = _run(lang, 11, min_conf=min_conf_primary)
                if elems is None or stats is None:
                    continue
                if best_lang is None or _score(stats) > _score(best_stats):
                    best_elements = elems
                    best_stats = stats
                    best_lang = lang
                    best_psm = 11

            # If the first pass looks low recall, try a couple more modes.
            if best_lang is not None:
                if (
                    int(best_stats.get("lines") or 0) < 12
                    and int(best_stats.get("words") or 0) < 80
                ):
                    psm_candidates = [11, 6, 3]

            for psm in psm_candidates:
                if psm == 11:
                    continue
                for lang in lang_candidates:
                    elems, stats = _run(lang, psm, min_conf=min_conf_primary)
                    if elems is None or stats is None:
                        continue
                    if best_lang is None or _score(stats) > _score(best_stats):
                        best_elements = elems
                        best_stats = stats
                        best_lang = lang
                        best_psm = int(psm)

            if best_lang is None:
                # All tesseract runs failed (e.g. binary not installed).
                raise RuntimeError(
                    "Tesseract OCR failed for all languages"
                ) from last_error

            # If the configured min_conf is too strict, Tesseract can return an
            # empty/near-empty result on scan-heavy slides. In that case we
            # retry with a lower confidence threshold so we at least get line
            # geometry; downstream can filter obvious noise and (optionally)
            # refine text with an AI vision model.
            if min_conf_primary > 25.0:
                lines_n = int(best_stats.get("lines") or 0)
                words_n = int(best_stats.get("words") or 0)
                looks_empty = (not best_elements) or (lines_n < 8 and words_n < 40)
                if looks_empty:
                    low_min_conf = 25.0
                    low_best_elems: list[dict] = []
                    low_best_stats: dict = {"words": 0, "lines": 0, "avg_conf": 0.0}
                    low_best_lang: str | None = None
                    low_best_psm: int | None = None

                    # Start from the best (lang, psm) choice, but also probe a
                    # couple other modes to avoid pathological edge cases.
                    psm_probe: list[int] = []
                    if best_psm is not None:
                        psm_probe.append(int(best_psm))
                    for p in (11, 6, 3):
                        if p not in psm_probe:
                            psm_probe.append(p)

                    for psm in psm_probe:
                        for lang in lang_candidates:
                            elems, stats = _run(lang, int(psm), min_conf=low_min_conf)
                            if elems is None or stats is None:
                                continue
                            if low_best_lang is None or _score(stats) > _score(
                                low_best_stats
                            ):
                                low_best_elems = elems
                                low_best_stats = stats
                                low_best_lang = lang
                                low_best_psm = int(psm)

                    if (
                        low_best_lang is not None
                        and low_best_elems
                        and _score(low_best_stats) > _score(best_stats)
                    ):
                        logger.info(
                            "Tesseract OCR lowered min_conf from %s to %s (lines=%s words=%s).",
                            min_conf_primary,
                            low_min_conf,
                            low_best_stats.get("lines"),
                            low_best_stats.get("words"),
                        )
                        best_elements = low_best_elems
                        best_stats = low_best_stats
                        best_lang = low_best_lang
                        best_psm = (
                            low_best_psm if low_best_psm is not None else best_psm
                        )
                        used_min_conf = float(low_min_conf)

            if best_lang and best_lang != primary_lang:
                logger.info(
                    "Tesseract OCR auto-switched lang from %s to %s (lines=%s words=%s).",
                    primary_lang or "<empty>",
                    best_lang,
                    best_stats.get("lines"),
                    best_stats.get("words"),
                )

            logger.info(
                "Tesseract OCR extracted %s text elements from %s (lang=%s, psm=%s, min_conf=%s)",
                len(best_elements),
                image_path,
                best_lang or primary_lang or "<unknown>",
                best_psm if best_psm is not None else 11,
                used_min_conf,
            )
            return best_elements

        except Exception as e:
            logger.error(f"Tesseract OCR failed on {image_path}: {str(e)}")
            raise


class PaddleOcrClient(OcrProvider):
    """PaddleOCR local client implementation."""

    def __init__(self, language: str = "ch"):
        self.language = _normalize_paddle_language(language)
        self._engine: Any | None = None
        # PaddleOCR 3.x (PaddleX pipeline) can be memory-hungry on large page
        # renders. Downscale long-edge to keep CPU inference stable.
        self._max_side_px: int = 2200

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "paddleocr package not installed. Install with: pip install paddleocr"
            )

        self._PaddleOCR = PaddleOCR
        logger.info("PaddleOCR client initialized (lang=%s)", self.language)

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        last_error: Exception | None = None
        constructors: list[dict[str, Any]] = [
            # PaddleOCR 3.x uses a PaddleX pipeline wrapper internally. On some
            # CPU builds, enabling MKL-DNN / oneDNN can trigger runtime errors in
            # the new executor. Keep it off by default for stability.
            {
                "lang": self.language,
                "use_textline_orientation": True,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "enable_mkldnn": False,
                "enable_cinn": False,
                "device": "cpu",
            },
            {
                "lang": self.language,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "enable_mkldnn": False,
                "enable_cinn": False,
                "device": "cpu",
            },
        ]

        for kwargs in constructors:
            try:
                self._engine = self._PaddleOCR(**kwargs)
                return self._engine
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError("Failed to initialize PaddleOCR runtime") from last_error

    def ocr_image(self, image_path: str) -> List[Dict]:
        engine = self._ensure_engine()

        # PaddleOCR can run on file paths or numpy arrays. We downscale huge
        # images before inference and scale bboxes back to the original size.
        image_for_ocr: Any = image_path
        scale_x = 1.0
        scale_y = 1.0
        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            largest = max(w, h)
            if largest > int(self._max_side_px):
                ratio = float(self._max_side_px) / float(largest)
                new_w = max(32, int(round(float(w) * ratio)))
                new_h = max(32, int(round(float(h) * ratio)))
                image_small = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                try:
                    import numpy as np

                    image_for_ocr = np.array(image_small)
                    scale_x = float(w) / float(new_w)
                    scale_y = float(h) / float(new_h)
                except Exception:
                    image_for_ocr = image_path
        except Exception:
            pass

        raw_result: Any = None
        last_error: Exception | None = None

        ocr_calls = [
            # PaddleOCR 3.x deprecates `.ocr()` in favor of `.predict()` and no
            # longer accepts the legacy `cls=` kwarg.
            lambda: engine.predict(image_for_ocr),
            lambda: engine.predict(input=image_for_ocr),
            lambda: engine.ocr(image_for_ocr),
        ]
        for fn in ocr_calls:
            try:
                raw_result = fn()
                last_error = None
                break
            except Exception as e:
                last_error = e
                continue

        if raw_result is None and hasattr(engine, "predict"):
            predict_calls = [
                lambda: engine.predict(input=image_for_ocr),
                lambda: engine.predict(image_for_ocr),
            ]
            for fn in predict_calls:
                try:
                    raw_result = fn()
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    continue

        if raw_result is None:
            if last_error is not None:
                logger.warning("PaddleOCR failed to produce output: %s", last_error)
            raise RuntimeError("PaddleOCR failed to produce output") from last_error

        elements: list[dict] = []

        def _append_from_paddlex_ocr_payload(payload: dict[str, Any]) -> bool:
            """Parse PaddleOCR 3.x / PaddleX pipeline output dict."""

            texts = payload.get("rec_texts")
            polys = payload.get("rec_polys")
            scores = payload.get("rec_scores")
            if texts is None or polys is None:
                # Some variants expose detection polys; use them only if they line
                # up with recognized texts.
                texts = payload.get("texts") or payload.get("text") or texts
                polys = payload.get("polys") or payload.get("dt_polys") or polys
                scores = payload.get("scores") or payload.get("rec_scores") or scores

            if texts is not None and hasattr(texts, "tolist"):
                try:
                    texts = texts.tolist()
                except Exception:
                    pass
            if polys is not None and hasattr(polys, "tolist"):
                try:
                    polys = polys.tolist()
                except Exception:
                    pass
            if scores is not None and hasattr(scores, "tolist"):
                try:
                    scores = scores.tolist()
                except Exception:
                    pass

            if not isinstance(texts, list) or not isinstance(polys, list) or not texts:
                return False
            if len(polys) != len(texts):
                return False

            used = False
            for i, (text_raw, poly) in enumerate(zip(texts, polys)):
                text = str(text_raw or "").strip()
                bbox = _coerce_bbox_xyxy(poly)
                if not text or not bbox:
                    continue

                confidence_raw: Any = None
                if isinstance(scores, list) and i < len(scores):
                    confidence_raw = scores[i]
                try:
                    confidence = (
                        float(confidence_raw) if confidence_raw is not None else 0.85
                    )
                except Exception:
                    confidence = 0.85
                if confidence > 1.0:
                    confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
                confidence = max(0.0, min(confidence, 1.0))

                elements.append(
                    {
                        "text": text,
                        "bbox": [
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ],
                        "confidence": confidence,
                    }
                )
                used = True

            return used

        def _append_from_dict(candidate: dict[str, Any]) -> bool:
            text = str(
                candidate.get("text")
                or candidate.get("transcription")
                or candidate.get("content")
                or candidate.get("label")
                or ""
            ).strip()
            bbox = _coerce_bbox_xyxy(
                candidate.get("bbox")
                or candidate.get("box")
                or candidate.get("points")
                or candidate.get("polygon")
                or candidate.get("position")
                or candidate.get("coordinates")
                or candidate.get("block_bbox")
            )
            if not text or not bbox:
                return False

            confidence_raw = (
                candidate.get("confidence")
                or candidate.get("score")
                or candidate.get("prob")
            )
            try:
                confidence = (
                    float(confidence_raw) if confidence_raw is not None else 0.85
                )
            except Exception:
                confidence = 0.85
            if confidence > 1.0:
                confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
            confidence = max(0.0, min(confidence, 1.0))

            elements.append(
                {
                    "text": text,
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ],
                    "confidence": confidence,
                }
            )
            return True

        # Fast-path: PaddleOCR 3.x (PaddleX pipeline) returns a list of dicts
        # containing rec_texts/rec_polys arrays instead of the legacy
        # `[[quad], (text, score)]` layout.
        if (
            isinstance(raw_result, list)
            and raw_result
            and all(isinstance(v, dict) for v in raw_result)
        ):
            used_any = False
            for payload in raw_result:
                used_any = _append_from_paddlex_ocr_payload(payload) or used_any
            if used_any:
                if scale_x != 1.0 or scale_y != 1.0:
                    for el in elements:
                        bbox = el.get("bbox")
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            continue
                        x0, y0, x1, y1 = [float(v) for v in bbox]
                        el["bbox"] = [
                            x0 * scale_x,
                            y0 * scale_y,
                            x1 * scale_x,
                            y1 * scale_y,
                        ]
                logger.info(
                    "PaddleOCR extracted %s text elements from %s (paddlex pipeline)",
                    len(elements),
                    image_path,
                )
                return elements

        stack: list[Any] = [raw_result]
        max_nodes = 20000
        visited = 0

        while stack and visited < max_nodes:
            visited += 1
            node = stack.pop()

            if isinstance(node, dict):
                used = _append_from_dict(node)
                if not used:
                    for value in node.values():
                        stack.append(value)
                continue

            if isinstance(node, (list, tuple)):
                if len(node) >= 2:
                    bbox = _coerce_bbox_xyxy(node[0])
                    text = ""
                    confidence_raw: Any = None

                    second = node[1]
                    if isinstance(second, (list, tuple)):
                        if second:
                            text = str(second[0] or "").strip()
                        if len(second) > 1:
                            confidence_raw = second[1]
                    elif isinstance(second, str):
                        text = second.strip()
                        if len(node) > 2:
                            confidence_raw = node[2]

                    if text and bbox:
                        try:
                            confidence = (
                                float(confidence_raw)
                                if confidence_raw is not None
                                else 0.85
                            )
                        except Exception:
                            confidence = 0.85
                        if confidence > 1.0:
                            confidence = (
                                confidence / 100.0 if confidence <= 100.0 else 1.0
                            )
                        confidence = max(0.0, min(confidence, 1.0))

                        elements.append(
                            {
                                "text": text,
                                "bbox": [
                                    float(bbox[0]),
                                    float(bbox[1]),
                                    float(bbox[2]),
                                    float(bbox[3]),
                                ],
                                "confidence": confidence,
                            }
                        )
                        continue

                for item in node:
                    stack.append(item)

        if not elements:
            raise RuntimeError("PaddleOCR returned no valid text elements")

        if scale_x != 1.0 or scale_y != 1.0:
            for el in elements:
                bbox = el.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = [float(v) for v in bbox]
                el["bbox"] = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]

        logger.info(
            "PaddleOCR extracted %s text elements from %s", len(elements), image_path
        )
        return elements


class LazyPaddleOcrClient(OcrProvider):
    """Lazy wrapper for local PaddleOCR fallback.

    Loading PaddleOCR can be expensive; explicit cloud OCR providers should not
    pay that startup cost unless fallback is actually needed.
    """

    def __init__(self, *, language: str = "ch"):
        self.language = _normalize_paddle_language(language)
        self._provider: PaddleOcrClient | None = None

    def _ensure_provider(self) -> PaddleOcrClient:
        if self._provider is None:
            self._provider = PaddleOcrClient(language=self.language)
        return self._provider

    def ocr_image(self, image_path: str) -> List[Dict]:
        return self._ensure_provider().ocr_image(image_path)


class OcrManager:
    """
    OCR manager with strict provider behavior.

    Policy:
    - If provider is explicitly `tesseract`/`local`, use Tesseract only.
    - For explicit providers (`aiocr`, `paddle`, `paddle_local`, `baidu`), strict
      mode keeps the provider pure. Non-strict mode may add local OCR fallbacks.
    - With `strict_no_fallback=True`, `auto` mode does not use local fallback
      providers; it requires AI OCR and fails fast on setup/runtime failures.
    - With `strict_no_fallback=False`, `auto` mode keeps hybrid fallback behavior.
    """

    def __init__(
        self,
        provider: str | None = None,
        *,
        route_kind: str | None = None,
        ai_provider: str | None = None,
        ai_api_key: str | None = None,
        ai_base_url: str | None = None,
        ai_model: str | None = None,
        ai_layout_model: str | None = None,
        paddle_doc_max_side_px: int | None = None,
        layout_block_max_concurrency: int | None = None,
        request_rpm_limit: int | None = None,
        request_tpm_limit: int | None = None,
        request_max_retries: int | None = None,
        baidu_app_id: str | None = None,
        baidu_api_key: str | None = None,
        baidu_secret_key: str | None = None,
        tesseract_min_confidence: float | None = None,
        tesseract_language: str | None = None,
        strict_no_fallback: bool = False,
        allow_paddle_model_downgrade: bool = False,
    ):
        """Initialize OCR manager with primary and fallback providers."""
        self.providers: list[OcrProvider] = []
        self.primary_provider: Optional[OcrProvider] = None
        self.fallback_provider: Optional[OcrProvider] = None
        self.last_provider_name: str | None = None
        self.last_provider_error: str | None = None
        self.last_fallback_reason: str | None = None
        self.last_quality_notes: list[str] = []
        self.last_image_regions: list[list[float]] = []
        self.provider_id: str = "auto"
        self.route_kind: str = ROUTE_KIND_HYBRID_AUTO
        self.strict_no_fallback: bool = bool(strict_no_fallback)
        self.allow_paddle_model_downgrade: bool = bool(allow_paddle_model_downgrade)
        self.ai_provider_disabled: bool = False
        self.ai_provider_disabled_reason: str | None = None
        # Keep typed references so `auto` mode can combine results.
        self.baidu_provider: BaiduOcrClient | None = None
        self.tesseract_provider: TesseractOcrClient | None = None
        self.paddle_provider: AiOcrClient | PaddleOcrClient | None = None
        self.paddle_local_fallback_provider: OcrProvider | None = None
        self.ai_provider: AiOcrClient | None = None

        provider_id = (provider or "auto").strip().lower()
        # Backward compatibility: legacy ids map to canonical provider names.
        if provider_id in {"remote", "ai"}:
            provider_id = "aiocr"
        if provider_id in {"paddle-local", "local_paddle"}:
            provider_id = "paddle_local"
        if provider_id not in {
            "auto",
            "aiocr",
            "baidu",
            "tesseract",
            "local",
            "paddle",
            "paddle_local",
        }:
            raise ValueError(f"Unsupported OCR provider: {provider_id}")
        self.provider_id = provider_id
        self.route_kind = normalize_ocr_route_kind(
            route_kind,
            default=(ROUTE_KIND_HYBRID_AUTO if provider_id == "auto" else "unknown"),
        )

        tesseract_min_conf = (
            float(tesseract_min_confidence)
            if tesseract_min_confidence is not None
            else None
        )
        # Prefer a bilingual default for scanned PDFs. Both language packs are
        # installed in the Docker image.
        tesseract_lang = (tesseract_language or "chi_sim+eng").strip() or "chi_sim+eng"

        def _maybe_add_tesseract_fallback(*, reason: str) -> None:
            """Add local Tesseract as a best-effort fallback provider.

            In strict mode we keep explicit providers "pure" (no implicit
            fallback). In non-strict mode, a Tesseract fallback makes explicit
            AI/cloud OCR options more reliable in open-source deployments.
            """

            if self.strict_no_fallback:
                return
            if self.tesseract_provider is not None:
                return
            try:
                self.tesseract_provider = TesseractOcrClient(
                    min_confidence=tesseract_min_conf or 50.0,
                    language=tesseract_lang,
                )
                self.providers.append(self.tesseract_provider)
                logger.info(
                    "Added Tesseract OCR as fallback provider (reason=%s)",
                    reason,
                )
            except Exception as e:
                logger.warning(
                    "Tesseract OCR fallback unavailable (reason=%s): %s",
                    reason,
                    e,
                )

        def _maybe_add_paddle_local_fallback(*, reason: str) -> None:
            """Add local PaddleOCR as a lazy fallback provider in non-strict mode."""

            if self.strict_no_fallback:
                return
            if any(
                isinstance(provider_obj, (PaddleOcrClient, LazyPaddleOcrClient))
                for provider_obj in self.providers
            ):
                return

            paddle_lang = "en" if tesseract_lang.strip().lower() == "eng" else "ch"
            try:
                self.paddle_local_fallback_provider = LazyPaddleOcrClient(
                    language=paddle_lang
                )
                self.providers.append(self.paddle_local_fallback_provider)
                logger.info(
                    "Added local PaddleOCR as lazy fallback provider (reason=%s, lang=%s)",
                    reason,
                    paddle_lang,
                )
            except Exception as e:
                logger.warning(
                    "Local PaddleOCR fallback unavailable (reason=%s): %s",
                    reason,
                    e,
                )

        if provider_id == "aiocr":
            if not ai_api_key:
                raise ValueError("AI OCR requires api_key")
            remote_spec = resolve_remote_ocr_client_spec(
                provider_id=provider_id,
                ai_provider=ai_provider,
                ai_base_url=ai_base_url,
                ai_model=ai_model,
                route_kind=route_kind,
            )
            self.route_kind = remote_spec.route_kind
            self.ai_provider = create_remote_ocr_client(
                requested_provider=provider_id,
                route_kind=route_kind,
                ai_provider=ai_provider,
                ai_api_key=ai_api_key,
                ai_base_url=ai_base_url,
                ai_model=ai_model,
                ai_layout_model=ai_layout_model,
                paddle_doc_max_side_px=paddle_doc_max_side_px,
                layout_block_max_concurrency=layout_block_max_concurrency,
                request_rpm_limit=request_rpm_limit,
                request_tpm_limit=request_tpm_limit,
                request_max_retries=request_max_retries,
                allow_paddle_model_downgrade=self.allow_paddle_model_downgrade,
            )
            self.providers.append(self.ai_provider)
            logger.info(
                "Using AI OCR as primary provider (route=%s, vendor=%s, model=%s, base_url=%s)",
                self.ai_provider.route_kind,
                self.ai_provider.provider_id,
                self.ai_provider.model,
                self.ai_provider.base_url or "<default>",
            )
            _maybe_add_tesseract_fallback(reason=remote_spec.route_kind)
            _maybe_add_paddle_local_fallback(reason=remote_spec.route_kind)
        if provider_id in {"baidu"}:
            self.baidu_provider = BaiduOcrClient(
                app_id=baidu_app_id,
                api_key=baidu_api_key,
                secret_key=baidu_secret_key,
            )
            self.providers.append(self.baidu_provider)
            logger.info("Using Baidu OCR (explicit)")
            _maybe_add_tesseract_fallback(reason="baidu")
            _maybe_add_paddle_local_fallback(reason="baidu")
        if provider_id in {"tesseract", "local"}:
            self.tesseract_provider = TesseractOcrClient(
                min_confidence=tesseract_min_conf or 50.0,
                language=tesseract_lang,
            )
            self.providers.append(self.tesseract_provider)
            logger.info("Using Tesseract OCR (explicit)")
        if provider_id == "paddle_local":
            paddle_lang = "en" if tesseract_lang.strip().lower() == "eng" else "ch"
            self.paddle_provider = PaddleOcrClient(language=paddle_lang)
            self.providers.append(self.paddle_provider)
            logger.info("Using local PaddleOCR (explicit, lang=%s)", paddle_lang)
            _maybe_add_tesseract_fallback(reason="paddle_local")
        if provider_id == "paddle":
            if not ai_api_key:
                raise ValueError("Paddle OCR requires api_key")
            remote_spec = resolve_remote_ocr_client_spec(
                provider_id=provider_id,
                ai_provider=ai_provider,
                ai_base_url=ai_base_url,
                ai_model=ai_model,
                route_kind=route_kind,
            )
            self.route_kind = remote_spec.route_kind
            self.paddle_provider = create_remote_ocr_client(
                requested_provider=provider_id,
                route_kind=route_kind,
                ai_provider=ai_provider,
                ai_api_key=ai_api_key,
                ai_base_url=ai_base_url,
                ai_model=ai_model,
                ai_layout_model=ai_layout_model,
                paddle_doc_max_side_px=paddle_doc_max_side_px,
                layout_block_max_concurrency=layout_block_max_concurrency,
                request_rpm_limit=request_rpm_limit,
                request_tpm_limit=request_tpm_limit,
                request_max_retries=request_max_retries,
                allow_paddle_model_downgrade=self.allow_paddle_model_downgrade,
            )
            self.providers.append(self.paddle_provider)
            logger.info(
                "Using PaddleOCR-VL as primary provider (route=%s, vendor=%s, model=%s, base_url=%s)",
                self.paddle_provider.route_kind,
                self.paddle_provider.provider_id,
                self.paddle_provider.model,
                self.paddle_provider.base_url or "<default>",
            )
            _maybe_add_tesseract_fallback(reason=remote_spec.route_kind)
            _maybe_add_paddle_local_fallback(reason=remote_spec.route_kind)

        if provider_id == "auto":
            if self.strict_no_fallback:
                if not ai_api_key:
                    raise RuntimeError(
                        "Strict OCR mode with provider=auto requires AI OCR credentials; "
                        "set ocr_provider=paddle/aiocr (recommended) or disable strict mode explicitly."
                    )

                remote_spec = resolve_remote_ocr_client_spec(
                    provider_id="aiocr",
                    ai_provider=ai_provider,
                    ai_base_url=ai_base_url,
                    ai_model=ai_model,
                    route_kind=route_kind,
                )
                self.route_kind = remote_spec.route_kind
                self.ai_provider = create_remote_ocr_client(
                    requested_provider="aiocr",
                    route_kind=route_kind,
                    ai_provider=ai_provider,
                    ai_api_key=ai_api_key,
                    ai_base_url=ai_base_url,
                    ai_model=ai_model,
                    ai_layout_model=ai_layout_model,
                    paddle_doc_max_side_px=paddle_doc_max_side_px,
                    layout_block_max_concurrency=layout_block_max_concurrency,
                    request_rpm_limit=request_rpm_limit,
                    request_tpm_limit=request_tpm_limit,
                    request_max_retries=request_max_retries,
                    allow_paddle_model_downgrade=self.allow_paddle_model_downgrade,
                )
                self.providers.append(self.ai_provider)
                logger.info(
                    "Using AI OCR as primary provider in strict auto mode (route=%s, vendor=%s, model=%s)",
                    self.ai_provider.route_kind,
                    self.ai_provider.provider_id,
                    self.ai_provider.model,
                )
            else:
                # Default behavior for scanned PDFs: prefer bbox-accurate machine OCR
                # for *geometry* (line bboxes), then optionally merge/refine with AI.
                try:
                    self.baidu_provider = BaiduOcrClient(
                        app_id=baidu_app_id,
                        api_key=baidu_api_key,
                        secret_key=baidu_secret_key,
                    )
                    self.providers.append(self.baidu_provider)
                    logger.info("Using Baidu OCR as primary provider")
                except (ValueError, ImportError) as e:
                    logger.warning("Baidu OCR not available: %s", e)

                # In auto mode, allow local Tesseract as fallback.
                try:
                    self.tesseract_provider = TesseractOcrClient(
                        min_confidence=tesseract_min_conf or 50.0,
                        language=tesseract_lang,
                    )
                    self.providers.append(self.tesseract_provider)
                    logger.info("Using Tesseract OCR as fallback provider in auto mode")
                except (ImportError, RuntimeError) as e:
                    logger.warning("Tesseract OCR not available in auto mode: %s", e)

                # In auto mode, allow local PaddleOCR as fallback.
                if not self.providers:
                    try:
                        self.paddle_provider = PaddleOcrClient()
                        self.providers.append(self.paddle_provider)
                        logger.info("Using PaddleOCR as fallback provider in auto mode")
                    except (ImportError, RuntimeError) as e:
                        logger.warning("PaddleOCR not available in auto mode: %s", e)

                try:
                    if ai_api_key:
                        remote_spec = resolve_remote_ocr_client_spec(
                            provider_id="aiocr",
                            ai_provider=ai_provider,
                            ai_base_url=ai_base_url,
                            ai_model=ai_model,
                            route_kind=route_kind,
                        )
                        self.ai_provider = create_remote_ocr_client(
                            requested_provider="aiocr",
                            route_kind=route_kind,
                            ai_provider=ai_provider,
                            ai_api_key=ai_api_key,
                            ai_base_url=ai_base_url,
                            ai_model=ai_model,
                            ai_layout_model=ai_layout_model,
                            paddle_doc_max_side_px=paddle_doc_max_side_px,
                            allow_paddle_model_downgrade=self.allow_paddle_model_downgrade,
                        )
                        self.providers.append(self.ai_provider)
                        logger.info(
                            "Using AI OCR as supplementary provider in auto mode (route=%s)",
                            self.ai_provider.route_kind,
                        )
                except Exception as e:
                    logger.warning("AI OCR not available: %s", e)

        if not self.providers:
            raise RuntimeError(
                "No OCR provider available. Install baidu-aip, pytesseract, or paddleocr."
            )

        self.primary_provider = self.providers[0]
        self.fallback_provider = self.providers[1] if len(self.providers) > 1 else None

    def ocr_image_lines(
        self, image_path: str, *, image_width: int, image_height: int
    ) -> list[dict]:
        """Return *line-level* OCR items (best-effort).

        In `auto` mode we combine available sources (for example
        Baidu / Tesseract / AI OCR) to reduce missed lines on scan-heavy PDFs.
        """

        W = int(image_width)
        H = int(image_height)
        self.last_quality_notes = []
        self.last_image_regions = []
        if W <= 0 or H <= 0:
            return []

        if self.provider_id != "auto":
            raw = self.ocr_image(image_path)
            # Providers like Baidu and AI OCR typically return line-level items
            # already. Re-merging can create huge paragraph-like boxes.
            if self.provider_id == "baidu":
                return _normalize_ocr_items_as_lines(raw, image_width=W, image_height=H)
            if self.provider_id in {"aiocr", "paddle"}:
                normalized = _normalize_ocr_items_as_lines(
                    raw, image_width=W, image_height=H
                )
                primary_model = None
                if self.provider_id == "aiocr" and self.ai_provider is not None:
                    primary_model = getattr(self.ai_provider, "model", None)
                elif self.provider_id == "paddle" and self.paddle_provider is not None:
                    primary_model = getattr(self.paddle_provider, "model", None)
                self.last_quality_notes = _build_primary_ocr_quality_notes(
                    normalized,
                    image_width=W,
                    image_height=H,
                    provider_name=self.last_provider_name,
                    model_name=primary_model,
                )

                # Defensive: some remote OCR models still return word-level
                # boxes even when prompted for line-level output. If we see a
                # very fragmented result, merge into line-level to keep PPT
                # shape count reasonable and improve wrap/size fitting.
                widths: list[float] = []
                heights: list[float] = []
                for it in normalized:
                    if not isinstance(it, dict):
                        continue
                    bbox_n = _normalize_bbox_px(it.get("bbox"))
                    if bbox_n is None:
                        continue
                    x0, y0, x1, y1 = bbox_n
                    w = float(x1 - x0)
                    h = float(y1 - y0)
                    if w > 0 and h > 0:
                        widths.append(w)
                        heights.append(h)

                allow_merge = False
                if widths and heights and len(widths) >= 140:
                    widths.sort()
                    heights.sort()
                    median_w = float(widths[len(widths) // 2])
                    median_h = float(heights[len(heights) // 2])
                    # Word-level output tends to have narrow boxes compared to
                    # page width and relative to glyph height.
                    if median_w <= max(0.18 * float(W), 2.9 * float(median_h)):
                        allow_merge = True

                if allow_merge:
                    return _merge_ocr_items_to_lines(
                        normalized,
                        image_width=W,
                        image_height=H,
                        allow_merge=True,
                    )
                return normalized
            if self.provider_id == "paddle_local":
                # PaddleOCR local output format varies across versions/models.
                # Some pipelines emit per-word boxes (very fragmented), which
                # leads to thousands of PPT shapes and poor line wrapping/font
                # fitting downstream. Detect this case and merge into
                # line-level boxes.
                widths: list[float] = []
                heights: list[float] = []
                for it in raw:
                    if not isinstance(it, dict):
                        continue
                    bbox_n = _normalize_bbox_px(it.get("bbox"))
                    if bbox_n is None:
                        continue
                    x0, y0, x1, y1 = bbox_n
                    w = float(x1 - x0)
                    h = float(y1 - y0)
                    if w > 0 and h > 0:
                        widths.append(w)
                        heights.append(h)

                allow_merge = False
                if widths and heights and len(widths) >= 80:
                    widths.sort()
                    heights.sort()
                    median_w = float(widths[len(widths) // 2])
                    median_h = float(heights[len(heights) // 2])
                    # Word-level output tends to have narrow boxes compared to
                    # the page width and relative to glyph height.
                    if median_w <= max(0.22 * float(W), 3.2 * float(median_h)):
                        allow_merge = True

                return _merge_ocr_items_to_lines(
                    raw,
                    image_width=W,
                    image_height=H,
                    allow_merge=allow_merge,
                )
            return _merge_ocr_items_to_lines(
                raw,
                image_width=W,
                image_height=H,
                allow_merge=False,
            )

        last_error: Exception | None = None
        baidu_lines: list[dict] = []
        tesseract_lines: list[dict] = []
        paddle_lines: list[dict] = []
        ai_lines: list[dict] = []
        ai_image_regions: list[list[float]] = []

        if self.baidu_provider is not None:
            try:
                raw_baidu = self.baidu_provider.ocr_image(image_path)
                baidu_lines = _normalize_ocr_items_as_lines(
                    raw_baidu, image_width=W, image_height=H
                )
            except Exception as e:
                last_error = e
                logger.warning("Baidu OCR failed (auto mode): %s", e)

        if self.tesseract_provider is not None:
            try:
                raw_tess = self.tesseract_provider.ocr_image(image_path)
                tesseract_lines = _merge_ocr_items_to_lines(
                    raw_tess,
                    image_width=W,
                    image_height=H,
                    allow_merge=False,
                )
            except Exception as e:
                last_error = e
                logger.warning("Tesseract OCR failed (auto mode): %s", e)

        if self.paddle_provider is not None:
            try:
                raw_paddle = self.paddle_provider.ocr_image(image_path)
                paddle_lines = _merge_ocr_items_to_lines(
                    raw_paddle,
                    image_width=W,
                    image_height=H,
                    allow_merge=True,
                )
            except Exception as e:
                last_error = e
                logger.warning("Paddle OCR failed (auto mode): %s", e)

        if self.ai_provider is not None:
            try:
                raw_ai = self.ai_provider.ocr_image(image_path)
                ai_image_regions = [
                    list(region)
                    for region in getattr(self.ai_provider, "last_image_regions_px", [])
                    if isinstance(region, list) and len(region) == 4
                ]
                ai_lines = _normalize_ocr_items_as_lines(
                    raw_ai, image_width=W, image_height=H
                )
            except Exception as e:
                last_error = e
                logger.warning("AI OCR failed (auto mode): %s", e)

        def _median_line_height(items: list[dict]) -> float:
            hs: list[float] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                bbox_n = _normalize_bbox_px(it.get("bbox"))
                if bbox_n is None:
                    continue
                _, y0, _, y1 = bbox_n
                h = float(y1 - y0)
                if h > 0:
                    hs.append(h)
            if not hs:
                return 0.0
            hs.sort()
            return max(1.0, float(hs[len(hs) // 2]))

        def _prune_ai_supplement(items: list[dict], *, baseline_h: float) -> list[dict]:
            """Drop likely coarse AI paragraph boxes when machine OCR exists."""

            out: list[dict] = []
            baseline_h = max(0.0, float(baseline_h))
            for it in items:
                if not isinstance(it, dict):
                    continue
                text = str(it.get("text") or "").strip()
                bbox_n = _normalize_bbox_px(it.get("bbox"))
                if not text or bbox_n is None:
                    continue
                if _is_probably_noise_line(text, bbox_n, image_width=W, image_height=H):
                    continue
                x0, y0, x1, y1 = bbox_n
                w = max(1.0, float(x1 - x0))
                h = max(1.0, float(y1 - y0))

                # Coarse paragraph-like boxes are harmful in hybrid mode:
                # they over-erase backgrounds and break text/image separation.
                if baseline_h > 0.0:
                    if h >= max(3.0 * baseline_h, 0.14 * float(H)) and (
                        w >= 0.20 * float(W) or len(text) >= 8
                    ):
                        continue
                    if w >= 0.90 * float(W) and h >= max(
                        1.8 * baseline_h, 0.08 * float(H)
                    ):
                        continue
                else:
                    if h >= 0.16 * float(H) and w >= 0.20 * float(W):
                        continue

                out.append({**it, "text": text, "bbox": [x0, y0, x1, y1]})

            out.sort(
                key=lambda it: ((it["bbox"][1] + it["bbox"][3]) / 2.0, it["bbox"][0])
            )
            return out

        machine_lines: list[dict] = []
        if baidu_lines:
            machine_lines.extend(baidu_lines)
        if tesseract_lines:
            machine_lines.extend(tesseract_lines)
        if paddle_lines:
            machine_lines.extend(paddle_lines)
        if ai_lines and machine_lines:
            machine_h = _median_line_height(machine_lines)
            ai_lines = _prune_ai_supplement(ai_lines, baseline_h=machine_h)

        # Merge available line lists in a preferred order.
        merged: list[dict] = []
        providers_used: list[str] = []

        def _merge_in(items: list[dict], label: str) -> None:
            nonlocal merged, providers_used
            if not items:
                return
            if not merged:
                merged = list(items)
                providers_used = [label]
                return
            merged = _merge_line_items_prefer_primary(
                merged, items, image_width=W, image_height=H
            )
            if label not in providers_used:
                providers_used.append(label)

        # Choose base ordering (machine OCR first for geometry, AI as supplement).
        if baidu_lines:
            _merge_in(baidu_lines, "Baidu")
        if tesseract_lines:
            _merge_in(tesseract_lines, "Tesseract")
        if paddle_lines:
            _merge_in(paddle_lines, "Paddle")
        if ai_lines:
            _merge_in(ai_lines, "AI")

        if merged:
            self.last_image_regions = [list(region) for region in ai_image_regions]
            self.last_provider_name = (
                f"HybridOcr({'+'.join(providers_used)})"
                if len(providers_used) > 1
                else (
                    "BaiduOcrClient"
                    if providers_used[0] == "Baidu"
                    else (
                        "TesseractOcrClient"
                        if providers_used[0] == "Tesseract"
                        else (
                            "PaddleOcrClient"
                            if providers_used[0] == "Paddle"
                            else "AiOcrClient"
                        )
                    )
                )
            )
            return merged

        # Defensive fallback: re-run AI OCR directly if all merged lists are empty.
        if self.ai_provider is not None:
            try:
                raw_ai = self.ai_provider.ocr_image(image_path)
                self.last_image_regions = [
                    list(region)
                    for region in getattr(self.ai_provider, "last_image_regions_px", [])
                    if isinstance(region, list) and len(region) == 4
                ]
                self.last_provider_name = "AiOcrClient"
                return _normalize_ocr_items_as_lines(
                    raw_ai, image_width=W, image_height=H
                )
            except Exception as e:
                last_error = e
                logger.warning("AI OCR failed (auto mode): %s", e)

        if self.paddle_provider is not None:
            try:
                raw_paddle = self.paddle_provider.ocr_image(image_path)
                self.last_provider_name = "PaddleOcrClient"
                return _merge_ocr_items_to_lines(
                    raw_paddle,
                    image_width=W,
                    image_height=H,
                    allow_merge=True,
                )
            except Exception as e:
                last_error = e
                logger.warning("Paddle OCR failed (auto mode): %s", e)

        raise RuntimeError("All OCR providers failed") from last_error

    def ocr_image(self, image_path: str) -> List[Dict]:
        """
        Perform OCR with automatic fallback.

        Args:
            image_path: Path to the image file

        Returns:
            List of text elements with bbox and confidence
        """
        last_error: Exception | None = None
        self.last_provider_error = None
        self.last_fallback_reason = None
        self.last_quality_notes = []
        self.last_image_regions = []
        for provider in self.providers:
            if self.ai_provider_disabled and isinstance(provider, AiOcrClient):
                continue
            try:
                out = provider.ocr_image(image_path)
                self.last_provider_name = provider.__class__.__name__
                self.last_image_regions = [
                    list(region)
                    for region in getattr(provider, "last_image_regions_px", [])
                    if isinstance(region, list) and len(region) == 4
                ]
                if isinstance(provider, AiOcrClient):
                    self.last_fallback_reason = None
                elif self.ai_provider_disabled:
                    self.last_fallback_reason = (
                        self.ai_provider_disabled_reason
                        or "aiocr_disabled_after_runtime_failure"
                    )
                return out
            except Exception as e:
                last_error = e
                self.last_provider_error = str(e)
                logger.warning(f"OCR provider failed: {str(e)}")
                if isinstance(provider, AiOcrClient) and not self.strict_no_fallback:
                    err = str(e).strip()
                    err_l = err.lower()
                    disable_markers = (
                        "ai ocr returned no items",
                        "ai ocr returned no parseable items",
                        "ai ocr returned empty",
                        "plain text without bbox",
                        "structural gibberish",
                        "timed out",
                        "timeout",
                    )
                    if any(marker in err_l for marker in disable_markers):
                        self.ai_provider_disabled = True
                        self.ai_provider_disabled_reason = (
                            f"aiocr_runtime_failure:{err or 'unknown'}"
                        )
                        logger.warning(
                            "Disabling AI OCR provider for remaining pages: %s",
                            err or "unknown",
                        )
                continue

        raise RuntimeError("All OCR providers failed") from last_error

    def detect_image_regions(self, image_path: str) -> list[list[float]]:
        if self.last_image_regions:
            return [list(region) for region in self.last_image_regions]

        if self.ai_provider_disabled:
            return []

        candidate_provider: OcrProvider | None = None
        if self.provider_id == "aiocr" and self.ai_provider is not None:
            candidate_provider = self.ai_provider
        elif self.provider_id == "paddle" and isinstance(
            self.paddle_provider, AiOcrClient
        ):
            candidate_provider = self.paddle_provider
        elif self.ai_provider is not None:
            candidate_provider = self.ai_provider
        elif isinstance(self.paddle_provider, AiOcrClient):
            candidate_provider = self.paddle_provider

        if candidate_provider is None:
            return []

        detect = getattr(candidate_provider, "detect_image_regions", None)
        if not callable(detect):
            return []

        try:
            regions = detect(image_path)
        except Exception as e:
            logger.warning("OCR image-region detection failed: %s", e)
            self.last_image_regions = []
            return []

        self.last_image_regions = [
            list(region)
            for region in regions
            if isinstance(region, list) and len(region) == 4
        ]
        return [list(region) for region in self.last_image_regions]

    def convert_bbox_to_pdf_coords(
        self,
        bbox: List[float],
        image_width: int,
        image_height: int,
        page_width_pt: float,
        page_height_pt: float,
    ) -> List[float]:
        """
        Convert OCR bounding box from image coordinates to PDF points.

        Args:
            bbox: [x0, y0, x1, y1] in image coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            page_width_pt: PDF page width in points
            page_height_pt: PDF page height in points

        Returns:
            [x0, y0, x1, y1] in PDF points
        """
        x0, y0, x1, y1 = bbox

        # Scale factors
        scale_x = page_width_pt / image_width
        scale_y = page_height_pt / image_height

        # Convert coordinates
        pdf_x0 = x0 * scale_x
        pdf_y0 = y0 * scale_y
        pdf_x1 = x1 * scale_x
        pdf_y1 = y1 * scale_y

        return [pdf_x0, pdf_y0, pdf_x1, pdf_y1]


def create_ocr_manager(
    provider: str | None = None,
    *,
    route_kind: str | None = None,
    ai_provider: str | None = None,
    ai_api_key: str | None = None,
    ai_base_url: str | None = None,
    ai_model: str | None = None,
    ai_layout_model: str | None = None,
    paddle_doc_max_side_px: int | None = None,
    layout_block_max_concurrency: int | None = None,
    request_rpm_limit: int | None = None,
    request_tpm_limit: int | None = None,
    request_max_retries: int | None = None,
    baidu_app_id: str | None = None,
    baidu_api_key: str | None = None,
    baidu_secret_key: str | None = None,
    tesseract_min_confidence: float | None = None,
    tesseract_language: str | None = None,
    strict_no_fallback: bool = False,
    allow_paddle_model_downgrade: bool = False,
) -> OcrManager:
    """
    Factory function to create OCR manager.

    Returns:
        Configured OcrManager instance
    """
    return OcrManager(
        provider=provider,
        route_kind=route_kind,
        ai_provider=ai_provider,
        ai_api_key=ai_api_key,
        ai_base_url=ai_base_url,
        ai_model=ai_model,
        ai_layout_model=ai_layout_model,
        paddle_doc_max_side_px=paddle_doc_max_side_px,
        layout_block_max_concurrency=layout_block_max_concurrency,
        request_rpm_limit=request_rpm_limit,
        request_tpm_limit=request_tpm_limit,
        request_max_retries=request_max_retries,
        baidu_app_id=baidu_app_id,
        baidu_api_key=baidu_api_key,
        baidu_secret_key=baidu_secret_key,
        tesseract_min_confidence=tesseract_min_confidence,
        tesseract_language=tesseract_language,
        strict_no_fallback=strict_no_fallback,
        allow_paddle_model_downgrade=allow_paddle_model_downgrade,
    )


def _clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _sample_text_color(image: Image.Image, bbox: List[float]) -> str:
    width, height = image.size
    if width <= 0 or height <= 0:
        return "#000000"

    x0, y0, x1, y1 = bbox
    x0 = _clamp_int(x0, 0, width - 1)
    y0 = _clamp_int(y0, 0, height - 1)
    x1 = _clamp_int(x1, 0, width - 1)
    y1 = _clamp_int(y1, 0, height - 1)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    # Estimate local background from samples just *outside* the bbox.
    pad = 3
    bg_points = [
        (x0 - pad, y0 - pad),
        (x1 + pad, y0 - pad),
        (x0 - pad, y1 + pad),
        (x1 + pad, y1 + pad),
        (x0 - pad, cy),
        (x1 + pad, cy),
        (cx, y0 - pad),
        (cx, y1 + pad),
    ]
    bg_samples: list[tuple[int, int, int]] = []
    for px, py in bg_points:
        px = _clamp_int(px, 0, width - 1)
        py = _clamp_int(py, 0, height - 1)
        r, g, b = image.getpixel((px, py))  # type: ignore[misc]
        bg_samples.append((int(r), int(g), int(b)))

    if not bg_samples:
        br, bg, bb = 255.0, 255.0, 255.0
    else:
        # Median is more robust than mean when the outside samples hit a glyph
        # or a nearby icon/highlight.
        rs = sorted(c[0] for c in bg_samples)
        gs = sorted(c[1] for c in bg_samples)
        bs = sorted(c[2] for c in bg_samples)
        mid = len(rs) // 2
        br, bg, bb = float(rs[mid]), float(gs[mid]), float(bs[mid])

    bg_luma = 0.2126 * br + 0.7152 * bg + 0.0722 * bb

    # Candidate "foreground" samples inside bbox. Prefer the most contrasting,
    # but make the result less noisy by averaging a few extreme samples.
    fg_points: list[tuple[int, int]] = []
    grid_x = 6
    grid_y = 4
    for gx in range(1, grid_x):
        for gy in range(1, grid_y):
            px = x0 + (x1 - x0) * gx // grid_x
            py = y0 + (y1 - y0) * gy // grid_y
            fg_points.append((int(px), int(py)))

    candidates: list[
        tuple[float, float, tuple[int, int, int]]
    ] = []  # (dist, luma, rgb)
    for px, py in fg_points:
        px = _clamp_int(px, 0, width - 1)
        py = _clamp_int(py, 0, height - 1)
        r, g, b = image.getpixel((px, py))  # type: ignore[misc]
        dist = (float(r) - br) ** 2 + (float(g) - bg) ** 2 + (float(b) - bb) ** 2
        luma = 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)
        candidates.append((dist, luma, (int(r), int(g), int(b))))

    if not candidates:
        return "#000000"

    # Keep only pixels that are meaningfully different from background.
    candidates.sort(key=lambda t: t[0], reverse=True)
    top = [c for c in candidates[:10] if c[0] >= 400.0]  # (>=20 rgb distance)
    if not top:
        top = candidates[:5]

    # If the background is light, text tends to be dark (lower luma), and vice
    # versa. Pick a few candidates consistent with that and average.
    if bg_luma >= 128.0:
        top.sort(key=lambda t: t[1])  # darker first
    else:
        top.sort(key=lambda t: t[1], reverse=True)  # lighter first

    chosen = top[:3] if len(top) >= 3 else top[:1]
    r = int(round(sum(c[2][0] for c in chosen) / len(chosen)))
    g = int(round(sum(c[2][1] for c in chosen) / len(chosen)))
    b = int(round(sum(c[2][2] for c in chosen) / len(chosen)))
    return _rgb_to_hex((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))


def _contains_cjk(text: str) -> bool:
    # Rough test: any character in common CJK blocks.
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
            or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
            or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
        ):
            return True
    return False


def _is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
        or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
    )


def _should_insert_space(prev: str, nxt: str) -> bool:
    if not prev or not nxt:
        return False
    if _contains_cjk(prev) or _contains_cjk(nxt):
        return False
    # Insert spaces for Latin words/numbers where OCR gives tokens without spaces.
    return prev[-1].isalnum() and nxt[0].isalnum()


def _normalize_bbox_px(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
    except Exception:
        return None
    if math.isnan(x0) or math.isnan(y0) or math.isnan(x1) or math.isnan(y1):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _merge_ocr_items_to_lines(
    items: list[dict],
    *,
    image_width: int,
    image_height: int,
    allow_merge: bool = True,
) -> list[dict]:
    """Merge word-level OCR items into line-level items.

    Many OCR engines return per-word boxes which creates thousands of PPT shapes.
    Merging improves editability and fidelity when masking over a background render.
    """

    if not items:
        return []

    # If items contain Tesseract's structural fields, merge by (block, paragraph,
    # line) first. This is significantly more stable than purely geometric
    # clustering for multi-column pages and tables.
    if any(
        isinstance(it, dict)
        and it.get("line_num") is not None
        and it.get("block_num") is not None
        for it in items
    ):
        words: list[dict] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            bbox_n = _normalize_bbox_px(it.get("bbox"))
            if not text or bbox_n is None:
                continue
            try:
                block_num = int(it.get("block_num") or 0)
                par_num = int(it.get("par_num") or 0)
                line_num = int(it.get("line_num") or 0)
                word_num = int(it.get("word_num") or 0)
            except Exception:
                continue

            x0, y0, x1, y1 = bbox_n
            if x1 <= x0 or y1 <= y0:
                continue
            # Clamp.
            x0 = max(0.0, min(x0, float(image_width - 1)))
            x1 = max(0.0, min(x1, float(image_width)))
            y0 = max(0.0, min(y0, float(image_height - 1)))
            y1 = max(0.0, min(y1, float(image_height)))
            if x1 <= x0 or y1 <= y0:
                continue

            words.append(
                {
                    "text": text,
                    "bbox": [x0, y0, x1, y1],
                    "confidence": float(it.get("confidence") or 0.0),
                    "block_num": block_num,
                    "par_num": par_num,
                    "line_num": line_num,
                    "word_num": word_num,
                }
            )

        if words:
            heights = sorted(max(1.0, it["bbox"][3] - it["bbox"][1]) for it in words)
            median_h = heights[len(heights) // 2] if heights else 10.0
            median_h = max(4.0, float(median_h))
            # Split "line" groups when a large horizontal gap is present.
            #
            # Tesseract can sometimes assign the same (block,par,line) to text
            # tokens that are on the same Y baseline but belong to different
            # visual regions (e.g. paragraph text + a nearby diagram label).
            # Using a slightly stricter gap threshold reduces these accidental
            # cross-region merges while keeping normal word spacing intact.
            gap_thresh = max(1.8 * median_h, 0.025 * float(image_width))

            groups: dict[tuple[int, int, int], list[dict]] = {}
            for w in words:
                key = (int(w["block_num"]), int(w["par_num"]), int(w["line_num"]))
                groups.setdefault(key, []).append(w)

            merged: list[dict] = []
            for group in groups.values():
                # Tesseract occasionally assigns the same (block,par,line) to
                # tokens from multiple visual lines (especially in dense
                # paragraphs on scanned slides). Before splitting by horizontal
                # gaps, we split by Y-center to avoid merging multiple lines
                # into a single tall paragraph-like box.
                def _y_center_word(it: dict) -> float:
                    y0, y1 = float(it["bbox"][1]), float(it["bbox"][3])
                    return (y0 + y1) / 2.0

                y_thresh = max(0.70 * float(median_h), 0.006 * float(image_height))

                by_y = sorted(
                    group, key=lambda it: (_y_center_word(it), float(it["bbox"][0]))
                )
                sublines: list[list[dict]] = []
                current: list[dict] = []
                current_y: float | None = None
                for it in by_y:
                    yc = _y_center_word(it)
                    if not current:
                        current = [it]
                        current_y = yc
                        continue
                    assert current_y is not None
                    if abs(float(yc) - float(current_y)) > y_thresh:
                        sublines.append(current)
                        current = [it]
                        current_y = yc
                    else:
                        n = len(current)
                        current.append(it)
                        current_y = (float(current_y) * float(n) + float(yc)) / float(
                            n + 1
                        )
                if current:
                    sublines.append(current)

                for line_words in sublines:
                    group_sorted = sorted(
                        line_words,
                        key=lambda it: (
                            int(it.get("word_num") or 0),
                            float(it["bbox"][0]),
                        ),
                    )

                    segment: list[dict] = []
                    prev = None
                    for it in group_sorted:
                        if not segment:
                            segment = [it]
                            prev = it
                            continue
                        assert prev is not None
                        gap = float(it["bbox"][0]) - float(prev["bbox"][2])
                        if gap > gap_thresh:
                            merged.append(_merge_segment(segment))
                            segment = [it]
                        else:
                            segment.append(it)
                        prev = it
                    if segment:
                        merged.append(_merge_segment(segment))

            merged.sort(
                key=lambda it: ((it["bbox"][1] + it["bbox"][3]) / 2.0, it["bbox"][0])
            )
            out: list[dict] = []
            for m in merged:
                if not isinstance(m, dict):
                    continue
                text = str(m.get("text") or "").strip()
                bbox_n = _normalize_bbox_px(m.get("bbox"))
                if not text or bbox_n is None:
                    continue
                if _is_probably_noise_line(
                    text,
                    bbox_n,
                    image_width=int(image_width),
                    image_height=int(image_height),
                ):
                    continue
                out.append(m)
            return out

    normalized: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        bbox_n = _normalize_bbox_px(it.get("bbox"))
        if not text or bbox_n is None:
            continue
        x0, y0, x1, y1 = bbox_n
        if x1 <= x0 or y1 <= y0:
            continue
        # Clamp.
        x0 = max(0.0, min(x0, float(image_width - 1)))
        x1 = max(0.0, min(x1, float(image_width - 1)))
        y0 = max(0.0, min(y0, float(image_height - 1)))
        y1 = max(0.0, min(y1, float(image_height - 1)))
        if x1 <= x0 or y1 <= y0:
            continue
        normalized.append(
            {
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "confidence": float(it.get("confidence") or 0.0),
            }
        )

    if not normalized:
        return []

    if not allow_merge:
        normalized.sort(
            key=lambda it: ((it["bbox"][1] + it["bbox"][3]) / 2.0, it["bbox"][0])
        )
        out_no_merge: list[dict] = []
        for it in normalized:
            bbox_n = _normalize_bbox_px(it.get("bbox"))
            if bbox_n is None:
                continue
            text_value = str(it.get("text") or "").strip()
            if not text_value:
                continue
            if _is_probably_noise_line(
                text_value,
                bbox_n,
                image_width=int(image_width),
                image_height=int(image_height),
            ):
                continue
            out_no_merge.append({**it, "bbox": list(bbox_n), "text": text_value})
        return out_no_merge

    heights = sorted(max(1.0, it["bbox"][3] - it["bbox"][1]) for it in normalized)
    median_h = heights[len(heights) // 2] if heights else 10.0
    median_h = max(4.0, float(median_h))

    def y_center(it: dict) -> float:
        y0, y1 = it["bbox"][1], it["bbox"][3]
        return (y0 + y1) / 2.0

    normalized.sort(key=lambda it: (y_center(it), it["bbox"][0]))

    # Band clustering by vertical proximity/overlap.
    #
    # IMPORTANT: scanned slides often have multiple "cards" (left/right columns)
    # whose text lines share similar Y ranges. Purely Y-based banding can merge
    # unrelated items across columns; a tall bbox on the right column can then
    # expand the band's Y range and accidentally merge multiple rows from the
    # left column (we observed this with Baidu OCR tokens in small tables).
    #
    # To keep line merging stable, we also gate band membership by horizontal
    # proximity (x-gap threshold).
    bands: list[list[dict]] = []
    band_stats: list[
        dict[str, float]
    ] = []  # min_y0, max_y1, min_x0, max_x1, center_y, n
    for it in normalized:
        x0, y0, x1, y1 = it["bbox"]
        yc = y_center(it)
        if not bands:
            bands.append([it])
            band_stats.append(
                {
                    "min_y0": float(y0),
                    "max_y1": float(y1),
                    "min_x0": float(x0),
                    "max_x1": float(x1),
                    "center_y": float(yc),
                    "n": 1.0,
                }
            )
            continue

        st = band_stats[-1]
        min_y0 = float(st.get("min_y0", 0.0))
        max_y1 = float(st.get("max_y1", 0.0))
        min_x0 = float(st.get("min_x0", 0.0))
        max_x1 = float(st.get("max_x1", 0.0))
        center_y = float(st.get("center_y", (min_y0 + max_y1) / 2.0))

        overlap = min(y1, max_y1) - max(y0, min_y0)
        band_h = max(1.0, max_y1 - min_y0)
        it_h = max(1.0, y1 - y0)

        # Horizontal gap between this item and the band's x-range.
        if x1 < min_x0:
            x_gap = float(min_x0 - x1)
        elif x0 > max_x1:
            x_gap = float(x0 - max_x1)
        else:
            x_gap = 0.0

        # Allow modest gaps for table columns (e.g. "label 70%"), but prevent
        # merging across distinct slide columns/cards.
        x_gap_thresh = max(0.04 * float(image_width), 6.0 * float(median_h))

        close = abs(float(yc) - center_y) <= 0.55 * float(median_h)
        same_line = (x_gap <= x_gap_thresh) and (
            close or (overlap >= 0.35 * min(band_h, it_h))
        )
        if same_line:
            bands[-1].append(it)
            st["min_y0"] = float(min(min_y0, y0))
            st["max_y1"] = float(max(max_y1, y1))
            st["min_x0"] = float(min(min_x0, x0))
            st["max_x1"] = float(max(max_x1, x1))
            n = int(float(st.get("n", 1.0) or 1.0))
            st["n"] = float(n + 1)
            st["center_y"] = float((center_y * n + float(yc)) / float(n + 1))
        else:
            bands.append([it])
            band_stats.append(
                {
                    "min_y0": float(y0),
                    "max_y1": float(y1),
                    "min_x0": float(x0),
                    "max_x1": float(x1),
                    "center_y": float(yc),
                    "n": 1.0,
                }
            )

    # Within each band, split by large horizontal gaps (multi-column / table cells).
    merged: list[dict] = []
    # Split segments on gaps that likely indicate a separate column/region.
    gap_thresh = max(1.8 * median_h, 0.025 * float(image_width))

    for band in bands:
        band_sorted = sorted(band, key=lambda it: it["bbox"][0])
        segment: list[dict] = []
        prev = None
        for it in band_sorted:
            if not segment:
                segment = [it]
                prev = it
                continue
            assert prev is not None
            gap = float(it["bbox"][0]) - float(prev["bbox"][2])
            if gap > gap_thresh:
                # Flush current segment.
                merged.append(_merge_segment(segment))
                segment = [it]
            else:
                segment.append(it)
            prev = it
        if segment:
            merged.append(_merge_segment(segment))

    # Filter empty merges.
    out: list[dict] = []
    for m in merged:
        if not isinstance(m, dict):
            continue
        text = str(m.get("text") or "").strip()
        bbox_n = _normalize_bbox_px(m.get("bbox"))
        if not text or bbox_n is None:
            continue
        if _is_probably_noise_line(
            text,
            bbox_n,
            image_width=int(image_width),
            image_height=int(image_height),
        ):
            continue
        out.append(m)
    return out


def _merge_segment(segment: list[dict]) -> dict:
    seg_sorted = sorted(segment, key=lambda it: it["bbox"][0])
    parts: list[str] = []
    prev_text = ""
    for it in seg_sorted:
        t = str(it.get("text") or "").strip()
        if not t:
            continue
        if parts and _should_insert_space(prev_text, t):
            parts.append(" ")
        parts.append(t)
        prev_text = t
    text = "".join(parts).strip()

    x0 = min(float(it["bbox"][0]) for it in seg_sorted)
    y0 = min(float(it["bbox"][1]) for it in seg_sorted)
    x1 = max(float(it["bbox"][2]) for it in seg_sorted)
    y1 = max(float(it["bbox"][3]) for it in seg_sorted)
    confs = [float(it.get("confidence") or 0.0) for it in seg_sorted]
    confidence = sum(confs) / len(confs) if confs else 0.0
    return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": confidence}


def _normalize_ocr_items_as_lines(
    items: list[dict],
    *,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Normalize OCR items that are already *line-level*.

    Some providers (notably Baidu's general/accurate OCR and many AI OCR
    prompts) output one item per visual line. Re-running the geometric merge
    step on such items can accidentally merge unrelated lines into huge boxes,
    which then causes over-masking and missing text in the generated PPT.
    """

    if not items:
        return []

    W = int(image_width)
    H = int(image_height)

    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        bbox_n = _normalize_bbox_px(it.get("bbox"))
        if not text or bbox_n is None:
            continue
        if _looks_like_ocr_prompt_echo_text(text):
            continue
        if _is_probably_noise_line(text, bbox_n, image_width=W, image_height=H):
            continue

        x0, y0, x1, y1 = bbox_n
        if x1 <= x0 or y1 <= y0:
            continue
        # Clamp to image bounds.
        x0 = max(0.0, min(x0, float(W - 1)))
        x1 = max(0.0, min(x1, float(W)))
        y0 = max(0.0, min(y0, float(H - 1)))
        y1 = max(0.0, min(y1, float(H)))
        if x1 <= x0 or y1 <= y0:
            continue

        out.append({**it, "text": text, "bbox": [x0, y0, x1, y1]})

    out.sort(key=lambda it: ((it["bbox"][1] + it["bbox"][3]) / 2.0, it["bbox"][0]))
    return out


def _build_primary_ocr_quality_notes(
    items: list[dict],
    *,
    image_width: int,
    image_height: int,
    provider_name: str | None,
    model_name: str | None,
) -> list[str]:
    """Emit lightweight quality notes when OCR output looks suspiciously coarse."""

    if str(provider_name or "") != "AiOcrClient":
        return []
    lowered_model = str(model_name or "").strip().lower()
    if "paddleocr-vl" not in lowered_model:
        return []
    if not items:
        return []

    W = max(1, int(image_width))
    H = max(1, int(image_height))
    if float(W) < (1.18 * float(H)):
        return []

    valid: list[tuple[tuple[float, float, float, float], str]] = []
    heights: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        bbox_n = _normalize_bbox_px(item.get("bbox"))
        if not text or bbox_n is None:
            continue
        x0, y0, x1, y1 = bbox_n
        if x1 <= x0 or y1 <= y0:
            continue
        compact = re.sub(r"\s+", "", text)
        valid.append((bbox_n, compact))
        heights.append(max(1.0, float(y1 - y0)))

    count = len(valid)
    if count < 4 or count > 18:
        return []

    heights.sort()
    median_h = heights[len(heights) // 2] if heights else max(10.0, 0.02 * float(H))
    median_h = max(8.0, float(median_h))

    large_boxes = 0
    small_boxes = 0
    right_boxes = 0
    for bbox_n, compact in valid:
        x0, y0, x1, y1 = bbox_n
        w = max(1.0, float(x1 - x0))
        h = max(1.0, float(y1 - y0))
        cx = (float(x0) + float(x1)) / 2.0

        if (
            h >= max(1.7 * median_h, 0.075 * float(H))
            or (w >= 0.22 * float(W) and len(compact) >= 20)
        ):
            large_boxes += 1
        if (
            w <= 0.18 * float(W)
            and h <= 0.08 * float(H)
            and len(compact) <= 24
        ):
            small_boxes += 1
        if cx >= 0.58 * float(W):
            right_boxes += 1

    if (
        large_boxes >= max(4, count - 2)
        and small_boxes <= 1
        and right_boxes <= 1
    ):
        return [
            "paddle_vl_sparse_slide_layout:"
            f" items={count}"
            f" large_boxes={large_boxes}"
            f" small_boxes={small_boxes}"
            f" right_boxes={right_boxes}"
        ]
    return []


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return float(inter) / float(max(1.0, union))


def _bbox_overlap_smaller(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return float(inter) / float(min(area_a, area_b))


def _normalize_text_for_dedupe(text: str) -> str:
    # Keep alnum/CJK, drop punctuation/whitespace for robust OCR text matching.
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())


def _texts_are_similar_for_dedupe(a: str, b: str) -> bool:
    na = _normalize_text_for_dedupe(a)
    nb = _normalize_text_for_dedupe(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        short = min(len(na), len(nb))
        long = max(len(na), len(nb))
        return short >= 3 and (float(short) / float(long)) >= 0.65
    return False


def _dedupe_overlapping_ocr_items(items: list[dict]) -> list[dict]:
    """Drop near-duplicate OCR items caused by multi-engine merge/refinement.

    For single-provider runs (for example pure AI OCR), we only remove exact-ish
    duplicates and keep potentially overlapping lines/paragraph splits. Aggressive
    overlap dedupe is used only for mixed-provider merges.
    """

    candidates: list[dict] = []
    providers_seen: set[str] = set()
    heights: list[float] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        bbox_n = _normalize_bbox_px(it.get("bbox"))
        if not text or bbox_n is None:
            continue
        x0, y0, x1, y1 = bbox_n
        if x1 <= x0 or y1 <= y0:
            continue
        conf = float(it.get("confidence") or 0.0)
        area = float((x1 - x0) * (y1 - y0))
        h = float(y1 - y0)
        heights.append(max(1.0, h))
        provider_name = (
            str(it.get("provider") or it.get("source") or "").strip().lower()
        )
        if provider_name:
            providers_seen.add(provider_name)
        candidates.append(
            {
                **it,
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "_bbox_t": (x0, y0, x1, y1),
                "_conf": conf,
                "_area": area,
                "_provider": provider_name,
                "_cx": float(x0 + x1) / 2.0,
                "_cy": float(y0 + y1) / 2.0,
                "_h": float(h),
            }
        )

    if len(candidates) <= 1:
        for it in candidates:
            it.pop("_bbox_t", None)
            it.pop("_conf", None)
            it.pop("_area", None)
            it.pop("_provider", None)
        return candidates

    # Prefer higher confidence, then smaller area (usually tighter line bbox).
    candidates.sort(key=lambda it: (-float(it["_conf"]), float(it["_area"])))

    multi_provider = len(providers_seen) >= 2
    heights.sort()
    median_h = float(heights[len(heights) // 2]) if heights else 10.0
    median_h = max(4.0, float(median_h))

    kept: list[dict] = []
    dropped = 0
    for cur in candidates:
        cur_bbox = cur["_bbox_t"]
        cur_text = str(cur.get("text") or "")
        cur_cx = float(cur.get("_cx") or 0.0)
        cur_cy = float(cur.get("_cy") or 0.0)
        duplicate = False
        for prev in kept:
            prev_bbox = prev["_bbox_t"]
            prev_cx = float(prev.get("_cx") or 0.0)
            prev_cy = float(prev.get("_cy") or 0.0)
            iou = _bbox_iou(cur_bbox, prev_bbox)
            overlap_small = _bbox_overlap_smaller(cur_bbox, prev_bbox)

            # Same-geometry duplicates can appear in malformed AI grounding output
            # (different text strings mapped to the exact same bbox). Keep only one.
            strong_same_bbox = overlap_small >= 0.985 and iou >= 0.90
            if strong_same_bbox:
                duplicate = True
                break

            # AI OCR (and some gateways) can also output near-identical boxes with
            # small jitter. Treat them as duplicates even if the text differs.
            near_same_bbox = overlap_small >= 0.965 and iou >= 0.85
            if near_same_bbox:
                duplicate = True
                break

            # Exact-ish duplicate candidate.
            exact_like = overlap_small >= 0.93 and _texts_are_similar_for_dedupe(
                cur_text, str(prev.get("text") or "")
            )
            if exact_like:
                duplicate = True
                break

            # In single-provider runs we are intentionally conservative, but we
            # still want to suppress obvious "same text, slightly shifted bbox"
            # duplicates which otherwise show up as stacked/offset glyphs.
            if not multi_provider:
                if _texts_are_similar_for_dedupe(cur_text, str(prev.get("text") or "")):
                    if overlap_small >= 0.85:
                        duplicate = True
                        break
                    # Some AI OCR engines (notably DeepSeek grounding outputs on
                    # gateways) can emit the same line twice with a slightly
                    # larger jitter (overlap ~0.70-0.85). Use a vertical-center
                    # guard to avoid deleting distinct nearby lines.
                    dy = abs(cur_cy - prev_cy)
                    if dy <= (0.55 * median_h) and (
                        overlap_small >= 0.70 or iou >= 0.55
                    ):
                        duplicate = True
                        break

            if multi_provider:
                # Only do aggressive overlap pruning for mixed-provider merges,
                # where stacked duplicate lines are common.
                if overlap_small >= 0.88 or iou >= 0.78:
                    duplicate = True
                    break
                if iou >= 0.62 and _texts_are_similar_for_dedupe(
                    cur_text, str(prev.get("text") or "")
                ):
                    duplicate = True
                    break

        if duplicate:
            dropped += 1
            continue
        kept.append(cur)

    if dropped > 0:
        logger.info("OCR dedupe dropped %s overlapping items", dropped)

    out: list[dict] = []
    for it in kept:
        cp = dict(it)
        cp.pop("_bbox_t", None)
        cp.pop("_conf", None)
        cp.pop("_area", None)
        cp.pop("_provider", None)
        cp.pop("_cx", None)
        cp.pop("_cy", None)
        cp.pop("_h", None)
        out.append(cp)

    # Stable reading order for downstream conversion.
    out.sort(key=lambda it: ((it["bbox"][1] + it["bbox"][3]) / 2.0, it["bbox"][0]))
    return out


def _is_probably_noise_line(
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> bool:
    t = str(text or "").strip()
    if not t:
        return True

    if _looks_like_ocr_prompt_echo_text(t):
        return True

    # Skip pure punctuation / dots (common false positives in scans).
    stripped = "".join(ch for ch in t if not ch.isspace())
    if stripped and all((not ch.isalnum()) for ch in stripped):
        return True
    if len(stripped) >= 6 and set(stripped) <= {"."}:
        return True

    cjk = _contains_cjk(t)
    has_digit = any(ch.isdigit() for ch in t)
    has_alpha = any(ch.isalpha() for ch in t)

    x0, y0, x1, y1 = bbox
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    area = w * h
    img_area = float(max(1, int(image_width) * int(image_height)))

    # Common acronyms that can appear as standalone tokens in slide decks.
    # We keep these even when other heuristics would treat them as noise.
    # Short Latin-only tokens inside small/odd boxes are frequently garbage
    # (icons/logos or screenshot UI chrome). However, 2-letter ALLCAPS tokens
    # like "AI" / "EF" can be meaningful abbreviations in decks, so we keep them
    # unless they are extremely tiny.
    if (not cjk) and (not has_digit) and has_alpha:
        alpha_only = "".join(ch for ch in stripped if ch.isalpha())
        if alpha_only and alpha_only.upper() in _ACRONYM_ALLOWLIST:
            return False

        if len(stripped) == 1:
            # Single-letter hits are almost always noise on scanned slides.
            if area / img_area < 0.002:
                return True
            if image_height > 0 and (h / float(image_height)) >= 0.08:
                return True
        elif len(stripped) == 2:
            if stripped.isupper():
                # Two-letter ALLCAPS tokens can be meaningful ("AI", "UI"), but
                # most random ones on scanned slides are icon false positives.
                # Keep a small allowlist and be stricter otherwise.
                min_area = (
                    0.00035 if stripped.upper() in _ACRONYM_ALLOWLIST else 0.00070
                )
                if area / img_area < min_area:
                    return True
                if image_height > 0 and (h / float(image_height)) >= 0.10:
                    return True
            else:
                if area / img_area < 0.0012:
                    return True
                # If the bbox is *very* tall relative to the page but contains
                # only 1-2 Latin letters, it is almost certainly an icon false
                # positive.
                if image_height > 0 and (h / float(image_height)) >= 0.08:
                    return True
        elif stripped.isupper() and 3 <= len(stripped) <= 4:
            # 3-4 uppercase tokens are often noise ("FRM", "GFE") produced by
            # icons / diagram strokes. Keep only if the bbox is reasonably large.
            if area / img_area < 0.0009:
                return True
            if image_height > 0 and (h / float(image_height)) >= 0.11:
                return True

    return False


def _filter_contextual_noise_items(
    items: list[dict],
    *,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Page-contextual OCR cleanup to reduce image-internal gibberish tokens.

    This is intentionally conservative and only applies stricter rules when the
    page is clearly CJK-dominant (common in scanned slides where icons/figures
    leak short Latin fragments like "T os", "RAN," etc.).
    """

    W = max(1, int(image_width))
    H = max(1, int(image_height))

    candidates: list[dict] = []
    cjk_chars = 0
    total_chars = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        bbox_n = _normalize_bbox_px(it.get("bbox"))
        if not text or bbox_n is None:
            continue
        candidates.append({**it, "text": text, "bbox": list(bbox_n)})
        for ch in text:
            if ch.isspace():
                continue
            total_chars += 1
            if _is_cjk_char(ch):
                cjk_chars += 1

    if not candidates:
        return []

    cjk_ratio = (float(cjk_chars) / float(total_chars)) if total_chars > 0 else 0.0
    cjk_dominant = cjk_ratio >= 0.32

    out: list[dict] = []
    for it in candidates:
        text = str(it.get("text") or "").strip()
        bbox = _normalize_bbox_px(it.get("bbox"))
        if not text or bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        w = max(1.0, float(x1 - x0))
        h = max(1.0, float(y1 - y0))
        area_ratio = (w * h) / float(W * H)

        stripped = "".join(ch for ch in text if not ch.isspace())
        alpha_only = "".join(ch for ch in stripped if ch.isalpha())
        has_alpha = bool(alpha_only)
        has_digit = any(ch.isdigit() for ch in stripped)
        has_cjk = any(_is_cjk_char(ch) for ch in stripped)
        conf = float(it.get("confidence") or 0.0)

        drop = False

        if cjk_dominant:
            if has_alpha and not has_digit:
                up = alpha_only.upper()
                # Very short Latin tokens on CJK pages are usually icon/UI noise.
                if len(alpha_only) <= 4 and up not in _ACRONYM_ALLOWLIST:
                    if area_ratio < 0.012:
                        drop = True
                # Lowercase short words are rarely meaningful in CJK titles/body.
                if len(alpha_only) <= 6 and alpha_only.islower() and area_ratio < 0.015:
                    drop = True
                # Long pure-Latin words on CJK-dominant pages are commonly
                # labels from embedded screenshots/diagrams (e.g. "Probability").
                # Keep larger heading-like words, drop tiny ones.
                if (not has_cjk) and len(alpha_only) >= 7 and area_ratio < 0.0035:
                    drop = True
                # Mixed short CJK+Latin fragments like "它crt".
                if has_cjk and len(stripped) <= 7 and len(alpha_only) <= 4:
                    drop = True

            # Repetitive ultra-short CJK fragments like "一国一一".
            if (not has_alpha) and has_cjk and len(stripped) <= 4:
                freq: dict[str, int] = {}
                for ch in stripped:
                    freq[ch] = freq.get(ch, 0) + 1
                max_freq = max(freq.values()) if freq else 0
                if max_freq >= max(3, len(stripped) - 1):
                    drop = True

            # Small mixed alpha+digit snippets on CJK pages are usually from
            # UI fragments in screenshots (e.g. "worst70%", "A1", "x3.2").
            if has_alpha and has_digit and (not has_cjk):
                if len(stripped) <= 14 and area_ratio < 0.0040:
                    drop = True

            # Tiny numeric-only fragments (e.g. "5%", "14%") are often chart
            # labels inside image regions and should not become editable text.
            if has_digit and (not has_alpha) and (not has_cjk):
                if len(stripped) <= 5 and area_ratio < 0.0015:
                    drop = True

        # Confidence-aware cleanup for tiny non-CJK snippets.
        if (not has_cjk) and len(stripped) <= 8 and conf > 0.0 and conf < 0.45:
            if area_ratio < 0.02:
                drop = True

        if not drop:
            out.append({**it, "bbox": [x0, y0, x1, y1]})

    out.sort(key=lambda it: ((it["bbox"][1] + it["bbox"][3]) / 2.0, it["bbox"][0]))
    return out


def _merge_line_items_prefer_primary(
    primary: list[dict],
    secondary: list[dict],
    *,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Merge two *line-level* OCR item lists.

    We keep all primary items and only add secondary items that do not overlap
    meaningfully with any primary bbox. This improves recall without producing
    duplicate lines.
    """

    W = int(image_width)
    H = int(image_height)

    prim: list[dict] = []
    prim_boxes: list[tuple[float, float, float, float]] = []
    prim_heights: list[float] = []

    for it in primary:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        bbox_n = _normalize_bbox_px(it.get("bbox"))
        if not text or bbox_n is None:
            continue
        if _is_probably_noise_line(text, bbox_n, image_width=W, image_height=H):
            continue
        x0, y0, x1, y1 = bbox_n
        if x1 <= x0 or y1 <= y0:
            continue
        prim.append({**it, "text": text, "bbox": [x0, y0, x1, y1]})
        prim_boxes.append((x0, y0, x1, y1))
        prim_heights.append(max(1.0, y1 - y0))

    prim_heights.sort()
    median_prim_h = prim_heights[len(prim_heights) // 2] if prim_heights else 10.0
    median_prim_h = max(4.0, float(median_prim_h))

    out: list[dict] = list(prim)

    def _matches_primary(bbox: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        for pb in prim_boxes:
            iou = _bbox_iou(bbox, pb)
            # Use a slightly stricter IoU threshold so we don't incorrectly
            # treat nearby-but-distinct lines as duplicates.
            if iou >= 0.45:
                return True
            px0, py0, px1, py1 = pb
            p_w = max(1.0, px1 - px0)
            p_h = max(1.0, py1 - py0)
            s_w = max(1.0, x1 - x0)
            s_h = max(1.0, y1 - y0)

            # Center-in-box match is helpful for minor jitter, but it's also
            # very aggressive when the primary box is abnormally large (e.g.
            # paragraph-level). In those cases we avoid suppressing secondary
            # lines which may contain the missing text geometry.
            primary_is_reasonable_line = p_h <= (2.2 * median_prim_h) and p_w <= (
                0.98 * float(W)
            )
            secondary_is_reasonable_line = s_h <= (2.6 * median_prim_h)
            if primary_is_reasonable_line and secondary_is_reasonable_line:
                if (
                    cx >= (px0 - 2.0)
                    and cx <= (px1 + 2.0)
                    and cy >= (py0 - 2.0)
                    and cy <= (py1 + 2.0)
                ):
                    return True
            # High overlap relative to the smaller box.
            ix0 = max(x0, px0)
            iy0 = max(y0, py0)
            ix1 = min(x1, px1)
            iy1 = min(y1, py1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            area_s = max(1.0, (x1 - x0) * (y1 - y0))
            area_p = max(1.0, (px1 - px0) * (py1 - py0))
            if inter >= 0.85 * float(min(area_s, area_p)):
                return True
        return False

    for it in secondary:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        bbox_n = _normalize_bbox_px(it.get("bbox"))
        if not text or bbox_n is None:
            continue
        if _is_probably_noise_line(text, bbox_n, image_width=W, image_height=H):
            continue
        if _matches_primary(bbox_n):
            continue
        x0, y0, x1, y1 = bbox_n
        out.append({**it, "text": text, "bbox": [x0, y0, x1, y1]})

    # Stable reading order.
    out.sort(key=lambda it: ((it["bbox"][1] + it["bbox"][3]) / 2.0, it["bbox"][0]))
    return out


def ocr_image_to_elements(
    image_path: str,
    *,
    page_width_pt: float,
    page_height_pt: float,
    ocr_manager: OcrManager,
    text_refiner: AiOcrTextRefiner | None = None,
    linebreak_refiner: AiOcrTextRefiner | None = None,
    strict_no_fallback: bool = True,
    linebreak_assist: bool | None = None,
) -> List[Dict]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        return []

    def _split_text_into_n_lines(text: str, *, n: int) -> list[str] | None:
        """Heuristically split a paragraph into N lines (no OCR re-run).

        This is a best-effort fallback used when we can *see* multi-line ink
        in a bbox (via projection), but do not have an AI vision model to
        split the text accurately. The main goal is layout fidelity (line
        count + approximate balance) rather than perfect linguistic wrapping.
        """

        n = int(n)
        raw = str(text or "").strip()
        if n <= 1 or not raw:
            return None

        # Punctuation line-breaking guards. We do not want a line to *start*
        # with closing punctuation (e.g. "：" or "）") because it is visually
        # jarring and causes obvious layout drift in PPT output.
        NO_BREAK_BEFORE = set(",.;:!?)]}、，。！？：；）】」』》〉%‰°")
        NO_BREAK_AFTER = set("([{（《【「『“‘")

        def _fix_punctuation_breaks(lines: list[str]) -> list[str]:
            if len(lines) <= 1:
                return lines

            out = [str(seg or "") for seg in lines]
            for _ in range(3):
                changed = False
                for i in range(1, len(out)):
                    prev = out[i - 1]
                    cur = out[i]
                    if not prev or not cur:
                        continue

                    # If current line begins with forbidden punctuation, move it
                    # to the end of previous line.
                    while cur and cur[0] in NO_BREAK_BEFORE and prev:
                        prev = prev + cur[0]
                        cur = cur[1:].lstrip()
                        changed = True
                        if not cur:
                            break

                    # If previous line ends with an opening punctuation, move it
                    # to the start of current line.
                    while prev and prev[-1] in NO_BREAK_AFTER and cur:
                        cur = prev[-1] + cur
                        prev = prev[:-1].rstrip()
                        changed = True
                        if not prev:
                            break

                    out[i - 1] = prev
                    out[i] = cur

                if not changed:
                    break

            return [seg for seg in (s.strip() for s in out) if seg]

        # If the upstream provider already inserted line breaks, do not
        # override them here.
        if "\n" in raw:
            lines = [seg.strip() for seg in raw.splitlines() if seg.strip()]
            if len(lines) >= 2:
                return _fix_punctuation_breaks(lines)
            return None

        is_cjk = _contains_cjk(raw)

        # Prefer word-level split when there is whitespace and we are not on a
        # CJK-heavy string.
        if (not is_cjk) and re.search(r"\s", raw):
            words = [w for w in re.split(r"\s+", raw) if w]
            if len(words) <= 1:
                return None
            total_chars = sum(len(w) for w in words) + max(0, len(words) - 1)
            target = max(1.0, float(total_chars) / float(n))
            lines: list[str] = []
            cur: list[str] = []
            cur_len = 0

            def _flush() -> None:
                nonlocal cur, cur_len
                if cur:
                    lines.append(" ".join(cur).strip())
                cur = []
                cur_len = 0

            for word in words:
                add_len = len(word) + (1 if cur else 0)
                if (
                    lines
                    and len(lines) < (n - 1)
                    and cur
                    and (float(cur_len + add_len) >= (1.12 * target))
                ):
                    _flush()
                cur.append(word)
                cur_len += add_len
            _flush()

            if len(lines) == n and all(lines):
                return _fix_punctuation_breaks(lines)
            # Try to rebalance by splitting the longest line(s).
            while len(lines) < n:
                longest_idx = max(range(len(lines)), key=lambda i: len(lines[i]))
                parts = lines[longest_idx].split()
                if len(parts) <= 1:
                    break
                mid = max(1, len(parts) // 2)
                left = " ".join(parts[:mid]).strip()
                right = " ".join(parts[mid:]).strip()
                if not left or not right:
                    break
                lines[longest_idx : longest_idx + 1] = [left, right]

            if len(lines) == n and all(lines):
                return _fix_punctuation_breaks(lines)
            return None

        # CJK or compact text: split by character count with punctuation-aware cuts.
        compact = re.sub(r"\s+", "", raw)
        if len(compact) < max(4, n * 2):
            return None

        break_chars = set("，。、；：！？,.!?:;）)】]》>、")
        breakpoints = [
            idx + 1
            for idx, ch in enumerate(compact)
            if ch in break_chars and idx + 1 < len(compact)
        ]
        target = float(len(compact)) / float(n)
        cuts: list[int] = []
        last = 0
        for k in range(1, n):
            ideal = int(round(float(k) * target))
            ideal = max(last + 1, min(len(compact) - 1, ideal))
            chosen = ideal
            # Pick a nearby punctuation breakpoint when available.
            if breakpoints:
                candidates = [
                    p for p in breakpoints if (last + 1) <= p <= (len(compact) - 1)
                ]
                if candidates:
                    nearest = min(candidates, key=lambda p: abs(p - ideal))
                    if abs(nearest - ideal) <= max(2, int(round(0.45 * target))):
                        chosen = nearest
            chosen = max(last + 1, min(len(compact) - 1, chosen))
            cuts.append(chosen)
            last = chosen

        parts: list[str] = []
        start = 0
        for cut in cuts + [len(compact)]:
            seg = compact[start:cut].strip()
            if seg:
                parts.append(seg)
            start = cut

        if len(parts) != n or not all(parts):
            return None
        return _fix_punctuation_breaks(parts)

    def _estimate_line_ranges_by_ink(
        bbox_n: tuple[float, float, float, float],
        *,
        typical_line_height: float,
        max_lines: int,
    ) -> list[tuple[float, float]] | None:
        """Estimate per-line vertical ranges using ink projection inside a bbox."""

        try:
            import numpy as np
        except Exception:
            return None

        x0, y0, x1, y1 = bbox_n
        W = int(width)
        H = int(height)

        xi0 = max(0, min(W - 1, int(math.floor(float(x0)))))
        yi0 = max(0, min(H - 1, int(math.floor(float(y0)))))
        xi1 = max(0, min(W, int(math.ceil(float(x1)))))
        yi1 = max(0, min(H, int(math.ceil(float(y1)))))
        if xi1 - xi0 < 6 or yi1 - yi0 < 10:
            return None

        try:
            gray = image.crop((xi0, yi0, xi1, yi1)).convert("L")
            arr = np.asarray(gray, dtype=np.float32)
        except Exception:
            return None

        if arr.ndim != 2 or arr.size <= 0:
            return None
        h_px, w_px = arr.shape
        if h_px < 10 or w_px < 6:
            return None

        p95 = float(np.percentile(arr, 95.0))
        p10 = float(np.percentile(arr, 10.0))
        contrast = max(1.0, p95 - p10)
        if contrast < 8.0:
            return None

        ink = np.clip((p95 - arr) / contrast, 0.0, 1.0)
        ink_mask = (ink >= 0.16).astype(np.float32)
        row_profile = ink_mask.mean(axis=1)
        if float(np.sum(row_profile)) <= max(0.02 * h_px, 1.0):
            return None

        k = max(1, int(round(h_px / 54.0)))
        if k > 1:
            kernel = np.ones((k,), dtype=np.float32) / float(k)
            smooth = np.convolve(row_profile, kernel, mode="same")
        else:
            smooth = row_profile

        # Use an adaptive threshold: above this value we consider the row part
        # of a text line. Keep a floor to avoid missing very light text.
        th = float(np.percentile(smooth, 70.0))
        th = max(0.055, min(0.20, th))

        active = smooth >= th
        segments: list[tuple[int, int]] = []
        start: int | None = None
        for idx, on in enumerate(active.tolist()):
            if on and start is None:
                start = idx
            elif (not on) and start is not None:
                segments.append((start, idx))
                start = None
        if start is not None:
            segments.append((start, h_px))

        if not segments:
            return None

        min_seg_h = max(2, int(round(0.25 * float(typical_line_height))))
        filtered: list[tuple[int, int]] = []
        for s, e in segments:
            if e - s < min_seg_h:
                continue
            filtered.append((s, e))
        segments = filtered
        if len(segments) < 2:
            return None

        # Merge segments separated by tiny gaps (diacritics / punctuation noise).
        merge_gap = max(1, int(round(0.22 * float(typical_line_height))))
        merged: list[tuple[int, int]] = []
        cur_s, cur_e = segments[0]
        for s, e in segments[1:]:
            if s - cur_e <= merge_gap:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        segments = merged

        if len(segments) < 2:
            return None
        if len(segments) > max(2, int(max_lines)):
            return None

        ranges: list[tuple[float, float]] = []
        prev_y = float(y0)
        for s, e in segments:
            ly0 = float(y0) + float(s)
            ly1 = float(y0) + float(e)
            # Clamp and enforce monotonic.
            ly0 = max(float(y0), min(float(y1) - 1.0, ly0))
            ly1 = max(ly0 + 1.0, min(float(y1), ly1))
            if ly0 < prev_y:
                ly0 = prev_y
            if ly1 <= ly0:
                continue
            ranges.append((ly0, ly1))
            prev_y = ly1

        if len(ranges) < 2:
            return None
        return ranges

    def _heuristic_assist_line_breaks(items: list[dict], *, force: bool) -> list[dict]:
        if not items:
            return items

        heights: list[float] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            bbox_n = _normalize_bbox_px(it.get("bbox"))
            if bbox_n is None:
                continue
            _, y0, _, y1 = bbox_n
            h = float(y1 - y0)
            if h > 0:
                heights.append(h)
        if heights:
            heights.sort()
            # Use a lower quantile to avoid paragraph boxes dominating the median.
            q_idx = int(round(0.35 * float(len(heights) - 1)))
            typical_h = max(4.0, float(heights[max(0, min(len(heights) - 1, q_idx))]))
        else:
            typical_h = max(10.0, 0.02 * float(height))

        max_lines = 8
        split_count = 0
        out: list[dict] = []

        candidates: list[tuple[int, dict, tuple[float, float, float, float], str]] = []
        for idx, original in enumerate(items):
            if not isinstance(original, dict):
                continue
            text = str(original.get("text") or "").strip()
            bbox_n = _normalize_bbox_px(original.get("bbox"))
            if not text or bbox_n is None:
                continue
            if "\n" in text:
                continue
            if _is_multiline_candidate_for_linebreak_assist(
                text=text,
                bbox=bbox_n,
                image_width=int(width),
                image_height=int(height),
                median_line_height=float(typical_h),
            ):
                candidates.append((idx, original, bbox_n, text))

        # Auto-mode guard: only apply the heuristic when we have enough strong
        # multiline candidates to justify splitting. This avoids accidentally
        # splitting a small number of tall headings on otherwise line-level OCR.
        if (not force) and len(candidates) < max(
            3, int(round(0.18 * float(len(items))))
        ):
            return items

        candidate_by_idx: dict[
            int, tuple[dict, tuple[float, float, float, float], str]
        ] = {int(idx): (orig, bb, txt) for idx, orig, bb, txt in candidates}

        for idx, original in enumerate(items):
            if not isinstance(original, dict):
                continue
            cand = candidate_by_idx.get(int(idx))
            if cand is None:
                out.append(dict(original))
                continue
            cand_original, bbox_n, text = cand

            x0, y0, x1, y1 = bbox_n
            box_h = max(1.0, float(y1 - y0))

            ranges = _estimate_line_ranges_by_ink(
                bbox_n,
                typical_line_height=float(typical_h),
                max_lines=max_lines,
            )

            n_lines = 0
            if ranges is not None:
                n_lines = len(ranges)
            else:
                est = int(round(box_h / max(1.0, float(typical_h))))
                n_lines = max(1, min(max_lines, est))
                if n_lines < 2:
                    out.append(dict(original))
                    continue
                total_h = float(y1 - y0)
                ranges = [
                    (
                        float(y0) + total_h * float(i) / float(n_lines),
                        float(y0) + total_h * float(i + 1) / float(n_lines),
                    )
                    for i in range(n_lines)
                ]

            if ranges is None or len(ranges) < 2:
                out.append(dict(cand_original))
                continue

            # Text split fallback: balance text across detected lines.
            lines = _split_text_into_n_lines(text, n=len(ranges))
            if not lines or len(lines) != len(ranges):
                out.append(dict(cand_original))
                continue

            for (ly0, ly1), text_line in zip(ranges, lines):
                cleaned_line = str(text_line or "").strip()
                if not cleaned_line:
                    continue
                if float(ly1 - ly0) < 1.0:
                    continue
                new_item = dict(cand_original)
                new_item["text"] = cleaned_line
                new_item["bbox"] = [float(x0), float(ly0), float(x1), float(ly1)]
                new_item["linebreak_assisted"] = True
                new_item["linebreak_assist_source"] = "heuristic"
                out.append(new_item)

            split_count += 1

        if split_count > 0:
            logger.info(
                "Heuristic line-break assist applied (no AI): split_boxes=%s/%s",
                split_count,
                len(items),
            )
        return out

    elements: List[Dict] = []
    merged_items = ocr_manager.ocr_image_lines(
        image_path, image_width=width, image_height=height
    )
    last_provider_name = str(getattr(ocr_manager, "last_provider_name", "") or "")
    provider_id = str(getattr(ocr_manager, "provider_id", "") or "").lower()
    ai_primary_fallback_mode = provider_id in {"aiocr", "paddle"}
    ai_provider_used_for_page = last_provider_name == "AiOcrClient"
    skip_ai_refiners_for_page = ai_primary_fallback_mode and not ai_provider_used_for_page
    effective_linebreak_refiner = (
        None if skip_ai_refiners_for_page else linebreak_refiner
    )
    effective_text_refiner = None if skip_ai_refiners_for_page else text_refiner

    if (
        effective_linebreak_refiner is not None
        and merged_items
        and linebreak_assist is True
    ):
        try:
            merged_items = effective_linebreak_refiner.assist_line_breaks(
                image_path,
                items=merged_items,
                # Heuristic line splitting is a local layout post-process, not
                # an OCR provider fallback. Keep it available even in strict
                # mode so coarse boxes from doc-oriented OCR models do not slip
                # through unchanged when the visual refiner returns no splits.
                allow_heuristic_fallback=True,
            )
        except Exception as e:
            logger.warning("AI OCR line-break assist failed: %s", e)
    elif linebreak_assist is True and merged_items:
        # Fallback: when user requests line-break assist (or backend auto-enabled
        # it) but no AI vision refiner is available, split coarse paragraph-like
        # boxes using pixel projection + text balancing. This is much better
        # than letting PPT guess wraps, and keeps the pipeline usable in fully
        # open-source deployments.
        try:
            merged_items = _heuristic_assist_line_breaks(merged_items, force=True)
        except Exception as e:
            logger.warning("Heuristic line-break assist failed: %s", e)
    elif (
        merged_items
        and linebreak_assist is None
        and (not strict_no_fallback)
        and effective_linebreak_refiner is None
    ):
        # Auto best-effort: AI OCR and some gateways return paragraph-like boxes
        # even when the user didn't enable explicit line-break assist. In
        # non-strict mode we can try a conservative heuristic split to reduce
        # wrap drift in PPT output.
        try:
            provider_id = str(getattr(ocr_manager, "provider_id", "") or "").lower()
            last_provider = str(getattr(ocr_manager, "last_provider_name", "") or "")
            should_try = (
                provider_id in {"aiocr", "paddle"} or last_provider == "AiOcrClient"
            )
            if should_try:
                merged_items = _heuristic_assist_line_breaks(merged_items, force=False)
        except Exception as e:
            logger.warning("Auto heuristic line-break assist failed: %s", e)

    if (
        effective_text_refiner is not None
        and merged_items
        and last_provider_name != "AiOcrClient"
    ):
        try:
            merged_items = effective_text_refiner.refine_items(
                image_path, items=merged_items
            )
        except Exception as e:
            logger.warning("AI OCR text refinement failed: %s", e)
    # Multi-engine merge + AI refinement can still leave near-identical line boxes.
    # Deduplicate here to prevent stacked text boxes in PPT output.
    merged_items = _dedupe_overlapping_ocr_items(merged_items)
    merged_items = _filter_contextual_noise_items(
        merged_items, image_width=width, image_height=height
    )
    for item in merged_items:
        bbox = item.get("bbox")
        text = str(item.get("text") or "").strip()
        if not bbox or not text:
            continue

        try:
            bbox_pt = ocr_manager.convert_bbox_to_pdf_coords(
                bbox=bbox,
                image_width=width,
                image_height=height,
                page_width_pt=page_width_pt,
                page_height_pt=page_height_pt,
            )
        except Exception:
            continue

        elements.append(
            {
                "type": "text",
                "bbox_pt": bbox_pt,
                "text": text,
                "confidence": item.get("confidence"),
                "source": "ocr",
                "color": _sample_text_color(image, bbox),
                # Lightweight provenance for downstream QA/dedupe (no secrets).
                "ocr_provider": item.get("provider") or item.get("source"),
                "ocr_model": item.get("model"),
                "ocr_linebreak_assisted": bool(item.get("linebreak_assisted")),
                "ocr_linebreak_assist_source": item.get("linebreak_assist_source"),
            }
        )

    return elements
