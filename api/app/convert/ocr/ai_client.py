"""AI OCR client and text refinement utilities."""

import contextvars
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Dict, List

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from PIL import Image

from .base import (
    _DEFAULT_PADDLE_OCR_VL_MODEL,
    _PADDLE_OCR_VL_MODEL_V1,
    _PADDLE_OCR_VL_MODEL_V15,
    _clean_str,
    _env_flag,
    _env_float,
    _is_probably_model_unsupported_error,
    _normalize_paddle_doc_backend,
    _normalize_paddle_doc_server_url,
    _resolve_paddle_doc_model_and_pipeline,
    _run_in_daemon_thread_with_timeout,
    OcrProvider,
)
from .deepseek_parser import (
    _extract_deepseek_tagged_items,
    _is_deepseek_ocr_model,
    _looks_like_ocr_prompt_echo_text,
)
from .json_extraction import (
    _extract_json_list,
    _extract_message_text,
    _extract_partial_json_object_list,
)
from .result_parsing import (
    _derive_paddle_doc_predict_max_pixels,
    _extract_deepseek_image_regions,
    _extract_image_regions_json,
    _extract_paddle_doc_parser_output,
    _normalize_bbox_px,
    _scale_paddle_doc_parser_output,
)
from .utils import (
    _coerce_bbox_xyxy,
    _is_paddleocr_vl_model,
    _looks_like_structural_gibberish,
)
from .vendors import (
    _create_ai_ocr_vendor_adapter,
    _normalize_ai_ocr_model_name,
    _should_send_image_first_for_ai_ocr,
)

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(str(raw).strip())
    except Exception:
        return int(default)
    return int(value)


class AiOcrClient(OcrProvider):
    """AI OCR using OpenAI-compatible vision models."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        paddle_doc_max_side_px: int | None = None,
    ):
        import openai

        if not api_key:
            raise ValueError("AI OCR api_key is required")

        self.api_key = str(api_key).strip()
        self._paddle_doc_parser: Any | None = None
        self._paddle_doc_parser_disabled: bool = False
        self._paddle_doc_effective_model: str | None = None
        self._paddle_doc_pipeline_version: str | None = None
        self._paddle_doc_server_url: str | None = None
        self._paddle_doc_backend: str | None = None
        self.last_image_regions_px: list[list[float]] = []
        self.last_layout_blocks: list[dict[str, Any]] = []
        self._last_layout_image_path: str | None = None
        self._image_region_cache_path: str | None = None
        self._image_region_cache_ready: bool = False

        self.vendor_adapter = _create_ai_ocr_vendor_adapter(
            provider=provider,
            base_url=base_url,
        )
        resolved_base_url = self.vendor_adapter.resolve_base_url(base_url)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = openai.OpenAI(**client_kwargs)
        resolved_model = self.vendor_adapter.resolve_model(model)
        self.model = (
            _normalize_ai_ocr_model_name(
                resolved_model,
                provider_id=self.vendor_adapter.provider_id,
            )
            or resolved_model
        )
        self.provider_id = self.vendor_adapter.provider_id
        self.base_url = resolved_base_url
        self.allow_model_downgrade: bool = _env_flag(
            "OCR_PADDLE_ALLOW_MODEL_DOWNGRADE",
            default=False,
        )
        self.allow_paddle_prompt_fallback: bool = _env_flag(
            "OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK",
            default=False,
        )
        self._paddle_doc_max_side_px_override: int | None = None
        if paddle_doc_max_side_px is not None:
            try:
                normalized_max_side = int(paddle_doc_max_side_px)
            except Exception:
                normalized_max_side = None
            if normalized_max_side is not None:
                self._paddle_doc_max_side_px_override = max(
                    0, min(6000, int(normalized_max_side))
                )

        if _is_paddleocr_vl_model(self.model):
            should_use_doc_parser = self._should_use_paddle_doc_parser()
            if not should_use_doc_parser and not self.allow_paddle_prompt_fallback:
                raise ValueError(
                    "Selected PaddleOCR-VL model requires dedicated structured OCR channel, "
                    f"but provider/base_url is not supported (provider={self.provider_id}). "
                    "Supported remote providers: siliconflow, novita. "
                    "To force prompt mode explicitly, set OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK=1."
                )
            if should_use_doc_parser:
                if not _clean_str(self.base_url):
                    raise ValueError(
                        "PaddleOCR-VL requires base_url (for example https://api.siliconflow.cn/v1)"
                    )
                try:
                    import paddleocr as _  # noqa: F401
                except Exception as e:
                    raise ValueError(
                        "PaddleOCR-VL requires `paddleocr` package. Install with: pip install paddleocr"
                    ) from e

    def _should_use_paddle_doc_parser(self) -> bool:
        if self._paddle_doc_parser_disabled:
            return False
        # Explicit env switch takes highest priority for debugging/rollout.
        explicit_env = os.getenv("OCR_PADDLE_VL_USE_DOCPARSER")
        if explicit_env is not None:
            return _env_flag("OCR_PADDLE_VL_USE_DOCPARSER", default=True)
        return self.vendor_adapter.should_use_paddle_doc_parser(
            base_url=self.base_url,
            model_name=self.model,
        )

    def _get_paddle_doc_parser(self) -> Any:
        if self._paddle_doc_parser is not None:
            return self._paddle_doc_parser

        try:
            from paddleocr import PaddleOCRVL
        except Exception as e:
            raise RuntimeError(
                "PaddleOCR-VL dedicated adapter requires `paddleocr` package"
            ) from e

        raw_server_url = (
            _clean_str(os.getenv("OCR_PADDLE_VL_REC_SERVER_URL"))
            or self.base_url
            or self.vendor_adapter.resolve_base_url(None)
        )
        server_url = _normalize_paddle_doc_server_url(
            raw_server_url,
            provider_id=self.provider_id,
        )
        if not server_url:
            raise RuntimeError("PaddleOCR-VL dedicated adapter requires base_url")

        backend = _normalize_paddle_doc_backend(os.getenv("OCR_PADDLE_VL_REC_BACKEND"))
        effective_model, pipeline_version = _resolve_paddle_doc_model_and_pipeline(
            model=self.model,
            provider_id=self.provider_id,
            allow_model_downgrade=self.allow_model_downgrade,
        )

        kwargs: dict[str, Any] = {
            "vl_rec_backend": backend,
            "vl_rec_server_url": server_url,
            "vl_rec_api_key": self.api_key,
            "vl_rec_api_model_name": effective_model,
        }
        kwargs.update(
            self._resolve_paddle_doc_parser_tuning_kwargs(
                effective_model=effective_model,
            )
        )
        if pipeline_version:
            kwargs["pipeline_version"] = pipeline_version

        init_timeout_s = _env_float("OCR_PADDLE_VL_DOCPARSER_INIT_TIMEOUT_S", 30.0)
        try:
            self._paddle_doc_parser = _run_in_daemon_thread_with_timeout(
                lambda: PaddleOCRVL(**kwargs),
                timeout_s=init_timeout_s,
                label="paddleocr-vl:init",
            )
        except TimeoutError as e:
            # Disable doc_parser for this client. Callers may optionally enable
            # prompt fallback via OCR_PADDLE_VL_ALLOW_PROMPT_FALLBACK.
            self._paddle_doc_parser_disabled = True
            raise RuntimeError(str(e)) from e
        except Exception:
            # Disable doc_parser for this client.
            self._paddle_doc_parser_disabled = True
            raise
        self._paddle_doc_effective_model = effective_model
        self._paddle_doc_pipeline_version = pipeline_version
        self._paddle_doc_server_url = server_url
        self._paddle_doc_backend = backend
        logger.info(
            "Initialized PaddleOCR-VL doc_parser adapter (provider=%s, requested_model=%s, effective_model=%s, pipeline_version=%s, base_url=%s, backend=%s, max_concurrency=%s, use_queues=%s)",
            self.provider_id,
            self.model,
            effective_model,
            pipeline_version or "<default>",
            server_url,
            backend,
            kwargs.get("vl_rec_max_concurrency"),
            kwargs.get("use_queues"),
        )
        return self._paddle_doc_parser

    def _resolve_paddle_doc_parser_tuning_kwargs(
        self, *, effective_model: str
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        lowered_model = str(effective_model or "").strip().lower()

        raw_max_concurrency = _clean_str(
            os.getenv("OCR_PADDLE_VL_DOCPARSER_MAX_CONCURRENCY")
        )
        if raw_max_concurrency is not None:
            try:
                parsed_max_concurrency = int(raw_max_concurrency)
            except Exception:
                parsed_max_concurrency = 0
            if parsed_max_concurrency > 0:
                kwargs["vl_rec_max_concurrency"] = parsed_max_concurrency
        elif (
            "paddleocr-vl-1.5" in lowered_model
            and str(self.provider_id or "").strip().lower() == "siliconflow"
        ):
            # PaddleOCR-VL upstream default max_concurrency=200 can overload some
            # OpenAI-compatible gateways (notably SiliconFlow) and cause long-tail
            # latency/timeouts. A small bounded fan-out keeps per-page latency
            # reasonable without reopening the 200-way burst.
            kwargs["vl_rec_max_concurrency"] = 4

        raw_use_queues = os.getenv("OCR_PADDLE_VL_DOCPARSER_USE_QUEUES")
        if raw_use_queues is not None:
            kwargs["use_queues"] = _env_flag(
                "OCR_PADDLE_VL_DOCPARSER_USE_QUEUES",
                default=True,
            )
        return kwargs

    def _resolve_paddle_doc_predict_timeout_s(self) -> float:
        default_timeout = max(
            10.0,
            _env_float("OCR_PADDLE_VL_DOCPARSER_PREDICT_TIMEOUT_S", 120.0),
        )
        lowered_model = str(self.model or "").strip().lower()
        provider_id = str(self.provider_id or "").strip().lower()
        if "paddleocr-vl-1.5" in lowered_model:
            if provider_id == "siliconflow":
                return max(
                    10.0,
                    _env_float(
                        "OCR_PADDLE_VL_DOCPARSER_PREDICT_TIMEOUT_S_V15_SILICONFLOW",
                        _env_float(
                            "OCR_PADDLE_VL_DOCPARSER_PREDICT_TIMEOUT_S_V15",
                            60.0,
                        ),
                    ),
                )
            v15_default_timeout = max(default_timeout, 180.0)
            return max(
                10.0,
                _env_float(
                    "OCR_PADDLE_VL_DOCPARSER_PREDICT_TIMEOUT_S_V15",
                    v15_default_timeout,
                ),
            )
        return default_timeout

    def _resolve_paddle_doc_retry_timeout_s(self, *, predict_timeout_s: float) -> float:
        default_retry_timeout_s = min(90.0, predict_timeout_s)
        lowered_model = str(self.model or "").strip().lower()
        provider_id = str(self.provider_id or "").strip().lower()
        if "paddleocr-vl-1.5" in lowered_model and provider_id == "siliconflow":
            return max(
                10.0,
                _env_float(
                    "OCR_PADDLE_VL_DOCPARSER_RETRY_TIMEOUT_S_V15_SILICONFLOW",
                    _env_float(
                        "OCR_PADDLE_VL_DOCPARSER_RETRY_TIMEOUT_S",
                        min(20.0, predict_timeout_s),
                    ),
                ),
            )
        return max(
            10.0,
            _env_float(
                "OCR_PADDLE_VL_DOCPARSER_RETRY_TIMEOUT_S",
                default_retry_timeout_s,
            ),
        )

    def _is_siliconflow_paddle_doc_v15(self) -> bool:
        lowered_model = str(self.model or "").strip().lower()
        provider_id = str(self.provider_id or "").strip().lower()
        return "paddleocr-vl-1.5" in lowered_model and provider_id == "siliconflow"

    def _should_retry_paddle_doc_timeout(self) -> bool:
        raw_specific = os.getenv(
            "OCR_PADDLE_VL_DOCPARSER_RETRY_ON_TIMEOUT_V15_SILICONFLOW"
        )
        raw_general = os.getenv("OCR_PADDLE_VL_DOCPARSER_RETRY_ON_TIMEOUT")
        if self._is_siliconflow_paddle_doc_v15():
            if raw_specific is not None:
                return _env_flag(
                    "OCR_PADDLE_VL_DOCPARSER_RETRY_ON_TIMEOUT_V15_SILICONFLOW",
                    default=False,
                )
            if raw_general is not None:
                return _env_flag(
                    "OCR_PADDLE_VL_DOCPARSER_RETRY_ON_TIMEOUT",
                    default=False,
                )
            return False
        return _env_flag("OCR_PADDLE_VL_DOCPARSER_RETRY_ON_TIMEOUT", default=True)

    def _should_use_paddle_doc_singleflight(self) -> bool:
        raw_specific = os.getenv(
            "OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT_V15_SILICONFLOW"
        )
        raw_general = os.getenv("OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT")
        if self._is_siliconflow_paddle_doc_v15():
            if raw_specific is not None:
                return _env_flag(
                    "OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT_V15_SILICONFLOW",
                    default=True,
                )
            if raw_general is not None:
                return _env_flag(
                    "OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT",
                    default=True,
                )
            return True
        return _env_flag("OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT", default=False)

    def _resolve_paddle_doc_singleflight_wait_s(self) -> float:
        default_wait_s = 1.0 if self._is_siliconflow_paddle_doc_v15() else 3.0
        return max(
            0.0,
            _env_float("OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT_WAIT_S", default_wait_s),
        )

    def _resolve_paddle_doc_singleflight_lock_path(self) -> Path:
        raw_lock_dir = _clean_str(
            os.getenv("OCR_PADDLE_VL_DOCPARSER_SINGLEFLIGHT_LOCK_DIR")
        )
        lock_dir = Path(raw_lock_dir) if raw_lock_dir else Path("/tmp")
        lock_key = "|".join(
            [
                str(self.provider_id or "").strip().lower(),
                str(self.base_url or "").strip().lower(),
                str(self.model or "").strip().lower(),
                str(self._paddle_doc_pipeline_version or "").strip().lower(),
            ]
        )
        digest = hashlib.sha1(lock_key.encode("utf-8")).hexdigest()[:24]
        return lock_dir / f"paddleocr-vl-docparser-{digest}.lock"

    def _resolve_paddle_doc_max_side_px(self) -> int:
        if self._paddle_doc_max_side_px_override is not None:
            return int(self._paddle_doc_max_side_px_override)
        return max(0, _env_int("OCR_PADDLE_VL_DOCPARSER_MAX_SIDE_PX", 2200))

    def _prepare_paddle_doc_predict_image(
        self, image_path: str
    ) -> tuple[str, float, float, Path | None]:
        max_side_px = self._resolve_paddle_doc_max_side_px()
        if max_side_px <= 0:
            return image_path, 1.0, 1.0, None

        try:
            with Image.open(image_path).convert("RGB") as image:
                width, height = image.size
                largest = max(int(width), int(height))
                if largest <= int(max_side_px):
                    return image_path, 1.0, 1.0, None

                ratio = float(max_side_px) / float(largest)
                new_width = max(32, int(round(float(width) * ratio)))
                new_height = max(32, int(round(float(height) * ratio)))
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                source_stat = Path(image_path).stat()
                digest = hashlib.sha1(
                    f"{image_path}|{source_stat.st_mtime_ns}|{source_stat.st_size}|{max_side_px}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:16]
                temp_path = Path(tempfile.gettempdir()) / (
                    f"paddleocr-vl-{digest}-{new_width}x{new_height}.png"
                )
                resized.save(temp_path)
                logger.info(
                    "Downscaled PaddleOCR-VL doc_parser image from %sx%s to %sx%s (max_side=%s)",
                    width,
                    height,
                    new_width,
                    new_height,
                    max_side_px,
                )
                return (
                    str(temp_path),
                    float(width) / float(new_width),
                    float(height) / float(new_height),
                    temp_path,
                )
        except Exception as e:
            logger.warning(
                "Failed to prepare downscaled PaddleOCR-VL image for %s: %s",
                image_path,
                e,
            )
        return image_path, 1.0, 1.0, None

    def _run_paddle_doc_predict_with_timeout(
        self,
        func: Any,
        *,
        timeout_s: float,
        label: str,
    ) -> Any:
        effective_timeout = max(1.0, float(timeout_s))
        if not self._should_use_paddle_doc_singleflight() or fcntl is None:
            return _run_in_daemon_thread_with_timeout(
                func,
                timeout_s=effective_timeout,
                label=label,
            )

        lock_path = self._resolve_paddle_doc_singleflight_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        wait_timeout_s = self._resolve_paddle_doc_singleflight_wait_s()
        wait_deadline = time.monotonic() + float(wait_timeout_s)
        lock_file = None

        while True:
            candidate = lock_path.open("a+b")
            try:
                fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_file = candidate
                break
            except BlockingIOError:
                candidate.close()
                if time.monotonic() >= wait_deadline:
                    raise TimeoutError(
                        f"{label} blocked by another in-flight PaddleOCR-VL doc_parser request "
                        f"after {wait_timeout_s:.1f}s"
                    )
                time.sleep(min(0.1, max(0.01, wait_deadline - time.monotonic())))

        done = threading.Event()
        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}
        ctx = contextvars.copy_context()

        def _runner() -> None:
            def _run_with_context() -> None:
                try:
                    result["value"] = func()
                except BaseException as exc:  # noqa: BLE001
                    error["error"] = exc
                finally:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    try:
                        lock_file.close()  # type: ignore[union-attr]
                    except Exception:
                        pass
                    done.set()

            ctx.run(_run_with_context)

        thread = threading.Thread(target=_runner, name=f"timeout:{label}", daemon=True)
        thread.start()

        if not done.wait(timeout=effective_timeout):
            raise TimeoutError(f"{label} timed out after {effective_timeout:.0f}s")
        if "error" in error:
            raise error["error"]
        return result.get("value")

    def _ocr_image_with_paddle_doc_parser(self, image_path: str) -> List[Dict]:
        max_side_px = self._resolve_paddle_doc_max_side_px()
        predict_image_path, scale_x, scale_y, temp_image_path = (
            self._prepare_paddle_doc_predict_image(image_path)
        )
        predict_kwargs: dict[str, Any] = {}
        predict_max_pixels = _derive_paddle_doc_predict_max_pixels(
            max_side_px=max_side_px,
            did_downscale=temp_image_path is not None,
        )
        if predict_max_pixels is not None:
            predict_kwargs["max_pixels"] = int(predict_max_pixels)
            logger.info(
                "Constraining PaddleOCR-VL doc_parser predict max_pixels=%s (max_side=%s, downscaled=%s)",
                predict_max_pixels,
                max_side_px,
                temp_image_path is not None,
            )

        def _predict_once() -> Any:
            parser_local = self._get_paddle_doc_parser()
            try:
                return parser_local.predict(input=predict_image_path, **predict_kwargs)
            except TypeError:
                try:
                    return parser_local.predict(predict_image_path, **predict_kwargs)
                except TypeError:
                    try:
                        return parser_local.predict(input=predict_image_path)
                    except TypeError:
                        return parser_local.predict(predict_image_path)

        try:
            try:
                predict_timeout_s = self._resolve_paddle_doc_predict_timeout_s()
                output = self._run_paddle_doc_predict_with_timeout(
                    _predict_once,
                    timeout_s=predict_timeout_s,
                    label="paddleocr-vl:predict",
                )
            except Exception as first_error:
                wants_v15 = (
                    str(self.model or "").strip().lower()
                    == _PADDLE_OCR_VL_MODEL_V15.lower()
                )
                can_downgrade = bool(self.allow_model_downgrade)
                error_to_raise: Exception | None = first_error
                if (
                    isinstance(first_error, TimeoutError)
                    and wants_v15
                    and self._should_retry_paddle_doc_timeout()
                ):
                    retry_timeout_s = self._resolve_paddle_doc_retry_timeout_s(
                        predict_timeout_s=predict_timeout_s
                    )
                    logger.warning(
                        "PaddleOCR-VL-1.5 predict timed out after %.0fs; retrying once with a fresh parser (retry_timeout=%.0fs)",
                        predict_timeout_s,
                        retry_timeout_s,
                    )
                    self._paddle_doc_parser = None
                    self._paddle_doc_parser_disabled = False
                    try:
                        output = self._run_paddle_doc_predict_with_timeout(
                            _predict_once,
                            timeout_s=retry_timeout_s,
                            label="paddleocr-vl:predict:retry",
                        )
                        error_to_raise = None
                    except Exception as retry_error:
                        error_to_raise = retry_error
                if (
                    error_to_raise is not None
                    and isinstance(error_to_raise, TimeoutError)
                ):
                    self._paddle_doc_parser_disabled = True
                if (
                    wants_v15
                    and can_downgrade
                    and error_to_raise is not None
                    and _is_probably_model_unsupported_error(error_to_raise)
                ):
                    logger.warning(
                        "PaddleOCR-VL-1.5 request failed and downgrade is allowed; retrying with %s",
                        _PADDLE_OCR_VL_MODEL_V1,
                    )
                    self._paddle_doc_parser = None
                    self._paddle_doc_effective_model = _PADDLE_OCR_VL_MODEL_V1
                    self._paddle_doc_pipeline_version = "v1"
                    self._paddle_doc_server_url = None
                    self._paddle_doc_backend = None
                    original_model = self.model
                    try:
                        self.model = _PADDLE_OCR_VL_MODEL_V1
                        output = _run_in_daemon_thread_with_timeout(
                            _predict_once,
                            timeout_s=predict_timeout_s,
                            label="paddleocr-vl:predict",
                        )
                    except Exception:
                        self.model = original_model
                        raise error_to_raise
                else:
                    if (
                        wants_v15
                        and (not can_downgrade)
                        and error_to_raise is not None
                        and _is_probably_model_unsupported_error(error_to_raise)
                    ):
                        raise RuntimeError(
                            "PaddleOCR-VL-1.5 is not available on current endpoint and strict mode forbids downgrade; "
                            "switch to PaddlePaddle/PaddleOCR-VL or disable strict mode explicitly."
                        ) from error_to_raise
                    if error_to_raise is not None:
                        raise error_to_raise

            raw_elements, image_regions, layout_blocks = _extract_paddle_doc_parser_output(
                output
            )
            raw_elements, image_regions, layout_blocks = _scale_paddle_doc_parser_output(
                raw_elements,
                image_regions,
                layout_blocks,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            self.last_layout_blocks = list(layout_blocks)
            self.last_image_regions_px = [list(region) for region in image_regions]
            self._last_layout_image_path = str(image_path)
            self._image_region_cache_path = str(image_path)
            self._image_region_cache_ready = True

            if not raw_elements:
                logger.warning(
                    "PaddleOCR-VL doc_parser produced no usable text blocks "
                    "(provider=%s, requested_model=%s, effective_model=%s, pipeline_version=%s)",
                    self.provider_id,
                    self.model,
                    self._paddle_doc_effective_model or self.model,
                    self._paddle_doc_pipeline_version or "<default>",
                )
                raise RuntimeError(
                    "PaddleOCR-VL doc_parser returned no valid text blocks in parsing_res_list"
                )

            logger.info(
                "PaddleOCR-VL doc_parser parsed %s text blocks and %s image-like regions",
                len(raw_elements),
                len(self.last_image_regions_px),
            )
            return raw_elements
        finally:
            if temp_image_path is not None:
                try:
                    temp_image_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _detect_image_regions_with_prompt(self, image_path: str) -> list[list[float]]:
        from ..llm_adapter import _validate_image_regions_px

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            return []

        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"

        effective_model = str(self.model)
        is_deepseek_model = _is_deepseek_ocr_model(effective_model)
        request_timeout_s = max(
            8.0,
            _env_float("OCR_AI_IMAGE_REGION_TIMEOUT_S", 30.0),
        )
        max_tokens_image_regions = self.vendor_adapter.clamp_max_tokens(
            1024, kind="ocr"
        )

        if is_deepseek_model:
            prompt = (
                "<image>\n<|grounding|>Locate <|ref|>screenshots, charts, diagrams, "
                "illustrations, photos, logos, and icons<|/ref|> in the image."
            )
        else:
            prompt = (
                "Locate standalone non-text visual regions on this page. "
                "Return ONLY minified JSON array, no markdown. "
                f"Image size: {width}x{height} px. "
                'Each item must be {"bbox":[x0,y0,x1,y1]}. '
                "Regions include screenshots, charts, diagrams, illustrations, photos, logos, and icons. "
                "Use tight pixel bbox around the visual asset only. "
                "Exclude surrounding text, card backgrounds, tables, formulas, and the full-page background. "
                "Keep output <= 12 items."
            )

        user_content = self.vendor_adapter.build_user_content(
            prompt=prompt,
            image_data_uri=data_uri,
            image_first=_should_send_image_first_for_ai_ocr(
                provider_id=self.provider_id,
                model_name=effective_model,
            ),
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "Return JSON array only, no markdown.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        if is_deepseek_model:
            messages = [
                {
                    "role": "user",
                    "content": user_content,
                }
            ]

        completion = self.client.with_options(
            timeout=request_timeout_s,
            max_retries=0,
        ).chat.completions.create(
            model=effective_model,
            temperature=0,
            max_tokens=max_tokens_image_regions,
            messages=messages,  # type: ignore[arg-type]
        )
        content_obj = (
            completion.choices[0].message.content
            if getattr(completion, "choices", None)
            else ""
        )
        content = _extract_message_text(content_obj)
        region_items = _extract_image_regions_json(content)
        if not region_items and (is_deepseek_model or "<|det|>" in (content or "")):
            region_items = _extract_deepseek_image_regions(content)
        if not region_items and (is_deepseek_model or "<|det|>" in (content or "")):
            tagged_items = _extract_deepseek_tagged_items(content)
            if tagged_items:
                region_items = []
                for item in tagged_items:
                    bbox = _coerce_bbox_xyxy(item.get("bbox"))
                    if bbox is None:
                        continue
                    region_items.append(
                        [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    )

        if region_items:
            normalized_candidates = [
                {
                    "text": "image_region",
                    "bbox": list(region),
                    "confidence": 1.0,
                }
                for region in region_items
                if isinstance(region, list) and len(region) == 4
            ]
            normalized_items, _ = self._normalize_items_to_pixels(
                normalized_candidates,
                image=image,
            )
            normalized_regions = [
                list(bbox)
                for bbox in (
                    item.get("bbox") if isinstance(item, dict) else None
                    for item in normalized_items
                )
                if isinstance(bbox, list) and len(bbox) == 4
            ]
            if normalized_regions:
                region_items = normalized_regions

        validated = _validate_image_regions_px(
            region_items or [],
            width_px=int(width),
            height_px=int(height),
            max_regions=12,
        )
        return validated or []

    def detect_image_regions(self, image_path: str) -> list[list[float]]:
        requested_path = str(image_path)
        if (
            self._image_region_cache_ready
            and str(self._image_region_cache_path or "") == requested_path
        ):
            return [list(region) for region in self.last_image_regions_px]

        self.last_image_regions_px = []
        self.last_layout_blocks = []
        self._last_layout_image_path = requested_path
        self._image_region_cache_path = requested_path
        self._image_region_cache_ready = False

        if _is_paddleocr_vl_model(self.model) and self._should_use_paddle_doc_parser():
            try:
                self._ocr_image_with_paddle_doc_parser(image_path)
                return [list(region) for region in self.last_image_regions_px]
            except Exception as e:
                logger.warning(
                    "PaddleOCR-VL image-region extraction failed; falling back to prompt detection: %s",
                    e,
                )

        try:
            self.last_image_regions_px = [
                list(region)
                for region in self._detect_image_regions_with_prompt(image_path)
            ]
        except Exception as e:
            logger.warning("AI OCR image-region detection failed: %s", e)
            self.last_image_regions_px = []

        self._image_region_cache_ready = True
        return [list(region) for region in self.last_image_regions_px]

    def _score_bbox_transform(
        self,
        *,
        image: Image.Image,
        gray: Image.Image,
        items: list[dict],
        base: float | tuple[float, float] | None,
        max_items: int = 60,
    ) -> tuple[float, dict]:
        """Score candidate bbox coordinate systems.

        Some vision models return bounding boxes in a *normalized* coordinate
        grid (often around 0..1000/1024) regardless of the actual image size.
        We evaluate a few plausible transforms and pick the best one.
        """

        width, height = image.size
        if width <= 0 or height <= 0 or not items:
            return (float("-inf"), {"reason": "empty"})

        if base is None:
            sx = 1.0
            sy = 1.0
            base_name = "identity"
        elif isinstance(base, tuple):
            try:
                base_x = float(base[0])
                base_y = float(base[1])
            except Exception:
                return (float("-inf"), {"reason": "invalid_base_xy"})
            if base_x <= 0 or base_y <= 0:
                return (float("-inf"), {"reason": "invalid_base_xy"})
            sx = float(width) / float(base_x)
            sy = float(height) / float(base_y)
            base_name = f"{int(round(base_x))}x{int(round(base_y))}"
        else:
            b = float(base)
            if b <= 0:
                return (float("-inf"), {"reason": "invalid_base"})
            sx = float(width) / b
            sy = float(height) / b
            base_name = str(int(b)) if b.is_integer() else str(b)

        # Take a stable subset (first N) to keep scoring fast on dense pages.
        subset = items[: max(1, min(len(items), int(max_items)))]

        x0s: list[float] = []
        x1s: list[float] = []
        y0s: list[float] = []
        y1s: list[float] = []
        stds: list[float] = []
        out_of_bounds = 0
        valid = 0

        for it in subset:
            bbox = it.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x0, y0, x1, y1 = (
                    float(bbox[0]) * sx,
                    float(bbox[1]) * sy,
                    float(bbox[2]) * sx,
                    float(bbox[3]) * sy,
                )
            except Exception:
                continue
            if math.isnan(x0) or math.isnan(y0) or math.isnan(x1) or math.isnan(y1):
                continue
            x0, x1 = (min(x0, x1), max(x0, x1))
            y0, y1 = (min(y0, y1), max(y0, y1))
            if x1 <= x0 or y1 <= y0:
                continue

            # Count OOB based on unclamped coords.
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                out_of_bounds += 1

            # Clamp for sampling.
            x0c = max(0, min(width - 1, int(round(x0))))
            y0c = max(0, min(height - 1, int(round(y0))))
            x1c = max(0, min(width, int(round(x1))))
            y1c = max(0, min(height, int(round(y1))))
            if x1c <= x0c or y1c <= y0c:
                continue

            x0s.append(float(x0c))
            x1s.append(float(x1c))
            y0s.append(float(y0c))
            y1s.append(float(y1c))
            valid += 1

            # Pixel-variance proxy: real text regions tend to have higher
            # local variance than blank/background regions.
            crop = gray.crop((x0c, y0c, x1c, y1c))
            if crop.width <= 0 or crop.height <= 0:
                continue
            target_w = max(8, min(64, crop.width // 8))
            target_h = max(8, min(64, crop.height // 8))
            small = crop.resize((target_w, target_h))
            pixels = list(small.getdata())  # type: ignore[arg-type]
            if not pixels:
                continue
            mean = sum(pixels) / len(pixels)
            var = sum((p - mean) ** 2 for p in pixels) / len(pixels)
            stds.append(float(var**0.5))

        if valid <= 0:
            return (float("-inf"), {"base": base_name, "reason": "no_valid_boxes"})

        def _percentile(sorted_vals: list[float], p: float) -> float:
            if not sorted_vals:
                return 0.0
            p = max(0.0, min(1.0, float(p)))
            idx = int(round((len(sorted_vals) - 1) * p))
            return sorted_vals[idx]

        x0s_s = sorted(x0s)
        x1s_s = sorted(x1s)
        y0s_s = sorted(y0s)
        y1s_s = sorted(y1s)

        x_span = (_percentile(x1s_s, 0.95) - _percentile(x0s_s, 0.05)) / float(width)
        y_span = (_percentile(y1s_s, 0.95) - _percentile(y0s_s, 0.05)) / float(height)
        coverage_score = max(0.0, min(1.0, x_span)) + max(0.0, min(1.0, y_span))  # 0..2

        median_std = sorted(stds)[len(stds) // 2] if stds else 0.0
        out_rate = float(out_of_bounds) / float(valid)

        # Weighted score: prioritize good coverage (boxes span the page) then
        # variance, penalize out-of-bounds.
        score = (1.6 * coverage_score) + (median_std / 32.0) - (2.0 * out_rate)
        details = {
            "base": base_name,
            "sx": sx,
            "sy": sy,
            "valid": valid,
            "median_std": median_std,
            "coverage_x": x_span,
            "coverage_y": y_span,
            "out_rate": out_rate,
        }
        return (float(score), details)

    def _normalize_items_to_pixels(
        self,
        items: list[dict],
        *,
        image: Image.Image,
    ) -> tuple[list[dict], dict]:
        """Return (items_px, debug) after auto-normalizing bbox coords to pixels."""

        width, height = image.size
        if width <= 0 or height <= 0 or not items:
            return (items, {"chosen": "none", "reason": "empty"})

        gray = image.convert("L")

        # Evaluate common coordinate grids + identity. Some gateways also return
        # bbox coordinates in the *resized* model-input pixel space (e.g. long
        # side normalized to 1024 while keeping aspect ratio). In that case the
        # X/Y bases differ; we add a few aspect-preserving candidates.
        uniform_candidates: list[float | None] = [
            None,
            1.0,
            100.0,
            1000.0,
            1024.0,
            2048.0,
            4096.0,
        ]

        def _resize_dims_for_target_side(
            target_side: float, *, mode: str
        ) -> tuple[float, float] | None:
            try:
                target = float(target_side)
            except Exception:
                return None
            if target <= 0:
                return None
            if mode == "short":
                denom = float(min(width, height))
            else:
                denom = float(max(width, height))
            if denom <= 0:
                return None
            scale = float(target) / denom
            if scale <= 0:
                return None
            bw = max(1.0, float(round(float(width) * scale)))
            bh = max(1.0, float(round(float(height) * scale)))
            if bw <= 0 or bh <= 0:
                return None
            return (bw, bh)

        seen: set[str] = set()
        candidates: list[float | tuple[float, float] | None] = []

        def _add_candidate(value: float | tuple[float, float] | None) -> None:
            if value is None:
                key = "identity"
            elif isinstance(value, tuple):
                key = f"xy:{int(round(float(value[0])))}x{int(round(float(value[1])))}"
            else:
                key = f"u:{float(value):.3f}"
            if key in seen:
                return
            seen.add(key)
            candidates.append(value)

        for base in uniform_candidates:
            _add_candidate(base)

        for side in (1000.0, 1024.0, 1536.0, 2048.0):
            cand = _resize_dims_for_target_side(side, mode="long")
            if cand is not None:
                _add_candidate(cand)

        for side in (1000.0, 1024.0):
            cand = _resize_dims_for_target_side(side, mode="short")
            if cand is not None:
                _add_candidate(cand)

        scored: list[tuple[float, float | tuple[float, float] | None, dict]] = []
        for base in candidates:
            score, details = self._score_bbox_transform(
                image=image, gray=gray, items=items, base=base
            )
            scored.append((score, base, details))

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_base, best_details = scored[0]

        # Apply best transform.
        if best_base is None:
            sx = 1.0
            sy = 1.0
        elif isinstance(best_base, tuple):
            bx, by = best_base
            bx = float(bx)
            by = float(by)
            sx = float(width) / float(max(1.0, bx))
            sy = float(height) / float(max(1.0, by))
        else:
            sx = float(width) / float(best_base)
            sy = float(height) / float(best_base)

        out: list[dict] = []
        for it in items:
            bbox = it.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x0, y0, x1, y1 = (
                    float(bbox[0]) * sx,
                    float(bbox[1]) * sy,
                    float(bbox[2]) * sx,
                    float(bbox[3]) * sy,
                )
            except Exception:
                continue
            if math.isnan(x0) or math.isnan(y0) or math.isnan(x1) or math.isnan(y1):
                continue
            x0, x1 = (min(x0, x1), max(x0, x1))
            y0, y1 = (min(y0, y1), max(y0, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            # Clamp to image bounds.
            x0 = max(0.0, min(x0, float(width - 1)))
            y0 = max(0.0, min(y0, float(height - 1)))
            x1 = max(0.0, min(x1, float(width)))
            y1 = max(0.0, min(y1, float(height)))
            if x1 <= x0 or y1 <= y0:
                continue
            new_it = dict(it)
            new_it["bbox"] = [x0, y0, x1, y1]
            out.append(new_it)

        debug = {
            "chosen_base": best_details.get("base"),
            "chosen_score": best_score,
            "chosen_details": best_details,
            "candidates": [d for _, _, d in scored[:3]],
        }
        return (out, debug)

    def _resolve_model_request_timeout_s(self, *, model_name: str | None) -> float:
        default_timeout = max(8.0, _env_float("OCR_AI_REQUEST_TIMEOUT_S", 25.0))
        lowered = str(model_name or "").strip().lower()
        if not lowered:
            return default_timeout

        if "qwen" in lowered and ("vl" in lowered or "omni" in lowered):
            return max(
                8.0,
                _env_float("OCR_AI_REQUEST_TIMEOUT_S_QWEN", default_timeout),
            )

        if "deepseek-ocr" in lowered or "deepseekocr" in lowered:
            return max(
                8.0,
                _env_float("OCR_AI_REQUEST_TIMEOUT_S_DEEPSEEK_OCR", default_timeout),
            )

        if "paddleocr-vl" in lowered:
            return max(
                8.0,
                _env_float("OCR_AI_REQUEST_TIMEOUT_S_PADDLE_VL", default_timeout),
            )

        return default_timeout

    def ocr_image(self, image_path: str) -> List[Dict]:
        self.last_image_regions_px = []
        self.last_layout_blocks = []
        self._last_layout_image_path = str(image_path)
        self._image_region_cache_path = None
        self._image_region_cache_ready = False
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            return []

        is_paddle_model = _is_paddleocr_vl_model(self.model)
        should_use_doc_parser = self._should_use_paddle_doc_parser()
        if is_paddle_model and should_use_doc_parser:
            try:
                return self._ocr_image_with_paddle_doc_parser(image_path)
            except Exception as e:
                if not self.allow_paddle_prompt_fallback:
                    logger.error(
                        "PaddleOCR-VL doc_parser failed with strict routing enabled (provider=%s, model=%s): %s",
                        self.provider_id,
                        self.model,
                        e,
                    )
                    raise RuntimeError(
                        f"PaddleOCR-VL dedicated channel failed: {e}"
                    ) from e
                logger.warning(
                    "PaddleOCR-VL doc_parser failed; prompt fallback is explicitly enabled: %s",
                    e,
                )
                self._paddle_doc_parser_disabled = True
                self._paddle_doc_parser = None
                should_use_doc_parser = self._should_use_paddle_doc_parser()

        model_candidates: list[str] = [str(self.model)]
        if is_paddle_model and not should_use_doc_parser:
            if not self.allow_paddle_prompt_fallback:
                raise RuntimeError(
                    "PaddleOCR-VL dedicated channel is unavailable for current provider/base_url; "
                    "prompt fallback is disabled."
                )
            # Prompt path is opt-in for advanced users who know their gateway can
            # emit bbox JSON without Paddle doc_parser protocol.
            fallback_model = _clean_str(os.getenv("OCR_PADDLE_PROMPT_FALLBACK_MODEL"))
            if fallback_model and fallback_model.lower() != str(self.model).lower():
                model_candidates.append(fallback_model)
                logger.info(
                    "PaddleOCR-VL prompt fallback enabled; trying requested model first, then fallback model=%s",
                    fallback_model,
                )
            else:
                logger.info(
                    "PaddleOCR-VL prompt fallback enabled; trying requested model=%s",
                    self.model,
                )

        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"

        last_error: Exception | None = None
        items: list[dict] | None = None
        max_attempts = max(1, min(5, _env_int("OCR_AI_MAX_ATTEMPTS", 3)))
        empty_response_break_after = max(
            1,
            min(3, _env_int("OCR_AI_EMPTY_RESPONSE_BREAK_AFTER", 2)),
        )
        for model_index, effective_model in enumerate(model_candidates, start=1):
            is_deepseek_model = _is_deepseek_ocr_model(effective_model)
            request_timeout_s = self._resolve_model_request_timeout_s(
                model_name=effective_model
            )

            def _make_prompt(*, item_limit: int) -> str:
                # Keep prompt compact to reduce instruction tokens and steer models
                # (especially OpenAI-compatible gateways) toward short structured output.
                if is_deepseek_model:
                    # Keep DeepSeek prompt minimal. In practice, adding verbose
                    # format constraints often leads to empty outputs on some
                    # OpenAI-compatible gateways.
                    return "<image>\n<|grounding|>OCR this image."
                return (
                    f"OCR task. Return ONLY minified JSON array, no markdown. "
                    f"Image size: {width}x{height} px. "
                    "Each item must be one visual text line with tight bbox. "
                    "Do not output duplicate lines/boxes; skip pure punctuation/noise. "
                    'Preferred item schema: {"t":"text","b":[x0,y0,x1,y1],"c":0.0-1.0}. '
                    'Also accepted: {"text":...,"bbox":...}. '
                    "Coordinates are pixel values, origin top-left. "
                    f"Keep output <= {int(item_limit)} items. "
                    "If dense page, merge words into line-level entries. "
                    "Stop immediately after JSON closes."
                )

            attempt_limits = [60, 40, 24, 16, 10]
            if is_deepseek_model:
                # DeepSeek grounding tags are fairly compact; allow more lines on
                # dense scanned pages while still retrying with smaller limits when
                # output truncates.
                attempt_limits = [180, 120, 90, 60, 40]
            attempt_limits = attempt_limits[:max_attempts]
            empty_response_streak = 0

            for attempt, item_limit in enumerate(attempt_limits, start=1):
                try:
                    prompt = _make_prompt(item_limit=item_limit)
                    requested_tokens = 8192
                    if is_deepseek_model:
                        # Each grounding item is short, but dense pages can easily
                        # exceed 60 lines. Allow enough output budget to avoid
                        # truncation while staying below common gateway limits.
                        requested_tokens = int(320 + int(item_limit) * 22)
                        requested_tokens = max(900, requested_tokens)
                        requested_tokens = min(3500, requested_tokens)
                    max_tokens_ocr = self.vendor_adapter.clamp_max_tokens(
                        requested_tokens, kind="ocr"
                    )
                    system_content = "Return JSON array only, no markdown."
                    if is_deepseek_model:
                        system_content = (
                            "You are an OCR engine. Output only DeepSeek grounding tags "
                            "(<|ref|>...<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|>) or JSON array with bbox."
                        )

                    user_content = self.vendor_adapter.build_user_content(
                        prompt=prompt,
                        image_data_uri=data_uri,
                        image_first=_should_send_image_first_for_ai_ocr(
                            provider_id=self.provider_id,
                            model_name=effective_model,
                        ),
                    )
                    messages: list[dict[str, Any]] = [
                        {
                            "role": "system",
                            "content": system_content,
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ]
                    if is_deepseek_model:
                        messages = [
                            {
                                "role": "user",
                                "content": user_content,
                            }
                        ]

                    completion = self.client.with_options(
                        timeout=request_timeout_s,
                        max_retries=0,
                    ).chat.completions.create(
                        model=effective_model,
                        temperature=0,
                        max_tokens=max_tokens_ocr,
                        messages=messages,  # type: ignore[arg-type]
                    )

                    content_obj = (
                        completion.choices[0].message.content
                        if getattr(completion, "choices", None)
                        else ""
                    )
                    content = _extract_message_text(content_obj)
                    finish_reason = None
                    try:
                        finish_reason = completion.choices[0].finish_reason
                    except Exception:
                        finish_reason = None

                    if not (content or "").strip():
                        empty_response_streak += 1
                        logger.warning(
                            "AI OCR returned empty content (model=%s, attempt=%s/%s, finish_reason=%s)",
                            effective_model,
                            attempt,
                            len(attempt_limits),
                            finish_reason,
                        )
                        if empty_response_streak >= empty_response_break_after:
                            last_error = RuntimeError(
                                "AI OCR returned empty content repeatedly"
                            )
                            # For some gateways/models this pattern is stable;
                            # stop early so OcrManager can move to fallback
                            # providers instead of burning the whole page timeout.
                            break
                    else:
                        empty_response_streak = 0

                    if is_deepseek_model and _looks_like_structural_gibberish(content):
                        preview = (content or "")[:220].replace("\n", " ").strip()
                        logger.warning(
                            "AI OCR returned structural gibberish (model=%s, attempt=%s, chars=%s, preview=%r)",
                            effective_model,
                            attempt,
                            len(content or ""),
                            preview,
                        )
                        raise RuntimeError("AI OCR returned structural gibberish")

                    items = _extract_json_list(content)
                    if not items and (is_deepseek_model or "<|det|>" in (content or "")):
                        items = _extract_deepseek_tagged_items(content)
                    if items:
                        logger.info(
                            "AI OCR parsed %s items (model=%s, attempt=%s, limit=%s, finish_reason=%s)",
                            len(items),
                            effective_model,
                            attempt,
                            item_limit,
                            finish_reason,
                        )
                        break

                    if finish_reason == "length":
                        partial_items = _extract_partial_json_object_list(content)
                        if not partial_items and (
                            is_deepseek_model or "<|det|>" in (content or "")
                        ):
                            tagged_partial = _extract_deepseek_tagged_items(content)
                            partial_items = tagged_partial or []
                        if partial_items:
                            logger.warning(
                                "AI OCR output truncated (model=%s, attempt=%s, limit=%s); recovered %s partial items.",
                                effective_model,
                                attempt,
                                item_limit,
                                len(partial_items),
                            )
                            items = partial_items
                            break
                        preview = (content or "")[:360].replace("\n", " ").strip()
                        logger.warning(
                            "AI OCR truncated with no recoverable JSON (model=%s, attempt=%s, limit=%s, chars=%s, preview=%r)",
                            effective_model,
                            attempt,
                            item_limit,
                            len(content or ""),
                            preview,
                        )
                        raise RuntimeError(
                            f"AI OCR output truncated (finish_reason=length, chars={len(content)})"
                        )

                    preview = (content or "")[:360].replace("\n", " ").strip()
                    logger.warning(
                        "AI OCR returned no parseable items (model=%s, attempt=%s, finish_reason=%s, chars=%s, preview=%r)",
                        effective_model,
                        attempt,
                        finish_reason,
                        len(content or ""),
                        preview,
                    )
                    plain_text_without_boxes = (
                        (not is_deepseek_model)
                        and bool(content and content.strip())
                        and ("{" not in (content or ""))
                        and ("[" not in (content or ""))
                        and ("<|det|>" not in (content or ""))
                    )
                    if plain_text_without_boxes:
                        # Some OCR-capable VLM endpoints return plain transcript
                        # text (without geometry) for prompt-based calls. Retries
                        # are typically useless and only consume page timeout.
                        # Fail this model fast so OcrManager can move to fallback
                        # providers (for example local PaddleOCR).
                        last_error = RuntimeError(
                            "AI OCR returned plain text without bbox/json"
                        )
                        logger.warning(
                            "AI OCR model produced plain text without geometry; skipping remaining attempts for model=%s",
                            effective_model,
                        )
                        break
                    raise RuntimeError("AI OCR returned no items")
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "AI OCR attempt failed (model=%s, attempt=%s): %s",
                        effective_model,
                        attempt,
                        e,
                    )
                    continue

            if items:
                break
            if model_index < len(model_candidates):
                logger.warning(
                    "AI OCR model candidate produced no usable items: %s. Trying next model candidate.",
                    effective_model,
                )

        if not items:
            raise RuntimeError("AI OCR returned no items") from last_error

        raw_elements: List[Dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            text = str(
                item.get("text")
                or item.get("t")
                or item.get("words")
                or item.get("content")
                or item.get("transcription")
                or item.get("value")
                or item.get("label")
                or ""
            ).strip()

            if _looks_like_ocr_prompt_echo_text(text):
                continue

            bbox_raw = item.get("bbox")
            if bbox_raw is None:
                for bbox_key in (
                    "b",
                    "box",
                    "bounding_box",
                    "location",
                    "rect",
                    "points",
                    "polygon",
                    "position",
                    "coordinates",
                    "quad",
                    "bbox_2d",
                ):
                    if bbox_key in item:
                        bbox_raw = item.get(bbox_key)
                        break
            bbox = _coerce_bbox_xyxy(bbox_raw)
            if not text or not bbox:
                continue

            try:
                x0, y0, x1, y1 = (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            except Exception:
                continue
            if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
                continue

            confidence_raw = item.get("confidence")
            if confidence_raw is None:
                confidence_raw = item.get("c")
            if confidence_raw is None:
                confidence_raw = item.get("score")
            if confidence_raw is None:
                confidence_raw = item.get("prob")

            try:
                confidence = (
                    float(confidence_raw) if confidence_raw is not None else 0.7
                )
            except Exception:
                confidence = 0.7
            if confidence > 1.0:
                confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
            confidence = max(0.0, min(confidence, 1.0))

            raw_elements.append(
                {
                    "text": text,
                    "bbox": [x0, y0, x1, y1],
                    "confidence": confidence,
                }
            )

        if not raw_elements:
            raise RuntimeError("AI OCR returned empty elements")

        # Normalize bbox coordinates into the real image pixel space.
        elements, debug = self._normalize_items_to_pixels(raw_elements, image=image)
        if not elements:
            raise RuntimeError("AI OCR bbox normalization produced no valid elements")

        # Lightweight sanity check: if bboxes cover only a tiny fraction of the
        # page and we have many items, treat it as a coordinate mismatch so
        # OcrManager can fall back to a bbox-accurate engine.
        try:
            if len(elements) >= 12:
                xs0 = sorted(float(it["bbox"][0]) for it in elements)
                xs1 = sorted(float(it["bbox"][2]) for it in elements)
                ys0 = sorted(float(it["bbox"][1]) for it in elements)
                ys1 = sorted(float(it["bbox"][3]) for it in elements)
                p05 = max(0, int(round((len(xs0) - 1) * 0.05)))
                p95 = max(0, int(round((len(xs1) - 1) * 0.95)))
                span_x = (xs1[p95] - xs0[p05]) / float(width)
                span_y = (ys1[p95] - ys0[p05]) / float(height)
                coverage_threshold = 0.24 if is_deepseek_model else 0.35
                if span_x < coverage_threshold or span_y < coverage_threshold:
                    raise RuntimeError(
                        f"AI OCR bbox coverage too small after normalization: span_x={span_x:.3f}, span_y={span_y:.3f}"
                    )
        except Exception as e:
            logger.warning("AI OCR bbox sanity check failed: %s debug=%s", e, debug)
            raise

        logger.info("AI OCR bbox normalization: %s", debug.get("chosen_details"))
        # Attach lightweight provenance for downstream dedupe/QA. Do NOT include
        # API keys or full URLs.
        try:
            for el in elements:
                if not isinstance(el, dict):
                    continue
                el.setdefault("provider", self.provider_id)
                el.setdefault("model", self.model)
        except Exception:
            pass
        return elements


def _is_multiline_candidate_for_linebreak_assist(
    *,
    text: str,
    bbox: tuple[float, float, float, float] | Any,
    image_width: int,
    image_height: int,
    median_line_height: float,
) -> bool:
    """Decide whether an OCR bbox likely contains multiple visual lines.

    This is a pre-filter before calling a vision model to split lines. Keeping
    it as a standalone helper makes behavior testable and easier to tune.
    """

    bbox_n = _normalize_bbox_px(bbox) if not isinstance(bbox, tuple) else bbox
    if bbox_n is None:
        return False

    x0, y0, x1, y1 = bbox_n
    w = max(1.0, float(x1 - x0))
    h = max(1.0, float(y1 - y0))
    width = max(1, int(image_width))
    height = max(1, int(image_height))
    median_h = max(0.0, float(median_line_height))
    if median_h <= 0.0:
        median_h = max(10.0, 0.02 * float(height))

    raw_text = str(text or "")
    compact = re.sub(r"\s+", "", raw_text)
    if "\n" in raw_text and len(compact) >= 3:
        return True
    if len(compact) < 8:
        return False

    # Wide banner-like titles are often single-line even with larger bboxes;
    # avoid over-splitting these into pseudo-lines.
    wide_banner_like = (
        w >= 0.28 * float(width)
        and (h / max(1.0, w)) <= 0.11
        and len(compact) <= 42
        and h <= max(3.6 * median_h, 0.16 * float(height))
    )
    if wide_banner_like:
        return False

    # PaddleOCR-VL doc parser (and some AI OCR providers) frequently returns
    # paragraph-like bboxes that are only ~1.5x the median line height. A
    # stricter 1.8x gate misses these, leaving the renderer to guess line
    # breaks and causing visible wrap drift in PPT.
    if h >= max(1.80 * median_h, 0.055 * float(height)):
        return True
    if h >= max(1.45 * median_h, 0.045 * float(height)) and (
        len(compact) >= 16 or w >= 0.30 * float(width)
    ):
        return True
    return False


class AiOcrTextRefiner:
    """Refine OCR line texts using an OpenAI-compatible vision model.

    This does NOT change bounding boxes. It is designed to run after a bbox-
    accurate OCR engine (e.g. Tesseract) and improve transcription quality.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ):
        import openai

        if not api_key:
            raise ValueError("AI refiner api_key is required")

        self.vendor_adapter = _create_ai_ocr_vendor_adapter(
            provider=provider,
            base_url=base_url,
        )
        resolved_base_url = self.vendor_adapter.resolve_base_url(base_url)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = openai.OpenAI(**client_kwargs)
        resolved_model = self.vendor_adapter.resolve_model(model)
        self.model = (
            _normalize_ai_ocr_model_name(
                resolved_model,
                provider_id=self.vendor_adapter.provider_id,
            )
            or resolved_model
        )
        self.provider_id = self.vendor_adapter.provider_id
        self.base_url = resolved_base_url

    def refine_items(
        self,
        image_path: str,
        *,
        items: list[dict],
        max_items_per_call: int = 80,
    ) -> list[dict]:
        """Return a new items list with refined `text` fields.

        Args:
            image_path: Path to the page image.
            items: List of dicts with keys: text (str) and bbox ([x0,y0,x1,y1] in px).
            max_items_per_call: Chunk size to reduce truncation risk.
        """

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0 or not items:
            return items

        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"

        def _chunks(seq: list[dict], n: int) -> list[list[dict]]:
            n = max(1, int(n))
            return [seq[i : i + n] for i in range(0, len(seq), n)]

        refined: list[dict] = [dict(it) for it in items]

        # Build a stable indexing so the model can return corrections by id.
        indexed: list[dict] = []
        for i, it in enumerate(items):
            text = str(it.get("text") or "")
            bbox = it.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            indexed.append({"i": i, "bbox": bbox, "text": text})

        if not indexed:
            return refined

        for part in _chunks(indexed, max_items_per_call):
            prompt = (
                "You are an OCR post-processor. You will be given a page image and a JSON array of OCR line boxes. "
                "Each item has {i, bbox:[x0,y0,x1,y1], text}. The bbox is in PIXELS in the page image "
                f"(origin top-left, width={width}, height={height}).\n\n"
                "Task: For each item, READ the text inside its bbox on the image and output ONLY a JSON array of "
                "objects {i:int, text:string}. Keep the same i values. Do NOT include bbox in the output. "
                "Do NOT add new items.\n\n"
                "Rules:\n"
                "- The provided `text` is noisy; treat it as a hint only.\n"
                "- Preserve the original language(s) and punctuation (Chinese/English/numbers/parentheses).\n"
                "- Do NOT hallucinate words that are not visible in the bbox.\n"
                "- If the bbox is unreadable or blank, return the original text for that i.\n\n"
                "Input items:\n"
                + json.dumps(part, ensure_ascii=True)
                + "\n\nOutput ONLY the JSON array."
            )

            messages_payload: Any = [
                {
                    "role": "system",
                    "content": "Return JSON array only, no markdown.",
                },
                {
                    "role": "user",
                    "content": self.vendor_adapter.build_user_content(
                        prompt=prompt,
                        image_data_uri=data_uri,
                        image_first=_should_send_image_first_for_ai_ocr(
                            provider_id=self.provider_id,
                            model_name=self.model,
                        ),
                    ),
                },
            ]
            completion = self.client.with_options(timeout=60).chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=self.vendor_adapter.clamp_max_tokens(4096, kind="refiner"),
                messages=messages_payload,
            )

            content = (
                completion.choices[0].message.content
                if getattr(completion, "choices", None)
                else ""
            )
            out = _extract_json_list(content or "")
            if not out:
                continue

            for item in out:
                if not isinstance(item, dict):
                    continue
                idx = item.get("i")
                if not isinstance(idx, int) or idx < 0 or idx >= len(refined):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    # Never overwrite a bbox's OCR text with empty output from the
                    # refiner. Some vision models return "" when they can't read
                    # a region; keeping the original Tesseract/Baidu text preserves
                    # coverage (the user can later fix/delete a few bad boxes).
                    new_text = text.strip()
                    if new_text:
                        refined[idx]["text"] = new_text

        return refined

    def assist_line_breaks(
        self,
        image_path: str,
        *,
        items: list[dict],
        max_items_per_call: int = 36,
        max_lines_per_item: int = 8,
        allow_heuristic_fallback: bool = False,
    ) -> list[dict]:
        """Split coarse OCR boxes into line-level boxes with visual guidance.

        Primarily split vertically, and then opportunistically tighten each
        line's horizontal bounds by local ink projection. This improves layout
        fidelity for downstream PPT text placement/color sampling.
        """

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0 or not items:
            return items

        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"

        normalized_rows: list[dict[str, Any]] = []
        line_heights: list[float] = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            bbox_n = _normalize_bbox_px(it.get("bbox"))
            if not text or bbox_n is None:
                continue
            x0, y0, x1, y1 = bbox_n
            h = max(1.0, float(y1 - y0))
            line_heights.append(h)
            normalized_rows.append(
                {
                    "i": i,
                    "text": text,
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "bbox_n": bbox_n,
                    "original": it,
                }
            )

        if not normalized_rows:
            return items

        def _median(values: list[float]) -> float:
            if not values:
                return 0.0
            ordered = sorted(float(v) for v in values)
            n = len(ordered)
            m = n // 2
            if n % 2 == 1:
                return ordered[m]
            return (ordered[m - 1] + ordered[m]) / 2.0

        median_h = _median(line_heights)
        if median_h <= 0:
            median_h = max(10.0, 0.02 * float(height))

        candidates: list[dict[str, Any]] = []
        for row in normalized_rows:
            if _is_multiline_candidate_for_linebreak_assist(
                text=str(row.get("text") or ""),
                bbox=row.get("bbox_n"),
                image_width=width,
                image_height=height,
                median_line_height=median_h,
            ):
                candidates.append(
                    {
                        "i": row["i"],
                        "bbox": row["bbox"],
                        "text": row["text"],
                    }
                )

        if not candidates:
            return items

        def _chunks(seq: list[dict], n: int) -> list[list[dict]]:
            n = max(1, int(n))
            return [seq[i : i + n] for i in range(0, len(seq), n)]

        split_map: dict[int, list[str]] = {}
        for part in _chunks(candidates, max_items_per_call):
            prompt = (
                "You are an OCR layout post-processor. You will get a page image and a JSON array "
                "of OCR text boxes that may contain multiple visual lines. Each item has {i, bbox, text}. "
                "bbox is in PIXELS in the image "
                f"(origin top-left, width={width}, height={height}).\n\n"
                "Task: For each item, read only the text inside its bbox and split it into visual lines "
                "(top to bottom). Return ONLY a JSON array of objects {i:int, lines:string[]}.\n\n"
                "Rules:\n"
                "- Keep original i values; do NOT add new items.\n"
                "- Keep language and punctuation as seen in the image.\n"
                "- If a box is single-line or uncertain, return lines with exactly one entry.\n"
                "- Do NOT include markdown or explanations.\n\n"
                "Input items:\n"
                + json.dumps(part, ensure_ascii=True)
                + "\n\nOutput ONLY the JSON array."
            )

            messages_payload: Any = [
                {
                    "role": "system",
                    "content": "Return JSON array only, no markdown.",
                },
                {
                    "role": "user",
                    "content": self.vendor_adapter.build_user_content(
                        prompt=prompt,
                        image_data_uri=data_uri,
                        image_first=_should_send_image_first_for_ai_ocr(
                            provider_id=self.provider_id,
                            model_name=self.model,
                        ),
                    ),
                },
            ]
            completion = self.client.with_options(timeout=60).chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=self.vendor_adapter.clamp_max_tokens(3072, kind="refiner"),
                messages=messages_payload,
            )

            content = (
                completion.choices[0].message.content
                if getattr(completion, "choices", None)
                else ""
            )
            out = _extract_json_list(content or "")
            if not out:
                continue

            for item in out:
                if not isinstance(item, dict):
                    continue
                idx = item.get("i")
                if not isinstance(idx, int):
                    continue
                raw_lines = item.get("lines")
                lines: list[str] = []
                if isinstance(raw_lines, str):
                    lines = [
                        seg.strip() for seg in raw_lines.splitlines() if seg.strip()
                    ]
                elif isinstance(raw_lines, list):
                    for seg in raw_lines:
                        if isinstance(seg, str):
                            cleaned = seg.strip()
                            if cleaned:
                                lines.append(cleaned)
                if lines:
                    split_map[idx] = lines[: max(1, int(max_lines_per_item))]

        row_map: dict[int, dict[str, Any]] = {
            int(row["i"]): row
            for row in normalized_rows
            if isinstance(row.get("i"), int)
        }
        candidate_idx_set: set[int] = {
            int(row["i"]) for row in candidates if isinstance(row.get("i"), int)
        }

        def _compact_text(text: str) -> str:
            return re.sub(r"\s+", "", text or "")

        def _split_is_plausible(
            original_text: str,
            lines: list[str],
            *,
            row: dict[str, Any] | None,
        ) -> bool:
            if len(lines) <= 1:
                return False
            compact_orig = _compact_text(original_text)
            compact_joined = _compact_text("".join(lines))
            if not compact_joined:
                return False
            if not compact_orig:
                return True

            contains_relation = (
                compact_orig in compact_joined or compact_joined in compact_orig
            )

            if contains_relation:
                diff = abs(len(compact_orig) - len(compact_joined))
                # For short/medium lines, don't accept splits that drop a visible
                # prefix/suffix chunk. This prevents title-like lines from losing
                # the first few glyphs after model line-splitting.
                if len(compact_orig) <= 44 and diff >= 3:
                    return False

            ratio = min(len(compact_orig), len(compact_joined)) / max(
                1, len(compact_orig), len(compact_joined)
            )
            min_ratio = 0.45
            # Moderate guard for short/medium boxes: strict enough to prevent
            # obvious truncation, but not so strict that valid line splits fail.
            if len(compact_orig) <= 64:
                min_ratio = 0.56
            if len(compact_orig) <= 36:
                min_ratio = 0.62
            if ratio < min_ratio:
                return False

            # Guard against unstable split outputs: for wide single-line titles,
            # splitting into a very short first segment + long remainder usually
            # hurts alignment and may later trigger noise filtering.
            if len(lines) == 2:
                lens = [len(_compact_text(seg)) for seg in lines]
                short_len = min(lens) if lens else 0
                long_len = max(lens) if lens else 0
                if short_len > 0 and long_len > 0:
                    imbalance = float(short_len) / float(long_len)
                    bbox_n = row.get("bbox_n") if isinstance(row, dict) else None
                    if isinstance(bbox_n, tuple) and len(bbox_n) == 4:
                        x0, y0, x1, y1 = bbox_n
                        w = max(1.0, float(x1 - x0))
                        h = max(1.0, float(y1 - y0))
                        wide_banner_like = (
                            w >= 0.25 * float(width) and (h / max(1.0, w)) <= 0.12
                        )
                        if wide_banner_like and short_len <= 5 and imbalance < 0.30:
                            return False

            return True

        def _split_bbox_by_ink_projection(
            row: dict[str, Any],
            *,
            n_lines: int,
        ) -> list[tuple[float, float]] | None:
            """Estimate vertical line ranges from image pixels inside a bbox.

            Returns a list of (y0, y1) in absolute image pixels for each line.
            """

            try:
                import numpy as np
            except Exception:
                return None

            bbox_n = row.get("bbox_n")
            if not isinstance(bbox_n, tuple) or len(bbox_n) != 4:
                return None
            if n_lines <= 1:
                return None

            x0, y0, x1, y1 = bbox_n
            xi0 = max(0, min(width - 1, int(math.floor(float(x0)))))
            yi0 = max(0, min(height - 1, int(math.floor(float(y0)))))
            xi1 = max(0, min(width, int(math.ceil(float(x1)))))
            yi1 = max(0, min(height, int(math.ceil(float(y1)))))
            if xi1 - xi0 < 4 or yi1 - yi0 < max(6, n_lines * 3):
                return None

            try:
                gray = image.crop((xi0, yi0, xi1, yi1)).convert("L")
                arr = np.asarray(gray, dtype=np.float32)
            except Exception:
                return None

            if arr.ndim != 2 or arr.size <= 0:
                return None

            h_px, w_px = arr.shape
            if h_px < max(6, n_lines * 3):
                return None

            p95 = float(np.percentile(arr, 95.0))
            p10 = float(np.percentile(arr, 10.0))
            contrast = max(1.0, p95 - p10)
            if contrast < 8.0:
                return None

            # Convert to rough "ink" intensity (0..1), then row profile.
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

            minima: list[int] = []
            low_th = float(np.percentile(smooth, 45.0))
            for pos in range(1, h_px - 1):
                v = float(smooth[pos])
                if v > low_th:
                    continue
                if v <= float(smooth[pos - 1]) and v <= float(smooth[pos + 1]):
                    minima.append(pos)

            target_cuts = max(1, int(n_lines) - 1)
            cuts: list[int] = []
            used: set[int] = set()
            max_dist = max(3, int(round(0.22 * h_px)))
            for k_idx in range(1, target_cuts + 1):
                target = int(round(float(k_idx) * float(h_px) / float(n_lines)))
                cands = [
                    m for m in minima if m not in used and abs(m - target) <= max_dist
                ]
                if not cands:
                    continue
                chosen = min(cands, key=lambda m: abs(m - target))
                cuts.append(chosen)
                used.add(chosen)

            # Fallback: quantiles by cumulative row ink.
            if len(cuts) < target_cuts:
                prof = smooth + 1e-6
                cum = np.cumsum(prof)
                total = float(cum[-1])
                if total > 0:
                    for k_idx in range(1, target_cuts + 1):
                        target_mass = total * (float(k_idx) / float(n_lines))
                        pos = int(np.searchsorted(cum, target_mass))
                        pos = max(1, min(h_px - 2, pos))
                        cuts.append(pos)

            if len(cuts) < target_cuts:
                return None

            cuts = sorted(set(cuts))
            if len(cuts) > target_cuts:
                # Keep cuts nearest to uniform targets for stability.
                targets = [
                    int(round(float(k_idx) * float(h_px) / float(n_lines)))
                    for k_idx in range(1, target_cuts + 1)
                ]
                selected: list[int] = []
                remaining = list(cuts)
                for t in targets:
                    if not remaining:
                        break
                    best = min(remaining, key=lambda c: abs(c - t))
                    selected.append(best)
                    remaining.remove(best)
                cuts = sorted(selected)

            bounds = [0] + cuts + [h_px]
            if len(bounds) != n_lines + 1:
                return None

            ranges: list[tuple[float, float]] = []
            prev_y = float(y0)
            for idx in range(n_lines):
                by0 = int(bounds[idx])
                by1 = int(bounds[idx + 1])
                if by1 - by0 < 1:
                    continue
                ly0 = float(y0) + float(by0)
                ly1 = float(y0) + float(by1)
                ly0 = max(float(y0), min(float(y1) - 1.0, ly0))
                ly1 = max(ly0 + 1.0, min(float(y1), ly1))
                if ly0 < prev_y:
                    ly0 = prev_y
                if ly1 <= ly0:
                    continue
                ranges.append((ly0, ly1))
                prev_y = ly1

            if len(ranges) != n_lines:
                return None

            heights = [max(0.0, ly1 - ly0) for (ly0, ly1) in ranges]
            if not heights:
                return None
            avg_h = float(sum(heights)) / float(max(1, len(heights)))
            min_h = min(heights)
            max_h = max(heights)
            # Guard against unstable projection cuts (over-compressed lines).
            # If line heights are too imbalanced, fallback to equal split.
            if avg_h > 0:
                if min_h < max(1.0, 0.55 * avg_h):
                    return None
                if max_h > (1.80 * avg_h):
                    return None

            return ranges

        def _tighten_line_bbox_x_by_ink(
            row: dict[str, Any],
            *,
            ly0: float,
            ly1: float,
            fallback_x0: float,
            fallback_x1: float,
        ) -> tuple[float, float]:
            """Best-effort horizontal tightening for a single split line bbox."""

            try:
                import numpy as np
            except Exception:
                return (float(fallback_x0), float(fallback_x1))

            bbox_n = row.get("bbox_n")
            if not isinstance(bbox_n, tuple) or len(bbox_n) != 4:
                return (float(fallback_x0), float(fallback_x1))
            x0, y0, x1, y1 = bbox_n
            base_x0 = float(min(x0, x1))
            base_x1 = float(max(x0, x1))
            if base_x1 - base_x0 < 6.0:
                return (float(fallback_x0), float(fallback_x1))

            seg_y0 = max(float(y0), min(float(y1) - 1.0, float(ly0)))
            seg_y1 = max(seg_y0 + 1.0, min(float(y1), float(ly1)))
            if seg_y1 - seg_y0 < 2.0:
                return (float(fallback_x0), float(fallback_x1))

            xi0 = max(0, min(width - 1, int(math.floor(base_x0))))
            xi1 = max(0, min(width, int(math.ceil(base_x1))))
            yi0 = max(0, min(height - 1, int(math.floor(seg_y0))))
            yi1 = max(0, min(height, int(math.ceil(seg_y1))))
            if (xi1 - xi0) < 6 or (yi1 - yi0) < 2:
                return (float(fallback_x0), float(fallback_x1))

            try:
                gray = image.crop((xi0, yi0, xi1, yi1)).convert("L")
                arr = np.asarray(gray, dtype=np.float32)
            except Exception:
                return (float(fallback_x0), float(fallback_x1))

            if arr.ndim != 2 or arr.size <= 0:
                return (float(fallback_x0), float(fallback_x1))
            h_px, w_px = arr.shape
            if w_px < 6 or h_px < 2:
                return (float(fallback_x0), float(fallback_x1))

            p95 = float(np.percentile(arr, 95.0))
            p10 = float(np.percentile(arr, 10.0))
            contrast = max(1.0, p95 - p10)
            if contrast < 8.0:
                return (float(fallback_x0), float(fallback_x1))

            ink = np.clip((p95 - arr) / contrast, 0.0, 1.0)
            ink_mask = (ink >= 0.16).astype(np.float32)
            col_profile = ink_mask.mean(axis=0)
            if float(np.sum(col_profile)) <= max(0.015 * w_px, 1.0):
                return (float(fallback_x0), float(fallback_x1))

            # Prefer robust, not over-tight, trimming.
            th = float(np.percentile(col_profile, 65.0))
            th = max(0.04, min(0.22, th))
            active = np.where(col_profile >= th)[0]
            if active.size == 0:
                return (float(fallback_x0), float(fallback_x1))

            left_idx = int(active[0])
            right_idx = int(active[-1]) + 1
            if right_idx - left_idx < 3:
                return (float(fallback_x0), float(fallback_x1))

            base_w = max(1.0, base_x1 - base_x0)
            margin_px = max(1, int(round(0.025 * float(base_w))))
            left_idx = max(0, left_idx - margin_px)
            right_idx = min(w_px, right_idx + margin_px)
            if right_idx - left_idx < 3:
                return (float(fallback_x0), float(fallback_x1))

            tx0 = float(xi0 + left_idx)
            tx1 = float(xi0 + right_idx)
            tightened_w = max(1.0, tx1 - tx0)

            # Guard against unstable over-shrink, especially for short lines.
            line_text = str(row.get("text") or "")
            compact_len = len(_compact_text(line_text))
            min_ratio = 0.28 if compact_len <= 8 else 0.22
            if tightened_w < (min_ratio * base_w):
                return (float(fallback_x0), float(fallback_x1))

            # Never expand beyond original fallback bounds.
            tx0 = max(float(fallback_x0), min(float(fallback_x1) - 1.0, tx0))
            tx1 = min(float(fallback_x1), max(float(fallback_x0) + 1.0, tx1))
            if tx1 <= tx0:
                return (float(fallback_x0), float(fallback_x1))
            return (float(tx0), float(tx1))

        def _estimate_target_lines(row: dict[str, Any]) -> int:
            bbox_n = row.get("bbox_n")
            if not isinstance(bbox_n, tuple) or len(bbox_n) != 4:
                return 1
            _, y0, _, y1 = bbox_n
            h = max(1.0, float(y1 - y0))
            baseline = max(8.0, float(median_h))
            est = int(round(h / baseline))
            if _is_multiline_candidate_for_linebreak_assist(
                text=str(row.get("text") or ""),
                bbox=row.get("bbox_n"),
                image_width=width,
                image_height=height,
                median_line_height=median_h,
            ):
                est = max(est, 2)
            est = max(1, min(est, max(2, int(max_lines_per_item))))
            return est

        def _split_into_sentences(text: str) -> list[str]:
            cleaned = " ".join(str(text or "").split()).strip()
            if not cleaned:
                return []
            out: list[str] = []
            buf = ""
            for ch in cleaned:
                buf += ch
                if ch in "。！？!?；;":
                    seg = buf.strip()
                    if seg:
                        out.append(seg)
                    buf = ""
            if buf.strip():
                out.append(buf.strip())
            return out

        def _fallback_split_lines(original_text: str, target_lines: int) -> list[str]:
            target_lines = max(1, int(target_lines))
            normalized = " ".join(str(original_text or "").split()).strip()
            if not normalized:
                return []
            if target_lines <= 1:
                return [normalized]

            sentences = _split_into_sentences(normalized)
            if not sentences:
                sentences = [normalized]

            if len(sentences) < target_lines:
                finer: list[str] = []
                for seg in sentences:
                    parts = [
                        p.strip() for p in re.split(r"(?<=[，,：:])", seg) if p.strip()
                    ]
                    if len(parts) > 1:
                        finer.extend(parts)
                    else:
                        finer.append(seg)
                if finer:
                    sentences = finer

            if len(sentences) <= 1:
                compact_len = len(_compact_text(normalized))
                if compact_len < target_lines * 4:
                    return [normalized]
                per_line = max(4, int(round(compact_len / float(target_lines))))
                out: list[str] = []
                buf = ""
                buf_len = 0
                for ch in normalized:
                    buf += ch
                    if ch.isspace():
                        continue
                    buf_len += 1
                    if buf_len >= per_line and len(out) < (target_lines - 1):
                        seg = buf.strip()
                        if seg:
                            out.append(seg)
                        buf = ""
                        buf_len = 0
                if buf.strip():
                    out.append(buf.strip())
                return [seg for seg in out if seg]

            desired = max(2, min(target_lines, max(2, int(max_lines_per_item))))
            total = sum(max(1, len(_compact_text(seg))) for seg in sentences)
            target_chars = max(6.0, float(total) / float(desired))

            out: list[str] = []
            cur_parts: list[str] = []
            cur_chars = 0.0
            for idx, seg in enumerate(sentences):
                seg_chars = float(max(1, len(_compact_text(seg))))
                cur_parts.append(seg)
                cur_chars += seg_chars

                slots_left = max(1, desired - len(out))
                segments_left = len(sentences) - idx - 1
                should_cut = len(out) < (desired - 1) and (
                    cur_chars >= target_chars or segments_left <= (slots_left - 1)
                )
                if should_cut:
                    merged = "".join(cur_parts).strip()
                    if merged:
                        out.append(merged)
                    cur_parts = []
                    cur_chars = 0.0

            if cur_parts:
                merged = "".join(cur_parts).strip()
                if merged:
                    out.append(merged)

            return [seg for seg in out if seg]

        def _has_strong_two_line_split_cue(text: str) -> bool:
            normalized = " ".join(str(text or "").split()).strip()
            if not normalized:
                return False

            stripped = normalized.lstrip()
            if stripped.startswith(("-", "•", "·", "●", "▪", "▶", "◆", "■", "*")):
                return True

            parts = re.split(r"[：:]", normalized, maxsplit=1)
            if len(parts) == 2:
                head = _compact_text(parts[0])
                tail = _compact_text(parts[1])
                if 2 <= len(head) <= 26 and len(tail) >= 2:
                    return True

            if re.match(r"^\s*[（(]?[0-9一二三四五六七八九十]+[）).、]", normalized):
                return True

            return False

        def _is_structured_multiline_text(text: str) -> bool:
            raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
            if "\n" not in raw:
                return False

            lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
            if len(lines) < 3:
                return False

            compact_lens = [len(_compact_text(ln)) for ln in lines]
            if not compact_lens:
                return False

            marker_lines = 0
            for ln in lines:
                if any(tok in ln for tok in ("【", "】", "[", "]", "#", "##")):
                    marker_lines += 1

            avg_len = float(sum(compact_lens)) / float(max(1, len(compact_lens)))
            max_len = max(compact_lens)

            # Template/spec-like multiline blocks are often already close to
            # intended line structure; avoid AI split from over-fragmenting.
            if (
                marker_lines >= max(2, int(round(0.35 * len(lines))))
                and avg_len <= 34.0
            ):
                return True
            if marker_lines >= 2 and len(lines) >= 5 and max_len <= 48:
                return True

            return False

        def _allow_split_for_row(
            *,
            original_text: str,
            lines: list[str],
            row: dict[str, Any],
        ) -> bool:
            if len(lines) <= 1:
                return False
            if "\n" in str(original_text or ""):
                return True
            if len(lines) != 2:
                return True

            estimated = _estimate_target_lines(row)
            if estimated >= 3:
                return True

            compact_len = len(_compact_text(original_text))
            if compact_len <= 34:
                return True

            # Paragraph-like long text with only two inferred lines is usually
            # more stable when kept in one bbox and rendered with adaptive wrap.
            return _has_strong_two_line_split_cue(original_text)

        split_count = 0
        fallback_split_count = 0
        x_tighten_count = 0
        out_items: list[dict] = []
        for idx, original in enumerate(items):
            if not isinstance(original, dict):
                continue
            lines = split_map.get(idx) or []
            row = row_map.get(idx)
            if row is None:
                out_items.append(dict(original))
                continue

            original_text = str(row.get("text") or "")

            if idx in candidate_idx_set and _is_structured_multiline_text(
                original_text
            ):
                out_items.append(dict(original))
                continue

            clean_lines = [
                str(seg).strip() for seg in (lines or []) if str(seg).strip()
            ]

            if (
                allow_heuristic_fallback
                and (not clean_lines or len(clean_lines) <= 1)
                and idx in candidate_idx_set
            ):
                estimated = _estimate_target_lines(row)
                if estimated >= 2:
                    fallback_lines = _fallback_split_lines(original_text, estimated)
                    if len(fallback_lines) >= 2:
                        clean_lines = fallback_lines
                        fallback_split_count += 1

            if not _allow_split_for_row(
                original_text=original_text,
                lines=clean_lines,
                row=row,
            ):
                out_items.append(dict(original))
                continue

            if not _split_is_plausible(original_text, clean_lines, row=row):
                out_items.append(dict(original))
                continue

            bbox_n = row.get("bbox_n")
            if not isinstance(bbox_n, tuple) or len(bbox_n) != 4:
                out_items.append(dict(original))
                continue

            x0, y0, x1, y1 = bbox_n
            n = max(1, len(clean_lines))
            total_h = max(1.0, float(y1 - y0))

            ranges = _split_bbox_by_ink_projection(row, n_lines=n)

            for line_idx, text_line in enumerate(clean_lines):
                if ranges is not None and line_idx < len(ranges):
                    ly0, ly1 = ranges[line_idx]
                else:
                    ly0 = y0 + total_h * float(line_idx) / float(n)
                    ly1 = y0 + total_h * float(line_idx + 1) / float(n)
                if ly1 - ly0 < 1.0:
                    continue

                tx0, tx1 = _tighten_line_bbox_x_by_ink(
                    row,
                    ly0=float(ly0),
                    ly1=float(ly1),
                    fallback_x0=float(x0),
                    fallback_x1=float(x1),
                )
                if (tx1 - tx0) < (x1 - x0):
                    x_tighten_count += 1

                new_item = dict(original)
                new_item["text"] = text_line
                new_item["bbox"] = [float(tx0), float(ly0), float(tx1), float(ly1)]
                new_item["linebreak_assisted"] = True
                new_item["linebreak_assist_source"] = (
                    "ai" if idx in split_map else "heuristic_fallback"
                )
                out_items.append(new_item)

            split_count += 1

        if split_count > 0:
            logger.info(
                "AI OCR line-break assist applied: split_boxes=%s/%s (fallback=%s, x_tightened=%s)",
                split_count,
                len(items),
                fallback_split_count,
                x_tighten_count,
            )

        return out_items
