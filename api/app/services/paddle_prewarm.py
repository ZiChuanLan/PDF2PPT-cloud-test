"""Optional PaddleOCR-VL doc_parser prewarm helpers for container startup."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time

from app.convert.ocr.ai_client import AiOcrClient
from app.convert.ocr.base import _clean_str, _env_flag
from app.convert.ocr.utils import _is_paddleocr_vl_model
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaddleDocPrewarmConfig:
    """Resolved config for an optional PaddleOCR-VL prewarm run."""

    provider: str
    api_key: str
    base_url: str | None
    model: str
    max_side_px: int | None
    required: bool
    service_role: str


def _env_int_or_none(name: str) -> int | None:
    raw_value = _clean_str(os.getenv(name))
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except Exception:
        return None


def _resolve_service_role(explicit_role: str | None = None) -> str:
    role = _clean_str(explicit_role) or _clean_str(os.getenv("APP_SERVICE_ROLE")) or ""
    return role.lower() or "unknown"


def _should_prewarm_for_role(service_role: str) -> bool:
    target_raw = (
        _clean_str(os.getenv("OCR_PADDLE_VL_PREWARM_TARGET")) or "worker"
    ).lower()
    if target_raw in {"all", "both", "*"}:
        return True
    targets = {
        token.strip().lower()
        for token in target_raw.split(",")
        if token.strip()
    }
    return service_role.lower() in targets


def resolve_paddle_doc_prewarm_config(
    *,
    service_role: str | None = None,
) -> PaddleDocPrewarmConfig | None:
    """Resolve optional prewarm config from environment variables."""

    if not _env_flag("OCR_PADDLE_VL_PREWARM", default=False):
        return None

    resolved_role = _resolve_service_role(service_role)
    if not _should_prewarm_for_role(resolved_role):
        return None

    model = _clean_str(os.getenv("OCR_PADDLE_VL_PREWARM_MODEL")) or _clean_str(
        os.getenv("SILICONFLOW_MODEL")
    )
    if not model or not _is_paddleocr_vl_model(model):
        return None

    api_key = _clean_str(os.getenv("OCR_PADDLE_VL_PREWARM_API_KEY")) or _clean_str(
        os.getenv("SILICONFLOW_API_KEY")
    )
    if not api_key:
        return None

    provider = (
        _clean_str(os.getenv("OCR_PADDLE_VL_PREWARM_PROVIDER")) or "siliconflow"
    )
    base_url = _clean_str(os.getenv("OCR_PADDLE_VL_PREWARM_BASE_URL")) or _clean_str(
        os.getenv("SILICONFLOW_BASE_URL")
    )
    max_side_px = _env_int_or_none("OCR_PADDLE_VL_PREWARM_MAX_SIDE_PX")
    if max_side_px is None:
        max_side_px = _env_int_or_none("OCR_PADDLE_VL_DOCPARSER_MAX_SIDE_PX")

    return PaddleDocPrewarmConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_side_px=max_side_px,
        required=_env_flag("OCR_PADDLE_VL_PREWARM_REQUIRED", default=False),
        service_role=resolved_role,
    )


def run_paddle_doc_prewarm(*, service_role: str | None = None) -> bool:
    """Prewarm PaddleOCR-VL doc_parser if explicitly enabled."""

    config = resolve_paddle_doc_prewarm_config(service_role=service_role)
    if config is None:
        logger.info(
            "Skipping PaddleOCR-VL prewarm (service_role=%s)",
            _resolve_service_role(service_role),
        )
        return False

    started_at = time.perf_counter()
    try:
        client = AiOcrClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            provider=config.provider,
            paddle_doc_max_side_px=config.max_side_px,
        )
        client._get_paddle_doc_parser()
        elapsed_s = time.perf_counter() - started_at
        logger.info(
            "PaddleOCR-VL prewarm finished in %.2fs (service_role=%s, provider=%s, model=%s)",
            elapsed_s,
            config.service_role,
            config.provider,
            config.model,
        )
        return True
    except Exception:
        elapsed_s = time.perf_counter() - started_at
        if config.required:
            logger.exception(
                "PaddleOCR-VL prewarm failed after %.2fs and is required",
                elapsed_s,
            )
            raise
        logger.exception(
            "PaddleOCR-VL prewarm failed after %.2fs; continuing without prewarm",
            elapsed_s,
        )
        return False


def main() -> int:
    """CLI entrypoint used by container startup scripts."""

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    run_paddle_doc_prewarm()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
