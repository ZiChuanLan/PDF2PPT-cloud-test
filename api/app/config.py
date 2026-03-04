"""Application configuration."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    max_file_mb: int = 100
    max_pages: int = 200
    job_ttl_minutes: int = 60
    # Keepalive heartbeat interval for long-running blocking stages.
    # Used to refresh job metadata TTL while no progress update is emitted.
    job_keepalive_interval_s: int = 15
    # Root directory for per-job runtime artifacts.
    # Relative paths are resolved under the `api/` directory.
    job_root_dir: str = "data/jobs"
    redis_url: str = "redis://redis:6379/0"
    log_level: str = "INFO"
    # Rendering quality knobs for scanned PDFs.
    #
    # - ocr_render_dpi: higher DPI improves OCR recall/accuracy on scan-heavy decks.
    # - scanned_render_dpi: controls the background render quality in the PPTX
    #   output (higher DPI looks sharper but increases file size).
    # NOTE: Higher values (250-300) can improve OCR on some documents but may
    # degrade on others and increases CPU/memory usage. Keep conservative defaults.
    ocr_render_dpi: int = 200
    scanned_render_dpi: int = 200
    # Parallel workers for text style fitting during PPT generation.
    # This does not reduce output quality; it only parallelizes CPU-heavy
    # text measurement/fitting on dense pages.
    ppt_text_fit_workers: int = 2
    siliconflow_api_key: str | None = None
    siliconflow_base_url: str | None = "https://api.siliconflow.cn/v1"
    siliconflow_model: str | None = "Pro/deepseek-ai/deepseek-ocr"
    # Per-page OCR timeout in seconds.  If a single page takes longer than
    # this the page is skipped with a warning instead of blocking the whole job.
    ocr_page_timeout_s: int = 300
    # Circuit-breaker for repeated page-level timeouts. When consecutive OCR
    # pages hit timeout this many times, skip remaining OCR pages so the job
    # can continue to PPTX generation instead of appearing stuck.
    ocr_max_consecutive_timeouts: int = 2
    # Overall OCR stage timeout in seconds.  When exceeded the remaining pages
    # are skipped and the job continues to PPTX generation.
    ocr_total_timeout_s: int = 3600
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_allow_origin_regex: str | None = None

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def parse_cors_allow_origins(raw: str | None) -> list[str]:
    value = str(raw or "").strip()
    if not value:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    if value == "*":
        return ["*"]
    items: list[str] = []
    for item in value.split(","):
        origin = item.strip()
        if origin and origin not in items:
            items.append(origin)
    return items or ["http://localhost:3000", "http://127.0.0.1:3000"]
