"""Performance-oriented runtime policy helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def resolve_page_artifact_export(
    *,
    enabled: bool | None,
    total_pages: int | None,
    max_pages: int | None = None,
) -> bool:
    """Return whether per-page artifact export should stay enabled."""

    if not bool(enabled):
        return False

    try:
        normalized_total_pages = max(0, int(total_pages or 0))
    except Exception:
        normalized_total_pages = 0

    try:
        normalized_max_pages = max(0, int(max_pages or 0))
    except Exception:
        normalized_max_pages = 0

    if normalized_max_pages > 0 and normalized_total_pages > normalized_max_pages:
        return False
    return True


@dataclass(frozen=True)
class ArtifactExportPolicy:
    layout_assist_debug_images: bool
    final_preview_images: bool


@dataclass(frozen=True)
class ArtifactExportSettings:
    ocr_overlay_images: bool = False
    layout_assist_debug_images: bool = False
    final_preview_images: bool = False
    final_preview_max_pages: int = 5

    @classmethod
    def from_settings(cls, settings: Any) -> "ArtifactExportSettings":
        return cls(
            ocr_overlay_images=bool(
                getattr(settings, "export_ocr_overlay_images", False)
            ),
            layout_assist_debug_images=bool(
                getattr(settings, "export_layout_assist_debug_images", False)
            ),
            final_preview_images=bool(
                getattr(settings, "export_final_preview_images", False)
            ),
            final_preview_max_pages=int(
                getattr(settings, "export_final_preview_max_pages", 5) or 0
            ),
        )

    def resolve_for_parsed_document(self, *, parsed_pages: int) -> ArtifactExportPolicy:
        return ArtifactExportPolicy(
            layout_assist_debug_images=resolve_page_artifact_export(
                enabled=self.layout_assist_debug_images,
                total_pages=parsed_pages,
            ),
            final_preview_images=resolve_page_artifact_export(
                enabled=self.final_preview_images,
                total_pages=parsed_pages,
                max_pages=self.final_preview_max_pages,
            ),
        )

    def resolve_ocr_overlay_images(self, *, ocr_target_pages: int) -> bool:
        return resolve_page_artifact_export(
            enabled=self.ocr_overlay_images,
            total_pages=ocr_target_pages,
        )


@dataclass(frozen=True)
class RuntimePerformanceSettings:
    ocr_render_dpi: int = 200
    scanned_render_dpi: int = 200
    keepalive_interval_s: float = 15.0
    artifact_exports: ArtifactExportSettings = field(
        default_factory=ArtifactExportSettings
    )

    @classmethod
    def from_settings(cls, settings: Any) -> "RuntimePerformanceSettings":
        return cls(
            ocr_render_dpi=int(getattr(settings, "ocr_render_dpi", 200) or 200),
            scanned_render_dpi=int(
                getattr(settings, "scanned_render_dpi", 200) or 200
            ),
            keepalive_interval_s=float(
                getattr(settings, "job_keepalive_interval_s", 15) or 15
            ),
            artifact_exports=ArtifactExportSettings.from_settings(settings),
        )
