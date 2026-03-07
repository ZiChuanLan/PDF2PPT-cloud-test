"""Runtime availability probes for local OCR engines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import (
    _clean_str,
    _normalize_paddle_language,
    _normalize_tesseract_language,
    _split_tesseract_languages,
)


def probe_local_tesseract(*, language: str | None = None) -> dict[str, Any]:
    """Probe local Tesseract runtime and language-pack availability."""

    requested_language = _normalize_tesseract_language(language)
    requested_languages = _split_tesseract_languages(requested_language)

    python_package_available = False
    binary_available = False
    languages_probe_ok = False
    version: str | None = None
    available_languages: list[str] = []
    missing_languages: list[str] = []
    issues: list[str] = []

    try:
        import pytesseract

        python_package_available = True
    except ImportError:
        issues.append("pytesseract_not_installed")
        return {
            "provider": "tesseract",
            "requested_language": requested_language,
            "requested_languages": requested_languages,
            "python_package_available": False,
            "binary_available": False,
            "version": None,
            "available_languages": [],
            "missing_languages": requested_languages,
            "issues": issues,
            "ready": False,
            "message": "pytesseract package is not installed",
        }

    try:
        version_raw = pytesseract.get_tesseract_version()
        version = str(version_raw).replace("\n", " ").strip() or None
        binary_available = True
    except Exception as e:
        issues.append(f"tesseract_binary_unavailable:{e!s}")

    if binary_available:
        try:
            raw_languages = pytesseract.get_languages(config="") or []
            unique_languages = {
                str(item).strip() for item in raw_languages if str(item).strip()
            }
            available_languages = sorted(unique_languages)
            languages_probe_ok = True
        except Exception as e:
            issues.append(f"tesseract_languages_probe_failed:{e!s}")

    if languages_probe_ok and requested_languages:
        available_set = {lang.lower() for lang in available_languages}
        missing_languages = [
            lang for lang in requested_languages if lang.lower() not in available_set
        ]
        if missing_languages:
            issues.append("tesseract_missing_languages")

    ready = (
        python_package_available
        and binary_available
        and (not languages_probe_ok or not missing_languages)
    )

    if not python_package_available:
        message = "pytesseract package is not installed"
    elif not binary_available:
        message = "tesseract executable is not available"
    elif missing_languages:
        message = f"Missing tesseract language packs: {', '.join(missing_languages)}"
    elif issues:
        message = "Local Tesseract OCR is available with warnings"
    else:
        message = "Local Tesseract OCR is ready"

    return {
        "provider": "tesseract",
        "requested_language": requested_language,
        "requested_languages": requested_languages,
        "python_package_available": python_package_available,
        "binary_available": binary_available,
        "version": version,
        "available_languages": available_languages,
        "missing_languages": missing_languages,
        "issues": issues,
        "ready": ready,
        "message": message,
    }


def probe_local_paddleocr(*, language: str | None = None) -> dict[str, Any]:
    """Probe local PaddleOCR runtime availability."""

    requested_language = _normalize_paddle_language(language)
    python_package_available = False
    runtime_available = False
    version: str | None = None
    available_languages: list[str] = [
        "ch",
        "en",
        "latin",
        "arabic",
        "cyrillic",
        "devanagari",
    ]
    missing_languages: list[str] = []
    issues: list[str] = []

    try:
        import paddleocr as paddleocr_module

        python_package_available = True
        version = (
            str(getattr(paddleocr_module, "__version__", "") or "").strip() or None
        )
    except ImportError:
        issues.append("paddleocr_not_installed")
        return {
            "provider": "paddle",
            "requested_language": requested_language,
            "requested_languages": [requested_language],
            "python_package_available": False,
            "binary_available": False,
            "version": None,
            "available_languages": available_languages,
            "missing_languages": [requested_language],
            "issues": issues,
            "ready": False,
            "message": "paddleocr package is not installed",
        }

    # For local packaging (e.g. exe), keep the probe lightweight and offline:
    # validate imports only, avoid constructing OCR engines that may download models.
    try:
        import paddle
        from paddleocr import PaddleOCR

        _ = getattr(paddle, "__version__", None)
        _ = PaddleOCR
        runtime_available = True
    except Exception as e:
        issues.append(f"paddleocr_runtime_unavailable:{e!s}")

    if requested_language not in available_languages:
        missing_languages.append(requested_language)
        issues.append("paddleocr_language_maybe_unsupported")

    ready = bool(python_package_available and runtime_available)
    if not python_package_available:
        message = "paddleocr package is not installed"
    elif not runtime_available:
        message = "PaddleOCR runtime is not ready"
    elif missing_languages:
        message = (
            "PaddleOCR runtime is ready, but requested language may be unsupported"
        )
    elif issues:
        message = "PaddleOCR is available with warnings"
    else:
        message = "Local PaddleOCR is ready"

    return {
        "provider": "paddle",
        "requested_language": requested_language,
        "requested_languages": [requested_language],
        "python_package_available": python_package_available,
        "binary_available": runtime_available,
        "version": version,
        "available_languages": available_languages,
        "missing_languages": missing_languages,
        "issues": issues,
        "ready": ready,
        "message": message,
    }


def probe_local_tesseract_models(*, language: str | None = None) -> dict[str, Any]:
    """Probe local Tesseract language-pack availability as model existence."""

    runtime = probe_local_tesseract(language=language)
    requested_language = str(
        runtime.get("requested_language") or _normalize_tesseract_language(language)
    ).strip()
    requested_languages = [
        str(item).strip()
        for item in (runtime.get("requested_languages") or [])
        if str(item).strip()
    ]
    available_languages = [
        str(item).strip()
        for item in (runtime.get("available_languages") or [])
        if str(item).strip()
    ]
    available_set = {item.lower() for item in available_languages}
    found_models = [
        lang for lang in requested_languages if lang and lang.lower() in available_set
    ]
    missing_models = [
        lang for lang in requested_languages if lang and lang.lower() not in available_set
    ]
    issues = [
        str(item).strip()
        for item in (runtime.get("issues") or [])
        if str(item).strip()
    ]
    version = str(runtime.get("version") or "").strip() or None
    model_root_dir = _clean_str(os.getenv("TESSDATA_PREFIX")) or None
    python_package_available = bool(runtime.get("python_package_available"))
    binary_available = bool(runtime.get("binary_available"))
    model_files: list[str] = []

    if model_root_dir:
        model_root = Path(model_root_dir)
        if model_root.exists() and model_root.is_dir():
            for lang in requested_languages:
                lang_clean = str(lang or "").strip()
                if not lang_clean:
                    continue
                candidate = model_root / f"{lang_clean}.traineddata"
                if candidate.exists():
                    model_files.append(str(candidate))

    if not python_package_available:
        message = "pytesseract package is not installed"
    elif not binary_available:
        message = "tesseract executable is not available"
    elif missing_models:
        message = f"Missing tesseract language packs: {', '.join(missing_models)}"
    else:
        message = "Tesseract language packs are ready"

    ready = bool(python_package_available and binary_available and not missing_models)
    return {
        "provider": "tesseract_models",
        "requested_language": requested_language,
        "requested_languages": requested_languages,
        "python_package_available": python_package_available,
        "binary_available": binary_available,
        "version": version,
        "available_languages": available_languages,
        "missing_languages": missing_models,
        "model_root_dir": model_root_dir,
        "required_models": requested_languages,
        "found_models": found_models,
        "missing_models": missing_models,
        "model_files": sorted(dict.fromkeys(model_files)),
        "issues": issues,
        "ready": ready,
        "message": message,
    }


def _resolve_paddle_model_roots() -> list[Path]:
    roots: list[Path] = []

    override_dir = _clean_str(os.getenv("PADDLE_OCR_MODEL_DIR"))
    if override_dir:
        roots.append(Path(override_dir).expanduser())

    paddleocr_home = _clean_str(os.getenv("PADDLEOCR_HOME"))
    if paddleocr_home:
        home_root = Path(paddleocr_home).expanduser()
        roots.extend([home_root / "whl", home_root])

    xdg_cache = _clean_str(os.getenv("XDG_CACHE_HOME"))
    if xdg_cache:
        cache_root = Path(xdg_cache).expanduser() / "paddleocr"
        roots.extend([cache_root / "whl", cache_root])

    local_default = Path.home() / ".paddleocr"
    roots.extend([local_default / "whl", local_default])

    uniq: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


def _is_paddle_model_file(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() in {".pdmodel", ".pdiparams", ".onnx"}:
        return True
    if name in {"inference.yml", "inference.json"}:
        return True
    return False


def probe_local_paddle_models(*, language: str | None = None) -> dict[str, Any]:
    """Probe whether local PaddleOCR model files exist in cache directories."""

    requested_language = _normalize_paddle_language(language)
    runtime = probe_local_paddleocr(language=requested_language)
    python_package_available = bool(runtime.get("python_package_available"))
    binary_available = bool(runtime.get("binary_available"))
    version = str(runtime.get("version") or "").strip() or None
    available_languages = [
        str(item).strip()
        for item in (runtime.get("available_languages") or [])
        if str(item).strip()
    ]
    runtime_missing_languages = [
        str(item).strip()
        for item in (runtime.get("missing_languages") or [])
        if str(item).strip()
    ]
    roots = _resolve_paddle_model_roots()
    existing_roots = [root for root in roots if root.exists()]

    # Need at least det + rec for normal OCR. cls is optional.
    required_tokens = ["det", "rec"]
    optional_tokens = ["cls"]
    token_matches: dict[str, list[str]] = {"det": [], "rec": [], "cls": []}
    issues: list[str] = [
        str(item).strip()
        for item in (runtime.get("issues") or [])
        if str(item).strip()
    ]

    if not existing_roots:
        if not python_package_available:
            message = "paddleocr package is not installed"
        elif not binary_available:
            message = "PaddleOCR runtime is not ready"
        else:
            message = "PaddleOCR model cache directory not found"
        return {
            "provider": "paddle_models",
            "requested_language": requested_language,
            "requested_languages": [requested_language],
            "python_package_available": python_package_available,
            "binary_available": binary_available,
            "version": version,
            "available_languages": available_languages,
            "missing_languages": runtime_missing_languages,
            "model_root_dir": str(roots[0]) if roots else None,
            "required_models": required_tokens,
            "found_models": [],
            "missing_models": required_tokens,
            "model_files": [],
            "issues": sorted(dict.fromkeys([*issues, "paddle_model_root_not_found"])),
            "ready": False,
            "message": message,
        }

    scan_roots: list[Path] = []
    for root in existing_roots:
        lang_root = root / requested_language
        if lang_root.exists():
            scan_roots.append(lang_root)
        scan_roots.append(root)

    deduped_scan_roots: list[Path] = []
    seen_scan_roots: set[str] = set()
    for root in scan_roots:
        key = str(root)
        if key in seen_scan_roots:
            continue
        seen_scan_roots.add(key)
        deduped_scan_roots.append(root)

    for root in deduped_scan_roots:
        try:
            for file_path in root.rglob("*"):
                if not file_path.is_file() or not _is_paddle_model_file(file_path):
                    continue
                full = file_path.as_posix().lower()
                rel = str(file_path)
                for token in (required_tokens + optional_tokens):
                    if token in full and len(token_matches[token]) < 5:
                        token_matches[token].append(rel)
        except Exception as e:
            issues.append(f"paddle_model_scan_failed:{e!s}")

    found_models = [
        token
        for token in required_tokens + optional_tokens
        if token_matches.get(token)
    ]
    missing_models = [token for token in required_tokens if not token_matches.get(token)]
    model_files: list[str] = []
    for token in required_tokens + optional_tokens:
        model_files.extend(token_matches.get(token) or [])

    model_files = sorted(dict.fromkeys(model_files))[:12]

    if not python_package_available:
        message = "paddleocr package is not installed"
    elif not binary_available:
        message = "PaddleOCR runtime is not ready"
    elif missing_models:
        message = f"PaddleOCR model files missing: {', '.join(missing_models)}"
    elif not token_matches.get("cls"):
        issues.append("paddle_cls_model_missing_optional")
        message = "PaddleOCR det/rec models are ready (cls optional model not found)"
    else:
        message = "PaddleOCR model files are ready"

    ready = bool(python_package_available and binary_available and not missing_models)
    return {
        "provider": "paddle_models",
        "requested_language": requested_language,
        "requested_languages": [requested_language],
        "python_package_available": python_package_available,
        "binary_available": binary_available,
        "version": version,
        "available_languages": available_languages,
        "missing_languages": runtime_missing_languages,
        "model_root_dir": str(existing_roots[0]),
        "required_models": required_tokens,
        "found_models": found_models,
        "missing_models": missing_models,
        "model_files": model_files,
        "issues": sorted(dict.fromkeys(issues)),
        "ready": ready,
        "message": message,
    }


__all__ = [
    "probe_local_paddle_models",
    "probe_local_paddleocr",
    "probe_local_tesseract",
    "probe_local_tesseract_models",
]
