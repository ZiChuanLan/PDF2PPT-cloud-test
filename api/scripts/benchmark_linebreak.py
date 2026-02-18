"""Benchmark OCR AI line-break assist with source-image fidelity metrics.

Usage example:
  REDIS_URL=memory:// PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  .venv/bin/python scripts/benchmark_linebreak.py \
    --pdf ../+AI智能体开发大学生指南.pdf \
    --pages 1-10 \
    --out-dir ../test/benchmark-linebreak \
    --ocr-provider paddle \
    --ocr-ai-provider siliconflow \
    --ocr-ai-key "$SILICONFLOW_API_KEY" \
    --ocr-ai-base-url https://api.siliconflow.cn/v1 \
    --ocr-ai-model PaddlePaddle/PaddleOCR-VL-1.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pymupdf


def _ensure_api_on_path() -> None:
    api_dir = Path(__file__).resolve().parents[1]
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))


def _parse_pages_spec(spec: str, *, max_pages: int) -> list[int]:
    out: set[int] = set()
    for token in [t.strip() for t in (spec or "").split(",") if t.strip()]:
        if "-" in token:
            a, b = token.split("-", 1)
            start = int(a)
            end = int(b)
            if start > end:
                start, end = end, start
            for page_no in range(start, end + 1):
                if 1 <= page_no <= max_pages:
                    out.add(page_no)
        else:
            page_no = int(token)
            if 1 <= page_no <= max_pages:
                out.add(page_no)
    return sorted(out)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class JobOutputs:
    job_id: str
    output_pptx: Path
    ir_ocr: Path
    ocr_debug: Path
    final_png: Path


def _run_single_page_job(
    *,
    page_no: int,
    linebreak_assist: bool,
    pdf_path: Path,
    jobs_root: Path,
    ocr_provider: str,
    ocr_ai_provider: str,
    ocr_ai_key: str,
    ocr_ai_base_url: str,
    ocr_ai_model: str,
    ocr_geometry_mode: str,
    ocr_strict_mode: bool,
) -> JobOutputs:
    _ensure_api_on_path()
    from app.worker import process_pdf_job

    tag = "on" if linebreak_assist else "off"
    job_id = f"bench-p{page_no:03d}-{tag}-{uuid.uuid4()}"
    job_dir = jobs_root / job_id
    _ensure_dir(job_dir)
    shutil.copy2(pdf_path, job_dir / "input.pdf")

    process_pdf_job(
        job_id,
        enable_ocr=True,
        parse_provider="local",
        page_start=page_no,
        page_end=page_no,
        ocr_provider=ocr_provider,
        ocr_ai_provider=ocr_ai_provider,
        ocr_ai_api_key=ocr_ai_key,
        ocr_ai_base_url=ocr_ai_base_url,
        ocr_ai_model=ocr_ai_model,
        ocr_geometry_mode=ocr_geometry_mode,
        ocr_ai_linebreak_assist=linebreak_assist,
        ocr_strict_mode=ocr_strict_mode,
    )

    return JobOutputs(
        job_id=job_id,
        output_pptx=job_dir / "output.pptx",
        ir_ocr=job_dir / "ir.ocr.json",
        ocr_debug=job_dir / "artifacts" / "ocr" / "ocr_debug.json",
        final_png=job_dir
        / "artifacts"
        / "final_preview"
        / f"page-{page_no - 1:04d}.final.png",
    )


def _render_source_page(
    pdf_path: Path, *, page_no: int, dpi: int, out_path: Path
) -> Path:
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(dpi=int(dpi), alpha=False)  # type: ignore[attr-defined]
        pix.save(str(out_path))
    return out_path


def _read_color_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img


def _align_image_like(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)


def _image_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mae = float(np.mean(diff)) / 255.0
    rmse = (
        float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))
        / 255.0
    )
    high_diff_ratio = float(np.mean((gray > 32).astype(np.float32)))
    return {
        "mae": mae,
        "rmse": rmse,
        "high_diff_ratio": high_diff_ratio,
    }


def _load_page_text_elements(ir_ocr_path: Path) -> list[dict[str, Any]]:
    data = json.loads(ir_ocr_path.read_text(encoding="utf-8"))
    pages = data.get("pages") or []
    if not pages:
        return []
    page0 = pages[0] if isinstance(pages[0], dict) else {}
    out: list[dict[str, Any]] = []
    for el in page0.get("elements") or []:
        if isinstance(el, dict) and el.get("type") == "text":
            out.append(el)
    return out


def _load_pdf_page_size_pt(pdf_path: Path, *, page_no: int) -> tuple[float, float]:
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc.load_page(page_no - 1)
        return (float(page.rect.width), float(page.rect.height))


def _build_text_region_mask(
    *,
    shape_hw: tuple[int, int],
    text_elements: Iterable[dict[str, Any]],
    page_w_pt: float,
    page_h_pt: float,
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if page_w_pt <= 0 or page_h_pt <= 0:
        return mask

    for el in text_elements:
        bbox = el.get("bbox_pt") if isinstance(el, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except Exception:
            continue

        X0 = max(0, min(w - 1, int(round(x0 / page_w_pt * w))))
        X1 = max(0, min(w - 1, int(round(x1 / page_w_pt * w))))
        Y0 = max(0, min(h - 1, int(round(y0 / page_h_pt * h))))
        Y1 = max(0, min(h - 1, int(round(y1 / page_h_pt * h))))
        if X1 > X0 and Y1 > Y0:
            cv2.rectangle(mask, (X0, Y0), (X1, Y1), (255,), thickness=-1)  # type: ignore[call-overload]

    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def _masked_mae(a: np.ndarray, b: np.ndarray, selector: np.ndarray) -> float:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    values = diff[selector]
    if values.size <= 0:
        return 0.0
    return float(values.mean()) / 255.0


def _save_side_by_side(
    *,
    left: np.ndarray,
    right: np.ndarray,
    left_label: str,
    right_label: str,
    out_path: Path,
) -> None:
    h, w = left.shape[:2]
    bar = np.full((48, w * 2, 3), 255, dtype=np.uint8)
    cv2.putText(
        bar,
        left_label,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        right_label,
        (w + 20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    canvas = np.hstack([left, right])
    cv2.imwrite(str(out_path), np.vstack([bar, canvas]))


def _save_heat_overlay(*, base: np.ndarray, other: np.ndarray, out_path: Path) -> None:
    diff = cv2.absdiff(base, other)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    mixed = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    cv2.imwrite(str(out_path), mixed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OCR AI linebreak assist")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--pages", default="1-10", help="Page spec, e.g. 1-10 or 1,3,5")
    parser.add_argument(
        "--dpi", type=int, default=200, help="Render DPI for comparison"
    )
    parser.add_argument(
        "--out-dir", default="../test/benchmark-linebreak", help="Output directory"
    )
    parser.add_argument("--jobs-root", default="data/jobs", help="Jobs directory")

    parser.add_argument("--ocr-provider", default="paddle")
    parser.add_argument("--ocr-ai-provider", default="siliconflow")
    parser.add_argument("--ocr-ai-base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--ocr-ai-model", default="PaddlePaddle/PaddleOCR-VL-1.5")
    parser.add_argument(
        "--ocr-geometry-mode",
        default="auto",
        choices=["auto", "local_tesseract", "direct_ai"],
        help="Geometry mode used when ocr-provider=aiocr",
    )
    parser.add_argument("--ocr-ai-key", default=os.getenv("SILICONFLOW_API_KEY", ""))
    parser.add_argument(
        "--ocr-strict-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict OCR mode (disable implicit fallbacks/downgrades)",
    )

    parser.add_argument(
        "--keep-job-artifacts", action="store_true", help="Keep full job dirs"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"ERROR: pdf not found: {pdf_path}", file=sys.stderr)
        return 2

    if not args.ocr_ai_key:
        print(
            "ERROR: --ocr-ai-key missing and SILICONFLOW_API_KEY not set",
            file=sys.stderr,
        )
        return 2

    out_dir = _ensure_dir(Path(args.out_dir).resolve())
    jobs_root = _ensure_dir(
        (Path(__file__).resolve().parents[1] / args.jobs_root).resolve()
    )

    with pymupdf.open(str(pdf_path)) as doc:
        page_numbers = _parse_pages_spec(str(args.pages), max_pages=int(doc.page_count))

    if not page_numbers:
        print("ERROR: no valid pages selected", file=sys.stderr)
        return 2

    report_pages: list[dict[str, Any]] = []

    for page_no in page_numbers:
        print(f"[bench] page {page_no}: running off/on ...", flush=True)

        off = _run_single_page_job(
            page_no=page_no,
            linebreak_assist=False,
            pdf_path=pdf_path,
            jobs_root=jobs_root,
            ocr_provider=args.ocr_provider,
            ocr_ai_provider=args.ocr_ai_provider,
            ocr_ai_key=args.ocr_ai_key,
            ocr_ai_base_url=args.ocr_ai_base_url,
            ocr_ai_model=args.ocr_ai_model,
            ocr_geometry_mode=str(args.ocr_geometry_mode),
            ocr_strict_mode=bool(args.ocr_strict_mode),
        )
        on = _run_single_page_job(
            page_no=page_no,
            linebreak_assist=True,
            pdf_path=pdf_path,
            jobs_root=jobs_root,
            ocr_provider=args.ocr_provider,
            ocr_ai_provider=args.ocr_ai_provider,
            ocr_ai_key=args.ocr_ai_key,
            ocr_ai_base_url=args.ocr_ai_base_url,
            ocr_ai_model=args.ocr_ai_model,
            ocr_geometry_mode=str(args.ocr_geometry_mode),
            ocr_strict_mode=bool(args.ocr_strict_mode),
        )

        page_dir = _ensure_dir(out_dir / f"page-{page_no:03d}")

        source_path = _render_source_page(
            pdf_path,
            page_no=page_no,
            dpi=int(args.dpi),
            out_path=page_dir / "source.png",
        )

        for src, name in [
            (off.final_png, "off.final.png"),
            (on.final_png, "on.final.png"),
            (off.ir_ocr, "off.ir.ocr.json"),
            (on.ir_ocr, "on.ir.ocr.json"),
            (off.ocr_debug, "off.ocr_debug.json"),
            (on.ocr_debug, "on.ocr_debug.json"),
            (off.output_pptx, "off.pptx"),
            (on.output_pptx, "on.pptx"),
        ]:
            if src.exists():
                shutil.copy2(src, page_dir / name)

        src_img = _read_color_image(source_path)
        off_img = _align_image_like(src_img, _read_color_image(off.final_png))
        on_img = _align_image_like(src_img, _read_color_image(on.final_png))

        source_vs_off = _image_metrics(src_img, off_img)
        source_vs_on = _image_metrics(src_img, on_img)
        off_vs_on = _image_metrics(off_img, on_img)

        page_w_pt, page_h_pt = _load_pdf_page_size_pt(pdf_path, page_no=page_no)
        on_text_elements = _load_page_text_elements(on.ir_ocr)
        mask = _build_text_region_mask(
            shape_hw=(src_img.shape[0], src_img.shape[1]),
            text_elements=on_text_elements,
            page_w_pt=page_w_pt,
            page_h_pt=page_h_pt,
        )

        text_selector = mask > 0
        non_text_selector = mask == 0

        region = {
            "off_text_mae": _masked_mae(src_img, off_img, text_selector),
            "on_text_mae": _masked_mae(src_img, on_img, text_selector),
            "off_non_text_mae": _masked_mae(src_img, off_img, non_text_selector),
            "on_non_text_mae": _masked_mae(src_img, on_img, non_text_selector),
            "text_region_ratio": float(np.mean(text_selector.astype(np.float32))),
        }

        off_text_count = len(_load_page_text_elements(off.ir_ocr))
        on_text_count = len(on_text_elements)

        _save_side_by_side(
            left=src_img,
            right=off_img,
            left_label=f"SOURCE PAGE{page_no}",
            right_label="OFF FINAL",
            out_path=page_dir / "compare-source-vs-off.png",
        )
        _save_side_by_side(
            left=src_img,
            right=on_img,
            left_label=f"SOURCE PAGE{page_no}",
            right_label="ON FINAL",
            out_path=page_dir / "compare-source-vs-on.png",
        )
        _save_side_by_side(
            left=off_img,
            right=on_img,
            left_label="OFF FINAL",
            right_label="ON FINAL",
            out_path=page_dir / "compare-off-vs-on.png",
        )

        _save_heat_overlay(
            base=src_img, other=off_img, out_path=page_dir / "heat-source-vs-off.png"
        )
        _save_heat_overlay(
            base=src_img, other=on_img, out_path=page_dir / "heat-source-vs-on.png"
        )

        page_report = {
            "page_no": page_no,
            "jobs": {
                "off": off.job_id,
                "on": on.job_id,
            },
            "metrics": {
                "source_vs_off": source_vs_off,
                "source_vs_on": source_vs_on,
                "off_vs_on": off_vs_on,
                "region": region,
            },
            "text_elements": {
                "off": off_text_count,
                "on": on_text_count,
                "delta": on_text_count - off_text_count,
            },
            "improvement": {
                "mae_drop": float(source_vs_off["mae"] - source_vs_on["mae"]),
                "rmse_drop": float(source_vs_off["rmse"] - source_vs_on["rmse"]),
                "text_mae_drop": float(region["off_text_mae"] - region["on_text_mae"]),
                "non_text_mae_drop": float(
                    region["off_non_text_mae"] - region["on_non_text_mae"]
                ),
            },
            "artifacts": {
                "dir": str(page_dir),
                "source": str(page_dir / "source.png"),
                "off_final": str(page_dir / "off.final.png"),
                "on_final": str(page_dir / "on.final.png"),
            },
        }
        report_pages.append(page_report)

        if not args.keep_job_artifacts:
            for job_id in [off.job_id, on.job_id]:
                job_dir = jobs_root / job_id
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)

    improved_pages = [p for p in report_pages if p["improvement"]["mae_drop"] > 0]
    text_improved_pages = [
        p for p in report_pages if p["improvement"]["text_mae_drop"] > 0
    ]

    overall = {
        "pages_total": len(report_pages),
        "pages_improved_by_mae": len(improved_pages),
        "pages_improved_text_mae": len(text_improved_pages),
        "avg_mae_drop": float(
            np.mean([p["improvement"]["mae_drop"] for p in report_pages])
        ),
        "avg_rmse_drop": float(
            np.mean([p["improvement"]["rmse_drop"] for p in report_pages])
        ),
        "avg_text_mae_drop": float(
            np.mean([p["improvement"]["text_mae_drop"] for p in report_pages])
        ),
        "avg_non_text_mae_drop": float(
            np.mean([p["improvement"]["non_text_mae_drop"] for p in report_pages])
        ),
    }

    ranking = sorted(
        [
            {
                "page_no": p["page_no"],
                "mae_drop": p["improvement"]["mae_drop"],
                "text_mae_drop": p["improvement"]["text_mae_drop"],
                "text_delta": p["text_elements"]["delta"],
            }
            for p in report_pages
        ],
        key=lambda x: (x["mae_drop"], x["text_mae_drop"], x["text_delta"]),
        reverse=True,
    )

    report = {
        "input": {
            "pdf": str(pdf_path),
            "pages": page_numbers,
            "dpi": int(args.dpi),
            "ocr_provider": args.ocr_provider,
            "ocr_ai_provider": args.ocr_ai_provider,
            "ocr_ai_base_url": args.ocr_ai_base_url,
            "ocr_ai_model": args.ocr_ai_model,
            "ocr_geometry_mode": args.ocr_geometry_mode,
            "ocr_strict_mode": bool(args.ocr_strict_mode),
        },
        "overall": overall,
        "ranking": ranking,
        "pages": report_pages,
    }

    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {"report": str(report_path), "overall": overall, "top3": ranking[:3]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
