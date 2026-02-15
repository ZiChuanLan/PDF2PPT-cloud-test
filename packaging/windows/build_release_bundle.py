#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

BUNDLE_DIRNAME = "ppt-opencode-win-x64"
LAUNCHER_EXE_NAME = "PPT-OpenCode-Launcher.exe"

TOP_LEVEL_FILES = [
    "docker-compose.yml",
    ".env.example",
    "README.md",
]

TOP_LEVEL_DIRS = [
    "api",
    "web",
]

SKIP_DIR_NAMES = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    ".sisyphus",
    ".omc",
    "node_modules",
    ".next",
    ".turbo",
    "dist",
    "build",
    "release",
    "test",
    "data",
    "fixtures",
}

# Runtime does not need test folders and local virtualenvs.
SKIP_REL_PATH_PREFIXES = {
    "api/tests",
    "api/.venv",
    "api/data",
    "web/node_modules",
    "web/.next",
}

SKIP_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
}


def _norm_rel(rel: Path) -> str:
    return str(rel.as_posix()).strip("/")


def _should_skip_path(src_root: Path, candidate: Path) -> bool:
    rel = candidate.resolve().relative_to(src_root.resolve())
    rel_str = _norm_rel(rel)
    if not rel_str:
        return False

    parts = rel.parts
    if any(part in SKIP_DIR_NAMES for part in parts):
        return True

    for prefix in SKIP_REL_PATH_PREFIXES:
        if rel_str == prefix or rel_str.startswith(prefix + "/"):
            return True

    if candidate.is_file() and candidate.suffix.lower() in SKIP_FILE_SUFFIXES:
        return True

    return False


def _copy_tree_filtered(src: Path, dst: Path, src_root: Path) -> None:
    if _should_skip_path(src_root, src):
        return

    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _copy_tree_filtered(child, dst / child.name, src_root)
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_runtime_readme(bundle_root: Path) -> None:
    content = """# PPT OpenCode Windows Bundle

这是可直接分发到 GitHub Release 的 Windows 运行包。

## 使用方式

1. 解压 zip 到任意目录（建议英文路径）。
2. 确保已安装并启动 Docker Desktop。
3. 双击 `PPT-OpenCode-Launcher.exe`（默认执行 start）。
4. 首次启动会自动构建镜像，时间较长属正常。
5. 浏览器打开 `http://127.0.0.1:3000` 即可使用。

## 常用命令（在该目录打开终端）

```bat
PPT-OpenCode-Launcher.exe start
PPT-OpenCode-Launcher.exe stop
PPT-OpenCode-Launcher.exe restart
PPT-OpenCode-Launcher.exe status
PPT-OpenCode-Launcher.exe logs --lines 200
```

## 说明

- 这是“全功能运行包”（web + api + worker + redis），运行时由 Docker Compose 拉起。
- 首次启动会构建容器镜像，并下载依赖，需联网。
- 建议通过前端设置页填写 OCR API Key，不要把密钥写入仓库。
"""
    (bundle_root / "README-Windows.md").write_text(content, encoding="utf-8")


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(src_dir.parent)
            zf.write(path, arcname)


def build_bundle(*, repo_root: Path, launcher_path: Path, out_dir: Path) -> Path:
    if not launcher_path.exists():
        raise FileNotFoundError(f"launcher exe not found: {launcher_path}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel_file in TOP_LEVEL_FILES:
        src = repo_root / rel_file
        if not src.exists():
            raise FileNotFoundError(f"required file missing: {src}")
        shutil.copy2(src, out_dir / rel_file)

    for rel_dir in TOP_LEVEL_DIRS:
        src_dir = repo_root / rel_dir
        if not src_dir.exists():
            raise FileNotFoundError(f"required dir missing: {src_dir}")
        _copy_tree_filtered(src_dir, out_dir / rel_dir, repo_root)

    # Ensure clean runtime workspace exists.
    (out_dir / "api" / "data" / "jobs").mkdir(parents=True, exist_ok=True)
    (out_dir / "api" / "data" / "jobs" / ".gitkeep").write_text("", encoding="utf-8")

    shutil.copy2(launcher_path, out_dir / LAUNCHER_EXE_NAME)
    _write_runtime_readme(out_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Windows release bundle zip for GitHub distribution."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path",
    )
    parser.add_argument(
        "--launcher",
        default=f"release/windows/{LAUNCHER_EXE_NAME}",
        help="Path to launcher exe",
    )
    parser.add_argument(
        "--out-dir",
        default=f"release/windows/{BUNDLE_DIRNAME}",
        help="Output bundle directory",
    )
    parser.add_argument(
        "--zip-path",
        default=f"release/windows/{BUNDLE_DIRNAME}.zip",
        help="Output zip path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    launcher_path = Path(args.launcher).resolve()
    out_dir = Path(args.out_dir).resolve()
    zip_path = Path(args.zip_path).resolve()

    bundle_root = build_bundle(
        repo_root=repo_root,
        launcher_path=launcher_path,
        out_dir=out_dir,
    )
    _zip_dir(bundle_root, zip_path)

    print(f"bundle_dir={bundle_root}")
    print(f"bundle_zip={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
