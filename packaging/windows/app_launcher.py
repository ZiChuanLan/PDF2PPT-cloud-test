#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Iterable, Sequence

WEB_URL = "http://127.0.0.1:3000"
API_HEALTH_URL = "http://127.0.0.1:8000/health"


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / "docker-compose.yml").exists()
        and (path / "api").exists()
        and (path / "web").exists()
    )


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _find_project_root() -> Path:
    candidates: list[Path] = []

    try:
        cwd = Path.cwd()
        candidates.extend([cwd, cwd.parent])
    except Exception:
        pass

    exe_path = Path(sys.executable).resolve()
    candidates.extend([exe_path.parent, exe_path.parent.parent])

    try:
        script_path = Path(__file__).resolve()
        candidates.extend([script_path.parent, script_path.parent.parent])
    except Exception:
        pass

    for candidate in _dedupe_paths(candidates):
        if _looks_like_project_root(candidate):
            return candidate

    raise RuntimeError(
        "未找到项目目录。请把 EXE 放在包含 docker-compose.yml / api / web 的目录中运行。"
    )


def _pick_compose_cmd() -> list[str]:
    probes = [["docker", "compose", "version"], ["docker-compose", "version"]]
    for probe in probes:
        try:
            result = subprocess.run(
                probe,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return probe[:2] if probe[0] == "docker" else [probe[0]]
        except FileNotFoundError:
            continue

    raise RuntimeError("未检测到 Docker Compose（docker compose 或 docker-compose）。")


def _ensure_docker_daemon() -> None:
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("未检测到 Docker 命令，请先安装 Docker Desktop。") from exc
    if result.returncode != 0:
        raise RuntimeError("Docker Desktop 未启动或不可用，请先启动后重试。")


def _compose_project_name(project_root: Path) -> str:
    raw = project_root.name.strip().lower() or "ppt-opencode"
    safe = "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-")
    safe = safe or "ppt-opencode"
    return safe[:42]


def _compose_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("COMPOSE_CONVERT_WINDOWS_PATHS", "1")
    env.setdefault("COMPOSE_PROJECT_NAME", _compose_project_name(project_root))
    return env


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        check=False,
        capture_output=capture,
    )


def _check_http(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            code = int(getattr(resp, "status", 0) or 0)
            return 200 <= code < 500
    except Exception:
        return False


def _wait_services(timeout_seconds: int) -> bool:
    started = time.time()
    while (time.time() - started) < timeout_seconds:
        api_ok = _check_http(API_HEALTH_URL)
        web_ok = _check_http(WEB_URL)
        if api_ok and web_ok:
            return True
        time.sleep(2)
    return False


def _print_header(title: str) -> None:
    print("=" * 66)
    print(title)
    print("=" * 66)


def cmd_start(
    *,
    project_root: Path,
    compose_cmd: list[str],
    env: dict[str, str],
    timeout: int,
    open_browser: bool,
    skip_build: bool,
) -> int:
    _print_header("PPT OpenCode 启动器")
    print(f"项目目录: {project_root}")
    print("正在启动服务（web/api/worker/redis）...")

    up_cmd = [*compose_cmd, "up", "-d", "--remove-orphans"]
    if not skip_build:
        up_cmd.append("--build")

    result = _run(up_cmd, cwd=project_root, env=env)
    if result.returncode != 0:
        print("启动失败。请检查 Docker Desktop 与网络。")
        return result.returncode or 1

    print("服务已提交启动，等待 API/Web 就绪...")
    ready = _wait_services(timeout)
    if not ready:
        print(f"等待超时（{timeout}s）。可执行 `logs` 查看详细日志。")
        return 2

    print(f"服务已就绪: {WEB_URL}")
    if open_browser:
        try:
            webbrowser.open(WEB_URL)
        except Exception:
            pass
    return 0


def cmd_stop(*, project_root: Path, compose_cmd: list[str], env: dict[str, str]) -> int:
    _print_header("停止服务")
    result = _run([*compose_cmd, "down"], cwd=project_root, env=env)
    if result.returncode == 0:
        print("已停止并移除容器。")
        return 0
    print("停止失败，请手动执行 docker compose down。")
    return result.returncode or 1


def cmd_restart(
    *,
    project_root: Path,
    compose_cmd: list[str],
    env: dict[str, str],
    timeout: int,
    open_browser: bool,
    skip_build: bool,
) -> int:
    stop_code = cmd_stop(project_root=project_root, compose_cmd=compose_cmd, env=env)
    if stop_code != 0:
        return stop_code
    return cmd_start(
        project_root=project_root,
        compose_cmd=compose_cmd,
        env=env,
        timeout=timeout,
        open_browser=open_browser,
        skip_build=skip_build,
    )


def cmd_status(
    *, project_root: Path, compose_cmd: list[str], env: dict[str, str]
) -> int:
    _print_header("服务状态")
    result = _run([*compose_cmd, "ps"], cwd=project_root, env=env)
    return result.returncode or 0


def cmd_logs(
    *,
    project_root: Path,
    compose_cmd: list[str],
    env: dict[str, str],
    lines: int,
    follow: bool,
) -> int:
    _print_header("服务日志")
    cmd = [*compose_cmd, "logs", "--tail", str(max(1, lines))]
    if follow:
        cmd.append("--follow")
    result = _run(cmd, cwd=project_root, env=env)
    return result.returncode or 0


def cmd_open() -> int:
    try:
        webbrowser.open(WEB_URL)
    except Exception:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PPT-OpenCode-Launcher",
        description="PPT OpenCode Windows 启动器",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=["start", "stop", "restart", "status", "logs", "open"],
        help="默认 start",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="启动等待秒数（默认 300）",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=120,
        help="logs 命令输出行数（默认 120）",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="logs 命令持续跟随输出",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start/restart 后不自动打开浏览器",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="start/restart 时跳过 --build（仅拉起现有镜像）",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        project_root = _find_project_root()
        compose_cmd = _pick_compose_cmd()
        _ensure_docker_daemon()
    except Exception as exc:
        print(f"[错误] {exc}")
        return 1

    env = _compose_env(project_root)

    if args.command == "start":
        return cmd_start(
            project_root=project_root,
            compose_cmd=compose_cmd,
            env=env,
            timeout=max(1, int(args.timeout)),
            open_browser=not args.no_browser,
            skip_build=args.skip_build,
        )
    if args.command == "stop":
        return cmd_stop(project_root=project_root, compose_cmd=compose_cmd, env=env)
    if args.command == "restart":
        return cmd_restart(
            project_root=project_root,
            compose_cmd=compose_cmd,
            env=env,
            timeout=max(1, int(args.timeout)),
            open_browser=not args.no_browser,
            skip_build=args.skip_build,
        )
    if args.command == "status":
        return cmd_status(project_root=project_root, compose_cmd=compose_cmd, env=env)
    if args.command == "logs":
        return cmd_logs(
            project_root=project_root,
            compose_cmd=compose_cmd,
            env=env,
            lines=max(1, int(args.lines)),
            follow=bool(args.follow),
        )
    if args.command == "open":
        return cmd_open()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
