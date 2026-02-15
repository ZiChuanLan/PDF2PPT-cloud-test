"""Small concurrency helpers used by OCR and worker flows."""

from __future__ import annotations

import threading
from typing import Any


def run_in_daemon_thread_with_timeout(
    func: Any,
    *,
    timeout_s: float,
    label: str,
) -> Any:
    """Run `func()` in a daemon thread and raise `TimeoutError` on timeout.

    This cannot forcibly terminate blocking C extensions/network calls, but it
    keeps the main pipeline responsive and bounded.
    """

    done = threading.Event()
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = func()
        except BaseException as exc:  # noqa: BLE001
            error["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_runner, name=f"timeout:{label}", daemon=True)
    thread.start()

    effective_timeout = max(1.0, float(timeout_s))
    if not done.wait(timeout=effective_timeout):
        raise TimeoutError(f"{label} timed out after {effective_timeout:.0f}s")
    if "error" in error:
        raise error["error"]
    return result.get("value")
