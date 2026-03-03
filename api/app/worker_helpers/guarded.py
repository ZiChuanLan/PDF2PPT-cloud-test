from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

from ..logging_config import get_logger


logger = get_logger(__name__)

T = TypeVar("T")


def run_blocking_with_guards(
    target: Callable[[], T],
    *,
    cancel_check: Callable[[], None],
    operation_name: str,
    heartbeat: Callable[[], None] | None = None,
    heartbeat_interval_s: float = 15.0,
    poll_interval_s: float = 0.2,
) -> T:
    result_holder: dict[str, T] = {}
    error_holder: dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            result_holder["value"] = target()
        except BaseException as e:
            error_holder["error"] = e
        finally:
            done.set()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    heartbeat_interval = max(0.5, float(heartbeat_interval_s))
    poll_interval = max(0.05, float(poll_interval_s))

    started_at = time.monotonic()
    last_heartbeat_at = started_at

    while not done.wait(timeout=poll_interval):
        cancel_check()
        now = time.monotonic()
        if heartbeat is not None and (now - last_heartbeat_at) >= heartbeat_interval:
            try:
                heartbeat()
            except Exception as e:
                logger.warning(
                    "Heartbeat callback failed during %s: %s", operation_name, e
                )
            last_heartbeat_at = now

    cancel_check()

    error = error_holder.get("error")
    if error is not None:
        raise error

    return result_holder["value"]
