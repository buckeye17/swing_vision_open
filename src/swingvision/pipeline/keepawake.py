"""Keep the computer from sleeping while a job runs (an overnight job on a laptop with the
default power plan would otherwise stop when Windows goes to sleep after ~30 idle minutes).

The display may still turn off. Elsewhere this is a no-op (use ``systemd-inhibit`` on Linux).
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator

from loguru import logger

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake() -> Iterator[None]:
    if sys.platform != "win32":
        yield
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        logger.warning("Could not ask Windows to stay awake during the job")
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
