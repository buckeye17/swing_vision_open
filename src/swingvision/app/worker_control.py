"""Start/inspect the background worker process from the app."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from filelock import Timeout
from loguru import logger

from swingvision.pipeline.worker import worker_lock
from swingvision.storage.library import Library, parse_iso

HEARTBEAT_STALE_S = 10.0


def worker_running(output_root: Path) -> bool:
    """True if some process holds the worker lock for this output root."""
    lock = worker_lock(output_root)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    lock.release()
    return False


def ensure_worker(output_root: Path | None) -> bool:
    """Spawn a worker if none is running. Returns True if one was started."""
    if output_root is None or not Path(output_root).exists():
        return False
    if worker_running(output_root):
        return False
    log_dir = Path(output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / "worker.log").open("a", encoding="utf-8")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, "-m", "swingvision", "worker", "--parent-pid", str(os.getpid())],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )
    logger.info("Started background worker for {}", output_root)
    return True


def worker_state(library: Library) -> tuple[str, str]:
    """(state, label) where state is one of idle | busy | stopped."""
    row = library.worker_status()
    if not row:
        return "stopped", "Worker stopped"
    hb = parse_iso(row.get("heartbeat_at"))
    age = (datetime.now(UTC) - hb).total_seconds() if hb else 1e9
    if row.get("state") == "stopped" or age > HEARTBEAT_STALE_S:
        return "stopped", "Worker stopped"
    if row.get("state") == "busy":
        return "busy", "Worker busy"
    return "idle", "Worker idle"
