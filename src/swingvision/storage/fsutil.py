"""Small filesystem helpers: atomic writes that tolerate Windows file locking."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from swingvision.storage import cache


def atomic_replace(tmp: Path, dst: Path, retries: int = 20, delay_s: float = 0.05) -> None:
    """``os.replace`` with retries: on Windows a reader holding ``dst`` open blocks the rename."""
    for attempt in range(retries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s)


def atomic_write(path: Path, writer: Callable[[Path], None], suffix: str = "") -> None:
    """Call ``writer(tmp_path)`` then atomically move the result to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix or ".tmp", dir=path.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        atomic_replace(tmp, path)
        cache.forget(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def read_json(path: Path) -> Any:
    return json.loads(cache.local(path).read_text(encoding="utf-8"))
