"""Per-request access to settings and the library (settings are re-read so edits apply live).

Reading the settings also switches the local read cache (``storage.cache``) on or off to
match them: on for an output folder on a network share, when the setting allows.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from swingvision.settings import AppSettings, load_settings
from swingvision.storage import cache
from swingvision.storage.library import Library
from swingvision.storage.session import Session


@dataclass
class AppOptions:
    start_worker: bool = True


OPTIONS = AppOptions()


_cache_lock = threading.Lock()
_cache_key: tuple | None = None


def settings() -> AppSettings:
    s = load_settings()
    _sync_cache(s)
    return s


def _sync_cache(s: AppSettings) -> None:
    global _cache_key
    key = (s.output_root, s.local_cache, s.cache_path(), s.cache_max_gb)
    if key == _cache_key:
        return
    with _cache_lock:
        if key == _cache_key:
            return
        local = None
        if s.output_root is not None and s.local_cache and _is_network(s.output_root):
            # One folder per output root, so switching roots never mixes their files.
            tag = hashlib.sha1(str(s.output_root).lower().encode()).hexdigest()[:12]
            local = cache.LocalCache(
                s.output_root, s.cache_path() / tag, int(s.cache_max_gb * 1024**3)
            )
        cache.activate(local)
        _cache_key = key


@lru_cache(maxsize=8)
def _is_network(root: Path) -> bool:
    return cache.is_network_path(root)


def library() -> Library | None:
    s = settings()
    if s.output_root is None:
        return None
    return Library(s.output_root).init().keep_open()


def session_for(session_id: str) -> tuple[Library, dict, Session] | None:
    lib = library()
    if lib is None:
        return None
    row = lib.get_session(session_id)
    if row is None:
        return None
    return lib, row, Session.open(lib.root, row["dir_name"])
