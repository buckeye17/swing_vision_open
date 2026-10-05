"""Per-request access to settings and the library (settings are re-read so edits apply live)."""

from __future__ import annotations

from dataclasses import dataclass

from swingvision.settings import AppSettings, load_settings
from swingvision.storage.library import Library
from swingvision.storage.session import Session


@dataclass
class AppOptions:
    start_worker: bool = True


OPTIONS = AppOptions()


def settings() -> AppSettings:
    return load_settings()


def library() -> Library | None:
    s = settings()
    if s.output_root is None:
        return None
    return Library(s.output_root).init()


def session_for(session_id: str) -> tuple[Library, dict, Session] | None:
    lib = library()
    if lib is None:
        return None
    row = lib.get_session(session_id)
    if row is None:
        return None
    return lib, row, Session.open(lib.root, row["dir_name"])
