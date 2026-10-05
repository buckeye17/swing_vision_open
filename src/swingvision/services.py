"""Operations shared by the CLI and the web app."""

from __future__ import annotations

import re
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path

from swingvision.io.probe import probe_video, source_info
from swingvision.settings import AppSettings
from swingvision.storage.library import Library, now_iso
from swingvision.storage.schemas import (
    MatchConfig,
    Mode,
    PlayersConfig,
    PracticeConfig,
    PracticeSubmode,
    SessionConfig,
)
from swingvision.storage.session import Session


def open_library(settings: AppSettings) -> Library:
    return Library(settings.require_output_root()).init()


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "session"


def _recorded_date(creation_time: str | None) -> str:
    if creation_time:
        try:
            return datetime.fromisoformat(creation_time.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    return datetime.now(UTC).date().isoformat()


def create_session(
    settings: AppSettings,
    source_path: Path,
    name: str | None = None,
    mode: Mode = "practice",
    practice_submode: PracticeSubmode = "self_feed",
) -> Session:
    library = open_library(settings)
    source = source_info(Path(source_path))
    video = probe_video(settings.ffprobe(), Path(source.path))
    session_id = secrets.token_hex(4)
    name = (name or Path(source.path).stem).strip()
    dir_name = f"{_recorded_date(video.creation_time)}_{slugify(name)}_{session_id}"
    session = Session.open(library.root, dir_name)
    session.path.mkdir(parents=True, exist_ok=False)

    config = SessionConfig(
        id=session_id,
        name=name,
        created_at=datetime.now(UTC),
        dir_name=dir_name,
        source=source,
        video=video,
        mode=mode,
        practice=PracticeConfig(submode=practice_submode) if mode == "practice" else None,
        match=MatchConfig() if mode == "match" else None,
        players=PlayersConfig(),
    )
    session.save_config(config)
    library.add_session(
        id=session_id,
        name=name,
        dir_name=dir_name,
        created_at=now_iso(),
        mode=mode,
        submode=practice_submode if mode == "practice" else None,
        source_path=source.path,
        source_hash=source.fast_hash,
        duration_s=video.duration_s,
    )
    return session


def session_by_id(settings: AppSettings, session_id: str) -> Session | None:
    library = open_library(settings)
    row = library.get_session(session_id)
    return Session.open(library.root, row["dir_name"]) if row else None


def enqueue(
    settings: AppSettings,
    session_id: str,
    targets: list[str] | None = None,
    force: list[str] | None = None,
) -> int:
    return open_library(settings).enqueue_job(session_id, targets, force)


def delete_session(settings: AppSettings, session_id: str) -> None:
    """Delete the session's *derived* data and library row. Never touches the source video."""
    library = open_library(settings)
    row = library.get_session(session_id)
    if row is None:
        return
    job = library.latest_job_for_session(session_id)
    if job and job.status in ("queued", "running"):
        raise ValueError("Cancel the session's active job before deleting it.")
    session_dir = (library.root / "sessions" / row["dir_name"]).resolve()
    if session_dir.parent != (library.root / "sessions").resolve():
        raise ValueError(f"Refusing to delete unexpected path {session_dir}")
    library.delete_session(session_id)
    shutil.rmtree(session_dir, ignore_errors=True)
