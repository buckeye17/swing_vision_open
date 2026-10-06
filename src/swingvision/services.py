"""Operations shared by the CLI and the web app."""

from __future__ import annotations

import json
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
    Profile,
    SessionConfig,
    SessionEdits,
    Target,
    TargetSet,
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
    me_profile_id: str | None = None,
    practice_targets: list[Target] | None = None,
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
        practice=(
            PracticeConfig(submode=practice_submode, targets=list(practice_targets or []))
            if mode == "practice"
            else None
        ),
        match=MatchConfig() if mode == "match" else None,
        players=PlayersConfig(me_profile_id=me_profile_id),
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


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def _profile(row: dict) -> Profile:
    return Profile.model_validate({k: row[k] for k in Profile.model_fields})


def list_profiles(settings: AppSettings) -> list[Profile]:
    return [_profile(r) for r in open_library(settings).list_profiles()]


def get_profile(settings: AppSettings, profile_id: str | None) -> Profile | None:
    if not profile_id:
        return None
    row = open_library(settings).get_profile(profile_id)
    return _profile(row) if row else None


def save_profile(
    settings: AppSettings,
    name: str,
    handedness: str = "right",
    backhand: str = "two_handed",
    height_m: float | None = None,
    profile_id: str | None = None,
) -> Profile:
    """Create a profile, or update ``profile_id``. Validates the values first."""
    name = (name or "").strip()
    if not name:
        raise ValueError("A profile needs a name.")
    ts = now_iso()
    candidate = Profile(
        id=profile_id or secrets.token_hex(4),
        name=name,
        handedness=handedness,  # type: ignore[arg-type]
        backhand=backhand,  # type: ignore[arg-type]
        height_m=height_m,
        created_at=ts,
        updated_at=ts,
    )
    library = open_library(settings)
    fields = candidate.model_dump(include=set(library.PROFILE_FIELDS))
    if profile_id:
        if library.get_profile(profile_id) is None:
            raise ValueError(f"Unknown profile {profile_id}")
        library.update_profile(profile_id, **fields)
    else:
        library.add_profile(id=candidate.id, **fields)
    return _profile(library.get_profile(candidate.id))


def delete_profile(settings: AppSettings, profile_id: str) -> int:
    """Delete a profile and unassign it from sessions. Returns how many sessions used it."""
    library = open_library(settings)
    cleared = 0
    for row in library.list_sessions():
        session = Session.open(library.root, row["dir_name"])
        if not session.config_path.exists():
            continue
        config = session.load_config()
        changed = False
        if config.players.me_profile_id == profile_id:
            config.players.me_profile_id = None
            changed = True
        if config.players.opponent_profile_id == profile_id:
            config.players.opponent_profile_id = None
            changed = True
        if changed:
            session.save_config(config)
            cleared += 1
    library.delete_profile(profile_id)
    return cleared


def set_session_player(settings: AppSettings, session_id: str, profile_id: str | None) -> None:
    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    if profile_id and open_library(settings).get_profile(profile_id) is None:
        raise ValueError(f"Unknown profile {profile_id}")
    config = session.load_config()
    config.players.me_profile_id = profile_id or None
    session.save_config(config)


# ---------------------------------------------------------------------------
# Practice (M5)
# ---------------------------------------------------------------------------

#: Stages cheap enough for the app to run itself after an edit (seconds, CPU).
CHEAP_STAGES = frozenset({"swings", "shots", "segments", "practice_eval"})


def set_practice(
    settings: AppSettings,
    session_id: str,
    *,
    submode: PracticeSubmode | None = None,
    targets: list[Target] | None = None,
) -> SessionConfig:
    """Change a practice session's type and/or targets (call :func:`refresh_practice` next)."""
    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    config = session.load_config()
    if config.practice is None:
        raise ValueError("Not a practice session")
    if submode is not None:
        config.practice.submode = submode
        open_library(settings).update_session(session_id, submode=submode)
    if targets is not None:
        config.practice.targets = list(targets)
    session.save_config(config)
    return config


def refresh_practice(settings: AppSettings, session_id: str) -> tuple[str, int | None]:
    """Bring ``segments`` and ``practice_eval`` (and the swings and shots they read) up to
    date after an edit.

    Runs them right here when nothing heavier is stale (``"ran"``); otherwise queues a job
    (``"queued"``, job id). ``"busy"``: the session is being processed, and the change is
    picked up when the job gets to these stages.
    """
    from swingvision.pipeline.runner import plan, run
    from swingvision.pipeline.stages import default_registry
    from swingvision.storage.library import ACTIVE_JOB_STATUSES

    library = open_library(settings)
    job = library.latest_job_for_session(session_id)
    if job is not None and job.status in ACTIVE_JOB_STATUSES:
        return "busy", job.id
    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    registry = default_registry()
    config = session.load_config()
    target = "practice_eval" if config.mode == "practice" else "shots"
    planned = plan(registry, session, config, settings, [target])
    stale = {p.stage.name for p in planned if not p.fresh}
    if not stale:
        return "ran", None
    if stale <= CHEAP_STAGES:
        result = run(registry, session, settings, targets=[target])
        if result.status != "done":
            raise RuntimeError(result.message or "Updating the session failed")
        return "ran", None
    return "queued", enqueue(settings, session_id, targets=[target])


def edit_practice_shot(
    settings: AppSettings,
    session_id: str,
    t: float,
    *,
    exclude: bool | None = None,
    landing: list[float] | bool | None = False,
    confirmed: bool | None = None,
    expected_version: int | None = None,
) -> SessionEdits:
    """Record a correction to the practice shot anchored at ``t`` (see ``storage.edits``)."""
    from swingvision.storage import edits

    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    return edits.update(
        session,
        lambda e: edits.set_practice_shot(
            e, t, exclude=exclude, landing=landing, confirmed=confirmed
        ),
        expected_version,
    )


def edit_swing_stroke(
    settings: AppSettings,
    session_id: str,
    t: float,
    stroke: str | None,
    expected_version: int | None = None,
) -> SessionEdits:
    """Set (``None``: clear) the user's stroke for the swing whose contact is at ``t``."""
    from swingvision.storage import edits

    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    return edits.update(session, lambda e: edits.set_swing_stroke(e, t, stroke), expected_version)


def _target_set(row: dict) -> TargetSet:
    return TargetSet(
        id=row["id"],
        name=row["name"],
        targets=[Target.model_validate(t) for t in json.loads(row["targets"] or "[]")],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_target_sets(settings: AppSettings) -> list[TargetSet]:
    return [_target_set(r) for r in open_library(settings).list_target_sets()]


def get_target_set(settings: AppSettings, set_id: str) -> TargetSet | None:
    row = open_library(settings).get_target_set(set_id)
    return _target_set(row) if row else None


def save_target_set(
    settings: AppSettings, name: str, targets: list[Target], set_id: str | None = None
) -> TargetSet:
    """Save targets under a name; a set with the same name (any case) is replaced."""
    name = (name or "").strip()
    if not name:
        raise ValueError("A target set needs a name.")
    library = open_library(settings)
    if set_id is None:
        same = [r for r in library.list_target_sets() if r["name"].lower() == name.lower()]
        set_id = same[0]["id"] if same else secrets.token_hex(4)
    payload = json.dumps([t.model_dump() for t in targets])
    library.save_target_set(set_id, name, payload)
    saved = get_target_set(settings, set_id)
    assert saved is not None
    return saved


def delete_target_set(settings: AppSettings, set_id: str) -> None:
    open_library(settings).delete_target_set(set_id)
