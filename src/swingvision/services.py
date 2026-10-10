"""Operations shared by the CLI and the web app."""

from __future__ import annotations

import json
import re
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path

from swingvision.io.probe import device_key, probe_video, source_info
from swingvision.settings import AppSettings
from swingvision.storage.library import Library, now_iso, recording_time
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
    return Library(settings.require_output_root()).init().keep_open()


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
    air_temp_c: float | None = None,
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
        device_key=device_key(video),
        air_temp_c=air_temp_c,
    )
    session.save_config(config)
    if config.device_key:
        library.add_device(config.device_key, video)
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
        recorded_on=recording_time(video.creation_time, config.created_at),
        profile_id=me_profile_id,
    )
    library.update_session(session_id, device_key=config.device_key, air_temp_c=air_temp_c)
    return session


# ---------------------------------------------------------------------------
# Source video: missing files and relinking (M7)
# ---------------------------------------------------------------------------


class RelinkError(ValueError):
    pass


def source_missing(row: dict) -> bool:
    """Whether a library session's source video is no longer where it was (cheap: a stat)."""
    return not Path(row["source_path"]).is_file()


def relink_session(settings: AppSettings, session_id: str, new_path: Path) -> list[str]:
    """Point a session at its source video's new location.

    The file must be the same video (same size and fast hash as when the session was
    created). Other sessions whose video is missing too are looked for next to it (moved
    together), by size and then hash. Returns the ids of every session relinked.
    """
    library = open_library(settings)
    row = library.get_session(session_id)
    if row is None:
        raise RelinkError(f"Unknown session {session_id}")
    new_path = Path(new_path)
    if not new_path.is_file():
        raise RelinkError(f"{new_path} is not a file")
    source = source_info(new_path)
    if source.fast_hash != row["source_hash"]:
        raise RelinkError(
            f"{new_path.name} is not the video this session was made from (different size or "
            "content). Pick the original file; a re-encoded or trimmed copy needs a new session."
        )
    _relink(library, row, source)
    relinked = [session_id]
    for other in library.list_sessions():
        if other["id"] == session_id or not source_missing(other):
            continue
        found = _find_by_hash(library, new_path.parent, other)
        if found is not None:
            _relink(library, other, found)
            relinked.append(other["id"])
    for sid in relinked:
        # A job that stopped because the video was missing picks up where it stopped.
        job = library.latest_job_for_session(sid)
        if job is not None and job.status == "needs_action" and job.action == "relink":
            library.enqueue_job(sid, job.targets, job.force)
    return relinked


def _relink(library: Library, row: dict, source) -> None:
    session = Session.open(library.root, row["dir_name"])
    config = session.load_config()
    config.source = source
    session.save_config(config)
    library.update_session(row["id"], source_path=source.path)


def _find_by_hash(library: Library, folder: Path, row: dict):
    """The file in ``folder`` with ``row``'s video content (size first: hashing is slower)."""
    from swingvision.io.probe import VIDEO_EXTENSIONS

    session = Session.open(library.root, row["dir_name"])
    size = session.load_config().source.size_bytes if session.config_path.exists() else None
    for f in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if size is not None and f.stat().st_size != size:
            continue
        info = source_info(f)
        if info.fast_hash == row["source_hash"]:
            return info
    return None


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
    shoe_length_m: float | None = None,
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
        shoe_length_m=shoe_length_m,
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
            library.update_session(row["id"], profile_id=config.players.me_profile_id)
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
    open_library(settings).update_session(session_id, profile_id=profile_id or None)


# ---------------------------------------------------------------------------
# Practice (M5)
# ---------------------------------------------------------------------------

#: Stages cheap enough for the app to run itself after an edit (seconds, CPU).
CHEAP_STAGES = frozenset(
    {"swings", "shots", "serve_contact", "speed_refs", "segments", "practice_eval", "stats"}
)


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


def refresh_practice(
    settings: AppSettings, session_id: str, target: str | None = None
) -> tuple[str, int | None]:
    """Bring ``segments``, ``practice_eval`` and ``stats`` (and the swings and shots they
    read) up to date after an edit.

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
    target = target or ("stats" if config.mode == "practice" else "shots")
    planned = plan(registry, session, config, settings, [target])
    stale = [p.stage for p in planned if not p.fresh]
    if not stale:
        return "ran", None

    def cheap(stage) -> bool:
        return stage.name in CHEAP_STAGES or stage.light(session, session.load_config(), settings)

    if all(cheap(st) for st in stale):
        result = run(registry, session, settings, targets=[target], allow=cheap)
        if result.status == "done":
            return "ran", None
        if result.status != "blocked":
            raise RuntimeError(result.message or "Updating the session failed")
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


def edit_serve(
    settings: AppSettings,
    session_id: str,
    t: float,
    *,
    frame: int | bool | None = False,
    toe: list[float] | bool | None = False,
    expected_version: int | None = None,
) -> SessionEdits:
    """Correct the contact frame and / or the toe tip of the serve at ``t`` (M7b; see
    ``storage.edits.set_serve``). The corrections are also ground truth for ``sv serves
    eval``."""
    from swingvision.storage import edits

    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    return edits.update(
        session, lambda e: edits.set_serve(e, t, frame=frame, toe=toe), expected_version
    )


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


# ---------------------------------------------------------------------------
# Session tags and saved Stats views (M7a)
# ---------------------------------------------------------------------------


def set_session_tags(settings: AppSettings, session_id: str, tags: list[str]) -> list[str]:
    library = open_library(settings)
    if library.get_session(session_id) is None:
        raise ValueError(f"Unknown session {session_id}")
    return library.set_session_tags(session_id, tags)


def save_view(settings: AppSettings, name: str, query: str) -> str:
    """Save a Stats view (its URL query) under a name; one with the same name is replaced."""
    name = (name or "").strip()
    if not name:
        raise ValueError("A view needs a name.")
    library = open_library(settings)
    same = [r for r in library.list_views() if r["name"].lower() == name.lower()]
    view_id = same[0]["id"] if same else secrets.token_hex(4)
    library.save_view(view_id, name, query.lstrip("?"))
    return view_id


# ---------------------------------------------------------------------------
# Devices and speed calibration (M7c)
# ---------------------------------------------------------------------------


def backfill_devices(settings: AppSettings) -> list[str]:
    """Give sessions probed before M7c their recording device: re-probe each source video
    that is still where it was (a second or less each) and store the device tags. Returns
    the ids of the sessions that got one."""
    library = open_library(settings)
    done = []
    for row in library.list_sessions():
        if row.get("device_key"):
            continue
        session = Session.open(library.root, row["dir_name"])
        if not session.config_path.exists():
            continue
        config = session.load_config()
        key = config.device_key or device_key(config.video)
        if key is None and not source_missing(row) and config.video is not None:
            try:
                probed = probe_video(settings.ffprobe(), Path(row["source_path"]))
            except Exception:  # a missing ffprobe or an unreadable file: try again later
                continue
            for f in ("device_make", "device_model", "device_lens"):
                setattr(config.video, f, getattr(probed, f))
            key = device_key(config.video)
        if key is None:
            continue
        config.device_key = key
        session.save_config(config)
        library.add_device(key, config.video)
        library.update_session(row["id"], device_key=key)
        done.append(row["id"])
    return done


def device_name(row: dict | None) -> str:
    """A ``devices`` row as people read it: the user's label, else "OnePlus Open · back_main
    · 3840×2160 60 fps"."""
    if not row:
        return "Unknown device"
    if row.get("label"):
        return row["label"]
    name = row.get("model") or row["device_key"]
    if row.get("make") and not name.lower().startswith(row["make"].lower()):
        name = f"{row['make']} {name}"
    bits = [name] + ([row["lens"]] if row.get("lens") else [])
    if row.get("width"):
        bits.append(f"{row['width']}×{row['height']} {round(row.get('fps') or 0)} fps")
    return " · ".join(bits)


def speed_status(settings: AppSettings, session: Session) -> tuple[dict | None, str | None]:
    """(the speed calibration the session's shots were made with, its device's name), for
    the badges."""
    from swingvision.analysis.stats import speed_calibration

    cal = speed_calibration(session)
    if cal is None:
        return None, None
    row = open_library(settings).get_device(cal.get("device_key") or "") or {}
    return cal, row.get("label") or row.get("model") or cal.get("device_key")


def set_air_temperature(settings: AppSettings, session_id: str, temp_c: float | None) -> None:
    """The air temperature during the recording (°C; ``None``: not known, 20 °C assumed)."""
    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    config = session.load_config()
    config.air_temp_c = None if temp_c is None else round(float(temp_c), 1)
    SessionConfig.model_validate(config.model_dump())  # range check
    session.save_config(config)
    open_library(settings).update_session(session_id, air_temp_c=config.air_temp_c)


def set_session_speed_calibration(
    settings: AppSettings, session_id: str, value: str | None
) -> None:
    """Which calibration the session's speeds use: ``None`` its device's active one,
    ``"none"`` none, else a calibration id."""
    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    if value not in (None, "none") and open_library(settings).get_calibration(value) is None:
        raise ValueError(f"Unknown calibration {value}")
    config = session.load_config()
    config.speed_calibration = value
    session.save_config(config)


def edit_speed_ref(
    settings: AppSettings,
    session_id: str,
    t: float,
    *,
    status: str | bool | None = False,
    marked: bool | None = None,
    t_racket: float | bool | None = False,
    t_tape: float | bool | None = False,
    expected_version: int | None = None,
) -> SessionEdits:
    """Review the reference serve at ``t`` (see ``storage.edits.set_speed_ref``); the
    session's reviews are also saved as ground truth (``training/speed_refs/``)."""
    from swingvision.storage import edits
    from swingvision.training import speed_labels

    session = session_by_id(settings, session_id)
    if session is None:
        raise ValueError(f"Unknown session {session_id}")
    out = edits.update(
        session,
        lambda e: edits.set_speed_ref(
            e, t, status=status, marked=marked, t_racket=t_racket, t_tape=t_tape
        ),
        expected_version,
    )
    speed_labels.save_from_edits(settings.require_output_root(), session.load_config(), out)
    return out


def device_references(settings: AppSettings, device: str) -> list[dict]:
    """Every accepted reference of the device's sessions, with its session id."""
    from swingvision.storage import tables

    library = open_library(settings)
    out = []
    for row in library.sessions_of_device(device):
        session = Session.open(library.root, row["dir_name"])
        if not session.speed_refs_path.exists():
            continue
        for r in tables.read_table(session.speed_refs_path).to_pylist():
            if r["status"] == "accepted" and r["ratio"] is not None:
                out.append({**r, "session_id": row["id"]})
    return out


#: What a calibration keeps of each reference (``speed_calibrations.refs_json``).
REF_KEYS = (
    "session_id", "ref_id", "t_contact", "ratio", "ratio_sigma", "v_ref_kmh", "v_fit_kmh",
    "rs_rate", "img_vy", "dt_s", "flags",
)  # fmt: skip


def calibrate_device(settings: AppSettings, device: str, save: bool = True):
    """Fit the device's speed calibration from its accepted references (§7.12) and, with
    ``save``, store it as the device's new active version. Returns (calibration or None
    when there are too few references, the stored row or None, the references used)."""
    from swingvision.ball import speed_refs as sr

    library = open_library(settings)
    if library.get_device(device) is None:
        raise ValueError(f"Unknown device {device}")
    refs = device_references(settings, device)
    cal = sr.fit_calibration(refs)
    if cal is None or not save:
        return cal, None, refs
    kept = [{k: r.get(k) for k in REF_KEYS} for r in refs]
    row = library.add_calibration(
        secrets.token_hex(4), device, kept, cal.diagnostics, model=cal.model, k=cal.k,
        k_sigma=cal.k_sigma, tau_s=cal.tau_s, tau_sigma_s=cal.tau_sigma_s, n_refs=cal.n_refs,
        loo_sd=cal.loo_sd,
    )  # fmt: skip
    return cal, row, refs


def set_active_calibration(settings: AppSettings, device: str, cal_id: str | None) -> None:
    open_library(settings).set_active_calibration(device, cal_id)


def rename_device(settings: AppSettings, device: str, label: str | None) -> None:
    open_library(settings).rename_device(device, (label or "").strip() or None)


def merge_devices(settings: AppSettings, src: str, dst: str) -> list[str]:
    """Treat ``src`` as the same phone and mode as ``dst``: its sessions move over (their
    ``session.json`` too) and ``src`` goes. Calibrate ``dst`` again to use their references."""
    library = open_library(settings)
    moved = library.merge_devices(src, dst)
    for sid in moved:
        session = session_by_id(settings, sid)
        if session is None:
            continue
        config = session.load_config()
        config.device_key = dst
        session.save_config(config)
    return moved


def refresh_device(settings: AppSettings, device: str) -> dict[str, list]:
    """Bring the device's sessions up to date after its calibration changed (``shots`` and
    what reads it): in-app where cheap, else queued. Returns session ids by outcome."""
    out: dict[str, list] = {"ran": [], "queued": [], "busy": []}
    for row in open_library(settings).sessions_of_device(device):
        try:
            how, _job = refresh_practice(settings, row["id"])
        except Exception:  # one broken session doesn't stop the others
            how = "queued"
            enqueue(settings, row["id"])
        out[how].append(row["id"])
    return out
