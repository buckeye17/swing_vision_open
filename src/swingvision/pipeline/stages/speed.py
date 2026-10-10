"""M7c stage ``speed_refs``: near-end serves that hit the net tape, measured as speed
references (PLAN.md §6 #15c, §7.12). The device's calibration is fitted from the accepted
ones across sessions in the library (``services.calibrate_device``) and applied in ``shots``.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from swingvision.ball import physics as ph
from swingvision.ball import speed_refs as sr
from swingvision.ball.flights import detected_points
from swingvision.court import calibration as calib
from swingvision.pipeline.stage import Stage, StageContext
from swingvision.pipeline.stages.serve import serve_rows
from swingvision.storage import tables
from swingvision.storage.fsutil import atomic_write_json, read_json
from swingvision.storage.schemas import SPEED_REFS


def serve_inputs(session, cal) -> list[sr.ServeInput]:
    """Every near-end serve with a fitted flight, ready to measure."""
    if not session.ball_flights_path.exists() or not session.ball_track_path.exists():
        return []
    flights = {
        f["start_event_id"]: f
        for f in tables.read_table(session.ball_flights_path).to_pylist()
        if f["start_kind"] == "hit" and f["start_event_id"] is not None
    }
    shots = (
        tables.read_table(session.shots_path, columns=["shot_id", "swing_id"]).to_pylist()
        if session.shots_path.exists()
        else []
    )
    shot_of = {s["swing_id"]: s["shot_id"] for s in shots if s["swing_id"] is not None}
    summary = read_json(session.swings_summary_path) if session.swings_summary_path.exists() else {}
    tosses = summary.get("tosses") or {}
    pts = detected_points(tables.read_table(session.ball_track_path))
    det_px = np.column_stack([pts["x"], pts["y"]])
    out = []
    for r in serve_rows(session):
        if r["side"] != -1 or r["hit_event_id"] is None:
            continue
        f = flights.get(r["hit_event_id"])
        if f is None:
            continue
        m = (pts["t"] > f["t0_s"]) & (pts["t"] <= f["t1_s"] + 1e-6)
        toss = tosses.get(str(r["swing_id"])) or {}

        def arr(k, toss=toss):
            return None if toss.get(k) is None else np.asarray(toss[k], float)

        out.append(
            sr.ServeInput(
                swing_id=r["swing_id"],
                shot_id=shot_of.get(r["swing_id"]),
                t_contact=float(r["t_contact"]),
                camera=calib.camera_at(cal, float(r["t_contact"])),
                flight=f,
                det_t=pts["t"][m],
                det_px=det_px[m],
                contact=arr("pos"),
                contact_cov=arr("cov"),
                toss_theta=arr("theta"),
                toss_t0=toss.get("t0"),
            )
        )
    return out


def build_speed_refs(session, config, edits=None, params: sr.RefParams | None = None):
    """(``SPEED_REFS`` table, summary) for one session."""
    from swingvision.storage import edits as ed

    params = params or sr.RefParams()
    edits = edits if edits is not None else ed.load(session)
    cal = calib.load(session.calibration_path)
    if cal is None:
        raise RuntimeError("No calibration")
    video = config.video
    serves = serve_inputs(session, cal)
    audio = sr.AudioClip.from_file(session.audio_path) if session.audio_path.exists() else None
    container = ((video.audio_start_time_s or 0.0) - video.start_time_s) if video else 0.0
    fps = (video.fps_avg or 60.0) if video else 60.0
    rotation = video.rotation_cw if video else 0
    temp = config.air_temp_c
    av, n_av = (0.0, 0)
    if audio is not None:
        av, n_av = sr.estimate_av_offset(serves, audio, container, temp, params)
    rows = []
    flags_count: dict[str, int] = {}
    measured = 0
    for s in serves:
        e = ed.speed_ref_edit(edits, s.t_contact)
        ref = sr.measure(
            s, audio, container_offset_s=container, av_offset_s=av, temp_c=temp, fps=fps,
            rotation_cw=rotation, p=params,
            t_racket=e.t_racket if e is not None else None,
            t_tape=e.t_tape if e is not None else None,
        )  # fmt: skip
        measured += ref.ratio is not None
        for fl in ref.flags:
            flags_count[fl] = flags_count.get(fl, 0) + 1
        user = e is not None and (e.marked or e.status is not None)
        if not ref.is_candidate and not user:
            continue
        rows.append(_row(ref, e, len(rows)))
    table = pa.table(
        {f.name: pa.array([r.get(f.name) for r in rows], f.type) for f in SPEED_REFS},
        schema=SPEED_REFS,
    )
    status = [r["status"] for r in rows]
    summary = {
        "near_serves": len(serves),
        "measured": measured,
        "references": len(rows),
        "candidates": status.count("candidate"),
        "accepted": status.count("accepted"),
        "rejected": status.count("rejected"),
        "container_offset_s": container,
        "av_offset_s": round(av, 5),
        "av_offset_serves": n_av,
        "sound_speed_mps": round(sr.sound_speed(temp), 2),
        "air_temp_c": temp,
        "flags": dict(sorted(flags_count.items())),
    }
    return table, summary


def _row(ref: sr.Reference, e, ref_id: int) -> dict:
    c = ref.contact if ref.contact is not None else [None] * 3
    t = ref.tape if ref.tape is not None else [None, None, None]
    return {
        "ref_id": ref_id,
        "shot_id": ref.shot_id,
        "swing_id": ref.swing_id,
        "flight_id": ref.flight_id,
        "t_contact": ref.t_contact,
        "t_cross": ref.t_cross,
        "t_racket_audio": ref.t_racket_audio,
        "t_tape_audio": ref.t_tape_audio,
        "onsets_by": ref.onsets_by,
        "t_racket_pred_audio": ref.t_racket_pred_audio,
        "t_tape_pred_audio": ref.t_tape_pred_audio,
        "snr_racket_db": ref.snr_racket_db,
        "snr_tape_db": ref.snr_tape_db,
        "contact_x": c[0],
        "contact_y": c[1],
        "contact_z": c[2],
        "tape_x": t[0],
        "tape_z": t[2],
        "d_contact_m": ref.d_contact_m,
        "d_tape_m": ref.d_tape_m,
        "sound_speed_mps": ref.sound_speed_mps,
        "dt_s": ref.dt_s,
        "dt_sigma_s": ref.dt_sigma_s,
        "path_m": ref.path_m,
        "path_sigma_m": ref.path_sigma_m,
        "v_ref_kmh": ref.v_ref_kmh,
        "v_fit_kmh": ref.v_fit_kmh,
        "ratio": ref.ratio,
        "ratio_sigma": ref.ratio_sigma,
        "img_vy": ref.img_vy,
        "rs_rate": ref.rs_rate,
        "clearance_m": ref.clearance_m,
        "end_kind": ref.end_kind,
        "av_offset_s": ref.av_offset_s,
        "status": (e.status if e is not None and e.status else "candidate"),
        "source": "auto" if ref.is_candidate else "user",
        "flags": list(dict.fromkeys(ref.flags)),
    }


class SpeedRefsStage(Stage):
    """Net-tape reference serves: sound-timed flight time, contact → tape distance,
    reference vs fitted speed (§6 #15c)."""

    name = "speed_refs"
    title = "Speed references"
    version = 1
    depends_on = ("ball_3d", "swings", "audio_onsets", "camera")
    weight = 0.1

    def config(self, session, config, settings):
        from swingvision.storage import edits as ed

        return {
            "params": sr.RefParams().as_config(),
            "fit": ph.FitParams().as_config(),
            "air_temp_c": config.air_temp_c,
            "edits": [e.model_dump() for e in ed.load(session).speed_refs],
        }

    def outputs(self, session):
        return [session.speed_refs_path, session.speed_refs_summary_path]

    def run(self, ctx: StageContext):
        table, summary = build_speed_refs(ctx.session, ctx.config)
        ctx.session.ball_dir.mkdir(parents=True, exist_ok=True)
        tables.write_table(table, ctx.session.speed_refs_path, SPEED_REFS)
        atomic_write_json(ctx.session.speed_refs_summary_path, summary)
        return {k: summary[k] for k in ("references", "candidates", "accepted", "av_offset_s")}


def speed_calibration_for(settings, config) -> dict | None:
    """The speed calibration a session's shots use: one picked by hand
    (``config.speed_calibration``), none (``"none"``), or its device's active one."""
    if config.speed_calibration == "none" or settings.output_root is None:
        return None
    from swingvision.io.probe import device_key
    from swingvision.storage.library import Library

    lib = Library(settings.output_root).init()
    if config.speed_calibration:
        row = lib.get_calibration(config.speed_calibration)
    else:
        key = config.device_key or device_key(config.video)
        row = lib.active_calibration(key) if key else None
    if row is None:
        return None
    from swingvision.analysis.shots import SPEED_SCALE_ERROR

    if not row.get("k") or 2 * row["k_sigma"] / row["k"] >= SPEED_SCALE_ERROR:
        return None  # no tighter than uncalibrated speeds: not worth a correction
    return {
        k: row[k]
        for k in ("id", "device_key", "version", "model", "k", "k_sigma", "tau_s", "n_refs")
    }
