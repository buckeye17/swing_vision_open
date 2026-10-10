"""M7b stages: ``serve_feet`` (foot keypoints around each serve's contact, GPU) and
``serve_contact`` (the contact point relative to the front toe in the contact frame; PLAN.md
§6 #15a-b, §7.11).

The contact frame itself comes from the toss path in ``swings`` (a serve's contact moves to
it); ``serve_contact`` reads that stage's toss contacts and adds the toe.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.ball.flights import detected_points
from swingvision.court import calibration as calib
from swingvision.io.frames import open_source
from swingvision.pipeline.stage import Stage, StageContext
from swingvision.pipeline.stages.ingest import _source_path
from swingvision.pipeline.stages.pose import box_track, me_movement
from swingvision.pose import feet as ft
from swingvision.pose import serve_contact as sc
from swingvision.storage import tables
from swingvision.storage.fsutil import atomic_write_json, read_json
from swingvision.storage.schemas import SERVE_CONTACT, SERVE_FEET

#: Foot keypoints from this long (s) before a serve's contact (the toss release is
#: ≈0.6-0.9 s before it) to this long after it, on a 0.1 s grid (a contact frame moved
#: by a frame or two seldom changes the windows, so ``serve_feet`` doesn't rerun).
FEET_BEFORE_S = 1.1
FEET_AFTER_S = 0.1


# ---------------------------------------------------------------------------
# Toss contacts (used by ``swings``)
# ---------------------------------------------------------------------------


def ball_points(session) -> sc.BallPoints:
    d = detected_points(tables.read_table(session.ball_track_path))
    return sc.BallPoints(d["frame"], d["t"], d["x"], d["y"])


def frame_clock(session) -> sc.FrameClock:
    """Frame times around the swings (every frame of the pose windows)."""
    p = tables.read_table(session.pose3d_path, columns=["frame", "t_s"])
    f, t = p.column("frame").to_numpy(), p.column("t_s").to_numpy()
    u, i = np.unique(f, return_index=True)
    return sc.FrameClock(u, t[i])


def toss_function(session, cal, params: sc.ContactParams | None = None) -> Callable:
    """``fn(t_guess, frame)`` → :func:`sc.toss_contact`, cached on disk by its inputs
    (``work/toss_contacts.json``): ``swings`` reruns after every stroke edit, and refitting
    every toss would take a minute."""
    params = params or sc.ContactParams()
    pts = ball_points(session)
    clock = frame_clock(session)
    path = session.toss_cache_path
    try:
        cache = read_json(path) if path.exists() else {}
    except (OSError, ValueError):
        cache = {}
    base = hashlib.sha1(json.dumps(params.as_config(), sort_keys=True).encode()).hexdigest()
    used: dict[str, dict] = {}

    def fn(t_guess: float, frame: int | None = None) -> sc.TossContact:
        cam = calib.camera_at(cal, t_guess)
        w = pts.window(t_guess - params.before_s, t_guess + params.after_s)
        h = hashlib.sha1(base.encode())
        for a in (w.frame, w.x, w.y, cam.params()):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        key = f"{t_guess:.4f}|{frame}|{h.hexdigest()[:16]}"
        if key in cache:
            used[key] = cache[key]
            return sc.toss_from_dict(cache[key])
        c = sc.toss_contact(pts, clock, cam, t_guess, params, frame_override=frame)
        used[key] = sc.toss_to_dict(c)
        return c

    def save() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, used)

    fn.save = save  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------------------
# serve_feet
# ---------------------------------------------------------------------------


def serve_rows(session) -> list[dict]:
    if not session.swings_path.exists():
        return []
    rows = tables.read_table(session.swings_path).to_pylist()
    return [r for r in rows if r["stroke_type"] == "serve" and r["t_contact"] is not None]


def feet_windows(session) -> list[list[float]]:
    """One window per serve, on a 0.1 s grid, merged where they overlap."""
    out: list[list[float]] = []
    for r in sorted(serve_rows(session), key=lambda r: r["t_contact"]):
        a = math.floor((r["t_contact"] - FEET_BEFORE_S) * 10) / 10
        b = math.ceil((r["t_contact"] + FEET_AFTER_S) * 10) / 10
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([round(a, 1), round(b, 1)])
    return out


#: A window counts as already done when the frames kept from the last run leave no gap
#: longer than this (s) in it.
REUSE_GAP_S = 0.05
#: Missing frames up to this long (s) in all make ``serve_feet`` light enough to run in the
#: app (a contact frame or two moved by the user).
LIGHT_S = 0.6


def feet_basis(session) -> dict:
    """What the stored keypoints were computed from: reusable only while this matches."""
    from swingvision.pipeline.stage import read_manifest

    mv = read_manifest(session, "movement") or {}
    return {
        "model": ft.DEFAULT_MODEL,
        "padding": ft.PADDING,
        "movement": mv.get("fingerprint"),
    }


def _stored_feet(session) -> pa.Table | None:
    """The last run's keypoints, if they were computed the way this run would."""
    from swingvision.storage.fsutil import read_json

    side = session.serve_feet_path.with_suffix(".json")
    if not session.serve_feet_path.exists() or not side.exists():
        return None
    try:
        if read_json(side).get("basis") != feet_basis(session):
            return None
    except (OSError, ValueError):
        return None
    return tables.read_table(session.serve_feet_path).sort_by("t_s")


def _gaps(t: np.ndarray, a: float, b: float) -> list[tuple[float, float]]:
    """The parts of [a, b) the stored frames (times ``t``, sorted) don't cover."""
    tt = t[(t >= a) & (t < b)]
    edges = np.r_[a, tt, b]
    out = []
    for lo, hi in itertools.pairwise(edges):
        if hi - lo > REUSE_GAP_S:
            # The frames at lo and hi are kept; decode strictly between them.
            out.append((float(lo) + (1e-4 if lo > a else 0.0), float(hi)))
    return out


def _missing_s(session, windows: list[list[float]]) -> float:
    old = _stored_feet(session)
    t = old.column("t_s").to_numpy() if old is not None else np.zeros(0)
    return float(sum(hi - lo for a, b in windows for lo, hi in _gaps(t, a, b)))


class ServeFeetStage(Stage):
    """Toe, heel and ankle keypoints around each serve's contact (PLAN.md §6 #15a)."""

    name = "serve_feet"
    title = "Feet at the serves"
    version = 1
    depends_on = ("movement",)
    # Ordering only: the windows come from the serves (config), so a stroke edit that
    # reruns ``swings`` doesn't rerun this GPU stage unless the serves changed.
    after = ("swings", "camera")
    uses_gpu = True
    weight = 0.5

    def config(self, session, config, settings):
        return {
            "model": ft.DEFAULT_MODEL,
            "padding": ft.PADDING,
            "windows": feet_windows(session),
        }

    def outputs(self, session):
        return [session.serve_feet_path]

    def light(self, session, config, settings):
        # No serves, or only a few frames to add to the last run's (a corrected contact).
        return _missing_s(session, feet_windows(session)) <= LIGHT_S

    def run(self, ctx: StageContext):
        from swingvision.storage.fsutil import atomic_write_json

        session, info = ctx.session, ctx.config.video
        assert info is not None, "ingest must run first"
        windows = feet_windows(session)
        boxes = box_track(me_movement(session))
        old = _stored_feet(session)
        old_t = old.column("t_s").to_numpy() if old is not None else np.zeros(0)
        rows: list[tuple] = []
        todo = []
        reused = 0
        for wi, (a, b) in enumerate(windows):
            if old is not None:
                keep = old.filter(
                    pc.and_(pc.greater_equal(old.column("t_s"), a), pc.less(old.column("t_s"), b))
                )
                for r in keep.to_pylist():
                    kp = np.asarray(r["kp"], np.float32).reshape(len(ft.FOOT), 3)
                    rect = np.array([r["x0"], r["y0"], r["x1"], r["y1"]], np.float32)
                    rows.append((r["frame"], r["t_s"], wi, kp, rect))
                reused += keep.num_rows
            todo += [(wi, lo, hi) for lo, hi in _gaps(old_t, a, b)]
        decoder = None
        if todo:
            ctx.progress(0.0, f"Loading {ft.DEFAULT_MODEL}")
            est = ft.FootPose(
                progress=lambda f: ctx.progress(0.0, f"Downloading {ft.DEFAULT_MODEL} {f:.0%}")
            )
            source = open_source(_source_path(ctx), info, ctx.settings.processing.decode_backend)
            decoder = source.backend
            with source:
                for n, (wi, a, b) in enumerate(todo):
                    ctx.check_cancel()
                    ctx.progress(n / len(todo), f"Serve {n + 1}/{len(todo)}")
                    pending: list[tuple] = []
                    for f in source.frames(a, b):
                        box = boxes.at(f.t_s)
                        if box is None:
                            continue
                        crop, rect = est.prepare(f.image, box)
                        pending.append((f.index, f.t_s, crop, rect))
                    if pending:
                        kp = est.run([x[2] for x in pending], np.array([x[3] for x in pending]))
                        rows += [
                            (fi, t, wi, k, r) for (fi, t, _, r), k in zip(pending, kp, strict=True)
                        ]
        session.pose_dir.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: r[1])
        tables.write_table(_feet_table(rows), session.serve_feet_path, SERVE_FEET)
        atomic_write_json(
            session.serve_feet_path.with_suffix(".json"), {"basis": feet_basis(session)}
        )
        return {
            "serves": len(windows),
            "frames": len(rows),
            "reused_frames": reused,
            "decoder": decoder,
        }


def _feet_table(rows: list[tuple]) -> pa.Table:
    if not rows:
        return SERVE_FEET.empty_table()
    kp = np.stack([r[3] for r in rows]).reshape(len(rows), -1).astype(np.float32)
    rect = np.stack([r[4] for r in rows]).astype(np.float32)
    return pa.table(
        {
            "frame": pa.array([r[0] for r in rows], pa.int64()),
            "t_s": pa.array([r[1] for r in rows], pa.float64()),
            "window": pa.array([r[2] for r in rows], pa.int32()),
            "kp": pa.FixedSizeListArray.from_arrays(pa.array(kp.ravel()), 3 * len(ft.FOOT)),
            "x0": rect[:, 0],
            "y0": rect[:, 1],
            "x1": rect[:, 2],
            "y1": rect[:, 3],
        },
        schema=SERVE_FEET,
    ).sort_by("t_s")


def load_feet(session) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``serve_feet.parquet`` → frames, times, keypoints (n, len(FOOT), 3)."""
    if not session.serve_feet_path.exists():
        return np.zeros(0, np.int64), np.zeros(0), np.zeros((0, len(ft.FOOT), 3))
    t = tables.read_table(session.serve_feet_path).sort_by("t_s")
    col = t.column("kp").combine_chunks()
    n = len(col)
    kp = (
        col.flatten().to_numpy(zero_copy_only=False).reshape(n, len(ft.FOOT), 3).astype(float)
        if n
        else np.zeros((0, len(ft.FOOT), 3))
    )
    return t.column("frame").to_numpy(), t.column("t_s").to_numpy(), kp


# ---------------------------------------------------------------------------
# serve_contact
# ---------------------------------------------------------------------------


def _profile(settings, config) -> dict:
    from swingvision.storage.library import Library

    pid = config.players.me_profile_id
    if not pid or settings.output_root is None:
        return {}
    row = Library(settings.output_root).get_profile(pid)
    if not row:
        return {}
    return {"height_m": row.get("height_m"), "shoe_length_m": row.get("shoe_length_m")}


def build_serve_contacts(
    serves: list[dict],
    tosses: dict[str, dict],
    feet: tuple[np.ndarray, np.ndarray, np.ndarray],
    cal,
    shots: list[dict],
    toe_edits: list[tuple[float, list[float]]] = (),
    profile: dict | None = None,
    params: sc.ToeParams | None = None,
) -> tuple[pa.Table, dict]:
    """One SERVE_CONTACT row per serve (PLAN.md §7.11).

    ``tosses``: the toss contacts from ``swings`` by swing id; ``feet``: ``serve_feet``;
    ``toe_edits``: (contact time, toe pixel) the user placed.
    """
    params = params or sc.ToeParams()
    profile = profile or {}
    height = profile.get("height_m") or None
    foot_len = profile.get("shoe_length_m") or params.foot_per_height * (height or 1.8)
    f_frames, f_t, f_kp = feet
    shot_of = {s["swing_id"]: s["shot_id"] for s in shots if s.get("swing_id") is not None}
    measured = [
        t["pos"][2] for t in tosses.values() if t.get("pos") is not None and t.get("source")
    ]
    usual_z = (
        float(np.median(measured)) if len(measured) >= 5 else CONTACT_PER_HEIGHT * (height or 1.75)
    )
    rows = []
    for r in serves:
        sid = r["swing_id"]
        toss = sc.toss_from_dict(tosses[str(sid)]) if str(sid) in tosses else None
        frame, tc = r["frame_contact"], r["t_contact"]
        cam = calib.camera_at(cal, tc)
        side = r["side"] if r["side"] is not None else -1
        hand = r["racket_hand"] or "right"
        front = "right" if hand == "left" else "left"
        flags = list(toss.flags) if toss is not None else ["toss_not_tracked"]
        if side == 1:
            flags.append("far")
        flags += [f for f in r["flags"] or [] if f in SWING_FLAGS]
        rec: dict = {
            "swing_id": sid,
            "shot_id": shot_of.get(sid),
            "t_contact": tc,
            "frame_contact": frame,
            "contact_source": toss.source if toss is not None else None,
            "side": side,
            "racket_hand": hand,
            "front_foot": front,
        }
        if toss is not None:
            rec.update(t_release=toss.t_release, toss_points=toss.n_toss)
            rec["toss_rms_px"] = toss.toss_rms_px
        # The toe.
        m = (f_t >= tc - FEET_BEFORE_S - 0.05) & (f_t <= tc + FEET_AFTER_S + 0.05)
        toe = None
        if m.any() and frame is not None:
            series = sc.FeetSeries(f_frames[m], f_t[m], f_kp[m])
            release = None
            if toss is not None and toss.t_release is not None:
                k = int(np.argmin(np.abs(series.t - toss.t_release)))
                if abs(series.t[k] - toss.t_release) < 0.05:
                    release = int(series.frame[k])
            edit = min(toe_edits, key=lambda e: abs(e[0] - tc), default=None)
            edit_px = (
                np.asarray(edit[1], float)
                if edit is not None and abs(edit[0] - tc) <= 0.25
                else None
            )
            toe = sc.toe_at_contact(
                series, cam, front, int(frame), params, foot_length_m=foot_len,
                release_frame=release, edit_px=edit_px,
            )  # fmt: skip
        if toss is None or toss.pos is None:
            _above_frame(flags, toe, cam, side, usual_z)
        if toe is None:
            flags.append("no_toe")
        else:
            flags += toe.flags
            rec.update(
                toe_x=float(toe.pos[0]),
                toe_y=float(toe.pos[1]),
                toe_z=float(toe.pos[2]),
                toe_sigma_m=toe.sigma_m,
                toe_source=toe.source,
                toe_on_ground=toe.on_ground,
                toe_moved_m=toe.moved_m,
                toe_to_baseline_m=sc.toe_to_baseline(toe.pos, side),
                toe_px_x=float(toe.px[0]),
                toe_px_y=float(toe.px[1]),
                serve_side=sc.serve_side_of(float(toe.pos[0]), side),
            )
        # The contact point and the offsets.
        if toss is not None and toss.pos is not None:
            pos, cov = toss.pos, toss.cov
            px = cam.project(pos[None])[0]
            rec.update(
                contact_x=float(pos[0]),
                contact_y=float(pos[1]),
                contact_z=float(pos[2]),
                contact_cov=None if cov is None else np.asarray(cov, float).ravel().tolist(),
                ball_px_x=float(px[0]),
                ball_px_y=float(px[1]),
            )
            if toe is not None:
                o = sc.offsets(pos, cov, toe.pos, toe.cov, side, hand)
                rec.update(
                    forward_m=o.forward_m,
                    lateral_m=o.lateral_m,
                    height_m=o.height_m,
                    height_rel=o.height_m / height if height else None,
                    forward_sigma_m=o.forward_sigma_m,
                    lateral_sigma_m=o.lateral_sigma_m,
                    height_sigma_m=o.height_sigma_m,
                )
        rec["flags"] = list(dict.fromkeys(flags))
        # A null fixed-size list doesn't read back (pyarrow): NaNs stand for "no covariance".
        if rec.get("contact_cov") is None:
            rec["contact_cov"] = [float("nan")] * 9
        rows.append(rec)
    table = pa.table(
        {f.name: pa.array([r.get(f.name) for r in rows], f.type) for f in SERVE_CONTACT},
        schema=SERVE_CONTACT,
    )
    with_offsets = [r for r in rows if r.get("forward_m") is not None]
    near = [r for r in rows if r["side"] == -1]
    summary = {
        "serves": len(rows),
        "near": len(near),
        "with_offsets": len(with_offsets),
        "near_with_offsets": sum(r.get("forward_m") is not None for r in near),
        "contact_frame": {
            k: sum(r["contact_source"] == k for r in rows)
            for k in ("toss_path", "inferred", "edit")
        },
        "flags": {
            f: sum(f in r["flags"] for r in rows)
            for f in sorted({f for r in rows for f in r["flags"]})
        },
        "forward_m_median": _median([r.get("forward_m") for r in with_offsets]),
        "forward_sigma_m_median": _median([r.get("forward_sigma_m") for r in with_offsets]),
    }
    return table, summary


#: The swing's flags a serve contact carries over (in the dark the foot keypoints jitter).
SWING_FLAGS = ("dark", "low_conf")
#: Without a measured contact, a serve is struck this high per metre of the player's height
#: (Oct 1: 2.77 m median for ≈1.78 m).
CONTACT_PER_HEIGHT = 1.55


def _above_frame(flags: list[str], toe, cam, side: int, contact_z: float) -> None:
    """``contact_above_frame`` when no toss path says where the contact was but the player's
    usual contact height over the toe is above the picture (the racket arm is then out of the
    picture too, so the pose can't tell)."""
    if "contact_above_frame" in flags or toe is None:
        return
    fwd = 0.3 if side < 0 else -0.3  # contacts are a little in front of the toe
    X = np.array([[toe.pos[0], toe.pos[1] + fwd, contact_z]])
    if cam.project(X)[0, 1] < 0:
        flags.append("contact_above_frame")
        if "contact_not_seen" in flags:
            flags.remove("contact_not_seen")


def _median(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return round(float(np.median(v)), 4) if v else None


class ServeContactStage(Stage):
    """Contact point vs the front toe in the contact frame, offsets, σ, flags (§6 #15b)."""

    name = "serve_contact"
    title = "Serve contact point"
    version = 1
    depends_on = ("swings", "serve_feet", "shots", "camera")
    weight = 0.1

    def config(self, session, config, settings):
        from swingvision.storage import edits as ed

        e = ed.load(session)
        return {
            "toe": sc.ToeParams().as_config(),
            "feet_window": [FEET_BEFORE_S, FEET_AFTER_S],
            "profile": _profile(settings, config),
            "edits": [[s.t, s.toe] for s in e.serves if s.toe is not None],
        }

    def outputs(self, session):
        return [session.serve_contact_path, session.serve_contact_summary_path]

    def run(self, ctx: StageContext):
        from swingvision.storage import edits as ed

        session = ctx.session
        cal = calib.load(session.calibration_path)
        if cal is None:
            raise RuntimeError("No calibration")
        summary = (
            read_json(session.swings_summary_path) if session.swings_summary_path.exists() else {}
        )
        shots = (
            tables.read_table(session.shots_path).to_pylist() if session.shots_path.exists() else []
        )
        table, out = build_serve_contacts(
            serve_rows(session),
            summary.get("tosses") or {},
            load_feet(session),
            cal,
            shots,
            toe_edits=[(s.t, s.toe) for s in ed.load(session).serves if s.toe is not None],
            profile=_profile(ctx.settings, ctx.config),
        )
        session.pose_dir.mkdir(parents=True, exist_ok=True)
        tables.write_table(table, session.serve_contact_path, SERVE_CONTACT)
        atomic_write_json(session.serve_contact_summary_path, out)
        return out
