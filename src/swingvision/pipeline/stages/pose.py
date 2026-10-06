"""M6 stages: ``pass2_pose`` (2D pose in swing windows, GPU), ``pose3d`` (lifting and court
placement) and ``swings`` (phases, kinematics, strokes) (PLAN.md §6, §7.7)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.court import calibration as calib
from swingvision.io.frames import open_source
from swingvision.pipeline.stage import Stage, StageContext
from swingvision.pipeline.stages.ingest import _source_path
from swingvision.pipeline.stages.players import player_height
from swingvision.pose import lift3d
from swingvision.pose import pose2d as p2d
from swingvision.pose import swings as sw
from swingvision.pose.windows import BoxTrack, merge_windows
from swingvision.storage import tables
from swingvision.storage.schemas import POSE2D, POSE3D, SWINGS

POSE_BATCH = 64
#: A gap of more than this many frames splits a window's pose into separate lifted clips.
CLIP_GAP_FRAMES = 3


def me_movement(session) -> pa.Table | None:
    if not session.movement_path.exists():
        return None
    mv = tables.read_table(session.movement_path)
    return mv.filter(pc.equal(mv.column("player"), "me"))


def box_track(mv: pa.Table | None) -> BoxTrack:
    if mv is None or mv.num_rows == 0:
        return BoxTrack(np.zeros(0), np.zeros((0, 4)))
    boxes = np.column_stack(
        [
            mv.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
            for c in ("bx0", "by0", "bx1", "by1")
        ]
    )
    return BoxTrack(mv.column("t_s").to_numpy(), boxes)


def pose_anchors(session, onset_z: float) -> np.ndarray:
    """The player's hits and strong impact sounds (unseen contacts)."""
    ev = tables.read_table(session.events_path, columns=["kind", "hitter", "t_s"])
    kind = ev.column("kind").to_pylist()
    hitter = ev.column("hitter").to_pylist()
    t = ev.column("t_s").to_pylist()
    anchors = [ti for k, h, ti in zip(kind, hitter, t, strict=True) if k == "hit" and h == "me"]
    if session.audio_onsets_path.exists():
        on = tables.read_table(session.audio_onsets_path, columns=["t_s", "strength"])
        st = on.column("strength").to_numpy()
        anchors += on.column("t_s").to_numpy()[st >= onset_z].tolist()
    return np.asarray(anchors, dtype=np.float64)


class Pass2PoseStage(Stage):
    """2D pose of the player on every frame of the windows around swings (PLAN.md §6 #13)."""

    name = "pass2_pose"
    title = "2D pose"
    version = 1
    depends_on = ("events", "movement", "audio_onsets")
    after = ("camera",)
    uses_gpu = True
    weight = 10.0

    def config(self, session, config, settings):
        p = settings.processing
        return {
            "model": p.pose_model,
            "before_s": p.pose_before_s,
            "after_s": p.pose_after_s,
            "onset_z": p.pose_onset_z,
            "flip_test": p.pose_flip_test,
            "crop": [p2d.PAD_H, p2d.PAD_W, p2d.UP_SHIFT],
        }

    def outputs(self, session):
        return [session.pose2d_path]

    def run(self, ctx: StageContext):
        session, info = ctx.session, ctx.config.video
        assert info is not None, "ingest must run first"
        p = ctx.settings.processing
        boxes = box_track(me_movement(session))
        anchors = [
            t for t in pose_anchors(session, p.pose_onset_z) if boxes.covered(t - 0.5, t + 0.5)
        ]
        windows = [
            w
            for w in merge_windows(
                np.array(anchors), p.pose_before_s, p.pose_after_s, info.duration_s
            )
            if boxes.covered(*w)
        ]
        total = sum(b - a for a, b in windows)
        chunk_of = [int(a // p.chunk_seconds) for a, _ in windows]
        chunk_ids = sorted(set(chunk_of))
        tracker = ctx.chunks(len(chunk_ids))
        parts = tracker.parts_dir / "pose2d"
        ctx.log.info(
            "Pose in {} windows ({:.0f} s of {:.0f} s) around {} anchors",
            len(windows), total, info.duration_s, len(anchors),
        )  # fmt: skip
        total = total or 1.0
        est = None
        done_s = 0.0
        source = open_source(_source_path(ctx), info, p.decode_backend)
        with source:
            for ci, cid in enumerate(chunk_ids):
                mine = [(wi, w) for wi, w in enumerate(windows) if chunk_of[wi] == cid]
                if tracker.is_done(ci):
                    done_s += sum(b - a for _, (a, b) in mine)
                    continue
                if est is None:
                    ctx.progress(done_s / total, f"Loading {p.pose_model}")
                    est = p2d.make_estimator(
                        p.pose_model,
                        flip_test=p.pose_flip_test,
                        progress=lambda f, d=done_s: ctx.progress(
                            d / total, f"Downloading {p.pose_model} {f:.0%}"
                        ),
                    )
                rows: list[tuple] = []
                pending: list[tuple] = []  # (frame, t, window, crop, rect)

                def flush(pending=pending, rows=rows, est=est) -> None:
                    if not pending:
                        return
                    kp = est.run([x[3] for x in pending], np.array([x[4] for x in pending]))
                    for (fi, t, wi, _, r), k in zip(pending, kp, strict=True):
                        rows.append((fi, t, wi, k, r))
                    pending.clear()

                for wi, (a, b) in mine:
                    for f in source.frames(a, b):
                        box = boxes.at(f.t_s)
                        if box is None:
                            continue
                        crop, rect = est.prepare(f.image, box)
                        pending.append((f.index, f.t_s, wi, crop, rect))
                        if len(pending) >= POSE_BATCH:
                            flush()
                            ctx.check_cancel()
                            ctx.progress(
                                (done_s + f.t_s - a) / total,
                                f"Window {wi + 1}/{len(windows)}",
                            )
                    flush()
                    done_s += b - a
                tables.write_part(_pose2d_table(rows), parts, ci, POSE2D)
                tracker.mark_done(ci)
        session.pose_dir.mkdir(parents=True, exist_ok=True)
        out = tables.consolidate_parts(parts, session.pose2d_path, POSE2D, sort_by="t_s")
        tracker.cleanup()
        score = out.column("score").to_numpy() if out.num_rows else np.zeros(0)
        return {
            "windows": len(windows),
            "window_s": round(total, 1),
            "frames": out.num_rows,
            "mean_score": round(float(score.mean()), 3) if len(score) else None,
            "decoder": source.backend,
        }


def _pose2d_table(rows: list[tuple]) -> pa.Table:
    if not rows:
        return POSE2D.empty_table()
    kp = np.stack([r[3] for r in rows]).reshape(len(rows), -1).astype(np.float32)
    rect = np.stack([r[4] for r in rows]).astype(np.float32)
    conf = kp[:, 2::3]
    return pa.table(
        {
            "frame": pa.array([r[0] for r in rows], pa.int64()),
            "t_s": pa.array([r[1] for r in rows], pa.float64()),
            "player": pa.array(["me"] * len(rows)),
            "window": pa.array([r[2] for r in rows], pa.int32()),
            "kp": pa.FixedSizeListArray.from_arrays(pa.array(kp.ravel()), 51),
            "x0": rect[:, 0],
            "y0": rect[:, 1],
            "x1": rect[:, 2],
            "y1": rect[:, 3],
            "score": conf.mean(1),
        },
        schema=POSE2D,
    )


def fixed_list(table: pa.Table, column: str, width: int = 17) -> np.ndarray:
    """A ``list<float32>[51]`` column → (n, width, 3) array."""
    col = table.column(column).combine_chunks()
    if len(col) == 0:
        return np.zeros((0, width, 3))
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), width, 3)


def clips_of(frames: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """Clip number per row: a new clip at every window change or frame gap."""
    if len(frames) == 0:
        return np.zeros(0, dtype=np.int32)
    new = np.r_[True, (np.diff(frames) > CLIP_GAP_FRAMES) | (np.diff(windows) != 0)]
    return (np.cumsum(new) - 1).astype(np.int32)


class Pose3DStage(Stage):
    """MotionBERT lifting, scaled to the player's height and placed on the court."""

    name = "pose3d"
    title = "3D pose"
    version = 1
    depends_on = ("pass2_pose", "camera", "movement")
    uses_gpu = True
    weight = 1.0

    def config(self, session, config, settings):
        return {
            "model": lift3d.LIFTER,
            "height_m": player_height(settings, config),
            "place": lift3d.PlaceParams().as_config(),
            "clip_gap": CLIP_GAP_FRAMES,
        }

    def outputs(self, session):
        return [session.pose3d_path]

    def run(self, ctx: StageContext):
        session = ctx.session
        cal = calib.load(session.calibration_path)
        if cal is None:
            raise RuntimeError("No calibration")
        p2 = tables.read_table(session.pose2d_path)
        p2 = p2.sort_by([("window", "ascending"), ("frame", "ascending")])
        frames = p2.column("frame").to_numpy()
        t = p2.column("t_s").to_numpy()
        kp = fixed_list(p2, "kp")
        clips = clips_of(frames, p2.column("window").to_numpy())
        height = player_height(ctx.settings, ctx.config)
        mv = me_movement(session)
        feet = sig = None
        mt = np.zeros(0)
        if mv is not None and mv.num_rows:
            mt = mv.column("t_s").to_numpy()
            feet = np.column_stack(
                [mv.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in ("x", "y")]
            )
            sig = mv.column("sigma_m").to_numpy(zero_copy_only=False).astype(np.float64)
        lifter = (
            lift3d.make_lifter(progress=lambda f: ctx.progress(0.0, f"Downloading {f:.0%}"))
            if len(t)
            else None
        )
        from swingvision.pose.skeleton import coco_to_h36m

        n_clips = int(clips.max()) + 1 if len(clips) else 0
        out_joints = np.full((len(t), 17, 3), np.nan, dtype=np.float32)
        out_rms = np.full(len(t), np.nan, dtype=np.float32)
        out_scale = np.full(len(t), np.nan, dtype=np.float32)
        starts = np.flatnonzero(np.r_[True, np.diff(clips) != 0])
        ends = np.r_[starts[1:], len(clips)]
        for ci, (a, b) in enumerate(zip(starts, ends, strict=True)):
            if ci % 20 == 0:
                ctx.check_cancel()
                ctx.progress(ci / max(n_clips, 1), f"Clip {ci + 1}/{n_clips}")
            if b - a < 5:
                continue
            h36 = coco_to_h36m(kp[a:b])
            assert lifter is not None
            lifted = lifter.lift(h36)
            cam = calib.camera_at(cal, float(np.median(t[a:b])))
            fxy = fs = None
            if feet is not None and len(mt):
                idx = np.clip(np.searchsorted(mt, t[a:b]), 1, len(mt) - 1)
                near = np.minimum(np.abs(mt[idx] - t[a:b]), np.abs(mt[idx - 1] - t[a:b])) < 0.3
                fxy = np.column_stack([np.interp(t[a:b], mt, feet[:, k]) for k in (0, 1)])
                fxy[~near] = np.nan
                fs = np.interp(t[a:b], mt, sig)
            placed = lift3d.place(lifted, h36, frames[a:b], cam, height, fxy, fs)
            out_joints[a:b] = placed.joints
            out_rms[a:b] = placed.reproj_px
            out_scale[a:b] = placed.scale
        conf = kp[..., 2].mean(1) if len(kp) else np.zeros(0)
        table = pa.table(
            {
                "frame": pa.array(frames, pa.int64()),
                "t_s": pa.array(t, pa.float64()),
                "player": p2.column("player"),
                "clip": pa.array(clips, pa.int32()),
                "joints": pa.FixedSizeListArray.from_arrays(
                    pa.array(out_joints.reshape(-1), pa.float32()), 51
                ),
                "conf": pa.array(conf, pa.float32()),
                "reproj_px": pa.array(out_rms, pa.float32()),
                "scale": pa.array(out_scale, pa.float32()),
            },
            schema=POSE3D,
        ).sort_by("t_s")
        session.pose_dir.mkdir(parents=True, exist_ok=True)
        tables.write_table(table, session.pose3d_path, POSE3D)
        ok = np.isfinite(out_joints[:, 0, 0])
        return {
            "frames": len(t),
            "placed": int(ok.sum()),
            "clips": n_clips,
            "reproj_px_median": round(float(np.nanmedian(out_rms)), 2) if ok.any() else None,
            "height_m": height,
        }


def swing_inputs(session, config, settings) -> sw.SwingInputs:
    """Pose, hits and sounds for the ``swings`` stage, from the session's files."""
    from swingvision.analysis import segmentation as sg
    from swingvision.storage.library import Library

    cal = calib.load(session.calibration_path)
    if cal is None:
        raise RuntimeError("No calibration")
    cam_xyz = calib.to_camera(cal.camera).center
    pose = sw.PoseSeries.from_table(tables.read_table(session.pose3d_path))
    events = tables.read_table(session.events_path).to_pylist()
    hits = [(e["event_id"], e["t_s"]) for e in events if e["kind"] == "hit" and e["hitter"] == "me"]
    inp = sw.SwingInputs(
        pose=pose,
        fps=config.video.fps_avg if config.video else 60.0,
        hits=hits,
        camera_xyz=cam_xyz,
    )
    pid = config.players.me_profile_id
    if pid and settings.output_root is not None:
        row = Library(settings.output_root).get_profile(pid)
        inp.profile_hand = row["handedness"] if row else None
    if session.pass1_frames_path.exists():
        fr = tables.read_table(session.pass1_frames_path, columns=["t_s", "luma"]).sort_by("t_s")
        inp.luma_t = fr.column("t_s").to_numpy()
        inp.luma = fr.column("luma").to_numpy(zero_copy_only=False).astype(np.float64)
    if session.audio_onsets_path.exists():
        on = tables.read_table(session.audio_onsets_path, columns=["t_s", "strength"]).sort_by(
            "t_s"
        )
        inp.onsets_t = on.column("t_s").to_numpy()
        inp.onsets_s = on.column("strength").to_numpy().astype(np.float64)
        ev_t = np.array([e["t_s"] for e in events])
        lone = [
            e
            for e in events
            if e["kind"] == "hit"
            and e["hitter"] == "me"
            and np.sum(np.abs(ev_t - e["t_s"]) < 0.35) == 1
        ]

        def xy(e):
            ok = e["court_x"] is not None and np.isfinite(e["court_x"])
            return (e["court_x"], e["court_y"]) if ok else None

        hit_t = np.array([e["t_s"] for e in lone])
        delay = np.array([sg.sound_delay(cam_xyz, xy(e)) for e in lone])
        p = sg.SegParams()
        gaps = sg.av_gaps(
            hit_t, delay, inp.onsets_t, inp.onsets_s, -p.av_offset_max_s, p.av_offset_max_s
        )
        inp.av_offset_n = len(gaps)
        if len(gaps):
            inp.av_offset_s = sg.offset_from_gaps(gaps, p.av_offset_max_s)
    return inp


def ball_context(session) -> sw.BallContext:
    events = tables.read_table(session.events_path).to_pylist()
    flights = (
        tables.read_table(session.ball_flights_path).to_pylist()
        if session.ball_flights_path.exists()
        else []
    )
    return sw.BallContext.from_tables(events, flights)


class SwingsStage(Stage):
    """Swings: contact, racket hand, phases, kinematics, stroke (PLAN.md §6 #15)."""

    name = "swings"
    title = "Swings and strokes"
    version = 1
    depends_on = ("pose3d", "events", "ball_3d", "audio_onsets", "camera")
    weight = 0.3

    def config(self, session, config, settings):
        from swingvision.pose import strokes as st
        from swingvision.storage import edits as ed

        model = st.active_model(settings.output_root, settings.processing.stroke_model)
        e = ed.load(session)
        pid = config.players.me_profile_id
        return {
            "params": sw.SwingParams().as_config(),
            "strokes": st.StrokeParams().as_config(),
            "profile": pid,
            "hand": _profile_hand(settings, pid),
            "model": model.card.get("trained_at") if model else None,
            "model_name": model.name if model else None,
            "edits": [[s.t, s.stroke] for s in e.swings],
        }

    def outputs(self, session):
        return [session.swings_path, session.swings_summary_path]

    def run(self, ctx: StageContext):
        from swingvision.pose import strokes as st
        from swingvision.storage import edits as ed
        from swingvision.storage.fsutil import atomic_write_json

        session, config = ctx.session, ctx.config
        ctx.progress(0.1, "Loading pose")
        inp = swing_inputs(session, config, ctx.settings)
        model = st.active_model(ctx.settings.output_root, ctx.settings.processing.stroke_model)
        ctx.progress(0.3, "Analyzing swings")
        table, summary = sw.build_swings(
            inp,
            ball_context(session),
            model=model,
            stroke_edits=[(s.t, s.stroke) for s in ed.load(session).swings],
        )
        session.pose_dir.mkdir(parents=True, exist_ok=True)
        tables.write_table(table, session.swings_path, SWINGS)
        atomic_write_json(session.swings_summary_path, summary)
        return summary


def _profile_hand(settings, pid) -> str | None:
    from swingvision.storage.library import Library

    if not pid or settings.output_root is None:
        return None
    row = Library(settings.output_root).get_profile(pid)
    return row["handedness"] if row else None
