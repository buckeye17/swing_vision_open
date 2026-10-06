"""M3 stages: ``ball_refine`` (GPU), ``ball_track`` and ``events`` (PLAN.md §6, §7.4-7.5).

``pass1_detect`` runs the ball detector at the sweep rate (or on every frame). With a sweep
rate, ``ball_refine`` finds the moments that need every frame (hits, bounces, strong audio
onsets, gaps where the sweep lost the ball) from a quick track of the sweep candidates and
re-runs the detector at the full frame rate in windows around them. ``ball_track`` links the
union of both candidate sets into one ball track, and ``events`` finds hits, bounces and net
contacts on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.ball import events as ev
from swingvision.ball.schedule import merge_windows, target_frames
from swingvision.ball.trajectory import LinkParams, PlayerBoxes, link
from swingvision.court import calibration as calib
from swingvision.io.frames import frame_table
from swingvision.pipeline.stage import NeedsUserAction, Stage, StageContext
from swingvision.pipeline.stages.ingest import _source_path
from swingvision.pipeline.stages.players import (
    ball_detection_crop,
    ball_spec,
    load_players_summary,
)
from swingvision.storage import tables
from swingvision.storage.schemas import BALL_CANDIDATES, BALL_FRAMES, BALL_TRACK, EVENTS

#: Audio onsets at least this strong (robust z) are refine moments.
ONSET_MOMENT = 8.0
#: A sweep track gap longer than this (but shorter than ``GAP_MAX_S``) is a refine moment.
GAP_MIN_S = 0.2
GAP_MAX_S = 2.0


def _player_boxes(session) -> PlayerBoxes | None:
    if not session.movement_path.exists():
        return None
    return PlayerBoxes.from_movement(tables.read_table(session.movement_path))


def event_context(session, config, cal, onsets: pa.Table | None) -> ev.EventContext:
    assert config.video is not None
    boxes = _player_boxes(session)
    player_xy = None
    if session.movement_path.exists():
        mv = tables.read_table(session.movement_path)
        mv = mv.filter(pc.equal(mv.column("player"), "me"))
        player_xy = (
            mv.column("t_s").to_numpy(),
            mv.column("x").to_numpy(zero_copy_only=False).astype(np.float64),
            mv.column("y").to_numpy(zero_copy_only=False).astype(np.float64),
        )
    machine = None
    summary = load_players_summary(session)
    if summary is not None and summary.machine is not None:
        machine = (summary.machine.x, summary.machine.y)
    cams: dict[int, object] = {}

    def camera_at(t: float):
        # Cache by drift window (camera_at is cheap but called per kink).
        key = int(t // 30)
        if key not in cams:
            cams[key] = calib.camera_at(cal, t)
        return cams[key]

    return ev.EventContext(
        camera_at=camera_at,
        width=config.video.display_width,
        players=boxes,
        player_xy=player_xy,
        onsets_t=onsets.column("t_s").to_numpy() if onsets is not None else np.zeros(0),
        onsets_s=onsets.column("strength").to_numpy() if onsets is not None else np.zeros(0),
        machine_xy=machine,
    )


def _onsets(session) -> pa.Table | None:
    p = session.audio_onsets_path
    return tables.read_table(p) if p.exists() else None


def refine_moments(track: pa.Table, events: pa.Table, onsets: pa.Table | None) -> list[float]:
    """Times worth a full-rate look: events, strong onsets, gaps and run ends of the sweep."""
    out: list[float] = []
    if events.num_rows:
        out += events.column("t_s").to_numpy().tolist()
    if onsets is not None and onsets.num_rows:
        s = onsets.column("strength").to_numpy()
        out += onsets.column("t_s").to_numpy()[s >= ONSET_MOMENT].tolist()
    if track.num_rows:
        t = track.column("t_s").to_numpy()
        d = np.diff(t)
        gaps = np.flatnonzero((d > GAP_MIN_S) & (d < GAP_MAX_S))
        out += (t[gaps] + d[gaps] / 2).tolist()
        # Where runs start and end the ball was hit, fed or lost.
        cut = np.flatnonzero(d >= GAP_MAX_S)
        out += t[np.r_[0, cut + 1]].tolist() + t[np.r_[cut, len(t) - 1]].tolist()
    return sorted(set(round(x, 4) for x in out))


def link_session(
    session, config, cand: pa.Table, frames: pa.Table, params: LinkParams | None = None
) -> pa.Table:
    assert config.video is not None
    ft = frame_table(str(_source_path_of(config)))
    allf = np.arange(len(ft), dtype=np.int64)
    track, _ = link(
        cand,
        frames,
        config.video.display_width,
        params,
        all_frames=(allf, ft.times()),
        player_boxes=_player_boxes(session),
    )
    return track


def _source_path_of(config) -> Path:
    src = Path(config.source.path)
    if not src.exists():
        raise NeedsUserAction("relink", f"Source video not found: {src}. Relink it in the Library.")
    return src


class BallRefineStage(Stage):
    name = "ball_refine"
    title = "Ball detection (full-rate windows)"
    version = 1
    depends_on = ("pass1_detect", "audio_onsets", "movement", "camera")
    uses_gpu = True
    weight = 6.0

    def config(self, session, config, settings):
        p = settings.processing
        crop = ball_detection_crop(session, settings)
        return {
            "detector": ball_spec(settings),
            "sweep_hz": p.ball_sweep_hz,
            "before_s": p.ball_refine_before_s,
            "after_s": p.ball_refine_after_s,
            "crop": crop.as_list() if crop else None,
            "link": LinkParams().as_config(),
            "events": ev.EventParams().as_config(),
            "view_min": p.view_min,
        }

    def outputs(self, session):
        return [session.ball_refine_path, session.ball_refine_frames_path]

    def run(self, ctx: StageContext):
        from swingvision.ball.detect import run_detector
        from swingvision.ball.detectors import make_ball_detector
        from swingvision.io.frames import open_source

        session, config = ctx.session, ctx.config
        p = ctx.settings.processing
        session.ball_dir.mkdir(parents=True, exist_ok=True)
        if not p.ball_sweep_hz:
            # Every frame was already processed in pass 1.
            tables.write_table(
                BALL_CANDIDATES.empty_table(), session.ball_refine_path, BALL_CANDIDATES
            )
            tables.write_table(
                BALL_FRAMES.empty_table(), session.ball_refine_frames_path, BALL_FRAMES
            )
            return {"windows": 0, "frames": 0}
        cal = calib.load(session.calibration_path)
        if cal is None:
            raise RuntimeError("No calibration")
        ctx.progress(0.02, "Finding moments")
        cand = tables.read_table(session.ball_sweep_path)
        frames = tables.read_table(session.ball_sweep_frames_path)
        cand, frames = drop_off_view(session, ctx.settings, cand, frames)
        track = link_session(session, config, cand, frames)
        onsets = _onsets(session)
        events, _ = ev.detect_events(track, event_context(session, config, cal, onsets))
        assert config.video is not None
        windows = merge_windows(
            refine_moments(track, events, onsets),
            p.ball_refine_before_s,
            p.ball_refine_after_s,
            config.video.duration_s,
        )
        spec = ball_spec(ctx.settings)
        det = make_ball_detector(spec)
        det.prepare(config.video, ball_detection_crop(session, ctx.settings))
        with open_source(_source_path(ctx), config.video, p.decode_backend) as src:
            targets = target_frames(src.table, windows)
            res = run_detector(
                src,
                det,
                targets,
                progress=lambda f: ctx.progress(0.05 + 0.95 * f, "Full-rate windows"),
                check_cancel=ctx.check_cancel,
            )
        tables.write_table(res.candidates, session.ball_refine_path, BALL_CANDIDATES)
        tables.write_table(res.frames, session.ball_refine_frames_path, BALL_FRAMES)
        covered = sum(w.t1 - w.t0 for w in windows)
        return {
            "windows": len(windows),
            "seconds_covered": round(covered, 2),
            "share_of_video": round(covered / max(config.video.duration_s, 1e-6), 4),
            "frames": res.frames.num_rows,
            "candidates": res.candidates.num_rows,
            "detect_seconds": round(res.detect_seconds, 2),
        }


def drop_off_view(session, settings, cand: pa.Table, frames: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Remove frames where the camera wasn't on the court (pass 1's view check, sampled at the
    person rate; the nearest sample within 0.1 s applies)."""
    if not session.pass1_frames_path.exists() or not frames.num_rows:
        return cand, frames
    pf = tables.read_table(session.pass1_frames_path, columns=["t_s", "view"])
    pt = pf.column("t_s").to_numpy()
    view = pf.column("view").to_numpy()
    if not len(pt):
        return cand, frames

    def bad(t: np.ndarray) -> np.ndarray:
        k = np.clip(np.searchsorted(pt, t), 1, len(pt) - 1)
        k = np.where(np.abs(pt[k - 1] - t) < np.abs(pt[k] - t), k - 1, k)
        near = np.abs(pt[k] - t) <= 0.1
        return near & (view[k] < settings.processing.view_min)

    off = bad(frames.column("t_s").to_numpy())
    if not off.any():
        return cand, frames
    off_frames = pa.array(frames.column("frame").to_numpy()[off])
    keep_c = pc.invert(pc.is_in(cand.column("frame"), off_frames))
    return cand.filter(keep_c), frames.filter(pa.array(~off))


def merged_candidates(session) -> tuple[pa.Table, pa.Table]:
    """Sweep + refine candidates; refined frames replace the sweep's for the same frames."""
    cand = tables.read_table(session.ball_sweep_path)
    frames = tables.read_table(session.ball_sweep_frames_path)
    if session.ball_refine_path.exists():
        rc = tables.read_table(session.ball_refine_path)
        rf = tables.read_table(session.ball_refine_frames_path)
        if rf.num_rows:
            refined = pa.array(np.unique(rf.column("frame").to_numpy()))
            cand = cand.filter(pc.invert(pc.is_in(cand.column("frame"), refined)))
            frames = frames.filter(pc.invert(pc.is_in(frames.column("frame"), refined)))
            cand = pa.concat_tables([cand, rc]).sort_by("frame")
            frames = pa.concat_tables([frames, rf]).sort_by("frame")
    return cand, frames


class BallTrackStage(Stage):
    name = "ball_track"
    title = "Ball tracking"
    version = 2
    depends_on = ("pass1_detect", "ball_refine", "movement")
    weight = 1.0

    def config(self, session, config, settings):
        return {"link": LinkParams().as_config(), "view_min": settings.processing.view_min}

    def outputs(self, session):
        return [session.ball_track_path]

    def run(self, ctx: StageContext):
        session = ctx.session
        ctx.progress(0.05, "Linking")
        cand, frames = merged_candidates(session)
        cand, frames = drop_off_view(session, ctx.settings, cand, frames)
        track = link_session(session, ctx.config, cand, frames)
        tables.write_table(track, session.ball_track_path, BALL_TRACK)
        src = np.array(track.column("source").to_pylist())
        return {
            "candidates": cand.num_rows,
            "frames_processed": frames.num_rows,
            "track_rows": track.num_rows,
            "detected": int((src == "detected").sum()),
            "interpolated": int((src == "interp").sum()),
        }


KINKS = pa.schema(
    [
        pa.field("frame", pa.int64()),
        pa.field("t_s", pa.float64()),
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
        *[pa.field(n, pa.float32()) for n in ev.FEATURE_NAMES],
    ],
    metadata={"name": "ball_kinks", "schema_version": "1"},
)


def kinks_table(feats: list[dict]) -> pa.Table:
    cols = {f.name: [d.get(f.name) for d in feats] for f in KINKS}
    return pa.table({f.name: pa.array(cols[f.name], f.type) for f in KINKS}, schema=KINKS)


class EventsStage(Stage):
    name = "events"
    title = "Hits and bounces"
    version = 1
    depends_on = ("ball_track", "audio_onsets", "movement", "camera", "players_track")
    weight = 0.5

    def config(self, session, config, settings):
        from swingvision.training.train_events import active_model_name

        return {
            "params": ev.EventParams().as_config(),
            "model": active_model_name(settings.output_root),
        }

    def outputs(self, session):
        return [session.events_path, session.ball_kinks_path]

    def run(self, ctx: StageContext):
        from swingvision.training.train_events import load_active_model

        session, config = ctx.session, ctx.config
        cal = calib.load(session.calibration_path)
        if cal is None:
            raise RuntimeError("No calibration")
        track = tables.read_table(session.ball_track_path)
        evctx = event_context(session, config, cal, _onsets(session))
        model = load_active_model(ctx.settings.output_root)
        events, feats = ev.detect_events(track, evctx, model=model)
        tables.write_table(events, session.events_path, EVENTS)
        tables.write_table(kinks_table(feats), session.ball_kinks_path, KINKS)
        kinds = events.column("kind").to_pylist()
        return {
            "kinks": len(feats),
            "hits": kinds.count("hit"),
            "bounces": kinds.count("bounce"),
            "net": kinds.count("net"),
            "model": model.name if model is not None else None,
        }
