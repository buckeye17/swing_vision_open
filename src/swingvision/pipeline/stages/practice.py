"""M5 stages: ``segments`` (practice shots and blocks) and ``practice_eval`` (line calls,
targets, accuracy). Both are CPU-only and take seconds, so the app reruns them itself when
the user edits targets, the practice type or a shot (PLAN.md §6, §7.8, §7.10, §9.3)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.analysis import practice as pr
from swingvision.analysis import segmentation as sg
from swingvision.analysis.shots import LANDING_PX_SIGMA
from swingvision.court import calibration as calib
from swingvision.court.homography import ground_sigma
from swingvision.pipeline.stage import Stage, StageContext, stable_hash
from swingvision.storage import edits as ed
from swingvision.storage import tables
from swingvision.storage.schemas import PRACTICE, SEGMENTS

PRACTICE_MODES = frozenset({"practice"})


def seg_params(settings) -> sg.SegParams:
    pd = settings.processing
    return sg.SegParams(
        pad_before_s=pd.practice_pad_before_s,
        pad_after_s=pd.practice_pad_after_s,
        block_gap_s=pd.practice_block_gap_s,
    )


def _opt(path) -> pa.Table | None:
    return tables.read_table(path) if path.exists() else None


def seg_inputs(session, config) -> sg.SegInputs:
    """Everything segmentation looks at, from the session's files."""
    cal = calib.load(session.calibration_path)
    if cal is None:
        raise RuntimeError("No calibration")
    cams: dict[int, object] = {}

    def landing_sigma(x: float, y: float, t: float) -> float | None:
        key = int(t // 30)
        if key not in cams:
            cams[key] = calib.camera_at(cal, t)
        cam = cams[key]
        sig = ground_sigma(cam, np.array([[x, y]]), LANDING_PX_SIGMA * cam.width / 3840)  # type: ignore[attr-defined]
        v = float(sig[0, 2])
        return v if np.isfinite(v) else None

    inp = sg.SegInputs(
        events=tables.read_table(session.events_path),
        shots=tables.read_table(session.shots_path),
        flights=_opt(session.ball_flights_path),
        duration_s=config.video.duration_s if config.video else 0.0,
        submode=config.practice.submode if config.practice else "self_feed",
        has_audio=bool(config.video and config.video.has_audio),
        camera_xyz=calib.to_camera(cal.camera).center,
        landing_sigma=landing_sigma,
    )
    onsets = _opt(session.audio_onsets_path)
    if onsets is not None:
        inp.onsets_t = onsets.column("t_s").to_numpy()
        inp.onsets_s = onsets.column("strength").to_numpy().astype(np.float64)
    mv = _opt(session.movement_path)
    if mv is not None and mv.num_rows:
        mv = mv.filter(pc.equal(mv.column("player"), "me"))
        inp.player_t = mv.column("t_s").to_numpy()
        inp.player_x = mv.column("x").to_numpy(zero_copy_only=False).astype(np.float64)
        inp.player_y = mv.column("y").to_numpy(zero_copy_only=False).astype(np.float64)
        inp.player_box = np.column_stack(
            [
                mv.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
                for c in ("bx0", "by0", "bx1", "by1")
            ]
        )
    track = _opt(session.ball_track_path)
    if track is not None:
        inp.ball_t = track.column("t_s").to_numpy()
        inp.ball_x = track.column("x").to_numpy(zero_copy_only=False).astype(np.float64)
        inp.ball_y = track.column("y").to_numpy(zero_copy_only=False).astype(np.float64)
    return inp


class SegmentsStage(Stage):
    name = "segments"
    title = "Segments"
    version = 1
    depends_on = ("shots", "events", "ball_3d", "ball_track", "movement", "audio_onsets", "camera")
    modes = PRACTICE_MODES
    weight = 0.1

    def config(self, session, config, settings):
        return {
            "params": seg_params(settings).as_config(),
            "submode": config.practice.submode if config.practice else None,
        }

    def outputs(self, session):
        return [session.segments_path]

    def run(self, ctx: StageContext):
        inp = seg_inputs(ctx.session, ctx.config)
        table, summary = sg.segment_practice(inp, seg_params(ctx.settings))
        tables.write_table(table, ctx.session.segments_path, SEGMENTS)
        return summary


class PracticeEvalStage(Stage):
    name = "practice_eval"
    title = "Practice accuracy"
    version = 1
    depends_on = ("segments", "shots", "ball_3d")
    modes = PRACTICE_MODES
    weight = 0.05

    def config(self, session, config, settings):
        targets = config.practice.targets if config.practice else []
        edits = ed.load(session)
        return {
            "targets": [t.model_dump() for t in targets],
            "edits": stable_hash(edits.model_dump(exclude={"version"})),
            "close_call_sigmas": pr.CLOSE_CALL_SIGMAS,
        }

    def outputs(self, session):
        return [session.practice_path]

    def run(self, ctx: StageContext):
        session, config = ctx.session, ctx.config
        table = pr.evaluate(
            tables.read_table(session.segments_path),
            tables.read_table(session.shots_path),
            config.practice.targets if config.practice else [],
            ed.load(session),
            _opt(session.ball_flights_path),
        )
        tables.write_table(table, session.practice_path, PRACTICE)
        s = pr.summarize(table.to_pylist())
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
