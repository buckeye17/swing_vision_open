"""``pass1_detect`` (GPU person detection + ball sweep), ``players_track``, ``movement``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from swingvision.court import calibration as calib
from swingvision.io.frames import open_source
from swingvision.pipeline.stage import (
    Stage,
    StageContext,
    chunk_ranges,
    read_manifest,
    stable_hash,
)
from swingvision.pipeline.stages.ingest import _source_path
from swingvision.players import movement as mv
from swingvision.players import track as tr
from swingvision.players.detect import Crop, make_detector, roi_crop
from swingvision.storage import tables
from swingvision.storage.fsutil import atomic_write_json, atomic_write_text, read_json
from swingvision.storage.library import Library
from swingvision.storage.schemas import (
    BALL_CANDIDATES,
    BALL_FRAMES,
    MOVEMENT,
    PASS1_FRAMES,
    PERSON_DETECTIONS,
    PLAYER_TRACKS,
    PlayersSummary,
)

BATCH = 8
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
VIEW_STRIDE = 16


def fill_nearest(items: list) -> list:
    """``None`` entries take the nearest earlier non-``None`` item, else the nearest later one
    (all ``None``: unchanged)."""
    out = list(items)
    last = None
    for i, v in enumerate(items):
        if v is None:
            out[i] = last
        else:
            last = v
    nxt = None
    for i in range(len(items) - 1, -1, -1):
        if items[i] is not None:
            nxt = items[i]
        elif out[i] is None:
            out[i] = nxt
    return out


class ViewCheck:
    """Correlates each frame (strided, gray) with the court background of its time window.

    The per-window median images from ``court_auto`` follow a camera that was re-aimed. Dark
    windows have none and use the nearest window's image (the camera doesn't move in the
    dark, and the session background may show it somewhere else); without any, the session
    background.
    """

    def __init__(self, session, cal, width: int, height: int, device: str):
        import cv2
        import torch

        from swingvision.pipeline.stages.court import read_rgb

        def vec(rgb):
            if rgb is None:
                return None
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(np.ascontiguousarray(rgb[::VIEW_STRIDE, ::VIEW_STRIDE]))
            return self._normalize(t.permute(2, 0, 1)[None].float().to(device))[0]

        self.main = vec(read_rgb(session.court_background_path))
        drift = cal.drift if cal is not None else []
        images = [
            vec(read_rgb(session.court_window_path(w.index))) if w.index is not None else None
            for w in drift
        ]
        self.windows = [
            (w.t0_s, w.t1_s, v) for w, v in zip(drift, fill_nearest(images), strict=True)
        ]

    @staticmethod
    def _normalize(small):
        import torch

        w = torch.tensor(LUMA_WEIGHTS, device=small.device).view(1, 3, 1, 1)
        g = (small * w).sum(1).flatten(1)
        g = g - g.mean(1, keepdim=True)
        return g / (g.norm(dim=1, keepdim=True) + 1e-6)

    def scores(self, small, times: list[float]) -> np.ndarray:
        """``small``: (B, 3, h, w) float frames strided like the references."""
        import torch

        refs = []
        for t in times:
            ref = next(
                (v for t0, t1, v in self.windows if t0 <= t < t1 and v is not None), self.main
            )
            refs.append(ref)
        if any(r is None for r in refs):
            return np.full(len(times), np.nan, dtype=np.float32)
        g = self._normalize(small)
        if g.shape[1] != refs[0].shape[0]:
            return np.full(len(times), np.nan, dtype=np.float32)
        return (g * torch.stack(refs)).sum(1).cpu().numpy()


def detection_crop(session, settings) -> Crop | None:
    """The court-ROI crop for every camera of the session's calibration (None: not yet)."""
    cal = calib.load(session.calibration_path)
    if cal is None:
        return None
    p = settings.processing
    cams = calib.all_cameras(cal)
    return roi_crop(cams, cal.camera.width, cal.camera.height, p.roi_behind_m, p.roi_beside_m)


def ball_spec(settings) -> str:
    from swingvision.ball.detectors import resolve_spec

    return resolve_spec(settings.processing.ball_detector, settings.output_root)


def ball_detection_crop(session, settings) -> Crop | None:
    from swingvision.ball.detect import ball_crop

    cal = calib.load(session.calibration_path)
    if cal is None:
        return None
    p = settings.processing
    return ball_crop(cal, p.roi_behind_m, p.roi_beside_m)


class Pass1DetectStage(Stage):
    """One decode pass over the video: person boxes and the ball sweep.

    Runs after the calibration gate because it crops to the court ROI, but only the coarse,
    64 px-snapped crops are part of its fingerprint: adjusting the calibration reruns the cheap
    CPU stages, not this one.

    The two parts have their own fingerprints (``pass1/persons.json`` remembers the persons
    part): changing only the ball detector decodes the video again but keeps the person boxes.
    """

    name = "pass1_detect"
    title = "Player and ball detection"
    version = 2
    depends_on = ("ingest",)
    after = ("camera",)
    uses_gpu = True
    weight = 24.0

    def persons_config(self, session, settings) -> dict:
        p = settings.processing
        crop = detection_crop(session, settings)
        return {
            "model": p.person_model,
            "rate_hz": p.person_rate_hz,
            "input_px": p.person_input_px,
            "conf": p.person_conf,
            "crop": crop.as_list() if crop else None,
        }

    def ball_config(self, session, settings) -> dict:
        p = settings.processing
        crop = ball_detection_crop(session, settings)
        return {
            "detector": ball_spec(settings),
            "sweep_hz": p.ball_sweep_hz,
            "crop": crop.as_list() if crop else None,
        }

    def config(self, session, config, settings):
        persons = self.persons_config(session, settings)
        _migrate_persons_state(session, persons)
        return {"persons": persons, "ball": self.ball_config(session, settings)}

    def outputs(self, session):
        return [
            session.person_detections_path,
            session.pass1_frames_path,
            session.ball_sweep_path,
            session.ball_sweep_frames_path,
        ]

    def run(self, ctx: StageContext):
        import torch

        from swingvision.ball.detect import StreamDetector, target_mask
        from swingvision.ball.detectors import make_ball_detector
        from swingvision.ball.schedule import Window, target_frames

        session = ctx.session
        src_path = _source_path(ctx)
        info = ctx.config.video
        assert info is not None, "ingest must run first"
        p = ctx.settings.processing
        crop = detection_crop(session, ctx.settings)
        bcrop = ball_detection_crop(session, ctx.settings)
        if crop is None or bcrop is None:
            raise RuntimeError("No calibration: the camera stage must run first")
        persons_hash = stable_hash(self.persons_config(session, ctx.settings))
        do_persons = not _persons_fresh(session, persons_hash)
        detector = None
        if do_persons:
            ctx.progress(0.0, f"Loading {p.person_model}")
            detector = make_detector(
                p.person_model,
                input_size=p.person_input_px,
                conf=p.person_conf,
                progress=lambda f: ctx.progress(0.0, f"Downloading {p.person_model} {f:.0%}"),
            )
        spec = ball_spec(ctx.settings)
        ball = make_ball_detector(spec)
        ball.prepare(info, bcrop)
        ranges = chunk_ranges(info.duration_s, p.chunk_seconds)
        tracker = ctx.chunks(len(ranges))
        det_dir, frames_dir = tracker.parts_dir / "det", tracker.parts_dir / "frames"
        bdir, bfdir = tracker.parts_dir / "ball", tracker.parts_dir / "ball_frames"
        source = open_source(src_path, info, p.decode_backend)
        table = source.table
        times = table.times()
        n = len(times)
        ball_targets = target_frames(table, [Window(0.0, info.duration_s + 1.0, p.ball_sweep_hz)])
        is_ball_target, is_ball_needed = target_mask(ball_targets, ball.context, n)
        ctx.log.info(
            "Detecting {}people with {} @ {} px, {} Hz, crop {}; ball with {} at {}, crop {}; "
            "decoder {}",
            "" if do_persons else "(kept) ", p.person_model, p.person_input_px,
            p.person_rate_hz, crop.as_list(), spec, p.ball_sweep_hz or "every frame",
            bcrop.as_list(), source.backend,
        )  # fmt: skip
        weights = torch.tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
        view = ViewCheck(
            session,
            calib.load(session.calibration_path),
            info.display_width,
            info.display_height,
            str(source.device),
        )

        def flush(batch, det_rows, frame_rows) -> None:
            imgs = [f.image for f in batch]
            boxes = detector.detect(imgs, crop)
            small = torch.stack([im[:, ::VIEW_STRIDE, ::VIEW_STRIDE] for im in imgs]).float()
            luma = (small * weights.to(small.device)).sum(1).mean((1, 2)).cpu().numpy()
            sim = view.scores(small, [f.t_s for f in batch])
            for f, b, lu, vs in zip(batch, boxes, luma, sim, strict=True):
                frame_rows.append((f.index, f.t_s, float(lu), float(vs)))
                for x0, y0, x1, y1, c in b.tolist():
                    det_rows.append((f.index, f.t_s, x0, y0, x1, y1, c))

        def report(i: int, t: float, a: float, b: float) -> None:
            ctx.check_cancel()
            ctx.progress(
                (i + (t - a) / max(b - a, 1e-6)) / len(ranges), f"Chunk {i + 1}/{len(ranges)}"
            )

        ext = (ball.context + 1) / max(info.fps_avg, 1.0)
        with source:
            for i, (a, b) in enumerate(ranges):
                if tracker.is_done(i):
                    continue
                ia, ib = table.index_at(a), table.index_at(b)
                person = np.zeros(n, dtype=bool)
                if do_persons:
                    person[target_frames(table, [Window(a, b, p.person_rate_hz)])] = True
                chunk_target = np.zeros(n, dtype=bool)
                chunk_target[ia:ib] = is_ball_target[ia:ib]
                needed = np.zeros(n, dtype=bool)
                lo, hi = max(0, ia - ball.context), min(n, ib + ball.context)
                needed[lo:hi] = is_ball_needed[lo:hi]
                stream = StreamDetector(ball, chunk_target, times)

                def keep(t, person=person, needed=needed):
                    k = table.index_at(t - 1e-6)
                    return bool(k < n and (person[k] or needed[k]))

                det_rows: list[tuple] = []
                frame_rows: list[tuple] = []
                batch = []
                for f in source.frames(max(0.0, a - ext), b + ext, keep):
                    if needed[f.index]:
                        stream.push(f)
                    if person[f.index]:
                        batch.append(f)
                        if len(batch) == BATCH:
                            flush(batch, det_rows, frame_rows)
                            batch = []
                            report(i, f.t_s, a, b)
                    elif f.index % 64 == 0:
                        report(i, f.t_s, a, b)
                if batch:
                    flush(batch, det_rows, frame_rows)
                stream.finish()
                cand, bframes = stream.take()
                if do_persons:
                    tables.write_part(
                        _rows(det_rows, PERSON_DETECTIONS), det_dir, i, PERSON_DETECTIONS
                    )
                    tables.write_part(_rows(frame_rows, PASS1_FRAMES), frames_dir, i, PASS1_FRAMES)
                tables.write_part(cand, bdir, i, BALL_CANDIDATES)
                tables.write_part(bframes, bfdir, i, BALL_FRAMES)
                tracker.mark_done(i)
                ctx.progress((i + 1) / len(ranges), f"Chunk {i + 1}/{len(ranges)}")
        session.players_dir.mkdir(parents=True, exist_ok=True)
        session.pass1_frames_path.parent.mkdir(parents=True, exist_ok=True)
        session.ball_dir.mkdir(parents=True, exist_ok=True)
        if do_persons:
            frames = tables.consolidate_parts(
                frames_dir, session.pass1_frames_path, PASS1_FRAMES, sort_by="t_s"
            )
            dets = tables.consolidate_parts(
                det_dir, session.person_detections_path, PERSON_DETECTIONS, sort_by="t_s"
            )
            atomic_write_json(_persons_state_path(session), {"hash": persons_hash})
        else:
            frames = tables.read_table(session.pass1_frames_path)
            dets = tables.read_table(session.person_detections_path)
        cand = tables.consolidate_parts(bdir, session.ball_sweep_path, BALL_CANDIDATES, "t_s")
        bframes = tables.consolidate_parts(
            bfdir, session.ball_sweep_frames_path, BALL_FRAMES, sort_by="t_s"
        )
        tracker.cleanup()
        return {
            "frames": frames.num_rows,
            "detections": dets.num_rows,
            "persons_rerun": do_persons,
            "ball_detector": spec,
            "ball_frames": bframes.num_rows,
            "ball_candidates": cand.num_rows,
            "decoder": source.backend,
            "crop": crop.as_list(),
            "ball_crop": bcrop.as_list(),
        }


def _persons_state_path(session) -> Path:
    return session.pass1_frames_path.parent / "persons.json"


def _persons_fresh(session, persons_hash: str) -> bool:
    path = _persons_state_path(session)
    if not (
        path.exists()
        and session.person_detections_path.exists()
        and session.pass1_frames_path.exists()
    ):
        return False
    try:
        return read_json(path).get("hash") == persons_hash
    except (OSError, ValueError):
        return False


def _migrate_persons_state(session, persons_cfg: dict) -> None:
    """Sessions detected before the ball sweep existed (``pass1_detect`` v1) had only the
    persons part, fingerprinted by its config hash: remember it so they keep their boxes."""
    path = _persons_state_path(session)
    if path.exists():
        return
    old = read_manifest(session, "pass1_detect")
    if old and old.get("version") == 1 and old.get("config_hash") == stable_hash(persons_cfg):
        atomic_write_json(path, {"hash": old["config_hash"]})


def _rows(rows: list[tuple], schema: pa.Schema) -> pa.Table:
    if not rows:
        return schema.empty_table()
    cols = list(zip(*rows, strict=True))
    return pa.table(
        {f.name: pa.array(c, f.type) for f, c in zip(schema, cols, strict=True)}, schema=schema
    )


def player_height(settings, config) -> float | None:
    """The "me" profile's height, if the session has a profile with one."""
    pid = config.players.me_profile_id
    if not pid or settings.output_root is None:
        return None
    row = Library(settings.output_root).get_profile(pid)
    return float(row["height_m"]) if row and row["height_m"] else None


def track_params(settings, config) -> tr.TrackParams:
    p = settings.processing
    height = player_height(settings, config)
    return tr.TrackParams(
        roi_behind_m=p.roi_behind_m,
        roi_beside_m=p.roi_beside_m,
        **({"person_height_m": height} if height else {}),
    )


def _period(frames: pa.Table, settings) -> float:
    t = frames.column("t_s").to_numpy()
    if len(t) > 1:
        return float(np.median(np.diff(t)))
    return 1.0 / settings.processing.person_rate_hz


class PlayersTrackStage(Stage):
    name = "players_track"
    title = "Player tracking"
    version = 1
    depends_on = ("pass1_detect", "camera")
    weight = 0.5

    def config(self, session, config, settings):
        practice = config.practice
        return {
            "params": track_params(settings, config).as_config(),
            "machine_xy": practice.machine_xy if practice else None,
            "auto_machine": bool(practice and practice.submode == "ball_machine"),
            "view_min": settings.processing.view_min,
        }

    def outputs(self, session):
        return [session.player_tracks_path, session.players_summary_path]

    def run(self, ctx: StageContext):
        session = ctx.session
        cal = calib.load(session.calibration_path)
        if cal is None:
            raise RuntimeError("No calibration: the camera stage must run first")
        det = tables.read_table(session.person_detections_path)
        frames = tables.read_table(session.pass1_frames_path)
        practice = ctx.config.practice
        view = frames.column("view").to_numpy()
        off_view = frames.column("frame").to_numpy()[view < ctx.settings.processing.view_min]
        ctx.progress(0.1, "Tracking")
        table, summary = tr.track_players(
            det,
            cal,
            cal.camera.width,
            cal.camera.height,
            _period(frames, ctx.settings),
            track_params(ctx.settings, ctx.config),
            machine_xy=practice.machine_xy if practice else None,
            auto_machine=bool(practice and practice.submode == "ball_machine"),
            ignore_frames=off_view,
        )
        tables.write_table(table, session.player_tracks_path, PLAYER_TRACKS)
        atomic_write_text(session.players_summary_path, summary.model_dump_json(indent=2))
        return {
            "detections": summary.n_detections,
            "in_roi": summary.n_in_roi,
            "tracklets": summary.n_tracklets,
            "me_detections": summary.me_detections,
            "static_objects": len(summary.static_objects),
            "machine": summary.machine.model_dump() if summary.machine else None,
        }


def load_players_summary(session) -> PlayersSummary | None:
    path = session.players_summary_path
    return PlayersSummary.model_validate(read_json(path)) if path.exists() else None


class MovementStage(Stage):
    name = "movement"
    title = "Movement"
    version = 1
    depends_on = ("players_track",)
    weight = 0.5

    def config(self, session, config, settings):
        p = settings.processing
        return {
            "params": mv.MovementParams().as_config(),
            "dark_luma": p.dark_luma,
            "view_min": p.view_min,
        }

    def outputs(self, session):
        return [session.movement_path]

    def run(self, ctx: StageContext):
        session = ctx.session
        tracks = tables.read_table(session.player_tracks_path)
        frames = tables.read_table(session.pass1_frames_path)
        params = mv.MovementParams()
        ctx.progress(0.1, "Smoothing")
        movement = mv.compute_movement(tracks, frames, params)
        tables.write_table(movement, session.movement_path, MOVEMENT)
        p = ctx.settings.processing
        stats = mv.summarize(movement, frames, p.dark_luma, params, p.view_min)
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}
