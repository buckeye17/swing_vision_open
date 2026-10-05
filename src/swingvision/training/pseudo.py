"""Pseudo-labels: more training data from confident detections (self-training, PLAN.md §10).

Hand labels are expensive; a ball in free flight is not. Sample clips are created at hitting
moments (strong audio onsets), the current detector + tracker runs over them, and only the
points of *confident* tracklets become labels (``src="assisted"``):

* long (≥ ``min_points`` detections) and smooth (RMS off a local quadratic ≤ ``max_rough_px``),
* clearly moving (90th-percentile image speed ≥ ``min_speed_px_s``: not a ball lying on the
  court or a slowly swinging racket),
* outside the player's box (rackets and limbs are where detectors are least reliable),
* yellow-green at the labeled spot.

The resulting clips have ``kind="sample"`` and ``split="train"``; they are never scored. Clips
within ``min_gap_s`` of a test clip are skipped so test footage doesn't leak into training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from swingvision.training import clipcache as cc
from swingvision.training.labels import BallClip, BallLabel, LabelStore


@dataclass(frozen=True)
class PseudoParams:
    min_points: int = 12
    max_rough_px: float = 2.0
    min_speed_px_s: float = 150.0
    min_yellow: float = 15.0
    clip_s: float = 2.5
    min_gap_s: float = 10.0


def _yellowness(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape[:2]
    xi, yi = round(x), round(y)
    if not (0 <= xi < w and 0 <= yi < h):
        return -1e9
    win = img[max(0, yi - 3) : yi + 4, max(0, xi - 3) : xi + 4].reshape(-1, 3).astype(np.int16)
    bg = img[max(0, yi - 25) : yi + 26, max(0, xi - 25) : xi + 26].reshape(-1, 3).astype(np.int16)
    yel = np.minimum(win[:, 0], win[:, 1]) - win[:, 2]
    ybg = np.median(np.minimum(bg[:, 0], bg[:, 1]) - bg[:, 2])
    return float(yel.max() - ybg)


def label_clip(
    store: LabelStore, clip: BallClip, session, config, settings, detector: str, p: PseudoParams
) -> int:
    """Detect + link over the clip and keep confident tracklet points as labels."""
    from pathlib import Path

    from swingvision.ball.detect import ball_crop, run_detector
    from swingvision.ball.detectors import make_ball_detector
    from swingvision.ball.schedule import Window, target_frames
    from swingvision.ball.trajectory import REF_WIDTH, PlayerBoxes, link
    from swingvision.court import calibration as calib
    from swingvision.io.frames import open_source
    from swingvision.storage import tables

    cal = calib.load(session.calibration_path)
    if cal is None or config.video is None:
        return 0
    pp = settings.processing
    det = make_ball_detector(detector)
    det.prepare(config.video, ball_crop(cal, pp.roi_behind_m, pp.roi_beside_m))
    with open_source(Path(config.source.path), config.video, pp.decode_backend) as src:
        times = src.table.times()
        t0 = max(0.0, float(times[clip.frame0]) - 0.5)
        t1 = float(times[min(clip.frame1, len(times) - 1)]) + 0.5
        res = run_detector(src, det, target_frames(src.table, [Window(t0, t1)]))
    boxes = None
    if session.movement_path.exists():
        boxes = PlayerBoxes.from_movement(tables.read_table(session.movement_path))
    width = config.video.display_width
    scale = width / REF_WIDTH
    _, tracklets = link(res.candidates, res.frames, width, player_boxes=boxes)
    n = 0
    for tr in tracklets:
        if len(tr) < p.min_points or tr.roughness() / scale > p.max_rough_px:
            continue
        if np.percentile(tr.speed(), 90) / scale < p.min_speed_px_s:
            continue
        inside = boxes.contains(tr.t, tr.x, tr.y) if boxes is not None else np.zeros(len(tr), bool)
        for f, x, y, ins in zip(tr.frame, tr.x, tr.y, inside, strict=True):
            f = int(f)
            if ins or not clip.frame0 <= f <= clip.frame1 or clip.label(f) is not None:
                continue
            img = cc.read_frame(store, clip, f)
            if img is None or _yellowness(img, x, y) < p.min_yellow:
                continue
            clip.set(
                f,
                BallLabel(
                    vis="visible", x=round(float(x), 2), y=round(float(y), 2), src="assisted"
                ),
            )
            n += 1
    clip.status = "done"
    store.save(clip)
    return n


def make_pseudo_clips(
    store: LabelStore,
    settings,
    session_id: str,
    n_clips: int,
    detector: str,
    params: PseudoParams | None = None,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[tuple[str, int]]:
    from swingvision import services
    from swingvision.io.frames import frame_table
    from swingvision.storage import tables

    p = params or PseudoParams()
    session = services.session_by_id(settings, session_id)
    if session is None:
        raise KeyError(session_id)
    config = session.load_config()
    on = tables.read_table(session.audio_onsets_path)
    t_on = on.column("t_s").to_numpy()[on.column("strength").to_numpy() >= 20]
    tests = [c.t0_s for c in store.list_clips(session_id) if c.split == "test"]
    taken = [c.t0_s for c in store.list_clips(session_id)]
    rng = np.random.default_rng(seed)
    rng.shuffle(t_on)
    tab = frame_table(config.source.path)
    times = tab.times()
    out = []
    for t in t_on:
        if len(out) >= n_clips:
            break
        start = float(t) - 0.6
        if any(abs(start - x) < p.min_gap_s for x in tests) or any(
            abs(start - x) < 3.0 for x in taken
        ):
            continue
        f0 = tab.index_at(start)
        f1 = min(len(times) - 1, tab.index_at(start + p.clip_s) - 1)
        clip = store.new_clip(
            session_id, f0, f1, float(times[f0]), float(times[f1]), kind="sample", split="train"
        )
        cc.ensure_cached(store, clip, config, settings.processing.decode_backend)
        n = label_clip(store, clip, session, config, settings, detector, p)
        taken.append(start)
        out.append((clip.clip_id, n))
        log(f"{session_id}/{clip.clip_id} at {start:.1f}s: {n} pseudo-labels")
    return out
