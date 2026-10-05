"""Assisted labeling: model predictions for a clip, and click snapping (PLAN.md §10).

* :func:`predict_clip` runs a ball detector and the trajectory linker over a clip (with some
  margin on both sides, so tracks entering the clip are linked) and stores the predicted ball
  position per frame next to the clip's frame cache. The labeling page shows them as hollow
  circles; *Accept* (Enter) turns one into a label.
* :func:`snap` moves a click onto the nearest moving blob, so labels land on the ball center
  even when the click is a few pixels off.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from swingvision.storage.fsutil import atomic_write_json
from swingvision.storage.schemas import SessionConfig
from swingvision.training.labels import BallClip, BallLabel, LabelStore

MARGIN_S = 0.75


def predictions_path(store: LabelStore, clip: BallClip) -> Path:
    return store.cache_dir(clip.session_id, clip.clip_id) / "predictions.json"


def load_predictions(store: LabelStore, clip: BallClip) -> dict[int, tuple[float, float, float]]:
    p = predictions_path(store, clip)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): tuple(v) for k, v in data.get("points", {}).items()}


def predict_clip(
    store: LabelStore,
    clip: BallClip,
    session,
    config: SessionConfig,
    settings,
    detector: str | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Detect + link around the clip; returns (and saves) frame → (x, y, score).

    ``detector``: a spec; default the one configured in Settings (``auto``: the newest
    trained model)."""
    from swingvision.ball.detect import ball_crop, run_detector
    from swingvision.ball.detectors import make_ball_detector, resolve_spec
    from swingvision.ball.schedule import Window, target_frames
    from swingvision.ball.trajectory import PlayerBoxes, link
    from swingvision.court import calibration as calib
    from swingvision.io.frames import open_source
    from swingvision.storage import tables

    detector = resolve_spec(detector or settings.processing.ball_detector, settings.output_root)
    cal = calib.load(session.calibration_path)
    if cal is None:
        raise RuntimeError("The session has no calibration yet")
    assert config.video is not None
    p = settings.processing
    crop = ball_crop(cal, p.roi_behind_m, p.roi_beside_m)
    det = make_ball_detector(detector)
    det.prepare(config.video, crop)
    src_path = Path(config.source.path)
    with open_source(src_path, config.video, p.decode_backend) as src:
        times = src.table.times()
        t0 = max(0.0, float(times[clip.frame0]) - MARGIN_S)
        t1 = min(float(times[-1]), float(times[min(clip.frame1, len(times) - 1)]) + MARGIN_S)
        targets = target_frames(src.table, [Window(t0, t1)])
        res = run_detector(src, det, targets)
    boxes = None
    if session.movement_path.exists():
        boxes = PlayerBoxes.from_movement(tables.read_table(session.movement_path))
    track, _ = link(res.candidates, res.frames, config.video.display_width, player_boxes=boxes)
    pts = {
        int(f): (round(float(x), 2), round(float(y), 2), round(float(s), 3))
        for f, x, y, s in zip(
            track.column("frame").to_numpy(),
            track.column("x").to_numpy(),
            track.column("y").to_numpy(),
            track.column("score").to_numpy(),
            strict=True,
        )
        if clip.frame0 <= f <= clip.frame1
    }
    atomic_write_json(
        predictions_path(store, clip),
        {"detector": detector, "points": {str(k): list(v) for k, v in pts.items()}},
    )
    return pts


def apply_predictions(
    clip: BallClip, preds: dict[int, tuple[float, float, float]], overwrite: bool = False
) -> int:
    """Turn predictions into ``assisted`` labels on frames that have none. Returns the count."""
    n = 0
    for f, (x, y, _) in preds.items():
        if not clip.frame0 <= f <= clip.frame1:
            continue
        if clip.label(f) is not None and not overwrite:
            continue
        clip.set(f, BallLabel(vis="visible", x=x, y=y, src="assisted"))
        n += 1
    return n


def snap(
    prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray, x: float, y: float, radius: int = 10
) -> tuple[float, float] | None:
    """Center of the nearest moving blob within ``radius`` px of (x, y), or None.

    ``prev``/``nxt``: full-resolution RGB frames a few frames before/after ``cur``
    (:data:`swingvision.training.clipcache.CONTEXT`).
    """
    import torch

    from swingvision.ball.detectors.motion import detect_blobs

    h, w = cur.shape[:2]
    pad = radius + 48
    x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
    x1, y1 = min(w, int(x) + pad + 1), min(h, int(y) + pad + 1)

    def t(a):
        return (
            torch.from_numpy(np.ascontiguousarray(a[y0:y1, x0:x1])).permute(2, 0, 1)[None].float()
        )

    blobs = detect_blobs(t(prev), t(cur), t(nxt), 16, 2.0)[0]
    if not len(blobs):
        return None
    d = np.hypot(blobs[:, 0] + x0 - x, blobs[:, 1] + y0 - y)
    j = int(np.argmin(d))
    if d[j] > radius:
        return None
    return float(blobs[j, 0] + x0), float(blobs[j, 1] + y0)
