"""Metrics against labeled ground truth (PLAN.md §7.4.1, §10).

Ball detection is scored per frame on ground-truth clips:

* a frame whose label is ``visible`` and has a prediction within the tolerance is a true
  positive; a prediction further away counts as both a false positive and a false negative;
  a missing prediction is a false negative;
* a frame labeled ``none`` with a prediction is a false positive;
* ``occluded`` and unlabeled frames are not scored.

**Tolerance.** PLAN.md asks for 4 px. A ball close to the camera is 20-40 px wide, so the
tolerance grows with the ball's apparent size: ``max(4 px, 0.4 × ball diameter)``. A fast
ball is a motion-blur streak up to its per-frame displacement long, and its label (the
streak's center) is uncertain along it: the default (*motion-aware*) tolerance is also at
least ``0.25 ×`` the displacement to the neighboring labeled frames. The strict variant
(size only) is reported alongside. The diameter is estimated from the distance to the point where the label's
viewing ray meets the ground, which is *beyond* a ball in the air, so the estimate is a lower
bound and the tolerance errs on the strict side. At the far baseline it is 4 px.

**Near / far.** A label whose viewing ray meets the ground on the far half (``y > 0``) is
*far*; everything else (near half, or above the horizon) is *near*.

Events (hits, bounces) match within ±2 frames, one-to-one, nearest first. Bounce positions
are compared on the ground, in meters, after both are refined the same way
(:func:`swingvision.ball.events.refine_contact`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np

from swingvision.court.camera import Camera
from swingvision.training.labels import BallClip

BALL_DIAMETER_M = 0.067
MIN_TOL_PX = 4.0
TOL_FRACTION = 0.4
MOTION_FRACTION = 0.25
EVENT_TOL_FRAMES = 2


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    errors: list[float] = field(default_factory=list)

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.errors += other.errors

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r and np.isfinite(p + r) else float("nan")

    def summary(self) -> dict:
        e = np.array(self.errors) if self.errors else np.array([np.nan])
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "median_err_px": round(float(np.nanmedian(e)), 2),
        }


def ground_info(cam: Camera, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For image points: (is_far, ball diameter estimate in px)."""
    px = np.atleast_2d(np.asarray(px, dtype=np.float64))
    g = cam.image_to_ground(px, 0.0)
    ok = np.isfinite(g).all(axis=1)
    far = np.zeros(len(px), dtype=bool)
    diam = np.full(len(px), np.nan)
    if ok.any():
        gg = g[ok]
        far[ok] = gg[:, 1] > 0
        X = np.column_stack([gg[:, 0], gg[:, 1], np.zeros(len(gg))])
        depth = cam.depth(X)
        good = depth > 0.5
        d = np.full(len(gg), np.nan)
        d[good] = cam.f * BALL_DIAMETER_M / depth[good]
        diam[ok] = d
    # Above the horizon (or behind): treat as near; diameter unknown → minimum tolerance.
    return far, diam


def tolerance(diam_px: np.ndarray) -> np.ndarray:
    d = np.where(np.isfinite(diam_px), diam_px, 0.0)
    return np.maximum(MIN_TOL_PX, TOL_FRACTION * d)


@dataclass
class BallScore:
    all: Counts = field(default_factory=Counts)
    near: Counts = field(default_factory=Counts)
    far: Counts = field(default_factory=Counts)
    none_frames: int = 0  # frames labeled "none" (where any prediction is a false positive)
    per_clip: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "all": self.all.summary(),
            "near": self.near.summary(),
            "far": self.far.summary(),
            "none_frames": self.none_frames,
        }


def motion_px(clip: BallClip, frames: list[int]) -> np.ndarray:
    """Largest image displacement to a neighboring labeled frame (px per frame)."""
    out = np.zeros(len(frames))
    for k, f in enumerate(frames):
        lab = clip.label(f)
        for g in (f - 1, f + 1):
            nb = clip.label(g)
            if nb is not None and nb.vis == "visible":
                out[k] = max(out[k], float(np.hypot(nb.x - lab.x, nb.y - lab.y)))
    return out


def score_clip(
    clip: BallClip,
    pred: dict[int, tuple[float, float]],
    cam: Camera,
    motion_aware: bool = True,
) -> tuple[Counts, Counts]:
    """(near, far) counts for one clip. ``pred``: frame → (x, y).

    ``motion_aware``: a ball moving ``v`` px per frame is a motion-blur streak whose center
    (the label) is uncertain along the streak, so the tolerance is at least ``MOTION_FRACTION
    · v`` (on top of the size-based one).
    """
    near, far = Counts(), Counts()
    frames = [f for f in clip.frames if clip.label(f) is not None]
    vis = [f for f in frames if clip.label(f).vis == "visible"]
    if vis:
        px = np.array([[clip.label(f).x, clip.label(f).y] for f in vis])
        is_far, diam = ground_info(cam, px)
        tol = tolerance(diam)
        if motion_aware:
            tol = np.maximum(tol, MOTION_FRACTION * motion_px(clip, vis))
    for f in frames:
        lab = clip.label(f)
        p = pred.get(f)
        if lab.vis == "occluded":
            continue
        if lab.vis == "none":
            if p is not None:
                # Attribute to the half where the false detection is.
                fa, _ = ground_info(cam, np.array([p]))
                (far if fa[0] else near).fp += 1
            continue
        k = vis.index(f)
        c = far if is_far[k] else near
        if p is None:
            c.fn += 1
            continue
        err = float(np.hypot(p[0] - lab.x, p[1] - lab.y))
        if err <= tol[k]:
            c.tp += 1
            c.errors.append(err)
        else:
            c.fp += 1
            c.fn += 1
    return near, far


def score_ball(
    clips: Iterable[BallClip],
    preds: dict[str, dict[int, tuple[float, float]]],
    cameras: dict[str, Camera],
    motion_aware: bool = True,
) -> BallScore:
    """``preds``: clip key (``session_id/clip_id``) → frame → (x, y)."""
    out = BallScore()
    for clip in clips:
        key = f"{clip.session_id}/{clip.clip_id}"
        near, far = score_clip(clip, preds.get(key, {}), cameras[clip.session_id], motion_aware)
        out.near.add(near)
        out.far.add(far)
        both = Counts()
        both.add(near)
        both.add(far)
        out.all.add(both)
        out.none_frames += sum(
            1 for f in clip.frames if (lab := clip.label(f)) is not None and lab.vis == "none"
        )
        out.per_clip[key] = both.summary()
    return out


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class EventScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    frame_errors: list[int] = field(default_factory=list)
    pos_errors_near: list[float] = field(default_factory=list)
    pos_errors_far: list[float] = field(default_factory=list)

    @property
    def f1(self) -> float:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else float("nan")

    def summary(self) -> dict:
        def med(v):
            return round(float(np.median(v)), 3) if v else None

        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")
        return {
            **{k: v for k, v in asdict(self).items() if isinstance(v, int)},
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(self.f1, 4),
            "median_frame_err": med([abs(e) for e in self.frame_errors]),
            "median_pos_err_near_m": med(self.pos_errors_near),
            "median_pos_err_far_m": med(self.pos_errors_far),
            "n_pos_near": len(self.pos_errors_near),
            "n_pos_far": len(self.pos_errors_far),
        }


def match_events(
    gt_frames: list[int], pred_frames: list[int], tol: int = EVENT_TOL_FRAMES
) -> list[tuple[int, int]]:
    """One-to-one matches (gt index, pred index), closest pairs first."""
    pairs = sorted(
        (abs(g - p), i, j)
        for i, g in enumerate(gt_frames)
        for j, p in enumerate(pred_frames)
        if abs(g - p) <= tol
    )
    used_g, used_p, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j))
    return out
