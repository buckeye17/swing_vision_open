"""Ball trajectory linking: candidates → one cleaned ball track (PLAN.md §7.4).

Detectors return several candidates per frame (the ball, but also rackets, limbs, leaves,
balls rolling elsewhere). Linking works in three steps, all in image space:

1. **Tracklets.** Seeds are candidate triplets on consecutive processed frames that move at a
   plausible, nearly constant velocity (``|c - (2b - a)|`` small). Seeds are grown forwards
   and backwards in order of quality: a quadratic fit over the last few points predicts the
   next position and the nearest unclaimed candidate inside a gate continues the tracklet,
   which may skip a few frames. Hits and bounces break the motion, so tracklets naturally end
   there.
2. **Selection.** Several balls can move at once (a ball rolling to the fence while the next
   one is fed). The *active* ball is chosen per frame with a Viterbi pass over the tracklets
   alive at each frame: detected points earn their confidence times a "flying" credit (from
   the tracklet's speed; a slow tracklet continuing a flight inherits it) and a smoothness
   factor; candidates inside the player's box count less (rackets, limbs); jumping to an
   unrelated tracklet costs more than a hand-off where one tracklet ends and another starts
   nearby (a hit or a bounce).
3. **Cleaning.** Points far off a local quadratic fit are dropped, and gaps of up to
   ``max_fill`` frames inside a tracklet or across a hand-off are filled from the fits on
   both sides (``source = interp``).

Pixel tolerances are given for 3840-pixel-wide frames and scaled to the actual width.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.storage.schemas import BALL_TRACK

REF_WIDTH = 3840.0


@dataclass(frozen=True)
class LinkParams:
    #: Fastest image motion considered (px/s at 3840 px wide; ~150 px per frame at 60 fps).
    vmax_px_s: float = 9000.0
    #: Image acceleration allowed between samples without an event (gravity, drag, perspective).
    accel_px_s2: float = 4000.0
    #: Prediction gate: base + rel · step length + accel · dt².
    gate_base_px: float = 6.0
    gate_rel: float = 0.15
    min_score: float = 0.2
    #: Longest stretch without a detection a tracklet may coast through (s).
    max_coast_s: float = 0.09
    #: Processed frames further apart than this are not linked directly (s).
    max_step_s: float = 0.35
    min_points: int = 4
    #: Viterbi rewards/costs.
    frame_cost: float = 0.3  # paid per frame on a tracklet (short junk doesn't pay off)
    coast_reward: float = 0.0
    #: A tracklet whose 90th-percentile speed is below this (px/s) is a ball (nearly) at rest
    #: or rolling: it earns nothing, full reward from twice this speed.
    slow_px_s: float = 60.0
    handoff_cost: float = 0.3
    jump_cost: float = 2.0
    start_cost: float = 0.5
    handoff_px: float = 60.0
    #: Gaps up to this long (s) are filled by interpolation.
    max_fill_s: float = 0.15
    outlier_px: float = 6.0
    #: Candidates inside the player's box (limbs, racket, clothes) keep this share of their
    #: score; the ball there is usually also visible outside the box before or after.
    in_player_factor: float = 0.4
    #: Tracklets wigglier than this (RMS off a sliding quadratic, px) earn less: limbs
    #: and cloth move smoothly only for a few frames, a ball in flight for many.
    smooth_px: float = 3.0

    def as_config(self) -> dict:
        return asdict(self)


@dataclass
class Tracklet:
    id: int
    t: np.ndarray  # detection times (s)
    frame: np.ndarray
    x: np.ndarray
    y: np.ndarray
    score: np.ndarray

    @property
    def t0(self) -> float:
        return float(self.t[0])

    @property
    def t1(self) -> float:
        return float(self.t[-1])

    def __len__(self) -> int:
        return len(self.t)

    def position(self, t: float | np.ndarray) -> np.ndarray:
        """Position at time(s) ``t`` from a local quadratic fit (extrapolates a little)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.empty((len(t), 2))
        for i, ti in enumerate(t):
            k = int(np.clip(np.searchsorted(self.t, ti), 0, len(self.t)))
            lo, hi = max(0, k - 3), min(len(self.t), k + 3)
            out[i] = _fit_eval(self.t[lo:hi], self.x[lo:hi], self.y[lo:hi], ti)
        return out

    def roughness(self) -> float:
        """RMS distance of points from a quadratic through their ±3 neighbors (px)."""
        n = len(self.t)
        if n < 5:
            return 0.0
        res = []
        for j in range(n):
            lo, hi = max(0, j - 3), min(n, j + 4)
            m = np.ones(hi - lo, dtype=bool)
            m[j - lo] = False
            if m.sum() < 4:
                continue
            pred = _fit_eval(self.t[lo:hi][m], self.x[lo:hi][m], self.y[lo:hi][m], self.t[j])
            res.append(np.hypot(self.x[j] - pred[0], self.y[j] - pred[1]))
        return float(np.sqrt(np.mean(np.square(res)))) if res else 0.0

    def speed(self) -> np.ndarray:
        """Per-point speed (px/s) from neighbors."""
        if len(self.t) < 2:
            return np.zeros(len(self.t))
        dx = np.gradient(self.x, self.t)
        dy = np.gradient(self.y, self.t)
        return np.hypot(dx, dy)


def _fit_eval(t: np.ndarray, x: np.ndarray, y: np.ndarray, at: float) -> np.ndarray:
    n = len(t)
    if n == 1:
        return np.array([x[0], y[0]])
    deg = 2 if n >= 4 else 1
    tc = t - t.mean()
    A = np.vander(tc, deg + 1)
    cx, *_ = np.linalg.lstsq(A, x, rcond=None)
    cy, *_ = np.linalg.lstsq(A, y, rcond=None)
    v = np.vander(np.array([at - t.mean()]), deg + 1)[0]
    return np.array([v @ cx, v @ cy])


@dataclass
class PlayerBoxes:
    """Player boxes over time (e.g. the tracked player from ``movement``), px."""

    t: np.ndarray
    box: np.ndarray  # (n, 4) x0, y0, x1, y1
    max_dt: float = 0.1
    pad: float = 0.05

    @classmethod
    def from_movement(cls, movement: pa.Table, player: str = "me") -> PlayerBoxes:
        m = movement.filter(pc.equal(movement.column("player"), player))
        box = np.column_stack(
            [m.column(c).to_numpy(zero_copy_only=False) for c in ("bx0", "by0", "bx1", "by1")]
        )
        ok = np.isfinite(box).all(axis=1)
        return cls(m.column("t_s").to_numpy()[ok], box[ok].astype(np.float64))

    def contains(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not len(self.t):
            return np.zeros(len(t), dtype=bool)
        k = np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1)
        k = np.where(np.abs(self.t[k - 1] - t) < np.abs(self.t[k] - t), k - 1, k)
        near = np.abs(self.t[k] - t) <= self.max_dt
        b = self.box[k]
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        px, py = self.pad * w, self.pad * h
        inside = (
            (x >= b[:, 0] - px) & (x <= b[:, 2] + px) & (y >= b[:, 1] - py) & (y <= b[:, 3] + py)
        )
        return near & inside


class _Frames:
    """Candidates grouped by processed frame."""

    def __init__(
        self,
        cand: pa.Table,
        frames: pa.Table,
        min_score: float,
        player_boxes: PlayerBoxes | None = None,
        in_player_factor: float = 1.0,
    ):
        f_all = frames.column("frame").to_numpy()
        t_all = frames.column("t_s").to_numpy()
        order = np.argsort(f_all)
        self.frame = f_all[order]
        self.t = t_all[order]
        cf = cand.column("frame").to_numpy()
        score = cand.column("score").to_numpy().astype(np.float64)
        cx = cand.column("x").to_numpy().astype(np.float64)
        cy = cand.column("y").to_numpy().astype(np.float64)
        if player_boxes is not None and in_player_factor < 1.0 and len(cf):
            inside = player_boxes.contains(cand.column("t_s").to_numpy(), cx, cy)
            score = np.where(inside, score * in_player_factor, score)
        keep = score >= min_score
        cf = cf[keep]
        o = np.argsort(cf, kind="stable")
        self.cf = cf[o]
        self.cx = cx[keep][o]
        self.cy = cy[keep][o]
        self.cs = score[keep][o]
        # candidate rows of processed frame k: start[k]:end[k]
        self.start = np.searchsorted(self.cf, self.frame, side="left")
        self.end = np.searchsorted(self.cf, self.frame, side="right")
        self.claimed = np.zeros(len(self.cf), dtype=bool)

    def __len__(self) -> int:
        return len(self.frame)

    def rows(self, k: int) -> np.ndarray:
        return np.arange(self.start[k], self.end[k])


def build_tracklets(fr: _Frames, p: LinkParams, scale: float) -> list[Tracklet]:
    """Seed on consistent triplets and grow, best seeds first (each candidate used once)."""
    seeds = []
    vmax = p.vmax_px_s * scale
    for k in range(1, len(fr) - 1):
        ta, tb, tc = fr.t[k - 1], fr.t[k], fr.t[k + 1]
        if tb - ta > p.max_step_s or tc - tb > p.max_step_s:
            continue
        ra, rb, rc = fr.rows(k - 1), fr.rows(k), fr.rows(k + 1)
        if not (len(ra) and len(rb) and len(rc)):
            continue
        dab = tb - ta
        dx = fr.cx[rb][:, None] - fr.cx[ra][None, :]
        dy = fr.cy[rb][:, None] - fr.cy[ra][None, :]
        ok = np.hypot(dx, dy) <= vmax * dab
        for ib, ia in zip(*np.nonzero(ok), strict=True):
            b, a = rb[ib], ra[ia]
            f = (tc - tb) / dab
            px = fr.cx[b] + (fr.cx[b] - fr.cx[a]) * f
            py = fr.cy[b] + (fr.cy[b] - fr.cy[a]) * f
            d = np.hypot(fr.cx[rc] - px, fr.cy[rc] - py)
            j = int(np.argmin(d))
            step = np.hypot(px - fr.cx[b], py - fr.cy[b])
            gate = scale * (p.gate_base_px + p.accel_px_s2 * (tc - ta) ** 2) + p.gate_rel * step
            if d[j] <= gate:
                q = fr.cs[a] + fr.cs[b] + fr.cs[rc[j]] - d[j] / gate
                seeds.append((q, k, a, b, rc[j]))
    seeds.sort(key=lambda s: -s[0])
    out: list[Tracklet] = []
    for _, k, a, b, c in seeds:
        if fr.claimed[a] or fr.claimed[b] or fr.claimed[c]:
            continue
        pts = [(k - 1, a), (k, b), (k + 1, c)]
        fr.claimed[[a, b, c]] = True
        fwd = _grow(fr, pts, +1, p, scale)
        bwd = _grow(fr, pts[::-1], -1, p, scale)
        pts = bwd[::-1][:-3] + pts + fwd[3:]
        if len(pts) < p.min_points:
            # Not worth keeping; free its candidates for other seeds.
            fr.claimed[[r for _, r in pts]] = False
            continue
        ks = np.array([kk for kk, _ in pts])
        rows = np.array([r for _, r in pts])
        out.append(
            Tracklet(len(out), fr.t[ks], fr.frame[ks], fr.cx[rows], fr.cy[rows], fr.cs[rows])
        )
    return out


def _grow(fr: _Frames, pts: list[tuple[int, int]], step: int, p: LinkParams, scale: float):
    pts = list(pts)
    k = k_last = pts[-1][0]
    while True:
        k += step
        if k < 0 or k >= len(fr) or abs(fr.t[k] - fr.t[k - step]) > p.max_step_s:
            break
        tk = fr.t[k]
        t_last = fr.t[k_last]
        dt = abs(tk - t_last)
        if dt > p.max_coast_s and k != k_last + step:
            break
        tail = pts[-5:]
        ts = fr.t[[q for q, _ in tail]]
        xs = fr.cx[[r for _, r in tail]]
        ys = fr.cy[[r for _, r in tail]]
        pred = _fit_eval(ts, xs, ys, tk)
        rows = fr.rows(k)
        rows = rows[~fr.claimed[rows]]
        if not len(rows):
            continue
        v = np.hypot(xs[-1] - xs[-2], ys[-1] - ys[-2]) / max(abs(ts[-1] - ts[-2]), 1e-6)
        gate = scale * (p.gate_base_px + p.accel_px_s2 * dt**2) + p.gate_rel * v * dt
        d = np.hypot(fr.cx[rows] - pred[0], fr.cy[rows] - pred[1])
        j = int(np.argmin(d))
        if d[j] <= gate:
            pts.append((k, int(rows[j])))
            fr.claimed[rows[j]] = True
            k_last = k
    return pts


# ---------------------------------------------------------------------------
# Selection (Viterbi over tracklets)
# ---------------------------------------------------------------------------


def flight_credit(tracklets: list[Tracklet], p: LinkParams, scale: float) -> np.ndarray:
    """0-1 per tracklet: is this a ball in flight (or the continuation of one)?

    From the tracklet's own speed (90th percentile of its image speed: a flight is slow near
    its apex but fast elsewhere, a ball lying or rolling on its own is slow throughout). A
    slow tracklet that starts where a flying one ended (a ball rolling out after its last
    bounce) inherits most of that credit.
    """
    own = np.array(
        [
            np.clip(
                (np.percentile(tr.speed(), 90) - p.slow_px_s * scale) / (p.slow_px_s * scale),
                0.0,
                1.0,
            )
            for tr in tracklets
        ]
    )
    credit = own.copy()
    order = np.argsort([tr.t0 for tr in tracklets])
    for i in order:
        tr = tracklets[i]
        for j in order:
            prev = tracklets[j]
            if j == i or credit[j] < 0.5 or not (-0.05 <= tr.t0 - prev.t1 <= p.max_step_s):
                continue
            end = prev.position(tr.t0)[0]
            if np.hypot(tr.x[0] - end[0], tr.y[0] - end[1]) <= p.handoff_px * scale:
                credit[i] = max(credit[i], 0.8 * credit[j])
    return credit


def select_active(
    tracklets: list[Tracklet], frame_t: np.ndarray, p: LinkParams, scale: float
) -> np.ndarray:
    """Per processed frame, the index of the active tracklet (or -1)."""
    n = len(frame_t)
    choice = np.full(n, -1, dtype=np.int64)
    if not tracklets:
        return choice
    moving = flight_credit(tracklets, p, scale)
    # Per tracklet: span over processed frames and per-frame rewards.
    spans = []
    rewards = []
    for tr, mv in zip(tracklets, moving, strict=True):
        k0 = int(np.searchsorted(frame_t, tr.t0 - 1e-6))
        k1 = int(np.searchsorted(frame_t, tr.t1 + 1e-6))
        ft = frame_t[k0:k1]
        r = np.full(len(ft), p.coast_reward - p.frame_cost)
        rough = tr.roughness() / scale
        smooth = float(np.exp(-max(0.0, rough - p.smooth_px) / p.smooth_px))
        idx = np.searchsorted(np.round(ft, 6), np.round(tr.t, 6))
        r[idx] = tr.score * mv * smooth - p.frame_cost
        spans.append((k0, k1))
        rewards.append(r)
    alive_at: list[list[int]] = [[] for _ in range(n)]
    for i, (k0, k1) in enumerate(spans):
        for k in range(k0, k1):
            alive_at[k].append(i)
    # Viterbi with state "none" = -1.
    prev_states = [-1]
    prev_score = np.array([0.0])
    back: list[tuple[list[int], np.ndarray]] = []
    for k in range(n):
        states = [-1, *alive_at[k]]
        score = np.full(len(states), -np.inf)
        arg = np.zeros(len(states), dtype=np.int64)
        for si, s in enumerate(states):
            emit = 0.0 if s < 0 else rewards[s][k - spans[s][0]]
            for pi, ps in enumerate(prev_states):
                c = (
                    prev_score[pi]
                    + emit
                    - _transition(ps, s, k, tracklets, spans, frame_t, p, scale)
                )
                if c > score[si]:
                    score[si], arg[si] = c, pi
        back.append((states, arg))
        prev_states, prev_score = states, score
    # Backtrack.
    si = int(np.argmax(prev_score))
    for k in range(n - 1, -1, -1):
        states, arg = back[k]
        choice[k] = states[si]
        si = int(arg[si])
    return choice


def _transition(ps, s, k, tracklets, spans, frame_t, p, scale) -> float:
    if ps == s:
        return 0.0
    if s < 0:
        return 0.0
    if ps < 0:
        return p.start_cost
    # Switching tracklets: a hand-off if s starts here near where ps is (or just ended).
    tr_s, tr_p = tracklets[s], tracklets[ps]
    if spans[s][0] == k:
        here = np.array([tr_s.x[0], tr_s.y[0]])
        there = tr_p.position(frame_t[k])[0]
        if np.hypot(*(here - there)) <= p.handoff_px * scale:
            return p.handoff_cost
    return p.jump_cost


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def link(
    cand: pa.Table,
    frames: pa.Table,
    width: int,
    params: LinkParams | None = None,
    all_frames: tuple[np.ndarray, np.ndarray] | None = None,
    player_boxes: PlayerBoxes | None = None,
) -> tuple[pa.Table, list[Tracklet]]:
    """Candidates → ball track (one row per frame with a ball) and all tracklets.

    ``all_frames``: (frame numbers, times) of every video frame in the processed span, used to
    fill short gaps (sweep schedules skip frames). Defaults to the processed frames.
    """
    p = params or LinkParams()
    scale = width / REF_WIDTH
    fr = _Frames(cand, frames, p.min_score, player_boxes, p.in_player_factor)
    tracklets = build_tracklets(fr, p, scale)
    choice = select_active(tracklets, fr.t, p, scale)
    rows = _assemble(tracklets, choice, fr, p, scale, all_frames)
    return rows, tracklets


def _assemble(tracklets, choice, fr: _Frames, p: LinkParams, scale: float, all_frames):
    if all_frames is None:
        all_f, all_t = fr.frame, fr.t
    else:
        all_f, all_t = all_frames
    # Detected points of the chosen tracklets, per processed frame.
    det: dict[int, tuple[float, float, float, int]] = {}
    for k, ti in enumerate(choice):
        if ti < 0:
            continue
        tr = tracklets[ti]
        j = np.nonzero(tr.frame == fr.frame[k])[0]
        if len(j):
            j = int(j[0])
            det[int(fr.frame[k])] = (tr.x[j], tr.y[j], tr.score[j], ti)
    # Outliers: points far from a quadratic through their neighbors in the same tracklet.
    for ti in set(int(c) for c in choice if c >= 0):
        tr = tracklets[ti]
        for j in range(len(tr)):
            f = int(tr.frame[j])
            if f not in det or det[f][3] != ti:
                continue
            lo, hi = max(0, j - 3), min(len(tr), j + 4)
            m = np.ones(hi - lo, dtype=bool)
            m[j - lo] = False
            if m.sum() < 4:
                continue
            pred = _fit_eval(tr.t[lo:hi][m], tr.x[lo:hi][m], tr.y[lo:hi][m], tr.t[j])
            if np.hypot(tr.x[j] - pred[0], tr.y[j] - pred[1]) > p.outlier_px * scale + 0.05 * (
                np.hypot(*np.diff(np.c_[tr.x[lo:hi], tr.y[lo:hi]], axis=0).T).mean()
            ):
                det.pop(f)
    # Rows: detected points, then fills.
    f_det = np.array(sorted(det), dtype=np.int64)
    rows = {f: (*det[f], "detected") for f in f_det}
    t_of = dict(zip(all_f.tolist(), all_t.tolist(), strict=True))
    t_of.update(zip(fr.frame.tolist(), fr.t.tolist(), strict=True))
    for a, b in itertools.pairwise(f_det):
        if b - a <= 1:
            continue
        ta, tb = t_of.get(int(a)), t_of.get(int(b))
        if ta is None or tb is None or tb - ta > p.max_fill_s:
            continue
        _xa, _ya, _sa, ia = det[int(a)]
        _xb, _yb, _sb, ib = det[int(b)]
        between = all_f[(all_f > a) & (all_f < b)]
        for f in between:
            t = t_of[int(f)]
            if ia == ib:
                xy = tracklets[ia].position(t)[0]
            else:
                # Across a hand-off (hit/bounce): each side's own fit, nearer side wins.
                side = ia if t - ta <= tb - t else ib
                xy = tracklets[side].position(t)[0]
            rows[int(f)] = (xy[0], xy[1], 0.0, ia if t - ta <= tb - t else ib, "interp")
    fs = np.array(sorted(rows), dtype=np.int64)
    if not len(fs):
        return BALL_TRACK.empty_table()
    vals = [rows[int(f)] for f in fs]
    return pa.table(
        {
            "frame": pa.array(fs, pa.int64()),
            "t_s": pa.array([t_of[int(f)] for f in fs], pa.float64()),
            "x": pa.array([v[0] for v in vals], pa.float32()),
            "y": pa.array([v[1] for v in vals], pa.float32()),
            "score": pa.array([v[2] for v in vals], pa.float32()),
            "tracklet": pa.array([v[3] for v in vals], pa.int32()),
            "source": pa.array([v[4] for v in vals], pa.string()),
        },
        schema=BALL_TRACK,
    )
