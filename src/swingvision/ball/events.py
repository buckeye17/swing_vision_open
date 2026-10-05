"""Ball events: hits, bounces, net (PLAN.md §7.5).

Candidates are the points where the ball's image motion breaks (*kinks*). At each point,
straight lines through the few points on each side (meeting there) are compared with one
quadratic through the whole window; a kink needs all of: the lines fit clearly better (a
smooth apex doesn't), a velocity change of several standard errors, and a relative velocity
change of at least 25%. Hand-offs between linked tracklets count when they also pass the
velocity tests.

Each kink is classified from geometry, motion and sound:

* **near the player** (inside the tracked player's box): a *bounce* when the contact is at
  the player's feet (bottom of the box) and the ball is kicked upwards, otherwise a *hit*;
* **inside the net's image region and nearly stopped**: *net* (a bounce right behind the
  net, seen through the mesh, keeps most of its speed);
* **elsewhere**: a *bounce* when the ball, moving down the picture, is kicked upwards (or
  slows sharply while coming towards the camera).

Racket impacts are loud broadband sounds. The sound reaches the phone ``distance / 343 m/s``
after the contact (about 70 ms, 4 frames, from the far baseline), so onsets are matched
after compensating that delay; a matching onset raises a hit's confidence and gives a
sub-frame contact time.

Contacts are refined by intersecting the quadratic fits before and after the kink
(:func:`refine_contact`); a bounce's ground position is the refined contact pixel's ray
intersected with the plane at ball-radius height.

Ball-machine feeds: a trajectory that *starts* near the ball machine's image position is a
``hit`` by ``machine``.

An optional learned classifier (:mod:`swingvision.training.train_events`) can replace the
rules: it scores every kink with the same features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pyarrow as pa

from swingvision.ball.trajectory import REF_WIDTH, PlayerBoxes
from swingvision.court import model as court_model
from swingvision.court.camera import Camera
from swingvision.storage.schemas import EVENTS

BALL_RADIUS_M = 0.0335
SPEED_OF_SOUND = 343.0
EVENT_KINDS = ("hit", "bounce", "net")


@dataclass(frozen=True)
class EventParams:
    window: int = 4  # points on each side of a candidate break
    noise_px: float = 1.5  # detection jitter (px at 3840 wide)
    #: A break needs: the two side lines explain the window better than one quadratic by
    #: this much (in noise variances; excludes smooth apexes), ...
    min_break: float = 3.0
    #: ... a velocity change of this many standard errors, ...
    min_dv_sigma: float = 6.0
    #: ... and a relative velocity change |v_after - v_before| / mean |v| of at least this.
    min_turn: float = 0.25
    max_run_gap_s: float = 0.25  # points further apart start a new run
    feet_frac: float = 0.25  # bottom share of the player box that counts as "at the feet"
    player_pad: float = 0.35  # player box padding (share of its height) for hits
    audio_window_s: float = 0.05
    audio_min_strength: float = 6.0
    machine_px: float = 120.0
    net_speed_ratio: float = 0.35
    min_event_gap_s: float = 0.08  # two events of one kind closer than this merge

    def as_config(self) -> dict:
        return asdict(self)


@dataclass
class Kink:
    i: int  # index into the run's points
    t: float
    frame: int
    x: float
    y: float
    score: float
    v_before: np.ndarray
    v_after: np.ndarray
    handoff: bool = False
    features: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def _quad_sse(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, max_deg: int = 2
) -> tuple[float, np.ndarray]:
    """SSE of a quadratic (linear for < 4 points) fit to x(t), y(t), and its coefficients."""
    deg = min(max_deg, 2 if len(t) >= 4 else 1)
    tc = t - t[0]
    A = np.vander(tc, deg + 1)
    cx, _rx, *_ = np.linalg.lstsq(A, x, rcond=None)
    cy, _ry, *_ = np.linalg.lstsq(A, y, rcond=None)
    sse = float(np.sum((A @ cx - x) ** 2 + (A @ cy - y) ** 2))
    return sse, np.stack([cx, cy])


def _poly_eval(coef: np.ndarray, t0: float, t: float | np.ndarray) -> np.ndarray:
    return np.array([np.polyval(coef[0], t - t0), np.polyval(coef[1], t - t0)])


def _poly_vel(coef: np.ndarray, t0: float, t: float) -> np.ndarray:
    return np.array(
        [np.polyval(np.polyder(coef[0]), t - t0), np.polyval(np.polyder(coef[1]), t - t0)]
    )


def refine_contact(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, i: int, w: int = 6
) -> tuple[float, float, float]:
    """Sub-frame contact (t, x, y) at a break near point ``i``.

    Quadratics are fitted to up to ``w`` points before and after (the point at the break is
    shared); the contact is where the two curves come closest within ±1.5 frames of it.
    """
    lo, hi = max(0, i - w), min(len(t), i + w + 1)
    if i - lo < 2 or hi - i < 3:
        return float(t[i]), float(x[i]), float(y[i])
    tb, ta = t[lo : i + 1], t[i:hi]
    _, cb = _quad_sse(tb, x[lo : i + 1], y[lo : i + 1])
    _, ca = _quad_sse(ta, x[i:hi], y[i:hi])
    dt = float(np.median(np.diff(t[lo:hi]))) if hi - lo > 1 else 1 / 60
    grid = np.linspace(t[i] - 1.5 * dt, t[i] + 1.5 * dt, 61)
    pb = _poly_eval(cb, tb[0], grid)
    pa_ = _poly_eval(ca, ta[0], grid)
    d = np.hypot(*(pb - pa_))
    k = int(np.argmin(d))
    p = (pb[:, k] + pa_[:, k]) / 2
    return float(grid[k]), float(p[0]), float(p[1])


# ---------------------------------------------------------------------------
# Kinks
# ---------------------------------------------------------------------------


def runs(track: pa.Table, max_gap_s: float) -> list[dict[str, np.ndarray]]:
    """Detected points of the track split into continuous runs."""
    src = np.array(track.column("source").to_pylist())
    det = src == "detected"
    t = track.column("t_s").to_numpy()[det]
    out = []
    if not len(t):
        return out
    cols = {
        "t": t,
        "frame": track.column("frame").to_numpy()[det],
        "x": track.column("x").to_numpy()[det].astype(np.float64),
        "y": track.column("y").to_numpy()[det].astype(np.float64),
        "tracklet": track.column("tracklet").to_numpy()[det],
    }
    cut = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)], strict=True):
        out.append({k: v[a:b] for k, v in cols.items()})
    return out


def find_kinks(run: dict[str, np.ndarray], p: EventParams, scale: float) -> list[Kink]:
    t, x, y = run["t"], run["x"], run["y"]
    n = len(t)
    if n < 5:
        return []
    W = p.window
    sigma2 = (p.noise_px * scale) ** 2
    fit = np.full(n, -np.inf)
    dv_sig = np.zeros(n)
    vb = np.zeros((n, 2))
    va = np.zeros((n, 2))
    for i in range(2, n - 2):
        lo, hi = max(0, i - W), min(n, i + W + 1)
        if i - lo < 2 or hi - i < 3:
            continue
        # A straight line on each side meeting at i vs one quadratic through the window:
        # over a few frames a ball flies almost straight; a quadratic on one side could
        # absorb a sharp V and misplace the break.
        sj, _ = _quad_sse(t[lo:hi], x[lo:hi], y[lo:hi])
        sb, cb = _quad_sse(t[lo : i + 1], x[lo : i + 1], y[lo : i + 1], max_deg=1)
        sa, ca = _quad_sse(t[i:hi], x[i:hi], y[i:hi], max_deg=1)
        fit[i] = (sj - sb - sa) / sigma2
        vb[i] = _poly_vel(cb, t[lo], t[i])
        va[i] = _poly_vel(ca, t[i], t[i])
        # Standard error of a line's slope: σ / sqrt(Σ (t - mean t)²), per side.
        sv2 = sum(
            sigma2 / max(float(np.sum((tt - tt.mean()) ** 2)), 1e-9)
            for tt in (t[lo : i + 1], t[i:hi])
        )
        dv_sig[i] = float(np.hypot(*(va[i] - vb[i]))) / np.sqrt(sv2)
    handoff = np.zeros(n, dtype=bool)
    tr = run["tracklet"]
    handoff[1:] = tr[1:] != tr[:-1]
    sp = np.hypot(vb[:, 0], vb[:, 1]) + np.hypot(va[:, 0], va[:, 1]) + 1e-6
    turn = np.hypot(va[:, 0] - vb[:, 0], va[:, 1] - vb[:, 1]) / sp * 2
    ok = (fit >= p.min_break) & (dv_sig >= p.min_dv_sigma) & (turn >= p.min_turn)
    out = []
    i = 2
    while i < n - 2:
        is_peak = dv_sig[i] >= dv_sig[max(0, i - 2) : i + 3].max()
        if is_peak and (ok[i] or (handoff[i] and turn[i] >= p.min_turn)):
            out.append(
                Kink(
                    i,
                    float(t[i]),
                    int(run["frame"][i]),
                    float(x[i]),
                    float(y[i]),
                    float(dv_sig[i]),
                    vb[i].copy(),
                    va[i].copy(),
                    bool(handoff[i]),
                    {"fit": float(fit[i])},
                )
            )
            i += 2
            continue
        i += 1
    return out


# ---------------------------------------------------------------------------
# Context and features
# ---------------------------------------------------------------------------


@dataclass
class EventContext:
    camera_at: object  # callable t -> Camera
    width: int
    players: PlayerBoxes | None = None
    #: Smoothed player court positions: (t, x, y) arrays, for audio delay and hitter position.
    player_xy: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    onsets_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    onsets_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    machine_xy: tuple[float, float] | None = None


def net_polygon(cam: Camera) -> np.ndarray:
    """Image quadrilateral of the net (posts' feet and tops)."""
    X = court_model.X_NET_POST
    pts = np.array(
        [
            [-X, 0, 0],
            [X, 0, 0],
            [X, 0, court_model.NET_HEIGHT_POST],
            [-X, 0, court_model.NET_HEIGHT_POST],
        ]
    )
    return cam.project(pts)


def _in_poly(poly: np.ndarray, x: float, y: float) -> bool:
    import cv2

    return cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0


def _player_at(ctx: EventContext, t: float) -> np.ndarray | None:
    pb = ctx.players
    if pb is None or not len(pb.t):
        return None
    k = int(np.clip(np.searchsorted(pb.t, t), 1, len(pb.t) - 1))
    k = k - 1 if abs(pb.t[k - 1] - t) < abs(pb.t[k] - t) else k
    if abs(pb.t[k] - t) > 0.15:
        return None
    return pb.box[k]


def _player_xy(ctx: EventContext, t: float) -> np.ndarray | None:
    if ctx.player_xy is None or not len(ctx.player_xy[0]):
        return None
    pt, px, py = ctx.player_xy
    k = int(np.clip(np.searchsorted(pt, t), 1, len(pt) - 1))
    k = k - 1 if abs(pt[k - 1] - t) < abs(pt[k] - t) else k
    if abs(pt[k] - t) > 0.3:
        return None
    return np.array([px[k], py[k]])


def audio_delay(cam: Camera, court_xy: np.ndarray | None, height: float = 1.0) -> float:
    if court_xy is None:
        return 0.0
    P = np.array([court_xy[0], court_xy[1], height])
    return float(np.linalg.norm(P - cam.center)) / SPEED_OF_SOUND


def kink_features(k: Kink, run, ctx: EventContext, p: EventParams) -> dict:
    scale = ctx.width / REF_WIDTH
    cam: Camera = ctx.camera_at(k.t)
    sb, sa = float(np.hypot(*k.v_before)), float(np.hypot(*k.v_after))
    f: dict = {
        "break": k.score,
        "fit": k.features.get("fit", np.nan),
        "turn": float(np.hypot(*(k.v_after - k.v_before)) / (sb + sa + 1e-6) * 2),
        "speed_before": sb / scale,
        "speed_after": sa / scale,
        "speed_ratio": sa / (sb + 1e-6),
        "vy_before": float(k.v_before[1]) / scale,
        "vy_after": float(k.v_after[1]) / scale,
        "dvy": float(k.v_after[1] - k.v_before[1]) / scale,
        "dvx": float(k.v_after[0] - k.v_before[0]) / scale,
        "reverse_x": float(np.sign(k.v_before[0]) != np.sign(k.v_after[0])),
        "handoff": float(k.handoff),
        "y_img": k.y / ctx.width,
        "in_net": float(_in_poly(net_polygon(cam), k.x, k.y)),
    }
    box = _player_at(ctx, k.t)
    if box is not None:
        h = box[3] - box[1]
        w = box[2] - box[0]
        pad = p.player_pad * h
        inside = box[0] - pad <= k.x <= box[2] + pad and box[1] - pad <= k.y <= box[3] + pad * 0.3
        f["near_player"] = float(inside)
        f["box_rel_y"] = float((k.y - box[1]) / max(h, 1))  # 0 head .. 1 feet
        f["box_rel_x"] = float((k.x - (box[0] + box[2]) / 2) / max(w, 1))
        f["box_h"] = float(h / scale)
    else:
        f["near_player"] = 0.0
        f["box_rel_y"] = np.nan
        f["box_rel_x"] = np.nan
        f["box_h"] = np.nan
    # Ground point of the ball (as if it were on the ground): distance from the player.
    g = cam.image_to_ground(np.array([[k.x, k.y]]), BALL_RADIUS_M)[0]
    pxy = _player_xy(ctx, k.t)
    f["ground_y"] = float(g[1]) if np.isfinite(g).all() else np.nan
    f["ground_dist_player"] = (
        float(np.hypot(*(g[:2] - pxy))) if pxy is not None and np.isfinite(g).all() else np.nan
    )
    # Sound: strongest onset in the window after the delay-compensated contact time.
    delay = audio_delay(cam, pxy if f["near_player"] else (g[:2] if np.isfinite(g).all() else None))
    f["audio_delay"] = delay
    if len(ctx.onsets_t):
        lo = np.searchsorted(ctx.onsets_t, k.t + delay - p.audio_window_s)
        hi = np.searchsorted(ctx.onsets_t, k.t + delay + p.audio_window_s)
        if hi > lo:
            j = lo + int(np.argmax(ctx.onsets_s[lo:hi]))
            f["audio_strength"] = float(ctx.onsets_s[j])
            f["audio_dt"] = float(ctx.onsets_t[j] - delay - k.t)
        else:
            f["audio_strength"] = 0.0
            f["audio_dt"] = np.nan
    else:
        f["audio_strength"] = 0.0
        f["audio_dt"] = np.nan
    return f


FEATURE_NAMES = (
    "break", "fit", "turn", "speed_before", "speed_after", "speed_ratio", "vy_before", "vy_after",
    "dvy", "dvx", "reverse_x", "handoff", "y_img", "in_net", "near_player", "box_rel_y",
    "box_rel_x", "box_h", "ground_y", "ground_dist_player", "audio_delay", "audio_strength",
    "audio_dt",
)  # fmt: skip


def classify_rules(f: dict, p: EventParams) -> tuple[str | None, float]:
    """(kind or None, confidence) from features."""
    audio = f["audio_strength"] >= p.audio_min_strength
    if f["near_player"]:
        at_feet = np.isfinite(f["box_rel_y"]) and f["box_rel_y"] >= 1 - p.feet_frac
        if at_feet and f["dvy"] < 0:
            return "bounce", 0.6
        conf = 0.55 + 0.1 * min(f["turn"], 2.0) + (0.25 if audio else 0.0)
        return "hit", min(conf, 0.99)
    # Inside the net's image region and nearly stopped: the net (a bounce keeps most of the
    # ball's speed, also one right behind the net seen through the mesh).
    if f["in_net"] and f["speed_ratio"] < p.net_speed_ratio:
        return "net", 0.6
    # Away from the player: the ball bounces when its image path, coming down the picture, is
    # kicked upwards.
    if f["dvy"] < 0 and f["vy_before"] > 0:
        conf = 0.6 + 0.1 * min(f["turn"], 2.0) + (0.1 if audio else 0.0)
        return "bounce", min(conf, 0.95)
    # A sharp slow-down while moving down the picture (a ball coming towards the camera).
    if f["dvy"] < 0 and f["turn"] > 0.5:
        conf = 0.55 + 0.1 * min(f["turn"], 2.0) + (0.1 if audio else 0.0)
        return "bounce", min(conf, 0.9)
    return None, 0.0


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def detect_events(
    track: pa.Table,
    ctx: EventContext,
    params: EventParams | None = None,
    model=None,
) -> tuple[pa.Table, list[dict]]:
    """Events table plus every kink's features (for training and debugging).

    ``model``: optional fitted classifier with ``predict_kinds(list[dict]) -> [(kind, conf)]``.
    """
    p = params or EventParams()
    scale = ctx.width / REF_WIDTH
    rows = []
    feats = []
    for run in runs(track, p.max_run_gap_s):
        kinks = find_kinks(run, p, scale)
        fs = [kink_features(k, run, ctx, p) for k in kinks]
        kinds = model.predict_kinds(fs) if model is not None else [classify_rules(f, p) for f in fs]
        for k, f, (kind, conf) in zip(kinks, fs, kinds, strict=True):
            feats.append({"frame": k.frame, "t_s": k.t, "x": k.x, "y": k.y, **f})
            if kind is None:
                continue
            tc, xc, yc = refine_contact(run["t"], run["x"], run["y"], k.i)
            rows.append(_event_row(kind, conf, k, tc, xc, yc, f, ctx))
        # Ball-machine feed: the run starts near the machine.
        if ctx.machine_xy is not None and len(run["t"]) >= 5:
            cam = ctx.camera_at(float(run["t"][0]))
            mpx = cam.project(np.array([[ctx.machine_xy[0], ctx.machine_xy[1], 0.6]]))[0]
            if np.hypot(run["x"][0] - mpx[0], run["y"][0] - mpx[1]) <= p.machine_px * scale:
                rows.append(
                    {
                        "kind": "hit",
                        "frame": int(run["frame"][0]),
                        "t_s": float(run["t"][0]),
                        "x": float(run["x"][0]),
                        "y": float(run["y"][0]),
                        "court_x": float(ctx.machine_xy[0]),
                        "court_y": float(ctx.machine_xy[1]),
                        "conf": 0.8,
                        "source": "rules",
                        "hitter": "machine",
                        "audio_dt": None,
                        "audio_strength": None,
                    }
                )
    rows = _merge_close(sorted(rows, key=lambda r: r["t_s"]), p.min_event_gap_s)
    return _events_table(rows), feats


def _event_row(kind, conf, k: Kink, tc, xc, yc, f, ctx: EventContext) -> dict:
    cam: Camera = ctx.camera_at(tc)
    court = (np.nan, np.nan)
    hitter = None
    if kind == "bounce":
        g = cam.image_to_ground(np.array([[xc, yc]]), BALL_RADIUS_M)[0]
        if np.isfinite(g).all():
            court = (float(g[0]), float(g[1]))
    elif kind == "hit":
        pxy = _player_xy(ctx, tc)
        if pxy is not None:
            court = (float(pxy[0]), float(pxy[1]))
        hitter = "me" if f.get("near_player") else "unknown"
        # A matching onset times the contact better than video frames can.
        if np.isfinite(f.get("audio_dt", np.nan)) and f["audio_strength"] >= 6.0:
            tc = float(k.t + f["audio_dt"])
    return {
        "kind": kind,
        "frame": k.frame,
        "t_s": float(tc),
        "x": float(xc),
        "y": float(yc),
        "court_x": court[0],
        "court_y": court[1],
        "conf": float(conf),
        "source": "rules",
        "hitter": hitter,
        "audio_dt": None if not np.isfinite(f.get("audio_dt", np.nan)) else float(f["audio_dt"]),
        "audio_strength": float(f.get("audio_strength", 0.0)),
    }


def _merge_close(rows: list[dict], gap: float) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if out and out[-1]["kind"] == r["kind"] and r["t_s"] - out[-1]["t_s"] < gap:
            if r["conf"] > out[-1]["conf"]:
                out[-1] = r
            continue
        out.append(r)
    return out


def _events_table(rows: list[dict]) -> pa.Table:
    if not rows:
        return EVENTS.empty_table()
    data = {f.name: [r.get(f.name) for r in rows] for f in EVENTS if f.name != "event_id"}
    data["event_id"] = list(range(len(rows)))
    return pa.table({f.name: pa.array(data[f.name], f.type) for f in EVENTS}, schema=EVENTS)


def split_segments(track: pa.Table, events: pa.Table) -> np.ndarray:
    """Segment number per track row: a new segment starts at every hit and bounce."""
    t = track.column("t_s").to_numpy()
    if not events.num_rows:
        return np.zeros(len(t), dtype=np.int32)
    et = np.sort(events.column("t_s").to_numpy())
    return np.searchsorted(et, t, side="right").astype(np.int32)
