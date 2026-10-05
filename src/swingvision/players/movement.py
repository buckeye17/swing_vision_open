"""Player movement on the court (PLAN.md §7.3).

"Me" detections become a court trajectory sampled at the detection rate:

* Gaps up to ``interp_gap_s`` are bridged (rows marked ``interp``); longer gaps split the
  trajectory into *runs*.
* Each run is smoothed with a constant-velocity Kalman filter and an RTS smoother. The
  measurement noise of every foot point comes from its ground-position σ, so far-court
  positions (large σ along the court) are smoothed more than near ones. Implausible jumps are
  down-weighted by an innovation gate.
* Speed comes from the smoothed velocity; distance only counts while moving faster than
  ``moving_speed`` so a standing player's residual jitter doesn't add up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.court import model as court_model
from swingvision.players.track import ROLE_ME
from swingvision.storage.schemas import MOVEMENT


@dataclass(frozen=True)
class MovementParams:
    interp_gap_s: float = 1.0
    accel_psd: float = 4.0  # white-acceleration spectral density, m²/s³
    gate_chi2: float = 25.0  # measurements beyond this Mahalanobis² are down-weighted ×100
    min_sigma_m: float = 0.05
    moving_speed: float = 0.6  # m/s: below this, movement doesn't add to the distance

    def as_config(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Kalman / RTS
# ---------------------------------------------------------------------------


def rts_smooth(
    t: np.ndarray, z: np.ndarray, r: np.ndarray, q: float, gate_chi2: float = 25.0
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity RTS smoother in 2D.

    ``z`` (n, 2) measurements with NaN rows where there is none; ``r`` (n, 2) measurement
    variances. Returns states (n, 4) ``[x, y, vx, vy]`` and their covariances (n, 4, 4).
    """
    n = len(t)
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    xf = np.zeros((n, 4))
    Pf = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4))
    Pp = np.zeros((n, 4, 4))
    F = np.eye(4)
    first = int(np.flatnonzero(np.isfinite(z[:, 0]))[0])
    x = np.array([z[first, 0], z[first, 1], 0.0, 0.0])
    P = np.diag([r[first, 0], r[first, 1], 16.0, 16.0])
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            F[0, 2] = F[1, 3] = dt
            q11, q12, q22 = q * dt**3 / 3, q * dt**2 / 2, q * dt
            Q = np.array([[q11, 0, q12, 0], [0, q11, 0, q12], [q12, 0, q22, 0], [0, q12, 0, q22]])
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if np.isfinite(z[k, 0]):
            R = np.diag(r[k])
            y = z[k] - x[:2]
            S = P[:2, :2] + R
            if y @ np.linalg.solve(S, y) > gate_chi2:
                R = R * 100.0
                S = P[:2, :2] + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P
        xf[k], Pf[k] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 2, -1, -1):
        dt = t[k + 1] - t[k]
        F[0, 2] = F[1, 3] = dt
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


def compute_movement(
    tracks: pa.Table, frames: pa.Table, params: MovementParams, player: str = ROLE_ME
) -> pa.Table:
    """``tracks`` (``players/tracks.parquet``) + processed frames → ``movement.parquet`` rows."""
    me = tracks.filter(pc.equal(tracks.column("role"), player)).sort_by("t_s")
    if me.num_rows == 0:
        return MOVEMENT.empty_table()
    ft = frames.column("t_s").to_numpy()
    ff = frames.column("frame").to_numpy()
    mt = me.column("t_s").to_numpy()
    mf = me.column("frame").to_numpy()
    mz = np.column_stack([me.column("court_x").to_numpy(), me.column("court_y").to_numpy()])
    sig = np.column_stack([me.column("sigma_x").to_numpy(), me.column("sigma_y").to_numpy()])
    sig = np.maximum(np.nan_to_num(sig, nan=1.0), params.min_sigma_m)
    mb = np.column_stack([me.column(c).to_numpy() for c in ("x0", "y0", "x1", "y1")])

    # Split into runs at long gaps; add the processed frames inside short gaps.
    breaks = np.flatnonzero(np.diff(mt) > params.interp_gap_s) + 1
    cols: dict[str, list] = {k: [] for k in MOVEMENT.names}
    for run, (a, b) in enumerate(zip(np.r_[0, breaks], np.r_[breaks, len(mt)], strict=True)):
        lo, hi = np.searchsorted(ft, mt[a]), np.searchsorted(ft, mt[b - 1], side="right")
        t = ft[lo:hi]
        frame = ff[lo:hi]
        pos = np.searchsorted(t, mt[a:b])
        if len(t) == 0 or not np.array_equal(t[np.clip(pos, 0, len(t) - 1)], mt[a:b]):
            # Detections not on the processed-frame grid (shouldn't happen): use them as-is.
            t, frame, pos = mt[a:b], mf[a:b], np.arange(b - a)
        z = np.full((len(t), 2), np.nan)
        r = np.full((len(t), 2), np.nan)
        z[pos] = mz[a:b]
        r[pos] = sig[a:b] ** 2
        xs, Ps = rts_smooth(t, z, r, params.accel_psd, params.gate_chi2)
        boxes = np.column_stack([np.interp(t, mt[a:b], mb[a:b, i]) for i in range(4)])
        source = np.full(len(t), "interp", dtype=object)
        source[pos] = "bbox"
        sigma = np.sqrt(np.max(np.linalg.eigvalsh(Ps[:, :2, :2]), axis=1))
        cols["frame"].append(frame)
        cols["t_s"].append(t)
        cols["player"].append(np.full(len(t), player, dtype=object))
        cols["x"].append(xs[:, 0])
        cols["y"].append(xs[:, 1])
        cols["vx"].append(xs[:, 2])
        cols["vy"].append(xs[:, 3])
        cols["speed"].append(np.hypot(xs[:, 2], xs[:, 3]))
        cols["sigma_m"].append(sigma)
        cols["source"].append(source)
        cols["run"].append(np.full(len(t), run, dtype=np.int32))
        for i, name in enumerate(("bx0", "by0", "bx1", "by1")):
            cols[name].append(boxes[:, i])
    data = {}
    for f in MOVEMENT:
        values = np.concatenate(cols[f.name])
        if pa.types.is_string(f.type):
            data[f.name] = pa.array(values.tolist(), f.type)
        else:
            data[f.name] = pa.array(values).cast(f.type)
    return pa.table(data, schema=MOVEMENT)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _rolling_mean_max(t: np.ndarray, v: np.ndarray, run: np.ndarray, window_s: float) -> float:
    best = 0.0
    for r in np.unique(run):
        m = run == r
        tt, vv = t[m], v[m]
        if len(tt) < 2:
            continue
        csum = np.r_[0.0, np.cumsum(vv[:-1] * np.diff(tt))]
        hi = np.searchsorted(tt, tt + window_s)
        ok = hi < len(tt)
        if not ok.any():
            continue
        lo_idx = np.flatnonzero(ok)
        span = tt[hi[ok]] - tt[lo_idx]
        avg = (csum[hi[ok]] - csum[lo_idx]) / np.maximum(span, 1e-6)
        best = max(best, float(avg.max()))
    return best


def usable_frames(frames: pa.Table, dark_luma: float, view_min: float | None = None):
    """Mask of processed frames that are bright enough and show the court."""
    ok = frames.column("luma").to_numpy() >= dark_luma
    if view_min is not None and "view" in frames.column_names:
        view = frames.column("view").to_numpy()
        ok &= ~(view < view_min)  # NaN (no reference image) counts as usable
    return ok


def summarize(
    movement: pa.Table,
    frames: pa.Table,
    dark_luma: float,
    params: MovementParams,
    view_min: float | None = None,
):
    """Headline movement stats for one player (all values plain floats for JSON).

    ``coverage`` is the share of usable processed frames (see :func:`usable_frames`) where the
    player was tracked; ``dark_frames`` counts the rest.
    """
    lit = usable_frames(frames, dark_luma, view_min)
    ff = frames.column("frame").to_numpy()
    lit_frames = set(ff[lit].tolist())
    out: dict = {
        "processed_frames": len(ff),
        "lit_frames": int(lit.sum()),
        "dark_frames": int((~lit).sum()),
    }
    if movement.num_rows == 0:
        out.update(
            coverage=0.0, tracked_s=0.0, distance_m=0.0, max_speed_mps=0.0,
            mean_moving_speed_mps=0.0, near_half_frac=0.0, n_runs=0,
        )  # fmt: skip
        return out
    t = movement.column("t_s").to_numpy()
    x = movement.column("x").to_numpy().astype(np.float64)
    y = movement.column("y").to_numpy().astype(np.float64)
    speed = movement.column("speed").to_numpy().astype(np.float64)
    run = movement.column("run").to_numpy()
    frame = movement.column("frame").to_numpy()
    same_run = run[1:] == run[:-1]
    dt = np.diff(t)
    step = np.hypot(np.diff(x), np.diff(y))
    moving = 0.5 * (speed[1:] + speed[:-1]) >= params.moving_speed
    tracked_lit = sum(1 for f in frame.tolist() if f in lit_frames)
    period = float(np.median(np.diff(frames.column("t_s").to_numpy()))) if len(ff) > 1 else 0.0
    out.update(
        coverage=tracked_lit / max(1, out["lit_frames"]),
        tracked_s=float(np.sum(dt[same_run]) + period * (run.max() + 1)),
        distance_m=float(np.sum(step[same_run & moving])),
        max_speed_mps=_rolling_mean_max(t, speed, run, 0.5),
        mean_moving_speed_mps=float(np.mean(speed[speed >= params.moving_speed]))
        if (speed >= params.moving_speed).any()
        else 0.0,
        near_half_frac=float(np.mean(y < 0)),
        n_runs=int(run.max() + 1),
    )
    return out


HEATMAP_X = court_model.HALF_DOUBLES + 4.0
HEATMAP_Y = court_model.HALF_LENGTH + 6.0


def heatmap(
    movement: pa.Table, cell_m: float = 0.5, fold: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seconds spent per court cell → (x centers, y centers, H[y, x]).

    ``fold`` mirrors far-half positions onto the near half (both ends in one picture).
    """
    xe = np.arange(-HEATMAP_X, HEATMAP_X + 1e-9, cell_m)
    ye = np.arange(-HEATMAP_Y, HEATMAP_Y + 1e-9, cell_m)
    if movement.num_rows < 2:
        return (
            0.5 * (xe[1:] + xe[:-1]),
            0.5 * (ye[1:] + ye[:-1]),
            np.zeros((len(ye) - 1, len(xe) - 1)),
        )
    t = movement.column("t_s").to_numpy()
    xy = np.column_stack([movement.column("x").to_numpy(), movement.column("y").to_numpy()]).astype(
        np.float64
    )
    if fold:
        far = xy[:, 1] > 0
        xy[far] = court_model.mirror(xy[far])
    w = np.r_[np.diff(t), 0.0]
    w = np.where(w > 1.0, 0.0, w)  # gaps between runs don't count
    H, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[ye, xe], weights=w)
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H
