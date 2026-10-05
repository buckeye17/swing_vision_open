"""Single-player tracking (PLAN.md §7.2, Phase 1).

1. **Court positions.** Each box's foot point (bottom center) is projected to the ground with
   the camera that applies at its time (piecewise calibration). The box top gives the person's
   height above that point, unless the box is cut off by the image edge. Each foot point also
   gets a ground-position σ: a few pixels of box jitter mean centimeters near the camera but
   tens of centimeters along the court at the far baseline.
2. **ROI.** Only detections on the court plus run-off count (neighboring courts, spectators
   beyond the fence and the playground are dropped).
3. **Tracklets.** Frame-to-frame assignment (Hungarian) on court distance, gated by plausible
   running speed plus the position σ. Duplicate boxes on the same person are merged first.
4. **Static objects.** Groups of motionless, short (< 1.3 m tall) detections that persist,
   such as a ball machine or a bag, are flagged and never become "me".
5. **"Me".** Practice sessions have one player. The tracklets that belong to them are the
   best-scoring *chain* of time-ordered tracklets: a chain gains each detection's confidence,
   links between tracklets must be physically possible, and starting over somewhere else
   (with no continuity) costs a penalty. That favors the one person who is there all session
   over brief false positives, and continuity over hopping between people.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pyarrow as pa

from swingvision.court import calibration as calib
from swingvision.court import model as court_model
from swingvision.court.homography import ground_sigma
from swingvision.storage.schemas import (
    PLAYER_TRACKS,
    Calibration,
    MachineInfo,
    PlayersSummary,
    StaticObject,
)

ROLE_ME, ROLE_OTHER, ROLE_STATIC, ROLE_OUTSIDE = "me", "other", "static", "outside"


@dataclass(frozen=True)
class TrackParams:
    roi_behind_m: float = 6.0
    roi_beside_m: float = 3.5
    #: Box-foot jitter in pixels, as a fraction of box height (min 2 px).
    foot_sigma_frac: float = 0.02
    #: Standing height used to place a person whose feet are below the frame (from the
    #: player's profile when set).
    person_height_m: float = 1.75
    max_speed: float = 9.0  # m/s, tracklet association gate
    height_ratio_gate: float = 1.5  # a track won't take a box this much taller/shorter
    max_age_s: float = 0.8  # a tracklet ends after this long without a detection
    link_gap_s: float = 4.0  # longest gap bridged between tracklets of one chain
    #: Chained tracklets may overlap this long if they're on the same spot (a duplicate box
    #: started a second track on the same person, which then took over).
    link_overlap_s: float = 2.0
    link_speed: float = 7.0  # m/s, average speed allowed across a gap
    restart_penalty: float = 4.0  # chain score lost when jumping with no continuity
    #: A detection adds ``conf - conf_offset`` to a chain, so runs of very weak boxes (shadows,
    #: dark corners) cost more than they add unless they connect stronger ones.
    conf_offset: float = 0.15
    static_min_s: float = 30.0
    static_spread_m: float = 0.35
    static_max_height_m: float = 1.3
    machine_radius_m: float = 0.8

    def as_config(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    court_x: np.ndarray
    court_y: np.ndarray
    height_m: np.ndarray  # NaN when the box top is cut off
    sigma_x: np.ndarray  # 1-σ foot position, meters
    sigma_y: np.ndarray
    sigma: np.ndarray  # major semi-axis
    foot_cut: np.ndarray  # box bottom at the image edge: no usable position
    from_head: np.ndarray  # feet below the frame: placed from the head at a standing height


def detection_geometry(
    det: pa.Table,
    cal: Calibration,
    width: int,
    height: int,
    foot_sigma_frac: float = 0.02,
    person_height_m: float = 1.75,
) -> Geometry:
    t = det.column("t_s").to_numpy()
    x0, y0, x1, y1 = (det.column(c).to_numpy().astype(np.float64) for c in ("x0", "y0", "x1", "y1"))
    n = len(t)
    cx = 0.5 * (x0 + x1)
    foot = np.column_stack([cx, y1])
    top = np.column_stack([cx, y0])
    sigma_px = np.maximum(2.0, foot_sigma_frac * (y1 - y0))
    g = np.full((n, 3), np.nan)
    hgt = np.full(n, np.nan)
    sig = np.full((n, 3), np.nan)
    cams, which = calib.cameras_for_times(cal, t)
    for k, cam in enumerate(cams):
        idx = np.flatnonzero(which == k)
        if not len(idx):
            continue
        g[idx] = cam.image_to_ground(foot[idx])
        # Height: where the top-of-box ray passes closest to the vertical through the foot.
        C = cam.center
        r = cam.rays(top[idx])
        d = g[idx] - C
        b = r[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            hgt[idx] = (b * np.sum(r * d, axis=1) - d[:, 2]) / (1 - b * b)
        ok = np.isfinite(g[idx]).all(axis=1)
        if ok.any():
            sig[idx[ok]] = ground_sigma(cam, g[idx[ok], :2]) * sigma_px[idx[ok], None]
        # Feet below the frame (the player right in front of the camera): the head is at
        # about standing height, so place the person under the head ray at that height.
        cut = idx[(y1[idx] >= height - 2.0) & (y0[idx] > 2.0)]
        if len(cut):
            head = cam.image_to_ground(top[cut], z=person_height_m)
            g[cut] = np.column_stack([head[:, :2], np.zeros(len(cut))])
            ok = np.isfinite(head).all(axis=1)
            if ok.any():
                s0 = ground_sigma(cam, head[ok, :2], z=person_height_m) * sigma_px[cut[ok], None]
                sig[cut[ok]] = np.sqrt(s0**2 + 0.3**2)  # ±10 cm of height ≈ ±0.3 m here
    top_cut = y0 <= 2.0
    bottom_cut = y1 >= height - 2.0
    hgt[top_cut | bottom_cut] = np.nan
    return Geometry(
        court_x=g[:, 0],
        court_y=g[:, 1],
        height_m=hgt,
        sigma_x=sig[:, 0],
        sigma_y=sig[:, 1],
        sigma=sig[:, 2],
        foot_cut=bottom_cut & (top_cut | ~np.isfinite(g[:, 0])),
        from_head=bottom_cut & ~top_cut & np.isfinite(g[:, 0]),
    )


def in_roi(x: np.ndarray, y: np.ndarray, behind_m: float, beside_m: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(x)
            & np.isfinite(y)
            & (np.abs(x) <= court_model.HALF_DOUBLES + beside_m)
            & (np.abs(y) <= court_model.HALF_LENGTH + behind_m)
        )


# ---------------------------------------------------------------------------
# Duplicates and tracklets
# ---------------------------------------------------------------------------


def _frame_groups(frame: np.ndarray) -> list[np.ndarray]:
    """Row indices per frame, in frame order (``frame`` must be sorted)."""
    if not len(frame):
        return []
    cuts = np.flatnonzero(np.diff(frame)) + 1
    return np.split(np.arange(len(frame)), cuts)


def suppress_duplicates(
    boxes: np.ndarray, conf: np.ndarray, groups: list[np.ndarray], containment: float = 0.8
) -> np.ndarray:
    """Mask of boxes to keep: drop a box mostly inside a more confident one in the same frame
    (a partial-body box on the same person)."""
    keep = np.ones(len(conf), dtype=bool)
    for g in groups:
        if len(g) < 2:
            continue
        order = g[np.argsort(-conf[g])]
        b = boxes[order]
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        for i in range(len(order)):
            if not keep[order[i]]:
                continue
            for j in range(i + 1, len(order)):
                if not keep[order[j]]:
                    continue
                iw = min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0])
                ih = min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1])
                if iw > 0 and ih > 0 and iw * ih >= containment * min(area[i], area[j]):
                    keep[order[j]] = False
    return keep


def build_tracklets(
    t: np.ndarray,
    groups: list[np.ndarray],
    xy: np.ndarray,
    sigma: np.ndarray,
    valid: np.ndarray,
    params: TrackParams,
    height_m: np.ndarray | None = None,
) -> np.ndarray:
    """Tracklet id per row (-1 for invalid rows), by frame-to-frame Hungarian assignment.

    Besides position, a detection must have a similar physical height to continue a track
    (when both are known), so a person walking past a ball machine or a bag doesn't swap
    tracks with it.
    """
    from scipy.optimize import linear_sum_assignment

    if height_m is None:
        height_m = np.full(len(t), np.nan)
    max_log_ratio = np.log(params.height_ratio_gate)
    ids = np.full(len(t), -1, dtype=np.int32)
    # Active tracks: [id, last_t, x, y, vx, vy, height (NaN: unknown)]
    active: list[list[float]] = []
    next_id = 0
    for g in groups:
        g = g[valid[g]]
        now = float(t[g[0]]) if len(g) else None
        if now is None:
            continue
        active = [a for a in active if now - a[1] <= params.max_age_s]
        matched_dets: set[int] = set()
        if active and len(g):
            A = np.array(active)
            dt = now - A[:, 1]
            pred = A[:, 2:4] + A[:, 4:6] * dt[:, None]
            d = np.linalg.norm(pred[:, None, :] - xy[g][None, :, :], axis=2)
            gate = 0.6 + params.max_speed * dt[:, None] + 2.5 * np.nan_to_num(sigma[g])[None, :]
            cost = d / gate
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.abs(np.log(height_m[g][None, :] / A[:, 6:7]))
            known = np.isfinite(ratio)
            cost[known] += 0.5 * ratio[known] / max_log_ratio
            cost[known & (ratio > max_log_ratio)] = 1e6
            cost[cost > 1.0] = 1e6
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] > 1.0:
                    continue
                row = g[c]
                a = active[r]
                step = max(now - a[1], 1e-3)
                v = (xy[row] - np.array(a[2:4])) / step
                speed = np.linalg.norm(v)
                if speed > params.max_speed:
                    v *= params.max_speed / speed
                a[4:6] = list(0.5 * np.array(a[4:6]) + 0.5 * v)
                a[1], a[2], a[3] = now, xy[row, 0], xy[row, 1]
                h = height_m[row]
                if np.isfinite(h):
                    a[6] = h if not np.isfinite(a[6]) else 0.7 * a[6] + 0.3 * h
                ids[row] = int(a[0])
                matched_dets.add(int(c))
        for c, row in enumerate(g):
            if c not in matched_dets:
                ids[row] = next_id
                active.append([next_id, now, xy[row, 0], xy[row, 1], 0.0, 0.0, height_m[row]])
                next_id += 1
    return ids


# ---------------------------------------------------------------------------
# Tracklet features, static objects, the "me" chain
# ---------------------------------------------------------------------------


@dataclass
class Tracklet:
    id: int
    rows: np.ndarray
    t0: float
    t1: float
    start_xy: np.ndarray
    end_xy: np.ndarray
    start_sigma: float
    end_sigma: float
    median_xy: np.ndarray
    spread: float  # 90th percentile distance from the median position
    height: float | None  # median, from boxes whose top is visible
    score: float
    ts: np.ndarray | None = None  # detection times
    xys: np.ndarray | None = None  # positions (n, 2)
    confs: np.ndarray | None = None

    def position_at(self, t: float) -> np.ndarray:
        if self.ts is None or self.xys is None:
            return self.end_xy
        return np.array([np.interp(t, self.ts, self.xys[:, k]) for k in range(2)])


def tracklet_table(
    ids: np.ndarray,
    t: np.ndarray,
    xy: np.ndarray,
    sigma: np.ndarray,
    conf: np.ndarray,
    h,
    conf_offset: float = 0.0,
) -> list[Tracklet]:
    out = []
    order = np.argsort(ids, kind="stable")
    sids = ids[order]
    starts = np.searchsorted(sids, np.unique(sids[sids >= 0]))
    ends = np.append(starts[1:], len(sids))
    for s, e in zip(starts, ends, strict=True):
        rows = order[s:e]
        rows = rows[np.argsort(t[rows], kind="stable")]
        p = xy[rows]
        med = np.median(p, axis=0)
        k = min(3, len(rows))
        heights = h[rows][np.isfinite(h[rows])]
        out.append(
            Tracklet(
                id=int(ids[rows[0]]),
                rows=rows,
                t0=float(t[rows[0]]),
                t1=float(t[rows[-1]]),
                start_xy=np.median(p[:k], axis=0),
                end_xy=np.median(p[-k:], axis=0),
                start_sigma=float(np.nanmax(sigma[rows[:k]], initial=0.0)),
                end_sigma=float(np.nanmax(sigma[rows[-k:]], initial=0.0)),
                median_xy=med,
                spread=float(np.percentile(np.linalg.norm(p - med, axis=1), 90)),
                height=float(np.median(heights)) if len(heights) else None,
                score=float((conf[rows] - conf_offset).sum()),
                ts=t[rows],
                xys=p,
                confs=conf[rows] - conf_offset,
            )
        )
    return out


def find_static_objects(
    tracklets: list[Tracklet], period_s: float, params: TrackParams
) -> tuple[list[StaticObject], set[int]]:
    """Clusters of motionless, short tracklets that add up to ``static_min_s`` of presence."""
    still = [
        tr
        for tr in tracklets
        if tr.spread <= params.static_spread_m
        and tr.t1 - tr.t0 >= 1.0
        and tr.height is not None
        and tr.height <= params.static_max_height_m
    ]
    clusters: list[list[Tracklet]] = []
    for tr in sorted(still, key=lambda tr: -(tr.t1 - tr.t0)):
        for cl in clusters:
            if np.linalg.norm(cl[0].median_xy - tr.median_xy) <= 2 * params.static_spread_m:
                cl.append(tr)
                break
        else:
            clusters.append([tr])
    objects, static_ids = [], set()
    for cl in clusters:
        seen = sum(len(tr.rows) for tr in cl) * period_s
        if seen < params.static_min_s:
            continue
        xy = np.median([tr.median_xy for tr in cl], axis=0)
        heights = [tr.height for tr in cl if tr.height is not None]
        objects.append(
            StaticObject(
                x=round(float(xy[0]), 3),
                y=round(float(xy[1]), 3),
                t0_s=round(min(tr.t0 for tr in cl), 3),
                t1_s=round(max(tr.t1 for tr in cl), 3),
                seen_s=round(seen, 2),
                height_m=round(float(np.median(heights)), 2) if heights else None,
            )
        )
        static_ids.update(tr.id for tr in cl)
    objects.sort(key=lambda o: -o.seen_s)
    return objects, static_ids


def select_chain(
    tracklets: list[Tracklet], period_s: float, params: TrackParams
) -> list[tuple[int, float]]:
    """The best-scoring chain of tracklets (see the module docstring).

    Returns ``(tracklet id, from_t)`` pairs: each tracklet counts from ``from_t`` on (the end of
    its predecessor). Tracklets are processed by end time, so a chain can continue into a
    tracklet that started *before* its predecessor ended (a duplicate track on the same person
    that took over), provided the two agree on where the person was at the junction.
    """
    if not tracklets:
        return []
    trs = sorted(tracklets, key=lambda tr: (tr.t1, tr.t0))
    n = len(trs)
    t1s = np.array([tr.t1 for tr in trs])
    cums = [np.cumsum(tr.confs) if tr.confs is not None else None for tr in trs]

    def score_after(j: int, t: float) -> float:
        tr = trs[j]
        if tr.ts is None or cums[j] is None or t < tr.t0:
            return tr.score
        k = int(np.searchsorted(tr.ts, t, side="right"))
        return float(cums[j][-1] - (cums[j][k - 1] if k > 0 else 0.0))

    best = np.zeros(n)
    prev = np.full(n, -1)
    prefix_best = np.zeros(n)  # max(best[:k+1])
    prefix_arg = np.zeros(n, dtype=np.int64)
    for j, tr in enumerate(trs):
        # Start a chain here, or restart (no continuity, so it costs a penalty) after the best
        # chain that ended before this tracklet starts.
        best_j, arg = tr.score, -1
        k = int(np.searchsorted(t1s[:j], tr.t0, side="right")) - 1
        if k >= 0 and prefix_best[k] - params.restart_penalty > 0:
            best_j, arg = tr.score + prefix_best[k] - params.restart_penalty, int(prefix_arg[k])
        # Links from tracklets that ended up to ``link_gap_s`` before j started, or while j ran.
        lo = int(np.searchsorted(t1s[:j], tr.t0 - params.link_gap_s, side="left"))
        for i in range(lo, j):
            pi = trs[i]
            if pi.t1 >= tr.t1:
                continue
            gap = tr.t0 - pi.t1
            sig = 2.5 * max(pi.end_sigma, tr.start_sigma)
            if gap >= -2.5 * period_s:
                reach = params.link_speed * max(gap, period_s) + 0.8 + sig
                ok = np.linalg.norm(pi.end_xy - tr.start_xy) <= reach
                gain = tr.score
            else:
                # Overlap: j must be where i ended, when i ended; j counts from there on.
                reach = params.link_speed * period_s + 0.8 + sig
                ok = np.linalg.norm(tr.position_at(pi.t1) - pi.end_xy) <= reach
                gain = score_after(j, pi.t1)
            if ok and best[i] + gain > best_j:
                best_j, arg = best[i] + gain, i
        best[j], prev[j] = best_j, arg
        if j == 0 or best_j > prefix_best[j - 1]:
            prefix_best[j], prefix_arg[j] = best_j, j
        else:
            prefix_best[j], prefix_arg[j] = prefix_best[j - 1], prefix_arg[j - 1]
    j = int(np.argmax(best))
    chain: list[tuple[int, float]] = []
    while j >= 0:
        i = int(prev[j])
        from_t = trs[i].t1 if i >= 0 and trs[j].t0 < trs[i].t1 else -np.inf
        chain.append((trs[j].id, float(from_t)))
        j = i
    return chain[::-1]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def track_players(
    det: pa.Table,
    cal: Calibration,
    width: int,
    height: int,
    period_s: float,
    params: TrackParams,
    machine_xy: list[float] | None = None,
    auto_machine: bool = False,
    ignore_frames: np.ndarray | None = None,
) -> tuple[pa.Table, PlayersSummary]:
    """Annotate every detection (court position, tracklet, role) and pick out "me".

    Detections in ``ignore_frames`` (the camera wasn't showing the court) are labelled
    ``outside`` and take no part in tracking.
    """
    det = det.sort_by([("frame", "ascending"), ("conf", "descending")])
    n = det.num_rows
    t = det.column("t_s").to_numpy()
    frame = det.column("frame").to_numpy()
    conf = det.column("conf").to_numpy().astype(np.float64)
    boxes = np.column_stack(
        [det.column(c).to_numpy().astype(np.float64) for c in ("x0", "y0", "x1", "y1")]
    ).reshape(n, 4)
    geo = detection_geometry(
        det, cal, width, height, params.foot_sigma_frac, params.person_height_m
    )
    xy = np.column_stack([geo.court_x, geo.court_y]).reshape(n, 2)
    roi = in_roi(geo.court_x, geo.court_y, params.roi_behind_m, params.roi_beside_m)
    if ignore_frames is not None and len(ignore_frames):
        roi &= ~np.isin(frame, ignore_frames)
    groups = _frame_groups(frame)
    valid = roi & suppress_duplicates(boxes, conf, groups) & ~geo.foot_cut
    ids = build_tracklets(t, groups, xy, geo.sigma, valid, params, geo.height_m)
    tracklets = tracklet_table(ids, t, xy, geo.sigma, conf, geo.height_m, params.conf_offset)

    statics, static_ids = find_static_objects(tracklets, period_s, params)
    machine = None
    if machine_xy is not None:
        machine = MachineInfo(x=machine_xy[0], y=machine_xy[1], source="user")
    elif auto_machine and statics:
        machine = MachineInfo(x=statics[0].x, y=statics[0].y, source="auto")
    if machine is not None:
        m = np.array([machine.x, machine.y])
        for tr in tracklets:
            short = tr.height is None or tr.height <= params.static_max_height_m + 0.2
            if np.linalg.norm(tr.median_xy - m) <= params.machine_radius_m and short:
                static_ids.add(tr.id)

    candidates = [tr for tr in tracklets if tr.id not in static_ids]
    chain = select_chain(candidates, period_s, params)
    me_ids = {tid for tid, _ in chain}

    role = np.full(n, ROLE_OUTSIDE, dtype=object)
    role[ids >= 0] = ROLE_OTHER
    role[np.isin(ids, list(static_ids))] = ROLE_STATIC
    me_rows = np.zeros(n, dtype=bool)
    for tid, from_t in chain:
        me_rows |= (ids == tid) & (t > from_t)
    # Where chained tracklets overlap in time, keep one "me" per frame (the most confident).
    for g in groups:
        m = g[me_rows[g]]
        if len(m) > 1:
            me_rows[m[np.argsort(-conf[m])[1:]]] = False
    role[me_rows] = ROLE_ME
    # Inside the ROI but dropped as a duplicate or a cut-off box: not a separate person.
    role[(ids < 0) & roi] = ROLE_OTHER

    table = pa.table(
        {
            **{name: det.column(name) for name in det.column_names},
            "court_x": geo.court_x.astype(np.float32),
            "court_y": geo.court_y.astype(np.float32),
            "height_m": geo.height_m.astype(np.float32),
            "sigma_x": geo.sigma_x.astype(np.float32),
            "sigma_y": geo.sigma_y.astype(np.float32),
            "track_id": ids,
            "role": pa.array(role.tolist(), pa.string()),
        }
    )
    summary = PlayersSummary(
        n_detections=n,
        n_in_roi=int(roi.sum()),
        n_tracklets=len(tracklets),
        me_tracklets=sorted(me_ids),
        me_detections=int(me_rows.sum()),
        static_objects=statics,
        machine=machine,
    )
    return table.cast(PLAYER_TRACKS), summary
