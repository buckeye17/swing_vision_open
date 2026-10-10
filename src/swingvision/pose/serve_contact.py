"""Serve contact point relative to the front toe, in the contact frame (PLAN.md §7.11).

**Contact frame: the last frame on the toss path.** Once the ball leaves the tossing hand it
flies freely until the racket strikes it. The toss is fitted in 3D (gravity, drag; the same
physics as every other flight) from its apex backwards to the release and forwards frame by
frame. The first detection that has left the path (the racket takes the ball from ≈1 m/s to
≈50 m/s in one frame, so tens of pixels) is the first frame after contact; the contact frame
is the frame before it. The contact point is the toss path at that frame's time, its
covariance sampled from the fit. Near the apex the ball moves < 2 m/s, so snapping to a frame
costs at most ≈3 cm at 60 fps.

When the ball isn't detected around the contact (hidden by the racket, blurred), the serve
flight's first detections are extrapolated back to where they meet the toss path and the
contact frame is the last frame before that (``contact_frame_inferred``). The serve fit's
contact prior at the hitter's feet plays no part here: it would pull the offsets toward the
feet.

**The toe in that frame** (:func:`toe_at_contact`) comes from foot keypoints (``serve_feet``):
the ray through the 2D big-toe keypoint meets the plane at the keypoint's height above the
sole, and the shoe's tip is a little further on along the foot (both measured against
hand-labeled toe tips, ``docs/spikes/m7b-toe-pose.md``). The toe is on the ground when its
ground point stayed still over the frames before the contact; a lifted toe (``toe_lifted``)
gets a large σ along its ray, since no 3D model measured its height reliably (the spike). A
toe keypoint the model isn't sure of (from behind, the shoe hides the toe tip) is replaced by
the heel plus the foot length along the heel → toe direction (``toe_estimated``).

**Offsets** (:func:`offsets`) are in the hitter's frame with the toe as the origin: forward
(toward the net), lateral (+ toward the racket-arm side, so left- and right-handers read the
same) and the ball's height above the ground, each with a σ from the toe and contact
covariances.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from swingvision.ball import physics as ph
from swingvision.court import model as court_model
from swingvision.court.camera import Camera

#: Bumped when the contact-frame algorithm changes (``swings`` reruns, cached tosses refit).
ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class ContactParams:
    #: Toss detections are searched from this long before the swing's contact estimate ...
    before_s: float = 1.6
    #: ... to this long after it.
    after_s: float = 0.35
    #: The seed fit uses the detections this long before the toss apex (free flight for sure).
    seed_s: float = 0.3
    #: Fewer toss detections than this: no toss fit (``toss_not_tracked``).
    min_points: int = 6
    #: A detection within this many σ (and at least ``gate_px``) of the path is on it.
    gate_sigma: float = 4.0
    gate_px: float = 5.0
    #: The path's own uncertainty counts up to this much (px) in those σ.
    cap_px: float = 2.0
    #: A detection beyond this many σ (and at least ``off_px``) has left the path.
    off_sigma: float = 6.0
    off_px: float = 9.0
    #: While growing the toss back to the release, refit after this many new detections
    #: (forwards to the contact: after every one).
    refit_every: int = 3
    #: Growing the toss backwards stops after this many detections off the path in a row
    #: (the ball is still in the hand).
    release_misses: int = 2
    #: The serve flight's first detections (at most this many) give the inferred contact.
    flight_points: int = 3
    #: No detection within this many frames of the contact frame: ``contact_above_frame``
    #: when the toss left the top of the picture, else ``contact_not_seen``.
    seen_frames: int = 3
    #: The toss leaves the top of the picture when its last detection is this close (px at
    #: 2160 high) to the top edge.
    top_margin_px: float = 60.0
    #: The contact frame may be at most this far (s) from the swing's own estimate.
    max_shift_s: float = 0.25
    #: Parameter samples for the contact covariance.
    n_samples: int = 64

    def as_config(self) -> dict:
        return {**asdict(self), "algorithm": ALGORITHM_VERSION}


@dataclass
class TossContact:
    """The contact frame and the ball's position on the toss path in it."""

    frame: int | None
    t: float | None
    #: Contact point (court m) and its covariance (3, 3).
    pos: np.ndarray | None = None
    cov: np.ndarray | None = None
    #: toss_path (the frame before the ball left the path was seen) | inferred | edit
    source: str | None = None
    flags: list[str] = field(default_factory=list)
    #: The toss fit (physics parameters), when there is one.
    theta: np.ndarray | None = None
    theta_cov: np.ndarray | None = None
    t0: float | None = None
    t_release: float | None = None
    n_toss: int = 0
    toss_rms_px: float | None = None
    #: Last frame with a detection on the path, first frame with one off it.
    last_on: int | None = None
    first_off: int | None = None
    #: The ball's detected pixel in the contact frame (when there is one).
    px: np.ndarray | None = None


@dataclass
class BallPoints:
    """Detected ball pixels (``ball/track.parquet`` rows with ``source == detected``)."""

    frame: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def window(self, t0: float, t1: float) -> BallPoints:
        m = (self.t >= t0) & (self.t <= t1)
        return BallPoints(self.frame[m], self.t[m], self.x[m], self.y[m])

    def px(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    def __len__(self) -> int:
        return len(self.t)


@dataclass
class FrameClock:
    """Time of every frame around the serve (VFR footage: frames aren't evenly spaced)."""

    frame: np.ndarray
    t: np.ndarray

    def time_of(self, frame: int) -> float | None:
        k = np.searchsorted(self.frame, frame)
        if k < len(self.frame) and self.frame[k] == frame:
            return float(self.t[k])
        return None

    def last_at_or_before(self, t: float) -> tuple[int, float] | None:
        k = int(np.searchsorted(self.t, t + 1e-9)) - 1
        if k < 0:
            return None
        return int(self.frame[k]), float(self.t[k])


def _fit(cam: Camera, t: np.ndarray, px: np.ndarray, p: ph.FitParams) -> ph.FlightFit:
    sigma = ph.point_sigma(px, t, cam.width, p)
    data = ph.FlightData(cam, t, px, sigma, ph.EndPoint("free", float(t[0])))
    return ph.fit_flight(data, p)


def _quick_fit(
    cam: Camera, t: np.ndarray, px: np.ndarray, p: ph.FitParams, prev: ph.FlightFit | None = None
) -> ph.FlightFit:
    """A lighter :func:`ph.fit_flight` for growing the toss point by point: the same model
    and residuals, warm-started from the previous fit, without outlier passes or outputs."""
    from scipy.optimize import least_squares

    sigma = ph.point_sigma(px, t, cam.width, p)
    t0 = float(t[0])
    data = ph.FlightData(cam, t, px, sigma, ph.EndPoint("free", t0))
    if prev is None:
        theta0 = np.concatenate([ph.linear_init(data, p), [0.0, ph.DRAG_CD]])
    else:  # the previous fit's state at this start time
        pos, vel = ph.simulate(prev.theta[None], prev.t0, np.array([t0]))
        theta0 = np.concatenate([pos[0, 0], vel[0, 0], prev.theta[6:]])
    steps = np.r_[np.full(6, 1e-4), 1e-5, 1e-4]

    def fun(th):
        return ph._residuals_batch(th[None], data, p)[0]

    def jac(th):
        r = ph._residuals_batch(np.vstack([th, th + np.diag(steps)]), data, p)
        return ((r[1:] - r[0]) / steps[:, None]).T

    sol = least_squares(fun, theta0, jac=jac, loss="soft_l1", f_scale=p.loss_scale, max_nfev=30)
    J = sol.jac
    dof = max(1, J.shape[0] - J.shape[1])
    chi2 = float(np.sum(sol.fun**2)) / dof
    cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2)
    pos, _ = ph.simulate(sol.x[None], t0, t)
    err = np.hypot(*(cam.project(pos[0]) - px).T)
    return ph.FlightFit(
        True, "", sol.x, cov, t0, float(t[-1]), len(t), float(np.sqrt(np.mean(err**2))), chi2
    )


def _samples(fit: ph.FlightFit, n: int) -> np.ndarray:
    th = fit.theta[None]
    if fit.cov is not None and np.isfinite(fit.cov).all():
        try:
            L = np.linalg.cholesky(fit.cov + 1e-12 * np.eye(len(fit.theta)))
            rng = np.random.default_rng(0)
            th = np.vstack([fit.theta, fit.theta + rng.standard_normal((n, len(fit.theta))) @ L.T])
        except np.linalg.LinAlgError:
            pass
    return th


def path_at(
    fit: ph.FlightFit, times: np.ndarray, n: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Toss-path positions at ``times`` (N, 3) and, with ``n`` samples, their covariances
    (N, 3, 3)."""
    times = np.atleast_1d(np.asarray(times, np.float64))
    order = np.argsort(times)
    th = _samples(fit, n) if n else fit.theta[None]
    pos, _ = ph.simulate(th, fit.t0, times[order])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    pos = pos[:, inv]
    cov = None
    if len(th) > 8:
        d = pos[1:] - pos[0][None]
        cov = np.einsum("bni,bnj->nij", d, d) / (len(th) - 1)
    return pos[0], cov


def _pred_sigma_px(fit: ph.FlightFit, cam: Camera, t: float, n: int = 24) -> float:
    """1-σ (px) of the path's projected position at ``t`` (fit uncertainty only)."""
    th = _samples(fit, n)
    if len(th) < 9:
        return 0.0
    pos, _ = ph.simulate(th, fit.t0, np.array([t]))
    px = cam.project(pos[:, 0])
    return float(np.sqrt(np.mean(np.sum((px[1:] - px[0]) ** 2, axis=1))))


def _deviation(
    fit: ph.FlightFit, cam: Camera, t: float, px: np.ndarray, cap_px: float
) -> tuple[float, float]:
    """Distance (px) of a detection from the toss path at its time, and the σ it's judged
    by: detection noise plus the path's own uncertainty there, at most ``cap_px`` (the path
    is refitted after every point, so it is never extrapolated far)."""
    pos, _ = ph.simulate(fit.theta[None], fit.t0, np.array([t]))
    pred = cam.project(pos[0, 0])
    noise = ph.FitParams().noise_px * cam.width / 3840
    s = np.hypot(noise, min(cap_px, _pred_sigma_px(fit, cam, t)))
    return float(np.hypot(*(px - pred))), float(s)


def find_toss(pts: BallPoints, t_guess: float, p: ContactParams) -> int | None:
    """Index of the toss apex: the highest detection (smallest image y) in the second before
    the contact estimate, with the ball seen rising into it."""
    m = np.flatnonzero((pts.t >= t_guess - 1.2) & (pts.t <= t_guess + 0.05))
    if len(m) < p.min_points:
        return None
    k = int(m[np.argmin(pts.y[m])])
    rising = np.flatnonzero((pts.t < pts.t[k]) & (pts.t >= pts.t[k] - p.seed_s))
    if len(rising) < 3 or not (pts.y[rising[0]] > pts.y[k] + 10):
        return None
    return k


def _left_top(pts: BallPoints, apex: int, cam: Camera, p: ContactParams) -> bool:
    """The toss's highest detection is at the top edge with no descent seen after it."""
    if pts.y[apex] > p.top_margin_px * cam.height / 2160:
        return False
    after = (pts.t > pts.t[apex]) & (pts.t <= pts.t[apex] + 0.15)
    return not (pts.y[after] > pts.y[apex] + 3).any()


def toss_contact(
    pts: BallPoints,
    clock: FrameClock,
    cam: Camera,
    t_guess: float,
    p: ContactParams | None = None,
    fit_params: ph.FitParams | None = None,
    frame_override: int | None = None,
) -> TossContact:
    """The serve's contact frame and contact point from the toss path (module docstring).

    ``t_guess``: the swing's contact estimate (hit, impact sound or pose). ``frame_override``:
    a contact frame the user set (the toss path still gives the position).
    """
    p = p or ContactParams()
    fp = fit_params or ph.FitParams()
    pts = pts.window(t_guess - p.before_s, t_guess + p.after_s)
    out = TossContact(None, None)
    apex = find_toss(pts, t_guess, p)
    if apex is None:
        out.flags.append("toss_not_tracked")
        return _without_toss(out, pts, clock, t_guess, p, frame_override)
    if _left_top(pts, apex, cam, p):
        # The toss went out of the top of the picture: its apex and the contact aren't
        # seen, and a few rising detections don't fix the depth.
        out.flags.append("contact_above_frame")
        return _without_toss(out, pts, clock, t_guess, p, frame_override)
    t, px = pts.t, pts.px()
    grown = _grow(pts, cam, apex, p, fp)
    if grown is None:
        out.flags.append("toss_not_tracked")
        return _without_toss(out, pts, clock, t_guess, p, frame_override)
    on, first_off = grown
    idx = np.array(sorted(on))
    fit = _fit(cam, t[idx], px[idx], fp)
    out.theta, out.theta_cov, out.t0 = fit.theta, fit.cov, fit.t0
    out.n_toss, out.toss_rms_px = len(idx), float(fit.rms_px)
    out.t_release = float(t[idx[0]])
    if not fit.ok:
        out.flags.append("poor_toss_fit")
    last_on = int(idx[-1])
    out.last_on = int(pts.frame[last_on])
    out.first_off = int(pts.frame[first_off]) if first_off is not None else None
    # The contact frame: the frame before the first detection off the path; when that
    # detection doesn't follow right after the last one on it (the ball hidden or blurred at
    # the contact), the time the serve flight left the path decides (inferred).
    if frame_override is not None:
        frame, source = int(frame_override), "edit"
    elif first_off is not None and out.first_off == out.last_on + 1:
        frame, source = out.last_on, "toss_path"
    else:
        t_star = _leaves_path(fit, cam, pts, last_on, first_off, t_guess, p)
        lo = out.last_on
        hi = out.first_off - 1 if out.first_off is not None else None
        hit = clock.last_at_or_before(t_star)
        frame = hit[0] if hit is not None else lo
        frame = max(lo, frame if hi is None else min(hi, frame))
        seen = set(pts.frame.tolist())
        if hi is not None and all(f in seen for f in range(lo, hi + 2)):
            source = "toss_path"  # every frame seen: detections between the gates decided
        else:
            source = "inferred"
            out.flags.append("contact_frame_inferred")
    tf = clock.time_of(frame)
    if tf is None:
        tf = float(np.interp(frame, pts.frame, pts.t))
    out.frame, out.t, out.source = frame, tf, source
    pos, cov = path_at(fit, np.array([tf]), p.n_samples)
    out.pos, out.cov = pos[0], (cov[0] if cov is not None else None)
    k = np.flatnonzero(pts.frame == frame)
    if len(k):
        out.px = px[int(k[0])]
    _seen_flags(out, pts, p, cam)
    return out


def release_index(pts: BallPoints, apex: int, max_gap: int = 3) -> int:
    """Index of the release: in the hand the ball accelerates up the image, in free flight it
    slows down, so the release is where the upward image speed peaks (on the run of
    detections leading into the apex). Points after it are free flight."""
    k = apex
    while k > 0 and pts.frame[k] - pts.frame[k - 1] <= max_gap:
        k -= 1
    idx = np.arange(k, apex + 1)
    if len(idx) < 5:
        return int(k) - 1
    vy = np.gradient(pts.y[idx], pts.t[idx])
    if len(vy) >= 5:
        vy = np.convolve(vy, np.ones(3) / 3, mode="same")
        vy[0], vy[-1] = vy[1], vy[-2]
    j = int(np.argmin(vy[: max(1, len(vy) - 3)]))
    return int(idx[j])


def _grow(
    pts: BallPoints, cam: Camera, apex: int, p: ContactParams, fp: ph.FitParams
) -> tuple[set[int], int | None] | None:
    """The toss's detections and the first one after contact (index, or ``None``).

    Seeded on the rise into the apex (surely free flight), then grown point by point with
    refits: backwards to the release (:func:`release_index`; detections off the path end it
    early), forwards until the first detection that has left the path, confirmed by the next
    one so a stray detection on the racket isn't taken for the contact.
    """
    t, px = pts.t, pts.px()
    first = release_index(pts, apex) + 1
    seed = np.flatnonzero((t >= t[apex] - p.seed_s) & (t <= t[apex]))
    seed = seed[seed >= first]
    if len(seed) < 3:
        return None
    on = {int(i) for i in seed}
    fit = _quick_fit(cam, t[seed], px[seed], fp)
    if not np.isfinite(fit.theta).all():
        return None

    pending = 0

    def accept(i: int, dev: float, gate: float, every: int):
        # Refit after every few points, or right away for a point near the gate (the path
        # is then never extrapolated more than a few frames).
        nonlocal fit, pending
        on.add(i)
        pending += 1
        if pending >= every or dev > 0.5 * gate:
            idx = np.array(sorted(on))
            fit = _quick_fit(cam, t[idx], px[idx], fp, fit)
            pending = 0

    misses = 0
    for i in range(int(seed[0]) - 1, first - 1, -1):
        dev, s = _deviation(fit, cam, t[i], px[i], p.cap_px)
        gate = max(p.gate_px, p.gate_sigma * s)
        if dev <= gate:
            accept(i, dev, gate, p.refit_every)
            misses = 0
        else:
            misses += 1
            if misses >= p.release_misses:
                break
    if pending:
        idx = np.array(sorted(on))
        fit = _quick_fit(cam, t[idx], px[idx], fp, fit)
        pending = 0
    first_off = None
    for i in range(apex + 1, len(t)):
        dev, s = _deviation(fit, cam, t[i], px[i], p.cap_px)
        gate = max(p.gate_px, p.gate_sigma * s)
        if dev <= gate:
            # Towards the contact every frame counts: refit after each one.
            accept(i, dev, gate, 1)
            continue
        if dev > max(p.off_px, p.off_sigma * s):
            nxt = i + 1
            if nxt < len(t) and pts.frame[nxt] - pts.frame[i] <= 3:
                d2, s2 = _deviation(fit, cam, t[nxt], px[nxt], p.cap_px)
                if d2 <= max(p.off_px, p.off_sigma * s2):
                    continue  # back near the path: a stray detection
            first_off = i
            break
        # Between the gates (a contact late in the frame interval moved the ball only a little
        # by this frame): neither on the path nor clearly off; the serve's extrapolation
        # decides (see toss_contact).
    if len(on) < p.min_points:
        return None
    return on, first_off


def _leaves_path(
    fit: ph.FlightFit,
    cam: Camera,
    pts: BallPoints,
    last_on: int,
    first_off: int | None,
    t_guess: float,
    p: ContactParams,
) -> float:
    """When the serve flight left the toss path: its first detections extrapolated back (in
    the image, at constant velocity: the racket has just launched it) to where they meet
    the path's projection. Without serve-flight detections: the swing's own estimate."""
    t_lo = float(pts.t[last_on])
    if first_off is None:
        return float(np.clip(t_guess, t_lo, t_lo + p.max_shift_s))
    j = np.arange(first_off, min(len(pts.t), first_off + p.flight_points))
    j = j[pts.t[j] - pts.t[first_off] < 0.1]
    t_hi = float(pts.t[first_off])
    grid = np.linspace(t_lo, t_hi, max(2, int((t_hi - t_lo) * 600) + 1))
    path_px = cam.project(ph.simulate(fit.theta[None], fit.t0, grid)[0][0])
    if len(j) >= 2:
        A = np.column_stack([pts.t[j] - t_hi, np.ones(len(j))])
        cx = np.linalg.lstsq(A, pts.x[j], rcond=None)[0]
        cy = np.linalg.lstsq(A, pts.y[j], rcond=None)[0]
        fl = np.column_stack([cx[0] * (grid - t_hi) + cx[1], cy[0] * (grid - t_hi) + cy[1]])
    else:
        fl = np.tile([pts.x[first_off], pts.y[first_off]], (len(grid), 1))
    return float(grid[int(np.argmin(np.hypot(*(fl - path_px).T)))])


def _without_toss(
    out: TossContact,
    pts: BallPoints,
    clock: FrameClock,
    t_guess: float,
    p: ContactParams,
    frame_override: int | None,
) -> TossContact:
    """No toss path: the contact frame from the swing's estimate (or the user), no point."""
    if frame_override is not None:
        out.frame, out.source = int(frame_override), "edit"
        out.t = clock.time_of(out.frame)
    else:
        hit = clock.last_at_or_before(t_guess)
        if hit is not None:
            out.frame, out.t = hit
            out.source = "inferred"
            out.flags.append("contact_frame_inferred")
    _seen_flags(out, pts, p, None)
    return out


def _seen_flags(out: TossContact, pts: BallPoints, p: ContactParams, cam: Camera | None) -> None:
    if out.frame is None or {"contact_above_frame", "contact_not_seen"} & set(out.flags):
        return
    near = np.abs(pts.frame - out.frame) <= p.seen_frames
    if near.any():
        return
    top = pts.y.min() if len(pts) else np.inf
    height = cam.height if cam is not None else 2160
    if top <= p.top_margin_px * height / 2160 or (
        out.pos is not None and cam is not None and cam.project(out.pos[None])[0, 1] < 0
    ):
        out.flags.append("contact_above_frame")
    else:
        out.flags.append("contact_not_seen")


# ---------------------------------------------------------------------------
# Offsets in the hitter's frame
# ---------------------------------------------------------------------------


def hitter_axes(side: int, racket_hand: str) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (court x, y) of the hitter's forward (toward the net) and lateral
    (toward the racket-arm side) directions."""
    fwd = np.array([0.0, 1.0]) if side < 0 else np.array([0.0, -1.0])
    # Facing the net, the hitter's right is +x at the near end, -x at the far end.
    right = np.array([1.0, 0.0]) if side < 0 else np.array([-1.0, 0.0])
    lat = right if racket_hand == "right" else -right
    return fwd, lat


@dataclass
class Offsets:
    forward_m: float
    lateral_m: float
    height_m: float
    forward_sigma_m: float
    lateral_sigma_m: float
    height_sigma_m: float


def offsets(
    contact: np.ndarray,
    contact_cov: np.ndarray | None,
    toe: np.ndarray,
    toe_cov: np.ndarray | None,
    side: int,
    racket_hand: str,
) -> Offsets:
    """The contact point relative to the toe in the hitter's frame, with σ."""
    fwd, lat = hitter_axes(side, racket_hand)
    d = np.asarray(contact, float) - np.asarray(toe, float)
    cov = np.zeros((3, 3))
    if contact_cov is not None and np.isfinite(contact_cov).all():
        cov = cov + contact_cov
    if toe_cov is not None and np.isfinite(toe_cov).all():
        cov = cov + toe_cov
    A = np.array([[fwd[0], fwd[1], 0.0], [lat[0], lat[1], 0.0], [0.0, 0.0, 1.0]])
    s = np.sqrt(np.maximum(np.diag(A @ cov @ A.T), 0.0))
    # The height is the ball's above the ground; its σ is the contact's alone.
    hs = float(np.sqrt(max(contact_cov[2, 2], 0.0))) if contact_cov is not None else float(s[2])
    return Offsets(
        float(fwd @ d[:2]), float(lat @ d[:2]), float(contact[2]), float(s[0]), float(s[1]), hs
    )


def toe_to_baseline(toe: np.ndarray, side: int) -> float:
    """How far behind the baseline the toe is (m; negative: inside the court)."""
    return (
        float(-court_model.HALF_LENGTH - toe[1])
        if side < 0
        else float(toe[1] - court_model.HALF_LENGTH)
    )


# ---------------------------------------------------------------------------
# The front toe in the contact frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToeParams:
    #: The big-toe keypoint sits this high (m) above the sole: its ray meets the ground
    #: further on than the toe (≈1.9 cm per cm of height near the camera).
    kp_height_m: float = 0.02
    #: The shoe's tip is this far (m) ahead of the big-toe keypoint, along the foot.
    tip_ahead_m: float = 0.028
    #: Below this keypoint score (RTMW runs ≈0-10) the toe is estimated from the heel.
    min_score: float = 3.0
    #: The toe is smoothed over this many frames on each side of the contact frame.
    smooth_frames: int = 2
    #: On the ground: the toe's ground point moved less than this (m) over the last
    #: ``still_frames`` frames up to the contact (median of the first half vs the second, so
    #: one jittery keypoint doesn't decide). A lifted foot moves several times that.
    still_frames: int = 6
    still_m: float = 0.03
    #: A lifted toe is placed on the ground with this height uncertainty (m).
    lifted_sigma_m: float = 0.05
    #: Keypoint noise (px at 3840 wide) behind the toe's σ, and the spread of where the shoe
    #: tip is relative to the keypoint (m).
    kp_sigma_px: float = 2.0
    tip_sigma_m: float = 0.015
    #: Foot length (heel keypoint to shoe tip) per metre of height when the profile has no
    #: shoe length.
    foot_per_height: float = 0.15

    def as_config(self) -> dict:
        return asdict(self)


@dataclass
class FeetSeries:
    """One serve window's foot keypoints (``serve_feet``) sorted by frame."""

    frame: np.ndarray
    t: np.ndarray
    kp: np.ndarray  # (n, len(FOOT), 3): x, y px, score

    def at(self, frame: int, half: int = 0) -> np.ndarray:
        """Indices of frames within ``half`` frames of ``frame``."""
        return np.flatnonzero(np.abs(self.frame - frame) <= half)


@dataclass
class Toe:
    pos: np.ndarray  # (3,) court m; z is the toe's height
    px: np.ndarray  # (2,) the toe tip in the contact frame (image)
    cov: np.ndarray  # (3, 3)
    sigma_m: float
    source: str  # toe | heel_length | edit
    on_ground: bool | None
    moved_m: float | None = None
    flags: list[str] = field(default_factory=list)


def _ground_cov(cam: Camera, xy: np.ndarray, px_sigma: float, z: float = 0.0) -> np.ndarray:
    """2×2 covariance (m²) of a point on the plane ``z`` from isotropic pixel noise."""
    px = cam.project(np.array([[xy[0], xy[1], z]]))[0]
    ex, ey = np.array([0.5, 0.0]), np.array([0.0, 0.5])
    gx = cam.image_to_ground(px + ex, z) - cam.image_to_ground(px - ex, z)
    gy = cam.image_to_ground(px + ey, z) - cam.image_to_ground(px - ey, z)
    J = np.stack([gx[:2], gy[:2]], axis=1)
    return px_sigma**2 * J @ J.T


def _toe_ground(
    cam: Camera, kp: np.ndarray, toe_i: int, heel_i: int, p: ToeParams, z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shoe tip on the plane ``z`` from one frame's keypoints (the big toe's ray at its
    height, moved ahead along the foot), the heel's ground point, and the heel → toe
    direction on the ground."""
    toe = cam.image_to_ground(kp[toe_i, :2], z + p.kp_height_m)[:2]
    heel = cam.image_to_ground(kp[heel_i, :2], 0.0)[:2]
    u = toe - heel
    n = float(np.linalg.norm(u))
    u = u / n if np.isfinite(n) and n > 1e-6 else np.zeros(2)
    return toe + p.tip_ahead_m * u, heel, u


def toe_at_contact(
    feet: FeetSeries,
    cam: Camera,
    front: str,
    frame: int,
    p: ToeParams | None = None,
    *,
    foot_length_m: float = 0.27,
    release_frame: int | None = None,
    edit_px: np.ndarray | None = None,
) -> Toe | None:
    """The front foot's toe tip in the contact frame (module docstring).

    ``edit_px``: the toe tip placed by the user in the contact frame.
    """
    from swingvision.pose.feet import foot_names

    p = p or ToeParams()
    _ank_i, toe_i, _small_i, heel_i = foot_names(front)
    near = feet.at(frame, p.smooth_frames)
    if not len(near):
        return None
    kp = np.median(feet.kp[near], axis=0)
    score = float(kp[toe_i, 2])
    flags: list[str] = []
    noise = p.kp_sigma_px * cam.width / 3840
    on_ground = toe_still(feet, cam, toe_i, heel_i, frame, p)
    if on_ground is False:
        flags.append("toe_lifted")
    if edit_px is not None:
        px = np.asarray(edit_px, float)
        g = cam.image_to_ground(px, 0.0)[:2]
        source = "edit"
        cov2 = _ground_cov(cam, g, noise)
    else:
        tip, heel, u = _toe_ground(cam, kp, toe_i, heel_i, p, 0.0)
        source, g = "toe", tip
        if score < p.min_score or not np.isfinite(tip).all():
            # The shoe hides the toe tip: the heel plus the foot length along the foot.
            if not np.isfinite(heel).all() or not u.any():
                return None
            g = heel + foot_length_m * u
            source = "heel_length"
            flags.append("toe_estimated")
        px = cam.project(np.array([[g[0], g[1], 0.0]]))[0]
        cov2 = _ground_cov(cam, g, noise, p.kp_height_m) + np.eye(2) * p.tip_sigma_m**2
        if source == "heel_length":
            cov2 = cov2 + np.eye(2) * 0.03**2
    cov = np.zeros((3, 3))
    cov[:2, :2] = cov2
    cov[2, 2] = 0.01**2
    if on_ground is False:
        # Off the ground the toe is on its ray at an unknown height: along the ray's
        # horizontal direction (about forward / backward) it is uncertain by the height's share.
        d = cam.rays(px[None])[0]
        h = float(np.hypot(d[0], d[1]))
        along = np.array([d[0], d[1]]) / max(h, 1e-9)
        cov[:2, :2] += np.outer(along, along) * (p.lifted_sigma_m * h / max(1e-6, -d[2])) ** 2
        cov[2, 2] = p.lifted_sigma_m**2
    sigma = float(np.sqrt(np.linalg.eigvalsh(cov[:2, :2])[-1]))
    toe = Toe(np.array([g[0], g[1], 0.0]), px, cov, sigma, source, on_ground, flags=flags)
    if release_frame is not None:
        rel = feet.at(release_frame, p.smooth_frames)
        if len(rel):
            r = _toe_ground(cam, np.median(feet.kp[rel], axis=0), toe_i, heel_i, p, 0.0)[0]
            ref = _toe_ground(cam, kp, toe_i, heel_i, p, 0.0)[0] if source == "edit" else g
            if np.isfinite(r).all() and np.isfinite(ref).all():
                toe.moved_m = float(np.hypot(*(ref - r)))
    return toe


def toe_still(
    feet: FeetSeries, cam: Camera, toe_i: int, heel_i: int, frame: int, p: ToeParams
) -> bool | None:
    """The on-ground test: the toe's ground point didn't move over the frames before the
    contact (``None``: too few frames)."""
    idx = np.flatnonzero((feet.frame <= frame) & (feet.frame > frame - p.still_frames))
    if len(idx) < max(4, p.still_frames - 2):
        return None
    pts = np.array([_toe_ground(cam, feet.kp[i], toe_i, heel_i, p, 0.0)[0] for i in idx])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 4:
        return None
    h = len(pts) // 2
    return bool(np.hypot(*(np.median(pts[:h], axis=0) - np.median(pts[h:], axis=0))) <= p.still_m)


def serve_side_of(x: float | None, side: int, min_x: float = 0.15) -> str | None:
    """Deuce when the server stands right of the center mark (in their own frame)."""
    if x is None or not np.isfinite(x) or abs(x) < min_x:
        return None
    return "deuce" if -side * x > 0 else "ad"


def toss_to_dict(c: TossContact) -> dict:
    """A :class:`TossContact` as JSON."""

    def arr(a):
        return None if a is None else np.asarray(a, float).tolist()

    return {
        "frame": c.frame,
        "t": c.t,
        "pos": arr(c.pos),
        "cov": arr(c.cov),
        "source": c.source,
        "flags": list(c.flags),
        "theta": arr(c.theta),
        "theta_cov": arr(c.theta_cov),
        "t0": c.t0,
        "t_release": c.t_release,
        "n_toss": c.n_toss,
        "toss_rms_px": c.toss_rms_px,
        "last_on": c.last_on,
        "first_off": c.first_off,
        "px": arr(c.px),
    }


def toss_from_dict(d: dict) -> TossContact:
    def arr(k):
        return None if d.get(k) is None else np.asarray(d[k], float)

    return TossContact(
        frame=d.get("frame"),
        t=d.get("t"),
        pos=arr("pos"),
        cov=arr("cov"),
        source=d.get("source"),
        flags=list(d.get("flags") or []),
        theta=arr("theta"),
        theta_cov=arr("theta_cov"),
        t0=d.get("t0"),
        t_release=d.get("t_release"),
        n_toss=int(d.get("n_toss") or 0),
        toss_rms_px=d.get("toss_rms_px"),
        last_on=d.get("last_on"),
        first_off=d.get("first_off"),
        px=arr("px"),
    )


def toss_fit(c: TossContact) -> ph.FlightFit | None:
    """The toss path of a contact (for drawing it and sampling it at other frames)."""
    if c.theta is None or c.t0 is None:
        return None
    return ph.FlightFit(True, "", c.theta, c.theta_cov, c.t0, c.t or c.t0, c.n_toss, 0.0, 0.0)
