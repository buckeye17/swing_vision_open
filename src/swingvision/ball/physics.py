"""3D ball flight from one calibrated camera (PLAN.md §7.6).

A *flight* is the ball's path between two events: from a hit or bounce (or the start of a
tracked run: a toss, a drop, a missed contact) to the next bounce, net contact or hit (or
where the track ends). Its 3D path is fitted to the 2D detections through the calibrated
camera, with a physical model:

    a = g + drag + Magnus,   drag = -k(C_d) |v| v,   Magnus = s |v| (ê × v)

``k = ½ ρ C_d A / m`` (57 g ball, 6.7 cm diameter; ``C_d`` has a prior of 0.55 ± 0.05, so
k ≈ 0.020 /m). ``ê`` is the horizontal unit vector to the *left* of the initial direction
of travel, so ``s > 0`` pushes the ball down (topspin) and ``s < 0`` lifts it (slice,
backspin). ``s`` has a prior around 0; its sign is reported only when the data demand it.

**Why depth is recoverable from one camera.** Each detection gives a ray; where the ball is
along the rays comes from physics (gravity is known, so how far a curve falls in the image
fixes its distance) and from the ends: a bounce is at ball-radius height, so its pixel is a
3D point, and a contact is where the incoming flight ended (or near the hitter's tracked
feet). The calibration supplies the metric scale; a fit with ``g`` free recovers 9.8 m/s²
on real footage (``docs/m4-ball-3d.md``).

**Fit.** A linear ballistic solve (no drag: each ray constraint ``d × (p(t) − C) = 0`` is
linear in the initial position and velocity) starts a robust nonlinear least-squares fit
of the full model to reprojection errors, end constraints and priors. The Jacobian comes
from integrating all perturbed parameter sets in one batched RK4 pass; output
uncertainties from sampled parameter sets (fit covariance), integrated in one batch too.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from swingvision.court import model as court_model
from swingvision.court.camera import Camera

G = 9.81
BALL_MASS = 0.057  # kg (ITF: 56.0-59.4 g)
BALL_RADIUS = 0.0335  # m (diameter 6.54-6.86 cm)
AIR_DENSITY = 1.2  # kg/m³ (20 °C, sea level)
DRAG_CD = 0.55
K_PER_CD = 0.5 * AIR_DENSITY * np.pi * BALL_RADIUS**2 / BALL_MASS
K_DRAG = K_PER_CD * DRAG_CD

#: Parameters: initial position (3), initial velocity (3), Magnus coefficient, drag
#: coefficient, and gravity when it's fitted too.
N_BASE = 8
I_SPIN, I_CD, I_G = 6, 7, 8

START_KINDS = ("hit", "machine", "bounce", "free")
END_KINDS = ("bounce", "net", "hit")


@dataclass(frozen=True)
class FitParams:
    noise_px: float = 1.5  # detection jitter at 3840 px wide
    blur_frac: float = 0.15  # extra σ per pixel of image motion per frame (streak centers)
    end_px: float = 2.0  # σ of an event's refined contact pixel (at 3840 px)
    ground_sigma_m: float = 0.02  # σ of the ball's height at a bounce
    net_sigma_m: float = 0.08  # σ of the ball's distance from the net plane at a net contact
    reach_sigma_m: float = 1.0  # σ of a contact point around the hitter's feet
    machine_sigma_m: float = 0.4  # σ of a feed's start around the ball machine
    contact_z: float = 1.2  # prior contact height (m) ...
    contact_z_sigma: float = 1.0  # ... and its σ (serves ~2.7 m, low volleys ~0.4 m)
    contact_sigma_min_m: float = 0.05  # floor for a contact from the incoming flight
    spin_sigma: float = 0.01  # prior σ of the Magnus coefficient (1/m; C_L 0.3 ≈ 0.011)
    cd_sigma: float = 0.05  # prior σ of the drag coefficient
    loss_scale: float = 2.5  # soft-L1 knee in σ units
    max_step_s: float = 1 / 30  # RK4 step (< 0.1 µm error over 1.5 s)
    min_points: int = 5
    #: A fit whose reprojection RMS (at 3840 px) is above this didn't follow one ballistic
    #: path (a missed contact inside the flight, a wrong link): not ok, flagged ``poor_fit``.
    max_rms_px: float = 4.0
    #: Faster than any serve ever measured (263 km/h): the depth wasn't constrained.
    max_speed: float = 72.0
    #: An end the fitted path misses by more than this (px at 3840) wasn't on this path.
    max_end_px: float = 12.0
    outlier_sigma: float = 5.0  # points beyond this are dropped before a refit
    n_samples: int = 48  # parameter samples for output uncertainties

    def as_config(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def spin_axis(v0: np.ndarray) -> np.ndarray:
    """Unit horizontal vectors (B, 3) to the left of the horizontal directions of ``v0``."""
    v0 = np.atleast_2d(v0)
    h = np.stack([-v0[:, 1], v0[:, 0], np.zeros(len(v0))], axis=-1)
    n = np.linalg.norm(h, axis=-1, keepdims=True)
    return np.where(n > 1e-6, h / np.maximum(n, 1e-12), np.array([1.0, 0.0, 0.0]))


def _accel(v: np.ndarray, kd: np.ndarray, s: np.ndarray, ex, ey, g) -> np.ndarray:
    """Acceleration for velocities ``v`` (3, B); the spin axis (ex, ey, 0) is horizontal."""
    vx, vy, vz = v
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    d = kd * speed
    m = s * speed
    # ê × v with ê_z = 0: (e_y v_z, -e_x v_z, e_x v_y - e_y v_x)
    return np.stack(
        [
            -d * vx + m * ey * vz,
            -d * vy - m * ex * vz,
            -d * vz + m * (ex * vy - ey * vx) - g,
        ]
    )


def simulate(
    theta: np.ndarray, t0: float, times: np.ndarray, max_step: float = 1 / 30
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate parameter sets ``theta`` (B, 8 or 9) from ``t0`` to sorted ``times`` (N,).

    Returns positions and velocities (B, N, 3). Times before ``t0`` are integrated
    backwards (from ``t0``, in descending order).
    """
    theta = np.atleast_2d(np.asarray(theta, np.float64))
    B = len(theta)
    times = np.asarray(times, np.float64)
    s = theta[:, I_SPIN].copy()
    kd = K_PER_CD * theta[:, I_CD]
    g = theta[:, I_G].copy() if theta.shape[1] > N_BASE else np.full(B, G)
    e = spin_axis(theta[:, 3:6])
    ex, ey = e[:, 0].copy(), e[:, 1].copy()
    pos = np.empty((B, len(times), 3))
    vel = np.empty((B, len(times), 3))

    def run(order: np.ndarray, p: np.ndarray, v: np.ndarray) -> None:
        t = t0
        for k in order:
            dt_total = times[k] - t
            n = max(1, int(np.ceil(abs(dt_total) / max_step - 1e-9))) if dt_total else 0
            h = dt_total / n if n else 0.0
            for _ in range(n):
                a1 = _accel(v, kd, s, ex, ey, g)
                a2 = _accel(v + 0.5 * h * a1, kd, s, ex, ey, g)
                a3 = _accel(v + 0.5 * h * a2, kd, s, ex, ey, g)
                a4 = _accel(v + h * a3, kd, s, ex, ey, g)
                p = p + h * (v + h / 6 * (a1 + a2 + a3))
                v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            t = times[k]
            pos[:, k] = p.T
            vel[:, k] = v.T

    p0 = theta[:, 0:3].T.copy()
    v0 = theta[:, 3:6].T.copy()
    # Wild trial parameters during a fit may overflow; their residuals become large.
    with np.errstate(over="ignore", invalid="ignore"):
        run(np.flatnonzero(times >= t0), p0, v0)
        run(np.flatnonzero(times < t0)[::-1], p0, v0)
    return pos, vel


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass
class EndPoint:
    """What is known about one end of a flight."""

    kind: str  # start: hit | machine | bounce | free; end: bounce | net | hit
    t: float
    #: Refined contact pixel. Used for bounce/net ends and hit ends (a hit *start* is
    #: constrained by ``contact`` or ``near_xy`` instead: its pixel is often on the racket).
    px: np.ndarray | None = None
    #: Hitter's feet (or the ball machine) on the court, m, for a hit/machine start.
    near_xy: np.ndarray | None = None
    #: Contact point (3,) and covariance (3, 3) from the incoming flight's end.
    contact: np.ndarray | None = None
    contact_cov: np.ndarray | None = None


@dataclass
class FlightData:
    camera: Camera
    t: np.ndarray  # (N,) observation times
    px: np.ndarray  # (N, 2) detected ball pixels
    sigma: np.ndarray  # (N,) 1-σ in px
    start: EndPoint
    end: EndPoint | None = None
    keep: np.ndarray | None = None  # observations in use (after outlier rejection)

    @property
    def scale(self) -> float:
        return self.camera.width / 3840


@dataclass
class FlightFit:
    ok: bool
    message: str
    theta: np.ndarray
    cov: np.ndarray | None
    t0: float
    t1: float  # end of the flight (end event, or the last observation)
    n_obs: int
    rms_px: float
    chi2_dof: float
    outputs: dict = field(default_factory=dict)
    sigmas: dict = field(default_factory=dict)
    #: Position at ``t1`` and its covariance (the next flight's contact prior).
    end_pos: np.ndarray | None = None
    end_cov: np.ndarray | None = None
    flags: list[str] = field(default_factory=list)


def point_sigma(px: np.ndarray, t: np.ndarray, width: int, p: FitParams) -> np.ndarray:
    """Per-detection σ: jitter plus a share of the frame-to-frame motion (blurred streaks)."""
    scale = width / 3840
    n = len(t)
    motion = np.zeros(n)
    if n >= 2:
        d = np.hypot(*np.diff(px, axis=0).T) / np.maximum(np.diff(t) * 60, 1e-3)
        motion[1:] = d
        motion[:-1] = np.maximum(motion[:-1], d)
    return np.hypot(p.noise_px * scale, p.blur_frac * motion)


def _start_priors(ep: EndPoint, p: FitParams) -> list[tuple[np.ndarray, np.ndarray]]:
    """Priors on a hit/machine start position: (mean (3,), W (3, 3)) with Wᵀ W = cov⁻¹.

    The incoming flight's end (when it was tracked) and the hitter's feet are independent
    evidence; both apply.
    """
    if ep.kind not in ("hit", "machine"):
        return []
    out = []
    if ep.contact is not None and ep.contact_cov is not None:
        cov = ep.contact_cov + np.eye(3) * p.contact_sigma_min_m**2
        with contextlib.suppress(np.linalg.LinAlgError):
            out.append((np.asarray(ep.contact, float), np.linalg.cholesky(np.linalg.inv(cov)).T))
    if ep.near_xy is not None:
        s_xy = p.machine_sigma_m if ep.kind == "machine" else p.reach_sigma_m
        mean = np.array([ep.near_xy[0], ep.near_xy[1], p.contact_z])
        out.append((mean, np.diag([1 / s_xy, 1 / s_xy, 1 / p.contact_z_sigma])))
    return out


def _uses_px(ep: EndPoint, is_start: bool) -> bool:
    return ep.px is not None and not (is_start and ep.kind in ("hit", "machine"))


# ---------------------------------------------------------------------------
# Initialization: linear ballistic solve
# ---------------------------------------------------------------------------


def _skew(d: np.ndarray) -> np.ndarray:
    z = np.zeros(len(d))
    return np.stack(
        [
            np.stack([z, -d[:, 2], d[:, 1]], axis=1),
            np.stack([d[:, 2], z, -d[:, 0]], axis=1),
            np.stack([-d[:, 1], d[:, 0], z], axis=1),
        ],
        axis=1,
    )


def linear_init(data: FlightData, p: FitParams, g: float = G) -> np.ndarray:
    """Drag-free (p0, v0) from rays + end constraints + priors, by linear least squares."""
    cam = data.camera
    C = cam.center
    t_ref = data.start.t
    rows, rhs = [], []
    depth = 15.0  # rough ray distance: ray residuals are metric, weight ≈ f / (σ_px · depth)

    def add_rays(px: np.ndarray, t: np.ndarray, sig_px: np.ndarray):
        d = cam.rays(px)
        ok = np.isfinite(d).all(axis=1)
        d, t, w = d[ok], t[ok], cam.f / (sig_px[ok] * depth)
        S = _skew(d)  # S @ x = d × x
        tau = t - t_ref
        A = np.concatenate([S, S * tau[:, None, None]], axis=2)  # (n, 3, 6)
        b = np.einsum("nij,nj->ni", S, C + np.outer(0.5 * g * tau**2, [0, 0, 1.0]))
        rows.append((A * w[:, None, None]).reshape(-1, 6))
        rhs.append((b * w[:, None]).reshape(-1))

    def add_row(coef: np.ndarray, value: float, sigma: float):
        rows.append(coef[None] / sigma)
        rhs.append(np.array([value / sigma]))

    add_rays(data.px, data.t, data.sigma)
    for is_start, ep in ((True, data.start), (False, data.end)):
        if ep is None:
            continue
        tau = ep.t - t_ref
        if _uses_px(ep, is_start):
            add_rays(ep.px[None], np.array([ep.t]), np.array([p.end_px * data.scale]))
        if ep.kind == "bounce":  # z(τ) = r
            add_row(
                np.array([0, 0, 1, 0, 0, tau]), BALL_RADIUS + 0.5 * g * tau**2, p.ground_sigma_m
            )
        elif ep.kind == "net":  # y(τ) = 0
            add_row(np.array([0, 1, 0, 0, tau, 0]), 0.0, p.net_sigma_m)
        for mean, W in _start_priors(ep, p) if is_start else []:
            rows.append(np.concatenate([W, np.zeros((3, 3))], axis=1))
            rhs.append(W @ mean)
    x, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)
    return x


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _residuals_batch(theta: np.ndarray, data: FlightData, p: FitParams) -> np.ndarray:
    """Normalized residuals (B, M) for parameter sets (B, P)."""
    keep = data.keep if data.keep is not None else np.ones(len(data.t), bool)
    t_obs = data.t[keep]
    ends = [(True, data.start)] + ([(False, data.end)] if data.end is not None else [])
    t_all = np.concatenate([t_obs, [ep.t for _, ep in ends]])
    order = np.argsort(t_all, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    pos_sorted, _ = simulate(theta, data.start.t, t_all[order], p.max_step_s)
    pos = pos_sorted[:, inv]
    B = len(theta)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _residual_parts(theta, pos, t_obs, keep, ends, data, p, B)


def _residual_parts(theta, pos, t_obs, keep, ends, data: FlightData, p: FitParams, B: int):
    cam = data.camera
    proj = cam.project(pos[:, : len(t_obs)].reshape(-1, 3)).reshape(B, len(t_obs), 2)
    parts = [((proj - data.px[keep][None]) / data.sigma[keep][None, :, None]).reshape(B, -1)]
    for j, (is_start, ep) in enumerate(ends):
        P = pos[:, len(t_obs) + j]
        if _uses_px(ep, is_start):
            parts.append((cam.project(P) - ep.px[None]) / (p.end_px * data.scale))
        if ep.kind == "bounce":
            parts.append(((P[:, 2] - BALL_RADIUS) / p.ground_sigma_m)[:, None])
        elif ep.kind == "net":
            parts.append((P[:, 1] / p.net_sigma_m)[:, None])
        for mean, W in _start_priors(ep, p) if is_start else []:
            parts.append((theta[:, 0:3] - mean[None]) @ W.T)
    parts.append((theta[:, I_SPIN] / p.spin_sigma)[:, None])
    parts.append(((theta[:, I_CD] - DRAG_CD) / p.cd_sigma)[:, None])
    out = np.concatenate(parts, axis=1)
    return np.clip(np.nan_to_num(out, nan=1e3), -1e4, 1e4)


def _px_errors(theta: np.ndarray, data: FlightData, p: FitParams) -> np.ndarray:
    keep = data.keep if data.keep is not None else np.ones(len(data.t), bool)
    t = data.t[keep]
    if not len(t):
        return np.zeros(0)
    pos, _ = simulate(theta[None], data.start.t, t, p.max_step_s)
    return np.hypot(*(data.camera.project(pos[0]) - data.px[keep]).T)


def fit_flight(
    data: FlightData, params: FitParams | None = None, free_g: bool = False
) -> FlightFit:
    """Fit a 3D flight (see module docstring). ``free_g`` also fits gravity (validation)."""
    p = params or FitParams()
    t_last = float(data.t[-1]) if len(data.t) else data.start.t
    t1 = data.end.t if data.end is not None else t_last
    if len(data.t) < p.min_points:
        n = N_BASE + int(free_g)
        return FlightFit(
            False, "too few points", np.full(n, np.nan), None, data.start.t, t1, len(data.t),
            np.nan, np.nan, flags=["few_points"],
        )  # fmt: skip
    x0 = linear_init(data, p)
    theta0 = np.concatenate([x0, [0.0, DRAG_CD], [G] if free_g else []])
    steps = np.r_[np.full(6, 1e-4), 1e-5, 1e-4, [1e-4] if free_g else []]

    def fun(th):
        return _residuals_batch(th[None], data, p)[0]

    def jac(th):
        r = _residuals_batch(np.vstack([th, th + np.diag(steps)]), data, p)
        return ((r[1:] - r[0]) / steps[:, None]).T

    data.keep = None
    sol = None
    for _ in range(3):
        sol = least_squares(
            fun, theta0, jac=jac, loss="soft_l1", f_scale=p.loss_scale, max_nfev=50, x_scale="jac"
        )
        theta0 = sol.x
        # Drop gross outliers (a candidate on the racket, another ball) and refit.
        keep = data.keep if data.keep is not None else np.ones(len(data.t), bool)
        idx = np.flatnonzero(keep)
        bad = _px_errors(sol.x, data, p) / data.sigma[idx] > p.outlier_sigma
        if not bad.any() or len(idx) - bad.sum() < p.min_points:
            break
        keep = keep.copy()
        keep[idx[bad]] = False
        data.keep = keep
    assert sol is not None
    theta = sol.x
    err = _px_errors(theta, data, p)
    J = sol.jac
    dof = max(1, J.shape[0] - J.shape[1])
    chi2 = float(np.sum(sol.fun**2)) / dof
    try:
        cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2)
    except np.linalg.LinAlgError:
        cov = None
    fit = FlightFit(
        bool(sol.success), str(sol.message), theta, cov, data.start.t, t1, len(err),
        float(np.sqrt(np.mean(err**2))) if len(err) else np.nan, chi2,
    )  # fmt: skip
    if data.keep is not None and (~data.keep).any():
        fit.flags.append("outliers_dropped")
    if not fit.rms_px <= p.max_rms_px * data.scale:
        fit.ok = False
        fit.flags.append("poor_fit")
    ends_on_ground = data.end is not None and data.end.kind == "bounce"
    fit.outputs, fit.sigmas, fit.end_pos, fit.end_cov = flight_outputs(
        theta, cov, fit.t0, fit.t1, ends_on_ground, p
    )
    if not fit.outputs["speed0"] <= p.max_speed:
        fit.ok = False
        fit.flags.append("implausible")
    end = data.end
    if end is not None and end.px is not None and np.isfinite(fit.end_pos).all():
        miss = np.hypot(*(data.camera.project(fit.end_pos[None])[0] - end.px))
        far_off_ground = end.kind == "bounce" and abs(fit.end_pos[2] - BALL_RADIUS) > 0.25
        if not miss <= p.max_end_px * data.scale or far_off_ground:
            fit.ok = False
            fit.flags.append("end_mismatch")
    return fit


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

OUTPUT_NAMES = (
    "speed0", "speed_net", "speed_end", "speed_avg", "net_x", "net_z", "net_clearance",
    "apex_z", "end_x", "end_y", "end_z", "vz0", "vz_end", "landing_x", "landing_y",
    "landing_t",
)  # fmt: skip


def net_height(x: np.ndarray | float) -> np.ndarray:
    """Height of the net's top at ``x`` (straight from the center strap to the posts)."""
    ax = np.minimum(np.abs(np.asarray(x, np.float64)), court_model.X_NET_POST)
    rise = court_model.NET_HEIGHT_POST - court_model.NET_HEIGHT_CENTER
    return court_model.NET_HEIGHT_CENTER + rise * ax / court_model.X_NET_POST


def _first_crossing(values: np.ndarray, level: float, falling_only: bool) -> tuple[int, float]:
    """Index k and fraction f where ``values`` first crosses ``level`` between k and k+1."""
    a, b = values[:-1] - level, values[1:] - level
    hit = (a > 0) & (b <= 0) if falling_only else (a * b < 0)
    ks = np.flatnonzero(hit)
    if not len(ks):
        return -1, np.nan
    k = int(ks[0])
    return k, float(a[k] / (a[k] - b[k]))


def _outputs_from_paths(pos, vel, grid, ext) -> dict[str, np.ndarray]:
    """Outputs per sample from dense paths (B, N, 3) on ``grid`` (t0 … t1)."""
    B = pos.shape[0]
    speed = np.linalg.norm(vel, axis=-1)
    out = {k: np.full(B, np.nan) for k in OUTPUT_NAMES}
    out["speed0"], out["speed_end"] = speed[:, 0], speed[:, -1]
    dur = grid[-1] - grid[0]
    if dur > 0:
        out["speed_avg"] = np.linalg.norm(np.diff(pos, axis=1), axis=-1).sum(axis=1) / dur
    out["apex_z"] = pos[:, :, 2].max(axis=1)
    out["end_x"], out["end_y"], out["end_z"] = pos[:, -1, 0], pos[:, -1, 1], pos[:, -1, 2]
    out["vz0"], out["vz_end"] = vel[:, 0, 2], vel[:, -1, 2]
    for b in range(B):
        k, f = _first_crossing(pos[b, :, 1], 0.0, falling_only=False)
        if k >= 0:
            P = pos[b, k] + f * (pos[b, k + 1] - pos[b, k])
            out["net_x"][b], out["net_z"][b] = P[0], P[2]
            out["net_clearance"][b] = P[2] - float(net_height(P[0]))
            out["speed_net"][b] = speed[b, k] + f * (speed[b, k + 1] - speed[b, k])
        if ext is not None:
            ext_pos, ext_grid = ext
            k, f = _first_crossing(ext_pos[b, :, 2], BALL_RADIUS, falling_only=True)
            if k >= 0:
                P = ext_pos[b, k] + f * (ext_pos[b, k + 1] - ext_pos[b, k])
                out["landing_x"][b], out["landing_y"][b] = P[0], P[1]
                out["landing_t"][b] = ext_grid[k] + f * (ext_grid[k + 1] - ext_grid[k])
    return out


def flight_outputs(
    theta: np.ndarray,
    cov: np.ndarray | None,
    t0: float,
    t1: float,
    ends_on_ground: bool = False,
    p: FitParams | None = None,
) -> tuple[dict, dict, np.ndarray, np.ndarray | None]:
    """Outputs at the fitted parameters, their 1-σ (sampled from ``cov``), and the end
    position with its covariance.

    A flight that ends on the ground lands at its end; otherwise the landing is where the
    path, extended up to 2 s, comes down to ball height.
    """
    p = p or FitParams()
    grid = np.linspace(t0, t1, max(2, int(np.ceil((t1 - t0) * 240)) + 1))
    thetas = theta[None]
    if cov is not None and np.isfinite(cov).all():
        try:
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(theta)))
            rng = np.random.default_rng(0)
            draws = rng.standard_normal((p.n_samples, len(theta))) @ L.T
            thetas = np.vstack([theta, theta + draws])
        except np.linalg.LinAlgError:
            pass
    pos, vel = simulate(thetas, t0, grid, p.max_step_s)
    ext = None
    if not ends_on_ground:
        ext_grid = np.linspace(t0, t1 + 2.0, int(np.ceil((t1 + 2.0 - t0) * 120)) + 1)
        ext = (simulate(thetas, t0, ext_grid, p.max_step_s)[0], ext_grid)
    # Samples of badly constrained parameters can be wild (non-finite outputs are skipped).
    with np.errstate(over="ignore", invalid="ignore"):
        out = _outputs_from_paths(pos, vel, grid, ext)
    if ends_on_ground:
        out["landing_x"], out["landing_y"] = out["end_x"], out["end_y"]
        out["landing_t"] = np.full(len(thetas), t1)
    best = {k: float(v[0]) for k, v in out.items()}
    sig = {}
    for k, v in out.items():
        s = v[1:][np.isfinite(v[1:])]
        sig[k] = float(np.std(s)) if len(s) >= 8 else np.nan
    for name, i in (("spin", I_SPIN), ("cd", I_CD), ("g", I_G)):
        if i < len(theta):
            best[name] = float(theta[i])
            sig[name] = float(np.sqrt(cov[i, i])) if cov is not None else np.nan
    end_pos = pos[0, -1].copy()
    end_cov = np.cov(pos[1:, -1].T) if len(thetas) > 8 else None
    return best, sig, end_pos, end_cov


def path_points(theta: np.ndarray, t0: float, t1: float, hz: float = 60.0) -> tuple:
    """(times, positions (n, 3)) along a fitted flight, for drawing."""
    grid = np.linspace(t0, t1, max(2, int(np.ceil((t1 - t0) * hz)) + 1))
    pos, _ = simulate(theta[None], t0, grid)
    return grid, pos[0]
