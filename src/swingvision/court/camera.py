"""Full camera model: pinhole intrinsics, radial lens distortion, and pose.

All image coordinates are **upright display pixels at full source resolution**
(what the proxy and the calibration editor show, scaled up).

Lens distortion uses the *division model* around the principal point ``c``,
with radii normalized by the image half-diagonal ``s``::

    p_u = c + (p_d - c) / (1 + k1 r_d² + k2 r_d⁴),     r_d = |p_d - c| / s

Undistortion is closed form, which keeps line-based fitting fast; distortion
(projection) is closed form for ``k2 = 0`` and takes a few Newton steps
otherwise. Barrel distortion has ``k1 < 0``. The principal point may be
off-center: phone video is often cropped for stabilization.

Fitting (:func:`fit_camera`) minimizes, jointly over focal length, distortion,
principal point and pose: reprojection error of point correspondences (court
keypoints, net points) plus perpendicular distances of observed line-center
pixels to the projected court lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector → 3×3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def rotation_to_rvec(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → axis-angle vector."""
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = float(np.arccos(cos))
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near 180°: axis from the diagonal of (R + I) / 2.
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        axis = M[:, i] / np.sqrt(M[i, i])
        return axis / np.linalg.norm(axis) * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * np.sin(theta)) * theta


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World→camera rotation for a camera at ``center`` looking at ``target`` (x right, y down)."""
    fwd = np.asarray(target, float) - np.asarray(center, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    width: int
    height: int
    f: float
    k1: float
    rvec: np.ndarray
    tvec: np.ndarray
    cx: float | None = None
    cy: float | None = None
    k2: float = 0.0

    #: Order of :meth:`params`.
    PARAM_NAMES = ("log_f", "k1", "k2", "cx", "cy", "r0", "r1", "r2", "t0", "t1", "t2")

    def __post_init__(self) -> None:
        self.rvec = np.asarray(self.rvec, dtype=np.float64).reshape(3)
        self.tvec = np.asarray(self.tvec, dtype=np.float64).reshape(3)
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0

    # -- derived ---------------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        return rodrigues(self.rvec)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @property
    def c(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float64)

    @property
    def scale(self) -> float:
        """Distortion radius normalization (image half-diagonal)."""
        return float(np.hypot(self.width, self.height) / 2)

    @property
    def center(self) -> np.ndarray:
        """Camera position in court coordinates."""
        return -self.R.T @ self.tvec

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / (2 * self.f))))

    def homography(self) -> np.ndarray:
        """Court ground plane (x, y, 1) → *undistorted* image pixels."""
        R = self.R
        H = self.K @ np.column_stack([R[:, 0], R[:, 1], self.tvec])
        return H / H[2, 2]

    # -- distortion ------------------------------------------------------------
    def distortion_factor(self, px: np.ndarray) -> np.ndarray:
        """``1 + k1 r² + k2 r⁴`` at distorted pixels (how much undistortion shrinks radii)."""
        d = np.asarray(px, np.float64) - self.c
        r2 = np.sum(d * d, axis=-1) / self.scale**2
        return 1 + self.k1 * r2 + self.k2 * r2 * r2

    def undistort(self, px: np.ndarray) -> np.ndarray:
        """Distorted → undistorted pixels. NaN where the model folds over (far outside
        the image, for strong distortion)."""
        px = np.asarray(px, dtype=np.float64)
        factor = self.distortion_factor(px)
        factor = np.where(factor > 0.05, factor, np.nan)
        return self.c + (px - self.c) / factor[..., None]

    def distort(self, px_u: np.ndarray) -> np.ndarray:
        return distort_points(np.asarray(px_u, np.float64), self.k1, self.k2, self.c, self.scale)

    # -- projection ----------------------------------------------------------
    def to_camera(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.R.T + self.tvec

    def project(self, X: np.ndarray) -> np.ndarray:
        """Court points (..., 3) → distorted image pixels (..., 2)."""
        return self.distort(self.project_undistorted(X))

    def project_undistorted(self, X: np.ndarray) -> np.ndarray:
        Xc = self.to_camera(X)
        return self.f * Xc[..., :2] / Xc[..., 2:3] + self.c

    def depth(self, X: np.ndarray) -> np.ndarray:
        return self.to_camera(X)[..., 2]

    def rays(self, px: np.ndarray) -> np.ndarray:
        """Unit view rays in court coordinates for image pixels (..., 2)."""
        u = self.undistort(px)
        d = np.concatenate([(u - self.c) / self.f, np.ones((*u.shape[:-1], 1))], axis=-1)
        w = d @ self.R  # R.T @ d for row vectors
        return w / np.linalg.norm(w, axis=-1, keepdims=True)

    def image_to_ground(self, px: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane at height ``z`` → (..., 3).

        Rays that never reach the plane (above the horizon) give NaN.
        """
        rays = self.rays(px)
        C = self.center
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (z - C[2]) / rays[..., 2]
        lam = np.where(lam > 0, lam, np.nan)
        return C + rays * lam[..., None]

    # -- serialization ---------------------------------------------------------
    def params(self) -> np.ndarray:
        return np.concatenate(
            [[np.log(self.f), self.k1, self.k2, self.cx, self.cy], self.rvec, self.tvec]
        )

    def with_params(self, p: np.ndarray) -> Camera:
        return Camera(
            self.width, self.height, float(np.exp(p[0])), float(p[1]), p[5:8], p[8:11],
            float(p[3]), float(p[4]), float(p[2]),
        )  # fmt: skip

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "f": float(self.f),
            "k1": float(self.k1),
            "k2": float(self.k2),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "rvec": [float(v) for v in self.rvec],
            "tvec": [float(v) for v in self.tvec],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Camera:
        return cls(
            int(d["width"]), int(d["height"]), float(d["f"]), float(d["k1"]),
            np.array(d["rvec"]), np.array(d["tvec"]), d.get("cx"), d.get("cy"),
            float(d.get("k2", 0.0)),
        )  # fmt: skip

    def describe(self) -> dict:
        """Human-readable summary (camera height, distance behind the near baseline, …)."""
        from swingvision.court.model import HALF_LENGTH

        C = self.center
        fwd = self.R[2]
        return {
            "height_m": float(C[2]),
            "behind_baseline_m": float(-HALF_LENGTH - C[1]),
            "offset_x_m": float(C[0]),
            "tilt_down_deg": float(np.degrees(np.arcsin(-fwd[2]))),
            "hfov_deg": self.hfov_deg,
            "f_px": float(self.f),
            "k1": float(self.k1),
            "k2": float(self.k2),
            "principal_offset_px": [
                float(self.cx - self.width / 2),
                float(self.cy - self.height / 2),
            ],
        }


def distort_points(
    px_u: np.ndarray, k1: float, k2: float, c: np.ndarray, scale: float
) -> np.ndarray:
    """Inverse of the division model: solve ``r_u = r_d / (1 + k1 r_d² + k2 r_d⁴)`` for r_d.

    Far outside the image the model can have no solution (the mapping folds over); those
    points come back as NaN.
    """
    d = px_u - c
    r_u = np.sqrt(np.sum(d * d, axis=-1, keepdims=True)) / scale
    disc = 1 - 4 * k1 * r_u * r_u
    r_d = 2 * r_u / (1 + np.sqrt(np.clip(disc, 0.0, None)))  # exact when k2 = 0
    invalid = disc < 0
    if k2 != 0.0:
        for _ in range(10):
            r2 = r_d * r_d
            den = 1 + k1 * r2 + k2 * r2 * r2
            dg = (1 - k1 * r2 - 3 * k2 * r2 * r2) / (den * den)
            r_d = r_d - (r_d / den - r_u) / np.where(np.abs(dg) > 1e-6, dg, 1e-6)
            r_d = np.clip(r_d, 0.0, 4 * r_u + 1.0)
        r2 = r_d * r_d
        den = 1 + k1 * r2 + k2 * r2 * r2
        invalid = (np.abs(r_d / den - r_u) > 1e-7 * (1 + r_u)) | (den <= 0.05)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(r_u > 1e-12, r_d / r_u, 1.0)
    factor = np.where(invalid, np.nan, factor)
    return c + d * factor


def default_camera(width: int, height: int) -> Camera:
    """A plausible starting point when nothing was detected: 3.5 m high, 5 m behind the
    near baseline, looking at the court center with a 70° horizontal field of view."""
    from swingvision.court.model import HALF_LENGTH

    f = width / (2 * np.tan(np.radians(70) / 2))
    C = np.array([0.0, -HALF_LENGTH - 5.0, 3.5])
    R = look_at(C, np.array([0.0, -2.0, 0.0]))
    return Camera(width, height, f, 0.0, rotation_to_rvec(R), -R @ C)


def camera_from_homography(H: np.ndarray, width: int, height: int) -> Camera | None:
    """Decompose a ground-plane homography (court xy → undistorted px) into a camera.

    Assumes square pixels and the principal point at the image center; focal length
    comes from the two orthonormality constraints on the rotation's first two columns.
    Returns ``None`` if the homography is inconsistent with a camera above the court.
    """
    c = np.array([width / 2.0, height / 2.0])
    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1.0]])
    Hc = T @ (H / np.linalg.norm(H))
    h1, h2 = Hc[:, 0], Hc[:, 1]
    # Linear in w = 1/f²:  w·a + b = 0
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    w = -float(a @ b) / float(a @ a) if float(a @ a) > 0 else -1.0
    f = 1 / np.sqrt(w) if w > 0 else None
    if f is None or not (0.15 * width < f < 10 * width):
        f = width / (2 * np.tan(np.radians(70) / 2))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    r1, r2, t = Kinv @ Hc[:, 0], Kinv @ Hc[:, 1], Kinv @ Hc[:, 2]
    lam = 2 / (np.linalg.norm(r1) + np.linalg.norm(r2))
    if t[2] * lam < 0:
        lam = -lam
    r1, r2, t = r1 * lam, r2 * lam, t * lam
    R = nearest_rotation(np.column_stack([r1, r2, np.cross(r1, r2)]))
    cam = Camera(width, height, float(f), 0.0, rotation_to_rvec(R), t)
    if cam.center[2] <= 0:
        return None
    return cam


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class PointObs:
    court: np.ndarray  # (N, 3)
    image: np.ndarray  # (N, 2) distorted px
    weight: np.ndarray  # (N,)


@dataclass
class LineObs:
    a: np.ndarray  # (M, 3) court-line endpoint
    b: np.ndarray  # (M, 3) other endpoint
    image: np.ndarray  # (M, 2) observed line-center pixel (distorted)
    weight: np.ndarray  # (M,)
    line_index: np.ndarray = field(default_factory=lambda: np.zeros(0, int))


@dataclass
class FitResult:
    camera: Camera
    rms_points_px: float | None
    rms_lines_px: float | None
    point_residuals: np.ndarray | None
    line_residuals: np.ndarray | None
    cov: np.ndarray | None
    success: bool
    cost: float


def line_distances(cam: Camera, lines: LineObs) -> np.ndarray:
    """Signed perpendicular distance (≈ image px) of observations to the projected lines."""
    a = cam.project_undistorted(lines.a)
    b = cam.project_undistorted(lines.b)
    ha = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    hb = np.concatenate([b, np.ones((len(b), 1))], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        lh = np.cross(ha, hb)
        lh /= np.linalg.norm(lh[:, :2], axis=1, keepdims=True)
    u = cam.undistort(lines.image)
    d = lh[:, 0] * u[:, 0] + lh[:, 1] * u[:, 1] + lh[:, 2]
    # Undistortion scales distances by ~1/factor; convert back to image pixels.
    return d * cam.distortion_factor(lines.image)


@dataclass
class FitOptions:
    fix_f: bool = False
    fit_k1: bool = True
    fit_k2: bool = False
    fit_center: bool = False
    k1_sigma: float = 0.15
    k2_sigma: float = 0.1
    center_sigma: float = 0.03  # fraction of the image width
    loss: str = "soft_l1"
    f_scale_px: float = 2.0


def fit_camera(
    init: Camera,
    points: PointObs | None = None,
    lines: LineObs | None = None,
    options: FitOptions | None = None,
) -> FitResult:
    """Least-squares camera fit to point and/or line observations (see module docstring).

    ``k1``, ``k2`` and the principal point get Gaussian priors (around 0 and the image
    center) so they stay put when the observations don't constrain them.
    """
    o = options or FitOptions()
    p0 = init.params()
    n_par = len(p0)
    free = np.ones(n_par, dtype=bool)
    free[0] = not o.fix_f
    free[1] = o.fit_k1
    free[2] = o.fit_k2
    free[3] = free[4] = o.fit_center
    c0 = np.array([init.width / 2, init.height / 2])
    c_sigma = o.center_sigma * init.width

    def unpack(x):
        p = p0.copy()
        p[free] = x
        return init.with_params(p)

    def residuals(x):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            r = _residuals(x)
        # Undefined projections (far off-frame, or a wild intermediate step) count as large.
        return np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3)

    def _residuals(x):
        cam = unpack(x)
        parts = []
        if points is not None and len(points.court):
            r = (cam.project(points.court) - points.image) * np.sqrt(points.weight)[:, None]
            parts.append(r.ravel())
        if lines is not None and len(lines.a):
            parts.append(line_distances(cam, lines) * np.sqrt(lines.weight))
        if o.fit_k1:
            parts.append(np.array([cam.k1 / o.k1_sigma]))
        if o.fit_k2:
            parts.append(np.array([cam.k2 / o.k2_sigma]))
        if o.fit_center:
            parts.append((cam.c - c0) / c_sigma)
        return np.concatenate(parts)

    sol = least_squares(
        residuals, p0[free], loss=o.loss, f_scale=o.f_scale_px, x_scale="jac", max_nfev=300
    )
    cam = unpack(sol.x)

    pres = lres = None
    rms_p = rms_l = None
    if points is not None and len(points.court):
        pres = np.linalg.norm(cam.project(points.court) - points.image, axis=1)
        w = points.weight
        rms_p = float(np.sqrt(np.sum(w * pres**2) / max(np.sum(w), 1e-12)))
    if lines is not None and len(lines.a):
        lres = line_distances(cam, lines)
        w = lines.weight
        rms_l = float(np.sqrt(np.sum(w * lres**2) / max(np.sum(w), 1e-12)))

    cov = None
    try:
        J = sol.jac
        dof = max(1, J.shape[0] - J.shape[1])
        s2 = float(np.sum(sol.fun**2)) / dof
        cov = np.zeros((n_par, n_par))
        cov[np.ix_(free, free)] = np.linalg.pinv(J.T @ J) * s2
    except (np.linalg.LinAlgError, ValueError):
        cov = None
    return FitResult(cam, rms_p, rms_l, pres, lres, cov, bool(sol.success), float(sol.cost))
