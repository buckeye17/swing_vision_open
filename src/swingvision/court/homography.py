"""Ground-plane homographies and position uncertainty (PLAN.md §7.1 steps 3 and 5)."""

from __future__ import annotations

import numpy as np

from swingvision.court.camera import Camera


def _normalizer(pts: np.ndarray) -> np.ndarray:
    mean = pts.mean(axis=0)
    d = np.sqrt(np.sum((pts - mean) ** 2, axis=1)).mean()
    s = np.sqrt(2) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * mean[0]], [0, s, -s * mean[1]], [0, 0, 1.0]])


def fit_homography(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Normalized (weighted) DLT: H such that dst ~ H @ src. Needs ≥ 4 points."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if len(src) < 4:
        raise ValueError("A homography needs at least 4 points")
    Ts, Td = _normalizer(src), _normalizer(dst)
    s = apply_h(Ts, src)
    d = apply_h(Td, dst)
    n = len(s)
    A = np.zeros((2 * n, 9))
    A[0::2, 0:2] = -s
    A[0::2, 2] = -1
    A[0::2, 6:8] = s * d[:, :1]
    A[0::2, 8] = d[:, 0]
    A[1::2, 3:5] = -s
    A[1::2, 5] = -1
    A[1::2, 6:8] = s * d[:, 1:2]
    A[1::2, 8] = d[:, 1]
    if weights is not None:
        A *= np.repeat(np.sqrt(weights), 2)[:, None]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    h = pts @ H[:, :2].T + H[:, 2]
    return h[..., :2] / h[..., 2:3]


def ground_sigma(
    camera: Camera, xy: np.ndarray, px_sigma: float = 1.0, z: float = 0.0
) -> np.ndarray:
    """1-σ ground-position uncertainty (meters) from ``px_sigma`` of image noise.

    Propagates isotropic pixel noise through the Jacobian of the image→ground mapping
    at each court point. Returns (..., 3): σ across (x), σ along (y), and the major
    semi-axis of the error ellipse. Far-court points have a large σ along y.
    """
    xy = np.atleast_2d(np.asarray(xy, np.float64))
    X = np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), z)])
    px = camera.project(X)
    eps = 0.5
    ex, ey = np.array([eps, 0.0]), np.array([0.0, eps])
    gx = (camera.image_to_ground(px + ex, z) - camera.image_to_ground(px - ex, z)) / (2 * eps)
    gy = (camera.image_to_ground(px + ey, z) - camera.image_to_ground(px - ey, z)) / (2 * eps)
    J = np.stack([gx[:, :2], gy[:, :2]], axis=2)  # (N, 2 ground, 2 pixel)
    cov = px_sigma**2 * J @ np.transpose(J, (0, 2, 1))
    major = np.sqrt(np.linalg.eigvalsh(cov)[:, -1])
    return np.column_stack([np.sqrt(cov[:, 0, 0]), np.sqrt(cov[:, 1, 1]), major])
