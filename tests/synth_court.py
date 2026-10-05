"""Render synthetic court images from a known camera (for calibration tests)."""

from __future__ import annotations

import cv2
import numpy as np

from swingvision.court import model
from swingvision.court.camera import Camera, look_at, rotation_to_rvec

RED = (178, 82, 100)
GREEN = (72, 122, 96)
TEAL = (110, 190, 185)
WHITE = (232, 232, 236)


def make_camera(
    width: int = 1920,
    height: int = 1080,
    hfov_deg: float = 80.0,
    height_m: float = 3.2,
    behind_m: float = 6.0,
    offset_x: float = 0.3,
    look_y: float = 1.0,
    k1: float = 0.02,
    k2: float = -0.01,
    principal_offset: tuple[float, float] = (0.0, 0.0),
) -> Camera:
    f = width / (2 * np.tan(np.radians(hfov_deg) / 2))
    C = np.array([offset_x, -model.HALF_LENGTH - behind_m, height_m])
    R = look_at(C, np.array([0.0, look_y, 0.0]))
    return Camera(
        width, height, float(f), k1, rotation_to_rvec(R), -R @ C,
        width / 2 + principal_offset[0], height / 2 + principal_offset[1], k2,
    )  # fmt: skip


def _poly(cam: Camera, pts: np.ndarray, ss: int) -> np.ndarray | None:
    pts = pts[cam.depth(pts) > 0.05]
    if len(pts) < 3:
        return None
    px = cam.project(pts)
    px = np.clip(px[np.isfinite(px).all(axis=1)], -1e5, 1e5)
    if len(px) < 3:
        return None
    return np.round(px * ss * 16).astype(np.int32)  # 4 fractional bits


def _strip(p0, p1, half_w: float, n: int = 200) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    nrm = np.array([-d[1], d[0], 0.0]) * half_w
    s = np.linspace(0, 1, n)[:, None]
    center = (1 - s) * p0 + s * p1
    return np.concatenate([center + nrm, (center - nrm)[::-1]])


def _rect(x0, y0, x1, y1, n: int = 60) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    a, b, c, d = (np.array((*p, 0.0)) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    return np.concatenate(
        [(1 - t) * a + t * b, (1 - t) * b + t * c, (1 - t) * c + t * d, (1 - t) * d + t * a]
    )


def render(
    cam: Camera,
    distractors: bool = True,
    noise: float = 3.0,
    ss: int = 2,
    seed: int = 0,
    occluder: bool = True,
) -> np.ndarray:
    """RGB uint8 image of a court (plus pickleball lines, a neighboring court and a
    "player") seen by ``cam``. Rendered ``ss``× supersampled for sub-pixel accuracy."""
    W, H = cam.width * ss, cam.height * ss
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (120, 150, 120)  # grass/sky beyond the courts

    def fill(pts, color):
        poly = _poly(cam, pts, ss)
        if poly is not None:
            cv2.fillPoly(img, [poly], color, lineType=cv2.LINE_AA, shift=4)

    fill(_rect(-30, -30, 30, 40), RED)
    fill(_rect(-model.HALF_DOUBLES - 1.5, -model.HALF_LENGTH - 4, model.HALF_DOUBLES + 1.5,
               model.HALF_LENGTH + 4), GREEN)  # fmt: skip
    half = model.LINE_WIDTH / 2
    if distractors:
        # Neighboring court to the right, and teal pickleball lines inside ours.
        dx = model.DOUBLES_WIDTH + 7.0
        fill(_rect(dx - model.HALF_DOUBLES - 1.5, -model.HALF_LENGTH - 4, dx + model.HALF_DOUBLES + 1.5,
                   model.HALF_LENGTH + 4), GREEN)  # fmt: skip
        for line in model.COURT_LINES:
            fill(_strip(np.add(line.p0, (dx, 0, 0)), np.add(line.p1, (dx, 0, 0)), half), WHITE)
        for y in (2.13, -2.13, 6.70, -6.70):
            fill(_strip((-3.05, y, 0), (3.05, y, 0), half), TEAL)
        for x in (-3.05, 3.05):
            fill(_strip((x, -6.7, 0), (x, 6.7, 0), half), TEAL)
    for line in model.COURT_LINES:
        fill(_strip(line.p0, line.p1, half), WHITE)
    # Net: dark mesh band under a white tape.
    for x0, x1 in ((-model.X_NET_POST, 0.0), (0.0, model.X_NET_POST)):
        h0 = model.NET_HEIGHT_POST if x0 < 0 else model.NET_HEIGHT_CENTER
        h1 = model.NET_HEIGHT_CENTER if x0 < 0 else model.NET_HEIGHT_POST
        s = np.linspace(0, 1, 50)[:, None]
        top = np.column_stack([x0 + (x1 - x0) * s[:, 0], np.zeros(50), h0 + (h1 - h0) * s[:, 0]])
        mesh = np.concatenate([top - [0, 0, 0.06], np.column_stack([top[::-1, :2], np.zeros(50)])])
        fill(mesh, (40, 45, 45))
        tape = np.concatenate([top, (top - [0, 0, 0.06])[::-1]])
        fill(tape, WHITE)
    if occluder:
        # A "player" standing on the near service line.
        fill(
            np.array([[-0.3, -6.3, 0], [0.3, -6.3, 0], [0.3, -6.3, 1.8], [-0.3, -6.3, 1.8]]),
            (60, 200, 60),
        )
    out = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA).astype(np.float32)
    rng = np.random.default_rng(seed)
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def keypoint_error(a: Camera, b: Camera) -> float:
    """RMS distance (px) between the in-frame ground keypoints of two cameras."""
    X = model.keypoint_array(list(model.GROUND_KEYPOINTS))
    pa, pb = a.project(X), b.project(X)
    m = (pa[:, 0] >= 0) & (pa[:, 0] < a.width) & (pa[:, 1] >= 0) & (pa[:, 1] < a.height)
    return float(np.sqrt(np.mean(np.sum((pa[m] - pb[m]) ** 2, axis=1))))
