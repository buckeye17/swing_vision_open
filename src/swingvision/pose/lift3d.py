"""2D keypoint sequences → 3D joints on the court (PLAN.md §7.7).

1. **Lifting**: COCO keypoints become Human3.6M joints, each clip (≤ 243 frames) is
   normalized to [-1, 1] by the player's extent over the clip (as MotionBERT's in-the-wild
   inference does), and MotionBERT-Lite predicts 3D joints in that normalized camera space,
   averaged with the prediction for the mirrored clip.
2. **Scale**: root-relative joints are scaled so the bones from ankle to head add up to the
   profile's height (:data:`~swingvision.pose.skeleton.HEIGHT_PER_CHAIN`).
3. **Orientation**: the network sees a crop, i.e. a virtual camera looking straight at the
   player. That camera is rotated onto the real ray through the player's pelvis, then the
   calibrated camera's rotation takes the joints into court axes (x across, y along, z up).
4. **Position**: per frame, one translation that puts every joint on its camera ray
   (confidence- and size-weighted linear least squares), softly held to the ground (the lower
   ankle ≈ 8 cm up) and to the tracked feet; then smoothed over time. Depth along the
   camera's axis is the least certain part, especially for the far player (PLAN.md §14).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from swingvision.court.camera import Camera
from swingvision.pose.skeleton import H36M_FLIP, HEIGHT_PER_CHAIN, H, chain_length

MAX_CLIP = 243
DEFAULT_HEIGHT_M = 1.78


@dataclass(frozen=True)
class PlaceParams:
    #: Ankle joint height when standing (m), and how firmly the lower ankle is held there.
    ankle_z_m: float = 0.08
    ground_sigma_m: float = 0.15
    #: Keypoint position error as a fraction of the player's image height.
    kp_sigma_frac: float = 0.02
    #: Extra σ (m) of the tracked feet position as a prior on the ankles' midpoint.
    feet_extra_sigma_m: float = 0.1
    #: Savitzky-Golay smoothing of the translation (frames, odd; 1 = off).
    smooth_frames: int = 9
    min_conf: float = 0.2
    #: Joints are moved onto their 2D keypoint's ray when it passes this close (m; 0: off),
    #: then smoothed over this many frames.
    snap_max_m: float = 0.35
    snap_min_conf: float = 0.5
    snap_smooth_frames: int = 5

    def as_config(self) -> dict:
        return asdict(self)


def normalize_clip(kp: np.ndarray, min_conf: float = 0.0) -> np.ndarray:
    """(T, 17, 3) pixel keypoints → [-1, 1] by the clip's extent (MotionBERT ``crop_scale``)."""
    kp = np.asarray(kp, dtype=np.float64)
    ok = kp[..., 2] > min_conf
    out = kp.copy()
    if ok.sum() < 4:
        out[..., :2] = 0
        return out
    xy = kp[..., :2][ok]
    lo, hi = xy.min(0), xy.max(0)
    scale = float(max(hi - lo))
    if scale <= 0:
        out[..., :2] = 0
        return out
    origin = (lo + hi - scale) / 2
    out[..., :2] = np.clip(((kp[..., :2] - origin) / scale - 0.5) * 2, -1, 1)
    return out


def split_clip(n: int, max_len: int = MAX_CLIP) -> list[tuple[int, int]]:
    """Split ``n`` frames into near-equal parts no longer than ``max_len``."""
    parts = max(1, -(-n // max_len))
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in itertools.pairwise(edges) if b > a]


class Lifter:
    """MotionBERT-Lite on clips of H36M keypoints."""

    def __init__(
        self,
        name: str = "motionbert-lite",
        device: str | None = None,
        flip: bool = True,
        progress: Callable[[float], None] | None = None,
    ):
        import torch

        from swingvision.models import registry
        from swingvision.pose import motionbert

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.flip = flip
        self.model = motionbert.load(registry.ensure(name, progress), self.device)

    def lift(self, kp_h36m: np.ndarray) -> np.ndarray:
        """(T, 17, 3) pixel H36M keypoints with confidence → (T, 17, 3) root-relative 3D in
        normalized camera units (x right, y down, z away)."""
        import torch

        out = np.zeros((len(kp_h36m), 17, 3), dtype=np.float64)
        for a, b in split_clip(len(kp_h36m)):
            x = normalize_clip(kp_h36m[a:b])
            t = torch.from_numpy(x.astype(np.float32))[None].to(self.device)
            with torch.inference_mode():
                y = self.model(t)
                if self.flip:
                    tf = t.clone()
                    tf[..., 0] *= -1
                    yf = self.model(tf[:, :, list(H36M_FLIP)])[:, :, list(H36M_FLIP)]
                    yf[..., 0] *= -1
                    y = (y + yf) / 2
            y = y[0].cpu().numpy().astype(np.float64)
            out[a:b] = y - y[:, :1]
        return out


#: name → factory(**kwargs) of lifters (tests register fakes).
LIFTERS: dict[str, Callable[..., Lifter]] = {
    "motionbert-lite": lambda **kw: Lifter("motionbert-lite", **kw)
}
LIFTER = "motionbert-lite"


def make_lifter(name: str = LIFTER, **kwargs) -> Lifter:
    return LIFTERS[name](**kwargs)


def scale_to_height(rel: np.ndarray, height_m: float) -> tuple[np.ndarray, float]:
    """Scale root-relative joints so their ankle-to-head chain matches ``height_m``."""
    chain = chain_length(rel)
    med = float(np.median(chain[np.isfinite(chain)])) if np.isfinite(chain).any() else 0.0
    if med <= 1e-6:
        return rel, 0.0
    s = height_m / (HEIGHT_PER_CHAIN * med)
    return rel * s, s


def _rotation_z_to(d: np.ndarray) -> np.ndarray:
    """(T, 3) unit vectors → (T, 3, 3) minimal rotations taking +z onto each."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = d @ z
    vx = np.zeros((len(d), 3, 3))
    vx[:, 0, 1], vx[:, 0, 2] = -v[:, 2], v[:, 1]
    vx[:, 1, 0], vx[:, 1, 2] = v[:, 2], -v[:, 0]
    vx[:, 2, 0], vx[:, 2, 1] = -v[:, 1], v[:, 0]
    k = 1.0 / (1.0 + np.maximum(c, -0.999))
    return np.eye(3)[None] + vx + (vx @ vx) * k[:, None, None]


def to_court_axes(rel_cam: np.ndarray, kp_h36m: np.ndarray, cam: Camera) -> np.ndarray:
    """Root-relative joints in the virtual (crop) camera → court axes (still root-relative)."""
    pelvis = kp_h36m[:, H["pelvis"], :2]
    u = cam.undistort(pelvis)
    d = np.column_stack([(u - cam.c) / cam.f, np.ones(len(u))])
    bad = ~np.isfinite(d).all(1)
    d[bad] = [0.0, 0.0, 1.0]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Rv = _rotation_z_to(d)
    in_cam = np.einsum("tij,tkj->tki", Rv, rel_cam)
    return in_cam @ cam.R  # cam → court for row vectors: X_court = R^T X_cam


def solve_translation(
    rel: np.ndarray,
    kp_h36m: np.ndarray,
    cam: Camera,
    feet_xy: np.ndarray | None,
    feet_sigma: np.ndarray | None,
    p: PlaceParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame translation putting court-axis joints ``rel`` (T, 17, 3) on their rays.

    Returns (T, 3) translations and (T,) reprojection RMS (px).
    """
    T = len(rel)
    R, t = cam.R, cam.tvec
    u = cam.undistort(kp_h36m[..., :2])  # (T, 17, 2)
    ab = (u - cam.c) / cam.f
    conf = np.where(np.isfinite(ab).all(-1), kp_h36m[..., 2], 0.0)
    conf = np.where(conf >= p.min_conf, conf, 0.0)
    ab = np.nan_to_num(ab)
    ys = kp_h36m[..., 1]
    span = np.where(conf > 0, ys, np.nan)
    with np.errstate(invalid="ignore"):
        h_px = np.nanmax(span, axis=1) - np.nanmin(span, axis=1)
    h_px = np.where(np.isfinite(h_px) & (h_px > 10), h_px, 100.0)
    ankles = [H["l_ankle"], H["r_ankle"]]
    out = np.full((T, 3), np.nan)
    rms = np.full(T, np.nan)
    # Start: feet on the ground at the tracked position, else under the pelvis ray.
    ground = cam.image_to_ground(kp_h36m[:, H["pelvis"], :2], z=0.9)
    for i in range(T):
        w = np.sqrt(conf[i])
        if (w > 0).sum() < 6:
            continue
        q = rel[i] @ R.T + t  # camera coords of joints for T = 0
        guess = (
            np.array([*feet_xy[i], 0.9])
            if feet_xy is not None and np.isfinite(feet_xy[i]).all()
            else ground[i]
        )
        if not np.isfinite(guess).all():
            continue
        Ti = np.asarray(guess, dtype=np.float64).copy()  # rel is pelvis-relative
        for _ in range(2):
            depth = np.maximum((q + R @ Ti)[:, 2], 0.5)
            wj = w * cam.f / (depth * p.kp_sigma_frac * h_px[i])
            rows = []
            rhs = []
            for k in (0, 1):
                A = R[k][None, :] - ab[i, :, k][:, None] * R[2][None, :]
                b = ab[i, :, k] * q[:, 2] - q[:, k]
                rows.append(A * wj[:, None])
                rhs.append(b * wj)
            low = ankles[int(np.argmin(rel[i, ankles, 2]))]
            rows.append(np.array([[0.0, 0.0, 1.0]]) / p.ground_sigma_m)
            rhs.append(np.array([p.ankle_z_m - rel[i, low, 2]]) / p.ground_sigma_m)
            if feet_xy is not None and np.isfinite(feet_xy[i]).all():
                sig = p.feet_extra_sigma_m + (
                    float(feet_sigma[i])
                    if feet_sigma is not None and np.isfinite(feet_sigma[i])
                    else 0.0
                )
                mid = rel[i, ankles, :2].mean(0)
                rows.append(np.array([[1.0, 0, 0], [0, 1.0, 0]]) / sig)
                rhs.append((feet_xy[i] - mid) / sig)
            A = np.vstack(rows)
            b = np.concatenate(rhs)
            Ti = np.linalg.lstsq(A, b, rcond=None)[0]
        out[i] = Ti
        px = cam.project(rel[i] + Ti)
        err = np.linalg.norm(px - kp_h36m[i, :, :2], axis=1)
        rms[i] = float(np.sqrt(np.average(err**2, weights=np.maximum(conf[i], 1e-6))))
    return out, rms


def smooth_runs(x: np.ndarray, frames: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing (order 2) of (T, D) values over runs of consecutive frames."""
    from scipy.signal import savgol_filter

    x = x.copy()
    if window < 3 or len(x) == 0:
        return x
    breaks = np.flatnonzero(np.diff(frames) > 1) + 1
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(frames)], strict=True):
        seg = x[a:b]
        ok = np.isfinite(seg).all(1)
        if ok.sum() < 5:
            continue
        idx = np.flatnonzero(ok)
        filled = np.column_stack(
            [np.interp(np.arange(len(seg)), idx, seg[idx, k]) for k in range(seg.shape[1])]
        )
        w = min(window, len(seg) - (1 - len(seg) % 2))
        if w >= 5:
            filled = savgol_filter(filled, w, 2, axis=0)
        filled[~ok] = np.nan
        x[a:b] = filled
    return x


@dataclass
class Placed:
    joints: np.ndarray  # (T, 17, 3) court meters
    scale: float  # meters per normalized unit
    reproj_px: np.ndarray  # (T,)


def place(
    lifted: np.ndarray,
    kp_h36m: np.ndarray,
    frames: np.ndarray,
    cam: Camera,
    height_m: float | None,
    feet_xy: np.ndarray | None = None,
    feet_sigma: np.ndarray | None = None,
    p: PlaceParams | None = None,
) -> Placed:
    """Lifted root-relative joints of one clip → court coordinates (meters)."""
    p = p or PlaceParams()
    rel, s = scale_to_height(lifted, height_m or DEFAULT_HEIGHT_M)
    rel = to_court_axes(rel, kp_h36m, cam)
    trans, rms = solve_translation(rel, kp_h36m, cam, feet_xy, feet_sigma, p)
    trans = smooth_runs(trans, frames, p.smooth_frames)
    joints = rel + trans[:, None, :]
    if p.snap_max_m > 0:
        joints = snap_to_rays(joints, kp_h36m, cam, p)
        flat = smooth_runs(joints.reshape(len(joints), -1), frames, p.snap_smooth_frames)
        joints = flat.reshape(joints.shape)
    return Placed(joints, s, rms)


def snap_to_rays(
    joints: np.ndarray, kp_h36m: np.ndarray, cam: Camera, p: PlaceParams
) -> np.ndarray:
    """Move each confident joint onto the camera ray through its 2D keypoint, keeping its
    distance from the camera (MotionBERT's ``gt_2d`` idea: the network supplies depth, the
    2D detector the fast image-plane motion it smooths away). A keypoint whose ray misses
    the lifted joint by more than ``snap_max_m`` (a left/right swap, an outlier) is not used.
    """
    out = joints.copy()
    C = cam.center
    ok = (kp_h36m[..., 2] >= p.snap_min_conf) & np.isfinite(joints).all(-1)
    rays = cam.rays(kp_h36m[..., :2])
    ok &= np.isfinite(rays).all(-1)
    d = joints - C
    along = np.einsum("tjk,tjk->tj", d, np.nan_to_num(rays))
    on_ray = C + np.nan_to_num(rays) * along[..., None]
    # Keep the joint's distance from the camera, not just its projection onto the ray.
    dist = np.linalg.norm(d, axis=-1)
    snapped = C + np.nan_to_num(rays) * dist[..., None]
    miss = np.linalg.norm(on_ray - joints, axis=-1)
    use = ok & (miss <= p.snap_max_m) & (along > 0)
    out[use] = snapped[use]
    return out
