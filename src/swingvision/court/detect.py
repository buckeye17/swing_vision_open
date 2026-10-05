"""Automatic court detection and line-based refinement (PLAN.md §7.1 steps 1–4).

The input is a *background* image: the per-pixel median of frames sampled across
the video, which removes the players and balls. Detection is classical, so it
needs no training data and copes with views a broadcast-trained keypoint CNN
has never seen (low corner cameras, wide-angle lenses, partially visible courts,
pickleball lines, neighboring courts):

1. **Line response**: white top-hat of ``min(R, G, B)``. Painted white lines are
   bright in every channel; the green and red court surfaces and teal pickleball
   lines are not.
2. **Line candidates**: probabilistic Hough on the thresholded response at 1920 px
   width, merged into long lines and split into an *across* family (near-horizontal,
   parallel to the baselines) and an *along* family (the sidelines).
3. **Hypotheses**: every pair of across lines × pair of along lines, matched to every
   ordered pair of model lines, gives a homography. Each is scored by how much of the
   projected court lands on line pixels minus how much lands off them.
4. **Refinement**: the best hypotheses are refined by sampling the response
   perpendicular to each projected model line, taking the sub-pixel line center, and
   fitting the full camera (focal length, lens ``k1``, pose) to those samples. The
   net tape adds non-planar evidence once the ground fit has converged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from swingvision.court import model
from swingvision.court.camera import (
    Camera,
    FitOptions,
    FitResult,
    LineObs,
    camera_from_homography,
    fit_camera,
)
from swingvision.court.homography import apply_h

WORK_WIDTH = 1920
MIN_DEPTH_M = 0.5  # ignore line points this close to the camera plane
MAX_LINE_WIDTH_PX = 80.0
ALL_LINES = model.COURT_LINES + model.NET_LINES


# ---------------------------------------------------------------------------
# 1. Line response
# ---------------------------------------------------------------------------


def line_response(rgb: np.ndarray) -> np.ndarray:
    """White-line response (float32, same size as the image)."""
    mn = rgb.min(axis=2)
    k = max(9, round(61 * rgb.shape[1] / 3840) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(mn, cv2.MORPH_TOPHAT, kernel).astype(np.float32)


def line_mask(small_resp: np.ndarray, threshold: float, max_width: int = 25) -> np.ndarray:
    """Thresholded response at working scale with blobs wider than any painted line removed
    (sky, water, buildings), so Hough sees mostly thin line structures."""
    mask = (small_resp > threshold).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_width, max_width))
    blobs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    blobs = cv2.dilate(blobs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return cv2.bitwise_and(mask, cv2.bitwise_not(blobs))


def response_threshold(resp: np.ndarray) -> float:
    p = float(np.percentile(resp[::4, ::4], 99.5))
    return float(np.clip(0.35 * p, 12.0, 60.0))


# ---------------------------------------------------------------------------
# 2. Line candidates
# ---------------------------------------------------------------------------


@dataclass
class ImageLine:
    """An infinite image line ``n · p = rho`` (unit normal ``n``) at working scale."""

    n: np.ndarray
    rho: float
    length: float
    angle_deg: float  # direction angle from horizontal in (-90, 90]

    @property
    def homog(self) -> np.ndarray:
        return np.array([self.n[0], self.n[1], -self.rho])


def _fit_line(pts: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    mean = (pts * w[:, None]).sum(0) / w.sum()
    d = pts - mean
    cov = (d * w[:, None]).T @ d
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, 0]
    return n, float(n @ mean)


def candidate_lines(
    mask: np.ndarray, max_per_family: int = 12
) -> tuple[list[ImageLine], list[ImageLine]]:
    """Hough segments merged into long lines → (across, along) families."""
    w = mask.shape[1]
    segs = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 720, threshold=50, minLineLength=max(30, w // 40), maxLineGap=8
    )
    if segs is None:
        return [], []
    segs = np.asarray(segs).reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lengths)
    clusters: list[dict] = []
    for i in order:
        x0, y0, x1, y1 = segs[i]
        p = np.array([[x0, y0], [x1, y1]])
        placed = False
        for c in clusters:
            dist = np.abs(p @ c["n"] - c["rho"])
            direction = np.array([-c["n"][1], c["n"][0]])
            seg_dir = (p[1] - p[0]) / max(lengths[i], 1e-9)
            if dist.max() < 4.0 and abs(seg_dir @ direction) > np.cos(np.radians(2.5)):
                c["pts"].append(p)
                c["w"].append(np.full(2, lengths[i]))
                c["len"] += lengths[i]
                allp = np.concatenate(c["pts"])
                c["n"], c["rho"] = _fit_line(allp, np.concatenate(c["w"]))
                placed = True
                break
        if not placed:
            n = np.array([-(y1 - y0), x1 - x0]) / max(lengths[i], 1e-9)
            clusters.append(
                {
                    "pts": [p],
                    "w": [np.full(2, lengths[i])],
                    "len": lengths[i],
                    "n": n,
                    "rho": float(n @ p[0]),
                }
            )
    lines = []
    for c in clusters:
        if c["len"] < w / 16:
            continue
        n = c["n"]
        angle = float(np.degrees(np.arctan2(-n[0], n[1])))  # direction = (-n1, n0) rotated
        angle = (angle + 90) % 180 - 90
        lines.append(ImageLine(n, c["rho"], c["len"], angle))
    lines.sort(key=lambda ln: -ln.length)
    across = [ln for ln in lines if abs(ln.angle_deg) < 35][:max_per_family]
    along = [ln for ln in lines if abs(ln.angle_deg) > 20][:max_per_family]
    return across, along


# ---------------------------------------------------------------------------
# 3. Hypotheses
# ---------------------------------------------------------------------------


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[..., :2] / p[..., 2:3]


def _homographies_from_quads(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Batched 4-point homographies: src, dst (N, 4, 2) → H (N, 3, 3) with dst ~ H src."""
    n = len(src)
    A = np.zeros((n, 8, 8))
    b = np.zeros((n, 8))
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    A[:, 0::2, 0] = x
    A[:, 0::2, 1] = y
    A[:, 0::2, 2] = 1
    A[:, 0::2, 6] = -x * u
    A[:, 0::2, 7] = -y * u
    A[:, 1::2, 3] = x
    A[:, 1::2, 4] = y
    A[:, 1::2, 5] = 1
    A[:, 1::2, 6] = -x * v
    A[:, 1::2, 7] = -y * v
    b[:, 0::2] = u
    b[:, 1::2] = v
    ok = np.abs(np.linalg.det(A)) > 1e-12
    h = np.full((n, 9), np.nan)
    h[ok, :8] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    h[:, 8] = 1.0
    return h.reshape(n, 3, 3)


@dataclass
class _Samples:
    pts: np.ndarray  # (S, 2) ground xy
    line: np.ndarray  # (S,)
    first: np.ndarray  # (S,) bool: first sample of its line


def _score_samples(spacing: float = 0.4) -> _Samples:
    pts, idx = model.line_samples(spacing)
    first = np.concatenate([[True], idx[1:] != idx[:-1]])
    return _Samples(pts[:, :2], idx, first)


def score_homographies(
    H: np.ndarray, dt: np.ndarray, samples: _Samples, tol: float = 3.0, miss_penalty: float = 0.6
) -> np.ndarray:
    """Image length of projected court lines on line pixels minus a penalty for length off them.

    ``dt`` is the distance transform to the nearest line pixel (working scale).
    """
    h, w = dt.shape
    P = np.concatenate([samples.pts, np.ones((len(samples.pts), 1))], axis=1)  # (S, 3)
    proj = np.einsum("nij,sj->nsi", H, P)  # (N, S, 3)
    z = proj[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = proj[..., :2] / z[..., None]
    # Image length represented by each sample: distance to the previous one on the same line.
    seg = np.linalg.norm(np.diff(xy, axis=1, prepend=xy[:, :1]), axis=2)
    seg[:, samples.first] = 0
    seg = np.minimum(seg, w / 20)  # don't let one sample claim a huge span
    inside = (xy[..., 0] >= 0) & (xy[..., 0] < w - 1) & (xy[..., 1] >= 0) & (xy[..., 1] < h - 1)
    inside &= np.isfinite(xy).all(axis=2)
    xi = np.clip(np.nan_to_num(xy[..., 0]), 0, w - 1).astype(np.int32)
    yi = np.clip(np.nan_to_num(xy[..., 1]), 0, h - 1).astype(np.int32)
    d = dt[yi, xi]
    hit = inside & (d <= tol)
    miss = inside & (d > tol)
    score = (seg * hit).sum(1) - miss_penalty * (seg * miss).sum(1)
    # The whole court must be in front of the camera.
    score[(z <= 0).any(axis=1)] = -np.inf
    score[~np.isfinite(score)] = -np.inf
    return score


def hypotheses(
    across: list[ImageLine],
    along: list[ImageLine],
    dt: np.ndarray,
    top_k: int = 12,
    batch: int = 8000,
) -> list[tuple[float, np.ndarray]]:
    """Best (score, H court→working-image) hypotheses, distinct and best first."""
    if len(across) < 2 or len(along) < 2:
        return []
    h, w = dt.shape
    A = np.array([ln.homog for ln in across])
    V = np.array([ln.homog for ln in along])
    ia, ib = np.triu_indices(len(A), 1)
    ka, kb = np.triu_indices(len(V), 1)
    # All image quads: (across pair) × (along pair)
    qa = np.repeat(np.arange(len(ia)), len(ka))
    qv = np.tile(np.arange(len(ka)), len(ia))
    a1, a2, v1, v2 = A[ia[qa]], A[ib[qa]], V[ka[qv]], V[kb[qv]]
    p11, p12 = _intersect(a1, v1), _intersect(a1, v2)
    p21, p22 = _intersect(a2, v1), _intersect(a2, v2)
    # Order: across line "top" has the smaller mean y; along line "left" has the smaller x.
    swap_a = (p11[:, 1] + p12[:, 1]) > (p21[:, 1] + p22[:, 1])
    p11, p12, p21, p22 = (
        np.where(swap_a[:, None], p21, p11),
        np.where(swap_a[:, None], p22, p12),
        np.where(swap_a[:, None], p11, p21),
        np.where(swap_a[:, None], p12, p22),
    )
    swap_v = (p11[:, 0] + p21[:, 0]) > (p12[:, 0] + p22[:, 0])
    p11, p12, p21, p22 = (
        np.where(swap_v[:, None], p12, p11),
        np.where(swap_v[:, None], p11, p12),
        np.where(swap_v[:, None], p22, p21),
        np.where(swap_v[:, None], p21, p22),
    )
    quads = np.stack([p11, p12, p22, p21], axis=1)  # TL, TR, BR, BL
    # Plausibility: finite, not absurdly far outside the image, convex with clockwise order.
    lim = 3 * max(w, h)
    ok = np.isfinite(quads).all(axis=(1, 2)) & (np.abs(quads) < lim).all(axis=(1, 2))
    e = np.roll(quads, -1, axis=1) - quads
    cross = (
        e[:, :, 0] * np.roll(e, -1, axis=1)[:, :, 1] - e[:, :, 1] * np.roll(e, -1, axis=1)[:, :, 0]
    )
    ok &= (cross > 0).all(axis=1)
    quads = quads[ok]
    if not len(quads):
        return []

    my = model.ACROSS_Y
    mx = model.ALONG_X
    mya, myb = np.triu_indices(len(my), 1)  # far (larger y) first since ACROSS_Y is descending
    mxa, mxb = np.triu_indices(len(mx), 1)  # left first
    model_quads = []
    for i, j in zip(mya, myb, strict=True):
        for k, m in zip(mxa, mxb, strict=True):
            # The center service line only runs between the service lines.
            center = mx[k] == 0.0 or mx[m] == 0.0
            if center and not (abs(my[i]) < 7 and abs(my[j]) < 7):
                continue
            model_quads.append([[mx[k], my[i]], [mx[m], my[i]], [mx[m], my[j]], [mx[k], my[j]]])
    model_quads = np.array(model_quads, dtype=np.float64)

    samples = _score_samples()
    src = np.repeat(model_quads[None], len(quads), axis=0).reshape(-1, 4, 2)
    dst = np.repeat(quads[:, None], len(model_quads), axis=1).reshape(-1, 4, 2)
    best: list[tuple[float, np.ndarray]] = []
    scores_all, H_all = [], []
    for s in range(0, len(src), batch):
        Hb = _homographies_from_quads(src[s : s + batch], dst[s : s + batch])
        valid = np.isfinite(Hb).all(axis=(1, 2))
        sc = np.full(len(Hb), -np.inf)
        if valid.any():
            sc[valid] = score_homographies(Hb[valid], dt, samples)
        keep = np.argsort(-sc)[: top_k * 4]
        scores_all.append(sc[keep])
        H_all.append(Hb[keep])
    scores = np.concatenate(scores_all)
    Hs = np.concatenate(H_all)
    order = np.argsort(-scores)
    # Distinct hypotheses: compare where they put the court's 14 keypoints.
    kp = model.keypoint_array(list(model.GROUND_KEYPOINTS))[:, :2]
    seen: list[np.ndarray] = []
    for i in order:
        if not np.isfinite(scores[i]) or scores[i] <= 0:
            break
        proj = apply_h(Hs[i], kp)
        if any(np.median(np.linalg.norm(proj - p, axis=1)) < 10 for p in seen):
            continue
        seen.append(proj)
        best.append((float(scores[i]), Hs[i]))
        if len(best) >= top_k:
            break
    return best


# ---------------------------------------------------------------------------
# 4. Ridge sampling and refinement
# ---------------------------------------------------------------------------


@dataclass
class RidgeSamples:
    obs: np.ndarray  # (M, 2) observed line-center px
    line: np.ndarray  # (M,) index into ALL_LINES
    peak: np.ndarray  # (M,) response at the center

    def line_obs(self, net_weight: float = 0.5) -> LineObs:
        a = np.array([ALL_LINES[i].p0 for i in self.line]).reshape(-1, 3)
        b = np.array([ALL_LINES[i].p1 for i in self.line]).reshape(-1, 3)
        w = np.where(self.line >= len(model.COURT_LINES), net_weight, 1.0)
        return LineObs(a, b, self.obs, w, self.line)


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        img, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderValue=0.0
    )


def ridge_samples(
    resp: np.ndarray,
    camera: Camera,
    halfwidth: float,
    threshold: float,
    lines: tuple[int, ...] | None = None,
    step_px: float = 6.0,
) -> RidgeSamples:
    """Find line centers near each projected court line by searching along its normal."""
    h, w = resp.shape
    lines = lines if lines is not None else tuple(range(len(model.COURT_LINES)))
    obs_all, line_all, peak_all = [], [], []
    for li in lines:
        line = ALL_LINES[li]
        # Dense 3D samples, then thin to ~step_px in the image.
        dense = line.samples(0.02)
        depth = camera.depth(dense)
        dense = dense[depth > MIN_DEPTH_M]
        if len(dense) < 3:
            continue
        px = camera.project(dense)
        # The search window must also hold the whole painted line, which near the camera
        # can be wider than the positional uncertainty.
        width_px = _line_width_px(camera, line, dense, px, w, h)
        hw = halfwidth + 0.6 * width_px + 2
        offsets = np.arange(-hw, hw + 0.25, 0.5)
        tang = np.gradient(px, axis=0)
        tn = np.linalg.norm(tang, axis=1)
        # Only the part of the line that is (nearly) in frame.
        near = (px[:, 0] > -50) & (px[:, 0] < w + 50) & (px[:, 1] > -50) & (px[:, 1] < h + 50)
        good = near & np.isfinite(tn) & (tn > 1e-9)
        px, tang, tn = px[good], tang[good], tn[good]
        if len(px) < 3:
            continue
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(px, axis=0), axis=1))])
        keep = np.searchsorted(arc, np.arange(step_px / 2, arc[-1], step_px))
        keep = np.clip(keep, 0, len(px) - 1)
        px, tang, tn = px[keep], tang[keep], tn[keep]
        nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1) / tn[:, None]
        margin = hw + 2
        inside = (
            (px[:, 0] > margin)
            & (px[:, 0] < w - margin)
            & (px[:, 1] > margin)
            & (px[:, 1] < h - margin)
        )
        px, nrm = px[inside], nrm[inside]
        if not len(px):
            continue
        X = px[:, None, 0] + offsets[None, :] * nrm[:, None, 0]
        Y = px[:, None, 1] + offsets[None, :] * nrm[:, None, 1]
        prof = _bilinear(resp, X, Y)  # (n, k)
        center, peak, ok = _profile_centers(prof, offsets, threshold)
        obs = px + center[:, None] * nrm
        obs_all.append(obs[ok])
        line_all.append(np.full(int(ok.sum()), li))
        peak_all.append(peak[ok])
    if not obs_all:
        return RidgeSamples(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0))
    return RidgeSamples(np.concatenate(obs_all), np.concatenate(line_all), np.concatenate(peak_all))


def _line_width_px(
    camera: Camera, line, dense: np.ndarray, px: np.ndarray, w: int, h: int
) -> float:
    """Largest on-screen width (px) of the painted line (or net tape) along its visible part."""
    d = np.subtract(line.p1, line.p0)
    if line.family == "net":
        delta = np.array([0.0, 0.0, model.NET_TAPE_HALF])
    else:
        delta = np.array([-d[1], d[0], 0.0]) / np.linalg.norm(d) * (model.LINE_WIDTH / 2)
    inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    if not inside.any():
        return 0.0
    pts = dense[inside]
    width = np.linalg.norm(camera.project(pts + delta) - camera.project(pts - delta), axis=1)
    width = width[np.isfinite(width)]
    # A wildly wrong camera can put line points near the image plane; cap the window.
    return float(min(np.max(width), MAX_LINE_WIDTH_PX)) if len(width) else 0.0


def _profile_centers(prof: np.ndarray, offsets: np.ndarray, threshold: float):
    """Sub-pixel center of the strongest ridge in each profile, with ambiguity rejection."""
    n, k = prof.shape
    i = np.argmax(prof, axis=1)
    peak = prof[np.arange(n), i]
    ok = (peak >= threshold) & (i > 1) & (i < k - 2)
    half = peak / 2
    above = prof >= half[:, None]
    # Contiguous run above half-max around the peak.
    idx = np.arange(k)[None, :]
    left_break = np.where(~above & (idx < i[:, None]), idx, -1).max(axis=1)
    right_break = np.where(~above & (idx > i[:, None]), idx, k).min(axis=1)
    run = (idx > left_break[:, None]) & (idx < right_break[:, None])
    wts = np.where(run, prof - half[:, None], 0.0)
    center = (wts * offsets[None, :]).sum(1) / np.maximum(wts.sum(1), 1e-9)
    ok &= (left_break >= 0) & (right_break < k)  # the ridge must fall off on both sides
    # Ambiguity: another strong ridge outside the run.
    others = np.where(run, 0.0, prof)
    ok &= others.max(axis=1) < 0.6 * peak
    return center, peak, ok


@dataclass
class Detection:
    ok: bool
    camera: Camera | None
    rms_px: float | None = None
    n_samples: int = 0
    n_expected: int = 0
    n_net_samples: int = 0
    score: float = 0.0
    message: str = ""
    samples: RidgeSamples | None = None
    fit: FitResult | None = field(default=None, repr=False)

    @property
    def coverage(self) -> float:
        return self.n_samples / self.n_expected if self.n_expected else 0.0


def expected_samples(camera: Camera, step_px: float = 6.0) -> int:
    """How many ridge samples the visible court lines would give with nothing occluded."""
    total = 0.0
    for line in model.COURT_LINES:
        dense = line.samples(0.05)
        dense = dense[camera.depth(dense) > MIN_DEPTH_M]
        if len(dense) < 2:
            continue
        px = camera.project(dense)
        inside = (
            (px[:, 0] >= 0)
            & (px[:, 0] < camera.width)
            & (px[:, 1] >= 0)
            & (px[:, 1] < camera.height)
        )
        seg = np.linalg.norm(np.diff(px, axis=0), axis=1)
        total += float(seg[inside[1:] & inside[:-1]].sum())
    return int(total / step_px)


def inlier_limit(abs_residuals: np.ndarray) -> float:
    """Residual (px) above which a line sample counts as an outlier: 3.5 robust sigmas,
    but never below 2.5 px."""
    if not len(abs_residuals):
        return 2.5
    sigma = 1.4826 * float(np.median(abs_residuals)) + 0.3
    return max(2.5, 3.5 * sigma)


#: Final-stage camera model. Phone video is often cropped/warped for stabilization and
#: corrected for lens distortion in-camera, so the principal point and k2 are freed
#: (with priors) once the line samples are trustworthy.
FULL_MODEL = FitOptions(fit_k1=True, fit_k2=True, fit_center=True)


def refine(
    resp: np.ndarray,
    camera: Camera,
    threshold: float,
    schedule: tuple[float, ...] = (30.0, 16.0, 10.0, 7.0),
    use_net: bool = True,
    model_options: FitOptions = FULL_MODEL,
    early_options: FitOptions | None = None,
) -> Detection:
    """Iteratively re-sample line centers and re-fit the camera, shrinking the search window.

    By default the first iteration fits pose and focal length only and the second adds
    ``k1`` (``early_options`` overrides both); the last two use ``model_options`` (and
    the net tape, if ``use_net``).
    """
    cam = camera
    fit = None
    samples = None
    n = len(schedule)
    for it, hw in enumerate(schedule):
        late = it >= n - 2
        lines = tuple(range(len(model.COURT_LINES)))
        if use_net and late:
            lines += tuple(range(len(model.COURT_LINES), len(ALL_LINES)))
        samples = ridge_samples(resp, cam, hw, threshold, lines)
        if (samples.line < len(model.COURT_LINES)).sum() < 20:
            return Detection(False, cam, message="Too few line samples near the projected court")
        if late:
            opts = model_options
        elif early_options is not None:
            opts = early_options
        else:
            opts = FitOptions(fit_k1=it > 0)
        opts = replace(opts, f_scale_px=max(1.0, hw / 6))
        fit = fit_camera(cam, lines=samples.line_obs(), options=opts)
        # Drop outliers, then re-fit on the inliers.
        res = np.abs(fit.line_residuals)
        inl = res < inlier_limit(res)
        if inl.sum() >= 20 and inl.sum() < len(res):
            samples = RidgeSamples(samples.obs[inl], samples.line[inl], samples.peak[inl])
            fit = fit_camera(fit.camera, lines=samples.line_obs(), options=opts)
        cam = fit.camera
    assert fit is not None and samples is not None
    n_net = int((samples.line >= len(model.COURT_LINES)).sum())
    ground_res = fit.line_residuals[samples.line < len(model.COURT_LINES)]
    rms = float(np.sqrt(np.mean(ground_res**2))) if len(ground_res) else None
    return Detection(
        ok=True,
        camera=cam,
        rms_px=rms,
        n_samples=len(ground_res),
        n_expected=expected_samples(cam),
        n_net_samples=n_net,
        samples=samples,
        fit=fit,
        # Soft inlier count: many well-fitting samples beat a few perfect ones.
        score=float(np.sum(np.exp(-(ground_res**2) / 8.0))),
    )


def _plausible(cam: Camera) -> bool:
    d = cam.describe()
    off = np.hypot(*d["principal_offset_px"])
    return (
        0.8 < d["height_m"] < 40
        and -60 < d["behind_baseline_m"] < 80
        and 15 < d["hfov_deg"] < 150
        and abs(cam.k1) < 0.6
        and abs(cam.k2) < 0.6
        and off < 0.25 * cam.width
    )


@dataclass
class Prepared:
    """Line response and threshold for one background image (reused across fits)."""

    resp: np.ndarray
    threshold: float

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> Prepared:
        resp = line_response(rgb)
        return cls(resp, response_threshold(resp))

    @property
    def ridge_threshold(self) -> float:
        return 0.6 * self.threshold


def detect_court(rgb: np.ndarray | Prepared, top_k: int = 6, use_net: bool = True) -> Detection:
    """Detect the court in a background image and fit the camera (full-resolution px).

    Each hypothesis gets a quick refinement (pinhole + k1, no net); the best one is then
    refined with the full camera model.
    """
    prep = rgb if isinstance(rgb, Prepared) else Prepared.from_rgb(rgb)
    resp = prep.resp
    h, w = resp.shape
    s = WORK_WIDTH / w
    small = cv2.resize(resp, (WORK_WIDTH, round(h * s)), interpolation=cv2.INTER_AREA)
    mask = line_mask(small, prep.threshold)
    across, along = candidate_lines(mask)
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    hyps = hypotheses(across, along, dt, top_k=top_k)
    if not hyps:
        return Detection(False, None, message="No court-like line arrangement found")
    S = np.diag([1 / s, 1 / s, 1.0])
    best: Detection | None = None
    for _score, Hs in hyps:
        cam0 = camera_from_homography(S @ Hs, w, h)
        if cam0 is None or not _plausible(cam0):
            continue
        quick = refine(
            resp, cam0, prep.ridge_threshold, schedule=(30.0, 16.0, 10.0), use_net=False,
            model_options=FitOptions(),
        )  # fmt: skip
        if not quick.ok or quick.camera is None or not _plausible(quick.camera):
            continue
        if best is None or quick.score > best.score:
            best = quick
    if best is None or best.camera is None:
        return Detection(False, None, message="Court hypotheses did not survive refinement")
    final = refine(resp, best.camera, prep.ridge_threshold, (10.0, 8.0, 7.0), use_net=use_net)
    if not final.ok or final.camera is None or not _plausible(final.camera):
        final = best
    final.message = f"{final.n_samples} line samples, RMS {final.rms_px:.2f} px"
    return final


def refine_from(
    rgb: np.ndarray | Prepared,
    camera: Camera,
    use_net: bool = True,
    schedule: tuple[float, ...] = (24.0, 14.0, 9.0, 7.0),
) -> Detection:
    """Snap an approximate camera (from the editor, or another time window) onto the lines."""
    prep = rgb if isinstance(rgb, Prepared) else Prepared.from_rgb(rgb)
    det = refine(prep.resp, camera, prep.ridge_threshold, schedule=schedule, use_net=use_net)
    if det.ok:
        det.message = f"{det.n_samples} line samples, RMS {det.rms_px:.2f} px"
    return det
