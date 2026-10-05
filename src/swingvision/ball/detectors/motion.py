"""Classical ball detector: frame differencing (no training data needed).

A flying ball is a small blob that differs from *both* an earlier and a later frame at its
current position. The differences against each neighbor contain the ball twice (at the old
and the new position); their minimum keeps only the current one. Neighbors ``gap`` frames
away (default 3) keep the whole ball even when it moves less than its own diameter per frame
(a serve toss near its apex, a ball flying straight away from the camera), where adjacent
frames would leave only a sliver and bias the position.

Response per pixel, in intensity levels (0-255):

    m   = min(max_c |I_t - I_t-gap|, max_c |I_t - I_t+gap|)
    r_s = mean_s×s(m) - 0.9 · mean_3s×3s(m)        for blob sizes s ∈ {3, 7, 13} px
    r   = max_s r_s

The center-surround filter favors compact blobs: a walking player is large, so the surround
cancels most of it. Peaks are refined to the centroid of the differencing blob (pixels above
half the peak's level connected to it, which also centers motion-blur streaks), then scored
``r / noise`` (noise: the frame's median ``m`` plus one level) and mapped to a 0-1
confidence ``q / (q + 15)``.
Balls lying still on the court are invisible to it, which is what the tracker wants.

A shaking camera (wind, a ball hitting the tripod) turns every court line into a moving edge.
The neighbor frames are therefore first aligned to the current one: a global translation from
phase correlation on a 1/4-scale gray image, applied with sub-pixel interpolation when it
exceeds a quarter pixel.
"""

from __future__ import annotations

import numpy as np

from swingvision.ball.detectors.base import (
    BallDetector,
    DetectorSpec,
    crop_resize,
    register,
    to_full_res,
)

#: Response (noise multiples) below which a peak isn't reported.
THRESHOLD = 4.0
#: Response that maps to a confidence of 0.5.
REF = 15.0
SCALES = (3, 7, 13)
SURROUND = 3
NMS = 15
#: Half-size of the window a blob (or motion-blur streak) may extend over, px at full res.
STREAK_PX = 64
BATCH = 8


class MotionDetector(BallDetector):
    name = "motion"
    max_candidates = 12

    def __init__(
        self,
        spec: DetectorSpec,
        device: str | None = None,
        threshold: float = THRESHOLD,
        gap: int = 3,
    ):
        super().__init__(spec, device)
        self.threshold = threshold
        self.gap = gap
        self.context = gap

    def detect(self, frames, targets):
        import torch

        assert self.crop is not None, "prepare() first"
        if not targets:
            return []
        scale = self.scale
        g = self.gap
        dtype = torch.half if self.device.startswith("cuda") else torch.float32
        out: list[np.ndarray] = []
        with torch.inference_mode():
            # Crop and scale only the frames each batch needs (4K float frames are large).
            for b in range(0, len(targets), BATCH):
                tb = targets[b : b + BATCH]
                need = sorted({j for i in tb for j in (i - g, i, i + g)})
                pos = {j: k for k, j in enumerate(need)}
                imgs = crop_resize([frames[j] for j in need], self.crop, scale, dtype, self.device)
                imgs = imgs.mul_(255)
                cur = imgs[[pos[i] for i in tb]]
                prv = align(imgs[[pos[i - g] for i in tb]], cur)
                nxt = align(imgs[[pos[i + g] for i in tb]], cur)
                for c in detect_blobs(prv, cur, nxt, self.max_candidates, self.threshold, scale):
                    c[:, 2] = c[:, 2] / (c[:, 2] + REF)
                    out.append(to_full_res(c, self.crop, scale))
                del imgs, cur, prv, nxt
        return out

    @staticmethod
    def response(prv, cur, nxt):
        """(B, h, w) noise-normalized blob response from three (B, 3, h, w) frames (0-255)."""
        r, _, _, noise = _response(prv, cur, nxt, SCALES)
        return r / noise[:, None, None]


#: Alignment works on frames downscaled by this factor.
ALIGN_DOWN = 4
#: Shifts below this (px) are left alone.
ALIGN_MIN_PX = 0.25


def global_shift(mov, ref):
    """(B, 2) translation ``(dx, dy)`` in pixels that moves ``mov`` onto ``ref`` (B, 3, h, w)."""
    import torch
    import torch.nn.functional as F

    def gray(x):
        g = x.float().mean(1, keepdim=True)
        g = F.avg_pool2d(g, ALIGN_DOWN)[:, 0]
        return g - g.mean((1, 2), keepdim=True)

    a, b = gray(ref), gray(mov)
    h, w = a.shape[-2:]
    win = torch.outer(torch.hann_window(h, device=a.device), torch.hann_window(w, device=a.device))
    fa = torch.fft.rfft2(a * win)
    fb = torch.fft.rfft2(b * win)
    r = fa * fb.conj()
    r = r / (r.abs() + 1e-6)
    corr = torch.fft.irfft2(r, s=(h, w))
    flat = corr.flatten(1)
    k = flat.argmax(1)
    py, px = k // w, k % w

    def sub(c0, cm, cp):
        den = cm - 2 * c0 + cp
        return torch.where(den.abs() > 1e-9, 0.5 * (cm - cp) / den, torch.zeros_like(den))

    bi = torch.arange(len(k), device=a.device)
    c0 = corr[bi, py, px]
    dy = py.float() + sub(c0, corr[bi, (py - 1) % h, px], corr[bi, (py + 1) % h, px])
    dx = px.float() + sub(c0, corr[bi, py, (px - 1) % w], corr[bi, py, (px + 1) % w])
    dy = torch.where(dy > h / 2, dy - h, dy)
    dx = torch.where(dx > w / 2, dx - w, dx)
    return torch.stack([dx, dy], 1) * ALIGN_DOWN


def translate(x, shift):
    """Sample ``x`` (B, C, h, w) at ``p - shift`` (bilinear): content moves by ``shift`` px."""
    import torch
    import torch.nn.functional as F

    B, C, h, w = x.shape
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=torch.float32)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -shift[:, 0] * 2 / w
    theta[:, 1, 2] = -shift[:, 1] * 2 / h
    grid = F.affine_grid(theta, (B, C, h, w), align_corners=False).to(x.dtype)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def refine_shift(mov, ref, shift, iters: int = 2, down: int = 2):
    """Gauss-Newton (Lucas-Kanade) refinement of a global translation on 1/``down`` scale.

    Pixels with large residuals (moving players and balls) are down-weighted.
    """
    import torch
    import torch.nn.functional as F

    def gray(x):
        return F.avg_pool2d(x.float().mean(1, keepdim=True), down)

    a, b = gray(ref), gray(mov)
    gx = (a[..., 1:-1, 2:] - a[..., 1:-1, :-2]) / 2
    gy = (a[..., 2:, 1:-1] - a[..., :-2, 1:-1]) / 2
    s = shift.clone() / down
    for _ in range(iters):
        warped = translate(b, s)
        e = (warped - a)[..., 1:-1, 1:-1]
        wgt = 1.0 / (1.0 + (e / 10.0) ** 2)
        sxx = (wgt * gx * gx).sum((1, 2, 3))
        sxy = (wgt * gx * gy).sum((1, 2, 3))
        syy = (wgt * gy * gy).sum((1, 2, 3))
        bx = (wgt * gx * e).sum((1, 2, 3))
        by = (wgt * gy * e).sum((1, 2, 3))
        det = (sxx * syy - sxy * sxy).clamp_min(1e-6)
        dx = (syy * bx - sxy * by) / det
        dy = (sxx * by - sxy * bx) / det
        s = s + torch.stack([dx, dy], 1)
    return s * down


def align(mov, ref):
    """``mov`` shifted onto ``ref`` where the camera moved (frames that didn't are untouched)."""
    shift = refine_shift(mov, ref, global_shift(mov, ref))
    need = shift.abs().amax(1) > ALIGN_MIN_PX
    if not bool(need.any()):
        return mov
    out = mov.clone()
    out[need] = translate(mov[need], shift[need]).to(mov.dtype)
    return out


def _difference(prv, cur, nxt):
    import torch

    d1 = (cur - prv).abs().amax(1)
    d2 = (cur - nxt).abs().amax(1)
    return torch.minimum(d1, d2)[:, None].float()


def _response(prv, cur, nxt, scales):
    import torch
    import torch.nn.functional as F

    m = _difference(prv, cur, nxt)
    best = None
    which = None
    for k, s in enumerate(scales):
        center = F.avg_pool2d(m, s, 1, s // 2)
        big = s * SURROUND
        sur = F.avg_pool2d(m, big, 1, big // 2, count_include_pad=False)
        r = (center - 0.9 * sur)[:, 0]
        if best is None:
            best, which = r, torch.zeros_like(r, dtype=torch.int8)
        else:
            better = r > best
            best = torch.where(better, r, best)
            which = torch.where(better, torch.full_like(which, k), which)
    noise = m[:, 0, ::8, ::8].flatten(1).median(1).values + 1.0
    return best, which, m[:, 0], noise


def detect_blobs(prv, cur, nxt, k: int, threshold: float, scale: float = 1.0) -> list[np.ndarray]:
    """Top-``k`` blobs per frame: (n, 3) arrays ``x, y, response`` in map pixels."""
    import torch
    import torch.nn.functional as F

    scales = tuple(max(3, round(s * scale) | 1) for s in SCALES)
    r, _, m, noise = _response(prv, cur, nxt, scales)
    q = r / noise[:, None, None]
    B, h, w = q.shape
    nms = max(5, round(NMS * scale) | 1)
    mx = F.max_pool2d(q[:, None], nms, 1, nms // 2)[:, 0]
    keep = (q == mx) & (q > threshold)
    vals = torch.where(keep, q, torch.full_like(q, -1.0)).flatten(1)
    kk = min(k, vals.shape[1])
    top_v, top_i = torch.topk(vals, kk, dim=1)
    ys, xs = top_i // w, top_i % w
    # Centroid of the whole differencing blob around each peak: pixels above half the peak's
    # level that connect to it (a motion-blur streak can be 100+ px long; its center is the
    # ball at mid-exposure).
    R = max(8, round(STREAK_PX * scale))
    offs = torch.arange(-R, R + 1, device=q.device)
    dy, dx = torch.meshgrid(offs, offs, indexing="ij")
    yi = (ys[..., None, None] + dy).clamp(0, h - 1)
    xi = (xs[..., None, None] + dx).clamp(0, w - 1)
    b4 = torch.arange(B, device=q.device)[:, None, None, None]
    win = m[b4, yi, xi]  # (B, k, 2R+1, 2R+1)
    b2 = torch.arange(B, device=q.device)[:, None]
    peak = m[b2, ys, xs]
    above = (win >= 0.5 * peak[..., None, None]).float()
    seed = torch.zeros_like(above)
    seed[..., R, R] = 1.0
    grown = seed.flatten(0, 1)[:, None]
    mask = above.flatten(0, 1)[:, None]
    for _ in range(-(-R // 3)):
        grown = F.max_pool2d(grown, 7, 1, 3) * mask
    grown = grown[:, 0].view_as(above)
    wgt = (win - 0.5 * peak[..., None, None]).clamp_min(0) * grown
    wsum = wgt.sum((-1, -2))
    good = wsum > 0
    wsum = wsum.clamp_min(1e-6)
    cy = torch.where(good, ys.float() + (wgt * dy).sum((-1, -2)) / wsum, ys.float())
    cx = torch.where(good, xs.float() + (wgt * dx).sum((-1, -2)) / wsum, xs.float())
    out = torch.stack([cx, cy, top_v], -1).cpu().numpy()
    return [o[o[:, 2] > threshold].astype(np.float32) for o in out]


@register("motion")
def _motion(spec: DetectorSpec, device: str | None = None) -> BallDetector:
    return MotionDetector(spec, device)
