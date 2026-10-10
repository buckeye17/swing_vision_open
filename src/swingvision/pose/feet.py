"""Foot keypoints around serve contacts (PLAN.md §7.11, M7b).

COCO-17 (ViTPose) stops at the ankle, so the toe comes from a whole-body model: RTMW-l
(OpenMMLab, distilled from RTMW-x, COCO-WholeBody's 133 keypoints at 384×288), run with
ONNX Runtime on the player's crops in the frames around each serve's contact only. Of its
keypoints the ankles, big toes, small toes and heels are kept (``FOOT``). The model choice and
how it was measured are in ``docs/spikes/m7b-toe-pose.md``.

The crop follows the model's training pipeline (MMPose ``TopDownGetBboxCenterScale`` with
padding 1.25, widened to the 3:4 input, ``TopDownAffine``): the player's box from
``movement`` is sampled straight from the decoded 4K frame on the GPU with the same affine
map, so keypoints map back to full-resolution pixels exactly. Keypoints are decoded from the
SimCC distributions (the peak, refined by a parabola through its neighbours); the score is
the mean of the two peaks (larger is surer; RTMW's scores run ≈0–10).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

#: Kept keypoints, in this order.
FOOT = ("l_ankle", "r_ankle", "l_big_toe", "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel")  # fmt: skip
F = {name: i for i, name in enumerate(FOOT)}
#: Their COCO-WholeBody indices.
WHOLEBODY_INDEX = (15, 16, 17, 18, 19, 20, 21, 22)
INPUT_W, INPUT_H = 288, 384
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
PADDING = 1.25
SIMCC_SPLIT = 2.0
DEFAULT_MODEL = "rtmw-l-wholebody"


def foot_names(side: str) -> tuple[int, int, int, int]:
    """Indices into :data:`FOOT` of one foot's (ankle, big toe, small toe, heel)."""
    p = side[0]
    return F[f"{p}_ankle"], F[f"{p}_big_toe"], F[f"{p}_small_toe"], F[f"{p}_heel"]


def crop_rect(box: np.ndarray) -> np.ndarray:
    """Player box(es) ``(..., 4)`` → the crop rectangle(s) the model sees (padded 1.25,
    widened to the 3:4 input)."""
    box = np.asarray(box, dtype=np.float64)
    cx = (box[..., 0] + box[..., 2]) / 2
    cy = (box[..., 1] + box[..., 3]) / 2
    w = (box[..., 2] - box[..., 0]) * PADDING
    h = (box[..., 3] - box[..., 1]) * PADDING
    aspect = INPUT_W / INPUT_H
    w, h = np.maximum(w, h * aspect), np.maximum(h, w / aspect)
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def decode_simcc(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """SimCC distributions (N, K, Wx), (N, K, Wy) → (N, K, 3): x, y in input pixels, score."""
    out = np.zeros((*sx.shape[:2], 3))
    for axis, sim in ((0, sx), (1, sy)):
        i = sim.argmax(-1)
        n = sim.shape[-1]
        lo = np.take_along_axis(sim, np.clip(i - 1, 0, n - 1)[..., None], -1)[..., 0]
        mid = np.take_along_axis(sim, i[..., None], -1)[..., 0]
        hi = np.take_along_axis(sim, np.clip(i + 1, 0, n - 1)[..., None], -1)[..., 0]
        den = lo - 2 * mid + hi
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(den < -1e-6, 0.5 * (lo - hi) / den, 0.0)
        out[..., axis] = (i + np.clip(off, -0.5, 0.5)) / SIMCC_SPLIT
    out[..., 2] = 0.5 * (sx.max(-1) + sy.max(-1))
    return out


class FootPose:
    """Top-down whole-body pose (ONNX Runtime) → :data:`FOOT` keypoints in image pixels."""

    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch: int = 16,
        progress: Callable[[float], None] | None = None,
    ):
        # Importing torch loads the CUDA and cuDNN libraries the CUDA provider needs.
        import onnxruntime as ort
        import torch

        from swingvision.models import registry

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        path = registry.ensure(name, progress)
        providers = ["CPUExecutionProvider"]
        if (
            self.device.startswith("cuda")
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._mean = torch.tensor(MEAN, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(3, 1, 1)

    def prepare(self, image: torch.Tensor, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One frame (3, H, W uint8) and player box → (network input (3, 384, 288), crop
        rectangle). Input pixel (i, j) samples the frame at (x0 + i·sx, y0 + j·sy)."""
        import torch
        import torch.nn.functional as Fn

        rect = crop_rect(np.asarray(box, dtype=np.float64))
        x0, y0, x1, y1 = (float(v) for v in rect)
        sx, sy = (x1 - x0) / INPUT_W, (y1 - y0) / INPUT_H
        xs = x0 + np.arange(INPUT_W) * sx
        ys = y0 + np.arange(INPUT_H) * sy
        _, H, W = image.shape
        ix0, iy0 = max(0, int(np.floor(xs[0])) - 1), max(0, int(np.floor(ys[0])) - 1)
        ix1, iy1 = min(W, int(np.ceil(xs[-1])) + 2), min(H, int(np.ceil(ys[-1])) + 2)
        if ix1 - ix0 < 2 or iy1 - iy0 < 2:
            return np.zeros((3, INPUT_H, INPUT_W), np.float32), rect
        sub = image[:, iy0:iy1, ix0:ix1].to(self.device, non_blocking=True).float()
        gx = torch.tensor(2 * (xs - ix0) / (ix1 - ix0 - 1) - 1, device=self.device)
        gy = torch.tensor(2 * (ys - iy0) / (iy1 - iy0 - 1) - 1, device=self.device)
        grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), dim=-1)[None].float()
        crop = Fn.grid_sample(sub[None], grid, mode="bilinear", align_corners=True)[0]
        crop = (crop - self._mean) / self._std
        return crop.cpu().numpy().astype(np.float32), rect

    def run(self, crops: list[np.ndarray], rects: np.ndarray) -> np.ndarray:
        """Prepared crops + their rectangles → (n, len(FOOT), 3): x, y (image px), score."""
        n = len(crops)
        out = np.zeros((n, len(FOOT), 3), dtype=np.float64)
        rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
        idx = list(WHOLEBODY_INDEX)
        for s in range(0, n, self.batch):
            e = min(n, s + self.batch)
            sx, sy = self.session.run(None, {self.input_name: np.stack(crops[s:e])})[:2]
            kp = decode_simcc(sx[:, idx], sy[:, idx])
            r = rects[s:e]
            scale_x = (r[:, 2] - r[:, 0]) / INPUT_W
            scale_y = (r[:, 3] - r[:, 1]) / INPUT_H
            out[s:e, :, 0] = r[:, None, 0] + kp[..., 0] * scale_x[:, None]
            out[s:e, :, 1] = r[:, None, 1] + kp[..., 1] * scale_y[:, None]
            out[s:e, :, 2] = kp[..., 2]
        return out.astype(np.float32)
