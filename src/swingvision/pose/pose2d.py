"""2D pose on full-resolution player crops (PLAN.md §7.7).

The player's box (from ``movement``) is padded (a raised racket arm and the toss reach far
above the box YOLO drew a frame earlier), widened to the network's 3:4 aspect, cut from the
4K frame on the GPU, and resized to 256×192. ViTPose (Hugging Face ``transformers``) returns
17 COCO heatmaps per crop; with ``flip_test`` the mirrored crop's heatmaps are averaged in.
Peaks get a sub-pixel offset from a quadratic fit to the log heatmap. Keypoints come back in
full-resolution display pixels with the heatmap peak as confidence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from swingvision.pose.skeleton import COCO_FLIP

if TYPE_CHECKING:
    import torch

INPUT_H, INPUT_W = 256, 192
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
#: Box padding: the crop is this much taller / wider than the player's box, and reaches
#: further up than down (arms and the racket go up, feet stay on the ground).
PAD_H = 1.35
PAD_W = 1.4
UP_SHIFT = 0.08


def crop_box(box: np.ndarray) -> np.ndarray:
    """Player box(es) ``(..., 4)`` → crop rectangle(s) with the network's aspect (w:h = 3:4)."""
    box = np.asarray(box, dtype=np.float64)
    x0, y0, x1, y1 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    h = (y1 - y0) * PAD_H
    w = np.maximum((x1 - x0) * PAD_W, h * INPUT_W / INPUT_H)
    h = np.maximum(h, w * INPUT_H / INPUT_W)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 - UP_SHIFT * (y1 - y0)
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def vitpose_config():
    """The ``usyd-community/vitpose-base-simple`` architecture (weights come from the
    registry, pinned by SHA-256)."""
    from transformers import VitPoseBackboneConfig, VitPoseConfig

    backbone = VitPoseBackboneConfig(out_features=["stage12"], out_indices=[12])
    return VitPoseConfig(
        backbone_config=backbone, use_simple_decoder=True, scale_factor=4, num_labels=17
    )


def decode_heatmaps(hm: np.ndarray) -> np.ndarray:
    """(N, K, h, w) heatmaps → (N, K, 3) peak ``x, y`` in heatmap pixels plus peak value."""
    n, k, h, w = hm.shape
    flat = hm.reshape(n, k, -1)
    idx = flat.argmax(-1)
    score = np.take_along_axis(flat, idx[..., None], -1)[..., 0]
    py, px = idx // w, idx % w
    out = np.zeros((n, k, 3), dtype=np.float64)
    out[..., 2] = score
    log = np.log(np.maximum(hm, 1e-10))
    ii, kk = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")
    for axis, (pos, size) in enumerate(((px, w), (py, h))):
        lo = np.clip(pos - 1, 0, size - 1)
        hi = np.clip(pos + 1, 0, size - 1)
        if axis == 0:
            a, b, c = log[ii, kk, py, lo], log[ii, kk, py, px], log[ii, kk, py, hi]
        else:
            a, b, c = log[ii, kk, lo, px], log[ii, kk, py, px], log[ii, kk, hi, px]
        denom = a - 2 * b + c
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where((denom < -1e-6) & (lo != pos) & (hi != pos), (a - c) / (2 * denom), 0)
        out[..., axis] = pos + np.clip(off, -0.5, 0.5)
    return out


class VitPose2D:
    """Top-down 2D pose: frames + player boxes → COCO-17 keypoints in image pixels."""

    def __init__(
        self,
        name: str = "vitpose-base-simple",
        device: str | None = None,
        flip_test: bool = True,
        batch: int = 32,
        progress: Callable[[float], None] | None = None,
    ):
        import torch
        from safetensors.torch import load_file
        from transformers import VitPoseForPoseEstimation

        from swingvision.models import registry

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")
        self.flip_test = flip_test
        self.batch = batch
        weights = registry.ensure(name, progress)
        model = VitPoseForPoseEstimation(vitpose_config())
        model.load_state_dict(load_file(str(weights)), strict=True)
        model = model.to(self.device)
        self.model = (model.half() if self.half else model.float()).eval()
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def _crop(self, image: torch.Tensor, rect: np.ndarray) -> torch.Tensor:
        """Cut ``rect`` (may extend past the frame: black) and resize to the network input."""
        import torch
        import torch.nn.functional as F

        _, H, W = image.shape
        x0, y0, x1, y1 = (round(float(v)) for v in rect)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            return torch.zeros((3, INPUT_H, INPUT_W), device=self.device)
        c = image[:, cy0:cy1, cx0:cx1].to(self.device, non_blocking=True).float().div_(255)
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            c = F.pad(c, (cx0 - x0, x1 - cx1, cy0 - y0, y1 - cy1))
        return F.interpolate(
            c[None], size=(INPUT_H, INPUT_W), mode="bilinear", antialias=True, align_corners=False
        )[0]

    def _heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        x = ((x - self._mean) / self._std).to(dtype)
        hm = self.model(pixel_values=x).heatmaps.float()
        if self.flip_test:
            hf = self.model(pixel_values=x.flip(-1)).heatmaps.float()
            hm = (hm + hf.flip(-1)[:, list(COCO_FLIP)]) / 2
        return hm

    def prepare(self, image: torch.Tensor, box: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """One frame and player box → (network-sized crop on the device, crop rectangle).

        Lets callers crop each decoded 4K frame right away and free it."""
        rect = crop_box(np.asarray(box, dtype=np.float64))
        return self._crop(image, rect), rect

    def run(self, crops: list[torch.Tensor], rects: np.ndarray) -> np.ndarray:
        """Prepared crops + their rectangles → (n, 17, 3) keypoints in image pixels."""
        import torch

        n = len(crops)
        out = np.zeros((n, 17, 3), dtype=np.float64)
        rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
        for s in range(0, n, self.batch):
            e = min(n, s + self.batch)
            with torch.inference_mode():
                hm = self._heatmaps(torch.stack(crops[s:e])).cpu().numpy()
            kp = decode_heatmaps(hm)
            h, w = hm.shape[-2:]
            r = rects[s:e]
            sx = (r[:, 2] - r[:, 0]) / w
            sy = (r[:, 3] - r[:, 1]) / h
            out[s:e, :, 0] = r[:, None, 0] + (kp[..., 0] + 0.5) * sx[:, None]
            out[s:e, :, 1] = r[:, None, 1] + (kp[..., 1] + 0.5) * sy[:, None]
            out[s:e, :, 2] = kp[..., 2]
        return out.astype(np.float32)

    def estimate(self, images: list[torch.Tensor], boxes: np.ndarray) -> np.ndarray:
        """``images``: (3, H, W) uint8 frames; ``boxes``: (n, 4) player boxes, one per image.

        Returns (n, 17, 3): ``x, y`` in full-resolution pixels and the keypoint confidence.
        """
        if not images:
            return np.zeros((0, 17, 3), dtype=np.float32)
        prepared = [self.prepare(im, b) for im, b in zip(images, boxes, strict=True)]
        return self.run([c for c, _ in prepared], np.array([r for _, r in prepared]))


#: name → factory(**kwargs) of 2D pose estimators (tests register fakes).
ESTIMATORS: dict[str, Callable[..., VitPose2D]] = {
    "vitpose-base-simple": lambda **kw: VitPose2D("vitpose-base-simple", **kw)
}


def make_estimator(name: str, **kwargs) -> VitPose2D:
    try:
        factory = ESTIMATORS[name]
    except KeyError:
        raise KeyError(f"Unknown pose model {name!r}. Known: {', '.join(ESTIMATORS)}") from None
    return factory(**kwargs)
