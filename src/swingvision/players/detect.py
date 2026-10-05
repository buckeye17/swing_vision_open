"""Person detection on decoded frames (PLAN.md §7.2).

Frames are cropped to the court region of interest (the court plus run-off, up to head height,
:func:`roi_crop`), scaled on the GPU so the crop's long side is ``input_size`` pixels, padded
to the network stride, and run through a COCO person detector in batches. Boxes come back in
full-resolution display pixels.

Detectors are looked up by name in :data:`DETECTORS`, so tests (and later experiments) can
register their own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from swingvision.court import model as court_model
from swingvision.court.camera import Camera

if TYPE_CHECKING:
    import torch

#: Person height used to extend the ROI crop above the ground (m).
CROP_HEAD_M = 2.6
CROP_QUANTUM = 64


@dataclass(frozen=True)
class Crop:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def roi_crop(
    cameras: Sequence[Camera],
    width: int,
    height: int,
    behind_m: float,
    beside_m: float,
    margin_frac: float = 0.03,
) -> Crop:
    """Image rectangle covering the playing area (and a standing player) for every camera.

    The ROI ground polygon and the same polygon raised to head height are projected; the box
    is padded, clipped to the image, and snapped outward to 64 px so small calibration edits
    don't change it (it is part of ``pass1_detect``'s fingerprint).
    """
    poly = court_model.roi_polygon(behind_m, beside_m)
    # Densify the outline: lens distortion bends straight edges.
    s = np.linspace(0, 1, 25)[:, None]
    edges = [(1 - s) * poly[i] + s * poly[(i + 1) % 4] for i in range(4)]
    ground = np.concatenate(edges)
    pts = np.concatenate([ground, ground + np.array([0.0, 0.0, CROP_HEAD_M])])
    xs, ys = [], []
    for cam in cameras:
        front = cam.depth(pts) > 0.5
        px = cam.project(pts[front])
        px = px[np.isfinite(px).all(axis=1)]
        xs.append(px[:, 0])
        ys.append(px[:, 1])
    if not xs or not np.concatenate(xs).size:
        return Crop(0, 0, width, height)
    x, y = np.concatenate(xs), np.concatenate(ys)
    pad_x, pad_y = margin_frac * width, margin_frac * height
    q = CROP_QUANTUM
    x0 = max(0, int(np.floor((x.min() - pad_x) / q)) * q)
    y0 = max(0, int(np.floor((y.min() - pad_y) / q)) * q)
    x1 = min(width, int(np.ceil((x.max() + pad_x) / q)) * q)
    y1 = min(height, int(np.ceil((y.max() + pad_y) / q)) * q)
    if x1 - x0 < q or y1 - y0 < q:
        return Crop(0, 0, width, height)
    return Crop(x0, y0, x1, y1)


class PersonDetector(Protocol):
    name: str

    def detect(self, images: list[torch.Tensor], crop: Crop) -> list[np.ndarray]:
        """``images``: (3, H, W) uint8 frames. Returns one (n, 5) array per image:
        ``x0, y0, x1, y1, conf`` in full-resolution pixels."""
        ...


class YoloPersonDetector:
    """Ultralytics YOLO (COCO) run directly on GPU tensors: FP16, class 0 only."""

    STRIDE = 32
    PAD_VALUE = 114 / 255

    def __init__(
        self,
        name: str,
        input_size: int = 1920,
        conf: float = 0.1,
        iou: float = 0.6,
        device: str | None = None,
        progress: Callable[[float], None] | None = None,
    ):
        import torch
        from ultralytics.nn.tasks import load_checkpoint

        from swingvision.models import registry

        self.name = name
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")
        weights = registry.ensure(name, progress)
        model, _ = load_checkpoint(str(weights), device=self.device, fuse=True)
        model = model.half() if self.half else model.float()
        self.model = model.eval()
        if self.half:
            torch.backends.cudnn.benchmark = True

    def _prepare(self, images: list[torch.Tensor], crop: Crop) -> tuple[torch.Tensor, float]:
        import torch
        import torch.nn.functional as F

        scale = min(1.0, self.input_size / max(crop.width, crop.height))
        h = max(self.STRIDE, round(crop.height * scale))
        w = max(self.STRIDE, round(crop.width * scale))
        ph = -(-h // self.STRIDE) * self.STRIDE
        pw = -(-w // self.STRIDE) * self.STRIDE
        dtype = torch.half if self.half else torch.float32
        x = torch.stack([im[:, crop.y0 : crop.y1, crop.x0 : crop.x1] for im in images])
        x = x.to(self.device, non_blocking=True).to(dtype).div_(255)
        if scale < 1.0:
            x = F.interpolate(x, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
        if (ph, pw) != (h, w):
            x = F.pad(x, (0, pw - w, 0, ph - h), value=self.PAD_VALUE)
        return x.contiguous(memory_format=torch.channels_last), scale

    def detect(self, images, crop):
        import torch
        from ultralytics.utils.nms import non_max_suppression

        if not images:
            return []
        x, scale = self._prepare(images, crop)
        with torch.inference_mode():
            preds = self.model(x)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds
            out = non_max_suppression(
                preds.float(),
                conf_thres=self.conf,
                iou_thres=self.iou,
                classes=[0],
                max_det=50,
                max_time_img=1.0,
            )
        result = []
        for det in out:
            d = det[:, :5].cpu().numpy().astype(np.float64)
            d[:, :4] /= scale
            d[:, [0, 2]] += crop.x0
            d[:, [1, 3]] += crop.y0
            result.append(d.astype(np.float32))
        return result


#: name → factory(**kwargs). Every registered YOLO model is available by its name.
DETECTORS: dict[str, Callable[..., PersonDetector]] = {}


def _register_yolo() -> None:
    from swingvision.models.registry import REGISTRY

    for name, spec in REGISTRY.items():
        if spec.task.startswith("person"):
            DETECTORS[name] = lambda name=name, **kw: YoloPersonDetector(name, **kw)


_register_yolo()


def make_detector(name: str, **kwargs) -> PersonDetector:
    try:
        factory = DETECTORS[name]
    except KeyError:
        raise KeyError(f"Unknown person detector {name!r}. Known: {', '.join(DETECTORS)}") from None
    return factory(**kwargs)
