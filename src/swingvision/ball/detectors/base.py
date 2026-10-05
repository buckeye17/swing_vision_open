"""The ``BallDetector`` interface and the detector registry (PLAN.md §7.4.1).

A detector sees short runs of consecutive decoded frames and returns ball *candidates* for
some of them: several per frame, each ``(x, y, score)`` in full-resolution display pixels.
Picking the real ball among the candidates is the trajectory linker's job
(:mod:`swingvision.ball.trajectory`), so detectors should favor recall.

Detectors are chosen with a spec string ``name[:weights][@size]``:

* ``motion`` - classical three-frame differencing, no training (full resolution)
* ``motion@1920`` - the same on the crop scaled to 1920 px wide
* ``unet:ball-unet-v1@1280`` - the slim TrackNet-style U-Net with trained weights
* ``yolo11m@1920`` - a COCO YOLO's "sports ball" class
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from swingvision.players.detect import Crop

if TYPE_CHECKING:
    import torch

    from swingvision.storage.schemas import VideoInfo

SPEC_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?::(?P<weights>[^@]+))?(?:@(?P<size>\w+))?$")


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    weights: str | None = None
    #: Width the crop is scaled to (long side) before the detector runs; ``None``: native.
    input_px: int | None = None

    @classmethod
    def parse(cls, text: str) -> DetectorSpec:
        m = SPEC_RE.match(text.strip())
        if not m:
            raise ValueError(f"Bad detector spec {text!r} (expected name[:weights][@size])")
        size = m["size"]
        if size in (None, "full", "native"):
            px = None
        elif size.isdigit():
            px = int(size)
        else:
            raise ValueError(f"Bad input size {size!r} in {text!r}")
        return cls(m["name"], m["weights"], px)

    def __str__(self) -> str:
        s = self.name
        if self.weights:
            s += f":{self.weights}"
        return s + (f"@{self.input_px}" if self.input_px else "")


class BallDetector(ABC):
    """Finds ball candidates on frames of one video.

    ``context`` frames on each side of a target frame must be passed along with it (3-frame
    models need the previous and the next frame).
    """

    name: str = ""
    context: int = 1
    #: Candidates kept per frame (best first).
    max_candidates: int = 8

    def __init__(self, spec: DetectorSpec, device: str | None = None):
        import torch

        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.crop: Crop | None = None
        self.info: VideoInfo | None = None

    def prepare(self, info: VideoInfo, crop: Crop, background: np.ndarray | None = None) -> None:
        """Called once per video before :meth:`detect`. ``background``: (H, W, 3) RGB median
        image of the court without players (full resolution), if the detector wants it."""
        self.info = info
        self.crop = crop

    @abstractmethod
    def detect(self, frames: Sequence[torch.Tensor], targets: Sequence[int]) -> list[np.ndarray]:
        """``frames``: consecutive (3, H, W) uint8 frames (full resolution, upright).
        ``targets``: indices into ``frames`` to detect on; each has ``context`` neighbors on
        both sides. Returns one ``(n, 3)`` float32 array ``x, y, score`` per target."""

    @property
    def scale(self) -> float:
        """Crop → model input scale (≤ 1)."""
        assert self.crop is not None
        if not self.spec.input_px:
            return 1.0
        return min(1.0, self.spec.input_px / max(self.crop.width, self.crop.height))

    def describe(self) -> dict:
        return {"spec": str(self.spec), "context": self.context}


# ---------------------------------------------------------------------------
# Shared tensor helpers
# ---------------------------------------------------------------------------


def crop_resize(
    frames: Sequence[torch.Tensor], crop: Crop, scale: float, dtype, device
) -> torch.Tensor:
    """(N, 3, h, w) float frames cropped to ``crop`` and scaled by ``scale`` (0-1 range)."""
    import torch
    import torch.nn.functional as F

    x = torch.stack([f[:, crop.y0 : crop.y1, crop.x0 : crop.x1] for f in frames])
    x = x.to(device, non_blocking=True).to(dtype).div_(255)
    if scale < 1.0:
        h, w = round(crop.height * scale), round(crop.width * scale)
        x = F.interpolate(x, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
    return x


def peaks(
    heat: torch.Tensor, k: int, threshold: float, nms: int, refine: int = 2
) -> list[np.ndarray]:
    """Top-``k`` local maxima per map of ``heat`` (B, h, w) above ``threshold``.

    Positions are refined to sub-pixel with the response-weighted centroid of a
    ``(2*refine+1)²`` window. Returns (n, 3) arrays ``x, y, value`` in map pixels.
    """
    import torch
    import torch.nn.functional as F

    B, h, w = heat.shape
    hm = heat[:, None].float()
    mx = F.max_pool2d(hm, nms, 1, nms // 2)
    keep = (hm == mx) & (hm > threshold)
    vals = torch.where(keep, hm, torch.full_like(hm, -1.0)).flatten(1)
    kk = min(k, vals.shape[1])
    top_v, top_i = torch.topk(vals, kk, dim=1)
    ys = (top_i // w).float()
    xs = (top_i % w).float()
    if refine > 0:
        r = refine
        pad = F.pad(hm, (r, r, r, r))
        offs = torch.arange(-r, r + 1, device=heat.device)
        dy, dx = torch.meshgrid(offs, offs, indexing="ij")
        yi = (ys.long()[..., None] + r + dy.flatten()).clamp(0, h + 2 * r - 1)
        xi = (xs.long()[..., None] + r + dx.flatten()).clamp(0, w + 2 * r - 1)
        bidx = torch.arange(B, device=heat.device)[:, None, None]
        win = pad[bidx, 0, yi, xi].clamp_min(0)  # (B, k, (2r+1)²)
        wsum = win.sum(-1).clamp_min(1e-6)
        ys = ys + (win * dy.flatten()).sum(-1) / wsum
        xs = xs + (win * dx.flatten()).sum(-1) / wsum
    out = torch.stack([xs, ys, top_v], -1).cpu().numpy()
    return [o[o[:, 2] > threshold].astype(np.float32) for o in out]


def to_full_res(cands: np.ndarray, crop: Crop, scale: float) -> np.ndarray:
    """Map (n, 3) map-pixel candidates back to full-resolution display pixels."""
    out = cands.copy()
    # Pixel centers: map pixel i covers [i, i+1) / scale in crop pixels.
    out[:, 0] = (cands[:, 0] + 0.5) / scale - 0.5 + crop.x0
    out[:, 1] = (cands[:, 1] + 0.5) / scale - 0.5 + crop.y0
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: name → factory(spec, device) for every known detector family.
DETECTORS: dict[str, Callable[..., BallDetector]] = {}


def register(name: str):
    def deco(factory):
        DETECTORS[name] = factory
        return factory

    return deco


def make_ball_detector(spec: str | DetectorSpec, device: str | None = None) -> BallDetector:
    from swingvision.ball import detectors as _  # noqa: F401  (registers the built-ins)

    spec = DetectorSpec.parse(spec) if isinstance(spec, str) else spec
    try:
        factory = DETECTORS[spec.name]
    except KeyError:
        raise KeyError(
            f"Unknown ball detector {spec.name!r}. Known: {', '.join(sorted(DETECTORS))}"
        ) from None
    return factory(spec, device=device)
