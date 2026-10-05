"""Slim TrackNet-style ball detector (PLAN.md §3, §7.4).

A small U-Net looks at three frames (``t - 2``, ``t``, ``t + 2``, RGB, stacked to 9 channels)
of the court crop scaled so its long side is ``input_px`` (default 1920, half of 4K) and
predicts a ball heatmap at half that resolution. The M0 spike sized it: a TrackNet-size
network at the resolution the far court needs runs at 20 fps, this one at about 190.

Frames two apart (rather than adjacent) still show a slow ball (a toss near its apex) in
different places, and the network learns the ball's look (round, yellow-green, often a
motion-blur streak), which the classical motion detector can't: a racket frame crossing a
court line moves like a ball but doesn't look like one.

Weights are trained on the user's labels (``sv train ball``) and live in
``<output_root>/models/ball/<name>.pt`` with a model card ``<name>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from swingvision.ball.detectors.base import (
    BallDetector,
    DetectorSpec,
    crop_resize,
    peaks,
    register,
    to_full_res,
)

#: Frame offsets the network sees, relative to the target frame.
OFFSETS = (-2, 0, 2)
THRESHOLD = 0.1
BATCH = 4
STRIDE = 16


def build_model(width: int = 32, in_frames: int = len(OFFSETS)):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def cbr(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class SlimBallNet(nn.Module):
        """Encoder 1/2 → 1/16, decoder back to 1/2 with skips; one logit map at 1/2."""

        def __init__(self):
            super().__init__()
            w = width
            self.stem = nn.Sequential(cbr(3 * in_frames, w, 2), cbr(w, w))
            self.e1 = nn.Sequential(cbr(w, 2 * w, 2), cbr(2 * w, 2 * w))
            self.e2 = nn.Sequential(cbr(2 * w, 3 * w, 2), cbr(3 * w, 3 * w))
            self.e3 = nn.Sequential(cbr(3 * w, 4 * w, 2), cbr(4 * w, 4 * w), cbr(4 * w, 4 * w))
            self.d2 = cbr(7 * w, 3 * w)
            self.d1 = cbr(5 * w, 2 * w)
            self.d0 = nn.Sequential(cbr(3 * w, w), cbr(w, w))
            self.head = nn.Conv2d(w, 1, 1)
            nn.init.constant_(self.head.bias, -4.0)  # start near "no ball anywhere"

        def forward(self, x):
            s = self.stem(x)
            e1 = self.e1(s)
            e2 = self.e2(e1)
            e3 = self.e3(e2)

            def up(a, b):
                return torch.cat([F.interpolate(a, size=b.shape[-2:], mode="nearest"), b], 1)

            d2 = self.d2(up(e3, e2))
            d1 = self.d1(up(d2, e1))
            d0 = self.d0(up(d1, s))
            return self.head(d0)[:, 0]

    return SlimBallNet()


def models_dir(output_root: Path) -> Path:
    return Path(output_root) / "models" / "ball"


def weights_path(output_root: Path, name: str) -> Path:
    return models_dir(output_root) / f"{name}.pt"


def card_path(output_root: Path, name: str) -> Path:
    return models_dir(output_root) / f"{name}.json"


def list_weights(output_root: Path) -> list[dict]:
    d = models_dir(output_root)
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            card = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if (d / f"{p.stem}.pt").exists():
            out.append(card)
    return out


def stack_inputs(imgs, idx: list[list[int]]):
    """(B, 9, H, W) network input from (N, 3, H, W) frames and per-sample frame indices."""
    import torch

    return torch.cat([imgs[[i[k] for i in idx]] for k in range(len(idx[0]))], 1)


def pad_to(x, mult: int = STRIDE):
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    ph, pw = -(-h // mult) * mult - h, -(-w // mult) * mult - w
    return F.pad(x, (0, pw, 0, ph)) if ph or pw else x


class UNetDetector(BallDetector):
    name = "unet"
    context = max(abs(o) for o in OFFSETS)
    max_candidates = 8

    def __init__(self, spec: DetectorSpec, device: str | None = None):
        import torch

        super().__init__(spec, device)
        from swingvision.settings import load_settings

        if not spec.weights:
            raise ValueError("The unet detector needs weights: unet:<name>[@size]")
        root = load_settings().require_output_root()
        wp = weights_path(root, spec.weights)
        if not wp.exists():
            raise FileNotFoundError(f"No ball model {spec.weights!r} in {models_dir(root)}")
        card = json.loads(card_path(root, spec.weights).read_text(encoding="utf-8"))
        self.card = card
        self.offsets = tuple(card.get("offsets", OFFSETS))
        self.context = max(abs(o) for o in self.offsets)
        self.threshold = float(card.get("threshold", THRESHOLD))
        if spec.input_px is None:
            self.spec = DetectorSpec(spec.name, spec.weights, int(card.get("input_px", 1920)))
        model = build_model(card.get("width", 32), len(self.offsets))
        state = torch.load(wp, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.half = self.device.startswith("cuda")
        model = model.to(self.device).eval()
        if self.half:
            model = model.half().to(memory_format=torch.channels_last)
        self.model = model

    def detect(self, frames, targets):
        import torch

        assert self.crop is not None, "prepare() first"
        if not targets:
            return []
        scale = self.scale
        dtype = torch.half if self.half else torch.float32
        need = sorted({j + o for j in targets for o in self.offsets})
        pos = {j: k for k, j in enumerate(need)}
        out: list[np.ndarray] = []
        with torch.inference_mode():
            imgs = crop_resize([frames[j] for j in need], self.crop, scale, dtype, self.device)
            h, w = imgs.shape[-2:]
            for b in range(0, len(targets), BATCH):
                tb = targets[b : b + BATCH]
                x = stack_inputs(imgs, [[pos[i + o] for o in self.offsets] for i in tb])
                x = pad_to(x).contiguous(memory_format=torch.channels_last)
                heat = torch.sigmoid(self.model(x).float())[:, : -(-h // 2), : -(-w // 2)]
                for c in peaks(heat, self.max_candidates, self.threshold, 5):
                    out.append(to_full_res(c, self.crop, scale / 2))
        return out

    def describe(self) -> dict:
        return {**super().describe(), "card": self.card}


@register("unet")
def _unet(spec: DetectorSpec, device: str | None = None) -> BallDetector:
    return UNetDetector(spec, device)
