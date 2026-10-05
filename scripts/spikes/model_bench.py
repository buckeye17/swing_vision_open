"""M0 spike (b): person-detector and ball-network throughput on this GPU.

Usage:  uv run python scripts/spikes/model_bench.py [VIDEO]

Networks are built from their architecture definitions with random weights:
compute cost is identical to trained weights, and nothing is downloaded.
Measures pure network forward passes (FP16, channels-last) at candidate input
sizes, then an end-to-end "pass 1" estimate: NVDEC decode → GPU resize →
YOLO + ball net on every frame.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNetLike(nn.Module):
    """TrackNetV2/V3-sized U-Net: 3 RGB frames + background (12 ch) → 3 heatmaps."""

    def __init__(self, in_ch: int = 12, out_ch: int = 3):
        super().__init__()

        def block(cin, cout, n):
            layers = []
            for i in range(n):
                layers += [
                    nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                ]
            return nn.Sequential(*layers)

        self.e1, self.e2, self.e3 = block(in_ch, 64, 2), block(64, 128, 2), block(128, 256, 3)
        self.bott = block(256, 512, 3)
        self.d3, self.d2, self.d1 = block(768, 256, 3), block(384, 128, 2), block(192, 64, 2)
        self.head = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return torch.sigmoid(self.head(d1))


def yolo(name: str) -> nn.Module:
    from ultralytics import YOLO

    return YOLO(f"{name}.yaml").model


def prep(m: nn.Module) -> nn.Module:
    return m.cuda().half().eval().to(memory_format=torch.channels_last)


@torch.inference_mode()
def time_forward(m: nn.Module, shape: tuple[int, ...], iters: int = 30) -> float:
    x = torch.rand(shape, device="cuda", dtype=torch.half).to(memory_format=torch.channels_last)
    for _ in range(5):
        m(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        m(x)
    torch.cuda.synchronize()
    return shape[0] * iters / (time.perf_counter() - t0)  # images per second


@torch.inference_mode()
def end_to_end(video: str, det: nn.Module, ball: nn.Module, det_hw, ball_hw, n=600, batch=8):
    import sys

    sys.path.insert(0, "scripts/spikes")
    from decode_bench import pynvvideocodec_frames

    from swingvision.io.probe import probe_video

    rot = probe_video("ffprobe", video).rotation_cw
    frames = pynvvideocodec_frames(video, 300.0, n, rot)
    bg = torch.zeros((1, 3, *ball_hw), device="cuda", dtype=torch.half)
    prev: list[torch.Tensor] = []
    buf: list[torch.Tensor] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    count = 0
    for f in frames:
        x = f[None].half().div_(255)
        d = F.interpolate(x, size=det_hw, mode="bilinear", antialias=False)
        b = F.interpolate(x, size=ball_hw, mode="bilinear", antialias=False)
        prev = ([*prev, b])[-3:]
        if len(prev) == 3:
            buf.append((d, torch.cat([*prev, bg], 1)))
        if len(buf) == batch:
            det(torch.cat([p[0] for p in buf]).contiguous(memory_format=torch.channels_last))
            ball(torch.cat([p[1] for p in buf]).contiguous(memory_format=torch.channels_last))
            buf.clear()
        count += 1
    torch.cuda.synchronize()
    return count / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}))

    det_hw = (736, 1280)  # 4K → 1280 wide, padded to a multiple of 32
    for name in ("yolo11s", "yolo11m", "yolo11l", "yolo11x"):
        m = prep(yolo(name))
        for bs in (1, 8):
            ips = time_forward(m, (bs, 3, *det_hw))
            print(
                json.dumps({"model": name, "input": det_hw, "batch": bs, "img_per_s": round(ips)})
            )
        del m
        torch.cuda.empty_cache()

    ball = prep(TrackNetLike())
    params = sum(p.numel() for p in ball.parameters()) / 1e6
    for hw in ((288, 512), (576, 1024), (720, 1280)):
        for bs in (1, 8):
            torch.cuda.reset_peak_memory_stats()
            ips = time_forward(ball, (bs, 12, *hw))
            print(
                json.dumps(
                    {
                        "model": f"tracknet_like({params:.1f}M)",
                        "input": hw,
                        "batch": bs,
                        "img_per_s": round(ips),
                        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    }
                )
            )

    if args.video:
        det = prep(yolo("yolo11l"))
        torch.cuda.reset_peak_memory_stats()
        fps = end_to_end(args.video, det, ball, det_hw, (720, 1280))
        print(
            json.dumps(
                {
                    "end_to_end": "nvdec→resize→yolo11l@1280 + tracknet@1280x720, batch 8",
                    "fps": round(fps, 1),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
            )
        )


if __name__ == "__main__":
    main()
