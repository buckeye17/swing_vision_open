"""M0 spike (a): 4K60 decode backends on Windows → upright RGB torch tensors on the GPU.

Usage:  uv run python scripts/spikes/decode_bench.py VIDEO [--start 300] [--frames 600]

Each backend decodes the same run of frames and must hand the pipeline a
CUDA uint8 tensor (3, H, W), upright (rotation applied). Throughput includes
everything needed to get there.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable, Iterator

import numpy as np
import torch

from swingvision.io.probe import probe_video


def _upright(t: torch.Tensor, rotation_cw: int) -> torch.Tensor:
    """t: (3, H, W). Rotate clockwise by rotation_cw."""
    if rotation_cw == 180:
        return t.flip(-1).flip(-2)
    if rotation_cw == 90:
        return t.transpose(-1, -2).flip(-1)
    if rotation_cw == 270:
        return t.transpose(-1, -2).flip(-2)
    return t


def pynvvideocodec_frames(path: str, start_s: float, n: int, rot: int) -> Iterator[torch.Tensor]:
    import PyNvVideoCodec as nvc

    dec = nvc.SimpleDecoder(
        path, gpu_id=0, use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
    )
    dec.seek_to_index(dec.get_index_from_time_in_seconds(start_s))
    got = 0
    while got < n:
        batch = dec.get_batch_frames(min(16, n - got))
        if not batch:
            return
        for f in batch:
            t = torch.from_dlpack(f)  # zero-copy, (3, H, W) uint8 on cuda
            yield _upright(t, rot)
            got += 1


def pyav_frames(
    path: str, start_s: float, n: int, rot: int, hwaccel: bool
) -> Iterator[torch.Tensor]:
    import av

    kwargs = {}
    if hwaccel:
        from av.codec.hwaccel import HWAccel

        kwargs["hwaccel"] = HWAccel(device_type="cuda", allow_software_fallback=False)
    with av.open(path, **kwargs) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(start_s / s.time_base), stream=s)
        got = 0
        for frame in c.decode(s):
            if frame.time is None or frame.time < start_s:
                continue
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) on CPU
            t = torch.from_numpy(arr).cuda(non_blocking=True).permute(2, 0, 1)
            yield _upright(t, rot)
            got += 1
            if got >= n:
                return


def ffmpeg_pipe_frames(
    path: str, start_s: float, n: int, rot: int, w: int, h: int
) -> Iterator[torch.Tensor]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(start_s),
        "-hwaccel",
        "cuda",
        "-i",
        path,
        "-frames:v",
        str(n),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    size = w * h * 3  # ffmpeg auto-rotates, so the frame is already upright
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(h, w, 3)
            yield t.cuda(non_blocking=True).permute(2, 0, 1)
    finally:
        proc.kill()


def bench(name: str, make: Callable[[], Iterator[torch.Tensor]], n: int) -> dict:
    torch.cuda.synchronize()
    try:
        it = make()
        first = next(it)  # exclude open/seek/first-frame latency from throughput
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        count, checksum = 1, float(first.float().mean())
        for t in it:
            count += 1
            if count % 50 == 0:
                checksum += float(t.float().mean())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        fps = (count - 1) / dt if dt > 0 else float("nan")
        res = {
            "backend": name,
            "frames": count,
            "fps": round(fps, 1),
            "shape": list(first.shape),
            "device": str(first.device),
            "mean0": round(float(first.float().mean()), 2),
        }
    except Exception as exc:  # report, don't crash the whole spike
        res = {"backend": name, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=300.0)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()
    info = probe_video("ffprobe", args.video)
    rot = info.rotation_cw
    print(
        json.dumps(
            {
                "video": args.video,
                "size": [info.display_width, info.display_height],
                "codec": info.codec,
                "fps": info.fps_avg,
                "rotation_cw": rot,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
    )
    backends = {
        "pynvvideocodec_nvdec_gpu": lambda: pynvvideocodec_frames(
            args.video, args.start, args.frames, rot
        ),
        "pyav_nvdec_hwaccel": lambda: pyav_frames(args.video, args.start, args.frames, rot, True),
        "ffmpeg_nvdec_pipe": lambda: ffmpeg_pipe_frames(
            args.video, args.start, args.frames, rot, info.display_width, info.display_height
        ),
        "pyav_cpu": lambda: pyav_frames(args.video, args.start, min(args.frames, 240), rot, False),
    }
    results = [bench(k, v, args.frames) for k, v in backends.items() if k not in args.skip]

    # Cross-check orientation/content: compare first frames' mean across backends.
    means = {r["backend"]: r.get("mean0") for r in results if "mean0" in r}
    print(
        json.dumps(
            {
                "first_frame_means": means,
                "spread": float(np.ptp(list(means.values()))) if means else None,
            }
        )
    )


if __name__ == "__main__":
    main()
