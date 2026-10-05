"""Sparse frame access: grab individual upright full-resolution frames.

Used for court calibration, which needs a few dozen frames spread over the
video. ffmpeg handles seeking and auto-rotation. Dense, GPU-resident decoding
(``FrameSource`` with PyNvVideoCodec, PLAN.md §3) arrives with ``pass1_detect``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from swingvision.io.ffmpeg import FFmpegError
from swingvision.storage.schemas import VideoInfo


def grab_frame(ffmpeg: str, src: Path, t_s: float, info: VideoInfo) -> np.ndarray:
    """One upright RGB frame (H, W, 3) uint8 at ``t_s`` seconds (display size)."""
    w, h = info.display_width, info.display_height
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{max(0.0, t_s):.3f}",
        "-i",
        str(src),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, check=False)
    expected = w * h * 3
    if out.returncode != 0 or len(out.stdout) < expected:
        msg = out.stderr.decode(errors="replace").strip() or "no frame decoded"
        raise FFmpegError(f"Could not grab frame at {t_s:.2f}s: {msg}")
    return np.frombuffer(out.stdout[:expected], dtype=np.uint8).reshape(h, w, 3)


def grab_frames(
    ffmpeg: str,
    src: Path,
    times: Iterable[float],
    info: VideoInfo,
    workers: int = 3,
    on_frame: Callable[[int], None] | None = None,
) -> list[np.ndarray | None]:
    """Grab several frames in parallel; failed grabs come back as ``None``."""
    times = list(times)
    done = 0

    def one(t: float) -> np.ndarray | None:
        nonlocal done
        try:
            return grab_frame(ffmpeg, src, t, info)
        except FFmpegError:
            return None
        finally:
            done += 1
            if on_frame:
                on_frame(done)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, times))


def median_image(frames: list[np.ndarray], band_rows: int = 128) -> np.ndarray:
    """Per-pixel median of same-size uint8 frames, computed in row bands to bound memory.

    Removes moving players and balls, leaving the static court.
    """
    stack = np.stack(frames)
    out = np.empty(stack.shape[1:], dtype=np.uint8)
    for r in range(0, stack.shape[1], band_rows):
        out[r : r + band_rows] = np.median(stack[:, r : r + band_rows], axis=0).astype(np.uint8)
    return out


def luminance(frame: np.ndarray) -> float:
    """Mean luma of a downsampled copy (0-255), to skip frames that are too dark to use."""
    small = frame[::8, ::8].astype(np.float32)
    return float(np.mean(small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)))
