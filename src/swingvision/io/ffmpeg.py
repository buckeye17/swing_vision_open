"""Run ffmpeg with machine-readable progress and cooperative cancellation."""

from __future__ import annotations

import base64
import contextlib
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from functools import cache
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


class FFmpegCancelled(RuntimeError):
    pass


@cache
def available_encoders(ffmpeg: str) -> frozenset[str]:
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
    names = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)


def run_ffmpeg(
    ffmpeg: str,
    args: list[str],
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> None:
    """Run ``ffmpeg <args>``; report output-time progress as a 0..1 fraction of ``duration_s``."""
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-progress", "pipe:1", *args]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stderr_tail: deque[str] = deque(maxlen=40)
    reader = threading.Thread(
        target=lambda: [stderr_tail.append(line.rstrip()) for line in proc.stderr], daemon=True
    )
    reader.start()
    cancelled = False
    assert proc.stdout is not None
    for line in proc.stdout:
        if is_cancelled and is_cancelled():
            cancelled = True
            proc.kill()
            break
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and on_progress and duration_s:
            with contextlib.suppress(ValueError):
                on_progress(min(1.0, max(0.0, int(value) / 1e6 / duration_s)))
    proc.wait()
    reader.join(timeout=5)
    if cancelled:
        raise FFmpegCancelled()
    if proc.returncode != 0:
        tail = "\n".join(stderr_tail)
        raise FFmpegError(f"ffmpeg exited with {proc.returncode}:\n{tail}")


def thumbnail_time(duration_s: float) -> float:
    """10% in: past camera setup at the start, but early enough to be representative."""
    return max(0.0, duration_s * 0.1)


def thumbnail_jpeg(ffmpeg: str, src: Path, t_s: float, width: int = 640) -> bytes:
    """Grab one upright frame as JPEG bytes (ffmpeg auto-rotates software frames)."""
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{t_s:.3f}",
        "-i",
        str(src),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "4",
        "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0 or not out.stdout:
        raise FFmpegError(out.stderr.decode(errors="replace").strip() or "thumbnail failed")
    return out.stdout


def thumbnail_data_uri(ffmpeg: str, src: Path, t_s: float, width: int = 640) -> str:
    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(thumbnail_jpeg(ffmpeg, src, t_s, width)).decode()
    )
