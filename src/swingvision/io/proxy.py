"""Browser-playable H.264 proxy of the source footage.

The proxy keeps source timestamps (``-fps_mode passthrough``) so proxy time ==
session time, including for variable-frame-rate phone footage.

Primary path (see docs/spikes/m0-video-pipeline.md): NVDEC decode → ``scale_cuda``
→ download → explicit rotation filters → NVENC. ``-display_rotation 0`` stops
ffmpeg auto-rotating *and* stops the rotation flag leaking into the output.
Fallback: CPU scaling with ffmpeg's normal auto-rotation, then any H.264 encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loguru import logger

from swingvision.io.ffmpeg import FFmpegCancelled, FFmpegError, available_encoders, run_ffmpeg
from swingvision.storage.fsutil import atomic_write
from swingvision.storage.schemas import VideoInfo

ROTATION_FILTERS = {
    0: [],
    90: ["transpose=clock"],
    180: ["hflip", "vflip"],
    270: ["transpose=cclock"],
}


def _video_encoder_args(encoders: frozenset[str], gop: int) -> list[str]:
    if "h264_nvenc" in encoders:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "25",
            "-b:v",
            "0",
            "-g",
            str(gop),
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-g",
        str(gop),
        "-pix_fmt",
        "yuv420p",
    ]


def _common_output_args(dst: Path, video_args: list[str]) -> list[str]:
    return [
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-fps_mode",
        "passthrough",
        *video_args,
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        "-y",
        str(dst),
    ]


def gpu_proxy_args(
    src: Path, dst: Path, info: VideoInfo, height: int, gop: int, encoders: frozenset[str]
) -> list[str]:
    vf = ",".join(
        [
            f"scale_cuda=-2:{height}",
            "hwdownload",
            "format=nv12",
            *ROTATION_FILTERS.get(info.rotation_cw, []),
        ]
    )
    if info.rotation_cw in (90, 270):  # height applies before the transpose
        vf = vf.replace(f"scale_cuda=-2:{height}", f"scale_cuda={height}:-2")
    return [
        "-display_rotation",
        "0",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(src),
        "-vf",
        vf,
        *_common_output_args(dst, _video_encoder_args(encoders, gop)),
    ]


def cpu_proxy_args(
    src: Path, dst: Path, height: int, gop: int, encoders: frozenset[str]
) -> list[str]:
    return [
        "-i",
        str(src),
        "-vf",
        f"scale=-2:{height}:flags=bicubic,format=yuv420p",
        *_common_output_args(dst, _video_encoder_args(encoders, gop)),
    ]


def make_proxy(
    ffmpeg: str,
    src: Path,
    dst: Path,
    info: VideoInfo,
    height: int = 720,
    gop: int = 30,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> str:
    """Write the proxy atomically; returns which path was used ("gpu" or "cpu")."""
    encoders = available_encoders(ffmpeg)
    attempts: list[tuple[str, Callable[[Path], list[str]]]] = []
    if "h264_nvenc" in encoders:
        attempts.append(("gpu", lambda tmp: gpu_proxy_args(src, tmp, info, height, gop, encoders)))
    attempts.append(("cpu", lambda tmp: cpu_proxy_args(src, tmp, height, gop, encoders)))

    last_error: Exception | None = None
    for name, build in attempts:
        try:
            atomic_write(
                dst,
                lambda tmp, build=build: run_ffmpeg(
                    ffmpeg, build(tmp), info.duration_s, on_progress, is_cancelled
                ),
                suffix=".mp4",
            )
            return name
        except FFmpegCancelled:
            raise
        except FFmpegError as exc:
            logger.warning("Proxy via {} path failed, trying next: {}", name, exc)
            last_error = exc
    raise FFmpegError(f"All proxy encode paths failed: {last_error}")
