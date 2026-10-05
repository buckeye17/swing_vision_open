"""ffprobe wrapper → :class:`VideoInfo`, plus a fast content hash for relinking moved files."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from swingvision.storage.schemas import SourceInfo, VideoInfo

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
HASH_BLOCK = 8 * 1024 * 1024
VFR_TOLERANCE = 0.005  # relative difference between nominal and average fps


class ProbeError(RuntimeError):
    pass


def run_ffprobe(ffprobe: str, path: Path) -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, text=True, encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        raise ProbeError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc
    return json.loads(out.stdout)


def _fraction(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rotation_cw(stream: dict[str, Any]) -> int:
    """Clockwise rotation needed for upright display.

    ffprobe reports the display-matrix rotation counter-clockwise (e.g. ``-180``
    or ``-90`` for phones); legacy files use a ``rotate`` tag that is clockwise.
    """
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            return round(-float(side["rotation"])) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        return int(tag) % 360
    return 0


def parse_probe(data: dict[str, Any]) -> VideoInfo:
    streams = data.get("streams", [])
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise ProbeError("No video stream found")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    fps_nominal = _fraction(video.get("r_frame_rate"))
    fps_avg = _fraction(video.get("avg_frame_rate")) or fps_nominal
    duration = _float(video.get("duration")) or _float(fmt.get("duration")) or 0.0
    nb_frames = video.get("nb_frames")
    n_frames = (
        int(nb_frames) if nb_frames and str(nb_frames).isdigit() else round(duration * fps_avg)
    )
    rotation = _rotation_cw(video)
    w, h = int(video["width"]), int(video["height"])
    dw, dh = (h, w) if rotation in (90, 270) else (w, h)
    is_vfr = bool(fps_nominal) and abs(fps_nominal - fps_avg) / fps_nominal > VFR_TOLERANCE

    return VideoInfo(
        codec=video.get("codec_name", "unknown"),
        profile=video.get("profile"),
        width=w,
        height=h,
        rotation_cw=rotation,
        display_width=dw,
        display_height=dh,
        pix_fmt=video.get("pix_fmt"),
        fps_nominal=fps_nominal,
        fps_avg=fps_avg,
        is_vfr=is_vfr,
        duration_s=duration,
        n_frames_est=n_frames,
        bit_rate=int(video["bit_rate"]) if str(video.get("bit_rate", "")).isdigit() else None,
        start_time_s=_float(video.get("start_time")) or 0.0,
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        audio_channels=audio.get("channels") if audio else None,
        audio_start_time_s=_float(audio.get("start_time")) if audio else None,
        creation_time=(fmt.get("tags") or {}).get("creation_time"),
    )


def probe_video(ffprobe: str, path: Path) -> VideoInfo:
    return parse_probe(run_ffprobe(ffprobe, path))


def fast_hash(path: Path) -> str:
    """Hash of size + first and last 8 MB. Identifies a file cheaply even if it moves."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(HASH_BLOCK))
        if size > HASH_BLOCK:
            f.seek(max(HASH_BLOCK, size - HASH_BLOCK))
            h.update(f.read(HASH_BLOCK))
    return h.hexdigest()[:32]


def source_info(path: Path) -> SourceInfo:
    path = Path(path).resolve()
    st = path.stat()
    return SourceInfo(
        path=str(path), size_bytes=st.st_size, fast_hash=fast_hash(path), mtime=st.st_mtime
    )
