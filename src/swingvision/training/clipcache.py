"""Decoded frames of labeling clips, cached as full-resolution JPEGs.

Seeking an HEVC phone video to an exact frame takes up to a second, which is far too slow to
step through a clip while labeling. Opening a clip decodes it once (plus a few frames of context
on each side, which the detectors need) into
``training/ball/<session_id>/cache/<clip_id>/f<frame>.jpg``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from swingvision.io.frames import frame_table, open_source
from swingvision.storage.fsutil import atomic_write
from swingvision.storage.schemas import SessionConfig
from swingvision.training.labels import BallClip, LabelStore

#: Frames of context on each side (the motion detector compares with frames 3 away).
CONTEXT = 3
QUALITY = 93


def frame_path(store: LabelStore, clip: BallClip, frame: int) -> Path:
    return store.cache_dir(clip.session_id, clip.clip_id) / f"f{frame:07d}.jpg"


def cached_frames(clip: BallClip) -> range:
    return range(max(0, clip.frame0 - CONTEXT), clip.frame1 + CONTEXT + 1)


def is_cached(store: LabelStore, clip: BallClip) -> bool:
    return all(frame_path(store, clip, f).exists() for f in cached_frames(clip))


def ensure_cached(
    store: LabelStore,
    clip: BallClip,
    config: SessionConfig,
    backend: str = "auto",
    progress: Callable[[float], None] | None = None,
) -> Path:
    """Decode the clip's frames into the cache (no-op when they're all there)."""
    out = store.cache_dir(clip.session_id, clip.clip_id)
    if is_cached(store, clip):
        return out
    assert config.video is not None
    src_path = Path(config.source.path)
    table = frame_table(str(src_path))
    times = table.times()
    frames = cached_frames(clip)
    f_hi = min(frames.stop - 1, len(times) - 1)
    t0, t1 = float(times[frames.start]) - 1e-4, float(times[f_hi]) + 1e-4
    n = len(frames)
    with open_source(src_path, config.video, backend) as src:
        for k, f in enumerate(src.frames(t0, t1)):
            if f.index not in frames:
                continue
            rgb = f.image.permute(1, 2, 0).cpu().numpy()
            write_jpeg(frame_path(store, clip, f.index), rgb)
            if progress:
                progress((k + 1) / n)
    return out


def write_jpeg(path: Path, rgb: np.ndarray, quality: int = QUALITY) -> None:
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    atomic_write(path, lambda tmp: tmp.write_bytes(buf.tobytes()), suffix=".jpg")


def read_frame(store: LabelStore, clip: BallClip, frame: int) -> np.ndarray | None:
    """RGB (H, W, 3) uint8, or None when not cached."""
    p = frame_path(store, clip, frame)
    if not p.exists():
        return None
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
