"""Frame access.

* Sparse: :func:`grab_frame` / :func:`grab_frames` decode individual upright full-resolution
  frames with ffmpeg (court calibration needs a few dozen spread over the video).
* Dense: :class:`FrameSource` decodes a time range to upright RGB tensors, on the GPU with
  PyNvVideoCodec (NVDEC, ≈300 fps of 4K) or on the CPU with PyAV as a fallback (PLAN.md §3).

Frame numbers are **presentation-order indices** from the container's packet table, which is
the same for both backends (NVDEC's own indices match it), and times come from each frame's
PTS relative to the first video frame, so they line up with the proxy and audio onsets even
for variable-frame-rate phone video.
"""

from __future__ import annotations

import contextlib
import queue
import subprocess
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from swingvision.io.ffmpeg import FFmpegError
from swingvision.storage.schemas import VideoInfo

if TYPE_CHECKING:
    import torch


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


# ---------------------------------------------------------------------------
# Dense decoding
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    index: int  # presentation-order frame number
    t_s: float  # seconds since the first video frame
    image: torch.Tensor  # (3, H, W) uint8, upright (display orientation)


class RateSampler:
    """Keeps the first frame in each ``1/rate_hz`` time slot (deterministic across chunks).

    ``rate_hz=None`` keeps every frame.
    """

    def __init__(self, rate_hz: float | None):
        self.rate_hz = rate_hz
        self._last_slot: int | None = None

    def __call__(self, t_s: float) -> bool:
        if not self.rate_hz:
            return True
        slot = int(np.floor(t_s * self.rate_hz + 1e-6))
        if slot == self._last_slot:
            return False
        self._last_slot = slot
        return True


@dataclass(frozen=True)
class FrameTable:
    """Presentation timestamps of every video frame, from the container's packet index."""

    pts: np.ndarray  # int64, sorted
    time_base: float
    start_pts: int

    def times(self) -> np.ndarray:
        return (self.pts - self.start_pts) * self.time_base

    def index_of_pts(self, pts: int) -> int:
        return int(np.searchsorted(self.pts, pts))

    def index_at(self, t_s: float) -> int:
        """First frame at or after ``t_s``."""
        target = self.start_pts + t_s / self.time_base - 0.5
        return int(np.searchsorted(self.pts, target))

    def __len__(self) -> int:
        return len(self.pts)


@cache
def frame_table(path: str) -> FrameTable:
    """Demux (no decode) every video packet: about 3 s for a 12 GB, 28-minute 4K file."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        pts = np.fromiter((p.pts for p in c.demux(s) if p.pts is not None), dtype=np.int64)
        pts.sort()
        start = s.start_time if s.start_time is not None else (int(pts[0]) if len(pts) else 0)
        return FrameTable(pts=pts, time_base=float(s.time_base), start_pts=int(start))


def upright(t: torch.Tensor, rotation_cw: int) -> torch.Tensor:
    """Rotate a (..., H, W) tensor clockwise by ``rotation_cw`` degrees (0/90/180/270)."""
    if rotation_cw == 180:
        return t.flip(-1).flip(-2)
    if rotation_cw == 90:
        return t.transpose(-1, -2).flip(-1)
    if rotation_cw == 270:
        return t.transpose(-1, -2).flip(-2)
    return t


class FrameSource(ABC):
    """Decode a time range of a video to upright RGB tensors.

    ``frames(t0, t1, keep)`` yields every frame with ``t0 <= t < t1`` for which ``keep(t)`` is
    true. Frames that ``keep`` rejects are still decoded (codecs can't skip them) but never
    copied.
    """

    backend: str = ""

    def __init__(self, path: Path, info: VideoInfo, device: str):
        self.path = Path(path)
        self.info = info
        self.device = device
        self.table = frame_table(str(self.path))

    @abstractmethod
    def frames(
        self, t0: float, t1: float, keep: Callable[[float], bool] | None = None
    ) -> Iterator[Frame]: ...

    def close(self) -> None:
        return None

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class NvdecSource(FrameSource):
    """PyNvVideoCodec ``SimpleDecoder`` → zero-copy DLPack tensors on the GPU.

    A background thread decodes ahead so NVDEC runs alongside inference. Kept frames are
    cloned because the decoder reuses its output surfaces.
    """

    backend = "nvdec"
    BATCH = 8
    SEEK_MARGIN = 8
    QUEUE = 12

    def __init__(self, path: Path, info: VideoInfo, device: str = "cuda"):
        super().__init__(path, info, device)
        import PyNvVideoCodec as nvc

        gpu = int(device.split(":")[1]) if ":" in device else 0
        self._dec = nvc.SimpleDecoder(
            str(self.path),
            gpu_id=gpu,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGBP,
        )

    def frames(self, t0, t1, keep=None):
        import torch

        start = self.table.index_at(t0)
        if start >= len(self.table):
            return
        q: queue.Queue = queue.Queue(maxsize=self.QUEUE)
        stop = threading.Event()
        tb, start_pts, rot = self.table.time_base, self.table.start_pts, self.info.rotation_cw

        def produce() -> None:
            try:
                # NVDEC's demuxer ignores MP4 edit lists, so its frame indices can be offset by
                # a few pre-roll frames (negative PTS). Seek a little early, select by time, and
                # take the index from the PTS.
                self._dec.seek_to_index(max(0, start - self.SEEK_MARGIN))
                while not stop.is_set():
                    batch = self._dec.get_batch_frames(self.BATCH)
                    if not batch:
                        return
                    for f in batch:
                        pts = f.getPTS()
                        t = (pts - start_pts) * tb
                        if t >= t1:
                            return
                        if t >= t0 - 1e-9 and (keep is None or keep(t)):
                            img = upright(torch.from_dlpack(f), rot).clone()
                            q.put(Frame(self.table.index_of_pts(pts), t, img))
            except BaseException as exc:  # re-raised in the consumer
                q.put(exc)
            finally:
                q.put(None)

        thread = threading.Thread(target=produce, name="nvdec", daemon=True)
        thread.start()
        try:
            while (item := q.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            while thread.is_alive():  # unblock a producer waiting on a full queue
                with contextlib.suppress(queue.Empty):
                    q.get(timeout=0.05)
            thread.join()


class PyAVSource(FrameSource):
    """CPU decode with PyAV (≈30 fps of 4K HEVC): the fallback, and what CPU-only tests use."""

    backend = "pyav"

    def frames(self, t0, t1, keep=None):
        import av
        import torch

        tb, start_pts = self.table.time_base, self.table.start_pts
        with av.open(str(self.path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            # Seek to the keyframe at or before t0, then decode forward.
            c.seek(int(start_pts + max(0.0, t0) / tb), stream=s, backward=True, any_frame=False)
            for frame in c.decode(s):
                if frame.pts is None:
                    continue
                t = (frame.pts - start_pts) * tb
                if t < t0 - 1e-9:
                    continue
                if t >= t1:
                    break
                if keep is not None and not keep(t):
                    continue
                arr = frame.to_ndarray(format="rgb24")
                img = upright(torch.from_numpy(arr).permute(2, 0, 1), self.info.rotation_cw)
                img = img.contiguous()
                if self.device != "cpu":
                    img = img.to(self.device, non_blocking=True)
                yield Frame(self.table.index_of_pts(frame.pts), t, img)


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def open_source(
    path: Path, info: VideoInfo, backend: str = "auto", device: str | None = None
) -> FrameSource:
    """``backend``: ``auto`` (NVDEC if it works, else PyAV), ``nvdec`` or ``pyav``."""
    device = device or default_device()
    if backend in ("auto", "nvdec") and device.startswith("cuda"):
        try:
            return NvdecSource(path, info, device)
        except Exception as exc:  # missing package, unsupported codec, no NVDEC
            if backend == "nvdec":
                raise
            logger.warning("NVDEC unavailable ({}); decoding on the CPU", exc)
    return PyAVSource(path, info, device)
