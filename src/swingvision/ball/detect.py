"""Run a ball detector over a set of frames of a video (PLAN.md §7.4).

The frames to detect on come from a schedule (:mod:`swingvision.ball.schedule`). They are
decoded with their temporal context, cropped to the ball region (the court ROI extended up
to the top of the picture, for lobs and serve tosses), and passed to the detector in runs of
consecutive frames. The result is a candidates table (several per frame, full-resolution
pixels) plus the list of frames that were looked at, so the tracker can tell "no ball found"
from "not looked at".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from swingvision.ball.detectors.base import BallDetector
from swingvision.ball.schedule import decode_spans, with_context
from swingvision.court import calibration as calib
from swingvision.io.frames import Frame, FrameSource
from swingvision.players.detect import Crop, roi_crop
from swingvision.storage.schemas import BALL_CANDIDATES, BALL_FRAMES, Calibration

#: Frames passed to the detector per call (plus context).
RUN = 24


def ball_crop(cal: Calibration, behind_m: float, beside_m: float) -> Crop:
    """The court ROI crop extended to the top of the frame (lobs and tosses go above it)."""
    cams = calib.all_cameras(cal)
    c = roi_crop(cams, cal.camera.width, cal.camera.height, behind_m, beside_m)
    return Crop(c.x0, 0, c.x1, c.y1)


@dataclass
class DetectResult:
    candidates: pa.Table
    frames: pa.Table
    seconds: float  # wall time spent (decode + detect)
    detect_seconds: float  # time inside the detector only


class StreamDetector:
    """Feeds decoded frames (in increasing frame order) to a detector in contiguous runs.

    ``is_target[i]`` says whether frame ``i`` should be detected on; the caller pushes every
    frame a target needs (``with_context``). Detections accumulate until :meth:`take`.
    """

    def __init__(self, detector: BallDetector, is_target: np.ndarray, times: np.ndarray):
        self.detector = detector
        self.is_target = is_target
        self.times = times
        self.buf: list[Frame] = []
        self.rows: list[tuple] = []
        self.done: list[int] = []
        self.processed: set[int] = set()
        self.detect_seconds = 0.0

    def push(self, f: Frame) -> None:
        if self.buf and f.index != self.buf[-1].index + 1:
            self._process(final=True)
        self.buf.append(f)
        if len(self.buf) >= RUN + 2 * self.detector.context:
            self._process(final=False)

    def finish(self) -> None:
        if self.buf:
            self._process(final=True)

    def _process(self, final: bool) -> None:
        import time

        import torch

        buf, ctx = self.buf, self.detector.context
        hi = len(buf) if final else len(buf) - ctx
        pos = [
            k
            for k in range(ctx, hi)
            if self.is_target[buf[k].index]
            and buf[k].index not in self.processed
            and k + ctx < len(buf)
            and buf[k + ctx].index == buf[k].index + ctx
            and buf[k - ctx].index == buf[k].index - ctx
        ]
        if pos:
            t0 = time.perf_counter()
            res = self.detector.detect([f.image for f in buf], pos)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.detect_seconds += time.perf_counter() - t0
            for k, cands in zip(pos, res, strict=True):
                f = buf[k]
                self.processed.add(f.index)
                self.done.append(f.index)
                for x, y, sc in cands.tolist():
                    self.rows.append((f.index, f.t_s, x, y, sc))
        self.buf = [] if final else buf[-2 * ctx :]

    def take(self) -> tuple[pa.Table, pa.Table]:
        """Candidates and processed frames since the last call."""
        cand = _table(self.rows, BALL_CANDIDATES)
        done = np.array(sorted(self.done), dtype=np.int64)
        frames = pa.table(
            {"frame": pa.array(done, pa.int64()), "t_s": pa.array(self.times[done], pa.float64())},
            schema=BALL_FRAMES,
        )
        self.rows, self.done = [], []
        return cand, frames


def target_mask(targets: np.ndarray, context: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(is_target, is_needed) boolean masks over all ``n`` frames."""
    targets = np.unique(np.asarray(targets, dtype=np.int64))
    targets = targets[(targets >= context) & (targets < n - context)]
    is_target = np.zeros(n, dtype=bool)
    is_target[targets] = True
    is_needed = np.zeros(n, dtype=bool)
    is_needed[with_context(targets, context, n)] = True
    return is_target, is_needed


def run_detector(
    source: FrameSource,
    detector: BallDetector,
    targets: np.ndarray,
    progress: Callable[[float], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> DetectResult:
    """Detect on every frame in ``targets`` (frame indices). The detector must be prepared."""
    import time

    table = source.table
    times = table.times()
    is_target, is_needed = target_mask(targets, detector.context, len(times))
    needed = np.flatnonzero(is_needed)
    total = max(1, int(is_target.sum()))
    stream = StreamDetector(detector, is_target, times)
    t_start = time.perf_counter()

    def keep(t: float) -> bool:
        i = table.index_at(t - 1e-6)
        return bool(i < len(is_needed) and is_needed[i])

    for i0, i1 in decode_spans(needed, times):
        t0 = float(times[i0]) - 1e-4
        t1 = float(times[i1]) + 1e-4
        for k, f in enumerate(source.frames(t0, t1, keep)):
            stream.push(f)
            if k % RUN == 0:
                if check_cancel:
                    check_cancel()
                if progress:
                    progress(len(stream.processed) / total)
        stream.finish()
    cand, frames = stream.take()
    return DetectResult(cand, frames, time.perf_counter() - t_start, stream.detect_seconds)


def _table(rows: list[tuple], schema: pa.Schema) -> pa.Table:
    if not rows:
        return schema.empty_table()
    cols = list(zip(*rows, strict=True))
    return pa.table(
        {f.name: pa.array(c, f.type) for f, c in zip(schema, cols, strict=True)}, schema=schema
    )
