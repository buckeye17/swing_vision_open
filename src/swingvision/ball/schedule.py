"""Frame-rate schedules for ball detection (PLAN.md §7.4.2).

A schedule is a list of time windows, each processed at a rate (``None``: every frame).

* **Sweep**: the whole video at a reduced rate (e.g. 15 Hz). Three-frame detectors still get
  their context from the neighboring frames, which are decoded anyway.
* **Refine**: short windows around important moments (hits, bounces, lost ball, audio
  onsets) at the full frame rate, found from the sweep by :func:`moments`.

Between refined windows the trajectory comes from the sweep detections plus interpolation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from swingvision.io.frames import FrameTable


@dataclass(frozen=True)
class Window:
    t0: float
    t1: float
    rate_hz: float | None = None  # None: every frame


@dataclass(frozen=True)
class Schedule:
    """A sweep rate plus refine-window sizes; ``sweep_hz=None`` means full rate everywhere."""

    sweep_hz: float | None = None
    before_s: float = 0.25
    after_s: float = 0.35

    @property
    def full_rate(self) -> bool:
        return not self.sweep_hz

    def label(self) -> str:
        if self.full_rate:
            return "full"
        return f"{self.sweep_hz:g}Hz+{self.before_s:g}/{self.after_s:g}s"

    def as_config(self) -> dict:
        return {"sweep_hz": self.sweep_hz, "before_s": self.before_s, "after_s": self.after_s}


def target_frames(table: FrameTable, windows: Iterable[Window]) -> np.ndarray:
    """Sorted unique frame indices to run the detector on."""
    times = table.times()
    out = []
    for w in windows:
        i0, i1 = table.index_at(w.t0), table.index_at(w.t1)
        idx = np.arange(i0, min(i1, len(times)))
        if w.rate_hz:
            slots = np.floor(times[idx] * w.rate_hz + 1e-6).astype(np.int64)
            first = np.ones(len(idx), dtype=bool)
            first[1:] = slots[1:] != slots[:-1]
            idx = idx[first]
        out.append(idx)
    if not out:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(out))


def with_context(targets: np.ndarray, context: int, n_frames: int) -> np.ndarray:
    """Target frames plus ``context`` neighbors on each side (clipped to the video)."""
    if not len(targets):
        return targets
    offs = np.arange(-context, context + 1)
    idx = (targets[:, None] + offs[None, :]).ravel()
    return np.unique(idx[(idx >= 0) & (idx < n_frames)])


def decode_spans(
    needed: np.ndarray, times: np.ndarray, max_gap_s: float = 1.5
) -> list[tuple[int, int]]:
    """Group needed frames into index spans ``[i0, i1]`` decoded in one go.

    Decoding through a short gap is cheaper than seeking (a seek restarts at a keyframe).
    """
    if not len(needed):
        return []
    spans = []
    start = prev = int(needed[0])
    for i in needed[1:]:
        i = int(i)
        if times[i] - times[prev] > max_gap_s:
            spans.append((start, prev))
            start = i
        prev = i
    spans.append((start, prev))
    return spans


def merge_windows(
    centers: Sequence[float], before_s: float, after_s: float, duration_s: float
) -> list[Window]:
    """Full-rate windows ``[c - before, c + after]`` around moments, overlapping ones merged."""
    if not len(centers):
        return []
    iv = sorted((max(0.0, c - before_s), min(duration_s, c + after_s)) for c in centers)
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [Window(a, b, None) for a, b in merged if b > a]
