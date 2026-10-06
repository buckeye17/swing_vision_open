"""Where to run pose: windows around swings, and the player's box on every frame."""

from __future__ import annotations

import numpy as np


def merge_windows(
    anchors: np.ndarray, before_s: float, after_s: float, duration_s: float | None = None
) -> list[tuple[float, float]]:
    """``[t - before, t + after]`` around each anchor time, overlapping ones merged."""
    out: list[list[float]] = []
    for t in np.sort(np.asarray(anchors, dtype=np.float64)):
        lo, hi = max(0.0, t - before_s), t + after_s
        if duration_s is not None:
            hi = min(hi, duration_s)
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


class BoxTrack:
    """The player's image box over time, from ``movement`` (15 Hz, gaps bridged).

    :meth:`at` interpolates linearly between the two nearest rows when they are close in
    time, and returns ``None`` where the player wasn't tracked.
    """

    def __init__(self, t: np.ndarray, boxes: np.ndarray, max_gap_s: float = 0.3):
        order = np.argsort(t)
        self.t = np.asarray(t, dtype=np.float64)[order]
        self.boxes = np.asarray(boxes, dtype=np.float64)[order]
        ok = np.isfinite(self.boxes).all(1)
        self.t, self.boxes = self.t[ok], self.boxes[ok]
        self.max_gap_s = max_gap_s

    def at(self, t: float) -> np.ndarray | None:
        n = len(self.t)
        if n == 0:
            return None
        i = int(np.searchsorted(self.t, t))
        if 0 < i < n:
            t0, t1 = self.t[i - 1], self.t[i]
            if t1 - t0 <= self.max_gap_s:
                w = (t - t0) / max(t1 - t0, 1e-9)
                return (1 - w) * self.boxes[i - 1] + w * self.boxes[i]
        j = min(max(i, 0), n - 1)
        if i > 0 and abs(self.t[i - 1] - t) < abs(self.t[j] - t):
            j = i - 1
        return self.boxes[j].copy() if abs(self.t[j] - t) <= self.max_gap_s / 2 else None

    def covered(self, t0: float, t1: float) -> bool:
        """Any tracked box in [t0, t1]?"""
        a, b = np.searchsorted(self.t, [t0, t1])
        return bool(b > a)
