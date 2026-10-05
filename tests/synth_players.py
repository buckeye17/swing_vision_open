"""Synthetic person detections from a known camera (for tracking and movement tests)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa

from swingvision.court import calibration as calib
from swingvision.court.camera import Camera
from swingvision.storage.schemas import PASS1_FRAMES, PERSON_DETECTIONS, Calibration


def calibration_for(cam: Camera) -> Calibration:
    return Calibration(source="auto", created_at=datetime.now(UTC), camera=calib.to_params(cam))


def person_box(cam: Camera, x: float, y: float, height: float = 1.8, width: float = 0.6):
    """Image box (x0, y0, x1, y1) of an upright person standing at court (x, y)."""
    pts = np.array(
        [[x + dx, y, z] for dx in (-width / 2, width / 2) for z in (0.0, height)], dtype=float
    )
    px = cam.project(pts)
    # The foot point is the bottom center: put the box bottom exactly at the projected feet.
    foot = cam.project(np.array([[x, y, 0.0]]))[0]
    return (px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), foot[1])


class DetectionBuilder:
    """Accumulates per-frame detections at ``rate_hz`` and returns the two pass-1 tables."""

    def __init__(self, cam: Camera, duration_s: float, rate_hz: float = 15.0, seed: int = 0):
        self.cam = cam
        self.rate = rate_hz
        self.t = np.arange(0, duration_s, 1 / rate_hz)
        self.frame = np.round(self.t * 60).astype(np.int64)
        self.rows: list[tuple] = []
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        xy_at,
        height: float = 1.8,
        conf: float = 0.85,
        jitter_px: float = 1.5,
        present=None,
        label: str = "",
    ) -> np.ndarray:
        """``xy_at(t) -> (x, y)``; ``present(t) -> bool``. Returns the row indices added."""
        start = len(self.rows)
        for f, t in zip(self.frame, self.t, strict=True):
            if present is not None and not present(t):
                continue
            x, y = xy_at(t)
            b = np.array(person_box(self.cam, x, y, height))
            b += self.rng.normal(0, jitter_px, 4)
            self.rows.append((int(f), float(t), *b.tolist(), conf, label))
        return np.arange(start, len(self.rows))

    def tables(self) -> tuple[pa.Table, pa.Table, list[str]]:
        rows = self.rows  # insertion order: add() returned indices into it
        cols = list(zip(*rows, strict=True))
        det = pa.table(
            {f.name: pa.array(c, f.type) for f, c in zip(PERSON_DETECTIONS, cols[:7], strict=True)},
            schema=PERSON_DETECTIONS,
        )
        frames = pa.table(
            {
                "frame": pa.array(self.frame, pa.int64()),
                "t_s": pa.array(self.t, pa.float64()),
                "luma": pa.array(np.full(len(self.t), 120.0), pa.float32()),
                "view": pa.array(np.full(len(self.t), 0.9), pa.float32()),
            },
            schema=PASS1_FRAMES,
        )
        return det, frames, list(cols[7])
