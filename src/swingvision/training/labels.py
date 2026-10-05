"""Ball and event labels (PLAN.md §10).

Labels live in the output folder, one JSON file per *clip* (a contiguous frame range of one
session): ``<output_root>/training/ball/<session_id>/<clip_id>.json``.

* **Ground-truth clips** (``kind="gt"``) are labeled at the full frame rate: every frame says
  where the ball in play is, or that it isn't visible. The detector benchmark and ``sv eval``
  score against them (PLAN.md §7.4.1).
* **Sample clips** (``kind="sample"``) are single frames (or short runs) picked for training:
  low-confidence detections, frames near events, track gaps, random frames.

Each frame label is a ball position (full-resolution display pixels) with a visibility:
``visible``; ``occluded`` (in play but hidden, e.g. behind the player: not scored);
``none`` (no ball in play in view). Clips also carry event labels (hit, bounce, net) at the
frame closest to the contact.

Keyframe interpolation: label every few frames, then :func:`interpolate` fills the frames in
between with a quadratic fit through the neighboring keyframes (never across an event, where
the motion breaks), marked ``src="interp"`` so the labeler can verify them.
"""

from __future__ import annotations

import itertools
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field

from swingvision.storage.fsutil import atomic_write_text

LABELS_DIR = "training"
Visibility = Literal["visible", "occluded", "none"]
LabelSource = Literal["click", "accepted", "interp", "assisted", "snapped"]
EventKind = Literal["hit", "bounce", "net"]
ClipKind = Literal["gt", "sample"]

#: Visibility values that the detector is scored on.
SCORED = ("visible", "none")


class BallLabel(BaseModel):
    vis: Visibility = "visible"
    x: float | None = None
    y: float | None = None
    src: LabelSource = "click"


class EventLabel(BaseModel):
    frame: int
    kind: EventKind
    x: float | None = None
    y: float | None = None


class BallClip(BaseModel):
    schema_version: int = 1
    session_id: str
    clip_id: str
    kind: ClipKind = "gt"
    frame0: int  # first frame (inclusive)
    frame1: int  # last frame (inclusive)
    t0_s: float
    t1_s: float
    #: ``train`` / ``test``; ``None`` lets the trainer decide (by session).
    split: Literal["train", "test"] | None = None
    status: Literal["in_progress", "done"] = "in_progress"
    keyframe_step: int = 5
    #: Frame number (as a string key, for JSON) → label.
    labels: dict[str, BallLabel] = Field(default_factory=dict)
    events: list[EventLabel] = Field(default_factory=list)
    #: False when the clip's events weren't labeled (event metrics skip it).
    events_labeled: bool = True
    notes: str = ""
    updated_at: str = ""

    @property
    def frames(self) -> range:
        return range(self.frame0, self.frame1 + 1)

    def label(self, frame: int) -> BallLabel | None:
        return self.labels.get(str(frame))

    def set(self, frame: int, label: BallLabel | None) -> None:
        if label is None:
            self.labels.pop(str(frame), None)
        else:
            self.labels[str(frame)] = label

    @property
    def n_labeled(self) -> int:
        return sum(1 for f in self.frames if str(f) in self.labels)

    @property
    def complete(self) -> bool:
        return self.n_labeled == len(self.frames)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class LabelStore:
    """All ball labels of one output folder."""

    def __init__(self, output_root: Path):
        self.root = Path(output_root) / LABELS_DIR / "ball"

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def clip_path(self, session_id: str, clip_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", session_id
        ):
            raise ValueError("bad id")
        return self.session_dir(session_id) / f"{clip_id}.json"

    def cache_dir(self, session_id: str, clip_id: str) -> Path:
        """Decoded frames of a clip (JPEG), for the labeling page."""
        return self.session_dir(session_id) / "cache" / clip_id

    def list_clips(self, session_id: str | None = None) -> list[BallClip]:
        dirs = [self.session_dir(session_id)] if session_id else sorted(self.root.glob("*"))
        out = []
        for d in dirs:
            for p in sorted(d.glob("*.json")):
                try:
                    out.append(BallClip.model_validate_json(p.read_text(encoding="utf-8")))
                except ValueError:
                    continue
        return sorted(out, key=lambda c: (c.session_id, c.frame0))

    def get(self, session_id: str, clip_id: str) -> BallClip | None:
        p = self.clip_path(session_id, clip_id)
        if not p.exists():
            return None
        return BallClip.model_validate_json(p.read_text(encoding="utf-8"))

    def save(self, clip: BallClip) -> None:
        clip.updated_at = _now()
        atomic_write_text(
            self.clip_path(clip.session_id, clip.clip_id), clip.model_dump_json(indent=1)
        )

    def delete(self, session_id: str, clip_id: str) -> None:
        import shutil

        self.clip_path(session_id, clip_id).unlink(missing_ok=True)
        shutil.rmtree(self.cache_dir(session_id, clip_id), ignore_errors=True)

    def new_clip(
        self,
        session_id: str,
        frame0: int,
        frame1: int,
        t0_s: float,
        t1_s: float,
        kind: ClipKind = "gt",
        split: Literal["train", "test"] | None = None,
    ) -> BallClip:
        existing = {c.clip_id for c in self.list_clips(session_id)}
        n = 1
        while f"{kind}{n:04d}" in existing:
            n += 1
        clip = BallClip(
            session_id=session_id,
            clip_id=f"{kind}{n:04d}",
            kind=kind,
            frame0=frame0,
            frame1=frame1,
            t0_s=t0_s,
            t1_s=t1_s,
            split=split,
            keyframe_step=5 if kind == "gt" else 1,
        )
        self.save(clip)
        return clip

    def table(self, session_id: str | None = None, kinds: tuple[str, ...] = ("gt", "sample")):
        """Every frame label as a table (one row per labeled frame)."""
        rows = []
        for c in self.list_clips(session_id):
            if c.kind not in kinds:
                continue
            for f in c.frames:
                lab = c.label(f)
                if lab is None:
                    continue
                rows.append(
                    (c.session_id, c.clip_id, c.kind, c.split, f, lab.vis, lab.x, lab.y, lab.src)
                )
        cols = list(zip(*rows, strict=True)) if rows else [[] for _ in LABELS_SCHEMA]
        return pa.table(
            {f.name: pa.array(col, f.type) for f, col in zip(LABELS_SCHEMA, cols, strict=True)},
            schema=LABELS_SCHEMA,
        )


LABELS_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string()),
        pa.field("clip_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("split", pa.string()),
        pa.field("frame", pa.int64()),
        pa.field("vis", pa.string()),
        pa.field("x", pa.float64()),
        pa.field("y", pa.float64()),
        pa.field("src", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Keyframe interpolation
# ---------------------------------------------------------------------------


def interpolate(clip: BallClip, max_gap: int | None = None, overwrite: bool = False) -> int:
    """Fill unlabeled (or previously interpolated) frames between visible keyframes.

    Each gap is filled from a quadratic in time through up to two keyframes on each side
    (linear with fewer), never across an event label or a non-visible keyframe. Returns the
    number of frames filled.
    """
    max_gap = max_gap or 3 * clip.keyframe_step
    event_frames = sorted(e.frame for e in clip.events)
    keys = [
        f
        for f in clip.frames
        if (lab := clip.label(f)) is not None and lab.src != "interp" and lab.vis == "visible"
    ]
    hard = {
        f
        for f in clip.frames
        if (lab := clip.label(f)) is not None and lab.src != "interp" and lab.vis != "visible"
    }
    filled = 0
    for a, b in itertools.pairwise(keys):
        if b - a <= 1 or b - a > max_gap:
            continue
        if any(a < h < b for h in hard):
            continue
        # An event strictly inside the gap breaks the motion; an event *at* a keyframe is fine
        # (the keyframe is the contact point and belongs to both sides).
        if any(a < e < b for e in event_frames):
            continue
        seg_lo = max([e for e in event_frames if e <= a], default=clip.frame0 - 1)
        seg_hi = min([e for e in event_frames if e >= b], default=clip.frame1 + 1)
        left = [k for k in keys if seg_lo <= k <= a][-2:]
        right = [k for k in keys if b <= k <= seg_hi][:2]
        support = left + right
        t = np.array(support, dtype=np.float64)
        xs = np.array([clip.label(k).x for k in support], dtype=np.float64)
        ys = np.array([clip.label(k).y for k in support], dtype=np.float64)
        deg = 2 if len(support) >= 3 else 1
        cx = np.polyfit(t - a, xs, deg)
        cy = np.polyfit(t - a, ys, deg)
        for f in range(a + 1, b):
            cur = clip.label(f)
            if cur is not None and cur.src != "interp" and not overwrite:
                continue
            clip.set(
                f,
                BallLabel(
                    vis="visible",
                    x=round(float(np.polyval(cx, f - a)), 2),
                    y=round(float(np.polyval(cy, f - a)), 2),
                    src="interp",
                ),
            )
            filled += 1
    return filled
