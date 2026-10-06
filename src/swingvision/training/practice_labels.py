"""Practice-shot ground truth (M5): which practice shots a session really contains.

One file per session, ``<output_root>/training/segments/<session_id>.json``: the audited
time spans and, inside them, every practice shot's contact time (within about a second is
enough), the hitter's end and whether the ball went into the net. ``sv eval segments``
scores the session's segments against it with :func:`score_segments`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from swingvision.storage.fsutil import atomic_write_text

SEGMENT_LABELS_DIR = ("training", "segments")
#: A labeled shot and a segment's contact match within this.
MATCH_TOL_S = 1.2


class ShotLabel(BaseModel):
    t: float  # contact time, s (approximate)
    side: Literal[-1, 1]  # hitter's half: -1 near (camera end), +1 far
    end: Literal["land", "net"] = "land"
    note: str = ""


class SegmentLabels(BaseModel):
    session_id: str
    #: Time spans that were audited completely; shots outside aren't scored.
    spans: list[tuple[float, float]] = Field(default_factory=list)
    shots: list[ShotLabel] = Field(default_factory=list)
    method: str = ""


def labels_path(output_root: Path, session_id: str) -> Path:
    return Path(output_root).joinpath(*SEGMENT_LABELS_DIR, f"{session_id}.json")


def load_labels(output_root: Path, session_id: str) -> SegmentLabels | None:
    path = labels_path(output_root, session_id)
    if not path.exists():
        return None
    return SegmentLabels.model_validate_json(path.read_text(encoding="utf-8"))


def save_labels(output_root: Path, labels: SegmentLabels) -> Path:
    path = labels_path(output_root, labels.session_id)
    atomic_write_text(path, labels.model_dump_json(indent=1))
    return path


@dataclass
class SegmentScore:
    n_labeled: int = 0
    n_segments: int = 0
    #: Labeled shots with a segment of the same hitter's end within the tolerance ...
    matched: int = 0
    #: ... whose clip covers the contact and the start of the flight.
    covered: int = 0
    end_agree: int = 0  # matched shots whose net / over-the-net call agrees
    missed: list[float] = field(default_factory=list)
    uncovered: list[float] = field(default_factory=list)
    false: list[float] = field(default_factory=list)  # segments matching no labeled shot

    @property
    def accuracy(self) -> float:
        """Shots segmented correctly over all shots and spurious segments."""
        denom = self.n_labeled + len(self.false)
        return self.covered / denom if denom else 1.0

    @property
    def recall(self) -> float:
        return self.matched / self.n_labeled if self.n_labeled else 1.0

    @property
    def precision(self) -> float:
        return (self.n_segments - len(self.false)) / self.n_segments if self.n_segments else 1.0


def _inside(t: float, spans: list[tuple[float, float]]) -> bool:
    return not spans or any(a <= t <= b for a, b in spans)


def score_segments(
    shots: list[dict], labels: SegmentLabels, tol_s: float = MATCH_TOL_S
) -> SegmentScore:
    """Score practice-shot segment rows (``segments.parquet`` rows of kind practice_shot)."""
    preds = [r for r in shots if _inside(r["t_contact"], labels.spans)]
    gts = sorted((g for g in labels.shots if _inside(g.t, labels.spans)), key=lambda g: g.t)
    sc = SegmentScore(n_labeled=len(gts), n_segments=len(preds))
    used: set[int] = set()
    for g in gts:
        best, best_d = None, tol_s
        for i, r in enumerate(preds):
            if i in used or (r["side"] is not None and r["side"] != g.side):
                continue
            d = abs(r["t_contact"] - g.t)
            if d <= best_d:
                best, best_d = i, d
        if best is None:
            sc.missed.append(g.t)
            continue
        used.add(best)
        r = preds[best]
        sc.matched += 1
        if r["start_t"] <= min(g.t, r["t_contact"]) - 0.3 and r["end_t"] >= g.t + 0.5:
            sc.covered += 1
        else:
            sc.uncovered.append(g.t)
        if (r["end_reason"] == "net") == (g.end == "net"):
            sc.end_agree += 1
    sc.false = [r["t_contact"] for i, r in enumerate(preds) if i not in used]
    return sc
