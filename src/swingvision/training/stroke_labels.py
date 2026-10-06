"""Stroke ground truth and the swing evaluation behind M6's exit criteria (``sv swings eval``).

Ground truth lives in ``<output_root>/training/strokes/<session_id>.json``::

    {"session_id": "...", "spans": [[t0, t1], ...],
     "strokes": [{"t": 41.69, "stroke": "serve", "side": -1, "visible": true}, ...],
     "note": "how it was audited"}

Within the spans (the whole session when empty) ``strokes`` lists **every** stroke the player
hit; anything else the swing detector finds there (dribbles, tosses, picking up balls,
walking) must come out as ``other``. ``visible: false`` marks strokes whose player is out of
the picture at the contact: they are left out, as are the swings around them.

Scoring (per side, near / far):

* **Strokes** are matched one-to-one to predicted strokes (anything but ``other``) within
  ``match_s`` of their time, nearest first; a stroke is correct when its match has the same
  class. An unmatched stroke is wrong (missed, or called ``other``).
* **Other**: every other swing in the spans must not be called a stroke.
* Accuracy = correct / (strokes + other swings).
* **Contact timing**: for strokes with a seen ball hit, the contact estimated from the pose
  alone (``t_contact_pose``) against the hit event, in frames; for the others, against the
  impact sound (after the session's audio/video offset and the sound's travel time).
* **Phase consistency**: the spread of each phase's duration over repeated strokes of the
  same class and end (robust coefficient of variation: 1.4826 × MAD / median).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

STROKE_DIR = "strokes"
MATCH_S = 0.8
PHASES = ("prep_s", "forward_s", "follow_s")


def labels_path(output_root: Path, session_id: str) -> Path:
    return Path(output_root) / "training" / STROKE_DIR / f"{session_id}.json"


def load_labels(output_root: Path, session_id: str) -> dict | None:
    path = labels_path(output_root, session_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@dataclass
class SideReport:
    strokes: int = 0
    strokes_correct: int = 0
    others: int = 0
    others_correct: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    contact_errors: list[int] = field(default_factory=list)  # frames, vs the ball hit
    #: Frames, vs the impact sound (contacts the ball tracker didn't see).
    contact_errors_audio: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.strokes + self.others

    @property
    def accuracy(self) -> float | None:
        return (self.strokes_correct + self.others_correct) / self.total if self.total else None

    def add(self, truth: str, pred: str) -> None:
        row = self.confusion.setdefault(truth, {})
        row[pred] = row.get(pred, 0) + 1


def _in_spans(t: float, spans: list) -> bool:
    return not spans or any(a <= t <= b for a, b in spans)


def evaluate(
    swings: pa.Table,
    labels: dict,
    fps: float,
    hit_times: dict[int, float] | None = None,
    match_s: float = MATCH_S,
    exclude_flags: frozenset[str] = frozenset(),
) -> dict:
    """Score one session's swings against its labels → {"near": SideReport, "far": ...,
    "phases": {...}}. Swings with any of ``exclude_flags`` (e.g. ``dark``) are left out,
    and so are labeled strokes whose best candidate swing has one."""
    spans = labels.get("spans") or []
    rows = [r for r in swings.to_pylist() if _in_spans(r["t_contact"], spans)]
    gt = [s for s in labels["strokes"] if _in_spans(s["t"], spans)]
    hidden = [s["t"] for s in gt if not s.get("visible", True)]
    rows = [r for r in rows if all(abs(r["t_contact"] - t) > match_s for t in hidden)]
    gt = [s for s in gt if s.get("visible", True)]

    def flagged(r: dict) -> bool:
        return bool(exclude_flags & set(r["flags"] or []))

    def is_stroke(r: dict) -> bool:
        return r["stroke_type"] not in (None, "other")

    reports = {"near": SideReport(), "far": SideReport()}
    pairs = sorted(
        (abs(r["t_contact"] - s["t"]), i, j)
        for i, s in enumerate(gt)
        for j, r in enumerate(rows)
        if is_stroke(r) and abs(r["t_contact"] - s["t"]) <= match_s
    )
    match: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in pairs:
        if i not in match and j not in used:
            match[i] = j
            used.add(j)
    near_gt = set()
    for i, s in enumerate(gt):
        cand = [j for j, r in enumerate(rows) if abs(r["t_contact"] - s["t"]) <= match_s]
        near_gt.update(cand)
        best = match.get(i)
        if best is None and cand:
            best = min(cand, key=lambda j: abs(rows[j]["t_contact"] - s["t"]))
        if best is not None and flagged(rows[best]):
            continue
        rep = reports["far" if s["side"] == 1 else "near"]
        pred = rows[match[i]]["stroke_type"] if i in match else "other/missed"
        rep.strokes += 1
        rep.strokes_correct += pred == s["stroke"]
        rep.add(s["stroke"], pred)
    for j, r in enumerate(rows):
        if j in used or flagged(r):
            continue
        if j in near_gt and not is_stroke(r):
            continue  # the toss or follow-through of a labeled stroke
        rep = reports["far" if r["side"] == 1 else "near"]
        rep.others += 1
        rep.others_correct += not is_stroke(r)
        rep.add("other", r["stroke_type"] or "other")
    # Contact timing: the pose alone vs the seen ball hit (or the impact sound).
    for i, j in match.items():
        r = rows[j]
        if r["stroke_type"] != gt[i]["stroke"] or r["t_contact_pose"] is None or flagged(r):
            continue
        rep = reports["far" if r["side"] == 1 else "near"]
        if r["hit_event_id"] is not None:
            t_hit = (hit_times or {}).get(r["hit_event_id"], r["t_contact"])
            rep.contact_errors.append(round((r["t_contact_pose"] - t_hit) * fps))
        elif r["contact_source"] == "audio":
            rep.contact_errors_audio.append(round((r["t_contact_pose"] - r["t_contact"]) * fps))
    phases = phase_consistency(
        [rows[j] for i, j in match.items() if rows[j]["stroke_type"] == gt[i]["stroke"]]
    )
    return {**reports, "phases": phases}


def phase_consistency(rows: list[dict]) -> dict:
    """Per stroke class and end: median and robust CV of each phase duration, and the share
    of swings whose phases came out in order."""
    out: dict = {}
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["stroke_type"], "far" if r["side"] == 1 else "near"), []).append(r)
    for (stroke, side), rs in sorted(groups.items()):
        g: dict = {"n": len(rs)}
        for ph in PHASES:
            v = np.array([r[ph] for r in rs if r[ph] is not None], dtype=np.float64)
            if len(v) >= 3:
                med = float(np.median(v))
                mad = float(np.median(np.abs(v - med)))
                g[ph] = {"n": len(v), "median_s": med, "cv": 1.4826 * mad / med if med else None}
        ordered = [
            r
            for r in rs
            if r["t_start"] is not None
            and r["t_backswing_end"] is not None
            and r["t_follow_end"] is not None
        ]
        g["in_order"] = (
            sum(
                r["t_start"] < r["t_backswing_end"] < r["t_contact"] < r["t_follow_end"]
                for r in ordered
            )
            / len(ordered)
            if ordered
            else None
        )
        g["with_all_phases"] = len(ordered)
        out[f"{stroke}/{side}"] = g
    return out
