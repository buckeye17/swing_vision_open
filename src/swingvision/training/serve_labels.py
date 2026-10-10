"""Serve contact ground truth and the evaluation behind M7b's exit criteria
(``sv serves eval``).

Ground truth lives in ``<output_root>/training/serve_contact/<session_id>.json``::

    {"session_id": "...", "note": "how it was labeled",
     "serves": [{"t": 41.68, "frame": 2503, "toe": [1846, 1212], "toe_on_ground": true,
                 "ball": [1807.6, 370.6], "toe_visibility": "visible"}, ...]}

``t`` is the serve's contact time when it was labeled (it finds the serve again: the nearest
serve within ``MATCH_S``). ``frame``: the contact frame, the last frame before the ball moves
toward the net (null: not labeled). ``toe``: the front toe tip's pixel in that frame (for
serves whose contact is above the picture, in the swing's own contact frame, ``toe_frame``).
``ball``: the ball's center in the contact frame. The Swings page's toe and contact-frame
corrections are ground truth too: ``edits.json`` serves are merged in (an edit wins).

Scoring:

* **Contact frame**: predicted minus labeled, exact and within ±1 (near serves).
* **Toe**: the predicted toe vs the labeled pixel cast onto the ground, in cm; the on-ground
  test vs the label.
* **Contact point**: the contact point projected into the contact frame vs the labeled ball
  pixel (px at 4K), and the forward σ.
* **Coverage**: near serves with offsets, and flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

SERVE_DIR = "serve_contact"
MATCH_S = 0.3


def labels_path(output_root: Path, session_id: str) -> Path:
    return Path(output_root) / "training" / SERVE_DIR / f"{session_id}.json"


def load_labels(output_root: Path, session_id: str) -> dict | None:
    path = labels_path(output_root, session_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_labels(output_root: Path, session_id: str, labels: dict) -> Path:
    from swingvision.storage.fsutil import atomic_write_json

    path = labels_path(output_root, session_id)
    atomic_write_json(path, labels)
    return path


def with_edits(labels: dict | None, edits, session_id: str) -> dict:
    """``labels`` with the user's serve corrections (``edits.json``) merged in."""
    out = {"session_id": session_id, "serves": [dict(s) for s in (labels or {}).get("serves", [])]}
    for e in getattr(edits, "serves", []) or []:
        best = min(out["serves"], key=lambda s: abs(s["t"] - e.t), default=None)
        if best is None or abs(best["t"] - e.t) > MATCH_S:
            best = {"t": e.t}
            out["serves"].append(best)
        if e.frame is not None:
            best["frame"] = e.frame
        if e.toe is not None:
            best["toe"] = list(e.toe)
            best.pop("toe_frame", None)
    return out


@dataclass
class ServeReport:
    frame_errors: list[int] = field(default_factory=list)
    toe_errors_m: list[float] = field(default_factory=list)
    on_ground_agree: list[bool] = field(default_factory=list)
    ball_errors_px: list[float] = field(default_factory=list)
    forward_sigma_m: list[float] = field(default_factory=list)
    near: int = 0
    near_with_offsets: int = 0
    near_contact_in_frame: int = 0
    near_contact_in_frame_with_offsets: int = 0
    above_frame: int = 0
    flags: dict[str, int] = field(default_factory=dict)

    def merge(self, other: ServeReport) -> None:
        for k in ("frame_errors", "toe_errors_m", "on_ground_agree", "ball_errors_px",
                  "forward_sigma_m"):  # fmt: skip
            getattr(self, k).extend(getattr(other, k))
        for k in ("near", "near_with_offsets", "near_contact_in_frame",
                  "near_contact_in_frame_with_offsets", "above_frame"):  # fmt: skip
            setattr(self, k, getattr(self, k) + getattr(other, k))
        for k, v in other.flags.items():
            self.flags[k] = self.flags.get(k, 0) + v

    def summary(self) -> dict:
        fe = np.array(self.frame_errors)
        te = np.array(self.toe_errors_m)
        be = np.array(self.ball_errors_px)
        fs = np.array(self.forward_sigma_m)
        return {
            "contact_frames": len(fe),
            "frame_exact": float(np.mean(fe == 0)) if len(fe) else None,
            "frame_within_1": float(np.mean(np.abs(fe) <= 1)) if len(fe) else None,
            "toes": len(te),
            "toe_median_m": float(np.median(te)) if len(te) else None,
            "toe_p90_m": float(np.percentile(te, 90)) if len(te) else None,
            "toe_within_6cm": float(np.mean(te <= 0.06)) if len(te) else None,
            "on_ground_agree": float(np.mean(self.on_ground_agree))
            if self.on_ground_agree
            else None,
            "balls": len(be),
            "ball_median_px": float(np.median(be)) if len(be) else None,
            "forward_sigma_median_m": float(np.median(fs)) if len(fs) else None,
            "near": self.near,
            "near_with_offsets": self.near_with_offsets,
            "coverage_contact_in_frame": (
                self.near_contact_in_frame_with_offsets / self.near_contact_in_frame
                if self.near_contact_in_frame
                else None
            ),
            "above_frame": self.above_frame,
            "flags": dict(sorted(self.flags.items())),
        }


def evaluate(contacts: pa.Table, labels: dict, camera_at, fps: float) -> ServeReport:
    """Score one session's ``serve_contact`` rows against its labels."""
    rows = contacts.to_pylist()
    rep = ServeReport()
    for r in rows:
        for f in r["flags"] or []:
            rep.flags[f] = rep.flags.get(f, 0) + 1
        if r["side"] != -1:
            continue
        rep.near += 1
        has = r["forward_m"] is not None
        rep.near_with_offsets += has
        if "contact_above_frame" in (r["flags"] or []):
            rep.above_frame += 1
        else:
            rep.near_contact_in_frame += 1
            rep.near_contact_in_frame_with_offsets += has
        if has and r["forward_sigma_m"] is not None:
            rep.forward_sigma_m.append(float(r["forward_sigma_m"]))
    for lab in labels.get("serves", []):
        r = min(rows, key=lambda x: abs(x["t_contact"] - lab["t"]), default=None)
        if r is None or abs(r["t_contact"] - lab["t"]) > MATCH_S or r["side"] != -1:
            continue
        if lab.get("frame") is not None and r["frame_contact"] is not None:
            rep.frame_errors.append(int(r["frame_contact"]) - int(lab["frame"]))
        cam = camera_at(r["t_contact"])
        if lab.get("toe") is not None and r["toe_x"] is not None:
            g = cam.image_to_ground(np.asarray(lab["toe"], float))[:2]
            rep.toe_errors_m.append(float(np.hypot(r["toe_x"] - g[0], r["toe_y"] - g[1])))
            if lab.get("toe_on_ground") is not None and r["toe_on_ground"] is not None:
                rep.on_ground_agree.append(bool(r["toe_on_ground"]) == bool(lab["toe_on_ground"]))
        if (
            lab.get("ball") is not None
            and r["contact_x"] is not None
            and lab.get("frame") == r["frame_contact"]
        ):
            px = cam.project(np.array([[r["contact_x"], r["contact_y"], r["contact_z"]]]))[0]
            rep.ball_errors_px.append(float(np.hypot(*(px - np.asarray(lab["ball"], float)))))
    return rep
