"""Stroke classification: rules, and a small learned model that replaces them once it is
trained on enough labels and beats them on held-out swings (PLAN.md §7.7).

Classes: serve, forehand, backhand, forehand_volley, backhand_volley, overhead, other.

**Rules** (in the hitter's frame, mirrored for left-handers, see ``kinematics``):

* *other*: no ball was hit (no hit event, no impact sound: a shadow swing, picking up a
  ball), the racket wrist never gets fast or barely travels in the last 0.3 s before the
  contact (a dribble on the racket, a tap, catching the ball), or the other hand is up at the
  head while the racket hand is low (a serve's toss and windup).
* *serve* / *overhead*: the racket hand is above the head at contact. A serve has a toss (the
  other hand above the head before the contact; checked first, so a serve whose contact time
  is a little off still counts) or is hit from behind the baseline with no ball coming in;
  otherwise it is an overhead. This is the pose-based serve detection the
  segmentation also uses: it doesn't depend on match state.
* *forehand* / *backhand*: votes from the contact point's side of the body, the shoulder turn
  at the end of the backswing (racket side back: forehand) and the wrist's sideways motion
  at contact (across the body: forehand).
* *volley*: the incoming ball wasn't bounced on the player's half and the player is inside
  the service line area.

**Model**: a GRU over the hitter-frame joints (relative to the pelvis) sampled every 50 ms
from 0.6 s before to 0.6 s after the contact, plus three ball features; ``sv train strokes``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from swingvision.pose.kinematics import FrameKinematics
from swingvision.pose.skeleton import H

STROKES = (
    "serve",
    "forehand",
    "backhand",
    "forehand_volley",
    "backhand_volley",
    "overhead",
    "other",
)
STROKE_LABELS = {
    "serve": "Serve",
    "forehand": "Forehand",
    "backhand": "Backhand",
    "forehand_volley": "Forehand volley",
    "backhand_volley": "Backhand volley",
    "overhead": "Overhead",
    "other": "Other",
}


@dataclass(frozen=True)
class StrokeParams:
    #: Not a stroke (a dribble, a tap, a toss, a catch): the racket wrist's peak speed below
    #: ``other_max_speed`` or its average over the last 0.3 s below ``other_max_avg`` (m/s).
    #: A groundstroke's wrist covers well over a meter in that time, a serve's about one.
    other_max_speed: float = 4.0
    other_max_avg: float = 2.5
    above_head_m: float = 0.05
    #: The other hand this high relative to the head (m) while the racket hand is low: a
    #: toss.
    toss_hand_m: float = -0.1
    #: Distance from the net (m) beyond which the player is behind the baseline (the far
    #: player's depth along the camera's axis is the least certain part of the pose).
    baseline_m: float = 10.8
    #: Distance from the net (m) inside which an unbounced ball is a volley.
    volley_m: float = 7.5
    side_scale_m: float = 0.3
    turn_scale_deg: float = 45.0
    lateral_scale: float = 3.0  # m/s

    def as_config(self) -> dict:
        return asdict(self)


@dataclass
class StrokeFeatures:
    wrist_speed_peak: float
    #: The racket wrist's straight-line distance in the last 0.3 s before contact, per s.
    wrist_speed_avg: float | None
    arm_above_head_m: float  # highest within 0.15 s of the contact
    toss: bool
    depth_m: float  # distance from the net (hitter's frame)
    contact_side_m: float
    turn_backswing_deg: float
    lateral_speed: float  # racket wrist sideways speed at contact (+: to the racket side)
    incoming: bool | None  # a ball came to the player (None: unknown)
    bounced: bool | None  # it bounced on the player's half first
    #: A ball was hit: the ball tracker's hit or an impact sound (not a shadow swing).
    ball_contact: bool = True
    #: The other hand's highest point near the contact, above the head (m).
    off_above_head_m: float = -1.0

    def ball_vector(self) -> list[float]:
        return [
            -1.0 if self.incoming is None else float(self.incoming),
            -1.0 if self.bounced is None else float(self.bounced),
            self.depth_m / 12.0,
        ]


def features(
    kin: FrameKinematics,
    t_contact: float,
    t_backswing_end: float | None,
    incoming: bool | None,
    bounced: bool | None,
    toss: bool = False,
    wrist_speed_avg: float | None = None,
    ball_contact: bool = True,
) -> StrokeFeatures:
    t = kin.t
    ic = int(np.argmin(np.abs(t - t_contact)))
    j = kin.joints
    lat = np.gradient(j[:, H["r_wrist"], 0], t) if len(t) > 2 else np.zeros(len(t))
    ib = int(np.argmin(np.abs(t - t_backswing_end))) if t_backswing_end is not None else ic
    win = (t >= t_contact - 0.3) & (t <= t_contact + 0.15)
    near = np.abs(t - t_contact) <= 0.15
    if not near.any():
        near = t == t[ic]
    around = np.abs(t - t_contact) <= 0.3
    if not around.any():
        around = near
    off_z = kin.off_wrist_rel[around, 2] + kin.pelvis[around, 2]
    return StrokeFeatures(
        wrist_speed_peak=float(kin.wrist_speed[win].max()) if win.any() else 0.0,
        arm_above_head_m=float(np.max(j[near, H["r_wrist"], 2] - j[near, H["head"], 2])),
        wrist_speed_avg=wrist_speed_avg,
        toss=toss,
        depth_m=float(-kin.pelvis[ic, 1]),
        contact_side_m=float(kin.wrist_rel[ic, 0]),
        turn_backswing_deg=float(kin.shoulder_turn[ib]),
        lateral_speed=float(lat[ic]),
        incoming=incoming,
        bounced=bounced,
        ball_contact=ball_contact,
        off_above_head_m=float(np.max(off_z - j[around, H["head"], 2])),
    )


def classify_rules(f: StrokeFeatures, p: StrokeParams | None = None) -> tuple[str, float]:
    """(stroke, confidence 0-1) from the rules in the module docstring."""
    p = p or StrokeParams()
    above = f.arm_above_head_m > p.above_head_m
    behind = f.depth_m >= p.baseline_m
    if above and f.toss:
        # A toss and the racket overhead: a serve, however the contact time came out.
        return "serve", min(0.95, 0.65 + 0.15 * (behind + (f.incoming is False)))
    if not f.ball_contact:
        return "other", 0.6  # no hit, no impact sound: a shadow swing, picking up a ball
    if f.wrist_speed_peak < p.other_max_speed or (
        f.wrist_speed_avg is not None and f.wrist_speed_avg < p.other_max_avg
    ):
        return "other", 0.7
    if not above and f.off_above_head_m > p.toss_hand_m:
        return "other", 0.6  # the other hand up at the head: a serve's toss and windup
    if above:
        if behind and f.incoming is not True:
            return "serve", 0.65 + 0.15 * (f.incoming is False)
        conf = 0.8 if f.incoming else 0.6
        return "overhead", conf
    score = (
        np.clip(f.contact_side_m / p.side_scale_m, -1, 1)
        + np.clip(f.turn_backswing_deg / p.turn_scale_deg, -1, 1)
        + 0.5 * np.clip(-f.lateral_speed / p.lateral_scale, -1, 1)
    )
    base = "forehand" if score >= 0 else "backhand"
    conf = float(0.5 + 0.5 * min(abs(score) / 1.5, 1.0))
    volley = f.incoming is True and f.bounced is False and f.depth_m < p.volley_m
    return (f"{base}_volley" if volley else base), conf


# ---------------------------------------------------------------------------
# Learned model
# ---------------------------------------------------------------------------

SEQ_BEFORE_S = 0.6
SEQ_AFTER_S = 0.6
SEQ_STEP_S = 0.05
N_BALL = 3


def sequence(kin: FrameKinematics, t_contact: float) -> np.ndarray | None:
    """(steps, 51) hitter-frame joints relative to the pelvis, resampled around the contact."""
    tq = np.arange(-SEQ_BEFORE_S, SEQ_AFTER_S + 1e-9, SEQ_STEP_S) + t_contact
    t = kin.t
    if len(t) < 5 or t[0] > tq[0] + 0.2 or t[-1] < tq[-1] - 0.2:
        return None
    rel = kin.joints - kin.pelvis[:, None, :]
    flat = rel.reshape(len(t), -1)
    out = np.column_stack([np.interp(tq, t, flat[:, k]) for k in range(flat.shape[1])])
    return out.astype(np.float32)


def _net(n_in: int, n_classes: int, hidden: int = 64):
    import torch
    from torch import nn

    class StrokeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(
                nn.Linear(2 * hidden + N_BALL, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
            )

        def forward(self, seq, ball):
            _, h = self.gru(seq)
            z = torch.cat([h[0], h[1], ball], dim=1)
            return self.head(z)

    return StrokeNet()


@dataclass
class StrokeModel:
    """A trained classifier (``<output_root>/models/strokes/<name>.pt`` + ``.json`` card)."""

    name: str
    classes: list[str]
    card: dict
    net: object

    @classmethod
    def load(cls, path: Path) -> StrokeModel:
        import torch

        card = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        net = _net(51, len(card["classes"]), card.get("hidden", 64))
        net.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
        net.eval()
        return cls(path.stem, list(card["classes"]), card, net)

    def predict(self, seqs: np.ndarray, ball: np.ndarray) -> tuple[list[str], np.ndarray]:
        import torch

        if len(seqs) == 0:
            return [], np.zeros(0)
        with torch.inference_mode():
            logits = self.net(torch.from_numpy(seqs).float(), torch.from_numpy(ball).float())
            prob = logits.softmax(1).numpy()
        k = prob.argmax(1)
        return [self.classes[i] for i in k], prob[np.arange(len(k)), k]


def models_dir(output_root: Path) -> Path:
    return Path(output_root) / "models" / "strokes"


def active_model(output_root: Path | None, spec: str = "auto") -> StrokeModel | None:
    """``auto``: the newest validated model; ``rules``: none; else a model by name."""
    if output_root is None or spec == "rules":
        return None
    d = models_dir(output_root)
    if spec != "auto":
        path = d / f"{spec}.pt"
        return StrokeModel.load(path) if path.exists() else None
    best = None
    for path in sorted(d.glob("*.pt"), key=lambda q: q.stat().st_mtime):
        card = path.with_suffix(".json")
        if card.exists() and json.loads(card.read_text(encoding="utf-8")).get("validated"):
            best = path
    return StrokeModel.load(best) if best else None
