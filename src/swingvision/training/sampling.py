"""Where to label next (PLAN.md §10).

Suggestions come from what a processed session already knows: moments where the ball track
has gaps while the player is hitting (strong audio onsets with no track nearby), detected
events, low-confidence stretches, and random moments for an unbiased sample. Times are
spread out (one suggestion per ``min_spacing_s``) and skip ones already labeled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swingvision.storage import tables
from swingvision.training.labels import BallClip

ONSET_STRENGTH = 15.0


@dataclass(frozen=True)
class Suggestion:
    t_s: float
    reason: str


def suggest(
    session,
    duration_s: float,
    existing: list[BallClip],
    n: int = 12,
    min_spacing_s: float = 20.0,
    seed: int = 0,
) -> list[Suggestion]:
    rng = np.random.default_rng(seed)
    cands: list[Suggestion] = []
    track_t = np.zeros(0)
    if session.ball_track_path.exists():
        track_t = (
            tables.read_table(session.ball_track_path, columns=["t_s"]).column("t_s").to_numpy()
        )
    if session.audio_onsets_path.exists():
        on = tables.read_table(session.audio_onsets_path)
        t = on.column("t_s").to_numpy()[on.column("strength").to_numpy() >= ONSET_STRENGTH]
        if len(track_t):
            k = np.clip(np.searchsorted(track_t, t), 1, len(track_t) - 1)
            near = np.minimum(np.abs(track_t[k] - t), np.abs(track_t[k - 1] - t))
            for ti in t[near > 0.3]:
                cands.append(Suggestion(float(ti) - 0.5, "loud hit, no ball track"))
        for ti in rng.choice(t, size=min(len(t), n), replace=False) if len(t) else []:
            cands.append(Suggestion(float(ti) - 0.8, "hitting (audio onset)"))
    if session.events_path.exists():
        ev = tables.read_table(session.events_path)
        low = ev.filter(np.asarray(ev.column("conf").to_numpy()) < 0.6)
        for ti, kind in zip(
            low.column("t_s").to_numpy(), low.column("kind").to_pylist(), strict=True
        ):
            cands.append(Suggestion(float(ti) - 1.0, f"uncertain {kind}"))
    for ti in rng.uniform(0, max(duration_s - 3, 1), size=n):
        cands.append(Suggestion(float(ti), "random"))
    rng.shuffle(cands)
    order = {"loud hit, no ball track": 0, "uncertain hit": 1, "uncertain bounce": 1}
    cands.sort(key=lambda s: order.get(s.reason, 2))
    taken: list[float] = [c.t0_s for c in existing]
    out: list[Suggestion] = []
    for s in cands:
        if s.t_s < 0 or s.t_s > duration_s - 1:
            continue
        if any(abs(s.t_s - t) < min_spacing_s for t in taken):
            continue
        out.append(s)
        taken.append(s.t_s)
        if len(out) >= n:
            break
    return sorted(out, key=lambda s: s.t_s)
