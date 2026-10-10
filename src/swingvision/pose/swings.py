"""Swings from 3D pose: detection, contact, phases, kinematic metrics (PLAN.md §7.7).

**Detection.** Every hit of the player's (``events``) with pose around it is a swing, and so
is every other burst of racket-wrist speed (≥ ``min_peak_speed``) and every strong impact
sound with the racket wrist moving: that finds the contacts the ball tracker didn't see. A burst's contact time comes from the impact sound when there is
one (after the session's audio/video offset and the sound's travel time), else from the pose
(the racket wrist's speed peak, shifted by the offset measured on seen hits).

**Racket hand.** The faster wrist at the session's clear contacts. The profile's handedness
is only a prior: it decides when the pose evidence is thin, and a mismatch is flagged.

**Phases** (times in the swing record):

* *preparation* starts with the unit turn (the shoulder turn starts to build) — for a serve,
  when the tossing arm starts up from its lowest point;
* *split step*: a quick dip-and-rise of the pelvis just before the preparation (not serves);
* *backswing end* / forward swing start: the racket wrist furthest back (for a serve, the
  racket drop: the racket elbow at its tightest bend, racket behind the back);
* *contact*; *follow-through* ends when the wrist speed stays below ``follow_frac`` of its
  peak (for a serve: when the racket wrist has come most of the way down); *recovery* ends
  when the hand and trunk are still again.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pyarrow as pa

from swingvision.pose.kinematics import FrameKinematics, frame_kinematics
from swingvision.pose.skeleton import H
from swingvision.storage.schemas import SWINGS

#: Savitzky-Golay window (frames) for joint speeds.
SMOOTH = 7
#: Span (s) of the racket wrist's average speed into the contact.
CHORD_S = 0.3
#: Flags :func:`analyze` sets (the others come later in :func:`build_swings`).
ANALYZE_FLAGS = frozenset({"little_pose", "far", "implausible_speed", "low_conf"})
#: An overhead contact is judged on the racket wrist's height this close (s) to the contact.
OVERHEAD_WINDOW_S = 0.15


@dataclass(frozen=True)
class SwingParams:
    #: A racket-wrist speed peak this fast (m/s) is a swing even without a ball event.
    min_peak_speed: float = 7.0
    #: Swings closer than this (s) are one swing.
    min_gap_s: float = 0.8
    #: A speed peak this close (s) to one of the player's hits is that hit's swing.
    hit_merge_s: float = 0.35
    #: Two strokes can't be closer than this (s): of a closer pair, the weaker one (a toss,
    #: the follow-through, walking off) becomes ``other``.
    stroke_gap_s: float = 1.0
    #: The audio/video offset needs this many hit sounds; with fewer seen hits it is measured
    #: from overhead swings instead (the racket wrist's highest point is the contact).
    av_min_samples: int = 8
    #: Swings in frames darker than this mean luma (0-255) are flagged ``dark``.
    dark_luma: float = 40.0
    #: Racket-wrist speeds above this (m/s) are pose errors (flagged).
    max_plausible_speed: float = 45.0
    #: Racket-wrist speed peak searched this close (s) to a hit's contact.
    peak_window_s: float = 0.25
    #: Impact sound from this long (s) before to twice this long after a pose-only swing's
    #: contact estimate gives its contact.
    audio_window_s: float = 0.15
    audio_min_z: float = 8.0
    #: An impact sound this strong (z) with the racket wrist at least this fast (m/s) around
    #: it is a swing too.
    audio_anchor_z: float = 15.0
    audio_anchor_speed: float = 4.0
    #: Phase search before / after the contact (s).
    before_s: float = 1.6
    after_s: float = 1.1
    #: Overhead swing: the racket wrist this much above the head (m) near the contact; a toss:
    #: the other hand at least this high above the head (m) before it. The lifted arm comes
    #: out short of a fully stretched one, so the margins are small.
    overhead_above_head_m: float = 0.05
    toss_above_head_m: float = 0.0
    follow_frac: float = 0.35
    #: Serves: the toss starts when the tossing hand has risen this much (m) from its low
    #: point; the follow-through ends when the racket wrist has come down this share of the
    #: way from the contact to its lowest point after it.
    toss_rise_m: float = 0.1
    follow_drop_frac: float = 0.85
    recovery_speed: float = 1.5  # m/s
    recovery_trunk_av: float = 90.0  # deg/s
    split_min_dip_m: float = 0.03
    #: Racket hand: a contact votes for the faster wrist when it is this much faster.
    hand_ratio: float = 1.25
    #: ... or at an overhead contact, when it is this much higher than the other hand (m).
    hand_overhead_gap_m: float = 0.25
    #: A hand peaking this high above the head (m) votes for itself when it is this fast
    #: there (m/s, the racket's upswing) and for the other hand when it is this still (a toss).
    hand_peak_above_m: float = 0.1
    hand_fast_peak: float = 1.5
    hand_still_peak: float = 0.8
    hand_min_votes: int = 5
    hand_min_share: float = 0.7
    #: Minimum pose frames (fraction of the expected) around the contact.
    min_coverage: float = 0.5

    def as_config(self) -> dict:
        return asdict(self)


@dataclass
class PoseSeries:
    """One player's 3D pose sorted by time (``pose3d``)."""

    t: np.ndarray
    frame: np.ndarray
    clip: np.ndarray
    joints: np.ndarray  # (n, 17, 3)
    conf: np.ndarray

    @classmethod
    def from_table(cls, table: pa.Table) -> PoseSeries:
        table = table.sort_by("t_s")
        col = table.column("joints").combine_chunks()
        n = len(col)
        joints = (
            col.flatten().to_numpy(zero_copy_only=False).reshape(n, 17, 3).astype(np.float64)
            if n
            else np.zeros((0, 17, 3))
        )
        return cls(
            t=table.column("t_s").to_numpy(),
            frame=table.column("frame").to_numpy(),
            clip=table.column("clip").to_numpy(),
            joints=joints,
            conf=table.column("conf").to_numpy(zero_copy_only=False).astype(np.float64),
        )

    def around(self, t: float, before: float, after: float) -> np.ndarray:
        """Indices within [t - before, t + after] in the clip nearest to ``t`` (valid joints)."""
        if not len(self.t):
            return np.zeros(0, dtype=np.int64)
        k = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if k > 0 and abs(self.t[k - 1] - t) < abs(self.t[k] - t):
            k -= 1
        if abs(self.t[k] - t) > 0.1:
            return np.zeros(0, dtype=np.int64)
        lo, hi = np.searchsorted(self.t, [t - before, t + after])
        idx = np.arange(lo, hi)
        idx = idx[(self.clip[idx] == self.clip[k]) & np.isfinite(self.joints[idx, 0, 0])]
        return idx


@dataclass
class SwingInputs:
    pose: PoseSeries
    fps: float
    #: The player's hits: (event_id, t_s).
    hits: list[tuple[int, float]] = field(default_factory=list)
    onsets_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    onsets_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    av_offset_s: float = 0.0
    camera_xyz: np.ndarray | None = None
    profile_hand: str | None = None
    #: Hit-sound gaps behind ``av_offset_s`` (fewer than ``av_min_samples``: not trusted).
    av_offset_n: int = 0
    #: Frame brightness (``pass1/frames.parquet``): time and mean luma.
    luma_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    luma: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class Swing:
    t_contact: float
    contact_source: str
    hit_event_id: int | None = None
    t_peak: float | None = None
    t_contact_pose: float | None = None
    rec: dict = field(default_factory=dict)
    kin: FrameKinematics | None = None
    idx: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def racket_hand(
    pose: PoseSeries, times: list[float], profile_hand: str | None, p: SwingParams
) -> tuple[str, dict]:
    """The racket hand from the pose, with the profile as the prior.

    Votes: every time a hand peaks above the head (:func:`overhead_votes`), and every hit
    where one wrist is clearly faster. The overhead votes decide when there are enough of
    them: a serve's toss and contact (often above the frame) are easier to see than a hit.
    """
    over = overhead_votes(pose, p)
    if sum(over.values()) >= p.hand_min_votes:
        return _decide(over, "overhead", profile_hand, p)
    votes = dict(over)
    for t in times:
        idx = pose.around(t, p.peak_window_s, p.peak_window_s)
        if len(idx) < 5:
            continue
        # Overhead contact: the hand clearly above the head and the other hand.
        k = idx[int(np.argmin(np.abs(pose.t[idx] - t)))]
        lz, rz = pose.joints[k, H["l_wrist"], 2], pose.joints[k, H["r_wrist"], 2]
        head = pose.joints[k, H["head"], 2]
        if max(lz, rz) > head + 0.1 and abs(lz - rz) > p.hand_overhead_gap_m:
            votes["left" if lz > rz else "right"] += 1
            continue
        lm = float(joint_speed(pose, idx, H["l_wrist"]).max())
        rm = float(joint_speed(pose, idx, H["r_wrist"]).max())
        if max(lm, rm) < p.min_peak_speed:
            continue  # a dribble or a tap: either hand
        if lm > p.hand_ratio * rm:
            votes["left"] += 1
        elif rm > p.hand_ratio * lm:
            votes["right"] += 1
    return _decide(votes, "hits", profile_hand, p)


def _decide(
    votes: dict[str, int], kind: str, profile_hand: str | None, p: SwingParams
) -> tuple[str, dict]:
    n = votes["left"] + votes["right"]
    best = max(votes, key=lambda k: votes[k])
    evidence = n >= p.hand_min_votes and votes[best] >= p.hand_min_share * n
    info = {"votes": votes, "evidence": kind, "profile": profile_hand}
    if evidence:
        hand = best
        info["source"] = "pose"
        info["mismatch"] = bool(profile_hand and profile_hand != best)
    else:
        hand = profile_hand or "right"
        info["source"] = "profile" if profile_hand else "default"
        info["mismatch"] = False
    return hand, info


def overhead_votes(pose: PoseSeries, p: SwingParams) -> dict[str, int]:
    """Racket-hand votes from hands peaking above the head with the other hand down.

    The racket hand passes its highest point fast (the serve's or smash's upswing), the
    tossing hand hangs at the top almost still: a fast peak votes for its hand, a still one
    for the other hand.
    """
    votes = {"left": 0, "right": 0}
    if not len(pose.t):
        return votes
    starts = np.flatnonzero(np.r_[True, np.diff(pose.clip) != 0])
    ends = np.r_[starts[1:], len(pose.t)]
    for a, b in zip(starts, ends, strict=True):
        idx = np.arange(a, b)
        idx = idx[np.isfinite(pose.joints[idx, 0, 0])]
        if len(idx) < 2 * SMOOTH:
            continue
        gap = max(3, round(p.min_gap_s / float(np.median(np.diff(pose.t[idx])))))
        head = pose.joints[idx, H["head"], 2]
        for hand, other in (("left", "right"), ("right", "left")):
            z = pose.joints[idx, H[f"{hand[0]}_wrist"], 2] - head
            zo = pose.joints[idx, H[f"{other[0]}_wrist"], 2] - head
            v = joint_speed(pose, idx, H[f"{hand[0]}_wrist"])
            k = 0
            while k < len(z):
                lo, hi = max(0, k - gap), min(len(z), k + gap + 1)
                if (
                    z[k] >= p.hand_peak_above_m
                    and z[k] == z[lo:hi].max()
                    and zo[k] < z[k] - p.hand_overhead_gap_m
                ):
                    if v[k] >= p.hand_fast_peak:
                        votes[hand] += 1
                    elif v[k] <= p.hand_still_peak:
                        votes[other] += 1
                    k = hi
                else:
                    k += 1
    return votes


def joint_speed(pose: PoseSeries, idx: np.ndarray, joint: int) -> np.ndarray:
    """Speed (m/s) of a joint over consecutive indices, from smoothed positions."""
    t = pose.t[idx]
    if len(t) < 3:
        return np.zeros(len(t))
    x = pose.joints[idx, joint]
    if len(t) >= SMOOTH:
        from scipy.signal import savgol_filter

        x = savgol_filter(x, SMOOTH, 2, axis=0)
    return np.linalg.norm(np.gradient(x, t, axis=0), axis=1)


def raw_contact(
    pose: PoseSeries, t: float, half: float, wrist: int
) -> tuple[float, float, bool] | None:
    """Pose-only contact near ``t``: (time, peak racket-wrist speed, overhead).

    Overhead swings (serves, smashes) meet the ball at the racket wrist's highest point, found
    from 0.15 s before to 0.3 s after the speed peak (the racket drop is fast too); other
    swings at the speed peak.
    """
    idx = pose.around(t, half, half)
    if len(idx) < 5:
        return None
    tt = pose.t[idx]
    v = joint_speed(pose, idx, wrist)
    k = int(np.argmax(v))
    wz = pose.joints[idx, wrist, 2]
    head = pose.joints[idx, H["head"], 2]
    win = (tt >= tt[k] - 0.15) & (tt <= tt[k] + 0.3)
    j = np.flatnonzero(win)[int(np.argmax(wz[win]))]
    if wz[j] > head[j] + 0.05:
        return float(tt[j]), float(v[k]), True
    return float(tt[k]), float(v[k]), False


def _speed_peaks(
    pose: PoseSeries, wrist: int, min_speed: float, min_gap: float
) -> list[tuple[float, float]]:
    """(time, speed) of racket-wrist speed peaks above ``min_speed``, per clip."""
    out: list[tuple[float, float]] = []
    if not len(pose.t):
        return out
    starts = np.flatnonzero(np.r_[True, np.diff(pose.clip) != 0])
    ends = np.r_[starts[1:], len(pose.t)]
    for a, b in zip(starts, ends, strict=True):
        idx = np.arange(a, b)
        idx = idx[np.isfinite(pose.joints[idx, wrist, 0])]
        if len(idx) < SMOOTH:
            continue
        t = pose.t[idx]
        v = joint_speed(pose, idx, wrist)
        order = np.argsort(-v)
        taken: list[float] = []
        for k in order:
            if v[k] < min_speed:
                break
            if 0 < k < len(v) - 1 and all(abs(t[k] - x) >= min_gap for x in taken):
                taken.append(float(t[k]))
                out.append((float(t[k]), float(v[k])))
    return sorted(out)


def detect(inp: SwingInputs, hand: str, p: SwingParams, lags: dict[bool, float]) -> list[Swing]:
    """Swings at the player's hits plus pose-only speed bursts (``lags``: pose-only contact
    correction for overhead / other swings, see :func:`pose_contact_lags`)."""
    wrist = H[f"{'l' if hand == 'left' else 'r'}_wrist"]
    pose = inp.pose
    swings: list[Swing] = []
    for eid, t in inp.hits:
        if len(pose.around(t, 0.3, 0.2)) < 5:
            continue
        s = Swing(t_contact=t, contact_source="hit", hit_event_id=eid)
        rc = raw_contact(pose, t, p.peak_window_s, wrist)
        if rc is not None:
            s.t_peak = rc[0]
            s.t_contact_pose = rc[0] + lags[rc[2]]
        swings.append(s)
    hit_times = [s.t_contact for s in swings]
    taken: list[float] = []
    on_t, on_s = inp.onsets_t, inp.onsets_s
    for t_pk, _ in _speed_peaks(pose, wrist, p.min_peak_speed, p.min_gap_s):
        if any(abs(t_pk - x) < p.hit_merge_s for x in hit_times):
            continue
        rc = raw_contact(pose, t_pk, 0.1, wrist)
        t_pose = t_pk + lags[False] if rc is None else rc[0] + lags[rc[2]]
        if any(abs(t_pose - x) < p.min_gap_s for x in taken):
            continue
        s = Swing(t_contact=t_pose, contact_source="pose", t_peak=t_pk, t_contact_pose=t_pose)
        if len(on_t):
            delay = _sound_delay(inp, pose, t_pk)
            centre = t_pose + inp.av_offset_s + delay
            lo, hi = np.searchsorted(
                on_t, [centre - p.audio_window_s, centre + 2 * p.audio_window_s]
            )
            if hi > lo:
                k = lo + int(np.argmax(on_s[lo:hi]))
                if on_s[k] >= p.audio_min_z:
                    s.t_contact = float(on_t[k] - inp.av_offset_s - delay)
                    s.contact_source = "audio"
        swings.append(s)
        taken.append(s.t_contact)
    # Strong impact sounds where the racket wrist moves: contacts the ball tracker and the
    # speed peaks missed (a serve hit above the top of the frame, a far player's swing).
    for t_on, z in zip(on_t, on_s, strict=True):
        if z < p.audio_anchor_z:
            continue
        t_c = float(t_on - inp.av_offset_s)
        t_c -= _sound_delay(inp, pose, t_c)
        if any(abs(t_c - x) < p.hit_merge_s for x in hit_times) or any(
            abs(t_c - x) < p.min_gap_s for x in taken
        ):
            continue
        idx = pose.around(t_c, 0.3, 0.2)
        if len(idx) < SMOOTH or joint_speed(pose, idx, wrist).max() < p.audio_anchor_speed:
            continue
        rc = raw_contact(pose, t_c, p.peak_window_s, wrist)
        s = Swing(t_contact=t_c, contact_source="audio")
        if rc is not None:
            s.t_peak = rc[0]
            s.t_contact_pose = rc[0] + lags[rc[2]]
        swings.append(s)
        taken.append(t_c)
    swings.sort(key=lambda s: s.t_contact)
    return swings


def _sound_delay(inp: SwingInputs, pose: PoseSeries, t: float) -> float:
    if inp.camera_xyz is None:
        return 0.0
    idx = pose.around(t, 0.05, 0.05)
    if not len(idx):
        return 0.0
    pelvis = pose.joints[idx[0], H["pelvis"]]
    return float(np.linalg.norm(pelvis + np.array([0.0, 0.0, 0.3]) - inp.camera_xyz)) / 343.0


def pose_av_offset(inp: SwingInputs, hand: str, p: SwingParams) -> tuple[float, int]:
    """Audio minus video lag (s) from overhead swings: the racket wrist's highest point (the
    contact, to a frame on seen hits) against the loudest impact sound just after it."""
    from swingvision.analysis.segmentation import av_gaps, offset_from_gaps

    if not len(inp.onsets_t):
        return 0.0, 0
    wrist = H[f"{'l' if hand == 'left' else 'r'}_wrist"]
    times = []
    for t_pk, _ in _speed_peaks(inp.pose, wrist, p.min_peak_speed, p.min_gap_s):
        rc = raw_contact(inp.pose, t_pk, 0.1, wrist)
        if rc is not None and rc[2]:
            times.append(rc[0])
    if not times:
        return 0.0, 0
    delay = np.array([_sound_delay(inp, inp.pose, t) for t in times])
    gaps = av_gaps(np.array(times), delay, inp.onsets_t, inp.onsets_s, -0.1, 0.35)
    if len(gaps) < p.av_min_samples:
        return 0.0, len(gaps)
    return offset_from_gaps(gaps, 0.35), len(gaps)


def pose_contact_lags(
    inp: SwingInputs, hand: str, p: SwingParams
) -> tuple[dict[bool, float], dict]:
    """Median (hit time - pose-only contact) over the seen hits with a real swing, for
    overhead (``True``) and other swings: the correction applied to pose-only contacts."""
    wrist = H[f"{'l' if hand == 'left' else 'r'}_wrist"]
    lags: dict[bool, list[float]] = {True: [], False: []}
    for _, t in inp.hits:
        rc = raw_contact(inp.pose, t, p.peak_window_s, wrist)
        if rc is not None and rc[1] >= p.min_peak_speed:
            lags[rc[2]].append(t - rc[0])
    out = {k: float(np.median(v)) if len(v) >= 5 else 0.0 for k, v in lags.items()}
    info = {
        "overhead_s": round(out[True], 4),
        "overhead_n": len(lags[True]),
        "other_s": round(out[False], 4),
        "other_n": len(lags[False]),
    }
    return out, info


# ---------------------------------------------------------------------------
# Phases and metrics
# ---------------------------------------------------------------------------


def _first(mask: np.ndarray) -> int | None:
    nz = np.flatnonzero(mask)
    return int(nz[0]) if len(nz) else None


def _last(mask: np.ndarray) -> int | None:
    nz = np.flatnonzero(mask)
    return int(nz[-1]) if len(nz) else None


def _f(v) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None


def analyze(
    s: Swing, pose: PoseSeries, side: int | None, hand: str, fps: float, p: SwingParams
) -> dict:
    """Phases and metrics of one swing (a SWINGS row without ids and stroke fields)."""
    idx = pose.around(s.t_contact, p.before_s, p.after_s)
    rec: dict = {"n_frames": len(idx), "flags": []}
    expected = (p.before_s + p.after_s) * fps
    near = pose.around(s.t_contact, 0.3, 0.2)
    if len(idx) < 10 or len(near) < p.min_coverage * 0.5 * fps:
        rec["flags"].append("little_pose")
        rec["pose_quality"] = float(len(idx) / max(expected, 1))
        return rec
    t = pose.t[idx]
    k = frame_kinematics(t, pose.joints[idx], side, hand)
    s.kin, s.idx = k, idx
    tc = s.t_contact
    ic = int(np.argmin(np.abs(t - tc)))
    head_z = k.joints[:, H["head"], 2]
    wrist_z = k.joints[:, H["r_wrist"], 2]
    pelvis_z = k.pelvis[:, 2]
    before = (t < tc - 0.05) & (t >= tc - 1.3)
    # Serve-like: racket hand above the head at contact.
    near_c = np.abs(t - tc) <= OVERHEAD_WINDOW_S
    overhead = bool(np.max(wrist_z[near_c] - head_z[near_c]) > p.overhead_above_head_m)
    toss_peak = None
    if overhead:
        win = (t >= tc - 1.6) & (t <= tc - 0.1)
        off_z = k.off_wrist_rel[:, 2] + pelvis_z
        if win.any():
            j = np.flatnonzero(win)[int(np.argmax(off_z[win] - head_z[win]))]
            if off_z[j] > head_z[j] + p.toss_above_head_m:
                toss_peak = j
                rec["toss_height_m"] = float(off_z[j])
    # Backswing end.
    i_bs = None
    if overhead:
        # The racket drop: the racket elbow at its tightest, racket behind the back.
        win = (t >= tc - 0.6) & (t < tc - 0.03)
        if win.any():
            i_bs = np.flatnonzero(win)[int(np.argmin(k.elbow[win]))]
    elif before.any():
        i_bs = np.flatnonzero(before)[int(np.argmin(k.wrist_rel[before, 1]))]
    # Preparation start.
    i_start = None
    if toss_peak is not None:
        off_z = k.off_wrist_rel[:, 2] + pelvis_z
        win = (t >= t[toss_peak] - 1.2) & (t < t[toss_peak])
        if win.any():
            # The toss starts when the tossing hand first rises from its low point.
            w = np.flatnonzero(win)
            j = _last(off_z[w] <= off_z[w].min() + p.toss_rise_m)
            i_start = int(w[j]) if j is not None else int(w[0])
    elif i_bs is not None:
        turn = k.shoulder_turn
        win = (t >= tc - p.before_s) & (t <= t[i_bs])
        if win.sum() >= 3:
            w = np.flatnonzero(win)
            sign = 1.0 if turn[i_bs] >= np.median(turn[w]) else -1.0
            tt = sign * turn[w]
            lo, hi = float(tt.min()), float(tt[-1])
            if hi - lo > 10:
                j = _last(tt <= lo + 0.15 * (hi - lo))
                i_start = int(w[j]) if j is not None else int(w[0])
    # Split step (groundstrokes): pelvis dips and rises just before the preparation.
    if i_start is not None and not overhead:
        win = (t >= t[i_start] - 0.8) & (t <= t[i_start] + 0.1)
        if win.sum() >= 6:
            w = np.flatnonzero(win)
            j = w[int(np.argmin(pelvis_z[w]))]
            pre = pelvis_z[w[0] : j + 1].max() - pelvis_z[j]
            post = pelvis_z[j : w[-1] + 1].max() - pelvis_z[j]
            if min(pre, post) >= p.split_min_dip_m and w[0] < j < w[-1]:
                rec["t_split"] = float(t[j])
    # Follow-through and recovery.
    after = t > tc
    win = (t >= tc - 0.3) & (t <= tc + 0.15)
    peak_i = np.flatnonzero(win)[int(np.argmax(k.wrist_speed[win]))] if win.any() else ic
    peak_v = float(k.wrist_speed[peak_i])
    i_follow = None
    if overhead:
        # Serves and smashes: the racket wrist has come down to its low point after contact.
        win = after & (t <= tc + 0.8)
        if win.sum() >= 3:
            w = np.flatnonzero(win)
            low, top = float(wrist_z[w].min()), float(wrist_z[ic])
            j = _first(wrist_z[w] <= top - p.follow_drop_frac * (top - low))
            i_follow = int(w[j]) if j is not None else None
    elif after.any():
        slow = after & (np.arange(len(t)) > peak_i) & (k.wrist_speed < p.follow_frac * peak_v)
        hold = max(1, int(0.1 * fps))
        for j in np.flatnonzero(slow):
            if slow[j : j + hold].all():
                i_follow = int(j)
                break
    i_rec = None
    if i_follow is not None:
        still = (k.wrist_speed < p.recovery_speed) & (np.abs(k.trunk_av) < p.recovery_trunk_av)
        hold = max(1, int(0.15 * fps))
        for j in range(i_follow, len(t) - hold):
            if still[j : j + hold].all():
                i_rec = j
                break
    tt = {
        "t_start": t[i_start] if i_start is not None else None,
        "t_backswing_end": t[i_bs] if i_bs is not None else None,
        "t_follow_end": t[i_follow] if i_follow is not None else None,
        "t_recovery_end": t[i_rec] if i_rec is not None else None,
    }
    for name, v in tt.items():
        rec[name] = _f(v)
    if rec.get("t_start") is not None and rec.get("t_backswing_end") is not None:
        rec["prep_s"] = rec["t_backswing_end"] - rec["t_start"]
    if rec.get("t_backswing_end") is not None:
        rec["forward_s"] = tc - rec["t_backswing_end"]
    if rec.get("t_follow_end") is not None:
        rec["follow_s"] = rec["t_follow_end"] - tc
        if rec.get("t_recovery_end") is not None:
            rec["recovery_s"] = rec["t_recovery_end"] - rec["t_follow_end"]
    if rec.get("prep_s") and rec.get("forward_s"):
        rec["tempo"] = rec["prep_s"] / rec["forward_s"]
    # Contact and body metrics.
    wr = k.wrist_rel[ic]
    rec.update(
        contact_height_m=float(wrist_z[ic]),
        contact_front_m=float(wr[1]),
        contact_side_m=float(wr[0]),
        contact_dist_m=float(np.hypot(wr[0], wr[1])),
        wrist_speed_peak=peak_v,
        wrist_speed_avg=_chord_speed(t, k.joints[:, H["r_wrist"]], tc, CHORD_S),
        t_wrist_peak=float(t[peak_i] - tc),
        shoulder_turn_contact=float(k.shoulder_turn[ic]),
        hip_turn_contact=float(k.hip_turn[ic]),
        knee_flex_contact=float(k.knee_flex[ic]),
        elbow_contact=float(k.elbow[ic]),
        trunk_lean_contact=float(k.trunk_lean[ic]),
        stance_width_m=float(k.stance[ic]),
    )
    a = i_start if i_start is not None else int(np.argmin(np.abs(t - (tc - 1.0))))
    prep = slice(a, ic + 1)
    for name, arr in (
        ("shoulder_turn_max", k.shoulder_turn),
        ("hip_turn_max", k.hip_turn),
        ("separation_max", k.separation),
    ):
        seg = arr[prep]
        rec[name] = float(seg[int(np.argmax(np.abs(seg)))]) if len(seg) else None
    rec["knee_flex_max"] = float(np.max(k.knee_flex[prep])) if ic >= a else None
    standing = pelvis_z[: a + 1] if a > 0 else pelvis_z[:1]
    stand_z = float(np.median(standing))
    rec["com_drop_m"] = float(stand_z - pelvis_z[prep].min()) if ic >= a else None
    rec["jump_m"] = float(pelvis_z[ic] - stand_z)
    # Kinetic chain: peak rotation speeds during the forward swing.
    lo = i_bs if i_bs is not None else max(0, ic - int(0.4 * fps))
    fw = slice(max(0, lo - int(0.1 * fps)), min(len(t), ic + int(0.1 * fps) + 1))
    peaks = {}
    for name, arr in (("pelvis", k.hip_av), ("trunk", k.trunk_av), ("elbow", k.elbow_av)):
        seg = np.abs(arr[fw])
        if len(seg):
            j = fw.start + int(np.argmax(seg))
            rec[f"{name}_av_peak"] = float(seg.max())
            rec[f"t_{name}_peak"] = float(t[j] - tc)
            peaks[name] = float(t[j])
    if len(peaks) == 3:
        order = [peaks["pelvis"], peaks["trunk"], peaks["elbow"], float(t[peak_i])]
        tol = 1.5 / fps
        rec["chain_in_order"] = all(b >= a - tol for a, b in itertools.pairwise(order))
    conf = pose.conf[idx]
    #: Mean keypoint confidence over the swing (not a SWINGS column; the stroke rules use it).
    rec["pose_conf"] = float(np.mean(conf))
    rec["pose_quality"] = float(
        np.clip(rec["pose_conf"] / 0.8, 0, 1) * min(1.0, len(idx) / expected)
    )
    if side is not None and side > 0:
        rec["flags"].append("far")
    if peak_v > p.max_plausible_speed:
        rec["flags"].append("implausible_speed")
    if rec["pose_conf"] < 0.5:
        rec["flags"].append("low_conf")
    return rec


def _chord_speed(t: np.ndarray, x: np.ndarray, tc: float, span: float) -> float | None:
    """Straight-line distance the joint covers from ``tc - span`` to ``tc``, per second: unlike
    the peak speed, keypoint jitter doesn't add up."""
    if t[0] > tc - span + 0.05 or t[-1] < tc - 0.05:
        return None
    a = np.array([np.interp(tc - span, t, x[:, i]) for i in range(3)])
    b = np.array([np.interp(tc, t, x[:, i]) for i in range(3)])
    return float(np.linalg.norm(b - a) / span)


def side_of(pose: PoseSeries, t: float) -> int | None:
    idx = pose.around(t, 0.1, 0.1)
    if not len(idx):
        return None
    y = float(np.median(pose.joints[idx, H["pelvis"], 1]))
    return 1 if y > 0 else -1


def swings_table(rows: list[dict]) -> pa.Table:
    cols = {f.name: [r.get(f.name) for r in rows] for f in SWINGS}
    return pa.table({f.name: pa.array(cols[f.name], f.type) for f in SWINGS}, schema=SWINGS)


# ---------------------------------------------------------------------------
# The whole session
# ---------------------------------------------------------------------------


@dataclass
class BallContext:
    """What the ball did before each of the player's hits (from ``events`` and flights)."""

    #: hit event id → (a ball came over the net to the player, it bounced on their half).
    incoming: dict[int, tuple[bool | None, bool | None]] = field(default_factory=dict)

    @classmethod
    def from_tables(cls, events: list[dict], flights: list[dict]) -> BallContext:
        by_end = {f["end_event_id"]: f for f in flights if f["end_event_id"] is not None}
        out: dict[int, tuple[bool | None, bool | None]] = {}
        for e in events:
            if e["kind"] != "hit":
                continue
            side = _side_of_y(e["court_y"])
            f = by_end.get(e["event_id"])
            if f is None or side is None or f["p0_y"] is None:
                out[e["event_id"]] = (None, None)
                continue
            if _side_of_y(f["p0_y"]) == -side:
                out[e["event_id"]] = (True, False)
            elif f["start_kind"] == "bounce":
                prev = by_end.get(f["start_event_id"])
                crossed = (
                    prev is not None
                    and prev["p0_y"] is not None
                    and _side_of_y(prev["p0_y"]) == -side
                )
                out[e["event_id"]] = (crossed, True)
            else:
                out[e["event_id"]] = (False, None)
        return cls(out)


def _side_of_y(y) -> int | None:
    if y is None or not np.isfinite(y):
        return None
    return 1 if y > 0 else -1


def build_swings(
    inp: SwingInputs,
    ball: BallContext,
    p: SwingParams | None = None,
    sp=None,
    model=None,
    stroke_edits: list[tuple[float, str]] = (),
    toss_fn: Callable | None = None,
    serve_frames: list[tuple[float, int]] = (),
) -> tuple[pa.Table, dict]:
    """Every swing of the session → (SWINGS table, summary).

    ``toss_fn(t_guess, frame)`` → :class:`~swingvision.pose.serve_contact.TossContact`: a
    serve's contact from its toss path (``frame``: a contact frame the user set, from
    ``serve_frames`` (contact time, frame)). The serve's contact moves to that frame and its
    phases and metrics are measured from there; the summary keeps each serve's toss contact
    (``tosses``, by swing id) for ``serve_contact``.
    """
    from swingvision.pose import strokes as st

    p = p or SwingParams()
    sp = sp or st.StrokeParams()
    hand, hand_info = racket_hand(inp.pose, [t for _, t in inp.hits], inp.profile_hand, p)
    av_source = "hits" if inp.av_offset_n >= p.av_min_samples else "none"
    if av_source == "none":
        off, n = pose_av_offset(inp, hand, p)
        if n >= p.av_min_samples:
            inp.av_offset_s, av_source = off, "pose"
    lags, lag_info = pose_contact_lags(inp, hand, p)
    swings = detect(inp, hand, p, lags)
    rows: list[dict] = []
    seqs, balls, seq_rows = [], [], []
    for s in swings:
        side = side_of(inp.pose, s.t_contact)
        rec = analyze(s, inp.pose, side, hand, inp.fps, p)
        rec.update(
            player="me",
            t_contact=s.t_contact,
            frame_contact=_frame_at(inp.pose, s.t_contact),
            contact_source=s.contact_source,
            t_contact_pose=s.t_contact_pose,
            hit_event_id=s.hit_event_id,
            side=side,
            racket_hand=hand,
        )
        incoming, bounced = (
            ball.incoming.get(s.hit_event_id, (None, None))
            if s.hit_event_id is not None
            else (None, None)
        )
        if s.kin is not None:
            f = st.features(
                s.kin,
                s.t_contact,
                rec.get("t_backswing_end"),
                incoming,
                bounced,
                toss=rec.get("toss_height_m") is not None,
                ball_contact=s.contact_source in ("hit", "audio"),
                wrist_speed_avg=rec.get("wrist_speed_avg"),
                pose_conf=rec.get("pose_conf"),
            )
            stroke, conf = st.classify_rules(f, sp)
            rec.update(
                stroke_type=stroke, stroke_conf=conf, stroke_source="rules", stroke_rules=stroke
            )
            seq = st.sequence(s.kin, s.t_contact)
            if seq is not None:
                seqs.append(seq)
                balls.append(f.ball_vector())
                seq_rows.append(len(rows))
        else:
            rec.update(stroke_type=None, stroke_conf=None, stroke_source=None)
        rows.append(rec)
    _strongest_strokes(rows, p)
    if model is not None and seqs:
        labels, conf = model.predict(np.stack(seqs), np.array(balls, dtype=np.float32))
        for i, lab, c in zip(seq_rows, labels, conf, strict=True):
            # The model only speaks for the classes it learned; others keep the rules.
            if rows[i]["stroke_rules"] in model.classes:
                rows[i].update(stroke_type=lab, stroke_conf=float(c), stroke_source="model")
    for t_edit, stroke in stroke_edits:
        best = min(rows, key=lambda r: abs(r["t_contact"] - t_edit), default=None)
        if best is not None and abs(best["t_contact"] - t_edit) <= 0.25:
            best.update(stroke_type=stroke, stroke_conf=1.0, stroke_source="user")
    for r in rows:
        if len(inp.luma_t):
            k = int(np.clip(np.searchsorted(inp.luma_t, r["t_contact"]), 0, len(inp.luma_t) - 1))
            if inp.luma[k] < p.dark_luma:
                r["flags"].append("dark")
                if r.get("stroke_conf") is not None and r.get("stroke_source") == "rules":
                    r["stroke_conf"] = min(r["stroke_conf"], 0.5)
    tosses = {}
    if toss_fn is not None:
        tosses = _serve_contacts(rows, swings, inp, hand, p, toss_fn, serve_frames)
    for i, r in enumerate(rows):
        r["swing_id"] = i
    table = swings_table(rows)
    strokes: dict[str, int] = {}
    for r in rows:
        if r.get("stroke_type"):
            strokes[r["stroke_type"]] = strokes.get(r["stroke_type"], 0) + 1
    summary = {
        "swings": len(rows),
        "from_hits": sum(r["contact_source"] == "hit" for r in rows),
        "from_audio": sum(r["contact_source"] == "audio" for r in rows),
        "pose_only": sum(r["contact_source"] == "pose" for r in rows),
        "strokes": strokes,
        "racket_hand": hand,
        "hand": hand_info,
        "pose_contact_lag": lag_info,
        "av_offset_s": round(inp.av_offset_s, 4),
        "av_offset_source": av_source,
        "stroke_model": getattr(model, "name", None),
        "tosses": tosses,
    }
    return table, summary


def _serve_contacts(
    rows: list[dict],
    swings: list[Swing],
    inp: SwingInputs,
    hand: str,
    p: SwingParams,
    toss_fn: Callable,
    serve_frames: list[tuple[float, int]],
) -> dict[str, dict]:
    """Serves' contacts from their toss paths (PLAN.md §7.11): the swing's contact moves to
    the contact frame, and its phases and metrics are measured again from there."""
    from swingvision.pose import serve_contact as sc

    out = {}
    for i, (r, s) in enumerate(zip(rows, swings, strict=True)):
        if r.get("stroke_type") != "serve":
            continue
        edit = min(serve_frames, key=lambda e: abs(e[0] - r["t_contact"]), default=None)
        frame = edit[1] if edit is not None and abs(edit[0] - r["t_contact"]) <= 0.25 else None
        c = toss_fn(r["t_contact"], frame)
        if c is None:
            continue
        out[str(i)] = sc.toss_to_dict(c)
        moved = (
            c.frame is not None and c.t is not None and (c.theta is not None or frame is not None)
        )
        if not moved or abs(c.t - r["t_contact"]) < 1e-6:
            continue
        s.t_contact = c.t
        rec = analyze(s, inp.pose, r["side"], hand, inp.fps, p)
        keep = {k: r[k] for k in r if k.startswith("stroke") or k in ("player", "hit_event_id")}
        extra = [f for f in r["flags"] if f not in ANALYZE_FLAGS]
        r.update(rec)
        r.update(keep)
        r["flags"] = rec["flags"] + extra
        r.update(t_contact=c.t, frame_contact=c.frame, contact_source="toss_path")
    return out


def _strongest_strokes(rows: list[dict], p: SwingParams) -> None:
    """Of strokes closer than ``stroke_gap_s``, keep the strongest (a serve's toss and
    follow-through, or walking off, can look like a swing next to the real one)."""
    strokes = [r for r in rows if r.get("stroke_type") not in (None, "other")]

    def strength(r: dict) -> tuple:
        return (r["stroke_type"] in ("serve", "overhead"), r.get("wrist_speed_avg") or 0.0)

    for r in sorted(strokes, key=strength, reverse=True):
        if r["stroke_type"] == "other":
            continue
        for q in strokes:
            if (
                q is not r
                and q["stroke_type"] != "other"
                and abs(q["t_contact"] - r["t_contact"]) < p.stroke_gap_s
            ):
                q.update(stroke_type="other", stroke_conf=0.6, stroke_rules="other")
                q["flags"].append("near_stronger_swing")


def _frame_at(pose: PoseSeries, t: float) -> int | None:
    if not len(pose.t):
        return None
    k = int(np.clip(np.searchsorted(pose.t, t), 0, len(pose.t) - 1))
    if k > 0 and abs(pose.t[k - 1] - t) < abs(pose.t[k] - t):
        k -= 1
    if abs(pose.t[k] - t) > 0.05:
        return None
    return int(pose.frame[k])
