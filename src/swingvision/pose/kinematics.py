"""Per-frame body kinematics from 3D joints (PLAN.md §7.7).

Everything is measured in the **hitter's frame**: the player's end is turned to the near end
(far-end joints rotated 180° about the vertical), so +y points at the net and +x to the
player's right; for a left-hander x is also mirrored, so every metric reads as for a
right-hander and the racket side is +x. Rotations are angles of the hip and shoulder lines in
the horizontal plane: 0° is square to the net, positive is turned with the racket side back
(the unit turn of a forehand or a serve), negative the other way (a backhand's turn).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swingvision.pose.skeleton import H

#: Savitzky-Golay window (frames) applied to the joints before angles and speeds.
SMOOTH = 7


def hitter_frame(joints: np.ndarray, side: int | None, racket_hand: str) -> np.ndarray:
    """(T, 17, 3) court joints → hitter's frame (see module docstring)."""
    j = np.asarray(joints, dtype=np.float64).copy()
    if side is not None and side > 0:
        j[..., 0] *= -1
        j[..., 1] *= -1
    if racket_hand == "left":
        j[..., 0] *= -1
        # Mirroring swaps the body's sides: relabel so "r_*" is the racket side again.
        swap = [H[f"r_{n}"] for n in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")]
        other = [H[f"l_{n}"] for n in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")]
        j[:, swap + other] = j[:, other + swap]
    return j


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at ``b`` (deg) between ``b→a`` and ``b→c``."""
    u, v = a - b, c - b
    cos = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _turn(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Line from the non-racket to the racket side in the horizontal plane → turn (deg).

    Square to the net the line points +x (0°); racket side back (−y) is a positive turn.
    """
    d = right - left
    return np.degrees(np.unwrap(np.arctan2(-d[..., 1], d[..., 0])))


def _rate(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.zeros_like(x)
    return np.gradient(x, t, axis=0)


@dataclass
class FrameKinematics:
    t: np.ndarray
    joints: np.ndarray  # hitter's frame
    hip_turn: np.ndarray  # deg
    shoulder_turn: np.ndarray
    separation: np.ndarray  # shoulder minus hip turn
    trunk_lean: np.ndarray  # deg from vertical
    knee_flex: np.ndarray  # deg, mean of both legs (0 = straight)
    elbow: np.ndarray  # racket elbow angle (180 = straight)
    wrist_speed: np.ndarray  # racket wrist, m/s
    wrist_rel: np.ndarray  # (T, 3) racket wrist relative to the pelvis
    off_wrist_rel: np.ndarray  # (T, 3) the other (toss) hand relative to the pelvis
    pelvis: np.ndarray  # (T, 3)
    hip_av: np.ndarray  # deg/s
    trunk_av: np.ndarray
    elbow_av: np.ndarray
    stance: np.ndarray  # m, horizontal distance between the ankles

    def curves(self) -> dict[str, np.ndarray]:
        """Named per-frame curves for plotting."""
        return {
            "shoulder_turn": self.shoulder_turn,
            "hip_turn": self.hip_turn,
            "separation": self.separation,
            "knee_flex": self.knee_flex,
            "elbow": self.elbow,
            "trunk_lean": self.trunk_lean,
            "wrist_speed": self.wrist_speed,
            "wrist_height": self.joints[:, H["r_wrist"], 2],
        }


CURVE_LABELS = {
    "shoulder_turn": ("Shoulder turn", "deg"),
    "hip_turn": ("Hip turn", "deg"),
    "separation": ("Hip-shoulder separation", "deg"),
    "knee_flex": ("Knee bend", "deg"),
    "elbow": ("Racket elbow angle", "deg"),
    "trunk_lean": ("Trunk lean", "deg"),
    "wrist_speed": ("Racket wrist speed", "m/s"),
    "wrist_height": ("Racket wrist height", "m"),
}


def frame_kinematics(
    t: np.ndarray, joints: np.ndarray, side: int | None, racket_hand: str
) -> FrameKinematics:
    """Kinematics of consecutive frames ``t`` (s) of court joints (T, 17, 3)."""
    t = np.asarray(t, dtype=np.float64)
    j = hitter_frame(joints, side, racket_hand)
    if len(t) >= SMOOTH:
        from scipy.signal import savgol_filter

        j = savgol_filter(j, SMOOTH, 2, axis=0)
    hip_turn = _turn(j[:, H["l_hip"]], j[:, H["r_hip"]])
    sh_turn = _turn(j[:, H["l_shoulder"]], j[:, H["r_shoulder"]])
    trunk = j[:, H["thorax"]] - j[:, H["pelvis"]]
    lean = np.degrees(np.arctan2(np.linalg.norm(trunk[:, :2], axis=1), trunk[:, 2]))
    knee = (
        180 - _angle(j[:, H["l_hip"]], j[:, H["l_knee"]], j[:, H["l_ankle"]])
        + 180 - _angle(j[:, H["r_hip"]], j[:, H["r_knee"]], j[:, H["r_ankle"]])
    ) / 2  # fmt: skip
    elbow = _angle(j[:, H["r_shoulder"]], j[:, H["r_elbow"]], j[:, H["r_wrist"]])
    wrist = j[:, H["r_wrist"]]
    speed = np.linalg.norm(_rate(wrist, t), axis=1)
    stance = np.linalg.norm(j[:, H["l_ankle"], :2] - j[:, H["r_ankle"], :2], axis=1)
    return FrameKinematics(
        t=t,
        joints=j,
        hip_turn=hip_turn,
        shoulder_turn=sh_turn,
        separation=sh_turn - hip_turn,
        trunk_lean=lean,
        knee_flex=knee,
        elbow=elbow,
        wrist_speed=speed,
        wrist_rel=wrist - j[:, H["pelvis"]],
        off_wrist_rel=j[:, H["l_wrist"]] - j[:, H["pelvis"]],
        pelvis=j[:, H["pelvis"]],
        hip_av=_rate(hip_turn, t),
        trunk_av=_rate(sh_turn, t),
        elbow_av=_rate(elbow, t),
        stance=stance,
    )
