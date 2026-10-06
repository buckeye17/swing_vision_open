"""Synthetic 3D swings (Human3.6M joints, court meters) for the pose tests."""

from __future__ import annotations

import numpy as np

from swingvision.pose.skeleton import H

#: A player standing at the near end facing the net (+y), right-handed, pelvis at the origin
#: of x/y (z up). Ankle-to-head chain ≈ 1.67 m → 1.75 m tall.
STANDING = {
    "pelvis": (0.0, 0.0, 0.95),
    "r_hip": (0.13, 0.0, 0.93),
    "r_knee": (0.14, 0.03, 0.51),
    "r_ankle": (0.14, 0.0, 0.08),
    "l_hip": (-0.13, 0.0, 0.93),
    "l_knee": (-0.14, 0.03, 0.51),
    "l_ankle": (-0.14, 0.0, 0.08),
    "spine": (0.0, 0.0, 1.18),
    "thorax": (0.0, 0.0, 1.42),
    "nose": (0.0, 0.08, 1.60),
    "head": (0.0, 0.02, 1.75),
    "l_shoulder": (-0.19, 0.0, 1.42),
    "l_elbow": (-0.22, 0.02, 1.14),
    "l_wrist": (-0.24, 0.08, 0.90),
    "r_shoulder": (0.19, 0.0, 1.42),
    "r_elbow": (0.22, 0.02, 1.14),
    "r_wrist": (0.24, 0.08, 0.90),
}
UPPER = ("spine", "thorax", "nose", "head", "l_shoulder", "l_elbow", "l_wrist",
         "r_shoulder", "r_elbow", "r_wrist")  # fmt: skip


def standing() -> np.ndarray:
    j = np.zeros((17, 3))
    for name, xyz in STANDING.items():
        j[H[name]] = xyz
    return j


def _minjerk(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    s = np.clip((t - t0) / (t1 - t0), 0, 1)
    return s**3 * (10 - 15 * s + 6 * s * s)


def _rot_z(points: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    out = points.copy()
    out[..., 0] = c * points[..., 0] - s * points[..., 1]
    out[..., 1] = s * points[..., 0] + c * points[..., 1]
    return out


def swing(
    kind: str,
    t_contact: float,
    fps: float = 60.0,
    before: float = 1.8,
    after: float = 1.2,
    hand: str = "right",
    side: int = -1,
    x: float = 0.0,
    y: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(t, joints (n, 17, 3)) of one ``forehand``, ``backhand``, ``serve`` or ``still`` around
    ``t_contact``, for a player on ``side`` (-1 near, +1 far) at (x, y)."""
    t = np.arange(t_contact - before, t_contact + after, 1 / fps)
    tc = t_contact
    base = standing()
    n = len(t)
    j = np.repeat(base[None], n, axis=0)
    if kind in ("forehand", "backhand"):
        sign = 1.0 if kind == "forehand" else -1.0
        # Unit turn (racket side back for a forehand), uncoil through the contact.
        turn = 70 * _minjerk(t, tc - 1.3, tc - 0.35) - 100 * _minjerk(t, tc - 0.3, tc + 0.3)
        turn += 30 * _minjerk(t, tc + 0.4, tc + 1.0)
        turn *= sign
        for k in range(n):
            idx = [H[u] for u in UPPER]
            j[k, idx] = _rot_z(base[idx] - base[H["pelvis"]], -turn[k]) + base[H["pelvis"]]
            hips = [H["l_hip"], H["r_hip"]]
            j[k, hips] = _rot_z(base[hips] - base[H["pelvis"]], -0.5 * turn[k]) + base[H["pelvis"]]
        # Racket wrist on an arc around the body: behind → beside → in front → across.
        alpha = np.radians(30 + 190 * _minjerk(t, tc - 0.3, tc + 0.3))
        ready = _minjerk(t, tc - 1.2, tc - 0.3)
        r = 0.7
        wx = sign * r * np.sin(alpha)
        wy = -r * np.cos(alpha)
        # Before the backswing the hand is in front; it travels to the arc's start.
        wx = (1 - ready) * sign * 0.25 + ready * wx
        wy = (1 - ready) * 0.35 + ready * wy
        j[:, H["r_wrist"]] = np.column_stack([wx, wy, np.full(n, 1.0)])
        j[:, H["r_elbow"]] = (j[:, H["r_wrist"]] + j[:, H["r_shoulder"]]) / 2 + [0, 0, -0.05]
    elif kind == "serve":
        y = -12.3 if y is None else y
        # Toss: the left hand goes up above the head and comes down again.
        toss = _minjerk(t, tc - 1.3, tc - 0.6) - _minjerk(t, tc - 0.5, tc)
        j[:, H["l_wrist"], 2] = 0.9 + 1.1 * toss
        j[:, H["l_wrist"], 1] = 0.08 + 0.2 * toss
        j[:, H["l_elbow"], 2] = 1.14 + 0.5 * toss
        # Racket: behind the back (the drop), up to the contact above the head, down across.
        drop = _minjerk(t, tc - 1.0, tc - 0.2)
        up = _minjerk(t, tc - 0.2, tc)
        down = _minjerk(t, tc + 0.02, tc + 0.4)
        wz = 0.9 + 0.6 * drop + 0.9 * up - 1.4 * down
        wy = 0.08 - 0.4 * drop + 0.7 * up + 0.1 * down
        wx = 0.24 - 0.04 * up - 0.5 * down
        j[:, H["r_wrist"]] = np.column_stack([wx, wy, wz])
        # Elbow: bent most at the racket drop (0.2 s before contact), straight at contact.
        bend = 0.25 * np.exp(-(((t - (tc - 0.2)) / 0.1) ** 2))
        j[:, H["r_elbow"]] = (j[:, H["r_wrist"]] + j[:, H["r_shoulder"]]) / 2 + np.column_stack(
            [bend, -0.6 * bend, np.zeros(n)]
        )
    y = (-12.0 if y is None else y) if kind != "serve" else y
    # Hitter's frame → court: mirror for a left-hander, turn 180° for the far end.
    if hand == "left":
        j[..., 0] *= -1
        names = ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")
        a = [H[f"r_{q}"] for q in names]
        b = [H[f"l_{q}"] for q in names]
        j[:, a + b] = j[:, b + a]
    if side > 0:
        j[..., 0] *= -1
        j[..., 1] *= -1
    j[..., 0] += x if side < 0 else -x
    j[..., 1] += y if side < 0 else -y
    return t, j
