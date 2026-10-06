"""Joint conventions: COCO-17 (2D pose) and Human3.6M-17 (3D lifting) (PLAN.md §7.7)."""

from __future__ import annotations

import numpy as np

COCO = (
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
    "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)  # fmt: skip
C = {name: i for i, name in enumerate(COCO)}

COCO_EDGES = (
    (C["l_ankle"], C["l_knee"]), (C["l_knee"], C["l_hip"]), (C["r_ankle"], C["r_knee"]),
    (C["r_knee"], C["r_hip"]), (C["l_hip"], C["r_hip"]), (C["l_shoulder"], C["l_hip"]),
    (C["r_shoulder"], C["r_hip"]), (C["l_shoulder"], C["r_shoulder"]),
    (C["l_shoulder"], C["l_elbow"]), (C["r_shoulder"], C["r_elbow"]),
    (C["l_elbow"], C["l_wrist"]), (C["r_elbow"], C["r_wrist"]), (C["nose"], C["l_eye"]),
    (C["nose"], C["r_eye"]), (C["l_eye"], C["l_ear"]), (C["r_eye"], C["r_ear"]),
)  # fmt: skip

#: COCO keypoint order after a horizontal flip (left ↔ right).
COCO_FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)

H36M = (
    "pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "spine", "thorax",
    "nose", "head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
)  # fmt: skip
H = {name: i for i, name in enumerate(H36M)}

H36M_EDGES = (
    (H["pelvis"], H["r_hip"]), (H["r_hip"], H["r_knee"]), (H["r_knee"], H["r_ankle"]),
    (H["pelvis"], H["l_hip"]), (H["l_hip"], H["l_knee"]), (H["l_knee"], H["l_ankle"]),
    (H["pelvis"], H["spine"]), (H["spine"], H["thorax"]), (H["thorax"], H["nose"]),
    (H["nose"], H["head"]), (H["thorax"], H["l_shoulder"]), (H["l_shoulder"], H["l_elbow"]),
    (H["l_elbow"], H["l_wrist"]), (H["thorax"], H["r_shoulder"]),
    (H["r_shoulder"], H["r_elbow"]), (H["r_elbow"], H["r_wrist"]),
)  # fmt: skip

#: H36M order after a horizontal flip.
H36M_FLIP = (0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13)

#: Bones that add up to standing height (with the foot and the skull above the "head"
#: joint): shank, thigh, pelvis → spine → thorax → nose → head.
HEIGHT_CHAIN = (
    (H["r_ankle"], H["r_knee"]), (H["r_knee"], H["r_hip"]),
    (H["l_ankle"], H["l_knee"]), (H["l_knee"], H["l_hip"]),
    (H["pelvis"], H["spine"]), (H["spine"], H["thorax"]), (H["thorax"], H["nose"]),
    (H["nose"], H["head"]),
)  # fmt: skip
#: Standing height / length of :data:`HEIGHT_CHAIN` (legs averaged). Human3.6M's subjects:
#: ≈1.62 m of chain for ≈1.70 m of height (ankle-to-sole and the top of the skull).
HEIGHT_PER_CHAIN = 1.05


def coco_to_h36m(kp: np.ndarray) -> np.ndarray:
    """COCO-17 keypoints (..., 17, D) → Human3.6M-17 (..., 17, D).

    Pelvis = mid-hips, thorax = mid-shoulders, spine = midway between them; COCO has no head
    top, so it is extrapolated from the neck through the mid-ears (as far again as the ears
    are above the shoulders, halved). For the confidence channel (D = 3) derived joints take
    the smaller confidence of their sources.
    """
    kp = np.asarray(kp, dtype=np.float64)
    out = np.zeros_like(kp)

    def mid(a, b):
        m = (kp[..., C[a], :] + kp[..., C[b], :]) / 2
        if kp.shape[-1] == 3:
            m[..., 2] = np.minimum(kp[..., C[a], 2], kp[..., C[b], 2])
        return m

    pelvis = mid("l_hip", "r_hip")
    thorax = mid("l_shoulder", "r_shoulder")
    ears = mid("l_ear", "r_ear")
    out[..., H["pelvis"], :] = pelvis
    out[..., H["thorax"], :] = thorax
    spine = (pelvis + thorax) / 2
    if kp.shape[-1] == 3:
        spine[..., 2] = np.minimum(pelvis[..., 2], thorax[..., 2])
    out[..., H["spine"], :] = spine
    head = ears.copy()
    head[..., :2] = ears[..., :2] + 0.5 * (ears[..., :2] - thorax[..., :2])
    out[..., H["head"], :] = head
    for name in ("nose", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"):
        out[..., H[name], :] = kp[..., C[name], :]
    for name in ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"):
        out[..., H[name], :] = kp[..., C[name], :]
    return out


def chain_length(joints: np.ndarray) -> np.ndarray:
    """Length of :data:`HEIGHT_CHAIN` per frame (legs averaged) for (..., 17, 3) joints."""
    j = np.asarray(joints, dtype=np.float64)

    def bone(a, b):
        return np.linalg.norm(j[..., a, :] - j[..., b, :], axis=-1)

    legs = sum(bone(a, b) for a, b in HEIGHT_CHAIN[:4]) / 2
    trunk = sum(bone(a, b) for a, b in HEIGHT_CHAIN[4:])
    return legs + trunk


def hand_joints(handedness: str) -> tuple[int, int, int, int]:
    """H36M (wrist, elbow, shoulder, hip) of the racket arm, from the profile's handedness."""
    s = "l" if handedness == "left" else "r"
    return H[f"{s}_wrist"], H[f"{s}_elbow"], H[f"{s}_shoulder"], H[f"{s}_hip"]
