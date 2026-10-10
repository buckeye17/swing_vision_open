"""Serve contact point (M7b): the contact frame from the toss path, the toe in that frame, and
the offsets, on synthetic serves with a known toe and contact (PLAN.md §7.11).

The toss and the serve come from an independent integrator (``tests.synth_ball.flight_3d``,
drag 0.58 vs the fit's 0.55 prior) seen through a camera like the user's with 1 px noise.
Before the release the ball rides up in the hand (accelerating, so not on the free path);
after the contact the serve leaves at 44 m/s, crosscourt. The front toe's keypoints are drawn where the
whole-body model puts them (above the sole, short of the shoe's tip).
"""

from __future__ import annotations

import numpy as np
import pytest

from swingvision.pose import feet as ft
from swingvision.pose import serve_contact as sc
from tests.synth_ball import flight_3d
from tests.synth_court import make_camera

CAM = make_camera(
    width=3840, height=2160, hfov_deg=87.5, height_m=3.26, behind_m=6.4, offset_x=0.1,
    look_y=-1.0,
)  # fmt: skip
FPS = 59.2
CD = 0.58
TOE_TIP = np.array([-0.15, -12.05, 0.0])  # the left-hander's right toe, just behind the line
FOOT_DIR = np.array([0.6, 0.8]) / 1.0  # heel → toe on the ground (normalized below)
FOOT_DIR = FOOT_DIR / np.linalg.norm(FOOT_DIR)


def _frames(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    frames = np.arange(1000, 1000 + n)
    return frames, (frames - 1000) / FPS


def serve(
    rng,
    t_contact: float = 3.0,
    release_h: float = 1.75,
    up: float = 4.6,
    hidden: tuple[int, ...] = (),
    jitter: float = 1.0,
) -> dict:
    """A synthetic serve: frames, detections, the true contact frame and contact point."""
    frames, t = _frames()
    hand_t = 0.35  # the hand pushes the ball up for this long before the release
    # Toss: back-solve the release so the ball passes the contact time on its way down.
    t_rel = t_contact - 0.9
    rel = np.array([TOE_TIP[0] + 0.25, TOE_TIP[1] + 0.35, release_h])
    v_rel = np.array([0.05, 0.35, up])
    pos, _, _ = flight_3d(rel, v_rel, t_rel, t, cd=CD)
    # Before the release: in the hand, accelerating evenly from rest to the release velocity.
    hand = (t >= t_rel - hand_t) & (t < t_rel)
    s = t[hand] - t_rel
    pos[hand] = rel + np.outer(s + s**2 / (2 * hand_t), v_rel)
    # The contact frame: the last frame at or before the contact.
    k_c = int(np.flatnonzero(t <= t_contact)[-1])
    contact_true, _, _ = flight_3d(rel, v_rel, t_rel, np.array([t[k_c]]), cd=CD)
    ball_c, _, _ = flight_3d(rel, v_rel, t_rel, np.array([t_contact]), cd=CD)
    # After the contact: the serve.
    # A crosscourt serve: ≈30 px per frame across the image right after the contact, as on
    # the real footage.
    p_serve, _, _ = flight_3d(ball_c[0], (-8.0, 43.0, -4.0), t_contact, t, cd=CD)
    after = t > t_contact
    pos[after] = p_serve[after]
    seen = np.isfinite(pos).all(axis=1) & (t >= t_rel - hand_t)
    px = CAM.project(np.where(seen[:, None], pos, 0.0)) + rng.normal(0, jitter, (len(t), 2))
    seen &= (px[:, 1] > 0) & (px[:, 1] < CAM.height) & (t < t_contact + 0.3)
    for h in hidden:
        seen[k_c + h] = False
    return {
        "pts": sc.BallPoints(frames[seen], t[seen], px[seen, 0], px[seen, 1]),
        "clock": sc.FrameClock(frames, t),
        "frame": int(frames[k_c]),
        "t_frame": float(t[k_c]),
        "contact": contact_true[0],
        "t_release": t_rel,
        "frames": frames,
        "t": t,
    }


def toe_keypoints(frames, t, rng, lift_from: float | None = None, jitter: float = 0.7):
    """FOOT keypoints (n, 8, 3) of a still right foot (or one rising from ``lift_from``)."""
    p = sc.ToeParams()
    n = len(frames)
    kp = np.zeros((n, len(ft.FOOT), 3))
    ank, big, small, heel = ft.foot_names("right")
    lift = np.zeros(n)
    if lift_from is not None:
        lift = np.clip(t - lift_from, 0, None) * 1.2  # rising at 1.2 m/s
    toe_kp = np.r_[TOE_TIP[:2] - p.tip_ahead_m * FOOT_DIR, p.kp_height_m]
    heel_pt = np.r_[TOE_TIP[:2] - 0.28 * FOOT_DIR, 0.0]
    for name, P in ((big, toe_kp), (small, toe_kp + np.array([0.03, -0.02, 0.0])), (heel, heel_pt)):
        X = np.tile(P, (n, 1))
        X[:, 2] += lift
        kp[:, name, :2] = CAM.project(X) + rng.normal(0, jitter, (n, 2))
        kp[:, name, 2] = 6.0
    kp[:, ank, :2] = CAM.project(np.tile(heel_pt + np.array([0.05, 0.05, 0.1]), (n, 1)))
    kp[:, ank, 2] = 6.0
    return kp


def _check(seed: int, jitter: float, phase: float):
    """One synthetic serve (contact ``phase`` of a frame interval after a frame): returns
    the toss contact, the truth, and the measured and true offsets."""
    rng = np.random.default_rng(seed)
    t_c = (round(3.0 * FPS) + phase) / FPS  # ``phase`` of a frame interval after a frame
    s = serve(rng, t_contact=t_c, up=rng.uniform(4.3, 5.0), jitter=jitter)
    c = sc.toss_contact(s["pts"], s["clock"], CAM, t_c + rng.normal(0, 0.03))
    m = np.abs(s["t"] - t_c) < 1.0
    kp = toe_keypoints(s["frames"][m], s["t"][m], rng, jitter=jitter)
    toe = sc.toe_at_contact(sc.FeetSeries(s["frames"][m], s["t"][m], kp), CAM, "right", c.frame)
    got = sc.offsets(c.pos, c.cov, toe.pos, toe.cov, -1, "left")
    want = sc.offsets(s["contact"], None, TOE_TIP, None, -1, "left")
    return c, s, toe, got, want


@pytest.mark.parametrize("seed", range(6))
def test_contact_frame_and_offsets_exact_without_noise(seed):
    """The exit criterion: offsets within 2 cm, the contact frame exact (noise-free
    detections and keypoints; the toss and serve still from the other integrator)."""
    phase = np.random.default_rng(100 + seed).uniform(0.05, 0.85)
    c, s, toe, got, want = _check(seed, 0.0, phase)
    assert c.frame == s["frame"] and c.source == "toss_path", c.flags
    assert abs(c.t_release - s["t_release"]) < 0.04
    assert toe.on_ground and toe.source == "toe"
    for k in ("forward_m", "lateral_m", "height_m"):
        assert abs(getattr(got, k) - getattr(want, k)) < 0.02, k


@pytest.mark.parametrize("seed", range(8))
def test_offsets_with_detection_noise_stay_within_their_sigma(seed):
    """1 px detection and keypoint noise: depth along the camera axis (≈ forward) rests on the
    toss's gravity curvature, a few cm (docs/m7b-serve-contact.md); the errors stay within
    the reported σ. Contacts up to 70% of a frame interval after a frame give the exact
    frame; a later one may leave the next frame's ball inside the gate (moved ≤ 6 px, a few
    cm), and that frame is taken as the contact (``test_late_contact_within_one_frame``)."""
    phase = np.random.default_rng(200 + seed).uniform(0.05, 0.7)
    c, s, toe, got, want = _check(seed, 1.0, phase)
    assert c.frame == s["frame"] and c.source == "toss_path", c.flags
    assert np.linalg.norm(toe.pos[:2] - TOE_TIP[:2]) < 0.02
    for k in ("forward_m", "lateral_m", "height_m"):
        err = abs(getattr(got, k) - getattr(want, k))
        assert err < max(0.02, 3 * getattr(got, k.replace("_m", "_sigma_m"))), k
    assert got.forward_sigma_m < 0.1 and got.lateral_sigma_m < 0.02


def test_late_contact_within_one_frame():
    for seed in range(4):
        c, s, *_ = _check(seed, 1.0, 0.97)
        assert abs(c.frame - s["frame"]) <= 1


def test_offsets_axes_follow_the_hitter():
    """Forward is toward the net; lateral is toward the racket arm for either hand."""
    toe = np.array([0.0, -12.0, 0.0])
    ball = np.array([0.2, -11.7, 2.8])  # in front, to the right (as seen from the near end)
    left = sc.offsets(ball, None, toe, None, -1, "left")
    right = sc.offsets(ball, None, toe, None, -1, "right")
    assert left.forward_m == pytest.approx(0.3) and left.height_m == pytest.approx(2.8)
    assert right.lateral_m == pytest.approx(0.2) and left.lateral_m == pytest.approx(-0.2)
    far = sc.offsets(-ball, None, -toe, None, 1, "right")  # the same serve from the far end
    assert far.forward_m == pytest.approx(0.3) and far.lateral_m == pytest.approx(0.2)


def test_contact_frame_inferred_when_the_ball_is_hidden():
    rng = np.random.default_rng(11)
    t_c = (round(3.0 * FPS) + 0.4) / FPS
    s = serve(rng, t_contact=t_c, hidden=(-1, 0, 1))
    c = sc.toss_contact(s["pts"], s["clock"], CAM, t_c)
    assert "contact_frame_inferred" in c.flags and c.source == "inferred"
    assert abs(c.frame - s["frame"]) <= 1


def test_toss_above_the_frame_is_flagged():
    rng = np.random.default_rng(3)
    cam_t = 3.0
    s = serve(rng, t_contact=cam_t, up=9.0, release_h=2.2)
    c = sc.toss_contact(s["pts"], s["clock"], CAM, cam_t)
    assert "contact_above_frame" in c.flags and c.pos is None


def test_no_toss_tracked():
    frames, t = _frames()
    pts = sc.BallPoints(frames[:0], t[:0], t[:0], t[:0])
    c = sc.toss_contact(pts, sc.FrameClock(frames, t), CAM, 3.0)
    assert "toss_not_tracked" in c.flags and c.pos is None and c.frame is not None


def test_contact_frame_edit_keeps_the_toss_path():
    rng = np.random.default_rng(5)
    s = serve(rng)
    c = sc.toss_contact(s["pts"], s["clock"], CAM, 3.0, frame_override=s["frame"] - 1)
    assert c.frame == s["frame"] - 1 and c.source == "edit" and c.pos is not None


def test_lifted_toe_is_flagged_with_a_larger_sigma():
    rng = np.random.default_rng(2)
    frames, t = _frames()
    m = np.abs(t - 3.0) < 1.0
    still = sc.FeetSeries(frames[m], t[m], toe_keypoints(frames[m], t[m], rng))
    lifting = sc.FeetSeries(frames[m], t[m], toe_keypoints(frames[m], t[m], rng, lift_from=2.85))
    f = int(frames[np.flatnonzero(t <= 3.0)[-1]])
    down = sc.toe_at_contact(still, CAM, "right", f)
    up = sc.toe_at_contact(lifting, CAM, "right", f)
    assert down.on_ground and not up.on_ground and "toe_lifted" in up.flags
    assert up.sigma_m > 2 * down.sigma_m


def test_hidden_toe_from_the_heel_and_foot_length():
    rng = np.random.default_rng(4)
    frames, t = _frames()
    m = np.abs(t - 3.0) < 1.0
    kp = toe_keypoints(frames[m], t[m], rng, jitter=0.0)
    kp[:, ft.F["r_big_toe"], 2] = 1.0  # not confident: the shoe hides it
    feet = sc.FeetSeries(frames[m], t[m], kp)
    toe = sc.toe_at_contact(feet, CAM, "right", int(frames[m][60]), foot_length_m=0.28)
    assert toe.source == "heel_length" and "toe_estimated" in toe.flags
    assert np.linalg.norm(toe.pos[:2] - TOE_TIP[:2]) < 0.03


def test_toe_edit_wins():
    rng = np.random.default_rng(6)
    frames, t = _frames()
    m = np.abs(t - 3.0) < 1.0
    feet = sc.FeetSeries(frames[m], t[m], toe_keypoints(frames[m], t[m], rng))
    target = np.array([0.1, -12.2, 0.0])
    px = CAM.project(target[None])[0]
    toe = sc.toe_at_contact(feet, CAM, "right", int(frames[m][60]), edit_px=px)
    assert toe.source == "edit" and np.linalg.norm(toe.pos[:2] - target[:2]) < 1e-3


def test_toss_round_trips_as_json():
    rng = np.random.default_rng(8)
    s = serve(rng)
    c = sc.toss_contact(s["pts"], s["clock"], CAM, 3.0)
    back = sc.toss_from_dict(sc.toss_to_dict(c))
    assert back.frame == c.frame and np.allclose(back.pos, c.pos) and back.flags == c.flags
    fit = sc.toss_fit(back)
    pos, _ = sc.path_at(fit, np.array([c.t]))
    assert np.allclose(pos[0], c.pos, atol=1e-9)


def test_serve_side():
    assert sc.serve_side_of(0.5, -1) == "deuce" and sc.serve_side_of(-0.5, -1) == "ad"
    assert sc.serve_side_of(-0.5, 1) == "deuce" and sc.serve_side_of(0.05, -1) is None
