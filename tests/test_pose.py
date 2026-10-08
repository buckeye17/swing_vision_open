"""Pose (M6): joint conventions, heatmap decoding, windows, 3D placement, kinematics, swing
detection and phases, stroke rules, the evaluation, and stroke edits."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from swingvision.pose import lift3d, pose2d, strokes
from swingvision.pose import swings as sw
from swingvision.pose.kinematics import frame_kinematics, hitter_frame
from swingvision.pose.skeleton import COCO, HEIGHT_PER_CHAIN, H, chain_length, coco_to_h36m
from swingvision.pose.windows import BoxTrack, merge_windows
from swingvision.storage import edits as ed
from swingvision.storage.schemas import SWINGS, SessionEdits
from swingvision.training.stroke_labels import evaluate, phase_consistency
from tests.synth_court import make_camera
from tests.synth_pose import standing, swing

# ---------------------------------------------------------------------------
# Conventions and 2D
# ---------------------------------------------------------------------------


def test_coco_to_h36m_derived_joints():
    kp = np.zeros((17, 3))
    for i in range(len(COCO)):
        kp[i] = (i, 10 * i, 0.9)
    kp[COCO.index("l_hip"), 2] = 0.4
    h = coco_to_h36m(kp)
    hips = (kp[COCO.index("l_hip"), :2] + kp[COCO.index("r_hip"), :2]) / 2
    assert np.allclose(h[H["pelvis"], :2], hips)
    assert h[H["pelvis"], 2] == pytest.approx(0.4)  # the weaker source's confidence
    assert np.allclose(h[H["r_wrist"]], kp[COCO.index("r_wrist")])
    shoulders = (kp[COCO.index("l_shoulder"), :2] + kp[COCO.index("r_shoulder"), :2]) / 2
    assert np.allclose(h[H["thorax"], :2], shoulders)


def test_decode_heatmaps_subpixel():
    yy, xx = np.mgrid[0:64, 0:48]
    hm = np.exp(-((xx - 20.3) ** 2 + (yy - 31.7) ** 2) / (2 * 2.0**2))[None, None]
    out = pose2d.decode_heatmaps(hm)
    assert out[0, 0, 0] == pytest.approx(20.3, abs=0.05)
    assert out[0, 0, 1] == pytest.approx(31.7, abs=0.05)
    assert out[0, 0, 2] == pytest.approx(1.0, abs=0.05)


def test_every_pose_architecture_is_registered():
    from swingvision.models import registry

    for name in pose2d.ARCHS:
        assert registry.get(name).task.startswith("pose2d")
        assert name in pose2d.ESTIMATORS
    huge = pose2d.vitpose_config("vitpose-plus-huge")
    assert huge.backbone_config.num_experts == 6
    assert huge.backbone_config.out_indices == [32] and not huge.use_simple_decoder
    assert huge.num_labels == 17


def test_mixture_of_experts_pose_uses_the_coco_expert(monkeypatch):
    torch = pytest.importorskip("torch")
    from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation

    # A tiny two-expert ViTPose+ (no weights) run through VitPose2D's heatmap path.
    backbone = VitPoseBackboneConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_experts=2,
        part_features=8, out_features=["stage2"], out_indices=[2],
    )  # fmt: skip
    model = VitPoseForPoseEstimation(
        VitPoseConfig(backbone_config=backbone, use_simple_decoder=False, num_labels=17)
    ).eval()
    seen = []
    forward = model.forward
    monkeypatch.setattr(
        model, "forward", lambda **kw: seen.append(kw.get("dataset_index")) or forward(**kw)
    )
    est = pose2d.VitPose2D.__new__(pose2d.VitPose2D)
    est.model, est.moe, est.flip_test = model, True, True
    est._mean = torch.tensor(pose2d.MEAN).view(1, 3, 1, 1)
    est._std = torch.tensor(pose2d.STD).view(1, 3, 1, 1)
    with torch.inference_mode():
        hm = est._heatmaps(torch.rand(3, 3, pose2d.INPUT_H, pose2d.INPUT_W))
    assert hm.shape == (3, 17, 64, 48)
    assert len(seen) == 2 and all(d is not None and d.tolist() == [0, 0, 0] for d in seen)


def test_crop_box_aspect_and_padding():
    box = np.array([100.0, 200.0, 180.0, 500.0])
    x0, y0, x1, y1 = pose2d.crop_box(box)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(192 / 256)
    assert x0 < 100 and x1 > 180 and y0 < 200 and y1 > 480
    assert y0 < 200 - (500 - y1)  # reaches further up than down


def test_windows_merge_and_box_track():
    assert merge_windows(np.array([10.0, 11.0, 30.0]), 1.8, 1.2, 31.0) == [
        (8.2, 12.2),
        (28.2, 31.0),
    ]
    track = BoxTrack(
        np.array([0.0, 1 / 15, 2 / 15, 5.0]), np.array([[0, 0, 10, 20]] * 3 + [[50, 0, 60, 20]])
    )
    b = track.at(0.1)
    assert b is not None and b[0] == pytest.approx(0)
    assert track.at(2.5) is None  # gap: not tracked
    assert track.covered(4.9, 5.1) and not track.covered(1.0, 4.0)


# ---------------------------------------------------------------------------
# 3D placement
# ---------------------------------------------------------------------------


def _lifted_from_truth(joints: np.ndarray, cam, scale: float = 0.31) -> np.ndarray:
    """What an ideal lifter would output: root-relative joints in the virtual camera that looks
    straight at the pelvis (normalized units)."""
    rel_court = joints - joints[:, :1]
    rel_cam = rel_court @ cam.R.T
    px = cam.project(joints[:, 0])
    u = cam.undistort(px)
    d = np.column_stack([(u - cam.c) / cam.f, np.ones(len(u))])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Rv = lift3d._rotation_z_to(d)
    return np.einsum("tji,tkj->tki", Rv, rel_cam) / scale


@pytest.mark.parametrize("side", [-1, 1])
def test_place_recovers_skeleton(side):
    cam = make_camera(3840, 2160)
    t, truth = swing("forehand", 1.0, side=side)
    px = cam.project(truth.reshape(-1, 3)).reshape(len(t), 17, 2)
    kp = np.concatenate([px, np.full((len(t), 17, 1), 0.9)], axis=-1)
    lifted = _lifted_from_truth(truth, cam)
    height = HEIGHT_PER_CHAIN * float(np.median(chain_length(truth)))
    feet = truth[:, [H["l_ankle"], H["r_ankle"]], :2].mean(1)
    placed = lift3d.place(lifted, kp, np.arange(len(t)), cam, height, feet, np.full(len(t), 0.05))
    err = np.linalg.norm(placed.joints - truth, axis=-1)
    assert np.median(err) < 0.03
    assert np.percentile(err, 95) < 0.08
    assert np.nanmedian(placed.reproj_px) < 2.0


def test_scale_to_height():
    rel = (standing() - standing()[0])[None] * 0.25
    out, s = lift3d.scale_to_height(rel, 1.75)
    assert HEIGHT_PER_CHAIN * chain_length(out)[0] == pytest.approx(1.75)
    assert s == pytest.approx(
        1 / 0.25 * 1.75 / (HEIGHT_PER_CHAIN * chain_length(standing()[None])[0])
    )


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("hand", "side"), [("right", -1), ("right", 1), ("left", -1), ("left", 1)])
def test_kinematics_hitter_frame(hand, side):
    t, j = swing("forehand", 1.0, hand=hand, side=side)
    k = frame_kinematics(t, j, side, hand)
    # Every combination reads as a right-hander at the near end facing +y.
    i_back = int(np.argmin(np.abs(t - 0.65)))
    assert k.shoulder_turn[i_back] == pytest.approx(70, abs=6)  # racket side turned back
    assert k.hip_turn[i_back] == pytest.approx(35, abs=6)
    assert k.pelvis[0, 1] < -11
    ic = int(np.argmin(np.abs(t - 1.0)))
    assert k.wrist_rel[ic, 0] > 0.3  # contact on the racket side
    assert k.wrist_speed.max() > 6


def test_hitter_frame_is_an_involution_for_far_lefties():
    j = swing("serve", 1.0)[1]
    back = hitter_frame(hitter_frame(j, 1, "left"), 1, "left")
    assert np.allclose(back, j)


# ---------------------------------------------------------------------------
# Swings and strokes
# ---------------------------------------------------------------------------


def _series(clips: list[tuple[np.ndarray, np.ndarray]]) -> sw.PoseSeries:
    t = np.concatenate([c[0] for c in clips])
    j = np.concatenate([c[1] for c in clips])
    clip = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(clips)])
    return sw.PoseSeries(
        t=t, frame=np.round(t * 60).astype(int), clip=clip, joints=j, conf=np.full(len(t), 0.9)
    )


@pytest.mark.parametrize("hand", ["right", "left"])
def test_build_swings_classifies_and_phases(hand):
    clips = [
        swing("serve", 10.0, hand=hand),
        swing("forehand", 20.0, hand=hand),
        swing("backhand", 30.0, hand=hand),
        swing("serve", 40.0, hand=hand, side=1),
        swing("forehand", 50.0, hand=hand, side=1),
    ]
    inp = sw.SwingInputs(
        pose=_series(clips),
        fps=60.0,
        hits=[(1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0), (5, 50.0)],
    )
    table, summary = sw.build_swings(inp, sw.BallContext())
    assert table.schema.equals(SWINGS, check_metadata=False)
    rows = table.to_pylist()
    assert summary["racket_hand"] == hand
    strokes_found = {round(r["t_contact"]): r["stroke_type"] for r in rows}
    assert strokes_found == {
        10: "serve",
        20: "forehand",
        30: "backhand",
        40: "serve",
        50: "forehand",
    }
    for r in rows:
        assert r["t_start"] < r["t_backswing_end"] < r["t_contact"] < r["t_follow_end"]
        assert r["contact_source"] == "hit"
        assert abs(r["t_contact_pose"] - r["t_contact"]) <= 2 / 60  # contact frame from the pose
    serve = rows[0]
    assert serve["toss_height_m"] > 1.75
    assert serve["contact_height_m"] > 2.0
    assert serve["forward_s"] == pytest.approx(0.2, abs=0.05)  # racket drop → contact


def test_pose_only_and_audio_swings():
    clips = [swing("forehand", 20.0), swing("forehand", 30.0), swing("still", 40.0)]
    inp = sw.SwingInputs(
        pose=_series(clips),
        fps=60.0,
        onsets_t=np.array([20.0, 40.0]),
        onsets_s=np.array([30.0, 30.0]),
    )
    rows = sw.build_swings(inp, sw.BallContext())[0].to_pylist()
    by_t = {round(r["t_contact"]): r for r in rows}
    # The sound gives the first contact; the second has none (a shadow swing: not a stroke);
    # standing still with a loud sound isn't a swing.
    assert by_t[20]["contact_source"] == "audio" and by_t[20]["stroke_type"] == "forehand"
    assert by_t[30]["contact_source"] == "pose" and by_t[30]["stroke_type"] == "other"
    assert 40 not in by_t


def test_stroke_edits_override_and_strongest_stroke_wins():
    clips = [swing("serve", 10.0)]
    inp = sw.SwingInputs(pose=_series(clips), fps=60.0, hits=[(1, 10.0), (2, 9.3)])
    rows = sw.build_swings(inp, sw.BallContext())[0].to_pylist()
    by_t = {round(r["t_contact"], 1): r for r in rows}
    assert by_t[10.0]["stroke_type"] == "serve"
    assert by_t[9.3]["stroke_type"] == "other"  # the toss next to it
    rows = sw.build_swings(inp, sw.BallContext(), stroke_edits=[(10.02, "overhead")])[0].to_pylist()
    r = next(r for r in rows if abs(r["t_contact"] - 10.0) < 0.01)
    assert (r["stroke_type"], r["stroke_source"], r["stroke_rules"]) == (
        "overhead",
        "user",
        "serve",
    )


def _f(**kw) -> strokes.StrokeFeatures:
    base = dict(
        wrist_speed_peak=9.0, wrist_speed_avg=4.0, arm_above_head_m=-0.6, toss=False,
        depth_m=11.5, contact_side_m=0.4, turn_backswing_deg=60.0, lateral_speed=-2.0,
        incoming=True, bounced=True,
    )  # fmt: skip
    base.update(kw)
    return strokes.StrokeFeatures(**base)


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, "forehand"),
        ({"contact_side_m": -0.4, "turn_backswing_deg": -60.0, "lateral_speed": 2.0}, "backhand"),
        ({"bounced": False, "depth_m": 5.0}, "forehand_volley"),
        (
            {"bounced": False, "depth_m": 5.0, "contact_side_m": -0.4, "turn_backswing_deg": -50.0},
            "backhand_volley",
        ),
        ({"arm_above_head_m": 0.3, "toss": True, "incoming": False}, "serve"),
        ({"arm_above_head_m": 0.3, "toss": True, "wrist_speed_avg": 1.0}, "serve"),
        ({"arm_above_head_m": 0.3, "incoming": None, "depth_m": 12.0}, "serve"),
        ({"arm_above_head_m": 0.3, "depth_m": 6.0}, "overhead"),
        ({"wrist_speed_peak": 2.0}, "other"),
        ({"wrist_speed_avg": 1.0}, "other"),
        ({"ball_contact": False}, "other"),
        ({"off_above_head_m": 0.1}, "other"),
    ],
)
def test_stroke_rules(kw, expected):
    assert strokes.classify_rules(_f(**kw))[0] == expected


def test_ball_context():
    events = [
        {"event_id": 1, "kind": "hit", "court_y": 11.0},
        {"event_id": 2, "kind": "bounce", "court_y": -8.0},
        {"event_id": 3, "kind": "hit", "court_y": -12.0},
        {"event_id": 4, "kind": "hit", "court_y": -3.0},
    ]
    flights = [
        {"flight_id": 0, "start_kind": "hit", "start_event_id": 1, "end_event_id": 2, "p0_y": 11.0},
        {
            "flight_id": 1,
            "start_kind": "bounce",
            "start_event_id": 2,
            "end_event_id": 3,
            "p0_y": -8.0,
        },
        {"flight_id": 2, "start_kind": "hit", "start_event_id": 3, "end_event_id": 4, "p0_y": 11.5},
    ]
    ctx = sw.BallContext.from_tables(events, flights)
    assert ctx.incoming[3] == (True, True)  # came over, bounced: a groundstroke
    assert ctx.incoming[4] == (True, False)  # came over, no bounce: a volley
    assert ctx.incoming[1] == (None, None)


# ---------------------------------------------------------------------------
# Evaluation and edits
# ---------------------------------------------------------------------------


def _row(i, t, stroke, side=-1, **kw):
    r = {f.name: None for f in SWINGS}
    r.update(swing_id=i, t_contact=t, stroke_type=stroke, side=side, flags=[], hit_event_id=None)
    r.update(kw)
    return r


def test_evaluate_matches_strokes_one_to_one():
    rows = [
        _row(0, 10.0, "serve", hit_event_id=1, t_contact_pose=10.016),
        _row(1, 9.2, "serve"),  # a second "serve" next to the first: a false stroke
        _row(2, 20.4, "other"),  # the labeled serve came out as other
        _row(3, 30.0, "other"),
        _row(4, 40.0, "forehand", side=1),
        _row(5, 60.0, "serve", flags=["dark"]),
    ]
    table = pa.table({f.name: pa.array([r[f.name] for r in rows], f.type) for f in SWINGS})
    labels = {
        "spans": [],
        "strokes": [
            {"t": 10.1, "stroke": "serve", "side": -1},
            {"t": 20.3, "stroke": "serve", "side": -1},
            {"t": 40.0, "stroke": "forehand", "side": 1},
            {"t": 50.0, "stroke": "serve", "side": 1, "visible": False},
            {"t": 60.0, "stroke": "serve", "side": -1},
        ],
    }
    rep = evaluate(table, labels, 60.0, hit_times={1: 10.0}, exclude_flags=frozenset({"dark"}))
    near, far = rep["near"], rep["far"]
    assert (near.strokes, near.strokes_correct) == (2, 1)
    assert (near.others, near.others_correct) == (2, 1)  # 9.2 wrong, 30.0 right
    assert (far.strokes, far.strokes_correct, far.others) == (1, 1, 0)
    assert near.contact_errors == [1]


def test_phase_consistency():
    rows = [
        _row(i, 10.0 * i, "serve", t_start=10.0 * i - 1.0, t_backswing_end=10.0 * i - 0.2,
             t_follow_end=10.0 * i + 0.3, prep_s=0.8, forward_s=0.2 + 0.01 * (i % 3), follow_s=0.3)
        for i in range(1, 8)
    ]  # fmt: skip
    g = phase_consistency(rows)["serve/near"]
    assert g["in_order"] == 1.0
    assert g["forward_s"]["cv"] < 0.1


def test_swing_edits():
    edits = SessionEdits()
    ed.set_swing_stroke(edits, 12.3456, "forehand")
    ed.set_swing_stroke(edits, 12.40, "backhand")  # same swing
    assert [(e.t, e.stroke) for e in edits.swings] == [(12.346, "backhand")]
    ed.set_swing_stroke(edits, 12.35, None)
    assert edits.swings == []
