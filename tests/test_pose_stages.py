"""pass2_pose → pose3d → swings → shots/segments on the rendered ball video (stage contract
tests). A fake 2D estimator (a standing skeleton in the player's box) and a fake lifter stand in
for ViTPose and MotionBERT, so the tests run on the CPU without weights."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from swingvision import services
from swingvision.pipeline.runner import run
from swingvision.pipeline.stage import read_manifest
from swingvision.pipeline.stages import default_registry
from swingvision.pose import lift3d, pose2d
from swingvision.pose.skeleton import COCO, H
from swingvision.storage import tables
from swingvision.storage.schemas import EVENTS, POSE2D, POSE3D, SHOTS, SWINGS
from tests.synth_pose import standing
from tests.test_players_stages import players_settings  # noqa: F401  (fixture)

pytestmark = pytest.mark.ffmpeg

#: Where each COCO keypoint sits in the player's box (fraction of width, height).
LAYOUT = {
    "nose": (0.5, 0.06), "l_eye": (0.53, 0.04), "r_eye": (0.47, 0.04), "l_ear": (0.58, 0.05),
    "r_ear": (0.42, 0.05), "l_shoulder": (0.7, 0.2), "r_shoulder": (0.3, 0.2),
    "l_elbow": (0.75, 0.37), "r_elbow": (0.25, 0.37), "l_wrist": (0.78, 0.52),
    "r_wrist": (0.22, 0.52), "l_hip": (0.62, 0.53), "r_hip": (0.38, 0.53),
    "l_knee": (0.62, 0.75), "r_knee": (0.38, 0.75), "l_ankle": (0.62, 0.97),
    "r_ankle": (0.38, 0.97),
}  # fmt: skip


class FakeEstimator:
    def __init__(self, **_):
        self.crops = 0

    def prepare(self, image, box):
        return None, np.asarray(box, dtype=np.float64)

    def run(self, crops, rects):
        self.crops += len(crops)
        out = np.zeros((len(rects), 17, 3), np.float32)
        for i, (x0, y0, x1, y1) in enumerate(rects):
            for k, name in enumerate(COCO):
                fx, fy = LAYOUT[name]
                out[i, k] = (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), 0.9)
        return out


class FakeLifter:
    """A standing skeleton seen from behind, in normalized camera units (x right, y down)."""

    def __init__(self, **_):
        pass

    def lift(self, kp):
        s = standing()
        cam = np.column_stack([s[:, 0], -s[:, 2], s[:, 1]]) / 2.0
        cam -= cam[H["pelvis"]]
        return np.repeat(cam[None], len(kp), axis=0)


def test_pose_stages(players_settings, ball_video, monkeypatch):  # noqa: F811
    monkeypatch.setitem(pose2d.ESTIMATORS, "fake_pose", lambda **kw: FakeEstimator(**kw))
    monkeypatch.setitem(lift3d.LIFTERS, lift3d.LIFTER, lambda **kw: FakeLifter(**kw))
    path, _px, _bounces = ball_video
    s = players_settings
    s.processing.ball_detector = "motion"
    s.processing.pose_model = "fake_pose"
    s.processing.pose_before_s = 0.5
    s.processing.pose_after_s = 0.5
    session = services.create_session(s, path)
    reg = default_registry()
    assert run(reg, session, s, targets=["events"]).status == "done"
    # A hit of the player's at 1.2 s (the rendered "player" stands still all along).
    ev = tables.read_table(session.events_path)
    hit = {f.name: None for f in EVENTS}
    hit.update(event_id=99, kind="hit", frame=72, t_s=1.2, hitter="me", source="user", conf=1.0)
    rows = [r for r in ev.to_pylist() if r["kind"] != "hit"] + [hit]
    tables.write_table(
        pa.table({f.name: pa.array([r[f.name] for r in rows], f.type) for f in EVENTS}),
        session.events_path,
        EVENTS,
    )

    result = run(reg, session, s, targets=["practice_eval"])
    assert result.status == "done", result.message
    assert {"pass2_pose", "pose3d", "swings", "shots", "segments"} <= set(result.ran)

    p2 = tables.read_table(session.pose2d_path)
    assert p2.schema.equals(POSE2D, check_metadata=False)
    t = p2.column("t_s").to_numpy()
    # Every frame in the window around the hit (60 fps), none outside it.
    assert t.min() >= 0.65 and t.max() <= 1.75 and p2.num_rows >= 60
    assert read_manifest(session, "pass2_pose")["extra"]["windows"] == 1

    p3 = tables.read_table(session.pose3d_path)
    assert p3.schema.equals(POSE3D, check_metadata=False)
    joints = np.stack(p3.column("joints").to_numpy(zero_copy_only=False)).reshape(-1, 17, 3)
    # Placed at the player's feet on the near baseline, upright, at the default height.
    feet = joints[:, [H["l_ankle"], H["r_ankle"]]].mean(1)
    assert np.median(feet[:, 1]) < -10.5
    assert np.median(joints[:, H["head"], 2]) == pytest.approx(1.7, abs=0.15)

    sw = tables.read_table(session.swings_path)
    assert sw.schema.equals(SWINGS, check_metadata=False)
    rows = sw.to_pylist()
    assert len(rows) == 1 and rows[0]["hit_event_id"] == 99
    assert rows[0]["stroke_type"] == "other"  # standing still: not a stroke

    shots = tables.read_table(session.shots_path)
    assert shots.schema.equals(SHOTS, check_metadata=False)
    shot = next(r for r in shots.to_pylist() if r["hit_event_id"] == 99)
    assert shot["swing_id"] == rows[0]["swing_id"] and shot["stroke_type"] == "other"

    # A stroke correction reruns the cheap stages in-process, and the shot follows it.
    services.edit_swing_stroke(s, session.load_config().id, 1.2, "forehand")
    assert services.refresh_practice(s, session.load_config().id) == ("ran", None)
    shot = next(
        r for r in tables.read_table(session.shots_path).to_pylist() if r["hit_event_id"] == 99
    )
    assert shot["stroke_type"] == "forehand"
