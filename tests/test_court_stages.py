"""court_auto + camera stages on a rendered court video (stage contract tests, PLAN.md §12)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from swingvision import services
from swingvision.court import calibration as calib
from swingvision.pipeline.runner import plan, run
from swingvision.pipeline.stages import default_registry
from tests.conftest import COURT_CAMERA as TRUE_CAMERA
from tests.synth_court import keypoint_error, render

pytestmark = pytest.mark.ffmpeg


@pytest.fixture
def court_settings(settings):
    settings.processing.calibration_window_s = 1.0
    settings.processing.calibration_frames_per_window = 3
    return settings


def test_auto_calibration_accepted_under_threshold(court_settings, court_video):
    court_settings.processing.calibration_auto_accept_px = 2.0
    session = services.create_session(court_settings, court_video)
    result = run(default_registry(), session, court_settings, targets=["camera"])
    assert result.status == "done", result.message
    assert session.court_background_path.exists()
    cal = calib.load(session.calibration_path)
    assert cal is not None and cal.confirmed_by == "auto" and cal.ok
    assert cal.metrics.rms_line_px < 1.5  # unfiltered, on an H.264-compressed 720p video
    assert keypoint_error(TRUE_CAMERA, calib.to_camera(cal.camera)) < 1.0
    # The moving player is gone from the median background.
    bg = cv2.imread(str(session.court_background_path))
    assert (
        np.abs(
            bg.astype(int) - cv2.cvtColor(render(TRUE_CAMERA, occluder=False), cv2.COLOR_RGB2BGR)
        ).mean()
        < 6
    )
    # Drift check: three windows, camera never moved.
    auto = calib.load(session.court_auto_path)
    assert len(auto.drift) == 3 and not auto.drift_detected
    assert all(d.status == "ok" and d.shift_rms_px < 1.0 for d in auto.drift)


def test_review_gate_then_user_confirmation(court_settings, court_video):
    session = services.create_session(court_settings, court_video)
    reg = default_registry()
    result = run(reg, session, court_settings, targets=["camera"])
    assert result.status == "needs_action" and "Review the court calibration" in result.message
    assert not session.calibration_path.exists()

    # The user confirms (unchanged) in the editor → the job completes.
    auto = calib.load(session.court_auto_path)
    user = auto.model_copy(update={"source": "user"})
    calib.save(session.court_user_path, user)
    result = run(reg, session, court_settings, targets=["camera"])
    assert result.status == "done" and result.ran == ["camera"]
    cal = calib.load(session.calibration_path)
    assert cal.confirmed_by == "user"
    config = session.load_config()
    assert all(p.fresh for p in plan(reg, session, config, court_settings, ["camera"]))

    # Re-confirming the same geometry doesn't invalidate anything; moving the camera does.
    calib.save(session.court_user_path, user.model_copy(update={"message": "again"}))
    assert all(p.fresh for p in plan(reg, session, config, court_settings, ["camera"]))
    cam = calib.to_camera(user.camera)
    moved = calib.with_camera(user, cam.with_params(cam.params() + 1e-4))
    calib.save(session.court_user_path, moved)
    stale = [
        p.stage.name for p in plan(reg, session, config, court_settings, ["camera"]) if not p.fresh
    ]
    assert stale == ["camera"]
