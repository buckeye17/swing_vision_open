"""Court detection and calibration helpers on synthetic renders with known cameras."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from swingvision.court import calibration as calib
from swingvision.court import detect, model
from swingvision.storage.schemas import Calibration, CalibrationMetrics, DriftWindow
from tests.synth_court import keypoint_error, make_camera, render

CAMERAS = {
    # High behind the baseline, wide lens (like the user's 0.6x footage).
    "wide": {},
    # Narrower lens, far baseline at the top edge, off-center principal point.
    "close": {
        "hfov_deg": 57,
        "behind_m": 6.8,
        "height_m": 3.35,
        "look_y": -3.0,
        "k1": 0.05,
        "k2": -0.03,
        "principal_offset": (-23, -284),
    },
    # Low camera near a corner, barrel distortion.
    "corner": {
        "hfov_deg": 70,
        "offset_x": -4.0,
        "behind_m": 3.0,
        "height_m": 2.5,
        "k1": -0.05,
        "k2": 0.0,
    },
}


@pytest.mark.parametrize("name", list(CAMERAS))
def test_detect_recovers_known_camera(name):
    true = make_camera(**CAMERAS[name])
    det = detect.detect_court(render(true))
    assert det.ok, det.message
    cam = det.camera
    assert det.rms_px < 0.5
    assert keypoint_error(true, cam) < 1.0  # px at 1920×1080
    assert cam.f == pytest.approx(true.f, rel=0.01)
    assert np.allclose(cam.center, true.center, atol=0.1)


def test_detect_fails_cleanly_without_a_court():
    rng = np.random.default_rng(0)
    img = rng.integers(60, 90, size=(540, 960, 3), dtype=np.uint8)
    det = detect.detect_court(img)
    assert not det.ok and det.message


def test_snap_to_lines_recovers_from_a_rough_camera():
    """After a rough drag in the editor (same lens, pose a bit off), snapping finds the lines."""
    true = make_camera(**CAMERAS["close"])
    img = render(true)
    rough = make_camera(**{**CAMERAS["close"], "behind_m": 6.6, "height_m": 3.45, "offset_x": 0.45})
    assert keypoint_error(true, rough) > 10
    det = calib.snap_to_lines(img, rough)
    assert det.ok and keypoint_error(true, det.camera) < 1.0


def test_solve_from_points_follows_user_points():
    true = make_camera(k1=0.0, k2=0.0)
    init = make_camera(k1=0.0, k2=0.0, hfov_deg=75, height_m=3.6, offset_x=-0.5)
    names = [
        "near_doubles_left", "near_doubles_right", "near_service_center",
        "far_doubles_left", "far_doubles_right", "far_service_center", "net_center",
    ]  # fmt: skip
    pts = {n: true.project(np.array(model.KEYPOINTS[n])).tolist() for n in names}
    fit = calib.solve_from_points(init, pts)
    assert fit.rms_points_px < 0.5
    assert keypoint_error(true, fit.camera) < 2.0
    # One dragged point moves the camera toward it without wrecking the rest.
    one = calib.solve_from_points(
        true, {"near_doubles_left": (pts["near_doubles_left"][0] + 15, pts["near_doubles_left"][1])}
    )
    moved = one.camera.project(np.array(model.KEYPOINTS["near_doubles_left"]))
    assert abs(moved[0] - pts["near_doubles_left"][0]) > 10


def _cal(rms=1.0, ok=True, coverage=0.8, drift=False, window_rms=None, window_status=None):
    cam = make_camera()
    status = window_status or ("moved" if drift else "ok")
    window = DriftWindow(
        t0_s=0,
        t1_s=1,
        n_frames=3,
        status=status,
        rms_line_px=window_rms,
        camera=calib.to_params(cam) if window_rms is not None else None,
    )
    return Calibration(
        source="auto",
        created_at=datetime.now(UTC),
        ok=ok,
        camera=calib.to_params(cam),
        metrics=CalibrationMetrics(
            rms_line_px=rms, n_line_samples=int(coverage * 1000), n_expected_samples=1000
        ),
        drift=[window],
        drift_detected=status == "moved",
    )


def test_auto_acceptable_rules():
    assert not calib.auto_acceptable(_cal(), None)[0]  # auto-accept off
    assert calib.auto_acceptable(_cal(rms=1.0), 2.0)[0]
    assert not calib.auto_acceptable(_cal(rms=2.5), 2.0)[0]
    assert not calib.auto_acceptable(_cal(ok=False), 2.0)[0]
    assert not calib.auto_acceptable(_cal(coverage=0.1), 2.0)[0]
    assert not calib.auto_acceptable(_cal(drift=True), 2.0)[0]  # moved, no own camera
    # A moved window with its own good fit is fine unattended; a poor one needs review.
    ok, why = calib.auto_acceptable(_cal(drift=True, window_rms=1.2), 2.0)
    assert ok and "moves" in why
    assert not calib.auto_acceptable(_cal(drift=True, window_rms=2.4), 2.0)[0]
    assert not calib.auto_acceptable(_cal(window_status="failed"), 2.0)[0]


def test_calibration_file_roundtrip_and_geometry_key(tmp_path):
    cal = _cal()
    path = tmp_path / "c.json"
    calib.save(path, cal)
    back = calib.load(path)
    assert back is not None and calib.geometry_key(back) == calib.geometry_key(cal)
    cam = calib.to_camera(back.camera)
    moved = calib.with_camera(back, cam.with_params(cam.params() + 1e-3))
    assert calib.geometry_key(moved) != calib.geometry_key(cal)
    assert calib.load(tmp_path / "missing.json") is None


def test_projected_keypoints_mark_undefined_points():
    cam = make_camera(k1=0.3, k2=-0.4)  # folds over not far outside the frame
    kp = calib.projected_keypoints(cam)
    assert set(kp) == set(model.KEYPOINTS)
    assert all(v is None or len(v) == 2 for v in kp.values())


def test_sample_windows():
    wins = calib.sample_windows(1000.0, 300.0, 5)
    assert len(wins) == 4 and all(len(w.times) == 5 for w in wins)
    assert all(w.t0 <= t <= w.t1 for w in wins for t in w.times)
    assert len(calib.sample_windows(10.0, 300.0, 5)[0].times) == 9  # one window → more frames
    assert len(calib.sample_windows(10 * 3600, 300.0, 5)) == calib.MAX_WINDOWS
    assert calib.sample_windows(0, 300, 5) == []


def test_keypoint_shift():
    a = make_camera()
    assert calib.keypoint_shift(a, a) == (0.0, 0.0)
    b = make_camera(offset_x=0.35)
    rms, mx = calib.keypoint_shift(a, b)
    assert 0 < rms <= mx


def test_window_drift_refits_a_remounted_camera():
    """A window filmed from somewhere else entirely (another mount, another zoom) gets its
    own full fit instead of a failed pose-only refinement."""
    main = make_camera(**CAMERAS["wide"])
    remounted = make_camera(**CAMERAS["close"])
    images = {0: render(main, seed=1), 1: render(remounted, seed=2)}
    windows = [
        DriftWindow(index=i, t0_s=100.0 * i, t1_s=100.0 * (i + 1), n_frames=5, status="ok")
        for i in images
    ]
    out = calib.window_drift(main, windows, images.get, drift_px=3.0)
    assert out[0].status == "ok"
    assert out[1].status == "moved" and out[1].rms_line_px < calib.POOR_WINDOW_PX
    assert keypoint_error(calib.to_camera(out[1].camera), remounted) < 2.0


def test_dark_windows_keep_the_last_known_camera():
    """A camera that moved and then went dark stays where it was last seen."""
    main = make_camera(**CAMERAS["wide"])
    moved = make_camera(**{**CAMERAS["wide"], "offset_x": 1.0})
    m, mv = calib.to_params(main), calib.to_params(moved)
    cal = Calibration(
        source="auto",
        created_at=datetime.now(UTC),
        camera=m,
        drift=[
            DriftWindow(t0_s=0, t1_s=10, n_frames=0, status="dark"),  # before any light
            DriftWindow(t0_s=10, t1_s=20, n_frames=5, status="ok", shift_rms_px=0.3, camera=m),
            DriftWindow(t0_s=20, t1_s=30, n_frames=5, status="moved", shift_rms_px=40, camera=mv),
            DriftWindow(t0_s=30, t1_s=40, n_frames=0, status="dark"),
            DriftWindow(t0_s=40, t1_s=50, n_frames=0, status="failed"),
        ],
    )

    def x_of(cam):
        return cam.project(np.array([[0.0, 0.0, 0.0]]))[0, 0]

    assert x_of(calib.camera_at(cal, 5)) == pytest.approx(x_of(main))  # next known: main
    assert x_of(calib.camera_at(cal, 25)) == pytest.approx(x_of(moved))
    assert x_of(calib.camera_at(cal, 35)) == pytest.approx(x_of(moved))  # dark: stays moved
    cams, which = calib.cameras_for_times(cal, np.array([5.0, 15.0, 25.0, 35.0, 45.0, 60.0]))
    assert len(cams) == 2 and which.tolist() == [0, 0, 1, 1, 1, 0]
