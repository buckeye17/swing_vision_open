"""Court model, camera model, homography and uncertainty: pure math (PLAN.md §12)."""

from __future__ import annotations

import numpy as np
import pytest

from swingvision.court import model
from swingvision.court.camera import (
    Camera,
    FitOptions,
    LineObs,
    PointObs,
    camera_from_homography,
    fit_camera,
    rodrigues,
    rotation_to_rvec,
)
from swingvision.court.homography import apply_h, fit_homography, ground_sigma
from tests.synth_court import make_camera


def test_court_dimensions():
    kp = model.GROUND_KEYPOINTS
    width = kp["far_doubles_right"][0] - kp["far_doubles_left"][0]
    assert width == pytest.approx(model.DOUBLES_WIDTH - model.LINE_WIDTH)
    length = kp["far_singles_left"][1] - kp["near_singles_left"][1]
    assert length == pytest.approx(model.COURT_LENGTH - model.LINE_WIDTH)
    assert kp["far_service_center"][1] == pytest.approx(6.40 - 0.025)
    assert len(model.GROUND_KEYPOINTS) == 14 and len(model.NET_KEYPOINTS) == 3
    assert all(z > 0.9 for _, _, z in model.NET_KEYPOINTS.values())


def test_mirror_and_zones():
    xy = np.array([[1.0, 5.0], [-2.0, -3.0]])
    assert np.allclose(model.mirror(model.mirror(xy)), xy)
    assert np.allclose(model.mirror(xy)[0], [-1.0, -5.0])
    assert model.zone_at(-1.0, 3.0) == "deuce_box"
    assert model.zone_at(1.0, 3.0) == "ad_box"
    assert model.zone_at(1.0, -3.0) == "near:deuce_box"
    assert model.zone_at(0.0, 9.0) == "backcourt"
    assert model.zone_at(5.0, 9.0) == "right_alley"
    assert model.zone_at(0.0, 13.0) is None
    assert model.in_court(np.array([4.0, 4.2]), np.array([0.0, 0.0])).tolist() == [True, False]


def test_rodrigues_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = rng.normal(size=3)
        r *= rng.uniform(0.01, 3.1) / np.linalg.norm(r)
        assert np.allclose(rotation_to_rvec(rodrigues(r)), r, atol=1e-8)
    assert np.allclose(rodrigues(np.zeros(3)), np.eye(3))


@pytest.mark.parametrize(("k1", "k2"), [(0.0, 0.0), (-0.08, 0.0), (0.05, -0.03), (-0.1, 0.04)])
def test_distortion_roundtrip(k1, k2):
    cam = make_camera(k1=k1, k2=k2, principal_offset=(30, -200))
    rng = np.random.default_rng(0)
    px = rng.uniform([0, 0], [cam.width, cam.height], size=(500, 2))
    assert np.allclose(cam.distort(cam.undistort(px)), px, atol=1e-6)


def test_project_and_back_to_ground():
    cam = make_camera()
    X = np.array([[0, 0, 0], [3.0, 8.0, 0], [-4.0, -10.0, 0], [5.0, 11.0, 0]])
    G = cam.image_to_ground(cam.project(X))
    assert np.allclose(G, X, atol=1e-6)
    # Points at height z come back on the plane z.
    Xh = np.array([[1.0, 2.0, 1.5]])
    assert np.allclose(cam.image_to_ground(cam.project(Xh), z=1.5), Xh, atol=1e-6)
    # Above the horizon there is no ground intersection.
    assert np.isnan(cam.image_to_ground(np.array([[cam.width / 2, -300.0]]))).all()


def test_camera_from_homography_recovers_pinhole():
    cam = make_camera(k1=0.0, k2=0.0)
    est = camera_from_homography(cam.homography(), cam.width, cam.height)
    assert est is not None
    assert est.f == pytest.approx(cam.f, rel=1e-6)
    assert np.allclose(est.center, cam.center, atol=1e-6)


def test_camera_from_homography_rejects_mirrored_court():
    cam = make_camera(k1=0.0, k2=0.0)
    flip = np.diag([-1.0, 1.0, 1.0])  # x → -x: a left-handed court
    assert camera_from_homography(cam.homography() @ flip, cam.width, cam.height) is None


def test_fit_homography_exact():
    H = np.array([[1.2, 0.1, 30], [0.05, 0.9, -20], [1e-4, 2e-4, 1.0]])
    src = np.random.default_rng(2).uniform(-10, 10, size=(12, 2))
    assert np.allclose(fit_homography(src, apply_h(H, src)), H, atol=1e-8)


def _point_obs(cam, noise=0.0, seed=0):
    X = model.keypoint_array()
    px = cam.project(X) + np.random.default_rng(seed).normal(0, noise, (len(X), 2))
    return PointObs(X, px, np.ones(len(X)))


def test_fit_camera_from_points_with_net_recovers_camera():
    """PnP with non-planar net points: focal length and pose from 17 noisy points."""
    true = make_camera(k1=0.0, k2=0.0, hfov_deg=60)
    init = make_camera(k1=0.0, k2=0.0, hfov_deg=70, height_m=4.0, behind_m=4.0, offset_x=1.0)
    fit = fit_camera(init, points=_point_obs(true, 0.3), options=FitOptions(fit_k1=False))
    assert fit.camera.f == pytest.approx(true.f, rel=0.01)
    assert np.allclose(fit.camera.center, true.center, atol=0.05)
    assert fit.rms_points_px < 0.6


def test_fit_camera_from_lines_recovers_distortion_and_center():
    true = make_camera(k1=0.04, k2=-0.02, principal_offset=(-20, -150))
    a, b, obs = [], [], []
    for line in model.COURT_LINES + model.NET_LINES:
        pts = line.samples(0.1)
        px = true.project(pts)
        m = (px[:, 0] > 0) & (px[:, 0] < true.width) & (px[:, 1] > 0) & (px[:, 1] < true.height)
        a += [line.p0] * int(m.sum())
        b += [line.p1] * int(m.sum())
        obs.append(px[m])
    obs = np.concatenate(obs) + np.random.default_rng(3).normal(0, 0.3, (len(a), 2))
    lines = LineObs(np.array(a), np.array(b), obs, np.ones(len(a)))
    init = make_camera(k1=0.0, k2=0.0, hfov_deg=75, height_m=3.0)
    fit = fit_camera(
        init, lines=lines, options=FitOptions(fit_k2=True, fit_center=True, loss="linear")
    )
    cam = fit.camera
    assert fit.rms_lines_px < 0.4
    assert cam.f == pytest.approx(true.f, rel=0.01)
    # The principal point is only weakly observable from one view (its prior pulls it
    # toward the center); what matters is that the projection is right.
    assert abs(cam.cy - true.cy) < 40
    assert np.allclose(cam.center, true.center, atol=0.1)
    X = model.keypoint_array(list(model.GROUND_KEYPOINTS))
    assert np.abs(cam.project(X) - true.project(X)).max() < 1.0


def test_ground_sigma_grows_with_distance_and_pixel_noise():
    cam = make_camera()
    s = ground_sigma(cam, np.array([[0.0, -11.0], [0.0, 11.0]]), px_sigma=1.0)
    near, far = s
    assert far[1] > 3 * near[1]  # depth uncertainty explodes at the far baseline
    assert far[2] >= far[1] - 1e-12  # major axis ≥ along-court component
    s2 = ground_sigma(cam, np.array([[0.0, 11.0]]), px_sigma=2.0)
    assert s2[0, 2] == pytest.approx(2 * far[2], rel=1e-3)


def test_camera_serialization_roundtrip():
    cam = make_camera(k1=0.03, k2=-0.01, principal_offset=(5, -40))
    back = Camera.from_dict(cam.to_dict())
    X = model.keypoint_array()
    assert np.allclose(back.project(X), cam.project(X))
    assert np.allclose(cam.with_params(cam.params()).project(X), cam.project(X))
