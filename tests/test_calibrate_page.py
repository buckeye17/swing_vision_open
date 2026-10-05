"""Calibrate page: rendering, handle drags, confirm → camera stage, media routes."""

from __future__ import annotations

import pytest

from swingvision import services
from swingvision.app.main import create_app
from swingvision.court import calibration as calib
from swingvision.pipeline.runner import run
from swingvision.pipeline.stages import default_registry
from tests.conftest import COURT_CAMERA
from tests.synth_court import keypoint_error

pytestmark = pytest.mark.ffmpeg


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(scope="module")
def calibrate(app):
    from swingvision.app.pages import calibrate

    return calibrate


@pytest.fixture
def detected(settings, court_video):
    settings.processing.calibration_window_s = 1.0
    settings.processing.calibration_frames_per_window = 3
    session = services.create_session(settings, court_video)
    assert run(default_registry(), session, settings, targets=["court_auto"]).status == "done"
    return session


def _texts(component) -> str:
    return str(component.to_plotly_json())


def test_layouts_render(app, calibrate, detected):
    from swingvision.app.pages import session_review

    sid = detected.load_config().id
    page = calibrate.layout(session_id=sid)
    assert "cal-graph" in _texts(page)
    review = session_review.layout(session_id=sid)
    text = _texts(review)
    assert "review-overlay" in text and "Auto-detected, not reviewed" in text
    assert "not found" in _texts(calibrate.layout(session_id="nope")).lower()


def test_figure_has_a_handle_per_visible_keypoint(calibrate, detected):
    auto = calib.load(detected.court_auto_path)
    st = calibrate._state_from(auto, detected.load_config().id, "auto")
    fig = calibrate.build_figure(st)
    handles = calibrate._handles(st)
    assert len(fig.layout.shapes) == len(handles) >= 10
    assert fig.layout.images[0].source.endswith("/court_bg.jpg")
    # Viewing a drift window shows its image, without handles.
    win = calibrate.build_figure(st, "0")
    assert win.layout.images[0].source.endswith("/court_w00.jpg") and not win.layout.shapes


def test_drag_pins_a_point_and_moves_the_camera(calibrate, detected):
    auto = calib.load(detected.court_auto_path)
    st = calibrate._state_from(auto, detected.load_config().id, "auto")
    name, x, y, _ = calibrate._handles(st)[0]
    relayout = {"shapes[0].xanchor": x + 12, "shapes[0].yanchor": y}
    new = calibrate.drag_update(st, relayout)
    assert new["origin"] == "edited" and new["user_points"][name] == pytest.approx([x + 12, y])
    moved = {h[0]: h for h in calibrate._handles(new)}
    assert moved[name][3] is True  # pinned
    assert calibrate.drag_update(st, {"xaxis.range[0]": 5}) is None  # zoom, not a drag


def test_confirm_then_camera_stage_uses_user_calibration(settings, calibrate, detected):
    auto = calib.load(detected.court_auto_path)
    st = calibrate._state_from(auto, detected.load_config().id, "auto")
    cal = calibrate.confirmed_calibration(st, auto.created_at.isoformat())
    calib.save(detected.court_user_path, cal)
    result = run(default_registry(), detected, settings, targets=["camera"])
    assert result.status == "done"
    final = calib.load(detected.calibration_path)
    assert final.confirmed_by == "user" and final.source == "user"
    assert keypoint_error(COURT_CAMERA, calib.to_camera(final.camera)) < 1.0
    assert len(final.drift) == 3


def test_media_routes_for_calibration_images(app, detected):
    sid = detected.load_config().id
    client = app.server.test_client()
    bg = client.get(f"/media/{sid}/court_bg.jpg")
    assert bg.status_code == 200 and bg.mimetype == "image/jpeg"
    assert client.get(f"/media/{sid}/court_w00.jpg").status_code == 200
    assert client.get(f"/media/{sid}/court_w99.jpg").status_code == 404
    assert client.get(f"/media/{sid}/court_w0.jpg").status_code == 404
    assert client.get(f"/media/{sid}/auto.json").status_code == 404
    bg.close()
