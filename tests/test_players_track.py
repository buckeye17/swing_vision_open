"""Single-player tracking on synthetic detections (PLAN.md §7.2, §12)."""

from __future__ import annotations

import numpy as np
import pytest

from swingvision.court import model
from swingvision.players.detect import roi_crop
from swingvision.players.track import TrackParams, detection_geometry, select_chain, track_players
from tests.synth_court import make_camera
from tests.synth_players import DetectionBuilder, calibration_for, person_box

CAM = make_camera(width=3840, height=2160, hfov_deg=80.0, height_m=3.2, behind_m=6.0)
MACHINE = (1.0, 10.0)


def player_path(t: float) -> tuple[float, float]:
    """Near-half figure eight for a minute, walk round the net post, then the far half."""
    if t < 60:
        return 3.0 * np.sin(0.4 * t), -10.5 + 1.5 * np.sin(0.8 * t)
    if t < 70:
        s = (t - 60) / 10
        return 6.2, -10.5 + 21.5 * s
    return -2.5 * np.sin(0.5 * (t - 70)), 11.0 + 1.0 * np.cos(0.5 * (t - 70))


def scenario(seed: int = 0):
    b = DetectionBuilder(CAM, duration_s=120.0, seed=seed)
    gaps = [(50.0, 50.4), (80.0, 83.0)]
    me = b.add(
        player_path,
        conf=0.85,
        present=lambda t: not any(a <= t < z for a, z in gaps),
    )
    spectator = b.add(lambda t: (2.0, model.HALF_LENGTH + 9.0), conf=0.6)
    neighbor = b.add(lambda t: (model.HALF_DOUBLES + 5.5, -5.0 + 2 * np.sin(0.3 * t)), conf=0.7)
    machine = b.add(lambda t: MACHINE, height=0.9, conf=0.5, jitter_px=0.5)
    ghost = b.add(lambda t: (-4.0, 2.0), conf=0.3, present=lambda t: 20 <= t < 21)
    det, frames, _ = b.tables()
    return det, frames, {"me": me, "spectator": spectator, "neighbor": neighbor,
                         "machine": machine, "ghost": ghost}  # fmt: skip


def _roles_by_entity(table, b_rows_per_entity, det):
    """Map each synthetic entity to the roles its detections received (rows matched by box)."""
    key = lambda tb: list(  # noqa: E731
        zip(tb.column("frame").to_pylist(), np.round(tb.column("x0").to_numpy(), 2), strict=True)
    )
    role_of = dict(zip(key(table), table.column("role").to_pylist(), strict=True))
    det_keys = key(det)
    out = {}
    for name, rows in b_rows_per_entity.items():
        out[name] = [role_of[det_keys[i]] for i in rows]
    return out


def test_geometry_recovers_foot_position_and_height():
    det, _, _ = scenario()
    geo = detection_geometry(det, calibration_for(CAM), CAM.width, CAM.height)
    # The machine is a fixed point: its court position and height come back exactly.
    keys = list(zip(det.column("frame").to_pylist(), det.column("conf").to_pylist(), strict=True))
    rows = [i for i, (_, c) in enumerate(keys) if abs(c - 0.5) < 1e-6]
    assert np.allclose(np.median(geo.court_x[rows]), MACHINE[0], atol=0.05)
    assert np.allclose(np.median(geo.court_y[rows]), MACHINE[1], atol=0.15)
    assert np.nanmedian(geo.height_m[rows]) == pytest.approx(0.9, abs=0.1)
    # Far-court positions are less certain along the court than near ones.
    far = geo.sigma_y[rows].mean()
    near_rows = [i for i, (_, c) in enumerate(keys) if abs(c - 0.85) < 1e-6][:100]
    assert far > 2 * geo.sigma_y[near_rows].mean()  # despite 2x smaller boxes


def test_single_player_is_tracked_and_others_are_not():
    det, _, ent = scenario()
    table, summary = track_players(
        det, calibration_for(CAM), CAM.width, CAM.height, 1 / 15, TrackParams(), auto_machine=True
    )
    roles = _roles_by_entity(table, ent, det)
    me = np.array(roles["me"]) == "me"
    assert me.mean() > 0.99, f"player tracked in {me.mean():.1%} of detections"
    for other in ("spectator", "neighbor", "machine", "ghost"):
        assert "me" not in roles[other], other
    assert set(roles["spectator"]) == {"outside"}
    assert set(roles["neighbor"]) == {"outside"}
    assert roles["machine"].count("static") > 0.9 * len(roles["machine"])
    # The ball machine is found as a static object and adopted as the machine.
    assert summary.static_objects, "machine not found"
    obj = summary.static_objects[0]
    assert np.hypot(obj.x - MACHINE[0], obj.y - MACHINE[1]) < 0.3
    assert obj.height_m == pytest.approx(0.9, abs=0.15)
    assert summary.machine is not None and summary.machine.source == "auto"
    # One chain covers the whole session, including the walk round the net and both gaps.
    assert summary.me_detections == int(me.sum())


def test_user_placed_machine_is_never_me():
    det, _, ent = scenario()
    table, summary = track_players(
        det, calibration_for(CAM), CAM.width, CAM.height, 1 / 15, TrackParams(),
        machine_xy=[MACHINE[0], MACHINE[1]],
    )  # fmt: skip
    assert summary.machine.source == "user"
    roles = _roles_by_entity(table, ent, det)
    assert "me" not in roles["machine"]


def test_chain_prefers_continuity_over_a_brief_stronger_detection():
    """A second person appears for 2 s while the player is visible: the chain stays put."""
    b = DetectionBuilder(CAM, duration_s=30.0, seed=1)
    me = b.add(lambda t: (0.5 * np.sin(t), -11.0), conf=0.6)
    intruder = b.add(lambda t: (-3.0, -4.0), conf=0.95, present=lambda t: 10 <= t < 12)
    det, _, _ = b.tables()
    table, _ = track_players(
        det, calibration_for(CAM), CAM.width, CAM.height, 1 / 15, TrackParams()
    )
    roles = _roles_by_entity(table, {"me": me, "intruder": intruder}, det)
    assert set(roles["me"]) == {"me"}
    assert set(roles["intruder"]) == {"other"}


def _tr(i, t0, t1, xy0, xy1=None, conf=0.9, rate=15.0):
    from swingvision.players.track import Tracklet

    ts = np.arange(t0, t1 + 1e-9, 1 / rate)
    a, b = np.array(xy0, float), np.array(xy1 if xy1 is not None else xy0, float)
    xys = a + (b - a) * ((ts - t0) / max(t1 - t0, 1e-9))[:, None]
    return Tracklet(i, np.arange(len(ts)), t0, t1, a, b, 0.1, 0.1, (a + b) / 2, 0.0, 1.8,
                    conf * len(ts), ts, xys, np.full(len(ts), conf))  # fmt: skip


def test_select_chain_restart_penalty():
    """Two disconnected tracklets far apart in time and space are both kept (one restart)."""
    ids = lambda chain: [i for i, _ in chain]  # noqa: E731
    p = TrackParams()
    assert ids(select_chain([_tr(0, 0, 10, (0, -10)), _tr(1, 100, 110, (0, 10))], 1 / 15, p)) == [
        0,
        1,
    ]
    # A tiny tracklet isn't worth a restart.
    assert ids(select_chain([_tr(0, 0, 10, (0, -10)), _tr(1, 100, 100.1, (0, 10))], 1 / 15, p)) == [
        0
    ]


def test_select_chain_continues_into_a_tracklet_that_started_earlier():
    """A duplicate track starts on the player 1 s before the original ends, then takes over."""
    first = _tr(0, 0, 20, (0, -10), (2, -10))
    takeover = _tr(1, 19, 40, (1.9, -10), (-3, -5))  # walks off after the junction
    elsewhere = _tr(2, 19, 21, (-4, 8))  # someone else, briefly
    chain = select_chain([first, takeover, elsewhere], 1 / 15, TrackParams())
    assert [i for i, _ in chain] == [0, 1]
    assert chain[1][1] == pytest.approx(20.0)  # the takeover counts from the junction on


def test_roi_crop_covers_court_and_snaps():
    crop = roi_crop([CAM], CAM.width, CAM.height, 6.0, 3.5)
    assert crop.x0 % 64 == 0 and crop.y0 % 64 == 0
    assert crop.x1 == CAM.width or crop.x1 % 64 == 0
    # A player standing at the far ROI corner is inside the crop.
    x0, y0, x1, y1 = person_box(CAM, model.HALF_DOUBLES + 3.0, model.HALF_LENGTH + 5.0)
    assert crop.x0 <= x0 and crop.y0 <= y0 and x1 <= crop.x1 and y1 <= crop.y1
    # Small camera changes don't move it.
    nudged = CAM.with_params(CAM.params() + np.r_[0, 0, 0, 0.3, 0.3, 1e-5, 1e-5, 0, 0, 0, 0])
    assert roi_crop([nudged], CAM.width, CAM.height, 6.0, 3.5) == crop


def test_frames_without_the_court_view_are_ignored():
    """Camera being set up: boxes in frames that don't show the court never become "me"."""
    det, _, ent = scenario()
    frames = det.column("frame").to_numpy()
    setup = np.unique(frames[det.column("t_s").to_numpy() < 10.0])
    table, _ = track_players(
        det, calibration_for(CAM), CAM.width, CAM.height, 1 / 15, TrackParams(),
        ignore_frames=setup,
    )  # fmt: skip
    roles = np.array(table.column("role").to_pylist())
    in_setup = np.isin(table.column("frame").to_numpy(), setup)
    assert set(roles[in_setup]) == {"outside"}
    me_roles = np.array(_roles_by_entity(table, ent, det)["me"])
    me_t = det.column("t_s").to_numpy()[ent["me"]]
    assert (me_roles[me_t >= 10.0] == "me").mean() > 0.99


def test_fill_nearest():
    from swingvision.pipeline.stages.players import fill_nearest

    assert fill_nearest([None, 1, None, None, 2, None]) == [1, 1, 1, 1, 2, 2]
    assert fill_nearest([None, None]) == [None, None]
