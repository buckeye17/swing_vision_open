"""pass1_detect (ball sweep) → ball_refine → ball_track → events on a rendered video."""

from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
import pytest

from swingvision import services
from swingvision.pipeline.runner import plan, run
from swingvision.pipeline.stages import default_registry
from swingvision.storage import tables
from swingvision.storage.schemas import BALL_TRACK, EVENTS
from tests.conftest import BALL_FPS
from tests.test_players_stages import players_settings  # noqa: F401  (fixture)

pytestmark = pytest.mark.ffmpeg


def _recall(track, px, tol=3.0):
    f = track.column("frame").to_numpy()
    x, y = track.column("x").to_numpy(), track.column("y").to_numpy()
    truth = np.isfinite(px).all(axis=1)
    good = sum(
        1
        for fi, xi, yi in zip(f, x, y, strict=True)
        if fi < len(px) and truth[fi] and np.hypot(xi - px[fi, 0], yi - px[fi, 1]) <= tol
    )
    return good / truth.sum(), good / max(len(f), 1)


@pytest.mark.parametrize("sweep_hz", [None, 15.0])
def test_ball_pipeline(players_settings, ball_video, sweep_hz):  # noqa: F811
    path, px, bounces = ball_video
    s = players_settings
    s.processing.ball_detector = "motion"
    s.processing.ball_sweep_hz = sweep_hz
    session = services.create_session(s, path)
    reg = default_registry()
    result = run(reg, session, s, targets=["events"])
    assert result.status == "done", result.message
    assert {"pass1_detect", "ball_refine", "ball_track", "events"} <= set(result.ran)

    track = tables.read_table(session.ball_track_path)
    assert track.schema.equals(BALL_TRACK, check_metadata=False)
    recall, precision = _recall(track, px)
    assert recall > 0.85, recall
    assert precision > 0.9, precision

    events = tables.read_table(session.events_path)
    assert events.schema.equals(EVENTS, check_metadata=False)
    b = events.filter(pc.equal(events.column("kind"), "bounce"))
    assert b.num_rows >= 1
    assert np.min(np.abs(b.column("t_s").to_numpy() - bounces[0])) < 2 / BALL_FPS

    if sweep_hz:
        refine = tables.read_table(session.ball_refine_frames_path)
        sweep = tables.read_table(session.ball_sweep_frames_path)
        # The sweep looked at every 4th frame; refining filled in windows around moments.
        assert sweep.num_rows < 0.3 * len(px)
        assert refine.num_rows > 0

    # Everything is fresh now; changing only the ball detector keeps the person boxes.
    config = session.load_config()
    assert all(p.fresh for p in plan(reg, session, config, s, ["events"]))
