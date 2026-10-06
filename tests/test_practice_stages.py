"""``segments`` → ``practice_eval`` through the pipeline, and the app-side services that
rerun them after target, practice-type and shot edits."""

from __future__ import annotations

import pytest

from swingvision import services
from swingvision.pipeline.runner import plan, run
from swingvision.pipeline.stage import read_manifest
from swingvision.pipeline.stages import default_registry
from swingvision.storage import tables
from swingvision.storage.schemas import PRACTICE, SEGMENTS, Target
from tests.test_players_stages import players_settings  # noqa: F401  (fixture)

pytestmark = pytest.mark.ffmpeg


def test_practice_stages(players_settings, ball_video):  # noqa: F811
    path, _px, _bounces = ball_video
    s = players_settings
    s.processing.ball_detector = "motion"
    session = services.create_session(
        s,
        path,
        practice_targets=[
            Target(id="deep", name="Deep", shape="rect", x0=-4.1, y0=6.4, x1=4.1, y1=11.9)
        ],
    )
    config = session.load_config()
    assert config.practice.targets[0].id == "deep"
    reg = default_registry()
    result = run(reg, session, s, targets=["practice_eval"])
    assert result.status == "done", result.message
    assert {"segments", "practice_eval"} <= set(result.ran)
    seg = tables.read_table(session.segments_path)
    assert seg.schema.equals(SEGMENTS, check_metadata=False)
    prac = tables.read_table(session.practice_path)
    assert prac.schema.equals(PRACTICE, check_metadata=False)
    assert prac.num_rows == sum(k == "practice_shot" for k in seg.column("kind").to_pylist())
    assert read_manifest(session, "practice_eval")["extra"]["n"] == prac.num_rows

    # Up to date: nothing to do.
    assert services.refresh_practice(s, config.id) == ("ran", None)
    assert all(p.fresh for p in plan(reg, session, session.load_config(), s, ["practice_eval"]))

    # A new target reruns only practice_eval, in-process.
    before = read_manifest(session, "segments")["finished_at"]
    services.set_practice(
        s,
        config.id,
        targets=[Target(id="c", name="C", shape="circle", cx=0.0, cy=8.0, r=2.0)],
    )
    stale = [
        p.stage.name
        for p in plan(reg, session, session.load_config(), s, ["practice_eval"])
        if not p.fresh
    ]
    assert stale == ["practice_eval"]
    assert services.refresh_practice(s, config.id) == ("ran", None)
    assert read_manifest(session, "segments")["finished_at"] == before

    # A shot edit does the same; a new practice type reruns segmentation too.
    services.edit_practice_shot(s, config.id, 1.0, exclude=True)
    stale = [
        p.stage.name
        for p in plan(reg, session, session.load_config(), s, ["practice_eval"])
        if not p.fresh
    ]
    assert stale == ["practice_eval"]
    services.set_practice(s, config.id, submode="serve")
    stale = [
        p.stage.name
        for p in plan(reg, session, session.load_config(), s, ["practice_eval"])
        if not p.fresh
    ]
    assert stale == ["segments", "practice_eval"]
    assert services.refresh_practice(s, config.id) == ("ran", None)
    lib_row = services.open_library(s).get_session(config.id)
    assert lib_row["submode"] == "serve"


def test_target_sets(settings):
    t = [Target(id="a", name="A", shape="rect", x0=0, y0=0, x1=1, y1=1)]
    first = services.save_target_set(settings, "Serve targets", t)
    assert [x.name for x in services.list_target_sets(settings)] == ["Serve targets"]
    # Same name (any case) replaces it.
    again = services.save_target_set(settings, "serve TARGETS", t * 2)
    assert again.id == first.id and len(again.targets) == 2
    assert len(services.list_target_sets(settings)) == 1
    services.delete_target_set(settings, first.id)
    assert services.list_target_sets(settings) == []
    with pytest.raises(ValueError):
        services.save_target_set(settings, "  ", t)
