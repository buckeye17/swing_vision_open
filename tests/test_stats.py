"""Session statistics, the ``stats`` stage, exports and the Stats page (M7)."""

from __future__ import annotations

import io

import dash
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import pytest

from swingvision import services
from swingvision.analysis import export as ex
from swingvision.analysis import stats as st
from swingvision.app.components import stats_view as sv
from swingvision.court import model
from swingvision.pipeline.runner import run
from swingvision.pipeline.stage import read_manifest
from swingvision.pipeline.stages import default_registry
from swingvision.storage.fsutil import read_json
from swingvision.storage.schemas import PRACTICE, SHOTS
from tests.test_players_stages import players_settings  # noqa: F401  (fixture)


def _table(schema: pa.Schema, rows: list[dict]) -> pa.Table:
    full = [{f.name: r.get(f.name) for f in schema} for r in rows]
    return pa.Table.from_pylist(full, schema=schema)


def _shot(i, stroke, speed, x, y, side=-1, outcome="in", **kw):
    return {
        "shot_id": i, "session_id": "s", "t_contact": 10.0 * i, "stroke_type": stroke,
        "is_serve": stroke == "serve", "side": side, "landing_x": x, "landing_y": y,
        "speed_racket_kmh": speed, "speed_sigma_kmh": 2.0, "outcome": outcome,
        "quality_flags": [], **kw,
    }  # fmt: skip


SHOT_ROWS = [
    _shot(0, "serve", 150.0, 1.0, 5.0),
    _shot(1, "serve", 140.0, -1.0, -5.5, side=1),  # far end: mirrored to (1, 5.5)
    _shot(2, "forehand", 100.0, 0.5, 10.5),
    _shot(3, "forehand", 90.0, 0.0, 0.0, outcome="net"),
    _shot(4, "other", 20.0, 0.2, -3.0, outcome="own_side"),  # a dribble: not a shot
    _shot(5, "forehand", 95.0, 0.0, 9.0, quality_flags=["speed_uncertain"]),
]


def test_records_and_summaries():
    recs = st.build_records(_table(SHOTS, SHOT_ROWS), None)
    assert [r["shot_id"] for r in recs] == [0, 1, 2, 3, 5]
    far = recs[1]
    assert (far["rel_x"], far["rel_y"]) == (1.0, 5.5)
    s = st.summarize_shots(recs)
    assert s["n"] == 5 and s["n_called"] == 5 and s["n_in"] == 4 and s["n_net"] == 1
    assert s["n_speed"] == 4  # the uncertain speed is left out
    groups = st.by_group(recs)
    assert list(groups) == ["serve", "forehand"]
    serve = groups["serve"]
    assert serve["speed_median"] == pytest.approx(145.0)
    # Serves are measured against the service line, groundstrokes against the baseline.
    assert serve["depth_mean"] == pytest.approx(model.SERVICE_LINE_FROM_NET - 5.25)
    fh = groups["forehand"]
    assert fh["depth_mean"] == pytest.approx(model.HALF_LENGTH - 9.75)
    assert fh["deep_pct"] == pytest.approx(0.5)  # 10.5 m is deep, 9.0 m isn't


def test_practice_rows_override_shots():
    practice = _table(
        PRACTICE,
        [
            # Shot 0 with a user-placed landing that turns it out.
            {"segment_id": 0, "shot_id": 0, "t_contact": 0.0, "side": -1, "shot_kind": "serve",
             "stroke_type": "serve", "landing_x": 1.0, "landing_y": 7.0, "rel_x": 1.0,
             "rel_y": 7.0, "outcome": "out_long", "excluded": False, "flags": []},
            # Shot 2 marked "not a practice shot".
            {"segment_id": 1, "shot_id": 2, "t_contact": 20.0, "side": -1,
             "shot_kind": "groundstroke", "stroke_type": "forehand", "outcome": "in",
             "excluded": True, "flags": []},
            # An unseen contact: a landing, no speed.
            {"segment_id": 2, "shot_id": None, "t_contact": 30.0, "side": -1,
             "shot_kind": "serve", "stroke_type": None, "landing_x": -1.0, "landing_y": 4.0,
             "rel_x": -1.0, "rel_y": 4.0, "outcome": "in", "excluded": False, "flags": []},
        ],
    )  # fmt: skip
    recs = st.build_records(_table(SHOTS, SHOT_ROWS), practice)
    assert [r["segment_id"] for r in recs] == [0, 2]
    assert recs[0]["outcome"] == "out_long" and recs[0]["speed_kmh"] == 150.0
    assert recs[1]["group"] == "serve" and not recs[1]["contact_seen"]
    assert len(st.build_records(_table(SHOTS, SHOT_ROWS), practice, include_excluded=True)) == 3


def test_figures_and_filters():
    recs = st.build_records(_table(SHOTS, SHOT_ROWS), None)
    assert [r["shot_id"] for r in sv.filtered(recs, ["forehand"], "all")] == [2, 3, 5]
    assert [r["shot_id"] for r in sv.filtered(recs, None, "far")] == [1]
    assert len(sv.speed_figure(recs).data) == 2  # a box per stroke
    assert len(sv.depth_figure(recs).data) == 2
    assert len(sv.landing_figure(recs, "dots").data) == 2
    assert len(sv.landing_figure(recs, "heat").data) == 2  # density + the shots
    sv.strokes_table(recs, {})
    sv.speed_figure([])  # empty states render too
    sv.landing_figure([], "heat")


def test_csv_flattens_lists():
    t = _table(SHOTS, SHOT_ROWS[:1])
    t = t.set_column(t.schema.get_field_index("quality_flags"), "quality_flags",
                     pa.array([["a", "b"]], pa.list_(pa.string())))  # fmt: skip
    back = pa_csv.read_csv(io.BytesIO(ex.to_csv(t)))
    assert back.column("quality_flags").to_pylist() == ["a|b"]


@pytest.mark.ffmpeg
def test_stats_stage_export_and_page(players_settings, ball_video):  # noqa: F811
    path, _px, _bounces = ball_video
    s = players_settings
    s.processing.ball_detector = "motion"
    session = services.create_session(s, path)
    config = session.load_config()
    result = run(default_registry(), session, s, targets=["stats"])
    assert result.status == "done", result.message
    assert result.ran[-1] == "stats"
    stats = read_json(session.stats_path)
    assert stats["version"] == st.STATS_VERSION
    assert read_manifest(session, "stats")["extra"]["n_shots"] == stats["shots"]["n"]
    # An edit reruns practice_eval and stats in-process.
    services.edit_practice_shot(s, config.id, 1.0, exclude=True)
    assert services.refresh_practice(s, config.id) == ("ran", None)

    shots = pq.read_table(io.BytesIO(ex.export_bytes(session, "shots", "parquet")))
    assert "practice_outcome" in shots.column_names
    assert shots.schema.field("speed_racket_kmh").metadata[b"unit"] == b"km/h"
    for what in ex.EXPORTS:
        assert ex.export_bytes(session, what, "csv")
    with pytest.raises(ex.ExportError):
        ex.export_bytes(session, "pose", "csv")

    from swingvision.app.main import create_app

    app = create_app()
    from swingvision.app.pages import stats as stats_page

    page = next(
        p for p in dash.page_registry.values() if p.get("path_template") == "/stats/<session_id>"
    )
    page["layout"](session_id=config.id)
    page["layout"](session_id="nope")
    out = stats_page._render({"session": config.id}, [], "all", False, "heat")
    assert len(out) == 12
    assert session.stats_records_path.exists()
    client = app.server.test_client()
    r = client.get(f"/export/{config.id}/shots.csv")
    assert r.status_code == 200 and b"shot_id" in r.data
    assert client.get(f"/export/{config.id}/nope.csv").status_code == 404
