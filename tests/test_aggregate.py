"""Multi-session statistics (M7a): selections, the records round trip, pooled summaries,
library v4, the Stats page over a selection, export and CLI."""

from __future__ import annotations

import io
import json
import sqlite3
import time

import numpy as np
import plotly.utils
import pyarrow.csv as pa_csv
import pytest

from swingvision.analysis import aggregate as agg
from swingvision.analysis import export as ex
from swingvision.analysis import stats as st
from swingvision.players import movement as mv
from swingvision.storage import tables
from swingvision.storage.library import MIGRATIONS, Library, clean_tags, now_iso
from swingvision.storage.schemas import STATS_RECORDS
from tests.synth_stats import Spec, make_library, make_session, run_stats_stage

SPECS = [
    Spec("a1", "2026-10-01T21:05:33.000000Z", submode="serve", profile="p1",
         tags=["new racket"], calibration_by="user", seed=1),
    Spec("b2", "2026-10-04T15:00:00Z", submode="serve", profile="p1", tags=["indoor"], seed=2),
    Spec("c3", "2026-10-06T10:00:00Z", submode="self_feed", profile="p2",
         tags=["new racket", "indoor"], seed=3, minutes=6.0),
    Spec("d4", "2026-10-09T18:30:00Z", submode="ball_machine", seed=4, n_shots=25),
]  # fmt: skip


@pytest.fixture(autouse=True)
def _fresh_cache():
    agg.clear_cache()
    yield
    agg.clear_cache()


@pytest.fixture
def lib(tmp_path):
    return make_library(tmp_path / "out", SPECS)


def _session(lib, sid):
    from swingvision.storage.session import Session

    return Session.open(lib.root, lib.get_session(sid)["dir_name"])


def _js(obj) -> str:
    return json.dumps(obj, cls=plotly.utils.PlotlyJSONEncoder, sort_keys=True)


def _all_summaries(data: st.StatsData) -> dict:
    """Every summary the Stats page, ``stats.json`` and the CLI show."""
    out = {
        "session_stats": st.session_stats(data),
        "heatmap": [a.tolist() for a in st.movement_heatmap(data)],
        "trends": {k: st.trend(data, k) for k in st.TREND_KPIS},
    }
    for groups in (None, ["serve"], ["forehand", "backhand"]):
        for end in ("all", "near", "far"):
            rows = st.filter_records(data.records, groups, end)
            out[f"{groups}-{end}"] = {
                "shots": st.summarize_shots(rows),
                "groups": st.by_group(rows),
                "trend": st.trend(data, "speed_median", rows),
            }
    return out


# ---------------------------------------------------------------------------
# One session: both data paths give the same numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excluded", [False, True])
def test_one_session_selection_equals_single_session(lib, excluded):
    for spec in SPECS:
        single = st.load(_session(lib, spec.id), include_excluded=excluded)
        sel = agg.select(lib, agg.SessionFilter(include=(spec.id,)), include_excluded=excluded)
        data = sel.data
        assert [s["session_id"] for s in data.sessions] == [spec.id]
        assert data.records == single.records
        assert data.swings == single.swings
        assert data.movement == single.movement
        assert data.is_practice == single.is_practice
        assert data.duration_s == single.duration_s
        assert _js(_all_summaries(data)) == _js(_all_summaries(single))


def test_one_session_page_renders_identically(settings, tmp_path):
    from swingvision.app.main import create_app

    lib = make_library(settings.output_root, SPECS[:2])
    create_app()
    from swingvision.app.pages import stats as page

    for spec in SPECS[:2]:
        for excluded in (False, True):
            for groups, end in (([], "all"), (["serve"], "near")):
                one = page._render({"session": spec.id}, groups, end, excluded, "dots")
                many = page._render({"query": f"sessions={spec.id}"}, groups, end, excluded,
                                    "dots")  # fmt: skip
                # Notes, KPIs, every chart and table; the count and export only exist for a
                # selection.
                assert _js(one[1:9]) == _js(many[1:9])
                assert one[9] is None and many[9] is not None
    assert lib.schema_version() == len(MIGRATIONS)


# ---------------------------------------------------------------------------
# Many sessions: pooled numbers equal those of the pooled raw records
# ---------------------------------------------------------------------------


def test_pooled_summaries_equal_pooled_raw_records(lib):
    sel = agg.select(lib, agg.SessionFilter())
    data = sel.data
    singles = [st.load(_session(lib, s.id)) for s in sorted(SPECS, key=lambda s: s.recorded)]
    assert [s["session_id"] for s in data.sessions] == [
        s.sessions[0]["session_id"] for s in singles
    ]
    pooled = [r for s in singles for r in s.records]
    assert data.records == pooled  # date-then-time order
    assert st.summarize_shots(data.records) == st.summarize_shots(pooled)
    assert st.by_group(data.records) == st.by_group(pooled)
    assert st.summarize_swings(data.swings) == st.summarize_swings(
        [w for s in singles for w in s.swings]
    )

    # Counts and rates straight from the raw records.
    recs = [r for r in pooled if not r["excluded"]]
    called = [r for r in recs if r["outcome"] in st.CALLED]
    s = st.summarize_shots(data.records)
    assert s["n"] == len(recs)
    assert s["in_pct"] == pytest.approx(sum(r["outcome"] == "in" for r in called) / len(called))
    speeds = [r["speed_kmh"] for r in recs if r["speed_ok"]]
    assert s["speed_median"] == pytest.approx(float(np.median(speeds)))
    c = sel.counts()
    assert c["sessions"] == 4 and c["shots"] == len(recs)
    assert c["serves"] == sum(r["group"] == "serve" for r in recs)

    # Movement: time and distance add up, averages pool over the samples behind them.
    m = st.summarize_movement(data)
    raws = []
    for spec in SPECS:
        sess = _session(lib, spec.id)
        move, frames = (
            tables.read_table(sess.movement_path),
            tables.read_table(sess.pass1_frames_path),
        )
        raws.append((move, mv.summarize(move, frames, 4.0, mv.MovementParams(), 0.6)))
    assert m["n_sessions"] == 4
    assert m["distance_m"] == pytest.approx(sum(r["distance_m"] for _, r in raws), abs=1e-3)
    assert m["tracked_s"] == pytest.approx(sum(r["tracked_s"] for _, r in raws), abs=1e-3)
    assert m["max_speed_mps"] == pytest.approx(max(r["max_speed_mps"] for _, r in raws), abs=1e-4)
    speed = np.concatenate([mt.column("speed").to_numpy() for mt, _ in raws]).astype(np.float64)
    fast = speed[speed >= mv.MovementParams().moving_speed]
    assert m["mean_moving_speed_mps"] == pytest.approx(float(fast.mean()), abs=1e-4)
    y = np.concatenate([mt.column("y").to_numpy() for mt, _ in raws])
    assert m["near_half_frac"] == pytest.approx(float(np.mean(y < 0)), abs=1e-4)
    lit = sum(r["lit_frames"] for _, r in raws)
    tracked = sum(r["coverage"] * r["lit_frames"] for _, r in raws)
    assert m["coverage"] == pytest.approx(tracked / lit, abs=1e-4)
    _xc, _yc, H = st.movement_heatmap(data)
    np.testing.assert_allclose(H, sum(mv.heatmap(mt, fold=True)[2] for mt, _ in raws))

    # Per 5 minutes becomes per session.
    dist = st.distance_series(data)
    assert dist["per"] == "session" and dist["session_id"] == [x.sessions[0]["session_id"]
                                                               for x in singles]  # fmt: skip
    assert sum(dist["distance_m"]) == pytest.approx(m["distance_m"], abs=0.5)

    # Trends: one point per session, equal to that session's own number, with an interval.
    pts = st.trend(data, "in_pct")
    assert [p["session_id"] for p in pts] == [x.sessions[0]["session_id"] for x in singles]
    for p, single in zip(pts, singles, strict=True):
        own = st.summarize_shots(single.records)
        assert p["value"] == own["in_pct"] and p["n"] == own["n_called"]
        assert p["lo"] <= p["value"] <= p["hi"]


def test_intervals():
    lo, hi = st.wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    assert st.wilson(0, 0) == (None, None)
    v = list(range(1, 101))
    lo, hi = st.median_ci(v)
    assert lo <= 50.5 <= hi and hi - lo < 25
    assert st.median_ci([1.0, 2.0]) == (None, None)
    lo, hi = st.mean_ci([1.0, 2.0, 3.0])
    assert lo < 2.0 < hi


# ---------------------------------------------------------------------------
# Filters select exactly the expected sessions
# ---------------------------------------------------------------------------


def _ids(lib, **kw) -> list[str]:
    return [r["id"] for r in agg.resolve(lib, agg.SessionFilter.make(**kw))]


def test_filters_select_exactly(lib):
    lib.add_session(
        id="m5", name="Match", dir_name="2026-10-07_m5", created_at=now_iso(), mode="match",
        submode=None, source_path="C:/m.mp4", source_hash="m", duration_s=60.0,
        recorded_on="2026-10-07T12:00:00",
    )  # fmt: skip
    make_session(lib, Spec("u6", "2026-10-10T09:00:00Z", submode="serve", seed=6), False)
    assert _ids(lib) == ["a1", "b2", "c3", "m5", "d4", "u6"]  # recording order
    assert _ids(lib, date_from="2026-10-04", date_to="2026-10-07") == ["b2", "c3", "m5"]
    assert _ids(lib, date_to="2026-10-03") == ["a1"]
    assert _ids(lib, modes=["match"]) == ["m5"]
    assert _ids(lib, modes="practice") == ["a1", "b2", "c3", "d4", "u6"]
    assert _ids(lib, practice_types=["serve"]) == ["a1", "b2", "u6"]
    assert _ids(lib, practice_types="self_feed,ball_machine") == ["c3", "d4"]
    assert _ids(lib, profiles=["p1"]) == ["a1", "b2"]
    assert _ids(lib, tags=["new racket"]) == ["a1", "c3"]
    assert _ids(lib, tags=["New Racket", "indoor"]) == ["c3"]  # all of them, any case
    assert _ids(lib, include=["b2", "d4"]) == ["b2", "d4"]
    assert _ids(lib, exclude=["a1", "m5", "u6"]) == ["b2", "c3", "d4"]
    assert _ids(lib, practice_types=["serve"], tags=["indoor"]) == ["b2"]
    # Quality options act on the processed data; matching practice sessions without records
    # are listed (match sessions have no statistics until Phase 2).
    sel = agg.select(lib, agg.SessionFilter(user_calibration=True))
    assert [s["session_id"] for s in sel.data.sessions] == ["a1"]
    assert {r["id"] for r in sel.filtered_out} == {"b2", "c3", "d4"}
    assert [r["id"] for r in sel.missing] == ["u6"]
    assert any("no statistics yet" in n for n in sel.data.notes)
    assert agg.select(lib, agg.SessionFilter(calibrated_speeds=True)).data.sessions == []


def test_recording_date_is_local(lib):
    from datetime import datetime

    local = datetime.fromisoformat("2026-10-01T21:05:33+00:00").astimezone()
    assert lib.get_session("a1")["recorded_on"] == local.replace(tzinfo=None).isoformat(
        timespec="seconds"
    )


def test_query_round_trip():
    f = agg.SessionFilter.make(
        date_from="2026-10-01", date_to="2026-10-31T00:00", modes=["practice"],
        practice_types="serve,self_feed", profiles=["p1"], tags=["new racket", "indoor"],
        include=["a1", "b2"], exclude=["c3"], user_calibration=True, calibrated_speeds=True,
    )  # fmt: skip
    assert f.date_to == "2026-10-31"
    q = f.to_query()
    assert "from=2026-10-01" in q and "tag=new+racket%2Cindoor" in q and "cal=user" in q
    assert agg.SessionFilter.from_query(q) == f
    assert agg.SessionFilter.from_query("?" + q) == f
    assert agg.SessionFilter.from_query(dict(type="serve", back="x")).practice_types == ("serve",)
    assert agg.SessionFilter().to_query() == "" and agg.SessionFilter.from_query("").is_empty()


def test_cache_invalidated_by_a_reprocessed_session(lib):
    flt = agg.SessionFilter()
    first = agg.select(lib, flt)
    assert agg.select(lib, flt) is first
    # Reprocessing one session (new records file) invalidates the selections it's in.
    sess = _session(lib, "b2")
    time.sleep(0.01)
    run_stats_stage(sess)
    again = agg.select(lib, flt)
    assert again is not first and again.data.records == first.data.records
    # Tags are read from the library, not the records.
    lib.set_session_tags("d4", ["clay"])
    tagged = agg.select(lib, agg.SessionFilter(tags=("clay",)))
    assert [s["session_id"] for s in tagged.data.sessions] == ["d4"]
    assert tagged.data.sessions[0]["tags"] == ["clay"]


# ---------------------------------------------------------------------------
# Records file and library v4
# ---------------------------------------------------------------------------


def test_records_file(lib):
    sess = _session(lib, "a1")
    t = tables.read_table(sess.stats_records_path)
    assert t.schema.equals(STATS_RECORDS, check_metadata=False)
    kinds = t.column("kind").to_pylist()
    assert set(kinds) == {"shot", "swing", "movement"} and kinds.count("movement") == 1
    row = t.slice(0, 1).to_pylist()[0]
    assert row["session_id"] == "a1" and row["practice_type"] == "serve"
    assert row["profile_id"] == "p1" and row["calibration_by"] == "user"
    assert row["recorded_on"] == lib.get_session("a1")["recorded_on"]
    # Excluded shots are kept (flagged), so the page can show them on request.
    assert any(t.column("excluded").to_pylist())
    assert sess.stats_path.exists()


def test_library_v4_backfill(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    (root / "sessions").mkdir()
    conn = sqlite3.connect(root / "library.sqlite")
    for i, script in enumerate(MIGRATIONS[:3], start=1):
        conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {i};\nCOMMIT;")
    conn.execute(
        "INSERT INTO sessions (id, name, dir_name, created_at, updated_at, mode, submode, "
        "source_path, source_hash, duration_s) VALUES ('s1', 'S1', 'd1', ?, ?, 'practice', "
        "'serve', 'C:/v.mp4', 'h', 10)",
        ("2026-10-02T08:00:00+00:00", "2026-10-02T08:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO sessions (id, name, dir_name, created_at, updated_at, mode, submode, "
        "source_path, source_hash, duration_s) VALUES ('s2', 'S2', 'gone', ?, ?, 'practice', "
        "'serve', 'C:/w.mp4', 'h2', 10)",
        ("2026-10-03T08:00:00+00:00", "2026-10-03T08:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    (root / "sessions" / "d1").mkdir()
    (root / "sessions" / "d1" / "session.json").write_text(
        json.dumps({"video": {"creation_time": "2026-10-01T12:00:00.000000Z"},
                    "players": {"me_profile_id": "p9"}}),
        encoding="utf-8",
    )  # fmt: skip
    lib = Library(root).init()
    assert lib.schema_version() == 4
    s1, s2 = lib.get_session("s1"), lib.get_session("s2")
    assert s1["recorded_on"].startswith("2026-10-01") and s1["profile_id"] == "p9"
    assert s2["recorded_on"].startswith("2026-10-0") and s2["profile_id"] is None


def test_tags_and_views(lib):
    assert clean_tags([" new  racket ", "New Racket", "", "a,b"]) == ["new racket", "a b"]
    assert lib.set_session_tags("a1", ["x", "y", "x"]) == ["x", "y"]
    assert lib.session_tags("a1") == ["x", "y"]
    assert "x" in lib.all_tags() and lib.tags_by_session()["a1"] == ["x", "y"]
    lib.save_view("v1", "Serves", "type=serve")
    assert lib.get_view("v1")["query"] == "type=serve"
    assert [v["name"] for v in lib.list_views()] == ["Serves"]
    lib.delete_view("v1")
    assert lib.list_views() == []
    lib.delete_session("a1")
    assert "a1" not in lib.tags_by_session()  # cascades


def test_services_keep_the_library_in_sync(settings):
    from swingvision import services

    lib = Library(settings.output_root).init()
    make_session(lib, Spec("e5", "2026-10-02T10:00:00Z", profile=None, seed=5), False)
    lib.add_profile(id="p1", name="Me")
    services.set_session_player(settings, "e5", "p1")
    assert lib.get_session("e5")["profile_id"] == "p1"
    services.delete_profile(settings, "p1")
    assert lib.get_session("e5")["profile_id"] is None
    assert services.set_session_tags(settings, "e5", ["a"]) == ["a"]
    v = services.save_view(settings, "Mine", "?tag=a")
    assert services.save_view(settings, "mine", "tag=b") == v  # same name: replaced
    assert lib.get_view(v)["query"] == "tag=b"
    with pytest.raises(ValueError):
        services.save_view(settings, " ", "x")


# ---------------------------------------------------------------------------
# Export and the Stats page over a selection
# ---------------------------------------------------------------------------


def test_selection_export_matches_the_page(lib):
    sel = agg.select(lib, agg.SessionFilter(practice_types=("serve",)))
    rows = st.filter_records(sel.data.records, ["serve"], "near")
    csv = pa_csv.read_csv(io.BytesIO(ex.selection_bytes(sel.data, "shots", "csv", rows)))
    s = st.summarize_shots(rows)
    assert csv.num_rows == s["n"]
    out = csv.column("outcome").to_pylist()
    called = [o for o in out if o in st.CALLED]
    assert out.count("in") / len(called) == pytest.approx(s["in_pct"])
    assert set(csv.column("session_id").to_pylist()) == {"a1", "b2"}
    assert "tags" in csv.column_names and "recorded_on" in csv.column_names
    sw = ex.selection_table(sel.data, "swings")
    assert sw.num_rows == len(sel.data.swings)
    with pytest.raises(ex.ExportError):
        ex.selection_table(sel.data, "pose")


def test_stats_page_over_a_selection(settings, monkeypatch):
    import dash

    from swingvision.app.main import create_app

    lib = make_library(settings.output_root, SPECS)
    lib.save_view("v1", "Serves", "type=serve&tag=indoor")
    app = create_app()
    from swingvision.app.pages import stats as page

    multi = next(p for p in dash.page_registry.values() if p.get("path") == "/stats")
    multi["layout"](type="serve", back="a1")
    multi["layout"]()
    q = "type=serve,self_feed"
    out = page._render({"query": q}, [], "all", False, "heat")
    assert len(out) == 11
    assert "3 sessions" in _js(out[9])
    fig = page._trend({"query": q}, [], "all", False, "speed_median")
    assert len(fig.data[0].x) == 3
    for kpi in st.TREND_KPIS:
        page._trend({"query": q}, ["serve"], "near", True, kpi)
    assert page._filter(["2026-10-01", "2026-10-05"], ["practice"], [], [], [], [], [], False,
                        {"query": ""}) == {"query": "from=2026-10-01&to=2026-10-05&mode=practice"}  # fmt: skip
    assert page._open_view("v1")[2] == ["serve"]
    assert page.export_query("type=serve", ["serve"], "near", True) == (
        "type=serve&groups=serve&end=near&excluded=1"
    )
    # Nothing matches: the page still renders.
    page._render({"query": "tag=nothing"}, [], "all", False, "dots")
    # A matching session without statistics records: the page offers to update it (here it
    # needs real processing, so a job is queued).
    from swingvision.app import state

    monkeypatch.setattr(state.OPTIONS, "start_worker", False)
    make_session(lib, Spec("u6", "2026-10-10T09:00:00Z", submode="serve", seed=6), False)
    out = page._render({"query": q}, [], "all", False, "heat")
    assert "st-refresh" in _js(out[0])
    note, scope = page._refresh_missing(1, {"query": q})
    assert "queued" in _js(note) and scope["refreshed"] == 1
    assert lib.latest_job_for_session("u6") is not None

    client = app.server.test_client()
    r = client.get("/export-selection/shots.csv?type=serve&groups=serve&end=near")
    assert r.status_code == 200
    csv = pa_csv.read_csv(io.BytesIO(r.data))
    sel = agg.select(lib, agg.SessionFilter(practice_types=("serve",)))
    assert (
        csv.num_rows
        == st.summarize_shots(st.filter_records(sel.data.records, ["serve"], "near"))["n"]
    )
    assert client.get("/export-selection/swings.parquet?sessions=a1").status_code == 200
    assert client.get("/export-selection/pose.csv").status_code == 404
    # A shot clicked on the Stats page opens the session at that moment.
    from swingvision.app.pages import session_review

    assert session_review._start_fragment("12.5") == "#t=11.50"
    assert session_review._start_fragment("x") == "" and session_review._start_fragment(None) == ""


def test_cli_stats(settings, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from swingvision.cli import app

    make_library(settings.output_root, SPECS)
    runner = CliRunner()
    out = tmp_path / "shots.csv"
    r = runner.invoke(
        app, ["stats", "--type", "serve", "--from", "2026-10-01", "--trend", "in_pct",
              "--out", str(out)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "2 sessions" in r.output and "In % per session" in r.output
    assert out.exists()
    r = runner.invoke(app, ["tags", "a1", "clay", "outdoor"])
    assert r.exit_code == 0 and "clay, outdoor" in r.output
    r = runner.invoke(app, ["stats", "--tag", "clay"])
    assert "1 sessions" in r.output
    r = runner.invoke(app, ["sessions"])
    assert "[clay, outdoor]" in r.output


# ---------------------------------------------------------------------------
# Speed: 50 sessions
# ---------------------------------------------------------------------------


def test_fifty_sessions_render_fast(settings):
    from swingvision.app.main import create_app

    specs = [
        Spec(f"s{i:02d}", f"2026-{9 + i // 28:02d}-{1 + i % 28:02d}T10:00:00Z", seed=100 + i,
             n_shots=150, n_other_swings=150, minutes=60.0, tags=["even"] if i % 2 else [])
        for i in range(50)
    ]  # fmt: skip
    make_library(settings.output_root, specs)
    create_app()
    from swingvision.app.pages import stats as page

    scope = {"query": ""}
    t0 = time.perf_counter()
    out = page._render(scope, [], "all", False, "heat")
    page._trend(scope, [], "all", False, "in_pct")
    cold = time.perf_counter() - t0
    assert "50 sessions" in _js(out[9])
    t0 = time.perf_counter()
    page._render(scope, [], "all", False, "dots")
    page._trend(scope, ["serve"], "all", False, "speed_median")
    warm = time.perf_counter() - t0
    print(f"50 sessions: {cold:.2f} s, cached {warm:.2f} s")
    assert cold < 2.0
    assert warm < 0.5
