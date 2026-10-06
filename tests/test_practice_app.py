"""Practice page pieces and the target editor (M5), without a browser."""

from __future__ import annotations

from datetime import UTC, datetime

import dash
import pytest

from swingvision.analysis import practice as pr
from swingvision.app.components import practice_view as pv
from swingvision.app.components.target_editor import (
    N_COURT_SHAPES,
    PRESETS,
    apply_relayout,
    editor_figure,
    targets_from_store,
)
from swingvision.app.main import create_app
from swingvision.court import calibration as calib
from swingvision.storage.schemas import (
    Calibration,
    PracticeConfig,
    SessionConfig,
    SourceInfo,
    Target,
)
from tests.synth_court import make_camera
from tests.test_practice import two_ends_session


@pytest.fixture(scope="module")
def app():
    return create_app()


def test_practice_page_registered_and_renders(app, settings):
    paths = {p.get("path_template") or p["path"] for p in dash.page_registry.values()}
    assert "/practice/<session_id>" in paths
    page = next(
        p for p in dash.page_registry.values() if p.get("path_template") == "/practice/<session_id>"
    )
    page["layout"](session_id="nope")  # unknown session: a "not found" page


TARGETS = [
    Target(id="deep", name="Deep", shape="rect", x0=-2.0, y0=8.0, x1=0.0, y1=10.0),
    Target(id="c", name="C", shape="circle", cx=1.0, cy=5.0, r=1.0, frame="absolute"),
]


@pytest.fixture
def data():
    s, seg = two_ends_session()
    rows = pr.evaluate(seg, s.tables()[1], TARGETS).to_pylist()
    cam = make_camera()
    cal = Calibration(
        source="auto",
        created_at=datetime.now(UTC),
        camera=calib.to_params(cam),
    )
    config = SessionConfig(
        id="x",
        name="x",
        created_at=datetime.now(UTC),
        dir_name="x",
        source=SourceInfo(path="x.mp4", size_bytes=1, fast_hash="h", mtime=0),
        mode="practice",
        practice=PracticeConfig(targets=TARGETS),
    )
    from swingvision.analysis.segmentation import blocks_of

    return pv.PracticeData(
        config=config,
        rows=rows,
        blocks=blocks_of(seg),
        segments={r["segment_id"]: r for r in seg.to_pylist() if r["kind"] == "practice_shot"},
        cal=cal,
        edits_version=0,
    )


def test_court_map_views(data):
    fig = pv.court_map(data, data.rows, "hit", selected=data.rows[0]["segment_id"])
    landing_traces = [t for t in fig.data if t.name == "landings"]
    xs = sorted(x for t in landing_traces for x in t.x)
    ys = sorted(y for t in landing_traces for y in t.y)
    # In the hitter's frame every landing is on the far half.
    assert min(ys) > 0 and len(xs) == 4
    assert any(t.name == "selected" for t in fig.data)
    court = pv.court_map(data, data.rows, "court")
    ys = sorted(y for t in court.data if t.name == "landings" for y in t.y)
    assert min(ys) < 0 < max(ys)
    # Relative targets drawn for both ends on the court view, absolute ones once.
    n_targets = len(court.layout.shapes) - len(pv.court_figure().layout.shapes)
    assert n_targets == 3
    placing = pv.court_map(data, data.rows, "hit", placing=True)
    assert all(t.hoverinfo == "skip" for t in placing.data if t.name == "landings")


def test_panels_render(data):
    assert pv.rolling_figure(data.rows, True).data
    assert pv.kpis(data.rows, True) is not None
    assert pv.blocks_table(data, True) is not None
    assert pv.shots_table(data.rows, data.rows[0]["segment_id"], True) is not None
    assert pv.breakdown_table(data.rows, True) is not None
    assert pv.shot_detail(data.rows[0], placing=False) is not None
    assert pv.segments_card("x", data) is not None
    store = pv.shots_store(data)
    assert len(store["t0"]) == 4 and store["side"] == [-1, -1, 1, 1]
    assert pv.filtered(data, "all", "serve") == []
    assert len(pv.filtered(data, str(data.rows[0]["block_id"]), "all")) >= 1


def test_targets_overlay(data):
    ov = pv.targets_overlay(data.cal, data.targets)
    assert ov is not None and set(ov) == {"near", "far"}
    assert ov["near"].startswith("data:image/svg+xml;base64,") and ov["near"] != ov["far"]
    assert pv.targets_overlay(data.cal, []) is None


# ---------------------------------------------------------------------------
# Target editor
# ---------------------------------------------------------------------------


def _court_shapes():
    return [{}] * N_COURT_SHAPES


def test_editor_draw_move_erase():
    drawn = {"type": "rect", "x0": 3.6, "y0": 11.8, "x1": 0.6, "y1": 9.6}
    out = apply_relayout([], {"shapes": [*_court_shapes(), drawn]})
    assert len(out) == 1
    t = out[0]
    assert (t["x0"], t["x1"], t["y0"], t["y1"]) == (0.6, 3.6, 9.6, 11.8)
    # A circle from its bounding box.
    circ = {"type": "circle", "x0": -3.0, "x1": -1.0, "y0": 4.0, "y1": 6.0}
    out2 = apply_relayout(out, {"shapes": [*_court_shapes(), {**drawn, "name": t["id"]}, circ]})
    assert [d["shape"] for d in out2] == ["rect", "circle"]
    assert out2[0]["id"] == t["id"] and out2[1]["r"] == pytest.approx(1.0)
    # Move: Plotly reports the changed corners only.
    moved = apply_relayout(
        out2, {f"shapes[{N_COURT_SHAPES}].y0": 8.0, f"shapes[{N_COURT_SHAPES}].y1": 10.2}
    )
    assert (moved[0]["y0"], moved[0]["y1"]) == (8.0, 10.2)
    assert moved[1] == out2[1]
    # Erase: the full list without it.
    erased = apply_relayout(moved, {"shapes": [*_court_shapes(), {**circ, "name": out2[1]["id"]}]})
    assert [d["id"] for d in erased] == [out2[1]["id"]]
    # Unrelated relayouts (theme colours, autosize) and slivers change nothing.
    assert apply_relayout(erased, {"font.color": "#fff"}) is None
    sliver = {"type": "rect", "x0": 1.0, "x1": 1.01, "y0": 2.0, "y1": 5.0}
    assert len(apply_relayout(erased, {"shapes": [*_court_shapes(), sliver]})) == 0


def test_editor_figure_and_presets():
    targets = targets_from_store(
        [
            {
                "id": "a",
                "name": "A",
                "shape": "rect",
                "frame": "relative",
                "strokes": [],
                **{k: PRESETS["Deuce box"][k] for k in ("x0", "y0", "x1", "y1")},
            },
            {"id": "bad", "name": "?", "shape": "rect"},
        ]
    )
    assert [t.id for t in targets] == ["a"]
    fig = editor_figure(targets)
    assert len(fig.layout.shapes) == N_COURT_SHAPES + 1
    assert fig.layout.shapes[-1].name == "a" and fig.layout.shapes[-1].editable
    # Presets are valid targets in the hitter's frame (far half).
    for p in PRESETS.values():
        t = Target(id="p", name="p", shape="rect", **{k: p[k] for k in ("x0", "y0", "x1", "y1")})
        assert pr.target_valid(t) and pr.target_center(t)[1] > 0
