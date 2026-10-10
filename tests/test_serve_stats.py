"""Serve contact point (M7b), beyond the geometry in ``test_serve_contact``: the per-serve
records and their statistics (bins, grid, findings, regression) for one session or a
selection, the Stats page section, the exports, the serve edits, the ``serve_contact``
table builder, ``serve_feet``'s reuse of earlier frames, and the foot keypoint decoding."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime

import numpy as np
import plotly.utils
import pyarrow.csv as pa_csv
import pytest

from swingvision.analysis import aggregate as agg
from swingvision.analysis import export as ex
from swingvision.analysis import serve_stats as ss
from swingvision.analysis import stats as st
from swingvision.court import calibration as calib
from swingvision.pipeline.stages import serve as stage
from swingvision.pose import feet as ft
from swingvision.pose import serve_contact as sc
from swingvision.storage import edits as ed
from swingvision.storage.schemas import SERVE_CONTACT, Calibration, SessionEdits
from tests.synth_stats import Spec, make_library
from tests.test_serve_contact import CAM, FPS, serve, toe_keypoints

BIG = [
    Spec(f"s{i}", f"2026-10-0{i + 1}T18:00:00Z", submode="serve", n_shots=700, seed=10 + i)
    for i in range(4)
]


@pytest.fixture(autouse=True)
def _fresh_cache():
    agg.clear_cache()
    yield
    agg.clear_cache()


@pytest.fixture
def big(tmp_path):
    return make_library(tmp_path / "out", BIG)


def _js(obj) -> str:
    return json.dumps(obj, cls=plotly.utils.PlotlyJSONEncoder, sort_keys=True)


def test_statistics_recover_the_planted_effects(big):
    sel = agg.select(big, agg.SessionFilter())
    rows = ss.select(sel.data.serves)  # near end, flagged hidden
    assert rows and all(r["side"] == -1 for r in rows)
    assert len(rows) < sum(r["forward_m"] is not None for r in sel.data.serves)
    reg = ss.regression(rows)
    b, lo, hi = reg["speed"]["effects"]["forward_m"]
    assert abs(b - 8.0) < 1.0 and hi - lo < 2.0  # +8 km/h per 10 cm in front of the toe
    assert reg["in"]["effects"]["lateral_m"][2] < 0  # far to the racket side: fewer in
    found = {f["axis"]: f for f in ss.findings(rows)}
    assert found["forward_m"]["speed"][0] > 0
    assert found["lateral_m"]["in"] is not None and found["lateral_m"]["in"][0] != 0
    bins = ss.quantile_bins(rows, "forward_m")
    assert len(bins) >= 3 and all(b["n"] >= ss.MIN_BIN for b in bins)
    assert bins[-1]["speed_mean"] > bins[0]["speed_mean"]
    cells = ss.grid(rows)
    assert cells and all(c["n"] >= ss.MIN_CELL for c in cells)
    o = ss.overview(rows)
    assert o["n"] == len(rows) and o["forward_m_median"] == pytest.approx(0.15, abs=0.05)
    text = ss.summary_text(
        found["forward_m"], lambda lo, hi, a: f"{lo:.2f}–{hi:.2f}", lambda v: f"{v:+.0f}"
    )
    assert text.startswith("Contacts") and "vs. the rest" in text


def test_filters():
    rows = [
        {"forward_m": 0.1, "side": -1, "serve_side": "deuce", "serve_flags": [], "excluded": False},
        {"forward_m": 0.2, "side": -1, "serve_side": "ad", "serve_flags": ["dark"], "excluded": False},
        {"forward_m": 0.3, "side": 1, "serve_side": "ad", "serve_flags": ["far"], "excluded": False},
        {"forward_m": None, "side": -1, "serve_side": "ad", "serve_flags": [], "excluded": False},
        {"forward_m": 0.4, "side": -1, "serve_side": "ad", "serve_flags": [], "excluded": True},
    ]  # fmt: skip
    assert len(ss.select(rows)) == 1
    assert len(ss.select(rows, "ad", near_only=False, hide_flagged=False)) == 2
    assert ss.select(rows, "deuce")[0]["forward_m"] == 0.1
    assert not ss.findings([]) and not ss.grid([]) and ss.regression([])["speed"] is None


def test_serve_section_same_for_one_session_and_a_selection(settings):
    from swingvision.app.main import create_app

    make_library(settings.output_root, BIG[:1])
    create_app()
    from swingvision.app.pages import stats as page

    one = page._serves({"session": "s0"}, False, "all", True, True)
    many = page._serves({"query": "sessions=s0"}, False, "all", True, True)
    assert _js(one) == _js(many)
    assert one[1]["data"]  # the top-down map has dots


def test_exports_carry_the_serve_contacts(big):
    from swingvision.storage.session import Session

    session = Session.open(big.root, big.get_session("s0")["dir_name"])
    shots = ex.export_table(session, "shots")
    assert "serve_forward_m" in shots.column_names
    fwd = [v for v in shots.column("serve_forward_m").to_pylist() if v is not None]
    assert len(fwd) == sum(r["forward_m"] is not None for r in st.load(session, True).serves)
    csv = pa_csv.read_csv(io.BytesIO(ex.export_bytes(session, "shots", "csv")))
    assert "serve_flags" in csv.column_names
    sel = agg.select(big, agg.SessionFilter(include=("s0",)))
    t = ex.selection_table(sel.data, "serves")
    assert t.num_rows == len(sel.data.serves) and "lateral_sigma_m" in t.column_names


def test_serve_edits():
    e = SessionEdits()
    ed.set_serve(e, 10.0, frame=601)
    ed.set_serve(e, 10.1, toe=[1800.04, 1210.0])  # the same serve
    assert len(e.serves) == 1 and e.serves[0].frame == 601 and e.serves[0].toe == [1800.0, 1210.0]
    ed.set_serve(e, 10.0, frame=None)
    assert e.serves[0].frame is None and e.serves[0].toe is not None
    ed.set_serve(e, 10.0, toe=None)
    assert not e.serves
    ed.set_serve(e, 20.0, frame=1200)
    ed.set_serve(e, 5.0, frame=300)
    assert [s.t for s in e.serves] == [5.0, 20.0]


def _cal():
    return Calibration(
        source="auto", created_at=datetime(2026, 10, 9, tzinfo=UTC), camera=calib.to_params(CAM)
    )


def test_serve_contact_table_from_a_synthetic_serve():
    rng = np.random.default_rng(1)
    s = serve(rng, t_contact=(round(3.0 * FPS) + 0.4) / FPS)
    toss = sc.toss_contact(s["pts"], s["clock"], CAM, s["t_frame"])
    m = np.abs(s["t"] - s["t_frame"]) < 1.2
    feet = (s["frames"][m], s["t"][m], toe_keypoints(s["frames"][m], s["t"][m], rng))
    swing = {"swing_id": 7, "t_contact": toss.t, "frame_contact": toss.frame, "side": -1,
             "racket_hand": "left", "flags": [], "contact_height_m": 2.2}  # fmt: skip
    shots = [{"swing_id": 7, "shot_id": 3}]
    table, summary = stage.build_serve_contacts(
        [swing], {"7": sc.toss_to_dict(toss)}, feet, _cal(), shots, profile={"height_m": 1.78}
    )
    assert table.schema.equals(SERVE_CONTACT, check_metadata=False)
    r = table.to_pylist()[0]
    assert r["shot_id"] == 3 and r["front_foot"] == "right" and r["contact_source"] == "toss_path"
    assert r["toe_source"] == "toe" and r["toe_on_ground"] is True
    want = sc.offsets(s["contact"], None, np.array([-0.15, -12.05, 0.0]), None, -1, "left")
    assert abs(r["forward_m"] - want.forward_m) < 0.05
    assert abs(r["lateral_m"] - want.lateral_m) < 0.02
    assert r["height_rel"] == pytest.approx(r["height_m"] / 1.78)
    assert summary["near_with_offsets"] == 1 and not r["flags"]
    # A toe the user placed wins.
    px = CAM.project(np.array([[0.2, -12.3, 0.0]]))[0]
    table, _ = stage.build_serve_contacts(
        [swing], {"7": sc.toss_to_dict(toss)}, feet, _cal(), shots,
        toe_edits=[(toss.t, list(px))],
    )  # fmt: skip
    r = table.to_pylist()[0]
    assert r["toe_source"] == "edit" and r["toe_x"] == pytest.approx(0.2, abs=1e-3)
    # No toss: no contact point, and above the picture when the usual height would be.
    high = sc.TossContact(frame=toss.frame, t=toss.t, flags=["toss_not_tracked"])
    table, _ = stage.build_serve_contacts(
        [swing], {"7": sc.toss_to_dict(high)}, feet, _cal(), shots, profile={"height_m": 4.0}
    )
    r = table.to_pylist()[0]
    assert r["forward_m"] is None and "contact_above_frame" in r["flags"]
    assert np.isnan(r["contact_cov"]).all()


def test_serve_feet_reuses_covered_frames():
    t = np.arange(0, 2, 1 / 60)
    assert stage._gaps(t, 0.5, 1.5) == []
    t2 = t[(t < 0.8) | (t > 1.0)]
    gaps = stage._gaps(t2, 0.5, 1.5)
    assert len(gaps) == 1 and 0.78 < gaps[0][0] < 0.8 and 1.0 < gaps[0][1] < 1.02
    assert stage._gaps(np.zeros(0), 0.5, 1.5) == [(0.5, 1.5)]


def test_feet_windows_merge(tmp_path):
    class S:
        swings_path = tmp_path / "x.parquet"

    assert stage.feet_windows(S()) == []


def test_foot_crop_and_simcc_decoding():
    r = ft.crop_rect(np.array([100.0, 200.0, 200.0, 600.0]))
    w, h = r[2] - r[0], r[3] - r[1]
    assert w / h == pytest.approx(ft.INPUT_W / ft.INPUT_H) and h == pytest.approx(500.0)
    n = int(ft.INPUT_W * ft.SIMCC_SPLIT)
    x = np.arange(n)
    sx = np.exp(-0.5 * ((x - 100.3) / 3.0) ** 2)[None, None]
    sy = np.exp(-0.5 * ((np.arange(int(ft.INPUT_H * ft.SIMCC_SPLIT)) - 400.0) / 3.0) ** 2)[
        None, None
    ]
    kp = ft.decode_simcc(sx, sy)
    assert kp[0, 0, 0] == pytest.approx(100.3 / ft.SIMCC_SPLIT, abs=0.05)
    assert kp[0, 0, 1] == pytest.approx(200.0, abs=0.05) and kp[0, 0, 2] == pytest.approx(
        1.0, abs=0.01
    )
    assert ft.foot_names("right") == (1, 5, 6, 7) and ft.foot_names("left") == (0, 2, 3, 4)


@pytest.mark.gpu
def test_foot_pose_on_a_drawn_figure():
    """The ONNX model runs (CUDA when available) and maps keypoints back into the frame."""
    torch = pytest.importorskip("torch")
    from swingvision.models import registry

    if not registry.get(ft.DEFAULT_MODEL).available():
        pytest.skip("RTMW weights not downloaded")
    img = torch.full((3, 1080, 1920), 40, dtype=torch.uint8)
    box = np.array([900.0, 300.0, 1100.0, 900.0])
    est = ft.FootPose()
    crop, rect = est.prepare(img, box)
    assert crop.shape == (3, ft.INPUT_H, ft.INPUT_W)
    kp = est.run([crop], rect[None])
    assert kp.shape == (1, len(ft.FOOT), 3)
    # A blank picture puts the guesses anywhere in the crop (a peak at the edge: half a pixel out).
    assert ((kp[0, :, 0] >= rect[0] - 2) & (kp[0, :, 0] <= rect[2] + 2)).all()
