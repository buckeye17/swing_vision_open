"""Practice segmentation, targets and accuracy (M5) on synthetic event streams."""

from __future__ import annotations

import itertools

import numpy as np
import pyarrow as pa
import pytest

from swingvision.analysis import practice as pr
from swingvision.analysis.segmentation import (
    SegParams,
    av_offset,
    blocks_of,
    practice_shots,
    segment_practice,
)
from swingvision.storage import edits as ed
from swingvision.storage.schemas import SEGMENTS, PracticeShotEdit, SessionEdits, Target
from swingvision.training.practice_labels import SegmentLabels, ShotLabel, score_segments
from tests.synth_practice import Stream

# ---------------------------------------------------------------------------
# Geometry and calls
# ---------------------------------------------------------------------------


def rect(id_="t", x0=-2.0, y0=4.0, x1=0.0, y1=6.0, **kw) -> Target:
    return Target(id=id_, name=id_, shape="rect", x0=x0, y0=y0, x1=x1, y1=y1, **kw)


def test_hitter_frame_and_mirroring():
    assert pr.to_hitter_frame(1.0, 5.0, -1) == (1.0, 5.0)
    assert pr.to_hitter_frame(1.0, -5.0, 1) == (-1.0, 5.0)
    t = rect(x0=0.5, x1=2.0)
    far = pr.target_on_court(t, 1)  # a far-end hitter's target lies on the near half
    assert (far.x0, far.x1, far.y0, far.y1) == (-2.0, -0.5, -6.0, -4.0)
    absolute = t.model_copy(update={"frame": "absolute"})
    assert pr.target_on_court(absolute, 1) == absolute
    c = Target(id="c", name="c", shape="circle", cx=1.0, cy=5.0, r=0.5)
    assert (pr.target_on_court(c, 1).cx, pr.target_on_court(c, 1).cy) == (-1.0, -5.0)


def test_target_edge_rect_and_circle():
    t = rect()  # x -2..0, y 4..6
    assert pr.target_edge(t, -1.0, 5.0) == pytest.approx(1.0)
    assert pr.target_edge(t, -1.0, 6.5) == pytest.approx(-0.5)
    assert pr.target_edge(t, 0.3, 6.4) == pytest.approx(-0.5)  # corner: 0.3, 0.4 -> 0.5
    c = Target(id="c", name="c", shape="circle", cx=0.0, cy=5.0, r=1.0)
    assert pr.target_edge(c, 0.0, 5.5) == pytest.approx(0.5)
    assert pr.target_edge(c, 0.0, 6.5) == pytest.approx(-0.5)
    # Corners drawn in any order.
    flipped = pr.normalized(rect(x0=0.0, x1=-2.0, y0=6.0, y1=4.0))
    assert pr.target_edge(flipped, -1.0, 5.0) == pytest.approx(1.0)
    assert not pr.target_valid(rect(x1=-2.0))


def test_target_stroke_filter():
    assert pr.target_applies(rect(), "groundstroke", None)
    serve_only = rect(strokes=["serve"])
    assert pr.target_applies(serve_only, "serve", None)
    assert not pr.target_applies(serve_only, "groundstroke", None)
    assert pr.target_applies(rect(strokes=["forehand"]), "groundstroke", "forehand")


@pytest.mark.parametrize(
    "rx, ry, side, expect",
    [
        (-2.0, 4.0, "deuce", "in"),
        (0.02, 4.0, "deuce", "in"),  # center over the center line's far edge: in
        (0.5, 4.0, "deuce", "out_wide"),  # wrong box
        (-2.0, 6.6, "deuce", "out_long"),
        (2.0, 4.0, "ad", "in"),
        (-4.3, 2.0, "ad", "out_wide"),
        (2.0, -1.0, "ad", "own_side"),
    ],
)
def test_service_box_calls(rx, ry, side, expect):
    assert pr.service_box_call(rx, ry, side)[0] == expect


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def self_feed_stream(n: int = 6, t0: float = 10.0, period: float = 6.0, side: int = -1) -> Stream:
    """Drop-feed groundstrokes from the baseline: drop bounce at the feet, hit, landing."""
    s = Stream()
    y = side * 11.0
    s.stand(t0 - 3, t0 + n * period, 0.5, y)
    for k in range(n):
        t = t0 + k * period
        s.event("bounce", t - 0.5, 0.6, y - side * 0.3)  # the drop
        land = (-2.0 + 0.5 * k, -side * (8.0 + 0.2 * k))
        s.hit(t, 0.5, y, outcome="in", landing=land, land_t=t + 1.0, speed=95.0)
    return s


def test_self_feed_groundstrokes():
    s = self_feed_stream()
    seg, summary = segment_practice(s.inputs())
    assert seg.schema.equals(SEGMENTS, check_metadata=False)
    shots = practice_shots(seg)
    assert len(shots) == 6
    assert summary["blocks"] == 1
    for r in shots:
        assert r["shot_kind"] == "groundstroke"
        assert r["feed_kind"] == "drop"
        assert r["side"] == -1
        assert r["end_reason"] == "bounce"
        assert r["start_t"] <= r["t_feed"] < r["t_contact"] < r["t_end"] < r["end_t"]
    # No overlaps, and blocks cover their shots.
    for a, b in itertools.pairwise(shots):
        assert a["end_t"] <= b["start_t"] + 1e-9
    (block,) = blocks_of(seg)
    assert block["n_shots"] == 6 and block["shot_kind"] == "groundstroke"


def test_dribbles_and_taps_are_not_shots():
    s = Stream()
    s.stand(0, 20, 0.3, -12.0)
    for k in range(5):  # pre-serve dribbles with the racket: hit, bounce at the feet
        t = 5 + 0.4 * k
        s.hit(t, 0.3, -12.0, outcome="own_side", landing=(0.2, -12.0), land_t=t + 0.2, sound=8)
    seg, _ = segment_practice(s.inputs())
    assert practice_shots(seg) == []


def test_serve_seen_contact_and_sides():
    s = Stream()
    s.stand(0, 40, 0.4, -12.0)  # right of the center mark: deuce
    s.stand(40, 80, -0.4, -12.0)  # ad
    for t, land in ((10.0, (-2.5, 5.0)), (20.0, (-1.0, 7.0)), (50.0, (2.0, 4.0))):
        for d in (2.5, 2.1, 1.7):
            s.event("bounce", t - d, 0.3 if t < 40 else -0.3, -12.1)  # dribbles
        s.hit(t, 0.4, -12.0, outcome="in", landing=land, land_t=t + 0.6, height=2.75, speed=140)
    seg, summary = segment_practice(s.inputs())
    shots = practice_shots(seg)
    assert [r["shot_kind"] for r in shots] == ["serve"] * 3
    assert [r["serve_side"] for r in shots] == ["deuce", "deuce", "ad"]
    assert shots[0]["feed_kind"] == "dribble"
    assert summary["serves"] == 3

    table = pr.evaluate(seg, s.tables()[1], [])
    calls = [r["outcome"] for r in table.to_pylist()]
    assert calls == ["in", "out_long", "in"]  # against the service boxes
    assert {r["call_area"] for r in table.to_pylist()} == {"deuce_box", "ad_box"}


def test_unseen_contact_from_landing_and_sound():
    s = Stream()
    s.stand(0, 60, 0.4, -12.0)
    for k in range(4):
        t = 10.0 + 10 * k
        s.event("bounce", t - 2.0, 0.3, -12.1)
        s.event("bounce", t - 1.6, 0.3, -12.1)
        s.onsets.append((t + 0.02, 50.0))  # the impact (sound delay ≈ 20 ms)
        s.event("bounce", t + 0.55, -2.0, 5.0)  # landing in the far court
    seg, summary = segment_practice(s.inputs())
    shots = practice_shots(seg)
    assert len(shots) == 4 and summary["unseen"] == 4
    for k, r in enumerate(shots):
        assert r["contact_source"] == "audio"
        assert r["t_contact"] == pytest.approx(10.0 + 10 * k, abs=0.03)
        assert r["side"] == -1 and r["landing_y"] == pytest.approx(5.0)
        assert "contact_unseen" in r["flags"]


def test_unseen_needs_an_impact_sound_or_no_audio():
    s = Stream()
    s.stand(0, 30, 0.4, -12.0)
    s.event("bounce", 10.0, -2.0, 14.5)  # a ball beyond the far baseline, no sound
    s.onsets.append((3.0, 30.0))  # (the recording has sound, just not then)
    assert practice_shots(segment_practice(s.inputs())[0]) == []
    # Without an audio track the landing alone counts.
    shots = practice_shots(segment_practice(s.inputs(has_audio=False))[0])
    assert len(shots) == 1 and shots[0]["contact_source"] == "estimate"


def test_same_ball_bouncing_on_is_one_shot():
    s = Stream()
    s.stand(0, 30, 0.4, -12.0)
    s.onsets.append((10.02, 50.0))
    for k, (t, y) in enumerate(((10.6, 6.0), (11.4, 9.0), (12.0, 10.5), (13.0, 11.5))):
        s.event("bounce", t, -2.0 - 0.2 * k, y)
        s.onsets.append((t - 0.3, 20.0))
    shots = practice_shots(segment_practice(s.inputs())[0])
    assert len(shots) == 1


def test_net_shots():
    s = Stream()
    s.stand(0, 30, 0.4, -12.0)
    # Into the net: the first bounce is back on the hitter's half, right behind the net.
    s.hit(10.0, 0.4, -12.0, outcome="unknown", landing=(-0.5, -1.0), land_t=10.5, height=2.7)
    seg, _ = segment_practice(s.inputs())
    (r,) = practice_shots(seg)
    assert r["end_reason"] == "net" and "net_from_drop" in r["flags"]
    (row,) = pr.evaluate(seg, s.tables()[1], [rect()]).to_pylist()
    assert row["outcome"] == "net" and row["in_target"] is False


def test_blocks_split_on_pause_and_side_change():
    a = self_feed_stream(n=3, t0=10.0)
    b = self_feed_stream(n=3, t0=100.0)  # after a long pause
    c = self_feed_stream(n=3, t0=125.0, side=1)  # from the other end
    s = Stream()
    for part in (a, b, c):
        off = len(s.events)
        for e in part.events:
            s.events.append({**e, "event_id": e["event_id"] + off})
        for sh in part.shots:
            s.shots.append({**sh, "hit_event_id": sh["hit_event_id"] + off,
                            "shot_id": len(s.shots)})  # fmt: skip
        s.onsets += part.onsets
        s.player += part.player
    seg, summary = segment_practice(s.inputs())
    assert summary["blocks"] == 3
    blocks = blocks_of(seg)
    assert [b["n_shots"] for b in blocks] == [3, 3, 3]
    assert [b["side"] for b in blocks] == [-1, -1, 1]


def test_ball_machine_feeds():
    s = Stream()
    s.stand(0, 60, 0.0, -11.0)
    for k in range(4):
        t = 10.0 + 4 * k
        s.event("hit", t - 1.6, 0.0, 12.5, hitter="machine")  # feed from the far baseline
        s.event("bounce", t - 0.5, 0.3, -7.0)  # the feed bounces in front of the player
        s.hit(t, 0.3, -11.0, outcome="in", landing=(1.0, 9.0), land_t=t + 1.0, speed=100)
    seg, _ = segment_practice(s.inputs(submode="ball_machine"))
    shots = practice_shots(seg)
    assert len(shots) == 4
    assert all(r["feed_kind"] == "machine" and r["shot_kind"] == "groundstroke" for r in shots)
    assert all(r["feed_event_id"] is not None for r in shots)


def test_serve_practice_submode_makes_everything_a_serve():
    s = self_feed_stream(n=2)
    shots = practice_shots(segment_practice(s.inputs(submode="serve"))[0])
    assert {r["shot_kind"] for r in shots} == {"serve"}


def test_av_offset_estimate():
    rng = np.random.default_rng(0)
    hits = np.sort(rng.uniform(0, 300, 40))
    onsets = hits + 0.11 + rng.normal(0, 0.004, len(hits))
    noise = rng.uniform(0, 300, 200)
    t = np.concatenate([onsets, noise])
    s = np.concatenate([np.full(len(onsets), 40.0), np.full(len(noise), 12.0)])
    order = np.argsort(t)
    off = av_offset(hits, np.zeros(len(hits)), t[order], s[order])
    assert off == pytest.approx(0.11, abs=0.01)
    assert av_offset(hits[:3], np.zeros(3), t[order], s[order]) == 0.0


def test_padding_params():
    s = self_feed_stream(n=1)
    (r,) = practice_shots(
        segment_practice(s.inputs(), SegParams(pad_before_s=2.0, pad_after_s=3.0))[0]
    )
    assert r["start_t"] == pytest.approx(r["t_feed"] - 2.0)
    assert r["end_t"] == pytest.approx(r["t_end"] + 3.0)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def two_ends_session():
    """Two groundstrokes from each end, landing at mirrored spots."""
    s = Stream()
    s.stand(0, 40, 0.5, -11.0)
    s.stand(40, 80, -0.5, 11.0)
    for t, y, land in ((10, -11.0, (-1.0, 9.0)), (16, -11.0, (2.0, 5.0)),
                       (50, 11.0, (1.0, -9.0)), (56, 11.0, (-2.0, -5.0))):  # fmt: skip
        s.event("bounce", t - 0.5, 0.5, y * 0.97)
        s.hit(t, 0.5, y, outcome="in", landing=land, land_t=t + 1.0, speed=100.0)
    seg, _ = segment_practice(s.inputs())
    return s, seg


def test_relative_target_follows_the_hitter():
    s, seg = two_ends_session()
    deep_left = rect("deep", x0=-2.0, y0=8.0, x1=0.0, y1=10.0)
    rows = pr.evaluate(seg, s.tables()[1], [deep_left]).to_pylist()
    assert [r["in_target"] for r in rows] == [True, False, True, False]
    assert rows[0]["rel_x"] == pytest.approx(-1.0) and rows[2]["rel_x"] == pytest.approx(-1.0)
    assert rows[2]["rel_y"] == pytest.approx(9.0)
    # Same distance and errors from both ends.
    assert rows[0]["target_dist_m"] == pytest.approx(rows[2]["target_dist_m"])
    assert rows[1]["depth_err_m"] == pytest.approx(-4.0)
    assert rows[1]["width_err_m"] == pytest.approx(3.0)


def test_absolute_target_stays_put():
    s, seg = two_ends_session()
    far_deep = rect("abs", x0=-2.0, y0=8.0, x1=0.0, y1=10.0, frame="absolute")
    rows = pr.evaluate(seg, s.tables()[1], [far_deep]).to_pylist()
    assert [r["in_target"] for r in rows] == [True, False, False, False]
    near_deep = rect("abs2", x0=0.0, y0=-10.0, x1=2.0, y1=-8.0, frame="absolute")
    rows = pr.evaluate(seg, s.tables()[1], [near_deep]).to_pylist()
    assert [r["in_target"] for r in rows] == [False, False, True, False]
    # Errors are measured in the hitter's frame (deeper = +) for absolute targets too.
    assert rows[2]["depth_err_m"] == pytest.approx(0.0)


def test_edits_exclude_and_place_landing():
    s, seg = two_ends_session()
    edits = SessionEdits(
        practice_shots=[
            PracticeShotEdit(t=10.0, exclude=True),
            PracticeShotEdit(t=16.05, landing=[-1.0, 9.0]),
        ]
    )
    target = rect("deep", x0=-2.0, y0=8.0, x1=0.0, y1=10.0)
    rows = pr.evaluate(seg, s.tables()[1], [target], edits).to_pylist()
    assert rows[0]["excluded"] and not rows[1]["excluded"]
    assert rows[1]["landing_source"] == "user" and rows[1]["landing_confirmed"]
    assert rows[1]["in_target"] is True
    summary = pr.summarize(rows)
    assert summary["n"] == 3 and summary["n_target_hits"] == 2


def test_summaries_rolling_and_breakdown():
    s, seg = two_ends_session()
    rows = pr.evaluate(seg, s.tables()[1], [rect("deep", y0=8.0, y1=10.0)]).to_pylist()
    sm = pr.summarize(rows)
    assert sm["n"] == 4 and sm["in_pct"] == 1.0 and sm["target_pct"] == 0.5
    assert sm["depth_sd"] == pytest.approx(2.0)
    roll = pr.rolling(rows, window=10)
    assert roll["target"][:2] == [None, None] and roll["target"][3] == pytest.approx(0.5)
    names = dict(pr.breakdown(rows))
    assert "Groundstroke" in names and "80–110 km/h" in names


def test_no_targets_no_target_results():
    s, seg = two_ends_session()
    rows = pr.evaluate(seg, s.tables()[1], []).to_pylist()
    assert all(r["in_target"] is None and r["targets"] == [] for r in rows)
    assert pr.evaluate(SEGMENTS.empty_table(), None, []).num_rows == 0


# ---------------------------------------------------------------------------
# Edits, labels, scoring
# ---------------------------------------------------------------------------


def test_edits_file_versioning(tmp_path):
    from swingvision.storage.session import Session

    session = Session(tmp_path)
    e = ed.update(session, lambda x: ed.set_practice_shot(x, 12.0, exclude=True))
    assert e.version == 1 and ed.load(session).practice_shots[0].exclude
    with pytest.raises(ed.EditConflict):
        ed.update(session, lambda x: None, expected_version=0)
    ed.update(session, lambda x: ed.set_practice_shot(x, 12.1, confirmed=True), 1)
    assert len(ed.load(session).practice_shots) == 1  # matched within the tolerance
    # Back at the defaults: the edit disappears.
    ed.update(session, lambda x: ed.set_practice_shot(x, 12.0, exclude=False, confirmed=False))
    assert ed.load(session).practice_shots == []


def test_score_segments():
    rows = [
        {"t_contact": 10.0, "side": -1, "start_t": 8.0, "end_t": 12.0, "end_reason": "bounce"},
        {"t_contact": 20.1, "side": -1, "start_t": 19.9, "end_t": 22.0, "end_reason": "net"},
        {"t_contact": 40.0, "side": 1, "start_t": 38.0, "end_t": 42.0, "end_reason": "bounce"},
    ]
    labels = SegmentLabels(
        session_id="x",
        shots=[
            ShotLabel(t=10.2, side=-1),
            ShotLabel(t=20.0, side=-1, end="net"),
            ShotLabel(t=30.0, side=-1),
        ],
    )
    sc = score_segments(rows, labels)
    assert sc.matched == 2 and sc.missed == [30.0] and sc.false == [40.0]
    assert sc.covered == 1 and sc.uncovered == [20.0]  # starts too late to show the swing
    assert sc.accuracy == pytest.approx(1 / 4)
    assert sc.end_agree == 2


def test_practice_rows_schema():
    s, seg = two_ends_session()
    table = pr.evaluate(seg, s.tables()[1], [rect()])
    assert isinstance(table, pa.Table)
    from swingvision.storage.schemas import PRACTICE

    assert table.schema.equals(PRACTICE, check_metadata=False)
