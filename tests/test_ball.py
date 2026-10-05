"""Ball detection pieces: schedules, motion detector, linking, events, labels, metrics."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import torch

from swingvision.ball import events as ev
from swingvision.ball.detectors import DetectorSpec, make_ball_detector
from swingvision.ball.detectors.motion import align, global_shift, translate
from swingvision.ball.schedule import (
    Window,
    decode_spans,
    merge_windows,
    target_frames,
    with_context,
)
from swingvision.ball.trajectory import LinkParams, PlayerBoxes, link
from swingvision.io.frames import FrameTable
from swingvision.players.detect import Crop
from swingvision.training.evaluate import Counts, match_events, score_ball, tolerance
from swingvision.training.labels import BallClip, BallLabel, EventLabel, LabelStore, interpolate
from tests.synth_ball import Flight, candidates, draw_ball, image_track
from tests.synth_court import make_camera

CAM = make_camera(width=3840, height=2160, hfov_deg=75.0, height_m=3.0)
FPS = 60.0


def _table(n: int, fps: float = FPS) -> FrameTable:
    return FrameTable(
        pts=np.arange(n, dtype=np.int64) * 1000, time_base=1 / (1000 * fps), start_pts=0
    )


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_spec_parsing():
    s = DetectorSpec.parse("unet:ball-v1@1920")
    assert (s.name, s.weights, s.input_px) == ("unet", "ball-v1", 1920)
    assert str(DetectorSpec.parse("motion")) == "motion"
    with pytest.raises(ValueError):
        DetectorSpec.parse("motion@huge")


def test_target_frames_rates_and_context():
    tab = _table(600)
    full = target_frames(tab, [Window(1.0, 2.0)])
    assert len(full) == 60 and full[0] == 60
    sweep = target_frames(tab, [Window(0.0, 10.0, 15.0)])
    assert len(sweep) == 150 and np.all(np.diff(sweep) == 4)
    both = target_frames(tab, [Window(0.0, 10.0, 15.0), Window(1.0, 1.5)])
    assert len(both) == len(np.union1d(sweep, np.arange(60, 90)))
    ctx = with_context(np.array([0, 10, 599]), 2, 600)
    assert set(ctx) == {0, 1, 2, 8, 9, 10, 11, 12, 597, 598, 599}
    spans = decode_spans(np.array([0, 1, 2, 300, 301]), tab.times(), max_gap_s=1.0)
    assert spans == [(0, 2), (300, 301)]


def test_merge_windows():
    w = merge_windows([1.0, 1.2, 5.0], 0.25, 0.35, 10.0)
    assert [(round(x.t0, 2), round(x.t1, 2)) for x in w] == [(0.75, 1.55), (4.75, 5.35)]


# ---------------------------------------------------------------------------
# Motion detector
# ---------------------------------------------------------------------------


def _frames_with_ball(n=9, shift=(0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(100, 3, (360, 640, 3)).clip(0, 255).astype(np.uint8)
    base[200:206, :] = 230  # a court line
    base[:, 300:304] = 230
    frames, truth = [], []
    for i in range(n):
        img = np.roll(base, (round(shift[1] * i), round(shift[0] * i)), axis=(0, 1)).copy()
        x, y = 100 + 21.3 * i, 120 + 9.7 * i - 0.6 * i * i
        draw_ball(img, x, y, r=3.5)
        frames.append(torch.from_numpy(img).permute(2, 0, 1).contiguous())
        truth.append((x, y))
    return frames, truth


def test_motion_detector_finds_ball_subpixel():
    frames, truth = _frames_with_ball()
    det = make_ball_detector("motion", device="cpu")
    det.prepare(None, Crop(0, 0, 640, 360))
    targets = list(range(det.context, len(frames) - det.context))
    out = det.detect(frames, targets)
    for k, cands in zip(targets, out, strict=True):
        assert len(cands), k
        best = cands[np.argmax(cands[:, 2])]
        assert np.hypot(best[0] - truth[k][0], best[1] - truth[k][1]) < 1.0
        assert best[2] > 0.5


def test_motion_detector_ignores_camera_shake():
    """A shaking camera moves the court lines; aligned frames don't report them."""
    frames, truth = _frames_with_ball(shift=(1.0, 1.0))
    det = make_ball_detector("motion", device="cpu")
    det.prepare(None, Crop(0, 0, 640, 360))
    k = 4
    out = det.detect(frames, [k])[0]
    best = out[np.argmax(out[:, 2])]
    assert np.hypot(best[0] - truth[k][0], best[1] - truth[k][1]) < 1.5
    strong = out[out[:, 2] > 0.5]
    assert len(strong) == 1


def test_global_shift_and_translate():
    rng = np.random.default_rng(1)
    ref = torch.nn.functional.avg_pool2d(
        torch.from_numpy(rng.uniform(0, 255, (1, 3, 256, 384))).float(), 9, 1, 4
    )
    true = torch.tensor([[2.3, -1.7]])
    mov = translate(ref, -true)
    s = global_shift(mov, ref)
    assert torch.allclose(s, true, atol=1.0)
    al = align(mov, ref)
    assert (al - ref)[..., 20:-20, 20:-20].abs().mean() < 0.5 * (mov - ref)[
        ..., 20:-20, 20:-20
    ].abs().mean()


# ---------------------------------------------------------------------------
# Linking and events
# ---------------------------------------------------------------------------


def _rally(seed=0, clutter=4, miss=0.05):
    rng = np.random.default_rng(seed)
    n = 150
    frames = np.arange(n)
    t = frames / FPS
    # A groundstroke from the near baseline: hit at 0.1 s, bounce in the far court.
    fl = Flight(0.1, (0.5, -11.0, 1.0), (-1.0, 18.0, 4.5))
    px, bounces, points = image_track(CAM, fl, t)
    # A ball rolling slowly near the net (another ball), and a static one.
    roll = CAM.project(
        np.column_stack([np.linspace(-3, -2.8, n), np.full(n, 2.0), np.full(n, 0.03)])
    )
    cand, fr = candidates(px, t, frames, rng, CAM.width, CAM.height, clutter=clutter, miss=miss,
                          extra_tracks=[roll])  # fmt: skip
    return px, bounces, points, cand, fr, t


def test_link_recovers_flight_among_clutter():
    px, _, _, cand, fr, _t = _rally()
    track, _tracklets = link(cand, fr, CAM.width)
    f = track.column("frame").to_numpy()
    x, y = track.column("x").to_numpy(), track.column("y").to_numpy()
    truth = np.isfinite(px).all(axis=1)
    hit = 0
    for fi, xi, yi in zip(f, x, y, strict=True):
        if truth[fi] and np.hypot(xi - px[fi, 0], yi - px[fi, 1]) < 6:
            hit += 1
    recall = hit / truth.sum()
    precision = hit / len(f)
    assert recall > 0.93, recall
    assert precision > 0.93, precision
    # Missed detections inside the flight are filled.
    assert "interp" in set(track.column("source").to_pylist())


def test_link_downweights_player_box():
    _px, _, _, cand, fr, t = _rally(clutter=0, miss=0.0)
    # A "limb" moving smoothly inside a player box for the whole clip.
    n = len(t)
    limb = np.column_stack([1500 + 2 * np.arange(n), 1400 + np.sin(np.arange(n) / 5) * 30])
    rows = cand.to_pylist() + [
        {
            "frame": int(i),
            "t_s": float(t[i]),
            "x": float(limb[i, 0]),
            "y": float(limb[i, 1]),
            "score": 0.9,
        }
        for i in range(n)
    ]
    cand2 = pa.Table.from_pylist(rows, schema=cand.schema).sort_by("frame")
    boxes = PlayerBoxes(t, np.tile([1400.0, 1100.0, 1900.0, 1700.0], (n, 1)))
    track, _ = link(cand2, fr, CAM.width, player_boxes=boxes)
    xs = track.column("x").to_numpy()
    assert not np.any((xs > 1450) & (xs < 1850) & (track.column("y").to_numpy() > 1300))


def _ctx(players=None) -> ev.EventContext:
    return ev.EventContext(camera_at=lambda t: CAM, width=CAM.width, players=players)


def test_bounce_event_and_ground_position():
    errs = []
    for seed in range(5):
        _, bounces, points, cand, fr, _ = _rally(seed=seed, clutter=0, miss=0.0)
        track, _ = link(cand, fr, CAM.width, LinkParams())
        events, feats = ev.detect_events(track, _ctx())
        b = events.filter(pa.compute.equal(events.column("kind"), "bounce"))
        assert b.num_rows >= 1, feats
        assert abs(b.column("t_s").to_numpy()[0] - bounces[0]) < 1.5 / FPS
        errs.append(
            np.hypot(
                b.column("court_x").to_numpy()[0] - points[0][0],
                b.column("court_y").to_numpy()[0] - points[0][1],
            )
        )
    # 0.7 px of detection jitter; one pixel is ≈5 cm along the court where it lands.
    assert np.median(errs) < 0.15, errs


def test_hit_next_to_player():
    rng = np.random.default_rng(3)
    n = 120
    frames = np.arange(n)
    t = frames / FPS
    hit_t = 0.8
    contact = np.array([0.3, -11.5, 0.9])
    # Incoming: a straight line from mid-court down to the contact point; outgoing: a flight.
    start = np.array([-0.5, 0.0, 2.0])
    a = np.clip(t / hit_t, 0, 1)[:, None]
    X_in = start + a * (contact - start)
    X_out, _, _ = Flight(hit_t, tuple(contact), (-1.5, 25.0, 4.0)).positions(t)
    X = np.where((t < hit_t)[:, None], X_in, X_out)
    px = CAM.project(X)
    cand, fr = candidates(px, t, frames, rng, CAM.width, CAM.height, clutter=0, miss=0.0)
    track, _ = link(cand, fr, CAM.width)
    feet = CAM.project(np.array([[0.6, -11.6, 0.0]]))[0]
    head = CAM.project(np.array([[0.6, -11.6, 1.8]]))[0]
    h = feet[1] - head[1]
    box = np.array([feet[0] - 0.35 * h, head[1], feet[0] + 0.35 * h, feet[1]])
    players = PlayerBoxes(t, np.tile(box, (n, 1)))
    events, _ = ev.detect_events(track, _ctx(players))
    hits = events.filter(pa.compute.equal(events.column("kind"), "hit"))
    assert hits.num_rows >= 1
    assert abs(hits.column("t_s").to_numpy()[0] - hit_t) < 2 / FPS


def test_refine_contact_intersects_fits():
    t = np.arange(-5, 6) / FPS
    x = np.where(t < 0, -300 * t, 200 * t) + 100
    y = np.where(t < 0, 500 * t + 50, -400 * t + 50)
    tc, xc, yc = ev.refine_contact(t, x, y, 5)
    assert abs(tc) < 0.3 / FPS and abs(xc - 100) < 1 and abs(yc - 50) < 1


# ---------------------------------------------------------------------------
# Labels and metrics
# ---------------------------------------------------------------------------


def _clip(**kw) -> BallClip:
    return BallClip(session_id="s1", clip_id="gt0001", frame0=0, frame1=20, t0_s=0, t1_s=1, **kw)


def test_interpolate_quadratic_and_events():
    c = _clip()

    def xy(f):
        return 10 + 3 * f, 100 - 4 * f + 0.25 * f * f

    for f in (0, 5, 10):
        c.set(f, BallLabel(x=xy(f)[0], y=xy(f)[1]))
    assert interpolate(c) == 8
    for f in (2, 7):
        lab = c.label(f)
        assert lab.src == "interp"
        assert abs(lab.x - xy(f)[0]) < 0.05 and abs(lab.y - xy(f)[1]) < 0.05
    # An event strictly between keyframes blocks interpolation across it.
    c2 = _clip(events=[EventLabel(frame=17, kind="bounce")])
    for f in (10, 15, 20):
        c2.set(f, BallLabel(x=f, y=f))
    interpolate(c2)
    assert c2.label(12) is not None and c2.label(18) is None


def test_label_store_roundtrip(tmp_path):
    store = LabelStore(tmp_path)
    c = store.new_clip("s1", 100, 160, 1.0, 2.0, kind="gt", split="test")
    c.set(110, BallLabel(x=1.0, y=2.0))
    c.set(111, BallLabel(vis="none"))
    store.save(c)
    back = store.get("s1", c.clip_id)
    assert back.label(110).x == 1.0 and back.label(111).vis == "none"
    tab = store.table()
    assert tab.num_rows == 2 and set(tab.column("vis").to_pylist()) == {"visible", "none"}
    assert store.new_clip("s1", 0, 10, 0, 0.2).clip_id == "gt0002"
    with pytest.raises(ValueError):
        store.clip_path("s1", "../evil")


def test_score_ball_counts():
    c = _clip()
    far = CAM.project(np.array([[0.0, 10.0, 0.0]]))[0]
    near = CAM.project(np.array([[0.0, -10.0, 0.0]]))[0]
    c.set(1, BallLabel(x=far[0], y=far[1]))
    c.set(2, BallLabel(x=near[0], y=near[1]))
    c.set(3, BallLabel(x=near[0], y=near[1]))
    c.set(4, BallLabel(vis="none"))
    c.set(5, BallLabel(vis="occluded"))
    pred = {
        1: (far[0] + 2, far[1]),  # within 4 px
        2: (near[0] + 3, near[1]),  # near ball is big: within tolerance
        3: (near[0] + 200, near[1]),  # wrong place: FP + FN
        4: (10.0, 10.0),  # no ball: FP
        5: (0.0, 0.0),  # occluded: not scored
    }
    s = score_ball([c], {"s1/gt0001": pred}, {"s1": CAM})
    assert (s.far.tp, s.far.fp, s.far.fn) == (1, 0, 0)
    assert s.near.tp == 1 and s.near.fn == 1 and s.all.fp == 2
    assert tolerance(np.array([2.0, 100.0])).tolist() == [4.0, 40.0]
    assert isinstance(s.all, Counts)


def test_match_events_one_to_one():
    assert match_events([10, 20], [11, 12, 30]) == [(0, 0)]
    assert sorted(match_events([10, 12], [11, 12])) == [(0, 0), (1, 1)]


def test_pareto_front():
    from swingvision.training.bench_ball import pareto

    front = pareto(np.array([1.0, 2.0, 3.0, 2.0]), np.array([0.5, 0.8, 0.7, 0.6]))
    assert front.tolist() == [True, True, False, False]


def test_unet_forward_shape():
    from swingvision.ball.detectors.unet import build_model, pad_to

    m = build_model(width=8).eval()
    x = pad_to(torch.zeros(2, 9, 100, 170))
    with torch.inference_mode():
        y = m(x)
    assert y.shape == (2, x.shape[-2] // 2, x.shape[-1] // 2)


def _feats(**kw):
    base = {
        "near_player": 0.0, "box_rel_y": np.nan, "in_net": 0.0, "speed_ratio": 0.9,
        "dvy": -800.0, "vy_before": 600.0, "turn": 1.8, "audio_strength": 0.0,
    }  # fmt: skip
    return {**base, **kw}


@pytest.mark.parametrize(
    ("kw", "kind"),
    [
        ({}, "bounce"),
        ({"in_net": 1.0}, "bounce"),  # bounce right behind the net, seen through the mesh
        ({"in_net": 1.0, "speed_ratio": 0.2}, "net"),  # stopped by the net
        ({"near_player": 1.0, "box_rel_y": 0.4}, "hit"),
        ({"near_player": 1.0, "box_rel_y": 0.95}, "bounce"),  # at the player's feet
        ({"dvy": 300.0, "vy_before": -500.0, "turn": 0.3}, None),
    ],
)
def test_event_rules(kw, kind):
    assert ev.classify_rules(_feats(**kw), ev.EventParams())[0] == kind


def test_machine_feed_is_a_machine_hit():
    rng = np.random.default_rng(5)
    n = 90
    frames = np.arange(n)
    t = frames / FPS
    machine = (1.5, 11.0)
    feed = Flight(0.2, (machine[0], machine[1], 0.6), (-0.5, -16.0, 3.0))
    px, _, _ = image_track(CAM, feed, t)
    cand, fr = candidates(px, t, frames, rng, CAM.width, CAM.height, clutter=0, miss=0.0)
    track, _ = link(cand, fr, CAM.width)
    ctx = ev.EventContext(camera_at=lambda t: CAM, width=CAM.width, machine_xy=machine)
    events, _ = ev.detect_events(track, ctx)
    feeds = [r for r in events.to_pylist() if r["hitter"] == "machine"]
    assert len(feeds) == 1 and feeds[0]["kind"] == "hit"
    assert abs(feeds[0]["t_s"] - 0.2) < 3 / FPS
