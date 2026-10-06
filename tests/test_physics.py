"""3D ball flight (M4): integrator, fits on synthetic flights, flights from events, shots.

Synthetic flights come from an independent integrator (``scipy.integrate.solve_ivp``) with
deliberately different physics from the fit's model (drag coefficient 0.58 instead of the
prior's 0.55, and a spin axis that turns with the ball), seen through a camera like the
user's (3.3 m high, 6.4 m behind the baseline, 87° wide) with 1 px detection noise and 10%
missed detections.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from swingvision.analysis import shots as shots_mod
from swingvision.ball import flights as fl
from swingvision.ball import physics as ph
from swingvision.storage.schemas import BALL_TRACK, EVENTS, SHOTS
from tests.synth_ball import flight_3d
from tests.synth_court import make_camera

CAM = make_camera(
    width=3840, height=2160, hfov_deg=87.5, height_m=3.26, behind_m=6.4, offset_x=0.1,
    look_y=-1.0,
)  # fmt: skip
FPS = 60.0
CD_TRUE = 0.58
PRM = ph.FitParams()


def _observe(pos, t, mask, rng, jitter=1.0, miss=0.1):
    px = CAM.project(pos[mask]) + rng.normal(0, jitter, (int(mask.sum()), 2))
    tt = t[mask]
    keep = rng.random(len(tt)) >= miss
    return tt[keep], px[keep]


def _data(tt, px, start, end):
    return ph.FlightData(CAM, tt, px, ph.point_sigma(px, tt, CAM.width, PRM), start, end)


def _outgoing(p0, v0, t0, spin, rng):
    """Observations of a flight from p0 at t0 to its bounce, and the bounce end point."""
    t = np.arange(0, 5, 1 / FPS)
    pos, _, t_land = flight_3d(p0, v0, t0, t, spin=spin, cd=CD_TRUE, follow_spin=True)
    m = np.isfinite(pos).all(axis=1) & (t > t0 + 1.5 / FPS) & (t < t_land)
    tt, px = _observe(pos, t, m, rng)
    land, _, _ = flight_3d(p0, v0, t0, np.array([t_land - 1e-9]), spin=spin, cd=CD_TRUE)
    land_px = CAM.project(land)[0] + rng.normal(0, 1.0, 2)
    return tt, px, ph.EndPoint("bounce", t_land, land_px), land[0]


def _incoming_feed(rng):
    """Self-feed: the ball bounces next to the player and is hit on the way down."""
    t = np.arange(0, 5, 1 / FPS)
    bounce = np.array([0.9, -11.6, ph.BALL_RADIUS]) + np.r_[rng.normal(0, 0.2, 2), 0]
    tb = 0.3 + rng.uniform(0, 1 / FPS)
    t0 = tb + 0.55
    pos, _, _ = flight_3d(bounce, (0.0, 0.2, 4.2), tb, np.r_[t, t0], cd=CD_TRUE)
    m = np.isfinite(pos[:-1]).all(axis=1) & (t > tb + 1 / FPS) & (t < t0 - 1 / FPS)
    tt, px = _observe(pos[:-1], t, m, rng)
    start = ph.EndPoint("bounce", tb, CAM.project(bounce[None])[0] + rng.normal(0, 1, 2))
    return _data(tt, px, start, ph.EndPoint("hit", t0)), pos[-1], t0


def _incoming_toss(rng):
    """Serve toss: released at 1.7 m, hit on the way down at ≈2.7 m."""
    t = np.arange(0, 5, 1 / FPS)
    release = np.array([0.35, -12.0, 1.7])
    tr = 0.2 + rng.uniform(0, 1 / FPS)
    t0 = tr + 0.75
    pos, _, _ = flight_3d(release, (0.05, 0.4, 5.1), tr, np.r_[t, t0], cd=CD_TRUE)
    m = np.isfinite(pos[:-1]).all(axis=1) & (t > tr) & (t < t0 - 1 / FPS)
    tt, px = _observe(pos[:-1], t, m, rng)
    return _data(tt, px, ph.EndPoint("free", float(tt[0])), ph.EndPoint("hit", t0)), pos[-1], t0


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def test_simulate_matches_reference_integrator():
    p0, v0, spin = (0.5, -12.0, 1.0), (-2.0, 30.0, 3.0), 0.007
    t = np.linspace(0.1, 1.2, 12)
    ref, ref_v, _ = flight_3d(p0, v0, 0.1, t, spin=spin, cd=0.55)
    ok = np.isfinite(ref).all(axis=1)
    theta = np.r_[p0, v0, spin, 0.55]
    pos, vel = ph.simulate(theta, 0.1, t[ok])
    assert np.abs(pos[0] - ref[ok]).max() < 1e-6
    assert np.abs(vel[0] - ref_v[ok]).max() < 1e-6
    # Backwards from the end state returns to the start.
    th_end = np.r_[pos[0, -1], vel[0, -1], spin, 0.55]
    back, _ = ph.simulate(th_end, t[ok][-1], np.array([0.1]))
    # (the spin axis is taken from the end velocity, nearly parallel here)
    assert np.abs(back[0, 0] - p0).max() < 0.02


def test_drag_and_magnus_directions():
    theta = np.array([[0, -12, 1, 0, 30, 0, 0.0, 0.55], [0, -12, 1, 0, 30, 0, 0.01, 0.55]])
    pos, vel = ph.simulate(theta, 0.0, np.array([0.5]))
    assert vel[0, 0, 1] < 30 * 0.8  # drag slows a 108 km/h ball by > 20% in 0.5 s
    assert pos[1, 0, 2] < pos[0, 0, 2] - 0.2  # topspin (s > 0) pulls the ball down


def test_net_height():
    assert ph.net_height(0.0) == pytest.approx(0.914)
    assert ph.net_height(10.0) == pytest.approx(1.07)


# ---------------------------------------------------------------------------
# Speed recovery (M4 exit criterion: within 3%)
# ---------------------------------------------------------------------------

NEAR_CASES = {
    # name: (incoming, v0, spin)
    "groundstroke": ("feed", (-2.0, 27.0, 4.2), 0.006),
    "hard drive": ("feed", (-2.5, 34.0, 2.8), 0.008),
    "lob": ("feed", (0.5, 14.0, 9.5), 0.004),
    "slice": ("feed", (1.0, 20.0, 3.8), -0.006),
    "serve": ("toss", (-1.5, 44.0, -4.6), 0.0),
}


@pytest.mark.parametrize("name", list(NEAR_CASES))
def test_near_court_speed_within_3_percent(name):
    incoming, v0, spin = NEAR_CASES[name]
    for seed in range(3):
        rng = np.random.default_rng(seed)
        inc, contact, t0 = (_incoming_feed if incoming == "feed" else _incoming_toss)(rng)
        fit_in = ph.fit_flight(inc, PRM)
        assert fit_in.ok
        assert np.linalg.norm(fit_in.end_pos - contact) < 0.5
        tt, px, end, land = _outgoing(contact, v0, t0, spin, rng)
        # The hitter's feet: beside and behind the contact, tracked to ±15 cm.
        feet = contact[:2] + np.array([-0.7, -0.3]) + rng.normal(0, 0.15, 2)
        start = ph.EndPoint("hit", t0, None, feet, fit_in.end_pos, fit_in.end_cov)
        fit = ph.fit_flight(_data(tt, px, start, end), PRM)
        assert fit.ok, fit.flags
        speed = np.linalg.norm(v0)
        assert fit.outputs["speed0"] == pytest.approx(speed, rel=0.03), (name, seed)
        assert abs(fit.outputs["speed0"] - speed) < 3 * fit.sigmas["speed0"] + 0.01 * speed
        assert (
            np.hypot(*(np.array([fit.outputs["landing_x"], fit.outputs["landing_y"]]) - land[:2]))
            < 0.15
        )


def test_net_clearance_and_apex():
    rng = np.random.default_rng(7)
    inc, contact, t0 = _incoming_feed(rng)
    fit_in = ph.fit_flight(inc, PRM)
    v0, spin = (-2.0, 27.0, 4.2), 0.006
    tt, px, end, _ = _outgoing(contact, v0, t0, spin, rng)
    start = ph.EndPoint("hit", t0, None, contact[:2], fit_in.end_pos, fit_in.end_cov)
    fit = ph.fit_flight(_data(tt, px, start, end), PRM)
    t = np.linspace(t0, end.t, 2000)
    pos, _, _ = flight_3d(contact, v0, t0, t, spin=spin, cd=CD_TRUE, follow_spin=True)
    ok = np.isfinite(pos).all(axis=1)
    k = np.flatnonzero(np.diff(np.sign(pos[ok, 1])))[0]
    true_clear = pos[ok][k, 2] - ph.net_height(pos[ok][k, 0])
    assert fit.outputs["net_clearance"] == pytest.approx(true_clear, abs=0.06)
    assert fit.outputs["apex_z"] == pytest.approx(np.nanmax(pos[:, 2]), abs=0.06)
    assert fit.outputs["spin"] > 0  # topspin seen


@pytest.mark.parametrize(
    ("start_kind", "p0", "v0", "near_sigma", "offset", "tol"),
    [
        # A ball machine at the far baseline, its position known (placed or detected).
        ("machine", (0.0, 11.0, 0.8), (0.3, -20.0, 5.0), 0.1, (0.0, 0.0), 0.03),
        # A far-court hitter with only their feet as the contact prior: depth along the
        # camera axis is weakly determined (≈ ±2% per shot, tolerance 5%).
        ("hit", (-0.5, 12.0, 1.0), (1.0, -25.0, 4.0), 0.3, (0.7, 0.3), 0.05),
    ],
)
def test_far_court_starts(start_kind, p0, v0, near_sigma, offset, tol):
    for seed in range(3):
        rng = np.random.default_rng(seed)
        t0 = 0.3 + rng.uniform(0, 1 / FPS)
        tt, px, end, _ = _outgoing(np.array(p0), v0, t0, 0.005, rng)
        feet = np.array(p0[:2]) + np.array(offset) + rng.normal(0, near_sigma, 2)
        fit = ph.fit_flight(_data(tt, px, ph.EndPoint(start_kind, t0, None, feet), end), PRM)
        assert fit.ok
        assert fit.outputs["speed0"] == pytest.approx(np.linalg.norm(v0), rel=tol)


def test_free_gravity_recovers_g():
    """Bounce → bounce: both ends on the ground, so the metric scale fixes g."""
    rng = np.random.default_rng(3)
    t = np.arange(0, 3, 1 / FPS)
    b0 = np.array([-2.0, 5.0, ph.BALL_RADIUS])
    pos, _, t_land = flight_3d(b0, (0.5, 9.0, 4.5), 0.2, t, cd=CD_TRUE)
    m = np.isfinite(pos).all(axis=1) & (t > 0.2 + 1 / FPS) & (t < t_land)
    tt, px = _observe(pos, t, m, rng)
    land, _, _ = flight_3d(b0, (0.5, 9.0, 4.5), 0.2, np.array([t_land - 1e-9]), cd=CD_TRUE)
    start = ph.EndPoint("bounce", 0.2, CAM.project(b0[None])[0])
    end = ph.EndPoint("bounce", t_land, CAM.project(land)[0])
    fit = ph.fit_flight(_data(tt, px, start, end), PRM, free_g=True)
    assert fit.outputs["g"] == pytest.approx(ph.G, rel=0.03)
    assert fit.sigmas["g"] < 1.0  # far court: a few % per flight


def test_poor_fit_and_few_points_are_flagged():
    rng = np.random.default_rng(0)
    _, contact, t0 = _incoming_feed(rng)
    tt, px, end, _ = _outgoing(contact, (-2.0, 27.0, 4.2), t0, 0.0, rng)
    # A second half that belongs to another path (a missed contact inside the "flight").
    px = px.copy()
    px[len(px) // 2 :] += (
        np.array([0.0, 60.0]) * np.linspace(0, 1, len(px) - len(px) // 2)[:, None] * 3
    )
    fit = ph.fit_flight(_data(tt, px, ph.EndPoint("hit", t0, None, contact[:2]), end), PRM)
    assert not fit.ok and ("poor_fit" in fit.flags or "end_mismatch" in fit.flags)
    few = ph.fit_flight(_data(tt[:3], px[:3], ph.EndPoint("hit", t0), None), PRM)
    assert not few.ok and few.flags == ["few_points"]


# ---------------------------------------------------------------------------
# Flights from a track + events, and shots
# ---------------------------------------------------------------------------


def _track_and_events(rng, fake_hit: bool = False):
    """A self-fed groundstroke: bounce → hit → bounce in the far court → bounce; optionally a
    false "hit" by the near player while the ball is in the far court."""
    t = np.arange(0, 4, 1 / FPS)
    frames = np.arange(len(t))
    bounce0 = np.array([0.9, -11.6, ph.BALL_RADIUS])
    tb = 0.3
    t_hit = tb + 0.55
    pin, _, _ = flight_3d(bounce0, (0.0, 0.2, 4.2), tb, t, cd=CD_TRUE)
    contact, _, _ = flight_3d(bounce0, (0.0, 0.2, 4.2), tb, np.array([t_hit]), cd=CD_TRUE)
    v0 = (-2.0, 27.0, 4.2)
    pout, _, t_land = flight_3d(contact[0], v0, t_hit, t, spin=0.006, cd=CD_TRUE)
    land, vland, _ = flight_3d(
        contact[0], v0, t_hit, np.array([t_land - 1e-9]), spin=0.006, cd=CD_TRUE
    )
    v_after = vland[0] * np.array([0.7, 0.7, -0.75])
    pafter, _, t_land2 = flight_3d(land[0], v_after, t_land, t, cd=CD_TRUE)
    P = np.where((t < t_hit)[:, None], pin, np.where((t < t_land)[:, None], pout, pafter))
    ok = np.isfinite(P).all(axis=1) & (t > 0.1)
    px = np.full((len(t), 2), np.nan)
    px[ok] = CAM.project(P[ok]) + rng.normal(0, 1.0, (int(ok.sum()), 2))
    track = pa.table(
        {
            "frame": pa.array(frames[ok], pa.int64()),
            "t_s": pa.array(t[ok], pa.float64()),
            "x": pa.array(px[ok, 0], pa.float32()),
            "y": pa.array(px[ok, 1], pa.float32()),
            "score": pa.array(np.full(int(ok.sum()), 0.8), pa.float32()),
            "tracklet": pa.array(np.zeros(int(ok.sum())), pa.int32()),
            "source": pa.array(["detected"] * int(ok.sum())),
        },
        schema=BALL_TRACK,
    )
    feet = contact[0, :2] + np.array([-0.7, -0.3])

    def row(kind, tt, P3, hitter=None, court=None):
        f = round(tt * FPS)
        q = CAM.project(np.asarray(P3, float)[None])[0]
        c = court if court is not None else P3[:2]
        return {
            "kind": kind, "frame": f, "t_s": float(tt), "x": float(q[0]), "y": float(q[1]),
            "court_x": float(c[0]), "court_y": float(c[1]), "conf": 0.8, "source": "rules",
            "hitter": hitter, "audio_dt": None, "audio_strength": 0.0,
        }  # fmt: skip

    rows = [
        row("bounce", tb, bounce0),
        row("hit", t_hit, contact[0], "me", feet),
        row("bounce", t_land, land[0]),
    ]
    if t_land2 is not None and t_land2 < 3.9:
        land2, _, _ = flight_3d(land[0], v_after, t_land, np.array([t_land2 - 1e-9]), cd=CD_TRUE)
        rows.append(row("bounce", t_land2, land2[0]))
    if fake_hit:
        k = int(np.searchsorted(t, t_land + 0.25))
        rows.append(row("hit", t[k], P[k], "me", feet))
    rows.sort(key=lambda r: r["t_s"])
    for i, r in enumerate(rows):
        r["event_id"] = i
    events = pa.table(
        {f.name: pa.array([r.get(f.name) for r in rows], f.type) for f in EVENTS}, schema=EVENTS
    )
    return track, events, np.linalg.norm(v0), land[0]


def _ctx():
    return fl.FlightContext(lambda t: CAM, CAM.width, PRM)


def test_plan_flights_from_events():
    track, events, _, _ = _track_and_events(np.random.default_rng(0))
    specs = fl.plan_flights(fl.detected_points(track), events)
    kinds = [(s.start_kind, s.end_kind) for s in specs]
    assert kinds[:3] == [("bounce", "hit"), ("hit", "bounce"), ("bounce", "bounce")]
    hit = specs[1]
    assert hit.hitter == "me" and hit.start_event == 1 and hit.end_event == 2
    # Detections at the event frames are left to the event constraints.
    frames = fl.detected_points(track)["frame"][hit.rows]
    assert frames.min() > events.column("frame")[1].as_py()


def test_fit_session_chains_contact_and_rejects_false_hits():
    track, events, v_true, land = _track_and_events(np.random.default_rng(1), fake_hit=True)
    specs, fits, rejected = fl.fit_session(track, events, _ctx())
    assert rejected == {3}  # the "hit" while the ball was in the far court
    by_start = {s.start_event: (s, f) for s, f in zip(specs, fits, strict=True)}
    _, f = by_start[1]
    assert "contact_from_incoming" in f.flags
    assert f.outputs["speed0"] == pytest.approx(v_true, rel=0.03)
    flights = fl.flights_table(specs, fits)
    paths = fl.paths_table(flights)
    assert paths.num_rows > 0

    shots = shots_mod.assemble_shots("sid", events, flights, lambda t: CAM, rejected)
    assert shots.schema.equals(SHOTS, check_metadata=False)
    assert shots.num_rows == 1
    r = shots.to_pylist()[0]
    assert r["outcome"] == "in" and r["landing_in"] and r["side"] == -1
    assert r["landing_source"] == "bounce"
    assert (r["landing_x"], r["landing_y"]) == pytest.approx(tuple(land[:2]), abs=0.05)
    assert r["speed_racket_kmh"] == pytest.approx(v_true * 3.6, rel=0.03)
    assert r["spin_sign"] in (0, 1)
    assert r["net_clearance_m"] is not None and r["net_clearance_m"] > 0
    assert r["landing_zone"] in ("backcourt", "deuce_box", "ad_box")


@pytest.mark.parametrize(
    ("x", "y", "side", "outcome"),
    [
        (0.0, 5.0, -1, "in"),
        (0.0, 12.5, -1, "out_long"),
        (4.5, 5.0, -1, "out_wide"),
        (0.0, -3.0, -1, "own_side"),
        (0.0, -5.0, 1, "in"),
        (0.0, -12.5, 1, "out_long"),
    ],
)
def test_line_call(x, y, side, outcome):
    got, margin = shots_mod.line_call(x, y, side)
    assert got == outcome
    assert (margin > 0) == (outcome == "in") or outcome == "own_side"


def test_line_call_margin_and_zone():
    _, m = shots_mod.line_call(4.0, 5.0, -1)
    assert m == pytest.approx(4.115 - 4.0)
    _, m = shots_mod.line_call(0.0, 11.985, -1)
    assert m == pytest.approx(-0.1)
    assert shots_mod.landing_zone(-2.0, 3.0, -1) == "deuce_box"
    # A far hitter's deuce box (seen mirrored).
    assert shots_mod.landing_zone(2.0, -3.0, 1) == "deuce_box"


def test_spin_sign():
    assert shots_mod.spin_sign(0.006, 0.001) == 1
    assert shots_mod.spin_sign(-0.006, 0.001) == -1
    assert shots_mod.spin_sign(0.006, 0.004) == 0
    assert shots_mod.spin_sign(None, None) is None


def test_shots_view_pieces():
    """The session page's shot pieces render from real shot/flight tables."""
    from swingvision.app.components import shots_view as sv
    from swingvision.court import calibration as calib

    track, events, _, _ = _track_and_events(np.random.default_rng(2))
    specs, fits, rejected = fl.fit_session(track, events, _ctx())
    flights = fl.flights_table(specs, fits)
    paths = fl.paths_table(flights)
    shots = shots_mod.assemble_shots("sid", events, flights, lambda t: CAM, rejected)
    cal = calib.Calibration(
        source="auto", created_at="2026-10-05T00:00:00", camera=calib.to_params(CAM)
    )
    store = sv.shots_store(shots, paths, cal, CAM.width, CAM.height)
    assert store is not None and len(store["id"]) == 1
    path = np.array(store["path"][0])
    assert len(path) > 10 and (path >= 0).all() and (path <= 100).all()
    assert store["v"][0] == pytest.approx(shots.column("speed_racket_kmh")[0].as_py(), abs=0.5)
    summary = sv.shot_summary(shots)
    assert summary["n"] == 1 and summary["in"] == 1
    sv.shots_card(shots)
    sv.shot_detail(shots.to_pylist()[0])
    y = paths.column("y").to_numpy()
    fig = sv.side_view_figure((y, paths.column("z").to_numpy(), -1))
    assert len(fig.data) == 1
    assert len(sv.landing_traces(shots)) == 1
