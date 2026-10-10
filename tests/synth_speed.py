"""Synthetic net-tape serves (M7c): flights rendered through a camera whose video clock runs
fast and whose sensor reads its rows over a readout time (rolling shutter), fitted like
``ball_3d``, with the racket and tape sounds synthesized on an accurate audio clock at the
right sound delays.

Time bases. ``T`` is true time (the audio clock, up to a constant). Around each serve (from
``T_ref``, its impact) a video frame stamped ``tv`` exposed sensor readout fraction ``s`` at
``T = T_ref + (tv − T_ref) / clock + τ · s``: the clock error is a time scale, the audio and
video staying in sync over the session as phones keep them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swingvision.ball import physics as ph
from swingvision.ball import speed_refs as sr
from tests.synth_ball import flight_3d
from tests.synth_court import make_camera

CAM = make_camera(
    width=3840, height=2160, hfov_deg=87.5, height_m=3.26, behind_m=6.4, offset_x=0.1,
    look_y=-1.0,
)  # fmt: skip
FPS = 60.0
SR = 22050
CD_TRUE = 0.58
AV_OFFSET = 0.1  # the phone's audio runs this much behind its video
ROTATION = 180  # the phone upside down, like the user's: the sensor reads bottom to top


@dataclass
class SynthServe:
    serve: sr.ServeInput
    t_imp: float  # true times
    t_tape: float
    v_true_avg: float  # contact → tape
    contact: np.ndarray
    tape: np.ndarray


def _readout(px: np.ndarray) -> np.ndarray:
    return sr.readout_fraction(px, ROTATION, CAM.width, CAM.height)


def _aim(contact, speed, x_net, z_net, spin=0.0):
    """Initial velocity of ``speed`` that crosses the net plane at (x_net, z_net)."""
    lo, hi = -0.3, 0.2  # elevation, rad

    def z_at_net(el):
        d = np.array([x_net - contact[0], -contact[1], 0.0])
        d /= np.linalg.norm(d)
        v0 = speed * np.array([d[0] * np.cos(el), d[1] * np.cos(el), np.sin(el)])
        t = np.linspace(0, 1.2, 2401)
        pos, _, _ = flight_3d(contact, v0, 0.0, t, spin=spin, cd=CD_TRUE)
        ok = np.isfinite(pos).all(axis=1)
        k = np.flatnonzero((pos[ok][:-1, 1] < 0) & (pos[ok][1:, 1] >= 0))
        return (pos[ok][k[0], 2] if len(k) else -1.0), v0

    for _ in range(40):
        mid = (lo + hi) / 2
        z, _ = z_at_net(mid)
        lo, hi = (mid, hi) if z < z_net else (lo, mid)
    return z_at_net((lo + hi) / 2)[1]


def _observe(path, t_ref, t_start, t_end, clock: float, tau: float, rng, miss=0.1):
    """Detections (video time, px) of a path(T) → (n, 3) between true times."""
    n0 = int(np.ceil((t_ref + clock * (t_start - t_ref)) * FPS))
    n1 = int(np.floor((t_ref + clock * (t_end - t_ref)) * FPS))
    tv = np.arange(n0, n1 + 1) / FPS
    T = t_ref + (tv - t_ref) / clock
    for _ in range(4):
        px = CAM.project(path(T))
        T = t_ref + (tv - t_ref) / clock + tau * _readout(px)
    pos = path(T)
    ok = np.isfinite(pos).all(axis=1) & (t_start <= T) & (t_end >= T)
    px = CAM.project(pos[ok]) + rng.normal(0, 1.0, (int(ok.sum()), 2))
    tv = tv[ok]
    keep = rng.random(len(tv)) >= miss
    return tv[keep], px[keep]


def _video_time(T: float, P: np.ndarray, t_ref: float, clock: float, tau: float) -> float:
    """The video time at which a point P seen at true time T appears."""
    return t_ref + clock * (T - tau * float(_readout(CAM.project(P[None]))[0]) - t_ref)


def make_serve(i: int, t_imp: float, kind: str, clock: float, tau: float, rng) -> SynthServe:
    contact = np.array(
        [rng.uniform(-0.9, 0.9), -12.1 + rng.normal(0, 0.1), 2.8 + rng.normal(0, 0.08)]
    )
    speed = rng.uniform(130, 190) / 3.6
    x_net = -np.sign(contact[0]) * rng.uniform(0.3, 1.5)  # crosscourt
    z_net = float(ph.net_height(x_net)) + 0.5 * ph.BALL_RADIUS
    v0 = _aim(contact, speed, x_net, z_net)
    tgrid = t_imp + np.linspace(0, 1.5, 6001)
    pos, vel, _ = flight_3d(contact, v0, t_imp, tgrid, cd=CD_TRUE)
    ok = np.isfinite(pos).all(axis=1)
    k = int(np.flatnonzero((pos[ok][:-1, 1] < 0) & (pos[ok][1:, 1] >= 0))[0])
    f = -pos[ok][k, 1] / (pos[ok][k + 1, 1] - pos[ok][k, 1])
    t_tape = float(tgrid[ok][k] + f * (tgrid[1] - tgrid[0]))
    tape = pos[ok][k] + f * (pos[ok][k + 1] - pos[ok][k])
    v_tape = vel[ok][k]
    arc = float(np.linalg.norm(np.diff(pos[ok][: k + 1], axis=0), axis=1).sum())
    arc += float(np.linalg.norm(tape - pos[ok][k]))

    def before(T):
        return flight_3d(contact, v0, t_imp, np.atleast_1d(T), cd=CD_TRUE)[0]

    v_after = v_tape * np.array([0.95, 0.85, 0.3]) + np.array([0.0, 0.0, 0.6])
    t_land_after = None
    if kind == "let":
        _, _, t_land_after = flight_3d(tape, v_after, t_tape, np.array([t_tape]), cd=CD_TRUE)

    def path(T):
        T = np.atleast_1d(T)
        out = before(np.minimum(T, t_tape))
        late = t_tape < T
        if late.any():
            if kind == "let":
                out[late] = flight_3d(tape, v_after, t_tape, T[late], cd=CD_TRUE)[0]
            else:
                out[late] = np.nan  # the ball drops into the net, out of the tracker's view
        return out

    t_end = t_tape if kind == "net" else float(t_land_after) - 0.01
    det_t, det_px = _observe(path, t_imp, t_imp + 1.5 / FPS / clock, t_end, clock, tau, rng)
    t0v = _video_time(t_imp, contact, t_imp, clock, tau) + rng.normal(0, 0.002)
    contact_obs = contact + rng.normal(0, 0.02, 3)
    cov = np.eye(3) * 0.03**2
    start = ph.EndPoint("hit", t0v, None, None, contact_obs, cov)
    if kind == "net":
        end = ph.EndPoint(
            "net", _video_time(t_tape, tape, t_imp, clock, tau), CAM.project(tape[None])[0]
        )
    else:
        land = path(np.array([t_land_after - 1e-6]))[0]
        end = ph.EndPoint(
            "bounce", _video_time(float(t_land_after), land, t_imp, clock, tau),
            CAM.project(land[None])[0] + rng.normal(0, 1.0, 2),
        )  # fmt: skip
    data = ph.FlightData(
        CAM, det_t, det_px, ph.point_sigma(det_px, det_t, CAM.width, ph.FitParams()), start, end
    )
    fit = ph.fit_flight(data)
    sr_cols = ("p0_x", "p0_y", "p0_z", "v0_x", "v0_y", "v0_z", "spin", "cd")
    flight = {
        "flight_id": i, "start_kind": "hit", "end_kind": kind if kind == "net" else "bounce",
        "t0_s": fit.t0, "t1_s": fit.t1, "ok": fit.ok, "rms_px": fit.rms_px,
        **dict(zip(sr_cols, fit.theta, strict=True)),
        "speed_avg": fit.outputs["speed_avg"], "speed_avg_sigma": fit.sigmas["speed_avg"],
    }  # fmt: skip
    t_cf = np.floor(t0v * FPS) / FPS
    serve = sr.ServeInput(
        swing_id=i, shot_id=i, t_contact=float(t_cf), camera=CAM, flight=flight, det_t=det_t,
        det_px=det_px, contact=contact_obs, contact_cov=cov,
    )  # fmt: skip
    return SynthServe(serve, t_imp, t_tape, arc / (t_tape - t_imp), contact, tape)


def _click(t: np.ndarray, t0: float, amp: float, freqs, decay: float) -> np.ndarray:
    s = t - t0
    on = s >= 0
    out = np.zeros_like(t)
    ss = s[on]
    shape = np.exp(-ss / decay) * (1 - np.exp(-ss / 0.00008))
    out[on] = amp * shape * sum(np.sin(2 * np.pi * f * ss + k) for k, f in enumerate(freqs))
    return out / len(freqs)


def render(
    n: int = 10,
    kinds=("net",),
    clock: float = 1.02,
    tau: float = 0.015,
    temp_c: float = 12.0,
    seed: int = 0,
    spacing_s: float = 5.0,
):
    """``n`` serves and their audio (array at :data:`SR`, on the true clock + the offset)."""
    rng = np.random.default_rng(seed)
    serves = [
        make_serve(i, 2.0 + i * spacing_s + rng.uniform(0, 1 / FPS), kinds[i % len(kinds)],
                   clock, tau, rng)
        for i in range(n)
    ]  # fmt: skip
    duration = 2.0 + n * spacing_s + 2.0
    t = np.arange(int(duration * SR)) / SR
    y = rng.normal(0, 0.0015, len(t))
    y += 0.01 * np.sin(2 * np.pi * 90 * t)  # hum (high-passed away)
    c = sr.sound_speed(temp_c)
    C = CAM.center
    for s in serves:
        tr = s.t_imp + np.linalg.norm(s.contact - C) / c + AV_OFFSET
        tt = s.t_tape + np.linalg.norm(s.tape - C) / c + AV_OFFSET
        m = (t > tr - 0.01) & (t < tr + 0.3)
        y[m] += _click(t[m], tr, 0.3, (1200, 2600, 4100), 0.004)
        y[m] += _click(t[m], tr + 0.12, 0.04, (1200, 2600, 4100), 0.006)  # a wall's echo
        m = (t > tt - 0.01) & (t < tt + 0.05)
        y[m] += _click(t[m], tt, 0.06, (3300, 5200, 7500), 0.0015)
    return serves, y.astype(np.float32)
