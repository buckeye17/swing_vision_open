"""Synthetic ball flights through a known camera (for detection, linking and event tests)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pyarrow as pa

from swingvision.court.camera import Camera
from swingvision.storage.schemas import BALL_CANDIDATES, BALL_FRAMES

G = 9.81
RADIUS = 0.0335


@dataclass
class Flight:
    """A ball hit at ``t_hit`` from ``p0`` with velocity ``v0``; bounces with restitution."""

    t_hit: float
    p0: tuple[float, float, float]
    v0: tuple[float, float, float]
    restitution: float = 0.7
    friction: float = 0.75

    def positions(self, t: np.ndarray) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
        """(n, 3) positions (NaN before the hit), bounce times, bounce ground points."""
        out = np.full((len(t), 3), np.nan)
        p = np.array(self.p0, float)
        v = np.array(self.v0, float)
        t0 = self.t_hit
        bounces, points = [], []
        segments = []
        for _ in range(4):
            # Time until z = RADIUS: p_z + v_z s - g s²/2 = RADIUS.
            a, b, c = -G / 2, v[2], p[2] - RADIUS
            s = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
            segments.append((t0, t0 + s, p.copy(), v.copy()))
            p = p + v * s + np.array([0, 0, -G / 2 * s * s])
            v = np.array(
                [v[0] * self.friction, v[1] * self.friction, -(v[2] - G * s) * self.restitution]
            )
            t0 += s
            bounces.append(t0)
            points.append(p.copy())
        for ta, tb, pa_, va in segments:
            m = (t >= ta) & (t < tb)
            s = t[m] - ta
            out[m] = pa_ + va * s[:, None] + np.outer(-G / 2 * s * s, [0, 0, 1])
        return out, bounces, points


def image_track(cam: Camera, flight: Flight, t: np.ndarray):
    """Image positions (n, 2) (NaN where no ball / off-screen), bounce times, ground points."""
    X, bounces, points = flight.positions(t)
    px = np.full((len(t), 2), np.nan)
    ok = np.isfinite(X).all(axis=1)
    if ok.any():
        px[ok] = cam.project(X[ok])
    off = (px[:, 0] < 0) | (px[:, 0] >= cam.width) | (px[:, 1] < 0) | (px[:, 1] >= cam.height)
    px[off] = np.nan
    return px, bounces, points


def candidates(
    px: np.ndarray,
    t: np.ndarray,
    frames: np.ndarray,
    rng: np.random.Generator,
    width: int,
    height: int,
    jitter: float = 0.7,
    miss: float = 0.05,
    clutter: int = 4,
    extra_tracks: list[np.ndarray] | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Detector-like candidates: the ball (jittered, sometimes missed) plus random clutter and
    other moving objects (``extra_tracks``: (n, 2) arrays with NaN where absent)."""
    rows = []
    for k, (f, ti) in enumerate(zip(frames, t, strict=True)):
        if np.isfinite(px[k]).all() and rng.random() > miss:
            x, y = px[k] + rng.normal(0, jitter, 2)
            rows.append((int(f), float(ti), x, y, rng.uniform(0.6, 0.9)))
        for _ in range(rng.poisson(clutter)):
            rows.append(
                (
                    int(f),
                    float(ti),
                    rng.uniform(0, width),
                    rng.uniform(0, height),
                    rng.uniform(0.3, 0.8),
                )
            )
        for tr in extra_tracks or []:
            if np.isfinite(tr[k]).all():
                x, y = tr[k] + rng.normal(0, jitter, 2)
                rows.append((int(f), float(ti), x, y, rng.uniform(0.5, 0.8)))
    cols = list(zip(*rows, strict=True))
    cand = pa.table(
        {fl.name: pa.array(c, fl.type) for fl, c in zip(BALL_CANDIDATES, cols, strict=True)},
        schema=BALL_CANDIDATES,
    )
    fr = pa.table(
        {"frame": pa.array(frames, pa.int64()), "t_s": pa.array(t, pa.float64())},
        schema=BALL_FRAMES,
    )
    return cand, fr


def draw_ball(img: np.ndarray, x: float, y: float, r: float = 4.0, color=(225, 235, 90)) -> None:
    """Anti-aliased ball (RGB image, in place) at sub-pixel (x, y)."""
    s = 16
    cv2.circle(img, (round(x * s), round(y * s)), round(r * s), color, -1, cv2.LINE_AA, shift=4)


def flight_3d(
    p0, v0, t0: float, t: np.ndarray, spin: float = 0.0, cd: float = 0.55, follow_spin=False
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Reference flight with drag and Magnus by ``scipy.integrate.solve_ivp`` (independent of
    :func:`swingvision.ball.physics.simulate`), until the ball comes down to ball height.

    ``follow_spin`` turns the spin axis with the horizontal velocity (the fit keeps it fixed).
    Returns positions and velocities at ``t`` (NaN outside the flight) and the landing time.
    """
    from scipy.integrate import solve_ivp

    from swingvision.ball import physics

    k = 0.5 * physics.AIR_DENSITY * cd * np.pi * physics.BALL_RADIUS**2 / physics.BALL_MASS
    e0 = physics.spin_axis(np.asarray(v0, float)[None])[0]

    def rhs(_t, y):
        v = y[3:]
        e = physics.spin_axis(v[None])[0] if follow_spin else e0
        sp = np.linalg.norm(v)
        a = -k * sp * v + spin * sp * np.cross(e, v) - np.array([0, 0, G])
        return np.r_[v, a]

    def ground(_t, y):
        return y[2] - RADIUS

    ground.terminal, ground.direction = True, -1
    sol = solve_ivp(
        rhs, (t0, t0 + 6), np.r_[p0, v0], events=ground, dense_output=True, rtol=1e-10, atol=1e-10
    )
    t_land = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    end = t_land if t_land is not None else sol.t[-1]
    pos = np.full((len(t), 3), np.nan)
    vel = np.full((len(t), 3), np.nan)
    m = (t >= t0) & (t <= end)
    if m.any():
        y = sol.sol(t[m])
        pos[m], vel[m] = y[:3].T, y[3:].T
    return pos, vel, t_land
