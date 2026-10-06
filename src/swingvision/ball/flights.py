"""The ball track cut into flights between events, each fitted in 3D (PLAN.md §7.6).

Every hit and bounce starts a flight that runs to the next event (bounce, net, hit) or,
when the detections stop for longer than :data:`LOST_GAP_S`, to the last detection
("lost"). A hit whose incoming flight wasn't tracked from an event (a serve toss, a dropped
feed whose bounce was missed) gets a *free* flight from the start of its run.

Flights are fitted in time order so a hit can use where its incoming flight ended as the
contact point (with that fit's uncertainty), on top of the hitter's tracked feet.
Detections at the event frames themselves are left to the event constraints: at a hit the
ball is often on the racket.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.ball import physics as ph
from swingvision.court.camera import Camera
from swingvision.storage.schemas import BALL_FLIGHT_PATHS, BALL_FLIGHTS

#: A flight is at most this long (a high lob is ≈ 3 s).
FLIGHT_MAX_S = 4.0
#: Detections further apart than this end a flight: the ball was lost.
LOST_GAP_S = 0.35
#: A free flight (into a hit nothing tracked the ball into) uses at most this much of the
#: run before the hit: earlier, a toss is still in the hand or a feed still bouncing.
FREE_MAX_S = 0.8
#: A fitted contact further than this from the hitter's tracked feet isn't their hit (a
#: ball near the back fence seen inside the near player's image box).
CONTACT_MAX_M = 2.5
#: Points drawn per second of a fitted path (``ball/flight_paths.parquet``).
PATH_HZ = 30.0


#: ``ball/flights.parquet`` columns holding the fitted parameters (physics.py order).
PARAM_COLS = ("p0_x", "p0_y", "p0_z", "v0_x", "v0_y", "v0_z", "spin", "cd")


@dataclass
class FlightSpec:
    flight_id: int
    start_kind: str  # hit | machine | bounce | free
    start_event: int | None
    end_kind: str  # bounce | net | hit | lost
    end_event: int | None
    t0: float
    t1: float
    hitter: str | None
    rows: np.ndarray  # indices into the detected points


@dataclass
class _Events:
    id: np.ndarray
    kind: np.ndarray
    frame: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    hitter: list

    @classmethod
    def from_table(cls, events: pa.Table) -> _Events:
        ev = events.sort_by("t_s")

        def f(name):
            return ev.column(name).to_numpy(zero_copy_only=False).astype(np.float64)

        return cls(
            ev.column("event_id").to_numpy().astype(np.int64),
            np.array(ev.column("kind").to_pylist(), dtype=object),
            ev.column("frame").to_numpy().astype(np.int64),
            f("t_s"),
            f("x"),
            f("y"),
            f("court_x"),
            f("court_y"),
            ev.column("hitter").to_pylist(),
        )


def detected_points(track: pa.Table) -> dict[str, np.ndarray]:
    src = np.array(track.column("source").to_pylist())
    det = src == "detected"
    return {
        "frame": track.column("frame").to_numpy()[det],
        "t": track.column("t_s").to_numpy()[det],
        "x": track.column("x").to_numpy()[det].astype(np.float64),
        "y": track.column("y").to_numpy()[det].astype(np.float64),
    }


def _until_gap(t: np.ndarray, rows: np.ndarray, t_start: float) -> tuple[np.ndarray, bool]:
    """Rows up to the first gap > LOST_GAP_S (counted from ``t_start``); whether cut."""
    if not len(rows):
        return rows, True
    tt = np.r_[t_start, t[rows]]
    gaps = np.flatnonzero(np.diff(tt) > LOST_GAP_S)
    if len(gaps):
        return rows[: gaps[0]], True
    return rows, False


def plan_flights(pts: dict[str, np.ndarray], events: pa.Table) -> list[FlightSpec]:
    """Flights from every hit/bounce (plus free flights into otherwise untracked hits)."""
    ev = _Events.from_table(events)
    t, frame = pts["t"], pts["frame"]
    specs: list[FlightSpec] = []
    reached: set[int] = set()  # hit events an event-started flight ran into
    for i in range(len(ev.t)):
        kind = ev.kind[i]
        if kind not in ("hit", "bounce"):
            continue
        t0 = ev.t[i]
        j = i + 1 if i + 1 < len(ev.t) and ev.t[i + 1] - t0 <= FLIGHT_MAX_S else None
        f_end = ev.frame[j] if j is not None else np.iinfo(np.int64).max
        t_end = ev.t[j] if j is not None else t0 + FLIGHT_MAX_S
        rows = np.flatnonzero((frame > ev.frame[i]) & (frame < f_end) & (t < t_end))
        rows, cut = _until_gap(t, rows, t0)
        if j is not None and not cut and len(rows) and ev.t[j] - t[rows[-1]] > LOST_GAP_S:
            cut = True
        if not len(rows):
            continue
        if kind == "hit":
            start_kind = "machine" if ev.hitter[i] == "machine" else "hit"
        else:
            start_kind = "bounce"
        if j is not None and not cut:
            end_kind, end_event, t1 = str(ev.kind[j]), int(ev.id[j]), float(ev.t[j])
            if end_kind == "hit":
                reached.add(int(ev.id[j]))
        else:
            end_kind, end_event, t1 = "lost", None, float(t[rows[-1]])
        specs.append(
            FlightSpec(
                len(specs), start_kind, int(ev.id[i]), end_kind, end_event, float(t0), t1,
                ev.hitter[i] if kind == "hit" else None, rows,
            )
        )  # fmt: skip
    # Free flights into hits nothing tracked the ball into (toss, missed feed bounce).
    for i in range(len(ev.t)):
        if ev.kind[i] != "hit" or int(ev.id[i]) in reached or ev.hitter[i] == "machine":
            continue
        f_prev = ev.frame[i - 1] if i > 0 else -1
        rows = np.flatnonzero((frame > f_prev) & (frame < ev.frame[i]) & (t > ev.t[i] - FREE_MAX_S))
        if not len(rows):
            continue
        # Walk back from the hit to the last gap.
        tt = np.r_[t[rows], ev.t[i]]
        gaps = np.flatnonzero(np.diff(tt) > LOST_GAP_S)
        if len(gaps):
            if gaps[-1] == len(rows) - 1:
                continue  # the ball wasn't seen right before the hit
            rows = rows[gaps[-1] + 1 :]
        specs.append(
            FlightSpec(
                len(specs), "free", None, "hit", int(ev.id[i]), float(t[rows[0]]),
                float(ev.t[i]), None, rows,
            )
        )  # fmt: skip
    specs.sort(key=lambda s: s.t0)
    for k, s in enumerate(specs):
        s.flight_id = k
    return specs


@dataclass
class FlightContext:
    camera_at: Callable[[float], Camera]
    width: int
    params: ph.FitParams


def _endpoint(ev: _Events, event_id: int, kind: str, by_id: dict[int, int]) -> ph.EndPoint:
    k = by_id[event_id]
    near = None
    if np.isfinite(ev.cx[k]) and np.isfinite(ev.cy[k]):
        near = np.array([ev.cx[k], ev.cy[k]])
    return ph.EndPoint(kind, float(ev.t[k]), np.array([ev.x[k], ev.y[k]]), near)


def fit_flights(
    specs: list[FlightSpec],
    pts: dict[str, np.ndarray],
    events: pa.Table,
    ctx: FlightContext,
    progress: Callable[[float], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
    cache: dict | None = None,
) -> list[ph.FlightFit]:
    """Fit every flight in time order, chaining contact points into hits.

    ``cache`` (keyed by the flight's ends, detections and contact prior) lets a re-plan
    refit only the flights that changed.
    """
    ev = _Events.from_table(events)
    by_id = {int(e): k for k, e in enumerate(ev.id)}
    end_at: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    fits: list[ph.FlightFit] = []
    for n, s in enumerate(specs):
        if check_cancel is not None and n % 20 == 0:
            check_cancel()
        cam = ctx.camera_at(s.t0)
        if s.start_kind == "free":
            start = ph.EndPoint("free", s.t0)
        else:
            assert s.start_event is not None
            start = _endpoint(ev, s.start_event, s.start_kind, by_id)
            if s.start_kind in ("hit", "machine"):
                start.px = None
                incoming = end_at.get(s.start_event)
                if incoming is not None:
                    start.contact, start.contact_cov = incoming
        end = None
        if s.end_kind != "lost":
            assert s.end_event is not None
            end = _endpoint(ev, s.end_event, s.end_kind, by_id)
            end.near_xy = None
            if s.end_kind == "hit":
                end.px = None  # its time may be audio-refined; the pixel is from the video
        px = np.column_stack([pts["x"][s.rows], pts["y"][s.rows]])
        tt = pts["t"][s.rows]
        data = ph.FlightData(cam, tt, px, ph.point_sigma(px, tt, ctx.width, ctx.params), start, end)
        key = (
            s.start_kind, s.start_event, s.end_kind, s.end_event, s.rows.tobytes(),
            None if start.contact is None else start.contact.tobytes(),
        )  # fmt: skip
        if cache is not None and key in cache:
            fits.append(cache[key])
            _remember_end(s, cache[key], end_at)
            continue
        fit = ph.fit_flight(data, ctx.params)
        if cache is not None:
            cache[key] = fit
        if s.start_kind in ("hit", "machine"):
            fit.flags.append(
                "contact_from_incoming" if start.contact is not None else "contact_from_feet"
            )
            if (
                start.near_xy is not None
                and np.isfinite(fit.theta).all()
                and np.hypot(*(fit.theta[:2] - start.near_xy)) > CONTACT_MAX_M
            ):
                fit.flags.append("contact_not_at_hitter")
        fits.append(fit)
        _remember_end(s, fit, end_at)
        if progress is not None and n % 10 == 0:
            progress((n + 1) / max(len(specs), 1))
    return fits


def _remember_end(s: FlightSpec, fit: ph.FlightFit, end_at: dict) -> None:
    if (
        s.end_kind == "hit"
        and fit.ok
        and fit.end_pos is not None
        and fit.end_cov is not None
        and np.isfinite(fit.end_cov).all()
    ):
        end_at[int(s.end_event)] = (fit.end_pos, fit.end_cov)


def false_hits(specs: list[FlightSpec], fits: list[ph.FlightFit]) -> set[int]:
    """Hits whose own flight fits well but starts far from the hitter: not their contact
    (typically a far ball seen inside the near player's image box)."""
    out = set()
    for s, f in zip(specs, fits, strict=True):
        if (
            s.start_kind == "hit"
            and "contact_not_at_hitter" in f.flags
            and not ({"poor_fit", "few_points", "end_mismatch"} & set(f.flags))
        ):
            out.add(int(s.start_event))
    return out


def fit_session(
    track: pa.Table,
    events: pa.Table,
    ctx: FlightContext,
    progress: Callable[[float], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
    max_rounds: int = 3,
) -> tuple[list[FlightSpec], list[ph.FlightFit], set[int]]:
    """Plan and fit all flights. Hits the fits reject are dropped and the flights around
    them re-planned (only changed flights are refitted). Also returns the rejected hits."""
    pts = detected_points(track)
    rejected: set[int] = set()
    cache: dict = {}
    specs: list[FlightSpec] = []
    fits: list[ph.FlightFit] = []
    for rnd in range(max_rounds):
        used = events
        if rejected:
            ids = pa.array(sorted(rejected), pa.int32())
            used = events.filter(pc.invert(pc.is_in(events.column("event_id"), ids)))
        specs = plan_flights(pts, used)

        def prog(f, rnd=rnd):
            if progress is not None:
                progress(f if rnd == 0 else 0.9 + 0.1 * f)

        fits = fit_flights(specs, pts, used, ctx, prog, check_cancel, cache)
        new = false_hits(specs, fits) - rejected
        if not new:
            break
        rejected |= new
    return specs, fits, rejected


def _f(d: dict, k: str) -> float | None:
    v = d.get(k)
    return None if v is None or not np.isfinite(v) else float(v)


def flights_table(specs: list[FlightSpec], fits: list[ph.FlightFit]) -> pa.Table:
    rows = []
    for s, f in zip(specs, fits, strict=True):
        o, sg = f.outputs, f.sigmas
        th = f.theta if np.isfinite(f.theta).all() else np.full(ph.N_BASE, np.nan)
        rows.append(
            {
                "flight_id": s.flight_id,
                "start_kind": s.start_kind,
                "start_event_id": s.start_event,
                "end_kind": s.end_kind,
                "end_event_id": s.end_event,
                "hitter": s.hitter,
                "t0_s": s.t0,
                "t1_s": f.t1,
                "n_obs": f.n_obs,
                "rms_px": _f({"v": f.rms_px}, "v"),
                "chi2": _f({"v": f.chi2_dof}, "v"),
                "ok": bool(f.ok and np.isfinite(th).all()),
                **{n: _f(dict(zip(PARAM_COLS, th, strict=False)), n) for n in PARAM_COLS},
                **{n: _f(o, n) for n in ph.OUTPUT_NAMES},
                "speed0_sigma": _f(sg, "speed0"),
                "speed_avg_sigma": _f(sg, "speed_avg"),
                "net_clearance_sigma": _f(sg, "net_clearance"),
                "apex_sigma": _f(sg, "apex_z"),
                "landing_sigma": _landing_sigma(sg),
                "spin_sigma": _f(sg, "spin"),
                "flags": f.flags,
            }
        )
    if not rows:
        return BALL_FLIGHTS.empty_table()
    return pa.table(
        {fl.name: pa.array([r.get(fl.name) for r in rows], fl.type) for fl in BALL_FLIGHTS},
        schema=BALL_FLIGHTS,
    )


def _landing_sigma(sg: dict) -> float | None:
    a, b = _f(sg, "landing_x"), _f(sg, "landing_y")
    return None if a is None or b is None else float(np.hypot(a, b))


def theta_of(row: dict) -> np.ndarray:
    return np.array([row[c] for c in PARAM_COLS], dtype=np.float64)


def paths_table(flights: pa.Table, hz: float = PATH_HZ) -> pa.Table:
    """Sampled 3D positions of every fitted flight (for drawing)."""
    fid, ts, xs, ys, zs = [], [], [], [], []
    for r in flights.to_pylist():
        if not r["ok"]:
            continue
        g, P = ph.path_points(theta_of(r), r["t0_s"], r["t1_s"], hz)
        fid.append(np.full(len(g), r["flight_id"]))
        ts.append(g)
        xs.append(P[:, 0])
        ys.append(P[:, 1])
        zs.append(P[:, 2])
    if not fid:
        return BALL_FLIGHT_PATHS.empty_table()
    cols = [np.concatenate(c) for c in (fid, ts, xs, ys, zs)]
    return pa.table(
        {fl.name: pa.array(c, fl.type) for fl, c in zip(BALL_FLIGHT_PATHS, cols, strict=True)},
        schema=BALL_FLIGHT_PATHS,
    )
