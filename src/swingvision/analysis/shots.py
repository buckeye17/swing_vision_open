"""Shot records: one row per hit, from events + fitted flights (PLAN.md §8).

Landing comes from the detected bounce when there is one (its ground point is more precise
than the fitted path's end), otherwise from the fitted path extended to the ground. Line
calls are against the opponent's singles court, measured to the ball center: a center up
to the line's outer edge is in. Calls closer than ``CLOSE_CALL_SIGMAS`` landing σ are
flagged ``close_call``.

Stroke type, serve and swing fields stay empty until M6; segments until M5.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyarrow as pa

from swingvision.court import model as court_model
from swingvision.court.camera import Camera
from swingvision.court.homography import ground_sigma
from swingvision.storage.schemas import SHOTS

KMH = 3.6
#: 1-σ of a bounce's contact pixel (detection, refinement and calibration), at 3840 px.
LANDING_PX_SIGMA = 1.8
CLOSE_CALL_SIGMAS = 2.0
#: Racket speeds with a 1-σ above this share are flagged ``speed_uncertain`` (a stricter
#: share for flights without a detected end: their σ comes from a weaker fit, and on the
#: user's dusk footage those spread far more than their σ says).
SPEED_UNCERTAIN = 0.10
SPEED_UNCERTAIN_OPEN_END = 0.05
#: Spin is reported only beyond this many σ (and this magnitude, 1/m).
SPIN_SIGMAS = 2.0
SPIN_MIN = 0.002
#: Flags that mean the flight's numbers shouldn't be trusted.
BAD_FLIGHT_FLAGS = frozenset({"poor_fit", "few_points"})


def line_call(x: float, y: float, side: int) -> tuple[str, float]:
    """(outcome, margin m) of a landing for a hitter on ``side`` (-1 near, +1 far).

    Margin: distance inside (+) the opponent's singles court or outside it (-).
    """
    yr = -side * y  # positive on the opponent's half
    if yr < 0:
        return "own_side", float(yr)
    long_by = yr - court_model.HALF_LENGTH
    wide_by = abs(x) - court_model.HALF_SINGLES
    if long_by > 0 or wide_by > 0:
        return ("out_long" if long_by >= wide_by else "out_wide"), -float(max(long_by, wide_by))
    return "in", float(min(-long_by, -wide_by))


def landing_zone(x: float, y: float, side: int) -> str | None:
    """Zone in the hitter's frame (opponent's half plain, own half ``near:…``)."""
    if side > 0:
        x, y = -x, -y
    return court_model.zone_at(x, y)


def spin_sign(spin: float | None, sigma: float | None) -> int | None:
    if spin is None or not np.isfinite(spin):
        return None
    sigma = sigma if sigma is not None and np.isfinite(sigma) else np.inf
    if abs(spin) < SPIN_MIN or abs(spin) < SPIN_SIGMAS * sigma:
        return 0
    return 1 if spin > 0 else -1


def _num(v) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def assemble_shots(
    session_id: str,
    events: pa.Table,
    flights: pa.Table,
    camera_at: Callable[[float], Camera],
    rejected_hits: set[int] | None = None,
) -> pa.Table:
    """Shots from hit events (minus ``rejected_hits``, which the 3D fits showed weren't the
    hitter's contact)."""
    rejected_hits = rejected_hits or set()
    ev = events.sort_by("t_s").to_pylist()
    by_id = {e["event_id"]: e for e in ev}
    fl_by_start = {
        f["start_event_id"]: f
        for f in flights.to_pylist()
        if f["start_kind"] in ("hit", "machine") and f["start_event_id"] is not None
    }
    rows = []
    for e in ev:
        if e["kind"] != "hit" or e["event_id"] in rejected_hits:
            continue
        f = fl_by_start.get(e["event_id"])
        flags: list[str] = list(f["flags"]) if f is not None else ["no_flight"]
        good = f is not None and f["ok"] and not (BAD_FLIGHT_FLAGS & set(flags))
        if good:
            contact = (f["p0_x"], f["p0_y"], f["p0_z"])
        else:
            contact = (_num(e["court_x"]), _num(e["court_y"]), None)
        side = None
        if contact[1] is not None:
            side = -1 if contact[1] < 0 else 1
        end_kind = f["end_kind"] if f is not None else "none"
        land_x = land_y = land_sigma = None
        land_src = None
        land_t = None
        if end_kind == "bounce" and f is not None:
            b = by_id.get(f["end_event_id"])
            if b is not None and _num(b["court_x"]) is not None and _num(b["court_y"]) is not None:
                land_x, land_y, land_t = b["court_x"], b["court_y"], b["t_s"]
                land_src = "bounce"
                cam = camera_at(b["t_s"])
                px_sigma = LANDING_PX_SIGMA * cam.width / 3840
                sig = ground_sigma(cam, np.array([[land_x, land_y]]), px_sigma)
                land_sigma = _num(float(sig[0, 2]))
        elif end_kind == "lost" and good and _num(f["landing_x"]) is not None:
            land_x, land_y, land_t = f["landing_x"], f["landing_y"], f["landing_t"]
            land_src = "fit"
            land_sigma = _num(f["landing_sigma"])
            flags.append("landing_extrapolated")
        outcome, margin, zone, land_in = "unknown", None, None, None
        if end_kind == "net":
            outcome = "net"
        elif land_x is not None and side is not None:
            outcome, margin = line_call(land_x, land_y, side)
            zone = landing_zone(land_x, land_y, side)
            land_in = outcome == "in"
            if land_sigma is not None and abs(margin) < CLOSE_CALL_SIGMAS * land_sigma:
                flags.append("close_call")
        if "contact_not_at_hitter" in flags:
            outcome = "unknown"
        v0 = _num(f["speed0"]) if good else None
        v0s = _num(f["speed0_sigma"]) if good else None
        share = SPEED_UNCERTAIN_OPEN_END if end_kind == "lost" else SPEED_UNCERTAIN
        if v0 is not None and (v0s is None or v0s > share * v0):
            flags.append("speed_uncertain")
        t_end = land_t if land_t is not None else (f["t1_s"] if f is not None else None)
        rows.append(
            {
                "shot_id": len(rows),
                "session_id": session_id,
                "hitter": e["hitter"],
                "hit_event_id": e["event_id"],
                "flight_id": f["flight_id"] if f is not None else None,
                "frame_contact": e["frame"],
                "t_contact": e["t_s"],
                "contact_x": _num(contact[0]),
                "contact_y": _num(contact[1]),
                "contact_height": _num(contact[2]),
                "side": side,
                "end_kind": end_kind,
                "landing_x": land_x,
                "landing_y": land_y,
                "landing_sigma_m": land_sigma,
                "landing_source": land_src,
                "landing_in": land_in,
                "landing_margin_m": margin,
                "landing_zone": zone,
                "speed_racket_kmh": None if v0 is None else v0 * KMH,
                "speed_net_kmh": _kmh(f, "speed_net") if good else None,
                "speed_avg_kmh": _kmh(f, "speed_avg") if good else None,
                "speed_bounce_kmh": (
                    _kmh(f, "speed_end") if good and end_kind == "bounce" else None
                ),
                "speed_sigma_kmh": None if v0s is None else v0s * KMH,
                "net_clearance_m": _num(f["net_clearance"]) if good else None,
                "apex_m": _num(f["apex_z"]) if good else None,
                "spin_sign": spin_sign(f["spin"], f["spin_sigma"]) if good else None,
                "flight_time_s": None if t_end is None else float(t_end - e["t_s"]),
                "outcome": outcome,
                "fit_rms_px": _num(f["rms_px"]) if f is not None else None,
                "quality_flags": flags,
            }
        )
    if not rows:
        return SHOTS.empty_table()
    return pa.table(
        {fl.name: pa.array([r.get(fl.name) for r in rows], fl.type) for fl in SHOTS},
        schema=SHOTS,
    )


def _kmh(f: dict, key: str) -> float | None:
    v = _num(f[key])
    return None if v is None else v * KMH
