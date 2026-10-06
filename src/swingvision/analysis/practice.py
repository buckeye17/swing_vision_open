"""Practice accuracy (PLAN.md §7.10): line calls, targets, and block/session summaries.

**Hitter's frame**: every landing is also expressed as if the hitter stood at the near end
(camera side) hitting into the far half: a shot from the far end is rotated by 180°. In it
``+y`` is deeper and ``+x`` is the hitter's right, so depth and width errors read the same
from both ends.

**Targets** are drawn on the court diagram. ``relative`` targets (the default) are drawn in
the hitter's frame and follow the player when they change ends; ``absolute`` targets stay
where they were drawn. A target with a stroke list only applies to those strokes: the stroke
of the shot's swing (classified from the pose, M6), or ``serve`` for a serve without one.

**Calls**: serves against the service box diagonal from the server's side (deuce/ad), other
shots against the opponent's singles court, both with the ball center allowed up to the
line's outer edge (as in :mod:`swingvision.analysis.shots`).

User edits (``edits.json``): a shot marked "not a practice shot" is kept with
``excluded`` and left out of every summary; a landing placed by the user replaces the
detected one and counts as confirmed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyarrow as pa

from swingvision.analysis.segmentation import practice_shots
from swingvision.analysis.shots import KMH, line_call, speed_error_kmh
from swingvision.court import model as court_model
from swingvision.storage.edits import practice_shot_edit
from swingvision.storage.schemas import PRACTICE, SessionEdits, Target

STROKE_LABELS: dict[str, str] = {
    "serve": "Serve",
    "forehand": "Forehand",
    "backhand": "Backhand",
    "forehand_volley": "Forehand volley",
    "backhand_volley": "Backhand volley",
    "overhead": "Overhead",
}
KIND_LABELS = {"serve": "Serve", "groundstroke": "Groundstroke", "unknown": "Shot"}
CALLED = ("in", "out_long", "out_wide", "net")
CLOSE_CALL_SIGMAS = 2.0
SPEED_BANDS_KMH = (80.0, 110.0, 140.0)
#: A serve's speed off the racket below this is a fit to something else (the toss).
SERVE_MIN_KMH = 60.0
ROLLING_WINDOW = 10


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def to_hitter_frame(x: float, y: float, side: int | None) -> tuple[float, float]:
    """Court coordinates as seen by a hitter on ``side`` (-1 near: unchanged, +1 far: 180°)."""
    return (-x, -y) if side == 1 else (x, y)


def normalized(t: Target) -> Target:
    """The target with rectangle corners ordered (x0 < x1, y0 < y1)."""
    if t.shape == "rect" and None not in (t.x0, t.x1, t.y0, t.y1):
        return t.model_copy(
            update={
                "x0": min(t.x0, t.x1),  # type: ignore[type-var]
                "x1": max(t.x0, t.x1),  # type: ignore[type-var]
                "y0": min(t.y0, t.y1),  # type: ignore[type-var]
                "y1": max(t.y0, t.y1),  # type: ignore[type-var]
            }
        )
    return t


def target_center(t: Target) -> tuple[float, float]:
    if t.shape == "circle":
        return float(t.cx), float(t.cy)  # type: ignore[arg-type]
    return (float(t.x0) + float(t.x1)) / 2, (float(t.y0) + float(t.y1)) / 2  # type: ignore[arg-type]


def target_edge(t: Target, x: float, y: float) -> float:
    """Signed distance (m) from the target's boundary: + inside, - outside."""
    if t.shape == "circle":
        return float(t.r) - float(np.hypot(x - t.cx, y - t.cy))  # type: ignore[arg-type,operator]
    x0, x1, y0, y1 = (float(v) for v in (t.x0, t.x1, t.y0, t.y1))  # type: ignore[arg-type]
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    if dx > 0 or dy > 0:
        return -float(np.hypot(dx, dy))
    return min(x - x0, x1 - x, y - y0, y1 - y)


def target_applies(t: Target, kind: str | None, stroke_type: str | None) -> bool:
    if not t.strokes:
        return True
    stroke = stroke_type or ("serve" if kind == "serve" else None)
    return stroke is not None and stroke in t.strokes


def target_in_hitter_frame(t: Target, side: int | None) -> Target:
    """An absolute target expressed in a hitter's frame (relative ones already are)."""
    if t.frame == "relative" or side != 1:
        return t
    if t.shape == "circle":
        return t.model_copy(update={"cx": -t.cx, "cy": -t.cy})  # type: ignore[operator]
    return normalized(
        t.model_copy(update={"x0": -t.x1, "x1": -t.x0, "y0": -t.y1, "y1": -t.y0})  # type: ignore[operator]
    )


def target_on_court(t: Target, side: int | None) -> Target:
    """Where a target lies on the court for a hitter on ``side`` (court coordinates)."""
    if t.frame == "absolute" or side != 1:
        return t
    if t.shape == "circle":
        return t.model_copy(update={"cx": -t.cx, "cy": -t.cy})  # type: ignore[operator]
    return normalized(
        t.model_copy(update={"x0": -t.x1, "x1": -t.x0, "y0": -t.y1, "y1": -t.y0})  # type: ignore[operator]
    )


def target_valid(t: Target) -> bool:
    if t.shape == "circle":
        return all(v is not None and np.isfinite(v) for v in (t.cx, t.cy, t.r)) and t.r > 0  # type: ignore[operator]
    vals = (t.x0, t.x1, t.y0, t.y1)
    return (
        all(v is not None and np.isfinite(v) for v in vals)
        and abs(t.x1 - t.x0) > 0.05  # type: ignore[operator]
        and abs(t.y1 - t.y0) > 0.05  # type: ignore[operator]
    )


def service_box_call(rx: float, ry: float, serve_side: str) -> tuple[str, float]:
    """(outcome, margin m) of a serve landing in the server's frame. Deuce serves go to the
    box on the server's left across the net (negative x), ad serves to the right."""
    if ry < 0:
        return "own_side", float(ry)
    half_line = court_model.LINE_WIDTH / 2  # a center over the center service line is in
    if serve_side == "deuce":
        x_lo, x_hi = -court_model.HALF_SINGLES, half_line
    else:
        x_lo, x_hi = -half_line, court_model.HALF_SINGLES
    long_by = ry - court_model.SERVICE_LINE_FROM_NET
    wide_by = max(x_lo - rx, rx - x_hi)
    if long_by > 0 or wide_by > 0:
        return ("out_long" if long_by >= wide_by else "out_wide"), -float(max(long_by, wide_by))
    return "in", float(min(-long_by, -wide_by))


# ---------------------------------------------------------------------------
# Per-shot evaluation
# ---------------------------------------------------------------------------


def anchor_time(seg: dict) -> float:
    """The time an edit refers to: the hit, or the landing when the contact wasn't seen."""
    if seg["contact_source"] == "hit" or seg["t_end"] is None:
        return float(seg["t_contact"])
    return float(seg["t_end"])


def _num(v) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def evaluate(
    segments: pa.Table,
    shots: pa.Table | None,
    targets: Iterable[Target],
    edits: SessionEdits | None = None,
    flights: pa.Table | None = None,
) -> pa.Table:
    """One practice row per practice-shot segment."""
    targets = [normalized(t) for t in targets if target_valid(normalized(t))]
    edits = edits or SessionEdits()
    shot_by_id = {r["shot_id"]: r for r in (shots.to_pylist() if shots is not None else [])}
    feed_flight = {
        f["start_event_id"]: f
        for f in (flights.to_pylist() if flights is not None else [])
        if f["start_kind"] == "machine"
    }
    rows = []
    for seg in practice_shots(segments):
        shot = shot_by_id.get(seg["shot_id"]) if seg["shot_id"] is not None else None
        ed = practice_shot_edit(edits, anchor_time(seg))
        flags = list(seg["flags"] or [])
        side, kind = seg["side"], seg["shot_kind"]
        stroke = seg.get("stroke_type") or (shot["stroke_type"] if shot else None)
        lx, ly = _num(seg["landing_x"]), _num(seg["landing_y"])
        sigma, source = _num(seg["landing_sigma_m"]), seg["landing_source"]
        confirmed = bool(ed and ed.confirmed)
        net = seg["end_reason"] == "net"
        if ed is not None and ed.landing is not None:
            lx, ly = ed.landing
            sigma, source, confirmed, net = None, "user", True, False
        row = {
            "segment_id": seg["segment_id"],
            "block_id": seg["block_id"],
            "shot_id": seg["shot_id"],
            "t_contact": seg["t_contact"],
            "t_end": seg["t_end"],
            "side": side,
            "shot_kind": kind,
            "serve_side": seg["serve_side"],
            "stroke_type": stroke,
            "landing_x": lx,
            "landing_y": ly,
            "landing_sigma_m": sigma,
            "landing_source": source if lx is not None else None,
            "landing_confirmed": confirmed,
            "excluded": bool(ed and ed.exclude),
            "targets": [],
            "targets_hit": [],
        }
        outcome, margin, area = "unknown", None, None
        has_landing = lx is not None and ly is not None and side is not None
        if net:
            outcome = "net"
        elif has_landing:
            rx, ry = to_hitter_frame(lx, ly, side)  # type: ignore[arg-type]
            row["rel_x"], row["rel_y"] = rx, ry
            if kind == "serve" and seg["serve_side"] in ("deuce", "ad"):
                outcome, margin = service_box_call(rx, ry, seg["serve_side"])
                area = f"{seg['serve_side']}_box"
            else:
                outcome, margin = line_call(lx, ly, side)  # type: ignore[arg-type]
                area = "singles"
        elif side is None:
            flags.append("side_unknown")
        row.update(outcome=outcome, margin_m=margin, call_area=area)
        row["close_call"] = (
            margin is not None and sigma is not None and abs(margin) < CLOSE_CALL_SIGMAS * sigma
        )
        applicable = [t for t in targets if target_applies(t, kind, stroke)]
        row["targets"] = [t.id for t in applicable]
        if applicable and has_landing:
            best = None
            for t in applicable:
                pt = (row["rel_x"], row["rel_y"]) if t.frame == "relative" else (lx, ly)
                edge = target_edge(t, *pt)  # type: ignore[misc]
                if edge >= 0:
                    row["targets_hit"].append(t.id)
                th = target_in_hitter_frame(t, side)
                cx, cy = target_center(th)
                dist = float(np.hypot(row["rel_x"] - cx, row["rel_y"] - cy))
                cand = (edge < 0, dist, t.id, edge, row["rel_x"] - cx, row["rel_y"] - cy)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            assert best is not None
            row["in_target"] = bool(row["targets_hit"])
            row["target_id"], row["target_dist_m"], row["target_edge_m"] = best[2], best[1], best[3]
            row["width_err_m"], row["depth_err_m"] = best[4], best[5]
            if sigma is not None and abs(best[3]) < CLOSE_CALL_SIGMAS * sigma:
                flags.append("target_close_call")
        elif applicable and outcome == "net":
            row["in_target"] = False
        if shot is not None:
            v = _num(shot["speed_racket_kmh"])
            if v is not None and kind == "serve" and v < SERVE_MIN_KMH:
                # The fit only saw the toss or the next ball: not this serve's flight.
                v = None
                flags.append("speed_implausible")
            row["speed_kmh"] = v
            row["speed_err_kmh"] = speed_error_kmh(v, _num(shot["speed_sigma_kmh"]))
            if "speed_uncertain" in (shot["quality_flags"] or []):
                flags.append("speed_uncertain")
        f = feed_flight.get(seg["feed_event_id"]) if seg["feed_event_id"] is not None else None
        if f is not None and f["ok"]:
            row["feed_speed_kmh"] = None if _num(f["speed0"]) is None else f["speed0"] * KMH
            row["feed_land_x"], row["feed_land_y"] = _num(f["landing_x"]), _num(f["landing_y"])
        row["flags"] = flags
        rows.append(row)
    if not rows:
        return PRACTICE.empty_table()
    return pa.table(
        {fl.name: pa.array([r.get(fl.name) for r in rows], fl.type) for fl in PRACTICE},
        schema=PRACTICE,
    )


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _pct(a: int, b: int) -> float | None:
    return a / b if b else None


def _stat(values: list[float], fn) -> float | None:
    return float(fn(values)) if values else None


def summarize(rows: list[dict]) -> dict:
    """Accuracy numbers for a set of practice rows (excluded shots are left out)."""
    rows = [r for r in rows if not r["excluded"]]
    called = [r for r in rows if r["outcome"] in CALLED]
    n_in = sum(r["outcome"] == "in" for r in called)
    n_net = sum(r["outcome"] == "net" for r in called)
    targeted = [r for r in rows if r["in_target"] is not None]
    n_hit = sum(bool(r["in_target"]) for r in targeted)
    landed = [r for r in rows if r["rel_y"] is not None and r["outcome"] != "own_side"]
    dists = [r["target_dist_m"] for r in landed if r["target_dist_m"] is not None]
    depth = [r["rel_y"] for r in landed]
    width = [r["rel_x"] for r in landed]
    speeds = [
        r["speed_kmh"]
        for r in rows
        if r["speed_kmh"] is not None and "speed_uncertain" not in (r["flags"] or [])
    ]
    feeds = [r["feed_speed_kmh"] for r in rows if r["feed_speed_kmh"] is not None]
    feed_land = [(r["feed_land_x"], r["feed_land_y"]) for r in rows if r["feed_land_x"] is not None]
    return {
        "n": len(rows),
        "n_called": len(called),
        "n_in": n_in,
        "in_pct": _pct(n_in, len(called)),
        "n_net": n_net,
        "net_pct": _pct(n_net, len(called)),
        "n_out": len(called) - n_in - n_net,
        "n_targeted": len(targeted),
        "n_target_hits": n_hit,
        "target_pct": _pct(n_hit, len(targeted)),
        "n_landed": len(landed),
        "dist_median": _stat(dists, np.median),
        "dist_mean": _stat(dists, np.mean),
        "depth_mean": _stat(depth, np.mean),
        "depth_sd": _stat(depth, np.std) if len(depth) >= 2 else None,
        "width_sd": _stat(width, np.std) if len(width) >= 2 else None,
        "speed_median": _stat(speeds, np.median),
        "n_speed": len(speeds),
        "n_confirmed": sum(bool(r["landing_confirmed"]) for r in rows),
        "feed_speed_mean": _stat(feeds, np.mean),
        "feed_speed_sd": _stat(feeds, np.std) if len(feeds) >= 2 else None,
        "feed_spread_m": (
            float(np.sqrt(np.var([p[0] for p in feed_land]) + np.var([p[1] for p in feed_land])))
            if len(feed_land) >= 2
            else None
        ),
    }


def rolling(rows: list[dict], window: int = ROLLING_WINDOW) -> dict:
    """Rolling target-hit and in rates over the session, by shot (excluded left out).

    Returns per-shot arrays: time, shot number, and the rate over the last ``window`` shots
    that had a target (resp. a call), ``None`` before the window has 3 shots.
    """
    rows = sorted((r for r in rows if not r["excluded"]), key=lambda r: r["t_contact"])
    out: dict[str, list] = {"t": [], "n": [], "target": [], "in": [], "segment_id": []}
    tgt: list[bool] = []
    ins: list[bool] = []
    for i, r in enumerate(rows):
        if r["in_target"] is not None:
            tgt.append(bool(r["in_target"]))
        if r["outcome"] in CALLED:
            ins.append(r["outcome"] == "in")
        out["t"].append(r["t_contact"])
        out["n"].append(i + 1)
        out["segment_id"].append(r["segment_id"])
        out["target"].append(float(np.mean(tgt[-window:])) if len(tgt) >= 3 else None)
        out["in"].append(float(np.mean(ins[-window:])) if len(ins) >= 3 else None)
    return out


def speed_band(v: float | None) -> str | None:
    if v is None:
        return None
    lo = 0.0
    for hi in SPEED_BANDS_KMH:
        if v < hi:
            return f"{lo:.0f}–{hi:.0f}" if lo else f"< {hi:.0f}"
        lo = hi
    return f"≥ {lo:.0f}"


def breakdown(rows: list[dict]) -> list[tuple[str, dict]]:
    """Summaries by shot kind (serves split deuce/ad), stroke type (M6) and speed band."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        k = KIND_LABELS.get(r["shot_kind"], "Shot")
        if r["shot_kind"] == "serve" and r["serve_side"]:
            k += f" ({r['serve_side']})"
        groups.setdefault(k, []).append(r)
        if r["stroke_type"]:
            label = STROKE_LABELS.get(r["stroke_type"], r["stroke_type"].capitalize())
            groups.setdefault(f"Stroke: {label}", []).append(r)
        band = speed_band(r["speed_kmh"])
        if band and "speed_uncertain" not in (r["flags"] or []):
            groups.setdefault(f"{band} km/h", []).append(r)
    return [(k, summarize(v)) for k, v in groups.items()]
