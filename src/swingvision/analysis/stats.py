"""Session statistics (PLAN.md §6 stage 20, §9.2 Stats): speeds and landings by stroke,
depth, movement, swings.

One record per shot is assembled from ``shots.parquet`` and, for practice sessions, the
practice results (``practice.parquet``: landings after the user's edits, line calls,
excluded shots, and practice shots whose contact the ball tracker didn't see). The Stats
page draws its charts from these records; the ``stats`` stage writes the summary to
``stats.json`` for the Library and, later, cross-session trends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.court import model as court_model
from swingvision.players import movement as mv
from swingvision.pose.strokes import STROKE_LABELS
from swingvision.storage import tables
from swingvision.storage.schemas import MOVEMENT, PASS1_FRAMES
from swingvision.storage.session import Session

STATS_VERSION = 1

#: Groups shots are summarized by: the recognized stroke, else the practice shot kind.
GROUPS = ("serve", "forehand", "backhand", "forehand_volley", "backhand_volley", "overhead")
GROUP_LABELS = {**{k: STROKE_LABELS[k] for k in GROUPS}, "unknown": "Not recognized"}
CALLED = ("in", "out_long", "out_wide", "net")
#: Groundstrokes landing within this distance of the baseline count as deep.
DEEP_ZONE_M = 2.74  # the last 3 yards
#: Distance bins for the movement-over-time chart.
DISTANCE_BIN_S = 300.0


def stroke_group(stroke: str | None, shot_kind: str | None = None) -> str:
    if stroke in GROUPS:
        return stroke  # type: ignore[return-value]
    if shot_kind == "serve":
        return "serve"
    return "unknown"


@dataclass
class StatsData:
    """Everything the Stats page and the ``stats`` stage look at."""

    records: list[dict]
    swings: list[dict]
    movement: pa.Table
    frames: pa.Table
    is_practice: bool
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)


def _opt(path) -> pa.Table | None:
    return tables.read_table(path) if path.exists() else None


def _num(v) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def _hitter_frame(x, y, side) -> tuple[float | None, float | None]:
    """Court position seen from the hitter's end (the hitter at the near baseline)."""
    if x is None or y is None:
        return None, None
    return (-x, -y) if side == 1 else (x, y)


def _depth(group: str, rel_y: float | None) -> float | None:
    """How far short of the line the shot aimed at it landed (m): the service line for
    serves, the baseline otherwise. Negative: beyond it."""
    if rel_y is None or rel_y <= 0:
        return None
    line = court_model.SERVICE_LINE_FROM_NET if group == "serve" else court_model.HALF_LENGTH
    return line - rel_y


def _record_from_shot(s: dict) -> dict:
    group = stroke_group(s["stroke_type"], "serve" if s["is_serve"] else None)
    rel_x, rel_y = _hitter_frame(s["landing_x"], s["landing_y"], s["side"])
    flags = list(s["quality_flags"] or [])
    return {
        "t": s["t_contact"],
        "shot_id": s["shot_id"],
        "segment_id": s["segment_id"],
        "group": group,
        "side": s["side"],
        "speed_kmh": _num(s["speed_racket_kmh"]),
        "speed_sigma_kmh": _num(s["speed_sigma_kmh"]),
        "speed_ok": s["speed_racket_kmh"] is not None and "speed_uncertain" not in flags,
        "landing_x": _num(s["landing_x"]),
        "landing_y": _num(s["landing_y"]),
        "rel_x": _num(rel_x),
        "rel_y": _num(rel_y),
        "outcome": s["outcome"],
        "net_clearance_m": _num(s["net_clearance_m"]),
        "spin_sign": s["spin_sign"],
        "contact_seen": True,
        "excluded": False,
        "in_target": None,
    }


def build_records(
    shots: pa.Table | None, practice: pa.Table | None, include_excluded: bool = False
) -> list[dict]:
    """One record per shot over the net, practice results layered over the shot data.

    Practice rows win for what the user can correct (landing, call, exclusion) and add the
    practice shots whose contact wasn't seen (they have a landing but no speed).
    """
    shot_rows = shots.to_pylist() if shots is not None else []
    by_id = {
        s["shot_id"]: s
        for s in shot_rows
        if s["outcome"] != "own_side" and "contact_not_at_hitter" not in (s["quality_flags"] or [])
    }
    if practice is None:
        records = [_record_from_shot(s) for s in by_id.values()]
        return sorted(records, key=lambda r: r["t"] or 0.0)
    records = []
    for p in practice.to_pylist():
        if p["excluded"] and not include_excluded:
            continue
        s = by_id.get(p["shot_id"]) if p["shot_id"] is not None else None
        r = _record_from_shot(s) if s is not None else {
            "shot_id": None, "speed_kmh": None, "speed_sigma_kmh": None, "speed_ok": False,
            "net_clearance_m": None, "spin_sign": None, "contact_seen": False,
        }  # fmt: skip
        group = stroke_group(p["stroke_type"], p["shot_kind"])
        r.update(
            t=p["t_contact"],
            segment_id=p["segment_id"],
            group=group,
            side=p["side"],
            landing_x=_num(p["landing_x"]),
            landing_y=_num(p["landing_y"]),
            rel_x=_num(p["rel_x"]),
            rel_y=_num(p["rel_y"]),
            outcome=p["outcome"],
            excluded=bool(p["excluded"]),
            in_target=p["in_target"],
        )
        if s is None and p["speed_kmh"] is not None:
            flags = p["flags"] or []
            r.update(speed_kmh=_num(p["speed_kmh"]), speed_ok="speed_uncertain" not in flags)
        records.append(r)
    return sorted(records, key=lambda r: r["t"] or 0.0)


def load(session: Session, include_excluded: bool = False) -> StatsData:
    config = session.load_config()
    shots = _opt(session.shots_path)
    practice = _opt(session.practice_path) if config.practice is not None else None
    swings = _opt(session.swings_path)
    movement = _opt(session.movement_path)
    frames = _opt(session.pass1_frames_path)
    if movement is not None and movement.num_rows:
        movement = movement.filter(pc.equal(movement.column("player"), "me"))
    data = StatsData(
        records=build_records(shots, practice, include_excluded),
        swings=swings.to_pylist() if swings is not None else [],
        movement=movement if movement is not None else MOVEMENT.empty_table(),
        frames=frames if frames is not None else PASS1_FRAMES.empty_table(),
        is_practice=config.practice is not None,
        duration_s=config.video.duration_s if config.video else 0.0,
    )
    if shots is None:
        data.notes.append("No shots yet: processing hasn't got to the Shots stage.")
    if config.practice is not None and practice is None and shots is not None:
        data.notes.append("Practice results aren't ready, so landings ignore your corrections.")
    return data


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _pct(n: int, d: int) -> float | None:
    return n / d if d else None


def _stat(values, fn) -> float | None:
    v = [x for x in values if x is not None]
    return float(fn(v)) if v else None


def _sd(values) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.std(v)) if len(v) >= 2 else None


def summarize_shots(records: list[dict]) -> dict:
    """Counts, line calls, speeds and depth for a set of shot records."""
    records = [r for r in records if not r["excluded"]]
    called = [r for r in records if r["outcome"] in CALLED]
    n_in = sum(r["outcome"] == "in" for r in called)
    n_net = sum(r["outcome"] == "net" for r in called)
    speeds = [r["speed_kmh"] for r in records if r["speed_ok"]]
    landed = [r for r in records if r["rel_y"] is not None and r["rel_y"] > 0]
    depth = [_depth(r["group"], r["rel_y"]) for r in landed]
    landed_in = [r for r in landed if r["outcome"] == "in"]
    ground_in = [r for r in landed_in if r["group"] != "serve"]
    deep = [r for r in ground_in if (_depth(r["group"], r["rel_y"]) or 0.0) <= DEEP_ZONE_M]
    targeted = [r for r in records if r["in_target"] is not None]
    return {
        "n": len(records),
        "n_seen": sum(r["contact_seen"] for r in records),
        "n_called": len(called),
        "n_in": n_in,
        "in_pct": _pct(n_in, len(called)),
        "n_net": n_net,
        "net_pct": _pct(n_net, len(called)),
        "n_out": len(called) - n_in - n_net,
        "n_speed": len(speeds),
        "speed_median": _stat(speeds, np.median),
        "speed_p90": _stat(speeds, lambda v: np.percentile(v, 90)),
        "speed_max": _stat(speeds, np.max),
        "n_landed": len(landed),
        "depth_mean": _stat(depth, np.mean),
        "depth_sd": _sd(depth),
        "width_sd": _sd([r["rel_x"] for r in landed]),
        "deep_pct": _pct(len(deep), len(ground_in)),
        "n_targeted": len(targeted),
        "target_pct": _pct(sum(bool(r["in_target"]) for r in targeted), len(targeted)),
        "net_clearance_median": _stat(
            [r["net_clearance_m"] for r in records if r["outcome"] != "net"], np.median
        ),
    }


def by_group(records: list[dict]) -> dict[str, dict]:
    """:func:`summarize_shots` per stroke group, in :data:`GROUPS` order."""
    out = {}
    for g in (*GROUPS, "unknown"):
        rows = [r for r in records if r["group"] == g]
        if rows:
            out[g] = summarize_shots(rows)
    return out


def summarize_swings(swings: list[dict]) -> dict[str, dict]:
    """Swing counts and medians per stroke (strokes only, no 'other')."""
    out = {}
    for g in GROUPS:
        rows = [s for s in swings if s["stroke_type"] == g]
        if not rows:
            continue
        out[g] = {
            "n": len(rows),
            "wrist_speed_median": _stat([s["wrist_speed_peak"] for s in rows], np.median),
            "forward_s_median": _stat([s["forward_s"] for s in rows], np.median),
            "contact_height_median": _stat([s["contact_height_m"] for s in rows], np.median),
            "chain_in_order_pct": _pct(
                sum(bool(s["chain_in_order"]) for s in rows),
                sum(s["chain_in_order"] is not None for s in rows),
            ),
        }
    return out


def distance_over_time(movement: pa.Table, bin_s: float = DISTANCE_BIN_S) -> dict:
    """Distance covered per ``bin_s`` of video (m), counted like the movement summary."""
    if movement.num_rows < 2:
        return {"t": [], "distance_m": [], "bin_s": bin_s}
    t = movement.column("t_s").to_numpy()
    x = movement.column("x").to_numpy().astype(np.float64)
    y = movement.column("y").to_numpy().astype(np.float64)
    speed = movement.column("speed").to_numpy().astype(np.float64)
    run = movement.column("run").to_numpy()
    step = np.hypot(np.diff(x), np.diff(y))
    ok = (run[1:] == run[:-1]) & (
        0.5 * (speed[1:] + speed[:-1]) >= mv.MovementParams().moving_speed
    )
    b = np.floor(t[1:] / bin_s).astype(np.int64)
    n = int(b.max()) + 1
    dist = np.bincount(b[ok], weights=step[ok], minlength=n)
    return {
        "t": [float(i * bin_s) for i in range(n)],
        "distance_m": [round(float(d), 1) for d in dist],
        "bin_s": bin_s,
    }


def summarize_movement(data: StatsData, dark_luma: float, view_min: float | None) -> dict:
    if data.frames.num_rows == 0:
        return {}
    s = mv.summarize(data.movement, data.frames, dark_luma, mv.MovementParams(), view_min)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}


def session_stats(data: StatsData, dark_luma: float, view_min: float | None) -> dict:
    """The ``stats.json`` content."""
    return {
        "version": STATS_VERSION,
        "shots": summarize_shots(data.records),
        "strokes": by_group(data.records),
        "swings": summarize_swings(data.swings),
        "movement": summarize_movement(data, dark_luma, view_min),
        "distance_over_time": distance_over_time(data.movement),
    }
