"""Session statistics (PLAN.md §6 stage 20, §9.2 Stats, §7.13): speeds and landings by
stroke, depth, movement, swings — for one session or any selection of sessions.

One record per shot is assembled from ``shots.parquet`` and, for practice sessions, the
practice results (``practice.parquet``: landings after the user's edits, line calls,
excluded shots, and practice shots whose contact the ball tracker didn't see). Swings and
movement get records too: one per swing, and one per session holding the movement sums and
counts (distance, time, samples, the folded heatmap) that add up across sessions.

The ``stats`` stage writes a session's records to ``stats_records.parquet`` and its summary
to ``stats.json``. :mod:`swingvision.analysis.aggregate` reads the records of many sessions
back into the same :class:`StatsData`, so every summary here runs unchanged on one session or
on a selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from swingvision.court import model as court_model
from swingvision.players import movement as mv
from swingvision.pose.strokes import STROKE_LABELS
from swingvision.settings import ProcessingDefaults
from swingvision.storage import tables
from swingvision.storage.library import recording_time
from swingvision.storage.schemas import MOVEMENT, PASS1_FRAMES, STATS_RECORDS, Calibration
from swingvision.storage.session import Session

STATS_VERSION = 2

#: Groups shots are summarized by: the recognized stroke, else the practice shot kind.
GROUPS = ("serve", "forehand", "backhand", "forehand_volley", "backhand_volley", "overhead")
GROUP_LABELS = {**{k: STROKE_LABELS[k] for k in GROUPS}, "unknown": "Not recognized"}
CALLED = ("in", "out_long", "out_wide", "net")
#: Groundstrokes landing within this distance of the baseline count as deep.
DEEP_ZONE_M = 2.74  # the last 3 yards
#: Distance bins for the movement-over-time chart.
DISTANCE_BIN_S = 300.0

#: Record fields per kind (besides the session columns, ``kind``, ``t`` and ``side``).
SHOT_FIELDS = (
    "shot_id", "segment_id", "group", "speed_kmh", "speed_sigma_kmh", "speed_ok", "landing_x",
    "landing_y", "rel_x", "rel_y", "outcome", "net_clearance_m", "spin_sign", "contact_seen",
    "excluded", "in_target",
)  # fmt: skip
SWING_FIELDS = (
    "swing_id", "stroke_type", "wrist_speed_peak", "forward_s", "contact_height_m",
    "chain_in_order",
)  # fmt: skip
MOVEMENT_FIELDS = (
    "processed_frames", "lit_frames", "dark_frames", "tracked_lit_frames", "tracked_s",
    "distance_m", "max_speed_mps", "n_samples", "n_moving", "moving_speed_sum", "n_near",
    "n_runs", "heatmap_s", "distance_bins_m", "distance_bin_s",
)  # fmt: skip
#: Per-session columns of the records file (what a selection is filtered by).
SESSION_FIELDS = (
    "session_id", "recorded_on", "mode", "practice_type", "profile_id", "device_key",
    "calibration_by", "speeds_calibrated",
)  # fmt: skip

_DEFAULTS = ProcessingDefaults()


def stroke_group(stroke: str | None, shot_kind: str | None = None) -> str:
    if stroke in GROUPS:
        return stroke  # type: ignore[return-value]
    if shot_kind == "serve":
        return "serve"
    return "unknown"


@dataclass
class StatsData:
    """Everything the Stats page and the ``stats`` stage look at, for one or many sessions.

    ``records`` (shots) and ``swings`` carry their ``session_id``; ``movement`` has one
    record per session with player tracking; ``sessions`` describes each session (id, name,
    recording time, mode, practice type, tags, ...), in recording order.
    """

    records: list[dict]
    swings: list[dict]
    movement: list[dict]
    sessions: list[dict]
    is_practice: bool
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def multi(self) -> bool:
        return len(self.sessions) > 1


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


def swing_record(s: dict) -> dict:
    """What the swing summaries read from one ``swings.parquet`` row."""
    return {
        "t": s["t_contact"],
        "side": s["side"],
        "swing_id": s["swing_id"],
        "stroke_type": s["stroke_type"],
        "wrist_speed_peak": _num(s["wrist_speed_peak"]),
        "forward_s": _num(s["forward_s"]),
        "contact_height_m": _num(s["contact_height_m"]),
        "chain_in_order": s["chain_in_order"],
    }


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


def movement_record(
    movement: pa.Table, frames: pa.Table, dark_luma: float, view_min: float | None
) -> dict | None:
    """One session's movement as sums and counts that pool across sessions (``None``
    before the detection pass has run)."""
    if frames.num_rows == 0:
        return None
    params = mv.MovementParams()
    s = mv.summarize(movement, frames, dark_luma, params, view_min)
    speed = movement.column("speed").to_numpy().astype(np.float64)
    y = movement.column("y").to_numpy().astype(np.float64)
    fast = speed >= params.moving_speed
    _xc, _yc, H = mv.heatmap(movement, fold=True)
    dist = distance_over_time(movement)
    return {
        "processed_frames": int(s["processed_frames"]),
        "lit_frames": int(s["lit_frames"]),
        "dark_frames": int(s["dark_frames"]),
        "tracked_lit_frames": round(s["coverage"] * max(1, s["lit_frames"])),
        "tracked_s": float(s["tracked_s"]),
        "distance_m": float(s["distance_m"]),
        "max_speed_mps": float(s["max_speed_mps"]),
        "n_samples": int(movement.num_rows),
        "n_moving": int(fast.sum()),
        "moving_speed_sum": float(speed[fast].sum()),
        "n_near": int((y < 0).sum()),
        "n_runs": int(s["n_runs"]),
        "heatmap_s": [float(v) for v in H.ravel()],
        "distance_bins_m": [float(v) for v in dist["distance_m"]],
        "distance_bin_s": float(dist["bin_s"]),
    }


def _calibration_by(session: Session) -> str | None:
    if not session.calibration_path.exists():
        return None
    try:
        cal = Calibration.model_validate_json(session.calibration_path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return cal.confirmed_by


def session_info(session: Session, config=None) -> dict:
    """The per-session columns of the records (what a selection is filtered by)."""
    config = config or session.load_config()
    return {
        "session_id": config.id,
        "name": config.name,
        "recorded_on": recording_time(
            config.video.creation_time if config.video else None, config.created_at
        ),
        "duration_s": config.video.duration_s if config.video else 0.0,
        "mode": config.mode,
        "practice_type": config.practice.submode if config.practice else None,
        "profile_id": config.players.me_profile_id,
        "device_key": None,  # M7c
        "calibration_by": _calibration_by(session),
        "speeds_calibrated": False,  # M7c
        "tags": [],
    }


def load(
    session: Session,
    include_excluded: bool = False,
    dark_luma: float = _DEFAULTS.dark_luma,
    view_min: float | None = _DEFAULTS.view_min,
) -> StatsData:
    """One session's statistics records, straight from its tables (always current)."""
    config = session.load_config()
    shots = _opt(session.shots_path)
    practice = _opt(session.practice_path) if config.practice is not None else None
    swings = _opt(session.swings_path)
    movement = _opt(session.movement_path)
    frames = _opt(session.pass1_frames_path)
    if movement is not None and movement.num_rows:
        movement = movement.filter(pc.equal(movement.column("player"), "me"))
    info = session_info(session, config)
    sid = info["session_id"]
    records = build_records(shots, practice, include_excluded)
    swing_rows = [swing_record(s) for s in swings.to_pylist()] if swings is not None else []
    for r in (*records, *swing_rows):
        r["session_id"] = sid
    move = movement_record(
        movement if movement is not None else MOVEMENT.empty_table(),
        frames if frames is not None else PASS1_FRAMES.empty_table(),
        dark_luma,
        view_min,
    )
    if move is not None:
        move["session_id"] = sid
    data = StatsData(
        records=records,
        swings=swing_rows,
        movement=[move] if move is not None else [],
        sessions=[info],
        is_practice=config.practice is not None,
        duration_s=info["duration_s"],
    )
    if shots is None:
        data.notes.append("No shots yet: processing hasn't got to the Shots stage.")
    if config.practice is not None and practice is None and shots is not None:
        data.notes.append("Practice results aren't ready, so landings ignore your corrections.")
    return data


def records_table(data: StatsData) -> pa.Table:
    """``data`` as ``stats_records.parquet`` rows (load with ``include_excluded=True``)."""
    info = {s["session_id"]: s for s in data.sessions}
    rows = []
    for kind, recs in (("shot", data.records), ("swing", data.swings), ("movement", data.movement)):
        for r in recs:
            row = {k: info[r["session_id"]].get(k) for k in SESSION_FIELDS}
            row.update({k: v for k, v in r.items() if k in STATS_RECORDS.names})
            row["kind"] = kind
            rows.append(row)
    full = [{f.name: r.get(f.name) for f in STATS_RECORDS} for r in rows]
    return pa.Table.from_pylist(full, schema=STATS_RECORDS)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def filter_records(records: list[dict], groups: list[str] | None, end: str | None) -> list[dict]:
    """Shot records of the stroke ``groups`` (all when empty) hit from ``end``
    (``near`` | ``far`` | anything else: both)."""
    out = []
    for r in records:
        if groups and r["group"] not in groups:
            continue
        if end == "near" and r["side"] != -1:
            continue
        if end == "far" and r["side"] != 1:
            continue
        out.append(r)
    return out


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
        "n_deep_base": len(ground_in),
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


def summarize_movement(data: StatsData) -> dict:
    """Headline movement figures over every session in ``data``: frames, time and distance
    add up; coverage and the averages are pooled over the frames and samples behind them."""
    ms = data.movement
    if not ms:
        return {}

    def total(k: str):
        return sum(m[k] for m in ms)

    lit, n_samples, n_moving = total("lit_frames"), total("n_samples"), total("n_moving")
    out = {
        "processed_frames": total("processed_frames"),
        "lit_frames": lit,
        "dark_frames": total("dark_frames"),
        "coverage": total("tracked_lit_frames") / max(1, lit),
        "tracked_s": total("tracked_s"),
        "distance_m": total("distance_m"),
        "max_speed_mps": max(m["max_speed_mps"] for m in ms),
        "mean_moving_speed_mps": total("moving_speed_sum") / n_moving if n_moving else 0.0,
        "near_half_frac": total("n_near") / n_samples if n_samples else 0.0,
        "n_runs": total("n_runs"),
        "n_sessions": len(ms),
    }
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


_HEAT_XC, _HEAT_YC, _HEAT_EMPTY = mv.heatmap(MOVEMENT.empty_table())


def movement_heatmap(data: StatsData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seconds per court cell over every session (both ends folded onto the near half)."""
    H = np.zeros_like(_HEAT_EMPTY)
    for m in data.movement:
        h = np.asarray(m["heatmap_s"] or [], dtype=np.float64)
        if h.size == H.size:  # a grid from another version doesn't line up: left out
            H += h.reshape(H.shape)
    return _HEAT_XC, _HEAT_YC, H


def session_label(info: dict) -> str:
    """Short name for a session on a chart axis: its recording date and name."""
    when = (info.get("recorded_on") or "")[:10]
    return f"{when} · {info.get('name') or info['session_id']}" if when else info["session_id"]


def distance_series(data: StatsData) -> dict:
    """Distance covered: per ``DISTANCE_BIN_S`` of video for one session, per session for
    several (``per``: ``time`` | ``session``)."""
    if not data.multi:
        m = data.movement[0] if data.movement else None
        if m is None or not m["distance_bins_m"]:
            return {"per": "time", "t": [], "distance_m": [], "bin_s": DISTANCE_BIN_S}
        bin_s = m["distance_bin_s"]
        return {
            "per": "time",
            "t": [float(i * bin_s) for i in range(len(m["distance_bins_m"]))],
            "distance_m": list(m["distance_bins_m"]),
            "bin_s": bin_s,
        }
    by_sid = {m["session_id"]: m for m in data.movement}
    out: dict = {"per": "session", "session_id": [], "label": [], "distance_m": [], "tracked_s": []}
    for info in data.sessions:
        m = by_sid.get(info["session_id"])
        if m is None:
            continue
        out["session_id"].append(info["session_id"])
        out["label"].append(session_label(info))
        out["distance_m"].append(round(m["distance_m"], 1))
        out["tracked_s"].append(m["tracked_s"])
    return out


def session_stats(data: StatsData) -> dict:
    """The ``stats.json`` content."""
    dist = distance_series(data)
    return {
        "version": STATS_VERSION,
        "shots": summarize_shots(data.records),
        "strokes": by_group(data.records),
        "swings": summarize_swings(data.swings),
        "movement": summarize_movement(data),
        "distance_over_time": {k: dist[k] for k in ("t", "distance_m", "bin_s")}
        if dist["per"] == "time"
        else dist,
    }


# ---------------------------------------------------------------------------
# Trends: one value per session against its date, with n and a 95% interval
# ---------------------------------------------------------------------------

Z95 = 1.959964

#: KPI → (label, kind). Rates get a Wilson interval, medians a distribution-free one
#: (order statistics), means a normal one; totals none.
TREND_KPIS: dict[str, tuple[str, str]] = {
    "in_pct": ("In %", "rate"),
    "target_pct": ("On target %", "rate"),
    "net_pct": ("Net %", "rate"),
    "deep_pct": ("Deep %", "rate"),
    "speed_median": ("Speed (median)", "median"),
    "depth_mean": ("Short of the line (mean)", "mean"),
    "wrist_speed": ("Wrist speed (median)", "median"),
    "distance_m": ("Distance covered", "total"),
}


def wilson(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def median_ci(values: list[float], z: float = Z95) -> tuple[float | None, float | None]:
    """Distribution-free interval for the median: the order statistics n/2 ∓ z·√n/2."""
    v = sorted(values)
    n = len(v)
    if n < 3:
        return None, None
    lo = max(0, math.floor(n / 2 - z * math.sqrt(n) / 2))
    hi = min(n - 1, math.ceil(n / 2 + z * math.sqrt(n) / 2) - 1)
    return float(v[lo]), float(v[hi])


def mean_ci(values: list[float], z: float = Z95) -> tuple[float | None, float | None]:
    n = len(values)
    if n < 2:
        return None, None
    m = float(np.mean(values))
    h = z * float(np.std(values, ddof=1)) / math.sqrt(n)
    return m - h, m + h


def _trend_point(kpi: str, rows: list[dict], swings: list[dict], move: dict | None) -> dict:
    rows = [r for r in rows if not r["excluded"]]
    if kpi in ("in_pct", "net_pct"):
        called = [r for r in rows if r["outcome"] in CALLED]
        k = sum(r["outcome"] == ("in" if kpi == "in_pct" else "net") for r in called)
        lo, hi = wilson(k, len(called))
        return {"n": len(called), "value": _pct(k, len(called)), "lo": lo, "hi": hi}
    if kpi == "target_pct":
        tg = [r for r in rows if r["in_target"] is not None]
        k = sum(bool(r["in_target"]) for r in tg)
        lo, hi = wilson(k, len(tg))
        return {"n": len(tg), "value": _pct(k, len(tg)), "lo": lo, "hi": hi}
    if kpi == "deep_pct":
        s = summarize_shots(rows)
        n = s["n_deep_base"]
        k = round((s["deep_pct"] or 0.0) * n)
        lo, hi = wilson(k, n)
        return {"n": n, "value": s["deep_pct"], "lo": lo, "hi": hi}
    if kpi == "speed_median":
        v = [r["speed_kmh"] for r in rows if r["speed_ok"]]
        lo, hi = median_ci(v)
        return {"n": len(v), "value": _stat(v, np.median), "lo": lo, "hi": hi}
    if kpi == "depth_mean":
        v = [_depth(r["group"], r["rel_y"]) for r in rows if r["rel_y"] is not None]
        v = [x for x in v if x is not None]
        lo, hi = mean_ci(v)
        return {"n": len(v), "value": _stat(v, np.mean), "lo": lo, "hi": hi}
    if kpi == "wrist_speed":
        v = [s["wrist_speed_peak"] for s in swings if s["stroke_type"] in GROUPS]
        v = [x for x in v if x is not None]
        lo, hi = median_ci(v)
        return {"n": len(v), "value": _stat(v, np.median), "lo": lo, "hi": hi}
    if kpi == "distance_m":
        if move is None:
            return {"n": 0, "value": None, "lo": None, "hi": None}
        n = round(move["tracked_s"] / 60.0)  # minutes tracked
        return {"n": n, "value": move["distance_m"], "lo": None, "hi": None}
    raise ValueError(f"Unknown trend KPI {kpi!r}; choose one of {', '.join(TREND_KPIS)}")


def trend(
    data: StatsData,
    kpi: str,
    records: list[dict] | None = None,
    swings: list[dict] | None = None,
) -> list[dict]:
    """``kpi`` per session in recording order: session id, label, date, n, value, lo, hi.

    ``records`` / ``swings``: a filtered subset (default: all of ``data``).
    """
    records = data.records if records is None else records
    swings = data.swings if swings is None else swings
    by_rec: dict[str, list[dict]] = {}
    for r in records:
        by_rec.setdefault(r["session_id"], []).append(r)
    by_sw: dict[str, list[dict]] = {}
    for s in swings:
        by_sw.setdefault(s["session_id"], []).append(s)
    moves = {m["session_id"]: m for m in data.movement}
    out = []
    for info in data.sessions:
        sid = info["session_id"]
        p = _trend_point(kpi, by_rec.get(sid, []), by_sw.get(sid, []), moves.get(sid))
        out.append(
            {"session_id": sid, "label": session_label(info),
             "recorded_on": info.get("recorded_on"), **p}
        )  # fmt: skip
    return out
