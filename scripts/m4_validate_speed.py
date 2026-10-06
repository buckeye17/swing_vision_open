"""M4 speed-scale checks on real footage, without a radar gun (docs/m4-ball-3d.md).

Speeds from one camera are only as right as the metric scale (calibration) and the clock
(frame timestamps). Three physical references test both:

* **Gravity.** Flights that start and end on the ground (bounce → bounce) are refitted with
  g free. Their ends are 3D points from the calibration, so g comes out of the image
  curvature in metres and seconds: a length-scale error ``s`` shows up as g·s, a clock
  error ``c`` as g/c². Speeds scale as s/c, so |speed error| ≤ |g error| (pure length
  error) and ≈ |g error|/2 (pure clock error).
* **Drag.** Long, well-tracked hit flights are refitted with a loose prior on C_d. Drag
  deceleration ∝ v², so a speed scale error s changes the fitted C_d by 1/s. Published
  C_d of tennis balls: 0.5-0.6.
* **Bounce.** Vertical restitution e = v_z(after) / -v_z(before) from the flights on both
  sides of a bounce. Hard courts give 0.7-0.85 (ITF rebound test: 0.73-0.76 vertically).

Also compared: the average horizontal speed from the player's tracked feet to the detected
bounce over the measured flight time (no 3D fit involved) vs the fit's.

Usage: ``uv run python scripts/m4_validate_speed.py <session_dir> [<session_dir> …]``
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from swingvision.ball import flights as fl
from swingvision.ball import physics as ph
from swingvision.court import calibration as calib
from swingvision.storage import tables
from swingvision.storage.session import Session


def _stats(name: str, v: np.ndarray, w: np.ndarray | None = None) -> str:
    v = np.asarray(v, float)
    if not len(v):
        return f"{name}: none"
    q = np.percentile(v, [25, 50, 75])
    wm = ""
    if w is not None:
        wm = f", weighted mean {np.sum(v * w) / np.sum(w):.3f}"
    return f"{name}: n={len(v)}, median {q[1]:.3f} (IQR {q[0]:.3f}-{q[2]:.3f}){wm}"


def check_session(path: Path) -> dict:
    session = Session(path)
    config = session.load_config()
    cal = calib.load(session.calibration_path)
    assert cal is not None and config.video is not None
    track = tables.read_table(session.ball_track_path)
    events = tables.read_table(session.events_path)
    summary = session.ball_flights_summary_path
    rejected = set()
    if summary.exists():
        import json

        rejected = set(json.loads(summary.read_text())["rejected_hits"])
    if rejected:
        import pyarrow as pa
        import pyarrow.compute as pc

        ids = pa.array(sorted(rejected), pa.int32())
        events = events.filter(pc.invert(pc.is_in(events.column("event_id"), ids)))
    flights = tables.read_table(session.ball_flights_path).to_pylist()
    pts = fl.detected_points(track)
    specs = {s.flight_id: s for s in fl.plan_flights(pts, events)}
    ev = fl._Events.from_table(events)
    by_id = {int(e): k for k, e in enumerate(ev.id)}
    width = config.video.display_width
    base = ph.FitParams()
    out: dict = {"session": config.name}

    def data_for(r, params):
        s = specs[r["flight_id"]]
        cam = calib.camera_at(cal, s.t0)
        start = (
            ph.EndPoint("free", s.t0)
            if s.start_kind == "free"
            else fl._endpoint(ev, s.start_event, s.start_kind, by_id)
        )
        if s.start_kind in ("hit", "machine"):
            start.px = None
        end = None
        if s.end_kind != "lost":
            end = fl._endpoint(ev, s.end_event, s.end_kind, by_id)
            end.near_xy = None
            if s.end_kind == "hit":
                end.px = None
        px = np.column_stack([pts["x"][s.rows], pts["y"][s.rows]])
        tt = pts["t"][s.rows]
        return ph.FlightData(cam, tt, px, ph.point_sigma(px, tt, width, params), start, end)

    # Gravity from ground-to-ground flights long enough to curve (≥ 0.3 s, ≥ 15 points).
    gs, gw = [], []
    for r in flights:
        if not (
            r["ok"]
            and r["start_kind"] == "bounce"
            and r["end_kind"] == "bounce"
            and r["t1_s"] - r["t0_s"] >= 0.3
            and r["n_obs"] >= 15
        ):
            continue
        f = ph.fit_flight(data_for(r, base), base, free_g=True)
        g, sg = f.outputs.get("g"), f.sigmas.get("g")
        if f.ok and g is not None and sg is not None and np.isfinite(sg) and sg < 1.0:
            gs.append(g)
            gw.append(1 / sg**2)
    gs, gw = np.array(gs), np.array(gw)
    out["g"] = _stats("g (m/s^2) from bounce-to-bounce flights, g free", gs, gw)
    out["g_wmean"] = float(np.sum(gs * gw) / np.sum(gw)) if len(gs) else None
    out["g_n"] = len(gs)

    # Drag coefficient from long hit flights with a loose C_d prior.
    loose = replace(base, cd_sigma=1.0)
    cds, cdw = [], []
    for r in flights:
        if not (
            r["ok"]
            and r["start_kind"] in ("hit", "machine")
            and r["end_kind"] == "bounce"
            and r["n_obs"] >= 25
            and (r["speed0"] or 0) > 25
        ):
            continue
        f = ph.fit_flight(data_for(r, loose), loose)
        cd, s = f.outputs.get("cd"), f.sigmas.get("cd")
        if f.ok and s is not None and np.isfinite(s) and s < 0.3:
            cds.append(cd)
            cdw.append(1 / s**2)
    cds, cdw = np.array(cds), np.array(cdw)
    out["cd"] = _stats("C_d from fast hit flights, loose prior", cds, cdw)
    out["cd_wmean"] = float(np.sum(cds * cdw) / np.sum(cdw)) if len(cds) else None

    # Vertical restitution at bounces with good flights on both sides.
    by_end = {r["end_event_id"]: r for r in flights if r["ok"] and r["end_kind"] == "bounce"}
    es = []
    for r in flights:
        if not (r["ok"] and r["start_kind"] == "bounce"):
            continue
        a = by_end.get(r["start_event_id"])
        if a is None or a["vz_end"] is None or r["vz0"] is None or a["vz_end"] > -1.0:
            continue
        es.append(r["vz0"] / -a["vz_end"])
    out["cor"] = _stats("vertical restitution at bounces", np.array(es))

    # Feet → bounce average horizontal speed vs the fit's, for hits that bounce.
    ratios = []
    for r in flights:
        if not (r["ok"] and r["start_kind"] == "hit" and r["end_kind"] == "bounce"):
            continue
        h, b = by_id.get(r["start_event_id"]), by_id.get(r["end_event_id"])
        if h is None or b is None or not np.isfinite([ev.cx[h], ev.cy[h], ev.cx[b]]).all():
            continue
        dt = ev.t[b] - ev.t[h]
        d_ref = np.hypot(ev.cx[b] - ev.cx[h], ev.cy[b] - ev.cy[h])
        if d_ref < 8 or dt <= 0:
            continue
        d_fit = np.hypot(r["end_x"] - r["p0_x"], r["end_y"] - r["p0_y"])
        ratios.append((d_fit / dt) / (d_ref / dt))
    out["horizontal"] = _stats("fit / feet-to-bounce average horizontal speed", np.array(ratios))
    return out


def main(paths: list[str]) -> None:
    for p in paths:
        res = check_session(Path(p))
        print(f"== {res['session']}")
        for k in ("g", "cd", "cor", "horizontal"):
            print("  " + res[k])
        if res["g_wmean"]:
            err = res["g_wmean"] / ph.G - 1
            print(
                f"  -> g off by {err:+.1%}: speed scale error between {err / 2:+.1%} "
                f"(clock) and {err:+.1%} (length)"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
