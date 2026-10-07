"""Shots on the session page (M4): speeds, landings, fitted flights.

The browser gets one record per shot (time, speed, outcome, landing, and the fitted 3D
path projected into the video in percent of the frame). Client callbacks pick the current
shot from the playback time; the server only renders the side view when it changes.
"""

from __future__ import annotations

import json

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
import pyarrow as pa
from dash import dcc, html

from swingvision.analysis.shots import SPEED_SCALE_ERROR, speed_error_kmh, speed_error_text
from swingvision.app import units
from swingvision.app.components.ui import icon, stat_grid, stat_tile
from swingvision.court import calibration as calib
from swingvision.court import model as court_model
from swingvision.pose.strokes import STROKE_LABELS
from swingvision.storage import tables
from swingvision.storage.schemas import Calibration

OUTCOME_COLORS = {
    "in": "#51cf66",
    "out_long": "#ff6b6b",
    "out_wide": "#ff6b6b",
    "net": "#cc5de8",
    "own_side": "#868e96",
    "unknown": "#adb5bd",
}
OUTCOME_LABELS = {
    "in": "In",
    "out_long": "Out long",
    "out_wide": "Out wide",
    "net": "Net",
    "own_side": "Own side",
    "unknown": "Unknown",
}
#: A shot stays "current" this long after it lands (or its flight ends).
LINGER_S = 1.5
PATH_COLOR = "#ff922b"

STROKE_SHORT = {
    "serve": "Serve",
    "forehand": "FH",
    "backhand": "BH",
    "forehand_volley": "FH volley",
    "backhand_volley": "BH volley",
    "overhead": "Overhead",
    "other": "–",
}


def load_shots(session) -> tuple[pa.Table | None, pa.Table | None]:
    shots = tables.read_table(session.shots_path) if session.shots_path.exists() else None
    paths = (
        tables.read_table(session.ball_flight_paths_path)
        if session.ball_flight_paths_path.exists()
        else None
    )
    return shots, paths


def over_net(r: dict) -> bool:
    """Shots meant to cross the net (not a dribble or a drop feed on the hitter's side)."""
    return r["outcome"] != "own_side" and "contact_not_at_hitter" not in r["quality_flags"]


def _r(v, nd=1):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def shots_store(
    shots: pa.Table | None, paths: pa.Table | None, cal: Calibration | None, width: int, height: int
) -> dict | None:
    """Per-shot records for client callbacks (columns of equal length).

    ``v``/``e`` (speed and its error) are in display units; ``cx``/``cy`` stay court metres.
    """
    if shots is None or shots.num_rows == 0:
        return None
    un = units.current()
    rows = [r for r in shots.to_pylist() if over_net(r)]
    if not rows:
        return None
    by_flight: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if paths is not None and paths.num_rows and cal is not None:
        fid = paths.column("flight_id").to_numpy()
        t = paths.column("t_s").to_numpy()
        P = np.column_stack(
            [paths.column(c).to_numpy().astype(np.float64) for c in ("x", "y", "z")]
        )
        wanted = {r["flight_id"] for r in rows if r["flight_id"] is not None}
        cams, which = calib.cameras_for_times(cal, t)
        px = np.full((len(t), 2), np.nan)
        for k, cam in enumerate(cams):
            m = which == k
            if m.any():
                px[m] = cam.project(P[m])
        order = np.argsort(fid, kind="stable")
        starts = np.searchsorted(fid[order], np.unique(fid))
        for u, a, b in zip(np.unique(fid), starts, np.r_[starts[1:], len(order)], strict=True):
            if int(u) in wanted:
                idx = order[a:b]
                by_flight[int(u)] = (t[idx], px[idx])
    out: dict = {k: [] for k in ("id", "t0", "t1", "v", "e", "o", "cx", "cy", "path")}
    for r in rows:
        t_end = r["t_contact"] + (r["flight_time_s"] or 1.0)
        path = []
        f = by_flight.get(r["flight_id"]) if r["flight_id"] is not None else None
        if f is not None:
            q = f[1]
            ok = np.isfinite(q).all(axis=1)
            path = np.round(
                np.column_stack([q[ok, 0] / width * 100, q[ok, 1] / height * 100]), 2
            ).tolist()
        out["id"].append(r["shot_id"])
        out["t0"].append(round(r["t_contact"], 3))
        out["t1"].append(round(t_end, 3))
        out["v"].append(_r(un.speed(r["speed_racket_kmh"]), 0))
        out["e"].append(_r(un.speed(_error(r)), 0))
        out["o"].append(r["outcome"])
        out["cx"].append(_r(r["landing_x"], 2))
        out["cy"].append(_r(r["landing_y"], 2))
        out["path"].append(path)
    return out


def _error(r: dict) -> float | None:
    return speed_error_kmh(r["speed_racket_kmh"], r["speed_sigma_kmh"])


def _speed_pm(r: dict, u: units.Units | None = None) -> str:
    """'139 ± 10' (display speed units), with the uncalibrated error bound."""
    u = u or units.current()
    v, e = u.speed(r["speed_racket_kmh"]), u.speed(_error(r))
    if v is None:
        return "–"
    return f"{v:.0f}" + ("" if e is None else f" ± {e:.0f}")


def uncalibrated_badge():
    return dmc.Tooltip(
        dmc.Badge(
            f"Uncalibrated · ±{SPEED_SCALE_ERROR:.0%}",
            color="yellow",
            variant="light",
            size="md",
            tt="none",
            leftSection=icon("tabler:info-circle", 13),
            style={"cursor": "help"},
        ),
        label=speed_error_text(),
        multiline=True,
        w=320,
        withArrow=True,
        position="bottom",
    )


def shot_summary(shots: pa.Table | None) -> dict:
    rows = [r for r in (shots.to_pylist() if shots is not None else []) if over_net(r)]
    called = [r for r in rows if r["outcome"] in ("in", "out_long", "out_wide", "net")]
    speeds = np.array(
        [
            r["speed_racket_kmh"]
            for r in rows
            if r["speed_racket_kmh"] is not None and "speed_uncertain" not in r["quality_flags"]
        ]
    )
    return {
        "n": len(rows),
        "called": len(called),
        "in": sum(r["outcome"] == "in" for r in called),
        "median": float(np.median(speeds)) if len(speeds) else None,
        "max": float(speeds.max()) if len(speeds) else None,
        "n_speed": len(speeds),
        "hidden": (shots.num_rows if shots is not None else 0) - len(rows),
    }


def _fmt_t(t: float) -> str:
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


def shots_table(shots: pa.Table | None, video_id: str = "review-video"):
    """The shots over the net, one row each; clicking a row plays the shot.

    Rendered in the browser (``assets/sv_table.js``) from one component: as Dash components
    (one per cell) a session's 600 shots slowed every update on the page to ~0.4 s.
    """
    u = units.current()
    rows = [r for r in (shots.to_pylist() if shots is not None else []) if over_net(r)]
    body = []
    for r in rows:
        v = r["speed_racket_kmh"]
        uncertain = "speed_uncertain" in r["quality_flags"]
        o = r["outcome"]
        body.append(
            {
                "t": round(max(0.0, r["t_contact"] - 0.8), 3),
                "cells": [
                    _fmt_t(r["t_contact"]),
                    STROKE_SHORT.get(r.get("stroke_type") or "", "–"),
                    "–" if v is None else _speed_pm(r, u) + (" ?" if uncertain else ""),
                    {
                        "badge": OUTCOME_LABELS.get(o, o),
                        "color": OUTCOME_COLORS.get(o, OUTCOME_COLORS["unknown"]),
                    },
                    "–" if r["net_clearance_m"] is None else f"{u.len(r['net_clearance_m']):+.2f}",
                ],
            }
        )
    spec = {
        "head": [
            {"label": "Time"},
            {"label": "Stroke"},
            {"label": u.speed_unit, "align": "right"},
            {"label": "Landing"},
            {"label": f"Net {u.len_unit}", "align": "right"},
        ],
        "rows": body,
        "video": video_id,
    }
    return dmc.ScrollArea(
        html.Div(id="review-shots-table", **{"data-sv-table": json.dumps(spec)}), h=240
    )


def shots_card(shots: pa.Table | None):
    title = dmc.Title("Shots", order=5)
    if shots is None:
        body = [dmc.Text("Shot analysis (3D flight) hasn't run yet.", size="sm", c="dimmed")]
        return dmc.Paper([title, *body], p="md", withBorder=True)
    u = units.current()
    s = shot_summary(shots)
    in_pct = f"{s['in'] / s['called']:.0%}" if s["called"] else "–"
    body = [
        stat_grid(
            [
                stat_tile("Over the net", str(s["n"]), f"{s['called']} with a landing"),
                stat_tile("In", in_pct, f"{s['in']} of {s['called']}"),
                stat_tile(
                    "Median speed",
                    u.speed_str(s["median"]),
                    f"off the racket, ± {SPEED_SCALE_ERROR:.0%}",
                ),
                stat_tile("Fastest", u.speed_str(s["max"]), "off the racket"),
            ],
            plain=True,
        ),
        dmc.Divider(my=6),
        html.Div(
            dmc.Text("Play the video: the current shot shows here.", size="xs", c="dimmed"),
            id="review-shot-detail",
        ),
        dcc.Graph(
            id="review-shot-side",
            figure=side_view_figure(None),
            config={"displayModeBar": False},
            style={"height": 170},
        ),
        shots_table(shots),
        dmc.Text(
            f"{s['hidden']} hits that stayed on the hitter's side (dribbles, drop feeds) or "
            "weren't at the player are hidden. '?' marks uncertain speeds (left out of the "
            "median and fastest). "
            "Click a row to play the shot.",
            size="xs",
            c="dimmed",
        ),
    ]
    return dmc.Paper(
        [
            dmc.Group(
                [
                    dmc.Group([title, dmc.Badge(str(s["n"]), variant="light")], gap="xs"),
                    uncalibrated_badge(),
                ],
                justify="space-between",
            ),
            *body,
        ],
        p="md",
        withBorder=True,
    )


def shot_detail(r: dict | None):
    if r is None:
        return dmc.Text("No shot at this moment.", size="xs", c="dimmed")

    u = units.current()

    def kmh(v):
        return "–" if v is None else f"{u.speed(v):.0f}"

    o = r["outcome"]
    speed = _speed_pm(r, u)
    spin = {1: "topspin", -1: "slice/backspin", 0: "no clear spin"}.get(r["spin_sign"], "–")
    stroke = STROKE_LABELS.get(r.get("stroke_type") or "")
    parts = [
        f"{_fmt_t(r['t_contact'])} · "
        + (f"{stroke} · " if stroke else "")
        + f"{speed} {u.speed_unit} off the racket",
        f"net {kmh(r['speed_net_kmh'])} · before bounce {kmh(r['speed_bounce_kmh'])} "
        f"{u.speed_unit}"
        + (
            f" · ± is the uncalibrated error ({SPEED_SCALE_ERROR:.0%} + 2× fit σ "
            f"{u.speed(r['speed_sigma_kmh']):.0f})"
            if r["speed_sigma_kmh"] is not None
            else ""
        ),
    ]
    geo = []
    if r["net_clearance_m"] is not None:
        geo.append(f"{u.len_str(r['net_clearance_m'], 2, sign=True)} over the net")
    if r["apex_m"] is not None:
        geo.append(f"apex {u.len_str(r['apex_m'], 2)}")
    if r["contact_height"] is not None:
        geo.append(f"contact {u.len_str(r['contact_height'], 2)} high")
    geo.append(spin)
    land = "no landing"
    if r["landing_x"] is not None:
        land = f"landed ({u.len(r['landing_x']):.2f}, {u.len(r['landing_y']):.2f}) {u.len_unit}"
        if r["landing_margin_m"] is not None:
            land += f", {u.small_str(abs(r['landing_margin_m']))} " + (
                "inside" if r["landing_margin_m"] >= 0 else "outside"
            )
        if r["landing_sigma_m"] is not None:
            land += f" (± {u.small_str(r['landing_sigma_m'])})"
        if r["landing_source"] == "fit":
            land += ", extrapolated"
    return dmc.Stack(
        [
            dmc.Group(
                [
                    dmc.Text(parts[0], size="sm", fw=600),
                    dmc.Badge(
                        OUTCOME_LABELS.get(o, o), color=OUTCOME_COLORS.get(o, "gray"), size="sm"
                    ),
                ],
                gap="xs",
            ),
            dmc.Text(parts[1], size="xs"),
            dmc.Text(" · ".join(geo), size="xs"),
            dmc.Text(land, size="xs", c="dimmed"),
        ],
        gap=0,
    )


def side_view_figure(path: tuple[np.ndarray, np.ndarray, int] | None) -> go.Figure:
    """Height over distance along the court, from the hitter's baseline (left) to the
    opponent's (right). ``path``: (y, z, side) of the fitted flight, in metres; both axes
    are drawn in display length units."""
    u = units.current()
    k = u.len_factor
    L = court_model.HALF_LENGTH * k
    sl = court_model.SERVICE_LINE_FROM_NET * k
    fig = go.Figure()
    fig.add_shape(
        type="line", x0=-L - 2 * k, x1=L + 3 * k, y0=0, y1=0, line={"color": "#888", "width": 1}
    )
    fig.add_shape(
        type="line", x0=0, x1=0, y0=0, y1=court_model.NET_HEIGHT_CENTER * k,
        line={"color": "#aaa", "width": 3},
    )  # fmt: skip
    for x in (-L, -sl, sl, L):
        fig.add_shape(
            type="line", x0=x, x1=x, y0=0, y1=0.15 * k, line={"color": "#aaa", "width": 2}
        )
    top = 3.5 * k
    if path is not None:
        y, z, side = path
        d = -side * np.asarray(y, dtype=float) * k
        h = np.asarray(z, dtype=float) * k
        top = max(top, float(np.nanmax(h)) + 0.4 * k)
        fig.add_trace(
            go.Scatter(
                x=d,
                y=h,
                mode="lines",
                line={"color": PATH_COLOR, "width": 2.5},
                hovertemplate=(
                    f"%{{x:.1f}} {u.len_unit} · %{{y:.2f}} {u.len_unit} high<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=170,
        margin={"l": 30, "r": 6, "t": 6, "b": 24},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "range": [-L - 2 * k, L + 3 * k],
            "tickvals": [-L, 0, L],
            "ticktext": ["baseline", "net", "baseline"],
            "showgrid": False,
            "zeroline": False,
            "fixedrange": True,
        },
        yaxis={"range": [0, top], "title": u.len_unit, "showgrid": False, "fixedrange": True},
    )
    return fig


def landing_traces(shots: pa.Table | None) -> list[go.Scatter]:
    """Static landing dots for the court map, one trace per outcome color (court metres;
    the hover text in display units)."""
    u = units.current()
    rows = [
        r
        for r in (shots.to_pylist() if shots is not None else [])
        if over_net(r) and r["landing_x"] is not None
    ]
    out = []
    for o in ("in", "out_long", "out_wide", "unknown"):
        sel = [r for r in rows if r["outcome"] == o]
        if not sel:
            continue
        out.append(
            go.Scatter(
                x=[r["landing_x"] for r in sel],
                y=[r["landing_y"] for r in sel],
                mode="markers",
                marker={"color": OUTCOME_COLORS[o], "size": 6, "opacity": 0.55},
                customdata=[
                    [
                        r["t_contact"],
                        u.speed(r["speed_racket_kmh"] or float("nan")),
                        u.speed(_error(r) or 0.0),
                        u.len(r["landing_x"]),
                        u.len(r["landing_y"]),
                    ]
                    for r in sel
                ],
                hovertemplate=(
                    f"%{{customdata[1]:.0f}} ± %{{customdata[2]:.0f}} {u.speed_unit} · "
                    + OUTCOME_LABELS[o]
                    + f"<br>(%{{customdata[3]:.2f}}, %{{customdata[4]:.2f}}) {u.len_unit}"
                    + "<extra></extra>"
                ),
                name="landings:" + o,
            )
        )
    return out
