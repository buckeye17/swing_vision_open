"""Stats page pieces (M7, M7a): KPIs, speed by stroke, landing heatmap, depth, strokes
table, movement, trends over sessions. Everything is drawn from
:mod:`swingvision.analysis.stats` records, for one session or a selection of them.

Shots on the charts carry their session and time (``customdata``: ``[..., session_id, t]``),
so a click opens the session at that moment.
"""

from __future__ import annotations

from datetime import datetime

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go

from swingvision.analysis import stats as st
from swingvision.analysis.shots import speed_error_kmh
from swingvision.app import units
from swingvision.app.components.court_diagram import VIEW_Y, court_figure
from swingvision.app.components.shots_view import records_scale_text
from swingvision.app.components.swings_view import STROKE_COLORS
from swingvision.app.components.ui import fmt_duration, stat_grid, stat_tile
from swingvision.court import model as court_model

UNKNOWN_COLOR = "#868e96"
#: Landing density: one hue, transparent → opaque (a magnitude).
LANDING_RGB = (255, 146, 43)
GRID = "rgba(128,128,128,0.15)"
END_FILTERS = {"all": "Both ends", "near": "Near end", "far": "Far end"}


def group_color(g: str) -> str:
    return STROKE_COLORS.get(g, UNKNOWN_COLOR)


def group_label(g: str) -> str:
    return st.GROUP_LABELS.get(g, g)


def pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.0%}"


def num(v: float | None, fmt: str, unit: str = "") -> str:
    return "–" if v is None else f"{v:{fmt}}{unit}"


def fmt_t(t: float | None) -> str:
    if t is None:
        return "–"
    m = int(t // 60)
    return f"{m}:{t - 60 * m:04.1f}"


def short_labels(sessions: list[dict]) -> dict[str, str]:
    """Session id → a short hover prefix (``Oct 1 · ``) when there are several sessions."""
    if len(sessions) <= 1:
        return {}
    out = {}
    for s in sessions:
        when = s.get("recorded_on")
        try:
            dt = datetime.fromisoformat(when) if when else None
        except ValueError:
            dt = None
        day = f"{dt:%b} {dt.day}" if dt else (s.get("name") or s["session_id"])
        out[s["session_id"]] = f"{day} · "
    return out


def _where(r: dict, labels: dict[str, str]) -> str:
    return labels.get(r.get("session_id"), "") + fmt_t(r["t"])


def filtered(records: list[dict], groups: list[str] | None, end: str | None) -> list[dict]:
    return st.filter_records(records, groups, end)


def _layout(fig: go.Figure, height: int, **axes) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 48, "r": 12, "t": 10, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"namelength": 0},
        **axes,
    )
    return fig


def empty_figure(text: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font={"size": 13, "color": "#868e96"})
    return _layout(
        fig,
        height,
        xaxis={"visible": False, "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True},
    )


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------


def kpis(records: list[dict], movement: dict, sessions: list[dict] | None = None):
    u = units.current()
    s = st.summarize_shots(records)
    items = []
    if sessions is not None and len(sessions) != 1:
        video = sum(x.get("duration_s") or 0.0 for x in sessions)
        items.append(stat_tile("Sessions", str(len(sessions)), f"{fmt_duration(video)} of video"))
    items += [
        stat_tile("Shots", str(s["n"]), f"{s['n_seen']} with the contact seen"),
        stat_tile("In", pct(s["in_pct"]), f"{s['n_in']} of {s['n_called']} called"),
        stat_tile("Net", pct(s["net_pct"]), f"{s['n_net']} shots"),
        stat_tile(
            "Speed",
            u.speed_str(s["speed_median"]),
            f"median of {s['n_speed']}, {records_scale_text(records)}",
        ),
        stat_tile("Fastest", u.speed_str(s["speed_max"]), "off the racket"),
    ]
    if movement:
        items += [
            stat_tile(
                "Distance",
                f"{u.len(movement['distance_m']):,.0f} {u.len_unit}",
                f"tracked {fmt_duration(movement['tracked_s'])}",
            ),
            stat_tile(
                "Top speed", u.speed_str_mps(movement["max_speed_mps"]), "running, best 0.5 s"
            ),
        ]
    return stat_grid(items)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def speed_figure(records: list[dict], labels: dict[str, str] | None = None) -> go.Figure:
    """Speed off the racket per stroke: a box per stroke with every shot as a dot."""
    labels = labels or {}
    rows = [r for r in records if r["speed_ok"]]
    if not rows:
        return empty_figure("No shot speeds yet (the ball's contact must be seen).")
    u = units.current()
    fig = go.Figure()
    groups = [g for g in (*st.GROUPS, "unknown") if any(r["group"] == g for r in rows)]
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        errs = [
            speed_error_kmh(r["speed_kmh"], r["speed_sigma_kmh"], r.get("speed_scale_err"))
            for r in rs
        ]
        fig.add_trace(
            go.Box(
                x=[group_label(g)] * len(rs),
                y=[u.speed(r["speed_kmh"]) for r in rs],
                name=group_label(g),
                boxpoints="all",
                jitter=0.45,
                pointpos=0,
                marker={"color": group_color(g), "size": 8, "opacity": 0.75},
                line={"color": group_color(g), "width": 2},
                fillcolor="rgba(0,0,0,0)",
                customdata=[
                    [
                        _where(r, labels),
                        "" if e is None else f" ± {u.speed(e):.0f}",
                        r.get("session_id"),
                        r["t"],
                    ]
                    for r, e in zip(rs, errs, strict=True)
                ],
                hovertemplate=(
                    f"%{{customdata[0]}} · %{{y:.0f}}%{{customdata[1]}} {u.speed_unit}"
                    "<extra></extra>"
                ),
            )
        )
    return _layout(
        fig,
        300,
        showlegend=False,
        xaxis={"fixedrange": True, "showgrid": False},
        yaxis={
            "title": f"{u.speed_unit} off the racket",
            "fixedrange": True,
            "gridcolor": GRID,
        },
    )


def landing_figure(
    records: list[dict], view: str = "heat", labels: dict[str, str] | None = None
) -> go.Figure:
    """Landings seen from the hitter's end (both ends folded onto one picture)."""
    labels = labels or {}
    rows = [r for r in records if r["rel_x"] is not None and r["outcome"] != "own_side"]
    fig = court_figure(height=460)
    fig.update_yaxes(range=[-4.0, VIEW_Y])  # the opponent's half, where shots land
    if not rows:
        fig.add_annotation(
            text="No landings yet", showarrow=False, font={"color": "white", "size": 13}
        )
        return fig
    x = np.array([r["rel_x"] for r in rows])
    y = np.array([r["rel_y"] for r in rows])
    if view == "heat":
        from scipy.ndimage import gaussian_filter

        cell = 0.5
        xe = np.arange(-court_model.HALF_DOUBLES - 3.5, court_model.HALF_DOUBLES + 3.5 + 1e-9, cell)
        ye = np.arange(-court_model.HALF_LENGTH - 6, court_model.HALF_LENGTH + 6 + 1e-9, cell)
        H, _, _ = np.histogram2d(y, x, bins=[ye, xe])
        smooth = gaussian_filter(H, 1.2)
        top = float(smooth.max()) or 1.0
        z = np.where(smooth > 0.06 * top, smooth, np.nan)
        r_, g_, b_ = LANDING_RGB
        fig.add_trace(
            go.Heatmap(
                x=0.5 * (xe[1:] + xe[:-1]),
                y=0.5 * (ye[1:] + ye[:-1]),
                z=z,
                zmin=0,
                zmax=top,
                colorscale=[
                    [0.0, f"rgba({r_},{g_},{b_},0.08)"],
                    [0.35, f"rgba({r_},{g_},{b_},0.5)"],
                    [1.0, f"rgba({r_},{g_},{b_},0.95)"],
                ],
                zsmooth="best",
                showscale=False,
                hoverinfo="skip",
            )
        )
        # The shots themselves stay visible (small, white) on top of the density.
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={"color": "white", "size": 4, "opacity": 0.8},
                text=[f"{_where(r, labels)} · {group_label(r['group'])}" for r in rows],
                customdata=[[r.get("session_id"), r["t"]] for r in rows],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        return fig
    for g in (*st.GROUPS, "unknown"):
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        ins = [r["outcome"] == "in" for r in rs]
        fig.add_trace(
            go.Scatter(
                x=[r["rel_x"] for r in rs],
                y=[r["rel_y"] for r in rs],
                mode="markers",
                name=group_label(g),
                # Out and net shots are hollow, so the call doesn't rely on color.
                marker={
                    "color": group_color(g),
                    "size": 9,
                    "symbol": ["circle" if i else "circle-open" for i in ins],
                    "line": {"color": "white", "width": 1},
                },
                text=[f"{_where(r, labels)} · {group_label(g)} · {r['outcome']}" for r in rs],
                customdata=[[r.get("session_id"), r["t"]] for r in rs],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "y": 0.01,
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "white", "size": 11},
            "bgcolor": "rgba(0,0,0,0.35)",
        },
    )
    return fig


def depth_figure(records: list[dict]) -> go.Figure:
    """How far past the net shots landed, per stroke, with the service line and baseline."""
    rows = [r for r in records if r["rel_y"] is not None and r["rel_y"] > 0]
    if not rows:
        return empty_figure("No landings yet.")
    u = units.current()
    k = u.len_factor
    # Bins of ½ m, or 2 ft in imperial (0.61 m: close, and round in the unit shown).
    size = 2.0 if u.imperial else 0.5
    end = (court_model.HALF_LENGTH + 4) * k
    fig = go.Figure()
    for g in (*st.GROUPS, "unknown"):
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        fig.add_trace(
            go.Histogram(
                x=[r["rel_y"] * k for r in rs],
                name=group_label(g),
                xbins={"start": 0, "end": end, "size": size},
                marker={"color": group_color(g), "line": {"color": "rgba(0,0,0,0)", "width": 0}},
                hovertemplate=group_label(g) + f": %{{y}} shots at %{{x}} {u.len_unit}"
                "<extra></extra>",
            )
        )
    lines = [
        (court_model.SERVICE_LINE_FROM_NET * k, "service line"),
        (court_model.HALF_LENGTH * k, "baseline"),
    ]
    fig.update_layout(
        barmode="stack",
        bargap=0.12,
        shapes=[
            {
                "type": "line",
                "x0": v,
                "x1": v,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "rgba(128,128,128,0.8)", "width": 1.5, "dash": "dash"},
            }
            for v, _ in lines
        ],
        annotations=[
            {
                "x": v,
                "y": 1,
                "yref": "paper",
                "text": label,
                "showarrow": False,
                "xanchor": "right",
                "yanchor": "top",
                "font": {"size": 11, "color": "#868e96"},
            }
            for v, label in lines
        ],
        legend={"orientation": "h", "y": 1.14, "x": 0},
    )
    return _layout(
        fig,
        260,
        showlegend=True,
        xaxis={
            "title": f"landing distance past the net ({u.len_unit})",
            "range": [0, end],
            "fixedrange": True,
            "showgrid": False,
        },
        yaxis={"title": "shots", "fixedrange": True, "gridcolor": GRID},
    )


def distance_figure(dist: dict) -> go.Figure:
    """Distance covered per 5 minutes of one session, or per session for several."""
    if dist.get("per") == "session":
        return _distance_per_session(dist)
    if not dist["t"]:
        return empty_figure("No player tracking yet.", 200)
    u = units.current()
    minutes = dist["bin_s"] / 60
    fig = go.Figure(
        go.Bar(
            x=[t / 60 + minutes / 2 for t in dist["t"]],
            y=u.len(list(dist["distance_m"])),
            width=minutes * 0.85,
            marker={"color": "#ffd43b", "cornerradius": 4},
            customdata=[[f"{t / 60:.0f}–{t / 60 + minutes:.0f} min"] for t in dist["t"]],
            hovertemplate=f"%{{customdata[0]}}: %{{y:.0f}} {u.len_unit}<extra></extra>",
        )
    )
    return _layout(
        fig,
        200,
        showlegend=False,
        xaxis={"title": "minutes into the video", "fixedrange": True, "showgrid": False},
        yaxis={
            "title": f"{u.len_unit} per {minutes:.0f} min",
            "fixedrange": True,
            "gridcolor": GRID,
        },
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _depth_text(g: str, s: dict, u: units.Units | None = None) -> str:
    if s["depth_mean"] is None:
        return "–"
    u = u or units.current()
    line = "service line" if g == "serve" else "baseline"
    sd = "" if s["depth_sd"] is None else f" ± {u.len(s['depth_sd']):.1f}"
    return f"{u.len(s['depth_mean']):.1f}{sd} {u.len_unit} ({line})"


def strokes_table(records: list[dict], swings: dict[str, dict]):
    groups = st.by_group(records)
    if not groups:
        return dmc.Text("No shots yet.", size="sm", c="dimmed")
    u = units.current()
    head = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh(h)
                for h in (
                    "Stroke",
                    "Shots",
                    "In",
                    "Net",
                    "Speed (median / max)",
                    "Short of the line",
                    "Deep",
                    "Swings",
                    "Wrist speed",
                )
            ]
        )
    )
    body = []
    for g, s in groups.items():
        sw = swings.get(g, {})
        body.append(
            dmc.TableTr(
                [
                    dmc.TableTd(
                        dmc.Group(
                            [
                                dmc.Box(
                                    w=10,
                                    h=10,
                                    style={"borderRadius": 3, "background": group_color(g)},
                                ),
                                dmc.Text(group_label(g), size="sm"),
                            ],
                            gap=6,
                            wrap="nowrap",
                        )
                    ),
                    dmc.TableTd(str(s["n"])),
                    dmc.TableTd(pct(s["in_pct"])),
                    dmc.TableTd(pct(s["net_pct"])),
                    dmc.TableTd(
                        "–"
                        if s["speed_median"] is None
                        else f"{u.speed(s['speed_median']):.0f} / "
                        f"{u.speed(s['speed_max']):.0f} {u.speed_unit}"
                    ),
                    dmc.TableTd(_depth_text(g, s, u)),
                    dmc.TableTd("–" if g == "serve" else pct(s["deep_pct"])),
                    dmc.TableTd(str(sw.get("n", "–"))),
                    dmc.TableTd(u.limb_speed_str(sw.get("wrist_speed_median"))),
                ]
            )
        )
    return dmc.ScrollArea(
        dmc.Table(
            [head, dmc.TableTbody(body)],
            highlightOnHover=True,
            verticalSpacing=4,
            fz="sm",
        ),
        type="auto",
    )


def _distance_per_session(dist: dict) -> go.Figure:
    if not dist["distance_m"]:
        return empty_figure("No player tracking yet.", 200)
    u = units.current()
    fig = go.Figure(
        go.Bar(
            x=dist["label"],
            y=u.len(list(dist["distance_m"])),
            marker={"color": "#ffd43b", "cornerradius": 4},
            customdata=[
                [fmt_duration(s), sid]
                for s, sid in zip(dist["tracked_s"], dist["session_id"], strict=True)
            ],
            hovertemplate=f"%{{x}}: %{{y:,.0f}} {u.len_unit} in %{{customdata[0]}} tracked"
            "<extra></extra>",
        )
    )
    return _layout(
        fig,
        220,
        showlegend=False,
        xaxis={"fixedrange": True, "showgrid": False, "tickangle": -30, "automargin": True},
        yaxis={"title": f"{u.len_unit} per session", "fixedrange": True, "gridcolor": GRID},
    )


# ---------------------------------------------------------------------------
# Trends over sessions (M7a)
# ---------------------------------------------------------------------------

#: KPIs whose values are fractions (shown in %).
RATE_KPIS = frozenset(k for k, (_label, kind) in st.TREND_KPIS.items() if kind == "rate")


def trend_value(kpi: str, v: float | None, u: units.Units) -> float | None:
    """A trend value in display units (rates in %)."""
    if v is None:
        return None
    if kpi in RATE_KPIS:
        return 100.0 * v
    if kpi == "speed_median":
        return u.speed(v)
    if kpi in ("depth_mean", "distance_m"):
        return u.len(v)
    if kpi == "wrist_speed":
        return u.limb_speed(v)
    return v


def trend_unit(kpi: str, u: units.Units) -> str:
    if kpi in RATE_KPIS:
        return "%"
    if kpi == "speed_median":
        return u.speed_unit
    if kpi in ("depth_mean", "distance_m"):
        return u.len_unit
    if kpi == "wrist_speed":
        return u.limb_speed_unit
    return ""


def trend_figure(points: list[dict], kpi: str) -> go.Figure:
    """One KPI per session against its recording date, with n and the 95% interval.

    ``customdata[3]`` is the session id (a click opens that session's stats)."""
    pts = [p for p in points if p["value"] is not None]
    if not pts:
        return empty_figure("Nothing to show for this selection yet.", 280)
    u = units.current()
    unit = trend_unit(kpi, u)
    label = st.TREND_KPIS[kpi][0]
    y = [trend_value(kpi, p["value"], u) for p in pts]
    lo = [trend_value(kpi, p["lo"], u) for p in pts]
    hi = [trend_value(kpi, p["hi"], u) for p in pts]
    n_name = "minutes tracked" if kpi == "distance_m" else "n"
    dated = all(p["recorded_on"] for p in pts)
    ci = ["" if a is None else f" (95%: {a:.1f}–{b:.1f})" for a, b in zip(lo, hi, strict=True)]
    fig = go.Figure(
        go.Scatter(
            x=[p["recorded_on"] for p in pts] if dated else [p["label"] for p in pts],
            y=y,
            mode="lines+markers",
            line={"color": "rgba(255,212,59,0.45)", "width": 1.5},
            marker={"color": "#ffd43b", "size": 9, "line": {"color": "white", "width": 1}},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [0 if b is None else b - v for b, v in zip(hi, y, strict=True)],
                "arrayminus": [0 if a is None else v - a for a, v in zip(lo, y, strict=True)],
                "color": "rgba(255,212,59,0.6)",
                "thickness": 1.5,
                "width": 4,
                "visible": any(a is not None for a in lo),
            },
            customdata=[
                [p["label"], p["n"], c, p["session_id"]] for p, c in zip(pts, ci, strict=True)
            ],
            hovertemplate=(
                f"%{{customdata[0]}}<br>{label}: %{{y:.1f}} {unit}%{{customdata[2]}}"
                f"<br>{n_name}: %{{customdata[1]}}<extra></extra>"
            ),
        )
    )
    xaxis = {"fixedrange": True, "showgrid": False}
    if dated:
        xaxis.update(type="date", title="recorded", tickformat="%b %-d", hoverformat="%b %-d %H:%M")
    return _layout(
        fig,
        280,
        showlegend=False,
        xaxis=xaxis,
        yaxis={
            "title": f"{label} ({unit})" if unit else label,
            "fixedrange": True,
            "gridcolor": GRID,
            "rangemode": "tozero" if kpi in RATE_KPIS else "normal",
        },
    )
