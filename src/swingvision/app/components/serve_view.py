"""The Stats page's "Serve contact" section (PLAN.md §7.11, M7b): where the ball was struck
relative to the front toe, and how speed and in % change with it, for one session or a
selection. Everything is drawn from the ``serves`` records via
:mod:`swingvision.analysis.serve_stats`.

Serves carry their session and time (``customdata``: ``[..., session_id, t]``), so a click
plays the serve on its session page.
"""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from swingvision.analysis import serve_stats as ss
from swingvision.app import units
from swingvision.app.components.stats_view import GRID, _layout, _where, empty_figure
from swingvision.app.components.ui import stat_grid, stat_tile

#: Speed colours (low → high) and the fault marker.
SPEED_SCALE = "Viridis"
FAULT_COLOR = "#fa5252"
IN_COLOR = "#40c057"
NO_SPEED_COLOR = "#868e96"
SPEED_COLOR = "#4dabf7"


def _foot_outline(u) -> go.Scatter:
    """A shoe at the origin pointing forward (toe tip at 0, heel 27 cm back), in display units."""
    t = np.linspace(0, np.pi, 16)
    front = np.column_stack([0.05 * np.cos(t), -0.05 + 0.05 * np.sin(t)])  # rounded toe
    back = np.array([[-0.045, -0.24], [-0.035, -0.27], [0.035, -0.27], [0.045, -0.24]])
    pts = np.vstack([front, back, front[:1]])
    return go.Scatter(
        x=u.small(pts[:, 0]),
        y=u.small(pts[:, 1]),
        mode="lines",
        line={"color": "rgba(134,142,150,0.8)", "width": 1.5},
        fill="toself",
        fillcolor="rgba(134,142,150,0.15)",
        hoverinfo="skip",
        showlegend=False,
    )


def _hover(r: dict, labels: dict[str, str], u) -> str:
    call = {"in": "in", "out_long": "long", "out_wide": "wide", "net": "net"}.get(
        r.get("outcome") or "", "no call"
    )
    sp = u.speed_str(r["speed_kmh"]) if r.get("speed_ok") else "no speed"
    side = f" · {r['serve_side']}" if r.get("serve_side") else ""
    return (
        f"{_where(r, labels)}{side} · {sp} · {call}<br>"
        f"{u.small_str(r['forward_m'], sign=True)} ± {u.small_str(r['forward_sigma_m'])} forward, "
        f"{u.small_str(r['lateral_m'], sign=True)} ± {u.small_str(r['lateral_sigma_m'])} "
        f"to the racket side, {u.len_str(r['height_m'], 2)} high"
    )


def _dots(rows: list[dict], x_key: str, y_key: str, labels, u, scale_y=None) -> list[go.Scatter]:
    """Serves as dots coloured by speed (grey: no speed), hollow for faults."""
    traces = []
    speeds = [r["speed_kmh"] for r in rows if r.get("speed_ok")]
    lo, hi = (min(speeds), max(speeds)) if speeds else (0, 1)
    scale_y = scale_y or u.small
    shown_scale = False
    for faults in (False, True):
        for has_speed in (True, False):
            rs = [
                r
                for r in rows
                if (r.get("outcome") in ("out_long", "out_wide", "net")) == faults
                and bool(r.get("speed_ok")) == has_speed
            ]
            if not rs:
                continue
            marker = {
                "size": 10,
                "symbol": "circle-open" if faults else "circle",
                "line": {"width": 2 if faults else 0.5, "color": "rgba(0,0,0,0.35)"},
            }
            if has_speed:
                marker.update(
                    color=[u.speed(r["speed_kmh"]) for r in rs],
                    colorscale=SPEED_SCALE,
                    cmin=u.speed(lo),
                    cmax=u.speed(hi),
                    showscale=not shown_scale,
                    colorbar={"title": {"text": u.speed_unit}, "thickness": 10, "len": 0.8},
                )
                shown_scale = True
            else:
                marker["color"] = NO_SPEED_COLOR
            name = ("faults" if faults else "in / no call") + ("" if has_speed else ", no speed")
            traces.append(
                go.Scatter(
                    x=[u.small(r[x_key]) for r in rs],
                    y=[scale_y(r[y_key]) for r in rs],
                    mode="markers",
                    name=name,
                    marker=marker,
                    customdata=[[_hover(r, labels, u), r.get("session_id"), r["t"]] for r in rs],
                    hovertemplate="%{customdata[0]}<extra></extra>",
                )
            )
    return traces


def top_figure(rows: list[dict], labels: dict[str, str] | None = None) -> go.Figure:
    """Top-down: the toe at the origin, forward up, the racket side to the right."""
    if not rows:
        return empty_figure("No serve contact points yet.", 380)
    u = units.current()
    fig = go.Figure([_foot_outline(u), *_dots(rows, "lateral_m", "forward_m", labels or {}, u)])
    # The σ ellipses (1σ) as faint outlines, so the precision is visible at a glance.
    t = np.linspace(0, 2 * np.pi, 24)
    xs, ys = [], []
    for r in rows:
        xs += [*u.small(r["lateral_m"] + r["lateral_sigma_m"] * np.cos(t)), None]
        ys += [*u.small(r["forward_m"] + r["forward_sigma_m"] * np.sin(t)), None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"width": 0.6, "color": "rgba(134,142,150,0.35)"},
            hoverinfo="skip",
            visible="legendonly",
            name="1σ",
        )
    )
    base = [r.get("toe_to_baseline_m") for r in rows if r.get("toe_to_baseline_m") is not None]
    if base:
        b = float(np.median(base))
        fig.add_hline(
            y=u.small(b),
            line={"color": "rgba(255,255,255,0.55)", "width": 3},
            annotation_text="baseline (median)",
            annotation_position="top left",
            annotation_font_size=10,
        )
    span = max(
        0.6,
        *(abs(r["lateral_m"]) + 0.15 for r in rows),
        *(abs(r["forward_m"]) + 0.15 for r in rows),
    )
    rng = [-u.small(span), u.small(span)]
    return _layout(
        fig,
        380,
        legend={"orientation": "h", "y": -0.18, "font": {"size": 11}},
        xaxis={
            "title": f"to the racket side ({u.small_unit})",
            "range": rng,
            "zeroline": True,
            "zerolinecolor": GRID,
            "gridcolor": GRID,
            "fixedrange": True,
        },
        yaxis={
            "title": f"in front of the toe ({u.small_unit})",
            "range": rng,
            "scaleanchor": "x",
            "zeroline": True,
            "zerolinecolor": GRID,
            "gridcolor": GRID,
            "fixedrange": True,
        },
    )


def side_figure(rows: list[dict], labels: dict[str, str] | None = None) -> go.Figure:
    """Side view: contact height against forward of the toe."""
    if not rows:
        return empty_figure("No serve contact points yet.", 380)
    u = units.current()
    fig = go.Figure(_dots(rows, "forward_m", "height_m", labels or {}, u, scale_y=u.len))
    for tr in fig.data:
        tr.marker.showscale = False
    fig.add_vline(x=0, line={"color": "rgba(134,142,150,0.6)", "dash": "dot"})
    return _layout(
        fig,
        380,
        showlegend=False,
        xaxis={
            "title": f"in front of the toe ({u.small_unit})",
            "gridcolor": GRID,
            "fixedrange": True,
        },
        yaxis={"title": f"contact height ({u.len_unit})", "gridcolor": GRID, "fixedrange": True},
    )


def effect_figure(rows: list[dict]) -> go.Figure:
    """Per axis: quantile bins of ≥ 15 serves, mean speed ± 95% and in % (Wilson 95%)."""
    u = units.current()
    fig = make_subplots(
        rows=2, cols=3, shared_xaxes="columns", vertical_spacing=0.08, horizontal_spacing=0.06,
        subplot_titles=[ss.AXIS_LABELS[a] for a in ss.AXES],
    )  # fmt: skip
    any_bins = False
    for c, axis in enumerate(ss.AXES, start=1):
        bins = ss.quantile_bins(rows, axis)
        if not bins:
            continue
        any_bins = True
        conv = u.len if axis == "height_m" else u.small
        rng = [f"{conv(b['lo']):.0f}…{conv(b['hi']):.0f}" if axis != "height_m"
               else f"{conv(b['lo']):.2f}…{conv(b['hi']):.2f}" for b in bins]  # fmt: skip
        sp = [b for b in bins if b["speed_mean"] is not None]
        fig.add_trace(
            go.Scatter(
                x=[conv(b["mid"]) for b in sp],
                y=[u.speed(b["speed_mean"]) for b in sp],
                error_y={
                    "type": "data",
                    "array": [u.speed(b["speed_hi"]) - u.speed(b["speed_mean"]) if b["speed_hi"] else 0 for b in sp],
                    "arrayminus": [u.speed(b["speed_mean"]) - u.speed(b["speed_lo"]) if b["speed_lo"] else 0 for b in sp],
                },
                mode="lines+markers",
                line={"color": SPEED_COLOR},
                customdata=[[r, b["n_speed"]] for r, b in zip(rng, sp, strict=False)],
                hovertemplate=f"%{{customdata[0]}}: %{{y:.0f}} {u.speed_unit} (n = %{{customdata[1]}})<extra></extra>",
                showlegend=False,
            ),
            row=1, col=c,
        )  # fmt: skip
        ip = [b for b in bins if b["in_pct"] is not None]
        fig.add_trace(
            go.Scatter(
                x=[conv(b["mid"]) for b in ip],
                y=[100 * b["in_pct"] for b in ip],
                error_y={
                    "type": "data",
                    "array": [100 * (b["in_hi"] - b["in_pct"]) for b in ip],
                    "arrayminus": [100 * (b["in_pct"] - b["in_lo"]) for b in ip],
                },
                mode="lines+markers",
                line={"color": IN_COLOR},
                customdata=[[r, b["n_called"]] for r, b in zip(rng, ip, strict=False)],
                hovertemplate="%{customdata[0]}: %{y:.0f}% in (n = %{customdata[1]})<extra></extra>",
                showlegend=False,
            ),
            row=2, col=c,
        )  # fmt: skip
        fig.update_xaxes(
            title_text=f"{u.len_unit if axis == 'height_m' else u.small_unit}", row=2, col=c,
            gridcolor=GRID, fixedrange=True,
        )  # fmt: skip
        fig.update_xaxes(gridcolor=GRID, fixedrange=True, row=1, col=c)
    if not any_bins:
        return empty_figure(
            f"Effect charts need at least {ss.MIN_BIN} serves with a contact point.", 300
        )
    fig.update_yaxes(title_text=u.speed_unit, row=1, col=1)
    fig.update_yaxes(title_text="in %", row=2, col=1)
    fig.update_yaxes(gridcolor=GRID, fixedrange=True)
    fig.update_annotations(font_size=12)
    fig = _layout(fig, 440)
    fig.update_layout(margin={"t": 30})
    return fig


def grid_figure(rows: list[dict]) -> go.Figure:
    """Forward × lateral cells of 10 cm: in % (colour) and median speed (text)."""
    cells = ss.grid(rows)
    if not cells:
        return empty_figure(
            f"The grid shows cells with at least {ss.MIN_CELL} serves; none has that many yet.", 300
        )
    u = units.current()
    xs = sorted({c["lateral_lo"] for c in cells})
    ys = sorted({c["forward_lo"] for c in cells})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    Z = np.full((len(ys), len(xs)), np.nan)
    T = [["" for _ in xs] for _ in ys]
    for c in cells:
        i, j = yi[c["forward_lo"]], xi[c["lateral_lo"]]
        Z[i, j] = 100 * c["in_pct"] if c["in_pct"] is not None else np.nan
        sp = u.speed_str(c["speed_median"]) if c["speed_median"] is not None else "–"
        T[i][j] = f"{sp}<br>n = {c['n']}"

    def lab(v):
        return f"{u.small(v):.0f}…{u.small(v + ss.CELL_M):.0f}"

    fig = go.Figure(
        go.Heatmap(
            z=Z,
            x=[lab(v) for v in xs],
            y=[lab(v) for v in ys],
            text=T,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
            colorbar={"title": {"text": "in %"}, "thickness": 10},
            hovertemplate="%{y} forward, %{x} lateral: %{z:.0f}% in<br>%{text}<extra></extra>",
        )
    )
    return _layout(
        fig,
        300,
        xaxis={"title": f"to the racket side ({u.small_unit})", "fixedrange": True},
        yaxis={"title": f"in front of the toe ({u.small_unit})", "fixedrange": True},
    )


def summary(rows: list[dict], all_rows: list[dict]):
    """KPI tiles, the plain findings (only where the intervals separate), the regression
    details and the caveat."""
    u = units.current()
    o = ss.overview(rows)
    tiles = stat_grid(
        [
            stat_tile("Serves", str(o["n"]), f"of {len(all_rows)} with a contact point"),
            stat_tile(
                "In front",
                u.small_str(o["forward_m_median"]),
                f"median, ± {u.small_str(o['forward_sigma_median'])} each",
            ),
            stat_tile("To the racket side", u.small_str(o["lateral_m_median"]), "median"),
            stat_tile("Height", u.len_str(o["height_m_median"], 2), "median"),
            stat_tile(
                "Toe down",
                "–" if o["on_ground_pct"] is None else f"{o['on_ground_pct']:.0%}",
                "front toe on the ground at contact",
            ),
        ]
    )
    found = ss.findings(rows)
    if found:

        def rng(lo: float, hi: float, axis: str) -> str:
            if axis == "height_m":
                return f"{u.len(lo):.2f}–{u.len(hi):.2f} {u.len_unit}"
            return f"{u.small(lo):.0f}–{u.small(hi):.0f} {u.small_unit}"

        def speed(kmh: float) -> str:
            return f"{u.speed(kmh):+.0f} {u.speed_unit}"

        lines = [dmc.ListItem(ss.summary_text(f, rng, speed)) for f in found]
        text = dmc.List(lines, size="sm")
    else:
        text = dmc.Text(
            "No contact zone stands out yet: no bin's speed or in % differs from the rest with "
            "separated 95% intervals.",
            size="sm",
            c="dimmed",
        )
    reg = ss.regression(rows)
    details = []
    for key, label, unit in (("speed", "Speed", u.speed_unit), ("in", "In %", "points")):
        m = reg.get(key)
        if not m:
            details.append(dmc.Text(f"{label}: too few serves for a model.", size="xs", c="dimmed"))
            continue
        parts = []
        for a in ss.AXES:
            b, lo, hi = m["effects"][a]
            scale = u.speed if key == "speed" else (lambda v: v)
            parts.append(
                f"{ss.AXIS_LABELS[a].lower()} {scale(b):+.1f} ({scale(lo):+.1f} to {scale(hi):+.1f})"
            )
        details.append(
            dmc.Text(
                f"{label} ({'least squares' if key == 'speed' else 'logistic'}, n = {m['n']}), "
                f"per {u.small_str(0.1)} at the average contact, {unit}: " + "; ".join(parts),
                size="xs",
            )
        )
    return dmc.Stack(
        [
            tiles,
            text,
            dmc.Accordion(
                [
                    dmc.AccordionItem(
                        [
                            dmc.AccordionControl("Regression details"),
                            dmc.AccordionPanel(dmc.Stack(details, gap=4)),
                        ],
                        value="reg",
                    )
                ],
                variant="contained",
            ),
            dmc.Text(
                "Serve type (flat, slice, kick) changes both where you toss and the speed and in "
                "%, and isn't detected yet: compare serves of one kind where you can.",
                size="xs",
                c="dimmed",
            ),
        ],
        gap="xs",
    )
