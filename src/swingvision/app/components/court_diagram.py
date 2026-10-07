"""Top-down court diagram (Plotly, meters) for the minimap, heatmaps and, later, landings.

Orientation matches the camera: the near baseline is at the bottom, ``x`` grows to the right.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from swingvision.app import units
from swingvision.court import model

SURFACE = "#3d7a5a"
RUNOFF = "#b4505f"
LINE = "rgba(255,255,255,0.95)"
NET = "rgba(240,240,240,0.9)"
ME_COLOR = "#ffd43b"
MACHINE_COLOR = "#4dabf7"

#: Extent drawn around the doubles court (m): the tracking ROI.
VIEW_X = model.HALF_DOUBLES + 3.5
VIEW_Y = model.HALF_LENGTH + 6.0


def _line_shapes() -> list[dict]:
    shapes = [
        {
            "type": "rect",
            "x0": -VIEW_X,
            "x1": VIEW_X,
            "y0": -VIEW_Y,
            "y1": VIEW_Y,
            "fillcolor": RUNOFF,
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "x0": -model.HALF_DOUBLES,
            "x1": model.HALF_DOUBLES,
            "y0": -model.HALF_LENGTH,
            "y1": model.HALF_LENGTH,
            "fillcolor": SURFACE,
            "line": {"width": 0},
            "layer": "below",
        },
    ]
    for line in model.COURT_LINES:
        shapes.append(
            {
                "type": "line",
                "x0": line.p0[0],
                "y0": line.p0[1],
                "x1": line.p1[0],
                "y1": line.p1[1],
                "line": {"color": LINE, "width": 1.5},
                "layer": "above",
            }
        )
    shapes.append(
        {
            "type": "line",
            "x0": -model.X_NET_POST,
            "x1": model.X_NET_POST,
            "y0": 0,
            "y1": 0,
            "line": {"color": NET, "width": 2.5, "dash": "dot"},
            "layer": "above",
        }
    )
    return shapes


def court_figure(height: int = 460, title: str | None = None) -> go.Figure:
    """An empty court; add traces in court meters."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin={"l": 4, "r": 4, "t": 28 if title else 4, "b": 4},
        title={"text": title, "font": {"size": 13}, "x": 0.02} if title else None,
        shapes=_line_shapes(),
        xaxis={
            "range": [-VIEW_X, VIEW_X],
            "visible": False,
            "fixedrange": True,
            "constrain": "domain",
        },
        yaxis={
            "range": [-VIEW_Y, VIEW_Y],
            "visible": False,
            "fixedrange": True,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        dragmode=False,
    )
    return fig


def click_grid(step_m: float = 0.5) -> go.Scatter:
    """Invisible points covering the view so clicks anywhere report a court position."""
    xs = np.arange(-VIEW_X, VIEW_X + 1e-9, step_m)
    ys = np.arange(-VIEW_Y, VIEW_Y + 1e-9, step_m)
    gx, gy = np.meshgrid(xs, ys)
    return go.Scatter(
        x=np.round(gx.ravel(), 2),
        y=np.round(gy.ravel(), 2),
        mode="markers",
        marker={"opacity": 0, "size": 10},
        hoverinfo="skip",
        name="click",
    )


def minimap_figure(
    machine_xy: tuple[float, float] | None = None,
    clickable: bool = False,
    extra_traces: list | None = None,
):
    """Live minimap. Trace 0: trail, 1: player, 2: recent bounces, 3: the current shot's
    landing (all moved client-side), then ``extra_traces``, the machine, the click grid.

    In imperial units the player's and bounces' hover text reads ``customdata`` (x, y in
    feet), which the client sets along with their positions."""
    u = units.current()
    fig = court_figure(height=420)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line={"color": ME_COLOR, "width": 2},
            opacity=0.6,
            hoverinfo="skip",
            name="trail",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"color": ME_COLOR, "size": 13, "line": {"color": "#222", "width": 1.5}},
            customdata=[],
            hovertemplate=_xy_hover(u, "x ", 1) + "<extra>me</extra>",
            name="me",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"color": "#4dabf7", "size": 9, "line": {"color": "#fff", "width": 1}},
            customdata=[],
            hovertemplate=_xy_hover(u, "bounce x ", 2) + "<extra></extra>",
            name="bounces",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers+text",
            marker={"color": "#fff", "size": 14, "symbol": "x", "line": {"width": 0}},
            text=[],
            textposition="top center",
            textfont={"color": "white", "size": 12},
            hoverinfo="skip",
            name="shot landing",
        )
    )
    for tr in extra_traces or []:
        fig.add_trace(tr)
    if machine_xy is not None:
        fig.add_trace(
            go.Scatter(
                x=[machine_xy[0]],
                y=[machine_xy[1]],
                mode="markers+text",
                marker={"color": MACHINE_COLOR, "size": 12, "symbol": "square"},
                text=["machine"],
                textposition="bottom center",
                textfont={"color": "white", "size": 10},
                hoverinfo="skip",
                name="machine",
            )
        )
    if clickable:
        fig.add_trace(click_grid())
    return fig


def _xy_hover(u: units.Units, prefix: str, nd: int) -> str:
    """``x 1.2 m, y 3.4 m``: from the point itself in metres, from ``customdata`` (display
    units) otherwise."""
    if not u.imperial:
        return f"{prefix}%{{x:.{nd}f}} m, y %{{y:.{nd}f}} m"
    return (
        f"{prefix}%{{customdata[0]:.{nd}f}} {u.len_unit}, y %{{customdata[1]:.{nd}f}} {u.len_unit}"
    )


def heatmap_figure(xc: np.ndarray, yc: np.ndarray, H: np.ndarray, height: int = 460) -> go.Figure:
    """Time spent per cell (seconds) over the court.

    A magnitude, so one hue (the player's yellow) ramps from transparent to opaque over the
    court. Lightly smoothed, and scaled to the 99th percentile so one spot where the player
    stood for minutes doesn't wash out everything else.
    """
    from scipy.ndimage import gaussian_filter

    u = units.current()
    fig = court_figure(height=height)
    if not H.any():
        return fig
    smooth = gaussian_filter(H, 0.8)
    # Time per spot is heavy-tailed (a feeding spot gets minutes): color by its square root.
    root = np.sqrt(smooth)
    top = float(np.percentile(root[root > 0], 99)) or 1.0
    z = np.where(root > 0.05 * top, root, np.nan)
    r, g, b = (int(ME_COLOR[i : i + 2], 16) for i in (1, 3, 5))
    fig.add_trace(
        go.Heatmap(
            x=xc,
            y=yc,
            z=z,
            # Per cell: x, y (display units), seconds.
            customdata=np.dstack(
                np.broadcast_arrays(
                    u.len(np.asarray(xc, dtype=float))[None, :],
                    u.len(np.asarray(yc, dtype=float))[:, None],
                    smooth,
                )
            ),
            zmin=0,
            zmax=top,
            colorscale=[
                [0.0, f"rgba({r},{g},{b},0.05)"],
                [0.3, f"rgba({r},{g},{b},0.45)"],
                [1.0, f"rgba({r},{g},{b},0.95)"],
            ],
            zsmooth="best",
            showscale=False,
            hovertemplate=(
                f"x %{{customdata[0]:.1f}} {u.len_unit}, y %{{customdata[1]:.1f}} {u.len_unit}"
                "<br>%{customdata[2]:.0f} s<extra></extra>"
            ),
        )
    )
    return fig
