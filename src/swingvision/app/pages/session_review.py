"""Session review: proxy video + timeline synced to playback.

``assets/video_sync.js`` publishes the video's current time into the
``review-time`` store (~10 Hz) and moves the timeline cursor directly with
Plotly, so playback stays smooth without a server round trip per frame.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, clientside_callback, dcc, html

from swingvision.app import state
from swingvision.app.components.court_overlay import overlay_svg
from swingvision.app.components.ui import (
    fmt_duration,
    icon,
    no_output_root_alert,
    page_header,
    status_badge,
)
from swingvision.court import calibration as calib
from swingvision.pipeline.runner import plan
from swingvision.pipeline.stages import default_registry
from swingvision.storage import tables
from swingvision.storage.schemas import (
    AUDIO_ONSETS,
    PRACTICE_SUBMODE_LABELS,
    Calibration,
    SessionConfig,
)

CURSOR_COLOR = "#e8590c"
OVERLAY_STYLE = {
    "position": "absolute",
    "inset": 0,
    "width": "100%",
    "height": "100%",
    "pointerEvents": "none",
}


def _tick_labels(duration: float) -> tuple[list[float], list[str]]:
    step = next(
        (s for s in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800) if duration / s <= 12), 3600
    )
    vals = list(np.arange(0, duration + 1e-6, step))
    return vals, [fmt_duration(v) for v in vals]


def timeline_figure(config: SessionConfig, onsets_t: np.ndarray, onsets_s: np.ndarray) -> go.Figure:
    duration = config.video.duration_s if config.video else 1.0
    fig = go.Figure()
    # Invisible click-catcher so a click anywhere on the strip seeks.
    grid = np.arange(0, duration, 0.25)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=np.zeros_like(grid),
            mode="markers",
            marker={"opacity": 0, "size": 12},
            hoverinfo="skip",
            showlegend=False,
            name="seek",
        )
    )
    if len(onsets_t):
        fig.add_trace(
            go.Scatter(
                x=onsets_t,
                y=np.minimum(onsets_s, 40),
                mode="markers",
                name="Audio onsets",
                marker={
                    "size": 3 + np.clip(onsets_s, 0, 40) / 10,
                    "color": "#4c8bf5",
                    "opacity": 0.45,
                    "line": {"width": 0},
                },
                hovertemplate="%{x:.2f}s · strength %{customdata:.1f}<extra></extra>",
                customdata=onsets_s,
            )
        )
    ticks, labels = _tick_labels(duration)
    fig.update_layout(
        height=150,
        margin={"l": 40, "r": 10, "t": 10, "b": 30},
        xaxis={
            "range": [0, duration],
            "tickvals": ticks,
            "ticktext": labels,
            "showgrid": False,
            "fixedrange": False,
        },
        yaxis={"title": "onset", "fixedrange": True, "showgrid": False, "zeroline": False},
        hovermode="closest",
        dragmode="zoom",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": 0,
                "x1": 0,
                "y0": 0,
                "y1": 1,
                "line": {"color": CURSOR_COLOR, "width": 2},
            }
        ],
    )
    return fig


def _stage_status(session, config):
    s = state.settings()
    try:
        planned = plan(default_registry(), session, config, s)
    except Exception as exc:
        return dmc.Text(f"Cannot plan stages: {exc}", c="red", size="sm")
    rows = []
    for p in planned:
        status = "done" if p.fresh else "pending"
        rows.append(
            dmc.Group(
                [
                    dmc.Text(p.stage.title or p.stage.name, size="sm", w=130),
                    status_badge(status, "xs"),
                    dmc.Text(p.reason, size="xs", c="dimmed"),
                ],
                gap="xs",
            )
        )
    return dmc.Stack(rows, gap=4)


def _calibration(session) -> tuple[Calibration | None, str]:
    """The best calibration available and how it was accepted."""
    for path, label in (
        (session.calibration_path, None),
        (session.court_user_path, "Confirmed by you"),
        (session.court_auto_path, "Auto-detected, not reviewed"),
    ):
        cal = calib.load(path)
        if cal is not None:
            if label is None:
                label = "Confirmed by you" if cal.confirmed_by == "user" else "Auto-accepted"
            return cal, label
    return None, "Not calibrated yet"


def _overlays(cal: Calibration) -> dict:
    """Court-overlay SVGs: the session camera plus windows where the view had shifted."""
    windows = []
    for w in cal.drift:
        if w.camera is not None and (w.shift_rms_px or 0) > calib.PIECEWISE_MIN_SHIFT_PX:
            windows.append(
                {"t0": w.t0_s, "t1": w.t1_s, "src": overlay_svg(calib.to_camera(w.camera))}
            )
    return {"main": overlay_svg(calib.to_camera(cal.camera)), "windows": windows}


def _calibration_card(session_id: str, cal: Calibration | None, label: str):
    color = "gray" if cal is None else ("grape" if "not reviewed" in label else "green")
    rows = [
        dmc.Group(
            [dmc.Title("Calibration", order=5), dmc.Badge(label, variant="light", color=color)],
            justify="space-between",
        )
    ]
    if cal is not None:
        d = cal.camera_summary
        rms = cal.metrics.rms_line_px
        rows.append(
            dmc.Text(
                (f"Line RMS {rms:.2f} px · " if rms is not None else "")
                + f"camera {d.get('height_m', 0):.1f} m high, "
                f"{d.get('behind_baseline_m', 0):.1f} m behind the baseline",
                size="xs",
                c="dimmed",
            )
        )
        moved = [w for w in cal.drift if w.status == "moved"]
        if moved:
            rows.append(
                dmc.Text(
                    f"The camera moved in {len(moved)} of {len(cal.drift)} time windows; "
                    "those use their own camera.",
                    size="xs",
                    c="orange",
                )
            )
    rows.append(
        dmc.Anchor(
            dmc.Button(
                "Open calibration editor",
                size="xs",
                variant="light",
                leftSection=icon("tabler:target", 14),
                mt=4,
            ),
            href=f"/calibrate/{session_id}",
        )
    )
    return dmc.Paper(dmc.Stack(rows, gap=6), p="md", withBorder=True)


def layout(session_id: str | None = None, **_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Session"), no_output_root_alert()], size="xl", px=0)
    found = state.session_for(session_id or "")
    if found is None:
        return dmc.Container(
            [page_header("Session not found"), dmc.Anchor("Back to library", href="/")],
            size="xl",
            px=0,
        )
    _lib, row, session = found
    config = session.load_config()
    video = config.video
    fps = video.fps_avg if video else 60.0
    subtitle = config.mode.capitalize()
    if config.practice:
        subtitle += f" · {PRACTICE_SUBMODE_LABELS[config.practice.submode]}"
    if video:
        subtitle += (
            f" · {fmt_duration(video.duration_s)} · {video.display_width}×"
            f"{video.display_height} @ {video.fps_avg:.2f} fps"
        )

    if session.audio_onsets_path.exists():
        onsets = tables.read_table(session.audio_onsets_path)
    else:
        onsets = AUDIO_ONSETS.empty_table()
    onsets_t = onsets.column("t_s").to_numpy()
    onsets_s = onsets.column("strength").to_numpy()

    cal, cal_label = _calibration(session)
    has_proxy = session.proxy_path.exists()
    overlays = _overlays(cal) if cal is not None and has_proxy else None

    if has_proxy:
        player = html.Video(
            id="review-video",
            src=f"/media/{config.id}/proxy.mp4",
            controls=True,
            preload="auto",
            style={"width": "100%", "display": "block", "background": "#000", "borderRadius": 8},
            **{
                "data-sv-player": "1",
                "data-fps": f"{fps}",
                "data-time-store": "review-time",
                "data-timeline": "review-timeline",
            },
        )
    else:
        player = dmc.Alert(
            "The playback proxy isn't ready yet. Check the Jobs page.",
            title="No proxy",
            color="blue",
            icon=icon("tabler:hourglass"),
        )

    # The court overlay sits on top of the video, in the video's own (letterbox-free) box.
    player = html.Div(
        [
            player,
            html.Img(
                id="review-overlay",
                src=overlays["main"] if overlays else "",
                style={**OVERLAY_STYLE, "display": "block" if overlays else "none"},
            ),
        ],
        style={"position": "relative"},
    )

    header_right = dmc.Group(
        [
            dmc.Anchor(
                dmc.Button(
                    "Calibrate", variant="default", size="sm", leftSection=icon("tabler:target", 16)
                ),
                href=f"/calibrate/{config.id}",
            ),
            status_badge(row["status"], "lg"),
        ],
        gap="sm",
    )
    return dmc.Container(
        [
            page_header(config.name, subtitle, right=header_right),
            dcc.Store(id="review-overlays", data=overlays),
            dcc.Store(id="review-time"),
            dcc.Store(id="review-fps", data=fps),
            dcc.Store(id="review-seek"),
            dmc.Grid(
                [
                    dmc.GridCol(
                        dmc.Stack(
                            [
                                player,
                                dmc.Group(
                                    [
                                        dmc.Group(
                                            [
                                                dmc.Text(
                                                    id="review-readout", ff="monospace", size="sm"
                                                ),
                                                dmc.Switch(
                                                    id="review-overlay-on",
                                                    label="Court overlay",
                                                    size="xs",
                                                    checked=overlays is not None,
                                                    disabled=overlays is None,
                                                ),
                                            ],
                                            gap="md",
                                        ),
                                        dmc.Text(
                                            "Space play/pause · J/L ±5 s · ←/→ frame · "
                                            "Shift+←/→ 1 s",
                                            size="xs",
                                            c="dimmed",
                                        ),
                                    ],
                                    justify="space-between",
                                ),
                                dmc.Paper(
                                    dcc.Graph(
                                        id="review-timeline",
                                        figure=timeline_figure(config, onsets_t, onsets_s),
                                        config={"displayModeBar": False, "scrollZoom": True},
                                    ),
                                    withBorder=True,
                                    p=4,
                                ),
                                dmc.Text(
                                    f"{len(onsets_t)} audio onsets. Click the timeline to "
                                    "seek; drag to zoom, double-click to reset.",
                                    size="xs",
                                    c="dimmed",
                                ),
                            ],
                            gap="xs",
                        ),
                        span={"base": 12, "lg": 8},
                    ),
                    dmc.GridCol(
                        dmc.Stack(
                            [
                                dmc.Paper(
                                    [
                                        dmc.Title("Segments", order=5, mb="xs"),
                                        dmc.Text(
                                            "Shot segmentation arrives in M5.",
                                            size="sm",
                                            c="dimmed",
                                        ),
                                    ],
                                    p="md",
                                    withBorder=True,
                                ),
                                _calibration_card(config.id, cal, cal_label),
                                dmc.Paper(
                                    [
                                        dmc.Title("Pipeline", order=5, mb="xs"),
                                        _stage_status(session, config),
                                    ],
                                    p="md",
                                    withBorder=True,
                                ),
                                dmc.Paper(
                                    [
                                        dmc.Title("Source", order=5, mb="xs"),
                                        dmc.Text(
                                            config.source.path,
                                            size="xs",
                                            c="dimmed",
                                            style={"wordBreak": "break-all"},
                                        ),
                                    ],
                                    p="md",
                                    withBorder=True,
                                ),
                            ],
                            gap="sm",
                        ),
                        span={"base": 12, "lg": 4},
                    ),
                ],
                gutter="md",
            ),
        ],
        size="xl",
        px=0,
    )


dash.register_page(
    __name__,
    path_template="/session/<session_id>",
    title="Session · Swing Vision Open",
    layout=layout,
)


clientside_callback(
    """
    function(t, fps) {
        if (!t) { return "00:00.000"; }
        const s = t.t || 0;
        const m = Math.floor(s / 60);
        const sec = (s - m * 60).toFixed(3).padStart(6, "0");
        const frame = Math.round(s * (fps || 60));
        return `${String(m).padStart(2, "0")}:${sec} · frame ≈${frame}` +
               (t.paused ? " · paused" : "");
    }
    """,
    Output("review-readout", "children"),
    Input("review-time", "data"),
    State("review-fps", "data"),
)

clientside_callback(
    """
    function(t, on, overlays) {
        const style = {position: "absolute", inset: 0, width: "100%", height: "100%",
                       pointerEvents: "none", display: "none"};
        if (!overlays || !on) { return [window.dash_clientside.no_update, style]; }
        const now = (t && t.t) || 0;
        let src = overlays.main;
        for (const w of overlays.windows || []) {
            if (now >= w.t0 && now < w.t1) { src = w.src; break; }
        }
        return [src, Object.assign({}, style, {display: "block"})];
    }
    """,
    Output("review-overlay", "src"),
    Output("review-overlay", "style"),
    Input("review-time", "data"),
    Input("review-overlay-on", "checked"),
    State("review-overlays", "data"),
)

clientside_callback(
    """
    function(click) {
        if (!click || !click.points || !click.points.length) {
            return window.dash_clientside.no_update;
        }
        const v = document.getElementById("review-video");
        if (v) { v.currentTime = click.points[0].x; }
        return click.points[0].x;
    }
    """,
    Output("review-seek", "data"),
    Input("review-timeline", "clickData"),
)
