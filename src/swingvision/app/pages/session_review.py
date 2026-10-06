"""Session review: proxy video + timeline synced to playback, player tracking, movement.

``assets/video_sync.js`` publishes the video's current time into the
``review-time`` store (~10 Hz) and moves the timeline cursor directly with
Plotly, so playback stays smooth without a server round trip per frame. The
player box on the video and the court minimap follow the same store client-side,
from the tracked positions in ``review-track``.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    dcc,
    html,
    no_update,
)
from plotly.subplots import make_subplots

from swingvision import services
from swingvision.app import state
from swingvision.app.components import practice_view as pv
from swingvision.app.components import swings_view as sw_view
from swingvision.app.components.ball_view import (
    BOUNCE_COLOR,
    HIT_COLOR,
    NET_COLOR,
    ball_card,
    ball_store,
    events_store,
    load_ball,
)
from swingvision.app.components.court_diagram import ME_COLOR, heatmap_figure, minimap_figure
from swingvision.app.components.court_overlay import overlay_svg
from swingvision.app.components.players_view import (
    load_movement,
    movement_card,
    player_card,
    profile_facts,
    speed_series,
    track_store,
)
from swingvision.app.components.shots_view import (
    LINGER_S,
    OUTCOME_COLORS,
    PATH_COLOR,
    landing_traces,
    load_shots,
    over_net,
    shot_detail,
    shots_card,
    shots_store,
    side_view_figure,
)
from swingvision.app.components.ui import (
    fmt_duration,
    icon,
    no_output_root_alert,
    notification,
    page_header,
    status_badge,
)
from swingvision.app.worker_control import ensure_worker
from swingvision.court import calibration as calib
from swingvision.pipeline.runner import plan
from swingvision.pipeline.stages import default_registry
from swingvision.pipeline.stages.players import load_players_summary
from swingvision.players.movement import heatmap
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


def timeline_figure(
    config: SessionConfig,
    onsets_t: np.ndarray,
    onsets_s: np.ndarray,
    speed_t: np.ndarray | None = None,
    speed_v: np.ndarray | None = None,
    events: dict | None = None,
    shots: dict | None = None,
    segments: list[tuple[float, float, str]] | None = None,
) -> go.Figure:
    """Audio onsets and ball events on top; once tracked, the player's speed and the shots'
    speeds in their own rows below. ``segments``: (start, end, color) bands (practice
    shots).

    The rows share the time axis (zooming one zooms all); the cursor spans all.
    """
    duration = config.video.duration_s if config.video else 1.0
    has_speed = speed_t is not None and len(speed_t) > 0
    has_shots = bool(shots and any(v is not None for v in shots["v"]))
    heights = [0.5] + ([0.25] if has_speed else []) + ([0.25] if has_shots else [])
    rows = len(heights)
    speed_row = 2 if has_speed else None
    shots_row = rows if has_shots else None
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[h / sum(heights) for h in heights],
    )
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
        ),
        row=1,
        col=1,
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
            ),
            row=1,
            col=1,
        )
    for kind, color, y, symbol in (
        ("hit", HIT_COLOR, 44, "triangle-down"),
        ("bounce", BOUNCE_COLOR, 44, "circle"),
        ("net", NET_COLOR, 44, "x"),
    ):
        if not events:
            break
        et = [t for t, k in zip(events["t"], events["kind"], strict=True) if k == kind]
        if et:
            fig.add_trace(
                go.Scatter(
                    x=et,
                    y=[y] * len(et),
                    mode="markers",
                    name=kind,
                    marker={"color": color, "size": 8, "symbol": symbol},
                    hovertemplate="%{x:.2f}s<extra>" + kind + "</extra>",
                ),
                row=1,
                col=1,
            )
    if has_speed:
        fig.add_trace(
            go.Scatter(
                x=speed_t,
                y=speed_v * 3.6,
                mode="lines",
                name="Player speed",
                line={"color": ME_COLOR, "width": 1.5},
                connectgaps=False,
                hovertemplate="%{x:.1f}s · %{y:.1f} km/h<extra>player speed</extra>",
            ),
            row=speed_row,
            col=1,
        )
    if has_shots:
        assert shots is not None
        for outcome, color in OUTCOME_COLORS.items():
            k = [i for i, o in enumerate(shots["o"]) if o == outcome and shots["v"][i] is not None]
            if not k:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[shots["t0"][i] for i in k],
                    y=[shots["v"][i] for i in k],
                    customdata=[shots["e"][i] or 0 for i in k],
                    mode="markers",
                    name="shot " + outcome,
                    marker={"color": color, "size": 6},
                    hovertemplate="%{x:.1f}s · %{y:.0f} ± %{customdata:.0f} km/h"
                    + "<extra>"
                    + outcome
                    + " (uncalibrated)</extra>",
                ),
                row=shots_row,
                col=1,
            )
    ticks, labels = _tick_labels(duration)
    axis = {"range": [0, duration], "showgrid": False, "fixedrange": False}
    fig.update_xaxes(**axis)
    fig.update_xaxes(tickvals=ticks, ticktext=labels, row=rows, col=1)
    for r in range(1, rows):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    fig.update_yaxes(fixedrange=True, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="onset", row=1, col=1)
    if has_speed:
        fig.update_yaxes(title_text="km/h", rangemode="tozero", row=speed_row, col=1)
    if has_shots:
        fig.update_yaxes(title_text="shot km/h", rangemode="tozero", row=shots_row, col=1)
    fig.update_layout(
        height=150 + 70 * (rows - 1),
        margin={"l": 44, "r": 10, "t": 10, "b": 30},
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
            },
            # Practice shots as bands behind everything (the cursor stays shapes[0]).
            *(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": a,
                    "x1": b,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": color,
                    "opacity": 0.13,
                    "line": {"width": 0},
                    "layer": "below",
                }
                for a, b, color in (segments or [])
            ),
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


def _minimap_card(track: dict | None, machine, is_machine: bool, extra_traces=None):
    if is_machine:
        if machine is None:
            machine_text = "Ball machine not found yet. Turn on placing and click its spot."
        else:
            how = "placed by you" if machine.source == "user" else "found automatically"
            machine_text = f"Ball machine at ({machine.x:.1f}, {machine.y:.1f}) m, {how}."
    else:
        machine_text = ""
    return dmc.Paper(
        [
            dmc.Group(
                [
                    dmc.Title("Court", order=5),
                    dmc.Text(id="review-pos", size="xs", c="dimmed", ff="monospace"),
                ],
                justify="space-between",
            ),
            dcc.Graph(
                id="review-minimap",
                figure=minimap_figure(
                    (machine.x, machine.y) if machine else None,
                    clickable=is_machine,
                    extra_traces=extra_traces,
                ),
                config={"displayModeBar": False},
            ),
            dmc.Text(
                "Player tracking hasn't run yet." if track is None else "",
                size="xs",
                c="dimmed",
            ),
            dmc.Stack(
                [
                    dmc.Text(machine_text, id="review-machine-text", size="xs", c="dimmed"),
                    dmc.Switch(
                        id="review-machine-place",
                        label="Click the court to place the ball machine",
                        size="xs",
                    ),
                ],
                gap=4,
                style={} if is_machine else {"display": "none"},
            ),
        ],
        p="md",
        withBorder=True,
    )


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

    movement, frames = load_movement(session)
    track = (
        track_store(movement, video.display_width, video.display_height)
        if video and has_proxy
        else None
    )
    speed_t, speed_v = speed_series(movement)
    ball_track, ball_events = load_ball(session)
    ball = (
        ball_store(ball_track, video.display_width, video.display_height)
        if video and has_proxy
        else None
    )
    evs = events_store(ball_events)
    shots, flight_paths = load_shots(session)
    shot_data = (
        shots_store(shots, flight_paths, cal, video.display_width, video.display_height)
        if video
        else None
    )
    practice = pv.load(session) if config.practice else None
    seg_bands, seg_starts, seg_times = None, [], {}
    if practice is not None:
        seg_bands = [
            (practice.segments[r["segment_id"]]["start_t"],
             practice.segments[r["segment_id"]]["end_t"], pv.result_color(r))
            for r in practice.rows
            if r["segment_id"] in practice.segments
        ]  # fmt: skip
        seg_times = {str(k): v["start_t"] for k, v in practice.segments.items()}
        seg_starts = sorted(seg_times.values())
    summary = load_players_summary(session)
    machine = summary.machine if summary else None
    is_machine = bool(config.practice and config.practice.submode == "ball_machine")
    profiles = services.list_profiles(state.settings())

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
            html.Div(id="review-box", style={"display": "none"}),
            html.Img(id="review-ball", src="", style={"display": "none"}),
            html.Img(id="review-skel-img", src="", style={"display": "none"}),
            html.Img(id="review-shotpath", src="", style={"display": "none"}),
            html.Div(id="review-shot-label", style={"display": "none"}),
        ],
        style={"position": "relative"},
    )

    practice_button = (
        dmc.Anchor(
            dmc.Button(
                "Practice", variant="light", size="sm", leftSection=icon("tabler:target-arrow", 16)
            ),
            href=f"/practice/{config.id}",
        )
        if config.practice
        else None
    )
    swings_button = (
        dmc.Anchor(
            dmc.Button(
                "Swings", variant="light", size="sm", leftSection=icon("tabler:ball-tennis", 16)
            ),
            href=f"/swings/{config.id}",
        )
        if session.swings_path.exists()
        else None
    )
    stats_button = (
        dmc.Anchor(
            dmc.Button(
                "Stats", variant="light", size="sm", leftSection=icon("tabler:chart-bar", 16)
            ),
            href=f"/stats/{config.id}",
        )
        if session.shots_path.exists() or session.movement_path.exists()
        else None
    )
    has_pose = session.pose2d_path.exists() and video is not None and has_proxy
    header_right = dmc.Group(
        [
            practice_button,
            swings_button,
            stats_button,
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
            dcc.Store(id="review-track", data=track),
            dcc.Store(id="review-ball-store", data=ball),
            dcc.Store(id="review-events", data=evs),
            dcc.Store(id="review-shots", data=shot_data),
            dcc.Store(id="review-shot-id"),
            dcc.Store(id="review-session-id", data=config.id),
            dcc.Store(id="review-seg-starts", data=seg_starts),
            dcc.Store(id="review-seg-times", data=seg_times),
            dcc.Store(id="review-sink"),
            dcc.Store(id="review-skel-key"),
            dcc.Store(id="review-skel"),
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
                                                dmc.Switch(
                                                    id="review-box-on",
                                                    label="Player box",
                                                    size="xs",
                                                    checked=track is not None,
                                                    disabled=track is None,
                                                ),
                                                dmc.Switch(
                                                    id="review-ball-on",
                                                    label="Ball",
                                                    size="xs",
                                                    checked=ball is not None,
                                                    disabled=ball is None,
                                                ),
                                                dmc.Switch(
                                                    id="review-skel-on",
                                                    label="Skeleton",
                                                    size="xs",
                                                    checked=False,
                                                    disabled=not has_pose,
                                                ),
                                                dmc.Switch(
                                                    id="review-shot-on",
                                                    label="Shot path",
                                                    size="xs",
                                                    checked=shot_data is not None,
                                                    disabled=shot_data is None,
                                                ),
                                            ],
                                            gap="md",
                                        ),
                                        dmc.Text(
                                            "Space play/pause · J/L ±5 s · ←/→ frame · "
                                            "Shift+←/→ 1 s · N/P next/previous shot",
                                            size="xs",
                                            c="dimmed",
                                        ),
                                    ],
                                    justify="space-between",
                                ),
                                dmc.Paper(
                                    dcc.Graph(
                                        id="review-timeline",
                                        figure=timeline_figure(
                                            config,
                                            onsets_t,
                                            onsets_s,
                                            speed_t,
                                            speed_v,
                                            evs,
                                            shot_data,
                                            seg_bands,
                                        ),
                                        config={"displayModeBar": False, "scrollZoom": True},
                                    ),
                                    withBorder=True,
                                    p=4,
                                ),
                                dmc.Text(
                                    f"{len(onsets_t)} audio onsets"
                                    + (" and player speed" if len(speed_t) else "")
                                    + ". Click the timeline to seek; drag to zoom, "
                                    "double-click to reset.",
                                    size="xs",
                                    c="dimmed",
                                ),
                                html.Div(
                                    movement_card(
                                        movement,
                                        frames,
                                        state.settings().processing.dark_luma,
                                        state.settings().processing.view_min,
                                    ),
                                    id="review-movement",
                                ),
                            ],
                            gap="xs",
                        ),
                        span={"base": 12, "lg": 8},
                    ),
                    dmc.GridCol(
                        dmc.Stack(
                            [
                                _minimap_card(track, machine, is_machine, landing_traces(shots)),
                                shots_card(shots),
                                ball_card(
                                    ball_track, ball_events, video.duration_s if video else 0.0, fps
                                ),
                                pv.segments_card(config.id, practice) if config.practice else None,
                                player_card(config, profiles),
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


clientside_callback(
    """
    function(t, boxOn, track) {
        const hidden = {display: "none"};
        const host = document.getElementById("review-minimap");
        const gd = host ? host.querySelector(".js-plotly-plot") : null;
        if (!track || !track.t.length) { return [hidden, ""]; }
        const now = (t && t.t) || 0;
        const ts = track.t;
        // Last sample at or before now.
        let lo = 0, hi = ts.length - 1, i = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (ts[mid] <= now + 1e-3) { i = mid; lo = mid + 1; } else { hi = mid - 1; }
        }
        const live = i >= 0 && now - ts[i] <= 0.15;
        const tx = [], ty = [];
        let px = [], py = [];
        if (live) {
            px = [track.x[i]]; py = [track.y[i]];
            for (let k = i; k >= 0 && ts[k] >= now - 3 && track.run[k] === track.run[i]; k--) {
                tx.push(track.x[k]); ty.push(track.y[k]);
            }
        }
        if (gd && window.Plotly && gd.data && gd.data.length >= 2) {
            window.Plotly.restyle(gd, {x: [tx, px], y: [ty, py]}, [0, 1]);
        }
        let label = "not tracked";
        if (live) {
            let v = 0;
            if (i > 0 && track.run[i - 1] === track.run[i]) {
                const dt = ts[i] - ts[i - 1];
                v = Math.hypot(track.x[i] - track.x[i - 1], track.y[i] - track.y[i - 1]) / dt;
            }
            label = "x " + track.x[i].toFixed(1) + " y " + track.y[i].toFixed(1) + " m · " +
                    (v * 3.6).toFixed(1) + " km/h" + (track.interp[i] ? " · bridged" : "");
        }
        if (!boxOn || !live) { return [hidden, label]; }
        const b = track.b[i];
        return [{
            position: "absolute", pointerEvents: "none", boxSizing: "border-box",
            left: b[0] + "%", top: b[1] + "%", width: (b[2] - b[0]) + "%",
            height: (b[3] - b[1]) + "%",
            border: "2px " + (track.interp[i] ? "dashed " : "solid ") + "#ffd43b",
            borderRadius: "3px",
        }, label];
    }
    """,
    Output("review-box", "style"),
    Output("review-pos", "children"),
    Input("review-time", "data"),
    Input("review-box-on", "checked"),
    State("review-track", "data"),
)


clientside_callback(
    """
    function(t, on, ball, events) {
        const hidden = {display: "none"};
        const host = document.getElementById("review-minimap");
        const gd = host ? host.querySelector(".js-plotly-plot") : null;
        const now = (t && t.t) || 0;
        if (events && gd && window.Plotly && gd.data && gd.data.length >= 3) {
            const bx = [], by = [];
            for (let k = 0; k < events.t.length; k++) {
                if (events.kind[k] === "bounce" && events.cx[k] !== null &&
                    events.t[k] <= now && events.t[k] >= now - 3) {
                    bx.push(events.cx[k]); by.push(events.cy[k]);
                }
            }
            window.Plotly.restyle(gd, {x: [bx], y: [by]}, [2]);
        }
        if (!on || !ball || !ball.t.length) { return ["", hidden]; }
        const ts = ball.t;
        let lo = 0, hi = ts.length - 1, i = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (ts[mid] <= now + 1e-3) { i = mid; lo = mid + 1; } else { hi = mid - 1; }
        }
        if (i < 0 || now - ts[i] > 0.1) { return ["", hidden]; }
        const pts = [];
        for (let k = i; k >= 0 && ts[k] >= now - 0.4; k--) {
            if (k < i && ts[k + 1] - ts[k] > 0.1) { break; }
            pts.push(ball.x[k] + "," + ball.y[k]);
        }
        const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" ' +
            'preserveAspectRatio="none"><polyline points="' + pts.join(" ") +
            '" fill="none" stroke="#d8f5a2" stroke-opacity="0.7" stroke-width="2" ' +
            'vector-effect="non-scaling-stroke"/><circle cx="' + ball.x[i] + '" cy="' +
            ball.y[i] + '" r="0.6" fill="none" stroke="#e8590c" stroke-width="2" ' +
            'vector-effect="non-scaling-stroke"/></svg>';
        return ["data:image/svg+xml;utf8," + encodeURIComponent(svg),
                {position: "absolute", inset: 0, width: "100%", height: "100%",
                 pointerEvents: "none", display: "block"}];
    }
    """,
    Output("review-ball", "src"),
    Output("review-ball", "style"),
    Input("review-time", "data"),
    Input("review-ball-on", "checked"),
    State("review-ball-store", "data"),
    State("review-events", "data"),
)


@callback(
    Output("review-heatmap", "figure"),
    Input("review-heat-fold", "checked"),
    State("review-session-id", "data"),
    prevent_initial_call=True,
)
def _fold_heatmap(fold, session_id):
    found = state.session_for(session_id or "")
    if found is None:
        return no_update
    movement, _ = load_movement(found[2])
    return heatmap_figure(*heatmap(movement, fold=bool(fold)), height=440)


@callback(
    Output("review-profile-facts", "children"),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("review-profile", "value"),
    State("review-session-id", "data"),
    prevent_initial_call=True,
)
def _set_profile(profile_id, session_id):
    s = state.settings()
    try:
        services.set_session_player(s, session_id, profile_id)
    except ValueError as exc:
        return no_update, notification(str(exc), color="red")
    p = services.get_profile(s, profile_id)
    if p is None:
        return "No profile assigned.", no_update
    return profile_facts(p), notification(f"Player set to {p.name}.", icon_name="tabler:check")


@callback(
    Output("review-machine-text", "children"),
    Output("review-machine-place", "checked"),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("review-minimap", "clickData"),
    State("review-machine-place", "checked"),
    State("review-session-id", "data"),
    prevent_initial_call=True,
)
def _place_machine(click, placing, session_id):
    if not placing or not click or not click.get("points"):
        return no_update, no_update, no_update
    pt = click["points"][0]
    x, y = round(float(pt["x"]), 2), round(float(pt["y"]), 2)
    s = state.settings()
    session = services.session_by_id(s, session_id)
    if session is None:
        return no_update, no_update, no_update
    config = session.load_config()
    if config.practice is None:
        return no_update, no_update, no_update
    config.practice.machine_xy = [x, y]
    session.save_config(config)
    try:
        job_id = services.enqueue(s, session_id, targets=["movement"])
        msg = f"Ball machine placed. Tracking reruns (job #{job_id}); reload when it's done."
        if state.OPTIONS.start_worker:
            ensure_worker(s.output_root)
    except ValueError as exc:
        msg = f"Ball machine placed. {exc} Process the session again afterwards."
    return f"Ball machine at ({x:.1f}, {y:.1f}) m, placed by you.", False, notification(msg)


clientside_callback(
    """
    function(t, on, shots, prevId) {
        const hidden = {display: "none"};
        const nu = window.dash_clientside.no_update;
        const now = (t && t.t) || 0;
        let i = -1;
        if (shots) {
            for (let k = shots.t0.length - 1; k >= 0; k--) {
                if (shots.t0[k] <= now + 0.05) {
                    if (now <= shots.t1[k] + LINGER) { i = k; }
                    break;
                }
            }
        }
        const host = document.getElementById("review-minimap");
        const gd = host ? host.querySelector(".js-plotly-plot") : null;
        const id = i >= 0 ? shots.id[i] : null;
        if (gd && window.Plotly && gd.data && gd.data.length >= 4 && id !== prevId) {
            const lx = [], ly = [], txt = [];
            if (i >= 0 && shots.cx[i] !== null) {
                lx.push(shots.cx[i]); ly.push(shots.cy[i]);
                txt.push(shots.v[i] !== null ? shots.v[i] + " km/h" : "");
            }
            window.Plotly.restyle(gd, {x: [lx], y: [ly], text: [txt]}, [3]);
        }
        const idOut = id === prevId ? nu : id;
        if (i < 0) { return ["", hidden, "", hidden, idOut]; }
        const label = (shots.v[i] !== null ? shots.v[i] +
            (shots.e[i] !== null ? " ± " + shots.e[i] : "") + " km/h" : "speed ?") +
            " · " + shots.o[i].replace("_", " ");
        const labelStyle = {position: "absolute", left: "1%", top: "1.5%", padding: "2px 8px",
            borderRadius: "4px", background: "rgba(0,0,0,0.6)", color: "#fff",
            font: "600 14px system-ui, sans-serif", pointerEvents: "none"};
        const path = shots.path[i];
        if (!on || !path.length) { return ["", hidden, label, labelStyle, idOut]; }
        const pts = path.map(p => p[0] + "," + p[1]).join(" ");
        const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" ' +
            'preserveAspectRatio="none"><polyline points="' + pts + '" fill="none" ' +
            'stroke="PATH" stroke-opacity="0.9" stroke-width="2" stroke-dasharray="6 3" ' +
            'vector-effect="non-scaling-stroke"/></svg>';
        return ["data:image/svg+xml;utf8," + encodeURIComponent(svg),
                {position: "absolute", inset: 0, width: "100%", height: "100%",
                 pointerEvents: "none", display: "block"}, label, labelStyle, idOut];
    }
    """.replace("LINGER", str(LINGER_S)).replace("PATH", PATH_COLOR),
    Output("review-shotpath", "src"),
    Output("review-shotpath", "style"),
    Output("review-shot-label", "children"),
    Output("review-shot-label", "style"),
    Output("review-shot-id", "data"),
    Input("review-time", "data"),
    Input("review-shot-on", "checked"),
    State("review-shots", "data"),
    State("review-shot-id", "data"),
)


clientside_callback(
    """
    function(clicks, shots) {
        const ctx = window.dash_clientside.callback_context;
        const nu = window.dash_clientside.no_update;
        if (!shots || !ctx.triggered.length || !ctx.triggered[0].value) { return nu; }
        const id = JSON.parse(ctx.triggered[0].prop_id.split(".")[0]).index;
        const k = shots.id.indexOf(id);
        const v = document.getElementById("review-video");
        if (k < 0 || !v) { return nu; }
        v.currentTime = Math.max(0, shots.t0[k] - 0.8);
        v.play();
        return shots.t0[k];
    }
    """,
    Output("review-seek", "data", allow_duplicate=True),
    Input({"type": "review-shot-row", "index": ALL}, "n_clicks"),
    State("review-shots", "data"),
    prevent_initial_call=True,
)


@callback(
    Output("review-shot-detail", "children"),
    Output("review-shot-side", "figure"),
    Input("review-shot-id", "data"),
    State("review-session-id", "data"),
    prevent_initial_call=True,
)
def _show_shot(shot_id, session_id):
    found = state.session_for(session_id or "")
    if found is None:
        return no_update, no_update
    shots, paths = load_shots(found[2])
    if shot_id is None or shots is None:
        return shot_detail(None), side_view_figure(None)
    rows = [r for r in shots.to_pylist() if r["shot_id"] == shot_id and over_net(r)]
    if not rows:
        return shot_detail(None), side_view_figure(None)
    r = rows[0]
    path = None
    if paths is not None and r["flight_id"] is not None and r["side"] is not None:
        fid = paths.column("flight_id").to_numpy()
        m = fid == r["flight_id"]
        if m.any():
            y = paths.column("y").to_numpy()[m].astype(np.float64)
            z = paths.column("z").to_numpy()[m].astype(np.float64)
            path = (y, z, r["side"])
    return shot_detail(r), side_view_figure(path)


clientside_callback(
    """
    function(starts) {
        const v = document.getElementById("review-video");
        if (v && starts) { v.dataset.segments = JSON.stringify(starts); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("review-sink", "data"),
    Input("review-seg-starts", "data"),
)


clientside_callback(
    """
    function(clicks, times) {
        const ctx = window.dash_clientside.callback_context;
        const nu = window.dash_clientside.no_update;
        if (!times || !ctx.triggered.length || !ctx.triggered[0].value) { return nu; }
        const id = JSON.parse(ctx.triggered[0].prop_id.split(".")[0]).index;
        const v = document.getElementById("review-video");
        const t = times[String(id)];
        if (!v || t === undefined) { return nu; }
        v.currentTime = Math.max(0, t);
        v.play();
        return t;
    }
    """,
    Output("review-seek", "data", allow_duplicate=True),
    Input({"type": "review-seg-row", "index": ALL}, "n_clicks"),
    State("review-seg-times", "data"),
    prevent_initial_call=True,
)


# Skeleton overlay: the browser asks for the pose of the 8 s block it's playing in (a block
# key changes only when playback leaves the block), the server sends that block's keypoints.
clientside_callback(
    """
    function(t, on) {
        if (!on) { return null; }
        const k = Math.floor(((t && t.t) || 0) / SKEL_BLOCK_S);
        return k;
    }
    """.replace("SKEL_BLOCK_S", "8"),
    Output("review-skel-key", "data"),
    Input("review-time", "data"),
    Input("review-skel-on", "checked"),
)


@callback(
    Output("review-skel", "data"),
    Input("review-skel-key", "data"),
    State("review-session-id", "data"),
    prevent_initial_call=True,
)
def _skeleton_block(key, session_id):
    if key is None:
        return None
    found = state.session_for(session_id or "")
    if found is None:
        return no_update
    session = found[2]
    video = session.load_config().video
    if video is None:
        return no_update
    t0 = key * 8.0
    return sw_view.skeleton_store(
        session, t0 - 0.5, t0 + 8.5, video.display_width, video.display_height
    )


clientside_callback(
    sw_view.SKELETON_JS,
    Output("review-skel-img", "src"),
    Output("review-skel-img", "style"),
    Input("review-time", "data"),
    Input("review-skel", "data"),
    Input("review-skel-on", "checked"),
)
