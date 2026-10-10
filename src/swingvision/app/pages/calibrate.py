"""Calibrate: review and adjust the court calibration of one session (PLAN.md §9.1), and
(the Speed tab, M7c) its net-tape reference serves and the device's speed calibration.

The background image (players removed) is shown with the projected court on top.
Every court keypoint and the three net points are draggable handles: a dragged
point is *pinned*, and the camera is re-solved so the court passes through the
pinned points. "Snap to lines" then fits the full camera model to the painted
lines for sub-pixel accuracy. "Confirm" saves ``court/user.json`` and queues
processing, which resumes past the calibration gate.
"""

from __future__ import annotations

import re
import threading
from datetime import UTC, datetime
from functools import lru_cache

import cv2
import dash
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.app import state, units
from swingvision.app.components import speed_view
from swingvision.app.components.court_overlay import COURT_COLOR, NET_COLOR, polylines, trace_xy
from swingvision.app.components.ui import (
    fmt_duration,
    icon,
    no_output_root_alert,
    notification,
    page_header,
    session_header,
)
from swingvision.app.worker_control import ensure_worker
from swingvision.court import calibration as calib
from swingvision.court import model
from swingvision.court.camera import Camera
from swingvision.court.detect import Prepared, detect_court
from swingvision.storage.schemas import Calibration, CalibrationMetrics, CameraParams

PINNED_COLOR = "#ffd43b"
HANDLE_COLOR = "#ffffff"
SHAPE_KEY = re.compile(r"^shapes\[(\d+)\]\.(xanchor|yanchor|x0|x1|y0|y1)$")
HANDLE_RADIUS_PX = 7  # screen pixels, independent of zoom
SHORT = {
    "far": "F",
    "near": "N",
    "doubles": "D",
    "singles": "S",
    "service": "Sv",
    "left": "L",
    "right": "R",
    "center": "C",
    "net": "Net",
    "post": "post",
}


def _short(name: str) -> str:
    return " ".join(SHORT.get(p, p) for p in name.split("_"))


# ---------------------------------------------------------------------------
# Server-side helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _prepared(path: str, mtime: float) -> Prepared | None:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return Prepared.from_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _background(session) -> Prepared | None:
    path = session.court_background_path
    if not path.exists():
        return None
    return _prepared(str(path), path.stat().st_mtime)


def _state_from(cal: Calibration, sid: str, origin: str) -> dict:
    return {
        "sid": sid,
        "camera": cal.camera.model_dump(),
        "user_points": dict(cal.user_points),
        "metrics": cal.metrics.model_dump(),
        "ok": cal.ok,
        "message": cal.message,
        "origin": origin,  # auto | user | edited
        "drift": [
            d.model_dump(exclude={"camera"}) | {"has_camera": d.camera is not None}
            for d in cal.drift
        ],
    }


def _camera(st: dict) -> Camera:
    return calib.to_camera(CameraParams(**st["camera"]))


def _with_camera(st: dict, cam: Camera, metrics: CalibrationMetrics | None, **upd) -> dict:
    new = dict(st)
    new["camera"] = cam.to_dict()
    if metrics is not None:
        new["metrics"] = metrics.model_dump()
    new.update(upd)
    return new


def _handles(st: dict) -> list[tuple[str, float, float, bool]]:
    """Draggable handles: (name, x, y, pinned). Order defines the figure's shape indices."""
    cam = _camera(st)
    kp = calib.projected_keypoints(cam)
    pins = st.get("user_points") or {}
    out = []
    for name in model.KEYPOINTS:
        pinned = name in pins and pins[name] is not None
        xy = pins[name] if pinned else kp.get(name)
        if xy is None:
            continue
        x, y = xy
        if (
            -0.25 * cam.width <= x <= 1.25 * cam.width
            and -0.25 * cam.height <= y <= 1.25 * cam.height
        ):
            out.append((name, float(x), float(y), pinned))
    return out


def _window_camera(sid: str, index: int) -> Camera | None:
    found = state.session_for(sid)
    if found is None:
        return None
    _, _, session = found
    auto = calib.load(session.court_auto_path)
    for w in auto.drift if auto else []:
        if w.index == index and w.camera is not None:
            return calib.to_camera(w.camera)
    return None


def build_figure(st: dict, view: str = "bg") -> go.Figure:
    cam = _camera(st)
    W, H = cam.width, cam.height
    sid = st["sid"]
    fig = go.Figure()
    window = None if view in (None, "bg") else int(view)
    src = (
        f"/media/{sid}/court_bg.jpg" if window is None else f"/media/{sid}/court_w{window:02d}.jpg"
    )
    fig.add_layout_image(
        source=src, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H,
        xanchor="left", yanchor="top", sizing="stretch", layer="below",
    )  # fmt: skip

    def add_lines(c: Camera, dash_style: str | None = None, opacity: float = 0.9, name="Court"):
        for color, lines, label in (
            (COURT_COLOR, model.COURT_LINES, name),
            (NET_COLOR, model.NET_LINES, f"{name} net"),
        ):
            xs, ys = trace_xy(polylines(c, lines))
            fig.add_trace(
                go.Scattergl(
                    x=xs, y=ys, mode="lines", name=label, hoverinfo="skip", opacity=opacity,
                    line={"color": color, "width": 1.5, "dash": dash_style},
                )
            )  # fmt: skip

    shapes = []
    if window is not None:
        wcam = _window_camera(sid, window)
        add_lines(cam, "dot", 0.6, "Session camera")
        if wcam is not None:
            add_lines(wcam, None, 0.95, "This window")
    else:
        add_lines(cam)
        handles = _handles(st)
        r = HANDLE_RADIUS_PX
        for _name, x, y, pinned in handles:
            # Pixel-sized circles anchored at the point stay grabbable at any zoom; a drag
            # reports the new anchor.
            shapes.append(
                {
                    "type": "circle", "xref": "x", "yref": "y",
                    "xsizemode": "pixel", "ysizemode": "pixel", "xanchor": x, "yanchor": y,
                    "x0": -r, "x1": r, "y0": -r, "y1": r,
                    "line": {"color": PINNED_COLOR if pinned else HANDLE_COLOR, "width": 2},
                    "fillcolor": "rgba(255,212,59,0.45)" if pinned else "rgba(0,0,0,0.25)",
                    "editable": True,
                }
            )  # fmt: skip
        if handles:
            fig.add_trace(
                go.Scatter(
                    x=[h[1] for h in handles], y=[h[2] for h in handles], mode="text",
                    text=[_short(h[0]) for h in handles], textposition="top right",
                    textfont={"color": "#ffffff", "size": 10}, hoverinfo="text",
                    hovertext=[model.KEYPOINT_LABELS[h[0]] for h in handles], showlegend=False,
                )
            )  # fmt: skip
    pad = 0.04 * W
    fig.update_layout(
        shapes=shapes,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        dragmode="pan",
        uirevision=f"{sid}-{view}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111",
        xaxis={"range": [-pad, W + pad], "visible": False, "constrain": "domain"},
        yaxis={"range": [H + pad, -pad], "visible": False, "scaleanchor": "x", "scaleratio": 1},
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _metric(label: str, value: str, tip: str | None = None):
    row = dmc.Group(
        [dmc.Text(label, size="sm", c="dimmed"), dmc.Text(value, size="sm", fw=600)],
        justify="space-between",
    )
    return dmc.Tooltip(row, label=tip, multiline=True, w=260) if tip else row


def _panel(st: dict):
    u = units.current()
    m = CalibrationMetrics(**st["metrics"])
    cam = _camera(st)
    d = cam.describe()
    pins = [n for n, v in (st.get("user_points") or {}).items() if v is not None]
    rms = f"{m.rms_line_px:.2f} px" if m.rms_line_px is not None else "–"
    color = "green" if (m.rms_line_px or 99) < 2 and m.coverage > 0.5 else "orange"
    origin = {"auto": "Auto-detected", "user": "Confirmed by you", "edited": "Edited, not saved"}[
        st["origin"]
    ]
    children = [
        dmc.Group(
            [
                dmc.Title("Fit", order=5),
                dmc.Badge(
                    origin, variant="light", color="blue" if st["origin"] == "edited" else "gray"
                ),
            ],
            justify="space-between",
        ),
        _metric(
            "Line RMS",
            rms,
            "Distance between the painted line centers and the projected court lines. Below 2 px is good.",
        ),
        dmc.Progress(value=min(100, 100 * m.coverage), color=color, size="sm"),
        _metric(
            "Lines found",
            f"{m.coverage:.0%} ({m.n_line_samples} samples)",
            "Share of the visible court lines whose centers were found near the projection.",
        ),
        _metric("Pinned points", str(len(pins))),
        dmc.Divider(my=4),
        dmc.Title("Camera", order=5),
        _metric("Height", u.len_str(d["height_m"], 2)),
        _metric("Behind baseline", u.len_str(d["behind_baseline_m"], 2)),
        _metric("Sideways offset", u.len_str(d["offset_x_m"], 2, sign=True)),
        _metric("Tilt down", f"{d['tilt_down_deg']:.1f}°"),
        _metric("Horizontal FOV", f"{d['hfov_deg']:.1f}°"),
        _metric("Lens k1 / k2", f"{d['k1']:+.3f} / {d['k2']:+.3f}"),
    ]
    if not st["ok"]:
        children.insert(
            0,
            dmc.Alert(
                st["message"]
                or "The court was not detected. Drag the points onto the court, then snap.",
                color="orange",
                p="xs",
            ),
        )
    return children


def _drift_panel(st: dict):
    rows = []
    for w in st.get("drift") or []:
        status = w["status"]
        color = {"ok": "green", "moved": "orange", "dark": "gray", "failed": "red"}[status]
        shift = f"{w['shift_rms_px']:.1f} px" if w.get("shift_rms_px") is not None else ""
        rows.append(
            dmc.Group(
                [
                    dmc.Text(
                        f"{fmt_duration(w['t0_s'])}–{fmt_duration(w['t1_s'])}", size="xs", w=90
                    ),
                    dmc.Badge(status, color=color, variant="light", size="xs"),
                    dmc.Text(shift, size="xs", c="dimmed"),
                ],
                gap="xs",
            )
        )
    if not rows:
        return dmc.Text("No drift check for this session.", size="xs", c="dimmed")
    moved = sum(1 for w in st["drift"] if w["status"] == "moved")
    note = (
        dmc.Text(
            f"The camera sits elsewhere in {moved} window(s). Those windows use their own "
            "pose-refined camera; view one below to check it.",
            size="xs",
            c="orange",
        )
        if moved
        else dmc.Text("The camera stays put for the whole video.", size="xs", c="dimmed")
    )
    return dmc.Stack([note, *rows], gap=4)


def _view_options(st: dict) -> list[dict]:
    opts = [{"value": "bg", "label": "Background (edit)"}]
    for w in st.get("drift") or []:
        if w.get("index") is not None and w["status"] != "dark":
            label = f"{fmt_duration(w['t0_s'])}–{fmt_duration(w['t1_s'])} · {w['status']}"
            opts.append({"value": str(w["index"]), "label": label})
    return opts


def _tabs(court, session, tab: str | None):
    """The Court and Speed tabs (``?tab=speed`` opens the second)."""
    speed = html.Div(speed_view.speed_panel(session, session.load_config()), id="spd-body")
    return dmc.Tabs(
        [
            dmc.TabsList(
                [
                    dmc.TabsTab("Court", value="court", leftSection=icon("tabler:grid-dots", 14)),
                    dmc.TabsTab("Speed", value="speed", leftSection=icon("tabler:gauge", 14)),
                ],
                mb="sm",
            ),
            dmc.TabsPanel(court, value="court"),
            dmc.TabsPanel(
                [
                    speed,
                    dcc.Store(id="spd-version", data=0),
                    dcc.Store(id="spd-sink"),
                    dcc.Store(id="spd-drag"),  # onset-line drags (assets/speed_wave.js)
                ],
                value="speed",
            ),
        ],
        value=tab if tab in ("court", "speed") else "court",
        id="cal-tabs",
    )


def layout(session_id: str | None = None, tab: str | None = None, **_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Calibrate"), no_output_root_alert()], size="xl", px=0)
    found = state.session_for(session_id or "")
    if found is None:
        return dmc.Container(
            [page_header("Session not found"), dmc.Anchor("Back to library", href="/")],
            size="xl",
            px=0,
        )
    _lib, row, session = found
    user = calib.load(session.court_user_path)
    auto = calib.load(session.court_auto_path)
    header = session_header(session, session.load_config(), "calibration", row["status"])
    if (user is None and auto is None) or not session.court_background_path.exists():
        return dmc.Container(
            [
                header,
                dmc.Alert(
                    [
                        dmc.Text(
                            "Court detection hasn't run for this session yet. It runs as part of processing."
                        ),
                        dmc.Button(
                            "Detect the court now",
                            id="cal-detect-job",
                            mt="sm",
                            size="xs",
                            leftSection=icon("tabler:player-play", 14),
                        ),
                    ],
                    color="blue",
                    icon=icon("tabler:info-circle"),
                ),
                dcc.Store(id="cal-sid", data=session_id),
            ],
            size="xl",
            px=0,
        )
    # Load the line response now, so the first drag doesn't wait for it.
    threading.Thread(target=_background, args=(session,), daemon=True).start()
    if user is not None:
        st = _state_from(user, session_id, "user")
        if auto is not None:
            st["drift"] = _state_from(auto, session_id, "auto")["drift"]
    else:
        st = _state_from(auto, session_id, "auto")
    view_opts = _view_options(st)
    graph = dcc.Graph(
        id="cal-graph",
        figure=build_figure(st),
        config={
            "edits": {"shapePosition": True},
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toImage"],
        },
        style={"height": "72vh"},
    )
    buttons = dmc.Stack(
        [
            dmc.Button(
                "Snap to lines",
                id="cal-snap",
                leftSection=icon("tabler:magnet"),
                variant="light",
                fullWidth=True,
            ),
            dmc.Group(
                [
                    dmc.Button("Clear pins", id="cal-clear", variant="default", size="xs", flex=1),
                    dmc.Button(
                        "Reset to auto",
                        id="cal-reset",
                        variant="default",
                        size="xs",
                        flex=1,
                        disabled=auto is None,
                    ),
                ],
                gap="xs",
                grow=True,
            ),
            dmc.Button(
                "Re-detect court",
                id="cal-redetect",
                variant="default",
                size="xs",
                fullWidth=True,
                leftSection=icon("tabler:scan", 14),
            ),
            dmc.Button(
                "Confirm and continue",
                id="cal-confirm",
                leftSection=icon("tabler:check"),
                fullWidth=True,
                mt="xs",
            ),
        ],
        gap="xs",
    )
    help_text = dmc.Text(
        "Drag a circle onto its court corner (or net post top / center strap); dragged points "
        "turn yellow and the court follows them. Then Snap to lines for a sub-pixel fit. "
        "Scroll to zoom, drag the background to pan, double-click to reset the view.",
        size="xs",
        c="dimmed",
    )
    court = [
        dmc.Text("Line up the court model with the painted lines.", c="dimmed", mb="md"),
        dcc.Store(id="cal-state", data=st),
        dmc.Grid(
            [
                dmc.GridCol(
                    dmc.Stack(
                        [
                            dmc.Paper(graph, withBorder=True, p=0, style={"overflow": "hidden"}),
                            dmc.Group(
                                [
                                    dmc.Select(
                                        id="cal-view",
                                        data=view_opts,
                                        value="bg",
                                        allowDeselect=False,
                                        size="xs",
                                        w=260,
                                        label="Image",
                                    ),
                                    help_text,
                                ],
                                align="flex-end",
                                wrap="nowrap",
                                gap="md",
                            ),
                        ],
                        gap="xs",
                    ),
                    span={"base": 12, "lg": 9},
                ),
                dmc.GridCol(
                    dmc.Stack(
                        [
                            dmc.Paper(buttons, p="md", withBorder=True),
                            dmc.Paper(
                                dmc.Stack(_panel(st), id="cal-panel", gap=6),
                                p="md",
                                withBorder=True,
                            ),
                            dmc.Paper(
                                [
                                    dmc.Title("Drift check", order=5, mb="xs"),
                                    html.Div(_drift_panel(st), id="cal-drift"),
                                ],
                                p="md",
                                withBorder=True,
                            ),
                        ],
                        gap="sm",
                    ),
                    span={"base": 12, "lg": 3},
                ),
            ],
            gutter="md",
        ),
    ]
    return dmc.Container(
        [header, dcc.Store(id="cal-sid", data=session_id), _tabs(court, session, tab)],
        fluid=True,
        px=0,
    )


dash.register_page(
    __name__,
    path_template="/calibrate/<session_id>",
    title="Calibrate · Swing Vision Open",
    layout=layout,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _session(sid: str):
    found = state.session_for(sid)
    return None if found is None else found[2]


def _evaluate(session, cam: Camera) -> CalibrationMetrics | None:
    prep = _background(session)
    return calib.evaluate(prep, cam) if prep is not None else None


@callback(
    Output("cal-graph", "figure"),
    Output("cal-panel", "children"),
    Input("cal-state", "data"),
    Input("cal-view", "value"),
)
def _render(st, view):
    if not st:
        return no_update, no_update
    return build_figure(st, view or "bg"), _panel(st)


def drag_update(st: dict, relayout: dict | None) -> dict | None:
    """Apply a handle drag (Plotly ``relayoutData``) to the editor state, or ``None``."""
    if not relayout:
        return None
    moved: dict[int, dict[str, float]] = {}
    for key, value in relayout.items():
        m = SHAPE_KEY.match(key)
        if m:
            moved.setdefault(int(m.group(1)), {})[m.group(2)] = float(value)
    if not moved:
        return None
    handles = _handles(st)
    pins = dict(st.get("user_points") or {})
    for i, c in moved.items():
        if i >= len(handles):
            continue
        name, x, y, _ = handles[i]
        if "xanchor" in c or "yanchor" in c:  # pixel-sized handle: the anchor moved
            pins[name] = [c.get("xanchor", x), c.get("yanchor", y)]
        elif len(c) == 4:  # data-sized shape: its box moved
            pins[name] = [(c["x0"] + c["x1"]) / 2, (c["y0"] + c["y1"]) / 2]
    fit = calib.solve_from_points(_camera(st), pins)
    return _with_camera(st, fit.camera, None, user_points=pins, origin="edited")


@callback(
    Output("cal-state", "data", allow_duplicate=True),
    Input("cal-graph", "relayoutData"),
    State("cal-state", "data"),
    prevent_initial_call=True,
)
def _drag(relayout, st):
    new = drag_update(st, relayout) if st else None
    if new is None:
        return no_update
    session = _session(st["sid"])
    metrics = _evaluate(session, _camera(new)) if session else None
    if metrics is not None:
        new["metrics"] = metrics.model_dump()
    return new


@callback(
    Output("cal-state", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("cal-snap", "n_clicks"),
    State("cal-state", "data"),
    running=[(Output("cal-snap", "loading"), True, False)],
    prevent_initial_call=True,
)
def _snap(n, st):
    if not n or not st:
        return no_update, no_update
    session = _session(st["sid"])
    prep = _background(session) if session else None
    if prep is None:
        return no_update, notification("The background image is missing.", color="red")
    det = calib.snap_to_lines(prep, _camera(st))
    if not det.ok or det.camera is None:
        return no_update, notification(
            f"Couldn't snap: {det.message}. Drag a few points closer to their corners first.",
            color="orange",
        )
    metrics = calib.evaluate(prep, det.camera)
    new = _with_camera(
        st, det.camera, metrics, user_points={}, origin="edited", ok=True, message=det.message
    )
    return new, notification(
        f"Snapped: line RMS {metrics.rms_line_px:.2f} px.", icon_name="tabler:magnet"
    )


@callback(
    Output("cal-state", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("cal-redetect", "n_clicks"),
    State("cal-state", "data"),
    running=[(Output("cal-redetect", "loading"), True, False)],
    prevent_initial_call=True,
)
def _redetect(n, st):
    if not n or not st:
        return no_update, no_update
    session = _session(st["sid"])
    prep = _background(session) if session else None
    if prep is None:
        return no_update, notification("The background image is missing.", color="red")
    det = detect_court(prep)
    if not det.ok or det.camera is None:
        return no_update, notification(f"Detection failed: {det.message}", color="orange")
    metrics = calib.evaluate(prep, det.camera)
    new = _with_camera(
        st, det.camera, metrics, user_points={}, origin="edited", ok=True, message=det.message
    )
    return new, notification(f"Court detected: line RMS {metrics.rms_line_px:.2f} px.")


@callback(
    Output("cal-state", "data", allow_duplicate=True),
    Input("cal-clear", "n_clicks"),
    Input("cal-reset", "n_clicks"),
    State("cal-state", "data"),
    prevent_initial_call=True,
)
def _clear_or_reset(_clear, _reset, st):
    if not st or not ctx.triggered[0]["value"]:
        return no_update
    if ctx.triggered_id == "cal-clear":
        return {**st, "user_points": {}}
    session = _session(st["sid"])
    auto = calib.load(session.court_auto_path) if session else None
    if auto is None:
        return no_update
    new = _state_from(auto, st["sid"], "auto")
    new["origin"] = "edited"
    return new


def confirmed_calibration(st: dict, based_on: str | None) -> Calibration:
    cam = _camera(st)
    pins = {k: v for k, v in (st.get("user_points") or {}).items() if v is not None}
    fit = calib.solve_from_points(cam, pins) if pins else None
    metrics = CalibrationMetrics(**st["metrics"])
    if fit is not None:
        metrics.rms_points_px = fit.rms_points_px
    return Calibration(
        source="user",
        created_at=datetime.now(UTC),
        ok=True,
        message="Confirmed in the calibration editor",
        camera=calib.to_params(cam),
        keypoints=calib.projected_keypoints(cam),
        user_points=pins,
        metrics=metrics,
        camera_summary=cam.describe(),
        based_on=based_on,
    )


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("cal-state", "data", allow_duplicate=True),
    Input("cal-confirm", "n_clicks"),
    State("cal-state", "data"),
    prevent_initial_call=True,
)
def _confirm(n, st):
    if not n or not st:
        return no_update, no_update
    s = state.settings()
    session = _session(st["sid"])
    if session is None:
        return notification("Session not found.", color="red"), no_update
    auto = calib.load(session.court_auto_path)
    cal = confirmed_calibration(st, auto.created_at.isoformat() if auto else None)
    calib.save(session.court_user_path, cal)
    try:
        job_id = services.enqueue(s, st["sid"])
        msg = f"Calibration saved. Processing continues as job #{job_id}."
    except ValueError:
        msg = (
            "Calibration saved. A job is already running for this session; it uses this "
            "calibration if it hasn't passed the calibration step yet (otherwise process "
            "the session again)."
        )
    if state.OPTIONS.start_worker:
        ensure_worker(s.output_root)
    return notification(msg, "Confirmed", icon_name="tabler:check"), {**st, "origin": "user"}


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("cal-detect-job", "n_clicks"),
    State("cal-sid", "data"),
    prevent_initial_call=True,
)
def _detect_job(n, sid):
    if not n:
        return no_update
    s = state.settings()
    try:
        job_id = services.enqueue(s, sid, targets=["court_auto"])
    except ValueError as exc:
        return notification(str(exc), color="red")
    if state.OPTIONS.start_worker:
        ensure_worker(s.output_root)
    return notification(f"Job #{job_id} queued. Reload this page when it finishes.")
