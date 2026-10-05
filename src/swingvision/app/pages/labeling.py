"""Labeling page: ball positions and events on short clips (PLAN.md §10).

A clip is a contiguous frame range of a session (3 s by default). Creating one decodes its
frames into a JPEG cache and runs the ball detector + tracker over it; those predictions
pre-fill the labels (*assisted*), so labeling is mostly checking and correcting:

* click the ball (snapped to the nearest moving blob unless *Snap* is off), or accept the
  prediction (Enter); mark frames without a ball in play (N) or with a hidden ball (O);
* label every few frames and *Interpolate* (I) fills the frames between them along a
  quadratic, never across an event;
* mark hits (H), bounces (B) and net contacts at their frame;
* *Done* puts the clip into the train or test split.

Ground-truth clips (``gt``) are labeled on every frame and scored by ``sv bench ball`` and
``sv eval``; sample clips feed training only.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, no_update

from swingvision.app import state
from swingvision.app.components.ui import (
    fmt_duration,
    icon,
    no_output_root_alert,
    notification,
    page_header,
)
from swingvision.training import clipcache as cc
from swingvision.training.assist import apply_predictions, load_predictions, predict_clip, snap
from swingvision.training.labels import BallLabel, EventLabel, LabelStore, interpolate

VIEWS = {"tight": 480, "near": 960, "wide": 1920, "full": 3840}
LABEL_COLOR = "#2fd16b"
PRED_COLOR = "#e64980"
TRAIL_COLOR = "#ffd43b"
EVENT_COLORS = {"hit": "#ff922b", "bounce": "#4dabf7", "net": "#cc5de8"}
SNAP_RADIUS = {"tight": 10, "near": 14, "wide": 24, "full": 40}


def _store() -> LabelStore | None:
    s = state.settings()
    return LabelStore(s.output_root) if s.output_root is not None else None


def _session_options() -> list[dict]:
    lib = state.library()
    if lib is None:
        return []
    return [
        {"value": r["id"], "label": f"{r['name']} ({r['id']})"}
        for r in lib.list_sessions()
        if r["status"] in ("ready", "needs_action", "processing", "failed")
    ]


def _clip_options(store: LabelStore, sid: str | None) -> list[dict]:
    if not sid:
        return []
    out = []
    for c in store.list_clips(sid):
        mark = "✓ " if c.status == "done" else ""
        out.append(
            {
                "value": c.clip_id,
                "label": f"{mark}{c.clip_id} · {fmt_duration(c.t0_s)} · {c.split or 'auto'} · "
                f"{c.n_labeled}/{len(c.frames)}",
            }
        )
    return out


def layout(**_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Labeling"), no_output_root_alert()], size="xl", px=0)
    sessions = _session_options()
    side = dmc.Stack(
        [
            dmc.Select(
                id="lab-session",
                label="Session",
                data=sessions,
                value=sessions[0]["value"] if sessions else None,
                searchable=True,
            ),
            dmc.Select(id="lab-clip", label="Clip", data=[], searchable=True),
            dmc.Divider(label="New clip", labelPosition="left"),
            dmc.Select(id="lab-suggest", label="Suggestions", data=[], clearable=True),
            dmc.Group(
                [
                    dmc.NumberInput(id="lab-new-t", label="Start (s)", value=0, min=0, w=110),
                    dmc.NumberInput(
                        id="lab-new-len", label="Length (s)", value=3, min=0.5, max=10, w=100
                    ),
                ],
                gap="xs",
            ),
            dmc.SegmentedControl(
                id="lab-new-kind",
                data=[
                    {"value": "gt", "label": "Ground truth"},
                    {"value": "sample", "label": "Training sample"},
                ],
                value="gt",
                size="xs",
                fullWidth=True,
            ),
            dmc.Button(
                "Create clip",
                id="lab-create",
                leftSection=icon("tabler:plus"),
                variant="light",
                fullWidth=True,
            ),
            dcc.Loading(dmc.Text(id="lab-create-status", size="xs"), type="dot"),
            dmc.Text(
                "Creating decodes the frames and runs the ball tracker (≈20-40 s).",
                size="xs",
                c="dimmed",
            ),
        ],
        gap="sm",
    )
    graph = dcc.Loading(
        dcc.Graph(
            id="lab-graph",
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"],
            },
            style={"height": "62vh"},
        ),
        delay_show=400,
    )

    def btn(id_, label, ic, **kw):
        return dmc.Button(
            label, id=id_, leftSection=icon(ic, 16), size="xs", variant="default", **kw
        )

    controls = dmc.Stack(
        [
            dmc.Group(
                [
                    btn("lab-prev5", "−5", "tabler:chevrons-left"),
                    btn("lab-prev", "Prev", "tabler:chevron-left"),
                    dmc.Text(id="lab-frame-text", fw=600, w=210, ta="center"),
                    btn("lab-next", "Next", "tabler:chevron-right"),
                    btn("lab-next5", "+5", "tabler:chevrons-right"),
                    dmc.SegmentedControl(
                        id="lab-view",
                        data=[{"value": k, "label": k} for k in VIEWS],
                        value="near",
                        size="xs",
                    ),
                    dmc.Switch(id="lab-snap", label="Snap", checked=True, size="sm"),
                ],
                gap="xs",
            ),
            dmc.Slider(id="lab-slider", min=0, max=1, step=1, value=0, size="sm"),
            dmc.Group(
                [
                    btn("lab-accept", "Accept prediction ⏎", "tabler:check"),
                    btn("lab-none", "No ball (N)", "tabler:circle-off"),
                    btn("lab-occ", "Hidden (O)", "tabler:eye-off"),
                    btn("lab-clear", "Clear (Del)", "tabler:eraser"),
                    btn("lab-interp", "Interpolate (I)", "tabler:route"),
                    btn("lab-fill", "Accept all predictions", "tabler:checks"),
                ],
                gap="xs",
            ),
            dmc.Group(
                [
                    btn("lab-hit", "Hit (H)", "tabler:ball-tennis"),
                    btn("lab-bounce", "Bounce (B)", "tabler:arrow-bounce"),
                    btn("lab-net", "Net", "tabler:grid-dots"),
                    dmc.Text(id="lab-events", size="sm", c="dimmed"),
                ],
                gap="xs",
            ),
            dmc.Group(
                [
                    dmc.SegmentedControl(
                        id="lab-split",
                        data=[
                            {"value": "train", "label": "Train"},
                            {"value": "test", "label": "Test"},
                        ],
                        value="train",
                        size="xs",
                    ),
                    dmc.Button(
                        "Done",
                        id="lab-done",
                        leftSection=icon("tabler:flag-check", 16),
                        size="xs",
                        color="teal",
                    ),
                    dcc.ConfirmDialogProvider(
                        dmc.Button(
                            "Delete clip",
                            id="lab-delete",
                            leftSection=icon("tabler:trash", 16),
                            size="xs",
                            color="red",
                            variant="subtle",
                        ),
                        id="lab-delete-confirm",
                        message="Delete this clip and all its labels? This can't be undone.",
                    ),
                    dmc.Text(id="lab-progress", size="sm"),
                ],
                gap="xs",
            ),
        ],
        gap="xs",
    )
    return dmc.Container(
        [
            page_header(
                "Labeling",
                "Label the ball on short clips. Predictions pre-fill the labels: check, correct, "
                "accept. Keys: ←/→ frame, Enter accept, N no ball, O hidden, I interpolate, "
                "H hit, B bounce.",
            ),
            dcc.Store(id="lab-frame", data=None),
            dcc.Store(id="lab-rev", data=0),
            dmc.Grid(
                [
                    dmc.GridCol(
                        dmc.Paper(side, p="md", withBorder=True), span={"base": 12, "md": 3}
                    ),
                    dmc.GridCol(
                        dmc.Paper(dmc.Stack([graph, controls], gap="xs"), p="sm", withBorder=True),
                        span={"base": 12, "md": 9},
                    ),
                ],
                gutter="md",
            ),
        ],
        fluid=True,
        px=0,
    )


dash.register_page(__name__, path="/labeling", title="Labeling · Swing Vision Open", layout=layout)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _center(clip, frame: int, preds: dict) -> tuple[float, float]:
    lab = clip.label(frame)
    if lab is not None and lab.vis == "visible":
        return lab.x, lab.y
    if frame in preds:
        return preds[frame][0], preds[frame][1]
    vis = [(f, clip.label(f)) for f in clip.frames if (lb := clip.label(f)) and lb.vis == "visible"]
    if vis:
        f, lab = min(vis, key=lambda v: abs(v[0] - frame))
        return lab.x, lab.y
    if preds:
        f = min(preds, key=lambda g: abs(g - frame))
        return preds[f][0], preds[f][1]
    return 1920.0, 1080.0


def build_figure(clip, frame: int, view: str, width: int = 3840, height: int = 2160) -> go.Figure:
    store = _store()
    preds = load_predictions(store, clip) if store else {}
    vw = VIEWS.get(view, 960)
    vh = round(vw * 9 / 16)
    cx, cy = _center(clip, frame, preds)
    x0 = int(np.clip(round(cx - vw / 2), 0, max(0, width - vw)))
    y0 = int(np.clip(round(cy - vh / 2), 0, max(0, height - vh)))
    fig = go.Figure()
    src = f"/labeling/frame/{clip.session_id}/{clip.clip_id}/{frame}.jpg?x0={x0}&y0={y0}&w={vw}&h={vh}"
    fig.add_layout_image(
        source=src, xref="x", yref="y", x=x0, y=y0, sizex=vw, sizey=vh,
        xanchor="left", yanchor="top", sizing="stretch", layer="below",
    )  # fmt: skip
    # Click catcher: a transparent heatmap over the view (clicks report its cell centers).
    step = max(1, vw // 480)
    xs = np.arange(x0, x0 + vw, step) + step / 2
    ys = np.arange(y0, y0 + vh, step) + step / 2
    fig.add_trace(
        go.Heatmap(
            x=xs, y=ys, z=np.zeros((len(ys), len(xs))), showscale=False, hoverinfo="none",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], name="click",
        )
    )  # fmt: skip
    # Trail of labels in the neighborhood of this frame.
    near = [f for f in range(frame - 15, frame + 16) if f != frame]
    tr = [(f, clip.label(f)) for f in near if (lb := clip.label(f)) and lb.vis == "visible"]
    if tr:
        fig.add_trace(
            go.Scatter(
                x=[lb.x for _, lb in tr], y=[lb.y for _, lb in tr], mode="markers",
                marker={"size": 4, "color": TRAIL_COLOR, "opacity": 0.7}, hoverinfo="text",
                hovertext=[str(f) for f, _ in tr], name="other frames",
            )
        )  # fmt: skip
    p = preds.get(frame)
    if p:
        fig.add_trace(
            go.Scatter(
                x=[p[0]], y=[p[1]], mode="markers", name="prediction", hoverinfo="name",
                marker={"size": 22, "color": "rgba(0,0,0,0)", "line": {"color": PRED_COLOR, "width": 2}},
            )
        )  # fmt: skip
    lab = clip.label(frame)
    if lab is not None and lab.vis == "visible":
        fig.add_trace(
            go.Scatter(
                x=[lab.x], y=[lab.y], mode="markers", name=f"label ({lab.src})", hoverinfo="name",
                marker={"size": 14, "color": "rgba(0,0,0,0)", "line": {"color": LABEL_COLOR, "width": 2}},
            )
        )  # fmt: skip
    for e in clip.events:
        if e.frame == frame and e.x is not None:
            fig.add_trace(
                go.Scatter(
                    x=[e.x], y=[e.y], mode="markers", name=e.kind, hoverinfo="name",
                    marker={"symbol": "x-thin", "size": 18, "line": {"width": 3, "color": EVENT_COLORS[e.kind]}},
                )
            )  # fmt: skip
    fig.update_xaxes(range=[x0, x0 + vw], visible=False, constrain="domain")
    fig.update_yaxes(range=[y0 + vh, y0], visible=False, scaleanchor="x", constrain="domain")
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        dragmode="pan",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111",
        uirevision=f"{clip.clip_id}-{view}-{x0}-{y0}",
    )
    return fig


def _frame_text(clip, frame: int) -> str:
    lab = clip.label(frame)
    status = (
        "unlabeled" if lab is None else (lab.vis if lab.vis != "visible" else f"ball · {lab.src}")
    )
    return f"{frame - clip.frame0 + 1}/{len(clip.frames)} · #{frame} · {status}"


def _progress(clip) -> str:
    counts = {"visible": 0, "none": 0, "occluded": 0}
    for f in clip.frames:
        lab = clip.label(f)
        if lab is not None:
            counts[lab.vis] += 1
    return (
        f"{clip.n_labeled}/{len(clip.frames)} labeled · {counts['visible']} ball · "
        f"{counts['none']} none · {counts['occluded']} hidden · {clip.status}"
    )


def _events_text(clip) -> str:
    if not clip.events:
        return "No events"
    return " · ".join(f"{e.kind} #{e.frame}" for e in clip.events)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("lab-clip", "data"),
    Output("lab-clip", "value"),
    Output("lab-suggest", "data"),
    Input("lab-session", "value"),
    Input("lab-rev", "data"),
    State("lab-clip", "value"),
)
def _clips(sid, _rev, current):
    store = _store()
    if store is None or not sid:
        return [], None, []
    opts = _clip_options(store, sid)
    values = [o["value"] for o in opts]
    value = current if current in values else (values[-1] if values else None)
    sugg = []
    found = state.session_for(sid)
    if found is not None and ctx.triggered_id == "lab-session":
        from swingvision.training.sampling import suggest

        _, _, session = found
        cfg = session.load_config()
        dur = cfg.video.duration_s if cfg.video else 0.0
        sugg = [
            {"value": f"{s.t_s:.2f}", "label": f"{fmt_duration(s.t_s)} · {s.reason}"}
            for s in suggest(session, dur, store.list_clips(sid))
        ]
        return opts, value, sugg
    return opts, value, no_update


@callback(Output("lab-new-t", "value"), Input("lab-suggest", "value"), prevent_initial_call=True)
def _use_suggestion(value):
    return float(value) if value else no_update


@callback(
    Output("lab-rev", "data", allow_duplicate=True),
    Output("lab-clip", "value", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("lab-create-status", "children"),
    Input("lab-create", "n_clicks"),
    State("lab-session", "value"),
    State("lab-new-t", "value"),
    State("lab-new-len", "value"),
    State("lab-new-kind", "value"),
    State("lab-rev", "data"),
    prevent_initial_call=True,
)
def _create(n_clicks, sid, t0, length, kind, rev):
    from swingvision.io.frames import frame_table

    if not n_clicks:  # Dash pages can fire this when the page mounts
        return no_update, no_update, no_update, no_update
    store = _store()
    found = state.session_for(sid or "")
    if store is None or found is None:
        return no_update, no_update, notification("Pick a session first", color="red"), ""
    _, _, session = found
    cfg = session.load_config()
    tab = frame_table(cfg.source.path)
    times = tab.times()
    f0 = tab.index_at(float(t0 or 0))
    f1 = min(len(times) - 1, tab.index_at(float(t0 or 0) + float(length or 3)) - 1)
    if f1 <= f0:
        return no_update, no_update, notification("That clip would be empty", color="red"), ""
    clip = store.new_clip(sid, f0, f1, float(times[f0]), float(times[f1]), kind=kind)
    settings = state.settings()
    cc.ensure_cached(store, clip, cfg, settings.processing.decode_backend)
    msg = f"Created {clip.clip_id} ({f1 - f0 + 1} frames)"
    try:
        preds = predict_clip(store, clip, session, cfg, settings)
        n = apply_predictions(clip, preds)
        store.save(clip)
        msg += f", {n} frames pre-filled"
    except Exception as exc:  # no calibration yet, …: label by hand
        msg += f" (no predictions: {exc})"
    return (rev or 0) + 1, clip.clip_id, notification(msg), msg


ACTIONS = (
    "lab-prev", "lab-next", "lab-prev5", "lab-next5", "lab-accept", "lab-none", "lab-occ",
    "lab-clear", "lab-interp", "lab-fill", "lab-hit", "lab-bounce", "lab-net", "lab-done",
)  # fmt: skip


@callback(
    Output("lab-graph", "figure"),
    Output("lab-frame", "data"),
    Output("lab-slider", "min"),
    Output("lab-slider", "max"),
    Output("lab-slider", "value"),
    Output("lab-frame-text", "children"),
    Output("lab-progress", "children"),
    Output("lab-events", "children"),
    Output("lab-split", "value"),
    Input("lab-clip", "value"),
    Input("lab-slider", "value"),
    Input("lab-view", "value"),
    Input("lab-graph", "clickData"),
    *[Input(a, "n_clicks") for a in ACTIONS],
    State("lab-session", "value"),
    State("lab-frame", "data"),
    State("lab-snap", "checked"),
    State("lab-split", "value"),
    prevent_initial_call=True,
)
def _edit(clip_id, slider, view, click, *rest):
    sid, frame, snap_on, split = rest[len(ACTIONS) :]
    store = _store()
    if store is None or not sid or not clip_id:
        return go.Figure(), None, 0, 1, 0, "", "", "", no_update
    clip = store.get(sid, clip_id)
    if clip is None:
        return go.Figure(), None, 0, 1, 0, "", "", "", no_update
    trig = ctx.triggered_id
    if trig == "lab-clip" or frame is None or not (clip.frame0 <= frame <= clip.frame1):
        frame = next((f for f in clip.frames if clip.label(f) is None), clip.frame0)
        split = clip.split or split
    if trig == "lab-slider" and slider is not None:
        frame = int(slider)
    step = {"lab-prev": -1, "lab-next": 1, "lab-prev5": -5, "lab-next5": 5}.get(trig)
    if step:
        frame = int(np.clip(frame + step, clip.frame0, clip.frame1))
    changed = False
    advance = False
    if trig == "lab-graph" and click and click.get("points"):
        pt = click["points"][0]
        x, y = float(pt["x"]), float(pt["y"])
        src = "click"
        if snap_on:
            imgs = [
                cc.read_frame(store, clip, g)
                for g in (frame - cc.CONTEXT, frame, frame + cc.CONTEXT)
            ]
            if all(i is not None for i in imgs):
                s = snap(*imgs, x, y, SNAP_RADIUS.get(view, 14))
                if s is not None:
                    x, y, src = s[0], s[1], "snapped"
        clip.set(frame, BallLabel(vis="visible", x=round(x, 2), y=round(y, 2), src=src))
        changed = advance = True
    elif trig == "lab-accept":
        p = load_predictions(store, clip).get(frame)
        lab = clip.label(frame)
        if p is not None:
            clip.set(frame, BallLabel(vis="visible", x=p[0], y=p[1], src="accepted"))
        elif lab is not None and lab.vis == "visible":
            lab.src = "accepted"
        changed = advance = True
    elif trig == "lab-none":
        clip.set(frame, BallLabel(vis="none", src="click"))
        changed = advance = True
    elif trig == "lab-occ":
        clip.set(frame, BallLabel(vis="occluded", src="click"))
        changed = advance = True
    elif trig == "lab-clear":
        clip.set(frame, None)
        changed = True
    elif trig == "lab-interp":
        interpolate(clip)
        changed = True
    elif trig == "lab-fill":
        apply_predictions(clip, load_predictions(store, clip))
        changed = True
    elif trig in ("lab-hit", "lab-bounce", "lab-net"):
        kind = trig.removeprefix("lab-")
        existing = [e for e in clip.events if e.frame == frame and e.kind == kind]
        if existing:
            clip.events = [e for e in clip.events if e not in existing]
        else:
            lab = clip.label(frame)
            xy = (lab.x, lab.y) if lab is not None and lab.vis == "visible" else (None, None)
            clip.events.append(EventLabel(frame=frame, kind=kind, x=xy[0], y=xy[1]))
            clip.events.sort(key=lambda e: e.frame)
        changed = True
    elif trig == "lab-done":
        clip.status = "in_progress" if clip.status == "done" else "done"
        clip.split = split
        changed = True
    if changed:
        store.save(clip)
    if advance:
        frame = min(clip.frame1, frame + 1)
    cfg_w, cfg_h = 3840, 2160
    found = state.session_for(sid)
    if found is not None:
        v = found[2].load_config().video
        if v is not None:
            cfg_w, cfg_h = v.display_width, v.display_height
    return (
        build_figure(clip, frame, view, cfg_w, cfg_h),
        frame,
        clip.frame0,
        clip.frame1,
        frame,
        _frame_text(clip, frame),
        _progress(clip),
        _events_text(clip),
        clip.split or split,
    )


@callback(
    Output("lab-rev", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("lab-delete-confirm", "submit_n_clicks"),
    State("lab-session", "value"),
    State("lab-clip", "value"),
    State("lab-rev", "data"),
    prevent_initial_call=True,
)
def _delete(n_clicks, sid, clip_id, rev):
    store = _store()
    if not n_clicks or store is None or not sid or not clip_id:
        return no_update, no_update
    store.delete(sid, clip_id)
    return (rev or 0) + 1, notification(f"Deleted {clip_id}", color="orange")
