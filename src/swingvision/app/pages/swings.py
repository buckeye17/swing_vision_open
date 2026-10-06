"""Swings page (M6): every swing of a session with its stroke, phases and kinematics, the 3D
skeleton, angle curves with the phases shaded, and a comparison with another swing or the
player's average for that stroke (PLAN.md §9.2).

Correcting a stroke writes ``edits.json`` and reruns the cheap stages (``swings`` →
``shots`` → ``segments`` → ``practice_eval``) in-process; the correction also becomes a
training label for ``sv train strokes``.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.app import state
from swingvision.app.components import swings_view as sv
from swingvision.app.components.ui import icon, no_output_root_alert, notification, page_header
from swingvision.pose.strokes import STROKES
from swingvision.storage.edits import EditConflict

OVERLAY_STYLE = {
    "position": "absolute",
    "inset": 0,
    "width": "100%",
    "height": "100%",
    "pointerEvents": "none",
    "display": "none",
}


def _missing(session_id: str):
    return dmc.Alert(
        [
            dmc.Text(
                "This session has no swings yet. They appear once processing gets through the "
                "2D pose, 3D pose and Swings stages."
            ),
            dmc.Anchor("Open the session", href=f"/session/{session_id}"),
        ],
        title="Not processed yet",
        color="blue",
        icon=icon("tabler:hourglass"),
    )


def _summary(data: sv.SwingData):
    s = data.summary
    hand = s.get("hand", {})
    src = {"pose": "from the pose", "profile": "from the profile", "default": "assumed"}.get(
        hand.get("source"), ""
    )
    strokes = sv.strokes_only(data.rows)
    counts = {}
    for r in strokes:
        counts[r["stroke_type"]] = counts.get(r["stroke_type"], 0) + 1
    badges = [
        dmc.Badge(
            f"{sv.stroke_label(k)} {v}",
            color=sv.STROKE_COLORS.get(k, "gray"),
            variant="light",
            styles={"root": {"color": sv.STROKE_COLORS.get(k, "gray")}},
        )
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    text = (
        f"{len(strokes)} strokes ({len(data.rows)} swings with the non-strokes: dribbles, "
        f"tosses, picking up balls) · racket hand {s.get('racket_hand', '?')} {src}"
    )
    if hand.get("mismatch"):
        text += " · differs from the profile"
    model = s.get("stroke_model")
    text += f" · strokes by {'model ' + model if model else 'rules'}"
    return dmc.Stack([dmc.Group(badges, gap=6), dmc.Text(text, size="xs", c="dimmed")], gap=4)


def layout(session_id: str | None = None, **_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Swings"), no_output_root_alert()], size="xl", px=0)
    found = state.session_for(session_id or "")
    if found is None:
        return dmc.Container(
            [page_header("Session not found"), dmc.Anchor("Back to library", href="/")],
            size="xl",
            px=0,
        )
    _lib, _row, session = found
    config = session.load_config()
    data = sv.load(session)
    links = [
        dmc.Anchor(
            dmc.Button(
                "Session", variant="default", size="sm", leftSection=icon("tabler:movie", 16)
            ),
            href=f"/session/{config.id}",
        )
    ]
    if config.practice is not None:
        links.append(
            dmc.Anchor(
                dmc.Button(
                    "Practice", variant="default", size="sm", leftSection=icon("tabler:target", 16)
                ),
                href=f"/practice/{config.id}",
            )
        )
    header = page_header(config.name, "Swings", right=dmc.Group(links, gap="sm"))
    if data is None:
        return dmc.Container([header, _missing(config.id)], size="xl", px=0)
    strokes = sv.strokes_only(data.rows)
    first = strokes[0]["swing_id"] if strokes else (data.rows[0]["swing_id"] if data.rows else None)
    present = sorted({r["stroke_type"] for r in strokes}, key=STROKES.index)
    if session.proxy_path.exists():
        player = html.Video(
            id="sw-video",
            src=f"/media/{config.id}/proxy.mp4",
            controls=True,
            preload="auto",
            style={"width": "100%", "display": "block", "background": "#000", "borderRadius": 8},
            **{"data-sv-player": "1", "data-fps": f"{data.fps}", "data-time-store": "sw-time"},
        )
    else:
        player = dmc.Alert("The playback proxy isn't ready yet.", color="blue")
    player = html.Div(
        [player, html.Img(id="sw-skel-img", src="", style=OVERLAY_STYLE)],
        style={"position": "relative"},
    )
    filters = dmc.Group(
        [
            dmc.MultiSelect(
                id="sw-strokes",
                data=[{"value": k, "label": sv.stroke_label(k)} for k in present],
                placeholder="All strokes",
                size="xs",
                w=220,
                clearable=True,
            ),
            dmc.SegmentedControl(
                id="sw-end",
                value="all",
                data=[
                    {"value": "all", "label": "Both ends"},
                    {"value": "near", "label": "Near"},
                    {"value": "far", "label": "Far"},
                ],
                size="xs",
            ),
            dmc.Switch(id="sw-other", label="Non-strokes", size="xs", checked=False),
        ],
        gap="sm",
    )
    left = dmc.Stack(
        [
            player,
            dmc.Group(
                [
                    dmc.Switch(id="sw-skel-on", label="Skeleton", size="xs", checked=True),
                    dmc.Button(
                        "Play swing",
                        id="sw-play",
                        size="xs",
                        variant="light",
                        leftSection=icon("tabler:player-play", 14),
                    ),
                    dmc.Text("Space play/pause · ←/→ one frame", size="xs", c="dimmed"),
                ],
                gap="sm",
            ),
            dmc.Paper(
                [
                    dmc.Group(
                        [
                            dmc.Text("Angles and speed around the contact", fw=600, size="sm"),
                            dmc.Select(
                                id="sw-compare",
                                placeholder="Compare with…",
                                size="xs",
                                w=240,
                                clearable=True,
                                data=[],
                            ),
                        ],
                        justify="space-between",
                    ),
                    dmc.Text(
                        "Shaded: preparation, forward swing, follow-through, recovery. "
                        "Dashed: the comparison, aligned at the contact.",
                        size="xs",
                        c="dimmed",
                    ),
                    dcc.Graph(id="sw-angles", config={"displayModeBar": False}),
                ],
                p="sm",
                withBorder=True,
            ),
        ],
        gap="sm",
    )
    right = dmc.Stack(
        [
            dmc.Paper([_summary(data)], p="sm", withBorder=True),
            dmc.Paper(
                [filters, html.Div(id="sw-list", style={"marginTop": 8})],
                p="sm",
                withBorder=True,
            ),
            dmc.Paper(
                [
                    dmc.Group(
                        [
                            html.Div(id="sw-title"),
                            dmc.Select(
                                id="sw-stroke-edit",
                                data=[{"value": k, "label": sv.stroke_label(k)} for k in STROKES],
                                size="xs",
                                w=170,
                                allowDeselect=False,
                                leftSection=icon("tabler:pencil", 14),
                            ),
                        ],
                        justify="space-between",
                    ),
                    dcc.Graph(id="sw-3d", config={"displayModeBar": False}),
                ],
                p="sm",
                withBorder=True,
            ),
            dmc.Paper(
                [dmc.Text("Metrics", fw=600, size="sm", mb=4), html.Div(id="sw-metrics")],
                p="sm",
                withBorder=True,
            ),
        ],
        gap="sm",
    )
    return dmc.Container(
        [
            header,
            dcc.Store(id="sw-session-id", data=config.id),
            dcc.Store(id="sw-selected", data=first),
            dcc.Store(id="sw-version", data=0),
            dcc.Store(id="sw-edits-version", data=0),
            dcc.Store(id="sw-time"),
            dcc.Store(id="sw-seek"),
            dcc.Store(id="sw-sink"),
            dcc.Store(id="sw-skel"),
            dmc.Grid(
                [
                    dmc.GridCol(left, span={"base": 12, "lg": 7}),
                    dmc.GridCol(right, span={"base": 12, "lg": 5}),
                ],
                gutter="md",
            ),
        ],
        size="xl",
        px=0,
    )


dash.register_page(
    __name__,
    path_template="/swings/<session_id>",
    title="Swings · Swing Vision Open",
    layout=layout,
)


def _session(session_id):
    found = state.session_for(session_id or "")
    return None if found is None else found[2]


@callback(
    Output("sw-list", "children"),
    Input("sw-strokes", "value"),
    Input("sw-end", "value"),
    Input("sw-other", "checked"),
    Input("sw-selected", "data"),
    Input("sw-version", "data"),
    State("sw-session-id", "data"),
)
def _list(strokes, end, other, selected, _version, session_id):
    session = _session(session_id)
    data = sv.load(session) if session else None
    if data is None:
        return no_update
    rows = sv.filtered(data.rows, strokes, end or "all", bool(other))
    if not rows:
        return dmc.Text("No swings match.", size="sm", c="dimmed")
    return sv.swing_table(rows, selected)


@callback(
    Output("sw-selected", "data", allow_duplicate=True),
    Input({"type": "sw-row", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _pick(clicks):
    if not isinstance(ctx.triggered_id, dict) or not any(clicks or []):
        return no_update
    return ctx.triggered_id["index"]


@callback(
    Output("sw-seek", "data"),
    Input("sw-selected", "data"),
    Input("sw-play", "n_clicks"),
    State("sw-session-id", "data"),
)
def _seek(swing_id, _play, session_id):
    """Cue the video to just before the swing (and play it for *Play swing*)."""
    session = _session(session_id)
    data = sv.load(session) if session else None
    row = data.by_id(swing_id) if data else None
    if row is None:
        return no_update
    t = (row["t_start"] or row["t_contact"] - 1.0) - 0.3
    return {"t": t, "play": ctx.triggered_id == "sw-play"}


@callback(
    Output("sw-title", "children"),
    Output("sw-stroke-edit", "value"),
    Output("sw-3d", "figure"),
    Output("sw-angles", "figure"),
    Output("sw-metrics", "children"),
    Output("sw-skel", "data"),
    Output("sw-compare", "data"),
    Input("sw-selected", "data"),
    Input("sw-compare", "value"),
    Input("sw-version", "data"),
    State("sw-session-id", "data"),
)
def _show(swing_id, compare, _version, session_id):
    session = _session(session_id)
    data = sv.load(session) if session else None
    row = data.by_id(swing_id) if data else None
    if data is None or row is None:
        return (
            dmc.Text("No swing selected", size="sm", c="dimmed"),
            None,
            sv.skeleton_3d([], [], None, "right"),
            sv.angle_figure(None, None),
            sv.metrics_table(None),
            None,
            [],
        )
    got = sv.swing_kinematics(session, row, data.hand)
    kin, joints = got if got else (None, None)
    stroke = row["stroke_type"] or "other"
    same = [
        r for r in sv.strokes_only(data.rows)
        if r["stroke_type"] == stroke and r["swing_id"] != row["swing_id"]
    ]  # fmt: skip
    options = []
    if same:
        options.append({"value": "avg", "label": f"My average {sv.stroke_label(stroke).lower()}"})
    options += [
        {
            "value": str(r["swing_id"]),
            "label": f"{sv.stroke_label(stroke)} at {sv.fmt_t(r['t_contact'])}",
        }
        for r in same
    ]
    cmp = None
    if compare == "avg" and same:
        same_end = [r for r in same if r["side"] == row["side"]] or same
        avg = sv.average_curves(session, same_end, data.hand)
        if avg is not None:
            cmp = {
                "label": f"average of {avg['n']}",
                "t": avg["t"],
                "curves": avg["curves"],
                "means": sv.mean_metrics(same_end),
            }
    elif compare and compare != "avg":
        other = data.by_id(int(compare))
        got2 = sv.swing_kinematics(session, other, data.hand) if other else None
        if got2 is not None:
            k2, _ = got2
            cmp = {
                "label": sv.fmt_t(other["t_contact"]),
                "t": k2.t - other["t_contact"],
                "curves": k2.curves(),
                "row": other,
            }
    t3 = kin.t if kin is not None else []
    title = dmc.Stack(
        [
            dmc.Text(
                f"{sv.stroke_label(stroke)} at {sv.fmt_t(row['t_contact'])} · "
                f"{ {-1: 'near', 1: 'far'}.get(row['side'], '?') } end",
                fw=600,
                size="sm",
            ),
            dmc.Text(
                f"contact from the {row['contact_source']}"
                f" · {row['stroke_source'] or 'no'} stroke"
                + (f" ({row['stroke_conf']:.0%})" if row["stroke_conf"] is not None else "")
                + (f" · {', '.join(row['flags'])}" if row["flags"] else ""),
                size="xs",
                c="dimmed",
            ),
        ],
        gap=0,
    )
    skel = sv.skeleton_store(
        session,
        row["t_contact"] - sv.WINDOW[0] - 0.5,
        row["t_contact"] + sv.WINDOW[1] + 0.5,
        data.width,
        data.height,
    )
    return (
        title,
        stroke,
        sv.skeleton_3d(t3, joints if joints is not None else [], row, data.hand),
        sv.angle_figure(kin, row, cmp),
        sv.metrics_table(row, cmp),
        skel,
        options,
    )


@callback(
    Output("sw-version", "data"),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("sw-stroke-edit", "value"),
    State("sw-selected", "data"),
    State("sw-session-id", "data"),
    State("sw-version", "data"),
    prevent_initial_call=True,
)
def _edit_stroke(stroke, swing_id, session_id, version):
    session = _session(session_id)
    data = sv.load(session) if session else None
    row = data.by_id(swing_id) if data else None
    if row is None or stroke is None or stroke == (row["stroke_type"] or "other"):
        return no_update, no_update
    s = state.settings()
    try:
        # Back to what the classifier said: drop the correction.
        same = stroke == (row["stroke_rules"] or "other") and row["stroke_source"] == "user"
        services.edit_swing_stroke(s, session_id, row["t_contact"], None if same else stroke)
        status, job = services.refresh_practice(s, session_id)
    except (EditConflict, ValueError, RuntimeError) as exc:
        return no_update, notification(str(exc), "Couldn't save", color="red")
    if status != "ran":
        msg = (
            f"Saved; queued job #{job} to update the session."
            if status == "queued"
            else "Saved; the running job picks it up."
        )
        return version + 1, notification(msg, "Stroke corrected", color="blue")
    return version + 1, notification(
        f"Marked as {sv.stroke_label(stroke).lower()}; shots and practice results updated.",
        "Stroke corrected",
    )


clientside_callback(
    """
    function(seek) {
        const v = document.getElementById("sw-video");
        if (!seek || !v) { return window.dash_clientside.no_update; }
        v.currentTime = Math.max(0, seek.t);
        if (seek.play) { v.play(); } else { v.pause(); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("sw-sink", "data"),
    Input("sw-seek", "data"),
    prevent_initial_call=True,
)

clientside_callback(
    sv.SKELETON_JS,
    Output("sw-skel-img", "src"),
    Output("sw-skel-img", "style"),
    Input("sw-time", "data"),
    Input("sw-skel", "data"),
    Input("sw-skel-on", "checked"),
)
