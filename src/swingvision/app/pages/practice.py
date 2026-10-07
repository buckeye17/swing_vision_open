"""Practice page (M5): where the shots landed against the targets, accuracy over the
session and per block, and every shot playable in the video.

The video and the stores are static; the analysis panels re-render from ``practice.parquet``
whenever ``pr-version`` changes (after a target, practice-type or shot edit has rerun the
cheap ``segments``/``practice_eval`` stages in-process).
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.app import state
from swingvision.app.components import practice_view as pv
from swingvision.app.components.target_editor import (
    register_target_editor,
    target_editor,
    targets_from_store,
)
from swingvision.app.components.ui import (
    fmt_duration,
    icon,
    no_output_root_alert,
    notification,
    page_header,
)
from swingvision.storage.edits import EditConflict
from swingvision.storage.schemas import PRACTICE_SUBMODE_LABELS

register_target_editor("pr-tgt")

OVERLAY_STYLE = {
    "position": "absolute",
    "inset": 0,
    "width": "100%",
    "height": "100%",
    "pointerEvents": "none",
}


def _missing(config, session_id: str):
    return dmc.Alert(
        [
            dmc.Text(
                "This session has no practice results yet. They appear once processing gets "
                "through the Segments and Practice accuracy stages."
            ),
            dmc.Anchor("Open the session", href=f"/session/{session_id}"),
        ],
        title="Not processed yet",
        color="blue",
        icon=icon("tabler:hourglass"),
    )


def layout(session_id: str | None = None, **_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Practice"), no_output_root_alert()], size="xl", px=0)
    found = state.session_for(session_id or "")
    if found is None:
        return dmc.Container(
            [page_header("Session not found"), dmc.Anchor("Back to library", href="/")],
            size="xl",
            px=0,
        )
    _lib, _row, session = found
    config = session.load_config()
    if config.practice is None:
        return dmc.Container([page_header(config.name, "Not a practice session")], size="xl", px=0)
    data = pv.load(session)
    video = config.video
    fps = video.fps_avg if video else 60.0
    subtitle = f"Practice · {PRACTICE_SUBMODE_LABELS[config.practice.submode]}"
    if video:
        subtitle += f" · {fmt_duration(video.duration_s)}"
    header_right = dmc.Group(
        [
            dmc.Anchor(
                dmc.Button(
                    "Session",
                    variant="default",
                    size="sm",
                    leftSection=icon("tabler:movie", 16),
                ),
                href=f"/session/{config.id}",
            ),
            dmc.Anchor(
                dmc.Button(
                    "Swings",
                    variant="default",
                    size="sm",
                    leftSection=icon("tabler:ball-tennis", 16),
                ),
                href=f"/swings/{config.id}",
            )
            if session.swings_path.exists()
            else None,
            dmc.Anchor(
                dmc.Button(
                    "Stats",
                    variant="default",
                    size="sm",
                    leftSection=icon("tabler:chart-bar", 16),
                ),
                href=f"/stats/{config.id}",
            ),
            dmc.Select(
                id="pr-submode",
                value=config.practice.submode,
                data=[{"value": k, "label": v} for k, v in PRACTICE_SUBMODE_LABELS.items()],
                size="sm",
                w=190,
                allowDeselect=False,
            ),
            dmc.Button(
                "Targets",
                id="pr-edit-targets",
                size="sm",
                leftSection=icon("tabler:target-arrow", 16),
            ),
        ],
        gap="sm",
    )
    has_proxy = session.proxy_path.exists()
    if has_proxy:
        player = html.Video(
            id="pr-video",
            src=f"/media/{config.id}/proxy.mp4",
            controls=True,
            preload="auto",
            style={"width": "100%", "display": "block", "background": "#000", "borderRadius": 8},
            **{
                "data-sv-player": "1",
                "data-fps": f"{fps}",
                "data-frame": "practice",  # overlays: assets/practice_frame.js
            },
        )
    else:
        player = dmc.Alert("The playback proxy isn't ready yet.", color="blue")
    overlays = pv.targets_overlay(data.cal, data.targets) if data and has_proxy else None
    player = html.Div(
        [
            player,
            html.Img(id="pr-overlay", src="", style={**OVERLAY_STYLE, "display": "none"}),
            html.Div(id="pr-label", style={"display": "none"}),
        ],
        style={"position": "relative"},
    )
    filters = dmc.Group(
        [
            dmc.SegmentedControl(
                id="pr-view",
                value="hit",
                data=[
                    {"value": "hit", "label": "As you hit"},
                    {"value": "court", "label": "On the court"},
                ],
                size="xs",
            ),
            dmc.Select(
                id="pr-block",
                value="all",
                data=[{"value": "all", "label": "All blocks"}]
                + [
                    {
                        "value": str(b["block_id"]),
                        "label": f"Block {b['block_id'] + 1} "
                        f"({pv.fmt_t(b['start_t'])}, {b['n_shots']} shots)",
                    }
                    for b in (data.blocks if data else [])
                ],
                size="xs",
                w=210,
                allowDeselect=False,
            ),
            dmc.SegmentedControl(
                id="pr-kind",
                value="all",
                data=[{"value": k, "label": v} for k, v in pv.KIND_FILTERS.items()],
                size="xs",
            ),
            dmc.Switch(id="pr-excluded", label="Show excluded", size="xs", checked=True),
        ],
        gap="sm",
    )
    modal = dmc.Modal(
        id="pr-targets-modal",
        title="Targets",
        size="80%",
        children=[
            target_editor("pr-tgt", config.practice.targets),
            dmc.Group(
                [
                    dmc.Button("Cancel", id="pr-targets-cancel", variant="default"),
                    dmc.Button(
                        "Save and recompute",
                        id="pr-targets-save",
                        leftSection=icon("tabler:check"),
                    ),
                ],
                justify="flex-end",
                mt="md",
            ),
        ],
    )
    body = (
        _missing(config, config.id)
        if data is None
        else dmc.Grid(
            [
                dmc.GridCol(
                    dmc.Stack(
                        [
                            player,
                            dmc.Text(
                                "Space play/pause · N/P next/previous shot · J/L ±5 s · "
                                "targets drawn in yellow follow your end",
                                size="xs",
                                c="dimmed",
                            ),
                            html.Div(id="pr-kpis"),
                            dmc.Paper(
                                [
                                    dmc.Text("Accuracy over the session", fw=600, size="sm"),
                                    dmc.Text(
                                        "Share of the last 10 shots; click a shot to play it.",
                                        size="xs",
                                        c="dimmed",
                                    ),
                                    dcc.Graph(id="pr-rolling", config={"displayModeBar": False}),
                                ],
                                p="sm",
                                withBorder=True,
                            ),
                            dmc.Paper(
                                [
                                    dmc.Text("Blocks", fw=600, size="sm", mb=4),
                                    html.Div(id="pr-blocks"),
                                ],
                                p="sm",
                                withBorder=True,
                            ),
                        ],
                        gap="sm",
                    ),
                    span={"base": 12, "lg": 7},
                ),
                dmc.GridCol(
                    dmc.Stack(
                        [
                            dmc.Paper(
                                [
                                    filters,
                                    # assets/shape_events.js: plot clicks → pr-court-click.
                                    html.Div(
                                        dcc.Graph(id="pr-court", config={"displayModeBar": False}),
                                        **{"data-click-store": "pr-court-click"},
                                    ),
                                    html.Div(id="pr-detail"),
                                ],
                                p="sm",
                                withBorder=True,
                            ),
                            dmc.Paper(
                                [
                                    dmc.Text("Shots", fw=600, size="sm", mb=4),
                                    html.Div(id="pr-shots-table"),
                                ],
                                p="sm",
                                withBorder=True,
                            ),
                            dmc.Paper(
                                [
                                    dmc.Text("Breakdown", fw=600, size="sm", mb=4),
                                    html.Div(id="pr-breakdown"),
                                ],
                                p="sm",
                                withBorder=True,
                            ),
                        ],
                        gap="sm",
                    ),
                    span={"base": 12, "lg": 5},
                ),
            ],
            gutter="md",
        )
    )
    return dmc.Container(
        [
            page_header(config.name, subtitle, right=header_right),
            dcc.Store(id="pr-session-id", data=config.id),
            dcc.Store(id="pr-version", data=0),
            dcc.Store(id="pr-edits-version", data=data.edits_version if data else 0),
            dcc.Store(id="pr-selected"),
            dcc.Store(id="pr-placing", data=False),
            dcc.Store(id="pr-seek"),
            dcc.Store(id="pr-sink"),
            dcc.Store(id="pr-court-click"),
            dcc.Store(id="pr-overlays", data=overlays),
            dcc.Store(id="pr-shots", data=pv.shots_store(data) if data else None),
            modal,
            body,
        ],
        size="xl",
        px=0,
    )


dash.register_page(
    __name__,
    path_template="/practice/<session_id>",
    title="Practice · Swing Vision Open",
    layout=layout,
)


def _data(session_id):
    found = state.session_for(session_id or "")
    return None if found is None else pv.load(found[2])


@callback(
    Output("pr-kpis", "children"),
    Output("pr-rolling", "figure"),
    Output("pr-blocks", "children"),
    Output("pr-shots-table", "children"),
    Output("pr-breakdown", "children"),
    Output("pr-shots", "data"),
    Output("pr-overlays", "data"),
    Output("pr-edits-version", "data"),
    Input("pr-version", "data"),
    Input("pr-block", "value"),
    Input("pr-kind", "value"),
    Input("pr-excluded", "checked"),
    Input("pr-selected", "data"),
    State("pr-session-id", "data"),
)
def _render(_version, block, kind, show_excluded, selected, session_id):
    data = _data(session_id)
    if data is None:
        return (no_update,) * 8
    rows = pv.filtered(data, block, kind, excluded=bool(show_excluded))
    has_targets = bool(data.targets)
    return (
        pv.kpis(rows, has_targets),
        pv.rolling_figure(rows, has_targets),
        pv.blocks_table(data, has_targets),
        pv.shots_table(rows, selected, has_targets),
        pv.breakdown_table(rows, has_targets),
        pv.shots_store(data),
        pv.targets_overlay(data.cal, data.targets),
        data.edits_version,
    )


@callback(
    Output("pr-court", "figure"),
    Output("pr-detail", "children"),
    Input("pr-version", "data"),
    Input("pr-view", "value"),
    Input("pr-block", "value"),
    Input("pr-kind", "value"),
    Input("pr-excluded", "checked"),
    Input("pr-selected", "data"),
    Input("pr-placing", "data"),
    State("pr-session-id", "data"),
)
def _court(_version, view, block, kind, show_excluded, selected, placing, session_id):
    data = _data(session_id)
    if data is None:
        return no_update, no_update
    rows = pv.filtered(data, block, kind, excluded=bool(show_excluded))
    sel = next((r for r in data.rows if r["segment_id"] == selected), None)
    fig = pv.court_map(data, rows, view or "hit", selected, bool(placing))
    return fig, pv.shot_detail(sel, bool(placing))


@callback(
    Output("pr-block", "value"),
    Input({"type": "pr-block-row", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _pick_block(clicks):
    if not isinstance(ctx.triggered_id, dict) or not any(clicks or []):
        return no_update
    return str(ctx.triggered_id["index"])


@callback(
    Output("pr-selected", "data"),
    Output("pr-seek", "data"),
    Output("pr-placing", "data", allow_duplicate=True),
    Output("pr-version", "data", allow_duplicate=True),
    Output("pr-edits-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("pr-court", "clickData"),
    Input("pr-court-click", "data"),
    Input("pr-rolling", "clickData"),
    Input({"type": "pr-shot-row", "index": ALL}, "n_clicks"),
    State("pr-placing", "data"),
    State("pr-selected", "data"),
    State("pr-view", "value"),
    State("pr-session-id", "data"),
    State("pr-version", "data"),
    State("pr-edits-version", "data"),
    prevent_initial_call=True,
)
def _select(
    court_click, plot_click, roll_click, row_clicks, placing, selected, view, sid, version, ev
):
    nu = no_update
    fired = {t["prop_id"].rsplit(".", 1)[0] for t in ctx.triggered}
    if placing:
        # Placing a landing: any click on the map counts, at its exact court position.
        if "pr-court-click" in fired and selected is not None and plot_click:
            return _place(sid, selected, plot_click["x"], plot_click["y"], view, version, ev)
        if fired <= {"pr-court", "pr-court-click"}:
            return (nu,) * 6
    elif fired == {"pr-court-click"}:
        return (nu,) * 6  # a click beside the landings
    trig = ctx.triggered_id
    if "pr-court" in fired:
        trig = "pr-court"
    if isinstance(trig, dict):
        if not any(row_clicks or []):
            return (nu,) * 6
        seg_id = trig["index"]
    else:
        click = court_click if trig == "pr-court" else roll_click
        pts = (click or {}).get("points") or []
        if not pts:
            return (nu,) * 6
        cd = pts[0].get("customdata")
        if cd is None:
            return (nu,) * 6
        seg_id = int(cd[0])
    data = _data(sid)
    if data is None:
        return (nu,) * 6
    seg = data.segments.get(seg_id)
    t = max(0.0, (seg or {}).get("start_t", 0.0))
    return seg_id, {"t": t, "play": True}, False, nu, nu, nu


def _place(sid, seg_id, x, y, view, version, ev):
    data = _data(sid)
    r = next((r for r in data.rows if r["segment_id"] == seg_id), None) if data else None
    if r is None:
        return (no_update,) * 6
    if view == "hit" and r["side"] == 1:  # the map shows the hitter's frame
        x, y = -x, -y
    return _edit(sid, r, version, ev, landing=[float(x), float(y)], confirmed=True)


def _anchor(sid, r) -> float:
    from swingvision.analysis.practice import anchor_time

    data = _data(sid)
    seg = data.segments.get(r["segment_id"]) if data else None
    return anchor_time(seg) if seg else r["t_contact"]


def _edit(sid, r, version, ev, **change):
    s = state.settings()
    try:
        services.edit_practice_shot(s, sid, _anchor(sid, r), expected_version=ev, **change)
        status, job = services.refresh_practice(s, sid)
    except EditConflict as exc:
        return (no_update,) * 5 + (notification(str(exc), color="red"),)
    except Exception as exc:
        return (no_update,) * 5 + (notification(str(exc), "Could not save", "red"),)
    msg = None
    if status == "busy":
        msg = notification("Saved; the session is being processed and picks it up there.")
    elif status == "queued":
        msg = notification(f"Saved; recomputing in job #{job}.")
    return no_update, no_update, False, (version or 0) + 1, (ev or 0) + 1, msg or no_update


@callback(
    Output("pr-selected", "data", allow_duplicate=True),
    Output("pr-seek", "data", allow_duplicate=True),
    Output("pr-placing", "data", allow_duplicate=True),
    Output("pr-version", "data", allow_duplicate=True),
    Output("pr-edits-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("pr-play", "n_clicks"),
    Input("pr-confirm", "n_clicks"),
    Input("pr-place", "n_clicks"),
    Input("pr-unplace", "n_clicks"),
    Input("pr-exclude", "n_clicks"),
    State("pr-selected", "data"),
    State("pr-placing", "data"),
    State("pr-session-id", "data"),
    State("pr-version", "data"),
    State("pr-edits-version", "data"),
    prevent_initial_call=True,
)
def _shot_action(_p, _c, _pl, _u, _x, selected, placing, sid, version, ev):
    nu = no_update
    if selected is None or not ctx.triggered or not ctx.triggered[0]["value"]:
        return (nu,) * 6
    data = _data(sid)
    r = next((r for r in data.rows if r["segment_id"] == selected), None) if data else None
    if r is None:
        return (nu,) * 6
    trig = ctx.triggered_id
    if trig == "pr-play":
        seg = data.segments.get(selected, {})
        return (
            nu,
            {"t": max(0.0, seg.get("start_t", r["t_contact"] - 1)), "play": True},
            nu,
            nu,
            nu,
            nu,
        )
    if trig == "pr-place":
        return nu, nu, not placing, nu, nu, nu
    if trig == "pr-confirm":
        return _edit(sid, r, version, ev, confirmed=not r["landing_confirmed"])
    if trig == "pr-unplace":
        return _edit(sid, r, version, ev, landing=None, confirmed=False)
    return _edit(sid, r, version, ev, exclude=not r["excluded"])


@callback(
    Output("pr-targets-modal", "opened"),
    Input("pr-edit-targets", "n_clicks"),
    Input("pr-targets-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def _toggle_modal(_open, _cancel):
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update
    return ctx.triggered_id == "pr-edit-targets"


@callback(
    Output("pr-targets-modal", "opened", allow_duplicate=True),
    Output("pr-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("pr-targets-save", "n_clicks"),
    State("pr-tgt-targets", "data"),
    State("pr-session-id", "data"),
    State("pr-version", "data"),
    running=[(Output("pr-targets-save", "loading"), True, False)],
    prevent_initial_call=True,
)
def _save_targets(n, targets, sid, version):
    if not n:
        return no_update, no_update, no_update
    return _apply(sid, version, targets=targets_from_store(targets), what="Targets saved")


@callback(
    Output("pr-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("pr-submode", "value"),
    State("pr-session-id", "data"),
    State("pr-version", "data"),
    prevent_initial_call=True,
)
def _set_submode(submode, sid, version):
    found = state.session_for(sid or "")
    config = found[2].load_config() if found else None
    if not submode or config is None or config.practice is None:
        return no_update, no_update
    if submode == config.practice.submode:  # the select mounting, not a change
        return no_update, no_update
    _, v, note = _apply(sid, version, submode=submode, what="Practice type changed")
    return v, note


def _apply(sid, version, what: str, **change):
    s = state.settings()
    try:
        services.set_practice(s, sid, **change)
        status, job = services.refresh_practice(s, sid)
    except Exception as exc:
        return no_update, no_update, notification(str(exc), "Could not save", "red")
    if status == "ran":
        msg = f"{what}; accuracy recomputed."
    elif status == "queued":
        msg = f"{what}; recomputing in job #{job}."
    else:
        msg = f"{what}; the session is being processed and picks it up there."
    return False, (version or 0) + 1, notification(msg, icon_name="tabler:check")


clientside_callback(
    """
    function(seek) {
        const v = document.getElementById("pr-video");
        if (!seek || !v) { return window.dash_clientside.no_update; }
        v.currentTime = seek.t;
        if (seek.play) { v.play(); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("pr-sink", "data"),
    Input("pr-seek", "data"),
    prevent_initial_call=True,
)

# Shot starts for the N/P keys (assets/video_sync.js).
clientside_callback(
    """
    function(shots) {
        const v = document.getElementById("pr-video");
        if (v && shots) { v.dataset.segments = JSON.stringify(shots.t0); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("pr-sink", "data", allow_duplicate=True),
    Input("pr-shots", "data"),
    prevent_initial_call="initial_duplicate",
)


# The overlays follow playback in assets/practice_frame.js; redraw them when the shots or
# targets change (a paused video doesn't redraw by itself).
clientside_callback(
    """
    function() {
        if (window.svFrameRefresh) { window.svFrameRefresh("pr-video"); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("pr-sink", "data", allow_duplicate=True),
    Input("pr-shots", "data"),
    Input("pr-overlays", "data"),
    prevent_initial_call="initial_duplicate",
)
