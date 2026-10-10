"""Library: all sessions in the output folder. Sessions can be tagged, and a selection of
them opened together in Stats (M7a)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.app import state, units
from swingvision.app.components.file_browser import file_browser, register_file_browser
from swingvision.app.components.ui import (
    fmt_duration,
    fmt_time,
    icon,
    no_output_root_alert,
    notification,
    page_header,
    status_badge,
)
from swingvision.app.worker_control import ensure_worker
from swingvision.io.probe import VIDEO_EXTENSIONS
from swingvision.pipeline.stage import read_manifest
from swingvision.pipeline.stages import default_registry
from swingvision.storage.schemas import PRACTICE_SUBMODE_LABELS
from swingvision.storage.session import Session

register_file_browser("lib-fb", mode="file", extensions=VIDEO_EXTENSIONS)


def layout(relink: str | None = None, **_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Library"), no_output_root_alert()], size="xl", px=0)
    return dmc.Container(
        [
            dmc.Group(
                [
                    dmc.Stack(
                        [
                            dmc.Title("Library", order=1),
                            dmc.Text(
                                str(state.settings().output_root),
                                size="xs",
                                c="dimmed",
                                ff="monospace",
                                style={"overflowWrap": "anywhere"},
                            ),
                        ],
                        gap=4,
                    ),
                    dmc.Anchor(
                        dmc.Button("New session", leftSection=icon("tabler:plus")), href="/new"
                    ),
                ],
                justify="space-between",
                align="flex-end",
                mb="lg",
            ),
            dcc.Interval(id="lib-poll", interval=3000),
            dcc.Store(id="lib-selected", data=[]),
            dmc.Group(
                [
                    dmc.Text(id="lib-sel-text", size="sm", c="dimmed"),
                    dmc.Anchor(
                        dmc.Button(
                            "Open in Stats",
                            id="lib-open-stats-btn",
                            size="xs",
                            variant="default",
                            disabled=True,
                            leftSection=icon("tabler:chart-bar", 14),
                        ),
                        id="lib-open-stats",
                        href="/stats",
                    ),
                ],
                justify="flex-end",
                gap="sm",
                mb="xs",
            ),
            dcc.Store(id="lib-tags-id"),
            dmc.Modal(
                id="lib-tags-modal",
                title="Session tags",
                children=dmc.Stack(
                    [
                        dmc.Text(
                            "Tags group sessions for Stats, e.g. “new racket” or “indoor”.",
                            size="sm",
                            c="dimmed",
                        ),
                        dmc.TagsInput(
                            id="lib-tags-input",
                            placeholder="Type a tag, press Enter",
                            clearable=True,
                            splitChars=[","],
                        ),
                        dmc.Group(
                            [
                                dmc.Button("Cancel", id="lib-tags-cancel", variant="default"),
                                dmc.Button("Save", id="lib-tags-save"),
                            ],
                            justify="flex-end",
                        ),
                    ]
                ),
            ),
            dcc.Store(id="lib-delete-id"),
            dcc.Store(id="lib-sig"),
            dcc.Store(id="lib-relink-id", data=relink),
            html.Div(id="lib-table"),
            _relink_modal(relink),
            file_browser("lib-fb", mode="file", title="Find the session's video"),
            dmc.Modal(
                id="lib-delete-modal",
                title="Delete session data?",
                children=dmc.Stack(
                    [
                        dmc.Text(id="lib-delete-text", size="sm"),
                        dmc.Group(
                            [
                                dmc.Button("Cancel", id="lib-delete-cancel", variant="default"),
                                dmc.Button("Delete", id="lib-delete-confirm", color="red"),
                            ],
                            justify="flex-end",
                        ),
                    ]
                ),
            ),
        ],
        size="xl",
        px=0,
    )


dash.register_page(__name__, path="/", title="Library · Swing Vision Open", order=0, layout=layout)


def _relink_text(sid: str | None):
    lib = state.library()
    row = lib.get_session(sid) if lib and sid else None
    if row is None:
        return "", None
    path = Path(row["source_path"])
    found = path.is_file()
    text = dmc.Stack(
        [
            dmc.Text(
                f"“{row['name']}” was made from {path.name}, which is "
                + ("still in place." if found else "no longer there:"),
                size="sm",
            ),
            dmc.Code(str(path), block=True),
            dmc.Text(
                "Choose the file where it is now. It must be the same video (it's checked); other "
                "sessions whose videos are missing are looked for in the same folder too. Results "
                "and the playback proxy don't need the video, only reprocessing does.",
                size="sm",
                c="dimmed",
            ),
        ],
        gap="xs",
    )
    start = str(path.parent) if path.parent.is_dir() else None
    return text, start


def _relink_modal(sid: str | None):
    text, _start = _relink_text(sid)
    return dmc.Modal(
        id="lib-relink-modal",
        title="Relink the source video",
        opened=bool(sid) and text != "",
        size="lg",
        children=dmc.Stack(
            [
                html.Div(text, id="lib-relink-text"),
                dmc.Group(
                    [
                        dmc.Button("Cancel", id="lib-relink-cancel", variant="default"),
                        dmc.Button(
                            "Choose file…",
                            id="lib-fb-open",
                            leftSection=icon("tabler:folder-open", 16),
                        ),
                    ],
                    justify="flex-end",
                ),
            ]
        ),
    )


def _manifest_extra(root, s: dict, stage: str) -> dict:
    m = read_manifest(Session.open(root, s["dir_name"]), stage)
    return (m or {}).get("extra", {})


def _metric(value: str, sub: str = ""):
    return dmc.TableTd(
        html.Div(
            [
                html.Div(value, className="sv-stat-value", style={"fontSize": 22}),
                html.Div(sub, className="sv-stat-sub"),
            ]
        ),
        ta="right",
    )


def _metrics(root, s: dict, u: units.Units) -> list:
    """Shots, in % and distance moved (from the stages' manifests)."""
    moved = _manifest_extra(root, s, "movement").get("distance_m")
    moved_cell = _metric(f"{u.len(moved):,.0f}", u.len_unit) if moved is not None else _metric("–")
    extra = _manifest_extra(root, s, "practice_eval") if s["mode"] == "practice" else {}
    if not extra.get("n"):
        return [_metric("–"), _metric("–"), moved_cell]
    in_pct = extra.get("in_pct")
    target = extra.get("target_pct")
    return [
        _metric(f"{extra['n']}", "practice shots"),
        _metric(
            "–" if in_pct is None else f"{in_pct:.0%}",
            "" if target is None else f"{target:.0%} on target",
        ),
        moved_cell,
    ]


def _row(root, s: dict, u: units.Units, tags: list[str], selected: bool):
    mode = s["mode"].capitalize()
    if s["submode"]:
        mode += f" · {PRACTICE_SUBMODE_LABELS.get(s['submode'], s['submode'])}"
    sid = s["id"]
    menu = dmc.Menu(
        [
            dmc.MenuTarget(dmc.ActionIcon(icon("tabler:dots"), variant="subtle", color="gray")),
            dmc.MenuDropdown(
                [
                    dmc.MenuItem(
                        "Process (stale stages)",
                        id={"type": "lib-process", "index": sid},
                        leftSection=icon("tabler:player-play", 14),
                    ),
                    dmc.MenuItem(
                        "Practice accuracy",
                        href=f"/practice/{sid}",
                        leftSection=icon("tabler:target-arrow", 14),
                        disabled=s["mode"] != "practice",
                    ),
                    dmc.MenuItem(
                        "Stats",
                        href=f"/stats/{sid}",
                        leftSection=icon("tabler:chart-bar", 14),
                    ),
                    dmc.MenuItem(
                        "Tags…",
                        id={"type": "lib-tags", "index": sid},
                        leftSection=icon("tabler:tags", 14),
                    ),
                    dmc.MenuItem(
                        "Export shots (CSV)",
                        href=f"/export/{sid}/shots.csv",
                        refresh=True,
                        leftSection=icon("tabler:download", 14),
                    ),
                    dmc.MenuItem(
                        "Calibrate court",
                        href=f"/calibrate/{sid}",
                        leftSection=icon("tabler:target", 14),
                    ),
                    dmc.MenuItem(
                        "Relink video…",
                        id={"type": "lib-relink", "index": sid},
                        leftSection=icon("tabler:link", 14),
                    ),
                    dmc.MenuItem(
                        "Reprocess from scratch",
                        id={"type": "lib-reprocess", "index": sid},
                        leftSection=icon("tabler:refresh", 14),
                    ),
                    dmc.MenuDivider(),
                    dmc.MenuItem(
                        "Delete session data",
                        id={"type": "lib-delete", "index": sid},
                        color="red",
                        leftSection=icon("tabler:trash", 14),
                    ),
                ]
            ),
        ],
        position="bottom-end",
    )
    status = [status_badge(s["status"])]
    if services.source_missing(s):
        status.append(
            dmc.Tooltip(
                html.Span(
                    dmc.Badge("video missing", color="red", variant="outline", size="sm"),
                    id={"type": "lib-relink-badge", "index": sid},
                    style={"cursor": "pointer"},
                ),
                label=f"Not found: {s['source_path']}. Click to relink.",
                multiline=True,
                w=320,
            )
        )
    links = []
    if s["status"] == "ready":
        if s["mode"] == "practice":
            links.append(("Practice", f"/practice/{sid}"))
        links.append(("Stats", f"/stats/{sid}"))
    recorded = (s.get("recorded_on") or "")[:10]
    when = f"recorded {recorded}" if recorded else f"created {fmt_time(s['created_at'])}"
    return dmc.TableTr(
        [
            dmc.TableTd(
                dmc.Checkbox(
                    id={"type": "lib-check", "index": sid},
                    checked=selected,
                    size="xs",
                    **{"aria-label": f"Select {s['name']}"},
                ),
                w=28,
            ),
            dmc.TableTd(
                dmc.Anchor(
                    dmc.Image(
                        src=f"/media/{sid}/thumb.jpg",
                        w=128,
                        h=72,
                        radius="md",
                        fallbackSrc="data:image/gif;base64,R0lGODlhAQABAAAAACw=",
                    ),
                    href=f"/session/{sid}",
                    **{"aria-label": f"Open {s['name']}"},
                ),
                w=144,
            ),
            dmc.TableTd(
                dmc.Stack(
                    [
                        dmc.Anchor(
                            s["name"], href=f"/session/{sid}", fw=600, c="var(--mantine-color-text)"
                        ),
                        dmc.Text(
                            f"{mode} · {fmt_duration(s['duration_s'])} · {when}",
                            size="sm",
                            c="dimmed",
                        ),
                        dmc.Group(
                            [dmc.Badge(t, variant="light", color="gray", size="sm") for t in tags],
                            gap=4,
                            mt=2,
                        )
                        if tags
                        else None,
                        dmc.Group(
                            [
                                dmc.Anchor(
                                    dmc.Button(label, variant="default", size="compact-xs"),
                                    href=href,
                                )
                                for label, href in links
                            ],
                            gap=6,
                            mt=4,
                        )
                        if links
                        else None,
                    ],
                    gap=0,
                )
            ),
            *_metrics(root, s, u),
            dmc.TableTd(dmc.Group(status, gap=4, wrap="nowrap"), style={"whiteSpace": "nowrap"}),
            dmc.TableTd(menu),
        ]
    )


@callback(
    Output("lib-table", "children"),
    Output("lib-sig", "data"),
    Input("lib-poll", "n_intervals"),
    State("lib-sig", "data"),
    State("lib-selected", "data"),
)
def _render(_, last_sig, selected):
    lib = state.library()
    if lib is None:
        return no_output_root_alert(), None
    sessions = lib.list_sessions()
    # Re-rendering closes open row menus, so only re-render when something changed.
    sig = repr(
        [
            (s["id"], s["status"], s["updated_at"], s["name"], services.source_missing(s))
            for s in sessions
        ]
    )
    if sig == last_sig:
        return no_update, no_update
    return _table(lib.root, sessions, lib.tags_by_session(), set(selected or [])), sig


def _table(root, sessions: list[dict], tags: dict | None = None, selected: set | None = None):
    if not sessions:
        return dmc.Paper(
            dmc.Stack(
                [
                    icon("tabler:video-off", 40, color="gray"),
                    dmc.Text("No sessions yet.", c="dimmed"),
                    dmc.Anchor("Create your first session", href="/new"),
                ],
                align="center",
                gap="xs",
            ),
            p="xl",
            withBorder=True,
        )
    u = units.current()
    head = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh(""),
                dmc.TableTh(""),
                dmc.TableTh("Session"),
                dmc.TableTh("Shots", ta="right"),
                dmc.TableTh("In", ta="right"),
                dmc.TableTh("Moved", ta="right"),
                dmc.TableTh("Status"),
                dmc.TableTh(""),
            ]
        )
    )
    waiting = [x for x in sessions if x["status"] == "needs_action"]
    banner = (
        dmc.Alert(
            dmc.Group(
                [
                    dmc.Text(
                        (
                            f"“{waiting[0]['name']}” is waiting for you."
                            if len(waiting) == 1
                            else f"{len(waiting)} sessions are waiting for you."
                        )
                        + " Processing continues once you've had a look.",
                        size="sm",
                    ),
                    dmc.Anchor(dmc.Button("Open Jobs", color="violet", size="xs"), href="/jobs"),
                ],
                justify="space-between",
            ),
            title="Needs action",
            color="violet",
            icon=icon("tabler:alert-triangle"),
            mb="md",
        )
        if waiting
        else None
    )
    return html.Div(
        [
            banner,
            dmc.Paper(
                dmc.TableScrollContainer(
                    dmc.Table(
                        [
                            head,
                            dmc.TableTbody(
                                [
                                    _row(
                                        root,
                                        s,
                                        u,
                                        (tags or {}).get(s["id"], []),
                                        s["id"] in (selected or set()),
                                    )
                                    for s in sessions
                                ]
                            ),
                        ],
                        highlightOnHover=True,
                        verticalSpacing="sm",
                        horizontalSpacing="md",
                    ),
                    minWidth=820,
                ),
                withBorder=True,
            ),
        ]
    )


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("lib-delete-id", "data"),
    Output("lib-delete-modal", "opened"),
    Output("lib-delete-text", "children"),
    Input({"type": "lib-process", "index": ALL}, "n_clicks"),
    Input({"type": "lib-reprocess", "index": ALL}, "n_clicks"),
    Input({"type": "lib-delete", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _menu(_p, _r, _d):
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not ctx.triggered[0]["value"]:
        return no_update, no_update, no_update, no_update
    sid = trig["index"]
    s = state.settings()
    lib = state.library()
    row = lib.get_session(sid) if lib else None
    if row is None:
        return no_update, no_update, no_update, no_update
    if trig["type"] == "lib-delete":
        text = (
            f"This deletes everything the app derived for “{row['name']}” (proxy, "
            f"analysis, edits). The source video is not touched: {row['source_path']}"
        )
        return no_update, sid, True, text
    force = default_registry().names() if trig["type"] == "lib-reprocess" else None
    try:
        job_id = services.enqueue(s, sid, force=force)
    except ValueError as exc:
        return notification(str(exc), color="red"), no_update, no_update, no_update
    if state.OPTIONS.start_worker:
        ensure_worker(s.output_root)
    return (
        notification(f"Job #{job_id} queued for “{row['name']}”."),
        no_update,
        no_update,
        no_update,
    )


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("lib-delete-modal", "opened", allow_duplicate=True),
    Output("lib-poll", "n_intervals"),
    Input("lib-delete-confirm", "n_clicks"),
    Input("lib-delete-cancel", "n_clicks"),
    State("lib-delete-id", "data"),
    prevent_initial_call=True,
)
def _delete(confirm, _cancel, sid):
    if ctx.triggered_id != "lib-delete-confirm" or not confirm or not sid:
        return no_update, False, no_update
    try:
        services.delete_session(state.settings(), sid)
    except ValueError as exc:
        return notification(str(exc), color="red"), False, no_update
    return notification("Session data deleted.", color="gray"), False, 0


@callback(
    Output("lib-relink-id", "data"),
    Output("lib-relink-modal", "opened"),
    Output("lib-relink-text", "children"),
    Output("lib-fb-start", "data"),
    Input({"type": "lib-relink", "index": ALL}, "n_clicks"),
    Input({"type": "lib-relink-badge", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _open_relink(_menu, _badge):
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not ctx.triggered[0]["value"]:
        return no_update, no_update, no_update, no_update
    text, start = _relink_text(trig["index"])
    return trig["index"], True, text, start


@callback(
    Output("lib-fb-start", "data", allow_duplicate=True),
    Input("lib-relink-id", "data"),
    prevent_initial_call="initial_duplicate",
)
def _relink_start(sid):
    """Open the file browser in the video's old folder when it still exists."""
    return _relink_text(sid)[1] if sid else no_update


@callback(
    Output("lib-relink-modal", "opened", allow_duplicate=True),
    Input("lib-relink-cancel", "n_clicks"),
    Input("lib-fb-open", "n_clicks"),
    prevent_initial_call=True,
)
def _close_relink(cancel, browse):
    # "Choose file…" opens the file browser; this dialog closes behind it.
    return False if (cancel or browse) else no_update


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("lib-poll", "n_intervals", allow_duplicate=True),
    Input("lib-fb-result", "data"),
    State("lib-relink-id", "data"),
    prevent_initial_call=True,
)
def _relink(result, sid):
    if not result or not sid:
        return no_update, no_update
    s = state.settings()
    try:
        relinked = services.relink_session(s, sid, Path(result["path"]))
    except services.RelinkError as exc:
        return notification(str(exc), "Not relinked", color="red"), no_update
    others = len(relinked) - 1
    msg = f"Relinked to {Path(result['path']).name}."
    if others:
        msg += f" Found the videos of {others} more session{'s' if others > 1 else ''} there too."
    if state.OPTIONS.start_worker:
        ensure_worker(s.output_root)
    return notification(msg, icon_name="tabler:link"), 0


@callback(
    Output("lib-selected", "data"),
    Output("lib-sel-text", "children"),
    Output("lib-open-stats", "href"),
    Output("lib-open-stats-btn", "disabled"),
    Input({"type": "lib-check", "index": ALL}, "checked"),
    State({"type": "lib-check", "index": ALL}, "id"),
)
def _selection(checked, ids):
    """Selected sessions open together in Stats."""
    chosen = [i["index"] for i, c in zip(ids or [], checked or [], strict=False) if c]
    if not chosen:
        return [], "Select sessions to see their statistics together.", "/stats", True
    text = f"{len(chosen)} session{'s' if len(chosen) > 1 else ''} selected"
    return chosen, text, "/stats?sessions=" + quote(",".join(chosen), safe=","), False


@callback(
    Output("lib-tags-id", "data"),
    Output("lib-tags-modal", "opened"),
    Output("lib-tags-input", "value"),
    Output("lib-tags-input", "data"),
    Input({"type": "lib-tags", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _open_tags(_clicks):
    trig = ctx.triggered_id
    lib = state.library()
    if not isinstance(trig, dict) or not ctx.triggered[0]["value"] or lib is None:
        return no_update, no_update, no_update, no_update
    sid = trig["index"]
    return sid, True, lib.session_tags(sid), lib.all_tags()


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("lib-tags-modal", "opened", allow_duplicate=True),
    Output("lib-poll", "n_intervals", allow_duplicate=True),
    Input("lib-tags-save", "n_clicks"),
    Input("lib-tags-cancel", "n_clicks"),
    State("lib-tags-id", "data"),
    State("lib-tags-input", "value"),
    prevent_initial_call=True,
)
def _save_tags(save, _cancel, sid, tags):
    if ctx.triggered_id != "lib-tags-save" or not save or not sid:
        return no_update, False, no_update
    try:
        stored = services.set_session_tags(state.settings(), sid, tags or [])
    except ValueError as exc:
        return notification(str(exc), color="red"), False, no_update
    text = ", ".join(stored) if stored else "no tags"
    return notification(f"Tags saved: {text}.", icon_name="tabler:tags"), False, 0
