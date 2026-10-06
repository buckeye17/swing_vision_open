"""Library: all sessions in the output folder."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.app import state
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
from swingvision.pipeline.stage import read_manifest
from swingvision.pipeline.stages import default_registry
from swingvision.storage.schemas import PRACTICE_SUBMODE_LABELS
from swingvision.storage.session import Session


def layout(**_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Library"), no_output_root_alert()], size="xl", px=0)
    return dmc.Container(
        [
            page_header(
                "Library",
                str(state.settings().output_root),
                right=dmc.Anchor(
                    dmc.Button("New session", leftSection=icon("tabler:plus")), href="/new"
                ),
            ),
            dcc.Interval(id="lib-poll", interval=3000),
            dcc.Store(id="lib-delete-id"),
            dcc.Store(id="lib-sig"),
            html.Div(id="lib-table"),
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


def _moved(root, s: dict) -> str:
    """Headline stat: distance covered (from the movement stage's manifest)."""
    m = read_manifest(Session.open(root, s["dir_name"]), "movement")
    dist = (m or {}).get("extra", {}).get("distance_m")
    return f"{dist:,.0f} m" if dist is not None else "–"


def _accuracy(root, s: dict):
    """Headline stat for practice sessions: in % (and target hits), linking to the Practice
    page (from the practice_eval stage's manifest)."""
    m = read_manifest(Session.open(root, s["dir_name"]), "practice_eval")
    extra = (m or {}).get("extra", {})
    if s["mode"] != "practice" or not extra.get("n"):
        return "–"
    text = f"{extra['n']} shots"
    if extra.get("in_pct") is not None:
        text += f" · {extra['in_pct']:.0%} in"
    if extra.get("target_pct") is not None:
        text += f" · {extra['target_pct']:.0%} on target"
    return dmc.Anchor(text, href=f"/practice/{s['id']}", size="sm")


def _row(root, s: dict):
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
                        "Calibrate court",
                        href=f"/calibrate/{sid}",
                        leftSection=icon("tabler:target", 14),
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
    return dmc.TableTr(
        [
            dmc.TableTd(
                dmc.Image(
                    src=f"/media/{sid}/thumb.jpg",
                    w=96,
                    h=54,
                    radius="sm",
                    fallbackSrc="data:image/gif;base64,R0lGODlhAQABAAAAACw=",
                )
            ),
            dmc.TableTd(dmc.Anchor(s["name"], href=f"/session/{sid}", fw=600)),
            dmc.TableTd(mode),
            dmc.TableTd(fmt_duration(s["duration_s"])),
            dmc.TableTd(_moved(root, s)),
            dmc.TableTd(_accuracy(root, s)),
            dmc.TableTd(fmt_time(s["created_at"])),
            dmc.TableTd(status_badge(s["status"]), style={"whiteSpace": "nowrap"}),
            dmc.TableTd(menu),
        ]
    )


@callback(
    Output("lib-table", "children"),
    Output("lib-sig", "data"),
    Input("lib-poll", "n_intervals"),
    State("lib-sig", "data"),
)
def _render(_, last_sig):
    lib = state.library()
    if lib is None:
        return no_output_root_alert(), None
    sessions = lib.list_sessions()
    # Re-rendering closes open row menus, so only re-render when something changed.
    sig = repr([(s["id"], s["status"], s["updated_at"], s["name"]) for s in sessions])
    if sig == last_sig:
        return no_update, no_update
    return _table(lib.root, sessions), sig


def _table(root, sessions: list[dict]):
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
    head = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh(h)
                for h in (
                    "",
                    "Name",
                    "Mode",
                    "Length",
                    "Moved",
                    "Practice",
                    "Created",
                    "Status",
                    "",
                )
            ]
        )
    )
    return dmc.Paper(
        dmc.Table(
            [head, dmc.TableTbody([_row(root, s) for s in sessions])],
            highlightOnHover=True,
            verticalSpacing="xs",
        ),
        withBorder=True,
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
