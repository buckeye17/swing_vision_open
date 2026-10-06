"""Jobs: live queue with per-stage progress, ETA, cancel/retry, and log tails."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, callback, ctx, dcc, html, no_update

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
from swingvision.pipeline.stages import default_registry
from swingvision.pipeline.stages.court import CameraStage
from swingvision.storage.library import RUNNING, Job, Library, parse_iso
from swingvision.storage.session import Session

LOG_TAIL_LINES = 12


def layout(**_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Jobs"), no_output_root_alert()], size="lg", px=0)
    return dmc.Container(
        [
            page_header(
                "Jobs",
                "Processing runs in a background worker; you can close this page.",
                right=dmc.Group(
                    [
                        dmc.SegmentedControl(
                            id="jobs-filter",
                            value="all",
                            size="xs",
                            data=[
                                {"value": "active", "label": "Active"},
                                {"value": "all", "label": "All"},
                            ],
                        ),
                        dmc.Switch(id="jobs-logs", label="Logs", size="sm"),
                    ],
                    gap="md",
                ),
            ),
            dcc.Interval(id="jobs-poll", interval=1500),
            html.Div(id="jobs-list"),
        ],
        size="lg",
        px=0,
    )


dash.register_page(__name__, path="/jobs", title="Jobs · Swing Vision Open", order=2, layout=layout)


def _eta(job: Job) -> str | None:
    started = parse_iso(job.started_at)
    if job.status != RUNNING or started is None or job.progress < 0.02:
        return None
    elapsed = (datetime.now(UTC) - started).total_seconds()
    return fmt_duration(elapsed * (1 - job.progress) / job.progress)


def _log_tail(lib: Library, job: Job) -> str:
    row = lib.get_session(job.session_id)
    if row is None:
        return ""
    path = Session.open(lib.root, row["dir_name"]).job_log_path(job.id)
    if not path.exists():
        return "(no log yet)"
    with path.open(encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=LOG_TAIL_LINES))


def _job_card(lib: Library, job: Job, sessions: dict, titles: dict, show_logs: bool):
    session = sessions.get(job.session_id)
    name = session["name"] if session else job.session_id
    stages = lib.job_stages(job.id)
    eta = _eta(job)
    active = job.status in ("queued", "running")
    actions = []
    if active:
        actions.append(
            dmc.Button(
                "Cancel",
                id={"type": "job-cancel", "index": job.id},
                size="xs",
                variant="light",
                color="red",
                disabled=job.cancel_requested,
                leftSection=icon("tabler:player-stop", 14),
            )
        )
    elif job.status in ("failed", "cancelled", "needs_action"):
        actions.append(
            dmc.Button(
                "Retry",
                id={"type": "job-retry", "index": job.id},
                size="xs",
                variant="light",
                leftSection=icon("tabler:refresh", 14),
            )
        )
    if session and job.status == "needs_action" and job.action == "relink":
        actions.insert(
            0,
            dmc.Anchor(
                dmc.Button("Relink video", size="xs", leftSection=icon("tabler:link", 14)),
                href=f"/?relink={job.session_id}",
            ),
        )
    elif session and job.status == "needs_action" and job.current_stage == CameraStage.name:
        actions.insert(
            0,
            dmc.Anchor(
                dmc.Button("Review calibration", size="xs", leftSection=icon("tabler:target", 14)),
                href=f"/calibrate/{job.session_id}",
            ),
        )
    if session:
        actions.append(
            dmc.Anchor(
                dmc.Button("Open session", size="xs", variant="default"),
                href=f"/session/{job.session_id}",
            )
        )

    stage_rows = [
        dmc.Group(
            [
                dmc.Text(titles.get(s["stage"], s["stage"]), size="sm", w=150),
                dmc.Progress(
                    value=100 * (s["progress"] or 0),
                    flex=1,
                    size="sm",
                    color="gray" if s["status"] == "skipped" else "teal",
                    animated=s["status"] == "running",
                ),
                dmc.Box(status_badge(s["status"], "xs"), w=96),
                dmc.Text(
                    "" if s["status"] == "skipped" else (s["message"] or ""),
                    size="xs",
                    c="dimmed",
                    w=220,
                    truncate="end",
                ),
            ],
            gap="sm",
            wrap="nowrap",
        )
        for s in stages
    ]
    meta = [f"Job #{job.id}", f"queued {fmt_time(job.created_at)}"]
    if job.finished_at:
        meta.append(f"finished {fmt_time(job.finished_at)}")
    if eta:
        meta.append(f"ETA {eta}")
    if job.cancel_requested and active:
        meta.append("cancelling…")

    children = [
        dmc.Group(
            [
                dmc.Group([dmc.Text(name, fw=600), status_badge(job.status)], gap="sm"),
                dmc.Group(actions, gap="xs"),
            ],
            justify="space-between",
        ),
        dmc.Text(" · ".join(meta), size="xs", c="dimmed"),
        dmc.Progress(value=100 * job.progress, size="lg", mt="xs", animated=job.status == RUNNING),
        dmc.Stack(stage_rows, gap=4, mt="sm") if stage_rows else None,
    ]
    if job.status in ("failed", "needs_action") and job.message:
        children.append(
            dmc.Alert(
                job.message, color="red" if job.status == "failed" else "grape", mt="sm", p="xs"
            )
        )
    if show_logs:
        children.append(
            dmc.Code(
                _log_tail(lib, job),
                block=True,
                mt="sm",
                style={"whiteSpace": "pre-wrap", "fontSize": 11},
            )
        )
    return dmc.Paper(children, p="md", withBorder=True)


@callback(
    Output("jobs-list", "children"),
    Input("jobs-poll", "n_intervals"),
    Input("jobs-filter", "value"),
    Input("jobs-logs", "checked"),
)
def _render(_, filt, show_logs):
    lib = state.library()
    if lib is None:
        return no_output_root_alert()
    jobs = lib.list_jobs(40)
    if filt == "active":
        jobs = [j for j in jobs if j.status in ("queued", "running")]
    if not jobs:
        return dmc.Paper(
            dmc.Stack(
                [dmc.Text("No jobs.", c="dimmed"), dmc.Anchor("Create a session", href="/new")],
                gap=4,
                align="center",
            ),
            p="xl",
            withBorder=True,
        )
    sessions = {s["id"]: s for s in lib.list_sessions()}
    titles = {s.name: s.title or s.name for s in default_registry().for_mode("practice")}
    titles.update({s.name: s.title or s.name for s in default_registry().for_mode("match")})
    return dmc.Stack([_job_card(lib, j, sessions, titles, bool(show_logs)) for j in jobs], gap="sm")


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input({"type": "job-cancel", "index": ALL}, "n_clicks"),
    Input({"type": "job-retry", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _actions(_cancel, _retry):
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not ctx.triggered[0]["value"]:
        return no_update
    lib = state.library()
    if lib is None:
        return no_update
    job = lib.get_job(trig["index"])
    if job is None:
        return no_update
    if trig["type"] == "job-cancel":
        lib.request_cancel(job.id)
        return notification(f"Cancelling job #{job.id}.", color="orange")
    try:
        new_id = lib.enqueue_job(job.session_id, job.targets, job.force)
    except ValueError as exc:
        return notification(str(exc), color="red")
    if state.OPTIONS.start_worker:
        ensure_worker(lib.root)
    return notification(f"Re-queued as job #{new_id}.")
