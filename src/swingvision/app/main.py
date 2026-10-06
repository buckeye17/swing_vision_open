"""Dash app: Mantine AppShell + pages. Run with ``sv app``."""

from __future__ import annotations

import importlib

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc
from loguru import logger

from swingvision.app import state
from swingvision.app.components.ui import STATUS_COLORS, icon
from swingvision.app.server_routes import register_routes
from swingvision.app.worker_control import ensure_worker, worker_state

PAGE_MODULES = [
    "library",
    "new_session",
    "jobs",
    "session_review",
    "practice",
    "swings",
    "stats",
    "calibrate",
    "profiles",
    "labeling",
    "settings_page",
]

NAV = [
    ("Library", "/", "tabler:books", "exact"),
    ("New session", "/new", "tabler:video-plus", "exact"),
    ("Jobs", "/jobs", "tabler:list-check", "exact"),
    ("Profiles", "/profiles", "tabler:users", "exact"),
    ("Labeling", "/labeling", "tabler:target", "exact"),
    ("Settings", "/settings", "tabler:settings", "exact"),
]

THEME = {
    "primaryColor": "teal",
    "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    "defaultRadius": "md",
}


def _layout():
    header = dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Group(
                    [
                        dmc.Burger(id="nav-burger", size="sm", hiddenFrom="sm", opened=False),
                        icon("tabler:ball-tennis", 26, color="#9acd32"),
                        dmc.Title("Swing Vision Open", order=4),
                    ],
                    gap="xs",
                ),
                dmc.Group(
                    [
                        dmc.Badge("Worker", id="worker-badge", variant="dot", color="gray"),
                        dmc.ColorSchemeToggle(
                            id="color-scheme-toggle",
                            lightIcon=icon("tabler:sun", 18),
                            darkIcon=icon("tabler:moon", 18),
                            variant="default",
                            size="lg",
                        ),
                    ],
                    gap="sm",
                ),
            ],
            justify="space-between",
            h="100%",
            px="md",
        )
    )
    navbar = dmc.AppShellNavbar(
        [
            dmc.NavLink(label=label, href=href, leftSection=icon(ic), active=match)
            for label, href, ic, match in NAV
        ],
        p="sm",
    )
    return dmc.MantineProvider(
        theme=THEME,
        defaultColorScheme="auto",
        children=[
            dcc.Location(id="url", refresh="callback-nav"),
            dmc.NotificationContainer(id="notify", position="top-right"),
            dcc.Interval(id="worker-poll", interval=3000),
            dmc.AppShell(
                [header, navbar, dmc.AppShellMain(dash.page_container)],
                id="appshell",
                header={"height": 56},
                navbar={"width": 220, "breakpoint": "sm", "collapsed": {"mobile": True}},
                padding="lg",
            ),
        ],
    )


@callback(
    Output("appshell", "navbar"),
    Input("nav-burger", "opened"),
    State("appshell", "navbar"),
)
def _toggle_nav(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar


@callback(
    Output("worker-badge", "children"),
    Output("worker-badge", "color"),
    Input("worker-poll", "n_intervals"),
)
def _worker_badge(_):
    lib = state.library()
    if lib is None:
        return "No output folder", "gray"
    status, label = worker_state(lib)
    if status == "stopped" and state.OPTIONS.start_worker:
        ensure_worker(lib.root)  # also revives a crashed worker
        return "Worker starting", "blue"
    return label, STATUS_COLORS[status]


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder="",
        title="Swing Vision Open",
        suppress_callback_exceptions=True,
        update_title=None,
    )
    for mod in PAGE_MODULES:
        importlib.import_module(f"swingvision.app.pages.{mod}")
    app.layout = _layout()
    register_routes(app.server)
    return app


def serve(port: int | None = None, start_worker: bool = True, debug: bool = False) -> None:
    settings = state.settings()
    state.OPTIONS.start_worker = start_worker
    app = create_app()
    if start_worker:
        ensure_worker(settings.output_root)
    port = port or settings.port
    logger.info("Swing Vision Open on http://{}:{}", settings.host, port)
    app.run(host=settings.host, port=port, debug=debug, use_reloader=False, threaded=True)
