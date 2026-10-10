"""Dash app: Mantine AppShell + pages. Run with ``sv app``."""

from __future__ import annotations

import functools
import importlib
import threading

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc, html
from loguru import logger

from swingvision.app import state, units
from swingvision.app.components.ui import error_page, icon
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
    ("Stats", "/stats", "tabler:chart-bar", "exact"),
    ("New session", "/new", "tabler:video-plus", "exact"),
    ("Jobs", "/jobs", "tabler:list-check", "exact"),
    ("Profiles", "/profiles", "tabler:users", "exact"),
    ("Labeling", "/labeling", "tabler:target", "exact"),
    ("Settings", "/settings", "tabler:settings", "exact"),
]

FONTS_CSS = (
    "https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700"
    "&family=Barlow+Condensed:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
)

#: Optic yellow: the one colour for what you act on (primary buttons, active tab, cursor).
BALL = [
    "#fafde6",
    "#f2fac6",
    "#e8f59e",
    "#def277",
    "#d7f25a",
    "#c3de45",
    "#a6bf32",
    "#7f9324",
    "#5d6c19",
    "#3c460f",
]
#: Graphite, slightly green. Mantine reads dark[0] as text, [2] dimmed, [4] borders,
#: [5] hover, [6] inputs, [7] the page.
GRAPHITE = [
    "#e9edeb",
    "#b9c2be",
    "#9ba5a0",
    "#6e7873",
    "#2e3633",
    "#252d2a",
    "#1c2220",
    "#0e1110",
    "#0a0c0b",
    "#070908",
]

THEME = {
    "colors": {"ball": BALL, "dark": GRAPHITE},
    "primaryColor": "ball",
    "primaryShade": {"light": 7, "dark": 4},
    "autoContrast": True,
    "luminanceThreshold": 0.45,
    "fontFamily": "Barlow, system-ui, -apple-system, Segoe UI, sans-serif",
    "fontFamilyMonospace": "'JetBrains Mono', ui-monospace, Consolas, monospace",
    "headings": {
        "fontFamily": "'Barlow Condensed', Barlow, system-ui, sans-serif",
        "fontWeight": "650",
        "sizes": {
            "h1": {"fontSize": "34px", "lineHeight": "1.05"},
            "h2": {"fontSize": "30px", "lineHeight": "1.1"},
            "h3": {"fontSize": "22px", "lineHeight": "1.2"},
            "h4": {"fontSize": "20px", "lineHeight": "1.2"},
            "h5": {"fontSize": "18px", "lineHeight": "1.25"},
            "h6": {"fontSize": "15px", "lineHeight": "1.3"},
        },
    },
    "defaultRadius": "md",
    "radius": {"md": "8px", "lg": "12px"},
    "components": {
        "Paper": {"defaultProps": {"radius": "lg"}},
        "Card": {"defaultProps": {"radius": "lg"}},
        "Badge": {"defaultProps": {"radius": "xl", "tt": "none"}},
    },
}


def _rail_logo():
    return dmc.Anchor(
        dmc.Group(
            [
                icon("tabler:ball-tennis", 26, color="var(--mantine-primary-color-filled)"),
                html.Span(["Swing Vision ", html.Span("Open", className="sv-dim")]),
            ],
            gap=10,
            wrap="nowrap",
        ),
        href="/",
        underline="never",
        className="sv-logo",
    )


def _layout():
    # The header only carries the menu button on phones; on wider screens the rail has it all.
    header = dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Burger(id="nav-burger", size="sm", opened=False),
                _rail_logo(),
            ],
            h="100%",
            px="md",
            gap="sm",
        ),
        hiddenFrom="sm",
    )
    navbar = dmc.AppShellNavbar(
        [
            dmc.AppShellSection(_rail_logo(), visibleFrom="sm", mb="lg", px=6),
            dmc.AppShellSection(
                [
                    dmc.NavLink(
                        label=label,
                        href=href,
                        leftSection=icon(ic),
                        active=match,
                        className="sv-navlink",
                    )
                    for label, href, ic, match in NAV
                ],
                grow=True,
            ),
            dmc.AppShellSection(
                dmc.Stack(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.Span(className="sv-dot"), "Worker"],
                                    id="worker-label",
                                    className="sv-worker-label",
                                ),
                                html.Div(id="worker-detail", className="sv-worker-detail"),
                            ],
                            id="worker-card",
                            className="sv-worker",
                        ),
                        dmc.Group(
                            [
                                dmc.Anchor(
                                    id="units-label",
                                    href="/settings",
                                    size="xs",
                                    c="dimmed",
                                    underline="hover",
                                ),
                                dmc.ColorSchemeToggle(
                                    id="color-scheme-toggle",
                                    lightIcon=icon("tabler:sun", 18),
                                    darkIcon=icon("tabler:moon", 18),
                                    variant="default",
                                    size="lg",
                                    **{"aria-label": "Switch light or dark theme"},
                                ),
                            ],
                            justify="space-between",
                            px=4,
                        ),
                    ],
                    gap="sm",
                )
            ),
        ],
        p="sm",
        pt="lg",
    )
    return dmc.MantineProvider(
        theme=THEME,
        defaultColorScheme="dark",
        children=[
            dcc.Location(id="url", refresh="callback-nav"),
            dmc.NotificationContainer(id="notify", position="top-right"),
            dcc.Interval(id="worker-poll", interval=3000),
            dmc.AppShell(
                [header, navbar, dmc.AppShellMain(dash.page_container)],
                id="appshell",
                header={"height": {"base": 56, "sm": 0}},
                navbar={"width": 216, "breakpoint": "sm", "collapsed": {"mobile": True}},
                padding="xl",
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
    Output("worker-label", "children"),
    Output("worker-label", "data-status"),
    Output("worker-detail", "children"),
    Output("units-label", "children"),
    Input("worker-poll", "n_intervals"),
)
def _worker_status(_):
    u = units.current()
    unit_text = f"Units: {u.speed_unit} · {u.len_unit}"
    lib = state.library()
    if lib is None:
        return [html.Span(className="sv-dot"), "No output folder"], "stopped", "", unit_text
    status, label = worker_state(lib)
    detail = {"idle": "Ready for new jobs", "busy": "Processing a job", "stopped": ""}[status]
    if status == "stopped" and state.OPTIONS.start_worker:
        ensure_worker(lib.root)  # also revives a crashed worker
        status, label, detail = "busy", "Worker starting", ""
    return [html.Span(className="sv-dot"), label], status, detail, unit_text


def _safe_layout(layout, module: str):
    """A page that fails to build (a damaged or half-written file, say) explains itself
    instead of leaving the screen blank; the traceback goes to the log."""

    @functools.wraps(layout)
    def safe(**kwargs):
        try:
            return layout(**kwargs)
        except Exception as exc:
            logger.exception("Page {} failed to render ({})", module, kwargs)
            return error_page(exc)

    safe._safe = True  # type: ignore[attr-defined]
    return safe


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder="",
        title="Swing Vision Open",
        suppress_callback_exceptions=True,
        update_title=None,
        external_stylesheets=[FONTS_CSS],
    )
    for mod in PAGE_MODULES:
        importlib.import_module(f"swingvision.app.pages.{mod}")
    for page in dash.page_registry.values():
        if callable(page.get("layout")) and not getattr(page["layout"], "_safe", False):
            page["layout"] = _safe_layout(page["layout"], page["module"])
    app.layout = _layout()
    register_routes(app.server)
    return app


def _backfill_devices(settings) -> None:
    """Sessions made before M7c get their recording device (a quick re-probe each)."""
    from swingvision import services

    try:
        done = services.backfill_devices(settings)
    except Exception as exc:  # never keep the app from starting
        logger.warning("Device backfill failed: {}", exc)
        return
    if done:
        logger.info("Recording device found for {} sessions", len(done))


def serve(port: int | None = None, start_worker: bool = True, debug: bool = False) -> None:
    settings = state.settings()
    state.OPTIONS.start_worker = start_worker
    app = create_app()
    if settings.output_root is not None:
        threading.Thread(target=_backfill_devices, args=(settings,), daemon=True).start()
    if start_worker:
        ensure_worker(settings.output_root)
    port = port or settings.port
    logger.info("Swing Vision Open on http://{}:{}", settings.host, port)
    app.run(host=settings.host, port=port, debug=debug, use_reloader=False, threaded=True)
