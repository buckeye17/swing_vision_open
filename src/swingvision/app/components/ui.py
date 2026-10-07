"""Small shared UI helpers and formatters."""

from __future__ import annotations

import re
from datetime import datetime

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

STATUS_COLORS = {
    "new": "gray",
    "queued": "blue",
    "processing": "yellow",
    "running": "yellow",
    "ready": "green",
    "done": "green",
    "skipped": "gray",
    "pending": "gray",
    "failed": "red",
    "cancelled": "orange",
    "needs_action": "grape",
    "idle": "green",
    "busy": "yellow",
    "stopped": "red",
}

STATUS_LABELS = {"needs_action": "needs action", "skipped": "up to date"}


def icon(name: str, size: int = 18, **kwargs) -> DashIconify:
    return DashIconify(icon=name, width=size, **kwargs)


def status_badge(status: str, size: str = "sm") -> dmc.Badge:
    return dmc.Badge(
        STATUS_LABELS.get(status, status),
        color=STATUS_COLORS.get(status, "gray"),
        variant="light",
        size=size,
    )


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    s = round(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return ""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return ""


def fmt_time(iso: str | None) -> str:
    if not iso:
        return "–"
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


def notification(
    message: str, title: str | None = None, color: str = "green", icon_name: str | None = None
) -> list[dict]:
    """Payload for ``NotificationContainer.sendNotifications``."""
    n: dict = {"action": "show", "message": message, "color": color, "autoClose": 5000}
    if title:
        n["title"] = title
    if icon_name:
        n["icon"] = icon(icon_name)
    return [n]


def page_header(title: str, subtitle: str | None = None, right=None) -> dmc.Group:
    left = dmc.Stack(
        [
            dmc.Title(title, order=1),
            dmc.Text(subtitle, c="dimmed", size="sm") if subtitle else None,
        ],
        gap=4,
    )
    return dmc.Group([left, right], justify="space-between", align="flex-end", mb="lg")


#: "88 mph", "± 15.3 ft", "2,272 ft" → number and unit; "13%" and "–" stay whole.
_VALUE_UNIT = re.compile(r"^(.*\d)\s+([^\d\s]+)$")


def stat_tile(label: str, value: str, sub: str | None = None) -> html.Div:
    """One measurement: label, the value in big numerals with its unit set small, a note."""
    m = _VALUE_UNIT.match(value or "")
    shown = [m.group(1), html.Span(m.group(2), className="sv-stat-unit")] if m else value
    return html.Div(
        [
            html.Div(label, className="sv-stat-label"),
            html.Div(shown, className="sv-stat-value"),
            html.Div(sub or "", className="sv-stat-sub"),
        ],
        className="sv-stat",
    )


def stat_grid(tiles: list, plain: bool = False) -> html.Div:
    """Tiles as one bordered strip (a page's headline numbers), or ``plain`` inside a card."""
    return html.Div(tiles, className="sv-stats-plain" if plain else "sv-stats")


def key_hints(hints: list[tuple[list[str], str]]) -> html.Div:
    """Keyboard shortcuts: ``[(["J", "L"], "±5 s"), …]``."""
    return html.Div(
        [html.Span([*(dmc.Kbd(k, mr=3) for k in keys), " ", label]) for keys, label in hints],
        className="sv-keys",
    )


def session_meta(config) -> list[str]:
    """Mode, length and format of a session, for its header."""
    from swingvision.storage.schemas import PRACTICE_SUBMODE_LABELS

    mode = config.mode.capitalize()
    if config.practice:
        mode += f" · {PRACTICE_SUBMODE_LABELS[config.practice.submode]}"
    parts = [mode]
    if config.video:
        v = config.video
        parts += [
            fmt_duration(v.duration_s),
            f"{v.display_width} × {v.display_height} · {v.fps_avg:.2f} fps",
        ]
    return parts


def export_menu(session, config) -> dmc.Menu:
    """Download the session's shots, practice shots or swings as CSV or Parquet."""
    from swingvision.analysis import export as ex

    available = {
        "shots": session.shots_path.exists(),
        "practice": session.practice_path.exists(),
        "swings": session.swings_path.exists(),
    }
    items = [
        dmc.MenuItem(
            f"{label} ({fmt.upper() if fmt == 'csv' else 'Parquet'})",
            href=f"/export/{config.id}/{what}.{fmt}",
            refresh=True,
            disabled=not available[what],
            leftSection=icon("tabler:file-spreadsheet" if fmt == "csv" else "tabler:database", 14),
        )
        for what, label in ex.EXPORTS.items()
        for fmt in ex.FORMATS
    ]
    return dmc.Menu(
        [
            dmc.MenuTarget(
                dmc.Button(
                    "Export",
                    variant="default",
                    disabled=not any(available.values()),
                    leftSection=icon("tabler:download", 16),
                )
            ),
            dmc.MenuDropdown(items),
        ],
        position="bottom-end",
    )


def session_header(session, config, active: str, status: str | None = None, right=None):
    """The header every session page shares: where you are, the session, and its views as
    tabs (Overview, Practice, Swings, Stats, Calibration) where they have something to show.
    """
    sid = config.id
    tabs = [("overview", "Overview", f"/session/{sid}")]
    if config.practice:
        tabs.append(("practice", "Practice", f"/practice/{sid}"))
    if session.swings_path.exists():
        tabs.append(("swings", "Swings", f"/swings/{sid}"))
    if session.shots_path.exists() or session.movement_path.exists():
        tabs.append(("stats", "Stats", f"/stats/{sid}"))
    tabs.append(("calibration", "Calibration", f"/calibrate/{sid}"))
    title = dmc.Stack(
        [
            html.Div([dmc.Anchor("Library", href="/"), " / Session"], className="sv-crumb"),
            dmc.Group(
                [dmc.Title(config.name, order=1), status_badge(status, "md") if status else None],
                gap="sm",
            ),
            html.Div([html.Span(p) for p in session_meta(config)], className="sv-meta"),
        ],
        gap=6,
    )
    return html.Div(
        [
            dmc.Group([title, right], justify="space-between", align="flex-start", mb="sm"),
            html.Nav(
                [
                    dmc.Anchor(
                        label,
                        href=href,
                        className="sv-tab",
                        underline="never",
                        **(
                            {"data-active": "true", "aria-current": "page"} if key == active else {}
                        ),
                    )
                    for key, label, href in tabs
                ],
                className="sv-tabs",
                **{"aria-label": "Session views"},
            ),
        ]
    )


def no_output_root_alert() -> dmc.Alert:
    return dmc.Alert(
        [
            dmc.Text("Choose an output folder first. All session data is stored there."),
            dmc.Anchor("Open Settings", href="/settings"),
        ],
        title="No output folder",
        color="yellow",
        icon=icon("tabler:folder-question"),
    )


def error_page(exc: BaseException) -> dmc.Container:
    """Shown instead of a page that failed to build."""
    return dmc.Container(
        [
            page_header("This page couldn't be shown"),
            dmc.Alert(
                [
                    dmc.Text(f"{type(exc).__name__}: {exc}", size="sm", ff="monospace"),
                    dmc.Text(
                        "A file this page reads may be missing or damaged, for example while the "
                        "session is being reprocessed. Try again in a moment, or reprocess the "
                        "session from the Library. The full error is in the app's log.",
                        size="sm",
                        mt="xs",
                    ),
                ],
                color="red",
                icon=icon("tabler:alert-triangle"),
            ),
            dmc.Anchor("Back to the library", href="/", mt="md"),
        ],
        size="xl",
        px=0,
    )
