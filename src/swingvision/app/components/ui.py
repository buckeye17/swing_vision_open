"""Small shared UI helpers and formatters."""

from __future__ import annotations

from datetime import datetime

import dash_mantine_components as dmc
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
            dmc.Title(title, order=2),
            dmc.Text(subtitle, c="dimmed", size="sm") if subtitle else None,
        ],
        gap=2,
    )
    return dmc.Group([left, right], justify="space-between", align="flex-end", mb="md")


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
