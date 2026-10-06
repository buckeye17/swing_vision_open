"""Server-side file/folder browser modal.

A browser upload can't reveal a local path, and the CV pipeline needs the
original file in place, so the app lists the local filesystem itself. The app
binds to 127.0.0.1 only.

Usage on a page::

    register_file_browser("x", mode="file", extensions=(".mp4",))  # module level, at import
    layout = [..., dmc.Button("Browse", id="x-open"), file_browser("x", mode="file")]
    @callback(Output(...), Input("x-result", "data"))  # {"path": "..."} after Select

Callbacks must exist before the app serves its first request, so they are
registered at import time rather than when the layout is built.
"""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from swingvision.app.components.ui import fmt_bytes, icon

MAX_ENTRIES = 2000
_REGISTERED: set[str] = set()


@dataclass
class Entry:
    name: str
    path: str
    is_dir: bool
    size: int | None
    modified: float | None


def list_roots() -> list[str]:
    if hasattr(os, "listdrives"):
        try:
            return list(os.listdrives())
        except OSError:
            pass
    if os.name == "nt":
        return [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
    return ["/"]


def quick_places() -> list[dict[str, str]]:
    home = Path.home()
    places = [{"value": str(home), "label": "Home"}]
    for name in ("Videos", "Movies", "Downloads", "Desktop", "Documents"):
        p = home / name
        if p.is_dir():
            places.append({"value": str(p), "label": name})
    places += [{"value": r, "label": r} for r in list_roots()]
    return places


def list_dir(
    path: str, mode: str, extensions: tuple[str, ...] = ()
) -> tuple[list[Entry], str | None]:
    """Directory listing (folders first). Returns (entries, error message)."""
    p = Path(path)
    try:
        scanned = list(os.scandir(p))
    except OSError as exc:
        return [], f"Cannot open {p}: {exc.strerror or exc}"
    dirs, files = [], []
    for e in scanned:
        if e.name.startswith((".", "$")) or e.name in ("System Volume Information",):
            continue
        try:
            is_dir = e.is_dir()
            st = e.stat() if not is_dir else None
        except OSError:
            continue
        if is_dir:
            dirs.append(Entry(e.name, e.path, True, None, None))
        elif mode == "file" and (not extensions or e.name.lower().endswith(extensions)):
            files.append(
                Entry(
                    e.name, e.path, False, st.st_size if st else None, st.st_mtime if st else None
                )
            )
    dirs.sort(key=lambda x: x.name.lower())
    files.sort(key=lambda x: x.name.lower())
    entries = (dirs + files)[:MAX_ENTRIES]
    return entries, None


def _breadcrumb_paths(path: str) -> list[tuple[str, str]]:
    p = Path(path)
    parts = [p, *p.parents]
    parts.reverse()
    return [(str(q), q.name or str(q)) for q in parts]


def register_file_browser(prefix: str, mode: str = "file", extensions: tuple[str, ...] = ()):
    """Register the modal's callbacks (once per prefix). Call at module import time."""
    if prefix not in _REGISTERED:
        _register(prefix, mode, extensions)
        _REGISTERED.add(prefix)


def file_browser(prefix: str, mode: str = "file", title: str = "Choose a file") -> html.Div:
    """Modal layout. ``register_file_browser`` must have been called for ``prefix``."""
    if prefix not in _REGISTERED:
        raise RuntimeError(f"register_file_browser({prefix!r}) was not called at import time")
    select_label = "Select this folder" if mode == "folder" else "Select file"
    return html.Div(
        [
            dcc.Store(id=f"{prefix}-cwd"),
            dcc.Store(id=f"{prefix}-entries"),
            dcc.Store(id=f"{prefix}-selected"),
            dcc.Store(id=f"{prefix}-result"),
            dcc.Store(id=f"{prefix}-start"),  # page may set an initial directory
            dcc.Store(id=f"{prefix}-last", storage_type="local"),
            dmc.Modal(
                id=f"{prefix}-modal",
                title=title,
                size="xl",
                opened=False,
                children=dmc.Stack(
                    [
                        dmc.Group(
                            [
                                dmc.ActionIcon(
                                    icon("tabler:arrow-up"),
                                    id=f"{prefix}-up",
                                    variant="default",
                                    size="lg",
                                ),
                                dmc.TextInput(
                                    id=f"{prefix}-path",
                                    flex=1,
                                    placeholder="Type a path and press Enter",
                                ),
                                dmc.Select(
                                    id=f"{prefix}-places",
                                    data=quick_places(),
                                    w=170,
                                    placeholder="Go to…",
                                    clearable=False,
                                    allowDeselect=False,
                                    searchable=False,
                                ),
                            ],
                            gap="xs",
                        ),
                        html.Div(id=f"{prefix}-crumbs"),
                        dmc.Text(id=f"{prefix}-error", c="red", size="sm"),
                        dmc.ScrollArea(
                            html.Div(id=f"{prefix}-list"), h=420, type="auto", offsetScrollbars=True
                        ),
                        dmc.Group(
                            [
                                dmc.Text(
                                    id=f"{prefix}-selected-label",
                                    size="sm",
                                    c="dimmed",
                                    truncate="start",
                                    maw=520,
                                ),
                                dmc.Group(
                                    [
                                        dmc.Button(
                                            "Cancel", id=f"{prefix}-cancel", variant="default"
                                        ),
                                        dmc.Button(
                                            select_label,
                                            id=f"{prefix}-select",
                                            disabled=mode == "file",
                                        ),
                                    ],
                                    gap="xs",
                                ),
                            ],
                            justify="space-between",
                            wrap="nowrap",
                        ),
                    ],
                    gap="sm",
                ),
            ),
        ]
    )


def _render_list(entries: list[Entry], selected: str | None, prefix: str):
    if not entries:
        return dmc.Text("Empty folder", c="dimmed", size="sm", p="md")
    rows = []
    for i, e in enumerate(entries):
        is_sel = selected is not None and e.path == selected
        rows.append(
            dmc.UnstyledButton(
                dmc.Group(
                    [
                        icon(
                            "tabler:folder" if e.is_dir else "tabler:movie",
                            18,
                            color="#e8a33d" if e.is_dir else "#4c8bf5",
                        ),
                        dmc.Text(e.name, size="sm", flex=1, truncate="end"),
                        dmc.Text(fmt_bytes(e.size), size="xs", c="dimmed", w=80, ta="right"),
                        dmc.Text(
                            datetime.fromtimestamp(e.modified).strftime("%Y-%m-%d %H:%M")
                            if e.modified
                            else "",
                            size="xs",
                            c="dimmed",
                            w=120,
                            ta="right",
                        ),
                    ],
                    gap="sm",
                    wrap="nowrap",
                ),
                id={"type": f"{prefix}-entry", "index": i},
                className="sv-fb-row" + (" sv-fb-row-selected" if is_sel else ""),
                w="100%",
            )
        )
    return dmc.Stack(rows, gap=0)


def render_listing(prefix: str, mode: str, exts: tuple[str, ...], cwd: str, selected: str | None):
    """The modal's contents for folder ``cwd``: list, entries, breadcrumbs, path box, error,
    selection label, select-button disabled, places reset."""
    entries, error = list_dir(cwd, mode, exts)
    crumbs = dmc.Breadcrumbs(
        [
            # A plain span: dmc.Anchor requires an href (and would navigate).
            html.Span(
                label,
                id={"type": f"{prefix}-crumb", "index": i},
                style={
                    "cursor": "pointer",
                    "fontSize": "var(--mantine-font-size-sm)",
                    "color": "var(--mantine-color-anchor)",
                },
            )
            for i, (_, label) in enumerate(_breadcrumb_paths(cwd))
        ],
        separator="›",
    )
    label = selected if mode == "file" else cwd
    can_select = bool(selected) if mode == "file" else error is None
    return (
        _render_list(entries, selected, prefix),
        [e.__dict__ for e in entries],
        crumbs,
        cwd,
        error or "",
        label or "No file selected",
        not can_select,
        None,
    )


def _register(prefix: str, mode: str, extensions: tuple[str, ...]) -> None:
    exts = tuple(e.lower() for e in extensions)

    @callback(
        Output(f"{prefix}-modal", "opened", allow_duplicate=True),
        Output(f"{prefix}-cwd", "data", allow_duplicate=True),
        Output(f"{prefix}-selected", "data", allow_duplicate=True),
        Input(f"{prefix}-open", "n_clicks"),
        State(f"{prefix}-start", "data"),
        State(f"{prefix}-last", "data"),
        prevent_initial_call=True,
    )
    def _open(n, start, last):
        if not n:
            return no_update, no_update, no_update
        for candidate in (start, last, str(Path.home())):
            if candidate and Path(candidate).is_dir():
                return True, candidate, None
            if candidate and Path(candidate).parent.is_dir():
                return True, str(Path(candidate).parent), None
        return True, list_roots()[0], None

    @callback(
        Output(f"{prefix}-cwd", "data", allow_duplicate=True),
        Output(f"{prefix}-selected", "data", allow_duplicate=True),
        Input({"type": f"{prefix}-entry", "index": ALL}, "n_clicks"),
        Input({"type": f"{prefix}-crumb", "index": ALL}, "n_clicks"),
        Input(f"{prefix}-up", "n_clicks"),
        Input(f"{prefix}-path", "n_submit"),
        Input(f"{prefix}-places", "value"),
        State(f"{prefix}-path", "value"),
        State(f"{prefix}-cwd", "data"),
        State(f"{prefix}-entries", "data"),
        prevent_initial_call=True,
    )
    def _navigate(_entry_clicks, _crumb_clicks, up, _submit, place, typed, cwd, entries):
        trig = ctx.triggered_id
        value = ctx.triggered[0]["value"] if ctx.triggered else None
        if trig is None or (isinstance(trig, dict) and not value):
            return no_update, no_update
        if trig == f"{prefix}-up":
            if not cwd:
                return no_update, no_update
            parent = Path(cwd).parent
            return (str(parent) if str(parent) != cwd else no_update), None
        if trig == f"{prefix}-path":
            p = Path(typed or "")
            if typed and p.is_dir():
                return str(p), None
            if typed and p.is_file() and mode == "file":
                return str(p.parent), str(p)
            return no_update, no_update
        if trig == f"{prefix}-places":
            return (place, None) if place else (no_update, no_update)
        if isinstance(trig, dict) and trig["type"] == f"{prefix}-crumb":
            crumbs = _breadcrumb_paths(cwd) if cwd else []
            idx = trig["index"]
            return (crumbs[idx][0], None) if idx < len(crumbs) else (no_update, no_update)
        if isinstance(trig, dict) and entries:
            e = entries[trig["index"]]
            if e["is_dir"]:
                return e["path"], None
            return no_update, e["path"]
        return no_update, no_update

    @callback(
        Output(f"{prefix}-list", "children"),
        Output(f"{prefix}-entries", "data"),
        Output(f"{prefix}-crumbs", "children"),
        Output(f"{prefix}-path", "value"),
        Output(f"{prefix}-error", "children"),
        Output(f"{prefix}-selected-label", "children"),
        Output(f"{prefix}-select", "disabled"),
        Output(f"{prefix}-places", "value"),
        Input(f"{prefix}-cwd", "data"),
        Input(f"{prefix}-selected", "data"),
        prevent_initial_call=True,
    )
    def _render(cwd, selected):
        if not cwd:
            return (no_update,) * 8
        return render_listing(prefix, mode, exts, cwd, selected)

    @callback(
        Output(f"{prefix}-result", "data"),
        Output(f"{prefix}-modal", "opened", allow_duplicate=True),
        Output(f"{prefix}-last", "data"),
        Input(f"{prefix}-select", "n_clicks"),
        State(f"{prefix}-cwd", "data"),
        State(f"{prefix}-selected", "data"),
        prevent_initial_call=True,
    )
    def _select(n, cwd, selected):
        if not n:
            return no_update, no_update, no_update
        path = selected if mode == "file" else cwd
        if not path:
            return no_update, no_update, no_update
        return {"path": path}, False, cwd

    @callback(
        Output(f"{prefix}-modal", "opened", allow_duplicate=True),
        Input(f"{prefix}-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def _cancel(n):
        return False if n else no_update
