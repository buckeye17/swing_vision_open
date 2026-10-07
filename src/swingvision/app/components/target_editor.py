"""Practice target editor: draw rectangles and circles on a court, name them, pick a frame
and a stroke filter, add presets, save and load target sets (PLAN.md §7.10, §9.1).

Targets live in a ``dcc.Store`` (``<prefix>-targets``, a list of :class:`Target` dicts); the
court's editable shapes mirror it. The court is drawn as the hitter sees it: they stand at
the bottom (near) end and hit up into the far half. ``relative`` targets follow the hitter
when they serve or hit from the other end; ``absolute`` ones stay where they are drawn.

Use :func:`target_editor` in a layout and call :func:`register_target_editor` once per
prefix at import time.
"""

from __future__ import annotations

import secrets

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.analysis import practice as pr
from swingvision.app import state, units
from swingvision.app.components.court_diagram import ME_COLOR, court_figure
from swingvision.app.components.ui import icon, notification
from swingvision.court import model as cm
from swingvision.storage.schemas import Target

TARGET_COLOR = "#ffd43b"
ABSOLUTE_COLOR = "#74c0fc"

_HS, _SB, _HL = cm.HALF_SINGLES, cm.SERVICE_LINE_FROM_NET, cm.HALF_LENGTH
#: Presets in the hitter's frame (hitter at the near end, hitting into the far half). The
#: far receiver faces the camera, so their deuce court is at negative x.
PRESETS: dict[str, dict] = {
    "Deuce box": {"x0": -_HS, "y0": 0.0, "x1": 0.0, "y1": _SB, "strokes": ["serve"]},
    "Ad box": {"x0": 0.0, "y0": 0.0, "x1": _HS, "y1": _SB, "strokes": ["serve"]},
    "Deuce T": {"x0": -1.2, "y0": _SB - 2.0, "x1": 0.0, "y1": _SB, "strokes": ["serve"]},
    "Deuce wide": {"x0": -_HS, "y0": _SB - 2.0, "x1": -_HS + 1.2, "y1": _SB, "strokes": ["serve"]},
    "Ad T": {"x0": 0.0, "y0": _SB - 2.0, "x1": 1.2, "y1": _SB, "strokes": ["serve"]},
    "Ad wide": {"x0": _HS - 1.2, "y0": _SB - 2.0, "x1": _HS, "y1": _SB, "strokes": ["serve"]},
    "Deep (past the service line)": {"x0": -_HS, "y0": _SB + 1.5, "x1": _HS, "y1": _HL},
    "Deep left corner": {"x0": -_HS, "y0": _HL - 3.0, "x1": -_HS + 2.5, "y1": _HL},
    "Deep right corner": {"x0": _HS - 2.5, "y0": _HL - 3.0, "x1": _HS, "y1": _HL},
}
STROKE_OPTIONS = [{"value": k, "label": v} for k, v in pr.STROKE_LABELS.items()]
#: Shapes the court itself draws before any target (court_figure's surface and lines).
N_COURT_SHAPES = len(court_figure().layout.shapes)


def new_id() -> str:
    return secrets.token_hex(3)


def _valid(targets: list[dict] | None) -> list[Target]:
    out = []
    for d in targets or []:
        try:
            t = pr.normalized(Target.model_validate(d))
        except ValueError:
            continue
        if pr.target_valid(t):
            out.append(t)
    return out


def target_shape(t: Target, editable: bool = True, opacity: float = 0.25) -> dict:
    color = ABSOLUTE_COLOR if t.frame == "absolute" else TARGET_COLOR
    base = {
        "name": t.id,
        "editable": editable,
        "layer": "above",
        "line": {"color": color, "width": 2},
        "fillcolor": color,
        "opacity": 1.0,
        "xref": "x",
        "yref": "y",
    }
    rgba = _rgba(color, opacity)
    base["fillcolor"] = rgba
    if t.shape == "circle":
        return {
            **base,
            "type": "circle",
            "x0": t.cx - t.r,  # type: ignore[operator]
            "x1": t.cx + t.r,  # type: ignore[operator]
            "y0": t.cy - t.r,  # type: ignore[operator]
            "y1": t.cy + t.r,  # type: ignore[operator]
        }
    return {**base, "type": "rect", "x0": t.x0, "x1": t.x1, "y0": t.y0, "y1": t.y1}


def _rgba(hex_color: str, a: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


def target_labels(targets: list[Target]) -> go.Scatter:
    """Target names, at each target's center."""
    xs, ys, names = [], [], []
    for t in targets:
        cx, cy = pr.target_center(t)
        xs.append(cx)
        ys.append(cy)
        names.append(t.name)
    return go.Scatter(
        x=xs,
        y=ys,
        mode="text",
        text=names,
        textfont={"color": "white", "size": 11},
        hoverinfo="skip",
        name="target names",
    )


def editor_figure(targets: list[Target], height: int = 520) -> go.Figure:
    fig = court_figure(height=height)
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[-cm.HALF_LENGTH - 1.2],
            mode="markers+text",
            marker={"color": ME_COLOR, "size": 12, "line": {"color": "#222", "width": 1}},
            text=["hitter"],
            textposition="bottom center",
            textfont={"color": "white", "size": 11},
            hoverinfo="skip",
        )
    )
    fig.add_trace(target_labels(targets))
    fig.update_layout(
        shapes=list(fig.layout.shapes) + [target_shape(t) for t in targets],
        dragmode="drawrect",
        newshape={
            "line": {"color": TARGET_COLOR, "width": 2},
            "fillcolor": _rgba(TARGET_COLOR, 0.25),
        },
        # New every render, so the server's list replaces whatever Plotly drew or moved
        # (with a fixed uirevision Plotly would keep user-edited shapes over the figure's).
        uirevision=new_id(),
    )
    return fig


def _graph_config() -> dict:
    return {
        "displaylogo": False,
        "modeBarButtonsToAdd": ["drawrect", "drawcircle", "eraseshape"],
        "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "autoScale2d"],
        "displayModeBar": True,
    }


def target_editor(prefix: str, targets: list[Target] | None = None, saved_sets: bool = True):
    targets = list(targets or [])
    data = [t.model_dump() for t in targets]
    sets = []
    if saved_sets and state.settings().output_root is not None:
        sets = [
            {"value": s.id, "label": s.name} for s in services.list_target_sets(state.settings())
        ]
    preset_menu = dmc.Menu(
        [
            dmc.MenuTarget(
                dmc.Button(
                    "Add preset", variant="light", size="xs", leftSection=icon("tabler:plus", 14)
                )
            ),
            dmc.MenuDropdown(
                [
                    dmc.MenuItem(name, id={"type": f"{prefix}-preset", "index": name}, n_clicks=0)
                    for name in PRESETS
                ]
            ),
        ]
    )
    set_controls = (
        dmc.Group(
            [
                dmc.Select(
                    id=f"{prefix}-set",
                    data=sets,
                    placeholder="Load a saved set…" if sets else "No saved sets yet",
                    clearable=True,
                    size="xs",
                    w=200,
                ),
                dmc.TextInput(id=f"{prefix}-set-name", placeholder="Set name", size="xs", w=160),
                dmc.Button(
                    "Save set",
                    id=f"{prefix}-set-save",
                    size="xs",
                    variant="default",
                    leftSection=icon("tabler:device-floppy", 14),
                ),
                dmc.ActionIcon(
                    icon("tabler:trash", 14),
                    id=f"{prefix}-set-delete",
                    variant="subtle",
                    color="red",
                    size="sm",
                ),
            ],
            gap="xs",
        )
        if saved_sets
        else None
    )
    return dmc.Stack(
        [
            dcc.Store(id=f"{prefix}-targets", data=data),
            dcc.Store(id=f"{prefix}-shapes"),
            dmc.Text(
                "Draw targets on the far half with the rectangle or circle tool (top right of "
                "the court), drag them to adjust, or add a preset. You hit from the bottom end; "
                "relative targets (yellow) follow you when you play from the other end, "
                "absolute ones (blue) stay put.",
                size="xs",
                c="dimmed",
            ),
            dmc.Grid(
                [
                    dmc.GridCol(
                        # assets/shape_events.js forwards shape edits to <prefix>-shapes.
                        html.Div(
                            dcc.Graph(
                                id=f"{prefix}-court",
                                figure=editor_figure(targets),
                                config=_graph_config(),
                            ),
                            **{"data-shapes-store": f"{prefix}-shapes"},
                        ),
                        span={"base": 12, "md": 6},
                    ),
                    dmc.GridCol(
                        dmc.Stack(
                            [
                                dmc.Group([preset_menu, set_controls], gap="sm"),
                                html.Div(
                                    id=f"{prefix}-list", children=target_rows(prefix, targets)
                                ),
                            ],
                            gap="sm",
                        ),
                        span={"base": 12, "md": 6},
                    ),
                ],
                gutter="md",
            ),
        ],
        gap="xs",
    )


def target_rows(prefix: str, targets: list[Target]):
    if not targets:
        return dmc.Text("No targets yet.", size="sm", c="dimmed")
    u = units.current()
    rows = []
    for t in targets:
        cx, cy = pr.target_center(t)
        size = (
            f"⌀ {u.len_str(2 * t.r, 1)}"  # type: ignore[operator]
            if t.shape == "circle"
            else f"{u.len(t.x1 - t.x0):.1f} × {u.len_str(t.y1 - t.y0, 1)}"  # type: ignore[operator]
        )
        rows.append(
            dmc.Paper(
                dmc.Stack(
                    [
                        dmc.Group(
                            [
                                dmc.TextInput(
                                    id={"type": f"{prefix}-tname", "index": t.id},
                                    value=t.name,
                                    size="xs",
                                    flex=1,
                                    debounce=True,
                                ),
                                dmc.SegmentedControl(
                                    id={"type": f"{prefix}-tframe", "index": t.id},
                                    value=t.frame,
                                    data=[
                                        {"value": "relative", "label": "Relative"},
                                        {"value": "absolute", "label": "Absolute"},
                                    ],
                                    size="xs",
                                ),
                                dmc.ActionIcon(
                                    icon("tabler:x", 14),
                                    id={"type": f"{prefix}-tdel", "index": t.id},
                                    variant="subtle",
                                    color="red",
                                    n_clicks=0,
                                ),
                            ],
                            gap="xs",
                            wrap="nowrap",
                        ),
                        dmc.Group(
                            [
                                dmc.MultiSelect(
                                    id={"type": f"{prefix}-tstrokes", "index": t.id},
                                    data=STROKE_OPTIONS,
                                    value=list(t.strokes),
                                    placeholder="All strokes",
                                    size="xs",
                                    clearable=True,
                                    flex=1,
                                ),
                                dmc.Text(
                                    f"{t.shape}, {size}, center "
                                    f"({u.len(cx):.1f}, {u.len(cy):.1f}) {u.len_unit}",
                                    size="xs",
                                    c="dimmed",
                                ),
                            ],
                            gap="xs",
                            wrap="nowrap",
                        ),
                    ],
                    gap=4,
                ),
                p="xs",
                withBorder=True,
            )
        )
    return dmc.Stack(rows, gap="xs")


def apply_relayout(targets: list[dict], relayout: dict) -> list[dict] | None:
    """New target list after a draw/move/erase on the court, or ``None`` if unrelated."""
    if not relayout:
        return None
    by_id = {t["id"]: dict(t) for t in targets}
    order = [t["id"] for t in targets]
    if "shapes" in relayout:
        out = []
        for sh in relayout["shapes"][N_COURT_SHAPES:]:
            tid = sh.get("name") if sh.get("name") in by_id else None
            t = by_id.get(tid) if tid else None
            if t is None:
                t = {
                    "id": new_id(),
                    "name": f"Target {len(out) + 1}",
                    "frame": "relative",
                    "strokes": [],
                }
            out.append(_with_geometry(t, sh))
        return [d for d in out if pr.target_valid(Target.model_validate(d))]
    changed = False
    for key, value in relayout.items():
        if not key.startswith("shapes["):
            continue
        idx = int(key[7 : key.index("]")]) - N_COURT_SHAPES
        prop = key.split(".")[-1]
        if not 0 <= idx < len(order) or prop not in ("x0", "x1", "y0", "y1"):
            continue
        t = by_id[order[idx]]
        sh = dict(t.get("_box") or _box_of(t))
        sh[prop] = value
        t["_box"] = sh
        changed = True
    if not changed:
        return None
    out = []
    for tid in order:
        t = by_id[tid]
        box = t.pop("_box", None)
        if box is not None:
            box["type"] = "circle" if t["shape"] == "circle" else "rect"
            t = _with_geometry(t, box)
        out.append(t)
    return out


def _box_of(t: dict) -> dict:
    if t["shape"] == "circle":
        return {"x0": t["cx"] - t["r"], "x1": t["cx"] + t["r"],
                "y0": t["cy"] - t["r"], "y1": t["cy"] + t["r"]}  # fmt: skip
    return {"x0": t["x0"], "x1": t["x1"], "y0": t["y0"], "y1": t["y1"]}


def _with_geometry(t: dict, sh: dict) -> dict:
    x0, x1 = sorted((float(sh["x0"]), float(sh["x1"])))
    y0, y1 = sorted((float(sh["y0"]), float(sh["y1"])))
    t = {k: v for k, v in t.items() if k not in ("x0", "x1", "y0", "y1", "cx", "cy", "r")}
    if sh.get("type") == "circle":
        t.update(shape="circle", cx=round((x0 + x1) / 2, 2), cy=round((y0 + y1) / 2, 2),
                 r=round((x1 - x0 + y1 - y0) / 4, 2))  # fmt: skip
    else:
        t.update(shape="rect", x0=round(x0, 2), x1=round(x1, 2), y0=round(y0, 2), y1=round(y1, 2))
    return t


def register_target_editor(prefix: str) -> None:
    @callback(
        Output(f"{prefix}-targets", "data", allow_duplicate=True),
        Input(f"{prefix}-shapes", "data"),
        State(f"{prefix}-targets", "data"),
        prevent_initial_call=True,
    )
    def _drawn(relayout, targets):
        relayout = {k: v for k, v in (relayout or {}).items() if k != "_ts"}
        out = apply_relayout(targets or [], relayout)
        return no_update if out is None else out

    @callback(
        Output(f"{prefix}-court", "figure"),
        Output(f"{prefix}-list", "children"),
        Input(f"{prefix}-targets", "data"),
    )
    def _render(targets):
        valid = _valid(targets)
        return editor_figure(valid), target_rows(prefix, valid)

    @callback(
        Output(f"{prefix}-targets", "data", allow_duplicate=True),
        Input({"type": f"{prefix}-tname", "index": ALL}, "value"),
        Input({"type": f"{prefix}-tframe", "index": ALL}, "value"),
        Input({"type": f"{prefix}-tstrokes", "index": ALL}, "value"),
        Input({"type": f"{prefix}-tdel", "index": ALL}, "n_clicks"),
        State(f"{prefix}-targets", "data"),
        prevent_initial_call=True,
    )
    def _row_edit(_names, _frames, _strokes, _dels, targets):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not ctx.triggered:
            return no_update
        value = ctx.triggered[0]["value"]
        out = []
        changed = False
        for t in targets or []:
            if t["id"] != trig["index"]:
                out.append(t)
                continue
            kind = trig["type"].removeprefix(f"{prefix}-")
            if kind == "tdel":
                if value:
                    changed = True
                    continue
                out.append(t)
            elif kind == "tname" and value is not None and value != t["name"]:
                out.append({**t, "name": value.strip() or t["name"]})
                changed = True
            elif kind == "tframe" and value and value != t["frame"]:
                out.append({**t, "frame": value})
                changed = True
            elif kind == "tstrokes" and list(value or []) != list(t.get("strokes") or []):
                out.append({**t, "strokes": list(value or [])})
                changed = True
            else:
                out.append(t)
        return out if changed else no_update

    @callback(
        Output(f"{prefix}-targets", "data", allow_duplicate=True),
        Input({"type": f"{prefix}-preset", "index": ALL}, "n_clicks"),
        State(f"{prefix}-targets", "data"),
        prevent_initial_call=True,
    )
    def _preset(clicks, targets):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not any(clicks or []):
            return no_update
        p = PRESETS[trig["index"]]
        t = {"id": new_id(), "name": trig["index"], "shape": "rect", "frame": "relative",
             "strokes": list(p.get("strokes", [])),
             **{k: p[k] for k in ("x0", "y0", "x1", "y1")}}  # fmt: skip
        return [*(targets or []), t]

    @callback(
        Output(f"{prefix}-targets", "data", allow_duplicate=True),
        Output(f"{prefix}-set-name", "value"),
        Input(f"{prefix}-set", "value"),
        prevent_initial_call=True,
    )
    def _load_set(set_id):
        if not set_id:
            return no_update, no_update
        ts = services.get_target_set(state.settings(), set_id)
        if ts is None:
            return no_update, no_update
        return [t.model_copy(update={"id": new_id()}).model_dump() for t in ts.targets], ts.name

    @callback(
        Output(f"{prefix}-set", "data"),
        Output(f"{prefix}-set", "value"),
        Output("notify", "sendNotifications", allow_duplicate=True),
        Input(f"{prefix}-set-save", "n_clicks"),
        Input(f"{prefix}-set-delete", "n_clicks"),
        State(f"{prefix}-set-name", "value"),
        State(f"{prefix}-set", "value"),
        State(f"{prefix}-targets", "data"),
        prevent_initial_call=True,
    )
    def _save_set(_save, _delete, name, set_id, targets):
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return no_update, no_update, no_update
        s = state.settings()
        try:
            if ctx.triggered_id == f"{prefix}-set-delete":
                if not set_id:
                    return (
                        no_update,
                        no_update,
                        notification("Pick a saved set first.", color="yellow"),
                    )
                services.delete_target_set(s, set_id)
                msg, value = "Target set deleted.", None
            else:
                if not _valid(targets):
                    return (
                        no_update,
                        no_update,
                        notification("Draw a target first.", color="yellow"),
                    )
                saved = services.save_target_set(s, name, _valid(targets))
                msg, value = f"Saved “{saved.name}”.", saved.id
        except ValueError as exc:
            return no_update, no_update, notification(str(exc), color="red")
        data = [{"value": x.id, "label": x.name} for x in services.list_target_sets(s)]
        return data, value, notification(msg, icon_name="tabler:check")


def targets_from_store(data: list[dict] | None) -> list[Target]:
    return _valid(data)
