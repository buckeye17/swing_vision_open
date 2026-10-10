"""Stats page (M7, M7a): speeds by stroke, where the shots landed, how deep, the strokes side
by side, and how much you moved — for one session (``/stats/<session>``) or any selection of
sessions (``/stats?from=…&type=…&tag=…``, see ``analysis.aggregate``).

Both scopes render through the same callback from the same records. The selection's filters
live in the URL (bookmarkable) and can be saved as named views; its records export as CSV or
Parquet (``/export-selection/<shots|swings>.<format>?<filter>``). A click on a shot opens its
session at that moment.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import (
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    dcc,
    html,
    no_update,
)

from swingvision import services
from swingvision.analysis import aggregate as agg
from swingvision.analysis import export as ex
from swingvision.analysis import serve_stats as ssv
from swingvision.analysis import stats as st
from swingvision.app import state, units
from swingvision.app.components import serve_view
from swingvision.app.components import stats_view as sv
from swingvision.app.components.court_diagram import heatmap_figure
from swingvision.app.components.shots_view import speed_badge, uncalibrated_badge
from swingvision.app.components.ui import (
    export_menu,
    fmt_duration,
    icon,
    no_output_root_alert,
    notification,
    page_header,
    session_header,
)
from swingvision.storage.schemas import PRACTICE_SUBMODE_LABELS

ALL_GROUPS = (*st.GROUPS, "unknown")


def _card(title: str, *children, right=None, hint: str | None = None, **kw):
    return dmc.Paper(
        [
            dmc.Group([dmc.Text(title, fw=600, size="sm"), right], justify="space-between"),
            dmc.Text(hint, size="xs", c="dimmed") if hint else None,
            *children,
        ],
        p="sm",
        withBorder=True,
        **kw,
    )


def _graph(id_: str):
    return dcc.Graph(id=id_, config={"displayModeBar": False})


def _shot_filters(groups: list[str], is_practice: bool):
    return dmc.Group(
        [
            dmc.MultiSelect(
                id="st-groups",
                data=[{"value": g, "label": sv.group_label(g)} for g in groups],
                value=[],
                placeholder="All strokes",
                clearable=True,
                size="xs",
                w=320,
            ),
            dmc.SegmentedControl(
                id="st-end",
                value="all",
                data=[{"value": k, "label": v} for k, v in sv.END_FILTERS.items()],
                size="xs",
            ),
            dmc.Switch(
                id="st-excluded",
                label="Include shots marked “not a practice shot”",
                size="xs",
                checked=False,
                display="block" if is_practice else "none",
            ),
        ],
        gap="md",
    )


def _body(trends: bool):
    u = units.current()
    cols = [
        dmc.GridCol(
            dmc.Stack(
                [
                    _card(
                        "Speed by stroke",
                        _graph("st-speed"),
                        right=html.Div(uncalibrated_badge(), id="st-speed-badge"),
                        hint="Every shot whose contact was seen; the box spans the middle "
                        "half, the line is the median. Click a shot to watch it.",
                    ),
                    _card(
                        "Depth",
                        _graph("st-depth"),
                        hint="Where shots came down, measured from the net on the "
                        "opponent's side (both ends folded together).",
                    ),
                ],
                gap="sm",
            ),
            span={"base": 12, "lg": 7},
        ),
        dmc.GridCol(
            _card(
                "Landings",
                _graph("st-landings"),
                right=dmc.SegmentedControl(
                    id="st-landing-view",
                    value="heat",
                    data=[
                        {"value": "heat", "label": "Heatmap"},
                        {"value": "dots", "label": "By stroke"},
                    ],
                    size="xs",
                ),
                hint="As you hit them: every shot drawn from the near end, so both "
                "ends add up. Hollow dots: out or net.",
            ),
            span={"base": 12, "lg": 5},
        ),
    ]
    if trends:
        cols.append(
            dmc.GridCol(
                _card(
                    "Over time",
                    _graph("st-trend"),
                    right=dmc.Select(
                        id="st-trend-kpi",
                        data=[{"value": k, "label": v[0]} for k, v in st.TREND_KPIS.items()],
                        value="in_pct",
                        allowDeselect=False,
                        size="xs",
                        w=220,
                    ),
                    hint="One point per session, at its recording date, with the 95% interval "
                    "and how many shots it rests on (hover). Follows the stroke and end filters. "
                    "Click a point to open that session's stats.",
                ),
                span=12,
            )
        )
    cols.append(dmc.GridCol(_serve_card(), span=12))
    cols += [
        dmc.GridCol(
            _card(
                "Strokes",
                html.Div(id="st-strokes"),
                hint="Depth: how far short of the line (−: past it) shots landed, the service "
                "line for serves and the baseline for the rest. Deep: groundstrokes in, within "
                f"{u.len_str(st.DEEP_ZONE_M)} of the baseline. Wrist speed: the "
                "racket wrist's peak (swings page).",
            ),
            span=12,
        ),
        dmc.GridCol(
            _card(
                "Movement",
                dmc.Grid(
                    [
                        dmc.GridCol(_graph("st-heatmap"), span={"base": 12, "md": 5}),
                        dmc.GridCol(
                            dmc.Stack(
                                [html.Div(id="st-move-facts"), _graph("st-distance")], gap="xs"
                            ),
                            span={"base": 12, "md": 7},
                        ),
                    ]
                ),
                hint="Where you spent your time (both ends folded onto the near half) and "
                + (
                    "the distance you covered in each session."
                    if trends
                    else "the distance you covered through the session."
                ),
            ),
            span=12,
        ),
    ]
    return dmc.Grid(cols, gutter="md")


def _serve_card():
    """Serve contact point vs the front toe (M7b)."""
    filters = dmc.Group(
        [
            dmc.SegmentedControl(
                id="st-sv-side",
                value="all",
                data=[
                    {"value": "all", "label": "Both sides"},
                    {"value": "deuce", "label": "Deuce"},
                    {"value": "ad", "label": "Ad"},
                ],
                size="xs",
            ),
            dmc.Switch(id="st-sv-near", label="Near end only", size="xs", checked=True),
            dmc.Switch(id="st-sv-flagged", label="Hide flagged", size="xs", checked=True),
        ],
        gap="md",
    )

    def panel(title: str, graph: str, span):
        return dmc.GridCol([dmc.Text(title, size="xs", fw=600), _graph(graph)], span=span)

    half = {"base": 12, "md": 6}
    return _card(
        "Serve contact",
        filters,
        html.Div(id="st-sv-summary", style={"marginTop": 8}),
        dmc.Grid(
            [
                panel("From above", "st-sv-top", half),
                panel("From the side", "st-sv-side-fig", half),
                panel(
                    "Speed (top) and in % (bottom) by where you hit it: bins of at least "
                    f"{ssv.MIN_BIN} serves, with 95% intervals",
                    "st-sv-effects",
                    12,
                ),
                panel(
                    f"In % and median speed per 10 cm cell (cells with at least {ssv.MIN_CELL} "
                    "serves)",
                    "st-sv-grid",
                    12,
                ),
            ],
            gutter="sm",
        ),
        hint="Where the ball was struck relative to your front foot's toe tip, in the frame of "
        "contact: forward toward the net, sideways toward your racket arm. Dots: one serve each, "
        "coloured by speed, hollow for faults; hover for the uncertainty, click to watch it. "
        "Serves from the camera's end; flagged ones (far end, contact above the picture, toss "
        "not seen, in the dark) are hidden unless you switch them on.",
    )


def _scope_switch(value: str, other_ok: bool):
    return dmc.SegmentedControl(
        id="st-scope-switch",
        value=value,
        data=[
            {
                "value": "session",
                "label": "This session",
                "disabled": value != "session" and not other_ok,
            },
            {"value": "sessions", "label": "Sessions…"},
        ],
        size="sm",
    )


def _placeholders():
    """Outputs the shared render callback writes that only the selection page shows."""
    return [html.Div(id="st-count", hidden=True), html.Div(id="st-export", hidden=True)]


def layout(session_id: str | None = None, **_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Stats"), no_output_root_alert()], size="xl", px=0)
    found = state.session_for(session_id or "")
    if found is None:
        return dmc.Container(
            [page_header("Session not found"), dmc.Anchor("Back to library", href="/")],
            size="xl",
            px=0,
        )
    _lib, _row, session = found
    config = session.load_config()
    data = st.load(session)
    groups = [g for g in ALL_GROUPS if any(r["group"] == g for r in data.records)]
    right = dmc.Group([_scope_switch("session", True), export_menu(session, config)], gap="sm")
    return dmc.Container(
        [
            session_header(session, config, "stats", _row["status"], right),
            dcc.Store(id="st-scope", data={"session": config.id}),
            *_placeholders(),
            dmc.Stack(
                [
                    html.Div(id="st-notes"),
                    html.Div(id="st-kpis"),
                    _shot_filters(groups, config.practice is not None),
                    _body(trends=False),
                ],
                gap="sm",
            ),
        ],
        size="xl",
        px=0,
    )


dash.register_page(
    __name__,
    path_template="/stats/<session_id>",
    title="Stats · Swing Vision Open",
    layout=layout,
)


# ---------------------------------------------------------------------------
# A selection of sessions (M7a)
# ---------------------------------------------------------------------------


def _session_label(row: dict) -> str:
    return f"{(row.get('recorded_on') or row['created_at'] or '')[:10]} · {row['name']}"


def _views_data(lib) -> list[dict]:
    return [{"value": v["id"], "label": v["name"]} for v in lib.list_views()]


def _filter_panel(flt: agg.SessionFilter, lib):
    rows = lib.list_sessions()
    sessions = [{"value": r["id"], "label": _session_label(r)} for r in rows]
    profiles = [{"value": p.id, "label": p.name} for p in services.list_profiles(state.settings())]
    devices = [
        {"value": d["device_key"], "label": services.device_name(d)} for d in lib.list_devices()
    ]
    known = {r["id"] for r in rows}
    tags = sorted(set(lib.all_tags()) | set(flt.tags), key=str.lower)
    ms = {"size": "xs", "clearable": True, "searchable": True}
    return dmc.Paper(
        dmc.Stack(
            [
                dmc.Group(
                    [
                        dmc.DatePickerInput(
                            id="st-f-dates",
                            type="range",
                            label="Recorded",
                            placeholder="Any date",
                            value=[flt.date_from, flt.date_to],  # [None, None]: any date
                            clearable=True,
                            allowSingleDateInRange=True,
                            valueFormat="YYYY-MM-DD",
                            size="xs",
                            w=230,
                        ),
                        dmc.MultiSelect(
                            id="st-f-mode",
                            label="Mode",
                            data=[{"value": m, "label": m.capitalize()} for m in agg.MODES],
                            value=list(flt.modes),
                            placeholder="Any",
                            w=170,
                            **ms,
                        ),
                        dmc.MultiSelect(
                            id="st-f-type",
                            label="Practice type",
                            data=[
                                {"value": k, "label": v} for k, v in PRACTICE_SUBMODE_LABELS.items()
                            ],
                            value=list(flt.practice_types),
                            placeholder="Any",
                            w=240,
                            **ms,
                        ),
                        dmc.MultiSelect(
                            id="st-f-profile",
                            label="Player",
                            data=profiles,
                            value=list(flt.profiles),
                            placeholder="Anyone",
                            w=180,
                            **ms,
                        ),
                        dmc.MultiSelect(
                            id="st-f-device",
                            label="Recorded with",
                            data=devices,
                            value=[d for d in flt.devices if d in {x["value"] for x in devices}],
                            placeholder="Any device",
                            w=240,
                            **ms,
                        ),
                        dmc.MultiSelect(
                            id="st-f-tags",
                            label="Tags (all of)",
                            data=tags,
                            value=list(flt.tags),
                            placeholder="No tag filter" if tags else "Tag sessions in the Library",
                            w=220,
                            **ms,
                        ),
                    ],
                    gap="sm",
                    align="flex-end",
                ),
                dmc.Group(
                    [
                        dmc.MultiSelect(
                            id="st-f-include",
                            label="Only these sessions",
                            data=sessions,
                            value=[s for s in flt.include if s in known],
                            placeholder="All that match",
                            w=360,
                            **ms,
                        ),
                        dmc.MultiSelect(
                            id="st-f-exclude",
                            label="Leave out",
                            data=sessions,
                            value=[s for s in flt.exclude if s in known],
                            placeholder="None",
                            w=300,
                            **ms,
                        ),
                        dmc.Stack(
                            [
                                dmc.Switch(
                                    id="st-f-cal",
                                    label="Only court calibrations you confirmed",
                                    checked=flt.user_calibration,
                                    size="xs",
                                ),
                                dmc.Tooltip(
                                    dmc.Switch(
                                        id="st-f-speeds",
                                        label="Only calibrated speeds",
                                        checked=flt.calibrated_speeds,
                                        size="xs",
                                    ),
                                    label="Sessions whose speeds carry their recording "
                                    "device's calibration from net-tape serves.",
                                    multiline=True,
                                    w=260,
                                ),
                            ],
                            gap=6,
                        ),
                    ],
                    gap="sm",
                    align="flex-end",
                ),
                dmc.Group(
                    [
                        html.Div(id="st-count"),
                        dmc.Group(
                            [
                                dmc.Select(
                                    id="st-view",
                                    data=_views_data(lib),
                                    placeholder="Saved views",
                                    clearable=True,
                                    size="xs",
                                    w=200,
                                    leftSection=icon("tabler:bookmark", 14),
                                ),
                                dmc.ActionIcon(
                                    icon("tabler:trash", 14),
                                    id="st-view-delete",
                                    variant="subtle",
                                    color="gray",
                                    **{"aria-label": "Delete the saved view"},
                                ),
                                dmc.TextInput(
                                    id="st-view-name",
                                    placeholder="Name this view",
                                    size="xs",
                                    w=170,
                                ),
                                dmc.Button(
                                    "Save view",
                                    id="st-view-save",
                                    size="xs",
                                    variant="default",
                                    leftSection=icon("tabler:device-floppy", 14),
                                ),
                            ],
                            gap=6,
                        ),
                    ],
                    justify="space-between",
                ),
            ],
            gap="sm",
        ),
        p="sm",
        withBorder=True,
    )


def layout_sessions(back: str | None = None, **query):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Stats"), no_output_root_alert()], size="xl", px=0)
    lib = state.library()
    flt = agg.SessionFilter.from_query(query)
    sel = agg.select(lib, flt)
    back_ok = bool(back and lib.get_session(back))
    title = dmc.Stack(
        [
            html.Div([dmc.Anchor("Library", href="/"), " / Stats"], className="sv-crumb"),
            dmc.Title("Stats", order=1),
            dmc.Text(
                "The statistics of every session that matches, added up.", size="sm", c="dimmed"
            ),
        ],
        gap=6,
    )
    right = dmc.Group([_scope_switch("sessions", back_ok), html.Div(id="st-export")], gap="sm")
    return dmc.Container(
        [
            dmc.Group([title, right], justify="space-between", align="flex-start", mb="sm"),
            dcc.Store(
                id="st-scope", data={"query": flt.to_query(), "back": back if back_ok else None}
            ),
            dcc.Store(id="st-url-sink"),
            dmc.Stack(
                [
                    _filter_panel(flt, lib),
                    html.Div(id="st-notes"),
                    html.Div(id="st-kpis"),
                    _shot_filters(list(ALL_GROUPS), sel.data.is_practice or flt.is_empty()),
                    _body(trends=True),
                ],
                gap="sm",
            ),
        ],
        size="xl",
        px=0,
    )


dash.register_page(
    "swingvision.app.pages.stats_sessions",
    path="/stats",
    title="Stats · Swing Vision Open",
    layout=layout_sessions,
)


# ---------------------------------------------------------------------------
# Rendering (both scopes)
# ---------------------------------------------------------------------------


def _load(scope: dict | None, excluded: bool):
    """(data, selection or None) for the page's scope."""
    scope = scope or {}
    if scope.get("session"):
        found = state.session_for(scope["session"])
        if found is None:
            return None, None
        p = state.settings().processing
        return st.load(found[2], bool(excluded), p.dark_luma, p.view_min), None
    lib = state.library()
    if lib is None:
        return None, None
    sel = agg.select(lib, agg.SessionFilter.from_query(scope.get("query")), bool(excluded))
    return sel.data, sel


def _movement_facts(m: dict, n_sessions: int):
    if not m:
        return dmc.Text("Player tracking hasn't run yet.", size="sm", c="dimmed")
    u = units.current()
    moving = u.speed_str_mps(m["mean_moving_speed_mps"])
    over = f" over {n_sessions} sessions" if n_sessions > 1 else ""
    return dmc.Text(
        f"{u.len(m['distance_m']):,.0f} {u.len_unit} covered in {fmt_duration(m['tracked_s'])} "
        f"of tracking{over} ({m['coverage']:.0%} of the usable video) · average {moving} while "
        f"moving · {m['near_half_frac']:.0%} of the time on the near half.",
        size="sm",
    )


def _count_text(sel: agg.Selection | None):
    if sel is None:
        return None
    c = sel.counts()
    parts = [
        dmc.Text(
            f"{c['sessions']} session{'s' if c['sessions'] != 1 else ''} · {c['shots']:,} shots · "
            f"{c['serves']:,} serves · {c['swings']:,} swings",
            size="sm",
            fw=600,
        )
    ]
    extra = []
    if c["missing"]:
        extra.append(f"{c['missing']} more match but aren't processed yet")
    if c["filtered_out"]:
        extra.append(f"{c['filtered_out']} left out by the quality options")
    if extra:
        parts.append(dmc.Text(" · ".join(extra), size="xs", c="dimmed"))
    return dmc.Stack(parts, gap=0)


def export_query(query: str | None, groups, end, excluded) -> str:
    """The selection export's query: the filter plus the page's shot filters."""
    from urllib.parse import urlencode

    extra = {}
    if groups:
        extra["groups"] = ",".join(groups)
    if end and end != "all":
        extra["end"] = end
    if excluded:
        extra["excluded"] = "1"
    return "&".join(q for q in (query or "", urlencode(extra)) if q)


def _selection_export(query: str, groups, end, excluded, has_rows: bool):
    q = export_query(query, groups, end, excluded)
    items = [
        dmc.MenuItem(
            f"{label} ({fmt.upper() if fmt == 'csv' else 'Parquet'})",
            href=f"/export-selection/{what}.{fmt}" + (f"?{q}" if q else ""),
            refresh=True,
            leftSection=icon("tabler:file-spreadsheet" if fmt == "csv" else "tabler:database", 14),
        )
        for what, label in ex.SELECTION_EXPORTS.items()
        for fmt in ex.FORMATS
    ]
    return dmc.Menu(
        [
            dmc.MenuTarget(
                dmc.Button(
                    "Export",
                    variant="default",
                    disabled=not has_rows,
                    leftSection=icon("tabler:download", 16),
                )
            ),
            dmc.MenuDropdown(
                [
                    dmc.MenuLabel("The records shown (the stroke and end filters apply to shots)"),
                    *items,
                ]
            ),
        ],
        position="bottom-end",
    )


@callback(
    Output("st-notes", "children"),
    Output("st-kpis", "children"),
    Output("st-speed", "figure"),
    Output("st-depth", "figure"),
    Output("st-strokes", "children"),
    Output("st-landings", "figure"),
    Output("st-heatmap", "figure"),
    Output("st-move-facts", "children"),
    Output("st-distance", "figure"),
    Output("st-count", "children"),
    Output("st-export", "children"),
    Output("st-speed-badge", "children"),
    Input("st-scope", "data"),
    Input("st-groups", "value"),
    Input("st-end", "value"),
    Input("st-excluded", "checked"),
    Input("st-landing-view", "value"),
)
def _render(scope, groups, end, excluded, view):
    data, sel = _load(scope, excluded)
    if data is None:
        return (no_update,) * 12
    rows = sv.filtered(data.records, groups or None, end)
    movement = st.summarize_movement(data)
    labels = sv.short_labels(data.sessions)
    xc, yc, H = st.movement_heatmap(data)
    notes = [dmc.Alert(n, color="blue", p="xs") for n in data.notes]
    if sel is not None and sel.missing:
        notes.append(
            dmc.Button(
                "Update their statistics",
                id="st-refresh",
                size="xs",
                variant="light",
                leftSection=icon("tabler:refresh", 14),
            )
        )
    if sel is not None and not data.sessions:
        notes.append(
            dmc.Alert("No processed session matches these filters.", color="yellow", p="xs")
        )
    exports = (
        _selection_export(scope.get("query"), groups, end, excluded, bool(data.records))
        if sel is not None
        else None
    )
    return (
        notes,
        sv.kpis(rows, movement, data.sessions if sel is not None else None),
        sv.speed_figure(rows, labels),
        sv.depth_figure(rows),
        sv.strokes_table(rows, st.summarize_swings(data.swings)),
        sv.landing_figure(rows, view or "heat", labels),
        heatmap_figure(xc, yc, H, height=380),
        _movement_facts(movement, len(data.movement)),
        sv.distance_figure(st.distance_series(data)),
        _count_text(sel),
        exports,
        _speed_badge(data.sessions),
    )


def _speed_badge(sessions: list[dict]):
    """Calibrated / uncalibrated speeds of the sessions shown (M7c)."""
    n_cal = sum(bool(s.get("speeds_calibrated")) for s in sessions)
    if len(sessions) == 1 and n_cal:
        found = state.session_for(sessions[0]["session_id"])
        if found is not None:
            return speed_badge(*services.speed_status(state.settings(), found[2]))
    return speed_badge(mixed=(n_cal, len(sessions)) if sessions else None)


@callback(
    Output("st-sv-summary", "children"),
    Output("st-sv-top", "figure"),
    Output("st-sv-side-fig", "figure"),
    Output("st-sv-effects", "figure"),
    Output("st-sv-grid", "figure"),
    Input("st-scope", "data"),
    Input("st-excluded", "checked"),
    Input("st-sv-side", "value"),
    Input("st-sv-near", "checked"),
    Input("st-sv-flagged", "checked"),
)
def _serves(scope, excluded, side, near, hide_flagged):
    data, _sel = _load(scope, excluded)
    if data is None:
        return (no_update,) * 5
    if not any(r.get("forward_m") is not None for r in data.serves):
        text = dmc.Text(
            "No serve contact points: they need serves from the camera's end with the toss "
            "and the contact in the picture.",
            size="sm",
            c="dimmed",
        )
        return text, *(sv.empty_figure("No serve contact points.", h) for h in (380, 380, 300, 300))
    rows = ssv.select(
        data.serves,
        None if side in (None, "all") else side,
        near_only=bool(near),
        hide_flagged=bool(hide_flagged),
    )
    allrows = [r for r in data.serves if r.get("forward_m") is not None]
    labels = sv.short_labels(data.sessions)
    return (
        serve_view.summary(rows, allrows),
        serve_view.top_figure(rows, labels),
        serve_view.side_figure(rows, labels),
        serve_view.effect_figure(rows),
        serve_view.grid_figure(rows),
    )


@callback(
    Output("st-trend", "figure"),
    Input("st-scope", "data"),
    Input("st-groups", "value"),
    Input("st-end", "value"),
    Input("st-excluded", "checked"),
    Input("st-trend-kpi", "value"),
)
def _trend(scope, groups, end, excluded, kpi):
    data, _sel = _load(scope, excluded)
    if data is None:
        return no_update
    rows = sv.filtered(data.records, groups or None, end)
    swings = [s for s in data.swings if not groups or s["stroke_type"] in groups]
    kpi = kpi if kpi in st.TREND_KPIS else "in_pct"
    return sv.trend_figure(st.trend(data, kpi, rows, swings), kpi)


@callback(
    Output("st-scope", "data"),
    Input("st-f-dates", "value"),
    Input("st-f-mode", "value"),
    Input("st-f-type", "value"),
    Input("st-f-profile", "value"),
    Input("st-f-device", "value"),
    Input("st-f-tags", "value"),
    Input("st-f-include", "value"),
    Input("st-f-exclude", "value"),
    Input("st-f-cal", "checked"),
    Input("st-f-speeds", "checked"),
    State("st-scope", "data"),
    prevent_initial_call=True,
)
def _filter(dates, modes, types, profiles, devices, tags, include, exclude, cal, speeds, scope):
    dates = [d for d in (dates or []) if d] if isinstance(dates, list) else []
    flt = agg.SessionFilter.make(
        date_from=dates[0] if dates else None,
        date_to=dates[-1] if dates else None,
        modes=modes,
        practice_types=types,
        profiles=profiles,
        devices=devices,
        tags=tags,
        include=include,
        exclude=exclude,
        user_calibration=cal,
        calibrated_speeds=speeds,
    )
    query = flt.to_query()
    if scope and scope.get("query") == query:
        return no_update
    return {**(scope or {}), "query": query}


# The filters go into the address bar (replaceState: no page reload), so a view can be
# bookmarked or shared.
clientside_callback(
    """
    function(scope) {
        if (!scope || scope.session) { return window.dash_clientside.no_update; }
        const parts = [];
        if (scope.query) { parts.push(scope.query); }
        if (scope.back) { parts.push("back=" + encodeURIComponent(scope.back)); }
        const url = "/stats" + (parts.length ? "?" + parts.join("&") : "");
        if (window.location.pathname + window.location.search !== url) {
            window.history.replaceState(window.history.state, "", url);
        }
        return url;
    }
    """,
    Output("st-url-sink", "data"),
    Input("st-scope", "data"),
)

# A click on a shot (speed dots, landing dots) opens its session at that moment; a click on
# a session (trend point, distance bar) opens that session's stats.
clientside_callback(
    """
    function(speed, land, trend, dist, svTop, svSide) {
        const nu = window.dash_clientside.no_update;
        const trig = dash_clientside.callback_context.triggered;
        if (!trig.length || !trig[0].value) { return nu; }
        const id = trig[0].prop_id.split(".")[0];
        const pt = trig[0].value.points && trig[0].value.points[0];
        const cd = pt && pt.customdata;
        if (!cd) { return nu; }
        if (id === "st-speed" && cd[2]) {
            return "/session/" + cd[2] + "?t=" + Number(cd[3]).toFixed(2);
        }
        if (id === "st-landings" && cd[0]) {
            return "/session/" + cd[0] + "?t=" + Number(cd[1]).toFixed(2);
        }
        if ((id === "st-sv-top" || id === "st-sv-side-fig") && cd[1]) {
            return "/session/" + cd[1] + "?t=" + (Number(cd[2]) - 1.5).toFixed(2);
        }
        if (id === "st-trend" && cd[3]) { return "/stats/" + cd[3]; }
        if (id === "st-distance" && cd[1]) { return "/stats/" + cd[1]; }
        return nu;
    }
    """,
    Output("url", "href", allow_duplicate=True),
    Input("st-speed", "clickData"),
    Input("st-landings", "clickData"),
    Input("st-trend", "clickData"),
    Input("st-distance", "clickData"),
    Input("st-sv-top", "clickData"),
    Input("st-sv-side-fig", "clickData"),
    prevent_initial_call=True,
)

# This session ⇄ a selection of sessions.
clientside_callback(
    """
    function(value, scope) {
        const nu = window.dash_clientside.no_update;
        if (!scope) { return nu; }
        if (value === "sessions" && scope.session) {
            return "/stats?back=" + encodeURIComponent(scope.session);
        }
        if (value === "session" && scope.back) { return "/stats/" + scope.back; }
        return nu;
    }
    """,
    Output("url", "href", allow_duplicate=True),
    Input("st-scope-switch", "value"),
    State("st-scope", "data"),
    prevent_initial_call=True,
)


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("st-scope", "data", allow_duplicate=True),
    Input("st-refresh", "n_clicks"),
    State("st-scope", "data"),
    prevent_initial_call=True,
)
def _refresh_missing(clicks, scope):
    """Bring the matching sessions without statistics records up to date: in the app when
    only cheap stages are stale (e.g. after an upgrade), otherwise as a queued job."""
    lib = state.library()
    if not clicks or lib is None or not scope or scope.get("session"):
        return no_update, no_update
    sel = agg.select(lib, agg.SessionFilter.from_query(scope.get("query")))
    ran, queued, failed = 0, 0, []
    for row in sel.missing:
        try:
            status, _job = services.refresh_practice(state.settings(), row["id"])
        except (RuntimeError, ValueError) as exc:
            failed.append(f"{row['name']}: {exc}")
            continue
        ran += status == "ran"
        queued += status != "ran"
    parts = []
    if ran:
        parts.append(f"updated {ran}")
    if queued:
        parts.append(f"{queued} need processing (queued, see Jobs)")
    if failed:
        parts.append(f"{len(failed)} failed ({'; '.join(failed[:2])})")
    if queued and state.OPTIONS.start_worker:
        from swingvision.app.worker_control import ensure_worker

        ensure_worker(state.settings().output_root)
    text = "Statistics: " + (", ".join(parts) or "nothing to do") + "."
    return (
        notification(text, color="red" if failed else "blue", icon_name="tabler:refresh"),
        {**scope, "refreshed": (scope.get("refreshed") or 0) + 1},
    )


# ---------------------------------------------------------------------------
# Saved views
# ---------------------------------------------------------------------------


@callback(
    Output("st-f-dates", "value"),
    Output("st-f-mode", "value"),
    Output("st-f-type", "value"),
    Output("st-f-profile", "value"),
    Output("st-f-device", "value"),
    Output("st-f-tags", "value"),
    Output("st-f-include", "value"),
    Output("st-f-exclude", "value"),
    Output("st-f-cal", "checked"),
    Output("st-f-speeds", "checked"),
    Input("st-view", "value"),
    prevent_initial_call=True,
)
def _open_view(view_id):
    lib = state.library()
    view = lib.get_view(view_id) if lib and view_id else None
    if view is None:
        return (no_update,) * 10
    f = agg.SessionFilter.from_query(view["query"])
    dates = [f.date_from, f.date_to]
    return (
        dates,
        list(f.modes),
        list(f.practice_types),
        list(f.profiles),
        list(f.devices),
        list(f.tags),
        list(f.include),
        list(f.exclude),
        f.user_calibration,
        f.calibrated_speeds,
    )


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("st-view", "data"),
    Output("st-view", "value"),
    Output("st-view-name", "value"),
    Input("st-view-save", "n_clicks"),
    Input("st-view-name", "n_submit"),
    Input("st-view-delete", "n_clicks"),
    State("st-view-name", "value"),
    State("st-view", "value"),
    State("st-scope", "data"),
    prevent_initial_call=True,
)
def _save_view(_save, _submit, _delete, name, view_id, scope):
    lib = state.library()
    if lib is None or not ctx.triggered or not ctx.triggered[0]["value"]:
        return (no_update,) * 4
    if ctx.triggered_id == "st-view-delete":
        if not view_id:
            return notification("Pick a saved view to delete.", color="gray"), *(no_update,) * 3
        lib.delete_view(view_id)
        return notification("View deleted.", color="gray"), _views_data(lib), None, no_update
    try:
        new_id = services.save_view(state.settings(), name or "", (scope or {}).get("query", ""))
    except ValueError as exc:
        return notification(str(exc), color="red"), *(no_update,) * 3
    return (
        notification(f"Saved the view “{name.strip()}”.", icon_name="tabler:bookmark"),
        _views_data(lib),
        new_id,
        "",
    )
