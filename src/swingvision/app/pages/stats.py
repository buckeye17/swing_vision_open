"""Stats page (M7): one session's numbers — speeds by stroke, where the shots landed, how
deep, the strokes side by side, and how much you moved. Shot tables export as CSV or
Parquet (``/export/<session>/<table>.<format>``, see ``server_routes``)."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc, html, no_update

from swingvision.analysis import export as ex
from swingvision.analysis import stats as st
from swingvision.app import state, units
from swingvision.app.components import stats_view as sv
from swingvision.app.components.court_diagram import heatmap_figure
from swingvision.app.components.shots_view import uncalibrated_badge
from swingvision.app.components.ui import fmt_duration, icon, no_output_root_alert, page_header
from swingvision.players import movement as mv
from swingvision.storage.schemas import PRACTICE_SUBMODE_LABELS


def _card(title: str, *children, right=None, hint: str | None = None):
    return dmc.Paper(
        [
            dmc.Group([dmc.Text(title, fw=600, size="sm"), right], justify="space-between"),
            dmc.Text(hint, size="xs", c="dimmed") if hint else None,
            *children,
        ],
        p="sm",
        withBorder=True,
    )


def _export_menu(session_id: str, available: dict[str, bool]):
    items = []
    for what, label in ex.EXPORTS.items():
        for fmt in ex.FORMATS:
            items.append(
                dmc.MenuItem(
                    f"{label} ({fmt.upper() if fmt == 'csv' else 'Parquet'})",
                    href=f"/export/{session_id}/{what}.{fmt}",
                    refresh=True,
                    disabled=not available[what],
                    leftSection=icon(
                        "tabler:file-spreadsheet" if fmt == "csv" else "tabler:database", 14
                    ),
                )
            )
    return dmc.Menu(
        [
            dmc.MenuTarget(
                dmc.Button("Export", size="sm", leftSection=icon("tabler:download", 16))
            ),
            dmc.MenuDropdown(items),
        ],
        position="bottom-end",
    )


def _nav_button(label: str, href: str, ic: str):
    return dmc.Anchor(
        dmc.Button(label, variant="default", size="sm", leftSection=icon(ic, 16)), href=href
    )


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
    subtitle = "Stats"
    if config.practice is not None:
        subtitle += f" · Practice · {PRACTICE_SUBMODE_LABELS[config.practice.submode]}"
    if config.video:
        subtitle += f" · {fmt_duration(config.video.duration_s)}"
    available = {
        "shots": session.shots_path.exists(),
        "practice": session.practice_path.exists(),
        "swings": session.swings_path.exists(),
    }
    header_right = dmc.Group(
        [
            _nav_button("Session", f"/session/{config.id}", "tabler:movie"),
            _nav_button("Practice", f"/practice/{config.id}", "tabler:target-arrow")
            if config.practice is not None
            else None,
            _nav_button("Swings", f"/swings/{config.id}", "tabler:ball-tennis")
            if available["swings"]
            else None,
            _export_menu(config.id, available),
        ],
        gap="sm",
    )
    groups = [
        {"value": g, "label": sv.group_label(g)}
        for g in (*st.GROUPS, "unknown")
        if any(r["group"] == g for r in data.records)
    ]
    filters = dmc.Group(
        [
            dmc.MultiSelect(
                id="st-groups",
                data=groups,
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
                display="block" if config.practice is not None else "none",
            ),
        ],
        gap="md",
    )
    p = state.settings().processing
    movement = st.summarize_movement(data, p.dark_luma, p.view_min)
    xc, yc, H = mv.heatmap(data.movement, fold=True)
    notes = [dmc.Alert(n, color="blue", p="xs") for n in data.notes]
    body = dmc.Grid(
        [
            dmc.GridCol(
                dmc.Stack(
                    [
                        _card(
                            "Speed by stroke",
                            dcc.Graph(id="st-speed", config={"displayModeBar": False}),
                            right=uncalibrated_badge(),
                            hint="Every shot whose contact was seen; the box spans the middle "
                            "half, the line is the median.",
                        ),
                        _card(
                            "Depth",
                            dcc.Graph(id="st-depth", config={"displayModeBar": False}),
                            hint="Where shots came down, measured from the net on the "
                            "opponent's side (both ends folded together).",
                        ),
                    ],
                    gap="sm",
                ),
                span={"base": 12, "lg": 7},
            ),
            dmc.GridCol(
                dmc.Stack(
                    [
                        _card(
                            "Landings",
                            dcc.Graph(id="st-landings", config={"displayModeBar": False}),
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
                    ],
                    gap="sm",
                ),
                span={"base": 12, "lg": 5},
            ),
            dmc.GridCol(
                _card(
                    "Strokes",
                    html.Div(id="st-strokes"),
                    hint="Depth: how far short of the line (−: past it) shots landed, the service "
                    "line for serves and the baseline for the rest. Deep: groundstrokes in, within "
                    f"{units.current().len_str(st.DEEP_ZONE_M)} of the baseline. Wrist speed: the "
                    "racket wrist's peak (swings page).",
                ),
                span=12,
            ),
            dmc.GridCol(
                _card(
                    "Movement",
                    dmc.Grid(
                        [
                            dmc.GridCol(
                                dcc.Graph(
                                    figure=heatmap_figure(xc, yc, H, height=380),
                                    config={"displayModeBar": False},
                                ),
                                span={"base": 12, "md": 5},
                            ),
                            dmc.GridCol(
                                dmc.Stack(
                                    [
                                        _movement_facts(movement),
                                        dcc.Graph(
                                            figure=sv.distance_figure(
                                                st.distance_over_time(data.movement)
                                            ),
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    gap="xs",
                                ),
                                span={"base": 12, "md": 7},
                            ),
                        ]
                    ),
                    hint="Where you spent your time (both ends folded onto the near half) and "
                    "the distance you covered through the session.",
                ),
                span=12,
            ),
        ],
        gutter="md",
    )
    return dmc.Container(
        [
            page_header(config.name, subtitle, right=header_right),
            dcc.Store(id="st-session-id", data=config.id),
            dmc.Stack(
                [
                    *notes,
                    html.Div(id="st-kpis"),
                    filters,
                    body,
                ],
                gap="sm",
            ),
        ],
        size="xl",
        px=0,
    )


def _movement_facts(m: dict):
    if not m:
        return dmc.Text("Player tracking hasn't run yet.", size="sm", c="dimmed")
    u = units.current()
    moving = u.speed_str_mps(m["mean_moving_speed_mps"])
    return dmc.Text(
        f"{u.len(m['distance_m']):,.0f} {u.len_unit} covered in {fmt_duration(m['tracked_s'])} "
        f"of tracking ({m['coverage']:.0%} of the usable video) · average {moving} while moving · "
        f"{m['near_half_frac']:.0%} of the time on the near half.",
        size="sm",
    )


dash.register_page(
    __name__,
    path_template="/stats/<session_id>",
    title="Stats · Swing Vision Open",
    layout=layout,
)


@callback(
    Output("st-kpis", "children"),
    Output("st-speed", "figure"),
    Output("st-depth", "figure"),
    Output("st-strokes", "children"),
    Output("st-landings", "figure"),
    Input("st-groups", "value"),
    Input("st-end", "value"),
    Input("st-excluded", "checked"),
    Input("st-landing-view", "value"),
    State("st-session-id", "data"),
)
def _render(groups, end, excluded, view, session_id):
    found = state.session_for(session_id or "")
    if found is None:
        return (no_update,) * 5
    session = found[2]
    p = state.settings().processing
    data = st.load(session, include_excluded=bool(excluded))
    rows = sv.filtered(data.records, groups or None, end)
    movement = st.summarize_movement(data, p.dark_luma, p.view_min)
    return (
        sv.kpis(rows, movement),
        sv.speed_figure(rows),
        sv.depth_figure(rows),
        sv.strokes_table(rows, st.summarize_swings(data.swings)),
        sv.landing_figure(rows, view or "heat"),
    )
