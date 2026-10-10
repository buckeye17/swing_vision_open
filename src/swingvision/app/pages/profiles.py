"""Profiles: the players you analyze (basics only in Phase 1).

Handedness and backhand feed the stroke rules (M6); height scales 3D pose. Appearance
enrollment (recognizing you among other players) arrives with match mode in Phase 2.
"""

from __future__ import annotations

import math

import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update
from pydantic import ValidationError

from swingvision import services
from swingvision.app import state, units
from swingvision.app.components.ui import (
    icon,
    no_output_root_alert,
    notification,
    page_header,
)
from swingvision.storage.schemas import BACKHAND_LABELS, HANDEDNESS_LABELS, Profile

HEIGHT_RANGE_M = (1.0, 2.5)  # Profile.height_m's allowed range
SHOE_RANGE_M = (0.18, 0.40)  # Profile.shoe_length_m's


def _height_bounds(u: units.Units) -> tuple[int, int]:
    """The allowed height range in whole input units (100–250 cm, 40–98 in)."""
    lo, hi = HEIGHT_RANGE_M
    return math.ceil(lo * u.small_factor - 1e-9), math.floor(hi * u.small_factor + 1e-9)


def layout(**_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("Profiles"), no_output_root_alert()], size="md", px=0)
    u = units.current()
    h_min, h_max = _height_bounds(u)
    form = dmc.Stack(
        [
            dmc.TextInput(id="prof-name", label="Name", placeholder="e.g. Chris"),
            dmc.Stack(
                [
                    dmc.Text("Plays", size="sm", fw=500),
                    dmc.SegmentedControl(
                        id="prof-hand",
                        value="right",
                        data=[{"value": k, "label": v} for k, v in HANDEDNESS_LABELS.items()],
                    ),
                ],
                gap=4,
            ),
            dmc.Stack(
                [
                    dmc.Text("Backhand", size="sm", fw=500),
                    dmc.SegmentedControl(
                        id="prof-backhand",
                        value="two_handed",
                        data=[{"value": k, "label": v} for k, v in BACKHAND_LABELS.items()],
                    ),
                ],
                gap=4,
            ),
            dmc.NumberInput(
                id="prof-height",
                label=f"Height ({u.height_input_unit})",
                description="Scales the 3D swing skeleton. Optional.",
                min=h_min,
                max=h_max,
                step=1,
                allowDecimal=False,
            ),
            dmc.NumberInput(
                id="prof-shoe",
                label=f"Shoe length ({u.small_unit})",
                description="Heel to toe tip. Places the toe when the shoe hides it on serves "
                "(otherwise 15% of the height). Optional.",
                min=u.small(SHOE_RANGE_M[0]),
                max=u.small(SHOE_RANGE_M[1]),
                step=0.5,
                decimalScale=1,
            ),
            dmc.Text(id="prof-error", c="red", size="sm"),
            dmc.Group(
                [
                    dmc.Button(
                        "Delete",
                        id="prof-delete",
                        color="red",
                        variant="subtle",
                        leftSection=icon("tabler:trash", 16),
                        style={"visibility": "hidden"},
                    ),
                    dmc.Group(
                        [
                            dmc.Button("Cancel", id="prof-cancel", variant="default"),
                            dmc.Button("Save", id="prof-save", leftSection=icon("tabler:check")),
                        ],
                        gap="xs",
                    ),
                ],
                justify="space-between",
                mt="sm",
            ),
        ],
        gap="sm",
    )
    return dmc.Container(
        [
            page_header(
                "Profiles",
                "Who plays in your sessions. Handedness and backhand drive stroke "
                "classification; height scales the 3D swing analysis.",
                right=dmc.Button(
                    "New profile", id="prof-new", leftSection=icon("tabler:user-plus")
                ),
            ),
            dcc.Store(id="prof-edit-id"),
            dcc.Store(id="prof-version", data=0),
            html.Div(id="prof-list"),
            dmc.Modal(id="prof-modal", title="Profile", children=form, size="md"),
            dmc.Modal(
                id="prof-confirm",
                title="Delete profile?",
                children=dmc.Stack(
                    [
                        dmc.Text(
                            "Sessions that use this profile keep their data but lose the "
                            "player assignment.",
                            size="sm",
                        ),
                        dmc.Group(
                            [
                                dmc.Button("Keep", id="prof-confirm-no", variant="default"),
                                dmc.Button("Delete", id="prof-confirm-yes", color="red"),
                            ],
                            justify="flex-end",
                        ),
                    ]
                ),
            ),
        ],
        size="md",
        px=0,
    )


dash.register_page(
    __name__, path="/profiles", title="Profiles · Swing Vision Open", order=3, layout=layout
)


def _initials(name: str) -> str:
    words = name.split()
    return ("".join(w[0] for w in words[:2]) if len(words) > 1 else name[:2]).upper()


def _card(p: Profile):
    facts = [HANDEDNESS_LABELS[p.handedness], f"{BACKHAND_LABELS[p.backhand].lower()} backhand"]
    if p.height_m:
        facts.append(units.current().height_str(p.height_m))
    return dmc.Paper(
        dmc.Group(
            [
                dmc.Group(
                    [
                        dmc.Avatar(_initials(p.name), color="teal", radius="xl"),
                        dmc.Stack(
                            [
                                dmc.Text(p.name, fw=600),
                                dmc.Text(" · ".join(facts), size="sm", c="dimmed"),
                            ],
                            gap=0,
                        ),
                    ],
                    gap="sm",
                ),
                dmc.Button(
                    "Edit",
                    id={"type": "prof-edit", "index": p.id},
                    variant="light",
                    size="xs",
                    leftSection=icon("tabler:pencil", 14),
                ),
            ],
            justify="space-between",
        ),
        p="md",
        withBorder=True,
    )


@callback(Output("prof-list", "children"), Input("prof-version", "data"))
def _render(_):
    if state.library() is None:
        return no_output_root_alert()
    profiles = services.list_profiles(state.settings())
    if not profiles:
        return dmc.Paper(
            dmc.Stack(
                [
                    icon("tabler:user-question", 40, color="gray"),
                    dmc.Text("No profiles yet. Create one for yourself.", c="dimmed"),
                ],
                align="center",
                gap="xs",
            ),
            p="xl",
            withBorder=True,
        )
    return dmc.Stack([_card(p) for p in profiles], gap="sm")


@callback(
    Output("prof-modal", "opened"),
    Output("prof-modal", "title"),
    Output("prof-edit-id", "data"),
    Output("prof-name", "value"),
    Output("prof-hand", "value"),
    Output("prof-backhand", "value"),
    Output("prof-height", "value"),
    Output("prof-shoe", "value"),
    Output("prof-error", "children"),
    Output("prof-delete", "style"),
    Input("prof-new", "n_clicks"),
    Input({"type": "prof-edit", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _open(_new, _edits):
    trig = ctx.triggered_id
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return (no_update,) * 10
    if trig == "prof-new":
        return (
            True,
            "New profile",
            None,
            "",
            "right",
            "two_handed",
            None,
            None,
            "",
            {"visibility": "hidden"},
        )
    p = services.get_profile(state.settings(), trig["index"])
    if p is None:
        return (no_update,) * 10
    u = units.current()
    height = u.height_to_input(p.height_m) if p.height_m else None
    shoe = round(u.small(p.shoe_length_m), 1) if p.shoe_length_m else None
    return True, f"Edit {p.name}", p.id, p.name, p.handedness, p.backhand, height, shoe, "", {}


@callback(
    Output("prof-modal", "opened", allow_duplicate=True),
    Output("prof-error", "children", allow_duplicate=True),
    Output("prof-version", "data"),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("prof-save", "n_clicks"),
    State("prof-edit-id", "data"),
    State("prof-name", "value"),
    State("prof-hand", "value"),
    State("prof-backhand", "value"),
    State("prof-height", "value"),
    State("prof-shoe", "value"),
    State("prof-version", "data"),
    prevent_initial_call=True,
)
def _save(n, pid, name, hand, backhand, height_in, shoe_in, version):
    if not n:
        return no_update, no_update, no_update, no_update
    u = units.current()
    h_min, h_max = _height_bounds(u)
    range_msg = f"Height must be between {h_min} and {h_max} {u.height_input_unit}."
    try:
        height = u.height_from_input(height_in) if height_in not in (None, "") else None
        if height is not None and not HEIGHT_RANGE_M[0] <= height <= HEIGHT_RANGE_M[1]:
            return no_update, range_msg, no_update, no_update
        shoe = float(shoe_in) / u.small_factor if shoe_in not in (None, "") else None
        p = services.save_profile(
            state.settings(), name, hand, backhand, height, pid, shoe_length_m=shoe
        )
    except ValidationError as exc:
        fields = {str(e["loc"][0]) for e in exc.errors() if e.get("loc")}
        lo, hi = (round(u.small(v), 1) for v in SHOE_RANGE_M)
        msg = (
            range_msg
            if "height_m" in fields
            else f"Shoe length must be between {lo} and {hi} {u.small_unit}."
            if "shoe_length_m" in fields
            else str(exc)
        )
        return no_update, msg, no_update, no_update
    except ValueError as exc:
        return no_update, str(exc), no_update, no_update
    return False, "", (version or 0) + 1, notification(f"Saved {p.name}.", icon_name="tabler:check")


@callback(
    Output("prof-modal", "opened", allow_duplicate=True),
    Input("prof-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def _cancel(n):
    return False if n else no_update


@callback(
    Output("prof-confirm", "opened"),
    Output("prof-modal", "opened", allow_duplicate=True),
    Output("prof-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("prof-delete", "n_clicks"),
    Input("prof-confirm-yes", "n_clicks"),
    Input("prof-confirm-no", "n_clicks"),
    State("prof-edit-id", "data"),
    State("prof-version", "data"),
    prevent_initial_call=True,
)
def _delete(_d, _yes, _no, pid, version):
    trig = ctx.triggered_id
    if not ctx.triggered[0]["value"] or not pid:
        return no_update, no_update, no_update, no_update
    if trig == "prof-delete":
        return True, no_update, no_update, no_update
    if trig == "prof-confirm-no":
        return False, no_update, no_update, no_update
    cleared = services.delete_profile(state.settings(), pid)
    msg = "Profile deleted." + (f" Unassigned from {cleared} session(s)." if cleared else "")
    return False, False, (version or 0) + 1, notification(msg, color="gray")
