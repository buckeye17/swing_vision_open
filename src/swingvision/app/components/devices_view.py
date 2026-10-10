"""The Settings page's *Recording devices* card (PLAN.md §7.12, M7c): each phone and
recording mode the library has seen, with its speed calibration. Rename a device, merge two
keys that are the same phone and mode, pick which calibration version is active, or fit a
new one from the accepted net-tape references.
"""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.app import state
from swingvision.app.components.ui import icon, notification
from swingvision.ball import speed_refs as sr


def _cal_label(c: dict) -> str:
    rs = f", τ {1000 * c['tau_s']:.1f} ms" if c.get("tau_s") is not None else ""
    return (
        f"v{c['version']} · × {c['k']:.4f} ± {c['k_sigma'] / c['k']:.2%}{rs} · "
        f"{c['n_refs']} refs · {c['created_at'][:10]}"
    )


def devices_card():
    return dmc.Paper(
        [
            html.A(id="devices"),
            dmc.Title("Recording devices", order=4),
            dmc.Text(
                "Speeds are calibrated per phone and recording mode from serves that hit the "
                "net tape (review them on a session's Calibration → Speed tab). A device "
                "aggregates the accepted references of all its sessions.",
                size="sm",
                c="dimmed",
                mb="sm",
            ),
            html.Div(devices_body(), id="dev-body"),
            dcc.Store(id="dev-version", data=0),
        ],
        p="lg",
        withBorder=True,
    )


def devices_body(selected: str | None = None):
    lib = state.library()
    if lib is None:
        return dmc.Text("Choose an output folder first.", size="sm", c="dimmed")
    devices = lib.list_devices()
    if not devices:
        return dmc.Text(
            "No devices yet: they come from the videos' make and model tags.",
            size="sm",
            c="dimmed",
        )
    rows = []
    for d in devices:
        cal = lib.get_calibration(d["calibration_id"]) if d["calibration_id"] else None
        rows.append(
            dmc.TableTr(
                [
                    dmc.TableTd(services.device_name(d)),
                    dmc.TableTd(str(d["n_sessions"])),
                    dmc.TableTd(
                        _cal_label(cal) if cal else dmc.Text("uncalibrated", size="sm", c="dimmed")
                    ),
                ]
            )
        )
    table = dmc.Table(
        [
            dmc.TableThead(
                dmc.TableTr(
                    [dmc.TableTh("Device"), dmc.TableTh("Sessions"), dmc.TableTh("Calibration")]
                )
            ),
            dmc.TableTbody(rows),
        ],
        striped=True,
        fz="sm",
    )
    keys = [d["device_key"] for d in devices]
    selected = selected if selected in keys else keys[0]
    return dmc.Stack(
        [
            table,
            dmc.Select(
                id="dev-pick",
                label="Device",
                data=[
                    {"value": d["device_key"], "label": services.device_name(d)} for d in devices
                ],
                value=selected,
                allowDeselect=False,
                size="xs",
                w=420,
            ),
            html.Div(device_detail(selected), id="dev-detail"),
        ],
        gap="sm",
    )


def device_detail(key: str):
    lib = state.library()
    d = lib.get_device(key) if lib else None
    if d is None:
        return None
    cals = lib.list_calibrations(key)
    active = next((c for c in cals if c["active"]), None)
    others = [x for x in lib.list_devices() if x["device_key"] != key]
    refs = services.device_references(state.settings(), key)
    diag = []
    if active:
        dg = active.get("diagnostics") or {}
        if active.get("loo_sd") is not None:
            diag.append(f"leave-one-out spread {active['loo_sd']:.2%}")
        for name, label in (("speed", "speed"), ("rolling_shutter", "image motion")):
            t = dg.get(f"trend_{name}")
            if t:
                diag.append(
                    f"trend vs {label}: p = {t['p']:.2f}"
                    + (" (significant)" if t["significant"] else "")
                )
        bs = dg.get("by_session")
        if bs:
            diag.append(f"between sessions: p = {bs['p']:.2f}")
    return dmc.Stack(
        [
            dmc.Text(
                f"Key {key} · {len(lib.sessions_of_device(key))} sessions · "
                f"{len(refs)} accepted references",
                size="xs",
                c="dimmed",
            ),
            dmc.Group(
                [
                    dmc.TextInput(
                        id="dev-name",
                        label="Name",
                        value=d.get("label") or "",
                        placeholder=services.device_name({**d, "label": None}),
                        size="xs",
                        w=300,
                    ),
                    dmc.Button("Rename", id="dev-rename", size="xs", variant="default"),
                ],
                gap="xs",
                align="flex-end",
            ),
            dmc.Group(
                [
                    dmc.Select(
                        id="dev-cal",
                        label="Active speed calibration",
                        data=[{"value": "", "label": "None (uncalibrated)"}]
                        + [{"value": c["id"], "label": _cal_label(c)} for c in cals],
                        value=active["id"] if active else "",
                        allowDeselect=False,
                        size="xs",
                        w=420,
                    ),
                    dmc.Button("Use", id="dev-cal-use", size="xs", variant="default"),
                    dmc.Button(
                        "Update calibration",
                        id="dev-recal",
                        size="xs",
                        leftSection=icon("tabler:target-arrow", 14),
                        disabled=len(refs) < sr.MIN_REFS,
                    ),
                ],
                gap="xs",
                align="flex-end",
            ),
            dmc.Text(" · ".join(diag), size="xs", c="dimmed") if diag else None,
            dmc.Group(
                [
                    dmc.Select(
                        id="dev-merge",
                        label="Same phone and mode as",
                        data=[
                            {"value": x["device_key"], "label": services.device_name(x)}
                            for x in others
                        ],
                        placeholder="Pick a device" if others else "No other device",
                        disabled=not others,
                        size="xs",
                        w=300,
                    ),
                    dmc.Button("Merge into it", id="dev-merge-go", size="xs", variant="default"),
                ],
                gap="xs",
                align="flex-end",
            ),
        ],
        gap="xs",
    )


@callback(
    Output("dev-detail", "children"),
    Input("dev-pick", "value"),
    prevent_initial_call=True,
)
def _pick(key):
    return device_detail(key) if key else None


@callback(
    Output("dev-body", "children"),
    Input("dev-version", "data"),
    State("dev-pick", "value"),
    prevent_initial_call=True,
)
def _rebuild(_v, key):
    return devices_body(key)


@callback(
    Output("dev-version", "data"),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("dev-rename", "n_clicks"),
    Input("dev-cal-use", "n_clicks"),
    Input("dev-recal", "n_clicks"),
    Input("dev-merge-go", "n_clicks"),
    State("dev-pick", "value"),
    State("dev-name", "value"),
    State("dev-cal", "value"),
    State("dev-merge", "value"),
    State("dev-version", "data"),
    prevent_initial_call=True,
)
def _act(rename, use, recal, merge, key, name, cal_id, target, version):
    clicks = {"dev-rename": rename, "dev-cal-use": use, "dev-recal": recal, "dev-merge-go": merge}
    if not clicks.get(ctx.triggered_id) or not key:
        return no_update, no_update
    s = state.settings()
    try:
        if ctx.triggered_id == "dev-rename":
            services.rename_device(s, key, name)
            msg = "Renamed."
        elif ctx.triggered_id == "dev-merge-go":
            if not target:
                return no_update, notification("Pick the device to merge into.", color="yellow")
            moved = services.merge_devices(s, key, target)
            msg = f"Merged: {len(moved)} sessions moved. Update its calibration to use them."
        else:
            if ctx.triggered_id == "dev-cal-use":
                services.set_active_calibration(s, key, cal_id or None)
                head = "Calibration changed."
            else:
                cal, row, refs = services.calibrate_device(s, key)
                if cal is None:
                    return no_update, notification(
                        f"{len(refs)} accepted references: at least {sr.MIN_REFS} are needed.",
                        color="yellow",
                    )
                head = f"Version {row['version']}: × {cal.k:.4f} ± {cal.k_sigma / cal.k:.2%}."
            done = services.refresh_device(s, key)
            msg = (
                f"{head} Updated {len(done['ran'])} sessions"
                + (f", queued {len(done['queued'])}" if done["queued"] else "")
                + "."
            )
    except ValueError as exc:
        return no_update, notification(str(exc), "Couldn't change the device", color="red")
    return (version or 0) + 1, notification(msg, "Recording devices")
