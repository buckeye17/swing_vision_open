"""New session wizard: footage → session type and player → practice targets → review, then
create and enqueue processing. The match path (Phase 2) slots into the Session step later.
"""

from __future__ import annotations

from pathlib import Path

import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.app import state
from swingvision.app.components.file_browser import file_browser, register_file_browser
from swingvision.app.components.target_editor import (
    register_target_editor,
    target_editor,
    targets_from_store,
)
from swingvision.app.components.ui import (
    fmt_bytes,
    fmt_duration,
    icon,
    no_output_root_alert,
    notification,
    page_header,
)
from swingvision.app.worker_control import ensure_worker
from swingvision.io.ffmpeg import thumbnail_data_uri, thumbnail_time
from swingvision.io.probe import VIDEO_EXTENSIONS, ProbeError, probe_video
from swingvision.storage.schemas import PRACTICE_SUBMODE_LABELS, VideoInfo

register_file_browser("video-browser", mode="file", extensions=VIDEO_EXTENSIONS)
register_target_editor("ns-tgt")
HIDDEN = {"display": "none"}


STEPS = ("Footage", "Session", "Targets", "Review")


def _card(*children):
    return dmc.Paper(list(children), p="lg", withBorder=True)


def layout(**_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("New session"), no_output_root_alert()], size="lg", px=0)
    profiles = services.list_profiles(state.settings())
    footage = _card(
        dmc.Group(
            [
                dmc.Title("Footage", order=4),
                dmc.Button(
                    "Choose video…",
                    id="video-browser-open",
                    leftSection=icon("tabler:folder-open"),
                    variant="light",
                ),
            ],
            justify="space-between",
        ),
        html.Div(
            id="ns-video-info",
            children=dmc.Text("No video selected.", c="dimmed", size="sm", mt="sm"),
        ),
    )
    session = _card(
        dmc.Title("Session", order=4, mb="sm"),
        dmc.Stack(
            [
                dmc.TextInput(id="ns-name", label="Name", placeholder="e.g. Basket forehands"),
                dmc.Stack(
                    [
                        dmc.Text("Mode", size="sm", fw=500),
                        dmc.SegmentedControl(
                            id="ns-mode",
                            value="practice",
                            data=[
                                {"value": "practice", "label": "Practice"},
                                {
                                    "value": "match",
                                    "label": "Match (coming in Phase 2)",
                                    "disabled": True,
                                },
                            ],
                        ),
                    ],
                    gap=4,
                ),
                dmc.Stack(
                    [
                        dmc.Text("Practice type", size="sm", fw=500),
                        dmc.SegmentedControl(
                            id="ns-submode",
                            value="self_feed",
                            data=[
                                {"value": k, "label": v} for k, v in PRACTICE_SUBMODE_LABELS.items()
                            ],
                        ),
                        dmc.Text(
                            "Self-feed sessions recognize serves on their own; serve practice "
                            "calls every shot against the service box.",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                    gap=4,
                ),
                dmc.Group(
                    [
                        dmc.Select(
                            id="ns-profile",
                            label="Player",
                            description="Who is practicing (handedness and height feed the "
                            "swing analysis).",
                            data=[{"value": p.id, "label": p.name} for p in profiles],
                            value=profiles[0].id if len(profiles) == 1 else None,
                            placeholder="Choose a profile" if profiles else "No profiles yet",
                            clearable=True,
                            flex=1,
                        ),
                        dmc.Anchor("Manage profiles", href="/profiles", size="sm", pb=6),
                    ],
                    align="flex-end",
                ),
            ],
            gap="md",
        ),
    )
    targets = _card(
        dmc.Title("Targets", order=4, mb="xs"),
        dmc.Text(
            "Optional: where you're aiming. Accuracy is measured against these; you can change "
            "them later on the session's Practice page.",
            size="sm",
            c="dimmed",
            mb="sm",
        ),
        target_editor("ns-tgt"),
    )
    review = _card(dmc.Title("Review", order=4, mb="sm"), html.Div(id="ns-review"))
    steps = [footage, session, targets, review]
    panes = [
        html.Div(step, id={"type": "ns-pane", "index": i}, style={} if i == 0 else HIDDEN)
        for i, step in enumerate(steps)
    ]
    nav = dmc.Group(
        [
            dmc.Button(
                "Back",
                id="ns-back",
                variant="default",
                disabled=True,
                leftSection=icon("tabler:arrow-left"),
            ),
            dmc.Group(
                [
                    dmc.Button(
                        "Next", id="ns-next", disabled=True, rightSection=icon("tabler:arrow-right")
                    ),
                    dmc.Button(
                        "Create and process",
                        id="ns-create",
                        disabled=True,
                        leftSection=icon("tabler:player-play"),
                        style=HIDDEN,
                    ),
                ],
                gap="sm",
            ),
        ],
        justify="space-between",
        mt="md",
    )
    return dmc.Container(
        [
            page_header(
                "New session", "Pick footage, say what kind of practice it is, set targets."
            ),
            dcc.Store(id="ns-video"),
            dcc.Store(id="ns-step", data=0),
            dmc.Stepper(
                id="ns-stepper",
                active=0,
                size="sm",
                allowNextStepsSelect=False,
                children=[dmc.StepperStep(label=label) for label in STEPS],
                mb="md",
            ),
            *panes,
            nav,
            file_browser("video-browser", mode="file", title="Choose footage"),
        ],
        size="lg",
        px=0,
    )


dash.register_page(
    __name__, path="/new", title="New session · Swing Vision Open", order=1, layout=layout
)


def _video_warnings(info: VideoInfo) -> list[str]:
    warnings = []
    if info.fps_avg < 50:
        warnings.append(
            f"{info.fps_avg:.0f} fps: ball speed will be approximate (60 fps recommended)."
        )
    if info.display_height < 2160:
        warnings.append(f"{info.display_height}p: far-court ball detection works best at 4K.")
    if not info.has_audio:
        warnings.append("No audio track: hit detection loses its audio cue.")
    return warnings


def _info_card(path: Path, info: VideoInfo, thumb: str | None):
    rows = [
        ("File", f"{path.name} ({fmt_bytes(path.stat().st_size)})"),
        (
            "Resolution",
            f"{info.display_width}×{info.display_height}"
            + (f" (rotated {info.rotation_cw}°)" if info.rotation_cw else ""),
        ),
        ("Frame rate", f"{info.fps_avg:.2f} fps" + (" (variable)" if info.is_vfr else "")),
        ("Duration", fmt_duration(info.duration_s)),
        ("Codec", f"{info.codec} {info.profile or ''}".strip()),
        ("Audio", f"{info.audio_codec}, {info.audio_sample_rate} Hz" if info.has_audio else "none"),
    ]
    table = dmc.Table(
        dmc.TableTbody(
            [dmc.TableTr([dmc.TableTd(k, fw=500, w=110), dmc.TableTd(v)]) for k, v in rows]
        ),
        fz="sm",
        verticalSpacing=4,
    )
    warnings = [
        dmc.Alert(w, color="yellow", variant="light", p="xs") for w in _video_warnings(info)
    ]
    if info.is_vfr:
        warnings.append(
            dmc.Text(
                "Variable frame rate detected; timestamps come from the file, so this is handled.",
                size="xs",
                c="dimmed",
            )
        )
    return dmc.Stack(
        [
            dmc.Text(str(path), size="xs", c="dimmed"),
            dmc.Grid(
                [
                    dmc.GridCol(
                        dmc.Image(src=thumb, radius="sm") if thumb else None,
                        span={"base": 12, "sm": 5},
                    ),
                    dmc.GridCol(table, span={"base": 12, "sm": 7}),
                ],
                gutter="md",
            ),
            *warnings,
        ],
        gap="xs",
        mt="sm",
    )


@callback(
    Output("ns-video", "data"),
    Output("ns-video-info", "children"),
    Output("ns-name", "value"),
    Output("ns-next", "disabled"),
    Input("video-browser-result", "data"),
    State("ns-name", "value"),
    prevent_initial_call=True,
)
def _video_picked(result, name):
    if not result:
        return no_update, no_update, no_update, no_update
    path = Path(result["path"])
    s = state.settings()
    try:
        info = probe_video(s.ffprobe(), path)
    except (ProbeError, FileNotFoundError) as exc:
        return None, dmc.Alert(str(exc), color="red", title="Can't read this file"), no_update, True
    try:
        thumb = thumbnail_data_uri(s.ffmpeg(), path, thumbnail_time(info.duration_s))
    except Exception:
        thumb = None
    return (
        {"path": str(path), "duration_s": info.duration_s},
        _info_card(path, info, thumb),
        name or path.stem,
        False,
    )


@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("ns-create", "n_clicks"),
    State("ns-video", "data"),
    State("ns-name", "value"),
    State("ns-mode", "value"),
    State("ns-submode", "value"),
    State("ns-profile", "value"),
    State("ns-tgt-targets", "data"),
    running=[(Output("ns-create", "loading"), True, False)],
    prevent_initial_call=True,
)
def _create(n, video, name, mode, submode, profile_id, targets):
    if not n or not video:
        return no_update, no_update
    s = state.settings()
    try:
        session = services.create_session(
            s,
            Path(video["path"]),
            name,
            mode,
            submode,
            me_profile_id=profile_id or None,
            practice_targets=targets_from_store(targets),
        )
        config = session.load_config()
        job_id = services.enqueue(s, config.id)
    except Exception as exc:
        return no_update, notification(str(exc), "Could not create session", "red")
    if state.OPTIONS.start_worker:
        ensure_worker(s.output_root)
    return "/jobs", notification(
        f"Session “{config.name}” created; job #{job_id} queued.", icon_name="tabler:check"
    )


@callback(
    Output("ns-step", "data"),
    Input("ns-next", "n_clicks"),
    Input("ns-back", "n_clicks"),
    State("ns-step", "data"),
    prevent_initial_call=True,
)
def _step(_n, _b, step):
    step = step or 0
    if ctx.triggered_id == "ns-next":
        return min(step + 1, len(STEPS) - 1)
    return max(step - 1, 0)


@callback(
    Output("ns-stepper", "active"),
    Output({"type": "ns-pane", "index": ALL}, "style"),
    Output("ns-back", "disabled"),
    Output("ns-next", "style"),
    Output("ns-create", "style"),
    Output("ns-create", "disabled"),
    Input("ns-step", "data"),
    State("ns-video", "data"),
)
def _show_step(step, video):
    step = step or 0
    last = step == len(STEPS) - 1
    panes = [{} if i == step else HIDDEN for i in range(len(STEPS))]
    return step, panes, step == 0, HIDDEN if last else {}, {} if last else HIDDEN, not video


#: Processing time per footage time on an RTX A5000 laptop (M7 overnight run).
PROCESSING_X_REALTIME = 1.8


@callback(
    Output("ns-review", "children"),
    Input("ns-step", "data"),
    State("ns-video", "data"),
    State("ns-name", "value"),
    State("ns-submode", "value"),
    State("ns-profile", "value"),
    State("ns-tgt-targets", "data"),
)
def _review(step, video, name, submode, profile_id, targets):
    if step != len(STEPS) - 1:
        return no_update
    profile = services.get_profile(state.settings(), profile_id) if profile_id else None
    valid = targets_from_store(targets)
    rows = [
        ("Video", Path(video["path"]).name if video else "–"),
        ("Name", name or "–"),
        ("Practice type", PRACTICE_SUBMODE_LABELS.get(submode, submode)),
        ("Player", profile.name if profile else "not set"),
        (
            "Targets",
            ", ".join(t.name for t in valid) if valid else "none (accuracy shows in/out only)",
        ),
    ]
    table = dmc.Table(
        dmc.TableTbody(
            [dmc.TableTr([dmc.TableTd(k, fw=500, w=130), dmc.TableTd(v)]) for k, v in rows]
        ),
        fz="sm",
    )
    hours = (video or {}).get("duration_s", 0) / 3600 * PROCESSING_X_REALTIME
    auto = state.settings().processing.calibration_auto_accept_px is not None
    note = (
        "Processing continues on its own after court detection when the calibration fits well "
        "(Settings)."
        if auto
        else "Processing pauses a few minutes in for you to review the court calibration "
        "(Jobs page). To process unattended, e.g. overnight, turn on automatic acceptance in "
        "Settings → Court calibration."
    )
    if hours >= 0.25:
        note = f"Expect roughly {hours:.1f} h of processing. " + note
    return dmc.Stack([table, dmc.Text(note, size="sm", c="dimmed")], gap="xs")
