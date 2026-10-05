"""New session: pick a video, choose mode, create + enqueue processing.

M0 covers the practice path. Targets (M5), the player profile (M2), and the
match path (Phase 2) slot into this page later.
"""

from __future__ import annotations

from pathlib import Path

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc, html, no_update

from swingvision import services
from swingvision.app import state
from swingvision.app.components.file_browser import file_browser, register_file_browser
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


def layout(**_):
    if state.settings().output_root is None:
        return dmc.Container([page_header("New session"), no_output_root_alert()], size="md", px=0)
    return dmc.Container(
        [
            page_header("New session", "Pick footage, choose what kind of session it is."),
            dcc.Store(id="ns-video"),
            dmc.Stack(
                [
                    dmc.Paper(
                        [
                            dmc.Group(
                                [
                                    dmc.Title("1. Footage", order=4),
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
                                children=dmc.Text(
                                    "No video selected.", c="dimmed", size="sm", mt="sm"
                                ),
                            ),
                        ],
                        p="lg",
                        withBorder=True,
                    ),
                    dmc.Paper(
                        [
                            dmc.Title("2. Session", order=4, mb="sm"),
                            dmc.Stack(
                                [
                                    dmc.TextInput(
                                        id="ns-name",
                                        label="Name",
                                        placeholder="e.g. Basket forehands",
                                    ),
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
                                                    {"value": k, "label": v}
                                                    for k, v in PRACTICE_SUBMODE_LABELS.items()
                                                ],
                                            ),
                                        ],
                                        gap=4,
                                    ),
                                    dmc.Text(
                                        "Targets and player profile are added in later "
                                        "milestones; you'll be able to set them on existing "
                                        "sessions.",
                                        size="xs",
                                        c="dimmed",
                                    ),
                                ],
                                gap="md",
                            ),
                        ],
                        p="lg",
                        withBorder=True,
                    ),
                    dmc.Group(
                        dmc.Button(
                            "Create and process",
                            id="ns-create",
                            disabled=True,
                            leftSection=icon("tabler:player-play"),
                        ),
                        justify="flex-end",
                    ),
                ],
                gap="md",
            ),
            file_browser("video-browser", mode="file", title="Choose footage"),
        ],
        size="md",
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
    Output("ns-create", "disabled"),
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
        {"path": str(path)},
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
    running=[(Output("ns-create", "loading"), True, False)],
    prevent_initial_call=True,
)
def _create(n, video, name, mode, submode):
    if not n or not video:
        return no_update, no_update
    s = state.settings()
    try:
        session = services.create_session(s, Path(video["path"]), name, mode, submode)
        config = session.load_config()
        job_id = services.enqueue(s, config.id)
    except Exception as exc:
        return no_update, notification(str(exc), "Could not create session", "red")
    if state.OPTIONS.start_worker:
        ensure_worker(s.output_root)
    return "/jobs", notification(
        f"Session “{config.name}” created; job #{job_id} queued.", icon_name="tabler:check"
    )
