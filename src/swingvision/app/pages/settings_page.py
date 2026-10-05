"""Settings: output folder, tool paths, processing defaults, system info."""

from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, html, no_update

from swingvision.app import state
from swingvision.app.components.file_browser import file_browser, register_file_browser
from swingvision.app.components.ui import icon, notification, page_header
from swingvision.app.worker_control import ensure_worker
from swingvision.io.ffmpeg import available_encoders
from swingvision.settings import save_settings, settings_path
from swingvision.storage.library import Library

register_file_browser("out-browser", mode="folder")


@cache
def _gpu_info() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "No NVIDIA GPU detected"
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi not found"


def _tool_report(ffmpeg: str, ffprobe: str) -> list:
    rows = []
    s = state.settings().model_copy(update={"ffmpeg_path": ffmpeg, "ffprobe_path": ffprobe})
    for label, getter in (("ffmpeg", s.ffmpeg), ("ffprobe", s.ffprobe)):
        try:
            exe = getter()
            ver = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10)
            first = ver.stdout.splitlines()[0] if ver.stdout else "?"
            rows.append(
                dmc.Text([icon("tabler:check", 14, color="green"), f" {label}: {first}"], size="sm")
            )
        except (OSError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            rows.append(dmc.Text(f"✗ {label}: {exc}", size="sm", c="red"))
    try:
        enc = available_encoders(s.ffmpeg())
        nvenc = "h264_nvenc" in enc
        if nvenc:
            rows.append(
                dmc.Text(
                    [icon("tabler:check", 14, color="green"), " NVENC H.264 encoder available"],
                    size="sm",
                )
            )
        else:
            rows.append(dmc.Text("NVENC not available (CPU encoding)", size="sm", c="orange"))
    except FileNotFoundError:
        pass
    return rows


def layout(**_):
    s = state.settings()
    return dmc.Container(
        [
            page_header("Settings", f"Stored in {settings_path()}"),
            dmc.Stack(
                [
                    dmc.Paper(
                        [
                            dmc.Title("Output folder", order=4),
                            dmc.Text(
                                "Session data, the library database, and fine-tuned models live "
                                "here. Raw footage is never copied.",
                                size="sm",
                                c="dimmed",
                                mb="sm",
                            ),
                            dmc.Group(
                                [
                                    dmc.TextInput(
                                        id="set-output-root",
                                        flex=1,
                                        value=str(s.output_root or ""),
                                        placeholder="e.g. D:\\SwingVisionData",
                                    ),
                                    dmc.Button(
                                        "Browse…",
                                        id="out-browser-open",
                                        variant="default",
                                        leftSection=icon("tabler:folder-open"),
                                    ),
                                ],
                                gap="xs",
                            ),
                        ],
                        p="lg",
                        withBorder=True,
                    ),
                    dmc.Paper(
                        [
                            dmc.Title("Tools", order=4, mb="sm"),
                            dmc.SimpleGrid(
                                [
                                    dmc.TextInput(
                                        id="set-ffmpeg", label="ffmpeg", value=s.ffmpeg_path
                                    ),
                                    dmc.TextInput(
                                        id="set-ffprobe", label="ffprobe", value=s.ffprobe_path
                                    ),
                                ],
                                cols={"base": 1, "sm": 2},
                            ),
                            dmc.Stack(
                                id="set-tool-report",
                                gap=4,
                                mt="sm",
                                children=_tool_report(s.ffmpeg_path, s.ffprobe_path),
                            ),
                        ],
                        p="lg",
                        withBorder=True,
                    ),
                    dmc.Paper(
                        [
                            dmc.Title("Processing defaults", order=4, mb="sm"),
                            dmc.SimpleGrid(
                                [
                                    dmc.Select(
                                        id="set-proxy-height",
                                        label="Playback proxy height",
                                        data=[
                                            {"value": str(h), "label": f"{h}p"}
                                            for h in (540, 720, 1080)
                                        ],
                                        value=str(s.processing.proxy_height),
                                        allowDeselect=False,
                                    ),
                                    dmc.NumberInput(
                                        id="set-chunk",
                                        label="Checkpoint chunk length (s)",
                                        value=s.processing.chunk_seconds,
                                        min=10,
                                        max=1800,
                                        step=10,
                                    ),
                                ],
                                cols={"base": 1, "sm": 2},
                            ),
                        ],
                        p="lg",
                        withBorder=True,
                    ),
                    dmc.Paper(
                        [
                            dmc.Title("Court calibration", order=4),
                            dmc.Text(
                                "After court detection, processing pauses until you review the "
                                "calibration, unless it fits the painted lines this well.",
                                size="sm",
                                c="dimmed",
                                mb="sm",
                            ),
                            dmc.Switch(
                                id="set-cal-auto",
                                label="Continue without review when the fit is good",
                                checked=s.processing.calibration_auto_accept_px is not None,
                                mb="sm",
                            ),
                            dmc.SimpleGrid(
                                [
                                    dmc.NumberInput(
                                        id="set-cal-threshold",
                                        label="Accept if line RMS is below (px)",
                                        value=s.processing.calibration_auto_accept_px or 1.5,
                                        min=0.3,
                                        max=10,
                                        step=0.1,
                                        decimalScale=1,
                                    ),
                                    dmc.NumberInput(
                                        id="set-cal-drift",
                                        label="Flag camera movement above (px)",
                                        value=s.processing.calibration_drift_px,
                                        min=0.5,
                                        max=50,
                                        step=0.5,
                                        decimalScale=1,
                                    ),
                                ],
                                cols={"base": 1, "sm": 2},
                            ),
                        ],
                        p="lg",
                        withBorder=True,
                    ),
                    dmc.Paper(
                        [
                            dmc.Title("System", order=4, mb="xs"),
                            dmc.Text(f"GPU: {_gpu_info()}", size="sm"),
                        ],
                        p="lg",
                        withBorder=True,
                    ),
                    dmc.Group(
                        dmc.Button(
                            "Save settings", id="set-save", leftSection=icon("tabler:device-floppy")
                        ),
                        justify="flex-end",
                    ),
                ],
                gap="md",
            ),
            file_browser("out-browser", mode="folder", title="Choose output folder"),
            html.Div(id="set-dummy"),
        ],
        size="md",
        px=0,
    )


dash.register_page(
    __name__, path="/settings", title="Settings · Swing Vision Open", order=9, layout=layout
)


@callback(
    Output("set-output-root", "value"),
    Input("out-browser-result", "data"),
    prevent_initial_call=True,
)
def _picked_folder(result):
    return result["path"] if result else no_update


@callback(
    Output("out-browser-start", "data"),
    Input("set-output-root", "value"),
)
def _browser_start(value):
    return value or None


@callback(
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("set-tool-report", "children"),
    Input("set-save", "n_clicks"),
    State("set-output-root", "value"),
    State("set-ffmpeg", "value"),
    State("set-ffprobe", "value"),
    State("set-proxy-height", "value"),
    State("set-chunk", "value"),
    State("set-cal-auto", "checked"),
    State("set-cal-threshold", "value"),
    State("set-cal-drift", "value"),
    prevent_initial_call=True,
)
def _save(n, output_root, ffmpeg, ffprobe, proxy_height, chunk, cal_auto, cal_thr, cal_drift):
    if not n:
        return no_update, no_update
    s = state.settings()
    if output_root:
        root = Path(output_root).expanduser()
        if not root.exists():
            if not root.parent.exists():
                return notification(
                    f"Parent folder of {root} does not exist.", "Not saved", "red"
                ), no_update
            root.mkdir()
        try:
            Library(root.resolve()).init()
        except Exception as exc:  # unwritable folder etc.
            return notification(f"Cannot use {root}: {exc}", "Not saved", "red"), no_update
        s.output_root = root.resolve()
    else:
        s.output_root = None
    s.ffmpeg_path = ffmpeg or "ffmpeg"
    s.ffprobe_path = ffprobe or "ffprobe"
    s.processing.proxy_height = int(proxy_height)
    s.processing.chunk_seconds = float(chunk or 120)
    s.processing.calibration_auto_accept_px = float(cal_thr or 1.5) if cal_auto else None
    s.processing.calibration_drift_px = float(cal_drift or 3.0)
    save_settings(s)
    if state.OPTIONS.start_worker:
        ensure_worker(s.output_root)
    return notification("Settings saved.", icon_name="tabler:check"), _tool_report(
        s.ffmpeg_path, s.ffprobe_path
    )
