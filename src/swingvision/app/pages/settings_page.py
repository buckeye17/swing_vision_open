"""Settings: output folder, tool paths, processing defaults, system info."""

from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc, html, no_update

from swingvision.app import state
from swingvision.app.components.file_browser import file_browser, register_file_browser
from swingvision.app.components.ui import icon, notification, page_header
from swingvision.app.units import Units
from swingvision.app.worker_control import ensure_worker
from swingvision.io.ffmpeg import available_encoders
from swingvision.models.registry import REGISTRY, weights_dir
from swingvision.settings import save_settings, settings_path
from swingvision.storage import cache as read_cache
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


ROI_INPUT_METRIC = {"min": 0.5, "max": 8, "step": 0.5}
ROI_INPUT_IMPERIAL = {"min": 2, "max": 26, "step": 1}  # ft


def _roi_to_input(roi_m: float, u: Units) -> float:
    """The beside-the-court distance in display length units (one decimal in feet)."""
    return round(u.len(roi_m), 1) if u.imperial else roi_m


def _roi_from_input(value, stored_m: float, u: Units) -> float:
    """The beside-the-court distance typed in ``u``'s length units, in metres. An unchanged
    (rounded) value keeps the stored metres, so saving doesn't drift it."""
    if value in (None, ""):
        return stored_m
    value = float(value)
    if value == 0:  # as before: an empty or zero entry keeps the stored value
        return stored_m
    if u.imperial and value == _roi_to_input(stored_m, u):
        return stored_m
    return value / u.len_factor


def _players_card(s):
    u = Units(s.units)
    p = s.processing
    models = [
        {
            "value": name,
            "label": f"{name} ({spec.description.split(':')[0].split(';')[0].lower()})"
            + ("" if spec.available() else " · downloads on first use"),
        }
        for name, spec in REGISTRY.items()
        if spec.task.startswith("person detection")
    ]
    return dmc.Paper(
        [
            dmc.Title("Player detection", order=4),
            dmc.Text(
                "People are detected on the court region of each sampled frame. A higher "
                "rate or input size costs GPU time (about 0.6× realtime at the defaults on an "
                "RTX A5000 laptop).",
                size="sm",
                c="dimmed",
                mb="sm",
            ),
            dmc.SimpleGrid(
                [
                    dmc.Select(
                        id="set-person-model",
                        label="Detector",
                        data=models,
                        value=p.person_model,
                        allowDeselect=False,
                    ),
                    dmc.Select(
                        id="set-person-rate",
                        label="Frames analysed per second",
                        data=[{"value": str(v), "label": f"{v} Hz"} for v in (10, 15, 20, 30)],
                        value=str(int(p.person_rate_hz)),
                        allowDeselect=False,
                    ),
                    dmc.Select(
                        id="set-person-input",
                        label="Input size (long side)",
                        data=[
                            {"value": "1280", "label": "1280 px (faster, misses far feet)"},
                            {"value": "1920", "label": "1920 px (recommended for 4K)"},
                            {"value": "2560", "label": "2560 px (slow)"},
                        ],
                        value=str(p.person_input_px),
                        allowDeselect=False,
                    ),
                    dmc.NumberInput(
                        id="set-roi-beside",
                        label=f"Track up to this far beside the court ({u.len_unit})",
                        description="Smaller if a neighboring court is close",
                        value=_roi_to_input(p.roi_beside_m, u),
                        **(ROI_INPUT_IMPERIAL if u.imperial else ROI_INPUT_METRIC),
                        decimalScale=1,
                    ),
                ],
                cols={"base": 1, "sm": 2},
            ),
            dmc.Text(f"Weights are cached in {weights_dir()}", size="xs", c="dimmed", mt="sm"),
        ],
        p="lg",
        withBorder=True,
    )


def _ball_card(s):
    from swingvision.ball.detectors.unet import list_weights

    p = s.processing
    opts = [
        {"value": "auto", "label": "Automatic (newest trained model, else motion)"},
        {"value": "motion", "label": "Motion (classical, no training)"},
    ]
    if s.output_root is not None:
        for card in sorted(
            list_weights(s.output_root), key=lambda c: c.get("created_at", ""), reverse=True
        ):
            opts.append(
                {
                    "value": f"unet:{card['name']}",
                    "label": f"Trained U-Net {card['name']} ({card.get('train_frames', '?')} "
                    "labeled frames)",
                }
            )
    if p.ball_detector not in [o["value"] for o in opts]:
        opts.append({"value": p.ball_detector, "label": p.ball_detector})
    return dmc.Paper(
        [
            dmc.Title("Ball detection", order=4),
            dmc.Text(
                "Train your own model from labeled clips (Labeling page, then sv train ball). "
                "A sweep rate runs the detector on part of the frames and every frame only "
                "around hits, bounces and lost-ball gaps.",
                size="sm",
                c="dimmed",
                mb="sm",
            ),
            dmc.SimpleGrid(
                [
                    dmc.Select(
                        id="set-ball-detector",
                        label="Detector",
                        data=opts,
                        value=p.ball_detector,
                        allowDeselect=False,
                    ),
                    dmc.Select(
                        id="set-ball-sweep",
                        label="Frames analysed",
                        data=[
                            {"value": "full", "label": "Every frame"},
                            {"value": "30", "label": "30 Hz + full-rate windows"},
                            {"value": "15", "label": "15 Hz + full-rate windows"},
                        ],
                        value="full" if not p.ball_sweep_hz else str(int(p.ball_sweep_hz)),
                        allowDeselect=False,
                    ),
                ],
                cols={"base": 1, "sm": 2},
            ),
        ],
        p="lg",
        withBorder=True,
    )


#: Pose models offered on the Settings page (registry names, see ``pose.pose2d.ARCHS``).
POSE_MODELS = {
    "vitpose-plus-huge": "ViTPose+-H (recommended: steadiest keypoints)",
    "vitpose-base-simple": "ViTPose-B (≈3.5× faster, noisier far player)",
}


def _pose_card(s):
    p = s.processing
    opts = [
        {
            "value": name,
            "label": label + ("" if REGISTRY[name].available() else " · downloads on first use"),
        }
        for name, label in POSE_MODELS.items()
    ]
    if p.pose_model not in POSE_MODELS:
        opts.append({"value": p.pose_model, "label": p.pose_model})
    return dmc.Paper(
        [
            dmc.Title("Swing pose", order=4),
            dmc.Text(
                "The pose network runs on every frame around each swing. ViTPose+-H costs about "
                "55 GPU-minutes per footage hour on an RTX A5000 laptop, ViTPose-B about 15 "
                "minutes. Changing it re-runs pose on sessions you reprocess.",
                size="sm",
                c="dimmed",
                mb="sm",
            ),
            dmc.Select(
                id="set-pose-model",
                label="Pose model",
                data=opts,
                value=p.pose_model,
                allowDeselect=False,
                maw=420,
            ),
        ],
        p="lg",
        withBorder=True,
    )


def _units_card(s):
    return dmc.Paper(
        [
            dmc.Title("Units", order=4),
            dmc.Text(
                "How distances, heights and speeds are shown. Data and exports stay metric.",
                size="sm",
                c="dimmed",
                mb="sm",
            ),
            dmc.SegmentedControl(
                id="set-units",
                value=s.units,
                data=[
                    {"value": "metric", "label": "Metric (m, km/h)"},
                    {"value": "imperial", "label": "Imperial (ft, mph)"},
                ],
            ),
        ],
        p="lg",
        withBorder=True,
    )


def _cache_status(s) -> str:
    if s.output_root is None:
        return "No output folder yet."
    if not read_cache.is_network_path(s.output_root):
        return "The output folder is on this computer: there's nothing to cache."
    active = read_cache.active()
    if not s.local_cache or active is None:
        return "The output folder is on a network share; the cache is off."
    return (
        f"The output folder is on a network share. Cached: "
        f"{active.size_bytes() / 1024**3:.2f} GB in {active.dir}."
    )


def _cache_card(s):
    return dmc.Paper(
        [
            dmc.Title("Local cache", order=4),
            dmc.Text(
                "When the output folder is on a network share (a NAS), the app keeps local "
                "copies of the session data and playback videos it reads, which makes pages and "
                "seeking much faster. Copies refresh by themselves when processing changes the "
                "originals; the least recently used ones go when the cache is full.",
                size="sm",
                c="dimmed",
                mb="sm",
            ),
            dmc.Switch(
                id="set-cache-on",
                label="Cache a network output folder on this computer",
                checked=s.local_cache,
                mb="sm",
            ),
            dmc.Group(
                [
                    dmc.NumberInput(
                        id="set-cache-max",
                        label="Cache size limit (GB)",
                        value=s.cache_max_gb,
                        min=1,
                        max=2000,
                        step=5,
                        w=200,
                    ),
                    dmc.Button(
                        "Clear cache",
                        id="set-cache-clear",
                        variant="default",
                        leftSection=icon("tabler:trash"),
                    ),
                ],
                align="flex-end",
            ),
            dmc.Text(_cache_status(s), id="set-cache-status", size="xs", c="dimmed", mt="xs"),
        ],
        p="lg",
        withBorder=True,
    )


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
                                "calibration, unless it fits the painted lines this well (where "
                                "the camera moved, every moved stretch must fit this well with "
                                "its own camera). Turn this on to process overnight unattended.",
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
                    _players_card(s),
                    _ball_card(s),
                    _pose_card(s),
                    _units_card(s),
                    _cache_card(s),
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
            dcc.Store(id="set-units-shown", data=s.units),  # what the form's inputs are in
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
    Output("set-roi-beside", "value"),
    Output("set-roi-beside", "label"),
    Output("set-roi-beside", "min"),
    Output("set-roi-beside", "max"),
    Output("set-roi-beside", "step"),
    Output("set-units-shown", "data"),
    Input("set-save", "n_clicks"),
    State("set-output-root", "value"),
    State("set-ffmpeg", "value"),
    State("set-ffprobe", "value"),
    State("set-proxy-height", "value"),
    State("set-chunk", "value"),
    State("set-cal-auto", "checked"),
    State("set-cal-threshold", "value"),
    State("set-cal-drift", "value"),
    State("set-person-model", "value"),
    State("set-person-rate", "value"),
    State("set-person-input", "value"),
    State("set-roi-beside", "value"),
    State("set-ball-detector", "value"),
    State("set-ball-sweep", "value"),
    State("set-pose-model", "value"),
    State("set-units", "value"),
    State("set-cache-on", "checked"),
    State("set-cache-max", "value"),
    State("set-units-shown", "data"),
    prevent_initial_call=True,
)
def _save(
    n,
    output_root,
    ffmpeg,
    ffprobe,
    proxy_height,
    chunk,
    cal_auto,
    cal_thr,
    cal_drift,
    person_model,
    person_rate,
    person_input,
    roi_beside,
    ball_detector,
    ball_sweep,
    pose_model,
    units,
    cache_on,
    cache_max,
    units_shown,
):
    if not n:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )
    s = state.settings()
    # The units the form's inputs are in: as rendered, or as the last save switched them to.
    shown = Units(units_shown if units_shown in ("metric", "imperial") else s.units)
    if output_root:
        root = Path(output_root).expanduser()
        if not root.exists():
            if not root.parent.exists():
                return (
                    notification(f"Parent folder of {root} does not exist.", "Not saved", "red"),
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            root.mkdir()
        try:
            Library(root.resolve()).init()
        except Exception as exc:  # unwritable folder etc.
            return (
                notification(f"Cannot use {root}: {exc}", "Not saved", "red"),
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )
        s.output_root = root.resolve()
    else:
        s.output_root = None
    s.ffmpeg_path = ffmpeg or "ffmpeg"
    s.ffprobe_path = ffprobe or "ffprobe"
    s.processing.proxy_height = int(proxy_height)
    s.processing.chunk_seconds = float(chunk or 120)
    s.processing.calibration_auto_accept_px = float(cal_thr or 1.5) if cal_auto else None
    s.processing.calibration_drift_px = float(cal_drift or 3.0)
    s.processing.person_model = person_model or s.processing.person_model
    s.processing.person_rate_hz = float(person_rate or s.processing.person_rate_hz)
    s.processing.person_input_px = int(person_input or s.processing.person_input_px)
    s.processing.roi_beside_m = _roi_from_input(roi_beside, s.processing.roi_beside_m, shown)
    s.processing.ball_detector = ball_detector or s.processing.ball_detector
    s.processing.ball_sweep_hz = None if ball_sweep in (None, "full") else float(ball_sweep)
    s.processing.pose_model = pose_model or s.processing.pose_model
    s.units = units if units in ("metric", "imperial") else s.units
    s.local_cache = bool(cache_on)
    s.cache_max_gb = max(1.0, float(cache_max or s.cache_max_gb))
    save_settings(s)
    state.settings()  # apply the cache settings now
    if state.OPTIONS.start_worker:
        ensure_worker(s.output_root)
    u = Units(s.units)  # redraw the inputs in the units just saved
    bounds = ROI_INPUT_IMPERIAL if u.imperial else ROI_INPUT_METRIC
    return (
        notification("Settings saved.", icon_name="tabler:check"),
        _tool_report(s.ffmpeg_path, s.ffprobe_path),
        _roi_to_input(s.processing.roi_beside_m, u),
        f"Track up to this far beside the court ({u.len_unit})",
        bounds["min"],
        bounds["max"],
        bounds["step"],
        s.units,
    )


@callback(
    Output("set-cache-status", "children"),
    Input("set-cache-clear", "n_clicks"),
    prevent_initial_call=True,
)
def _clear_cache(n):
    active = read_cache.active()
    if active is not None:
        active.clear()
    return _cache_status(state.settings())
