"""The Calibrate page's Speed tab (PLAN.md §7.12, M7c): review the session's net-tape
reference serves, and the recording device's speed calibration.

For each candidate: the clip slowed down, the frames at the tape zoomed with the tape model
and the ball, the waveform ±60 ms around each sound with its onset (drag the red line, or
nudge it in 0.1 ms steps) and the window it was searched in, and the numbers. Accept or
reject it; any serve can also be marked as a tape hit. Decisions and moved onsets go to
``edits.json`` (and ``training/speed_refs/``); ``speed_refs`` reruns in-app.
"""

from __future__ import annotations

import base64
import threading
from collections import OrderedDict
from functools import lru_cache

import cv2
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from swingvision import services
from swingvision.app import state, units
from swingvision.app.components.ui import icon, notification, stat_grid, stat_tile
from swingvision.ball import physics as ph
from swingvision.ball import speed_refs as sr
from swingvision.court import calibration as calib
from swingvision.storage import tables
from swingvision.storage.edits import EditConflict
from swingvision.storage.fsutil import read_json

WAVE_HALF_S = 0.06
CROP = (180, 110)  # full-resolution px around the tape point
ZOOM = 3
WAVE_CONFIG = {"displayModeBar": False, "scrollZoom": True, "edits": {"shapePosition": True}}
STATUS_COLORS = {"accepted": "teal", "rejected": "red", "candidate": "yellow"}
_frames_cache: OrderedDict = OrderedDict()
_lock = threading.Lock()


def _fmt_t(t: float | None) -> str:
    if t is None:
        return "–"
    m, s = divmod(float(t), 60)
    return f"{int(m)}:{s:04.1f}"


def load_refs(session) -> list[dict]:
    if not session.speed_refs_path.exists():
        return []
    return tables.read_table(session.speed_refs_path).to_pylist()


def ref_summary(session) -> dict:
    p = session.speed_refs_summary_path
    return read_json(p) if p.exists() else {}


@lru_cache(maxsize=4)
def _audio(path: str, _mtime: float) -> sr.AudioClip:
    return sr.AudioClip.from_file(path)


def audio_of(session) -> sr.AudioClip | None:
    p = session.audio_path
    return _audio(str(p), p.stat().st_mtime) if p.exists() else None


def near_serves(session) -> list[dict]:
    """The session's near-end serves, for marking one as a tape hit."""
    if not session.swings_path.exists():
        return []
    rows = tables.read_table(
        session.swings_path, columns=["swing_id", "stroke_type", "side", "t_contact"]
    ).to_pylist()
    return [r for r in rows if r["stroke_type"] == "serve" and r["side"] == -1 and r["t_contact"]]


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def ref_label(r: dict) -> str:
    kind = "let" if "velocity_change" in (r["flags"] or []) else (r["end_kind"] or "?")
    ratio = "–" if r["ratio"] is None else f"{r['ratio']:.3f}"
    return f"#{r['ref_id']} · {_fmt_t(r['t_contact'])} · {kind} · ratio {ratio} · {r['status']}"


def waveform_figure(audio, r: dict, sound: str) -> tuple[go.Figure, float | None]:
    """The waveform around one sound with its onset (a draggable line) and its search
    window; returns the figure and the time its x axis (ms) counts from."""
    p = sr.RefParams()
    onset = r.get(f"t_{sound}_audio")
    pred = r.get(f"t_{sound}_pred_audio")
    ref_t = onset if onset is not None else pred
    fig = go.Figure()
    if audio is None or ref_t is None:
        fig.update_layout(height=170, margin={"l": 40, "r": 10, "t": 26, "b": 30})
        return fig, None
    tt, hp, env = sr.waveform(audio, ref_t, WAVE_HALF_S, p, 2400)
    x = (tt - ref_t) * 1000
    fig.add_trace(
        go.Scatter(x=x, y=hp, mode="lines", line={"width": 0.7, "color": "#74c0fc"},
                   name="waveform", hoverinfo="skip")
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=x, y=env, mode="lines", line={"width": 1.4, "color": "#ff922b"},
                   name="envelope", yaxis="y2", hovertemplate="%{x:.2f} ms<extra></extra>")
    )  # fmt: skip
    half = p.racket_window_s if sound == "racket" else p.tape_window_s
    # The onset line is always shapes[0] (a drag reports its new x as ``shapes[0].x0``);
    # without an onset it waits, dashed, at the predicted arrival for the user to place it.
    dash = None if onset is not None else "dash"
    shapes = [
        {"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1, "yref": "paper",
         "line": {"color": "#fa5252", "width": 3, "dash": dash}}
    ]  # fmt: skip
    if pred is not None:
        a, b = (pred - half - ref_t) * 1000, (pred + half - ref_t) * 1000
        shapes.append(
            {"type": "rect", "x0": a, "x1": b, "y0": 0, "y1": 1, "yref": "paper",
             "fillcolor": "rgba(250, 200, 50, 0.10)", "line": {"width": 0}, "layer": "below"}
        )  # fmt: skip
    snr = r.get(f"snr_{sound}_db")
    title = f"{'Racket' if sound == 'racket' else 'Tape'} sound" + (
        f" · onset {onset:.4f} s · SNR {snr:.0f} dB" if onset is not None and snr else ""
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 12}, "x": 0.01},
        height=180,
        margin={"l": 40, "r": 40, "t": 28, "b": 30},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=shapes,
        xaxis={"title": "ms (drag the red line to move the onset)", "range": [-60, 60],
               "zeroline": False, "gridcolor": "rgba(128,128,128,0.15)"},
        yaxis={"showticklabels": False, "zeroline": False},
        yaxis2={"overlaying": "y", "side": "right", "showticklabels": False, "rangemode": "tozero"},
        dragmode="pan",
    )  # fmt: skip
    return fig, ref_t


def _frames_near(session, frame: int, t: float) -> dict[int, np.ndarray]:
    key = (str(session.path), frame)
    with _lock:
        if key in _frames_cache:
            _frames_cache.move_to_end(key)
            return _frames_cache[key]
    from pathlib import Path

    from swingvision.io.frames import open_source

    config = session.load_config()
    fps = config.video.fps_avg if config.video else 60.0
    out: dict[int, np.ndarray] = {}
    with open_source(Path(config.source.path), config.video, "pyav", "cpu") as src:
        for f in src.frames(t - 1.6 / fps, t + 1.6 / fps):
            if abs(f.index - frame) <= 1:
                out[f.index] = f.image.permute(1, 2, 0).numpy()
    with _lock:
        _frames_cache[key] = out
        while len(_frames_cache) > 8:
            _frames_cache.popitem(last=False)
    return out


def tape_frames(session, r: dict) -> list[dict]:
    """The frames around the crossing, cut around the tape point, zoomed, with the tape model
    (magenta) and the ball's detections (cyan)."""
    if r["t_cross"] is None or r["tape_x"] is None:
        return []
    cal = calib.load(session.calibration_path)
    cam = calib.camera_at(cal, r["t_cross"])
    track = tables.read_table(session.ball_track_path, columns=["frame", "t_s", "x", "y", "source"])
    f_all = track.column("frame").to_numpy()
    t_all = track.column("t_s").to_numpy()
    k = int(np.argmin(np.abs(t_all - r["t_cross"])))
    config = session.load_config()
    fps = config.video.fps_avg if config.video else 60.0
    frame = int(f_all[k] + round((r["t_cross"] - t_all[k]) * fps))
    imgs = _frames_near(session, frame, float(r["t_cross"]))
    tp = cam.project(np.array([[r["tape_x"], 0.0, r["tape_z"]]]))[0]
    xs = np.linspace(r["tape_x"] - 2.0, r["tape_x"] + 2.0, 80)
    tape_px = cam.project(np.column_stack([xs, np.zeros_like(xs), ph.net_height(xs)]))
    xd, yd = track.column("x").to_numpy(), track.column("y").to_numpy()
    w, h = CROP
    x0, y0 = round(tp[0] - w / 2), round(tp[1] - h / 2)
    out = []
    for f in sorted(imgs):
        img = imgs[f]
        crop = np.zeros((h, w, 3), np.uint8)
        H, W = img.shape[:2]
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if xb > xa and yb > ya:
            crop[ya - y0 : yb - y0, xa - x0 : xb - x0] = img[ya:yb, xa:xb]
        z = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
        pts = np.round((tape_px - [x0, y0]) * ZOOM).astype(np.int32)
        cv2.polylines(z, [pts.reshape(-1, 1, 2)], False, (230, 60, 230), 1, cv2.LINE_AA)
        for i in np.flatnonzero(f_all == f):
            if np.isfinite(xd[i]):
                p = (round((xd[i] - x0) * ZOOM), round((yd[i] - y0) * ZOOM))
                cv2.drawMarker(z, p, (60, 220, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(z, cv2.COLOR_RGB2BGR))
        src = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""
        out.append({"frame": f, "src": src, "tape": f == frame})
    return out


def numbers(r: dict, summary: dict) -> html.Div:
    u = units.current()

    def ms(v, nd=1):
        return "–" if v is None else f"{1000 * v:.{nd}f} ms"

    tiles = [
        stat_tile("Flight time", ms(r["dt_s"]), f"± {ms(r['dt_sigma_s'], 2)}, sound delays out"),
        stat_tile(
            "Path",
            "–" if r["path_m"] is None else u.len_str(r["path_m"], 2),
            None if r["path_sigma_m"] is None else f"± {100 * r['path_sigma_m']:.0f} cm",
        ),
        stat_tile("Reference speed", u.speed_str(r["v_ref_kmh"]), "path / flight time"),
        stat_tile("Fitted speed", u.speed_str(r["v_fit_kmh"]), "the 3D fit, same part"),
        stat_tile(
            "Ratio",
            "–" if r["ratio"] is None else f"{r['ratio']:.4f}",
            None if r["ratio_sigma"] is None else f"± {r['ratio_sigma']:.4f}",
        ),
        stat_tile(
            "Over the tape",
            "–" if r["clearance_m"] is None else f"{100 * r['clearance_m']:+.0f} cm",
            "fitted ball center",
        ),
        stat_tile(
            "Audio offset",
            ms(r["av_offset_s"]),
            f"session {ms(summary.get('av_offset_s'))}",
        ),
    ]
    return stat_grid(tiles, plain=True)


def device_panel(session, config):
    """The recording device and its calibration, with *Update calibration*."""
    lib = state.library()
    key = config.device_key
    if lib is None or not key:
        return dmc.Alert(
            "This video's recording device isn't known (no make / model tags), so its speeds "
            "can't be calibrated per device.",
            color="gray",
            p="xs",
        )
    dev = lib.get_device(key)
    cal = lib.active_calibration(key)
    refs = services.device_references(state.settings(), key)
    n_sessions = len({r["session_id"] for r in refs})
    picked = config.speed_calibration
    if cal is not None:
        status = (
            f"× {cal['k']:.4f} ± {cal['k_sigma'] / cal['k']:.2%} "
            f"({'rolling shutter τ ' + format(1000 * cal['tau_s'], '.1f') + ' ms, ' if cal['tau_s'] is not None else ''}"
            f"version {cal['version']}, {cal['n_refs']} references)"
        )
    else:
        status = "not calibrated yet"
    note = (
        f"{len(refs)} accepted reference{'' if len(refs) == 1 else 's'} from {n_sessions} "
        f"session{'' if n_sessions == 1 else 's'}"
        + (f" (at least {sr.MIN_REFS} are needed)" if len(refs) < sr.MIN_REFS else "")
    )
    choices = [
        {"value": "auto", "label": "The device's active calibration"},
        {"value": "none", "label": "No calibration"},
    ] + [
        {"value": c["id"], "label": f"Version {c['version']}: × {c['k']:.4f}"}
        for c in lib.list_calibrations(key)
    ]
    override = dmc.Select(
        id="spd-cal-pick",
        label="This session's speeds use",
        data=choices,
        value=picked or "auto",
        allowDeselect=False,
        size="xs",
        w=260,
    )
    return dmc.Paper(
        dmc.Group(
            [
                dmc.Stack(
                    [
                        dmc.Group(
                            [
                                icon("tabler:device-mobile", 16),
                                dmc.Text(services.device_name(dev), fw=600, size="sm"),
                                dmc.Badge(
                                    "Calibrated" if cal else "Uncalibrated",
                                    color="teal" if cal else "yellow",
                                    variant="light",
                                ),
                            ],
                            gap=6,
                        ),
                        dmc.Text(f"Speed calibration: {status}", size="xs"),
                        dmc.Text(note, size="xs", c="dimmed"),
                        override,
                    ],
                    gap=2,
                ),
                dmc.Group(
                    [
                        dmc.NumberInput(
                            id="spd-temp",
                            label="Air temperature",
                            value=config.air_temp_c,
                            placeholder="20 °C assumed",
                            suffix=" °C",
                            min=-30,
                            max=50,
                            step=1,
                            decimalScale=1,
                            size="xs",
                            w=150,
                        ),
                        dmc.Button(
                            "Save",
                            id="spd-temp-save",
                            size="xs",
                            variant="default",
                        ),
                        dmc.Button(
                            "Update calibration",
                            id="spd-calibrate",
                            size="xs",
                            leftSection=icon("tabler:target-arrow", 14),
                            disabled=len(refs) < sr.MIN_REFS,
                        ),
                        dmc.Anchor("Devices in Settings", href="/settings#devices", size="xs"),
                    ],
                    gap="xs",
                    align="flex-end",
                ),
            ],
            justify="space-between",
            align="flex-start",
        ),
        p="sm",
        withBorder=True,
    )


def speed_panel(session, config):
    """The Speed tab's content (rebuilt when ``spd-version`` changes)."""
    refs = load_refs(session)
    summary = ref_summary(session)
    serves = near_serves(session)
    marked = {round(r["t_contact"], 1) for r in refs}
    mark_opts = [
        {"value": f"{r['t_contact']:.3f}", "label": f"Serve at {_fmt_t(r['t_contact'])}"}
        for r in serves
        if round(r["t_contact"], 1) not in marked
    ]
    head = [device_panel(session, config)]
    if not session.speed_refs_path.exists():
        head.append(
            dmc.Alert(
                [
                    dmc.Text(
                        "The session's serves haven't been checked for net-tape hits yet "
                        "(the Speed references stage)."
                    ),
                    dmc.Button("Find them now", id="spd-run", size="xs", mt="xs"),
                ],
                color="blue",
                p="xs",
            )
        )
        return dmc.Stack(head, gap="sm")
    counts = (
        f"{summary.get('references', 0)} reference serves from "
        f"{summary.get('near_serves', 0)} near-end serves · audio "
        f"{1000 * (summary.get('av_offset_s') or 0):.0f} ms behind the video "
        f"({summary.get('av_offset_serves', 0)} serves) · sound "
        f"{summary.get('sound_speed_mps', 343):.1f} m/s"
    )
    picker = dmc.Group(
        [
            dmc.Select(
                id="spd-select",
                data=[{"value": str(r["ref_id"]), "label": ref_label(r)} for r in refs],
                value=str(refs[0]["ref_id"]) if refs else None,
                placeholder="No reference serves found" if not refs else None,
                allowDeselect=False,
                size="xs",
                w=420,
                label="Reference serve",
            ),
            dmc.Select(
                id="spd-mark-select",
                data=mark_opts,
                placeholder="Pick a serve",
                searchable=True,
                size="xs",
                w=200,
                label="Mark a serve as a tape hit",
            ),
            dmc.Button("Mark", id="spd-mark", size="xs", variant="default"),
        ],
        gap="xs",
        align="flex-end",
    )
    return dmc.Stack(
        [
            *head,
            dmc.Text(counts, size="xs", c="dimmed"),
            picker,
            dcc.Loading(html.Div(id="spd-detail"), type="dot", delay_show=300),
        ],
        gap="sm",
    )


def detail(session, config, r: dict) -> list:
    summary = ref_summary(session)
    audio = audio_of(session)
    figs = {}
    refs_t = {}
    for sound in ("racket", "tape"):
        figs[sound], refs_t[sound] = waveform_figure(audio, r, sound)
    try:
        frames = tape_frames(session, r)
        frame_err = None
    except Exception as exc:  # the source video moved, a decode error
        frames, frame_err = [], str(exc)
    flags = r["flags"] or []
    status = r["status"]
    clip = None
    if session.proxy_path.exists() and r["t_cross"] is not None:
        clip = html.Video(
            id="spd-clip",
            src=f"/media/{config.id}/proxy.mp4",
            controls=True,
            muted=True,
            preload="auto",
            style={"width": "100%", "background": "#000", "borderRadius": 6},
            **{"data-a": f"{r['t_contact'] - 0.4:.3f}", "data-b": f"{r['t_cross'] + 0.5:.3f}"},
        )

    def nudges(sound):
        return dmc.Group(
            [
                dmc.Button(
                    lab,
                    id={"type": "spd-nudge", "sound": sound, "step": step},
                    size="compact-xs",
                    variant="default",
                )
                for lab, step in (("−1 ms", -1.0), ("−0.1", -0.1), ("+0.1", 0.1), ("+1 ms", 1.0))
            ]
            + [
                dmc.Button(
                    "Detected",
                    id={"type": "spd-nudge", "sound": sound, "step": 0},
                    size="compact-xs",
                    variant="subtle",
                )
            ],
            gap=4,
        )

    strip = [
        dmc.Stack(
            [
                dmc.Text(
                    f"frame {f['frame']}" + (" · at the tape" if f["tape"] else ""),
                    size="xs",
                    c="dimmed",
                    ta="center",
                ),
                html.Img(src=f["src"], style={"width": "100%", "borderRadius": 4}),
            ],
            gap=2,
            style={"flex": 1},
        )
        for f in frames
    ]
    return [
        dcc.Store(id="spd-wave-refs", data=refs_t),
        dcc.Store(id="spd-ref-t", data=r["t_contact"]),
        dmc.Grid(
            [
                dmc.GridCol(
                    dmc.Stack(
                        [
                            clip
                            if clip is not None
                            else dmc.Text("No playback proxy.", size="xs", c="dimmed"),
                            dmc.Text(
                                "Plays at ¼ speed, looping from the toss to past the net.",
                                size="xs",
                                c="dimmed",
                            ),
                            dmc.Group(strip, gap=4, grow=True, wrap="nowrap", align="flex-start")
                            if strip
                            else dmc.Text(
                                f"Frames unavailable{': ' + frame_err if frame_err else ''}.",
                                size="xs",
                                c="dimmed",
                            ),
                            dmc.Text(
                                "Magenta: the tape model; cyan: the ball's detections.",
                                size="xs",
                                c="dimmed",
                            ),
                        ],
                        gap=6,
                    ),
                    span={"base": 12, "lg": 5},
                ),
                dmc.GridCol(
                    dmc.Stack(
                        [
                            dcc.Graph(
                                id="spd-wave-racket",
                                figure=figs["racket"],
                                config=WAVE_CONFIG,
                            ),
                            nudges("racket"),
                            dcc.Graph(
                                id="spd-wave-tape",
                                figure=figs["tape"],
                                config=WAVE_CONFIG,
                            ),
                            nudges("tape"),
                        ],
                        gap=4,
                    ),
                    span={"base": 12, "lg": 7},
                ),
            ],
            gutter="md",
        ),
        numbers(r, summary),
        dmc.Group(
            [
                dmc.Badge(status, color=STATUS_COLORS.get(status, "gray"), variant="light"),
                dmc.Badge(
                    "found by the rules" if r["source"] == "auto" else "marked by you",
                    color="gray",
                    variant="outline",
                ),
                *(dmc.Badge(f, color="gray", variant="light", tt="none") for f in flags),
            ],
            gap=4,
        ),
        dmc.Group(
            [
                dmc.Button(
                    "Accept",
                    id="spd-accept",
                    size="xs",
                    color="teal",
                    leftSection=icon("tabler:check", 14),
                ),
                dmc.Button(
                    "Reject",
                    id="spd-reject",
                    size="xs",
                    color="red",
                    variant="light",
                    leftSection=icon("tabler:x", 14),
                ),
                dmc.Button("Reset review", id="spd-reset", size="xs", variant="default"),
            ],
            gap="xs",
        ),
        dmc.Text(
            "Accept a serve that clearly hit the tape: a sharp tick where the ball meets the "
            "tape in the frames. Reject a net-mesh hit (a dull sound, the net bellying back), "
            "a sound that isn't the tape, or a ball that passed clear of it.",
            size="xs",
            c="dimmed",
        ),
    ]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _session(sid):
    found = state.session_for(sid or "")
    return None if found is None else found[2]


@callback(
    Output("spd-body", "children"),
    Input("spd-version", "data"),
    State("cal-sid", "data"),
    prevent_initial_call=True,
)
def _rebuild(_version, sid):
    session = _session(sid)
    if session is None:
        return no_update
    return speed_panel(session, session.load_config())


@callback(
    Output("spd-detail", "children"),
    Input("spd-select", "value"),
    State("cal-sid", "data"),
)
def _detail(ref_id, sid):
    session = _session(sid)
    if session is None or ref_id is None:
        return None
    rows = [r for r in load_refs(session) if str(r["ref_id"]) == str(ref_id)]
    if not rows:
        return None
    return detail(session, session.load_config(), rows[0])


clientside_callback(
    """
    function(children) {
        const v = document.getElementById("spd-clip");
        if (!v) { return window.dash_clientside.no_update; }
        const a = parseFloat(v.dataset.a), b = parseFloat(v.dataset.b);
        const start = () => { v.playbackRate = 0.25; v.currentTime = Math.max(0, a); v.play(); };
        if (v.readyState >= 1) { start(); } else { v.addEventListener("loadedmetadata", start, {once: true}); }
        if (!v._svLoop) {
            v._svLoop = true;
            v.addEventListener("timeupdate", () => {
                const b2 = parseFloat(v.dataset.b), a2 = parseFloat(v.dataset.a);
                if (v.currentTime > b2) { v.currentTime = Math.max(0, a2); }
            });
            v.addEventListener("play", () => { v.playbackRate = 0.25; });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("spd-sink", "data"),
    Input("spd-detail", "children"),
    prevent_initial_call=True,
)


def _refresh(sid: str) -> tuple[str, str]:
    """Rerun ``speed_refs`` (cheap) → (message, color)."""
    status, job = services.refresh_practice(state.settings(), sid, target="speed_refs")
    if status == "ran":
        return "Updated.", "green"
    if status == "queued":
        return f"Saved; queued job #{job} to process the session first.", "blue"
    return "Saved; the running job picks it up.", "blue"


def _save(sid, t, version, **change):
    try:
        services.edit_speed_ref(state.settings(), sid, t, **change)
        msg, color = _refresh(sid)
    except (EditConflict, ValueError, RuntimeError) as exc:
        return no_update, notification(str(exc), "Couldn't save", color="red")
    return (version or 0) + 1, notification(msg, "Speed reference", color=color)


@callback(
    Output("spd-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("spd-accept", "n_clicks"),
    Input("spd-reject", "n_clicks"),
    Input("spd-reset", "n_clicks"),
    State("spd-ref-t", "data"),
    State("cal-sid", "data"),
    State("spd-version", "data"),
    prevent_initial_call=True,
)
def _review(accept, reject, reset, t, sid, version):
    clicks = {"spd-accept": accept, "spd-reject": reject, "spd-reset": reset}
    if not clicks.get(ctx.triggered_id) or t is None:
        return no_update, no_update
    if ctx.triggered_id == "spd-reset":
        return _save(sid, t, version, status=None, marked=False, t_racket=None, t_tape=None)
    status = "accepted" if ctx.triggered_id == "spd-accept" else "rejected"
    return _save(sid, t, version, status=status)


def drag_onset(x_ms: float | None, ref_t: float | None) -> float | None:
    """The onset (audio s) a drag of the red line put it at: ``x_ms`` on an axis counting
    from ``ref_t``."""
    if x_ms is None or ref_t is None:
        return None
    return round(ref_t + float(x_ms) / 1000, 5)


@callback(
    Output("spd-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("spd-drag", "data"),
    Input({"type": "spd-nudge", "sound": ALL, "step": ALL}, "n_clicks"),
    State("spd-wave-refs", "data"),
    State("spd-ref-t", "data"),
    State("spd-select", "value"),
    State("cal-sid", "data"),
    State("spd-version", "data"),
    prevent_initial_call=True,
)
def _onset(drag, _nudges, wave_refs, t, ref_id, sid, version):
    trig = ctx.triggered_id
    if trig is None or t is None:
        return no_update, no_update
    if isinstance(trig, dict):
        if not ctx.triggered[0].get("value"):
            return no_update, no_update  # mounting
        sound, step = trig["sound"], float(trig["step"])
        if step == 0:
            return _save(sid, t, version, **{f"t_{sound}": None})
        session = _session(sid)
        rows = [r for r in load_refs(session) if str(r["ref_id"]) == str(ref_id)] if session else []
        cur = rows[0][f"t_{sound}_audio"] if rows else None
        if cur is None:
            return no_update, no_update
        return _save(sid, t, version, **{f"t_{sound}": cur + step / 1000})
    # A drag of the onset line (assets/speed_wave.js): {sound, x (ms), n}.
    sound = (drag or {}).get("sound")
    new = drag_onset((drag or {}).get("x"), (wave_refs or {}).get(sound))
    if sound not in ("racket", "tape") or new is None:
        return no_update, no_update
    return _save(sid, t, version, **{f"t_{sound}": new})


@callback(
    Output("spd-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("spd-mark", "n_clicks"),
    State("spd-mark-select", "value"),
    State("cal-sid", "data"),
    State("spd-version", "data"),
    prevent_initial_call=True,
)
def _mark(n, value, sid, version):
    if not n or not value:
        return no_update, no_update
    return _save(sid, float(value), version, marked=True)


@callback(
    Output("spd-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("spd-temp-save", "n_clicks"),
    State("spd-temp", "value"),
    State("cal-sid", "data"),
    State("spd-version", "data"),
    prevent_initial_call=True,
)
def _temp(n, temp, sid, version):
    if not n:
        return no_update, no_update
    try:
        services.set_air_temperature(
            state.settings(), sid, None if temp in (None, "") else float(temp)
        )
        msg, color = _refresh(sid)
    except (ValueError, RuntimeError) as exc:
        return no_update, notification(str(exc), "Couldn't save", color="red")
    return (version or 0) + 1, notification(msg, "Air temperature", color=color)


@callback(
    Output("spd-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("spd-run", "n_clicks"),
    State("cal-sid", "data"),
    State("spd-version", "data"),
    prevent_initial_call=True,
)
def _run(n, sid, version):
    if not n:
        return no_update, no_update
    try:
        msg, color = _refresh(sid)
    except (ValueError, RuntimeError) as exc:
        return no_update, notification(str(exc), "Couldn't run it", color="red")
    return (version or 0) + 1, notification(msg, "Speed references", color=color)


@callback(
    Output("spd-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("spd-calibrate", "n_clicks"),
    State("cal-sid", "data"),
    State("spd-version", "data"),
    prevent_initial_call=True,
)
def _calibrate(n, sid, version):
    if not n:
        return no_update, no_update
    s = state.settings()
    session = _session(sid)
    key = session.load_config().device_key if session else None
    if not key:
        return no_update, notification("The recording device isn't known.", color="red")
    try:
        cal, row, refs = services.calibrate_device(s, key)
    except ValueError as exc:
        return no_update, notification(str(exc), "Couldn't calibrate", color="red")
    if cal is None:
        return no_update, notification(
            f"{len(refs)} accepted references: at least {sr.MIN_REFS} are needed.",
            "Not enough references",
            color="yellow",
        )
    done = services.refresh_device(s, key)
    msg = (
        f"× {cal.k:.4f} ± {cal.k_sigma / cal.k:.2%} from {cal.n_refs} references "
        f"(version {row['version']}). Updated {len(done['ran'])} sessions"
        + (f", queued {len(done['queued'])}" if done["queued"] else "")
        + "."
    )
    return (version or 0) + 1, notification(msg, "Speed calibration")


@callback(
    Output("spd-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("spd-cal-pick", "value"),
    State("cal-sid", "data"),
    State("spd-version", "data"),
    prevent_initial_call=True,
)
def _pick_calibration(value, sid, version):
    session = _session(sid)
    if session is None or value is None:
        return no_update, no_update
    current = session.load_config().speed_calibration or "auto"
    if value == current:
        return no_update, no_update
    s = state.settings()
    try:
        services.set_session_speed_calibration(s, sid, None if value == "auto" else value)
        status, job = services.refresh_practice(s, sid)
    except (ValueError, RuntimeError) as exc:
        return no_update, notification(str(exc), "Couldn't change it", color="red")
    msg = "Speeds updated." if status == "ran" else f"Saved; job #{job} updates the speeds."
    return (version or 0) + 1, notification(msg, "Speed calibration")
