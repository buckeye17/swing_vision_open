"""The Swings page's serve contact view (PLAN.md §7.11, M7b): the contact frame and the frames
on either side, cut around the ball and around the front foot, with the toss path, the
contact point and the toe drawn in, so a wrong toe or an off-by-one contact frame is easy to
spot. The toe crop of the contact frame is a figure: a click there places the toe.

Frames are decoded from the source video on the CPU (one short seek per serve) and kept in a
small in-memory cache.
"""

from __future__ import annotations

import base64
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import html

from swingvision.app import units
from swingvision.app.components.swings_view import num
from swingvision.court import calibration as calib
from swingvision.pose import serve_contact as sc
from swingvision.storage import tables
from swingvision.storage.fsutil import read_json

#: Crops (full-resolution px): around the ball, around the toe.
BALL_BOX = (200, 200)
TOE_BOX = (240, 160)
SCALE = 2
_cache: OrderedDict = OrderedDict()
_lock = threading.Lock()
CACHE_SIZE = 12


def contact_row(session, swing_id: int) -> dict | None:
    """The serve's ``serve_contact`` row (None: not a serve, or not processed yet)."""
    if not session.serve_contact_path.exists():
        return None
    rows = tables.read_table(
        session.serve_contact_path, filters=[("swing_id", "=", int(swing_id))]
    ).to_pylist()
    return rows[0] if rows else None


def _frames(session, frame: int, t: float) -> dict[int, np.ndarray]:
    """Frames ``frame - 1 … frame + 1`` (RGB, full resolution), cached."""
    key = (str(session.path), frame)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    from swingvision.io.frames import open_source

    config = session.load_config()
    out: dict[int, np.ndarray] = {}
    fps = config.video.fps_avg if config.video else 60.0
    with open_source(Path(config.source.path), config.video, "pyav", "cpu") as src:
        for f in src.frames(t - 1.6 / fps, t + 1.6 / fps):
            if abs(f.index - frame) <= 1:
                out[f.index] = f.image.permute(1, 2, 0).numpy()
    with _lock:
        _cache[key] = out
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def _crop(
    img: np.ndarray, cx: float, cy: float, box: tuple[int, int]
) -> tuple[np.ndarray, int, int]:
    w, h = box
    x0, y0 = round(cx - w / 2), round(cy - h / 2)
    H, W = img.shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb > xa and yb > ya:
        out[ya - y0 : yb - y0, xa - x0 : xb - x0] = img[ya:yb, xa:xb]
    return out, x0, y0


def _jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _marks(img, pts: list[tuple[float, float, tuple, str]], x0: int, y0: int) -> np.ndarray:
    z = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for x, y, color, kind in pts:
        p = (round((x - x0) * SCALE), round((y - y0) * SCALE))
        if kind == "circle":
            cv2.circle(z, p, 9 * SCALE // 2, color, 2, cv2.LINE_AA)
        elif kind == "dot":
            cv2.circle(z, p, 2, color, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(z, p, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    return z


def toss_points(session, row: dict, times: list[float]) -> list[np.ndarray | None]:
    """The toss path's position at ``times`` (court m), from the swing's toss fit."""
    path = session.swings_summary_path
    toss = (
        (read_json(path).get("tosses") or {}).get(str(row["swing_id"])) if path.exists() else None
    )
    fit = sc.toss_fit(sc.toss_from_dict(toss)) if toss else None
    if fit is None:
        return [None] * len(times)
    pos, _ = sc.path_at(fit, np.asarray(times, float))
    return list(pos)


def view(session, row: dict) -> dict:
    """Crops of the frames around the contact (base64 JPEG) with their geometry."""
    frame, t = row["frame_contact"], row["t_contact"]
    if frame is None or t is None:
        return {"frames": []}
    cal = calib.load(session.calibration_path)
    cam = calib.camera_at(cal, t)
    config = session.load_config()
    fps = config.video.fps_avg if config.video else 60.0
    imgs = _frames(session, int(frame), float(t))
    ball = (row["ball_px_x"], row["ball_px_y"]) if row["ball_px_x"] is not None else None
    toe = (row["toe_px_x"], row["toe_px_y"]) if row["toe_px_x"] is not None else None
    frames = sorted(imgs)
    path_t = np.linspace(t - 6 / fps, t + 2 / fps, 17)
    path = [p for p in toss_points(session, row, list(path_t)) if p is not None]
    path_px = cam.project(np.array(path)) if path else np.zeros((0, 2))
    out = []
    for f in frames:
        img = imgs[f]
        item = {"frame": f, "contact": f == frame}
        if ball is not None and np.isfinite(ball).all() and ball[1] > -BALL_BOX[1] / 2:
            c, x0, y0 = _crop(img, ball[0], max(ball[1], BALL_BOX[1] / 2), BALL_BOX)
            pts = [(px[0], px[1], (255, 200, 0), "dot") for px in path_px]
            if f == frame:
                pts.append((ball[0], ball[1], (0, 255, 255), "circle"))
            item["ball"] = _jpeg(_marks(c, pts, x0, y0))
        if toe is not None:
            c, x0, y0 = _crop(img, toe[0], toe[1] - 20, TOE_BOX)
            if f == frame:
                item["toe_img"] = _jpeg(
                    cv2.resize(c, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
                )
                item["toe_origin"] = [x0, y0]
            item["toe"] = _jpeg(_marks(c, [(toe[0], toe[1], (255, 60, 60), "cross")], x0, y0))
        out.append(item)
    return {"frames": out, "frame": frame, "toe": toe, "ball": ball}


def toe_figure(item: dict, toe: tuple[float, float] | None) -> go.Figure:
    """The contact frame's toe crop as a figure: a click there sets the toe."""
    w, h = TOE_BOX[0] * SCALE, TOE_BOX[1] * SCALE
    x0, y0 = item["toe_origin"]
    fig = go.Figure()
    fig.add_layout_image(
        source=item["toe_img"], x=x0, y=y0, sizex=TOE_BOX[0], sizey=TOE_BOX[1],
        xref="x", yref="y", sizing="stretch", layer="below",
    )  # fmt: skip
    # An invisible grid of points makes the whole image clickable.
    gx, gy = np.meshgrid(np.arange(x0, x0 + TOE_BOX[0], 2.0), np.arange(y0, y0 + TOE_BOX[1], 2.0))
    fig.add_trace(
        go.Scatter(
            x=gx.ravel(),
            y=gy.ravel(),
            mode="markers",
            marker={"size": 4, "opacity": 0},
            hovertemplate="click: the toe tip is here<extra></extra>",
            showlegend=False,
        )
    )
    if toe is not None:
        fig.add_trace(
            go.Scatter(
                x=[toe[0]],
                y=[toe[1]],
                mode="markers",
                marker={"symbol": "x-thin", "size": 16, "line": {"width": 2, "color": "#ff4040"}},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        width=w,
        height=h,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode=False,
        hovermode="closest",
        xaxis={"visible": False, "range": [x0, x0 + TOE_BOX[0]], "fixedrange": True},
        yaxis={"visible": False, "range": [y0 + TOE_BOX[1], y0], "fixedrange": True},
    )
    return fig


def metrics(row: dict):
    """The serve's contact numbers (a small table)."""
    u = units.current()
    src = {"toss_path": "seen leaving the toss", "inferred": "inferred (ball hidden)",
           "edit": "set by you"}.get(row["contact_source"] or "", "–")  # fmt: skip
    toe_src = {"toe": "toe keypoint", "heel_length": "heel + foot length", "edit": "set by you"}

    def sig(v, s):
        if v is None:
            return "–"
        return (
            f"{u.small_str(v, sign=True)} ± {u.small_str(s)}"
            if s is not None
            else u.small_str(v, sign=True)
        )

    lines = [
        ("Contact frame", f"{row['frame_contact'] or '–'} ({src})"),
        ("In front of the toe", sig(row["forward_m"], row["forward_sigma_m"])),
        ("To the racket side", sig(row["lateral_m"], row["lateral_sigma_m"])),
        ("Contact height", "–" if row["height_m"] is None else u.len_str(row["height_m"], 2)),
        (
            "Toe",
            f"{toe_src.get(row['toe_source'] or '', '–')}"
            + {True: ", on the ground", False: ", lifted", None: ""}[row["toe_on_ground"]],
        ),
        ("Toe behind the baseline", u.small_str(row["toe_to_baseline_m"], sign=True)),
        ("Toe moved since the toss", u.small_str(row["toe_moved_m"])),
        ("Serve side", row["serve_side"] or "–"),
        ("Toss", f"{row['toss_points'] or 0} points, {num(row['toss_rms_px'], '.1f', ' px')} RMS"),
    ]
    table = dmc.Table(
        [
            html.Tbody(
                [html.Tr([html.Td(k), html.Td(v, style={"textAlign": "right"})]) for k, v in lines]
            )
        ],
        fz="xs",
    )
    flags = row["flags"] or []
    return dmc.Stack(
        [table, dmc.Text(" · ".join(flags), size="xs", c="dimmed") if flags else None], gap=2
    )
