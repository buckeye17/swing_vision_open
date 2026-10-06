"""Swings: data access and figures for the Swings page and the skeleton on the session video
(PLAN.md §9.1 3D skeleton viewer, §9.2 Swings page)."""

from __future__ import annotations

from dataclasses import dataclass

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots

from swingvision.pose.kinematics import CURVE_LABELS, FrameKinematics, frame_kinematics
from swingvision.pose.skeleton import COCO_EDGES, H36M_EDGES, H
from swingvision.pose.strokes import STROKE_LABELS
from swingvision.storage import tables
from swingvision.storage.fsutil import read_json

STROKE_COLORS = {
    "serve": "#7048e8",
    "forehand": "#1c7ed6",
    "backhand": "#f08c00",
    "forehand_volley": "#0ca678",
    "backhand_volley": "#e8590c",
    "overhead": "#ae3ec9",
    "other": "#868e96",
}
PHASES = (
    ("Preparation", "t_start", "t_backswing_end", "rgba(112,72,232,0.10)"),
    ("Forward swing", "t_backswing_end", "t_contact", "rgba(240,140,0,0.16)"),
    ("Follow-through", "t_contact", "t_follow_end", "rgba(28,126,214,0.12)"),
    ("Recovery", "t_follow_end", "t_recovery_end", "rgba(134,142,150,0.10)"),
)
#: Seconds of pose around the contact the page loads for one swing.
WINDOW = (1.8, 1.2)
CURVE_ROWS = (
    ("Rotation (deg)", ("shoulder_turn", "hip_turn", "separation")),
    ("Joint angles (deg)", ("knee_flex", "elbow", "trunk_lean")),
    ("Racket wrist", ("wrist_speed",)),
)
CURVE_COLORS = {
    "shoulder_turn": "#1c7ed6",
    "hip_turn": "#f08c00",
    "separation": "#7048e8",
    "knee_flex": "#0ca678",
    "elbow": "#e8590c",
    "trunk_lean": "#868e96",
    "wrist_speed": "#c2255c",
    "wrist_height": "#2f9e44",
}


def stroke_label(stroke: str | None) -> str:
    return STROKE_LABELS.get(stroke or "", "–")


def fmt_t(t: float | None) -> str:
    if t is None:
        return "–"
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def num(v, fmt: str, unit: str = "") -> str:
    return "–" if v is None else f"{format(v, fmt)}{unit}"


@dataclass
class SwingData:
    rows: list[dict]
    summary: dict
    hand: str
    width: int
    height: int
    fps: float

    def by_id(self, swing_id: int | None) -> dict | None:
        return next((r for r in self.rows if r["swing_id"] == swing_id), None)


def load(session) -> SwingData | None:
    if not (session.swings_path.exists() and session.swings_summary_path.exists()):
        return None
    config = session.load_config()
    video = config.video
    summary = read_json(session.swings_summary_path)
    return SwingData(
        rows=tables.read_table(session.swings_path).to_pylist(),
        summary=summary,
        hand=summary.get("racket_hand", "right"),
        width=video.display_width if video else 3840,
        height=video.display_height if video else 2160,
        fps=video.fps_avg if video else 60.0,
    )


def strokes_only(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["stroke_type"] not in (None, "other")]


def filtered(rows: list[dict], strokes: list[str] | None, end: str, show_other: bool) -> list[dict]:
    out = []
    for r in rows:
        if r["stroke_type"] in (None, "other"):
            if not show_other:
                continue
        elif strokes and r["stroke_type"] not in strokes:
            continue
        if end != "all" and r["side"] != (1 if end == "far" else -1):
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Pose for one swing
# ---------------------------------------------------------------------------


def pose_window(session, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """3D joints (court) of the player between t0 and t1 → (t, joints (n, 17, 3)), the clip at
    the window's middle only (no jumps across gaps)."""
    if not session.pose3d_path.exists():
        return np.zeros(0), np.zeros((0, 17, 3))
    tab = tables.read_table(
        session.pose3d_path, filters=[("t_s", ">=", t0), ("t_s", "<=", t1)]
    ).sort_by("t_s")
    if tab.num_rows == 0:
        return np.zeros(0), np.zeros((0, 17, 3))
    t = tab.column("t_s").to_numpy()
    clip = tab.column("clip").to_numpy()
    col = tab.column("joints").combine_chunks()
    j = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), 17, 3).astype(np.float64)
    mid = clip[int(np.argmin(np.abs(t - (t0 + t1) / 2)))]
    keep = (clip == mid) & np.isfinite(j[:, 0, 0])
    return t[keep], j[keep]


def swing_kinematics(session, row: dict, hand: str) -> tuple[FrameKinematics, np.ndarray] | None:
    """Kinematics (hitter's frame) and court joints around one swing's contact."""
    tc = row["t_contact"]
    t, j = pose_window(session, tc - WINDOW[0], tc + WINDOW[1])
    if len(t) < 10:
        return None
    return frame_kinematics(t, j, row["side"], hand), j


def average_curves(session, rows: list[dict], hand: str, step: float = 1 / 60) -> dict | None:
    """Mean curves of several swings aligned at their contact (for "my average")."""
    grid = np.arange(-WINDOW[0] + 0.3, WINDOW[1] - 0.2 + 1e-9, step)
    acc: dict[str, list[np.ndarray]] = {}
    for r in rows[:40]:
        got = swing_kinematics(session, r, hand)
        if got is None:
            continue
        kin, _ = got
        rel = kin.t - r["t_contact"]
        if rel[0] > grid[0] or rel[-1] < grid[-1]:
            continue
        for name, curve in kin.curves().items():
            acc.setdefault(name, []).append(np.interp(grid, rel, curve))
    if not acc:
        return None
    n = len(next(iter(acc.values())))
    return {"t": grid, "n": n, "curves": {k: np.mean(v, axis=0) for k, v in acc.items()}}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _phase_shapes(row: dict, t0: float) -> list[dict]:
    shapes = []
    for _, a, b, color in PHASES:
        if row.get(a) is None or row.get(b) is None:
            continue
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": row[a] - t0,
                "x1": row[b] - t0,
                "y0": 0,
                "y1": 1,
                "fillcolor": color,
                "line": {"width": 0},
                "layer": "below",
            }
        )
    shapes.append(
        {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": 0,
            "x1": 0,
            "y0": 0,
            "y1": 1,
            "line": {"color": "#e03131", "width": 1.5, "dash": "dot"},
        }
    )
    return shapes


def angle_figure(kin: FrameKinematics | None, row: dict | None, compare: dict | None = None):
    """Angle and speed curves around the contact (t = 0), phases shaded; ``compare``:
    ``{"label", "t" (relative), "curves"}`` drawn dashed."""
    fig = make_subplots(
        rows=len(CURVE_ROWS),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[title for title, _ in CURVE_ROWS],
    )
    if kin is not None and row is not None:
        rel = kin.t - row["t_contact"]
        curves = kin.curves()
        for i, (_, names) in enumerate(CURVE_ROWS, start=1):
            for name in names:
                label, unit = CURVE_LABELS[name]
                fig.add_trace(
                    go.Scatter(
                        x=rel,
                        y=curves[name],
                        name=label,
                        legendgroup=name,
                        line={"color": CURVE_COLORS[name], "width": 2},
                        hovertemplate=f"{label}: %{{y:.1f}} {unit}<extra>%{{x:+.2f}} s</extra>",
                    ),
                    row=i,
                    col=1,
                )
                if compare is not None and name in compare["curves"]:
                    fig.add_trace(
                        go.Scatter(
                            x=compare["t"],
                            y=compare["curves"][name],
                            name=f"{label} ({compare['label']})",
                            legendgroup=name,
                            showlegend=False,
                            line={"color": CURVE_COLORS[name], "width": 1.5, "dash": "dash"},
                            hovertemplate=(
                                f"{compare['label']}: %{{y:.1f}} {unit}<extra>%{{x:+.2f}} s</extra>"
                            ),
                        ),
                        row=i,
                        col=1,
                    )
        fig.update_layout(shapes=_phase_shapes(row, row["t_contact"]))
    fig.update_xaxes(title_text="seconds from contact", row=len(CURVE_ROWS), col=1)
    fig.update_layout(
        height=520,
        margin={"l": 50, "r": 10, "t": 30, "b": 40},
        legend={"orientation": "h", "y": -0.12, "font": {"size": 11}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    return fig


def _bones(j: np.ndarray) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for a, b in H36M_EDGES:
        xs += [j[a, 0], j[b, 0], None]
        ys += [j[a, 1], j[b, 1], None]
        zs += [j[a, 2], j[b, 2], None]
    return xs, ys, zs


def skeleton_3d(t: np.ndarray, joints: np.ndarray, row: dict | None, hand: str, step: int = 2):
    """Animated 3D skeleton on the court (meters), with the racket wrist's path and a phase
    label per frame."""
    fig = go.Figure()
    if row is None or len(t) == 0:
        fig.update_layout(height=420, margin={"l": 0, "r": 0, "t": 0, "b": 0})
        return fig
    wrist = H[f"{'l' if hand == 'left' else 'r'}_wrist"]
    keep = np.arange(0, len(t), step)
    pelvis = joints[:, H["pelvis"]]
    cx, cy = np.nanmedian(pelvis[:, 0]), np.nanmedian(pelvis[:, 1])
    half = 1.6
    path = joints[:, wrist]
    ic = int(np.argmin(np.abs(t - row["t_contact"])))

    def phase(tt: float) -> str:
        name = "Ready"
        for label, a, b, _ in PHASES:
            if row.get(a) is not None and row.get(b) is not None and row[a] <= tt < row[b]:
                name = label
        if abs(tt - row["t_contact"]) < 0.5 / 60:
            name = "Contact"
        return name

    def frame_data(k: int) -> list:
        xs, ys, zs = _bones(joints[k])
        return [
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", line={"color": "#1c7ed6", "width": 6},
                hoverinfo="skip", name="Body",
            ),
            go.Scatter3d(
                x=joints[k, :, 0], y=joints[k, :, 1], z=joints[k, :, 2], mode="markers",
                marker={"size": 3, "color": "#1c7ed6"}, hoverinfo="skip", name="Joints",
            ),
            go.Scatter3d(
                x=[joints[k, wrist, 0]], y=[joints[k, wrist, 1]], z=[joints[k, wrist, 2]],
                mode="markers", marker={"size": 6, "color": "#e03131"}, name="Racket wrist",
                hoverinfo="skip",
            ),
        ]  # fmt: skip

    static = [
        go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines",
            line={"color": "rgba(224,49,49,0.35)", "width": 3}, name="Wrist path",
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=[path[ic, 0]], y=[path[ic, 1]], z=[path[ic, 2]], mode="markers",
            marker={"size": 5, "color": "#e03131", "symbol": "diamond"}, name="Contact",
            hoverinfo="skip",
        ),
    ]  # fmt: skip
    k0 = int(keep[np.argmin(np.abs(t[keep] - row["t_contact"]))])
    fig = go.Figure(
        data=frame_data(k0) + static,
        frames=[
            go.Frame(
                data=frame_data(int(k)), name=f"{t[k] - row['t_contact']:+.2f}", traces=[0, 1, 2]
            )
            for k in keep
        ],
    )
    steps = [
        {
            "label": f"{t[k] - row['t_contact']:+.2f} {phase(t[k])}",
            "method": "animate",
            "args": [
                [f"{t[k] - row['t_contact']:+.2f}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for k in keep
    ]
    fig.update_layout(
        height=460,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis": {"range": [cx - half, cx + half], "title": "x (m)"},
            "yaxis": {"range": [cy - half, cy + half], "title": "y (m)"},
            "zaxis": {"range": [0, 2.8], "title": "z (m)"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 2.8 / (2 * half)},
            "camera": {"eye": {"x": 1.4, "y": -1.6 if row["side"] != 1 else 1.6, "z": 0.6}},
        },
        sliders=[
            {
                "active": int(np.argmin(np.abs(t[keep] - row["t_contact"]))),
                "steps": steps,
                "currentvalue": {"prefix": "t = ", "font": {"size": 12}},
                "pad": {"t": 0, "b": 4},
                "len": 1.0,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": 1000 / 30, "redraw": True}, "fromcurrent": True},
                        ],
                    },
                    {
                        "label": "❚❚",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0}}],
                    },
                ],
            }
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def swing_table(rows: list[dict], selected: int | None):
    body = []
    for r in rows:
        style = {"cursor": "pointer"}
        if r["swing_id"] == selected:
            style["background"] = "var(--mantine-color-teal-light)"
        stroke = r["stroke_type"] or "other"
        flags = [f for f in (r["flags"] or []) if f in ("dark", "far", "low_conf", "little_pose")]
        body.append(
            html.Tr(
                [
                    html.Td(fmt_t(r["t_contact"])),
                    html.Td({-1: "near", 1: "far"}.get(r["side"], "–")),
                    html.Td(
                        dmc.Badge(
                            stroke_label(stroke),
                            color=STROKE_COLORS.get(stroke, "gray"),
                            variant="light",
                            size="xs",
                            styles={"root": {"color": STROKE_COLORS.get(stroke, "gray")}},
                        )
                    ),
                    html.Td("✎" if r["stroke_source"] == "user" else ""),
                    html.Td(num(r["forward_s"], ".2f", " s"), style={"textAlign": "right"}),
                    html.Td(num(r["wrist_speed_peak"], ".1f"), style={"textAlign": "right"}),
                    html.Td(", ".join(flags), style={"color": "var(--mantine-color-dimmed)"}),
                ],
                id={"type": "sw-row", "index": r["swing_id"]},
                n_clicks=0,
                style=style,
            )
        )
    head = html.Thead(
        html.Tr(
            [
                html.Th("Time"),
                html.Th("End"),
                html.Th("Stroke"),
                html.Th(""),
                html.Th("Forward", style={"textAlign": "right"}),
                html.Th("Wrist m/s", style={"textAlign": "right"}),
                html.Th(""),
            ]
        )
    )
    return html.Div(
        dmc.Table([head, html.Tbody(body)], striped=False, highlightOnHover=True, fz="xs"),
        style={"maxHeight": 360, "overflowY": "auto"},
    )


def metrics_table(row: dict | None, compare: dict | None = None):
    """Phase timing, contact and kinetic-chain numbers of one swing (and the compared one)."""
    if row is None:
        return dmc.Text("Pick a swing.", size="sm", c="dimmed")
    cmp_row = compare.get("row") if compare else None

    def line(label: str, key: str, fmt: str, unit: str = "", hint: str | None = None):
        cells = [
            html.Td(label),
            html.Td(num(row.get(key), fmt, unit), style={"textAlign": "right"}),
        ]
        if compare is not None:
            v = cmp_row.get(key) if cmp_row else compare.get("means", {}).get(key)
            cells.append(
                html.Td(
                    num(v, fmt, unit),
                    style={"textAlign": "right", "color": "var(--mantine-color-dimmed)"},
                )
            )
        cells.append(html.Td(hint or "", style={"color": "var(--mantine-color-dimmed)"}))
        return html.Tr(cells)

    order = row.get("chain_in_order")
    rows = [
        line("Preparation", "prep_s", ".2f", " s", "unit turn / toss to backswing end"),
        line("Forward swing", "forward_s", ".2f", " s", "backswing end / racket drop to contact"),
        line("Follow-through", "follow_s", ".2f", " s"),
        line("Recovery", "recovery_s", ".2f", " s"),
        line("Tempo", "tempo", ".1f", "", "preparation / forward swing"),
        line("Contact height", "contact_height_m", ".2f", " m", "racket wrist"),
        line("Contact in front", "contact_front_m", "+.2f", " m", "of the pelvis, toward the net"),
        line("Contact to the side", "contact_side_m", "+.2f", " m", "racket side +"),
        line("Wrist speed (peak)", "wrist_speed_peak", ".1f", " m/s", "racket speed proxy"),
        line("Wrist speed (last 0.3 s)", "wrist_speed_avg", ".1f", " m/s"),
        line("Shoulder turn (max)", "shoulder_turn_max", ".0f", "°", "racket side back +"),
        line("Hip turn (max)", "hip_turn_max", ".0f", "°"),
        line("Hip-shoulder separation", "separation_max", ".0f", "°"),
        line("Knee bend (max)", "knee_flex_max", ".0f", "°"),
        line("Elbow at contact", "elbow_contact", ".0f", "°", "180° = straight"),
        line("Trunk lean at contact", "trunk_lean_contact", ".0f", "°"),
        line("Stance width", "stance_width_m", ".2f", " m"),
        line("Jump", "jump_m", ".2f", " m", "pelvis at contact vs standing"),
        line("Toss height", "toss_height_m", ".2f", " m", "serves"),
        line("Hips peak", "t_pelvis_peak", "+.2f", " s", "rotation speed, from contact"),
        line("Trunk peak", "t_trunk_peak", "+.2f", " s"),
        line("Elbow peak", "t_elbow_peak", "+.2f", " s"),
        line("Wrist peak", "t_wrist_peak", "+.2f", " s"),
    ]
    head = [html.Th("Metric"), html.Th("This swing", style={"textAlign": "right"})]
    if compare is not None:
        head.append(html.Th(compare["label"], style={"textAlign": "right"}))
    head.append(html.Th(""))
    chain = (
        "–"
        if order is None
        else ("hips → trunk → elbow → wrist in order" if order else "out of order")
    )
    return dmc.Stack(
        [
            dmc.Table(
                [html.Thead(html.Tr(head)), html.Tbody(rows)],
                fz="xs",
                highlightOnHover=True,
            ),
            dmc.Text(f"Kinetic chain: {chain}", size="xs", c="dimmed"),
        ],
        gap=4,
    )


def mean_metrics(rows: list[dict]) -> dict:
    keys = [
        "prep_s", "forward_s", "follow_s", "recovery_s", "tempo", "contact_height_m",
        "contact_front_m", "contact_side_m", "wrist_speed_peak", "wrist_speed_avg",
        "shoulder_turn_max", "hip_turn_max", "separation_max", "knee_flex_max",
        "elbow_contact", "trunk_lean_contact", "stance_width_m", "jump_m", "toss_height_m",
        "t_pelvis_peak", "t_trunk_peak", "t_elbow_peak", "t_wrist_peak",
    ]  # fmt: skip
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.median(v)) if v else None
    return out


# ---------------------------------------------------------------------------
# 2D skeleton on the video
# ---------------------------------------------------------------------------


def skeleton_store(session, t0: float, t1: float, width: int, height: int) -> dict:
    """2D keypoints of the player from t0 to t1, in percent of the frame (for the overlay)."""
    out = {"t0": t0, "t1": t1, "t": [], "kp": []}
    if not session.pose2d_path.exists():
        return out
    tab = tables.read_table(
        session.pose2d_path, columns=["t_s", "kp"], filters=[("t_s", ">=", t0), ("t_s", "<=", t1)]
    ).sort_by("t_s")
    if tab.num_rows == 0:
        return out
    col = tab.column("kp").combine_chunks()
    kp = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), 17, 3).astype(np.float64)
    xy = np.round(
        np.stack([kp[..., 0] * 100 / width, kp[..., 1] * 100 / height, kp[..., 2]], -1), 2
    )
    out["t"] = np.round(tab.column("t_s").to_numpy(), 4).tolist()
    out["kp"] = xy.reshape(len(xy), -1).tolist()
    return out


#: Clientside: (time store, skeleton store, on) → [img src, img style]. Bones with a weak
#: keypoint (confidence < 0.3) are left out.
SKELETON_JS = (
    """
    function(t, skel, on) {
        const hidden = {display: "none"};
        if (on === false || !skel || !skel.t || !skel.t.length) { return ["", hidden]; }
        const now = (t && t.t) || 0;
        const ts = skel.t;
        let lo = 0, hi = ts.length - 1, i = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (ts[mid] <= now + 1e-3) { i = mid; lo = mid + 1; } else { hi = mid - 1; }
        }
        if (i < 0 || now - ts[i] > 0.05) { return ["", hidden]; }
        const kp = skel.kp[i];
        const edges = EDGES;
        let d = "";
        for (const [a, b] of edges) {
            if (kp[3 * a + 2] < 0.3 || kp[3 * b + 2] < 0.3) { continue; }
            d += "M" + kp[3 * a] + "," + kp[3 * a + 1] + "L" + kp[3 * b] + "," + kp[3 * b + 1];
        }
        const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" ' +
            'preserveAspectRatio="none"><path d="' + d + '" fill="none" stroke="#74c0fc" ' +
            'stroke-width="2.5" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>';
        return ["data:image/svg+xml;utf8," + encodeURIComponent(svg),
                {position: "absolute", inset: 0, width: "100%", height: "100%",
                 pointerEvents: "none", display: "block"}];
    }
    """
).replace("EDGES", str([list(e) for e in COCO_EDGES]))
