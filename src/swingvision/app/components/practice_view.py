"""Practice page pieces (M5): loading, the landing map with targets, rolling accuracy,
KPIs, block and shot tables, and target outlines projected onto the video."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import html

from swingvision.analysis import practice as pr
from swingvision.analysis.segmentation import blocks_of
from swingvision.analysis.shots import SPEED_SCALE_ERROR
from swingvision.app import units
from swingvision.app.components.court_diagram import court_figure
from swingvision.app.components.target_editor import (
    ABSOLUTE_COLOR,
    TARGET_COLOR,
    target_labels,
    target_shape,
)
from swingvision.court import calibration as calib
from swingvision.court.camera import Camera
from swingvision.storage import edits as ed
from swingvision.storage import tables
from swingvision.storage.schemas import Calibration, SessionConfig, Target
from swingvision.storage.session import Session

HIT_COLOR = "#51cf66"  # in a target
MISS_COLOR = "#ffa94d"  # in, but not in a target
OUT_COLOR = "#ff6b6b"
NET_COLOR = "#cc5de8"
IN_COLOR = "#51cf66"
EXCLUDED_COLOR = "#868e96"
OUTCOME_LABELS = {
    "in": "In",
    "out_long": "Long",
    "out_wide": "Wide",
    "net": "Net",
    "own_side": "Own side",
    "unknown": "No landing",
}
KIND_FILTERS = {"all": "All", "serve": "Serves", "groundstroke": "Groundstrokes"}


@dataclass
class PracticeData:
    config: SessionConfig
    rows: list[dict]
    blocks: list[dict]
    segments: dict[int, dict]
    cal: Calibration | None
    edits_version: int

    @property
    def targets(self) -> list[Target]:
        return [
            pr.normalized(t)
            for t in (self.config.practice.targets if self.config.practice else [])
            if pr.target_valid(pr.normalized(t))
        ]


def load(session: Session) -> PracticeData | None:
    if not session.practice_path.exists() or not session.segments_path.exists():
        return None
    seg = tables.read_table(session.segments_path)
    return PracticeData(
        config=session.load_config(),
        rows=tables.read_table(session.practice_path).to_pylist(),
        blocks=blocks_of(seg),
        segments={r["segment_id"]: r for r in seg.to_pylist() if r["kind"] == "practice_shot"},
        cal=calib.load(session.calibration_path),
        edits_version=ed.load(session).version,
    )


def filtered(
    data: PracticeData, block: str | None, kind: str | None, excluded: bool = True
) -> list[dict]:
    out = []
    for r in data.rows:
        if block not in (None, "all") and str(r["block_id"]) != block:
            continue
        if kind not in (None, "all") and r["shot_kind"] != kind:
            continue
        if r["excluded"] and not excluded:
            continue
        out.append(r)
    return out


def fmt_t(t: float | None) -> str:
    if t is None:
        return "–"
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


def pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.0%}"


def result_color(r: dict) -> str:
    if r["excluded"]:
        return EXCLUDED_COLOR
    if r["outcome"] == "net":
        return NET_COLOR
    if r["outcome"] in ("out_long", "out_wide", "own_side"):
        return OUT_COLOR
    if r["in_target"] is True:
        return HIT_COLOR
    if r["in_target"] is False:
        return MISS_COLOR
    return IN_COLOR if r["outcome"] == "in" else EXCLUDED_COLOR


def result_label(r: dict, u: units.Units | None = None) -> str:
    o = OUTCOME_LABELS.get(r["outcome"], r["outcome"])
    if r["outcome"] in ("out_long", "out_wide") and r["margin_m"] is not None:
        o += f" {(u or units.current()).small_str(abs(r['margin_m']))}"
    if r["in_target"] is True:
        return o + " · target"
    return o


def kind_label(r: dict) -> str:
    """The shot's kind, or its stroke when the pose says more (forehand, backhand, …)."""
    stroke = r.get("stroke_type")
    if stroke in pr.STROKE_LABELS and stroke != "serve":
        return pr.STROKE_LABELS[stroke]
    return pr.KIND_LABELS.get(r["shot_kind"], "Shot")


def shot_title(r: dict) -> str:
    kind = kind_label(r)
    if r["serve_side"]:
        kind += f" ({r['serve_side']})"
    end = {-1: "near", 1: "far"}.get(r["side"], "?")
    return f"{fmt_t(r['t_contact'])} · {kind} from the {end} end"


# ---------------------------------------------------------------------------
# Court map
# ---------------------------------------------------------------------------


def landing_xy(r: dict, view: str) -> tuple[float, float] | None:
    if r["landing_x"] is None:
        return None
    if view == "hit" and r["rel_x"] is not None:
        return r["rel_x"], r["rel_y"]
    return r["landing_x"], r["landing_y"]


def court_map(
    data: PracticeData,
    rows: list[dict],
    view: str = "hit",
    selected: int | None = None,
    placing: bool = False,
    height: int = 520,
) -> go.Figure:
    """Landings over the targets. ``view="hit"``: every shot as if hit from the near end
    (relative targets as drawn); ``"court"``: where things are on the court."""
    u = units.current()
    fig = court_figure(height=height)
    shapes = list(fig.layout.shapes)
    sides = sorted({r["side"] for r in rows if r["side"] is not None}) or [-1]
    labels: list[Target] = []
    for t in data.targets:
        # Relative targets sit where each end's hitter aims on the court view; absolute ones
        # (and everything in the hitter's view) as drawn.
        if view == "court" and t.frame == "relative":
            versions = [pr.target_on_court(t, s) for s in sides]
        else:
            versions = [t]
        for v in versions:
            shapes.append(target_shape(v, editable=False, opacity=0.18))
            labels.append(v)
    fig.update_layout(shapes=shapes)
    if labels:
        fig.add_trace(target_labels(labels))
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if landing_xy(r, view) is None:
            continue
        key = "excluded" if r["excluded"] else result_color(r)
        groups.setdefault(key, []).append(r)
    for key, members in groups.items():
        xy = [landing_xy(r, view) for r in members]
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in xy],
                y=[p[1] for p in xy],
                mode="markers",
                marker={
                    "color": EXCLUDED_COLOR if key == "excluded" else key,
                    "size": [12 if r["segment_id"] == selected else 8 for r in members],
                    "symbol": ["x" if r["excluded"] else "circle" for r in members],
                    "line": {
                        "color": ["white" if r["landing_confirmed"] else "#222" for r in members],
                        "width": [2 if r["landing_confirmed"] else 1 for r in members],
                    },
                    "opacity": 0.9,
                },
                customdata=[
                    [r["segment_id"], r["t_contact"], u.len(pt[0]), u.len(pt[1])]
                    for r, pt in zip(members, xy, strict=True)
                ],
                text=[f"{shot_title(r)}<br>{result_label(r, u)}" for r in members],
                # While placing a landing, clicks must reach the grid underneath (a
                # hovertemplate would override hoverinfo).
                hovertemplate=None
                if placing
                else (
                    f"%{{text}}<br>(%{{customdata[2]:.2f}}, %{{customdata[3]:.2f}}) "
                    f"{u.len_unit}<extra></extra>"
                ),
                hoverinfo="skip" if placing else None,
                name="landings",
            )
        )
    sel = next((r for r in rows if r["segment_id"] == selected), None)
    if sel is not None and landing_xy(sel, view) is not None:
        x, y = landing_xy(sel, view)  # type: ignore[misc]
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker={
                    "size": 20,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "white", "width": 2},
                },
                hoverinfo="skip",
                name="selected",
            )
        )
    fig.update_layout(
        clickmode="event",
        uirevision=f"{view}",
        annotations=[
            {
                "text": "you hit from here ↓" if view == "hit" else "camera end ↓",
                "x": 0,
                "y": -12.9,
                "showarrow": False,
                "font": {"color": "white", "size": 11},
            }
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Rolling accuracy
# ---------------------------------------------------------------------------


def rolling_figure(rows: list[dict], has_targets: bool) -> go.Figure:
    u = units.current()
    roll = pr.rolling(rows)
    fig = go.Figure()
    if has_targets:
        fig.add_trace(
            go.Scatter(
                x=roll["n"],
                y=[None if v is None else v * 100 for v in roll["target"]],
                mode="lines",
                name="target hits",
                line={"color": HIT_COLOR, "width": 2.5},
                hovertemplate="shot %{x}: %{y:.0f}% in a target (last 10)<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=roll["n"],
            y=[None if v is None else v * 100 for v in roll["in"]],
            mode="lines",
            name="in",
            line={"color": "#4dabf7", "width": 2, "dash": "dot" if has_targets else "solid"},
            hovertemplate="shot %{x}: %{y:.0f}% in (last 10)<extra></extra>",
        )
    )
    kept = sorted((r for r in rows if not r["excluded"]), key=lambda r: r["t_contact"])
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(kept) + 1)),
            y=[-6] * len(kept),
            mode="markers",
            marker={"color": [result_color(r) for r in kept], "size": 7, "symbol": "square"},
            customdata=[[r["segment_id"], r["t_contact"]] for r in kept],
            text=[f"{shot_title(r)}<br>{result_label(r, u)}" for r in kept],
            hovertemplate="%{text}<extra></extra>",
            name="shots",
        )
    )
    blocks = [r["block_id"] for r in kept]
    shapes = [
        {
            "type": "line",
            "x0": i + 0.5,
            "x1": i + 0.5,
            "y0": 0,
            "y1": 100,
            "line": {"color": "rgba(128,128,128,0.5)", "width": 1, "dash": "dot"},
        }
        for i in range(1, len(blocks))
        if blocks[i] != blocks[i - 1]
    ]
    fig.update_layout(
        height=220,
        margin={"l": 40, "r": 10, "t": 10, "b": 30},
        showlegend=True,
        legend={"orientation": "h", "y": 1.12, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=shapes,
        xaxis={"title": "shot", "showgrid": False, "fixedrange": True},
        yaxis={
            "range": [-10, 102],
            "ticksuffix": "%",
            "fixedrange": True,
            "showgrid": True,
            "gridcolor": "rgba(128,128,128,0.15)",
        },
    )
    return fig


# ---------------------------------------------------------------------------
# KPIs and tables
# ---------------------------------------------------------------------------


def _stat(label: str, value: str, sub: str = ""):
    return dmc.Paper(
        dmc.Stack(
            [
                dmc.Text(label, size="xs", c="dimmed"),
                dmc.Text(value, fw=700, size="xl"),
                dmc.Text(sub, size="xs", c="dimmed"),
            ],
            gap=0,
        ),
        p="xs",
        withBorder=True,
    )


def kpis(rows: list[dict], has_targets: bool):
    u = units.current()
    s = pr.summarize(rows)
    items = [
        _stat("Shots", str(s["n"]), f"{s['n_landed']} landings seen"),
        _stat("In", pct(s["in_pct"]), f"{s['n_in']} of {s['n_called']} called"),
        _stat("Net", pct(s["net_pct"]), f"{s['n_net']} shots"),
    ]
    if has_targets:
        items += [
            _stat(
                "Target hits",
                pct(s["target_pct"]),
                f"{s['n_target_hits']} of {s['n_targeted']} aimed",
            ),
            _stat(
                "To target",
                u.len_str(s["dist_median"], 1),
                "median distance to its center",
            ),
        ]
    items += [
        _stat(
            "Depth spread",
            "–" if s["depth_sd"] is None else f"± {u.len_str(s['depth_sd'], 1)}",
            "SD of landing depth",
        ),
        _stat(
            "Speed",
            u.speed_str(s["speed_median"]),
            f"median of {s['n_speed']}, ± {SPEED_SCALE_ERROR:.0%} uncalibrated",
        ),
    ]
    if s["feed_speed_mean"] is not None:
        items.append(
            _stat(
                "Machine feeds",
                u.speed_str(s["feed_speed_mean"]),
                "spread "
                + u.len_str(s["feed_spread_m"], 1)
                + (
                    ""
                    if s["feed_speed_sd"] is None
                    else f", speed ± {u.speed_str(s['feed_speed_sd'])}"
                ),
            )
        )
    return dmc.SimpleGrid(items, cols={"base": 2, "sm": 4, "lg": len(items)}, spacing="xs")


def blocks_table(data: PracticeData, has_targets: bool):
    u = units.current()
    body = []
    for b in data.blocks:
        rows = [r for r in data.rows if r["block_id"] == b["block_id"]]
        s = pr.summarize(rows)
        end = {-1: "near", 1: "far"}.get(b["side"], "?")
        body.append(
            html.Tr(
                [
                    html.Td(f"{b['block_id'] + 1}"),
                    html.Td(f"{fmt_t(b['start_t'])}–{fmt_t(b['end_t'])}"),
                    html.Td(f"{pr.KIND_LABELS.get(b['shot_kind'], 'Mixed')}s, {end} end"),
                    html.Td(str(s["n"]), style={"textAlign": "right"}),
                    html.Td(pct(s["in_pct"]), style={"textAlign": "right"}),
                    html.Td(pct(s["net_pct"]), style={"textAlign": "right"}),
                    html.Td(
                        pct(s["target_pct"]) if has_targets else "",
                        style={"textAlign": "right"},
                    ),
                    html.Td(
                        "–" if s["depth_sd"] is None else f"{u.len(s['depth_sd']):.1f}",
                        style={"textAlign": "right"},
                    ),
                    html.Td(
                        "–" if s["speed_median"] is None else f"{u.speed(s['speed_median']):.0f}",
                        style={"textAlign": "right"},
                    ),
                ],
                id={"type": "pr-block-row", "index": b["block_id"]},
                n_clicks=0,
                style={"cursor": "pointer"},
            )
        )
    head = html.Thead(
        html.Tr(
            [
                html.Th("#"),
                html.Th("Time"),
                html.Th("What"),
                html.Th("Shots", style={"textAlign": "right"}),
                html.Th("In", style={"textAlign": "right"}),
                html.Th("Net", style={"textAlign": "right"}),
                html.Th("Target" if has_targets else "", style={"textAlign": "right"}),
                html.Th(f"Depth SD {u.len_unit}", style={"textAlign": "right"}),
                html.Th(u.speed_unit, style={"textAlign": "right"}),
            ]
        )
    )
    return dmc.Table([head, html.Tbody(body)], highlightOnHover=True, fz="xs")


def shots_table(rows: list[dict], selected: int | None, has_targets: bool):
    u = units.current()
    body = []
    for i, r in enumerate(sorted(rows, key=lambda r: r["t_contact"])):
        flags = []
        if r["landing_confirmed"]:
            flags.append("✓ confirmed" if r["landing_source"] != "user" else "✓ placed")
        if r["close_call"] or "target_close_call" in (r["flags"] or []):
            flags.append("close")
        if "contact_unseen" in (r["flags"] or []):
            flags.append("contact unseen")
        speed = "–" if r["speed_kmh"] is None else f"{u.speed(r['speed_kmh']):.0f}"
        tgt = ""
        if r["in_target"] is not None:
            tgt = "hit" if r["in_target"] else "miss"
            if r["target_dist_m"] is not None:
                tgt += f" · {u.len_str(r['target_dist_m'], 1)}"
        style = {"cursor": "pointer"}
        if r["segment_id"] == selected:
            style["background"] = "var(--mantine-color-teal-light)"
        if r["excluded"]:
            style["opacity"] = 0.45
            style["textDecoration"] = "line-through"
        body.append(
            html.Tr(
                [
                    html.Td(str(i + 1)),
                    html.Td(fmt_t(r["t_contact"])),
                    html.Td(str(r["block_id"] + 1)),
                    html.Td(kind_label(r) + (f" ({r['serve_side']})" if r["serve_side"] else "")),
                    html.Td(
                        dmc.Badge(
                            result_label(r, u),
                            color=result_color(r),
                            variant="light",
                            size="xs",
                            styles={"root": {"color": result_color(r)}},
                        )
                    ),
                    html.Td(tgt) if has_targets else None,
                    html.Td(speed, style={"textAlign": "right"}),
                    html.Td(", ".join(flags), style={"color": "var(--mantine-color-dimmed)"}),
                ],
                id={"type": "pr-shot-row", "index": r["segment_id"]},
                n_clicks=0,
                style=style,
            )
        )
    head = html.Thead(
        html.Tr(
            [
                html.Th("#"),
                html.Th("Time"),
                html.Th("Block"),
                html.Th("Shot"),
                html.Th("Result"),
                html.Th("Target") if has_targets else None,
                html.Th(u.speed_unit, style={"textAlign": "right"}),
                html.Th(""),
            ]
        )
    )
    return dmc.ScrollArea(
        dmc.Table([head, html.Tbody(body)], highlightOnHover=True, fz="xs", verticalSpacing=2),
        h=360,
    )


def breakdown_table(rows: list[dict], has_targets: bool):
    u = units.current()
    groups = pr.breakdown(rows, u.speed_bands_kmh, u.speed_factor, u.speed_unit)
    if not groups:
        return dmc.Text("No shots.", size="sm", c="dimmed")
    body = [
        html.Tr(
            [
                html.Td(name),
                html.Td(str(s["n"]), style={"textAlign": "right"}),
                html.Td(pct(s["in_pct"]), style={"textAlign": "right"}),
                html.Td(pct(s["net_pct"]), style={"textAlign": "right"}),
                html.Td(pct(s["target_pct"]) if has_targets else "", style={"textAlign": "right"}),
                html.Td(
                    "–" if s["depth_mean"] is None else f"{u.len(s['depth_mean']):.1f}",
                    style={"textAlign": "right"},
                ),
            ]
        )
        for name, s in groups
    ]
    head = html.Thead(
        html.Tr(
            [
                html.Th(""),
                html.Th("Shots", style={"textAlign": "right"}),
                html.Th("In", style={"textAlign": "right"}),
                html.Th("Net", style={"textAlign": "right"}),
                html.Th("Target" if has_targets else "", style={"textAlign": "right"}),
                html.Th(f"Depth {u.len_unit}", style={"textAlign": "right"}),
            ]
        )
    )
    return dmc.Table([head, html.Tbody(body)], fz="xs")


def shot_detail(r: dict | None, placing: bool):
    if r is None:
        return dmc.Text(
            "Click a landing, a shot in the table or the accuracy chart to play it.",
            size="sm",
            c="dimmed",
        )
    u = units.current()
    lines = [dmc.Text(shot_title(r), fw=600, size="sm")]
    if r["landing_x"] is not None:
        src = {"bounce": "detected bounce", "fit": "extended flight", "user": "placed by you"}
        land = (
            f"Landed ({u.len(r['landing_x']):.2f}, {u.len(r['landing_y']):.2f}) {u.len_unit}, "
            f"{src.get(r['landing_source'], r['landing_source'])}"
        )
        if r["landing_sigma_m"] is not None:
            land += f", ± {u.small_str(r['landing_sigma_m'])}"
        lines.append(dmc.Text(land, size="xs"))
    call = result_label(r, u)
    if r["call_area"]:
        area = {"singles": "singles court", "deuce_box": "deuce box", "ad_box": "ad box"}
        call += f" (called against the {area.get(r['call_area'], r['call_area'])})"
    lines.append(dmc.Text(call, size="xs"))
    if r["target_id"] is not None:
        lines.append(
            dmc.Text(
                f"Nearest target {u.len_str(r['target_dist_m'], 2)} from its center: "
                f"{u.len_str(r['depth_err_m'], 2, sign=True)} deep, "
                f"{u.len_str(r['width_err_m'], 2, sign=True)} to your right",
                size="xs",
            )
        )
    if r["speed_kmh"] is not None:
        lines.append(
            dmc.Text(
                f"{u.speed(r['speed_kmh']):.0f} ± {u.speed(r['speed_err_kmh'] or 0):.0f} "
                f"{u.speed_unit} off the racket "
                "(uncalibrated)",
                size="xs",
            )
        )
    buttons = dmc.Group(
        [
            dmc.Button(
                "Play",
                id="pr-play",
                size="compact-xs",
                variant="light",
                leftSection=dmc.Text("▶", size="xs"),
            ),
            dmc.Button(
                "Landing is right" if not r["landing_confirmed"] else "Unconfirm",
                id="pr-confirm",
                size="compact-xs",
                variant="default",
                disabled=r["landing_x"] is None,
            ),
            dmc.Button(
                "Click the map to place it…" if placing else "Place landing",
                id="pr-place",
                size="compact-xs",
                variant="filled" if placing else "default",
                color="orange" if placing else None,
            ),
            dmc.Button(
                "Clear placed landing",
                id="pr-unplace",
                size="compact-xs",
                variant="subtle",
                disabled=r["landing_source"] != "user",
            ),
            dmc.Button(
                "Include" if r["excluded"] else "Not a practice shot",
                id="pr-exclude",
                size="compact-xs",
                variant="subtle",
                color="red" if not r["excluded"] else None,
            ),
        ],
        gap=6,
        mt=6,
    )
    return dmc.Stack([*lines, buttons], gap=2)


# ---------------------------------------------------------------------------
# Targets on the video
# ---------------------------------------------------------------------------


def _target_poly(cam: Camera, t: Target) -> np.ndarray:
    if t.shape == "circle":
        a = np.linspace(0, 2 * np.pi, 72)
        pts = np.column_stack([t.cx + t.r * np.cos(a), t.cy + t.r * np.sin(a)])  # type: ignore[operator]
    else:
        c = np.array([[t.x0, t.y0], [t.x1, t.y0], [t.x1, t.y1], [t.x0, t.y1]], dtype=float)
        pts = np.concatenate([np.linspace(c[i], c[(i + 1) % 4], 20) for i in range(4)])
    P = np.column_stack([pts, np.zeros(len(pts))])
    ok = cam.depth(P) > 0.3
    return cam.project(P[ok])


def targets_overlay(cal: Calibration | None, targets: list[Target]) -> dict | None:
    """SVG data URIs of the targets in the video for a near-end and a far-end hitter."""
    if cal is None or not targets:
        return None
    cam = calib.to_camera(cal.camera)
    out = {}
    for side, key in ((-1, "near"), (1, "far")):
        paths = []
        for t in targets:
            p = _target_poly(cam, pr.target_on_court(t, side))
            if len(p) < 3:
                continue
            color = ABSOLUTE_COLOR if t.frame == "absolute" else TARGET_COLOR
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p) + " Z"
            w = max(2.0, cam.width / 800)
            paths.append(
                f'<path d="{d}" fill="{color}" fill-opacity="0.18" stroke="{color}" '
                f'stroke-width="{w:.1f}" stroke-opacity="0.9"/>'
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cam.width} {cam.height}" '
            f'preserveAspectRatio="none">{"".join(paths)}</svg>'
        )
        out[key] = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return out


def shots_store(data: PracticeData) -> dict:
    """Per-shot times for client callbacks (current shot, overlay side, N/P keys)."""
    u = units.current()
    rows = sorted(data.rows, key=lambda r: r["t_contact"])
    out: dict = {k: [] for k in ("id", "t0", "t1", "tc", "side", "label")}
    for r in rows:
        seg = data.segments.get(r["segment_id"], {})
        out["id"].append(r["segment_id"])
        out["t0"].append(round(seg.get("start_t", r["t_contact"] - 1), 2))
        out["t1"].append(round(seg.get("end_t", r["t_contact"] + 2), 2))
        out["tc"].append(round(r["t_contact"], 3))
        out["side"].append(r["side"])
        out["label"].append(result_label(r, u))
    return out


def segments_card(session_id: str, data: PracticeData | None):
    """Session page: the practice blocks and their shots (click a shot to play it)."""
    from swingvision.app.components.ui import icon

    title = dmc.Group(
        [
            dmc.Title("Segments", order=5),
            dmc.Anchor(
                dmc.Button(
                    "Practice page",
                    size="compact-xs",
                    variant="light",
                    leftSection=icon("tabler:target-arrow", 14),
                ),
                href=f"/practice/{session_id}",
            ),
        ],
        justify="space-between",
    )
    if data is None:
        body = dmc.Text(
            "Shots are segmented once processing reaches the Segments stage.", size="sm", c="dimmed"
        )
        return dmc.Paper([title, body], p="md", withBorder=True)
    u = units.current()
    items = []
    for b in data.blocks:
        rows = sorted(
            (r for r in data.rows if r["block_id"] == b["block_id"]), key=lambda r: r["t_contact"]
        )
        s = pr.summarize(rows)
        end = {-1: "near", 1: "far"}.get(b["side"], "?")
        kind = pr.KIND_LABELS.get(b["shot_kind"], "mixed").lower()
        shots = [
            html.Div(
                [
                    html.Span(fmt_t(r["t_contact"]), style={"fontFamily": "monospace"}),
                    html.Span(
                        " " + result_label(r, u),
                        style={"color": result_color(r), "fontWeight": 600},
                    ),
                ],
                id={"type": "review-seg-row", "index": r["segment_id"]},
                n_clicks=0,
                style={"cursor": "pointer", "fontSize": 12, "padding": "1px 0"},
            )
            for r in rows
        ]
        items.append(
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        dmc.Text(
                            f"Block {b['block_id'] + 1} · {fmt_t(b['start_t'])} · "
                            f"{b['n_shots']} {kind}s from the {end} end · in {pct(s['in_pct'])}",
                            size="xs",
                        )
                    ),
                    dmc.AccordionPanel(dmc.ScrollArea(html.Div(shots), mah=220, type="auto")),
                ],
                value=str(b["block_id"]),
            )
        )
    s = pr.summarize(data.rows)
    body = [
        dmc.Text(
            f"{s['n']} practice shots in {len(data.blocks)} blocks; in {pct(s['in_pct'])}, "
            f"net {pct(s['net_pct'])}"
            + (
                " (serves called against the service box)"
                if any(r["shot_kind"] == "serve" for r in data.rows)
                else ""
            )
            + ". N/P jump to the next/previous shot.",
            size="xs",
            c="dimmed",
        ),
        dmc.Accordion(items, multiple=True, variant="contained", chevronPosition="left"),
    ]
    return dmc.Paper([title, *body], p="md", withBorder=True)
