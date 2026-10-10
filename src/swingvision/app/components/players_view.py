"""Session-page pieces for player tracking: client-side track data, movement stats, heatmap."""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import pyarrow as pa
from dash import dcc

from swingvision.app import units
from swingvision.app.components.court_diagram import heatmap_figure
from swingvision.app.components.ui import fmt_duration, icon, stat_grid, stat_tile
from swingvision.players import movement as mv
from swingvision.storage import tables
from swingvision.storage.schemas import (
    BACKHAND_LABELS,
    HANDEDNESS_LABELS,
    MOVEMENT,
    PASS1_FRAMES,
    Profile,
    SessionConfig,
)
from swingvision.storage.session import Session

#: The browser gets at most this many samples per second of the track.
CLIENT_RATE_HZ = 15.0


def load_movement(session: Session) -> tuple[pa.Table, pa.Table]:
    movement = (
        tables.read_table(session.movement_path)
        if session.movement_path.exists()
        else MOVEMENT.empty_table()
    )
    frames = (
        tables.read_table(session.pass1_frames_path)
        if session.pass1_frames_path.exists()
        else PASS1_FRAMES.empty_table()
    )
    return movement, frames


def track_store(movement: pa.Table, width: int, height: int) -> dict | None:
    """Compact columns for ``assets/players_view.js``-style client callbacks.

    Boxes are in percent of the frame so they map straight onto the video element.
    """
    if movement.num_rows == 0:
        return None
    t = movement.column("t_s").to_numpy()
    keep = np.r_[True, np.diff(np.floor(t * CLIENT_RATE_HZ)) > 0]
    m = movement.filter(pa.array(keep))
    col = lambda name: m.column(name).to_numpy().astype(np.float64)  # noqa: E731
    return {
        "t": np.round(col("t_s"), 3).tolist(),
        "x": np.round(col("x"), 2).tolist(),
        "y": np.round(col("y"), 2).tolist(),
        "run": m.column("run").to_numpy().tolist(),
        "interp": [s == "interp" for s in m.column("source").to_pylist()],
        "b": np.round(
            np.column_stack(
                [
                    col("bx0") / width * 100,
                    col("by0") / height * 100,
                    col("bx1") / width * 100,
                    col("by1") / height * 100,
                ]
            ),
            2,
        ).tolist(),
        "gap": 1.0,
    }


def speed_series(movement: pa.Table, bin_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Peak speed per ``bin_s`` (for the timeline), NaN between runs."""
    if movement.num_rows == 0:
        return np.array([]), np.array([])
    t = movement.column("t_s").to_numpy()
    v = movement.column("speed").to_numpy().astype(np.float64)
    b = np.floor(t / bin_s).astype(np.int64)
    uniq, start = np.unique(b, return_index=True)
    peak = np.maximum.reduceat(v, start)
    tb = uniq * bin_s + bin_s / 2
    # Break the line where tracking stopped.
    gaps = np.flatnonzero(np.diff(tb) > 1.5)
    tb = np.insert(tb, gaps + 1, tb[gaps] + bin_s)
    peak = np.insert(peak, gaps + 1, np.nan)
    return tb, peak


def movement_card(
    movement: pa.Table, frames: pa.Table, dark_luma: float, view_min: float, fold: bool = False
):
    if frames.num_rows == 0:
        return dmc.Paper(
            [
                dmc.Title("Movement", order=5, mb="xs"),
                dmc.Text("Player tracking hasn't run yet.", size="sm", c="dimmed"),
            ],
            p="md",
            withBorder=True,
        )
    u = units.current()
    s = mv.summarize(movement, frames, dark_luma, mv.MovementParams(), view_min)
    xc, yc, H = mv.heatmap(movement, fold=fold)
    dark = s["dark_frames"] / max(1, s["processed_frames"])
    stats = stat_grid(
        [
            stat_tile("Distance", u.len_str(s["distance_m"], 0)),
            stat_tile(
                "Tracked", fmt_duration(s["tracked_s"]), f"{s['coverage']:.0%} of usable video"
            ),
            stat_tile("Top speed", u.speed_str_mps(s["max_speed_mps"]), "best 0.5 s"),
            stat_tile("Avg moving", u.speed_str_mps(s["mean_moving_speed_mps"])),
        ],
        plain=True,
    )
    notes = []
    if dark > 0.02:
        notes.append(
            f"{dark:.0%} of the video is too dark or doesn't show the court, and isn't counted."
        )
    if movement.num_rows:
        notes.append(f"{s['near_half_frac']:.0%} of tracked time on the near half.")
    return dmc.Paper(
        [
            dmc.Group(
                [
                    dmc.Title("Movement", order=5),
                    dmc.Switch(
                        id="review-heat-fold",
                        label="Fold both ends together",
                        size="xs",
                        checked=fold,
                    ),
                ],
                justify="space-between",
                mb="sm",
            ),
            stats,
            dmc.Text(" ".join(notes), size="xs", c="dimmed", mt="xs"),
            dcc.Graph(
                id="review-heatmap",
                figure=heatmap_figure(xc, yc, H, height=440),
                config={"displayModeBar": False},
            ),
            dmc.Text(
                "Time spent per ½ m (1.6 ft) square."
                if u.imperial
                else "Time spent per ½ m square.",
                size="xs",
                c="dimmed",
                ta="center",
            ),
        ],
        p="md",
        withBorder=True,
    )


def profile_facts(p: Profile | None) -> str:
    if p is None:
        return "No profile assigned."
    return f"{HANDEDNESS_LABELS[p.handedness]}, {BACKHAND_LABELS[p.backhand].lower()} backhand" + (
        f", {units.current().height_str(p.height_m)}" if p.height_m else ""
    )


def player_card(config: SessionConfig, profiles: list[Profile]):
    current = next((p for p in profiles if p.id == config.players.me_profile_id), None)
    facts = profile_facts(current)
    return dmc.Paper(
        [
            dmc.Group(
                [
                    dmc.Title("Player", order=5),
                    dmc.Anchor("Manage profiles", href="/profiles", size="xs"),
                ],
                justify="space-between",
                mb="xs",
            ),
            dmc.Select(
                id="review-profile",
                data=[{"value": p.id, "label": p.name} for p in profiles],
                value=current.id if current else None,
                placeholder="Choose a profile" if profiles else "Create a profile first",
                disabled=not profiles,
                clearable=True,
                size="xs",
                leftSection=icon("tabler:user", 14),
            ),
            dmc.Text(facts, id="review-profile-facts", size="xs", c="dimmed", mt=4),
            dmc.NumberInput(
                id="review-temp",
                label="Air temperature",
                description="For the speed of sound in speed calibration (20 °C if not set).",
                value=config.air_temp_c,
                placeholder="20 °C assumed",
                suffix=" °C",
                min=-30,
                max=50,
                step=1,
                decimalScale=1,
                debounce=True,
                size="xs",
                mt="sm",
            ),
        ],
        p="md",
        withBorder=True,
    )
