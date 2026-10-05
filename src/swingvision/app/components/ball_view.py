"""Ball track and events on the session page (M3)."""

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import pyarrow as pa

from swingvision.storage import tables

#: Ball positions sent to the browser per second (the video publishes its time at ~10 Hz;
#: 30 Hz keeps the trail smooth without a multi-megabyte store).
CLIENT_RATE_HZ = 30.0
BALL_COLOR = "#d8f5a2"
HIT_COLOR = "#ff922b"
BOUNCE_COLOR = "#4dabf7"
NET_COLOR = "#cc5de8"


def load_ball(session) -> tuple[pa.Table | None, pa.Table | None]:
    track = tables.read_table(session.ball_track_path) if session.ball_track_path.exists() else None
    events = tables.read_table(session.events_path) if session.events_path.exists() else None
    return track, events


def ball_store(track: pa.Table | None, width: int, height: int) -> dict | None:
    """Ball image positions in percent of the frame (for the video overlay)."""
    if track is None or track.num_rows == 0:
        return None
    t = track.column("t_s").to_numpy()
    keep = np.r_[True, np.diff(np.floor(t * CLIENT_RATE_HZ)) > 0]
    t = t[keep]
    x = track.column("x").to_numpy()[keep].astype(np.float64) / width * 100
    y = track.column("y").to_numpy()[keep].astype(np.float64) / height * 100
    return {
        "t": np.round(t, 3).tolist(),
        "x": np.round(x, 2).tolist(),
        "y": np.round(y, 2).tolist(),
    }


def events_store(events: pa.Table | None) -> dict | None:
    if events is None or events.num_rows == 0:
        return None

    def col(name):
        v = events.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
        return [None if not np.isfinite(a) else round(float(a), 3) for a in v]

    return {
        "t": col("t_s"),
        "kind": events.column("kind").to_pylist(),
        "cx": col("court_x"),
        "cy": col("court_y"),
    }


def ball_card(
    track: pa.Table | None, events: pa.Table | None, duration_s: float, fps: float = 60.0
):
    if track is None:
        body = [dmc.Text("Ball tracking hasn't run yet.", size="sm", c="dimmed")]
    else:
        src = track.column("source").to_pylist() if track.num_rows else []
        tracked_s = len(src) / max(fps, 1.0)
        kinds = events.column("kind").to_pylist() if events is not None else []
        body = [
            dmc.SimpleGrid(
                [
                    _stat(
                        "Ball seen", f"{tracked_s:.0f} s", f"{tracked_s / max(duration_s, 1):.0%}"
                    ),
                    _stat("Hits", str(kinds.count("hit")), None),
                    _stat("Bounces", str(kinds.count("bounce")), None),
                    _stat("Net", str(kinds.count("net")), None),
                ],
                cols=4,
                spacing="xs",
            ),
            dmc.Text(
                "Hits and bounces are on the timeline; bounces of the last 3 s show on the court.",
                size="xs",
                c="dimmed",
            ),
        ]
    return dmc.Paper([dmc.Title("Ball", order=5, mb="xs"), *body], p="md", withBorder=True)


def _stat(label: str, value: str, sub: str | None):
    return dmc.Stack(
        [
            dmc.Text(label, size="xs", c="dimmed"),
            dmc.Text(value, fw=700),
            dmc.Text(sub or "", size="xs", c="dimmed"),
        ],
        gap=0,
    )
