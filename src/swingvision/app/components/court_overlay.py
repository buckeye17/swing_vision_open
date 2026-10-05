"""Projected court lines for drawing over images and video (Plotly traces or SVG)."""

from __future__ import annotations

import base64

import numpy as np

from swingvision.court import model
from swingvision.court.camera import Camera

COURT_COLOR = "#ff3df5"
NET_COLOR = "#2ee6ff"


def polylines(cam: Camera, lines=model.COURT_LINES, spacing_m: float = 0.1) -> list[np.ndarray]:
    """Image-space polylines (curved by lens distortion) for each visible line piece."""
    out = []
    lim = 0.5 * max(cam.width, cam.height)
    for line in lines:
        pts = line.samples(spacing_m)
        px = cam.project(pts)
        ok = (
            (cam.depth(pts) > 0.3)
            & np.isfinite(px).all(axis=1)
            & (px[:, 0] > -lim)
            & (px[:, 0] < cam.width + lim)
            & (px[:, 1] > -lim)
            & (px[:, 1] < cam.height + lim)
        )
        # Split into runs of valid points.
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        for run in np.split(idx, breaks + 1):
            if len(run) >= 2:
                out.append(px[run])
    return out


def trace_xy(polys: list[np.ndarray]) -> tuple[list, list]:
    """Concatenate polylines with ``None`` gaps for a single Plotly scatter trace."""
    xs: list = []
    ys: list = []
    for p in polys:
        xs += [*np.round(p[:, 0], 1).tolist(), None]
        ys += [*np.round(p[:, 1], 1).tolist(), None]
    return xs, ys


def overlay_svg(cam: Camera, stroke_px: float | None = None) -> str:
    """Transparent SVG (as a data URI) of the court lines in full-resolution pixel space."""
    w = stroke_px or max(2.0, cam.width / 900)
    paths = []
    for color, lines in ((COURT_COLOR, model.COURT_LINES), (NET_COLOR, model.NET_LINES)):
        for p in polylines(cam, lines):
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)
            paths.append(
                f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w:.1f}" '
                'stroke-linejoin="round" stroke-opacity="0.85"/>'
            )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cam.width} {cam.height}" '
        f'preserveAspectRatio="none">{"".join(paths)}</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
