"""ITF court geometry in meters (PLAN.md §7.1).

Court frame: origin at the court center on the ground, ``x`` across the court
(positive to the camera's right), ``y`` along the court (positive away from the
camera, so the *near* baseline is at negative ``y``), ``z`` up. The frame is
right-handed.

Painted lines lie *inside* the nominal dimensions (court measurements are to
the outer edges of the lines), so every line and keypoint here is on the line's
**centerline**, which is what an image line detector finds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_LINE_FROM_NET = 6.40
LINE_WIDTH = 0.05
NET_POST_OUTSIDE = 0.914  # net posts stand this far outside the doubles sidelines
NET_HEIGHT_POST = 1.07
NET_HEIGHT_CENTER = 0.914
NET_TAPE_HALF = 0.03  # the measured net height is the top of a ~6 cm tape

HALF_LENGTH = COURT_LENGTH / 2
HALF_DOUBLES = DOUBLES_WIDTH / 2
HALF_SINGLES = SINGLES_WIDTH / 2

# Centerline coordinates of the painted lines.
X_DOUBLES = HALF_DOUBLES - LINE_WIDTH / 2
X_SINGLES = HALF_SINGLES - LINE_WIDTH / 2
Y_BASELINE = HALF_LENGTH - LINE_WIDTH / 2
Y_SERVICE = SERVICE_LINE_FROM_NET - LINE_WIDTH / 2
X_NET_POST = HALF_DOUBLES + NET_POST_OUTSIDE

# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------

#: 14 ground keypoints (line-centerline intersections), in a fixed order.
GROUND_KEYPOINTS: dict[str, tuple[float, float, float]] = {
    "far_doubles_left": (-X_DOUBLES, Y_BASELINE, 0.0),
    "far_singles_left": (-X_SINGLES, Y_BASELINE, 0.0),
    "far_singles_right": (X_SINGLES, Y_BASELINE, 0.0),
    "far_doubles_right": (X_DOUBLES, Y_BASELINE, 0.0),
    "far_service_left": (-X_SINGLES, Y_SERVICE, 0.0),
    "far_service_center": (0.0, Y_SERVICE, 0.0),
    "far_service_right": (X_SINGLES, Y_SERVICE, 0.0),
    "near_service_left": (-X_SINGLES, -Y_SERVICE, 0.0),
    "near_service_center": (0.0, -Y_SERVICE, 0.0),
    "near_service_right": (X_SINGLES, -Y_SERVICE, 0.0),
    "near_doubles_left": (-X_DOUBLES, -Y_BASELINE, 0.0),
    "near_singles_left": (-X_SINGLES, -Y_BASELINE, 0.0),
    "near_singles_right": (X_SINGLES, -Y_BASELINE, 0.0),
    "near_doubles_right": (X_DOUBLES, -Y_BASELINE, 0.0),
}

#: Non-planar points on the net. They pin down focal length and camera height (PnP).
NET_KEYPOINTS: dict[str, tuple[float, float, float]] = {
    "net_post_left": (-X_NET_POST, 0.0, NET_HEIGHT_POST),
    "net_center": (0.0, 0.0, NET_HEIGHT_CENTER),
    "net_post_right": (X_NET_POST, 0.0, NET_HEIGHT_POST),
}

KEYPOINTS: dict[str, tuple[float, float, float]] = {**GROUND_KEYPOINTS, **NET_KEYPOINTS}

KEYPOINT_LABELS: dict[str, str] = {name: name.replace("_", " ").capitalize() for name in KEYPOINTS}


def keypoint_array(names: list[str] | None = None) -> np.ndarray:
    names = names or list(KEYPOINTS)
    return np.array([KEYPOINTS[n] for n in names], dtype=np.float64)


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CourtLine:
    name: str
    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    #: ``"across"`` lines run parallel to the baselines (constant y); ``"along"`` lines
    #: run parallel to the sidelines (constant x); ``"net"`` is the tape on top of the net.
    family: str

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.p1, self.p0)))

    def samples(self, spacing_m: float) -> np.ndarray:
        n = max(2, int(np.ceil(self.length / spacing_m)) + 1)
        s = np.linspace(0.0, 1.0, n)[:, None]
        return (1 - s) * np.asarray(self.p0) + s * np.asarray(self.p1)


def _ground(name, x0, y0, x1, y1, family) -> CourtLine:
    return CourtLine(name, (x0, y0, 0.0), (x1, y1, 0.0), family)


COURT_LINES: tuple[CourtLine, ...] = (
    _ground("far_baseline", -X_DOUBLES, Y_BASELINE, X_DOUBLES, Y_BASELINE, "across"),
    _ground("far_service_line", -X_SINGLES, Y_SERVICE, X_SINGLES, Y_SERVICE, "across"),
    _ground("near_service_line", -X_SINGLES, -Y_SERVICE, X_SINGLES, -Y_SERVICE, "across"),
    _ground("near_baseline", -X_DOUBLES, -Y_BASELINE, X_DOUBLES, -Y_BASELINE, "across"),
    _ground("left_doubles_sideline", -X_DOUBLES, -Y_BASELINE, -X_DOUBLES, Y_BASELINE, "along"),
    _ground("left_singles_sideline", -X_SINGLES, -Y_BASELINE, -X_SINGLES, Y_BASELINE, "along"),
    _ground("center_service_line", 0.0, -Y_SERVICE, 0.0, Y_SERVICE, "along"),
    _ground("right_singles_sideline", X_SINGLES, -Y_BASELINE, X_SINGLES, Y_BASELINE, "along"),
    _ground("right_doubles_sideline", X_DOUBLES, -Y_BASELINE, X_DOUBLES, Y_BASELINE, "along"),
)

#: The net tape centerline as two straight pieces (post → center strap → post).
NET_LINES: tuple[CourtLine, ...] = (
    CourtLine(
        "net_left",
        (-X_NET_POST, 0.0, NET_HEIGHT_POST - NET_TAPE_HALF),
        (0.0, 0.0, NET_HEIGHT_CENTER - NET_TAPE_HALF),
        "net",
    ),
    CourtLine(
        "net_right",
        (0.0, 0.0, NET_HEIGHT_CENTER - NET_TAPE_HALF),
        (X_NET_POST, 0.0, NET_HEIGHT_POST - NET_TAPE_HALF),
        "net",
    ),
)

#: Coordinates of the across (constant y) and along (constant x) lines, far→near and left→right.
ACROSS_Y = np.array([Y_BASELINE, Y_SERVICE, -Y_SERVICE, -Y_BASELINE])
ALONG_X = np.array([-X_DOUBLES, -X_SINGLES, 0.0, X_SINGLES, X_DOUBLES])


def line_samples(
    spacing_m: float = 0.25, include_net: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Points along every court line → ``(points (N,3), line_index (N,))``.

    ``line_index`` indexes ``COURT_LINES + NET_LINES``.
    """
    lines = COURT_LINES + (NET_LINES if include_net else ())
    pts, idx = [], []
    for i, line in enumerate(lines):
        s = line.samples(spacing_m)
        pts.append(s)
        idx.append(np.full(len(s), i))
    return np.concatenate(pts), np.concatenate(idx)


# ---------------------------------------------------------------------------
# Zones and helpers
# ---------------------------------------------------------------------------

#: Named rectangles (x0, y0, x1, y1) on the *far* half (the half the near player hits into).
#: :func:`zone_at` mirrors near-half points onto these. Deuce/ad follow the far
#: receiver, who faces the camera: their deuce (right) court is at negative x.
ZONES: dict[str, tuple[float, float, float, float]] = {
    "deuce_box": (-HALF_SINGLES, 0.0, 0.0, SERVICE_LINE_FROM_NET),
    "ad_box": (0.0, 0.0, HALF_SINGLES, SERVICE_LINE_FROM_NET),
    "backcourt": (-HALF_SINGLES, SERVICE_LINE_FROM_NET, HALF_SINGLES, HALF_LENGTH),
    "left_alley": (-HALF_DOUBLES, 0.0, -HALF_SINGLES, HALF_LENGTH),
    "right_alley": (HALF_SINGLES, 0.0, HALF_DOUBLES, HALF_LENGTH),
}


def mirror(xy: np.ndarray) -> np.ndarray:
    """Rotate court coordinates by 180° about the center (swap ends). Works on (..., 2|3)."""
    out = np.array(xy, dtype=np.float64, copy=True)
    out[..., 0] *= -1
    out[..., 1] *= -1
    return out


def zone_at(x: float, y: float) -> str | None:
    """Name of the zone containing (x, y) on the far half, ``"near:<zone>"`` on the near
    half, or ``None`` outside the doubles court."""
    prefix = ""
    if y < 0:
        x, y = -x, -y
        prefix = "near:"
    for name, (x0, y0, x1, y1) in ZONES.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return prefix + name
    return None


def in_court(x: np.ndarray, y: np.ndarray, singles: bool = True, margin: float = 0.0) -> np.ndarray:
    """Whether ground points are inside the (singles) court, ``margin`` meters of tolerance."""
    half_w = (HALF_SINGLES if singles else HALF_DOUBLES) + margin
    return (np.abs(x) <= half_w) & (np.abs(y) <= HALF_LENGTH + margin)


def roi_polygon(behind_m: float = 6.0, beside_m: float = 4.0) -> np.ndarray:
    """Ground polygon (4,3) of the playing area used to filter detections (PLAN.md §7.2)."""
    x = HALF_DOUBLES + beside_m
    y = HALF_LENGTH + behind_m
    return np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]], dtype=np.float64)
