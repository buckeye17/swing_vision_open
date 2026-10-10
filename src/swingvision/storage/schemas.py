"""Data contracts: pydantic models for JSON files, PyArrow schemas for Parquet files.

Session configs already carry match-mode fields (PLAN.md §1a) so Phase 2 does
not need a data migration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# JSON models
# ---------------------------------------------------------------------------

Mode = Literal["practice", "match"]
PracticeSubmode = Literal["self_feed", "ball_machine", "serve"]
PRACTICE_SUBMODE_LABELS: dict[str, str] = {
    "self_feed": "Self-feed / basket",
    "ball_machine": "Ball machine",
    "serve": "Serve practice",
}

SESSION_SCHEMA_VERSION = 1


class SourceInfo(BaseModel):
    path: str
    size_bytes: int
    fast_hash: str
    mtime: float


class VideoInfo(BaseModel):
    codec: str
    profile: str | None = None
    width: int  # coded size, before rotation
    height: int
    rotation_cw: int = 0  # clockwise degrees needed to display upright (0/90/180/270)
    display_width: int
    display_height: int
    pix_fmt: str | None = None
    fps_nominal: float
    fps_avg: float
    is_vfr: bool
    duration_s: float
    n_frames_est: int
    bit_rate: int | None = None
    start_time_s: float = 0.0
    has_audio: bool = False
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    audio_start_time_s: float | None = None
    creation_time: str | None = None


class Target(BaseModel):
    """Practice target in court meters (M5). ``frame='relative'`` mirrors with hitting side."""

    id: str
    name: str
    shape: Literal["rect", "circle"]
    frame: Literal["relative", "absolute"] = "relative"
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    cx: float | None = None
    cy: float | None = None
    r: float | None = None
    strokes: list[str] = Field(default_factory=list)


class TargetSet(BaseModel):
    """A named, reusable list of targets (``target_sets`` table)."""

    id: str
    name: str
    targets: list[Target] = Field(default_factory=list)
    created_at: str
    updated_at: str


class PracticeShotEdit(BaseModel):
    """A user correction to one practice shot, found again by its anchor time (the hit, or
    the landing when the contact wasn't seen) so it survives re-segmentation."""

    t: float
    exclude: bool = False
    #: Landing placed by the user, court meters.
    landing: list[float] | None = None
    #: The user checked the landing (detected or placed) on the video.
    confirmed: bool = False


class SwingEdit(BaseModel):
    """The user's stroke for the swing whose contact is at ``t`` (M6). Also a training label
    for the stroke classifier."""

    t: float
    stroke: str


class SessionEdits(BaseModel):
    """``edits.json``: user overrides layered over derived data (PLAN.md §9.3)."""

    version: int = 0
    practice_shots: list[PracticeShotEdit] = Field(default_factory=list)
    swings: list[SwingEdit] = Field(default_factory=list)


class PracticeConfig(BaseModel):
    submode: PracticeSubmode = "self_feed"
    targets: list[Target] = Field(default_factory=list)
    #: Ball-machine position on the court (m), placed by the user. ``None``: auto-detect.
    machine_xy: list[float] | None = None


class MatchFormat(BaseModel):
    """Phase 2. Present now so session files never need migrating."""

    preset: str | None = None
    sets_to_win: int = 2
    games_per_set: int = 6
    win_by_two_games: bool = True
    tiebreak_at: int | None = 6
    tiebreak_points: int = 7
    no_ad: bool = False
    final_set: Literal["normal", "match_tiebreak", "advantage"] = "normal"
    pro_set_games: int | None = None
    free_play: bool = False


class MatchConfig(BaseModel):
    format: MatchFormat = Field(default_factory=MatchFormat)
    first_server: Literal["auto", "me", "opponent"] = "auto"
    switch_ends: Literal["per_rules", "never", "auto"] = "per_rules"


class PlayersConfig(BaseModel):
    me_profile_id: str | None = None
    opponent_profile_id: str | None = None


class SessionConfig(BaseModel):
    schema_version: int = SESSION_SCHEMA_VERSION
    id: str
    name: str
    created_at: datetime
    dir_name: str
    source: SourceInfo
    video: VideoInfo | None = None
    mode: Mode
    practice: PracticeConfig | None = None
    match: MatchConfig | None = None
    players: PlayersConfig = Field(default_factory=PlayersConfig)
    notes: str = ""


Handedness = Literal["right", "left"]
Backhand = Literal["two_handed", "one_handed"]
HANDEDNESS_LABELS: dict[str, str] = {"right": "Right-handed", "left": "Left-handed"}
BACKHAND_LABELS: dict[str, str] = {"two_handed": "Two-handed", "one_handed": "One-handed"}


class Profile(BaseModel):
    """A player profile (``profiles`` table). Appearance galleries arrive in Phase 2."""

    id: str
    name: str
    handedness: Handedness = "right"
    backhand: Backhand = "two_handed"
    height_m: float | None = Field(default=None, ge=1.0, le=2.5)
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Players (PLAN.md §7.2)
# ---------------------------------------------------------------------------


class StaticObject(BaseModel):
    """Something person-like that never moves (ball machine, bag on a bench, a post)."""

    x: float
    y: float
    t0_s: float
    t1_s: float
    seen_s: float  # time covered by its detections
    height_m: float | None = None


class MachineInfo(BaseModel):
    x: float
    y: float
    source: Literal["user", "auto"]


class PlayersSummary(BaseModel):
    """``players/identity.json``: who is who, and how well "me" was tracked."""

    method: Literal["single_player"] = "single_player"
    n_detections: int = 0
    n_in_roi: int = 0
    n_tracklets: int = 0
    me_tracklets: list[int] = Field(default_factory=list)
    me_detections: int = 0
    static_objects: list[StaticObject] = Field(default_factory=list)
    machine: MachineInfo | None = None


# ---------------------------------------------------------------------------
# Court calibration (PLAN.md §7.1)
# ---------------------------------------------------------------------------

CALIBRATION_SCHEMA_VERSION = 1


class CameraParams(BaseModel):
    """Serialized :class:`swingvision.court.camera.Camera` (full-resolution display px)."""

    width: int
    height: int
    f: float
    k1: float
    k2: float = 0.0
    cx: float
    cy: float
    rvec: list[float]
    tvec: list[float]


class CalibrationMetrics(BaseModel):
    rms_line_px: float | None = None  # line-center samples vs projected court lines
    n_line_samples: int = 0
    n_expected_samples: int = 0  # samples the visible lines would give, unoccluded
    n_net_samples: int = 0
    rms_points_px: float | None = None  # user-placed points vs their projections

    @property
    def coverage(self) -> float:
        return self.n_line_samples / self.n_expected_samples if self.n_expected_samples else 0.0


class DriftWindow(BaseModel):
    """Calibration re-checked on one time window of the video (tripod bumped?)."""

    index: int | None = None  # window number; its median image is court/windows/wNN.jpg
    t0_s: float
    t1_s: float
    n_frames: int
    luminance: float | None = None
    status: Literal["ok", "moved", "dark", "failed"]
    #: Keypoint displacement vs the session calibration. Windows that "moved" keep their
    #: own pose-refined ``camera`` (piecewise calibration).
    shift_rms_px: float | None = None
    shift_max_px: float | None = None
    rms_line_px: float | None = None
    camera: CameraParams | None = None


class Calibration(BaseModel):
    """``court/auto.json`` (detected), ``court/user.json`` (confirmed in the editor), and
    ``calibration.json`` (the one downstream stages use) all share this model."""

    schema_version: int = CALIBRATION_SCHEMA_VERSION
    source: Literal["auto", "user"]
    created_at: datetime
    ok: bool = True  # False: detection failed and ``camera`` is only a starting guess
    message: str = ""
    camera: CameraParams
    #: Projected keypoints (``court.model.KEYPOINTS``), px; may lie outside the image.
    keypoints: dict[str, list[float] | None] = Field(default_factory=dict)
    #: Points the user placed by hand (they constrain the fit), px.
    user_points: dict[str, list[float]] = Field(default_factory=dict)
    metrics: CalibrationMetrics = Field(default_factory=CalibrationMetrics)
    camera_summary: dict = Field(default_factory=dict)  # height, distance, fov (for display)
    frame_times_s: list[float] = Field(default_factory=list)
    drift: list[DriftWindow] = Field(default_factory=list)
    drift_detected: bool = False
    dark_fraction: float = 0.0
    #: In ``calibration.json``: who accepted it ("user", or "auto" under the threshold).
    confirmed_by: Literal["user", "auto"] | None = None
    #: In ``court/user.json``: fingerprint of the auto calibration it started from.
    based_on: str | None = None


# ---------------------------------------------------------------------------
# Parquet (PyArrow) schemas
# ---------------------------------------------------------------------------


def field(
    name: str, type_: pa.DataType, unit: str | None = None, desc: str | None = None, nullable=True
) -> pa.Field:
    meta = {}
    if unit:
        meta["unit"] = unit
    if desc:
        meta["description"] = desc
    return pa.field(name, type_, nullable=nullable, metadata=meta or None)


def table_schema(name: str, version: int, fields: list[pa.Field]) -> pa.Schema:
    return pa.schema(fields, metadata={"name": name, "schema_version": str(version)})


AUDIO_ONSETS = table_schema(
    "audio_onsets",
    1,
    [
        field("t_s", pa.float64(), "s", "Onset time on the video timeline", nullable=False),
        field("strength", pa.float32(), "z", "Robust z-score of spectral flux", nullable=False),
        field("rms_db", pa.float32(), "dB", "Peak RMS in a ±10 ms window"),
        field("centroid_hz", pa.float32(), "Hz", "Spectral centroid at the onset"),
        field("flatness", pa.float32(), None, "Spectral flatness (1 = noise-like/broadband)"),
    ],
)

PASS1_FRAMES = table_schema(
    "pass1_frames",
    1,
    [
        field("frame", pa.int64(), None, "Presentation-order frame number", nullable=False),
        field("t_s", pa.float64(), "s", "Time on the video timeline", nullable=False),
        field("luma", pa.float32(), None, "Mean luma 0-255 (darkness check)"),
        field(
            "view",
            pa.float32(),
            None,
            "Correlation with the court background (< 0.6: camera not on the court)",
        ),
    ],
)

PERSON_DETECTIONS = table_schema(
    "person_detections",
    1,
    [
        field("frame", pa.int64(), None, "Presentation-order frame number", nullable=False),
        field("t_s", pa.float64(), "s", "Time on the video timeline", nullable=False),
        field("x0", pa.float32(), "px", "Box left (full-resolution display pixels)"),
        field("y0", pa.float32(), "px", "Box top"),
        field("x1", pa.float32(), "px", "Box right"),
        field("y1", pa.float32(), "px", "Box bottom"),
        field("conf", pa.float32(), None, "Detector confidence"),
    ],
)

#: Every person detection with its court position and tracking annotations.
PLAYER_TRACKS = table_schema(
    "player_tracks",
    1,
    [
        *PERSON_DETECTIONS,
        field("court_x", pa.float32(), "m", "Foot point on the court (x across)"),
        field("court_y", pa.float32(), "m", "Foot point on the court (y along, + away)"),
        field("height_m", pa.float32(), "m", "Box height at the foot point (NaN: box cut off)"),
        field("sigma_x", pa.float32(), "m", "1-σ foot position across, from box jitter"),
        field("sigma_y", pa.float32(), "m", "1-σ foot position along the court"),
        field("track_id", pa.int32(), None, "Tracklet id (-1: not tracked)"),
        field("role", pa.string(), None, "me | other | static | outside"),
    ],
)

MOVEMENT = table_schema(
    "movement",
    1,
    [
        field("frame", pa.int64(), None, "Presentation-order frame number", nullable=False),
        field("t_s", pa.float64(), "s", "Time on the video timeline", nullable=False),
        field("player", pa.string(), None, "me | opponent", nullable=False),
        field("x", pa.float32(), "m", "Smoothed court position (x across)"),
        field("y", pa.float32(), "m", "Smoothed court position (y along, + away)"),
        field("vx", pa.float32(), "m/s"),
        field("vy", pa.float32(), "m/s"),
        field("speed", pa.float32(), "m/s"),
        field("sigma_m", pa.float32(), "m", "1-σ position uncertainty (major axis)"),
        field("source", pa.string(), None, "bbox | interp (gap bridged, no detection)"),
        field("run", pa.int32(), None, "Continuous tracked stretch number"),
        field("bx0", pa.float32(), "px", "Player box (detected or interpolated)"),
        field("by0", pa.float32(), "px"),
        field("bx1", pa.float32(), "px"),
        field("by1", pa.float32(), "px"),
    ],
)

# ---------------------------------------------------------------------------
# Ball (PLAN.md §7.4-7.5)
# ---------------------------------------------------------------------------

BALL_CANDIDATES = table_schema(
    "ball_candidates",
    1,
    [
        field("frame", pa.int64(), None, "Presentation-order frame number", nullable=False),
        field("t_s", pa.float64(), "s", "Time on the video timeline", nullable=False),
        field("x", pa.float32(), "px", "Candidate center (full-resolution display pixels)"),
        field("y", pa.float32(), "px"),
        field("score", pa.float32(), None, "Detector score (detector-specific scale)"),
    ],
)

#: Frames a ball detector looked at (a frame without candidates had no ball found).
BALL_FRAMES = table_schema(
    "ball_frames",
    1,
    [
        field("frame", pa.int64(), None, "Presentation-order frame number", nullable=False),
        field("t_s", pa.float64(), "s", "Time on the video timeline", nullable=False),
    ],
)

#: The active ball per frame after linking (``ball_track``).
BALL_TRACK = table_schema(
    "ball_track",
    1,
    [
        field("frame", pa.int64(), None, "Presentation-order frame number", nullable=False),
        field("t_s", pa.float64(), "s", "Time on the video timeline", nullable=False),
        field("x", pa.float32(), "px", "Ball center (full-resolution display pixels)"),
        field("y", pa.float32(), "px"),
        field("score", pa.float32(), None, "Detection confidence 0-1 (0 for interpolated)"),
        field("tracklet", pa.int32(), None, "Linked tracklet the point belongs to"),
        field("source", pa.string(), None, "detected | interp (gap filled)"),
    ],
)

#: Hits, bounces and net contacts (``events``).
EVENTS = table_schema(
    "events",
    1,
    [
        field("event_id", pa.int32(), None, "Event number within the session", nullable=False),
        field("kind", pa.string(), None, "hit | bounce | net", nullable=False),
        field("frame", pa.int64(), None, "Frame nearest to the contact", nullable=False),
        field("t_s", pa.float64(), "s", "Contact time (sub-frame; audio-refined for hits)"),
        field("x", pa.float32(), "px", "Ball at the contact (full-resolution display pixels)"),
        field("y", pa.float32(), "px"),
        field("court_x", pa.float32(), "m", "Bounce: ground point; hit: hitter's position"),
        field("court_y", pa.float32(), "m"),
        field("conf", pa.float32(), None, "Confidence 0-1"),
        field("source", pa.string(), None, "rules | model | user"),
        field("hitter", pa.string(), None, "Hits: me | opponent | machine | unknown"),
        field("audio_dt", pa.float32(), "s", "Matched onset minus contact (after delay)"),
        field("audio_strength", pa.float32(), "z", "Strength of the matched audio onset"),
    ],
)

# ---------------------------------------------------------------------------
# 3D flights and shots (PLAN.md §7.6, §8)
# ---------------------------------------------------------------------------

#: One fitted 3D flight between two events (``ball_3d``).
BALL_FLIGHTS = table_schema(
    "ball_flights",
    1,
    [
        field("flight_id", pa.int32(), None, "Flight number within the session", nullable=False),
        field("start_kind", pa.string(), None, "hit | machine | bounce | free (run start)"),
        field("start_event_id", pa.int32(), None, "Event the flight starts at (events.parquet)"),
        field("end_kind", pa.string(), None, "bounce | net | hit | lost (track ended)"),
        field("end_event_id", pa.int32(), None, "Event the flight ends at"),
        field("hitter", pa.string(), None, "Flights from hits: me | opponent | machine | unknown"),
        field("t0_s", pa.float64(), "s", "Start (contact) time"),
        field("t1_s", pa.float64(), "s", "End time (end event, or the last detection)"),
        field("n_obs", pa.int32(), None, "Detections used by the fit"),
        field("rms_px", pa.float32(), "px", "Reprojection RMS of the detections"),
        field("chi2", pa.float32(), None, "Normalized fit cost per degree of freedom"),
        field("ok", pa.bool_(), None, "The fit converged"),
        field("p0_x", pa.float64(), "m", "Position at t0"),
        field("p0_y", pa.float64(), "m"),
        field("p0_z", pa.float64(), "m"),
        field("v0_x", pa.float64(), "m/s", "Velocity at t0"),
        field("v0_y", pa.float64(), "m/s"),
        field("v0_z", pa.float64(), "m/s"),
        field("spin", pa.float64(), "1/m", "Magnus coefficient (+ topspin, - backspin)"),
        field("cd", pa.float64(), None, "Drag coefficient"),
        field("speed0", pa.float32(), "m/s", "Speed at t0 (off the racket for hits)"),
        field("speed_net", pa.float32(), "m/s", "Speed crossing the net plane"),
        field("speed_end", pa.float32(), "m/s", "Speed at t1 (before the bounce)"),
        field("speed_avg", pa.float32(), "m/s", "Path length / flight time"),
        field("net_x", pa.float32(), "m", "Where the path crosses the net plane"),
        field("net_z", pa.float32(), "m", "Ball center height there"),
        field("net_clearance", pa.float32(), "m", "net_z minus the net's height there"),
        field("apex_z", pa.float32(), "m", "Highest point"),
        field("end_x", pa.float32(), "m", "Position at t1"),
        field("end_y", pa.float32(), "m"),
        field("end_z", pa.float32(), "m"),
        field("vz0", pa.float32(), "m/s", "Vertical velocity at t0"),
        field("vz_end", pa.float32(), "m/s", "Vertical velocity at t1"),
        field("landing_x", pa.float32(), "m", "End on the ground, or the extended path's"),
        field("landing_y", pa.float32(), "m"),
        field("landing_t", pa.float32(), "s"),
        field("speed0_sigma", pa.float32(), "m/s", "1-σ of speed0 (fit covariance)"),
        field("speed_avg_sigma", pa.float32(), "m/s"),
        field("net_clearance_sigma", pa.float32(), "m"),
        field("apex_sigma", pa.float32(), "m"),
        field("landing_sigma", pa.float32(), "m", "1-σ of the fitted landing point"),
        field("spin_sigma", pa.float32(), "1/m"),
        field("flags", pa.list_(pa.string()), None, "Quality flags"),
    ],
)

#: Sampled positions along each fitted flight (for drawing).
BALL_FLIGHT_PATHS = table_schema(
    "ball_flight_paths",
    1,
    [
        field("flight_id", pa.int32(), None, nullable=False),
        field("t_s", pa.float64(), "s", nullable=False),
        field("x", pa.float32(), "m"),
        field("y", pa.float32(), "m"),
        field("z", pa.float32(), "m"),
    ],
)

#: One row per hit (``shots``; PLAN.md §8). Stroke and swing fields arrive with M6,
#: segments with M5.
SHOTS = table_schema(
    "shots",
    1,
    [
        field("shot_id", pa.int32(), None, "Shot number within the session", nullable=False),
        field("session_id", pa.string(), None, nullable=False),
        field("segment_id", pa.int32(), None, "Practice shot / point segment (M5)"),
        field("hitter", pa.string(), None, "me | opponent | machine | unknown"),
        field("hit_event_id", pa.int32(), None, "The hit in events.parquet"),
        field("flight_id", pa.int32(), None, "The fitted flight (ball/flights.parquet)"),
        field("frame_contact", pa.int64(), None, "Frame nearest to the contact"),
        field("t_contact", pa.float64(), "s", "Contact time"),
        field("stroke_type", pa.string(), None, "serve | forehand | backhand | ... (M6)"),
        field("stroke_conf", pa.float32(), None),
        field("is_serve", pa.bool_(), None, "(M6/M9)"),
        field("serve_number", pa.int8(), None, "1 or 2 (match mode)"),
        field("contact_x", pa.float32(), "m", "Contact point (fit; hitter's feet without one)"),
        field("contact_y", pa.float32(), "m"),
        field("contact_height", pa.float32(), "m"),
        field("side", pa.int8(), None, "Hitter's half: -1 near (camera side), +1 far"),
        field("end_kind", pa.string(), None, "bounce | net | hit | lost | none"),
        field("landing_x", pa.float32(), "m", "Where the ball first came down"),
        field("landing_y", pa.float32(), "m"),
        field("landing_sigma_m", pa.float32(), "m", "1-σ landing position (major axis)"),
        field("landing_source", pa.string(), None, "bounce (detected) | fit (extended path)"),
        field("landing_in", pa.bool_(), None, "Inside the opponent's singles court"),
        field("landing_margin_m", pa.float32(), "m", "Distance inside (+) / outside (-) the lines"),
        field("landing_zone", pa.string(), None, "Zone on the opponent's half (own half: near:)"),
        field("speed_racket_kmh", pa.float32(), "km/h", "Speed off the racket"),
        field("speed_net_kmh", pa.float32(), "km/h", "Speed crossing the net"),
        field("speed_avg_kmh", pa.float32(), "km/h", "Path length / flight time"),
        field("speed_bounce_kmh", pa.float32(), "km/h", "Speed just before the bounce"),
        field("speed_sigma_kmh", pa.float32(), "km/h", "1-σ of the speed off the racket"),
        field("net_clearance_m", pa.float32(), "m", "Ball center above the net"),
        field("apex_m", pa.float32(), "m", "Highest point of the flight"),
        field("spin_sign", pa.int8(), None, "+1 topspin, -1 backspin/slice, 0 unclear"),
        field("flight_time_s", pa.float32(), "s", "Contact to landing (or end)"),
        field("outcome", pa.string(), None, "in | out_long | out_wide | net | own_side | unknown"),
        field("swing_id", pa.int32(), None, "(M6)"),
        field("fit_rms_px", pa.float32(), "px"),
        field("quality_flags", pa.list_(pa.string()), None),
    ],
)

# ---------------------------------------------------------------------------
# Segments and practice accuracy (PLAN.md §7.8, §7.10, §8)
# ---------------------------------------------------------------------------

#: Practice shots and the blocks they're grouped into (``segments``); match points and
#: warm-ups arrive with Phase 2 in the same file.
SEGMENTS = table_schema(
    "segments",
    2,
    [
        field("segment_id", pa.int32(), None, "Segment number within the session", nullable=False),
        field("kind", pa.string(), None, "practice_shot | block (point | warmup: Phase 2)"),
        field("start_t", pa.float64(), "s", "Segment start (feed minus padding)"),
        field("end_t", pa.float64(), "s", "Segment end (landing plus padding)"),
        field("block_id", pa.int32(), None, "Block the shot belongs to / the block's number"),
        field("shot_id", pa.int32(), None, "shots.parquet row (null: the contact wasn't seen)"),
        field("hit_event_id", pa.int32(), None, "The hit in events.parquet"),
        field("feed_event_id", pa.int32(), None, "Ball-machine feed (hit by machine)"),
        field("feed_kind", pa.string(), None, "machine | drop | dribble | toss | none"),
        field("t_feed", pa.float64(), "s", "Start of the feed (drop, dribbles, machine feed)"),
        field("t_contact", pa.float64(), "s", "Contact time (estimated when not seen)"),
        field("contact_source", pa.string(), None, "hit | audio (impact sound) | estimate"),
        field("t_end", pa.float64(), "s", "Landing, net contact or end of the tracked flight"),
        field("end_reason", pa.string(), None, "bounce | net | lost | none"),
        field("landing_event_id", pa.int32(), None, "Bounce the shot landed with"),
        field("landing_x", pa.float32(), "m", "Where the ball first came down (court)"),
        field("landing_y", pa.float32(), "m"),
        field("landing_sigma_m", pa.float32(), "m", "1-σ landing position (major axis)"),
        field("landing_source", pa.string(), None, "bounce (detected) | fit (extended path)"),
        field("hitter", pa.string(), None, "me | opponent | machine | unknown"),
        field("side", pa.int8(), None, "Hitter's half: -1 near (camera side), +1 far"),
        field("shot_kind", pa.string(), None, "serve | groundstroke | unknown"),
        field("serve_side", pa.string(), None, "Serves: deuce | ad (server's half)"),
        field("swing_id", pa.int32(), None, "The player's swing at the contact (M6)"),
        field("stroke_type", pa.string(), None, "serve | forehand | backhand | ... (M6)"),
        field("stroke_conf", pa.float32(), None),
        field("n_shots", pa.int32(), None, "Blocks: practice shots in the block"),
        field("conf", pa.float32(), None, "Confidence 0-1 that this is a practice shot"),
        field("flags", pa.list_(pa.string()), None),
        field("point_index", pa.int32(), None, "(Phase 2)"),
        field("server", pa.string(), None, "(Phase 2)"),
        field("first_serve_in", pa.bool_(), None, "(Phase 2)"),
        field("rally_length", pa.int32(), None, "(Phase 2)"),
    ],
)

#: One row per practice shot: line call, targets, accuracy (``practice_eval``).
PRACTICE = table_schema(
    "practice",
    1,
    [
        field("segment_id", pa.int32(), None, "The shot's segment", nullable=False),
        field("block_id", pa.int32(), None),
        field("shot_id", pa.int32(), None, "shots.parquet row (null: contact not seen)"),
        field("t_contact", pa.float64(), "s"),
        field("t_end", pa.float64(), "s", "Landing / net / end time"),
        field("side", pa.int8(), None, "Hitter's half: -1 near, +1 far"),
        field("shot_kind", pa.string(), None, "serve | groundstroke | unknown"),
        field("serve_side", pa.string(), None, "deuce | ad"),
        field("stroke_type", pa.string(), None, "(M6)"),
        field("landing_x", pa.float32(), "m", "Landing on the court (after user edits)"),
        field("landing_y", pa.float32(), "m"),
        field("landing_sigma_m", pa.float32(), "m"),
        field("landing_source", pa.string(), None, "bounce | fit | user"),
        field("landing_confirmed", pa.bool_(), None, "The user confirmed or placed the landing"),
        field("rel_x", pa.float32(), "m", "Landing in the hitter's frame (hitter at the near end)"),
        field("rel_y", pa.float32(), "m"),
        field("outcome", pa.string(), None, "in | out_long | out_wide | net | own_side | unknown"),
        field("call_area", pa.string(), None, "singles | deuce_box | ad_box (what was called)"),
        field("margin_m", pa.float32(), "m", "Inside (+) / outside (-) the called area"),
        field("close_call", pa.bool_(), None, "|margin| below 2 landing σ"),
        field("targets", pa.list_(pa.string()), None, "Ids of the targets that apply"),
        field("target_id", pa.string(), None, "Nearest applicable target"),
        field("in_target", pa.bool_(), None, "Landed in an applicable target (null: n/a)"),
        field("targets_hit", pa.list_(pa.string()), None),
        field("target_dist_m", pa.float32(), "m", "Landing to the nearest target's center"),
        field("target_edge_m", pa.float32(), "m", "Inside (+) / outside (-) that target"),
        field("depth_err_m", pa.float32(), "m", "Deeper (+) / shorter (-) than its center"),
        field("width_err_m", pa.float32(), "m", "Hitter's right (+) / left (-) of its center"),
        field("speed_kmh", pa.float32(), "km/h", "Speed off the racket (uncalibrated)"),
        field("speed_err_kmh", pa.float32(), "km/h", "Uncalibrated error bound"),
        field("feed_speed_kmh", pa.float32(), "km/h", "Ball-machine feed speed"),
        field("feed_land_x", pa.float32(), "m", "Ball-machine feed bounce"),
        field("feed_land_y", pa.float32(), "m"),
        field("excluded", pa.bool_(), None, "Marked 'not a practice shot' by the user"),
        field("flags", pa.list_(pa.string()), None),
    ],
)

# ---------------------------------------------------------------------------
# Pose and swings (PLAN.md §7.7, §8)
# ---------------------------------------------------------------------------

#: 2D pose of the player on every frame of the swing windows (``pass2_pose``).
POSE2D = table_schema(
    "pose2d",
    1,
    [
        field("frame", pa.int64(), None, "Presentation-order frame number", nullable=False),
        field("t_s", pa.float64(), "s", "Time on the video timeline", nullable=False),
        field("player", pa.string(), None, "me | opponent", nullable=False),
        field("window", pa.int32(), None, "Pose window (merged around hits and impact sounds)"),
        field(
            "kp",
            pa.list_(pa.float32(), 51),
            "px",
            "COCO-17 keypoints: x, y (full-resolution display pixels), confidence",
        ),
        field("x0", pa.float32(), "px", "Crop the pose network saw"),
        field("y0", pa.float32(), "px"),
        field("x1", pa.float32(), "px"),
        field("y1", pa.float32(), "px"),
        field("score", pa.float32(), None, "Mean keypoint confidence"),
    ],
)

#: 3D joints on the court (``pose3d``).
POSE3D = table_schema(
    "pose3d",
    1,
    [
        field("frame", pa.int64(), None, "Presentation-order frame number", nullable=False),
        field("t_s", pa.float64(), "s", "Time on the video timeline", nullable=False),
        field("player", pa.string(), None, "me | opponent", nullable=False),
        field("clip", pa.int32(), None, "Lifted clip (consecutive frames of one window)"),
        field(
            "joints",
            pa.list_(pa.float32(), 51),
            "m",
            "Human3.6M-17 joints x, y, z in court coordinates (z up)",
        ),
        field("conf", pa.float32(), None, "Mean 2D keypoint confidence"),
        field("reproj_px", pa.float32(), "px", "Joint reprojection RMS after placement"),
        field("scale", pa.float32(), "m", "Meters per normalized unit (from the player's height)"),
    ],
)

#: One row per swing (``swings``; PLAN.md §8).
SWINGS = table_schema(
    "swings",
    1,
    [
        field("swing_id", pa.int32(), None, "Swing number within the session", nullable=False),
        field("player", pa.string(), None, "me | opponent"),
        field("t_contact", pa.float64(), "s", "Contact time"),
        field("frame_contact", pa.int64(), None, "Frame nearest to the contact"),
        field("contact_source", pa.string(), None, "hit (ball event) | audio | pose"),
        field("t_contact_pose", pa.float64(), "s", "Contact estimated from the pose alone"),
        field("hit_event_id", pa.int32(), None, "The hit in events.parquet"),
        field("side", pa.int8(), None, "Player's half: -1 near (camera side), +1 far"),
        field("racket_hand", pa.string(), None, "left | right"),
        field(
            "stroke_type",
            pa.string(),
            None,
            "serve | forehand | backhand | forehand_volley | backhand_volley | overhead | other",
        ),
        field("stroke_conf", pa.float32(), None, "Confidence 0-1"),
        field("stroke_source", pa.string(), None, "rules | model | user"),
        field("stroke_rules", pa.string(), None, "What the rules said (before model/user)"),
        field("t_start", pa.float64(), "s", "Preparation (unit turn / toss) starts"),
        field("t_split", pa.float64(), "s", "Split step (vertical hop) before the preparation"),
        field(
            "t_backswing_end", pa.float64(), "s", "Racket hand furthest back: forward swing starts"
        ),
        field("t_follow_end", pa.float64(), "s", "Hand slowed down after the contact"),
        field("t_recovery_end", pa.float64(), "s", "Back to a ready position"),
        field("prep_s", pa.float32(), "s", "Preparation start to backswing end"),
        field("forward_s", pa.float32(), "s", "Forward swing: backswing end to contact"),
        field("follow_s", pa.float32(), "s", "Contact to follow-through end"),
        field("recovery_s", pa.float32(), "s", "Follow-through end to ready"),
        field("tempo", pa.float32(), None, "Preparation / forward swing duration"),
        field("contact_height_m", pa.float32(), "m", "Racket wrist height at contact"),
        field(
            "contact_front_m",
            pa.float32(),
            "m",
            "Racket wrist ahead (+) of the pelvis, toward the net",
        ),
        field(
            "contact_side_m", pa.float32(), "m", "Racket wrist to the racket side (+) of the pelvis"
        ),
        field("contact_dist_m", pa.float32(), "m", "Horizontal racket wrist to pelvis distance"),
        field(
            "wrist_speed_peak", pa.float32(), "m/s", "Peak racket wrist speed (racket speed proxy)"
        ),
        field(
            "wrist_speed_avg",
            pa.float32(),
            "m/s",
            "Racket wrist distance covered in the last 0.3 s before contact, per second",
        ),
        field("pelvis_av_peak", pa.float32(), "deg/s", "Peak hip rotation speed"),
        field("trunk_av_peak", pa.float32(), "deg/s", "Peak shoulder rotation speed"),
        field("elbow_av_peak", pa.float32(), "deg/s", "Peak elbow extension speed"),
        field("t_pelvis_peak", pa.float32(), "s", "Peak hip rotation speed, relative to contact"),
        field(
            "t_trunk_peak", pa.float32(), "s", "Peak shoulder rotation speed, relative to contact"
        ),
        field("t_elbow_peak", pa.float32(), "s", "Peak elbow extension speed, relative to contact"),
        field("t_wrist_peak", pa.float32(), "s", "Peak wrist speed, relative to contact"),
        field("chain_in_order", pa.bool_(), None, "Hips → trunk → elbow → wrist peaks in order"),
        field("shoulder_turn_max", pa.float32(), "deg", "Largest shoulder turn away from the net"),
        field("hip_turn_max", pa.float32(), "deg", "Largest hip turn away from the net"),
        field("separation_max", pa.float32(), "deg", "Largest hip-shoulder separation"),
        field("shoulder_turn_contact", pa.float32(), "deg", "Shoulder turn at contact"),
        field("hip_turn_contact", pa.float32(), "deg", "Hip turn at contact"),
        field("knee_flex_max", pa.float32(), "deg", "Deepest knee bend in the preparation"),
        field("knee_flex_contact", pa.float32(), "deg", "Knee bend at contact"),
        field(
            "elbow_contact", pa.float32(), "deg", "Racket elbow angle at contact (180 = straight)"
        ),
        field("trunk_lean_contact", pa.float32(), "deg", "Trunk lean from vertical at contact"),
        field("stance_width_m", pa.float32(), "m", "Feet apart at contact"),
        field("com_drop_m", pa.float32(), "m", "Pelvis lowered in the preparation"),
        field("jump_m", pa.float32(), "m", "Pelvis raised at contact vs standing"),
        field("toss_height_m", pa.float32(), "m", "Serves: tossing hand's highest point"),
        field("pose_quality", pa.float32(), None, "0-1: keypoint confidence and coverage"),
        field("n_frames", pa.int32(), None, "Pose frames in the swing window"),
        field("flags", pa.list_(pa.string()), None, "Quality flags"),
    ],
)

# ---------------------------------------------------------------------------
# Statistics records (PLAN.md §7.13)
# ---------------------------------------------------------------------------

#: The records a session's statistics are computed from (``stats``), after the user's edits:
#: one row per shot, per swing, and one movement row. Read across sessions with
#: ``pyarrow.dataset`` (``analysis.aggregate``). Numbers are float64 so a round trip through
#: this file changes nothing.
STATS_RECORDS = table_schema(
    "stats_records",
    1,
    [
        field("session_id", pa.string(), None, nullable=False),
        field("recorded_on", pa.string(), None, "Recording start, local time (ISO)"),
        field("mode", pa.string(), None, "practice | match"),
        field("practice_type", pa.string(), None, "self_feed | ball_machine | serve"),
        field("profile_id", pa.string(), None, "The player's profile ('me')"),
        field("device_key", pa.string(), None, "Recording device (M7c)"),
        field("calibration_by", pa.string(), None, "Court calibration accepted by: user | auto"),
        field("speeds_calibrated", pa.bool_(), None, "Speeds carry a device calibration (M7c)"),
        field("kind", pa.string(), None, "shot | swing | movement", nullable=False),
        field("t", pa.float64(), "s", "Contact time (shots, swings)"),
        field("side", pa.int8(), None, "Hitter's half: -1 near, +1 far"),
        # Shots
        field("shot_id", pa.int32(), None, "shots.parquet row (null: contact not seen)"),
        field("segment_id", pa.int32(), None, "Practice shot segment"),
        field("group", pa.string(), None, "Stroke group: serve | forehand | ... | unknown"),
        field("speed_kmh", pa.float64(), "km/h", "Speed off the racket"),
        field("speed_sigma_kmh", pa.float64(), "km/h"),
        field("speed_ok", pa.bool_(), None, "The speed is certain enough to count"),
        field("landing_x", pa.float64(), "m", "Landing on the court (after edits)"),
        field("landing_y", pa.float64(), "m"),
        field("rel_x", pa.float64(), "m", "Landing in the hitter's frame"),
        field("rel_y", pa.float64(), "m"),
        field("outcome", pa.string(), None, "in | out_long | out_wide | net | unknown"),
        field("net_clearance_m", pa.float64(), "m"),
        field("spin_sign", pa.int8(), None),
        field("contact_seen", pa.bool_(), None),
        field("excluded", pa.bool_(), None, "Marked 'not a practice shot'"),
        field("in_target", pa.bool_(), None, "Landed in an applicable target (null: n/a)"),
        # Swings
        field("swing_id", pa.int32(), None),
        field("stroke_type", pa.string(), None, "The swing's stroke (after edits)"),
        field("wrist_speed_peak", pa.float64(), "m/s"),
        field("forward_s", pa.float64(), "s"),
        field("contact_height_m", pa.float64(), "m"),
        field("chain_in_order", pa.bool_(), None),
        # Movement: what pooled movement figures need (sums and counts, not just ratios)
        field("processed_frames", pa.int64(), None),
        field("lit_frames", pa.int64(), None),
        field("dark_frames", pa.int64(), None),
        field("tracked_lit_frames", pa.int64(), None, "Usable frames with the player tracked"),
        field("tracked_s", pa.float64(), "s"),
        field("distance_m", pa.float64(), "m"),
        field("max_speed_mps", pa.float64(), "m/s", "Best 0.5 s running speed"),
        field("n_samples", pa.int64(), None, "Movement samples"),
        field("n_moving", pa.int64(), None, "Samples at running speed"),
        field("moving_speed_sum", pa.float64(), "m/s", "Sum of the speeds of those samples"),
        field("n_near", pa.int64(), None, "Samples on the near half"),
        field("n_runs", pa.int64(), None, "Tracked runs"),
        field("heatmap_s", pa.list_(pa.float64()), "s", "Time per cell, both ends folded"),
        field("distance_bins_m", pa.list_(pa.float64()), "m", "Distance per bin of video"),
        field("distance_bin_s", pa.float64(), "s"),
    ],
)

SCHEMAS: dict[str, pa.Schema] = {
    "audio_onsets": AUDIO_ONSETS,
    "pass1_frames": PASS1_FRAMES,
    "person_detections": PERSON_DETECTIONS,
    "player_tracks": PLAYER_TRACKS,
    "movement": MOVEMENT,
    "ball_candidates": BALL_CANDIDATES,
    "ball_frames": BALL_FRAMES,
    "ball_track": BALL_TRACK,
    "events": EVENTS,
    "ball_flights": BALL_FLIGHTS,
    "ball_flight_paths": BALL_FLIGHT_PATHS,
    "shots": SHOTS,
    "segments": SEGMENTS,
    "practice": PRACTICE,
    "pose2d": POSE2D,
    "pose3d": POSE3D,
    "swings": SWINGS,
    "stats_records": STATS_RECORDS,
}
