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


class PracticeConfig(BaseModel):
    submode: PracticeSubmode = "self_feed"
    targets: list[Target] = Field(default_factory=list)


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

SCHEMAS: dict[str, pa.Schema] = {"audio_onsets": AUDIO_ONSETS}
