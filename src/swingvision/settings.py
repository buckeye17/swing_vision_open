"""Application settings, stored as JSON in the user's config directory.

Settings live outside the output folder because they say *where* the output
folder is. Set ``SWINGVISION_CONFIG_DIR`` to override the location (tests do).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field, field_validator

from swingvision.storage.fsutil import atomic_write_text

APP_NAME = "swingvision"


def config_dir() -> Path:
    override = os.environ.get("SWINGVISION_CONFIG_DIR")
    return Path(override) if override else Path(user_config_dir(APP_NAME, appauthor=False))


def data_dir() -> Path:
    """App data (base model weights, caches). Not the user-selected output folder."""
    override = os.environ.get("SWINGVISION_DATA_DIR")
    return Path(override) if override else Path(user_data_dir(APP_NAME, appauthor=False))


class ProcessingDefaults(BaseModel):
    proxy_height: int = 720
    proxy_gop: int = 30
    chunk_seconds: float = 120.0
    segment_pad_before_s: float = 2.0
    segment_pad_after_s: float = 2.0
    #: Continue without review when the auto calibration's line RMS is below this (px).
    #: ``None`` always pauses the job for review (status ``needs_action``).
    calibration_auto_accept_px: float | None = None
    calibration_window_s: float = 300.0  # drift check: one re-calibration per window
    calibration_frames_per_window: int = 5
    calibration_drift_px: float = 3.0  # RMS keypoint shift that flags a moved camera
    #: Person detection (``pass1_detect``). The court-ROI crop is scaled so its long side is
    #: ``person_input_px``; 1920 (half of 4K) finds the far player far more reliably than 1280.
    person_model: str = "yolo11m"
    person_rate_hz: float = 15.0
    person_input_px: int = 1920
    person_conf: float = 0.1
    decode_backend: str = "auto"  # auto | nvdec | pyav
    #: Court region of interest: the court plus this much run-off (m) behind the baselines and
    #: beside the doubles sidelines. Neighboring courts usually start ≈3.7 m beside.
    roi_behind_m: float = 6.0
    roi_beside_m: float = 3.5
    #: Frames darker than this mean luma (0-255) count as unusable for player tracking. The
    #: detector still finds a player at dusk down to about 4 (measured on the Oct 1 session).
    dark_luma: float = 4.0
    #: Frames whose correlation with the court background is below this don't show the
    #: calibrated view (camera being set up, picked up, or knocked): detections there are
    #: ignored. Settled play scores 0.8-0.98 (also at dusk); a camera still being aimed ~0.5.
    view_min: float = 0.6


class AppSettings(BaseModel):
    output_root: Path | None = None
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    host: str = "127.0.0.1"
    port: int = 8050
    processing: ProcessingDefaults = Field(default_factory=ProcessingDefaults)

    @field_validator("output_root", mode="before")
    @classmethod
    def _blank_is_unset(cls, value):
        # Path("") would silently mean the current directory.
        return None if isinstance(value, str) and not value.strip() else value

    def ffmpeg(self) -> str:
        return _resolve_tool(self.ffmpeg_path)

    def ffprobe(self) -> str:
        return _resolve_tool(self.ffprobe_path)

    def require_output_root(self) -> Path:
        if self.output_root is None:
            raise RuntimeError("No output folder is configured. Set one on the Settings page.")
        return self.output_root


def _resolve_tool(name_or_path: str) -> str:
    found = shutil.which(name_or_path)
    if found:
        return found
    if Path(name_or_path).is_file():
        return name_or_path
    raise FileNotFoundError(f"Could not find executable {name_or_path!r}. Check Settings.")


def settings_path() -> Path:
    return config_dir() / "settings.json"


def load_settings() -> AppSettings:
    path = settings_path()
    if not path.exists():
        return AppSettings()
    return AppSettings.model_validate_json(path.read_text(encoding="utf-8"))


def save_settings(settings: AppSettings) -> None:
    atomic_write_text(settings_path(), settings.model_dump_json(indent=2))
