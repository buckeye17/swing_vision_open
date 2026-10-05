"""Application settings, stored as JSON in the user's config directory.

Settings live outside the output folder because they say *where* the output
folder is. Set ``SWINGVISION_CONFIG_DIR`` to override the location (tests do).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field

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


class AppSettings(BaseModel):
    output_root: Path | None = None
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    host: str = "127.0.0.1"
    port: int = 8050
    processing: ProcessingDefaults = Field(default_factory=ProcessingDefaults)

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
