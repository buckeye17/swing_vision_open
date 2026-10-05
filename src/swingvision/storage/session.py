"""One session directory under ``<output_root>/sessions/`` (layout in PLAN.md §5)."""

from __future__ import annotations

from pathlib import Path

from swingvision.storage.fsutil import atomic_write_text
from swingvision.storage.schemas import SessionConfig

SESSIONS_DIR = "sessions"
CONFIG_NAME = "session.json"


class Session:
    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def open(cls, output_root: Path, dir_name: str) -> Session:
        return cls(Path(output_root) / SESSIONS_DIR / dir_name)

    # -- config ------------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_NAME

    def load_config(self) -> SessionConfig:
        return SessionConfig.model_validate_json(self.config_path.read_text(encoding="utf-8"))

    def save_config(self, config: SessionConfig) -> None:
        atomic_write_text(self.config_path, config.model_dump_json(indent=2))

    # -- well-known paths ----------------------------------------------------------
    @property
    def manifests_dir(self) -> Path:
        return self.path / "manifests"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def work_dir(self) -> Path:
        """Scratch space for in-progress chunk parts."""
        return self.path / "work"

    @property
    def proxy_path(self) -> Path:
        return self.path / "proxy_720p.mp4"

    @property
    def thumb_path(self) -> Path:
        return self.path / "thumb.jpg"

    @property
    def audio_path(self) -> Path:
        return self.path / "audio.flac"

    @property
    def audio_onsets_path(self) -> Path:
        return self.path / "audio_onsets.parquet"

    def job_log_path(self, job_id: int) -> Path:
        return self.logs_dir / f"job-{job_id:05d}.log"

    def rel(self, path: Path) -> str:
        return Path(path).relative_to(self.path).as_posix()
