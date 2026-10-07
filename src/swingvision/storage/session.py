"""One session directory under ``<output_root>/sessions/`` (layout in PLAN.md §5)."""

from __future__ import annotations

from pathlib import Path

from swingvision.storage import cache
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
        text = cache.local(self.config_path).read_text(encoding="utf-8")
        return SessionConfig.model_validate_json(text)

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

    @property
    def court_dir(self) -> Path:
        return self.path / "court"

    @property
    def court_background_path(self) -> Path:
        """Median of sampled frames (players removed), full resolution."""
        return self.court_dir / "background.jpg"

    def court_window_path(self, index: int) -> Path:
        """Median image of one drift-check window."""
        return self.court_dir / "windows" / f"w{index:02d}.jpg"

    @property
    def court_auto_path(self) -> Path:
        return self.court_dir / "auto.json"

    @property
    def court_user_path(self) -> Path:
        """Calibration confirmed in the editor. Never written by the pipeline."""
        return self.court_dir / "user.json"

    @property
    def calibration_path(self) -> Path:
        """The calibration downstream stages use (written by the ``camera`` stage)."""
        return self.path / "calibration.json"

    @property
    def pass1_frames_path(self) -> Path:
        """Frames the detection pass processed (time, brightness)."""
        return self.path / "pass1" / "frames.parquet"

    @property
    def players_dir(self) -> Path:
        return self.path / "players"

    @property
    def person_detections_path(self) -> Path:
        """Raw person boxes in image coordinates (``pass1_detect``)."""
        return self.players_dir / "detections.parquet"

    @property
    def player_tracks_path(self) -> Path:
        return self.players_dir / "tracks.parquet"

    @property
    def players_summary_path(self) -> Path:
        return self.players_dir / "identity.json"

    @property
    def movement_path(self) -> Path:
        return self.players_dir / "movement.parquet"

    @property
    def ball_dir(self) -> Path:
        return self.path / "ball"

    @property
    def ball_sweep_path(self) -> Path:
        """Ball candidates from the detection pass (``pass1_detect``), image coordinates."""
        return self.ball_dir / "sweep.parquet"

    @property
    def ball_sweep_frames_path(self) -> Path:
        """Frames the ball detector looked at in the detection pass."""
        return self.ball_dir / "sweep_frames.parquet"

    @property
    def ball_refine_path(self) -> Path:
        """Ball candidates from the full-rate windows (``ball_refine``)."""
        return self.ball_dir / "refine.parquet"

    @property
    def ball_refine_frames_path(self) -> Path:
        return self.ball_dir / "refine_frames.parquet"

    @property
    def ball_track_path(self) -> Path:
        return self.ball_dir / "track.parquet"

    @property
    def ball_kinks_path(self) -> Path:
        """Every motion break the event detector considered, with its features."""
        return self.ball_dir / "kinks.parquet"

    @property
    def events_path(self) -> Path:
        return self.path / "events.parquet"

    @property
    def ball_flights_path(self) -> Path:
        """Fitted 3D flights between events (``ball_3d``)."""
        return self.ball_dir / "flights.parquet"

    @property
    def ball_flights_summary_path(self) -> Path:
        """Hits the 3D fits rejected (not at the hitter)."""
        return self.ball_dir / "flights.json"

    @property
    def ball_flight_paths_path(self) -> Path:
        return self.ball_dir / "flight_paths.parquet"

    @property
    def shots_path(self) -> Path:
        return self.path / "shots.parquet"

    @property
    def segments_path(self) -> Path:
        """Practice shots and blocks (match points from Phase 2)."""
        return self.path / "segments.parquet"

    @property
    def practice_path(self) -> Path:
        """Per-shot line calls and target accuracy (``practice_eval``)."""
        return self.path / "practice.parquet"

    @property
    def stats_path(self) -> Path:
        """Session aggregates (``stats``)."""
        return self.path / "stats.json"

    @property
    def pose_dir(self) -> Path:
        return self.path / "pose"

    @property
    def pose2d_path(self) -> Path:
        """2D keypoints in the swing windows (``pass2_pose``), image coordinates."""
        return self.pose_dir / "pose2d.parquet"

    @property
    def pose3d_path(self) -> Path:
        """3D joints on the court (``pose3d``)."""
        return self.pose_dir / "pose3d.parquet"

    @property
    def swings_path(self) -> Path:
        """One row per swing: phases, kinematics, stroke (``swings``)."""
        return self.pose_dir / "swings.parquet"

    @property
    def swings_summary_path(self) -> Path:
        """Racket hand, A/V offset and other session-level swing facts."""
        return self.pose_dir / "swings.json"

    @property
    def edits_path(self) -> Path:
        """User overrides (PLAN.md §9.3). Never written by the pipeline."""
        return self.path / "edits.json"

    def job_log_path(self, job_id: int) -> Path:
        return self.logs_dir / f"job-{job_id:05d}.log"

    def rel(self, path: Path) -> str:
        return Path(path).relative_to(self.path).as_posix()
