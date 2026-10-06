"""The stage registry. Registration order must be a valid topological order."""

from __future__ import annotations

from swingvision.pipeline.runner import Registry
from swingvision.pipeline.stages.ball import BallRefineStage, BallTrackStage, EventsStage
from swingvision.pipeline.stages.court import CameraStage, CourtAutoStage
from swingvision.pipeline.stages.ingest import AudioOnsetsStage, IngestStage, ProxyStage
from swingvision.pipeline.stages.players import (
    MovementStage,
    Pass1DetectStage,
    PlayersTrackStage,
)
from swingvision.pipeline.stages.shots import Ball3DStage, ShotsStage


def default_registry() -> Registry:
    # court_auto runs right after ingest so the calibration can be reviewed while the
    # proxy encodes; the camera gate comes before the GPU detection pass (which crops to the
    # court) so playback is ready even when it pauses.
    return Registry(
        [
            IngestStage(),
            CourtAutoStage(),
            ProxyStage(),
            AudioOnsetsStage(),
            CameraStage(),
            Pass1DetectStage(),
            PlayersTrackStage(),
            MovementStage(),
            BallRefineStage(),
            BallTrackStage(),
            EventsStage(),
            Ball3DStage(),
            ShotsStage(),
        ]
    )
