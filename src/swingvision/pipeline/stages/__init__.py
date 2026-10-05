"""The stage registry. Registration order must be a valid topological order."""

from __future__ import annotations

from swingvision.pipeline.runner import Registry
from swingvision.pipeline.stages.court import CameraStage, CourtAutoStage
from swingvision.pipeline.stages.ingest import AudioOnsetsStage, IngestStage, ProxyStage


def default_registry() -> Registry:
    # court_auto runs right after ingest so the calibration can be reviewed while the
    # proxy encodes; the camera gate comes last so playback is ready even when it pauses.
    return Registry(
        [
            IngestStage(),
            CourtAutoStage(),
            ProxyStage(),
            AudioOnsetsStage(),
            CameraStage(),
        ]
    )
