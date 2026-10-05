"""The stage registry. Registration order must be a valid topological order."""

from __future__ import annotations

from swingvision.pipeline.runner import Registry
from swingvision.pipeline.stages.ingest import AudioOnsetsStage, IngestStage, ProxyStage


def default_registry() -> Registry:
    return Registry([IngestStage(), ProxyStage(), AudioOnsetsStage()])
