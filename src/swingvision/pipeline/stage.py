"""Stage framework: versioned, config-hashed, resumable pipeline steps (PLAN.md §6).

A stage is *fresh* when its manifest's fingerprint equals the fingerprint
computed now from (stage version, stage config, upstream fingerprints) and its
outputs exist. Changing a stage's version or config, or anything upstream,
makes it and everything downstream stale.

Heavy stages work in time chunks. Each finished chunk is checkpointed, so a
crashed or cancelled job resumes at the first unfinished chunk.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from swingvision.settings import AppSettings
from swingvision.storage.fsutil import atomic_write_json, read_json
from swingvision.storage.library import now_iso
from swingvision.storage.schemas import SessionConfig
from swingvision.storage.session import Session

ALL_MODES = frozenset({"practice", "match"})


class Cancelled(Exception):
    """Raised inside a stage when the job was cancelled."""


class NeedsUserAction(Exception):
    """Raised by a stage that cannot continue without the user (e.g. calibration review)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()[:16]


@dataclass
class StageContext:
    session: Session
    config: SessionConfig
    settings: AppSettings
    fingerprint: str
    stage_name: str
    report: Callable[[float, str | None], None] = lambda frac, msg: None
    is_cancelled: Callable[[], bool] = lambda: False
    log: Any = field(default=logger)

    def progress(self, fraction: float, message: str | None = None) -> None:
        self.report(min(1.0, max(0.0, fraction)), message)

    def check_cancel(self) -> None:
        if self.is_cancelled():
            raise Cancelled()

    def chunks(self, n_chunks: int) -> ChunkTracker:
        return ChunkTracker(self.session, self.stage_name, self.fingerprint, n_chunks)


class Stage(ABC):
    name: ClassVar[str]
    version: ClassVar[int]
    depends_on: ClassVar[tuple[str, ...]] = ()
    #: Stages that must run first but whose fingerprints don't feed this one's. The stage's
    #: ``config()`` captures what it actually uses from them (e.g. ``pass1_detect`` keeps only
    #: a coarse crop rectangle from the calibration, so recalibrating doesn't rerun the GPU).
    after: ClassVar[tuple[str, ...]] = ()
    modes: ClassVar[frozenset[str]] = ALL_MODES
    phase: ClassVar[int] = 1
    uses_gpu: ClassVar[bool] = False
    weight: ClassVar[float] = 1.0  # relative cost, for overall job progress
    title: ClassVar[str] = ""

    def config(self, session: Session, config: SessionConfig, settings: AppSettings) -> dict:
        """Settings that affect this stage's output. Changing them invalidates the stage."""
        return {}

    def outputs(self, session: Session) -> list[Path]:
        return []

    @abstractmethod
    def run(self, ctx: StageContext) -> dict[str, Any] | None:
        """Do the work; return extra info to store in the manifest."""


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def manifest_path(session: Session, stage_name: str) -> Path:
    return session.manifests_dir / f"{stage_name}.json"


def read_manifest(session: Session, stage_name: str) -> dict[str, Any] | None:
    path = manifest_path(session, stage_name)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def write_manifest(session: Session, stage_name: str, manifest: dict[str, Any]) -> None:
    atomic_write_json(manifest_path(session, stage_name), manifest)


def clear_manifest(session: Session, stage_name: str) -> None:
    manifest_path(session, stage_name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Chunk checkpoints
# ---------------------------------------------------------------------------


def chunk_ranges(duration_s: float, chunk_s: float) -> list[tuple[float, float]]:
    if duration_s <= 0:
        return []
    n = max(1, int(-(-duration_s // chunk_s)))
    return [(i * chunk_s, min(duration_s, (i + 1) * chunk_s)) for i in range(n)]


class ChunkTracker:
    """Records finished chunks in ``manifests/<stage>.chunks.json``; parts live in ``work/<stage>/``."""

    def __init__(self, session: Session, stage_name: str, fingerprint: str, n_chunks: int):
        self.session = session
        self.stage_name = stage_name
        self.fingerprint = fingerprint
        self.n_chunks = n_chunks
        self.path = session.manifests_dir / f"{stage_name}.chunks.json"
        self.parts_dir = session.work_dir / stage_name
        state = read_json(self.path) if self.path.exists() else None
        if (
            not state
            or state.get("fingerprint") != fingerprint
            or state.get("n_chunks") != n_chunks
        ):
            # Different config/version/input: earlier parts are not reusable.
            shutil.rmtree(self.parts_dir, ignore_errors=True)
            state = {"fingerprint": fingerprint, "n_chunks": n_chunks, "done": []}
        self._done: set[int] = set(state["done"])
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self._save()

    @property
    def done(self) -> set[int]:
        return set(self._done)

    def is_done(self, index: int) -> bool:
        return index in self._done

    def mark_done(self, index: int) -> None:
        self._done.add(index)
        self._save()

    def all_done(self) -> bool:
        return len(self._done) == self.n_chunks

    def cleanup(self) -> None:
        self.path.unlink(missing_ok=True)
        shutil.rmtree(self.parts_dir, ignore_errors=True)

    def _save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "fingerprint": self.fingerprint,
                "n_chunks": self.n_chunks,
                "done": sorted(self._done),
                "updated_at": now_iso(),
            },
        )


def build_manifest(
    stage: Stage,
    session: Session,
    config_hash: str,
    inputs: dict[str, str],
    fingerprint: str,
    started: float,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "stage": stage.name,
        "version": stage.version,
        "config_hash": config_hash,
        "inputs": inputs,
        "fingerprint": fingerprint,
        "status": "done",
        "finished_at": now_iso(),
        "duration_s": round(time.time() - started, 3),
        "outputs": [session.rel(p) for p in stage.outputs(session)],
        "extra": extra or {},
    }
