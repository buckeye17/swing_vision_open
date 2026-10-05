"""The GPU worker process: pulls jobs from the library queue and runs the pipeline.

Exactly one worker per output root (enforced with a lock file). Started by the
app, or by hand with ``sv worker``. It exits when idle if its parent app has
exited or the configured output root changed.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from pathlib import Path

import psutil
from filelock import FileLock, Timeout
from loguru import logger

from swingvision.pipeline.runner import PlannedStage, Registry, RunHooks, run
from swingvision.pipeline.stages import default_registry
from swingvision.settings import AppSettings, load_settings
from swingvision.storage.library import (
    CANCELLED,
    DONE,
    FAILED,
    NEEDS_ACTION,
    STAGE_DONE,
    STAGE_PENDING,
    STAGE_RUNNING,
    STAGE_SKIPPED,
    Job,
    Library,
    now_iso,
)
from swingvision.storage.session import Session

LOCK_NAME = ".worker.lock"
HEARTBEAT_S = 2.0
STALE_AFTER_S = 30.0
PROGRESS_MIN_INTERVAL_S = 0.5


def worker_lock(output_root: Path) -> FileLock:
    return FileLock(str(Path(output_root) / LOCK_NAME))


class JobReporter:
    """Translates runner hooks into (throttled) library updates."""

    def __init__(self, library: Library, job: Job):
        self.library = library
        self.job = job
        self.weights: dict[str, float] = {}
        self.fractions: dict[str, float] = {}
        self._last_write = 0.0
        self._cancel = False
        self._cancel_checked = 0.0

    def is_cancelled(self) -> bool:
        now = time.monotonic()
        if now - self._cancel_checked > 0.5:
            self._cancel = self.library.is_cancel_requested(self.job.id)
            self._cancel_checked = now
        return self._cancel

    def overall(self) -> float:
        total = sum(self.weights.values()) or 1.0
        return sum(self.weights[n] * self.fractions.get(n, 0.0) for n in self.weights) / total

    def on_plan(self, planned: list[PlannedStage]) -> None:
        self.weights = {p.stage.name: p.stage.weight for p in planned}
        self.fractions = {p.stage.name: 1.0 if p.fresh else 0.0 for p in planned}
        self.library.set_job_stages(
            self.job.id,
            [
                (p.stage.name, p.stage.weight, STAGE_SKIPPED if p.fresh else STAGE_PENDING)
                for p in planned
            ],
        )
        self.library.update_job(self.job.id, progress=self.overall())

    def on_stage_start(self, name: str) -> None:
        self.library.update_job_stage(
            self.job.id,
            name,
            status=STAGE_RUNNING,
            started_at=now_iso(),
            progress=0.0,
            message=None,
        )
        self.library.update_job(self.job.id, current_stage=name, message=None)

    def on_stage_progress(self, name: str, fraction: float, message: str | None) -> None:
        self.fractions[name] = fraction
        now = time.monotonic()
        if now - self._last_write < PROGRESS_MIN_INTERVAL_S and fraction < 1.0:
            return
        self._last_write = now
        self.library.update_job_stage(self.job.id, name, progress=fraction, message=message)
        self.library.update_job(
            self.job.id, progress=self.overall(), message=message, heartbeat_at=now_iso()
        )

    def on_stage_end(self, name: str, status: str, message: str | None) -> None:
        if status in ("done", "skipped"):
            self.fractions[name] = 1.0
        fields = {"status": STAGE_DONE if status == "done" else status, "message": message}
        if status != "skipped":
            fields["finished_at"] = now_iso()
        if status == "done":
            fields["progress"] = 1.0
        self.library.update_job_stage(self.job.id, name, **fields)
        self.library.update_job(self.job.id, progress=self.overall())


def process_job(library: Library, settings: AppSettings, job: Job, registry: Registry) -> str:
    row = library.get_session(job.session_id)
    if row is None:
        library.finish_job(job.id, FAILED, "Session no longer exists")
        return FAILED
    session = Session.open(library.root, row["dir_name"])
    session.logs_dir.mkdir(parents=True, exist_ok=True)
    sink = logger.add(
        session.job_log_path(job.id),
        level="INFO",
        enqueue=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {message}",
    )
    reporter = JobReporter(library, job)
    try:
        logger.info("Job #{} for session {} ({})", job.id, row["name"], row["dir_name"])
        result = run(
            registry,
            session,
            settings,
            targets=job.targets,
            force=job.force,
            hooks=RunHooks(
                on_plan=reporter.on_plan,
                on_stage_start=reporter.on_stage_start,
                on_stage_progress=reporter.on_stage_progress,
                on_stage_end=reporter.on_stage_end,
                is_cancelled=reporter.is_cancelled,
            ),
        )
    except Exception as exc:  # planning errors etc.
        logger.exception("Job #{} crashed", job.id)
        library.finish_job(job.id, FAILED, str(exc), error=repr(exc))
        return FAILED
    finally:
        logger.remove(sink)
    status = {"done": DONE, "cancelled": CANCELLED, "needs_action": NEEDS_ACTION, "failed": FAILED}[
        result.status
    ]
    library.finish_job(job.id, status, result.message, result.error)
    logger.info("Job #{} finished: {} ({})", job.id, status, result.message)
    return status


def _output_root_changed(root: Path) -> bool:
    current = load_settings().output_root
    return current is None or Path(current).resolve() != Path(root).resolve()


class _Heartbeat(threading.Thread):
    def __init__(self, library: Library, worker_id: str, started_at: str):
        super().__init__(daemon=True)
        self.library = library
        self.worker_id = worker_id
        self.started_at = started_at
        self.state = "idle"
        self.job_id: int | None = None
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.library.worker_heartbeat(
                    self.worker_id, os.getpid(), self.state, self.started_at
                )
                if self.job_id is not None:
                    self.library.update_job(self.job_id, heartbeat_at=now_iso())
            except Exception as exc:  # never let the heartbeat kill the worker
                logger.warning("Heartbeat failed: {}", exc)
            self.stop_event.wait(HEARTBEAT_S)


def run_worker(
    settings: AppSettings | None = None,
    parent_pid: int | None = None,
    poll_s: float = 1.0,
    once: bool = False,
    registry: Registry | None = None,
) -> int:
    """Main loop. Returns a process exit code."""
    follow_settings_file = settings is None
    settings = settings or load_settings()
    root = settings.require_output_root()
    library = Library(root).init()
    lock = worker_lock(root)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        logger.info("Another worker already serves {}", root)
        return 0

    registry = registry or default_registry()
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    heartbeat = _Heartbeat(library, worker_id, now_iso())
    heartbeat.start()
    requeued = library.requeue_stale_jobs(STALE_AFTER_S)
    if requeued:
        logger.info("Re-queued interrupted jobs {}", requeued)
    logger.info("Worker {} serving {}", worker_id, root)
    try:
        while True:
            job = library.claim_next_job(worker_id)
            if job is None:
                if once:
                    return 0
                if parent_pid is not None and not psutil.pid_exists(parent_pid):
                    logger.info("Parent app exited; worker stopping")
                    return 0
                if follow_settings_file and _output_root_changed(root):
                    logger.info("Output root changed; worker stopping")
                    return 0
                time.sleep(poll_s)
                continue
            heartbeat.state, heartbeat.job_id = "busy", job.id
            try:
                process_job(library, settings, job, registry)
            finally:
                heartbeat.state, heartbeat.job_id = "idle", None
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
        return 130
    finally:
        heartbeat.stop_event.set()
        with contextlib.suppress(Exception):
            library.worker_heartbeat(worker_id, os.getpid(), "stopped", heartbeat.started_at)
        lock.release()
