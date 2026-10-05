"""SQLite library in the output root: sessions index, job queue, profiles, worker heartbeat.

The app (many Flask threads) and the worker process both use this file, so every
operation opens a short-lived connection in WAL mode.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DB_NAME = "library.sqlite"

# Job statuses
QUEUED, RUNNING, DONE, FAILED, CANCELLED, NEEDS_ACTION = (
    "queued",
    "running",
    "done",
    "failed",
    "cancelled",
    "needs_action",
)
ACTIVE_JOB_STATUSES = (QUEUED, RUNNING)

# Stage statuses within a job
STAGE_PENDING, STAGE_RUNNING, STAGE_DONE, STAGE_SKIPPED = "pending", "running", "done", "skipped"

MIGRATIONS: list[str] = [
    # v1
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        dir_name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        mode TEXT NOT NULL,
        submode TEXT,
        source_path TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        duration_s REAL,
        status TEXT NOT NULL DEFAULT 'new'
    );
    CREATE TABLE jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        targets TEXT,
        force TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        heartbeat_at TEXT,
        worker_id TEXT,
        progress REAL NOT NULL DEFAULT 0,
        current_stage TEXT,
        message TEXT,
        error TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX jobs_status_idx ON jobs(status, id);
    CREATE TABLE job_stages (
        job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        weight REAL NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'pending',
        progress REAL NOT NULL DEFAULT 0,
        message TEXT,
        started_at TEXT,
        finished_at TEXT,
        PRIMARY KEY (job_id, stage)
    );
    CREATE TABLE profiles (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        handedness TEXT NOT NULL DEFAULT 'right',
        backhand TEXT NOT NULL DEFAULT 'two_handed',
        height_m REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        data TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE worker (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        worker_id TEXT,
        pid INTEGER,
        started_at TEXT,
        heartbeat_at TEXT,
        state TEXT
    );
    """,
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def parse_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass
class Job:
    id: int
    session_id: str
    status: str
    targets: list[str] | None
    force: list[str]
    created_at: str
    started_at: str | None
    finished_at: str | None
    heartbeat_at: str | None
    worker_id: str | None
    progress: float
    current_stage: str | None
    message: str | None
    error: str | None
    cancel_requested: bool

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        d = dict(row)
        d["targets"] = json.loads(d["targets"]) if d["targets"] else None
        d["force"] = json.loads(d["force"] or "[]")
        d["cancel_requested"] = bool(d["cancel_requested"])
        return cls(**d)


class Library:
    def __init__(self, output_root: Path):
        self.root = Path(output_root)
        self.path = self.root / DB_NAME

    # -- connection / schema -------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def init(self) -> Library:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
                conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {i};\nCOMMIT;")
        return self

    def schema_version(self) -> int:
        with self.connect() as conn:
            return conn.execute("PRAGMA user_version").fetchone()[0]

    # -- sessions --------------------------------------------------------------
    def add_session(
        self,
        *,
        id: str,
        name: str,
        dir_name: str,
        created_at: str,
        mode: str,
        submode: str | None,
        source_path: str,
        source_hash: str,
        duration_s: float | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO sessions (id, name, dir_name, created_at, updated_at, mode, submode,
                                         source_path, source_hash, duration_s)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    id,
                    name,
                    dir_name,
                    created_at,
                    now_iso(),
                    mode,
                    submode,
                    source_path,
                    source_hash,
                    duration_s,
                ),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def update_session(self, session_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.connect() as conn:
            conn.execute(f"UPDATE sessions SET {cols} WHERE id = ?", (*fields.values(), session_id))

    def delete_session(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # -- jobs ------------------------------------------------------------------
    def enqueue_job(
        self, session_id: str, targets: list[str] | None = None, force: list[str] | None = None
    ) -> int:
        with self.transaction() as conn:
            active = conn.execute(
                "SELECT id FROM jobs WHERE session_id = ? AND status IN (?, ?)",
                (session_id, *ACTIVE_JOB_STATUSES),
            ).fetchone()
            if active:
                raise ValueError(f"Session already has an active job (#{active['id']}).")
            cur = conn.execute(
                """INSERT INTO jobs (session_id, status, targets, force, created_at, message)
                   VALUES (?, ?, ?, ?, ?, 'Waiting for worker')""",
                (
                    session_id,
                    QUEUED,
                    json.dumps(targets) if targets else None,
                    json.dumps(force or []),
                    now_iso(),
                ),
            )
            conn.execute(
                "UPDATE sessions SET status = 'queued', updated_at = ? WHERE id = ?",
                (now_iso(), session_id),
            )
            return int(cur.lastrowid)

    def claim_next_job(self, worker_id: str) -> Job | None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY id LIMIT 1", (QUEUED,)
            ).fetchone()
            if row is None:
                return None
            ts = now_iso()
            conn.execute(
                """UPDATE jobs SET status = ?, worker_id = ?, started_at = COALESCE(started_at, ?),
                          heartbeat_at = ?, message = 'Starting' WHERE id = ?""",
                (RUNNING, worker_id, ts, ts, row["id"]),
            )
            job = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            conn.execute(
                "UPDATE sessions SET status = 'processing', updated_at = ? WHERE id = ?",
                (ts, job["session_id"]),
            )
        return Job.from_row(job)

    def get_job(self, job_id: int) -> Job | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[Job]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [Job.from_row(r) for r in rows]

    def latest_job_for_session(self, session_id: str) -> Job | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE session_id = ? ORDER BY id DESC LIMIT 1", (session_id,)
            ).fetchone()
        return Job.from_row(row) if row else None

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def finish_job(
        self, job_id: int, status: str, message: str | None = None, error: str | None = None
    ) -> None:
        session_status = {
            DONE: "ready",
            FAILED: "failed",
            CANCELLED: "cancelled",
            NEEDS_ACTION: "needs_action",
        }[status]
        ts = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """UPDATE jobs SET status = ?, finished_at = ?, message = ?, error = ?,
                          progress = CASE WHEN ? = 'done' THEN 1.0 ELSE progress END
                   WHERE id = ?""",
                (status, ts, message, error, status, job_id),
            )
            conn.execute(
                """UPDATE sessions SET status = ?, updated_at = ?
                   WHERE id = (SELECT session_id FROM jobs WHERE id = ?)""",
                (session_status, ts, job_id),
            )

    def request_cancel(self, job_id: int) -> None:
        """Queued jobs are cancelled immediately; running jobs at the worker's next check."""
        with self.transaction() as conn:
            job = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                return
            if job["status"] == QUEUED:
                ts = now_iso()
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, message = 'Cancelled' WHERE id = ?",
                    (CANCELLED, ts, job_id),
                )
                conn.execute(
                    """UPDATE sessions SET status = 'cancelled', updated_at = ?
                       WHERE id = (SELECT session_id FROM jobs WHERE id = ?)""",
                    (ts, job_id),
                )
            elif job["status"] == RUNNING:
                conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))

    def is_cancel_requested(self, job_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return bool(row and row[0])

    def requeue_stale_jobs(self, stale_after_s: float = 60.0) -> list[int]:
        """Put RUNNING jobs whose worker stopped heartbeating back in the queue (they resume)."""
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_s)
        requeued = []
        with self.transaction() as conn:
            rows = conn.execute("SELECT id, heartbeat_at FROM jobs WHERE status = ?", (RUNNING,))
            for row in rows.fetchall():
                hb = parse_iso(row["heartbeat_at"])
                if hb is None or hb < cutoff:
                    conn.execute(
                        """UPDATE jobs SET status = ?, worker_id = NULL,
                                  message = 'Re-queued after worker stopped' WHERE id = ?""",
                        (QUEUED, row["id"]),
                    )
                    requeued.append(row["id"])
        return requeued

    # -- job stages --------------------------------------------------------------
    def set_job_stages(self, job_id: int, stages: list[tuple[str, float, str]]) -> None:
        """``stages`` = [(name, weight, initial_status)] in execution order."""
        with self.transaction() as conn:
            conn.execute("DELETE FROM job_stages WHERE job_id = ?", (job_id,))
            conn.executemany(
                """INSERT INTO job_stages (job_id, stage, ordinal, weight, status, progress)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (job_id, name, i, weight, status, 1.0 if status == STAGE_SKIPPED else 0.0)
                    for i, (name, weight, status) in enumerate(stages)
                ],
            )

    def update_job_stage(self, job_id: int, stage: str, **fields: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE job_stages SET {cols} WHERE job_id = ? AND stage = ?",
                (*fields.values(), job_id, stage),
            )

    def job_stages(self, job_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM job_stages WHERE job_id = ? ORDER BY ordinal", (job_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- profiles ----------------------------------------------------------------
    PROFILE_FIELDS = ("name", "handedness", "backhand", "height_m")

    def add_profile(
        self,
        *,
        id: str,
        name: str,
        handedness: str = "right",
        backhand: str = "two_handed",
        height_m: float | None = None,
    ) -> None:
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO profiles (id, name, handedness, backhand, height_m, created_at,
                                         updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (id, name, handedness, backhand, height_m, ts, ts),
            )

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return dict(row) if row else None

    def list_profiles(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM profiles ORDER BY name COLLATE NOCASE").fetchall()
        return [dict(r) for r in rows]

    def update_profile(self, profile_id: str, **fields: Any) -> None:
        unknown = set(fields) - set(self.PROFILE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown profile fields {sorted(unknown)}")
        if not fields:
            return
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.connect() as conn:
            conn.execute(f"UPDATE profiles SET {cols} WHERE id = ?", (*fields.values(), profile_id))

    def delete_profile(self, profile_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))

    # -- worker heartbeat ----------------------------------------------------------
    def worker_heartbeat(self, worker_id: str, pid: int, state: str, started_at: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO worker (id, worker_id, pid, started_at, heartbeat_at, state)
                   VALUES (1, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET worker_id = excluded.worker_id,
                       pid = excluded.pid, started_at = excluded.started_at,
                       heartbeat_at = excluded.heartbeat_at, state = excluded.state""",
                (worker_id, pid, started_at, now_iso(), state),
            )

    def worker_status(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM worker WHERE id = 1").fetchone()
        return dict(row) if row else None
