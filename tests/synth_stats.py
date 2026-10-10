"""Synthetic sessions for multi-session statistics tests: a library row, ``session.json`` and
the tables the ``stats`` stage reads (shots, practice results, swings, movement, frames, serve
contacts), then the stage itself, which writes ``stats.json`` and ``stats_records.parquet``.

Serve contacts (M7b) carry a planted effect: speed rises 8 km/h per 10 cm in front of the
toe, and serves struck more than 30 cm to the racket side go in less often."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa

from swingvision.pipeline.stage import StageContext
from swingvision.pipeline.stages.stats import StatsStage
from swingvision.settings import AppSettings
from swingvision.storage import tables
from swingvision.storage.fsutil import atomic_write_text
from swingvision.storage.library import Library, now_iso, recording_time
from swingvision.storage.schemas import (
    MOVEMENT,
    PASS1_FRAMES,
    PRACTICE,
    SERVE_CONTACT,
    SHOTS,
    SWINGS,
    Calibration,
    CameraParams,
    PlayersConfig,
    PracticeConfig,
    SessionConfig,
    SourceInfo,
    VideoInfo,
)
from swingvision.storage.session import Session

STROKES = ("serve", "forehand", "backhand", "forehand_volley", "other")
OUTCOMES = ("in", "in", "in", "out_long", "out_wide", "net")


@dataclass
class Spec:
    id: str
    recorded: str  # "YYYY-MM-DDTHH:MM:SSZ" (the video's creation_time)
    mode: str = "practice"
    submode: str = "serve"
    profile: str | None = None
    tags: list[str] = field(default_factory=list)
    calibration_by: str | None = "auto"
    n_shots: int = 40
    n_other_swings: int = 20
    minutes: float = 10.0
    seed: int = 0
    serve_contacts: bool = True


def _table(schema: pa.Schema, rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist([{f.name: r.get(f.name) for f in schema} for r in rows], schema)


def _f32(v):
    return None if v is None else float(np.float32(v))


def make_session(lib: Library, spec: Spec, write_records: bool = True) -> Session:
    rng = np.random.default_rng(spec.seed)
    dir_name = f"{spec.recorded[:10]}_{spec.id}"
    session = Session.open(lib.root, dir_name)
    session.path.mkdir(parents=True)
    duration = spec.minutes * 60.0
    config = SessionConfig(
        id=spec.id,
        name=f"Session {spec.id}",
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
        dir_name=dir_name,
        source=SourceInfo(path=f"C:/v/{spec.id}.mp4", size_bytes=1, fast_hash=spec.id, mtime=0.0),
        video=VideoInfo(
            codec="h264", width=3840, height=2160, display_width=3840, display_height=2160,
            fps_nominal=60.0, fps_avg=60.0, is_vfr=False, duration_s=duration,
            n_frames_est=int(duration * 60), creation_time=spec.recorded,
        ),
        mode=spec.mode,
        practice=PracticeConfig(submode=spec.submode) if spec.mode == "practice" else None,
        players=PlayersConfig(me_profile_id=spec.profile),
    )  # fmt: skip
    session.save_config(config)
    lib.add_session(
        id=spec.id, name=config.name, dir_name=dir_name, created_at=now_iso(), mode=spec.mode,
        submode=spec.submode if spec.mode == "practice" else None,
        source_path=config.source.path, source_hash=spec.id, duration_s=duration,
        recorded_on=recording_time(spec.recorded), profile_id=spec.profile,
    )  # fmt: skip
    if spec.tags:
        lib.set_session_tags(spec.id, spec.tags)
    if spec.calibration_by is not None:
        cal = Calibration(
            source="auto",
            created_at=datetime(2026, 10, 9, tzinfo=UTC),
            camera=CameraParams(width=3840, height=2160, f=3000.0, k1=0.0, cx=1920.0, cy=1080.0,
                                rvec=[0.0, 0.0, 0.0], tvec=[0.0, 0.0, 20.0]),
            confirmed_by=spec.calibration_by,
        )  # fmt: skip
        atomic_write_text(session.calibration_path, cal.model_dump_json())

    shots, practice, swings, contacts = [], [], [], []
    times = np.sort(rng.uniform(5, duration - 5, spec.n_shots))
    for i, t in enumerate(times):
        stroke = str(rng.choice(STROKES))
        side = int(rng.choice([-1, 1]))
        outcome = str(rng.choice(OUTCOMES))
        fwd, lat = float(rng.normal(0.15, 0.15)), float(rng.normal(0.2, 0.15))
        if stroke == "serve" and lat > 0.3 and rng.random() < 0.5:
            outcome = str(rng.choice(["out_long", "out_wide", "net"]))
        ly = float(rng.uniform(2.0, 12.5)) * -side if outcome != "net" else None
        lx = float(rng.uniform(-4.0, 4.0)) if ly is not None else None
        flags = ["speed_uncertain"] if rng.random() < 0.1 else []
        speed = float(rng.normal(160 if stroke == "serve" else 100, 15))
        if stroke == "serve":
            speed = float(150 + 80 * fwd + rng.normal(0, 5))
        seen = rng.random() > 0.15
        if seen:
            shots.append(
                {"shot_id": i, "session_id": spec.id, "segment_id": i, "t_contact": float(t),
                 "stroke_type": stroke, "is_serve": stroke == "serve", "side": side,
                 "landing_x": _f32(lx), "landing_y": _f32(ly),
                 "speed_racket_kmh": _f32(speed), "speed_sigma_kmh": _f32(3.0),
                 "outcome": outcome, "net_clearance_m": _f32(rng.uniform(0.1, 1.5)),
                 "spin_sign": int(rng.choice([-1, 0, 1])), "quality_flags": flags,
                 "swing_id": i}
            )  # fmt: skip
            if stroke == "serve" and spec.serve_contacts:
                contacts.append(
                    {"swing_id": i, "shot_id": i, "t_contact": float(t), "side": side,
                     "frame_contact": int(t * 60), "contact_source": "toss_path",
                     "serve_side": "deuce" if i % 2 else "ad", "racket_hand": "left",
                     "front_foot": "right", "forward_m": fwd, "lateral_m": lat,
                     "height_m": float(rng.normal(2.75, 0.06)), "height_rel": 1.55,
                     "forward_sigma_m": 0.06, "lateral_sigma_m": 0.015, "height_sigma_m": 0.004,
                     "toe_to_baseline_m": 0.05, "toe_moved_m": 0.08, "toe_on_ground": True,
                     "toe_source": "toe", "contact_cov": [0.0] * 9,
                     "flags": ["far"] if side == 1 else []}
                )  # fmt: skip
            swings.append(
                {"swing_id": i, "player": "me", "t_contact": float(t), "side": side,
                 "stroke_type": stroke, "wrist_speed_peak": _f32(rng.uniform(8, 20)),
                 "forward_s": _f32(rng.uniform(0.15, 0.3)),
                 "contact_height_m": _f32(rng.uniform(1.0, 2.8)),
                 "chain_in_order": bool(rng.random() < 0.7)}
            )  # fmt: skip
        if spec.mode == "practice":
            rx, ry = (None, None) if lx is None else (lx * -side, ly * -side)
            practice.append(
                {"segment_id": i, "shot_id": i if seen else None, "t_contact": float(t),
                 "side": side, "shot_kind": "serve" if stroke == "serve" else "groundstroke",
                 "stroke_type": stroke if seen else None, "landing_x": _f32(lx),
                 "landing_y": _f32(ly), "rel_x": _f32(rx), "rel_y": _f32(ry),
                 "outcome": outcome, "excluded": bool(rng.random() < 0.08),
                 "in_target": None if rng.random() < 0.3 else bool(rng.random() < 0.4),
                 "speed_kmh": None if seen else _f32(speed), "flags": flags}
            )  # fmt: skip
    for j in range(spec.n_other_swings):
        swings.append(
            {"swing_id": 10_000 + j, "player": "me", "t_contact": float(rng.uniform(0, duration)),
             "side": -1, "stroke_type": "other", "wrist_speed_peak": _f32(rng.uniform(2, 8))}
        )  # fmt: skip
    tables.write_table(_table(SHOTS, shots), session.shots_path, SHOTS)
    if spec.mode == "practice":
        tables.write_table(_table(PRACTICE, practice), session.practice_path, PRACTICE)
    swings.sort(key=lambda s: s["t_contact"])
    tables.write_table(_table(SWINGS, swings), session.swings_path, SWINGS)
    if spec.serve_contacts:
        session.pose_dir.mkdir(parents=True, exist_ok=True)
        tables.write_table(
            _table(SERVE_CONTACT, contacts), session.serve_contact_path, SERVE_CONTACT
        )

    # Frames at 5 Hz (some dark), movement at 15 Hz in a few runs.
    ft = np.arange(0, duration, 0.2)
    frames = pa.table(
        {"frame": (ft * 60).astype(np.int64), "t_s": ft,
         "luma": np.where(rng.random(len(ft)) < 0.05, 1.0, 120.0).astype(np.float32),
         "view": np.full(len(ft), 0.9, np.float32)}
    )  # fmt: skip
    tables.write_table(frames, session.pass1_frames_path, PASS1_FRAMES)
    mt = np.arange(0, duration, 1 / 15)
    keep = (mt % 120) < 100  # 20 s gaps between runs
    mt = mt[keep]
    run = (mt // 120).astype(np.int32)
    x = np.clip(np.cumsum(rng.normal(0, 0.08, len(mt))), -6, 6)
    y = np.clip(-12 + np.cumsum(rng.normal(0, 0.08, len(mt))), -16, 16)
    speed = np.abs(rng.normal(1.0, 0.8, len(mt)))
    move = pa.table(
        {"frame": np.round(mt * 60).astype(np.int64), "t_s": mt,
         "player": ["me"] * len(mt), "x": x.astype(np.float32), "y": y.astype(np.float32),
         "speed": speed.astype(np.float32), "run": run}
    )  # fmt: skip
    move = pa.table({f.name: move.column(f.name) if f.name in move.column_names else
                     pa.nulls(len(mt), f.type) for f in MOVEMENT})  # fmt: skip
    tables.write_table(move, session.movement_path, MOVEMENT)
    if write_records:
        run_stats_stage(session)
    return session


def run_stats_stage(session: Session, settings: AppSettings | None = None) -> None:
    ctx = StageContext(
        session=session,
        config=session.load_config(),
        settings=settings or AppSettings(),
        fingerprint="test",
        stage_name="stats",
    )
    StatsStage().run(ctx)


def make_library(root, specs: list[Spec]) -> Library:
    lib = Library(root).init()
    for spec in specs:
        make_session(lib, spec)
    return lib
