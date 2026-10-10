"""Devices, speed calibrations and their library and services plumbing (M7c)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from swingvision import services
from swingvision.io.probe import device_key
from swingvision.pipeline.stages.speed import speed_calibration_for
from swingvision.storage import tables
from swingvision.storage.library import MIGRATIONS, Library
from swingvision.storage.schemas import SPEED_REFS, SessionConfig, SourceInfo, VideoInfo
from swingvision.storage.session import Session
from swingvision.training import speed_labels

VIDEO = VideoInfo(
    codec="hevc", width=3840, height=2160, rotation_cw=180, display_width=3840,
    display_height=2160, fps_nominal=60.0, fps_avg=59.83, is_vfr=False, duration_s=100.0,
    n_frames_est=6000, device_make="OnePlus", device_model="OnePlus Open",
    device_lens="back_main",
)  # fmt: skip
KEY = "oneplus-open-back-main-3840x2160-60"


def _session(root, sid: str, video=VIDEO, key_in_config: bool = True) -> Session:
    dir_name = f"2026-10-01_s_{sid}"
    session = Session.open(root, dir_name)
    session.path.mkdir(parents=True)
    cfg = SessionConfig(
        id=sid, name=f"S {sid}", created_at=datetime.now(UTC), dir_name=dir_name,
        source=SourceInfo(path=f"C:/nowhere/{sid}.mp4", size_bytes=1, fast_hash="h", mtime=0.0),
        video=video, mode="practice", device_key=device_key(video) if key_in_config else None,
    )  # fmt: skip
    session.save_config(cfg)
    return session


def _register(lib: Library, session: Session) -> None:
    cfg = session.load_config()
    lib.add_session(
        id=cfg.id, name=cfg.name, dir_name=cfg.dir_name, created_at=cfg.created_at.isoformat(),
        mode="practice", submode="serve", source_path=cfg.source.path, source_hash="h",
        duration_s=100.0,
    )  # fmt: skip
    if cfg.device_key:
        lib.add_device(cfg.device_key, cfg.video)
        lib.update_session(cfg.id, device_key=cfg.device_key)


def _refs(session: Session, ratios, status="accepted") -> None:
    rows = [
        {"ref_id": i, "t_contact": 10.0 + 5 * i, "ratio": r, "ratio_sigma": 0.006,
         "v_fit_kmh": 120 + 5 * i, "v_ref_kmh": (120 + 5 * i) * r, "rs_rate": -0.15,
         "status": status, "source": "auto", "flags": []}
        for i, r in enumerate(ratios)
    ]  # fmt: skip
    session.ball_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {f.name: pa.array([r.get(f.name) for r in rows], f.type) for f in SPEED_REFS},
        schema=SPEED_REFS,
    )
    tables.write_table(table, session.speed_refs_path, SPEED_REFS)


def test_v6_migration_backfills_devices(tmp_path):
    root = tmp_path / "out"
    (root / "sessions").mkdir(parents=True)
    conn = sqlite3.connect(root / "library.sqlite")
    for i, script in enumerate(MIGRATIONS[:5], start=1):
        conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {i};\nCOMMIT;")
    a = _session(root, "aaaa1111", key_in_config=False)  # probed with tags, no key stored
    b = _session(root, "bbbb2222", VideoInfo(**{**VIDEO.model_dump(), "device_model": None}))
    for s in (a, b):
        cfg = s.load_config()
        conn.execute(
            "INSERT INTO sessions (id, name, dir_name, created_at, updated_at, mode, "
            "source_path, source_hash) VALUES (?, ?, ?, ?, ?, 'practice', 'x', 'h')",
            (cfg.id, cfg.name, cfg.dir_name, "2026-10-01", "2026-10-01"),
        )
    conn.commit()
    conn.close()
    lib = Library(root).init()
    assert lib.schema_version() == len(MIGRATIONS) == 6
    assert lib.get_session("aaaa1111")["device_key"] == KEY
    assert lib.get_session("bbbb2222")["device_key"] is None  # no model tag
    dev = lib.get_device(KEY)
    assert dev["model"] == "OnePlus Open" and dev["width"] == 3840 and dev["fps"] == 60.0


def test_calibrations_are_versioned_and_one_is_active(tmp_path):
    lib = Library(tmp_path).init()
    lib.add_device(KEY, VIDEO)
    c1 = lib.add_calibration(
        "c1", KEY, [{"ratio": 1.0}], {"chi2_dof": 1}, model="scalar", k=1.01, k_sigma=0.004,
        n_refs=4,
    )  # fmt: skip
    c2 = lib.add_calibration("c2", KEY, [], {}, model="scalar", k=1.02, k_sigma=0.003, n_refs=8)
    assert (c1["version"], c2["version"]) == (1, 2)
    assert lib.active_calibration(KEY)["id"] == "c2"
    assert [c["id"] for c in lib.list_calibrations(KEY)] == ["c2", "c1"]
    old = lib.get_calibration("c1")
    assert old["refs"] == [{"ratio": 1.0}] and not old["active"]
    lib.set_active_calibration(KEY, "c1")
    assert lib.active_calibration(KEY)["id"] == "c1"
    lib.set_active_calibration(KEY, None)
    assert lib.active_calibration(KEY) is None
    with pytest.raises(ValueError):
        lib.set_active_calibration(KEY, "nope")
    with pytest.raises(ValueError):
        lib.add_calibration("c3", KEY, [], {}, bogus=1)


def test_calibrate_a_device_and_apply_it(settings):
    lib = services.open_library(settings)
    s1, s2 = _session(lib.root, "s1aaaaaa"), _session(lib.root, "s2bbbbbb")
    for s in (s1, s2):
        _register(lib, s)
    _refs(s1, [1.018, 1.024, 1.021])
    _refs(s2, [1.02, 1.6], status="rejected")  # rejected ones don't count
    assert len(services.device_references(settings, KEY)) == 3
    cal, row, _refs_used = services.calibrate_device(settings, KEY, save=False)
    assert cal.k == pytest.approx(1.021, abs=0.003) and row is None
    cal, row, _refs_used = services.calibrate_device(settings, KEY)
    assert row["active"] and row["n_refs"] == 3 and len(row["refs"]) == 3
    assert {r["session_id"] for r in row["refs"]} == {"s1aaaaaa"}
    # Each session's shots use the device's active calibration ...
    applied = speed_calibration_for(settings, s2.load_config())
    assert applied["id"] == row["id"] and applied["k"] == pytest.approx(cal.k)
    # ... or none, or one picked by hand.
    services.set_session_speed_calibration(settings, "s2bbbbbb", "none")
    assert speed_calibration_for(settings, s2.load_config()) is None
    services.set_session_speed_calibration(settings, "s2bbbbbb", row["id"])
    assert speed_calibration_for(settings, s2.load_config())["id"] == row["id"]
    with pytest.raises(ValueError):
        services.set_session_speed_calibration(settings, "s2bbbbbb", "nope")
    # A calibration no tighter than uncalibrated speeds (2σ = 4% > 3%) isn't applied.
    lib.add_calibration("loose", KEY, [], {}, model="scalar", k=1.0, k_sigma=0.02, n_refs=3)
    assert speed_calibration_for(settings, s1.load_config()) is None


def test_too_few_references(settings):
    lib = services.open_library(settings)
    s1 = _session(lib.root, "s1aaaaaa")
    _register(lib, s1)
    _refs(s1, [1.01, 1.02])
    cal, row, refs = services.calibrate_device(settings, KEY)
    assert cal is None and row is None and len(refs) == 2
    with pytest.raises(ValueError):
        services.calibrate_device(settings, "unknown-device")


def test_merge_and_rename_devices(settings):
    lib = services.open_library(settings)
    other = VideoInfo(**{**VIDEO.model_dump(), "device_lens": None})
    a, b = _session(lib.root, "aaaa1111"), _session(lib.root, "bbbb2222", other)
    _register(lib, a)
    _register(lib, b)
    kb = device_key(other)
    assert kb != KEY and len(lib.list_devices()) == 2
    services.rename_device(settings, KEY, "  My phone ")
    assert services.device_name(lib.get_device(KEY)) == "My phone"
    services.rename_device(settings, KEY, "")
    assert "OnePlus Open" in services.device_name(lib.get_device(KEY))
    assert services.merge_devices(settings, kb, KEY) == ["bbbb2222"]
    assert lib.get_device(kb) is None
    assert lib.get_session("bbbb2222")["device_key"] == KEY
    assert b.load_config().device_key == KEY
    assert [d["n_sessions"] for d in lib.list_devices()] == [2]


def test_review_edits_are_ground_truth(settings):
    lib = services.open_library(settings)
    s = _session(lib.root, "s1aaaaaa")
    _register(lib, s)
    services.set_air_temperature(settings, "s1aaaaaa", 14.25)
    services.edit_speed_ref(settings, "s1aaaaaa", 12.0, status="accepted", t_tape=12.40123)
    services.edit_speed_ref(settings, "s1aaaaaa", 30.0, marked=True)
    gt = speed_labels.load_labels(lib.root, "s1aaaaaa")
    assert gt["device_key"] == KEY and gt["air_temp_c"] == pytest.approx(14.2, abs=0.06)
    assert [(r["t"], r["status"], r["marked"]) for r in gt["refs"]] == [
        (12.0, "accepted", False),
        (30.0, None, True),
    ]
    assert lib.get_session("s1aaaaaa")["air_temp_c"] == pytest.approx(14.2, abs=0.06)
    with pytest.raises(ValueError):
        services.set_air_temperature(settings, "s1aaaaaa", 80.0)
    assert json.loads(s.edits_path.read_text())["speed_refs"][0]["t_tape"] == 12.40123
