from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import pyarrow as pa
import pytest

from swingvision.pipeline.runner import Registry, RunHooks, plan, run
from swingvision.pipeline.stage import NeedsUserAction, Stage, StageContext, chunk_ranges
from swingvision.storage import tables
from swingvision.storage.schemas import AUDIO_ONSETS, PracticeConfig, SessionConfig, SourceInfo
from swingvision.storage.session import Session


def make_session(root, mode="practice") -> Session:
    s = Session(root / "sessions" / "s")
    s.path.mkdir(parents=True)
    s.save_config(
        SessionConfig(
            id="s",
            name="S",
            created_at=datetime.now(UTC),
            dir_name="s",
            mode=mode,
            source=SourceInfo(path="x.mp4", size_bytes=1, fast_hash="h", mtime=0),
            practice=PracticeConfig() if mode == "practice" else None,
        )
    )
    return s


class Recorder:
    def __init__(self):
        self.calls: list[str] = []


def stage_cls(name, deps=(), version=1, modes=None, cfg=None, rec=None, fail=None):
    class _S(Stage):
        pass

    _S.name, _S.depends_on, _S.version = name, tuple(deps), version
    if modes:
        _S.modes = frozenset(modes)

    def config(self, session, config, settings):
        return dict(cfg or {})

    def outputs(self, session):
        return [session.path / f"{name}.txt"]

    def run_(self, ctx: StageContext):
        rec.calls.append(name)
        if fail:
            raise fail
        (ctx.session.path / f"{name}.txt").write_text("ok")
        ctx.progress(1.0)
        return {"n": 1}

    _S.config, _S.outputs, _S.run = config, outputs, run_
    _S.__abstractmethods__ = frozenset()
    return _S


@pytest.fixture
def rec():
    return Recorder()


def test_runs_then_skips_fresh(tmp_path, settings, rec):
    s = make_session(tmp_path)
    reg = Registry([stage_cls("a", rec=rec)(), stage_cls("b", ["a"], rec=rec)()])
    assert run(reg, s, settings).status == "done"
    assert rec.calls == ["a", "b"]
    assert run(reg, s, settings).ran == []
    assert all(p.fresh for p in plan(reg, s, s.load_config(), settings))


def test_config_change_invalidates_downstream(tmp_path, settings, rec):
    s = make_session(tmp_path)
    run(Registry([stage_cls("a", rec=rec)(), stage_cls("b", ["a"], rec=rec)()]), s, settings)
    rec.calls.clear()
    reg2 = Registry([stage_cls("a", cfg={"k": 2}, rec=rec)(), stage_cls("b", ["a"], rec=rec)()])
    planned = plan(reg2, s, s.load_config(), settings)
    assert [p.reason for p in planned] == ["settings changed", "upstream changed (a)"]
    run(reg2, s, settings)
    assert rec.calls == ["a", "b"]


def test_version_bump_and_missing_output_and_force(tmp_path, settings, rec):
    s = make_session(tmp_path)
    run(Registry([stage_cls("a", rec=rec)()]), s, settings)
    p = plan(Registry([stage_cls("a", version=2, rec=rec)()]), s, s.load_config(), settings)
    assert "stage version 1 → 2" in p[0].reason
    (s.path / "a.txt").unlink()
    p = plan(Registry([stage_cls("a", rec=rec)()]), s, s.load_config(), settings)
    assert p[0].reason == "missing output a.txt"
    run(Registry([stage_cls("a", rec=rec)()]), s, settings)
    rec.calls.clear()
    run(Registry([stage_cls("a", rec=rec)()]), s, settings, force=["a"])
    assert rec.calls == ["a"]


def test_targets_pull_in_dependencies_and_mode_filter(tmp_path, settings, rec):
    s = make_session(tmp_path)
    reg = Registry(
        [
            stage_cls("a", rec=rec)(),
            stage_cls("m", ["a"], modes={"match"}, rec=rec)(),
            stage_cls("b", ["a"], rec=rec)(),
        ]
    )
    run(reg, s, settings, targets=["b"])
    assert rec.calls == ["a", "b"]
    with pytest.raises(ValueError):
        plan(reg, s, s.load_config(), settings, targets=["m"])


def test_failure_and_needs_action_stop_the_run(tmp_path, settings, rec):
    s = make_session(tmp_path)
    reg = Registry(
        [stage_cls("a", rec=rec, fail=RuntimeError("boom"))(), stage_cls("b", ["a"], rec=rec)()]
    )
    res = run(reg, s, settings)
    assert res.status == "failed" and "boom" in res.message and res.error
    assert rec.calls == ["a"]
    reg = Registry([stage_cls("a", rec=rec, fail=NeedsUserAction("calib", "check it"))()])
    assert run(reg, s, settings).status == "needs_action"


def test_cancel_before_stage(tmp_path, settings, rec):
    s = make_session(tmp_path)
    reg = Registry([stage_cls("a", rec=rec)()])
    res = run(reg, s, settings, hooks=RunHooks(is_cancelled=lambda: True))
    assert res.status == "cancelled" and rec.calls == []


def test_registry_rejects_unknown_dependency():
    with pytest.raises(ValueError):
        Registry([stage_cls("b", ["a"], rec=Recorder())()])


class ChunkyStage(Stage):
    """Writes one part per chunk; crashes once at chunk 2 to test resume."""

    name = "chunky"
    version = 1
    processed: ClassVar[list[int]] = []
    crash_at: int | None = 2

    def outputs(self, session):
        return [session.path / "chunky.parquet"]

    def run(self, ctx):
        ranges = chunk_ranges(10.0, 2.0)
        tracker = ctx.chunks(len(ranges))
        for i, (a, _b) in enumerate(ranges):
            if tracker.is_done(i):
                continue
            if i == ChunkyStage.crash_at:
                ChunkyStage.crash_at = None
                raise RuntimeError("simulated crash")
            ChunkyStage.processed.append(i)
            t = pa.table(
                {
                    "t_s": [a],
                    "strength": [1.0],
                    "rms_db": [0.0],
                    "centroid_hz": [0.0],
                    "flatness": [0.0],
                }
            )
            tables.write_part(t, tracker.parts_dir, i, AUDIO_ONSETS)
            tracker.mark_done(i)
        tables.consolidate_parts(tracker.parts_dir, self.outputs(ctx.session)[0], AUDIO_ONSETS)
        tracker.cleanup()


def test_chunked_stage_resumes_after_crash(tmp_path, settings):
    s = make_session(tmp_path)
    reg = Registry([ChunkyStage()])
    assert run(reg, s, settings).status == "failed"
    assert ChunkyStage.processed == [0, 1]
    assert run(reg, s, settings).status == "done"
    assert ChunkyStage.processed == [0, 1, 2, 3, 4]  # chunks 0-1 not redone
    out = tables.read_table(s.path / "chunky.parquet")
    assert out.column("t_s").to_pylist() == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert not (s.manifests_dir / "chunky.chunks.json").exists()


def test_chunk_ranges():
    assert chunk_ranges(5.0, 2.0) == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]
    assert chunk_ranges(0, 2.0) == []


def test_config_read_from_upstream_output_is_planned_lazily(tmp_path, settings, rec):
    """A stage whose config depends on a file its upstream writes must not be stale after
    one run (the runner re-plans each stage just before running it)."""
    s = make_session(tmp_path)

    class Writer(Stage):
        name, version = "writer", 1

        def outputs(self, session):
            return [session.path / "value.txt"]

        def run(self, ctx):
            (ctx.session.path / "value.txt").write_text("42")

    class Reader(Stage):
        name, version, depends_on = "reader", 1, ("writer",)

        def config(self, session, config, settings):
            path = session.path / "value.txt"
            return {"value": path.read_text() if path.exists() else None}

        def outputs(self, session):
            return [session.path / "reader.txt"]

        def run(self, ctx):
            (ctx.session.path / "reader.txt").write_text("ok")

    reg = Registry([Writer(), Reader()])
    assert run(reg, s, settings).ran == ["writer", "reader"]
    assert all(p.fresh for p in plan(reg, s, s.load_config(), settings))
    (s.path / "value.txt").write_text("43")  # e.g. the user edited a calibration
    assert [p.stage.name for p in plan(reg, s, s.load_config(), settings) if not p.fresh] == [
        "reader"
    ]
