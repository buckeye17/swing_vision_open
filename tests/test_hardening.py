"""M7 hardening: failure messages, the GPU out-of-memory retry, the disk-space check,
needs-action reasons, relinking a moved video, and pages that fail to build."""

from __future__ import annotations

import errno
import shutil
from collections import namedtuple

import pytest

from swingvision import services
from swingvision.pipeline import runner, worker
from swingvision.pipeline.errors import describe
from swingvision.pipeline.runner import Registry, run
from swingvision.pipeline.stage import NeedsUserAction
from swingvision.pipeline.worker import run_worker
from swingvision.storage.library import Job, Library
from tests.test_runner import Recorder, make_session, stage_cls
from tests.test_worker import _setup


class OutOfMemoryError(RuntimeError):  # stands in for torch.cuda.OutOfMemoryError
    pass


def test_describe_explains_common_failures():
    assert "GPU ran out of memory" in describe(OutOfMemoryError("CUDA out of memory. Tried"))
    assert "disk is full" in describe(OSError(errno.ENOSPC, "No space left on device"))
    assert "permission denied" in describe(PermissionError(13, "denied", "x.parquet"))
    assert "GPU error" in describe(RuntimeError("CUDA error: an illegal memory access"))
    assert describe(ValueError("plain")) == "plain"


def test_failure_message_names_the_stage_and_the_fix(tmp_path, settings):
    s = make_session(tmp_path)
    reg = Registry([stage_cls("a", rec=Recorder(), fail=OSError(errno.ENOSPC, "full"))()])
    res = run(reg, s, settings)
    assert res.status == "failed" and res.message.startswith("a (a) failed: the output disk")


def test_gpu_stage_retries_once_after_out_of_memory(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(runner, "OOM_RETRY_WAIT_S", 0.0)
    s = make_session(tmp_path)
    attempts = []

    cls = stage_cls("g", rec=Recorder())
    cls.uses_gpu = True
    first_run = cls.run

    def flaky(self, ctx):
        attempts.append(1)
        if len(attempts) == 1:
            raise OutOfMemoryError("CUDA out of memory")
        return first_run(self, ctx)

    cls.run = flaky
    assert run(Registry([cls()]), s, settings).status == "done"
    assert len(attempts) == 2


def test_needs_action_reason_reaches_the_job(settings):
    lib = _setup(settings)
    reg = Registry([stage_cls("a", rec=Recorder(), fail=NeedsUserAction("relink", "gone"))()])
    job_id = lib.enqueue_job("s")
    run_worker(settings, once=True, registry=reg)
    job = lib.get_job(job_id)
    assert job.status == "needs_action" and job.action == "relink"


def test_low_disk_space_pauses_the_job(settings, monkeypatch):
    lib = _setup(settings)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda p: usage(10, 9, 1024**3))
    rec = Recorder()
    job_id = lib.enqueue_job("s")
    run_worker(settings, once=True, registry=Registry([stage_cls("a", rec=rec)()]))
    job = lib.get_job(job_id)
    assert job.status == "needs_action" and job.action == "disk_space"
    assert "GB free" in job.message and rec.calls == []


def test_job_rows_ignore_unknown_columns():
    row = {f: None for f in Job.__dataclass_fields__} | {
        "id": 1, "session_id": "s", "status": "queued", "force": "[]", "created_at": "",
        "progress": 0.0, "cancel_requested": 0, "added_later": "x",
    }  # fmt: skip
    assert Job.from_row(row).id == 1  # type: ignore[arg-type]


@pytest.mark.ffmpeg
def test_relink_moved_videos(settings, synthetic_video, tmp_path):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    a_path = shutil.copy(synthetic_video, old_dir / "a.mp4")
    b_path = old_dir / "b.mp4"
    b_path.write_bytes(synthetic_video.read_bytes() + b"\0" * 4096)  # another "video"
    a = services.create_session(settings, a_path).load_config().id
    b = services.create_session(settings, b_path).load_config().id
    lib = services.open_library(settings)
    assert not services.source_missing(lib.get_session(a))

    shutil.move(old_dir, new_dir)  # the footage folder moved
    assert services.source_missing(lib.get_session(a))
    # b's job stopped because its video was missing; relinking picks it back up.
    job = lib.enqueue_job(b)
    lib.claim_next_job("w")
    lib.finish_job(job, "needs_action", "gone", action="relink")

    with pytest.raises(services.RelinkError, match="not the video"):
        services.relink_session(settings, a, new_dir / "b.mp4")
    relinked = services.relink_session(settings, a, new_dir / "a.mp4")
    assert relinked == [a, b]  # b was found in the same folder, by content
    for sid, name in ((a, "a.mp4"), (b, "b.mp4")):
        row = lib.get_session(sid)
        assert not services.source_missing(row) and row["source_path"].endswith(name)
        session = services.session_by_id(settings, sid)
        assert session.load_config().source.path == row["source_path"]
    assert lib.latest_job_for_session(b).status == "queued"


def test_broken_page_shows_an_error(settings):
    import dash

    from swingvision.app.main import create_app

    create_app()
    page = next(p for p in dash.page_registry.values() if p["path"] == "/")
    assert getattr(page["layout"], "_safe", False)

    from swingvision.app import main

    def boom(**_):
        raise ValueError("damaged file")

    out = main._safe_layout(boom, "x")()
    assert "damaged file" in str(out)


def test_library_migrates_jobs_action(settings):
    lib = Library(settings.output_root).init()
    assert lib.schema_version() >= 3


def test_library_opens_the_relink_dialog_from_a_link(settings, synthetic_video):
    import dash

    from swingvision.app.main import create_app

    create_app()
    sid = services.create_session(settings, synthetic_video).load_config().id
    page = next(p for p in dash.page_registry.values() if p["path"] == "/")
    assert "no longer there" not in str(page["layout"](relink=sid))  # the video is in place
    assert "Relink" in str(page["layout"](relink="nope"))  # unknown id: dialog stays closed
