from __future__ import annotations

from swingvision.pipeline.runner import Registry
from swingvision.pipeline.worker import run_worker, worker_lock
from swingvision.storage.library import Library, now_iso
from tests.test_runner import Recorder, make_session, stage_cls


def _setup(settings):
    lib = Library(settings.output_root).init()
    make_session(settings.output_root)
    lib.add_session(
        id="s",
        name="S",
        dir_name="s",
        created_at=now_iso(),
        mode="practice",
        submode=None,
        source_path="x.mp4",
        source_hash="h",
        duration_s=1.0,
    )
    return lib


def test_worker_processes_queue_and_records_stages(settings):
    lib = _setup(settings)
    rec = Recorder()
    reg = Registry([stage_cls("a", rec=rec)(), stage_cls("b", ["a"], rec=rec)()])
    job_id = lib.enqueue_job("s")
    assert run_worker(settings, once=True, registry=reg) == 0
    job = lib.get_job(job_id)
    assert job.status == "done" and job.progress == 1.0
    assert [(s["stage"], s["status"]) for s in lib.job_stages(job_id)] == [
        ("a", "done"),
        ("b", "done"),
    ]
    assert lib.get_session("s")["status"] == "ready"
    assert (settings.output_root / "sessions" / "s" / "logs" / f"job-{job_id:05d}.log").exists()
    assert lib.worker_status()["state"] == "stopped"

    # Second job: everything fresh → skipped.
    job2 = lib.enqueue_job("s")
    run_worker(settings, once=True, registry=reg)
    assert [s["status"] for s in lib.job_stages(job2)] == ["skipped", "skipped"]


def test_failed_stage_marks_job_failed(settings):
    lib = _setup(settings)
    reg = Registry([stage_cls("a", rec=Recorder(), fail=RuntimeError("nope"))()])
    job_id = lib.enqueue_job("s")
    run_worker(settings, once=True, registry=reg)
    job = lib.get_job(job_id)
    assert job.status == "failed" and "nope" in job.message
    assert lib.get_session("s")["status"] == "failed"


def test_second_worker_exits_when_lock_held(settings):
    _setup(settings)
    lock = worker_lock(settings.output_root)
    lock.acquire()
    try:
        assert run_worker(settings, once=True, registry=Registry()) == 0
    finally:
        lock.release()


def test_worker_resumes_job_orphaned_moments_ago(settings):
    """A worker that died seconds ago (fresh heartbeat) must not leave a zombie job."""
    lib = _setup(settings)
    rec = Recorder()
    reg = Registry([stage_cls("a", rec=rec)()])
    job_id = lib.enqueue_job("s")
    lib.claim_next_job("dead-worker")
    lib.update_job(job_id, heartbeat_at=now_iso())
    assert run_worker(settings, once=True, registry=reg) == 0
    assert lib.get_job(job_id).status == "done" and rec.calls == ["a"]
