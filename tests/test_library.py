from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from swingvision.storage.library import (
    CANCELLED,
    DONE,
    MIGRATIONS,
    QUEUED,
    RUNNING,
    Library,
    now_iso,
)


@pytest.fixture
def lib(tmp_path) -> Library:
    lib = Library(tmp_path).init()
    lib.add_session(
        id="s1",
        name="S1",
        dir_name="2026_s1",
        created_at=now_iso(),
        mode="practice",
        submode="self_feed",
        source_path="C:/v.mp4",
        source_hash="h",
        duration_s=10.0,
    )
    return lib


def test_migrations_idempotent(tmp_path):
    lib = Library(tmp_path).init()
    lib.init()
    assert lib.schema_version() == len(MIGRATIONS)


def test_enqueue_claim_finish(lib):
    job_id = lib.enqueue_job("s1", targets=["proxy"], force=["proxy"])
    assert lib.get_session("s1")["status"] == "queued"
    with pytest.raises(ValueError):
        lib.enqueue_job("s1")  # one active job per session
    job = lib.claim_next_job("w1")
    assert job.id == job_id and job.status == RUNNING
    assert job.targets == ["proxy"] and job.force == ["proxy"]
    assert lib.claim_next_job("w1") is None
    lib.finish_job(job_id, DONE, "ok")
    assert lib.get_job(job_id).progress == 1.0
    assert lib.get_session("s1")["status"] == "ready"


def test_cancel_queued_immediately_running_cooperatively(lib):
    j = lib.enqueue_job("s1")
    lib.request_cancel(j)
    assert lib.get_job(j).status == CANCELLED
    j2 = lib.enqueue_job("s1")
    lib.claim_next_job("w")
    lib.request_cancel(j2)
    assert lib.get_job(j2).status == RUNNING
    assert lib.is_cancel_requested(j2)


def test_requeue_stale_running_jobs(lib):
    j = lib.enqueue_job("s1")
    lib.claim_next_job("w")
    old = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    lib.update_job(j, heartbeat_at=old)
    assert lib.requeue_stale_jobs(60) == [j]
    assert lib.get_job(j).status == QUEUED


def test_job_stages_and_delete_cascade(lib):
    j = lib.enqueue_job("s1")
    lib.set_job_stages(j, [("ingest", 1.0, "skipped"), ("proxy", 6.0, "pending")])
    stages = lib.job_stages(j)
    assert [s["stage"] for s in stages] == ["ingest", "proxy"]
    assert stages[0]["progress"] == 1.0
    lib.update_job_stage(j, "proxy", status="running", progress=0.5)
    assert lib.job_stages(j)[1]["progress"] == 0.5
    lib.delete_session("s1")
    assert lib.get_job(j) is None


def test_blank_output_root_means_unset():
    from swingvision.settings import AppSettings

    assert AppSettings.model_validate_json('{"output_root": ""}').output_root is None
    assert AppSettings.model_validate_json('{"output_root": "  "}').output_root is None


def test_settings_saved_before_v2_move_to_the_new_pose_model():
    from swingvision.settings import AppSettings

    old = '{"processing": {"pose_model": "vitpose-base-simple", "person_model": "yolo11l"}}'
    s = AppSettings.model_validate_json(old)
    assert s.processing.pose_model == "vitpose-plus-huge"
    assert s.processing.person_model == "yolo11l"  # other choices are kept
    # Chosen after the change: kept.
    s.processing.pose_model = "vitpose-base-simple"
    again = AppSettings.model_validate_json(s.model_dump_json())
    assert again.processing.pose_model == "vitpose-base-simple"
