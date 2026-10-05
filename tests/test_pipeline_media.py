"""End-to-end M0 stages on a tiny synthetic video (needs ffmpeg)."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from swingvision import services
from swingvision.pipeline.runner import run
from swingvision.pipeline.stages import default_registry
from swingvision.storage import tables
from tests.conftest import CLICK_TIMES

pytestmark = pytest.mark.ffmpeg


def _stream_info(path):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_side_data=rotation",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)["streams"][0]


def test_create_session_and_run_m0_stages(settings, synthetic_video):
    session = services.create_session(settings, synthetic_video, "Synthetic", "practice", "serve")
    config = session.load_config()
    assert config.video.rotation_cw == 180
    assert config.practice.submode == "serve"

    m0 = ["proxy", "audio_onsets"]
    result = run(default_registry(), session, settings, targets=m0)
    assert result.status == "done", result.error
    assert result.ran == ["ingest", "proxy", "audio_onsets"]

    proxy = _stream_info(session.proxy_path)
    assert proxy["height"] == settings.processing.proxy_height
    assert not proxy.get("side_data_list"), "rotation must be baked in, not flagged"

    onsets = tables.read_table(session.audio_onsets_path)
    found = np.sort(onsets.column("t_s").to_numpy())
    for t in CLICK_TIMES:
        assert np.min(np.abs(found - t)) < 0.003, f"click at {t}s not detected: {found}"

    # Everything is fresh now; changing the proxy height re-runs only the proxy.
    assert run(default_registry(), session, settings, targets=m0).ran == []
    settings.processing.proxy_height = 240
    assert run(default_registry(), session, settings, targets=m0).ran == ["proxy"]
    assert _stream_info(session.proxy_path)["height"] == 240


def test_missing_source_needs_relink(settings, synthetic_video, tmp_path):
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(synthetic_video.read_bytes())
    session = services.create_session(settings, copy)
    copy.unlink()
    result = run(default_registry(), session, settings)
    assert result.status == "needs_action"
    assert "not found" in result.message


def test_delete_session_keeps_source(settings, synthetic_video):
    session = services.create_session(settings, synthetic_video)
    sid = session.load_config().id
    services.delete_session(settings, sid)
    assert not session.path.exists()
    assert synthetic_video.exists()
    assert services.open_library(settings).get_session(sid) is None
