from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from swingvision.settings import AppSettings, save_settings

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not on PATH")
    for item in items:
        if "ffmpeg" in item.keywords and not HAS_FFMPEG:
            item.add_marker(skip)


@pytest.fixture
def settings(tmp_path, monkeypatch) -> AppSettings:
    """Isolated settings file + output root."""
    monkeypatch.setenv("SWINGVISION_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("SWINGVISION_DATA_DIR", str(tmp_path / "data"))
    s = AppSettings(output_root=tmp_path / "out")
    s.processing.chunk_seconds = 2.0
    s.output_root.mkdir()
    save_settings(s)
    return s


CLICK_TIMES = (0.5, 1.25, 2.0, 3.1, 4.4)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """5 s, 640x360 @ 60 fps, rotated 180°, with sharp clicks at CLICK_TIMES over quiet noise."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("video")
    plain = d / "plain.mp4"
    clicks = "+".join(f"0.9*between(t,{t},{t}+0.004)*sin(2*PI*3000*t)" for t in CLICK_TIMES)
    audio = f"aevalsrc='0.002*random(0)+{clicks}':s=48000:d=5"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=60:duration=5",
            "-f",
            "lavfi",
            "-i",
            audio,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(plain),
        ],
        check=True,
    )
    rotated = d / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-display_rotation",
            "180",
            "-i",
            str(plain),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    return rotated
