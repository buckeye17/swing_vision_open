from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import pytest

from swingvision.settings import AppSettings, save_settings
from tests.synth_court import make_camera, render

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not on PATH")
    for item in items:
        if "ffmpeg" in item.keywords and not HAS_FFMPEG:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _default_settings(tmp_path, monkeypatch):
    """Every test sees default settings, never the user's own (units, output folder, ...).
    The data dir stays shared so tests can still find installed base weights."""
    monkeypatch.setenv("SWINGVISION_CONFIG_DIR", str(tmp_path / "config"))


@pytest.fixture
def settings(tmp_path, monkeypatch) -> AppSettings:
    """Isolated settings file + output root."""
    monkeypatch.setenv("SWINGVISION_DATA_DIR", str(tmp_path / "data"))
    s = AppSettings(output_root=tmp_path / "out")
    s.processing.chunk_seconds = 2.0
    s.output_root.mkdir()
    save_settings(s)
    return s


CLICK_TIMES = (0.5, 1.25, 2.0, 3.1, 4.4)
COURT_CAMERA = make_camera(width=1280, height=720, k1=0.02, k2=-0.01)


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


@pytest.fixture(scope="session")
def court_video(tmp_path_factory):
    """3 s, 1280×720 @ 30 fps of a court with a "player" walking across it."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("court") / "court.mp4"
    base = render(COURT_CAMERA, occluder=False)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1280x720", "-r", "30", "-i", "pipe:0",
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for i in range(90):
        frame = base.copy()
        x = 200 + 9 * i
        cv2.rectangle(frame, (x, 300), (x + 60, 520), (40, 200, 60), -1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


BALL_FPS = 60


@pytest.fixture(scope="session")
def ball_video(tmp_path_factory):
    """2.5 s, 1280×720 @ 60 fps: a "player" at the near baseline and a ball hit to the far
    court at 0.3 s. Returns (path, image positions (n, 2) with NaN where no ball, bounce times)."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")
    import numpy as np

    from tests.synth_ball import Flight, draw_ball, image_track

    path = tmp_path_factory.mktemp("ball") / "ball.mp4"
    base = render(COURT_CAMERA, occluder=False)
    n = int(2.5 * BALL_FPS)
    t = np.arange(n) / BALL_FPS
    fl = Flight(0.3, (1.0, -11.0, 1.0), (-1.5, 18.0, 4.5))
    px, bounces, _ = image_track(COURT_CAMERA, fl, t)
    X, _, _ = fl.positions(t)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1280x720", "-r", str(BALL_FPS), "-i", "pipe:0",
            "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for i in range(n):
        frame = base.copy()
        cv2.rectangle(frame, (700, 420), (760, 640), (40, 200, 60), -1)
        if np.isfinite(px[i]).all():
            depth = COURT_CAMERA.depth(X[i : i + 1])[0]
            r = max(2.2, 2 * COURT_CAMERA.f * 0.0335 / depth)
            draw_ball(frame, px[i, 0], px[i, 1], r=r)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path, px, bounces
