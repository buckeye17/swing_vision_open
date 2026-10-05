"""pass1_detect → players_track → movement on a rendered court video (stage contract tests).

A color-threshold "detector" stands in for YOLO so the tests run on the CPU without weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from swingvision import services
from swingvision.court import calibration as calib
from swingvision.io.frames import RateSampler, grab_frame, open_source
from swingvision.io.probe import probe_video
from swingvision.pipeline.runner import plan, run
from swingvision.pipeline.stages import default_registry
from swingvision.players import detect
from swingvision.storage import tables
from swingvision.storage.schemas import MOVEMENT, PERSON_DETECTIONS, PLAYER_TRACKS

pytestmark = pytest.mark.ffmpeg


class GreenBlobDetector:
    """Finds the bright green rectangle the court_video fixture moves across the court."""

    name = "fake_green"

    def __init__(self, **_):
        self.calls = 0

    def detect(self, images, crop):
        out = []
        for im in images:
            self.calls += 1
            im = im.cpu().numpy().astype(int)
            sub = im[:, crop.y0 : crop.y1, crop.x0 : crop.x1]
            mask = (sub[1] > 160) & (sub[0] < 110) & (sub[2] < 120)
            ys, xs = np.nonzero(mask)
            if len(xs) < 50:
                out.append(np.zeros((0, 5), np.float32))
                continue
            box = [xs.min() + crop.x0, ys.min() + crop.y0, xs.max() + 1 + crop.x0,
                   ys.max() + 1 + crop.y0, 0.9]  # fmt: skip
            out.append(np.array([box], np.float32))
        return out


@pytest.fixture
def players_settings(settings, monkeypatch):
    monkeypatch.setitem(detect.DETECTORS, "fake_green", lambda **kw: GreenBlobDetector(**kw))
    p = settings.processing
    p.person_model = "fake_green"
    p.decode_backend = "pyav"
    p.person_rate_hz = 15.0
    p.calibration_auto_accept_px = 2.0
    p.calibration_window_s = 1.0
    p.calibration_frames_per_window = 3
    return settings


def test_players_pipeline(players_settings, court_video):
    s = players_settings
    session = services.create_session(s, court_video)
    reg = default_registry()
    result = run(reg, session, s, targets=["movement"])
    assert result.status == "done", result.message
    assert {"camera", "pass1_detect", "players_track", "movement"} <= set(result.ran)

    frames = tables.read_table(session.pass1_frames_path)
    det = tables.read_table(session.person_detections_path)
    assert det.schema.equals(PERSON_DETECTIONS, check_metadata=False)
    # 3 s at 15 Hz from a 30 fps video: every other frame, in two 2 s chunks.
    assert frames.num_rows == 45
    fidx = frames.column("frame").to_numpy()
    assert np.array_equal(fidx, np.arange(0, 90, 2))
    assert np.allclose(frames.column("t_s").to_numpy(), fidx / 30.0, atol=1e-6)
    # Every frame shows the calibrated court (the walking blob barely changes the picture).
    assert frames.column("view").to_numpy().min() > 0.8
    assert det.num_rows == 45

    tracks = tables.read_table(session.player_tracks_path)
    assert tracks.schema.equals(PLAYER_TRACKS, check_metadata=False)
    assert tracks.column("role").to_pylist().count("me") == 45
    # The blob's foot point lands on the court, moving right (x grows) at a steady pace.
    x = tracks.column("court_x").to_numpy()
    assert np.all(np.diff(x) > 0)

    mv = tables.read_table(session.movement_path)
    assert mv.schema.equals(MOVEMENT, check_metadata=False)
    assert mv.num_rows == 45 and set(mv.column("source").to_pylist()) == {"bbox"}
    speed = mv.column("speed").to_numpy()
    true_speed = np.median(np.diff(x) / np.diff(tracks.column("t_s").to_numpy()))
    assert np.median(speed[10:-10]) == pytest.approx(true_speed, rel=0.05)
    summary = services.session_by_id(s, session.load_config().id)
    assert summary is not None


def test_recalibration_reruns_tracking_but_not_detection(players_settings, court_video):
    s = players_settings
    session = services.create_session(s, court_video)
    reg = default_registry()
    assert run(reg, session, s, targets=["movement"]).status == "done"
    config = session.load_config()
    assert all(p.fresh for p in plan(reg, session, config, s, ["movement"]))

    # The user nudges the calibration in the editor and confirms it.
    cal = calib.load(session.calibration_path)
    cam = calib.to_camera(cal.camera)
    moved = calib.with_camera(cal, cam.with_params(cam.params() + 2e-4), source="user")
    calib.save(session.court_user_path, moved)
    result = run(reg, session, s, targets=["movement"])
    assert result.status == "done"
    assert "pass1_detect" not in result.ran
    assert {"camera", "players_track", "movement"} <= set(result.ran)


def test_detection_resumes_after_cancel(players_settings, court_video):
    s = players_settings
    session = services.create_session(s, court_video)
    reg = default_registry()
    assert run(reg, session, s, targets=["camera"]).status == "done"

    from swingvision.pipeline.runner import RunHooks

    seen = []

    def cancel_after_first_chunk():
        return session.manifests_dir.joinpath("pass1_detect.chunks.json").exists() and bool(
            seen.append(1) or len(seen) > 3
        )

    result = run(
        reg, session, s, ["pass1_detect"], hooks=RunHooks(is_cancelled=cancel_after_first_chunk)
    )
    assert result.status == "cancelled"
    result = run(reg, session, s, ["pass1_detect"])
    assert result.status == "done"
    assert tables.read_table(session.pass1_frames_path).num_rows == 45


# ---------------------------------------------------------------------------
# FrameSource
# ---------------------------------------------------------------------------


def test_rate_sampler_keeps_one_frame_per_slot():
    t = np.arange(0, 2, 1 / 60)
    keep = RateSampler(15.0)
    kept = [x for x in t if keep(x)]
    assert len(kept) == 30
    assert np.allclose(np.diff(kept), 1 / 15)
    assert all(RateSampler(None)(x) for x in t)


def test_pyav_source_indices_times_and_rotation(settings, synthetic_video):
    info = probe_video("ffprobe", synthetic_video)
    assert info.rotation_cw == 180
    with open_source(synthetic_video, info, backend="pyav", device="cpu") as src:
        frames = list(src.frames(1.0, 2.0))
    idx = np.array([f.index for f in frames])
    t = np.array([f.t_s for f in frames])
    assert len(frames) == 60
    assert np.array_equal(idx, np.arange(60, 120))
    assert np.allclose(t, idx / 60, atol=1e-6)
    ref = grab_frame("ffmpeg", synthetic_video, t[5], info)  # ffmpeg auto-rotates
    img = frames[5].image.permute(1, 2, 0).numpy()
    assert img.shape == ref.shape
    assert np.abs(img.astype(int) - ref.astype(int)).mean() < 3


@pytest.mark.gpu
def test_nvdec_matches_pyav(settings, synthetic_video):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PyNvVideoCodec")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    info = probe_video("ffprobe", synthetic_video)
    with open_source(synthetic_video, info, backend="nvdec", device="cuda") as nv:
        a = list(nv.frames(1.0, 2.0, RateSampler(15.0)))
    with open_source(synthetic_video, info, backend="pyav", device="cpu") as av:
        b = list(av.frames(1.0, 2.0, RateSampler(15.0)))
    assert [f.index for f in a] == [f.index for f in b]
    assert np.allclose([f.t_s for f in a], [f.t_s for f in b])
    nv_img, av_img = a[3].image.cpu().int(), b[3].image.int()
    diff = (nv_img - av_img).abs().float().mean().item()
    flipped = (nv_img.flip(-1).flip(-2) - av_img).abs().float().mean().item()
    # Same upright frame; the YUV→RGB matrices differ slightly (≈1.5 levels on the 4K phone
    # video, ≈5 on this untagged 360p clip).
    assert diff < 8 and flipped > 3 * diff
