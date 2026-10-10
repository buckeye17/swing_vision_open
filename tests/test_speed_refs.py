"""Net-tape speed references and per-device speed calibration (M7c).

The synthetic serves (``tests/synth_speed.py``) are rendered with a video clock running 2%
fast and a 15 ms rolling-shutter readout, fitted like ``ball_3d``, with the racket and tape
sounds synthesized on an accurate clock at the right sound delays. Exit criterion: Δt
recovered within 0.5 ms and the factor within 0.3%.
"""

from __future__ import annotations

import numpy as np
import pytest

from swingvision.ball import speed_refs as sr
from swingvision.io.probe import device_key, device_label, device_tags, parse_probe
from swingvision.storage import edits as ed
from swingvision.storage.schemas import SessionEdits, VideoInfo
from tests import synth_speed as ss


@pytest.fixture(scope="module")
def synth():
    serves, y = ss.render(10, kinds=("net", "let"), clock=1.02, tau=0.015, temp_c=12.0, seed=0)
    audio = sr.AudioClip.from_array(y, ss.SR)
    av, n = sr.estimate_av_offset([s.serve for s in serves], audio, 0.0, 12.0)
    kw = {"container_offset_s": 0.0, "av_offset_s": av, "temp_c": 12.0, "fps": ss.FPS}
    refs = [sr.measure(s.serve, audio, rotation_cw=ss.ROTATION, **kw) for s in serves]
    return serves, refs, av, n


def test_audio_video_offset_is_measured(synth):
    _serves, _refs, av, n = synth
    assert n == 10
    # The phone's 0.1 s plus where the impact falls in its frame (the clock error only
    # stretches time around each serve).
    assert 0.09 < av < 0.13


def test_candidates_net_serves_and_refitted_lets(synth):
    serves, refs, _av, _n = synth
    kinds = [s.serve.flight["end_kind"] for s in serves]
    net = [r for r, k in zip(refs, kinds, strict=True) if k == "net"]
    lets = [r for r, k in zip(refs, kinds, strict=True) if k != "net"]
    assert all(r.is_candidate for r in net), [r.flags for r in net]
    assert sum(r.is_candidate for r in lets) >= 4
    assert all("velocity_change" in r.flags for r in lets)


def test_dt_within_half_a_millisecond(synth):
    serves, refs, _av, _n = synth
    for s, r in zip(serves, refs, strict=True):
        if r.is_candidate:
            assert abs(r.dt_s - (s.t_tape - s.t_imp)) < 0.0005, (r.flags, r.dt_s)
            assert r.v_ref_kmh / 3.6 == pytest.approx(s.v_true_avg, rel=0.006)


def test_factor_within_0_3_percent(synth):
    serves, refs, _av, _n = synth
    use = [(s, r) for s, r in zip(serves, refs, strict=True) if r.is_candidate]
    rows = [
        {"ratio": r.ratio, "ratio_sigma": r.ratio_sigma, "v_fit_kmh": r.v_fit_kmh,
         "rs_rate": r.rs_rate, "session_id": "s1"}
        for _s, r in use
    ]  # fmt: skip
    cal = sr.fit_calibration(rows)
    assert cal is not None and cal.n_refs == len(use)
    # The truth: what each serve's fitted speed should have been multiplied by, weighted
    # like the references (the fits' own scatter is common to both).
    w = np.array([1 / r.ratio_sigma**2 for _s, r in use])
    k_true = np.array([s.v_true_avg * 3.6 / r.v_fit_kmh for s, r in use])
    k_true_w = float(np.sum(w * k_true) / np.sum(w))
    rs = float(np.mean([r.rs_rate for _s, r in use]))
    assert sr.speed_factor(cal, rs) == pytest.approx(k_true_w, rel=0.003)
    assert cal.k == pytest.approx(1.02, abs=0.015)  # the clock, plus the fits' own bias
    assert cal.k_sigma < 0.01
    assert not cal.diagnostics["trend_speed"]["significant"]


def test_no_reference_without_the_tape_sound():
    serves, y = ss.render(3, kinds=("net",), seed=5)
    quiet = y.copy()
    audio = sr.AudioClip.from_array(quiet, ss.SR)
    # Cut the tape ticks out: the window holds only noise.
    c = sr.sound_speed(12.0)
    for s in serves:
        tt = s.t_tape + np.linalg.norm(s.tape - ss.CAM.center) / c + ss.AV_OFFSET
        a, b = int((tt - 0.005) * ss.SR), int((tt + 0.05) * ss.SR)
        quiet[a:b] = np.random.default_rng(0).normal(0, 0.0015, b - a)
    audio = sr.AudioClip.from_array(quiet, ss.SR)
    for s in serves:
        r = sr.measure(
            s.serve, audio, container_offset_s=0.0, av_offset_s=0.115, temp_c=12.0, fps=ss.FPS,
            rotation_cw=ss.ROTATION,
        )  # fmt: skip
        assert not r.is_candidate and "no_tape_sound" in r.flags


def test_user_onsets_override_the_detected_ones(synth):
    serves, refs, av, _n = synth
    s, r = serves[0], refs[0]
    audio = sr.AudioClip.from_array(ss.render(10, kinds=("net", "let"), seed=0)[1], ss.SR)
    moved = sr.measure(
        s.serve, audio, container_offset_s=0.0, av_offset_s=av, temp_c=12.0, fps=ss.FPS,
        rotation_cw=ss.ROTATION, t_tape=r.t_tape_audio + 0.001,
    )  # fmt: skip
    assert moved.onsets_by == "user"
    assert moved.dt_s == pytest.approx(r.dt_s + 0.001, abs=1e-6)


# ---------------------------------------------------------------------------
# Onsets
# ---------------------------------------------------------------------------


def _clicks(snr_amp: float, t0: float = 0.5, sr_: int = 22050, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(sr_) / sr_
    y = rng.normal(0, 0.001, len(t)) + ss._click(t, t0, snr_amp, (2500, 4300, 6100), 0.002)
    return sr.AudioClip.from_array(y, sr_)


@pytest.mark.parametrize("amp", [0.02, 0.05, 0.3])
def test_onset_rule_is_on_the_start_at_any_loudness(amp):
    """The rule extrapolates the rising edge back to the floor, so a quiet tick and a loud
    crack are timed alike (a plain threshold fires later on the quiet one)."""
    for seed in range(4):
        o = sr.onset_near(_clicks(amp, seed=seed), 0.5, 0.02, sr.RefParams())
        assert o is not None
        assert abs(o.t - 0.5) < 0.0003, (amp, seed, o.t - 0.5)


def test_onset_snr_and_floor():
    o = sr.onset_near(_clicks(0.05), 0.5, 0.02, sr.RefParams())
    assert 20 < o.snr_db < 40


def test_sound_speed():
    assert sr.sound_speed(20.0) == pytest.approx(343.2, abs=0.2)
    assert sr.sound_speed(None) == sr.sound_speed(20.0)
    assert sr.sound_speed(0.0) == pytest.approx(331.3)


# ---------------------------------------------------------------------------
# Calibration model
# ---------------------------------------------------------------------------


def _refs(k, tau, n=12, noise=0.004, seed=0, rs=(-0.6, 0.6), speeds=(120, 190)):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x = rng.uniform(*rs)
        v = rng.uniform(*speeds)
        ratio = 1 / (1 / k + tau * x) * (1 + rng.normal(0, noise))
        out.append({"ratio": ratio, "ratio_sigma": noise, "v_fit_kmh": v, "rs_rate": x,
                    "session_id": f"s{i % 3}"})  # fmt: skip
    return out


def test_scalar_when_there_is_no_trend():
    cal = sr.fit_calibration(_refs(1.015, 0.0))
    assert cal.model == "scalar"
    assert cal.k == pytest.approx(1.015, abs=0.004)
    assert cal.loo_sd == pytest.approx(0.004, rel=0.6)
    lo, hi = cal.diagnostics["k_ci95"]
    assert lo < cal.k < hi
    assert not cal.diagnostics["by_session"]["significant"]
    assert sr.speed_factor(cal, 0.5) == cal.k


def test_rolling_shutter_when_the_trend_is_significant():
    cal = sr.fit_calibration(_refs(1.01, 0.03, n=16))
    assert cal.diagnostics["trend_rolling_shutter"]["significant"]
    assert cal.model == "rolling_shutter"
    assert cal.tau_s == pytest.approx(0.03, abs=0.008)
    assert cal.k == pytest.approx(1.01, abs=0.004)
    # A flight crossing the sensor faster is corrected more.
    assert sr.speed_factor(cal, -0.5) > sr.speed_factor(cal, 0.5)


def test_too_few_references():
    assert sr.fit_calibration(_refs(1.0, 0.0, n=2)) is None
    assert sr.speed_factor(None) == 1.0


def test_session_trend_is_reported():
    a = _refs(1.00, 0.0, n=6, seed=1)
    b = _refs(1.04, 0.0, n=6, seed=2)
    for r in a:
        r["session_id"] = "a"
    for r in b:
        r["session_id"] = "b"
    cal = sr.fit_calibration(a + b)
    assert cal.diagnostics["by_session"]["significant"]


# ---------------------------------------------------------------------------
# Devices, readout, edits
# ---------------------------------------------------------------------------


def test_device_tags_from_android_and_iphone():
    assert device_tags(
        {"com.oplus.product.model": "OnePlus Open", "com.oplus.lens.model": "back_main",
         "location": "+39.9-086.0/"}
    ) == ("OnePlus", "OnePlus Open", "back_main")  # fmt: skip
    assert device_tags({"com.android.manufacturer": "Google", "com.android.model": "Pixel 8"}) == (
        "Google",
        "Pixel 8",
        None,
    )
    assert device_tags(
        {"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 15 Pro"}
    ) == ("Apple", "iPhone 15 Pro", None)
    assert device_tags({}) == (None, None, None)


def test_probe_reads_the_device_and_the_key():
    data = {
        "streams": [{
            "codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160,
            "r_frame_rate": "60/1", "avg_frame_rate": "60/1", "duration": "10",
        }],
        "format": {"tags": {"com.oplus.product.model": "OnePlus Open",
                            "com.oplus.lens.model": "back_main"}},
    }  # fmt: skip
    info = parse_probe(data)
    assert (info.device_make, info.device_model, info.device_lens) == (
        "OnePlus",
        "OnePlus Open",
        "back_main",
    )
    assert device_key(info) == "oneplus-open-back-main-3840x2160-60"
    assert "OnePlus Open" in device_label(info)
    bare = VideoInfo(**{**info.model_dump(), "device_model": None})
    assert device_key(bare) is None


@pytest.mark.parametrize(
    ("rotation", "px", "expected"),
    [(0, (100, 540), 0.25), (180, (100, 540), 0.75), (90, (270, 50), 0.75), (270, (270, 50), 0.25)],
)
def test_readout_fraction(rotation, px, expected):
    w, h = (1080, 1920) if rotation in (90, 270) else (1920, 2160)
    assert sr.readout_fraction(np.array(px, float), rotation, w, h) == pytest.approx(expected)


def test_speed_ref_edits_roundtrip():
    e = SessionEdits()
    ed.set_speed_ref(e, 10.0, status="accepted")
    ed.set_speed_ref(e, 10.1, t_tape=12.34567891)
    assert len(e.speed_refs) == 1
    r = ed.speed_ref_edit(e, 10.05)
    assert r.status == "accepted" and r.t_tape == pytest.approx(12.34568)
    ed.set_speed_ref(e, 20.0, marked=True)
    assert [x.t for x in e.speed_refs] == [10.0, 20.0]
    ed.set_speed_ref(e, 10.0, status=None, t_tape=None)
    ed.set_speed_ref(e, 20.0, marked=False)
    assert e.speed_refs == []


def test_stage_reads_a_session(tmp_path):
    """``speed_refs`` on a session on disk: serves from ``swings``, flights, the track, the
    audio file, the calibration and the user's reviews."""
    from datetime import UTC, datetime

    import pyarrow as pa
    import soundfile as sf

    from swingvision.court import calibration as calib
    from swingvision.pipeline.stages.speed import build_speed_refs
    from swingvision.storage import tables
    from swingvision.storage.fsutil import atomic_write_json
    from swingvision.storage.schemas import (
        BALL_FLIGHTS,
        BALL_TRACK,
        SWINGS,
        Calibration,
        SessionConfig,
        SourceInfo,
    )  # fmt: skip
    from swingvision.storage.session import Session

    serves, y = ss.render(4, kinds=("net",), seed=3)
    session = Session.open(tmp_path, "2026-10-06_s_abcd1234")
    for d in (session.path, session.ball_dir, session.pose_dir):
        d.mkdir(parents=True, exist_ok=True)
    video = VideoInfo(
        codec="hevc", width=3840, height=2160, rotation_cw=ss.ROTATION, display_width=3840,
        display_height=2160, fps_nominal=60, fps_avg=60, is_vfr=False, duration_s=30,
        n_frames_est=1800, has_audio=True, audio_start_time_s=0.0,
    )  # fmt: skip
    config = SessionConfig(
        id="abcd1234", name="synthetic", created_at=datetime.now(UTC),
        dir_name=session.path.name, mode="practice", video=video, air_temp_c=12.0,
        source=SourceInfo(path="C:/nowhere.mp4", size_bytes=1, fast_hash="h", mtime=0),
    )  # fmt: skip
    session.save_config(config)
    calib.save(
        session.calibration_path,
        Calibration(
            source="auto", created_at="2026-10-06T00:00:00", camera=calib.to_params(ss.CAM)
        ),
    )
    sf.write(str(session.audio_path), y, ss.SR, format="FLAC")
    swings, flights, det, tosses = [], [], [], {}
    for i, s in enumerate(serves):
        sv = s.serve
        swings.append({"swing_id": i, "player": "me", "t_contact": sv.t_contact,
                       "frame_contact": round(sv.t_contact * 60), "hit_event_id": 100 + i,
                       "side": -1, "stroke_type": "serve", "racket_hand": "left"})  # fmt: skip
        flights.append({**sv.flight, "flight_id": i, "start_event_id": 100 + i,
                        "speed0": sv.flight["speed_avg"], "flags": []})  # fmt: skip
        det += [(round(t * 60), t, *p) for t, p in zip(sv.det_t, sv.det_px, strict=True)]
        tosses[str(i)] = {"pos": sv.contact.tolist(), "cov": sv.contact_cov.tolist()}

    def table(rows, schema):
        return pa.table(
            {f.name: pa.array([r.get(f.name) for r in rows], f.type) for f in schema},
            schema=schema,
        )

    tables.write_table(table(swings, SWINGS), session.swings_path, SWINGS)
    atomic_write_json(session.swings_summary_path, {"tosses": tosses})
    tables.write_table(table(flights, BALL_FLIGHTS), session.ball_flights_path, BALL_FLIGHTS)
    track = [{"frame": f, "t_s": t, "x": x, "y": yy, "score": 0.9, "tracklet": 0,
              "source": "detected"} for f, t, x, yy in det]  # fmt: skip
    tables.write_table(table(track, BALL_TRACK), session.ball_track_path, BALL_TRACK)

    refs, summary = build_speed_refs(session, config)
    assert summary["near_serves"] == 4 and summary["av_offset_serves"] == 4
    assert summary["sound_speed_mps"] == pytest.approx(sr.sound_speed(12.0), abs=0.01)
    rows = refs.to_pylist()
    assert len(rows) == 4 and all(r["status"] == "candidate" for r in rows)
    for r, s in zip(rows, serves, strict=True):
        assert r["dt_s"] == pytest.approx(s.t_tape - s.t_imp, abs=0.0005)
    # A review: accepted, one rejected, and a serve marked by hand stays listed.
    e = SessionEdits()
    ed.set_speed_ref(e, rows[0]["t_contact"], status="accepted")
    ed.set_speed_ref(
        e, rows[1]["t_contact"], status="rejected", t_tape=rows[1]["t_tape_audio"] + 0.001
    )
    refs2, summary2 = build_speed_refs(session, config, edits=e)
    by = {r["swing_id"]: r for r in refs2.to_pylist()}
    assert by[0]["status"] == "accepted" and by[1]["status"] == "rejected"
    assert by[1]["onsets_by"] == "user"
    assert by[1]["dt_s"] == pytest.approx(rows[1]["dt_s"] + 0.001, abs=1e-5)  # 10 µs steps
    assert (summary2["accepted"], summary2["rejected"]) == (1, 1)
