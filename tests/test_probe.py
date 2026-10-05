from __future__ import annotations

import pytest

from swingvision.io.probe import fast_hash, parse_probe


def _probe(rotation=None, r="60/1", avg="60/1", w=3840, h=2160, audio=True):
    video = {
        "codec_type": "video",
        "codec_name": "hevc",
        "width": w,
        "height": h,
        "r_frame_rate": r,
        "avg_frame_rate": avg,
        "duration": "100.0",
        "start_time": "0.0",
        "pix_fmt": "yuv420p",
        "bit_rate": "56000000",
    }
    if rotation is not None:
        video["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": rotation}]
    streams = [video]
    if audio:
        streams.append(
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "start_time": "0.010",
            }
        )
    return {
        "streams": streams,
        "format": {"duration": "100.0", "tags": {"creation_time": "2026-10-01T23:25:10Z"}},
    }


def test_phone_180_rotation_and_vfr():
    info = parse_probe(_probe(rotation=-180, r="60/1", avg="1008860000/17037117"))
    assert info.rotation_cw == 180
    assert (info.display_width, info.display_height) == (3840, 2160)
    assert info.is_vfr
    assert info.fps_avg == pytest.approx(59.215, abs=1e-3)
    assert info.n_frames_est == round(100 * info.fps_avg)
    assert info.audio_start_time_s == pytest.approx(0.01)


def test_portrait_rotation_swaps_display_size():
    info = parse_probe(_probe(rotation=-90, w=1920, h=1080))
    assert info.rotation_cw == 90
    assert (info.display_width, info.display_height) == (1080, 1920)


def test_legacy_rotate_tag_and_cfr():
    data = _probe()
    data["streams"][0]["tags"] = {"rotate": "270"}
    info = parse_probe(data)
    assert info.rotation_cw == 270
    assert not info.is_vfr


def test_no_audio_and_attached_pic_ignored():
    data = _probe(audio=False)
    data["streams"].insert(
        0,
        {
            "codec_type": "video",
            "codec_name": "mjpeg",
            "width": 1,
            "height": 1,
            "disposition": {"attached_pic": 1},
        },
    )
    info = parse_probe(data)
    assert info.codec == "hevc"
    assert not info.has_audio


def test_fast_hash_depends_on_head_and_tail(tmp_path):
    a = tmp_path / "a.bin"
    data = bytearray(20 * 1024 * 1024)
    a.write_bytes(bytes(data))
    h1 = fast_hash(a)
    data[-1] = 1
    a.write_bytes(bytes(data))
    assert fast_hash(a) != h1
    data[-1] = 0
    data[10 * 1024 * 1024] = 1  # middle byte: not hashed (by design)
    a.write_bytes(bytes(data))
    assert fast_hash(a) == h1
