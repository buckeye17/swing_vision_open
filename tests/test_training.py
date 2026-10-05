"""Ball labels → training samples → a (tiny) trained U-Net that loads as a detector."""

from __future__ import annotations

import numpy as np
import torch

from swingvision.ball.detectors import make_ball_detector
from swingvision.ball.detectors.unet import list_weights
from swingvision.players.detect import Crop
from swingvision.training import clipcache as cc
from swingvision.training.labels import BallLabel, LabelStore
from swingvision.training.train_ball import (
    SCALE,
    BallDataset,
    TrainConfig,
    collect,
    ensure_half_cache,
    half_path,
    train,
)
from tests.synth_ball import draw_ball

W, H = 1280, 720


def _ball_xy(f: int) -> tuple[float, float]:
    return 300 + 9.3 * f, 200 + 4.1 * f


def _make_clip(store: LabelStore, n: int = 12):
    clip = store.new_clip("s1", 10, 10 + n - 1, 0.0, n / 60, kind="gt", split="train")
    rng = np.random.default_rng(0)
    base = rng.normal(90, 4, (H, W, 3)).clip(0, 255).astype(np.uint8)
    for f in cc.cached_frames(clip):
        img = base.copy()
        x, y = _ball_xy(f)
        draw_ball(img, x, y, r=5)
        cc.write_jpeg(cc.frame_path(store, clip, f), img, quality=98)
        if clip.frame0 <= f <= clip.frame1:
            clip.set(f, BallLabel(x=x, y=y))
    store.save(clip)
    return clip


def test_dataset_target_on_the_ball(tmp_path):
    store = LabelStore(tmp_path)
    clip = _make_clip(store)
    samples = collect(store, [clip])
    assert len(samples) == len(clip.frames)
    ensure_half_cache(store, samples)
    assert half_path(store, clip, clip.frame0).exists()
    ds = BallDataset(store, samples, {}, train=False, cfg=TrainConfig("t"))
    for i in (0, 5):
        x, y = ds[i]
        assert x.shape == (9, 320, 320) and y.shape == (160, 160)
        assert y.max() == 1.0
        # The positive pixel is where the detector would report the ball: invert the
        # heatmap → full-resolution mapping of base.to_full_res (scale SCALE / 2).
        iv, iu = np.unravel_index(np.argmax(y), y.shape)
        s = samples[i]
        bx, by = (s.x + 0.5) * SCALE - 0.5, (s.y + 0.5) * SCALE - 0.5
        # Validation patches are centered on the ball (±0.4 of the patch): recover the offset
        # from the heatmap and compare in model pixels.
        cx = (iu + 0.5) * 2 - 0.5
        cy = (iv + 0.5) * 2 - 0.5
        img = x[3:6].transpose(1, 2, 0)
        yel = np.minimum(img[..., 0], img[..., 1]) - img[..., 2]
        py, px = np.unravel_index(np.argmax(yel), yel.shape)
        assert abs(px - cx) <= 2.5 and abs(py - cy) <= 2.5, (px, py, cx, cy, bx, by)


def test_tiny_training_run_loads_as_detector(settings):
    store = LabelStore(settings.output_root)
    clip = _make_clip(store)
    cfg = TrainConfig("tiny", epochs=1, batch=4, width=8, workers=0)
    card = train(store, settings, cfg, [clip], [], log=lambda m: None, device="cpu")
    assert card["train_frames"] == len(clip.frames)
    assert [c["name"] for c in list_weights(settings.output_root)] == ["tiny"]
    det = make_ball_detector("unet:tiny@640", device="cpu")
    det.prepare(None, Crop(0, 0, W, H))
    frames = [
        torch.from_numpy(cc.read_frame(store, clip, f)).permute(2, 0, 1).contiguous()
        for f in range(clip.frame0 - 2, clip.frame0 + 3)
    ]
    out = det.detect(frames, [2])
    assert len(out) == 1 and out[0].shape[1] == 3
