"""Movement smoothing, distance and speed on synthetic trajectories (PLAN.md §7.3, §12)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from swingvision.players.movement import (
    MovementParams,
    compute_movement,
    heatmap,
    rts_smooth,
    summarize,
)
from swingvision.storage.schemas import MOVEMENT, PASS1_FRAMES, PLAYER_TRACKS

RATE = 15.0


def _frames(duration: float, luma: float = 120.0) -> pa.Table:
    t = np.arange(0, duration, 1 / RATE)
    return pa.table(
        {
            "frame": np.round(t * 60).astype(np.int64),
            "t_s": t,
            "luma": np.full(len(t), luma, np.float32),
            "view": np.full(len(t), 0.9, np.float32),
        },
        schema=PASS1_FRAMES,
    )


def _tracks(frames: pa.Table, xy: np.ndarray, sigma_xy, keep: np.ndarray | None = None):
    """ "me" rows at the given frames (``keep`` masks out missed detections)."""
    n = frames.num_rows
    keep = np.ones(n, bool) if keep is None else keep
    sx, sy = sigma_xy
    cols = {
        "frame": frames.column("frame").to_numpy()[keep],
        "t_s": frames.column("t_s").to_numpy()[keep],
        "x0": np.full(n, 100.0)[keep],
        "y0": np.full(n, 100.0)[keep],
        "x1": np.full(n, 150.0)[keep],
        "y1": np.full(n, 300.0)[keep],
        "conf": np.full(n, 0.9)[keep],
        "court_x": xy[keep, 0],
        "court_y": xy[keep, 1],
        "height_m": np.full(n, 1.8)[keep],
        "sigma_x": np.full(n, sx)[keep],
        "sigma_y": np.full(n, sy)[keep],
        "track_id": np.zeros(n, np.int32)[keep],
        "role": ["me"] * int(keep.sum()),
    }
    return pa.table(cols).cast(PLAYER_TRACKS)


def test_rts_recovers_constant_velocity():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 1 / RATE)
    truth = np.column_stack([3.0 * t, -10 + 0 * t])
    z = truth + rng.normal(0, 0.1, truth.shape)
    xs, _ = rts_smooth(t, z, np.full_like(z, 0.01), q=8.0)
    speed = np.hypot(xs[:, 2], xs[:, 3])
    assert np.median(speed) == pytest.approx(3.0, rel=0.03)
    assert np.sqrt(np.mean((xs[:, :2] - truth) ** 2)) < 0.07  # better than the raw 0.1 m


def test_known_path_distance_and_top_speed():
    """Shuttle runs: 5 m out and back at up to ~5 m/s, with far-court-like noise along y."""
    frames = _frames(60.0)
    t = frames.column("t_s").to_numpy()
    rng = np.random.default_rng(1)
    phase = 2 * np.pi * t / 6.0
    x = 2.5 * (1 - np.cos(phase))  # 0 → 5 → 0 m every 6 s
    truth = np.column_stack([x, 10.0 + 0 * t])
    true_dist = np.sum(np.hypot(np.diff(truth[:, 0]), np.diff(truth[:, 1])))
    true_top = np.max(2.5 * 2 * np.pi / 6.0 * np.abs(np.sin(phase)))
    noisy = truth + rng.normal(0, [0.05, 0.25], truth.shape)
    mv = compute_movement(_tracks(frames, noisy, (0.05, 0.25)), frames, MovementParams())
    s = summarize(mv, frames, 25.0, MovementParams())
    assert s["distance_m"] == pytest.approx(true_dist, rel=0.05)
    assert s["max_speed_mps"] == pytest.approx(true_top, rel=0.1)
    assert s["coverage"] == pytest.approx(1.0)


def test_standing_still_adds_almost_no_distance():
    frames = _frames(120.0)
    rng = np.random.default_rng(2)
    n = frames.num_rows
    noisy = np.column_stack([rng.normal(0, 0.05, n), 11.0 + rng.normal(0, 0.3, n)])
    mv = compute_movement(_tracks(frames, noisy, (0.05, 0.3)), frames, MovementParams())
    s = summarize(mv, frames, 25.0, MovementParams())
    assert s["distance_m"] < 3.0, s  # vs ~2,000 m if the raw jitter were summed


def test_gaps_bridge_or_split_runs():
    frames = _frames(30.0)
    t = frames.column("t_s").to_numpy()
    xy = np.column_stack([0.5 * t, -10 + 0 * t])
    keep = ~(((t >= 5) & (t < 5.5)) | ((t >= 15) & (t < 20)))
    mv = compute_movement(_tracks(frames, xy, (0.05, 0.05), keep), frames, MovementParams())
    assert mv.schema.equals(MOVEMENT, check_metadata=False)
    runs = mv.column("run").to_numpy()
    src = np.array(mv.column("source").to_pylist())
    assert runs.max() == 1  # the 0.5 s gap is bridged, the 5 s gap splits
    bridged = t[(t >= 5) & (t < 5.5)]
    mt = mv.column("t_s").to_numpy()
    assert np.isin(bridged, mt[src == "interp"]).all()
    assert not np.isin(t[(t > 15.1) & (t < 19.9)], mt).any()
    # Smoothed positions inside the bridged gap follow the walk.
    gap = (mt >= 5) & (mt < 5.5)
    assert np.allclose(mv.column("x").to_numpy()[gap], 0.5 * mt[gap], atol=0.05)
    s = summarize(mv, _frames(30.0), 25.0, MovementParams())
    assert s["coverage"] == pytest.approx(25.0 / 30.0, abs=0.01)
    assert s["n_runs"] == 2


def test_dark_frames_do_not_count_against_coverage():
    frames = _frames(20.0)
    luma = np.where(frames.column("t_s").to_numpy() >= 10, 5.0, 120.0).astype(np.float32)
    frames = frames.set_column(2, "luma", pa.array(luma))
    t = frames.column("t_s").to_numpy()
    keep = t < 10  # the player can't be seen in the dark half
    xy = np.column_stack([0 * t, -10 + 0 * t])
    mv = compute_movement(_tracks(frames, xy, (0.05, 0.05), keep), frames, MovementParams())
    s = summarize(mv, frames, 25.0, MovementParams())
    assert s["coverage"] == pytest.approx(1.0)
    assert s["dark_frames"] == int((~keep).sum())


def test_heatmap_sums_to_tracked_time_and_folds():
    frames = _frames(20.0)
    t = frames.column("t_s").to_numpy()
    xy = np.column_stack([1.0 + 0 * t, np.where(t < 10, -10.0, 10.0)])
    mv = compute_movement(_tracks(frames, xy, (0.05, 0.05)), frames, MovementParams())
    _, yc, H = heatmap(mv)
    assert H.sum() == pytest.approx(20.0, abs=0.2)
    near = H[yc < 0].sum()
    assert near == pytest.approx(10.0, abs=0.5)
    _, yc, Hf = heatmap(mv, fold=True)
    assert Hf[yc > 0].sum() == pytest.approx(0.0, abs=0.2)  # everything on the near half
