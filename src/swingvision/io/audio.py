"""Audio extraction and onset detection.

Racket impacts are short broadband transients, so they show up as sharp
spectral-flux peaks. Onsets are reported on the *video* timeline: audio and
video streams can start at slightly different PTS, which is corrected here.
Later stages (M3) also compensate the sound travel delay from far-side hits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import soundfile as sf

from swingvision.io.ffmpeg import run_ffmpeg
from swingvision.storage.fsutil import atomic_write
from swingvision.storage.schemas import AUDIO_ONSETS

SAMPLE_RATE = 22050


@dataclass(frozen=True)
class OnsetParams:
    sample_rate: int = SAMPLE_RATE
    hop_length: int = 128  # 5.8 ms at 22.05 kHz
    n_fft: int = 1024
    min_strength: float = 4.0  # robust z-score threshold
    min_gap_s: float = 0.04
    pad_s: float = 1.0  # context read either side of each chunk

    def as_config(self) -> dict:
        return self.__dict__.copy()


def extract_audio(
    ffmpeg: str,
    src: Path,
    dst: Path,
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
    on_progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> None:
    """Mono FLAC at ``sample_rate`` (lossless, ~1/3 the size of WAV)."""

    def write(tmp: Path) -> None:
        run_ffmpeg(
            ffmpeg,
            [
                "-i",
                str(src),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "flac",
                "-f",
                "flac",
                "-y",
                str(tmp),
            ],
            duration_s,
            on_progress,
            is_cancelled,
        )

    atomic_write(dst, write, suffix=".flac")


def audio_duration_s(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def _refine_onset(y: np.ndarray, center: int, sr: int, search_s: float = 0.035) -> int:
    """First sample near ``center`` where the high-passed envelope reaches half its local peak."""
    lo = max(0, center - int(search_s * sr))
    hi = min(len(y), center + int(search_s * sr))
    if hi - lo < 8:
        return center
    hp = np.abs(np.diff(y[lo:hi], prepend=y[lo]))
    k = max(1, int(0.0005 * sr))
    env = np.convolve(hp, np.ones(k) / k, mode="same")
    peak = env.max()
    if peak <= 0:
        return center
    return lo + int(np.argmax(env >= 0.5 * peak))


def detect_onsets(
    path: Path, start_s: float, end_s: float, params: OnsetParams, time_offset_s: float = 0.0
) -> pa.Table:
    """Onsets with ``start_s <= t < end_s`` (audio timeline), reported as ``t + time_offset_s``."""
    import librosa  # heavy import (numba); keep it out of module import time

    sr = params.sample_rate
    info = sf.info(str(path))
    if info.samplerate != sr:
        raise ValueError(f"Expected {sr} Hz audio, got {info.samplerate} Hz")
    read_start = max(0, int((start_s - params.pad_s) * sr))
    read_end = min(info.frames, int((end_s + params.pad_s) * sr))
    if read_end <= read_start:
        return AUDIO_ONSETS.empty_table()
    y, _ = sf.read(str(path), start=read_start, stop=read_end, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    hop = params.hop_length
    spec = np.abs(librosa.stft(y, n_fft=params.n_fft, hop_length=hop)) ** 2
    log_spec = librosa.power_to_db(spec, ref=np.max)
    flux = np.maximum(0.0, np.diff(log_spec, axis=1, prepend=log_spec[:, :1])).mean(axis=0)

    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) * 1.4826 + 1e-6
    z = (flux - med) / mad

    wait = max(1, int(params.min_gap_s * sr / hop))
    peaks = librosa.util.peak_pick(
        z, pre_max=3, post_max=3, pre_avg=20, post_avg=20, delta=params.min_strength, wait=wait
    )
    peaks = peaks[z[peaks] >= params.min_strength]

    # Spectral flux with a centred window fires up to n_fft/2 early; refine each onset to the
    # transient's leading edge in the time domain (sub-millisecond, vs. a 16.7 ms video frame).
    onset_samples = np.array([_refine_onset(y, p * hop, sr) for p in peaks], dtype=np.int64)
    times = (read_start + onset_samples) / sr
    keep = (times >= start_s) & (times < end_s)
    peaks, onset_samples, times = peaks[keep], onset_samples[keep], times[keep]
    if len(peaks) == 0:
        return AUDIO_ONSETS.empty_table()

    centroid = librosa.feature.spectral_centroid(S=np.sqrt(spec), sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(spec))[0]
    win = int(0.01 * sr)
    rms_db = []
    for c in onset_samples:
        seg = y[max(0, c - win) : c + win]
        rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
        rms_db.append(20 * np.log10(rms + 1e-9))

    return pa.table(
        {
            "t_s": pa.array(times + time_offset_s, pa.float64()),
            "strength": pa.array(z[peaks], pa.float32()),
            "rms_db": pa.array(rms_db, pa.float32()),
            "centroid_hz": pa.array(centroid[peaks], pa.float32()),
            "flatness": pa.array(flatness[peaks], pa.float32()),
        },
        schema=AUDIO_ONSETS,
    )
