# M0 spikes: video decode, model throughput, browser playback

Date: 2026-10-05 · Machine: RTX A5000 Laptop GPU (16 GB, 110 W max, on AC), Windows 11,
torch 2.11.0+cu128, ffmpeg 7.1.1 (gyan full build), PyNvVideoCodec 2.2.3, PyAV 19.0.1.

Test footage: the user's own practice recordings. These are phone 4K HEVC Main at about
56 Mb/s, AAC 48 kHz, with a 180° display-matrix rotation. One file is **variable frame rate**
(nominal 60, average 59.22 fps); the other is 60.04 fps.

Reproduce:

```bash
uv run python scripts/spikes/decode_bench.py VIDEO --start 300 --frames 600
uv run python scripts/spikes/model_bench.py VIDEO
```

## (a) 4K decode → upright RGB tensor on the GPU

Each backend has to deliver a CUDA `uint8` tensor `(3, 2160, 3840)` with the rotation
applied. Results are steady-state throughput over 600 frames, measured with no other GPU work.

| Backend | fps | Notes |
|---|---:|---|
| **PyNvVideoCodec `SimpleDecoder` (NVDEC, RGBP, DLPack → torch)** | **305** | Zero-copy. Rotation is applied with `torch.flip` (cheap). |
| PyAV 19 with `HWAccel("cuda")` | 81 | Frames are downloaded to the CPU and converted there |
| PyAV CPU decode | 29 | About half of realtime |
| ffmpeg subprocess → rawvideo pipe | 6 | Windows pipes can't move 24 MB frames fast enough |
| ffmpeg NVDEC → null (reference ceiling) | ≈250 | `-hwaccel cuda -hwaccel_output_format cuda -f null` |

When run while an NVENC proxy encode was also active, PyNvVideoCodec dropped to 126 fps
because NVDEC is shared. **The worker must not run the proxy encode and a GPU decode pass at
the same time.** It already runs one stage at a time.

**Correctness check.** PyNvVideoCodec and PyAV return the same upright frame (mean absolute
difference of 1.5 levels, against 37 levels for a flipped frame). PyNvVideoCodec's YUV→RGB
conversion comes out about 1.5 levels brighter (slightly different matrix/range).

**Decision.** `FrameSource` uses PyNvVideoCodec as the primary backend and PyAV CPU as the
fallback. Labels and inference must come from the **same** backend so the color conversion
matches. torchcodec was not evaluated: PyNvVideoCodec already beats the requirement, and
torchcodec on Windows needs shared FFmpeg DLLs that this machine doesn't have.

Rotation: NVDEC returns coded (unrotated) frames, so `FrameSource` must apply `rotation_cw`
itself. ffmpeg-based paths auto-rotate unless told otherwise.

## Proxy encode (implemented in `io/proxy.py`)

| Path | fps (4K60 HEVC → 720p H.264) | Output rotation flag |
|---|---:|---|
| `-hwaccel cuda` (frames downloaded) → CPU scale → NVENC | ≈120 | none (auto-rotated) |
| **NVDEC → `scale_cuda` → hwdownload → hflip,vflip → NVENC, `-display_rotation 0`** | **≈250** | none |
| same GPU path without `-display_rotation 0` | ≈250 | **leaks −180°**, so the browser would rotate again |

Measured end to end in the pipeline: a 28:24 session took **307 s for the proxy** (≈5.5×
realtime), 2.2 s for ingest (probe + thumbnail + FLAC), and 16.7 s for audio onsets. The proxy
keeps source timestamps (`-fps_mode passthrough`), so proxy time equals session time even for
the VFR file.

## (b) Model throughput (FP16, channels-last, random weights)

The networks are built from their architecture definitions, so compute cost is identical to
trained weights and nothing was downloaded. The numbers are network forward passes only.

| Model | Input (H×W) | Batch | img/s |
|---|---|---:|---:|
| YOLO11-s | 736×1280 | 8 | 220 |
| YOLO11-m | 736×1280 | 8 | 104 |
| YOLO11-l | 736×1280 | 8 | 80 |
| YOLO11-x | 736×1280 | 8 | 47 |
| YOLO11-s | 1088×1920 | 4 | 97 |
| YOLO11-s | 1440×2560 | 4 | 53 |
| TrackNet-size U-Net (11.3 M, 12-ch in, full-res heatmap) | 288×512 | 8 | 126 |
| same | 576×1024 | 8 | 32 (6.6 GB) |
| same | 720×1280 | 8 | 20 (10.3 GB) |
| **Slim U-Net (width 32, stride-2 stem, ½-res heatmap)** | 720×1280 | 8 | **191** |
| same | 448×1280 (court-ROI band) | 8 | 305 |
| same | 1088×1920 | 8 | 84 |

End to end (NVDEC → GPU resize → YOLO11-l @1280 + TrackNet-size @1280×720 on every frame):
**15 fps** with 7.8 GB peak VRAM, which is **≈4 h per hour of footage** for pass 1 alone.

### Consequences for the plan

1. **A TrackNet-size network at the resolution the far court needs is too slow.** PLAN.md §6
   assumed about 100 fps. M3 should use a **slim TrackNet-style U-Net** at about 1280 px wide
   (≈190 fps). Train it on our labels, and optionally distil from a pretrained TrackNetV3
   teacher on downscaled frames. Cropping to the court ROI band makes it faster still.
2. **Person detection does not need every frame or the largest model.** YOLO11-m at 1280 on
   every 2nd frame (30 Hz) with tracker interpolation costs about 0.3 ms/frame of video.
   YOLO11-l is affordable at 30 Hz if recall on the far player needs it.
3. Revised pass-1 estimate: decode (305) + slim ball net (190) + YOLO11-m @30 Hz (208 effective)
   ≈ **75–90 fps** combined, or **≈45 min per hour of footage**. TensorRT/FP8 remain
   optimizations for M7/M11.
4. Gate the expensive passes with cheap signals. In practice sessions a large share of the
   time is ball collection. Frame differencing plus the audio-onset density can skip dead time.

## (c) Browser playback (Chromium-based app pane)

* The proxy (H.264 High, 720p, faststart, GOP 30) loads and plays: `readyState 4`, 1280×720,
  and a duration of 1703.72 s that matches the source.
* Seeking uses HTTP range requests from Flask `send_file(conditional=True)` and is verified by
  a test (`206 Partial Content`).
* Sync is verified in the real app:
  * Seeking to 600 s updates the readout and timeline cursor to `10:00.000`.
  * Playback updates both.
  * `L` jumps +5 s, `→` steps exactly one frame (16.9 ms at 59.2 fps), and clicking the
    timeline seeks to the clicked time.
* `canPlayType('video/mp4; codecs="hvc1…"')` returns `"probably"` in this Chromium build.
  HEVC playback depends on the browser and OS, though, so the H.264 proxy stays the default.
* While the view was hidden, `requestAnimationFrame` was throttled and the readout lagged by
  about 1 s. A `timeupdate` listener was added as a fallback.

## Observations about the footage (affect M1–M3)

* The camera sits fairly **low behind one baseline corner**, so the far half of the court is
  heavily foreshortened. Landing precision on the far side will be the weakest point. The
  recording guide (raise the camera) matters.
* The sessions run into **darkness** (one file is nearly black by about 40 min). M1/M3 should
  flag unusable low-light spans instead of producing garbage.
* Audio onsets are plentiful (3,425 in 28 min: hits, bounces, footsteps, balls into the
  basket). M3 has to classify them, using the planned `centroid_hz`/`flatness`/strength
  features plus proximity to ball-trajectory events.
* Onset timing: plain spectral flux with a centered 1024-sample window fired about 17 ms
  early, which is a whole frame. A time-domain leading-edge refinement brings the error below
  0.1 ms on synthetic clicks (covered by a test).
