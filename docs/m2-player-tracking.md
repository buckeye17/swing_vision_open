# M2: player tracking, profiles, movement

Date: 2026-10-05. Code: `src/swingvision/players/` (detection, tracking, movement),
stages in `src/swingvision/pipeline/stages/players.py`, dense decoding in
`src/swingvision/io/frames.py`, pages in `src/swingvision/app/pages/` (session review,
profiles).

## Pipeline

| Stage | GPU | What it does | Output |
|---|---|---|---|
| `pass1_detect` | ✓ | One decode pass (NVDEC, PyAV fallback). Every frame of a 15 Hz sample is cropped to the court region and run through YOLO11m at 1920 px; per frame it also records brightness and how well the picture matches the court background. | `players/detections.parquet`, `pass1/frames.parquet` |
| `players_track` | – | Court positions, tracklets, static objects, the "me" chain | `players/tracks.parquet`, `players/identity.json` |
| `movement` | – | Gap bridging, Kalman/RTS smoothing, speeds, stats | `players/movement.parquet` |

`pass1_detect` runs after the calibration gate because it crops to the court, but only the
coarse crop rectangle (snapped to 64 px) is in its fingerprint. Adjusting the calibration
reruns `players_track` and `movement` (about 3 s for a 28-minute session), not the GPU pass.
The stage framework gained an `after` (ordering-only) dependency for this.

## Detection

* **Frames.** `FrameSource` decodes a time range to upright RGB tensors. Frame numbers come
  from the container's packet table (demuxing a 12 GB file takes ≈3 s), not from the
  decoder's own counter: NVDEC ignores MP4 edit lists and emits pre-roll frames with negative
  timestamps, so its indices are offset on files with B-frames (found by a test on an x264
  clip). NVDEC and PyAV now agree frame for frame.
* **Model choice.** 16 random frames from each session, checked by eye. Every model finds
  a fully visible player; the hard case is Oct 4, where the far baseline is at the frame's
  top edge and only the player's legs show (5 of the frames). Extra boxes are mostly people
  at the playground beyond the far fence, which the court ROI drops anyway.

  | Model @ input (long side) | far player, legs only | boxes in all 32 frames | speed |
  |---|---|---|---|
  | YOLO11m @ 1280 | 2 of 5 | 30 | ≈1.5× faster end to end |
  | **YOLO11m @ 1920** | **5 of 5** | 40 | 46 img/s |
  | YOLO11l @ 1920 | 5 of 5 | 40 | ≈0.77× (M0 spike ratio) |
  | YOLO26m @ 1920 | 5 of 5 | 53 | not measured |

* **Cost.** The detector, not decoding, is the bottleneck: YOLO11m at 1024×1920 runs 46
  images/s on this laptop GPU (power-limited to ≈90 W, 1.1 GHz). At 30 Hz that would be
  ≈40 min per footage hour for players alone, so the default is **15 Hz** (a sprinting
  player moves ≤ 0.6 m between samples). Measured end to end: **890 s for the 28-minute
  session (0.52× realtime)**. Rate, input size and model are in Settings.
* **Darkness.** The person detector is far more robust than the court detector: on the
  Oct 1 session the player is tracked in 100% of frames down to a median frame luma of ≈4.5
  (of 255), and lost below ≈4. "Too dark" for players is therefore luma < 4 (the calibration
  uses 25).
* **View check.** Each sampled frame is correlated (gray, 1/16 scale) with the median image
  of its drift window. Settled play scores 0.8–0.98 (99% of frames ≥ 0.87 on Oct 1, ≥ 0.79
  even in the darkest usable minutes); a camera in hand scores −0.3–0.45 and one still being
  aimed ≈0.5. Detections in frames below 0.6 are ignored. Without this, the first seconds of
  a recording produced a "player" on a hand, and a camera being aimed one on playground
  equipment.

## Tracking

1. **Court position.** Foot point (box bottom center) → ground through the camera of that
   time window. Each foot point gets a σ from the image→ground Jacobian (2% of box height as
   pixel jitter): a few cm near the camera, 0.2–0.3 m along the court at the far baseline.
   The box top gives the person's height. Boxes whose feet are below the frame (the player
   walking past the camera) are placed under the head ray at the profile's height (1.75 m
   by default).
2. **ROI.** Court + 6 m behind the baselines + 3.5 m beside (neighboring courts usually start
   ≈3.7 m beside; the plan's 4 m would reach into them).
3. **Tracklets.** Per-frame Hungarian assignment on court distance, gated by running speed
   plus σ, and by **physical height** (a 1.8 m person can't continue a 0.9 m object's track).
   Without the height gate, a player walking past a ball machine swapped tracks with it.
   Boxes mostly inside a more confident box (partial-body duplicates) are dropped.
4. **Static objects.** Clusters of motionless (< 0.35 m spread), short (< 1.3 m) tracklets
   adding up to ≥ 30 s: a ball machine, a bag, a basket on a stand. They never become "me".
5. **"Me" chain.** A dynamic program over tracklets ordered by end time. A chain gains
   `conf − 0.15` per detection; links must be physically possible (≤ 7 m/s across gaps of
   up to 4 s); jumping elsewhere without continuity costs a restart penalty. A link may
   continue into a tracklet that started *before* its predecessor ended, if both agree on
   where the player was at the junction. On real footage the player is often covered by two
   overlapping tracklets for a second (a duplicate box starts a second track that then takes
   over), and the first version of the DP lost whole minutes this way.

**Ball machine.** In ball-machine sessions the most persistent static object is adopted as
the machine (`identity.json`), and the session page can place it by clicking the court map.
Recognizing the machine from the origin of feed trajectories needs ball tracking (M3). No
ball-machine footage exists yet, so this path is covered by synthetic tests only.

## Movement

Positions are smoothed per continuous run with a constant-velocity Kalman filter + RTS
smoother; each measurement's noise is its ground σ, and implausible jumps are down-weighted.
Gaps up to 1 s are bridged (`source = interp`), longer gaps start a new run. Distance only
accumulates above 0.6 m/s, top speed is the best 0.5 s average.

Tuned on synthetic trajectories (`tests/test_movement.py`): shuttle runs with far-court noise
(σ 0.25 m along the court) give distance within 2% and top speed within 6%; a player
standing still for 2 minutes with σ 0.3 m adds < 1 m (summing raw positions would add
≈2 km).

## Results on the user's footage

Checked with `scripts/m2_eval_tracking.py`. There are no hit labels before M3, so **hitting
time** is approximated as ±0.5 s around strong audio onsets (strength ≥ 8: racket impacts
and bounces, plus some ball-collection noise). Only *usable* frames count (luma ≥ 4, view
check ≥ 0.6). The script also draws contact sheets of random hitting moments for a visual
precision check.

| | Oct 1 (28:24, self-feed, into dusk) | Oct 4 (50:58, far baseline at the top edge) |
|---|---|---|
| Detection pass | 883 s (0.52× realtime) | 1,651 s (0.54× realtime) |
| `players_track` + `movement` | 1.7 s + 1.4 s | 2.1 s + 1.9 s |
| Sampled frames, usable | 25,556, 23,314 | 45,882, 34,909 |
| Hitting-time frames tracked | **95.0%** of 13,116 | 85.2% of 22,395 |
| … with the player in the picture | **99.94%** (0.5 s missed, at dusk) | not labeled (see below) |
| Audit: box on the player | **48 of 48** | 18 of 18 tracked samples |
| Spectators / neighbors / objects as "me" | **none** | none |
| Distance, top speed, avg moving speed | 696 m, 2.75 m/s, 1.07 m/s | 736 m, 4.16 m/s, 1.08 m/s |

**Oct 1 misses**, all checked by eye: the end of camera setup (14–19 s, player not in the
picture), two walks off camera to fetch balls (625–645 s and 1279–1306 s; the player leaves
past the left edge and returns past the camera), and 0.5 s in near-darkness. Excluding the
three off-camera spans (686 frames) leaves 7 missed frames of 12,430.

**Oct 4 misses** are framing, not tracking: on all 12 frames sampled from the longest missed
spans, and on all 6 missed audit samples, the player is outside the picture, mostly behind
the far baseline above the top edge. Tracked frames there are often just the player's legs at
the top edge, which 1920 px input makes possible.

**Precision.** On Oct 1 the "me" boxes were on the player in all 48 random hitting moments,
including dusk frames at luma ≈5 and a frame where the player walks away from the camera with
the feet below the frame (placed from the head). People at the playground beyond the far
fence are detected but projected outside the court area and ignored. The only "me" tracklets
with implausible heights are 1–3-detection fragments of the player entering or leaving
through a frame edge.

**Speeds.** Smoothed speeds match the raw detections: the fastest 1-s movements on Oct 1 are
2.3–2.4 m/s from medians of raw foot points, and 1.9–2.2 m/s smoothed over the same second
(this was a basket-feeding session with little running).

**Exit criterion** (> 98% of hitting time tracked in a 30-minute practice session, no
spectators or neighboring-court players): **met on Oct 1** with 99.94% of hitting time while
the player is in the picture (95.0% if moments when the player is off camera count as
hitting time) and no one else ever tracked as the player.

## App

* **Session page:** the player's box drawn on the video (dashed while bridged), a live court
  minimap with a 3 s trail and position/speed readout, a player-speed row under the audio
  onsets on the timeline, and a Movement card (distance, tracked time and coverage, top and
  average speed, time-on-court heatmap with a "fold both ends" option).
* **Profiles page:** name, handedness, one- or two-handed backhand, height. New session picks
  the player (preselected when there's one profile); the session page can change it. The
  profile height is used to place players whose feet are out of frame, and later scales 3D
  pose (M6). Appearance enrollment arrives with match mode.
* **Library:** distance moved per session. **Settings:** detector, rate, input size, ROI width.
* CLI: `sv profiles list|add`, `sv models list|download`, `sv create --profile`.

## Limitations and follow-ups

* When the far baseline sits at the top edge of the frame (Oct 4), a player standing behind
  it is out of the picture and can't be tracked. The recording guide's "frame the area
  behind the far baseline" matters.
* Throughput is 0.5× realtime for detection alone. Batching the crop with M3's ball network
  in the same decode pass and TensorRT (M11) are the planned speedups.
* Static-object rules and ball-machine adoption need real ball-machine footage to confirm.
