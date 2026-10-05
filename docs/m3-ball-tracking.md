# M3: ball detection, trajectory, events, labeling

Date: 2026-10-05. Code: `src/swingvision/ball/` (detectors, schedules, linking, events),
`src/swingvision/training/` (labels, labeling helpers, benchmark, training, evaluation),
stages in `src/swingvision/pipeline/stages/ball.py` and `players.py` (`pass1_detect`), the
Labeling page in `src/swingvision/app/pages/labeling.py`.

## Results

Ground truth: 9 clips (2.8 s each, every frame labeled) held out from training, spread over
both sessions (daylight, dusk, near- and far-court play, a pre-serve dribble, net shots, a
clip with no ball in play): **934 frames with the ball, 467 without, 104 occluded; 18
bounces, 6 hits, 2 net contacts.** Scored by `sv bench ball` / `sv eval`.

| Exit criterion | Target | `unet:ball-v2`, every frame |
|---|---|---|
| Ball F1, near half | ≥ 0.85 | **0.908** (strict tolerance: 0.887) |
| Ball F1, far half | ≥ 0.75 | **0.951** (strict: 0.950) |
| Bounce F1 (±2 frames) | ≥ 0.85 | **0.941** (16 of 18, no false bounces) |
| Median bounce position error, near | ≤ 15 cm | **2.4 cm** (5 bounces) |
| Median bounce position error, far | ≤ 35 cm | **4.3 cm** (11 bounces) |

**All exit criteria are met.** Precision 0.96, recall 0.92, median position error 0.7 px.

Notes on the numbers:

* **Tolerance.** A detection counts within `max(4 px, 0.4 × ball diameter)` (a near ball is
  20-40 px wide). The primary metric also allows `0.25 ×` the ball's per-frame motion: a
  ball moving 60 px between frames is a 60 px streak whose center (the label) is uncertain
  along it. The *strict* column drops that term; it also passes.
* **Bounce position error** compares the detected bounce with the bounce derived from the
  labels the same way (fits on both sides of the contact, ray at ball height), through the
  same calibration. It measures detection and refinement, not calibration. The calibration
  adds (1 px of line RMS propagated to the ground at the test bounces): ≤ 2 cm near,
  4-7 cm in the far court, 11-13 cm behind the far baseline, still inside the targets.
* **Bias risk.** Five event-rule fixes (bounce vs net ordering, kink gating) were found
  while looking at test-clip failures; they are general (a bounce kicks a falling ball
  upwards; a net contact nearly stops it), but the event numbers may be slightly optimistic.
  On the 6 event-labeled *training* clips the same rules find 9 of 10 events and add 3
  bounces that aren't there (a streak leaving the frame edge, a ball rolling into the back
  fence, one in an unlabeled dark stretch): expect bounce precision nearer 0.85 than 1.0. One ground-truth bounce the labeling had
  missed (Oct 1 frame 3651, plain in the labeled positions) was added.

### Detector benchmark

`sv bench ball` on the same clips (cost: detector GPU time per hour of footage on the RTX
A5000 laptop; *refine* is the share of the clips' time re-run at the full rate).

| Detector | Schedule | F1 | near | far | strict near / far | raw top-1 | bounce F1 | hit F1 | GPU min/h | refine |
|---|---|---|---|---|---|---|---|---|---|---|
| `unet:ball-v2` | every frame | **0.940** | 0.908 | 0.951 | 0.887 / 0.950 | 0.849 | 0.941 | 0.800 | 46 | 0.00 |
| `unet:ball-v2` | 15Hz sweep + windows | 0.914 | 0.902 | 0.918 | 0.881 / 0.916 | 0.798 | 0.875 | 0.800 | 43 | 0.65 |
| `unet:ball-v2` | 30Hz sweep + windows | 0.936 | 0.908 | 0.946 | 0.887 / 0.944 | 0.825 | 0.941 | 0.800 | 56 | 0.72 |
| `unet:ball-v1` | every frame | 0.869 | 0.679 | 0.930 | 0.647 / 0.929 | 0.750 | 0.875 | 0.500 | 46 | 0.00 |
| `motion` | every frame | 0.700 | 0.428 | 0.793 | 0.424 / 0.793 | 0.321 | 0.667 | 0.316 | 206 | 0.00 |
| `motion` | 15Hz sweep + windows | 0.691 | 0.461 | 0.771 | 0.457 / 0.771 | 0.319 | 0.690 | 0.273 | 210 | 0.75 |
| `motion@1920` | every frame | 0.646 | 0.404 | 0.734 | 0.400 / 0.734 | 0.357 | 0.643 | 0.348 | 48 | 0.00 |
| `yolo11m@1920` | every frame | 0.402 | 0.502 | 0.348 | 0.498 / 0.339 | 0.182 | 0.452 | 0.600 | 179 | 0.00 |

* **The trained U-Net wins on accuracy and cost.** The classical motion detector needs no
  labels but confuses rackets, limbs and shoes with balls near the player (near F1 0.43) and
  costs 3.4 GPU-hours per footage hour at full resolution; at 1920 px it loses the far ball.
  YOLO11m's COCO "sports ball" class finds near balls only.
* **v1 → v2**: v1 (1,522 labeled frames) failed on a pre-serve dribble right in front of the
  camera (near F1 0.68). v2 added six dribble clips labeled from color + motion and checked by
  eye, negatives (clips without a ball in play), hard negatives at v1's false detections
  (racket rims), and 899 pseudo-labels from confident v1 tracklets: 3,377 frames, near F1
  0.91.
* **Schedules.** A 15 Hz sweep with full-rate windows costs only 7% less here, because these
  clips are all play and the windows (around events, strong onsets, track gaps) cover 65% of
  their time; it also loses bounces whose window was missed. 30 Hz costs *more* than every
  frame: a three-frame model reads frames t±2, so a sweep decodes and preprocesses nearly
  every frame anyway, and the windows are processed again. **Default: every frame.** The
  sweep saves time only in long sessions with much dead time (ball collection); see the
  full-session numbers below.
* The linker matters most for precision: raw best-candidate-per-frame F1 is 0.85 vs 0.94
  linked.

### Labels

| Set | Clips | Frames with ball | No ball | Occluded | Events | How |
|---|---|---|---|---|---|---|
| Test (ground truth) | 9 | 934 | 467 | 104 | 26 | every frame checked by eye; keyframes + snapping + interpolation |
| Train, hand-checked | 12 | ≈1,100 | 850 | 159 | 10 | same, plus color + motion labels for dribbles |
| Train, auto-cleaned | 21 | ≈530 | – | – | – | detector + tracker, snapped, color-checked, outside the player box |
| Train, pseudo-labels | 24 | 899 | – | – | – | confident v1 tracklets (`sv labels pseudo`) |


## Pipeline

| Stage | GPU | What it does | Output |
|---|---|---|---|
| `pass1_detect` | ✓ | One decode pass: person boxes at 15 Hz (M2) **and the ball detector** on every frame, or at the sweep rate | `ball/sweep.parquet` (+ frames looked at) |
| `ball_refine` | ✓ | With a sweep rate: links the sweep, finds moments (events, strong audio onsets, track gaps, run ends) and re-runs the detector at the full frame rate in windows around them (−0.25 … +0.35 s) | `ball/refine.parquet` |
| `ball_track` | – | Links sweep + refine candidates into one ball track, fills short gaps | `ball/track.parquet` |
| `events` | – | Hits, bounces, net contacts with sub-frame contact time, bounce ground position, audio match | `events.parquet`, `ball/kinks.parquet` |

`pass1_detect` (v2) keeps two fingerprints: changing only the ball detector decodes the video
again but reuses the person boxes (`pass1/persons.json`), and sessions processed by M2 keep
theirs.

## Detectors (`ball/detectors/`)

Every detector implements `BallDetector.detect(frames, targets)` and returns candidates
`(x, y, score)` in full-resolution pixels; a spec string picks one
(`name[:weights][@input_px]`).

* **`motion`**: classical, no training. Frame differencing against frames ±3 apart
  (adjacent frames lose a slow ball's body), a multi-scale center-surround blob filter, and a
  connected-blob centroid (motion-blur streaks are centered, not their leading edge). Camera
  shake is removed first: a global translation per neighbor frame (phase correlation +
  Lucas-Kanade refinement), applied when above ¼ px. Without it, wind on the tripod turned
  every court line into a "ball".
* **`unet:<weights>`**: slim TrackNet-style U-Net (3 frames t−2, t, t+2 → heatmap at half the
  input resolution; width 32, 1/16 bottleneck), trained on the user's labels with a CenterNet
  focal loss, flips, time reversal and dusk-like brightness/gamma augmentation.
* **`yolo11m` (s/l)**: a COCO YOLO's *sports ball* class, zero-shot.

## Linking (`ball/trajectory.py`)

1. **Tracklets** from candidate triplets with near-constant velocity, grown both ways with a
   quadratic predictor and a gate that widens with speed and time.
2. **Selection** of the active ball per frame by a Viterbi pass over tracklets: confidence ×
   smoothness × "is it flying" (90th-percentile image speed of the tracklet; a slow apex
   doesn't lose the ball, a ball rolling to the fence earns nothing), hand-offs cheaper than
   jumps. Candidates inside the tracked player's box count 40% (rackets, limbs).
3. **Cleaning**: points off a local quadratic are dropped; gaps ≤ 0.15 s are filled.

## Events (`ball/events.py`)

A **kink** needs all three: straight lines on each side (4 points) explain the window better
than one quadratic (rules out smooth apexes), a velocity change ≥ 6 standard errors, and a
relative velocity change ≥ 25%. Kinks are classified by rules (or the learned classifier,
`sv train events`): near the player's box → hit (or a bounce at the feet), inside the net's
image quadrilateral and slowing → net, elsewhere with the image path kicked upwards → bounce.
The contact is refined where the fits on both sides meet (sub-frame), a bounce's ground
point is that pixel's ray at ball-radius height. Hits are matched with audio onsets after
compensating the sound delay (distance / 343 m/s, ≈70 ms from the far baseline).

## Labeling (`training/`, Labeling page)

Clips are short frame ranges (3 s by default) labeled at every frame: ball position,
`none` (no ball in play in view), or `occluded` (not scored); events at their frames.
Creating a clip decodes its frames into a JPEG cache and pre-fills labels from the detector +
tracker; labeling is checking and correcting. Clicks snap to the nearest moving blob.
Keyframe interpolation fills between labels along a quadratic, never across an event.

Label policy used for the ground truth:

* the **ball in play** is labeled; balls lying or rolling elsewhere are `none`, a ball held in
  the hand is `occluded`;
* motion-blurred balls are labeled at the **center of the streak** (the ball at
  mid-exposure);
* a ball cut by the frame edge or hidden by the net tape or the player is `occluded`.

## Whole sessions

Oct 1 (28:24, self-feed into dusk), processed by the pipeline with `unet:ball-v2` on every
frame (RTX A5000 laptop):

| | Every frame | 15 Hz sweep + windows |
|---|---|---|
| Ball detection (GPU) | 1,433 s (0.84× realtime) on 100,882 frames | 635 s sweep + 845 s windows = 1,480 s (674 windows, 52% of the video) |
| Linking, events (CPU) | 40 s, 9 s | 28 s, 7 s |
| Ball in the track | 38,586 frames (36,323 detected) | 37,002 frames (29,120 detected) |
| Hits / bounces / net | 424 / 775 / 9 | 413 / 736 / 10; 93% of the full-rate hits and bounces matched |

Even over a whole session with ball collection and dead time, the sweep saves nothing: the
windows (around every strong onset and event) cover half of the video, and they decode and
preprocess their frames a second time. **Every frame is the default.**

Ignoring frames where pass 1's view check says the camera wasn't on the court (camera being
set up, picked up) removes events like a "bounce" in the sky at 0:08 (spot check of 24 random
bounces: 19 plausible, the self-feed drop bounce at the baseline is the most common; 4 too
dark to judge after 23 min; 1 at 0:08 during camera setup). With that filter: 424 hits,
760 bounces, 8 net contacts.

Pass 1 for one footage hour is therefore ≈50 min of ball detection plus ≈31 min of person
detection (M2) when both run, ≈1.4× realtime.

## App

* **Labeling** page (new): sessions and clips, a frame viewer at several zoom levels with the
  label, the prediction and neighboring labels, keyboard shortcuts, events, train/test,
  suggestions of where to label next.
* **Session** page: *Ball* switch draws the ball and a 0.4 s trail on the video; hits
  (orange) and bounces (blue) on the timeline; bounces of the last 3 s on the court map; a
  Ball card (time the ball was seen, hits, bounces, net).
* **Settings**: ball detector (automatic = newest trained model) and frame-rate schedule.
* CLI: `sv labels list|pseudo`, `sv bench ball`, `sv train ball|events`, `sv eval`.

## Limitations and follow-ups

* **Ground truth is small**: 9 clips, 18 bounces. The numbers are encouraging, not precise;
  label more test clips as more footage (especially ball-machine sessions and matches)
  comes in.
* **Events are rules.** `sv train events` trains a gradient-boosting classifier on kink
  features, but with 13 labeled kinks it isn't activated (threshold 200). Hits during a
  pre-serve dribble (racket taps) and the last tiny bounces of a dying ball are missed; a
  ball touching the net tape while hidden behind it isn't seen.
* Hits are "me" or "machine" only (single player); the hitter's side and the opponent come
  with match mode. Serve detection is pose-based (M6).
* Machine feeds (a trajectory starting at the ball machine) are tested on synthetic data
  only: there is no ball-machine footage yet.
* The full-rate U-Net is the throughput bottleneck now (≈46 GPU-min per footage hour);
  TensorRT/FP8 (M11) and a smaller crop when the ball's position is known are the next
  steps.

