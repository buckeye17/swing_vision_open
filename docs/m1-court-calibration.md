# M1: court calibration and camera model

Date: 2026-10-05. Code: `src/swingvision/court/`, stages in
`src/swingvision/pipeline/stages/court.py`, editor in `src/swingvision/app/pages/calibrate.py`.

## What the user's footage looks like

The two sessions recorded so far (4K HEVC phone video, 28 and 51 minutes) are the design
targets. Things a broadcast-trained keypoint CNN wouldn't handle well:

* The camera is **behind one baseline, about 3.3 m high and 6.4 m back**. The far court is
  strongly foreshortened, and in the Oct 4 session the far baseline sits at the very top edge of
  the frame.
* One session uses the **ultrawide lens** (87° horizontal FOV), the other the main lens (59°).
* The court has **teal pickleball lines** painted inside it and a **neighboring court** on the
  right.
* The phone's video is **not a plain pinhole image**: with the principal point fixed at the
  image center, the best fit to the Oct 4 session is 4.2 px RMS; freeing it (with a prior)
  gives 1.5 px. In-camera stabilization and lens correction are the likely cause.
* **The camera moved** in both sessions during the first 10–15 minutes (22–25 px at first,
  then a slow sag of a few pixels), and both run into **darkness** at the end.

## Approach

Classical, so it needs no training data (a CNN fine-tuned on confirmed calibrations remains an
option if a view ever defeats it):

1. **Background images.** The video is split into ≤ 12 windows (default 5 min). Five frames per
   window are decoded at full resolution (`io/frames.py`, ffmpeg seek, ≈0.6 s per 4K frame);
   frames darker than mean luma 25 are skipped; the per-pixel median removes players and balls.
2. **Line response.** White top-hat (61 px at 4K) of `min(R, G, B)`: paint is bright in every
   channel, the green/red surface and the teal pickleball lines are not. For line finding,
   blobs wider than any painted line (sky, water, houses) are removed from the thresholded
   response.
3. **Hypotheses.** Probabilistic Hough at 1920 px, segments merged into long lines and split
   into an *across* (baseline-like) and an *along* (sideline-like) family. Every pair × pair of
   image lines is matched to every ordered pair of model lines (≈10⁵ batched 4-point
   homographies). Each is scored by the image length of projected court lines on line pixels
   minus 0.6 × the length off them.
4. **Refinement.** For the six best hypotheses: decompose into a camera (focal length from the
   homography's orthonormality constraints), then sample the response perpendicular to every
   projected court line, take the sub-pixel center of the ridge (half-max centroid; profiles
   with a second strong ridge or no fall-off are rejected), and fit the camera to these samples
   with a robust loss. The search window shrinks over iterations (30 → 7 px plus the on-screen
   line width) and outliers beyond 3.5 robust σ are dropped. The best hypothesis by soft inlier
   count gets a final fit with the **full model**: pinhole, division-model distortion (`k1`,
   `k2`), principal point with a prior, and the **net tape** (post tops 1.07 m, center strap
   0.914 m) as non-planar evidence.
5. **Drift check / piecewise calibration.** The camera is pose-refined on every window. If the
   windows disagree, the session is re-calibrated on the median of the largest agreeing group
   (the position held longest). Windows whose pose differs from the session camera keep their
   own camera; `court.calibration.camera_at(cal, t)` returns it for downstream stages.
6. **Uncertainty.** `court.homography.ground_sigma` propagates image noise through the
   image→ground Jacobian (σ across, σ along, and the error-ellipse major axis per court point).

The reported **line RMS** is measured the same way everywhere (auto result, editor, Settings
threshold): line centers found within a few pixels of the projected lines, after the same
3.5-σ outlier rule as the refinement, so occluders and stray marks don't count.

## Results on the user's footage

| | Oct 1 session (28 min, ultrawide) | Oct 4 session (51 min, main lens) |
|---|---|---|
| **Line RMS, session camera** | **0.64 px** | **1.20 px** |
| Lines found (of visible, unoccluded) | 91% (1,705 samples) | 93% (2,403 samples) |
| Net-tape samples | 132 | 201 |
| Camera | 3.26 m high, 6.39 m behind, 87.5° HFOV | 3.26 m high, 6.43 m behind, 58.6° HFOV |
| Lens k1 / k2, principal-point offset | +0.044 / −0.048, (0, −130) px | +0.029 / −0.008, (4, −213) px |
| Drift windows (5 min) | 2 moved (25 and 8 px), 2 ok, 2 dark | 3 moved (22, 20, 4 px), 5 ok, 3 dark |
| Line RMS, per-window cameras | 0.66–0.91 px | 1.18–1.24 px |

Exit criterion (< 2 px) met on both sessions and on every usable time window.

Both sessions were also checked visually: the overlay sits on the painted lines over the full
court, on each drift window's own image (Calibrate page → *Image*), and over the playing video
(session page → *Court overlay*). The camera positions agree between the two sessions
(3.26 m high, 6.4 m behind the baseline), as expected for the same tripod spot.

`court_auto` takes about 45 s (28 min video) and 85 s (51 min video) per session (mostly decoding 4K frames).

## Synthetic tests

`tests/synth_court.py` renders a court (with a neighboring court, pickleball lines, a net and a
"player") through a known camera at 2× supersampling. Detection recovers the camera for a wide
high camera, a narrow lens with an off-center principal point (like the Oct 4 session), and a
low corner camera with barrel distortion: < 1 px keypoint error at 1920×1080, focal length
within 1%, camera position within 10 cm. The stage tests run the same through an H.264 encode,
the median background, the drift check and the review gate.

## Workflow

* `court_auto` runs right after `ingest`, so the Calibrate page is usable while the proxy
  encodes. The `camera` stage is the gate: it uses `court/user.json` if the user confirmed a
  calibration, otherwise the auto result only if Settings allows auto-accept and the fit
  passes (line RMS threshold, ≥ 30% of lines found, no camera movement). Otherwise the job
  stops with `needs_action` and Jobs shows **Review calibration**.
* The editor's handles are the 14 court keypoints and 3 net points. A dragged point is pinned;
  the camera is re-solved (pose and, with ≥ 4 points, focal length; distortion stays) with a
  weak pull toward the previous camera elsewhere. **Snap to lines** refits the full model to
  the background. **Confirm** writes `court/user.json` and re-queues the job; the `camera`
  stage then re-derives the per-window cameras from the confirmed one.
* The `camera` stage's fingerprint depends only on the chosen camera, so re-confirming an
  unchanged calibration invalidates nothing downstream.

## Limitations and follow-ups

* The principal point and tilt trade off against each other on a planar scene; the net tape
  pins them down, but parameters such as focal length can differ by a few percent between
  equally good fits. Ground projections are unaffected; heights (ball flight, M4) should be
  validated against a known reference (ball-machine speed, radar).
* Real courts deviate from ITF dimensions by a few centimeters, which limits how low the line
  RMS can go on near lines.
* Windows that are too dark are skipped; analysis stages should not trust them either.
