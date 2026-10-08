# Spike: do bigger, slower models buy accuracy?

Date: 2026-10-07 · Machine: RTX A5000 Laptop GPU (16 GB), Windows 11, transformers 5.18.

Question: the pipeline runs small models (a 0.9 M-parameter ball U-Net, ViTPose-B,
MotionBERT-Lite, YOLO11m). Would bigger ones measurably improve the outputs, and at what
cost? Everything was scored with the existing harnesses (`sv bench ball`, `sv swings eval`)
on the M3/M6 ground truth, in a copy of the scratch output root, so the sessions and
labels were not touched. The baselines reproduced exactly before any change.

**Answer: no, not with this footage and these labels.** Bigger models make the raw
outputs cleaner, but none of the measured pipeline results moved beyond the spread you get
from retraining the same model with another seed. The limits are data
(labels, framing, footage variety) and the rules downstream, not model size.

## Decision (2026-10-08)

**ViTPose+-H is now the default 2D pose model**, chosen for the steadiest keypoints (lowest
frame-to-frame jitter on the far player for every joint group, half of ViTPose-B's left/right
swaps). ViTPose-B stays selectable in Settings → Swing pose. The other defaults (ball U-Net,
MotionBERT-Lite, YOLO11m) are unchanged.

* **Cost**: Oct 1's `pass2_pose` took 29 min (1,750 s for 61,782 frames, ≈35 crops/s with
  decode) vs 6.9 min with ViTPose-B: ×4.2. Scaled to the M7 2-hour run, processing goes from
  1.8× to ≈2.4× realtime (≈5 h 20 min instead of 3 h 53 min). GPU memory ≈6 GB at batch 8.
* **Scores**: the real implementation reproduces the experiment exactly (keypoints within
  0.005% of the crop height): near 96.3%, far 84.6% in daylight (81.0% with dusk), contact
  within ±2 frames on every seen hit. That is a few swings below ViTPose-B (97.1% / 86.6%);
  the far false forehands (8 → 14) are the place to look when the rules are next retuned.
* Settings saved before this change move to the new model once (`settings_version` 2): the
  pose model wasn't selectable before, so a stored `vitpose-base-simple` was a written-out
  default, not a choice.
* Near-player jitter is a tie: ViTPose+-L is marginally steadier there (wrists 0.224 vs
  0.232), Huge on every far-player joint group.

## Ball detector

Same data, same 40-epoch recipe as `ball-v2`, scored on the 9 held-out clips (934 ball
frames, 18 bounces).

| Model | Params | F1 | near | far | raw top-1 near / far | bounce F1 | far bounce err | GPU min per footage h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ball-v2` (width 32, seed 0) | 0.9 M | **0.940** | 0.908 | **0.951** | 0.799 / 0.867 | **0.941** | 4.3 cm | **46** |
| width 32, seed 1 | 0.9 M | 0.937 | 0.908 | 0.947 | 0.752 / 0.904 | 0.909 | 5.8 cm | 46 |
| width 64 | 3.7 M | 0.932 | **0.943** | 0.928 | 0.834 / 0.886 | 0.909 | 3.7 cm | ≈83 |
| ensemble of both width-32 models | 1.9 M | 0.937 | 0.927 | 0.940 | — | 0.909 | — | ≈84 |

GPU costs for the new rows are estimated from network-only throughput measured with nothing
else running: width 32 at 94 frames/s, width 64 at 48 frames/s, at the 1920 px crop.

* **Seed noise is as large as the width effect.** The two width-32 runs differ by up to
  14 missed balls on a single clip (Oct 1 `gt0007`: 12, 25 and 36 misses for v2, seed 1 and
  width 64). `ball-v2`'s weaker near score comes almost entirely from 17 false detections on
  the clip with no ball in play (`4c981c00/gt0006`); both new models have none there.
* Width 64 is better near (precision 0.99) and worse far (recall 0.89). The raw per-frame
  detector improves (near top-1 0.80 → 0.83), but linking erases most of that difference.
* Averaging two models' heatmaps doesn't help either. That points to a data limit: hard
  frames that no model trained on these 3,377 frames gets right.
* **To decide between ball models at the 1-point level, the test set needs to grow**
  (bench differences under ≈1.5 F1 points are noise at 9 clips). Comparing 2–3 seeds per
  model would also help.

## Pose

Every pose variant reran `pass2_pose → pose3d → swings` on both sessions. The ViTPose+
models are the Hugging Face `usyd-community/vitpose-plus-*` checkpoints, run through the
same crops and flip test with the COCO expert (`dataset_index=0`). Stroke accuracy is the
M6 exit metric (daylight swings, 80 near and 21 far serves).

| 2D / 3D model | Params (2D) | Crops/s (2D network, flip, FP16) | ≈ pose min per footage h | near strokes | far strokes | near contact ±2 f | far serve phases CV (fwd / follow) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ViTPose-B / MB-Lite** (default) | 86 M | **353** | **15** | **97.1%** | 86.6% | **100%** | 24% / 20% |
| ViTPose+-B / MB-Lite | 86 M + MoE | 290 | ≈16 | 96.9% | 82.3% | 100% | 12% / 23% |
| ViTPose+-L / MB-Lite | 304 M + MoE | 109 | ≈28 | 96.7% | **87.2%** | 100% | 16% / 16% |
| ViTPose+-H / MB-Lite | 632 M + MoE | 42 (batch ≤ 16) | ≈57 | 96.3% | 84.6% | 100% | 16% / **14%** |
| ViTPose-B / **MotionBERT (full)** | — | — | 15 | 96.6% | 83.9% | 68% | 17% / 20% |

The pose-minutes column scales the measured 15 min/h (decode included) by the extra network
time, at ≈120k pose frames per footage hour. With the dusk swings included
(`--include-dark`), far strokes come out B 80.4%, B+ 79.7%, L 85.7%, H 81.0%.

**Downstream:** stroke accuracy and contact timing don't improve with model size. Near is
saturated: its 10 missed serves are Oct 4 serves hit above the top of the frame, which no
pose model can fix. Each far serve is worth 4.8 points, so the far column is ±1 serve
between models. The far false strokes (non-strokes called forehand) actually *rise* with
ViTPose+ (8 → 12–15) because its sharper far wrists trip the rule thresholds tuned on
ViTPose-B.

**MotionBERT (full, 3× the Lite parameters) is worse.** Near contact from the pose drops
from 100% to 68% within ±2 frames. The released full checkpoint is the H36M fine-tune, not
the "global" variant the Lite model is (the one MotionBERT's in-the-wild inference uses),
which may explain it. Not worth pursuing.

**Keypoints themselves do get better.** There are no keypoint labels, so these numbers
compare the models with each other (`pose_compare`, % of crop height, ≈1.35 × player
height):

| vs ViTPose+-H | median wrist distance near / far | left/right swaps (far) | wrist jitter near / far | torso jitter far |
|---|---:|---:|---:|---:|
| ViTPose-B | 0.48 / 1.10 | 12.2% | 0.29 / 0.95 | 0.78 |
| ViTPose+-B | 0.32 / 0.92 | 11.9% | 0.24 / 0.94 | 0.60 |
| ViTPose+-L | 0.26 / 0.68 | 6.9% | 0.22 / 0.82 | 0.58 |
| ViTPose+-H | — | — | 0.23 / 0.81 | 0.55 |

Jitter is the median frame-to-frame second difference. It doesn't depend on the
reference, and it falls steadily with size: the far player's torso is 25–30% steadier with
ViTPose+-L/H than with ViTPose-B. Left/right swaps relative to Huge halve with Large. Large
gets most of Huge's gain at 2.6× its speed.

**When this becomes worth revisiting:** when the 3D swing metrics (wrist speed, rotations,
phases) are trusted for the far player, or when groundstroke footage (where arms cross the
body and left/right swaps matter more) is labeled. ViTPose+-L is then the candidate to add
as an option, costing ≈+13 min per footage hour. Retune the forehand/backhand thresholds
alongside it, and validate on held-out footage.

## Person detector (not rerun)

M2 already answered this: YOLO11l and YOLO26m at 1920 px find the same 5/5 legs-only far
players as YOLO11m, and every tracking miss on both sessions is a framing miss (player
outside the picture), not a detection miss. A bigger detector has nothing measurable to fix.

## Practical notes

* ViTPose+-H at batch 32 needs 17 GB and spills into shared memory on a 16 GB card,
  dropping to 3 crops/s. Batch 8–16 runs at 42 crops/s (5.6–9.5 GB), so `pose2d.ARCHS`
  runs it at 8.
* ViTPose+-H is in `pose2d.ARCHS` and the model registry. The other experiment shims
  (ViTPose+-B/L, MotionBERT-full, a `uens:a+b` heatmap-averaging ball detector) stayed out of
  the codebase. Training a wider ball model needs no code: `TrainConfig(width=64)`; the card
  carries the width to the detector.
