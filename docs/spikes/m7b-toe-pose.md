# Spike: a pose model with toe keypoints (M7b)

Date: 2026-10-10 · Machine: RTX A5000 Laptop GPU (16 GB), Windows 11, ONNX Runtime 1.22 (CUDA 12
build, cuDNN 9 from PyTorch cu128). Script: `scripts/spikes/toe_pose_bench.py`.

Question: the serve analysis needs the front toe's tip in the contact frame. COCO-17 (ViTPose)
and Human3.6M-17 (MotionBERT) stop at the ankle, and the registry's ViTPose+-H checkpoint only
has the COCO head. Which model finds the toe best on the user's footage, and is there a 3D model
that gives the toe's height?

**Answer: RTMW-l (2D whole-body, OpenMMLab, Apache-2.0) through ONNX Runtime**, with two
measured offsets: its big-toe keypoint sits ≈2 cm above the sole, and the shoe's tip is
2.8 cm further on along the foot. Median toe error on the ground 2.2 cm, 90% within 5.6 cm on
44 hand-labeled toe tips. **No 3D model was adopted**: the toe's height comes from a 2D
on-ground test instead (the toe's ground point stays still over the frames before the
contact), which agrees with the labels on 95–98% of serves.

## Ground truth

* **Toe tips**: 44 near serves labeled by eye on 4K crops (zoomed, with a pixel grid) in the
  contact frame: 34 from Oct 1 (`ecc669e5`, every daylight near serve with the foot clearly
  visible; 10 left out where both feet overlap or the shoe is motion-blurred) and 10 from
  Oct 4 (`4c981c00`, every third near serve, in the swing's own contact frame since the
  contact is above the picture; 5 left out where the heel is raised toward the camera and
  hides the toe). Every labeled toe is on the ground. 25 are fully visible, 9 partly hidden,
  2 hidden behind the other shoe (estimated).
* **Contact frames**: 56 near serves of Oct 1 (all with the toss in view; the 5 darkest
  could not be labeled). See `docs/m7b-serve-contact.md`.
* Stored in `training/serve_contact/<session>.json` (the output root's training folder).

The pixel → ground scale near the server is ≈0.7 cm per pixel along the court (Oct 1's wide
lens) and ≈0.4 cm (Oct 4's zoomed framing), so the 3 cm target is 4–7 px.

## Candidates

All four run from ONNX files on the player's crop (the `movement` box padded ×1.25 as in
MMPose's top-down pipeline), on every frame from 0.35 s before to 0.12 s after each near
serve's contact (107 serves, 3,000 frames per model).

| Model | Keypoints | Source | License | Size | GPU speed |
|---|---|---|---|---:|---:|
| **RTMW-l 384×288** (distilled from RTMW-x) | COCO-WholeBody 133 | OpenMMLab `rtmw-dw-x-l_simcc-cocktail14_270e-384x288` | Apache-2.0 | 229 MB | 215 crops/s |
| RTMPose-x 384×288 | Halpe-26 (body + feet) | OpenMMLab `rtmpose-x_simcc-body7_pt-body7-halpe26` | Apache-2.0 | 200 MB | 206 crops/s |
| RTMW3D-x 384×288 | 133, 2D + relative depth (H3WB) | rtmlib's Hugging Face copy | Apache-2.0 | 369 MB | 72 crops/s (batch 1 export) |
| ViTPose-L whole-body | 133 | third-party ONNX export on Hugging Face | Apache-2.0 (weights' provenance unclear) | 1,235 MB | 113 crops/s |

Each serve needs ≈30 crops, so any of them costs seconds per session; speed doesn't decide.

## Toe on the ground

The front toe's keypoint (median of the 5 frames around the contact frame), cast onto a
horizontal plane, against the labeled toe tip cast onto the ground (cm). *After bias* removes
the median offset along / across the heel → toe direction, i.e. what a fixed correction can't
fix:

| Model | keypoint at | error median | bias along / across | after bias: median | 90% |
|---|---|---:|---:|---:|---:|
| **RTMW-l** | 0 cm | 7.4 | +4.8 / +4.5 | 3.2 | 5.9 |
| **RTMW-l** | **2 cm** | **3.8** | **+2.7 / +1.1** | **2.5** | **5.0** |
| RTMPose-x Halpe-26 | 0 cm | 6.5 | +4.7 / +2.6 | 3.3 | 6.8 |
| RTMPose-x Halpe-26 | 2 cm | 4.4 | +3.1 / −0.6 | 2.8 | 6.6 |
| RTMW3D-x | 0 cm | 8.0 | +6.0 / +5.4 | 3.4 | 6.8 |
| RTMW3D-x | 2 cm | 4.8 | +4.2 / +1.4 | 2.9 | 6.5 |
| ViTPose-L whole-body | 0 cm | 6.8 | +5.9 / +0.2 | 4.1 | 6.7 |
| ViTPose-L whole-body | 2 cm | 6.3 | +3.4 / −3.9 | 3.5 | 7.0 |

* Every model places the "big toe" short of the shoe's tip and a little above the sole (it was
  trained on toes, not shoe tips). Taking the keypoint 2 cm above the ground halves the error
  for all of them; what's left is mostly a shift along the foot.
* **RTMW-l has the smallest spread** (90% within 5.0 cm after the bias) and the smallest
  across-foot bias. With the tip 2.8 cm ahead of the keypoint along the foot, the production
  code (`pose/serve_contact.py`, `ToeParams`) gives **median 2.2 cm, 90% 5.6 cm, 91% within
  6 cm** on the 44 labels (`sv serves eval`).
* **Does the offset generalize?** Fitted on Oct 1 alone (3.1 cm) it gives Oct 4 a median of
  2.8 cm and 90% of 4.9 cm, the same as with the offset fitted on both. Without the offsets the
  median is 7.4 cm.

## Toe height: no 3D model needed (or good enough)

The plan's on-ground test was "the 2D toe is still before the contact *and* the 3D toe height is
below 3 cm". Two 3D sources were tried:

* **MotionBERT's ankle plus a fitted foot** (the toe on its ray at the ankle-to-toe distance
  measured earlier in the window): the court-placed 3D ankle at serve contact sits 0.2–0.77 m
  high (it should be ≈0.1–0.2 m), so the fitted foot put toes that are on the ground up to
  35 cm in the air. 21 of 44 grounded toes would have been called lifted.
* **RTMW3D-x's relative depth**: depth relative to the body's root, 2.17 m full scale. On this
  geometry a 5 cm depth error moves the toe's height by ≈2.5 cm, and its 2D toe was the
  second-worst. Not a usable height source either.

So the test is 2D only: the toe's ground point (keypoint ray at 2 cm) over the 6 frames up to
the contact frame, median of the first half against the second half (so one jittery keypoint
doesn't decide), moved less than 3 cm. On the labels that is **43 of 44 (97.7%)** in the spike,
**42 of 44 (95.5%)** in production (the production keypoints come from the stage's own
windows). A lifted foot moves several times that in 0.1 s. A toe that fails the test is placed
on the ground with a ±5 cm height uncertainty along its ray (`toe_lifted`, a forward σ of
≈10 cm). None of the labeled near serves has a lifted front toe: the user's right foot stays
down, as the plan expected; the lifted cases in production are the dark serves, where the
keypoints jitter.

## Install footprint

* `onnxruntime-gpu` adds 344 MB to the environment. Version 1.23+ is built for CUDA 13 and
  doesn't load with PyTorch's CUDA 12.8 libraries, so it is pinned below 1.23
  (`onnxruntime-gpu>=1.20,<1.23`). Importing torch first loads the cuDNN 9 libraries the CUDA
  provider needs; without CUDA it falls back to the CPU (≈10× slower, still seconds per
  session).
* The weights (229 MB) download from OpenMMLab on first use, verified by SHA-256 (registry
  name `rtmw-l-wholebody`; the zip's `end2end.onnx` is kept).
* Staying in PyTorch would have meant porting the RTMW architecture (a third-party Hugging Face
  port exists, `akore/rtmw-x-384x288`, but needs `trust_remote_code`); ONNX Runtime was
  smaller and quicker to trust.

## Decision

RTMW-l through ONNX Runtime for the foot keypoints (`pose/feet.py`, stage `serve_feet`); the
keypoint height and tip offset as measured here; the 2D on-ground test. Revisit the 3D height
when a body-model regressor with feet (SMPL-X) is worth its install size, or if serves with a
lifted front foot show up.
