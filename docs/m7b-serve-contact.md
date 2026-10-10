# M7b: Serve contact point

Date: 2026-10-10. Code: `src/swingvision/pose/serve_contact.py` (toss path, contact frame,
toe, offsets), `pose/feet.py` (foot keypoints), `pipeline/stages/serve.py` (stages
`serve_feet`, `serve_contact`), `pose/swings.py` + `pipeline/stages/pose.py` (`swings` v2:
serves' contacts move to the contact frame), `analysis/serve_stats.py` (bins, grid, findings,
regression), `analysis/stats.py` and `analysis/aggregate.py` (serve records), the Stats page
(`app/components/serve_view.py`), the Swings page (`app/components/serve_frames.py`),
`training/serve_labels.py` and `sv serves contact|eval`. Tests: `tests/test_serve_contact.py`
(synthetic serves), `tests/test_serve_stats.py`. The model spike:
[spikes/m7b-toe-pose.md](spikes/m7b-toe-pose.md).

## What it does

For every serve it finds where the ball was struck relative to the toe tip of the front foot
(the foot opposite the racket hand: the right foot for the left-handed user), **in the contact
frame**, the last frame before the ball heads for the net. The offsets are in the hitter's
frame: *forward* toward the net, *lateral* toward the racket-arm side (so left- and
right-handers read the same), and the ball's *height* above the ground, each with a σ. The
Stats page relates them to serve speed and serve-in %, for one session or any selection of
sessions; the Swings page shows the contact frame with the toe and the ball drawn in and lets
you correct both.

## How

**Contact frame from the toss path** (`toss_contact`). The ball tracker's detections from 1.6 s
before the swing's contact estimate are searched for the toss: the highest point with the ball
seen rising into it. The release is where the upward image speed peaks (in the hand the ball
accelerates up the picture, in free flight it slows down), so hand points never enter the fit.
The toss is fitted in 3D with the same physics as every flight (gravity, drag with its prior,
Magnus), seeded on the 0.3 s rise into the apex and grown point by point with refits:
backwards to the release, forwards until a detection leaves the path by more than 6σ (at least
9 px) and the next one confirms it. The frame before that is the contact frame. When the
detections around the contact are missing or between the gates, the serve's first detections
are extrapolated back (constant image velocity) to where they meet the toss path's projection,
and the contact frame is the last frame before that time (`contact_frame_inferred` when frames
are missing). The contact point is the toss path at the contact frame's time, its covariance
sampled from the fit. The serve flight's contact prior at the feet plays no part.

This runs inside `swings` (version 2): a serve's `t_contact` moves to the contact frame
(`contact_source = toss_path`) and its phases and metrics are measured from there, so
everything downstream (shots, segments, practice, stats) uses it. The toss fits are kept in
`swings.json` (`tosses`, by swing id) and cached on disk by their inputs
(`work/toss_contacts.json`), so a stroke correction doesn't refit every toss (first run ≈1.6 s
per serve: 140–180 s for Oct 1's 89 serves; a rerun 3–4 s).

**Foot keypoints** (`serve_feet`, GPU). RTMW-l (2D whole-body, ONNX Runtime) on the player's
crop in every frame from 1.1 s before to 0.1 s after each serve's contact (windows on a 0.1 s
grid, merged): ankles, big and small toes and heels of both feet. ≈75 frames per serve,
≈215 crops/s on the GPU. Frames already computed with the same model, crop and player track
are kept (a sidecar `serve_feet.json` records what they came from), so moving a contact frame
decodes only the frames that are new and runs in the app. Oct 1 (89 serves) took 8 min here,
almost all of it CPU decoding (PyNvVideoCodec wasn't installed on this machine; with NVDEC it
is the ≈30 s of network time).

**The toe** (`toe_at_contact`). The front foot's big-toe keypoint (median over ±2 frames)
cast onto a plane 2 cm above the ground (the keypoint sits above the sole), moved 2.8 cm ahead
along the heel → toe direction to the shoe's tip; both offsets measured in the spike. The toe
is on the ground when its ground point stayed within 3 cm over the 6 frames up to the contact
(median of the first half against the second); otherwise it's flagged `toe_lifted` and gets a
±5 cm height uncertainty along its ray. A toe keypoint the model is unsure of (score < 3 of ≈10)
is replaced by the heel plus the foot length (the profile's shoe length, else 15% of the
height) along the foot (`toe_estimated`). The σ combines 2 px of keypoint noise through the
image → ground Jacobian with 1.5 cm for where the tip is. `toe_moved_m` is how far the toe moved
since the toss release; `toe_to_baseline_m` its distance behind the baseline.

**Offsets** (`serve_contact`, CPU). Contact minus toe in the hitter's frame (mirrored for the
far end), σ from the summed covariances; the height is the ball's above the ground, also
relative to the player's height. A serve whose toss path isn't known is flagged
`contact_above_frame` when the toss left the top of the picture, or when the player's usual
contact height over the toe (the session's median when 5 tosses are measured, else 1.55 ×
the player's height) projects above it. Other flags: `toss_not_tracked`, `contact_not_seen`,
`poor_toss_fit`, `no_toe`, `far`, and the swing's `dark` / `low_conf`. One row per serve in
`pose/serve_contact.parquet` (schema `SERVE_CONTACT`), plus `serve_feet.parquet`.

**Statistics.** The `stats` stage (records v2) adds a `serve` record per contact, joined with
its shot record for the speed, the call and whether it's excluded; a selection of sessions
reads them like every other record. `analysis/serve_stats.py`:

* the selection shown: near end only and flagged serves hidden by default, deuce/ad;
* effect charts: per axis, as many equal-count bins as hold 15 serves each (at most 6), mean
  speed with a 95% interval and in % with a Wilson interval;
* a forward × lateral grid of 10 cm cells with in % and the median speed, cells of 8+ serves;
* findings: per axis the bin whose speed or in % differs most clearly from the rest's
  (difference of means with a normal interval, of proportions with Newcombe's interval), only
  when the 95% interval excludes 0: "Contacts 20–40 cm in front of the toe: +6 km/h and +12
  points of in % vs. the rest (n = 84 / 213)";
* a regression in the details panel: speed by least squares on a quadratic in the three
  offsets, in by a logistic (Newton), effects per 10 cm at the average contact with 95%
  intervals.

**Corrections.** On the Swings page: *Earlier* / *Later* step the contact frame, a click on the
toe picture places the toe, *Reset* drops both. They go to `edits.json` (`serves`: the serve's
time, frame, toe pixel); a frame change reruns `swings` → `shots` → `serve_feet` (only the new
frames) → `serve_contact` → ... → `stats` in the app, a toe change only `serve_contact` and
`stats`. The corrections count as ground truth in `sv serves eval`.

## Choices beyond the plan

* **No 3D foot model.** Neither MotionBERT's ankle plus a fitted foot nor RTMW3D's relative
  depth measured the toe's height well enough (the spike); the on-ground test is 2D only, and a
  lifted toe gets a large σ instead of a height.
* **The contact frame is computed in `swings`**, not `serve_contact`, so the swing's contact,
  phases and metrics and every downstream stage use it without a dependency cycle.
* **The release from the image speed**, not from the tossing hand's wrist: the pose wrist near
  the top of the picture is the least reliable joint, and the speed peak is exact.
* **Contacts above the picture** are also recognized without a toss (from the usual contact
  height), so all of Oct 4's near serves are flagged.
* **`serve_feet` keeps its frames**, so a corrected contact frame updates in-process (the
  runner gained an `allow` filter: the app runs only stages that are cheap or *light*).
* **One finding per axis** (the clearest bin), not every bin vs the rest: with two bins the
  halves would be mirror images of each other.
* **Library v5** is the profile's shoe length; M7c's devices and calibrations move to v6.
* **The serve's call and speed** come from the shot and practice records (after corrections), so
  calibrated speeds (M7c) will flow in without changes here.

## Exit criteria

All measured with `sv serves eval` on the ground truth in `training/serve_contact/` (labeled by
eye from 4K crops, 2026-10-10: 56 contact frames of Oct 1's near serves, 44 toe tips from
both sessions, 16 ball centers), unless noted.

### Synthetic serves with a known toe and contact: offsets within 2 cm; contact frame exact

**Met without detection noise; within σ with it.** `tests/test_serve_contact.py` renders serves
through a camera like the user's: a toss from an independent integrator (drag 0.58 vs the
fit's 0.55 prior) released from an accelerating hand, a 44 m/s crosscourt serve after a contact
anywhere in the frame interval, and the front foot's keypoints where RTMW puts them.

* Noise-free detections and keypoints (6 serves, contacts at 5–85% of the frame interval):
  contact frame exact, release within 0.04 s, forward / lateral / height offsets within 2 cm.
* 1 px noise (8 serves): contact frame exact for contacts up to 70% of the interval, within ±1
  frame later (the next frame's ball has moved < 6 px, inside the gate); toe within 2 cm;
  offsets within 3σ. The forward axis lies along the camera's line of sight, where the toss's
  depth comes only from its gravity curvature: over 12 serves the contact's error along it is
  2.3 cm median, 4.8 cm at most (1.1 / 2.5 cm with 0.5 px noise); the reported σ is ≈7 cm,
  conservative because the fit down-weights the fast rise as if blurred. The serve flight
  doesn't help: fitted alone, its start is uncertain by ≈46 cm along the line of sight.

### Contact frame vs the hand-labeled one: exact on ≥ 80%, within ±1 on ≥ 95% of near serves

**Met: 89.3% exact, 100% within ±1** (56 serves, Oct 1, including 6 at dusk and night). For
comparison, the hit event's frame (on the 54 of these with a seen hit) was exact on 81.5% and
within ±1 on all; the toss path also finds the frame of serves found only from the sound. The 6
misses are all one frame: 5 are frames where the racket overlaps the ball and I judged it
already moving (the detection is still within the path's gate), 1 at night.

### Toe vs the hand-labeled tip on near serves: median ≤ 3 cm, 90% ≤ 6 cm; on-ground test ≥ 95%

**Met: median 2.2 cm, 90% at 5.6 cm (91% within 6 cm)** over 44 toes (34 Oct 1, 10 Oct 4; Oct 4
alone 3.0 cm / 4.9 cm). **On-ground test: 42 of 44 (95.5%)**. Every labeled front toe is on the
ground, so the test's other side (a lifted toe) has no real examples; it is covered by the
synthetic test (a foot rising at 1.2 m/s is flagged).

### Contact point: reprojected within 4 px (median) of the labeled ball; median forward σ ≤ 8 cm

**Met: 2.9 px median** (16 ball centers in labeled contact frames: the centroid of the
ball-coloured blob, or by eye where the racket overlaps it). **Forward σ 7.5 cm median** (Oct 1's
near serves with offsets).

A caveat on the forward axis beyond the per-serve σ: the toss's depth scales with the camera's
focal length and the video's clock, so a 0.5% error in either moves every contact by ≈4 cm along
the line of sight. Comparisons between serves (the Stats page's question) don't depend on it;
the absolute forward offset carries this extra uncertainty (the M4 g-refit puts the scale right
to ≈1%; M7c's net-tape references will measure it).

### Coverage: ≥ 85% of Oct 1's near serves with the contact in frame; Oct 4 flagged

**Met: 57 of 61 (93.4%)** of Oct 1's near serves get offsets; the other 4 have no tracked toss
(night). **All 46 of Oct 4's near serves are flagged `contact_above_frame`** (33 from the toss
leaving the top edge, 13 with no toss seen, from the usual contact height).

### The Stats page shows the maps and effect charts for both sessions together

**Met.** A selection of both sessions (`/stats?sessions=ecc669e5,4c981c00`) shows the section:
40 serves (Oct 1's daylight near serves; Oct 4 adds none, its contacts are above the picture),
the top-down and side maps, the effect charts (2 bins per axis at this count), the summary and
the regression; the grid stays empty until a 10 cm cell holds 8 serves. A one-session selection
gives exactly the section of the session's own page (`test_serve_section_same_for_one_session_
and_a_selection`), and a synthetic library with a planted effect (+8 km/h per 10 cm in front;
fewer in when struck far to the racket side) is recovered by the regression, the bins and the
findings (`test_statistics_recover_the_planted_effects`).

## On the user's serves (Oct 1, 40 daylight near serves)

Median contact 9 cm in front of the toe (± 7 cm each), 26 cm to the racket side, 2.78 m high;
the front toe stays down on 90% (the rest are flagged, not necessarily lifted). One finding
clears the bar so far: contacts from −33 to +25 cm to the racket side went in 34 points more
often than those further out (n = 20 / 40). Too few serves for more: the grid needs a few
hundred, and serve type (flat, slice, kick) isn't separated yet.

## Open items

* Label more contact frames and toes as sessions come in (the Swings page's corrections count
  as labels), and record a session with the toss and contact in view at both ends.
* Serve type (flat / slice / kick) confounds the effects; a tag or a classifier would help.
* A real lifted-toe example to check the on-ground test's other side.
