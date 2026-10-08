# M6: Pose, swings, strokes

Date: 2026-10-06. Code: `src/swingvision/pose/` (`pose2d.py` ViTPose on player crops,
`motionbert.py` + `lift3d.py` 3D lifting and court placement, `kinematics.py`, `swings.py`
detection/contact/phases/metrics, `strokes.py` rules and learned model, `windows.py`),
`src/swingvision/pipeline/stages/pose.py` (`pass2_pose`, `pose3d`, `swings`), the Swings page
(`app/pages/swings.py`, `app/components/swings_view.py`), stroke ground truth and scoring
(`training/stroke_labels.py`, `sv swings eval`), the learned classifier
(`training/train_strokes.py`, `sv train strokes`).

## Results

| Exit criterion | Target | Result |
|---|---|---|
| Stroke classification, near player | ≥ 90% | **97.1%**: 69/80 strokes and 495/501 non-strokes right (daylight); 97.1% with the dusk swings too |
| Stroke classification, far player | ≥ 80% | **86.6%**: 16/21 strokes and 81/91 non-strokes (daylight); **80.4%** with the dusk swings |
| Contact frame within ±2 frames of hit events | — | Near: **39/39** (median 0 frames). Far: **3/3** seen hits (median 1), and 8/12 within ±2 frames of the impact sound for unseen far contacts (median 1) — too few seen far hits to call it measured |
| Phase timings consistent across repeated swings | — | Serves, Oct 1 daylight: phases in order in **100%**; preparation 0.99 s (CV 8%) near / 1.00 s (5%) far, forward swing 0.17 s (17%, ±2 frames) / 0.18 s (24%), follow-through 0.22 s (16%) / 0.28 s (20%) |

`uv run sv swings eval` prints the table (`--include-dark` adds the dusk swings). CV is the
robust coefficient of variation (1.4826 × MAD / median) over repeated serves from one end.

**Read these numbers with care.** Both sessions are serve practice (as in M5), so the strokes
scored are **serves** and the non-strokes the player's dribbles on the racket, ball tosses,
serve follow-throughs, picking up balls and walking: the classifier is tested on *serve vs.
not a stroke* only. Forehands, backhands, volleys and overheads are covered by synthetic
swings in `tests/test_pose.py` only; no groundstroke footage exists yet. The rules were
developed on both sessions (no held-out session). Record a self-fed groundstroke session and
label it (below) to measure the rest.

### Ground truth

`<output_root>/training/strokes/<session_id>.json` lists every stroke of a labeled span; every
other swing there must come out `other` (format in `training/stroke_labels.py`).

* **Oct 1** (28 min, into dusk): the 101 practice shots of the M5 ground truth (all serves;
  every hit and strong impact sound of the session was checked by eye on 4K strobes for M5).
  For M6, the swings more than 0.3 s from a labeled serve (512) were rendered as strobe sheets
  (player crop at −0.6, −0.3, −0.12, 0, +0.2 s); 23 of 64 sheets (≈180 swings) were checked
  frame by frame, plus spot checks of the far end, dusk and the session's end: racket
  dribbles, tosses, serve follow-throughs, picking up balls, walking — no other strokes.
* **Oct 4** (51 min): the 61 practice shots of the M5 ground truth in 80–700 s and 780–1020 s
  (all serves). The 21 serves of the far block 468–664 s are marked not visible: the server
  stands above the top edge of the picture (player box clipped at y = 0) and is left out with
  the swings around them. 8 swings called strokes outside labeled serves were checked by eye
  (picking up balls, tosses and windups, dribbles).

Scoring is event-based: a labeled stroke is matched one-to-one to a predicted stroke within
0.8 s (the M5 contact times of unseen contacts are good to ≈0.5 s), a stroke's class must
match, and every other swing must not be called a stroke. Swings flagged `dark` (frame luma
below 40: Oct 1 after ≈16:30) are left out unless `--include-dark`.

### Errors

| Kind | Count (daylight) | Why |
|---|---|---|
| Serve missed / called other | near 10 (Oct 4), far 2 | Oct 4's near serves are hit above the top of the frame: the racket wrist's peak and the impact sound are 0.1–0.2 s apart (see *Audio/video offset*), and two had no swing at all |
| Non-stroke called forehand/backhand | near 4, far 10 | The far player's toss arm and follow-through move fast in the noisier far pose |
| Serve called forehand | near 1, far 3 | Contact time of an unseen far serve off by > 0.15 s: the overhead test looks at the wrong moment |

## Pipeline

| Stage | What it does | Output | Oct 1 (28 min) | Oct 4 (51 min) |
|---|---|---|---|---|
| `pass2_pose` (GPU) | ViTPose-B on every frame from 1.8 s before to 1.2 s after each of the player's hits and each impact sound ≥ 15 z, on the player's box | `pose/pose2d.parquet` | 1,049 s of windows, 61,782 frames, 6.9 min (150 fps), mean keypoint confidence 0.82 | 1,505 s, 89,646 frames, 13.6 min (110 fps), 0.64 |
| `pose3d` (GPU) | MotionBERT-Lite lifting, height scaling, court placement | `pose/pose3d.parquet` | 69 s | 64 s |
| `swings` | Racket hand, swing detection, contact, phases, kinematics, stroke | `pose/swings.parquet`, `pose/swings.json` | < 1 s | < 1 s |

So pose costs ≈15 min per footage hour on the RTX A5000 laptop (decode included), on top of
pass 1. `shots` (v2) joins each hit's swing (stroke, `is_serve`, `swing_id`), `segments` (v2)
gives every practice shot its swing's stroke (`swing_id`, `stroke_type`, `stroke_conf`) and
uses a confident pose-based serve or groundstroke to decide an unclear shot kind, and
`practice_eval` (v2) takes the stroke from the segment, so target stroke filters work for
unseen contacts too. M5's segmentation results are unchanged on both sessions.

*Since 2026-10-08 the default 2D model is ViTPose+-H (`vitpose-plus-huge`, Apache-2.0,
3.6 GB), ≈4.2× slower in `pass2_pose`; the numbers on this page are ViTPose-B's. See
[spikes/bigger-models.md](spikes/bigger-models.md).*

Weights come from the model registry, pinned by SHA-256: `vitpose-base-simple`
(usyd-community, Apache-2.0, 344 MB) and `motionbert-lite` (MotionBERT-Lite in-the-wild
checkpoint, Apache-2.0, 64 MB; the DSTformer model code is vendored in `pose/motionbert.py`).

## How it works

### 2D pose (`pose2d.py`)

The player's box from `movement` (15 Hz, interpolated per frame) is padded (×1.35 tall, ×1.4
wide, shifted up: a raised arm and the toss reach far above a box YOLO drew a frame earlier),
widened to 3:4, cut from the 4K frame on the GPU and resized to 256×192. ViTPose-B runs in
FP16 with flip test (mirrored crop averaged in); peaks get a sub-pixel offset from a quadratic
fit to the log heatmap. Crops are made as frames are decoded, so a window never holds more
than a batch of 4K frames.

### 3D lifting and placement (`lift3d.py`)

1. COCO keypoints → Human3.6M joints (pelvis, spine, thorax derived; the head top extrapolated
   from the neck through the ears), each clip (≤ 243 frames) normalized by the player's extent
   over the clip as MotionBERT's in-the-wild inference does, lifted with and without a
   horizontal flip and averaged.
2. Root-relative joints are scaled so ankle-to-head bones add up to the profile's height
   (1.78 m without a profile; standing height ≈ 1.05 × that chain).
3. The network sees a crop, i.e. a virtual camera looking straight at the player: the joints
   are rotated from that camera onto the real ray through the pelvis, then into court axes
   with the calibrated camera's rotation (a wide phone lens makes this tens of degrees for
   the near player).
4. Position: one translation per frame from a confidence-weighted linear least-squares fit
   of every joint to its camera ray, softly held to the ground (lower ankle 8 cm up,
   σ 0.15 m) and horizontally to the tracked feet (σ 0.1 m + the track's own σ), then
   smoothed. The feet prior matters: without it the far player's reprojection put the
   server 1–3 m inside the baseline.
5. **Snapping**: MotionBERT smooths fast motion (a serve's racket wrist peaked at ≈5 m/s,
   130 ms early). Each confident joint (≥ 0.5) is moved onto the camera ray through its 2D
   keypoint at the distance the lift gave it (MotionBERT's own `gt_2d` idea: the network
   supplies depth, the detector the image-plane motion), unless the ray misses the lifted
   joint by more than 35 cm (a left/right swap), and then smoothed over 5 frames.

Depth along the camera's axis stays the least certain part: a near serve seen from behind
moves the racket mostly away from the camera, so its wrist speed (5–7 m/s) is low; compare
swings with the same player's own swings rather than absolute norms (PLAN.md §14).

### Racket hand

From the pose, with the profile as a prior. Every time a hand peaks above the head with the
other hand down, it votes: a fast peak (≥ 1.5 m/s at the top: the racket's upswing) for
itself, a still one (≤ 0.8 m/s: the toss) for the other hand. Without five such votes, the
faster wrist at the player's hits decides; a mismatch with the profile is flagged. Both
sessions: **left-handed** (Oct 1 132:19 votes, Oct 4 112:5). The sessions have no profile, so
the default (right) would have been wrong — set the player's profile or rely on this.

### Swings (`swings.py`)

* **Detection**: every hit of the player's (`events`) with pose around it; every racket-wrist
  speed peak ≥ 7 m/s more than 0.35 s from a hit; every impact sound ≥ 15 z with the racket
  wrist moving (≥ 4 m/s) — that finds contacts the ball tracker didn't see (a serve hit above
  the frame, a far player). Oct 1: 585 swings (424 hits, 96 sounds, 65 pose-only).
* **Contact from the pose alone** (`t_contact_pose`): overhead swings at the racket wrist's
  highest point (from 0.15 s before to 0.3 s after its speed peak), others at the speed peak,
  plus a per-session lag measured on seen hits (Oct 1: 0.009 s overhead, −0.031 s other). A
  pose-only swing takes the loudest impact sound near it as its contact (after the session's
  audio/video offset and the sound's travel time).
* **Phases**: preparation starts with the unit turn (the shoulder turn starts to build) or,
  for a serve, when the tossing hand starts up; a split step is a quick pelvis dip-and-rise
  just before it; the backswing ends with the racket wrist furthest back — for a serve the
  racket drop, the racket elbow's tightest bend; follow-through ends when the wrist speed
  stays below 35% of its peak for 0.1 s (serve: when the racket wrist has come 85% of the way
  down); recovery when hand and trunk are still again.
* **Metrics** (hitter's frame: the player's end turned to the near end, mirrored for a
  left-hander, so every number reads as for a right-hander facing the net): contact height,
  in front / to the side of the pelvis, peak and average wrist speed, peak hip / shoulder /
  elbow rotation speeds and their timing relative to contact (kinetic chain in order or not),
  shoulder and hip turn, hip-shoulder separation, knee bend, elbow angle, trunk lean, stance
  width, pelvis drop and jump, toss height.

### Strokes (`strokes.py`)

Rules, in order:

1. The racket hand overhead near the contact **and a toss** (the other hand at least at head
   height before it): **serve**.
2. No ball (no hit, no impact sound — a shadow swing, picking up a ball), a racket wrist that
   never gets fast (peak < 4 m/s) or barely travels in the last 0.3 s (< 2.5 m/s average — a
   groundstroke's wrist covers well over a meter, a dribble a few centimeters), or the other
   hand up at the head while the racket hand is low (a serve's windup): **other**.
3. Racket hand overhead: **serve** from behind the baseline with no ball coming in, else
   **overhead**.
4. Otherwise **forehand / backhand** by votes: the contact's side of the body, the shoulder
   turn at the end of the backswing, the wrist's sideways motion at contact; **volley** when
   the incoming ball came over the net and didn't bounce, inside 7.5 m of the net.
5. Two strokes can't be closer than 1 s: of a closer pair the weaker (a toss, a follow-through)
   becomes other. Swings in frames darker than luma 40 are flagged `dark`, confidence ≤ 0.5.

User corrections on the Swings page (`edits.json` → `swings`) override the classifier.

**Learned model** (`sv train strokes <name>`): a bidirectional GRU over the hitter-frame
joints (relative to the pelvis) every 50 ms from 0.6 s before to 0.6 s after the contact plus
three ball features, trained on the labeled swings (ground truth + corrections), scored by
leaving one session out. It replaces the rules (`stroke_model: auto`) only when it learned
from ≥ 100 swings, saw two classes ≥ 10 times and beat the rules on the held-out sessions —
and then only for the classes it was trained on, so a serve-only model can't hide a forehand
the rules see. On the two sessions it doesn't beat the rules (Oct 1 held out: 86.6% vs 94.7%;
Oct 4: 92.2% vs 92.5%, 876 labeled swings, serve vs other), so the rules stay in use.

### Audio/video offset

Phones record audio up to ≈0.1 s behind the picture. `swings` measures it like M5 (seen hits
vs the loudest onset) and falls back to the pose (overhead peaks vs the following impact
sound) when there are fewer than 8 hit sounds. On **Oct 4** the M3 `events` stage had already
moved 229 of 446 hit times onto their sounds (without an offset), so the hit-based estimate
is circular there (0.00 s) while the frames show ≈0.10 s (racket fully up at 171.60 s, impact
at 171.72 s, 24 ms of travel). Sound-based contacts on Oct 4 are therefore ≈0.1 s late, which
is why Oct 4's near serves score worst; fixing `events` (an M5 follow-up already) is the
remedy.

## App

* **Swings page** (`/swings/<id>`, linked from the session and Practice pages): stroke counts,
  racket hand and its source; a filterable swing list (stroke, end, non-strokes); the video
  cued to the selected swing with the 2D skeleton drawn over it; the animated 3D skeleton on
  the court with a phase slider and the racket wrist's path; angle and speed curves around the
  contact with the phases shaded; a metrics table; **compare** with another swing or "my
  average" for that stroke (curves aligned at contact, dashed; metrics side by side); a
  stroke correction menu (choosing the classifier's own answer removes the correction).
* **Session page**: *Skeleton* switch (pose loaded in 8 s blocks as the video plays), a
  *Stroke* column in the Shots card, *Swings* button.
* **Practice page**: target stroke filters enabled for every stroke, strokes in the shot
  table, a *Stroke: …* breakdown (fixed a collision where a serve without a deuce/ad side
  would have been counted twice), *Swings* button.
* **CLI**: `sv swings show <id> [--all]`, `sv swings eval [<id>...] [--include-dark]`,
  `sv train strokes <name>`, `sv models download vitpose-base-simple motionbert-lite`.

## Limitations and follow-ups

* **No groundstroke, volley or overhead footage**: those classes are tested on synthetic
  swings only. Record a self-fed groundstroke session, label it in
  `training/strokes/<id>.json` (or correct strokes on the Swings page) and rerun
  `sv swings eval` / `sv train strokes`.
* **Far contact timing** rests on 3 seen far hits (plus 12 sound-timed ones): measure again
  on footage where the far player's hits are tracked.
* **Oct 4's audio/video offset** (above): move the offset estimate into `events` before its
  audio refinement.
* Near serves seen from behind move the racket away from the camera: wrist speeds are low
  and the forward-swing time of Oct 4's above-the-frame serves is less consistent (CV 23%,
  follow-through 74%).
* At dusk (Oct 1 after ≈16:30) the far pose degrades (implausible wrist speeds are flagged).
* Pose runs only in the swing windows; sparse pose elsewhere (PLAN.md §6 #13) and ankles as
  the movement foot point are not used yet.
* The `ball_3d` contact prior still uses the tracked feet, not the racket wrist (PLAN.md §7.6).
