# Swing Vision Open

A personal, local-first tennis video analysis tool. Point a fixed camera at the court, record
a practice or a match, and let your NVIDIA GPU break the footage down. The roadmap and
architecture are in [PLAN.md](PLAN.md).

**Status: milestones M0–M6 are complete: practice mode is usable, with swing analysis.** The app ingests footage, builds a
browser-playable proxy, detects audio onsets, finds the court and fits a full camera model
(sub-pixel on real footage), checks whether the camera moved during the recording, and lets
you review the calibration. It tracks you on the court (also at dusk) and reports your
movement: distance, speeds, a live court map and a heatmap. It tracks the ball on every frame
with a detector trained on your own labeled footage and finds hits, bounces (with their spot
on the court) and net contacts. Every shot gets a 3D flight: speed off the racket, at the net
and before the bounce, net clearance, height, landing spot and line call. Practice sessions
are cut into one clip per shot (feed, hit, landing), grouped into blocks, and scored against
targets you draw on the court. Every swing gets a 2D and 3D pose, its phases (preparation,
forward swing, follow-through), body kinematics and a stroke type. Match scoring arrives in
M8–M10.

## Requirements

* Windows 11 or Linux with an NVIDIA GPU and a recent driver (developed on an RTX A5000, 16 GB)
* [uv](https://docs.astral.sh/uv/)
* `ffmpeg` and `ffprobe` on `PATH`, built with NVDEC/NVENC (e.g. the gyan.dev "full" build
  via `scoop install ffmpeg`)

## Setup

```bash
uv sync --extra nvdec
```

This installs Python 3.12, PyTorch with CUDA 12.8, and the rest of the dependencies into
`.venv`.

## Run

```bash
uv run sv app
```

Open http://127.0.0.1:8050. On first run, go to **Settings** and choose an **output folder**.
Every session's derived data, the library database, and (later) fine-tuned models live
there. Raw footage is never copied.

Then go to **New session**, choose a video, and click **Create and process**. A background
worker (started by the app) processes the job. You can watch it on **Jobs**, and open the
session from the **Library** when it's done.

### Court calibration

Right after ingest, the worker finds the court on a median "background" of frames sampled
across the video (players removed) and fits the camera: focal length, lens distortion, and
position. By default the job then **pauses for your review** (status *needs action*): click
**Review calibration** on the Jobs page (or **Calibrate** on the session page).

* The projected court is drawn over the background. If it's off, drag any circle onto its
  court corner (or the net post tops / center strap); dragged points turn yellow and the court
  follows. Then click **Snap to lines** for a sub-pixel fit. *Line RMS* under 2 px is good.
* **Drift check** lists the video in time windows. If the camera was bumped or re-aimed,
  those windows get their own camera automatically; pick one under *Image* to check it.
* **Confirm and continue** saves your calibration and resumes processing.

To skip the review when the automatic fit is good, enable *Continue without review* under
**Settings → Court calibration**. The session page's **Court overlay** switch draws the
court over the playing video.

### Player tracking and movement

After calibration the worker detects people in the video (YOLO11m on your GPU, 15 frames per
second, about half of realtime on an RTX A5000 laptop) and follows the one player on your
court. Spectators, people on neighboring courts, and static things like a ball machine or a
bag are ignored. The model weights (≈40 MB) download on first use.

On the session page:

* **Player box** draws the tracked player on the video (dashed while a short gap is bridged).
* **Court** shows where you are on a top-down court as the video plays, with a 3 s trail.
* The timeline gets a **player speed** row under the audio onsets.
* **Movement** sums it up: distance, tracked time, top and average speed, and a heatmap of
  where you spent your time (*Fold both ends together* if you switched ends).

For ball-machine sessions, the machine is recognized when it stands still in view; otherwise
turn on *Click the court to place the ball machine* under the court map and click its spot.

### Ball tracking and events

In the same decoding pass the worker looks for the ball on every frame (or, to save GPU
time, at a lower rate with full-rate windows around hits, bounces and lost-ball moments) and
links the detections into one ball track. Hits, bounces and net contacts are found where the
ball's path breaks; bounces get a position on the court, hits are timed with the racket
sound when one matches.

On the session page, *Ball* draws the ball and a short trail on the video, the timeline marks
hits (orange) and bounces (blue), and the court map shows recent bounces.

The ball detector works out of the box (a classical motion detector, slow at ≈3 GPU-hours per
footage hour and unreliable near the player) and gets much better and faster when trained on
your own footage (≈50 GPU-minutes per footage hour):

1. **Labeling** page: pick a session, create a 3 s clip (suggestions list moments where the
   tracker struggled). Labels are pre-filled from the current detector. Step through the
   frames (←/→), click the ball where it's wrong (clicks snap to the moving ball), *N* for no
   ball, *O* when it's hidden, *I* to interpolate between labeled frames, *H*/*B* for hits and
   bounces. Mark the clip *Done* as **Train** or **Test**.
2. `uv run sv train ball my-model` trains the detector on the Train clips (≈30 min on an RTX
   A5000 laptop); `uv run sv labels pseudo <session-id>` adds automatically labeled clips from
   confident detections.
3. `uv run sv bench ball motion unet:my-model "unet:my-model,15"` compares detectors and
   frame-rate schedules on the Test clips (HTML report with an accuracy-vs-cost chart);
   `uv run sv eval` checks the active detector against the targets.
4. In **Settings → Ball detection** pick the detector (*Automatic* uses your newest model),
   then reprocess sessions (person detections are kept).

### Shots: speed, net clearance, landing

From the ball track and the events, every flight between two events is fitted in 3D through
the calibrated camera with real ball physics (gravity, air drag, spin). Where the ball is
along the camera's line of sight comes from gravity and the flight's ends: a bounce is on the
ground, and a hit is where the incoming ball (a feed's bounce, a serve toss) was met, near
your tracked feet. Each shot gets its speed off the racket (± its uncertainty), at the net
and before the bounce, its net clearance and highest point, a rough topspin/slice sign, and
its landing spot with an in/out call against the singles court (*close call* when the
margin is within 2σ of the landing uncertainty). Hits the 3D fit shows weren't at the player
are dropped.

On the session page, the **Shots** card lists the shots over the net (click one to play it)
with in %, median and fastest speed; while the video plays it shows the current shot's
numbers and a side view of its flight, *Shot path* draws the fitted flight on the video, the
court map marks every landing, and the timeline has a row of shot speeds. `uv run sv shots
<session-id>` prints the same list.

Speeds are **not calibrated** against a radar gun or a ball machine yet, and the app says so:
every speed is shown as `value ± error`, where the error is 3% (the bound on a shared scale
error, from the gravity checks below) plus twice the shot's own fit uncertainty, e.g.
*139 ± 10 km/h* for a serve from the camera's end. The Shots card explains this; `sv shots`
prints the same note.

Accuracy: on synthetic flights, speeds come out within 3% for shots from the camera's end
(within 5% for a far-court hitter); on real footage, flights refitted with gravity left free
give 9.87 and 9.96 m/s² on two sessions (0.6% and 1.5% off), so the speed scale is right to
about 1-1.5%. No radar gun reading exists yet to compare with: if you have one, record a
session with it. Leave room above the far baseline in the picture: a serve that leaves the
top of the frame has no detected contact and gets no shot record. Details in
[docs/m4-ball-3d.md](docs/m4-ball-3d.md).

### Practice: shots, targets, accuracy

Every practice shot becomes its own clip, from the feed (a drop, your pre-serve bounces, a
ball-machine feed) to where the ball landed, and the shots are grouped into blocks (a pause
to collect balls, a change of end, or switching between serves and groundstrokes starts a
new one). Shots are found from the hits the ball tracker saw, and also when it didn't see
the contact: a serve hit above the top of the picture is found from its landing and the
sound of the impact, and at dusk from your toss and the impact sound. Serves are recognized
on their own (toss, contact height, pre-serve bounces) and called against the right service
box; pick *Serve practice* to treat every shot as a serve.

**Targets.** Draw rectangles or circles on the court when creating a session (or later on
the Practice page), or add presets (service boxes, T and wide corners, deep zones), and save
them as named target sets. Targets are *relative* by default: you draw them as if hitting
from the near end, and they follow you when you play from the other end. *Absolute* targets
stay where they are drawn. A target can be limited to some strokes (serves, forehands, …:
the stroke recognized from your pose, see below).

The **Practice** page (button on the session page, link in the Library) shows the video with
the targets drawn in, where every ball landed ("as you hit" or on the court; click one to
play it), in %, net %, target hits, distance to target, depth spread and speed, accuracy over
the session, a table per block and per shot, and a breakdown by serve side and speed. Fix
what the analysis got wrong right there: *Not a practice shot*, *Landing is right*, or
*Place landing* and click the map; accuracy updates in a second. On the session page the
Segments card lists the blocks and shots, the timeline shows each shot as a band, and N/P
jump to the next/previous shot.

Accuracy: on the user's two serve-practice sessions, 159 of 162 shots labeled by eye are
found and clipped correctly (98.1%) with no false clips, and every target hit/miss agrees
with the targets projected into the video. Details in
[docs/m5-practice.md](docs/m5-practice.md).

### Swings: pose, phases, strokes

Around every hit and every impact sound, the worker runs a pose network on the full-resolution
crop of you (ViTPose, ≈15 GPU-minutes per footage hour), lifts the 2D skeleton to 3D
(MotionBERT), scales it to your height and places it on the court. From that it finds your
swings — also those whose ball the tracker missed — and measures them:

* **Racket hand** from the pose (the hand that swings up fast over your head; your profile's
  handedness only decides when the footage shows too little).
* **Stroke**: serve, forehand, backhand, forehand/backhand volley, overhead, or *other*
  (dribbling the ball on the racket, the toss, picking up balls, shadow swings).
* **Phases**: preparation (unit turn, or the toss for a serve), split step, backswing end
  (racket drop for a serve), contact, follow-through and recovery, with their durations.
* **Kinematics**: contact height and position relative to your body, racket-wrist speed,
  shoulder and hip turn, hip-shoulder separation, knee bend, elbow angle, trunk lean, stance,
  jump, toss height, and whether hips → trunk → elbow → wrist peaked in order.

The **Swings** page (button on the session and Practice pages) lists the swings (filter by
stroke and end), plays the selected one with the skeleton drawn over the video, shows it as
an animated 3D skeleton on the court with a phase slider, plots the joint angles and the
wrist speed around the contact with the phases shaded, and compares it with another swing or
your average for that stroke. If a stroke is wrong, pick the right one there: shots and
practice results update, and the correction becomes a training example for
`uv run sv train strokes my-model` (the learned classifier replaces the rules only once it
beats them on sessions it didn't learn from). The session page has a *Skeleton* switch and
the stroke of each shot.

Accuracy, on the two serve-practice sessions (serves vs. everything else; no groundstroke
footage yet): 97.1% of swings right for the near player, 86.6% for the far one in daylight
(80.4% including dusk); the contact found from the pose alone was within 2 frames of the ball
hit on every seen hit (39 near, only 3 far). Depth along the camera's line of sight is the least certain part of a single-camera 3D
pose, especially for the far player: compare swings with your own rather than with absolute
norms. Details in [docs/m6-swings.md](docs/m6-swings.md).

### Profiles

Create a profile for yourself on **Profiles** (name, handedness, one- or two-handed
backhand, height) and pick it when creating a session (or on the session page). Height
places you correctly when your feet are out of frame and scales the 3D swing analysis;
handedness is the fallback when the footage doesn't show which hand holds the racket.

## CLI

```bash
uv run sv settings set-output-root D:\SwingVisionData
uv run sv profiles add "Chris" --height-cm 183   # then: sv profiles list
uv run sv create D:\footage\practice.mp4 --name "Basket forehands" --profile <id>  # + enqueue
uv run sv worker                       # process the queue in the foreground
uv run sv jobs                         # queue status
uv run sv process <session-id> --inline --force proxy   # rerun a stage in this process
uv run sv probe D:\footage\practice.mp4
uv run sv court detect D:\footage\practice.mp4 --out overlay.jpg  # court fit + overlay image
uv run sv models list                  # model weights, licenses, download status
uv run python scripts/m2_eval_tracking.py <session-id> --out sheets/  # tracking check
uv run sv labels list                  # labeled clips
uv run sv labels pseudo <session-id> --clips 20   # training clips from confident detections
uv run sv train ball my-model          # train the ball detector on your labels
uv run sv train events my-events       # learned hit/bounce classifier (optional)
uv run sv bench ball motion unet:my-model "unet:my-model,15"   # compare (HTML report)
uv run sv eval                         # metrics on the Test clips vs the M3 targets
uv run sv shots <session-id>           # shots: speed, net clearance, landing, line call
uv run python scripts/m4_validate_speed.py <session-dir>   # speed-scale checks (gravity, drag)
uv run sv practice show <session-id>   # practice blocks and shots: call, target, speed
uv run sv practice eval                # segmentation vs shots labeled by eye (M5 target: 95%)
uv run python scripts/m5_check_targets.py <session-id> --sets builtin  # target check in the image
uv run sv swings show <session-id>     # strokes, phases, wrist speed (--all: also non-strokes)
uv run sv swings eval                  # strokes, contact timing, phase spread vs labels (M6)
uv run sv train strokes my-strokes     # learned stroke classifier from labels + corrections
```

## Recording tips

* Mount the camera **centered behind a baseline and as high as practical (≥ 2.5–3 m)**. A
  low camera squashes the far half of the court and costs precision there.
* Make sure the camera sees as many court lines as possible, including the far baseline.
* Frame the entire court with some margin behind the far baseline. If the far baseline sits
  at the top edge of the picture, you can't be tracked while standing behind it.
* Record 4K at 60 fps, with exposure and focus locked if your phone allows it. Avoid recording
  into darkness.
* Leave room at the top of the picture (some sky above the far baseline): a serve's toss and
  contact that go out of the top of the frame can only be found from where the ball lands
  and the sound of the hit, and get no speed.
* Keep yourself in the picture, head to toe, at both ends: swing analysis needs to see you
  (a serve whose arm leaves the top of the frame is still recognized, but less reliably).
* Don't touch the camera once you start recording. (If you do, the drift check notices and
  calibrates those minutes separately, but it's better not to.)

## Development

```bash
uv run pytest                 # unit + ffmpeg integration tests
uv run ruff check src tests   # lint (pandas/polars imports are banned; use PyArrow)
uv run pre-commit install     # optional git hooks
```

Benchmarks from the M0 spikes live in `scripts/spikes/`, with results in
[docs/spikes/m0-video-pipeline.md](docs/spikes/m0-video-pipeline.md).

### Layout

```
src/swingvision/
  settings.py        app settings (user config dir)
  services.py        create/enqueue/delete sessions (shared by CLI + app)
  storage/           SQLite library + job queue, session dirs, PyArrow schemas/tables
  io/                ffprobe, ffmpeg runner, proxy encode, audio + onsets, frame decoding
  court/             court model, camera model, detection, calibration (M1)
  players/           person detection, tracking, movement (M2)
  ball/              ball detectors, frame-rate schedules, linking, events (M3),
                     3D flight physics and fitting (M4)
  analysis/          shot records (M4); practice segmentation, targets, accuracy (M5)
  pose/              2D pose, 3D lifting and placement, kinematics, swings, strokes (M6)
  training/          labels, labeling helpers, training, benchmark, evaluation (M3, M6)
  models/            pretrained weights registry (URLs, SHA-256, licenses)
  pipeline/          stage framework, DAG runner, worker process, stages/
  app/               Dash + Mantine UI (pages/, components/, assets/)
```
