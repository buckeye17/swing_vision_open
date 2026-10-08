# Swing Vision Open

A personal, local-first tennis video analysis tool. Point a fixed camera at the court, record
a practice or a match, and let your NVIDIA GPU break the footage down. The roadmap and
architecture are in [PLAN.md](PLAN.md).

**Status: v0.1, the practice-mode MVP (milestones M0–M7).** The app ingests footage, builds a
browser-playable proxy, detects audio onsets, finds the court and fits a full camera model
(sub-pixel on real footage), checks whether the camera moved during the recording, and lets
you review the calibration. It tracks you on the court (also at dusk) and reports your
movement: distance, speeds, a live court map and a heatmap. It tracks the ball on every frame
with a detector trained on your own labeled footage and finds hits, bounces (with their spot
on the court) and net contacts. Every shot gets a 3D flight: speed off the racket, at the net
and before the bounce, net clearance, height, landing spot and line call. Practice sessions
are cut into one clip per shot (feed, hit, landing), grouped into blocks, and scored against
targets you draw on the court. Every swing gets a 2D and 3D pose, its phases (preparation,
forward swing, follow-through), body kinematics and a stroke type. A Stats page sums up each
session (speeds by stroke, landing heatmaps, depth, movement) and exports the shots as CSV or
Parquet. A 2-hour session processes unattended overnight. Match scoring arrives in M8–M10.

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

### Processing overnight

Processing takes about 2.4 hours per hour of footage on an RTX A5000 laptop (a 2-hour session
≈5 h 20 min; 3 h 53 min measured with the faster ViTPose-B pose model, which you can pick in
**Settings → Swing pose**), so long sessions are best left to run overnight:

1. In **Settings → Court calibration**, turn on *Continue without review when the fit is
   good*. Otherwise every job pauses after court detection until you confirm the calibration
   (you can also confirm it in the first minutes, then leave). With it on, a calibration is
   used unattended when it fits the painted lines within the threshold, including the
   stretches where the camera moved (a mount sagging or a bumped tripod), each of which must
   fit just as well with its own camera.
2. Create the session and leave the app running (or run `uv run sv worker`). The worker keeps
   Windows from going to sleep while it works; the screen may still turn off.
3. In the morning the session is *ready*. If anything stopped it, the Jobs page says why and
   what to do (see [When something goes wrong](#when-something-goes-wrong)).

Jobs survive interruptions: if the app or the computer stops mid-job, the worker picks the job
up where it was the next time it starts (long stages resume from 2-minute checkpoints).

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
crop of you (ViTPose+-H, ≈1 GPU-hour per footage hour; ViTPose-B in Settings is ≈4× faster
with noisier far-player keypoints), lifts the 2D skeleton to 3D
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
footage yet): 96.3% of swings right for the near player, 84.6% for the far one in daylight
(81.0% including dusk); the contact found from the pose alone was within 2 frames of the ball
hit on every seen hit (39 near, only 3 far). ViTPose-B scored slightly higher (97.1% /
86.6%), a difference of a few swings, but ViTPose+-H's far-player keypoints jitter 15–30% less
from frame to frame ([docs/spikes/bigger-models.md](docs/spikes/bigger-models.md)). Depth along the camera's line of sight is the least certain part of a single-camera 3D
pose, especially for the far player: compare swings with your own rather than with absolute
norms. Details in [docs/m6-swings.md](docs/m6-swings.md).

### Stats and export

The **Stats** page (button on the session, Practice and Swings pages; *Stats* in the Library
row menu) sums up one session:

* **Headline numbers**: shots, in %, net %, median and fastest speed, distance moved, top
  running speed.
* **Speed by stroke**: every shot's speed off the racket as a dot, with a box for the middle
  half and the median (uncalibrated, see above).
* **Depth**: where shots came down, measured from the net, per stroke, with the service line
  and baseline marked.
* **Landings**: a heatmap (or dots colored by stroke, hollow when out) of where your shots
  landed, drawn as seen from your end so both ends add up.
* **Strokes**: calls, speed, depth (short of the service line for serves, of the baseline
  for the rest), share of deep groundstrokes, swing count and wrist speed per stroke.
* **Movement**: where you spent your time and the distance you covered every 5 minutes.

Filter by stroke and end. In practice sessions the numbers use your corrections from the
Practice page (shots marked *not a practice shot* are left out unless you include them).

**Export** (top right) downloads the session's shots (every shot record with its practice
result: call after your corrections, target, excluded), practice shots or swings, as CSV or
Parquet (Parquet keeps the units). `uv run sv export <session-id> --format parquet` does the
same from the command line.

### Profiles

Create a profile for yourself on **Profiles** (name, handedness, one- or two-handed
backhand, height) and pick it when creating a session (or on the session page). Height
places you correctly when your feet are out of frame and scales the 3D swing analysis;
handedness is the fallback when the footage doesn't show which hand holds the racket.

## When something goes wrong

The **Jobs** page shows each job's stages; *Show logs* adds the end of the job's log (the full
log is in the session folder under `logs/`). A job that stops says why:

* **Needs action: review the court calibration.** Click *Review calibration*, check the court
  overlay, adjust if needed and confirm; the job continues. To skip this step, see
  [Processing overnight](#processing-overnight).
* **Needs action: source video not found.** You moved or renamed the footage. Click *Relink
  video* (or *Relink video…* in the Library row menu, or the red *video missing* badge),
  pick the file where it is now, and the job resumes. The file must be the same video (its
  size and content are checked). If you moved a whole folder, relinking one session finds
  the others' videos in it too. Results and playback don't need the original video, only
  reprocessing does. Command line: `uv run sv relink <session-id> <new path>`.
* **Needs action: low disk space.** A job doesn't start with less than 5 GB free in the
  output folder. Free some space and click *Retry*.
* **Failed: the GPU ran out of memory.** Another program was using the GPU. GPU stages retry
  once on their own; if it happens again, close the other program and click *Retry*.
* **Failed: a GPU error / ffmpeg failed / permission denied.** *Retry* resumes where the job
  stopped. Permission errors usually mean another program (a video player, a spreadsheet)
  has one of the session's files open.

If a page shows *This page couldn't be shown*, one of the session's files is missing or
damaged (for example while it is being reprocessed): wait for the job, or reprocess the
session from the Library (*Process (stale stages)* or *Reprocess from scratch*).

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
uv run sv export <session-id> --what shots --format csv   # shots | practice | swings, csv | parquet
uv run sv relink <session-id> E:\footage\practice.mp4     # the video moved
```

## Recording guide

How you record matters more than any setting in the app. In order of importance:

**Camera position**

* Mount the camera **centered behind a baseline and as high as practical (2.5–3 m or more)**:
  a fence clamp or a light stand on the fence. Height is the biggest single factor in how
  well the far half of the court is measured; a low camera squashes it.
* Frame the **whole court**: all lines including the far baseline, some room behind the far
  baseline (you can't be tracked while standing behind a baseline at the edge of the
  picture), and **some sky above the far baseline**: a serve's toss and contact that leave
  the top of the frame get no speed and are found only from the landing and the sound.
* Keep yourself in the picture head to toe at both ends: swing analysis needs to see you.
* Make it rigid. A phone on a pole that sags or sways is handled (the drift check calibrates
  the minutes where it moved separately), but a steady camera is better.

**Phone settings**

* **4K at 60 fps.** The app reads the frame rate and resolution from the file; 4K60 is what
  it is tuned for.
* Lock exposure and focus if your camera app allows it, and use a fast shutter (1/1000 s or
  faster) in good light to keep the ball sharp.
* Keep the microphone uncovered: racket-impact sounds find hits the camera can't see.
* Storage: 4K60 HEVC takes about **25 GB per hour** (a 2-hour session ≈ 50 GB). Battery:
  plug the phone into a power bank for long sessions, and keep it out of direct sun so it
  doesn't overheat and stop recording.

**Light**

* Footage that gets too dark (dusk) is detected and skipped for tracking, so it costs nothing
  but processing time; stop recording when you can no longer see the ball.

**During the session**

* Start recording before you start hitting and don't touch the camera once it runs. If you
  do, the drift check notices and calibrates those minutes separately.
* One long recording is fine: a 2-hour file processes unattended overnight. Split recordings
  are fine too (one session each).

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
  analysis/          shot records (M4); practice segmentation, targets, accuracy (M5);
                     session statistics and exports (M7)
  pose/              2D pose, 3D lifting and placement, kinematics, swings, strokes (M6)
  training/          labels, labeling helpers, training, benchmark, evaluation (M3, M6)
  models/            pretrained weights registry (URLs, SHA-256, licenses)
  pipeline/          stage framework, DAG runner, worker process, stages/
  app/               Dash + Mantine UI (pages/, components/, assets/)
```
