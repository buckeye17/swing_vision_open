# Swing Vision Open

A personal, local-first tennis video analysis tool. Point a fixed camera at the court, record
a practice or a match, and let your NVIDIA GPU break the footage down. The roadmap and
architecture are in [PLAN.md](PLAN.md).

**Status: milestones M0–M3 are complete.** The app ingests footage, builds a
browser-playable proxy, detects audio onsets, finds the court and fits a full camera model
(sub-pixel on real footage), checks whether the camera moved during the recording, and lets
you review the calibration. It tracks you on the court (also at dusk) and reports your
movement: distance, speeds, a live court map and a heatmap. It tracks the ball on every frame
with a detector trained on your own labeled footage and finds hits, bounces (with their spot
on the court) and net contacts. Shot speeds and practice analytics arrive in M4–M7.

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

### Profiles

Create a profile for yourself on **Profiles** (name, handedness, one- or two-handed
backhand, height) and pick it when creating a session (or on the session page). Handedness
and backhand drive stroke classification later; height places you correctly when your feet
are out of frame and scales the 3D swing analysis.

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
```

## Recording tips

* Mount the camera **centered behind a baseline and as high as practical (≥ 2.5–3 m)**. A
  low camera squashes the far half of the court and costs precision there.
* Make sure the camera sees as many court lines as possible, including the far baseline.
* Frame the entire court with some margin behind the far baseline. If the far baseline sits
  at the top edge of the picture, you can't be tracked while standing behind it.
* Record 4K at 60 fps, with exposure and focus locked if your phone allows it. Avoid recording
  into darkness.
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
  ball/              ball detectors, frame-rate schedules, linking, events (M3)
  training/          labels, labeling helpers, training, benchmark, evaluation (M3)
  models/            pretrained weights registry (URLs, SHA-256, licenses)
  pipeline/          stage framework, DAG runner, worker process, stages/
  app/               Dash + Mantine UI (pages/, components/, assets/)
```
