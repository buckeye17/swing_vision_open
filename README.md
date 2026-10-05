# Swing Vision Open

A personal, local-first tennis video analysis tool. Point a fixed camera at the court, record
a practice or a match, and let your NVIDIA GPU break the footage down. The roadmap and
architecture are in [PLAN.md](PLAN.md).

**Status: milestones M0 (foundation) and M1 (court calibration) are complete.** The app
ingests footage, builds a browser-playable proxy, detects audio onsets, finds the court and
fits a full camera model (sub-pixel on real footage), checks whether the camera moved during
the recording, and lets you review and adjust the calibration. Ball and player tracking and
practice analytics arrive in M2–M7.

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

## CLI

```bash
uv run sv settings set-output-root D:\SwingVisionData
uv run sv create D:\footage\practice.mp4 --name "Basket forehands"   # creates + enqueues
uv run sv worker                       # process the queue in the foreground
uv run sv jobs                         # queue status
uv run sv process <session-id> --inline --force proxy   # rerun a stage in this process
uv run sv probe D:\footage\practice.mp4
uv run sv court detect D:\footage\practice.mp4 --out overlay.jpg  # court fit + overlay image
```

## Recording tips

* Mount the camera **centered behind a baseline and as high as practical (≥ 2.5–3 m)**. A
  low camera squashes the far half of the court and costs precision there.
* Make sure the camera sees as many court lines as possible, including the far baseline.
* Frame the entire court with some margin behind the far baseline.
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
  io/                ffprobe, ffmpeg runner, proxy encode, audio + onsets, frame grabs
  court/             court model, camera model, detection, calibration (M1)
  pipeline/          stage framework, DAG runner, worker process, stages/
  app/               Dash + Mantine UI (pages/, components/, assets/)
```
