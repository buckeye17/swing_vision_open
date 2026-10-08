# Swing Vision Open — Implementation Plan

A personal, local-first tennis video analysis tool. It ingests full-court footage from a fixed camera, runs computer vision on the local NVIDIA GPU, and presents results in a Dash Mantine Components web app.

---

## 1. Decisions captured

| Topic | Decision |
|---|---|
| Camera | Fixed, raised, behind one baseline, full court visible. Camera doesn't move during a recording. |
| Footage | 4K @ 60 fps (read fps/resolution from file; don't hard-code). |
| Recording length | 1–2 hours typical; overnight batch processing is acceptable. |
| Format | Singles only (≤ 2 players tracked). |
| Court calibration | Auto-detect court keypoints → user can drag to adjust → saved per video. |
| Player identity | Stored appearance profile (re-ID embeddings), matched automatically across videos. |
| Match scoring | CV infers point winners, user reviews/corrects in a timeline UI. |
| Match formats | Standard sets (ad/no-ad, tiebreaks), pro sets / first-to-N, match tiebreak, free play (no score). |
| Practice targets | Drawn on a top-down court diagram (rectangles/circles). |
| Segments | Timestamps only (no clip files); played in the in-app player. |
| Swing analysis | 3D pose lifting, stroke classification, swing phase timing. |
| Model training | User will label a few hundred frames; plan includes labeling UI + fine-tuning. |
| Hardware | NVIDIA RTX A5000 Laptop GPU, 16 GB VRAM, Windows 11. |
| Stack | Python 3.12 via `uv`, PyTorch (CUDA), Dash ≥ 3 + dash-mantine-components ≥ 2. |
| Priority | **Phase 1 (MVP) = practice mode**, including speed, movement, strokes, and 3D swing analysis. **Phase 2 = match mode.** Phase 3 = cross-session trends and polish. |

---

## 1a. Feature priorities

### Phase 1 — MVP: practice mode (one player on court, optionally a ball machine)

| Feature | Notes |
|---|---|
| App shell, settings, output folder, file-selection modal, job queue/worker | Foundation |
| Ingest, browser proxy, audio onsets | Foundation |
| Court calibration (auto + adjust) and full camera model incl. net points | Needed for landings and speed |
| Single-player detection/tracking, ball-machine handling | The only player in the court ROI is "me" (no re-ID needed) |
| Profile basics: name, handedness, 1H/2H backhand, height | Used by stroke rules and 3D pose scaling |
| Player movement (position, distance, speed, heatmaps) | |
| Ball detection, trajectory, hit/bounce/net events, labeling + fine-tuning | |
| Ball landing spots + **3D flight fit → speed**, net clearance, apex | |
| Practice segmentation: one segment per shot, grouped into blocks; auto hitting side | Sub-modes: self-feed, ball machine, serve practice |
| Targets drawn on court (relative/absolute, optional stroke filter) + accuracy tracking | Core practice deliverable |
| 2D pose → 3D lifting, kinematics, swing phases, stroke classification | |
| Session review, Practice, Swings, and per-session Stats pages; CSV/Parquet export | |

### Phase 2 — Match mode (human hitting partner)

| Feature | Notes |
|---|---|
| Two-player tracking (singles constraint), opponent tracking | |
| Appearance profiles: OSNet re-ID gallery, outfits, identity review flow | |
| Match segmentation: serve detection state machine, faults, lets, warm-up, match start marker | |
| Point outcome inference with confidence + line-call uncertainty | |
| Scoring engine (all formats), end-switching modes, server consistency checks | |
| Points review page (review queue, overrides, split/merge, live rescoring, lock) | |
| Match stats (serve %, points won on serve/return, rally length, opponent stats) | |

### Phase 3 — Later

Cross-session trends (`pyarrow.dataset`), annotated proxy render, TensorRT/performance pass, plus the §15 ideas.

**Forward compatibility rule for Phase 1:** schemas, the stage framework, and `session.json` must already carry match-mode fields (`mode`, `hitter`, `player_id`, `segment.kind`, `format`). That way Phase 2 adds stages and pages without migrating Phase 1 data. The stage DAG is mode-aware, so practice sessions skip match-only stages.

---

## 2. System overview

```
                ┌───────────────────────── Dash app (127.0.0.1) ─────────────────────────┐
                │ Library · New Session · Calibrate · Jobs · Review · Points · Practice   │
                │ Swings · Stats · Profiles · Labeling · Settings                         │
                └───────────────┬───────────────────────────────▲─────────────────────────┘
                     enqueue job │                               │ read parquet (pyarrow) / sqlite
                                 ▼                               │
                ┌──────────── Worker process (GPU) ──────────────┴───────┐
                │ Pipeline stages (resumable, versioned, cached)          │
                │ ingest → court → pass1 (players+ball) → identity →      │
                │ events → shots/3D ball → pass2 (pose) → swings →        │
                │ segmentation → outcomes/scoring | practice → stats      │
                └──────────────────────────┬──────────────────────────────┘
                                           ▼
                     <output_root>/  (user-selected external folder)
```

* **App process**: Dash UI only. It never runs heavy CV in callbacks.
* **Worker process**: started by the app (or `sv worker`). It pulls jobs from a SQLite queue in the output root and runs pipeline stages on the GPU.
* **Storage**: Parquet files written and read with **PyArrow** for per-frame and per-event data, JSON for configs and calibration, SQLite for the library index, job queue, and profiles. Cross-session queries use `pyarrow.dataset`. No pandas or polars (see §5.1).
* **CLI** (`sv`): headless equivalents of everything (process, train, evaluate, run app).

---

## 3. Technology choices

| Concern | Choice | Notes |
|---|---|---|
| Env / packaging | `uv`, `pyproject.toml`, `.python-version` = 3.12 | System Python is 3.8; uv manages 3.12. |
| Deep learning | `torch`, `torchvision` from the PyTorch CUDA 12.8 index | Configured with `[tool.uv.index]` + `[tool.uv.sources]`. |
| Video decode | `FrameSource` abstraction: **PyNvVideoCodec** (NVDEC → GPU tensor via DLPack) primary; **PyAV** CPU fallback | ✅ M0 spike: 305 fps of upright 4K RGB on the GPU (PyAV hwaccel 81, CPU 29). See `docs/spikes/m0-video-pipeline.md`. |
| Probe / audio / proxy | `ffmpeg`/`ffprobe` binaries (path set in Settings) | NVENC for 720p proxy encode. |
| Person detection | ✅ Ultralytics YOLO11m on the court-ROI crop at 1920 px, 15 Hz, raw model on GPU tensors | M2: 1920 px finds the far player's legs at the frame edge (5/5 frames vs 2/5 at 1280). AGPL, fine for personal use. Weights pinned by SHA-256 in `models/registry.py`. |
| Multi-object tracking | ✅ Own court-space tracker: Hungarian frame-to-frame association (speed + σ + physical-height gates) and a tracklet-chain DP for "me" | M2: simpler and more robust for one player than BoT-SORT; Phase 2 adds the two-player constraint and re-ID (`boxmot`/OSNet then). See `docs/m2-player-tracking.md`. |
| Re-ID | OSNet embeddings (via `boxmot` weights) | Appearance profile gallery. |
| Ball detection | ✅ **Slim** TrackNet-style heatmap U-Net (frames t−2, t, t+2; width 32; heatmap at ½ input), trained on own labels at 1920 px wide (half of 4K); a classical motion detector (no training) and a COCO YOLO's sports-ball class as alternatives | M0 spike: a TrackNet-size net at 1280×720 runs at only 20 fps, the slim variant at 190. M3: 1920 px is needed for the far ball; see `docs/m3-ball-tracking.md` for the benchmark. |
| Court keypoints | ✅ Classical line detector: white top-hat → Hough → court-model hypothesis search → sub-pixel ridge refinement + full camera fit | M1: needs no training data and handles low corner cameras, wide lenses, partly visible courts, pickleball lines and neighboring courts. A keypoint CNN fine-tuned on confirmed calibrations stays an option if a view ever defeats it. See `docs/m1-court-calibration.md`. |
| 2D pose | ✅ ViTPose+-H (HF `transformers` `VitPoseForPoseEstimation`, COCO expert of the MoE backbone) on 4K player crops, FP16 + flip test; ViTPose-B selectable | M6 shipped ViTPose-B (≈110–150 crops/s with decode). 2026-10-08: ViTPose+-H is the default for its steadier keypoints (far-player jitter −15–30%, half the left/right swaps), at ≈35 crops/s with decode (pose ≈1 h per footage hour); see `docs/spikes/bigger-models.md`. Weights pinned by SHA-256 in the registry. RTMPose via `rtmlib` stays the alternative. |
| 3D lifting | ✅ MotionBERT-Lite (vendored DSTformer, Apache-2.0, in-the-wild checkpoint) | COCO-17 → H36M-17 joint mapping; confident joints snapped back onto their 2D rays (M6, see `docs/m6-swings.md`). |
| Audio onsets | `librosa` | Racket-impact sounds help confirm hits. |
| Classical ML | `scikit-learn`, `lightgbm` | Bounce classifier, stroke classifier baseline. |
| Optimization | `scipy.optimize.least_squares` | Camera calibration, 3D ball trajectory fits. |
| Data | `pyarrow` (Parquet I/O, `pyarrow.compute`, `pyarrow.dataset`), `numpy`, `pydantic` v2 | No pandas or polars in project code (see §5.1). |
| UI | `dash>=3`, `dash-mantine-components>=2`, `dash-iconify`, `plotly` | Native `html.Video` + small clientside JS for sync. |
| CLI / misc | `typer`, `platformdirs`, `loguru`, `filelock` | |
| Dev | `ruff`, `pytest`, `pytest-cov`, `pre-commit`, `pyright` (basic) | |

> Check each pretrained model's license when vendoring/downloading weights; record it in `models/registry.py`.

`pyproject.toml` torch index snippet:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

---

## 4. Repository layout

```
swing_vision_open/
├─ pyproject.toml  uv.lock  .python-version  README.md  PLAN.md
├─ src/swingvision/
│  ├─ cli.py                    # typer: app, worker, process, train, eval, models
│  ├─ settings.py               # app settings (platformdirs), output_root, ffmpeg path
│  ├─ models/registry.py        # weight URLs, checksums, licenses, download/cache
│  ├─ io/
│  │  ├─ probe.py               # ffprobe → VideoInfo (fps, size, duration, rotation, codec)
│  │  ├─ frames.py              # FrameSource (NVDEC / PyAV), batched, seek-to-window
│  │  ├─ audio.py               # extract wav, onset detection
│  │  └─ proxy.py               # NVENC 720p proxy (+ optional annotated proxy)
│  ├─ storage/
│  │  ├─ library.py             # SQLite: sessions, jobs, profiles
│  │  ├─ session.py             # Session dir API, manifests, pyarrow.parquet read/write
│  │  ├─ schemas.py             # pydantic models (JSON) + pyarrow schemas (Parquet)
│  │  ├─ tables.py              # Arrow helpers: write/read, append parts, to_numpy, to_rows
│  │  └─ edits.py               # user overrides layered over derived data
│  ├─ court/
│  │  ├─ model.py               # ITF court geometry (meters), keypoints, lines, zones
│  │  ├─ detect.py              # line response, hypotheses, ridge refinement
│  │  ├─ homography.py          # image↔court plane, uncertainty (Jacobian)
│  │  ├─ camera.py              # pinhole + division-model distortion, PnP/line fits
│  │  └─ calibration.py         # frame sampling, drift windows, editor solves, files
│  ├─ players/
│  │  ├─ detect.py  track.py    # YOLO on ROI crops; court-space tracklets, static objects, "me" chain
│  │  ├─ reid.py                # OSNet embeddings, tracklet ↔ profile matching
│  │  └─ movement.py            # feet → court coords, smoothing, speed/distance/heatmaps
│  ├─ ball/
│  │  ├─ detectors/             # pluggable BallDetector implementations (slim U-Net, TrackNetV3, YOLO, …)
│  │  ├─ schedule.py            # adaptive frame-rate plan: sparse sweep → full-rate event windows
│  │  ├─ detect.py              # runs a detector over a schedule on court-ROI crops, background model
│  │  ├─ trajectory.py          # outlier rejection, gap fill, sub-tracks
│  │  ├─ events.py              # hit / bounce / net candidates + classifiers + audio
│  │  └─ physics.py             # 3D flight fit (drag + optional Magnus) → speed etc.
│  ├─ pose/
│  │  ├─ pose2d.py  lift3d.py   # ViTPose crops, MotionBERT lifting, world placement
│  │  ├─ kinematics.py          # joint angles, segment rotations, angular velocities
│  │  ├─ phases.py              # swing phase segmentation
│  │  └─ strokes.py             # rules + learned stroke classifier
│  ├─ analysis/
│  │  ├─ shots.py               # assemble Shot records
│  │  ├─ segmentation.py        # match points / practice shots & blocks
│  │  ├─ outcomes.py            # point-ending reason + winner inference + confidence
│  │  ├─ practice.py            # targets, accuracy metrics
│  │  └─ stats.py               # session + cross-session aggregates
│  ├─ scoring/
│  │  ├─ formats.py             # MatchFormat config + presets
│  │  └─ engine.py              # pure score state machine (server, ends, tiebreaks)
│  ├─ pipeline/
│  │  ├─ stage.py               # Stage base: version, deps, config hash, chunking
│  │  ├─ stages/*.py            # one module per stage
│  │  ├─ runner.py              # DAG resolution, resume, invalidation
│  │  └─ worker.py              # job loop, progress, cancel, GPU memory hygiene
│  ├─ training/
│  │  ├─ labels.py              # label stores (ball, court, events, strokes)
│  │  ├─ bench_ball.py          # ball-detector comparison: accuracy × cost on labelled clips
│  │  ├─ train_ball.py  train_court.py  train_events.py  train_strokes.py
│  │  └─ evaluate.py            # metrics vs ground truth
│  └─ app/
│     ├─ main.py  layout.py  server_routes.py   # Flask media routes (range requests)
│     ├─ components/            # file_browser, video_player, court_diagram,
│     │                         # calib_editor, skeleton3d, timeline, job_card
│     ├─ pages/                 # library, new_session, calibrate, jobs, review,
│     │                         # points, practice, swings, stats, profiles,
│     │                         # labeling, settings
│     └─ assets/                # video_sync.js, keyboard shortcuts, css
├─ tests/                       # unit + synthetic + golden-clip regression
└─ scripts/                     # benchmarks, one-off utilities
```

---

## 5. Output folder (external, chosen in UI)

```
<output_root>/
├─ library.sqlite                     # sessions, jobs queue, profiles, settings snapshot
├─ profiles/<profile_id>/
│  ├─ (name, handedness, height, backhand live in library.sqlite → profiles)
│  ├─ reid_gallery.npy  gallery.json  # embeddings grouped by "outfit"
│  └─ thumbs/
├─ sessions/<YYYY-MM-DD>_<slug>_<shortid>/
│  ├─ session.json                    # source path + fast hash, mode, format, targets, players
│  ├─ calibration.json                # camera (K, k1/k2, R|t), keypoints, metrics, drift windows, confirmed_by
│  ├─ court/background.jpg  auto.json  user.json  windows/wNN.jpg   # detection inputs/results, user edits
│  ├─ proxy_720p.mp4                  # browser playback (H.264, NVENC)
│  ├─ audio_onsets.parquet
│  ├─ manifests/<stage>.json          # stage version, config hash, input hashes, timing, status
│  ├─ pass1/frames.parquet             # sampled frames: time, brightness, view check
│  ├─ players/detections.parquet  tracks.parquet  identity.json  movement.parquet
│  ├─ ball/sweep.parquet  track.parquet  flights.parquet  flight_paths.parquet
│  ├─ events.parquet                  # hits, bounces, net, serve_toss (frame, xy, conf, source)
│  ├─ shots.parquet                   # one row per shot (see §8)
│  ├─ pose/pose2d.parquet  pose3d.parquet  swings.parquet
│  ├─ segments.parquet                # points (match) or shots/blocks (practice)
│  ├─ points.parquet  score_log.parquet   (match)
│  ├─ practice.parquet                (practice: per-shot accuracy)
│  ├─ stats.json
│  ├─ edits.json                      # user overrides (never mutate derived files)
│  └─ logs/
├─ training/{ball,court,events,strokes}/   # exported labels from UI + corrections
└─ models/                            # fine-tuned weights (base weights in app data dir)
```

Principles:

* Raw footage is **never copied**. The session stores its absolute path plus a fast content hash (size + hash of the first/last 8 MB) so it can be relinked if moved.
* Detections are stored in **image coordinates**, and court-coordinate data is derived from them. Recalibrating then reruns only the cheap projection and analysis stages, not the GPU passes.
* User edits live in `edits.json` and are applied on read. They're also exported as training labels.

### 5.1 Tabular data conventions (PyArrow only)

* **In-memory type**: `pyarrow.Table` is the only tabular type passed between modules. Numeric work (geometry, smoothing, fits) converts columns with `Table.column(...).to_numpy()` and builds results back with `pa.table({...}, schema=...)`.
* **Explicit schemas**: every Parquet file has a `pa.schema` defined in `storage/schemas.py`, with units in the field metadata (e.g. `{"unit": "m"}`) plus a `schema_version` in the file metadata. Writers validate against the schema, so there's no type inference.
  * Fixed-size arrays use `pa.list_(pa.float32(), n)`. For example, pose keypoints are `list<float32>[17*3]` per player-frame.
* **Writing**: `pyarrow.parquet.write_table` with `compression="zstd"`. Per-frame data uses sensible row-group sizes (e.g. one row group per 2-minute chunk).
  * Chunked GPU stages write one part file per chunk (`work/pass1_detect/det/part-00007.parquet`), which matches the resume checkpoints, and consolidate them into one file at the end.
  * Writes are atomic: write to a temp file, then rename.
* **Reading**: `pq.read_table(path, columns=[...], filters=[...])` reads only the columns and rows needed. Part-file directories are read with `pyarrow.dataset.dataset(dir)`.
* **Transforms**: filtering, joins, group-by, and aggregation use `pyarrow.compute` and `Table.join`/`Table.group_by`. When logic is clearer in Python (state machines, scoring), it iterates over `to_pylist()` or NumPy arrays.
* **Cross-session queries**: one `pyarrow.dataset` over `sessions/*/shots.parquet` (and similar), with a `session_id` column. These feed Phase 3 trends.
* **UI boundary**: Plotly figures use `plotly.graph_objects` fed with NumPy arrays or lists. Plotly Express needs a DataFrame, so it isn't used. Tables in the UI get `table.to_pylist()`. CSV export uses `pyarrow.csv.write_csv`.
* **Enforcement**: a ruff `flake8-tidy-imports` banned-API rule (`TID251`) blocks `import pandas`/`import polars` in `src/`. Transitive dependencies (e.g. Ultralytics) may still install pandas, and that's fine as long as project code never uses it.

---

## 6. Pipeline stages

Each stage declares `name`, `VERSION`, `depends_on`, and config keys. It writes outputs plus a manifest. The runner skips a stage when its manifest matches (version + config hash + upstream hashes). Heavy stages process in **chunks** (e.g. 2-minute windows) with per-chunk checkpoints, so an overnight job that crashes resumes mid-video.

| # | Stage | GPU | Phase | Inputs → Outputs |
|---|---|---|---|---|
| 1 | `ingest` | – | 1 | probe, fast hash, audio wav, `session.json` |
| 2 | `proxy` | NVENC | 1 | 720p H.264 proxy for browser playback |
| 3 | `audio_onsets` | – | 1 | onset times + strength + spectral features |
| 4 | `court_auto` | – | 1 | median background per time window, court detection on the dominant camera position, per-window drift check → `court/auto.json` |
| 5 | `camera` | – | 1 | calibration gate: user-confirmed (`court/user.json`) or auto-accepted calibration → `calibration.json` |
| 6 | `pass1_detect` | ✓ | 1 | **single decode pass**: person detections at 15 Hz (✅ M2) + ball candidates on every frame or at the sweep rate (✅ M3). Runs *after* `camera` (crops to the court) but only the 64 px-snapped crops are in its fingerprint, so recalibrating doesn't rerun it; the persons part has its own fingerprint, so changing the ball detector keeps the person boxes |
| 7 | `players_track` | – | 1 (single player) / 2 (two players) | ✅ court positions, ROI, tracklets, static objects/ball machine, the "me" chain. Phase 2 adds the top-2 singles constraint |
| 8 | `identity` | ✓ (light) | 2 | tracklet ↔ profile ("me" / "opponent"), confidence. In Phase 1 the only tracked player is "me" |
| 9 | `movement` | – | 1 | ✅ smoothed position/velocity per player (Kalman/RTS with per-point ground σ), short gaps bridged |
| 10a | `ball_refine` | ✓ | 1 | ✅ with a sweep rate: full-rate detection in windows around moments found from the sweep (events, strong audio onsets, track gaps) |
| 10 | `ball_track` | – | 1 | ✅ linked trajectory (tracklets + Viterbi selection of the ball in play), outliers dropped, short gaps filled |
| 11 | `events` | – | 1 | ✅ hits (hitter = me / machine), bounces with court position, net contacts, sub-frame contact time, audio match (serves are recognized from the pose in `swings`, M6) |
| 12 | `ball_3d` | – | 1 | ✅ 3D fit of every flight between events (hit/bounce → bounce/net/hit): speeds, net clearance, apex, landing, spin sign, uncertainties; contact chained from the incoming flight; rejects hits not at the hitter → `ball/flights.parquet` |
| 13 | `pass2_pose` | ✓ | 1 | ✅ 2D pose on 4K crops at full fps from 1.8 s before to 1.2 s after each of the player's hits and strong impact sounds (unseen contacts); sparse pose elsewhere not yet (nothing uses it) |
| 14 | `pose3d` | ✓ | 1 | ✅ MotionBERT lifting, height scaling, court placement (rays + ground + tracked feet), joints snapped to their 2D rays |
| 15 | `swings` | – | 1 | ✅ racket hand from the pose, swings at hits / speed peaks / impact sounds, contact from the pose, phases, kinematics, stroke (rules; learned model when validated), user corrections |
| 16 | `shots` | – | 1 | ✅ join events + ball_3d + swings → `shots.parquet` (M4: events + ball_3d, line calls; M6: stroke, `is_serve`, `swing_id`). Runs after `swings` |
| 17 | `segments` | – | 1 (practice) / 2 (match) | ✅ Practice (M5): one segment per shot (seen hits, unseen contacts from landing + impact sound, toss + sound), feeds, serve/groundstroke, deuce/ad, blocks. Match: points (+ warm-up) |
| 18 | `outcomes` + `scoring` | – | 2 | match only: point winner, reason, confidence, score log |
| 19 | `practice_eval` | – | 1 | ✅ practice only: line calls (service box for serves), per-shot target hit/miss, distance, depth/width error, with the user's edits; rerun in-app after target/shot edits |
| 20 | `stats` | – | 1 (session) / 2 (match stats) | ✅ session aggregates → `stats.json` (M7: shots and calls by stroke, speeds, depth, swings, movement); rerun in-app after edits |

**Calibration gating**: stages 4–5 run automatically. `court_auto` runs right after `ingest`, so the calibration can be reviewed while the proxy encodes. Unless the auto calibration passes the Settings threshold ("continue without review when line RMS < X px", off by default), `camera` stops the job with status `needs_action`; Jobs and the session page link to the Calibrate page, and confirming there re-queues the job. Stage 6 doesn't depend on calibration except for the court-ROI crop. The `camera` stage's fingerprint covers only the chosen camera, so re-confirming an unchanged calibration invalidates nothing; if the user later adjusts calibration, stages 7+ rerun on CPU in minutes. (The runner re-plans each stage just before running it, so a stage's config may read files that upstream stages or the user wrote.)

**Throughput** for 1 h of 4K60 (216k frames), measured in the M0 spike (`docs/spikes/m0-video-pipeline.md`): PyNvVideoCodec decode 305 fps; YOLO11-m @1280 FP16 104 img/s; slim ball U-Net @1280×720 190 img/s. *M2 measured:* YOLO11m at the 1920 px input the far player needs runs 46 img/s, so persons run at 15 Hz; the person pass alone is **0.52× realtime (≈31 min per hour)**. *M3 measured:* the slim ball U-Net at 1920 px on every frame runs at 0.84× realtime (≈50 min per hour); a 15 Hz sweep with full-rate windows saved nothing on a whole practice session, so pass 1 is ≈1.4× realtime when both detectors run (person boxes are reused when only the ball model changes). Pass 1 is ≈75–90 fps (**≈45 min per hour**). Add the proxy (≈11 min per hour), pose windows (≈30–40% of frames), and CPU stages. Expected total is **≈1.5–2.5 h per hour of footage**, which fits overnight. NVDEC/NVENC are shared, so GPU stages run one at a time. Optimizations if needed: TensorRT, skipping dead time (ball collection) using frame differencing and audio-onset density.

---

## 7. Core algorithms

### 7.1 Court calibration and camera model

✅ Built in M1 (details and measurements in `docs/m1-court-calibration.md`):

1. **Background images**: the video is split into ≤ 12 time windows (default 5 min). Five frames per window are decoded at full resolution, too-dark frames are skipped, and the per-pixel median removes the players and balls.
2. **Line response**: white top-hat of `min(R, G, B)`. White paint is bright in every channel; green/red surfaces and teal pickleball lines are not. Blobs wider than any painted line (sky, water, buildings) are removed before line finding.
3. **Hypotheses**: Hough lines merged into long lines, split into an across (baseline-like) and an along (sideline-like) family. Every pair × pair of image lines is matched to every ordered pair of model lines (batched 4-point homographies), and scored by projected line length on line pixels minus length off them. Hypotheses whose decomposed camera isn't above the court are rejected.
4. **Refinement**: the best hypotheses are refined by sampling the response perpendicular to each projected model line (the search window adapts to the on-screen line width) and taking the sub-pixel line center, with ambiguity and outlier rejection, then fitting the camera to those samples. The window shrinks over iterations.
5. **Full camera model** (needed for 3D ball flight and pose orientation):
   * Pinhole with square pixels; focal length initialized from the homography's orthonormality constraints.
   * Lens distortion with the division model (`k1`, `k2`; closed-form undistortion), fitted from line straightness.
   * Principal point free with a prior: the user's phone footage needs it (in-camera stabilization/lens correction), dropping line RMS from 4.2 to about 1.2–1.5 px on the Oct 4 video.
   * Non-planar evidence from the **net tape** (post tops at 1.07 m, center strap 0.914 m) fitted as a line, plus the editor's draggable net-post-top and strap points for PnP.
6. Per-point uncertainty: `court.homography.ground_sigma` propagates ±1 px through the image→ground Jacobian to σ (meters) for any court location. This feeds line-call confidence.
7. **Drift check**: the camera is pose-refined on every window. If windows disagree, the session is calibrated on the position held longest, and windows where the view shifted keep their own camera (`calibration.camera_at(cal, t)`, piecewise by time). Both of the user's first sessions show real camera movement in the first 10–15 minutes.

Court model (`court/model.py`): ITF dimensions, 23.77 × 10.97 m (singles 8.23 m), service line 6.40 m from net, line width 5 cm. Named zones: service boxes, deuce/ad, no-man's land, alleys. Includes a helper to **mirror coordinates by hitting side**.

### 7.2 Players: detection, tracking, identity

> **Phase 1** covers ROI filtering and tracking of the single player, who is assumed to be "me", plus ball-machine handling. The singles two-player constraint and appearance re-ID are **Phase 2**.

* Use YOLO person detections (15 Hz). Keep detections whose foot point (bbox bottom center, later ankle midpoint from pose) projects inside the **court ROI** (court + 6 m behind baselines + 3.5 m beside sidelines; neighboring courts start ≈3.7 m beside). This excludes neighboring courts and spectators. Frames that don't show the calibrated view (camera being handled) are ignored.
* ✅ Phase 1 (M2): tracklets by court-space Hungarian association gated by speed, ground σ and physical height; motionless short objects (ball machine, bags) are flagged static; "me" is the best chain of tracklets (confidence gained, physically possible links, restart penalty). Details in `docs/m2-player-tracking.md`.
* BoT-SORT tracking, then **singles constraint**: at most one "near-side" and one "far-side" player while in play. Short tracklets are stitched by position and time continuity plus appearance similarity.
* **Identity**: compute OSNet embeddings for sampled crops per tracklet and average them. Match tracklets to the "me" profile gallery (cosine similarity, max over outfits). Use Hungarian assignment between the two players for each continuous play period, so the higher-similarity player is "me".
  * If confidence is low (e.g. a new outfit), the session gets the flag `identity_review`. The UI shows thumbnails of both players, the user clicks themself once, and the new outfit embeddings are added to the gallery.
  * The opponent can optionally be a named profile too (for stats against a regular partner).
* **Ball machine**: in practice/ball-machine mode, the static machine is detected as a non-person stationary source of shots. Its location is either auto-detected from the origin of feed trajectories or marked by the user.

### 7.3 Movement

* Foot position comes from the ankle midpoint (pose) when available, otherwise the bbox bottom center. Project with `H`, then smooth with a constant-velocity Kalman/RTS smoother.
* Metrics: distance covered (total, per point/drill), speed and acceleration profiles, max sprint speed, court-position heatmaps, average position by context (serving/returning/rallying), recovery position and time after each shot, and distance-to-ball at contact.

### 7.4 Ball detection and trajectory

* **Crop to the court ROI** (court bbox projected in 4K with a margin above the image for lobs), then resize to the model input (default 1280×720, configurable). The far-court ball is ~6–8 px in 4K, and naive 512×288 downscaling would erase it.
* TrackNet-style network: 3 consecutive frames + a background median image, outputting a heatmap. Peaks are found with subpixel refinement (centroid on the heatmap), and multiple candidates per frame are kept with scores.
* Trajectory linking: a dynamic program / Kalman multi-hypothesis over candidates penalizes physically implausible jumps. Outliers are rejected, short gaps (≤ ~8 frames) are filled with local parabolic fits, and the track is split into **sub-tracks** at hits and bounces.
* A labeling-driven fine-tuning loop is in §10.

#### 7.4.1 Pluggable detectors and model comparison

Ball detection is expected to be the hardest part of the system, so it's built to try several
models side by side instead of committing to one.

* **`BallDetector` interface** (`ball/detectors/`): `prepare(video_info, roi)`, `frames_needed`
  (temporal context, e.g. 3 for TrackNet), `input_size`, and
  `detect(batch) → candidates (frame, x, y, score)` in full-resolution image coordinates.
  Detectors are registered by name with their weights, so a config such as
  `ball.detector = "tracknetv3@512x288"` picks one.
* **Candidates to compare** (initial list):
  * slim U-Net at about 1280 px (≈190 fps)
  * full TrackNetV3 at 512×288 and on court-ROI crops
  * YOLO11-s/m with a single ball class at 1920 px, or tiled (SAHI-style)
  * any of the above fine-tuned on the user's labels, vs pretrained only
* **Benchmark harness** (`sv bench ball`, `training/bench_ball.py`):
  * Runs each detector, *and each frame-rate schedule*, over the same held-out labelled clips
    and stores every run's raw candidates as Parquet, so trajectory linking and events can be
    re-scored without re-running models.
  * Metrics, split by court zone (near/far half) and ball state (in flight, near racket, at
    bounce):
    * detection precision/recall/F1 within 4 px and position error
    * track continuity (gaps, ID switches)
    * downstream hit/bounce event F1 and bounce landing error in meters
  * **Cost** in GPU-seconds per footage-hour, measured from the same runs.
  * Output: a comparison table plus an **accuracy-vs-cost Pareto chart** (HTML report), so
    detector choice is a measured trade-off.
* **Ground truth for a fair comparison** must be labelled at **full frame rate**. That means
  contiguous clips covering serves, rallies, near and far bounces, lobs, and dark/backlit
  spans, not just sampled frames. The keyframe-interpolation labeling tool (§10) makes this
  affordable.

#### 7.4.2 Adaptive frame-rate scheduling

Bigger, more accurate models can run at a **lower frame rate across the whole video** and then
at **full frame rate only around important moments**:

1. **Sweep**: run the detector at a reduced rate (configurable; e.g. every 4th frame = 15 Hz)
   over the full video. TrackNet-style models still get their 3-frame context from adjacent
   frames, decoded but not otherwise processed.
2. **Find moments** from the sweep plus cheap signals:
   * trajectory direction changes (candidate hits/bounces)
   * ball near a player's racket-side wrist
   * audio onsets (sound-delay compensated)
   * gaps where the sweep lost the ball
3. **Refine**: decode windows around each moment (e.g. −0.25 s … +0.35 s) and run the detector
   at the **full 60 fps**. Optionally use a bigger model or a tighter, higher-resolution crop
   around the predicted ball position. Windows are merged and processed in time order so
   NVDEC seeks stay cheap.
4. **Fill**: in between, the trajectory comes from sweep detections plus parabolic/physics
   interpolation (§7.6). That's sufficient in free flight, where nothing interesting happens
   between samples.

Schedules are first-class benchmark subjects: `{model} × {sweep rate} × {window size}` all run
through the same harness. This answers "is a big model at 15 Hz + full-rate windows better
than a slim model at 60 Hz?" with numbers. In the pipeline this splits ball detection into a
sweep in pass 1 and a `ball_refine` stage, which links the sweep and finds events itself to
place its windows; `ball_track` and `events` then run on the merged detections.
(✅ M3: see `docs/m3-ball-tracking.md` for what the measurements showed.)

### 7.5 Events: hits, bounces, net

* **Hit candidates**: sharp change in image velocity direction or magnitude while the ball is near a player's bbox or wrist, especially reversing toward the other side.
  * Fused with audio onsets: a racket impact is a sharp broadband onset. The expected audio delay is `distance(camera, hitter)/343 m/s`, about **70 ms (~4 frames) for the far player**, which is compensated before matching.
* **Bounce candidates**: change in the sign of vertical image acceleration plus a speed drop, away from players. The bounce point is refined by intersecting the fitted pre- and post-bounce curves, giving a subpixel ground contact.
* A **LightGBM event classifier** over trajectory windows (positions, velocities, accelerations, player proximity, audio features) is trained on labeled and corrected events, with rule-based candidates as the bootstrap.
* **Net**: the trajectory ends or reverses near the projected net line without crossing, or the 3D fit (7.6) crosses the net plane below net height.
* **Serve candidate**: a player stationary behind the baseline, ball rising above the head (toss), and the wrist high at contact. This is also used by segmentation.

### 7.6 3D ball flight, speed, and landing

The homography alone only gives positions on the ground. For speed:

* For each shot sub-track (hit → bounce, or hit → next hit for volleys), fit a **physical 3D trajectory** with gravity + quadratic drag (+ optional Magnus with 1–2 spin parameters) by minimizing 2D reprojection error of the detected ball positions through the calibrated camera.
* Constraints and priors:
  * Start point near the hitter's contact location: court position from feet plus contact height from the wrist in 3D pose (serve ~2.5–3 m, groundstroke ~0.5–1.3 m).
  * End point on the ground (z = ball radius) at the detected bounce.
  * Ball mass 57 g, diameter 6.7 cm, Cd ≈ 0.55.
* Outputs per shot: **speed off the racket**, speed at net, speed before bounce, average speed, **net clearance**, apex height, flight time, landing point (from the bounce detection, which is more precise than the 3D fit), and an optional rough spin estimate (sign of Magnus = topspin/slice).
* Each output has an uncertainty from the fit covariance.
* **Validation**: compare with known ball-machine speed settings and/or a radar gun on a few sessions.

✅ Built in M4 (details and measurements in `docs/m4-ball-3d.md`): every flight between two events is fitted (not just shots), in time order, so a hit's contact prior is where its incoming flight (a feed's bounce, a serve toss) ended, with that fit's covariance, plus the tracked feet (± 1 m) until pose arrives in M6. Drag C_d is fitted with a 0.55 ± 0.05 prior, Magnus with one coefficient around a horizontal axis. A linear drag-free solve (ray constraints are linear in p₀, v₀) starts each fit; the Jacobian and the uncertainty samples are integrated as one batch. Hits whose well-fitted flight starts > 2.5 m from the hitter are rejected (M3 false hits) and the flights re-planned. Without a radar, the speed scale is validated by refitting ground-to-ground flights with g free (9.87 m/s² on real footage, +0.6%).

### 7.7 Pose: 2D, 3D, kinematics, phases, strokes

* **2D pose**: ViTPose on **4K crops** of each player (the far player is still ~150–250 px tall). Run at full fps in windows of ± 1.5 s around each hit, and sparsely otherwise.
* **3D lifting**: MotionBERT on 2D sequences (padded/strided windows), giving root-relative 3D joints in the camera frame. These are rotated into the court frame with the camera extrinsics and translated to the court position from the feet. Bone lengths are scaled to the profile's height.
  * Caveat: depth along the camera axis is the least reliable, especially for the far player. Every metric carries a quality flag (near/far, keypoint confidence).
* **Kinematics** (per frame, per swing):
  * Knee flexion, hip and shoulder rotation relative to the baseline, **hip-shoulder separation**, trunk lean, elbow angle.
  * Wrist (racket hand) linear speed, as a proxy for racket speed.
  * Contact point relative to the body: in front/beside, height, distance.
  * Stance width, center-of-mass height drop.
  * **Kinetic chain sequencing**: timing of peak angular velocity for hips → trunk → shoulder → elbow → wrist.
* **Swing phases** (from 3D hand trajectory relative to the trunk, plus trunk rotation):
  1. Ready / split step (if detected: vertical CoM bounce before the opponent's contact)
  2. Unit turn / preparation: shoulder rotation away from the net begins
  3. Backswing: until the hand's maximum posterior displacement
  4. Forward swing: until contact (hit frame)
  5. Follow-through: until hand speed drops below a threshold
  6. Recovery: until the player resumes ready position or moves toward center

  Phase durations, tempo ratios, and contact timing are stored.

  ✅ Built in M6 (details and measurements in `docs/m6-swings.md`): pose in windows around
  hits and impact sounds; MotionBERT depth with the 2D detector's image-plane motion (joints
  snapped onto their rays), since MotionBERT alone smooths a serve's wrist to a third of its
  speed; placement held to the tracked feet (the far player otherwise drifts meters toward the
  camera); racket hand from the pose (a fast hand peaking over the head swings, a still one
  tosses); swings also from wrist-speed peaks and impact sounds; serve phases from the toss,
  the racket drop (tightest elbow bend) and the wrist coming down.
* **Stroke classification**: classes are serve, forehand, backhand, forehand volley, backhand volley, overhead (plus `other`).
  * Rules bootstrap: contact side relative to the dominant hand from 3D pose; volley = no bounce since the previous hit and the player inside the service line; serve = the point-start hit by the server; overhead = contact above the head mid-rally.
  * Learned model: a small temporal network (GRU or ST-GCN) over a ±0.6 s pose window + ball features, trained on rule labels + user corrections. The model replaces the rules once validated.
  * ✅ M6: rules in the hitter's frame (a serve needs a toss, or an overhead behind the baseline with no ball coming in; *other* for no ball contact, slow or short wrist travel, or a toss windup; strokes at least 1 s apart); a bidirectional GRU (`sv train strokes`) validated by leaving one session out and used only for the classes it learned. On the serve-practice sessions the GRU doesn't beat the rules yet.
  * Handedness and 1H/2H backhand come from the profile.

### 7.8 Segmentation

**Match mode** uses a per-frame state machine: `IDLE → SERVE_SETUP → IN_PLAY → DEAD`.

* `SERVE_SETUP`: a serve candidate (toss + server behind baseline).
* `IN_PLAY`: from serve contact through subsequent hits.
* `DEAD`: double bounce, ball out with no return within T, net with no recovery, or no ball activity for T seconds while players walk.
* **A point includes a first-serve fault plus the second serve.** Lets are detected heuristically (serve clips net, lands in box, play stops) and flagged.
* Rallies before the first serve, or explicitly marked by the user, become **warm-up** segments. The user can set a "match starts here" marker.
* Segment = `[first_event − pad_before, last_event + pad_after]`, with configurable padding (default 2 s / 2 s).

**Practice mode**: each segment is **one shot** (a feed or self-drop → hit → landing/net). Shots are grouped into **blocks** separated by long gaps (ball collection) or a change of hitting side.

* Sub-modes:
  * **Self-feed / basket**: hitting side auto-detected per shot, so the player can switch ends.
  * **Ball machine**: one fixed hitting side. Machine feeds are recognized as not-the-user's shots and reported as feed consistency.
  * **Serve practice**: serve targets in service boxes.

✅ Built in M5 (details and measurements in `docs/m5-practice.md`): shots come from the player's hits whose ball crosses the net (or the net; or drops back behind it), from **unseen contacts** (the first bounce opposite the player plus the impact sound, after a per-session audio/video offset; a serve's contact is often above the frame) and, at dusk, from a **toss** (the tracked ball rising above the player) plus a loud impact. Serves are recognized without the serve sub-mode (toss, contact ≥ 2.2 m, pre-serve dribbles, a fast unseen ball into a service box). 98.1% of 162 shots labeled by eye are segmented correctly with no spurious segments.

### 7.9 Point outcomes and scoring

* For each point, `outcomes.py` produces `(winner, reason, confidence, evidence)`. Reasons: `ace`, `service_winner`, `double_fault`, `winner`, `forced/unforced_error_out_long|wide`, `net`, `double_bounce`, `let`, `unknown`.
  * Line calls use the bounce position vs line edges with tolerance. If |margin| < 2σ (from §7.1 uncertainty), the call is **low confidence**.
  * Serve faults use the correct service box for the current server and side, which comes from the score state.
* **Scoring engine** (`scoring/engine.py`): a pure, deterministic, heavily unit-tested state machine. It's fed point winners and outputs a score log, server, and serving side per point.
  * `MatchFormat`: `sets_to_win`, `games_per_set`, `win_by_two_games`, `tiebreak_at` (or none), `tiebreak_points`, `no_ad`, `final_set` (`normal` | `match_tiebreak(10)` | `advantage`), `pro_set_games` (first-to-N), `free_play`.
  * Presets: "Best of 3 (7-pt TB)", "Best of 3, MTB 3rd", "Pro set to 8", "First to 6, no TB", "Fast4", "Free play".
  * **Ends**: `switch_ends` = `per_rules` | `never` | `auto` (from detected player sides per point). The first server is auto-detected from the first serve and confirmable.
* Consistency check: the detected server for each point should match the engine's expected server. Mismatches flag earlier points for review.
* **Review UI** (§9) applies overrides, then the engine re-runs instantly.

### 7.10 Practice accuracy

* Targets are drawn on the court diagram: rectangles/circles, named, optionally filtered to stroke types (e.g. "FH crosscourt deep" applies to forehands only).
* **Target frame**: `relative` (default: defined on the *opponent's* half relative to the hitter, so it auto-mirrors when the player switches ends) or `absolute`.
* Per shot: in-court, in-target, distance to target center, depth/width error, net.
* Per block/session: target hit %, in % and net %, mean/median distance, depth consistency (std), and a rolling accuracy curve over the session (fatigue/learning). Breakdowns by stroke type and speed bands.

✅ Built in M5: landings in the hitter's frame (`rel_x/rel_y`), serves called against the diagonal service box, other shots against the singles court, close calls within 2σ; targets with a stroke filter (every stroke since M6, from the shot's swing); user edits (exclude, confirm, place landing) applied by `practice_eval`. Every target hit/miss on both sessions agrees with the target outline projected into the image (`scripts/m5_check_targets.py`).

---

## 8. Key data records

`shots.parquet` (one row per hit; ✅ M4, schema `SHOTS` in `storage/schemas.py`):

```
shot_id, session_id, segment_id, hitter (me|opponent|machine|unknown), hit_event_id,
flight_id, frame_contact, t_contact, stroke_type, stroke_conf, is_serve, serve_number,
contact_x, contact_y, contact_height, side (-1 near | +1 far), end_kind,
landing_x, landing_y, landing_sigma_m, landing_source (bounce|fit), landing_in,
landing_margin_m, landing_zone, speed_racket_kmh, speed_net_kmh, speed_avg_kmh,
speed_bounce_kmh, speed_sigma_kmh, net_clearance_m, apex_m, spin_sign, flight_time_s,
outcome (in|out_long|out_wide|net|own_side|unknown; winner etc. with match mode),
swing_id, fit_rms_px, quality_flags
```

`ball/flights.parquet` (M4): one fitted flight per event-to-event stretch: start/end kind and
event, t0/t1, fitted state (p0, v0, spin, C_d), speeds, net crossing, apex, landing, σ of the
key outputs, fit RMS and flags.

`swings.parquet` (✅ M6, schema `SWINGS`): `swing_id, t_contact, contact_source (hit|audio|pose), t_contact_pose, hit_event_id, side, racket_hand, stroke_type/conf/source, stroke_rules, phase times (t_start, t_split, t_backswing_end, t_follow_end, t_recovery_end) and durations, tempo, contact metrics, wrist speeds, peak angular velocities and timings, chain_in_order, joint angle summary (max/at-contact), stance, CoM drop, jump, toss height, pose_quality, flags`. Shots and segments point to their swing (`swing_id`). Also `pose/pose2d.parquet` (COCO-17 keypoints per frame) and `pose/pose3d.parquet` (H36M-17 court joints per frame).

`segments.parquet` (✅ M5, schema `SEGMENTS`): `segment_id, kind (point|warmup|practice_shot|block), start_t, end_t, block_id, shot_id, hit_event_id, feed_event_id, feed_kind, t_feed, t_contact, contact_source (hit|audio|estimate), t_end, end_reason, landing_*, hitter, side, shot_kind, serve_side, n_shots, conf, flags, point_index, server, first_serve_in, rally_length`.

`practice.parquet` (✅ M5, schema `PRACTICE`): per practice shot: landing (after edits) and its hitter's-frame position, call (`outcome`, `call_area`, `margin_m`, `close_call`), applicable targets and hits, distance/depth/width error to the nearest target, speed, feed speed and bounce, `excluded`, `landing_confirmed`, flags.

`points.parquet` / `score_log.parquet`: `point_index, winner_inferred, winner_final, reason, confidence, evidence_json, score_before/after, server, ends, overridden`.

`movement.parquet`: `frame, t_s, player, x, y, vx, vy, speed, sigma_m, source (bbox|interp; pose from M6), run, bx0..by1`.

---

## 9. Web app (Dash + Mantine)

Global: `dmc.MantineProvider` (light/dark toggle), `dmc.AppShell` with a navbar, `dmc.NotificationProvider`. Bound to **127.0.0.1 only**. Multi-page via Dash Pages.

### 9.1 Components

* **Server-side file browser modal** (`components/file_browser.py`). A browser upload can't expose local paths, so a `dmc.Modal` lists drives (Windows) and folders from the server with breadcrumbs, a filter by extension (`.mp4 .mov .mkv .avi`), and file size/duration preview. The same component has a **folder-select mode** for choosing the output root. Optional "native dialog" button (tkinter in a subprocess) as a convenience.
* **Video player**: native `html.Video` served from Flask routes with HTTP range support (`send_file(conditional=True)`), restricted to registered session files. `assets/video_sync.js` (clientside) publishes `currentTime` to a `dcc.Store` at ~10 Hz and handles seek commands. Keyboard shortcuts: J/K/L, ←/→ frame step, N/P next/previous segment.
* **Timeline**: a Plotly strip under the video showing segments, events (hits/bounces), and flags. Click to seek.
* **Court diagram**: a Plotly top-down court in meters. Layers include ball landings (colored by stroke/outcome), player positions/heatmaps, and targets. Supports `drawrect`/`drawcircle` editing modes for targets.
* **Calibration editor**: a Plotly image of a representative frame with keypoints as draggable shapes. The projected court model is overlaid and updated live on drag, with reprojection error shown. Includes net post/strap points, a zoom loupe, and buttons for reset / re-detect / confirm.
* **3D skeleton viewer**: Plotly `scatter3d` animation of a swing in court coordinates with a phase slider, side by side with the video segment. Joint-angle curves have phase bands below.

### 9.2 Pages

Phase 1 (MVP): Settings, Profiles (basics), New session (practice path), Jobs, Calibrate, Library, Session review, Practice, Swings, Stats (per session), Labeling.
Phase 2 adds: the match path in New session, appearance enrollment in Profiles, Points, and match sections of Stats. Phase 3 adds cross-session trends in Stats.

1. **Settings / first run**: output root (folder modal), ffmpeg path, model weights status + download, GPU info, processing defaults (padding, auto-continue threshold, pose window).
2. **Profiles**: create a profile (name, handedness, 1H/2H backhand, height). Enroll appearance by picking a player track from any processed session. Manage outfits.
3. **New session** (`dmc.Stepper`):
   1. Pick video (file modal), with probe info + thumbnail
   2. Mode: Match / Practice (self-feed, ball machine, serve)
   3. Match: format preset or custom, first server (auto/manual), switch ends (per rules/never/auto). Practice: draw targets (or load a saved target set)
   4. Players: "me" profile, optional opponent profile
   5. Review & enqueue
4. **Jobs**: queue with per-stage progress bars, ETA, logs tail, cancel/retry. Sessions that need action (calibration review, identity review) show a call-to-action.
5. **Calibrate**: the calibration editor. Confirming triggers the dependent stages.
6. **Library**: sessions table (date, mode, duration, status, headline stats), search/filter, relink missing video, delete session data.
7. **Session review**: video + timeline + live court minimap (ball and player positions at the current time) + segment list. Filter segments by stroke type, outcome, or flags.
8. **Points** (match): point table (score, server, inferred winner, reason, confidence badge). Low-confidence rows are highlighted and sorted first via a "review queue" toggle. Overrides: winner dropdown, mark let, split/merge/insert/delete points, set the match start marker. The score recomputes live, and a "Confirm score" action locks it.
9. **Practice**: target overlay with landing scatter (hit/miss coloring), accuracy KPIs, rolling accuracy chart, per-block table, and clicking a landing plays that shot.
10. **Swings**: list of swings filtered by stroke type, 3D viewer, phase timing, metric table. **Compare** two swings (overlay angle curves aligned at contact), or one swing vs the user's average for that stroke.
11. **Stats**: per session and across sessions (`pyarrow.dataset` + `pyarrow.compute`). Speed distributions by stroke, landing heatmaps, depth, first-serve %, points won by serve/return, rally length, movement distance and heatmaps, trends over time.
12. **Labeling** (§10).

### 9.3 Event/edit plumbing

* All UI edits write to `edits.json` through a small API (`storage/edits.py`) with optimistic versioning. Cheap derived stages (`scoring`, `practice_eval`, `stats`) rerun synchronously in the app process.
* Edits that change events (e.g. "this was a forehand", "bounce here") enqueue a CPU-only partial rerun from the affected stage.

---

## 10. Labeling and model fine-tuning

Goal: high-quality ball detection on the user's camera setup with a few hundred hand-labeled frames, amplified by assisted labeling.

* **Frame sampling** for labeling (via the Labeling page, from any processed session):
  * Low-confidence frames, frames near detected events, frames where the track has gaps, plus random frames.
  * Stratified by near/far court and lighting.
* **Ball labeling UI**: the ROI crop at full res. Click the ball center, or mark "not visible"/"occluded".
  * **Assisted**: model prediction pre-filled, Enter to accept.
  * **Keyframe interpolation**: label every ~5th frame in a short clip, and a parabolic fit fills the gaps for user verification. This yields contiguous labeled sequences that TrackNet's 3-frame input needs.
* **Event labeling**: hit/bounce corrections in the Review page are stored as labels automatically.
* **Court labels**: every confirmed calibration is a training sample for the court keypoint model.
* **Stroke labels**: stroke corrections in the Swings/Review pages.
* **Training CLIs**: `sv train ball|court|events|strokes`.
  * Train/val split by session (to avoid leakage), mixed precision, early stopping.
  * Writes versioned weights to `<output_root>/models/` with a model card (data, metrics).
  * The app can switch the active weights per model.
* **Evaluation**: `sv eval`.
  * Ball: detection precision/recall/F1 within 4 px; trajectory metrics.
  * Bounces: position error (m); event F1 within ±2 frames.
  * Segmentation: boundary IoU; point winner accuracy.
  * Stroke: confusion matrix.
  * Runs against a held-out, fully labeled ground-truth session.

---

## 11. Milestones

Each milestone ends with tests passing, a demo on real footage, and a short README update. Milestones are ordered by priority. **Phase 1 ends with a usable practice-mode MVP release.**

### Phase 1 — MVP: practice mode

#### M0 — Foundation and spikes ✅ (done 2026-10-05)
* `uv init`, `pyproject.toml` (deps, torch index, ruff/pytest config), `.python-version` 3.12, pre-commit, `sv` CLI skeleton.
* Settings (platformdirs), output-root handling, SQLite library schema + migrations. Schemas carry match-mode fields from day one (see §1a).
* Dash app shell: AppShell, theme toggle, Settings page, **file/folder browser modal**, Library (empty), Jobs page.
* Worker process + job queue + mode-aware stage framework (manifests, chunking, resume, cancel, progress).
* `ingest`, `proxy`, `audio_onsets` stages. Session review page with proxy video playback + time sync.
* **Spikes** (with written results in `docs/spikes/`):
  * (a) 4K60 decode backends on Windows (PyNvVideoCodec vs PyAV vs torchcodec)
  * (b) YOLO + TrackNet throughput at candidate resolutions
  * (c) HEVC/H.264 playback in the browser
* ✅ Exit criteria: pick a video in the UI, enqueue it, watch progress, and play the proxy with synced timestamps.

#### M1 — Court calibration and camera model ✅ (done 2026-10-05)
* ITF court model, classical court detector (no CNN needed, see §3/§7.1), sub-pixel line refinement, homography + uncertainty, lens `k1`/`k2`, principal point, PnP with net points, net-tape fitting, drift check with piecewise per-window cameras.
* `court_auto` + `camera` stages, calibration gating (`needs_action` → Calibrate page → confirm re-queues the job), auto-accept threshold in Settings.
* Calibrate page (draggable keypoints and net points, live re-solve and fit metrics, snap to lines, re-detect, drift-window viewer, confirm); court overlay on the session review video; `sv court detect` CLI.
* Framework fixes found on the way: the runner re-plans each stage just before running it (configs may read files written upstream or by the user), and a starting worker re-queues every job left `running` by a dead worker (it holds the lock, so they're orphaned), instead of only ones with a 30 s-old heartbeat.
* ✅ Exit criteria:
  * Reprojection (line-center) RMS on the user's footage: **0.64 px** and **1.20 px** for the two sessions (< 2 px). Every usable time window with its own pose: 0.66–1.24 px.
  * The projected court overlay aligns on the background, on each drift window, and on the playing video.
  * Synthetic tests recover known cameras (wide, narrow with off-center principal point, low corner camera) to < 1 px keypoint error, focal length within 1%, camera position within 10 cm, including through an H.264 encode and the full stage pipeline.

#### M2 — Single player tracking, profile basics, movement ✅ (done 2026-10-05)
* `pass1_detect` (players part: dense NVDEC/PyAV `FrameSource`, YOLO11m on the court-ROI crop at 1920 px, 15 Hz, plus per-frame brightness and a view check that ignores frames where the camera is being handled), court-ROI filtering, single-player tracking (court-space tracklets with speed/σ/height gates, a tracklet-chain DP that tolerates overlapping duplicate tracks, feet-out-of-frame placement from the head), static-object detection with ball-machine adoption or click-to-place (feed-origin detection needs M3's ball tracks).
* Profiles page (basics only): name, handedness, 1H/2H backhand, height; profile choice in New session and on the session page; `sv profiles`.
* Movement stage (Kalman/RTS with per-point ground σ, gap bridging) + live court minimap and player box on the video + speed timeline row + movement stats/heatmaps; distance in the Library.
* Framework fixes found on the way: ordering-only stage dependencies (`after`), and a forced/rerun stage now reruns its dependents even when its fingerprint is unchanged.
* ✅ Exit criteria (details in `docs/m2-player-tracking.md`):
  * Oct 1 session (28 min, self-feed, into dusk): the player is tracked in **99.94%** of hitting time while in the picture (95.0% counting two walks off camera and the camera setup as hitting time; hitting time approximated from strong audio onsets until M3 labels hits).
  * **No** spectators, neighboring-court players or objects tracked as the player: 48 of 48 random hitting moments audited by eye have the box on the player; playground spectators beyond the fence are ignored.
  * Oct 4 (51 min): 85% of hitting time, the rest with the player outside the picture (behind the far baseline at the frame's top edge); 18 of 18 tracked audit samples on the player.

#### M3 — Ball detection, trajectory, events (+ labeling) ✅ (done 2026-10-05)
* Labeling page (clips cached as JPEGs, labels pre-filled by the current detector + tracker, click-to-snap, no-ball/hidden, keyframe interpolation, events, train/test split, suggestions), `sv labels list|pseudo`.
* `BallDetector` interface with three detector families: classical frame-differencing `motion` (camera-shake compensation, streak centroids), slim TrackNet-style `unet` trained on own labels, COCO YOLO "sports ball". Benchmark `sv bench ball` (raw candidates cached per subject, HTML table + accuracy-vs-cost Pareto chart); `unet` at every frame is the default (`auto` = newest trained model).
* Frame-rate schedules: ball sweep in `pass1_detect` + `ball_refine` windows, benchmarked against every frame (no gain on play-dense footage with a 3-frame model, see `docs/m3-ball-tracking.md`).
* `pass1_detect` v2 (persons + ball in one decode, separate fingerprints so a new ball model keeps the person boxes), `ball_track` (tracklets + Viterbi selection of the ball in play, gap fill), `events` (kinks: line fits + velocity-change test; rules for hit / bounce / net, sub-frame contact, bounce ground point, audio matching with sound-delay compensation, ball-machine feeds).
* `sv train ball` (half-size frame cache, focal loss, hard negatives at earlier false detections), `sv train events` (gradient boosting on kink features; activated only from 200 labeled kinks, so rules stay in use for now), `sv eval`.
* Session page: ball and trail on the video, hits/bounces on the timeline, recent bounces on the court map, Ball card; Settings card for the detector and schedule.
* Labels: 9 held-out ground-truth clips (934 frames with the ball, 467 without, 26 events), 2,500 hand-checked or color-labeled training frames, 900 pseudo-labels. No ball-machine footage exists yet: machine feeds are covered by synthetic tests only.
* ✅ Exit criteria (held-out clips, `unet:ball-v2`, every frame; details in `docs/m3-ball-tracking.md`):
  * Ball F1 **0.908** near (≥ 0.85) / **0.951** far (≥ 0.75); with the strict size-only tolerance 0.887 / 0.950
  * Bounce event F1 **0.941** (≥ 0.85)
  * Median bounce position error **2.4 cm** near (≤ 15 cm) / **4.3 cm** far (≤ 35 cm), vs label-derived bounces; calibration adds ≈2 cm near, 4–7 cm far

#### M4 — Shots, landings, 3D flight, speed ✅ (done 2026-10-05; radar comparison pending footage)
* `ball_3d` (`ball/physics.py`, `ball/flights.py`): every flight between events fitted in 3D through the calibrated camera (gravity, drag with a C_d prior, one-parameter Magnus), linear drag-free initialization from rays, batched RK4 Jacobians, sampled output uncertainties; flights fitted in time order so a hit's contact comes from where its incoming flight (feed bounce, serve toss) ended, plus the tracked feet; quality checks (`poor_fit`, `end_mismatch`, `implausible`); hits whose fitted contact isn't at the hitter are rejected and the flights re-planned.
* `shots` (`analysis/shots.py`): one row per hit (§8) with speeds off the racket / at the net / before the bounce, contact point, net clearance, apex, spin sign, landing (detected bounce, else the extended fit) with σ, singles-court line call with margin and close-call flag, quality flags. No stroke type yet (M6).
* Session page: Shots card (KPIs, current shot's numbers and side view, clickable shot list), fitted flight drawn on the video, landings on the court map, shot-speed timeline row; `sv shots`; `scripts/m4_validate_speed.py`.
* ✅ Exit criteria (details in `docs/m4-ball-3d.md`):
  * Synthetic flights (independent integrator, mismatched drag and spin axis, 1 px noise): speed off the racket within **3%** for every shot from the camera's end (typically 0.5-1%, serves ≤ 2% with the toss) and for far-end ball-machine feeds; a far-court hitter with only feet tracking within 5%.
  * Real-world radar/ball-machine comparison: **not possible yet**, no such footage exists. Substitute physical references on the user's footage: g refitted from ground-to-ground flights is 9.87 / 9.96 m/s² (+0.6% / +1.5%, two sessions), so the speed scale is right to ≈1-1.5%; bounce restitution 0.80 (hard court); fitted vs feet→bounce horizontal speed 0.98. Oct 1 serves: 139 km/h median ± 2.8 km/h per shot, contact 2.76 m high. **Open item: record one session with a radar gun or a ball machine and compare.**
  * Oct 4's framing cuts the serves' flights at the top of the frame, so it yields no shots over the net (recording guide: leave sky above the far baseline).

#### M5 — Practice mode: segmentation, targets, accuracy ✅ (done 2026-10-06)
* `segments` (`analysis/segmentation.py`): one segment per practice shot from seen hits, unseen contacts (landing + impact sound, audio/video offset measured per session) and toss + sound; feeds (dribbles, drop, ball-machine feed), serve/groundstroke kind with block-level filling, deuce/ad, hitting side, blocks (pause, end change, kind change), padding from Settings.
* `practice_eval` (`analysis/practice.py`): service-box calls for serves, relative/absolute rect/circle targets with stroke filters, distance/depth/width errors, summaries, rolling accuracy, breakdowns; user edits in `edits.json` (`storage/edits.py`, optimistic versioning): exclude, confirm, place landing. Both stages rerun in-process after an edit (`services.refresh_practice`).
* New-session wizard (footage → session → targets → review), target editor (draw/move/erase, presets, saved target sets in the library), Practice page (targets on the video, landing map as hit / on the court, KPIs, rolling accuracy, blocks, shots, breakdown, click-to-play, shot corrections, targets and practice type editable in place), Segments card + timeline bands + N/P keys on the session page, Practice column in the Library.
* `sv practice show|eval`, ground-truth store `training/segments/<id>.json`, `scripts/m5_check_targets.py`.
* ✅ Exit criteria (details in `docs/m5-practice.md`):
  * Practice shots segmented correctly: **98.1%** (159 of 162 labeled shots, no spurious segments; Oct 1 100/101, Oct 4 59/61 on the audited spans). Both sessions are serve practice; self-feed groundstrokes and ball-machine feeds are covered by synthetic tests only, as no such footage exists yet.
  * Target accuracy: **158 of 158** landings with a detected bounce agree with the targets projected into the image, for every target battery (relative/absolute, rectangles/circles, stroke filters); bounce pixels checked frame by frame against the court lines.
* 🎯 **First usable milestone**: practice sessions with landings, speed, movement, and accuracy.

#### M6 — Pose, swings, strokes ✅ (done 2026-10-06; groundstroke footage pending)
* `pass2_pose` (ViTPose-B on windowed 4K crops), `pose3d` (MotionBERT-Lite lifting, height scaling, court placement with ray snapping), `swings` (racket hand, detection, contact, phases, kinematics, strokes); `shots`/`segments`/`practice_eval` carry each shot's stroke.
* Kinematics, swing phases, kinetic-chain timing, stroke rules → learned classifier (`sv train strokes`, validated by leaving one session out, used only once it beats the rules).
  * Serve detection here is pose-based (toss + overhead contact behind the baseline). It doesn't depend on match state.
* Swings page with 3D viewer, phase-annotated angle curves, compare mode (another swing or "my average"), stroke corrections; skeleton on the session video; `sv swings show|eval`; stroke ground truth `training/strokes/<id>.json`.
* Target stroke filters and per-stroke accuracy breakdowns are enabled on the Practice page.
* ✅ Exit criteria (details in `docs/m6-swings.md`; both sessions are serve practice, so strokes are scored as serve vs. not a stroke, and the other stroke classes are tested on synthetic swings only):
  * Stroke classification ≥ 90% (near player) / ≥ 80% (far player): **97.1%** / **86.6%** in daylight (97.1% / 80.4% including dusk), over 101 labeled serves and 592 non-stroke swings.
  * Contact frame within ±2 frames of hit events: near **39/39**; far **3/3** seen hits plus 8/12 within ±2 frames of the impact sound. The far sample is too small to call measured.
  * Phase timings consistent across repeated swings: phases in order on 100% of serves; preparation CV 5–8%, forward swing 0.17–0.18 s (±2 frames), follow-through CV 16–20% (Oct 1); Oct 4's above-the-frame serves are less consistent.
  * **Open items**: record a self-fed groundstroke session (and a far-end session with tracked hits) and label it; move the audio/video offset into `events` (Oct 4's sound-based contacts are ≈0.1 s late).

#### M7 — MVP hardening and release ✅ (done 2026-10-06; tagged v0.1)
* `stats` stage (`analysis/stats.py` → `stats.json`) and the per-session Stats page: KPIs, speed by stroke, depth, landing heatmap / dots by stroke (hitter's frame), strokes table, movement heatmap and distance per 5 min; stroke/end filters. CSV/Parquet export of shots (with practice results), practice shots and swings (`analysis/export.py`, `/export` route, `sv export`).
* Error handling: plain-language failure messages, one retry after GPU out-of-memory, a 5 GB free-space preflight, keep-awake while a job runs, needs-action reasons on jobs (library v3: `jobs.action`), error pages instead of blank ones. Relink-missing-video flow (Library badge + dialog, folder search by content, job resumes; `sv relink`). README: overnight processing, Stats/export, troubleshooting, recording guide.
* Calibration fixes found by the overnight test: auto-accept with camera movement when every moved window fits on its own; drift windows stay 5 min long up to 3 h; full re-fit for windows the pose-only refit can't fit; dark windows keep the nearest known camera and view reference.
* ✅ Exit criteria (details in `docs/m7-release.md`): a **2:10 h** practice video (the two real sessions joined: Oct 4 + Oct 1 + Oct 4, two camera mounts) processed unattended in **3 h 53 min** (1.79× realtime; pass 1 2 h 42 min, pose 26 min, proxy 24 min), calibration auto-accepted, GPU memory ≤ 6.4 GB; every part matches the same footage processed on its own (Oct 1 100/100 shots, Oct 4 136/137 and 135/137). Tagged **v0.1 (MVP)**.
  * **Open item**: confirm on a real single-mount 2-hour recording (with groundstrokes).

### Phase 2 — Match mode

#### M8 — Two players and appearance profiles
* Singles two-player constraint, opponent tracking.
* OSNet re-ID gallery, enrollment from a processed session, outfits, `identity` stage + identity-review flow, optional opponent profile.
* ✅ Exit criteria: on a 30-minute match, "me" is correctly identified for > 98% of in-play time.

#### M9 — Match segmentation
* Serve/rally/dead-ball state machine, first/second serves, lets, warm-up segments, match-start marker.
* New-session wizard (match path): format presets, first server, switch-ends mode.
* ✅ Exit criteria: ≥ 95% of points segmented correctly on labeled matches.

#### M10 — Outcomes, scoring, review
* `scoring/engine.py` with exhaustive unit tests (all formats, tiebreak serve rotation, end changes, no-ad, MTB, pro sets).
  * The engine is pure and has no CV dependencies, so it can be built earlier if convenient.
* `outcomes.py` with confidence and evidence, server-consistency checks.
* Points page with review queue, overrides, split/merge, live rescoring, lock. Match sections on the Stats page.
* ✅ Exit criteria: ≥ 85% of point winners correct before review, ≤ 2 minutes of review per set, and the final score always matches after review.
* Tag **v0.2 (match mode)**.

### Phase 3 — Later

#### M11 — Trends and polish
* Cross-session trends (`pyarrow.dataset`): accuracy, speed, and movement over time.
* Optional annotated proxy render (ball trail, skeletons, landings).
* Performance pass (TensorRT/FP16, batching).
* ✅ Exit criteria: all pages are responsive with 50+ sessions in the library.

---

## 12. Testing strategy

* **Pure units**:
  * Scoring engine: table-driven, property-based with `hypothesis`, e.g. "a set always ends at a valid score"
  * Court geometry, homography/PnP: recover synthetic cameras
  * Physics fit: recover synthetic trajectories with noise
  * Segmentation and outcome rules: synthetic event streams
  * Target mirroring, edits layering
* **Stage contract tests**: each stage runs on a tiny synthetic session (rendered court + moving dots) on CPU, with checks on schema, manifest, and resume behavior.
* **Golden clip regression**: a ~2-minute labeled clip of the user's footage, kept outside git and referenced by path in a local config. `sv eval --golden` reports metric deltas vs the last baseline.
* **UI**: callback unit tests for the edit/score logic, plus a smoke test that starts the app and loads each page (`dash.testing` optional).
* CI (optional, GitHub Actions): ruff + CPU-only unit tests. GPU tests are marked `@pytest.mark.gpu` and run locally.

---

## 13. Recording guide (ships in README)

* Mount the camera **centered behind the baseline, as high as practical (≥ 2.5–3 m)**. Height is the single biggest factor in far-court accuracy.
* Frame the entire court, including all lines, the area behind the far baseline, and some sky above the far baseline for lobs.
* 4K60, with **locked exposure/focus** if the phone allows it, and fast shutter (≤ 1/1000 s) in good light to reduce ball blur.
* Don't touch the camera once recording starts. If it's bumped, the drift check will flag it.
* Hit the record button before warm-up, and optionally clap once at match start (an audio marker the app can detect).

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Far-court ball is tiny and blurred at 4K | ROI crop at higher model res, fine-tuning on own footage, multi-hypothesis tracking, physics gap fill. |
| Line calls near the far baseline have depth uncertainty of tens of cm | Propagate uncertainty, mark low-confidence calls, review queue. Advise a higher camera mount. |
| Single-camera 3D pose depth errors (far player especially) | Quality flags, profile-height scaling, report angles mostly in robust planes. Compare a swing to the same player's history rather than absolute norms. |
| Re-ID fails after clothing change | Outfit galleries, one-click confirm flow, continuity constraints within a session. |
| Windows GPU video decode tooling | M0 spike, `FrameSource` abstraction with PyAV fallback. |
| Ball speed bias from calibration errors | Full camera model with net points, validation against ball machine/radar, uncertainty reporting. |
| Lets, net cords, and odd endings mis-scored | Explicit `unknown`/`let` reasons, server-consistency checks, review UI. |
| Long processing times | Chunked resumable stages, windowed pose, FP16/TensorRT, overnight queue. |
| Model license constraints | Personal, non-distributed use. Licenses recorded in the model registry. |

---

## 15. Out of scope for v1 (future ideas)

Doubles; live/real-time processing; racket keypoint tracking and true racket-head speed; spin rate measurement; multi-camera fusion; moving/handheld cameras; automatic highlight reels and clip export; cloud sync; mobile capture app.
