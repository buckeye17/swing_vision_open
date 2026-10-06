# M7: MVP hardening and release (v0.1)

Date: 2026-10-06. Code: `src/swingvision/analysis/stats.py` and `stages/stats.py` (session
statistics → `stats.json`), `analysis/export.py` (CSV/Parquet), the Stats page
(`app/pages/stats.py`, `app/components/stats_view.py`), relinking (`services.relink_session`,
Library dialog, `sv relink`), failure handling (`pipeline/errors.py`, the out-of-memory retry
in `pipeline/runner.py`, the disk check and keep-awake in `pipeline/worker.py` /
`keepawake.py`), calibration fixes for long videos (`court/calibration.py`).

## Exit criterion: a 2-hour practice session processes unattended overnight

**Met.** 130 minutes of 4K60 footage went from *Create* to *ready* in **3 h 53 min**
(1.79× the footage length) with no interaction: the calibration was accepted automatically,
every stage ran, and the session opens with its Practice, Swings and Stats pages. Its
results match the two sessions processed on their own (below).

### The test video

No 2-hour recording exists yet, so the two real sessions were joined into one file without
re-encoding (`ffmpeg -f concat -c copy`; same phone, same codec settings): **Oct 4 (51 min) +
Oct 1 (28 min) + Oct 4 again (51 min) = 2:10:22**, 55 GB, 468,178 frames. This is harder than
a real 2-hour session: the camera sits on two different mounts (the Oct 1 part is about
500 px off the Oct 4 view), each part starts with its mount sagging for 10-15 minutes, and
each ends in darkness.

Setup: the scratch output root with the trained ball model (`ball-v2`) and pose weights,
Settings → *Continue without review* at 1.5 px. The worker ran as `sv worker --once` in a
detached process (no app), from a git worktree of the code being released.

There were two full runs. Run 1 (commit `40b83ee`, 04:56 → 08:46, 3 h 50 min) finished
unattended, but the comparison below showed the Oct 1 part's dusk minutes missing, which
led to fix 4. Run 2 (commit `93bd9f4`, from scratch) is the one reported here.

### Timeline (run 2, job #5, 2026-10-06 09:34 → 13:27)

| Stage | Time | Notes |
|---|---|---|
| ingest | 17 s | probe, hash, audio |
| court_auto | 3 min 17 s | 27 drift windows (5 min each), 10 with the camera moved |
| proxy | 23 min 31 s | 720p NVENC |
| audio_onsets | 41 s | |
| camera | 0 s | **auto-accepted**: line RMS 1.21 px; every moved window fits on its own (≤ 1.27 px) |
| pass1_detect | 2 h 42 min | persons (15 Hz) + ball (every frame), 48 fps; crop = whole frame (the two mounts) |
| players_track, movement | 10 s | |
| ball_track, events | 2 min 53 s | |
| pass2_pose | 26 min 18 s | |
| pose3d | 3 min 3 s | |
| ball_3d | 10 min 42 s | |
| swings, shots, segments, practice_eval, stats | 10 s | |
| **Total** | **3 h 53 min 3 s** | 1.79× realtime |

Resources (sampled every minute): GPU memory ≤ 6.4 of 16 GB, GPU ≤ 74 °C, mean GPU
utilization 73%, worker process ≤ 2.9 GB RAM. The session folder holds ≈ 3 GB. The decoder
logged 11 harmless `INVALID INDEX` warnings at the very end (it was asked for frames past the
last one; detection covers frames up to 468,176).

Pass 1 is the bulk (70%). It ran at 48 fps here, slower than on the separate sessions,
because the detection crop must cover both mounts' views (the whole frame); a real
single-mount session gets a tighter crop.

### Fixes the test forced

The first attempt stopped at the calibration gate, which would have meant waking up to a
paused job. Four problems, all of which would also hit real sessions, were fixed:

1. **Camera movement always required review.** Both real sessions sag on their mount in the
   first 10-15 minutes, so the auto-accept rule (*no movement*) could never pass. A
   calibration with movement is now accepted when every window where the camera moved fits
   the lines within the threshold with its own camera; windows that failed still need review.
2. **Drift windows were 11 minutes long on a 2-hour video** (at most 12 windows), so their
   median images blurred the sagging mount and fit at 8-9 px. Windows now stay at the
   configured 5 minutes for up to 3 hours (36 windows).
3. **A window far from the main camera couldn't be fitted** by a pose-only refinement (and one
   such fit latched onto a neighboring court with few lines, at a low RMS). Windows whose
   pose-only fit is poor or finds too few lines now get the better of a full refinement and
   a fresh detection: the Oct 1 part fits at 0.64-0.70 px.
4. **Dark windows used the session's main camera and background.** In run 1 the Oct 1 part
   matched its separate processing exactly until dusk (local 19 min), then found nothing
   (81 of its 100 shots): its dark windows fell back to the Oct 4 camera, and the pass-1
   view check, comparing frames with the Oct 4 background, scored them 0.21 instead of
   0.89 and ignored them. A window too dark to calibrate now keeps the camera and the
   reference image of the nearest window that has one. On the two real sessions the
   camera choice is unchanged at every second (their dark windows follow windows on the
   main camera).

### Results vs the sessions processed separately

Each part of the 2-hour video against the same footage processed as its own session
(practice shots; in = in / called; matched = the separate session's shots found within
0.5 s):

| Part | Practice shots | In | Matched | Median speed (seen contacts) |
|---|---|---|---|---|
| Oct 4, first copy | 139 (alone: 137) | 45/99 (41/97) | 136/137 | 64 km/h (64) |
| Oct 1 | 100 (alone: 100) | 27/90 (26/90) | **100/100** | 137 km/h (138) |
| Oct 4, second copy | 141 (alone: 137) | 43/99 (41/97) | 135/137 | 64 km/h (64) |

Distance moved: 2,137 m vs 2,167 m for the three parts on their own (−1.4%). The small
differences most likely come from the detection crop (the whole frame here, so the ball detector sees
the frame at a slightly different scale) and from per-session settings estimated over
the whole video (audio/video offset, racket hand). Oct 4's median speed is low because
its serves leave the top of the frame, so only a few shots there (no fast serves) have a seen
contact and a speed (see M4).

## Stats page and export

* `analysis/stats.py` builds one record per shot: `shots.parquet`, with practice results
  layered over it for practice sessions (corrected landings, calls, exclusions, and the
  practice shots whose contact the tracker didn't see). Summaries per stroke group (the
  recognized stroke, else serve from the practice shot kind): calls, speed median / p90 /
  max, depth short of the service line (serves) or baseline (the rest), share of deep
  groundstrokes (in, within 2.74 m of the baseline), width spread; swings per stroke; movement
  and distance per 5 minutes.
* The `stats` stage writes `stats.json` (≈0.1 s) and is rerun in-app with `practice_eval`
  after edits.
* The Stats page: KPI row, speed by stroke (box + every shot), depth histogram (stacked by
  stroke, service line and baseline marked), landings as a heatmap or dots by stroke (hollow
  = out/net, so the call doesn't rely on color), strokes table, movement heatmap and distance
  per 5 minutes. Filters: strokes, end, excluded shots.
* Export: shots (with `practice_*` columns: segment, block, excluded, call after edits,
  target), practice shots, swings; CSV (lists joined with `|`) or Parquet (units in the
  field metadata). `/export/<session>/<table>.<fmt>`, the Stats page and Library menus,
  `sv export`.
* Stroke colors were re-picked so neighbors stay apart under color-vision deficiencies
  (serve #c2255c, forehand #1c7ed6, backhand #e8590c, …), checked with a palette validator.

## Error handling

* **Relink**: the Library marks sessions whose video is missing (a red *video missing* badge
  and a *Relink video…* menu item); the dialog opens the file browser in the old folder. The
  chosen file must match the stored size + fast hash. Other sessions with missing videos are
  found in the same folder by content (renamed files too), and a job that stopped for the
  missing video is re-queued. The Jobs page (*Relink video*) and the session page link to
  it. `sv relink`. The CPU `ball_track` stage now asks for a relink too instead of failing
  in the demuxer.
* **Needs-action reasons** are stored with the job (library schema v3, `jobs.action`):
  `calibration_review`, `relink`, `disk_space`. Rows from a newer schema don't break an
  older worker.
* **Failure messages** say what probably happened and what to do: GPU out of memory, GPU
  errors, disk full, a file held open by another program, ffmpeg errors. GPU stages retry
  once after running out of memory (30 s later; chunked stages resume).
* **Preflight**: a job doesn't start with less than 5 GB free in the output folder.
* **Keep-awake**: the worker asks Windows not to sleep while a job runs (the default power
  plan sleeps after 30 idle minutes, which would stop an overnight job).
* **Pages** that fail to build show the error and a way back instead of a blank screen.
* The New-session review step estimates the processing time and says whether the job will
  pause for the calibration review.

## Open items

* A real (single-mount) 2-hour recording, ideally with groundstrokes, to confirm the timing
  and the M5/M6 numbers on new footage.
* The speed comparison against a radar gun or ball machine (from M4).
* Existing sessions don't rerun automatically for fix 4 (no stage version was bumped, to
  avoid reprocessing sessions it doesn't affect); reprocess a session whose camera moved
  before it got dark.
