# M7a: Multi-session statistics

Date: 2026-10-10. Code: `src/swingvision/analysis/aggregate.py` (selection, records read,
cache), `analysis/stats.py` (one `StatsData` for one or many sessions, pooled movement,
trends), `pipeline/stages/stats.py` (writes `stats_records.parquet`), `storage/library.py`
(library v4), `analysis/export.py` (selection export), the Stats page (`app/pages/stats.py`,
`app/components/stats_view.py`), the Library page (tags, *Open in Stats*), `sv stats` and
`sv tags`. Tests: `tests/test_aggregate.py`, synthetic sessions in `tests/synth_stats.py`.

## What it does

The Stats page shows the same numbers for one session (`/stats/<session>`) or for any
selection of sessions (`/stats?from=…&type=…&tag=…`): KPIs, speed by stroke, depth, landing
maps, the strokes table, swings and movement, plus a trend chart (one point per session at
its recording date). A selection is made by recording date, mode, practice type, player,
tags, explicit include / exclude, and a quality option (only calibrations the user
confirmed). Filters live in the URL and can be saved as named views; the Library opens the
ticked sessions in Stats; the selection's shots or swings export as CSV or Parquet; a shot
clicked on the page opens its session one second before it. `sv stats` prints the same
summary from the command line.

## Design

**One data path.** The `stats` stage (version 2) now writes the records its summaries are
computed from to `stats_records.parquet` (schema `STATS_RECORDS`), after the user's edits:
one row per shot (practice results layered over `shots.parquet`, excluded shots kept and
flagged), one per swing (the fields the swing summaries read), and one movement row per
session. Every row carries the session's columns (`session_id, recorded_on, mode,
practice_type, profile_id, device_key, calibration_by, speeds_calibrated`). A selection
reads the matching sessions' files with one `pyarrow.dataset` into the same `StatsData` a
single session builds from its tables (`stats.load`), so every summary function runs
unchanged. Numbers are float64 in the file, so the round trip is exact: a one-session
selection gives *identical* records, not just close ones.

**Movement adds up.** One session's movement summary used to be computed from the raw
movement and frame tables. It is now a record of sums and counts: frames processed / usable /
tracked, tracked time, distance, best 0.5 s speed, samples (all, at running speed, on the
near half), the sum of running speeds, the folded time-per-cell heatmap and the per-5-minute
distance bins. Pooling gives coverage = Σ tracked frames / Σ usable frames, mean running
speed = Σ speeds / Σ samples, and so on, exactly as if the raw samples were concatenated. The
single-session page uses the same code over one record. The per-5-minute distance chart
becomes a bar per session for a selection.

**Order.** Sessions are ordered by recording time, then shots by time within the session, so
everything that runs through the records (trends, the export) is in date-then-time order.
The recording time is the video's `creation_time` (UTC from the phone) in local time, or the
session's creation time when the video has none.

**Trends.** `stats.trend(data, kpi)` gives one value per session with its n and a 95%
interval: Wilson for rates (in %, net %, on target %, deep %), order statistics for medians
(speed, wrist speed), a normal interval for means (depth); distance has no interval (n is the
minutes tracked). The chart follows the page's stroke and end filters.

**Cache.** Selections are cached in memory (16 entries) by the filter, the excluded-shots
switch, and for each matching session its records file's mtime and size, name, recording
time and tags, plus the matching sessions without records. Reprocessing a session changes
only its file, so only the selections it's in are read again.

**Library v4** (backfilled from each `session.json` on first open): `sessions.recorded_on`,
`sessions.profile_id` (kept in sync when the player changes or a profile is deleted, so the
player filter doesn't read every session's config), `session_tags` (cascade on delete) and
`saved_views` (name → URL query).

**Choices beyond the plan.**

* Tags are read from the library when a selection is made (and added to the export), not
  stored in the records file: retagging a session needs no reprocessing.
* The *Only calibrated speeds* option and the `device_key` column are in place but empty
  until speed calibration (M7c); the switch is disabled with a note. Serve-contact records
  join the file with M7b.
* The rolling-accuracy chart stays on the Practice page (one session). Across sessions the
  trend chart shows accuracy per session.
* Sessions that match but have no records file (processed before this version) are counted
  and named on the page, with *Update their statistics*: `services.refresh_practice` runs the
  cheap stale stages in the app, or queues a job when heavier ones are stale. Match sessions
  aren't counted as missing (their statistics arrive with Phase 2).

## Exit criteria

### For a selection of one session, every number equals the single-session page

**Met.** `test_one_session_selection_equals_single_session` builds four synthetic sessions
(serve, self-feed and ball-machine practice; near and far shots; unseen contacts, excluded
shots, uncertain speeds, targets, dark frames, movement runs), runs the stage, and checks for
each session, with and without excluded shots: the records, swings and movement records are
equal, and so are `stats.json`'s content, the heatmap, every trend KPI, and the shot and
stroke summaries under nine stroke × end filters. `test_one_session_page_renders_identically`
renders the page both ways and compares the serialized KPIs, figures and tables.

On the two real sessions (Oct 1 `ecc669e5`, Oct 4 `4c981c00`, a copy of the bigger-models
output root), a one-session selection gives the same records, swings, movement and summary
as the session's own page for both. In the browser, the Oct 1 Stats page and the selection
"all sessions minus Oct 4" show the same tiles: 100 shots, 29% in, 17% net, 138 km/h median,
160 km/h fastest, 695 m, 9.9 m/s top speed.

### Synthetic multi-session libraries: pooled numbers and filters

**Met.** `test_pooled_summaries_equal_pooled_raw_records`: over four sessions, the records
equal the sessions' own records concatenated in date-then-time order; counts, in %, the speed
median and the stroke and swing summaries equal those of the pooled raw records; distance and
tracked time are the sums of the sessions' own movement summaries; coverage, mean running
speed, time on the near half and the heatmap equal the values computed from the concatenated
raw movement and frame tables; each trend point equals its session's own number and lies in
its interval.

`test_filters_select_exactly` (six sessions, one of them a match and one without records):
date range (inclusive, both ends), mode, practice type, player, tags (all of them, any case),
include, exclude and combinations select exactly the expected sessions in recording order;
the quality option keeps only the user-confirmed calibration and lists the others as left
out; the session without records is reported as missing. The filter survives a URL round
trip (`test_query_round_trip`).

### 50 synthetic sessions render in < 2 s (cached < 0.5 s)

**Met.** `test_fifty_sessions_render_fast`: 50 sessions of one hour, each with 150 shots,
150 strokes plus 150 other swings and an hour of movement (7,500 shots, 15,000 swings). The
page's render and trend callbacks take **0.67 s** the first time (reading 50 files, building
every figure) and **0.33 s** with the selection cached (laptop, RTX A5000 machine, CPU only).

### Both real sessions together on the Stats page; the CSV export matches the page

**Met.** The two sessions (copied tables only; the originals untouched) were run through the
new `stats` stage (0.18 s and 0.24 s) and opened together on the page: **2 sessions · 237
shots · 213 serves · 183 swings**; in 35.8% of 187 called, net 16%, median 138.2 km/h over 32
seen contacts (Oct 4's serves are hit above the frame, so it has no speeds), 1,431 m moved in
58 min tracked. Reading both record files takes 18 ms.

| Over time | Oct 1 | Oct 4 |
|---|---|---|
| In % (n called, 95%) | 28.9% (90; 20.5–39.0%) | 42.3% (97; 32.9–52.2%) |
| Speed median (n, 95%) | 138.2 km/h (32; 133.8–143.1) | – (0) |

The CSV export of the selection has 237 rows (the page's shot count), its outcomes give
35.83% in and its speeds a median of 138.16 km/h, both equal to the page; with the page
filtered to near-end serves, 121 rows, 29.55% in and 139.20 km/h on both sides.

Also checked in the browser: filter changes update the counts and the address bar without a
reload (`/stats?exclude=4c981c00`); a saved view comes back from the list; tagging in the
Library shows the tag; ticking both sessions opens `/stats?sessions=4c981c00,ecc669e5`; a
clicked speed dot opens `/session/ecc669e5?t=50.60` with the video at 0:49.6; *Sessions…* and
*This session* switch scope both ways.

## Notes

* Existing sessions get their records file the next time their `stats` stage runs (its
  version changed, so it's stale): *Update their statistics* on the selection page, or
  *Process (stale stages)* in the Library.
* The first app start on an existing output folder migrates the library to v4, reading every
  session's `session.json` once.
