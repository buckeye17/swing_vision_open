# M5: Practice mode — segmentation, targets, accuracy

Date: 2026-10-06. Code: `src/swingvision/analysis/segmentation.py` (practice shots and
blocks), `src/swingvision/analysis/practice.py` (targets, calls, accuracy),
`src/swingvision/pipeline/stages/practice.py` (`segments`, `practice_eval`),
`src/swingvision/storage/edits.py` (user corrections), the Practice page
(`app/pages/practice.py`, `app/components/practice_view.py`), the target editor
(`app/components/target_editor.py`), ground truth and scoring
(`training/practice_labels.py`, `sv practice eval`), and the image-space target check
(`scripts/m5_check_targets.py`).

## Results

| Exit criterion | Target | Result |
|---|---|---|
| Practice shots segmented correctly on labeled sessions | ≥ 95% | **98.1%**: 159 of 162 labeled shots, **no spurious segments** (Oct 1: 100/101 = 99.0%; Oct 4: 59/61 = 96.7% on the audited spans) |
| Target accuracy correct for every confirmed landing | every landing | **158 of 158** landings with a detected bounce agree with the targets projected into the video, for every target battery (relative and absolute rectangles, circles, serve-only and forehand-only filters); 3 bounces checked frame by frame on the video against the court lines |

`uv run sv practice eval` prints the first row; `uv run python scripts/m5_check_targets.py
<session-id>... --sets builtin` the second.

A shot counts as **segmented correctly** when exactly one segment matches it (same hitter's
end, contact within 1.2 s) and that segment's clip covers the swing and the start of the
flight. Accuracy = correct / (labeled shots + spurious segments), so a missed shot and a
segment with no shot both count against it.

### Ground truth

Both of the user's sessions are serve practice (each player end in turn, about one serve
every 10 s, collecting balls in between). There is no self-fed groundstroke or ball-machine
footage yet: those paths are covered by synthetic event streams in `tests/test_practice.py`.

* **Oct 1** (28 min, into dusk): every candidate moment — every hit, every bounce on the
  half opposite the player, every audio onset with a robust z-score ≥ 25, clustered into
  180 moments — was rendered as a time-coloured strobe from the 4K video (3-frame
  difference, maximum over the moment, blue → red over time) and checked by eye; moments
  longer than 6 s got continuation strobes, and the dark stretch brightened strobes.
  **101 practice shots** (all serves; 16 into the net), whole session.
* **Oct 4** (51 min, close framing: a near-end serve's toss and contact leave the top of the
  picture): each predicted shot was checked with a strobe of the swing and a strobe around
  its landing, and every gap in the serve rhythm longer than 12.5 s was rendered and checked
  for a missed serve. **61 shots** in 80–700 s and 780–1020 s (both near-end blocks and the
  far-end block where the server is tracked). The two far-end blocks where the server stands
  out of the picture were checked for false segments only (every predicted landing there is
  a serve arriving; the ball bouncing on after one landing was a false segment and is now
  rejected), not for misses: a serve out wide lands out of view and can't be told from no
  serve.

Labels are in `<output_root>/training/segments/<session_id>.json` (contact time, end,
net or over).

### Misses

| Session | Time | Why |
|---|---|---|
| Oct 1 | 25:02 | In the dark the ball is barely detected: a weak toss and a faint impact (z = 10) |
| Oct 4 | 2:28.6 | No landing detected (out wide on the right) and the contact is above the frame; impact z = 18 |
| Oct 4 | 15:57.7 | A very quiet serve (z ≤ 15) whose ball was never seen |

## How shots are found (`analysis/segmentation.py`)

The events and shots from M3/M4 already contain every hit the ball tracker saw, but a
practice shot is more than a hit: pre-serve dribbles, taps and drop feeds are hits too, and
the hit of half the serves in these sessions isn't seen at all. Three kinds of evidence,
in order:

1. **Seen contacts.** A hit of the player's whose ball crosses the net: its shot record has
   a call over the net, or the first bounce after it (before the next hit) is on the other
   half, or the ball hits the net — or drops back right behind it on the hitter's half (an
   into-the-net serve's ball usually isn't seen touching the tape). A ball coming down on
   the hitter's half within 1.2 s was a tap or a drop. A hit whose flight is lost counts
   when it's a serve: a fitted contact ≥ 2.2 m, or a **toss** (the tracked ball rising and
   falling above the player's image box, starting at head height) — and it isn't one of a
   run of taps or fitted below 1.8 m.
2. **Unseen contacts.** The first bounce of a ball on the half opposite the player (or a
   net contact) with no hit before it, when the **impact sound** is there: the loudest onset
   0.25–1.8 s before the landing, after the session's audio/video offset and the sound's
   travel time. It must be strong (z ≥ 15), or merely present (z ≥ 5) when the ball lands
   inside the court with the player tracked at the other end. A bounce deeper on a half
   where a shot landed less than 4 s earlier is that shot's ball. The contact time comes
   from the sound.
3. **Toss and sound only.** At dusk a serve's ball can vanish completely: a loud impact
   (z ≥ 25) right after a toss, away from every other shot.

**Audio/video offset.** The Oct 1 video's audio lags its picture by 0.10 s (Oct 4: 0.00 s).
It's measured per session as the most common gap between an isolated seen hit and the
loudest onset near it, so a sound-based contact time is right to a frame or two.

Each shot then gets its **feed** (dribbles, a drop, a ball-machine feed, or a toss), its
**kind** (serve: a serve-practice session, a toss, a high contact or pre-serve dribbling
behind the baseline, or an unseen contact whose ball comes down fast in a service box;
groundstroke: a bounce on the hitter's half just before the contact; unclear shots take
their block's kind), its **serve side** (where the server stood, else the box it went to),
and a segment from the feed to the landing, padded 1 s before and 1.5 s after (Settings:
`practice_pad_before_s`, `practice_pad_after_s`), never overlapping the next. **Blocks**
split on a pause longer than 45 s (`practice_block_gap_s`), a change of end, or serves
turning into groundstrokes.

Results on the two sessions: Oct 1 100 shots (69 seen contacts, 31 unseen) in 5 blocks;
Oct 4 137 shots (15 seen, 122 unseen) in 8 blocks; the stage takes 0.2–0.3 s.

## Targets and accuracy (`analysis/practice.py`)

* **Hitter's frame.** Every landing is also stored as if hit from the near end
  (`rel_x`, `rel_y`; a far-end shot is rotated by 180°): +y is deeper, +x the hitter's
  right. Depth and width errors and the Practice page's "As you hit" map use it, so both
  ends read the same.
* **Targets** are rectangles or circles drawn on the court. *Relative* targets (default) are
  drawn in the hitter's frame and follow the player to the other end; *absolute* ones stay
  where they are drawn. A stroke filter limits a target to some strokes; until stroke
  classification (M6) only *serve* is known (from the shot's kind).
* **Calls.** Serves are called against the service box diagonal from the server's side;
  other shots against the opponent's singles court; a ball center up to a line's outer edge
  is in (as in M4). Calls and target edges within 2 landing σ are flagged close.
* **Per shot:** call and margin, the targets that apply and which were hit, distance to the
  nearest target's center, depth/width error, speed (serve speeds under 60 km/h are fits to
  the toss and dropped), ball-machine feed speed and bounce.
* **Summaries** per session, block, shot kind / serve side, stroke (M6) and speed band: in
  %, net %, target hit %, median distance, depth and width spread, median speed, machine
  feed speed and spread; a rolling accuracy over the last 10 shots.

### Checking target accuracy

`scripts/m5_check_targets.py` shares nothing with the evaluation but the target
definitions: it projects each applicable target (for the hitter's end) into the image
through the calibrated camera and tests the bounce pixel against that outline with OpenCV.
Run with five target batteries on both sessions:

| Target set | Oct 1 (75 landings) | Oct 4 (83) |
|---|---|---|
| Deuce and ad boxes (relative) | 75 agree (27 hits) | 83 agree (40) |
| T and wide corners, serves only | 75 (6) | 82 (13) |
| Circles (relative) | 75 (9) | 83 (6) |
| Near/far left halves (absolute) | 75 (51) | 83 (70) |
| Forehands only (no strokes before M6) | 0 apply | 0 apply |

The landings themselves were checked on the video: for three Oct 4 serves the bounce pixel
sits on the ball in its lowest frame, with the projected service line where the paint is
(two long by 1.1 m and on the line, one in by 2 m). M3 measured the bounce positions at
2.4 cm (near) and 4.3 cm (far) median error against labels.

Many of Oct 4's near-end serves are called long (15 of 32 landings, 11 in): the
frame-by-frame check confirms the ones looked at bounce beyond the service line.

## Edits (`storage/edits.py`)

User corrections live in `edits.json` and are applied by `practice_eval`, never written into
derived files: *not a practice shot* (excluded from every summary), *landing is right*
(confirmed), and a landing placed by clicking the map (replaces the detected one). Each edit
is keyed by its shot's anchor time (the hit, or the landing for an unseen contact), so it
survives re-segmentation. Writes are versioned: a second tab editing an older version gets
an error instead of overwriting.

## Pipeline

| Stage | What it does | Output |
|---|---|---|
| `segments` | Practice shots, feeds, kinds, serve sides, blocks (practice sessions only; match points arrive in M9) | `segments.parquet` |
| `practice_eval` | Calls, targets, accuracy, with the user's edits | `practice.parquet` |

Both are CPU-only and take well under a second. Targets and edits are in `practice_eval`'s
fingerprint and the practice type in `segments`', so the app reruns just these two in-process
after an edit (`services.refresh_practice`); with anything heavier stale it queues a job.

## App

* **New session** is a four-step wizard: footage, session (practice type, player), targets
  (draw, presets, saved target sets), review.
* **Practice page** (`/practice/<id>`): the video with the targets drawn in (for the end
  being played) and the current shot's result; KPIs; rolling accuracy (click a shot to play
  it); blocks; the landing map ("As you hit" or "On the court", filters by block and kind,
  click a landing to play it); the selected shot's details with *Play*, *Landing is right*,
  *Place landing* (click the map), *Not a practice shot*; a shot table and a breakdown by
  kind, serve side and speed band. *Targets* opens the editor; the practice type can be
  changed in place. N/P jump to the next/previous shot.
* **Session page**: a Segments card (blocks and their shots, click to play), shots as bands
  on the timeline, N/P keys, a *Practice* button. **Library**: a Practice column (shots, in %,
  on target).
* **CLI**: `sv practice show <id>`, `sv practice eval [<id>...]`.

Two browser details: `assets/shape_events.js` forwards shape edits and plot clicks to
stores, because `theme_sync.js`'s re-colouring relayout replaces a graph's `relayoutData`
in the same tick; and the target editor's figure gets a fresh `uirevision` on every render,
otherwise Plotly keeps the shapes the user drew over the server's list.

## Limitations and follow-ups

* **No self-feed groundstroke or ball-machine footage** exists yet: drop feeds, machine
  feeds, feed consistency and groundstroke blocks are only tested on synthetic events.
  Record one of each and label it (`training/segments/<id>.json`) to extend the check.
* A serve whose landing is out of view, with a quiet impact and its toss above the frame,
  can't be found (Oct 4's two misses); a camera framing with room above the far baseline
  avoids most of these.
* The M3 `events` stage matches racket sounds without the audio/video offset, so on Oct 1
  (0.10 s lag) hits get no audio confirmation; segmentation measures and applies the offset
  itself. Moving the offset estimate into `events` would improve hit confidence and contact
  times there.
* Unseen contacts have no speed: their flight is seen only near the landing. Their contact
  time (from the sound), the server's position and the landing would allow a
  two-point-boundary speed estimate.
* The serve side of an untracked far-end server comes from the box the ball went to, so a
  serve into the wrong box can't be called wide.
