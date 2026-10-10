# M7c: Speed calibration from net-tape serves

Date: 2026-10-10. Code: `src/swingvision/ball/speed_refs.py` (sound onsets, one reference,
the calibration model), `pipeline/stages/speed.py` (stage `speed_refs`, the calibration a
session uses), `analysis/shots.py` (the calibration applied), `io/probe.py` (device tags and
keys), `storage/library.py` (library v6: devices, speed calibrations), `services.py` (devices,
reviews, calibrating), `training/speed_labels.py` (review ground truth), the Calibrate page's
Speed tab (`app/components/speed_view.py`, `app/assets/speed_wave.js`), the Settings card
(`app/components/devices_view.py`), the badges (`app/components/shots_view.py`) and
`sv speed refs|calibrate|show`. Tests: `tests/test_speed_refs.py` (synthetic serves,
`tests/synth_speed.py`), `tests/test_speed_calibration.py`, `test_physics.py::
test_calibrated_shots`. The spike: [spikes/m7c-tape-sound.md](spikes/m7c-tape-sound.md).

## What it does

A serve from the camera's end that hits the net tape gives its speed from two sounds and two
well-calibrated points, independently of the video clock, ball-detection timing, rolling
shutter and the 3D fit's contact prior. The app finds such serves, lets you review them, and
fits each recording device's speed correction from the accepted ones; every speed in the app
(shots, practice, stats, exports) then carries it: speeds × the device's factor, and the error
every speed shares is twice the calibration's uncertainty instead of 3%. The Shots card's
badge says *Calibrated · device · ±x%* instead of *Uncalibrated · ±3%*.

## How

**Time from sound.** Both impact sounds are on the audio clock, so the audio/video offset
cancels: `Δt = (t_tape − |P_tape − C|/c) − (t_racket − |P_contact − C|/c)`, C the camera
center (the phone's microphone), c from the session's air temperature (20 °C when it isn't
entered; editable in New session, on the session page and on the Speed tab). The tape is ≈12 m
further from the camera than the contact, ≈35 ms of sound travel, so the compensation matters;
an unknown temperature (±8 °C) moves Δt by ≈0.15%.

* **Where to look.** The phone's audio runs ≈100 ms behind its video (spike), so each session's
  offset is measured first from its serves' racket cracks (median of the strong ones). The
  racket sound is then searched ±20 ms around the video contact + delay + offset, the tape
  sound ±25 ms around the racket onset + the fit's flight time to the net + the extra delay.
* **One onset rule for both sounds** on the 22.05 kHz waveform: a causal 1 kHz high-pass, the
  rectified signal averaged over 0.25 ms, the strongest peak in the window, and its rising
  edge through 4× and 8× the noise floor (median over 35–4 ms before the peak) extrapolated
  back to the floor. A causal filter can't ring ahead of a loud crack (a zero-phase one put a
  synthetic racket onset 0.4 ms early); a rectified mean rises linearly over a sound's start
  and the extrapolation makes a quiet tick and a loud crack line up (a plain threshold fires
  later on the quieter one).

**The candidates** (stage `speed_refs`, CPU, ≈6 s for a 1 h session): every near-end serve with
a fitted flight whose path reaches the net plane with the ball's bottom within ±(ball radius
+ 5 cm) of the tape top, and that either ends there (`net`) or changes velocity at the crossing
(the detections after it leave a path fitted to the ones before it by > 4σ and > 8 px), with a
tape sound of ≥ 15 dB and a racket sound of ≥ 20 dB. The user can mark any serve as a tape hit;
it's measured whatever the rules say. Net-mesh hits fail the height test.

* **Lets are refitted.** `ball_3d` fits a let as one flight through the break, which bends it
  (the spike's shot 368 arrives 10 ms late). A let is refitted the way `ball_3d` fits a serve
  that ends in the net: the detections up to the break (a change point in the image track,
  refined where the quadratics either side meet), the toss contact prior, a net end at the
  break. Its ratio then measures what `ball_3d` gives a clean serve.

**Distance from geometry.**

* *Contact*: M7b's toss path at the impact's time (from the racket onset, the sound delay and
  the session's offset, kept within the contact frame), with the toss fit's covariance.
* *Tape*: the ball's ray in the frame nearest the crossing cast onto the net plane gives x;
  the height is the tape model's there (posts 1.07 m, center strap 0.914 m) plus half a ball
  radius.
* The path is the straight contact → tape distance times the fitted path's arc / chord.

**The reference** = path / Δt; the **ratio** = reference / the production fit's average speed
over the same part of the flight on its own (video) clock. Its σ combines the onsets (1 ms
each, measured in the spike), the temperature, the path (the toss covariance along the path,
≥ 2 cm; the tape point ±3 cm in x and z, little of it along the path) and the fit's own
average-speed σ (≥ 1.5% for a let: its end time is the break's).

**The correction model** (`fit_calibration`, chosen by the data from every accepted reference of
one device):

1. A scalar factor k: the weighted mean ratio, σ from the scatter (χ² scaled) or a bootstrap,
   whichever is larger, plus 0.2% shared systematics (tape model, microphone position,
   temperature common to a session). Leave-one-out spread and a 95% interval are kept.
2. Diagnostics: weighted trends of the ratio vs speed and vs the rolling-shutter rate (the
   sensor readout fraction the ball crosses per second, signed by the video's rotation), and a
   χ² test between sessions.
3. Only when the rolling-shutter trend is significant (p < 0.05, ≥ 6 references): a readout
   time τ from `1/ratio = 1/k + τ·rate`.

**Per device.** `probe` reads the container's make, model and lens tags (iPhone
`com.apple.quicktime.*`, Android `com.android.*`, vendor prefixes such as `com.oplus.*`;
location tags are never read); the device key is make + model + lens + coded size + frame rate
(`oneplus-open-back-main-3840x2160-60`). Library v6 keeps `devices`, versioned
`speed_calibrations` (one active per device) and each session's `device_key` and `air_temp_c`;
sessions probed before M7c get their device from a quick re-probe when the app starts. In
Settings → *Recording devices* a device can be renamed, merged into another key (the same phone
and mode), given another active version, or calibrated again; a session can use its device's
calibration, none, or one picked by hand (Speed tab).

**Applied in `shots`** (v3): the calibration in use is in its config, so a new or changed one
reruns `shots` → `segments` → `practice_eval` → `stats` only (the app does it in-process where
it can). Speeds × the factor (with a readout time, per flight from its own readout rate, the
first-order effect of fitting with corrected times, so `ball_3d` doesn't depend on its own
calibration); new columns `speed_factor`, `speed_scale_err` (the shared bound: 3% uncalibrated,
else 2σ/k) and `speed_calibration`. Practice results and stats records (v3) carry the bound;
records also carry the device and whether speeds are calibrated, so Stats can select by device
and by "only calibrated speeds". A calibration no tighter than 3% (2σ ≥ 3%) is never applied.

**Review** (Calibrate page → Speed tab, `/calibrate/<id>?tab=speed`): the device and its
calibration with *Update calibration*, the air temperature, the session's references; for the
one picked: the clip at ¼ speed looping from the toss to past the net, the frames around the
tape zoomed ×3 with the tape model and the ball's detections, the waveform ±60 ms around each
sound with its onset (drag the red line, or nudge it by 0.1 / 1 ms) and its search window, and
the numbers (flight time, path, reference and fitted speed, ratio, height over the tape, audio
offset). *Accept* / *Reject* / *Reset review*; *Mark a serve as a tape hit* for one the rules
missed. Reviews and moved onsets go to `edits.json` and, as ground truth, to
`training/speed_refs/<id>.json`; `speed_refs` reruns in a few seconds.

**CLI**: `sv speed refs <session> [--accept/--reject/--reset <id>] [--mark <t>] [--temp °C]`,
`sv speed calibrate <device> | --session <id> [--dry-run]`, `sv speed show [<device>]`.

## Choices beyond the plan

* **The audio/video offset is measured per session.** The plan only needed it to cancel; it
  doesn't, for *finding* the sounds: they come ≈100 ms after the video says.
* **The onset rule** extrapolates the rising edge to the floor on a causal, rectified envelope
  (the plan: the crossing of k × floor). Same rule for both sounds, as planned.
* **Lets are refitted to the break** rather than compared with their bent production fit.
* **The onset σ is 1 ms**, not 0.3 ms: measured in the spike (a racket crack's soft start
  moves with the noise floor). A reference's ratio σ is ≈1.7–2% (mostly the toss's depth along
  the path and the fit's own scatter), so ≈10 references give k to ≈0.6%.
* **A rolling-shutter calibration is applied in `shots`** per flight (first order), not inside
  the `ball_3d` fits. Same effect for monotone image motion, no dependency cycle.
* **Feed speeds** (ball machine) take the session's factor too (they come from flights, not
  shots).
* **No device without a model tag**: such sessions stay uncalibrated rather than share a key
  with other phones.

## Exit criteria

### Synthetic: Δt within 0.5 ms, the factor within 0.3% (clock +2%, rolling shutter 15 ms)

**Met.** `tests/synth_speed.py` renders serves through a camera like the user's (3.3 m high,
6.4 m behind the baseline, the phone upside down as the user's is, so the sensor reads the
picture bottom to top), with the video clock running 2% fast and a 15 ms readout, flights from
an independent integrator (drag 0.58 vs the fit's 0.55), 1 px detection noise, 10% missed
detections, fitted like `ball_3d` (net-ending serves and lets that carry on at 85–95% speed),
and the racket and tape sounds synthesized on the true clock at their sound delays, 100 ms
behind the video, with an echo, hum and noise (the tape tick ≈25 dB over the floor, the crack
≈40 dB). Over 5 seeds × 10 serves each:

| Clock, readout | Candidates | Δt error (median / max) | Factor error per seed |
|---|---|---|---|
| +2%, 15 ms | 47 / 50 | 0.10 / 0.30 ms | −0.07% … +0.08% |
| +2%, none | 48 / 50 | 0.15 / 0.43 ms | −0.12% … +0.02% |
| none, none | 48 / 50 | 0.11 / 0.31 ms | −0.11% … +0.07% |

The factor is compared with what each serve's fitted speed truly needed (true average speed /
fitted, weighted like the references): k = 1.021 ± 0.004 (true 1.022) with the clock error,
0.999 without (the fits' own drag bias, ≈ −0.1% here). For serves from behind the rolling
shutter is small (the ball moves slowly in the picture): ≈0.07% here. The missed 2–3 per 50 are
lets whose break wasn't found (their window then misses the tick); the unit tests also cover the
onset rule at three loudness levels, a serve without a tape sound, user-moved onsets, the
model choice (scalar / rolling shutter / between-session trend) and the stage on a session on
disk.

### Real: ≥ 8 accepted references on one device; leave-one-out spread ≤ 1.5%; factor σ ≤ 1%; no significant trend

**Not met yet: the footage has 2 references** (spike). In 3.5 h (Oct 1, Oct 4, Oct 6, all the
user's OnePlus Open at 4K60) only two serves touched the tape, both lets on Oct 6: ratios
1.034 ± 0.020 and 1.012 ± 0.017, both saying the fitted speeds are 1–3% low (consistent with
M4's synthetic drag bias and gravity checks). They are left for the user to review (the
decisions are ground truth). **A calibration session is needed**: 30–40 serves aimed at the
tape from the camera's end, from both courts, quiet surroundings, the air temperature noted.
Then review them on the Speed tab (or `sv speed refs`), *Update calibration* (or
`sv speed calibrate --session <id>`), and the leave-one-out spread, σ and trends print with it.

### The Shots card shows calibrated speeds with the measured uncertainty; docs/m4-ball-3d.md updated

**Met (built and tested; with the user's footage it shows *Uncalibrated · ±3%* until a
calibration exists).** With a calibration, the card's badge reads *Calibrated · OnePlus Open ·
±0.8%*, the median tile *± 0.8% calibrated*, each shot *± is the calibrated error (0.8% + 2× fit
σ)*, and every speed is × k (checked on Oct 1 with a test calibration in a scratch copy of the
library, then removed; `test_calibrated_shots`). The Stats page's badge says *Calibrated*,
*Uncalibrated* or *Calibrated in m of n sessions*. [m4-ball-3d.md](m4-ball-3d.md) points here.

### Open item stays: a radar-gun comparison

Still the independent check of the whole chain (the tape references and the speeds off the
racket, which the calibration scales by the same factor).

## Limitations

* A calibration applies one factor (or one readout time) to every speed of a device: it is
  measured on serves from the camera's end over contact → tape; groundstrokes and far-end shots
  share the clock and scale errors but not necessarily the fit's own bias.
* The onset rule can pick another transient in the window (a footstep, a ball on the next
  court) when the tick is quiet; the review is there for that.
* The tape model is two straight pieces (no sag modelled between strap and posts; the error is
  < 1 cm near the strap, where serves cross).
