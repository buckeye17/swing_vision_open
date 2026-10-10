# Spike: net-tape serves in the existing footage (M7c)

Date: 2026-10-10. Script: `scripts/spikes/tape_sound_spike.py` (it runs the `speed_refs` stage's
own measurement, `ball/speed_refs.py`); pictures were made with throwaway plotting scripts.

Questions (PLAN.md M7c): are there near-end serves that hit the net tape in the footage we
have? How strong is the tape sound, and how repeatable is its onset? If there are too few,
record a calibration session.

**Answer: two, both lets, in 3.5 h of footage. Too few: a calibration session has to be
recorded.** The tape ticks are clear (24 and 42 dB over the noise floor; the racket crack is
37 dB) and line up with the frame where the ball reaches the tape. Onsets repeat to
0.25–0.75 ms for a tape tick and ≈1.35 ms for a racket crack. Three findings changed the plan:
the phone's audio runs ≈100 ms behind its video, a let's production fit bends through the
break, and clear serves often have some other transient in the tape window.

## Footage

| Session | Length | Near-end serves with a fitted flight | Tape hits |
|---|---|---|---|
| Oct 1 (`ecc669e5`, serve practice into dusk) | 28 min | 56 | none |
| Oct 4 (`4c981c00`) | 51 min | none: the contact is above the picture, so serves have no hit event or flight (M4, M7b) | – |
| Oct 6 (`92e516bd`, serve practice, new this milestone) | 62 min | 103 | 2 lets: shots 346 and 368 |

Oct 6 was processed on the NAS with the user's settings; its `swings` and `shots` were rerun
here with the current code (swings v2 has M7b's toss contacts). Its net serves (`end_kind`
net) are all mesh hits: shot 118's ball is in the net 25 cm below the tape (the frames show it),
and its sound, though sharp, is the mesh's.

How a tape hit was told from a near miss: the fitted path reaches the net plane with the ball
within a few cm of the tape top; the ball's image breaks at the tape (it slows and jumps); a
sharp broadband tick sits where the ball meets the tape in the frames, after the sound delay.
For shot 368 the ball's detections stop right at the tape band (it is hidden behind it for
three frames) and the tick comes at that moment; for shot 346 the tick (42 dB) comes as the
ball reaches the tape in frame 90199, and the ball then rolls along the tape.

## Findings

**Audio ≈100 ms behind the video.** Measured from the racket cracks of each session's serves:
the onset minus the video contact minus the sound's travel is 99.9 ms on Oct 6 (88 serves) and
104.1 ms on Oct 1 (55 serves), IQR ±3.5 ms (the video contact is frame-quantized). The container
says 9–13 ms. The offset cancels out of Δt (both sounds are on the audio clock), but it is
needed to know where to look for the racket sound, so `speed_refs` measures it per session
(the M6 open item "Oct 4's sound-based contacts are ≈0.1 s late" is this offset).

**The tape window is busy.** Of the 48 serves that cleared the tape by more than 30 cm (both
sessions), 8 had a transient of ≥ 15 dB in the ±25 ms window where a tape sound would be, and
2 of ≥ 20 dB: footsteps and the landing after the serve, balls on other courts. An SNR gate
alone would make false references, so a candidate also needs the geometry (the ball's bottom
within ±(r + 5 cm) of the tape) and a net end or a velocity change at the crossing, and every
candidate is reviewed by the user.

**A let's production fit is bent.** `ball_3d` fits one flight from the contact to the bounce
through the break at the tape, so its path arrives at the net ≈10 ms late for shot 368 (its
residuals reach 11 px, the contact sits 25 cm behind the toss's). The reference is meant to
measure what `ball_3d` gives a normal serve, so a let is refitted the way `ball_3d` fits a
serve that ends in the net: the detections up to the break (found as a change point in the
image track), the toss contact prior, a net end at the break. Synthetic lets confirm it (the
break time is found within a frame; refitted lets' ratios match the truth as well as net
serves'). A fit of the pre-tape detections with no end is not a substitute: without the end its
depth is loose (shot 368: 158 km/h off the racket against 138).

**Onset repeatability.** Each candidate's tick, and a racket crack, was cut out (2 ms before
its onset to 30 ms after) and pasted into the tape windows of the 31 clear Oct 6 serves (real
background, ±15 ms of jitter), at its own level, ½ and ¼:

| Sound | Level | Timed | Bias | SD | Lost to another transient |
|---|---|---|---|---|---|
| Tape, shot 368 (24 dB) | ×1 | 29 | +0.02 ms | 0.25 ms | 2 |
| | ×½ | 28 | +0.35 | 0.76 | 3 |
| | ×¼ | 23 | +0.57 | 0.71 | 8 |
| Tape, shot 346 (42 dB) | ×1 | 31 | +0.63 | 0.63 | 0 |
| | ×½ | 31 | +1.15 | 0.52 | 0 |
| | ×¼ | 30 | +1.30 | 0.49 | 1 |
| Racket crack, shot 346 | ×1 | 29 | −1.00 | 1.35 | 2 |
| | ×½ | 27 | +0.01 | 1.33 | 4 |
| | ×¼ | 25 | +2.86 | 1.77 | 6 |

A racket crack starts softly (the strings before the frame), so where its first arrival crosses
the noise floor moves by ≈1 ms with the floor; a tape tick is sharper. Per serve that is
≈1.4 ms on Δt (0.4% of a 0.35 s flight), which `RefParams.onset_sigma_s` (1 ms per onset) now
carries; the plan's ±0.3 ms per onset was optimistic. "Lost" pastes are another transient in
the window being picked: the review shows those at a glance (the tick is not where the ball meets
the tape).

**The onset rule.** The first version (envelope through k × floor on a zero-phase high-pass)
timed a synthetic racket crack 0.4 ms early: the filter rings ahead of a loud sound's low
frequencies. The rule is now a causal high-pass, a rectified 0.25 ms mean (it rises linearly
over a sound's start), and the rising edge through k and 2k × the floor extrapolated back to
the floor, the same for both sounds; on synthetic serves Δt comes back within 0.43 ms.

## The two references

| Shot | Kind | Over the tape | SNR racket / tape | Δt | Reference | Fitted | Ratio |
|---|---|---|---|---|---|---|---|
| 346 (25:01.9) | let, refitted | +3 cm | 37 / 42 dB | 322.4 ms | 132.6 km/h | 128.2 km/h | 1.034 ± 0.020 |
| 368 (25:33.5) | let, refitted | +3 cm | 37 / 24 dB | 371.1 ms | 120.1 km/h | 118.6 km/h | 1.012 ± 0.017 |

Both say the fitted speeds are low by 1–3%, consistent with M4's synthetic drag bias (≈ −0.7%)
and gravity checks (+0.6 to +1.5%), but two references can't calibrate (at least 3 are
needed, and the exit criterion asks for 8 with a leave-one-out spread ≤ 1.5%). They are left
unreviewed for the user: the decisions are ground truth (`training/speed_refs/`).

## Next: a calibration session

30–40 serves aimed at the tape from the camera's end, from both courts, in quiet surroundings,
with the air temperature noted in the session (README, recording guide). Serves that drop
into the net off the tape are the cleanest references (their production fit already ends at
the net); lets work too, refitted to the break. Every tape hit has to be reviewed, so the
count that matters is the accepted ones: the exit criterion needs 8.
