# M4: 3D ball flight, speed, shots

Date: 2026-10-05. Code: `src/swingvision/ball/physics.py` (flight model and fit),
`src/swingvision/ball/flights.py` (flights from the track and events, chaining, false-hit
rejection), `src/swingvision/analysis/shots.py` (shot records, line calls), stages in
`src/swingvision/pipeline/stages/shots.py`, the Shots card in
`src/swingvision/app/components/shots_view.py`, checks in `scripts/m4_validate_speed.py`.

## Results

| Exit criterion | Target | Result |
|---|---|---|
| Synthetic flights: speed off the racket | within 3% | **met** for every shot from the camera's end: groundstroke, hard drive, lob, slice and serve, 3 noise draws each (`tests/test_physics.py`); typical error 0.5-1%, worst ≈2%. Ball-machine feeds from the far end: within 3%. A far-court *hitter* with only feet tracking: within 5% (depth along the camera axis is weakly determined; M6's pose will help) |
| Real footage: speeds vs ball-machine/radar | within ~10% | **no reference exists yet** (no radar gun, no ball-machine footage). Physical references instead: gravity recovered to **+0.6%** (Oct 1, 109 flights) and **+1.5%** (Oct 4, 31 flights), so the speed scale is right to ≈1-1.5%; see below |

The radar half of the exit criterion can't be measured with the footage available. The
gravity check measures the same thing a radar comparison would catch (a wrong metric scale or
clock), more precisely than a consumer radar gun (±1-2 km/h, ±1-3%), but it can't catch an
error that only affects the horizontal speed of fast flights. Two observations speak against
one: the fitted contact points are where a server's racket is (median 2.76 m high), and the
fitted path's horizontal distance from contact to bounce agrees with the tracked feet → bounce
distance (median ratio 0.983, the contact being a little in front of the feet). **To close
the criterion, record a session with a radar gun or a ball machine with known speed settings
and compare with `uv run sv shots <session-id>`.**

### Synthetic accuracy

Flights from an independent integrator (`scipy.integrate.solve_ivp`) with deliberately
different physics from the fit (C_d 0.58 vs the prior's 0.55, a spin axis that turns with the
ball), through a camera like the user's (3.3 m high, 6.4 m behind the baseline, 87° wide),
1 px detection noise, 10% missed detections, the first 1-2 frames after contact missing.
Monte Carlo, 15 noise draws per case:

| Case | Speed | Mean error | SD | Worst |
|---|---|---|---|---|
| Self-fed groundstroke | 99 km/h | -0.7% | 0.5% | 1.4% |
| Hard drive | 123 km/h | -0.6% | 0.6% | 1.9% |
| Lob | 61 km/h | -0.8% | 0.3% | 1.3% |
| Slice | 73 km/h | -0.7% | 0.4% | 1.4% |
| Serve (contact from the toss) | 159 km/h | -1.1% | 0.4% | 2.0% |
| Serve (feet only, no toss) | 159 km/h | -0.2% | 1.5% | 2.9% |
| Ball machine, far baseline | 74 km/h | -1.2% | 0.9% | 2.5% |
| Far-court hitter (feet only) | 91 km/h | -1.3% | 1.8% | 4.5% |

With the true C_d equal to the prior (spin axis still turning) the mean errors of the chained
fits are −0.2% to +0.03%, so the ≈ −0.7% above is the drag mismatch: one flight barely
identifies C_d, so it stays near its prior.
Landing points from the fit are within ≈5 cm of the truth; net clearance and apex within a
few cm.

### Real footage: Oct 1 session (28 min, serve practice into dusk)

`ball_3d` fits 1,266 flights (916 good) in 250 s on the CPU; `shots` makes 409 shot records
from the 424 hits (15 rejected, see below). 321 of them stay on the hitter's side: pre-serve
dribbles with the racket and drop feeds, kept for M5's segmentation but hidden on the page.
**74 shots go over the net**: 37 in, 17 out (9 long, 8 wide), 2 net, 18 without a landing.

| | Serves from the near end | Serves from the far end |
|---|---|---|
| Shots with a certain speed | 25 (22 ending in a detected bounce) | 7 |
| Speed off the racket, median (IQR) | 139 km/h (134-146) | 130 km/h (126-136) |
| 1-σ per shot, median | 2.8 km/h (2.0%) | 6.2 km/h (4.5%) |
| Contact height, median | 2.76 m | 2.71 m |
| Net clearance, median | 0.58 m | 0.63 m |

The same server at both ends gives the same contact height and a similar speed: the far-end
depth isn't off. Landing σ (from 1.8 px of bounce-pixel uncertainty through the camera):
median 16 cm, mostly along the court. Line calls are against the singles court; serves are
called against the service box once serve detection exists (M6).

### Real footage: Oct 4 session (51 min, self-feed and serves)

`ball_3d` fits 1,385 flights (701 good) in 200 s; 10 of 446 hits are rejected. This session
gives **no shots over the net**: its framing has almost no room above the far baseline, so a
serve toss and most of the serve's flight leave the top of the frame (the ball comes back
into view a few frames before it bounces in the far court). Without the contact there is no
hit event and no shot; the 436 hits are pre-serve dribbles and drop feeds at the near
baseline. The recording guide's advice (sky above the far baseline) matters for speeds too.
Its bounce → bounce flights still serve the scale check below.

### Speed-scale checks (`scripts/m4_validate_speed.py`)

| Reference | Expected | Oct 1 | Oct 4 |
|---|---|---|---|
| g from bounce → bounce flights refitted with g free (weighted mean) | 9.81 m/s² | **9.87** (n = 109, median 9.94, IQR 9.80-10.07) | **9.96** (n = 31, median 9.97, IQR 9.84-10.15) |
| Vertical restitution v_z(after) / −v_z(before) at bounces | 0.7-0.85 on hard courts | **0.80** (n = 317, IQR 0.73-0.86) | **0.80** (n = 126, IQR 0.76-0.86) |
| Fitted / feet → bounce horizontal average speed (hits bouncing ≥ 8 m away) | ≈1 (contact ahead of the feet: slightly below) | **0.983** (n = 28, IQR 0.972-0.993) | – (no tracked contacts, see below) |
| C_d from fast flights refitted with a loose prior | 0.5-0.6 (published) | 0.46 (n = 10, IQR 0.32-0.51) | – |

* **Gravity** is the precise one. A bounce → bounce flight starts and ends at 3D points
  (ground + calibrated pixel), so how far the ball falls in the image fixes g in metres and
  seconds. A length-scale error `s` would show as g·s, a clock error `c` as g/c²; speeds
  scale as s/c, so the speed scale is off by at most the g error (+0.6% and +1.5%) and about
  half of it if the clock were to blame.
* **Restitution** matches hard-court physics (the ITF rebound test gives 0.73-0.76 for a
  vertical drop; oblique bounces with spin bounce a little higher).
* **Drag** is a weak check: one flight barely constrains C_d (the spread is wide) and a used,
  spinning ball differs from wind-tunnel values. A 0.46 would mean speeds 20% too high *if*
  C_d were really 0.55, which gravity and the direct distance / time comparison rule out.

### Speed error shown in the app; calibration (M7c)

Since M7c, speeds are calibrated per recording device from serves that hit the net tape
([m7c-speed-calibration.md](m7c-speed-calibration.md)): two impact sounds on one clock and two
well-calibrated points give a reference speed, and every speed of the device is multiplied by
the factor fitted from the accepted references. On synthetic serves with a 2% clock error and a
15 ms rolling shutter the factor comes back within 0.12%; the user's footage so far has two
such serves (ratios 1.034 and 1.012, i.e. fitted speeds 1–3% low, consistent with the drag
bias and gravity checks below), too few to calibrate: a calibration session is needed.

Every speed in the app and in `sv shots` is shown as `value ± error`, with error = the bound
every speed shares (`speed_scale_err`: 2σ of the device's calibration, or uncalibrated
`SPEED_SCALE_ERROR`, 3%, in `analysis/shots.py`) × speed + 2 × the shot's fit σ. The 3% bounds what every speed would
share: the gravity checks (+0.6%, +1.5%), the drag model's ≈1% bias in the synthetic tests,
and rolling shutter (< 1%). Typical uncalibrated results: serves from the camera's
end ± 7-15 km/h (5-10%), far-end shots more. The Shots card carries an *Uncalibrated · ±3%*
badge (*Calibrated · device · ±x%* with a calibration) and a note explaining it; the median and
fastest speeds are labeled the same way.

## Pipeline

| Stage | What it does | Output |
|---|---|---|
| `ball_3d` | Cuts the ball track into flights between events, fits each in 3D in time order (a hit's contact prior comes from where its incoming flight ended), rejects hits whose fitted contact isn't at the hitter and re-plans the flights around them | `ball/flights.parquet`, `ball/flight_paths.parquet` (30 Hz samples for drawing), `ball/flights.json` (rejected hits) |
| `shots` | One record per hit (PLAN.md §8): speeds, contact, net clearance, apex, spin sign, landing (detected bounce, or the fitted path extended to the ground), line call with margin and close-call flag, quality flags | `shots.parquet` |

Recalibrating reruns both (CPU only). `shots` will also depend on `swings` once M6 adds
stroke types.

## Flight model and fit (`ball/physics.py`)

* **Model**: gravity + quadratic drag (C_d prior 0.55 ± 0.05; 57 g, 6.7 cm ball, ρ = 1.2) +
  Magnus with one coefficient around a horizontal spin axis (prior 0 ± 0.01 /m; + topspin,
  − slice). RK4 at 1/30 s steps (error < 0.1 µm over 1.5 s), all parameter sets of a
  Jacobian or a covariance sample integrated in one batched pass.
* **Observations**: detected ball pixels (σ = 1.5 px + 15% of the per-frame image motion for
  blurred streaks), robust soft-L1 loss, points beyond 5σ dropped and refitted.
* **Ends**: a bounce is at ball-radius height ± 2 cm at its refined pixel; a net contact is
  in the net plane; a hit *end* fixes only the time. A hit *start* uses the incoming flight's
  end position and covariance (when there was one) plus the hitter's tracked feet (± 1 m;
  the contact is beside and ahead of the feet) and a broad height prior (1.2 ± 1 m); a ball
  machine's position ± 0.4 m.
* **Start**: a linear least-squares solve of the drag-free trajectory (each ray constraint
  `d × (p(t) − C) = 0` is linear in the initial position and velocity), so no flight needs a
  hand-tuned initial guess.
* **Uncertainty**: covariance from the Jacobian (scaled by χ²/dof when above 1), propagated
  to every output by 48 sampled parameter sets.
* **Checks**: reprojection RMS > 4 px (`poor_fit`: a missed contact inside the "flight", a
  wrong link), speed > 260 km/h (`implausible`), a fitted path missing its end event by
  > 12 px or ending off the ground (`end_mismatch`).

Why chaining matters: from the camera's end a serve flies almost straight away from the
camera, so where it started along the line of sight is the main unknown. The toss (a short
free flight) pins the contact to a few cm, reducing the per-serve speed spread from 1.5% to
0.4% in the synthetic tests.

## False hits

M3's rules call a kink a hit when the ball is inside the player's (padded) image box. The
near player's box covers part of the far court in the image, so a ball at the far fence can
become a "hit by me". In 3D that hit's flight starts metres from the player: a well-fitted
flight starting > 2.5 m from the hitter's feet rejects the hit, the flights around it are
re-planned (unchanged flights come from a cache) and the shot is dropped. Oct 1: 15 of 424
hits rejected.

## App

* **Session page**: a *Shots* card (shots over the net, in %, median and fastest speed; the
  current shot's speeds, net clearance, apex, contact height, spin, landing and call; a side
  view of its flight; a table of shots, click to play), a *Shot path* switch that draws the
  fitted flight on the video with a speed/outcome label, landings on the court map (colored
  by outcome, the current shot's marked with its speed), and a shot-speed row on the timeline.
* **CLI**: `sv shots <session-id> [--all]`.

## Limitations and follow-ups

* **No radar / ball-machine reference** yet (see above); the net-tape calibration (M7c) is the
  in-app reference, a radar gun stays the independent check.
* **Rolling shutter** is not modelled in the fits: a phone sensor reads rows over ≈10-20 ms,
  which shifts a fast ball's apparent time by up to that much between the top and bottom of
  the frame. For a serve moving a third of the frame height over 0.5 s that's < 1% of its
  speed. M7c's calibration measures it (a readout time, only if the references show a trend)
  and corrects speeds per flight.
* **Shots ending in the dark** (the Oct 1 session after 23 min) often lose the ball before
  the bounce: their speed comes from a weaker fit, and they are flagged `speed_uncertain`
  above 5% σ (10% for flights with a detected end).
* **Spin** is only a sign, and only when the data demand it; serves mostly come out "no clear
  spin" from this camera angle.
* Hits that stay on the hitter's side (dribbles, drop feeds) and hits by an unknown hitter
  remain shot rows; M5's segmentation decides what counts as a practice shot.
* `ball_3d` takes ≈8 CPU-minutes per footage hour (one process); fitting independent rallies
  in parallel would cut that several-fold if it ever matters.
