"""Net-tape reference serves and per-device speed calibration (PLAN.md §7.12, M7c).

A serve from the camera's end that hits the net tape gives a speed that doesn't depend on the
video clock, ball-detection timing, rolling shutter or the fit's contact prior:

* **Time from sound.** The racket impact and the tape impact are both sharp sounds on the
  same (audio) clock, so the audio/video offset cancels. Each reaches the phone after
  ``|P − C| / c`` (C: the camera center, where the microphone is; c from the air
  temperature), so the flight time is
  ``Δt = (t_tape − |P_tape − C|/c) − (t_racket − |P_contact − C|/c)``. The tape is ≈12 m
  further from the camera than the contact, ≈35 ms of sound travel (10% of the flight), so
  the compensation is essential; a 10 °C temperature error moves Δt by ≈0.15%.
* **One onset rule for both sounds** on the raw 22.05 kHz waveform: high-passed, a 0.25 ms
  envelope, its rising edge before the peak through k and 2k × the noise floor extrapolated
  back to the floor. A half-peak rule would bias Δt (a tape tick and a racket crack rise
  differently), and so would a plain threshold (it fires later on a quieter sound).
* **Distance from geometry.** The contact is the ball on the toss path (M7b) at the impact;
  the tape point is the ball's ray in the frame nearest the crossing cast onto the net plane,
  at the tape's height there plus half a ball radius. Both lie in the best-calibrated part of
  the view.
* **Reference** = path length / Δt; the per-serve ratio = reference / the 3D fit's average
  speed over the same part of the flight on its own (video) clock.

Candidates are near-end serves whose fitted path reaches the net plane with the ball's bottom
within ±(ball radius + 5 cm) of the tape, that end there (``net``) or change velocity at the
crossing (a let), with a tape sound in a ±25 ms window around its predicted arrival. Net-mesh
hits fail the height test. The user reviews each candidate (accept / reject, onsets moved on
the waveform) and can mark any serve as a tape hit.

**Correction model** (:func:`fit_calibration`), from every accepted reference of one device:
a scalar factor k (weighted mean ratio, bootstrap interval), diagnostics (ratio vs speed, vs
the rolling-shutter rate, vs session) and, only when the rolling-shutter trend is
significant, a readout time τ: an observation at sensor readout fraction s was taken at
``t_frame + τ·s``, so a flight that crosses Δs of the sensor in Δt of video time was really
``Δt/k + τ·Δs`` long and its speeds are off by ``1 / (1/k + τ·Δs/Δt)``. Shots apply that
factor per flight (:func:`speed_factor`), which is the first-order effect of fitting with
the corrected times, without making ``ball_3d`` depend on its own calibration.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from swingvision.ball import physics as ph
from swingvision.court.camera import Camera

KMH = 3.6
#: Speed of sound in dry air at 0 °C (m/s); c(T) = c0 √(1 + T / 273.15).
SOUND_SPEED_0C = 331.3
DEFAULT_AIR_TEMP_C = 20.0
#: 1-σ of the air temperature: not entered (a range of playing weather) / entered.
AIR_TEMP_SIGMA_UNSET = 8.0
AIR_TEMP_SIGMA_SET = 2.0
#: Shared by every reference of a device (not reduced by averaging): the tape height model
#: (±1 cm of 0.9-1.07 m over the ≈12 m path ≈ 0.1%), the camera center as the microphone
#: (a few cm along the line of sight), the temperature's error common to a session.
SYSTEMATIC = 0.002
#: Too few accepted references to calibrate a device.
MIN_REFS = 3


def sound_speed(temp_c: float | None) -> float:
    t = DEFAULT_AIR_TEMP_C if temp_c is None else float(temp_c)
    return SOUND_SPEED_0C * math.sqrt(1.0 + t / 273.15)


@dataclass(frozen=True)
class RefParams:
    highpass_hz: float = 1000.0
    env_s: float = 0.00025  # envelope window
    #: The onset is where the envelope last rose through this many times the noise floor.
    k_floor: float = 4.0
    floor_s: tuple[float, float] = (0.035, 0.004)  # noise floor: this long before the peak
    racket_window_s: float = 0.02  # search ± around the predicted racket sound
    tape_window_s: float = 0.025  # ... and the tape sound
    racket_min_snr_db: float = 20.0
    tape_min_snr_db: float = 15.0
    #: 1-σ of an onset (each): two real tape clicks pasted into 31 real tape windows came
    #: back with SD 0.25-0.75 ms, a racket crack with 1.35 ms (its soft start moves with the
    #: noise floor); together ≈1.4 ms on Δt (docs/spikes/m7c-tape-sound.md).
    onset_sigma_s: float = 0.001
    tape_sigma_m: float = 0.03  # 1-σ of the tape point (height model, where it touched)
    contact_sigma_min_m: float = 0.02
    #: The ball's bottom within ± (radius + this) of the tape top.
    tape_tol_m: float = 0.05
    #: Detections after the crossing this far (median, in σ and px at 3840) from the path
    #: fitted before it: the ball changed velocity at the tape.
    deflect_sigmas: float = 4.0
    deflect_px: float = 8.0
    #: Only paths passing this close to the tape (production fit, m) are checked for a let:
    #: the bent production fit of a let still passes within a few cm.
    let_gate_m: float = 0.4
    #: The audio/video offset is searched this far after the video contact + sound delay.
    av_search_s: tuple[float, float] = (-0.05, 0.30)
    av_min_serves: int = 3

    def as_config(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rolling shutter
# ---------------------------------------------------------------------------


def readout_fraction(px: np.ndarray, rotation_cw: int, width: int, height: int) -> np.ndarray:
    """Where in the sensor's row-by-row readout each display pixel (…, 2) was read, 0 … 1.

    The sensor reads its coded rows top to bottom; a video displayed rotated by
    ``rotation_cw`` maps those rows to display rows (0/180) or columns (90/270).
    """
    px = np.asarray(px, np.float64)
    r = rotation_cw % 360
    if r == 0:
        return px[..., 1] / height
    if r == 180:
        return 1.0 - px[..., 1] / height
    if r == 90:
        return 1.0 - px[..., 0] / width
    return px[..., 0] / width


# ---------------------------------------------------------------------------
# Sound onsets
# ---------------------------------------------------------------------------


@dataclass
class Onset:
    t: float  # audio-file time of the onset (s)
    snr_db: float
    t_peak: float
    floor: float


class AudioClip:
    """Windows of the session's audio (mono, any rate) on the audio file's clock."""

    def __init__(self, read: Callable[[int, int], np.ndarray], sample_rate: int, n: int):
        self._read = read
        self.sr = sample_rate
        self.n = n

    @classmethod
    def from_file(cls, path) -> AudioClip:
        import soundfile as sf

        info = sf.info(str(path))

        def read(a: int, b: int) -> np.ndarray:
            y, _ = sf.read(str(path), start=a, stop=b, dtype="float32", always_2d=False)
            return y.mean(axis=1) if y.ndim > 1 else y

        return cls(read, info.samplerate, info.frames)

    @classmethod
    def from_array(cls, y: np.ndarray, sample_rate: int) -> AudioClip:
        y = np.asarray(y, np.float32)
        return cls(lambda a, b: y[a:b], sample_rate, len(y))

    def window(self, t0: float, t1: float) -> tuple[np.ndarray, float]:
        """Samples between ``t0`` and ``t1`` (clipped to the file) and the first one's time."""
        a = max(0, math.floor(t0 * self.sr))
        b = min(self.n, math.ceil(t1 * self.sr))
        if b <= a:
            return np.zeros(0, np.float32), a / self.sr
        return np.asarray(self._read(a, b), np.float64), a / self.sr


def envelope(y: np.ndarray, sr: int, p: RefParams) -> np.ndarray:
    """High-passed, rectified and averaged over ``env_s`` (same length as ``y``). A mean of
    |y| rises linearly while its window slides over a sound's start (an RMS would rise as a
    square root), so the onset rule's extrapolation lands on the start."""
    if len(y) < 32:
        return np.abs(y)
    hp = highpass(y, sr, p)
    k = max(1, round(p.env_s * sr))
    return np.convolve(np.abs(hp), np.ones(k) / k, mode="same")


def highpass(y: np.ndarray, sr: int, p: RefParams) -> np.ndarray:
    """Causal high-pass: nothing of a sound can show before it starts (a zero-phase filter
    rings ahead of a loud crack, putting its onset early)."""
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(4, p.highpass_hz, "highpass", fs=sr, output="sos"), y)


def onset_near(audio: AudioClip, t_center: float, half: float, p: RefParams) -> Onset | None:
    """The strongest transient within ``t_center ± half`` (audio clock) and its onset."""
    pad = p.floor_s[0] + 0.01
    y, t_first = audio.window(t_center - half - pad, t_center + half + 0.01)
    if len(y) < 64:
        return None
    sr = audio.sr
    env = envelope(y, sr, p)
    tt = t_first + np.arange(len(env)) / sr
    win = np.flatnonzero((tt >= t_center - half) & (tt <= t_center + half))
    if not len(win):
        return None
    j = int(win[np.argmax(env[win])])
    return onset_at_peak(env, tt, j, p)


def onset_at_peak(env: np.ndarray, tt: np.ndarray, j: int, p: RefParams) -> Onset | None:
    """Onset of the transient peaking at sample ``j``, the same rule for every sound: the
    envelope's rising edge through ``k_floor`` and 2 × ``k_floor`` times the noise floor
    (median envelope over ``floor_s`` before the peak), extrapolated back to the floor. A
    plain threshold crossing comes later for a quieter sound (the rise to k × floor takes a
    larger share of it), which would lengthen Δt for a tape tick against a racket crack."""
    pre = (tt >= tt[j] - p.floor_s[0]) & (tt <= tt[j] - p.floor_s[1])
    if pre.sum() < 8:
        return None
    floor = float(np.median(env[pre])) + 1e-12
    snr = float(20 * math.log10(max(env[j], 1e-12) / floor))
    level = p.k_floor * floor
    if env[j] <= level:
        return Onset(float(tt[j]), snr, float(tt[j]), floor)
    m = j
    while m > 0 and env[m - 1] > level:
        m -= 1

    def crossing(i: int, lev: float) -> float:
        if i == 0:
            return float(tt[0])
        a, b = env[i - 1], env[i]
        frac = (lev - a) / (b - a) if b > a else 0.0
        return float(tt[i - 1] + frac * (tt[i] - tt[i - 1]))

    t1 = crossing(m, level)
    level2 = 2 * level
    if env[j] > level2:
        n = m
        while n < j and env[n] <= level2:
            n += 1
        t2 = crossing(n, level2)
        if t2 > t1:
            t_on = t1 - (t2 - t1) * (level - floor) / (level2 - level)
            return Onset(float(max(t_on, t1 - 0.001)), snr, float(tt[j]), floor)
    return Onset(t1, snr, float(tt[j]), floor)


def onset_at(audio: AudioClip, t: float, p: RefParams) -> Onset | None:
    """The onset measured at a time the user picked (its SNR from the peak just after)."""
    o = onset_near(audio, t + 0.002, 0.002, p)
    if o is None:
        return None
    return Onset(float(t), o.snr_db, o.t_peak, o.floor)


def waveform(audio: AudioClip, t_center: float, half: float, p: RefParams, n_max: int = 1600):
    """(times, high-passed waveform, envelope) around ``t_center`` for drawing, decimated
    to at most ``n_max`` points by min/max pairs."""
    y, t_first = audio.window(t_center - half, t_center + half)
    if len(y) < 32:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    hp = highpass(y, audio.sr, p)
    env = envelope(y, audio.sr, p)
    tt = t_first + np.arange(len(y)) / audio.sr
    step = max(1, len(y) // n_max)
    if step > 1:
        n = len(y) // step * step
        tt, hp, env = (a[:n].reshape(-1, step) for a in (tt, hp, env))
        tt = tt[:, 0]
        hp = np.where(np.arange(len(hp)) % 2 == 0, hp.max(axis=1), hp.min(axis=1))
        env = env.max(axis=1)
    return tt, hp, env


# ---------------------------------------------------------------------------
# One reference
# ---------------------------------------------------------------------------


@dataclass
class ServeInput:
    """What one near-end serve brings to a reference measurement."""

    swing_id: int
    shot_id: int | None
    t_contact: float  # video clock (the contact frame's time)
    camera: Camera
    #: The serve's fitted flight (``ball/flights.parquet`` row).
    flight: dict
    #: Detections of the serve's flight (video clock, display px).
    det_t: np.ndarray
    det_px: np.ndarray
    #: The toss fit (M7b): position + covariance at the contact frame, and its parameters.
    contact: np.ndarray | None = None
    contact_cov: np.ndarray | None = None
    toss_theta: np.ndarray | None = None
    toss_t0: float | None = None


@dataclass
class Reference:
    swing_id: int
    shot_id: int | None
    flight_id: int | None
    t_contact: float
    t_cross: float | None = None
    t_racket_audio: float | None = None
    t_tape_audio: float | None = None
    t_racket_pred_audio: float | None = None
    t_tape_pred_audio: float | None = None
    onsets_by: str = "auto"
    snr_racket_db: float | None = None
    snr_tape_db: float | None = None
    contact: np.ndarray | None = None
    tape: np.ndarray | None = None
    d_contact_m: float | None = None
    d_tape_m: float | None = None
    sound_speed_mps: float | None = None
    dt_s: float | None = None
    dt_sigma_s: float | None = None
    path_m: float | None = None
    path_sigma_m: float | None = None
    v_ref_kmh: float | None = None
    v_fit_kmh: float | None = None
    ratio: float | None = None
    ratio_sigma: float | None = None
    img_vy: float | None = None
    rs_rate: float | None = None
    clearance_m: float | None = None
    end_kind: str | None = None
    av_offset_s: float | None = None
    is_candidate: bool = False
    flags: list[str] = field(default_factory=list)


def _theta(f: dict) -> np.ndarray:
    return np.array(
        [f[k] for k in ("p0_x", "p0_y", "p0_z", "v0_x", "v0_y", "v0_z", "spin", "cd")], float
    )


def net_crossing(theta: np.ndarray, t0: float, t1: float, side: int = -1, dt: float = 0.005):
    """(time, position) where a fitted path first reaches the net plane heading away from
    the hitter's side, searched up to 0.3 s past ``t1`` (a net end can stop short of it).
    Interpolated between ``dt`` samples (the path bends < 0.2 mm in 5 ms)."""
    grid = np.arange(t0, t1 + 0.3 + dt, dt)
    pos = ph.simulate(theta[None], t0, grid)[0][0]
    y = pos[:, 1] * (-side)  # > 0 once past the net
    k = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
    if not len(k):
        return None, None
    k = int(k[0])
    f = -y[k] / (y[k + 1] - y[k])
    return float(grid[k] + f * dt), pos[k] + f * (pos[k + 1] - pos[k])


def arc_length(theta: np.ndarray, t0: float, ta: float, tb: float, n: int = 120) -> float:
    grid = np.linspace(ta, tb, n)
    pos = ph.simulate(theta[None], t0, grid)[0][0]
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def _deflected(s: ServeInput, t_cross: float, fps: float, p: RefParams, fp: ph.FitParams):
    """Whether the detections after the crossing left the path fitted to the ones before it
    (the ball changed velocity at the tape), and that fit (``None``: too few points). The
    fit only tells a let from a clean pass: without an end it is a weaker fit than the
    production one, so it never stands in for it."""
    t, px = s.det_t, s.det_px
    pre = (t > s.flight["t0_s"]) & (t <= t_cross)
    post = t > t_cross + 0.5 / fps
    if pre.sum() < fp.min_points or post.sum() < 3:
        return None, None
    start = ph.EndPoint(
        "hit", float(s.flight["t0_s"]), None, None,
        None if s.contact is None else np.asarray(s.contact, float),
        None if s.contact_cov is None else np.asarray(s.contact_cov, float),
    )  # fmt: skip
    if start.contact is None:
        start.contact = _theta(s.flight)[:3]
        start.contact_cov = np.eye(3) * 0.05**2
    sig = ph.point_sigma(px[pre], t[pre], s.camera.width, fp)
    fit = ph.fit_flight(ph.FlightData(s.camera, t[pre], px[pre], sig, start, None), fp)
    if not np.isfinite(fit.theta).all():
        return None, None
    pred = s.camera.project(ph.simulate(fit.theta[None], fit.t0, t[post])[0][0])
    err = np.hypot(*(pred - px[post]).T)
    sig_post = ph.point_sigma(px[post], t[post], s.camera.width, fp)
    scale = s.camera.width / 3840
    dev = float(np.median(err / sig_post))
    return bool(dev > p.deflect_sigmas and np.median(err) > p.deflect_px * scale), fit


def _quad(t: np.ndarray, px: np.ndarray):
    """Quadratics x(t), y(t) through image points (linear below 4), and their SSE."""
    deg = 2 if len(t) >= 4 else 1
    A = np.vander(t - t[0], deg + 1)
    coef, *_ = np.linalg.lstsq(A, px, rcond=None)
    return coef, float(np.sum((A @ coef - px) ** 2)), t[0]


def tape_kink(t: np.ndarray, px: np.ndarray, t_x: float, fps: float, half_s: float = 0.12):
    """Where a let's image track breaks at the tape: (time, pixel). The detections within
    ``half_s`` of the fitted crossing ``t_x`` are split where a quadratic either side fits
    best (a change point), and the break is where the two curves come closest between the
    last point before and the first after, like ``events`` refines a contact. ``None``: too
    few detections on one side."""
    idx = np.flatnonzero(np.abs(t - t_x) <= half_s)
    if len(idx) < 8:
        return None
    tw, pw = t[idx], px[idx]
    best = None
    for i in range(3, len(tw) - 4):  # ≥ 4 points either side
        cb, eb, tb = _quad(tw[: i + 1], pw[: i + 1])
        ca, ea, ta = _quad(tw[i + 1 :], pw[i + 1 :])
        if best is None or eb + ea < best[0]:
            best = (eb + ea, i, cb, tb, ca, ta)
    if best is None:
        return None
    _, i, cb, tb, ca, ta = best
    grid = np.linspace(tw[i], tw[i + 1], 121)

    def ev(c, t0):
        return np.column_stack([np.polyval(c[:, 0], grid - t0), np.polyval(c[:, 1], grid - t0)])

    pb, pa_ = ev(cb, tb), ev(ca, ta)
    k = int(np.argmin(np.hypot(*(pb - pa_).T)))
    return float(grid[k]), (pb[k] + pa_[k]) / 2


def let_fit(s: ServeInput, t_x: float, fps: float, fp: ph.FitParams):
    """A let's flight fitted the way ``ball_3d`` fits a serve that ends in the net: the
    detections up to the break at the tape, the toss contact prior, a net end at the break.
    (``ball_3d`` fitted one flight through the break, which bends it.) ``None``: no break."""
    kink = tape_kink(s.det_t, s.det_px, t_x, fps)
    if kink is None or s.contact is None:
        return None
    t_k, px_k = kink
    pre = (s.det_t > s.flight["t0_s"]) & (s.det_t < t_k)
    if pre.sum() < fp.min_points:
        return None
    start = ph.EndPoint(
        "hit", float(s.flight["t0_s"]), None, None, np.asarray(s.contact, float),
        np.asarray(s.contact_cov if s.contact_cov is not None else np.eye(3) * 0.05**2, float),
    )  # fmt: skip
    t, px = s.det_t[pre], s.det_px[pre]
    data = ph.FlightData(
        s.camera, t, px, ph.point_sigma(px, t, s.camera.width, fp), start,
        ph.EndPoint("net", t_k, np.asarray(px_k, float)),
    )  # fmt: skip
    fit = ph.fit_flight(data, fp)
    return fit if fit.ok else None


def measure(
    s: ServeInput,
    audio: AudioClip | None,
    *,
    container_offset_s: float,
    av_offset_s: float,
    temp_c: float | None,
    fps: float,
    rotation_cw: int,
    p: RefParams | None = None,
    fp: ph.FitParams | None = None,
    t_racket: float | None = None,
    t_tape: float | None = None,
) -> Reference:
    """Measure one near-end serve as a net-tape reference.

    ``container_offset_s``: the audio stream's start minus the video's (container times);
    ``av_offset_s``: how much later than that the sounds arrive (measured on the session's
    serves). ``t_racket`` / ``t_tape``: onsets the user placed (audio clock).
    """
    p = p or RefParams()
    fp = fp or ph.FitParams()
    f = s.flight
    ref = Reference(s.swing_id, s.shot_id, f.get("flight_id"), s.t_contact)
    ref.end_kind = f.get("end_kind")
    theta = _theta(f)
    t0, t1 = float(f["t0_s"]), float(f["t1_s"])
    if not np.isfinite(theta).all():
        ref.flags.append("no_flight_fit")
        return ref
    # A let's production fit may fail on the break (poor fit): it still says where the
    # break is, and the let is refitted below.
    prod_ok = bool(f.get("ok"))
    t_x, P_x = net_crossing(theta, t0, t1)
    if t_x is None or (ref.end_kind != "net" and t_x > t1):
        ref.flags.append("no_net_crossing")
        return ref
    ref.t_cross = t_x
    C = s.camera.center
    c = sound_speed(temp_c)
    ref.sound_speed_mps = c

    # The contact: the toss path at the impact (from the sound) or the contact frame.
    contact = s.contact if s.contact is not None else theta[:3]
    if s.contact is None:
        ref.flags.append("contact_from_fit")

    # Velocity change at the crossing (a let), or the flight ends there. A let's production
    # fit bends through the break, so it is refitted up to the tape like a net serve.
    deflected = None
    prod_clear = float(P_x[2] - ph.net_height(P_x[0]))
    if ref.end_kind != "net" and abs(prod_clear) <= p.let_gate_m:
        deflected, _pre = _deflected(s, t_x, fps, p, fp)
        if deflected:
            ref.flags.append("velocity_change")
            lf = let_fit(s, t_x, fps, fp)
            if lf is None:
                ref.flags.append("let_not_refitted")
            else:
                t_lx, P_lx = net_crossing(lf.theta, lf.t0, lf.t1)
                if t_lx is not None:
                    theta, t0, t_x, P_x = lf.theta, lf.t0, t_lx, P_lx
                    ref.t_cross = t_x
                    ref.flags.append("let_refitted")
        elif deflected is None:
            ref.flags.append("no_detections_after_net")
    if not prod_ok and "let_refitted" not in ref.flags:
        ref.flags.append("no_flight_fit")
        return ref

    tape_top = float(ph.net_height(P_x[0]))
    ref.clearance_m = float(P_x[2] - tape_top)
    near_tape = abs(P_x[2] - ph.BALL_RADIUS - tape_top) <= ph.BALL_RADIUS + p.tape_tol_m
    if not near_tape:
        ref.flags.append("above_tape" if ref.clearance_m > 0 else "below_tape")

    # The ball at the tape: its ray in the frame nearest the crossing, on the net plane.
    k = int(np.argmin(np.abs(s.det_t - t_x))) if len(s.det_t) else -1
    if k >= 0 and abs(s.det_t[k] - t_x) <= 1.5 / fps:
        ray = s.camera.rays(s.det_px[k][None])[0]
        lam = -C[1] / ray[1] if abs(ray[1]) > 1e-9 else np.nan
        tape_x = float(C[0] + lam * ray[0]) if np.isfinite(lam) and lam > 0 else float(P_x[0])
    else:
        tape_x = float(P_x[0])
        ref.flags.append("tape_from_fit")
    tape = np.array([tape_x, 0.0, float(ph.net_height(tape_x)) + 0.5 * ph.BALL_RADIUS])

    # Sounds.
    ref.d_contact_m = float(np.linalg.norm(contact - C))
    ref.d_tape_m = float(np.linalg.norm(tape - C))
    if audio is None:
        ref.flags.append("no_audio")
        return ref
    pred_r = s.t_contact + ref.d_contact_m / c - container_offset_s + av_offset_s
    ref.t_racket_pred_audio = pred_r
    on_r = (
        onset_at(audio, t_racket, p)
        if t_racket is not None
        else onset_near(audio, pred_r, p.racket_window_s, p)
    )
    if on_r is None:
        ref.flags.append("no_racket_sound")
        return ref
    ref.t_racket_audio, ref.snr_racket_db = on_r.t, on_r.snr_db
    if on_r.snr_db < p.racket_min_snr_db:
        ref.flags.append("weak_racket_sound")
    ref.av_offset_s = on_r.t - (s.t_contact + ref.d_contact_m / c - container_offset_s)
    pred_t = on_r.t + (t_x - t0) + (ref.d_tape_m - ref.d_contact_m) / c
    ref.t_tape_pred_audio = pred_t
    on_t = (
        onset_at(audio, t_tape, p)
        if t_tape is not None
        else onset_near(audio, pred_t, p.tape_window_s, p)
    )
    if t_racket is not None or t_tape is not None:
        ref.onsets_by = "user"
    if on_t is None:
        ref.flags.append("no_tape_sound")
        return ref
    ref.t_tape_audio, ref.snr_tape_db = on_t.t, on_t.snr_db
    if on_t.snr_db < p.tape_min_snr_db:
        ref.flags.append("no_tape_sound")

    # The impact on the video clock, from the racket sound: the toss path there.
    t_imp = on_r.t - ref.d_contact_m / c + container_offset_s - av_offset_s
    t_imp = float(np.clip(t_imp, s.t_contact - 0.5 / fps, s.t_contact + 1.5 / fps))
    if s.toss_theta is not None and s.toss_t0 is not None:
        contact = ph.simulate(np.asarray(s.toss_theta, float)[None], s.toss_t0, np.array([t_imp]))
        contact = contact[0][0, 0]
        ref.d_contact_m = float(np.linalg.norm(contact - C))
    ref.contact, ref.tape = np.asarray(contact, float), tape

    # Flight time and path.
    dt = (on_t.t - ref.d_tape_m / c) - (on_r.t - ref.d_contact_m / c)
    ref.dt_s = float(dt)
    t_sig = AIR_TEMP_SIGMA_SET if temp_c is not None else AIR_TEMP_SIGMA_UNSET
    c_rel = 0.5 * t_sig / (273.15 + (DEFAULT_AIR_TEMP_C if temp_c is None else temp_c))
    delay_sig = (ref.d_tape_m - ref.d_contact_m) / c * c_rel
    ref.dt_sigma_s = float(math.sqrt(2 * p.onset_sigma_s**2 + delay_sig**2))
    chord = tape - contact
    L = float(np.linalg.norm(chord))
    u = chord / max(L, 1e-9)
    # The production fit (``ball_3d``) over the same part of the flight, on its own clock;
    # its arc / chord bends the straight contact → tape distance into a path.
    arc = arc_length(theta, t0, t0, t_x)
    straight = float(np.linalg.norm(P_x - theta[:3]))
    ref.path_m = L * (arc / straight if straight > 0 else 1.0)
    cov_c = s.contact_cov if s.contact_cov is not None else np.eye(3) * 0.05**2
    sig_c = max(float(np.sqrt(max(u @ np.asarray(cov_c, float) @ u, 0.0))), p.contact_sigma_min_m)
    sig_t = p.tape_sigma_m * float(np.hypot(u[0], u[2]))  # the tape point is uncertain in x, z
    ref.path_sigma_m = float(math.hypot(sig_c, max(sig_t, 0.005)))
    if dt <= 0.05:
        ref.flags.append("implausible_dt")
        return ref
    ref.v_ref_kmh = ref.path_m / dt * KMH
    ref.v_fit_kmh = arc / (t_x - t0) * KMH
    ref.ratio = ref.v_ref_kmh / ref.v_fit_kmh
    fit_rel = (f.get("speed_avg_sigma") or 0.0) / max(f.get("speed_avg") or 1.0, 1e-6)
    if "let_refitted" in ref.flags:
        fit_rel = max(fit_rel, 0.015)  # its end time is the break's (≈ ±5 ms)
    ref.ratio_sigma = ref.ratio * math.sqrt(
        (ref.dt_sigma_s / dt) ** 2 + (ref.path_sigma_m / ref.path_m) ** 2 + fit_rel**2
    )
    # Image motion contact → tape (rolling shutter).
    W, H = s.camera.width, s.camera.height
    px_c, px_t = s.camera.project(np.vstack([theta[:3], P_x]))
    ref.img_vy = float((px_t[1] - px_c[1]) / (t_x - t0))
    sc, st = readout_fraction(np.vstack([px_c, px_t]), rotation_cw, W, H)
    ref.rs_rate = float((st - sc) / (t_x - t0))
    ref.is_candidate = bool(
        near_tape
        and (ref.end_kind == "net" or (deflected and "let_refitted" in ref.flags))
        and on_t.snr_db >= p.tape_min_snr_db
        and on_r.snr_db >= p.racket_min_snr_db
    )
    return ref


def estimate_av_offset(
    serves: list[ServeInput],
    audio: AudioClip,
    container_offset_s: float,
    temp_c: float | None,
    p: RefParams | None = None,
) -> tuple[float, int]:
    """The session's audio/video offset: how much later than the video contact plus the
    sound delay the racket sounds of its serves arrive (median of the strong ones), and how
    many serves measured it. Phones record audio ≈0.1 s behind their video."""
    p = p or RefParams()
    c = sound_speed(temp_c)
    lo, hi = p.av_search_s
    vals = []
    for s in serves:
        contact = s.contact if s.contact is not None else _theta(s.flight)[:3]
        if not np.isfinite(contact).all():
            continue
        base = s.t_contact + float(np.linalg.norm(contact - s.camera.center)) / c
        base -= container_offset_s
        o = onset_near(audio, base + (lo + hi) / 2, (hi - lo) / 2, p)
        if o is not None and o.snr_db >= p.racket_min_snr_db:
            vals.append(o.t - base)
    if len(vals) < p.av_min_serves:
        return 0.0, len(vals)
    v = np.asarray(vals)
    med = float(np.median(v))
    close = v[np.abs(v - med) <= 0.02]
    return float(np.median(close)), len(close)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    """A device's speed correction: speeds × :func:`speed_factor`."""

    model: str  # scalar | rolling_shutter
    k: float
    k_sigma: float
    tau_s: float | None = None
    tau_sigma_s: float | None = None
    n_refs: int = 0
    loo_sd: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def speed_factor(cal: dict | Calibration | None, rs_rate: float | None = None) -> float:
    """The factor a speed is multiplied by: k, or with a readout time τ
    ``1 / (1/k + τ · rs_rate)`` for a flight crossing ``rs_rate`` of the sensor per second."""
    if cal is None:
        return 1.0
    d = cal.as_dict() if isinstance(cal, Calibration) else cal
    k = float(d["k"])
    tau = d.get("tau_s")
    if d.get("model") == "rolling_shutter" and tau is not None and rs_rate is not None:
        return 1.0 / (1.0 / k + float(tau) * float(rs_rate))
    return k


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
    """Weighted straight line y = a + b x: (a, b, σ_b, t-statistic of b), the σ scaled by
    the residual scatter when that is larger than the weights say."""
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    A = X.T @ W @ X
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan"), 0.0
    a, b = cov @ X.T @ W @ y
    r = y - (a + b * x)
    dof = max(1, len(x) - 2)
    chi2 = float(np.sum(w * r * r)) / dof
    sb = float(math.sqrt(cov[1, 1] * max(1.0, chi2)))
    return float(a), float(b), sb, float(b / sb) if sb > 0 else 0.0


def _p_two_sided(t: float, dof: int) -> float:
    from scipy import stats

    return float(2 * stats.t.sf(abs(t), max(1, dof)))


def fit_calibration(refs: list[dict], n_boot: int = 2000, seed: int = 0) -> Calibration | None:
    """The correction model from accepted references (dicts with ``ratio``, ``ratio_sigma``,
    ``v_fit_kmh``, ``rs_rate``, ``session_id``), chosen by the data (§7.12)."""
    refs = [
        r for r in refs
        if r.get("ratio") is not None and np.isfinite(r["ratio"]) and r.get("ratio_sigma")
    ]  # fmt: skip
    n = len(refs)
    if n < MIN_REFS:
        return None
    ratio = np.array([r["ratio"] for r in refs], float)
    sig = np.array([r["ratio_sigma"] for r in refs], float)
    w = 1.0 / sig**2
    k = _wmean(ratio, w)
    resid = ratio - k
    chi2 = float(np.sum(w * resid**2)) / max(1, n - 1)
    k_stat = math.sqrt(1.0 / np.sum(w) * max(1.0, chi2))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    boot = np.sum(w[idx] * ratio[idx], axis=1) / np.sum(w[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    k_sigma = math.hypot(max(k_stat, float(np.std(boot))), SYSTEMATIC * k)
    loo = []
    for i in range(n):
        m = np.arange(n) != i
        loo.append(ratio[i] / _wmean(ratio[m], w[m]) - 1.0)
    diag: dict = {
        "chi2_dof": round(chi2, 3),
        "k_ci95": [float(lo), float(hi)],
        "k_stat": k_stat,
        "loo": [float(v) for v in loo],
    }
    # Trends: vs speed, vs the rolling-shutter rate, vs session.
    trend_rs = None
    for name, key in (("speed", "v_fit_kmh"), ("rolling_shutter", "rs_rate")):
        x = np.array([r.get(key) if r.get(key) is not None else np.nan for r in refs], float)
        ok = np.isfinite(x)
        if ok.sum() >= 4 and np.ptp(x[ok]) > 0:
            a, b, sb, t = _wls(x[ok], ratio[ok], w[ok])
            pval = _p_two_sided(t, int(ok.sum()) - 2)
            diag[f"trend_{name}"] = {
                "slope": b, "slope_sigma": sb, "p": pval, "n": int(ok.sum()),
                "significant": bool(pval < 0.05),
            }  # fmt: skip
            if name == "rolling_shutter":
                trend_rs = diag[f"trend_{name}"]
    sessions = sorted({r.get("session_id") for r in refs if r.get("session_id")})
    if len(sessions) >= 2:
        from scipy import stats

        chi = 0.0
        per = {}
        for sid in sessions:
            m = np.array([r.get("session_id") == sid for r in refs])
            km = _wmean(ratio[m], w[m])
            per[sid] = {"k": km, "n": int(m.sum())}
            chi += float(np.sum(w[m]) * (km - k) ** 2 / max(1.0, chi2))
        pval = float(stats.chi2.sf(chi, len(sessions) - 1))
        diag["by_session"] = {"sessions": per, "p": pval, "significant": bool(pval < 0.05)}
    cal = Calibration(
        "scalar", k, k_sigma, n_refs=n, loo_sd=float(np.std(loo, ddof=1)) if n > 2 else None,
        diagnostics=diag,
    )  # fmt: skip
    if trend_rs is not None and trend_rs["significant"] and n >= 6:
        # 1/ratio = 1/k + τ · rs_rate, weighted by the ratio's σ carried over.
        x = np.array([r["rs_rate"] for r in refs], float)
        y = 1.0 / ratio
        wy = 1.0 / (sig / ratio**2) ** 2
        a, b, sb, _t = _wls(x, y, wy)
        if a > 0:
            X = np.column_stack([np.ones_like(x), x])
            cov = np.linalg.inv(X.T @ np.diag(wy) @ X)
            ka = 1.0 / a
            ka_sig = math.hypot(math.sqrt(cov[0, 0]) / a**2, SYSTEMATIC * ka)
            cal = Calibration(
                "rolling_shutter", ka, ka_sig, tau_s=b, tau_sigma_s=sb, n_refs=n,
                loo_sd=cal.loo_sd, diagnostics=diag,
            )  # fmt: skip
    return cal
