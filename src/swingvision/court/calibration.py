"""Session-level calibration: frame sampling, detection, drift check, editor solves, files.

Files (PLAN.md §5):

* ``court/background.jpg``: median of frames sampled across the video (players removed).
* ``court/auto.json``: what :func:`auto_calibrate` detected (``court_auto`` stage).
* ``court/user.json``: what the user confirmed in the calibration editor.
* ``calibration.json``: the calibration downstream stages use (``camera`` stage). It is
  ``user.json`` when present, otherwise ``auto.json`` if it may be auto-accepted.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from swingvision.court import model
from swingvision.court.camera import (
    Camera,
    FitOptions,
    FitResult,
    PointObs,
    default_camera,
    fit_camera,
    line_distances,
)
from swingvision.court.detect import (
    Detection,
    Prepared,
    detect_court,
    expected_samples,
    inlier_limit,
    refine,
    refine_from,
    ridge_samples,
)
from swingvision.io.frames import grab_frames, luminance, median_image
from swingvision.storage.fsutil import atomic_write_text
from swingvision.storage.schemas import (
    Calibration,
    CalibrationMetrics,
    CameraParams,
    DriftWindow,
    VideoInfo,
)

DARK_LUMINANCE = 25.0  # mean luma below which a frame is too dark to calibrate from
#: Up to 3 hours at the default 5-minute windows: longer windows blur a slowly sagging
#: mount in their median image.
MAX_WINDOWS = 36
#: A drift window whose pose-only refit is worse than this (line RMS) gets a full re-fit.
POOR_WINDOW_PX = 2.0

#: Drift check: a bumped tripod changes the pose, not the lens.
POSE_ONLY = FitOptions(fix_f=True, fit_k1=False)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def to_params(cam: Camera) -> CameraParams:
    return CameraParams(**cam.to_dict())


def to_camera(params: CameraParams) -> Camera:
    return Camera.from_dict(params.model_dump())


def projected_keypoints(cam: Camera) -> dict[str, list[float] | None]:
    """Projected keypoints (px); ``None`` where the projection is undefined (far off-frame)."""
    names = list(model.KEYPOINTS)
    X = model.keypoint_array(names)
    px = cam.project(X)
    ok = np.isfinite(px).all(axis=1) & (cam.depth(X) > 0)
    return {
        n: [round(float(p[0]), 2), round(float(p[1]), 2)] if good else None
        for n, p, good in zip(names, px, ok, strict=True)
    }


def visible(cam: Camera, px: np.ndarray, margin: float = 0.0) -> np.ndarray:
    return (
        (px[..., 0] >= -margin)
        & (px[..., 0] < cam.width + margin)
        & (px[..., 1] >= -margin)
        & (px[..., 1] < cam.height + margin)
    )


def keypoint_shift(a: Camera, b: Camera) -> tuple[float, float] | None:
    """RMS and max displacement (px) of the in-frame ground keypoints between two cameras."""
    X = model.keypoint_array(list(model.GROUND_KEYPOINTS))
    pa, pb = a.project(X), b.project(X)
    m = visible(a, pa) & (a.depth(X) > 0)
    if not m.any():
        return None
    d = np.linalg.norm(pa[m] - pb[m], axis=1)
    return float(np.sqrt(np.mean(d**2))), float(d.max())


def calibration_from_detection(
    det: Detection, width: int, height: int, source: str = "auto", message: str = ""
) -> Calibration:
    cam = det.camera if det.ok and det.camera is not None else default_camera(width, height)
    metrics = CalibrationMetrics()
    if det.ok:
        metrics = CalibrationMetrics(
            rms_line_px=det.rms_px,
            n_line_samples=det.n_samples,
            n_expected_samples=det.n_expected,
            n_net_samples=det.n_net_samples,
        )
    return Calibration(
        source=source,  # type: ignore[arg-type]
        created_at=datetime.now(UTC),
        ok=det.ok,
        message=message or det.message,
        camera=to_params(cam),
        keypoints=projected_keypoints(cam),
        metrics=metrics,
        camera_summary=cam.describe(),
    )


# ---------------------------------------------------------------------------
# Automatic calibration over a whole video
# ---------------------------------------------------------------------------


@dataclass
class Window:
    t0: float
    t1: float
    times: list[float]


def sample_windows(duration_s: float, window_s: float, per_window: int) -> list[Window]:
    """Split the video into ≤ MAX_WINDOWS windows and pick evenly spaced frames in each."""
    if duration_s <= 0:
        return []
    n = max(1, min(MAX_WINDOWS, math.ceil(duration_s / window_s)))
    per = per_window if n > 1 else max(per_window, 9)
    length = duration_s / n
    out = []
    for i in range(n):
        t0, t1 = i * length, (i + 1) * length
        lo, hi = t0 + min(1.0, 0.05 * length), t1 - min(1.0, 0.05 * length)
        out.append(Window(t0, t1, [lo + (hi - lo) * (k + 0.5) / per for k in range(per)]))
    return out


@dataclass
class AutoResult:
    calibration: Calibration
    background: np.ndarray  # RGB, full resolution
    #: Per-window median images (RGB) by window index, for re-checking drift later.
    window_images: dict[int, np.ndarray]


def pose_refine(image: np.ndarray | Prepared, camera: Camera) -> Detection:
    """Re-fit only the pose of ``camera`` to the lines in ``image`` (a moved tripod)."""
    prep = image if isinstance(image, Prepared) else Prepared.from_rgb(image)
    return refine(
        prep.resp, camera, prep.ridge_threshold, schedule=(16.0, 10.0, 8.0, 7.0),
        use_net=False, model_options=POSE_ONLY, early_options=POSE_ONLY,
    )  # fmt: skip


def _dominant_group(cams: dict[int, Camera], drift_px: float) -> list[int]:
    """Largest set of windows whose cameras agree (within ``drift_px``) with one window."""
    best: list[int] = []
    for ci in cams.values():
        group = [
            j for j, cj in cams.items() if (keypoint_shift(ci, cj) or (0.0, 0.0))[0] <= drift_px
        ]
        if len(group) > len(best):
            best = group
    return sorted(best)


def window_drift(
    main: Camera,
    windows: list[DriftWindow],
    images: Callable[[int], np.ndarray | None],
    drift_px: float,
    progress: Callable[[float, str], None] = lambda f, m: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> list[DriftWindow]:
    """Pose-refine ``main`` on every usable window image; flag windows where it moved."""
    out = []
    for k, win in enumerate(windows):
        check_cancel()
        progress(k / max(1, len(windows)), f"Drift check ({k + 1}/{len(windows)})")
        if win.status == "dark" or win.index is None:
            out.append(win)
            continue
        img = images(win.index)
        prep = Prepared.from_rgb(img) if img is not None else None
        fit = _window_fit(prep, main) if prep is not None else None
        if fit is None:
            out.append(win.model_copy(update={"status": "failed", "camera": None}))
            continue
        cam, line_rms = fit
        shift = keypoint_shift(main, cam)
        rms, mx = shift if shift else (None, None)
        out.append(
            win.model_copy(
                update={
                    "status": "moved" if rms is not None and rms > drift_px else "ok",
                    "shift_rms_px": rms,
                    "shift_max_px": mx,
                    "rms_line_px": line_rms,
                    "camera": to_params(cam),
                }
            )
        )
    return out


def _window_fit(prep: Prepared, main: Camera) -> tuple[Camera, float | None] | None:
    """The camera of one drift window: ``main`` pose-refined, or, when that fits poorly (the
    camera was re-aimed or remounted far from ``main``), the best of that, a full
    refinement and a fresh detection: the lowest line RMS among the fits that found most
    of the lines."""
    fits: list[tuple[Camera, CalibrationMetrics]] = []
    det = pose_refine(prep, main)
    if det.ok and det.camera is not None:
        m = evaluate(prep, det.camera)
        # A pose far off can still match a few lines (a neighboring court) closely, so a
        # good fit must also find about as many line samples as the main camera expects.
        if (
            m.rms_line_px is not None
            and m.rms_line_px <= POOR_WINDOW_PX
            and m.n_line_samples >= 0.5 * expected_samples(main)
        ):
            return det.camera, m.rms_line_px
        fits.append((det.camera, m))
    for redo in (lambda: refine_from(prep, main), lambda: detect_court(prep)):
        d = redo()
        if d.ok and d.camera is not None:
            fits.append((d.camera, evaluate(prep, d.camera)))
    fits = [f for f in fits if f[1].rms_line_px is not None and np.isfinite(f[1].rms_line_px)]
    if not fits:
        return None
    most = max(f[1].n_line_samples for f in fits)
    cam, m = min(
        (f for f in fits if f[1].n_line_samples >= 0.6 * most),
        key=lambda f: f[1].rms_line_px,  # type: ignore[arg-type,return-value]
    )
    return cam, m.rms_line_px


def auto_calibrate(
    ffmpeg: str,
    src: Path,
    info: VideoInfo,
    window_s: float = 300.0,
    per_window: int = 5,
    drift_px: float = 3.0,
    progress: Callable[[float, str], None] = lambda f, m: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> AutoResult:
    """Calibrate on the dominant camera position and re-check every time window.

    1. Each window's frames → a median image (dark frames skipped).
    2. Detect the court on the median of all windows.
    3. If windows disagree (the camera was moved), re-calibrate on the median of the
       largest agreeing group of windows.
    4. Pose-refine the result on every window: windows where the camera sits elsewhere
       are flagged ``moved`` and keep their own camera (piecewise calibration).
    """
    windows = sample_windows(info.duration_s, window_s, per_window)
    medians: dict[int, np.ndarray] = {}
    entries: list[DriftWindow] = []
    used_times: list[float] = []
    dark_frames = total_frames = 0
    fallback: list[np.ndarray] = []
    for i, win in enumerate(windows):
        check_cancel()
        frames = grab_frames(ffmpeg, src, win.times, info)
        good, good_times, wl = [], [], []
        for t, fr in zip(win.times, frames, strict=True):
            if fr is None:
                continue
            total_frames += 1
            lum = luminance(fr)
            wl.append(lum)
            if lum < DARK_LUMINANCE:
                dark_frames += 1
                continue
            good.append(fr)
            good_times.append(t)
        if not good and not fallback:
            fallback = [f for f in frames if f is not None]
        if good:
            medians[i] = median_image(good) if len(good) >= 2 else good[0]
        entries.append(
            DriftWindow(
                index=i, t0_s=win.t0, t1_s=win.t1, n_frames=len(good),
                luminance=float(np.mean(wl)) if wl else None,
                status="ok" if good else "dark",
            )
        )  # fmt: skip
        used_times += good_times
        progress(0.55 * (i + 1) / len(windows), f"Sampling frames ({i + 1}/{len(windows)} windows)")

    w, h = info.display_width, info.display_height
    if not medians:
        background = (
            median_image(fallback)
            if len(fallback) >= 2
            else (fallback[0] if fallback else np.zeros((h, w, 3), np.uint8))
        )
        det = Detection(False, None, message="Every sampled frame is too dark to calibrate")
        cal = calibration_from_detection(det, w, h)
        cal.dark_fraction = 1.0
        cal.drift = entries
        return AutoResult(cal, background, {})

    usable = list(medians.values())
    background = usable[0] if len(usable) == 1 else median_image(usable)
    check_cancel()
    progress(0.57, "Detecting court lines")
    det = detect_court(Prepared.from_rgb(background))

    if det.ok and det.camera is not None and len(medians) > 1:
        progress(0.7, "Checking for camera movement")
        window_cams = {}
        for i, med in medians.items():
            check_cancel()
            wd = pose_refine(med, det.camera)
            if wd.ok and wd.camera is not None:
                window_cams[i] = wd.camera
        group = _dominant_group(window_cams, drift_px)
        if group and len(group) < len(medians):
            # The camera moved: calibrate on the position it held longest.
            imgs = [medians[i] for i in group]
            bg = imgs[0] if len(imgs) == 1 else median_image(imgs)
            redo = refine_from(Prepared.from_rgb(bg), det.camera, schedule=(16.0, 10.0, 8.0, 7.0))
            if redo.ok and redo.camera is not None:
                det, background = redo, bg

    cal = calibration_from_detection(det, w, h)
    if det.ok and det.camera is not None:
        # Report the same measurement the editor shows (all samples near the projection,
        # no outlier rejection), so numbers are comparable before and after edits.
        cal.metrics = evaluate(Prepared.from_rgb(background), det.camera)
    cal.frame_times_s = [round(t, 3) for t in used_times]
    cal.dark_fraction = dark_frames / total_frames if total_frames else 0.0
    if det.ok and det.camera is not None and len(medians) > 1:
        cal.drift = window_drift(
            det.camera, entries, medians.get, drift_px,
            progress=lambda f, m: progress(0.85 + 0.15 * f, m), check_cancel=check_cancel,
        )  # fmt: skip
    else:
        cal.drift = entries
    cal.drift_detected = any(d.status == "moved" for d in cal.drift)
    if cal.drift_detected:
        cal.message += ". The camera moves during the video (see the drift check)."
    progress(1.0, cal.message)
    return AutoResult(cal, background, medians)


#: Windows whose pose-refined camera differs from the session camera by more than this
#: use their own camera (slow sag or a bumped tripod); below it, the difference is noise.
PIECEWISE_MIN_SHIFT_PX = 1.0


def _window_cameras(cal: Calibration) -> list[tuple[float, float, CameraParams | None]]:
    """(t0, t1, camera) per drift window, ``None`` meaning the session camera.

    A window too dark (or failed) to calibrate keeps the camera of the nearest earlier
    window that had one (else the nearest later one): a camera doesn't move in the dark,
    and the session camera may be from a part of the video where it sat elsewhere.
    """
    own: list[CameraParams | bool | None] = []  # False: no measurement in this window
    for win in cal.drift:
        if win.camera is None:
            own.append(False)
        else:
            moved = (win.shift_rms_px or 0.0) > PIECEWISE_MIN_SHIFT_PX
            own.append(win.camera if moved else None)
    filled = list(own)
    last: CameraParams | bool | None = False
    for i, c in enumerate(own):
        if c is False:
            filled[i] = last
        else:
            last = c
    nxt: CameraParams | bool | None = False
    for i in range(len(filled) - 1, -1, -1):
        if own[i] is not False:
            nxt = own[i]
        elif filled[i] is False:
            filled[i] = nxt
    return [
        (w.t0_s, w.t1_s, None if c is False else c)  # type: ignore[misc]
        for w, c in zip(cal.drift, filled, strict=True)
    ]


def camera_at(cal: Calibration, t_s: float) -> Camera:
    """The camera for time ``t_s``: the window's own camera where the view had shifted."""
    for t0, t1, cam in _window_cameras(cal):
        if t0 <= t_s < t1 and cam is not None:
            return to_camera(cam)
    return to_camera(cal.camera)


def cameras_for_times(cal: Calibration, t_s: np.ndarray) -> tuple[list[Camera], np.ndarray]:
    """Vectorized :func:`camera_at`: the distinct cameras and, per time, which one applies."""
    cams = [to_camera(cal.camera)]
    which = np.zeros(len(t_s), dtype=np.int64)
    t_s = np.asarray(t_s, dtype=np.float64)
    seen: dict[int, int] = {}
    for t0, t1, cam in _window_cameras(cal):
        if cam is None:
            continue
        inside = (t_s >= t0) & (t_s < t1)
        if inside.any():
            if id(cam) not in seen:
                cams.append(to_camera(cam))
                seen[id(cam)] = len(cams) - 1
            which[inside] = seen[id(cam)]
    return cams, which


def all_cameras(cal: Calibration) -> list[Camera]:
    """The session camera plus every window's own camera (for crops that must fit all)."""
    return [to_camera(cal.camera)] + [
        to_camera(w.camera)
        for w in cal.drift
        if w.camera is not None and (w.shift_rms_px or 0.0) > PIECEWISE_MIN_SHIFT_PX
    ]


# ---------------------------------------------------------------------------
# Editor solves
# ---------------------------------------------------------------------------

#: Weight of keypoints the user did not place, relative to placed ones. They act as a
#: prior that keeps the camera near its previous estimate where the user gave no input.
PRIOR_WEIGHT = 0.01
MIN_POINTS_NO_PRIOR = 6


def solve_from_points(
    init: Camera, user_points: dict[str, list[float] | tuple[float, float]]
) -> FitResult:
    """Fit the camera to user-placed points (plus a weak pull toward ``init`` elsewhere).

    Lens distortion and principal point stay fixed (few points can't constrain them);
    "Snap to lines" refits everything against the background image.
    """
    names = list(model.KEYPOINTS)
    X = model.keypoint_array(names)
    img = init.project(X)
    placed = [n in user_points and user_points[n] is not None for n in names]
    # With enough placed points the fit is fully determined; the prior would only bias it.
    w = np.full(len(names), PRIOR_WEIGHT if sum(placed) < MIN_POINTS_NO_PRIOR else 0.0)
    for i, n in enumerate(names):
        if placed[i]:
            img[i] = user_points[n]
            w[i] = 1.0
    ok = np.isfinite(img).all(axis=1) & (init.depth(X) > 0) & (w > 0)
    pts = PointObs(X[ok], img[ok], w[ok])
    n_user = int((w[ok] == 1.0).sum())
    opts = FitOptions(fit_k1=False, fix_f=n_user < 4, loss="linear")
    fit = fit_camera(init, points=pts, options=opts)
    user = w[ok] == 1.0
    if n_user and fit.point_residuals is not None:
        fit.rms_points_px = float(np.sqrt(np.mean(fit.point_residuals[user] ** 2)))
    else:
        fit.rms_points_px = None
    return fit


def evaluate(
    image: np.ndarray | Prepared, cam: Camera, halfwidth: float = 6.0
) -> CalibrationMetrics:
    """How well ``cam`` fits the court lines in ``image``, without changing it.

    Line centers are searched only a few pixels around the projected lines, so a
    misaligned camera shows up as both a higher RMS and fewer lines found.
    """
    prep = image if isinstance(image, Prepared) else Prepared.from_rgb(image)
    lines = tuple(range(len(model.COURT_LINES) + len(model.NET_LINES)))
    samples = ridge_samples(prep.resp, cam, halfwidth, prep.ridge_threshold, lines)
    ground = samples.line < len(model.COURT_LINES)
    if ground.sum() == 0:
        return CalibrationMetrics(n_expected_samples=expected_samples(cam))
    res = np.abs(line_distances(cam, samples.line_obs()))
    # Same outlier rule as the refinement: occluders and stray marks don't count.
    inlier = res < inlier_limit(res[ground])
    g = ground & inlier
    return CalibrationMetrics(
        rms_line_px=float(np.sqrt(np.mean(res[g] ** 2))) if g.any() else None,
        n_line_samples=int(g.sum()),
        n_expected_samples=expected_samples(cam),
        n_net_samples=int((~ground & inlier).sum()),
    )


def snap_to_lines(background: np.ndarray | Prepared, init: Camera) -> Detection:
    """Refine an approximate camera against the line pixels of the background image."""
    prep = background if isinstance(background, Prepared) else Prepared.from_rgb(background)
    det = refine(prep.resp, init, prep.ridge_threshold, schedule=(40.0, 24.0, 14.0, 9.0, 7.0))
    if det.ok:
        det.message = f"{det.n_samples} line samples, RMS {det.rms_px:.2f} px"
    return det


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def load(path: Path) -> Calibration | None:
    if not path.exists():
        return None
    try:
        return Calibration.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(path: Path, cal: Calibration) -> None:
    atomic_write_text(path, cal.model_dump_json(indent=2))


def geometry_key(cal: Calibration) -> dict:
    """What downstream results depend on: the camera, rounded to remove float noise."""
    out = {}
    for k, v in cal.camera.model_dump().items():
        if isinstance(v, list):
            v = [round(x, 9) for x in v]
        elif isinstance(v, float):
            v = round(v, 9)
        out[k] = v
    return out


def with_camera(cal: Calibration, cam: Camera, **updates) -> Calibration:
    """Copy of ``cal`` with a new camera (keypoints and summary recomputed)."""
    return cal.model_copy(
        update={
            "camera": to_params(cam),
            "keypoints": projected_keypoints(cam),
            "camera_summary": cam.describe(),
            **updates,
        }
    )


def auto_acceptable(cal: Calibration, threshold_px: float | None) -> tuple[bool, str]:
    """Whether the auto calibration may be used without review, and why not."""
    if threshold_px is None:
        return False, "Auto-accept is off in Settings"
    if not cal.ok:
        return False, cal.message or "Court detection failed"
    rms = cal.metrics.rms_line_px
    if rms is None or rms > threshold_px:
        return (
            False,
            f"Line RMS {rms if rms is not None else float('nan'):.2f} px exceeds {threshold_px:g} px",
        )
    if cal.metrics.coverage < 0.3:
        return False, f"Only {cal.metrics.coverage:.0%} of the visible lines were found"
    if any(w.status == "failed" for w in cal.drift):
        return False, "The court wasn't found in part of the video (see the drift check)"
    if cal.drift_detected:
        # A sagging mount or a bumped tripod is fine unattended when every window where
        # the camera sat elsewhere fits the lines just as well with its own camera.
        moved = [w for w in cal.drift if w.status == "moved"]
        bad = [
            w
            for w in moved
            if w.camera is None or w.rms_line_px is None or w.rms_line_px > threshold_px
        ]
        if bad:
            return False, (
                f"The camera moves during the video and {len(bad)} of its {len(moved)} "
                "moved time windows don't fit well on their own"
            )
        worst = max(w.rms_line_px for w in moved)  # type: ignore[type-var]
        return True, (
            f"Line RMS {rms:.2f} px ≤ {threshold_px:g} px; the camera moves in "
            f"{len(moved)} time windows, each fitted on its own (≤ {worst:.2f} px)"
        )
    return True, f"Line RMS {rms:.2f} px ≤ {threshold_px:g} px"
