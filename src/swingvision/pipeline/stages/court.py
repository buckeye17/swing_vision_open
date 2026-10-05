"""M1 stages: ``court_auto`` (detect + drift check) and ``camera`` (calibration gate)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from swingvision.court import calibration as calib
from swingvision.pipeline.stage import NeedsUserAction, Stage, StageContext
from swingvision.pipeline.stages.ingest import _source_path
from swingvision.storage.fsutil import atomic_write

CALIBRATION_REVIEW = "calibration_review"


def write_jpeg(path: Path, rgb: np.ndarray, quality: int = 92) -> None:
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    atomic_write(path, lambda tmp: tmp.write_bytes(buf.tobytes()), suffix=".jpg")


def read_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class CourtAutoStage(Stage):
    name = "court_auto"
    title = "Court detection"
    version = 1
    depends_on = ("ingest",)
    weight = 1.5

    def config(self, session, config, settings):
        p = settings.processing
        return {
            "window_s": p.calibration_window_s,
            "per_window": p.calibration_frames_per_window,
            "drift_px": p.calibration_drift_px,
        }

    def outputs(self, session):
        return [session.court_auto_path, session.court_background_path]

    def run(self, ctx: StageContext):
        src = _source_path(ctx)
        info = ctx.config.video
        assert info is not None, "ingest must run first"
        p = ctx.settings.processing
        result = calib.auto_calibrate(
            ctx.settings.ffmpeg(),
            src,
            info,
            window_s=p.calibration_window_s,
            per_window=p.calibration_frames_per_window,
            drift_px=p.calibration_drift_px,
            progress=ctx.progress,
            check_cancel=ctx.check_cancel,
        )
        session = ctx.session
        write_jpeg(session.court_background_path, result.background)
        shutil.rmtree(session.court_window_path(0).parent, ignore_errors=True)
        for i, img in result.window_images.items():
            write_jpeg(session.court_window_path(i), img, quality=88)
        cal = result.calibration
        calib.save(session.court_auto_path, cal)
        ctx.log.info("Court detection: {}", cal.message)
        return {
            "ok": cal.ok,
            "message": cal.message,
            "rms_line_px": cal.metrics.rms_line_px,
            "coverage": round(cal.metrics.coverage, 3),
            "drift_detected": cal.drift_detected,
            "dark_fraction": round(cal.dark_fraction, 3),
            "camera": cal.camera_summary,
        }


class CameraStage(Stage):
    """Resolve the calibration downstream stages use, pausing for review when needed.

    ``court/user.json`` (confirmed in the editor) wins. Otherwise the auto calibration is
    used only if it passes the auto-accept threshold in Settings; if not, the job stops
    with ``needs_action`` until the user confirms a calibration on the Calibrate page.
    The per-window drift cameras are re-derived from the confirmed camera.
    """

    name = "camera"
    title = "Camera calibration"
    version = 1
    depends_on = ("court_auto",)
    weight = 0.5

    def config(self, session, config, settings):
        # Downstream results depend only on the chosen camera, so confirming an unchanged
        # calibration doesn't invalidate anything.
        user = calib.load(session.court_user_path)
        if user is not None:
            return {
                "source": "user",
                "camera": calib.geometry_key(user),
                "drift_px": settings.processing.calibration_drift_px,
            }
        auto = calib.load(session.court_auto_path)
        if auto is not None:
            return {"source": "auto", "camera": calib.geometry_key(auto)}
        return {"source": None}

    def outputs(self, session):
        return [session.calibration_path]

    def run(self, ctx: StageContext):
        session = ctx.session
        auto = calib.load(session.court_auto_path)
        user = calib.load(session.court_user_path)
        if user is not None:
            drift = auto.drift if auto is not None else []
            if drift:
                ctx.progress(0.0, "Re-checking camera movement")
                drift = calib.window_drift(
                    calib.to_camera(user.camera),
                    drift,
                    lambda i: read_rgb(session.court_window_path(i)),
                    ctx.settings.processing.calibration_drift_px,
                    progress=ctx.progress,
                    check_cancel=ctx.check_cancel,
                )
            cal = user.model_copy(
                update={
                    "confirmed_by": "user",
                    "drift": drift,
                    "drift_detected": any(d.status == "moved" for d in drift),
                }
            )
            note = "confirmed by user"
        else:
            if auto is None:
                raise RuntimeError("court_auto produced no calibration")
            threshold = ctx.settings.processing.calibration_auto_accept_px
            ok, why = calib.auto_acceptable(auto, threshold)
            if not ok:
                session.calibration_path.unlink(missing_ok=True)
                raise NeedsUserAction(
                    CALIBRATION_REVIEW,
                    f"Review the court calibration ({why}). Open Calibrate, adjust if needed, "
                    "and confirm.",
                )
            cal = auto.model_copy(update={"confirmed_by": "auto"})
            note = f"auto-accepted: {why}"
        calib.save(session.calibration_path, cal)
        ctx.log.info("Calibration {}", note)
        return {
            "confirmed_by": cal.confirmed_by,
            "note": note,
            "drift_detected": cal.drift_detected,
            "camera": cal.camera_summary,
        }
