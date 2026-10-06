"""M4 stages: ``ball_3d`` (3D flights between events) and ``shots`` (PLAN.md §6, §7.6, §8)."""

from __future__ import annotations

from swingvision.analysis.shots import assemble_shots
from swingvision.ball import flights as fl
from swingvision.ball import physics as ph
from swingvision.court import calibration as calib
from swingvision.pipeline.stage import Stage, StageContext
from swingvision.storage import tables
from swingvision.storage.fsutil import atomic_write_json, read_json
from swingvision.storage.schemas import BALL_FLIGHT_PATHS, BALL_FLIGHTS, SHOTS


def _camera_at(cal):
    cams: dict[int, object] = {}

    def camera_at(t: float):
        key = int(t // 30)  # drift windows are minutes long
        if key not in cams:
            cams[key] = calib.camera_at(cal, t)
        return cams[key]

    return camera_at


class Ball3DStage(Stage):
    name = "ball_3d"
    title = "3D ball flight"
    version = 1
    depends_on = ("events", "ball_track", "camera")
    weight = 2.0

    def config(self, session, config, settings):
        return {
            "fit": ph.FitParams().as_config(),
            "flight_max_s": fl.FLIGHT_MAX_S,
            "lost_gap_s": fl.LOST_GAP_S,
            "free_max_s": fl.FREE_MAX_S,
            "contact_max_m": fl.CONTACT_MAX_M,
        }

    def outputs(self, session):
        return [
            session.ball_flights_path,
            session.ball_flight_paths_path,
            session.ball_flights_summary_path,
        ]

    def run(self, ctx: StageContext):
        session, config = ctx.session, ctx.config
        assert config.video is not None
        cal = calib.load(session.calibration_path)
        if cal is None:
            raise RuntimeError("No calibration")
        track = tables.read_table(session.ball_track_path)
        events = tables.read_table(session.events_path)
        fctx = fl.FlightContext(_camera_at(cal), config.video.display_width, ph.FitParams())
        specs, fits, rejected = fl.fit_session(
            track,
            events,
            fctx,
            progress=lambda f: ctx.progress(0.95 * f, "Fitting flights"),
            check_cancel=ctx.check_cancel,
        )
        flights = fl.flights_table(specs, fits)
        summary = {"rejected_hits": sorted(rejected)}
        atomic_write_json(session.ball_flights_summary_path, summary)
        tables.write_table(flights, session.ball_flights_path, BALL_FLIGHTS)
        ctx.progress(0.97, "Sampling paths")
        tables.write_table(
            fl.paths_table(flights), session.ball_flight_paths_path, BALL_FLIGHT_PATHS
        )
        ok = flights.column("ok").to_pylist()
        kinds = flights.column("start_kind").to_pylist()
        return {
            "flights": flights.num_rows,
            "ok": sum(bool(v) for v in ok),
            "from_hits": sum(k in ("hit", "machine") for k in kinds),
            "rejected_hits": len(rejected),
        }


class ShotsStage(Stage):
    name = "shots"
    title = "Shots"
    version = 1
    depends_on = ("ball_3d", "events", "camera")
    weight = 0.2

    def config(self, session, config, settings):
        from swingvision.analysis import shots as sh

        return {
            "landing_px_sigma": sh.LANDING_PX_SIGMA,
            "close_call_sigmas": sh.CLOSE_CALL_SIGMAS,
            "speed_uncertain": [sh.SPEED_UNCERTAIN, sh.SPEED_UNCERTAIN_OPEN_END],
            "spin": [sh.SPIN_SIGMAS, sh.SPIN_MIN],
        }

    def outputs(self, session):
        return [session.shots_path]

    def run(self, ctx: StageContext):
        session = ctx.session
        cal = calib.load(session.calibration_path)
        if cal is None:
            raise RuntimeError("No calibration")
        summary = read_json(session.ball_flights_summary_path)
        shots = assemble_shots(
            ctx.config.id,
            tables.read_table(session.events_path),
            tables.read_table(session.ball_flights_path),
            _camera_at(cal),
            rejected_hits=set(summary.get("rejected_hits", [])),
        )
        tables.write_table(shots, session.shots_path, SHOTS)
        speeds = [v for v in shots.column("speed_racket_kmh").to_pylist() if v is not None]
        outcomes = shots.column("outcome").to_pylist()
        return {
            "shots": shots.num_rows,
            "with_speed": len(speeds),
            "in": outcomes.count("in"),
            "out": sum(o.startswith("out") for o in outcomes),
            "net": outcomes.count("net"),
        }
