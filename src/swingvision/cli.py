"""``sv`` command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from swingvision.settings import load_settings, save_settings, settings_path

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Swing Vision Open")
settings_app = typer.Typer(no_args_is_help=True, help="Show or change app settings.")
app.add_typer(settings_app, name="settings")
court_app = typer.Typer(no_args_is_help=True, help="Court detection and calibration tools.")
app.add_typer(court_app, name="court")


@app.command("app")
def run_app(
    port: Annotated[int | None, typer.Option(help="Port (default from settings)")] = None,
    no_worker: Annotated[bool, typer.Option("--no-worker", help="Don't start a worker")] = False,
    debug: bool = False,
) -> None:
    """Start the web app (and a background worker)."""
    from swingvision.app.main import serve

    serve(port=port, start_worker=not no_worker, debug=debug)


@app.command()
def worker(
    parent_pid: Annotated[
        int | None, typer.Option(help="Exit when idle if this pid is gone")
    ] = None,
    once: Annotated[bool, typer.Option(help="Process queued jobs, then exit")] = False,
) -> None:
    """Run the GPU worker in the foreground."""
    from swingvision.pipeline.worker import run_worker

    raise typer.Exit(run_worker(parent_pid=parent_pid, once=once))


@app.command()
def probe(video: Path) -> None:
    """Print video metadata as the pipeline sees it."""
    from swingvision.io.probe import probe_video

    info = probe_video(load_settings().ffprobe(), video)
    typer.echo(info.model_dump_json(indent=2))


@app.command()
def create(
    video: Path,
    name: Annotated[str | None, typer.Option()] = None,
    mode: Annotated[str, typer.Option(help="practice | match")] = "practice",
    submode: Annotated[str, typer.Option(help="self_feed | ball_machine | serve")] = "self_feed",
    process: Annotated[bool, typer.Option(help="Enqueue processing right away")] = True,
) -> None:
    """Create a session from a video (and enqueue processing)."""
    from swingvision import services

    settings = load_settings()
    session = services.create_session(settings, video, name, mode, submode)  # type: ignore[arg-type]
    config = session.load_config()
    typer.echo(f"Created session {config.id} at {session.path}")
    if process:
        job_id = services.enqueue(settings, config.id)
        typer.echo(f"Enqueued job #{job_id}. Run `sv worker` or `sv app` to process it.")


@app.command()
def process(
    session_id: str,
    stages: Annotated[list[str] | None, typer.Option("--stage", help="Target stage(s)")] = None,
    force: Annotated[list[str] | None, typer.Option(help="Re-run these stages")] = None,
    inline: Annotated[bool, typer.Option(help="Run here instead of via the worker")] = False,
) -> None:
    """Enqueue (or run inline) processing for a session."""
    from swingvision import services

    settings = load_settings()
    if not inline:
        job_id = services.enqueue(settings, session_id, stages, force)
        typer.echo(f"Enqueued job #{job_id}")
        return
    from swingvision.pipeline.runner import RunHooks, run
    from swingvision.pipeline.stages import default_registry

    session = services.session_by_id(settings, session_id)
    if session is None:
        raise typer.BadParameter(f"Unknown session {session_id}")

    def show(name: str, frac: float, msg: str | None) -> None:
        sys.stdout.write(f"\r{name:<14} {frac * 100:5.1f}%  {msg or '':<40}")
        sys.stdout.flush()

    result = run(
        default_registry(),
        session,
        settings,
        stages,
        force or [],
        RunHooks(
            on_stage_progress=show,
            on_stage_end=lambda n, s, m: typer.echo(f"\r{n:<14} {s:<10} {m or '':<40}"),
        ),
    )
    typer.echo(f"Result: {result.status} {result.message or ''}")
    if result.error:
        typer.echo(result.error, err=True)
        raise typer.Exit(1)


@app.command()
def jobs(limit: int = 20) -> None:
    """List recent jobs."""
    from swingvision import services

    library = services.open_library(load_settings())
    for job in library.list_jobs(limit):
        typer.echo(
            f"#{job.id:<5} {job.status:<12} {job.progress * 100:5.1f}%  "
            f"{job.session_id}  {job.current_stage or '':<14} {job.message or ''}"
        )


@app.command()
def sessions() -> None:
    """List sessions in the library."""
    from swingvision import services

    for s in services.open_library(load_settings()).list_sessions():
        typer.echo(f"{s['id']}  {s['status']:<12} {s['mode']:<9} {s['name']}")


@court_app.command("detect")
def court_detect(
    path: Annotated[Path, typer.Argument(help="A video, or an image of the court")],
    times: Annotated[
        list[float] | None, typer.Option("--time", help="Video time(s) to sample (s)")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Write an overlay JPEG here")] = None,
) -> None:
    """Detect the court and fit the camera; print the fit and camera summary."""
    import json

    import cv2
    import numpy as np

    from swingvision.court import calibration as calib
    from swingvision.court.detect import Prepared, detect_court

    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise typer.BadParameter(f"Cannot read image {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        from swingvision.io.frames import grab_frames, median_image
        from swingvision.io.probe import probe_video

        settings = load_settings()
        info = probe_video(settings.ffprobe(), path)
        ts = times or list(np.linspace(0.05, 0.3, 9) * info.duration_s)
        frames = [f for f in grab_frames(settings.ffmpeg(), path, ts, info) if f is not None]
        if not frames:
            raise typer.BadParameter("No frames could be decoded")
        rgb = median_image(frames) if len(frames) > 1 else frames[0]
    prep = Prepared.from_rgb(rgb)
    det = detect_court(prep)
    if not det.ok or det.camera is None:
        typer.echo(f"Not detected: {det.message}")
        raise typer.Exit(1)
    metrics = calib.evaluate(prep, det.camera)
    typer.echo(
        f"Line RMS {metrics.rms_line_px:.2f} px, lines found {metrics.coverage:.0%} "
        f"({metrics.n_line_samples} samples, {metrics.n_net_samples} on the net)"
    )
    typer.echo(json.dumps(det.camera.describe(), indent=2))
    if out is not None:
        from swingvision.app.components.court_overlay import polylines
        from swingvision.court import model

        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        thick = max(1, round(det.camera.width / 1500))
        for color, lines in (
            ((245, 61, 255), model.COURT_LINES),
            ((255, 230, 46), model.NET_LINES),
        ):
            for poly in polylines(det.camera, lines):
                pts = np.round(poly * 4).astype(np.int32)
                cv2.polylines(vis, [pts], False, color, thick, cv2.LINE_AA, shift=2)
        cv2.imwrite(str(out), vis)
        typer.echo(f"Overlay written to {out}")


@settings_app.command("show")
def settings_show() -> None:
    typer.echo(f"# {settings_path()}")
    typer.echo(load_settings().model_dump_json(indent=2))


@settings_app.command("set-output-root")
def settings_set_output_root(path: Path) -> None:
    settings = load_settings()
    settings.output_root = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    save_settings(settings)
    from swingvision.storage.library import Library

    Library(settings.output_root).init()
    typer.echo(f"Output root set to {settings.output_root}")


if __name__ == "__main__":
    app()
