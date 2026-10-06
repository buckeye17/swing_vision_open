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
models_app = typer.Typer(no_args_is_help=True, help="Pretrained model weights.")
app.add_typer(models_app, name="models")
profiles_app = typer.Typer(no_args_is_help=True, help="Player profiles.")
app.add_typer(profiles_app, name="profiles")
labels_app = typer.Typer(no_args_is_help=True, help="Ball and event labels.")
app.add_typer(labels_app, name="labels")
bench_app = typer.Typer(no_args_is_help=True, help="Benchmarks.")
app.add_typer(bench_app, name="bench")
train_app = typer.Typer(no_args_is_help=True, help="Train models on your labels.")
app.add_typer(train_app, name="train")


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
    profile: Annotated[
        str | None, typer.Option(help="Player profile id (sv profiles list)")
    ] = None,
    process: Annotated[bool, typer.Option(help="Enqueue processing right away")] = True,
) -> None:
    """Create a session from a video (and enqueue processing)."""
    from swingvision import services

    settings = load_settings()
    if profile and services.get_profile(settings, profile) is None:
        raise typer.BadParameter(f"Unknown profile {profile}")
    session = services.create_session(
        settings,
        video,
        name,
        mode,  # type: ignore[arg-type]
        submode,  # type: ignore[arg-type]
        me_profile_id=profile,
    )
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


@app.command()
def shots(
    session_id: str,
    all_hits: Annotated[
        bool, typer.Option("--all", help="Also hits that stayed on the hitter's side")
    ] = False,
) -> None:
    """List a session's shots: speed, net clearance, landing and line call (M4)."""
    from swingvision import services
    from swingvision.app.components.shots_view import over_net, shot_summary
    from swingvision.storage import tables

    session = services.session_by_id(load_settings(), session_id)
    if session is None:
        raise typer.BadParameter(f"Unknown session {session_id}")
    if not session.shots_path.exists():
        raise typer.BadParameter("No shots yet: process the session up to the 'shots' stage")
    table = tables.read_table(session.shots_path)

    def num(v, fmt: str) -> str:
        return "-" if v is None else format(v, fmt)

    typer.echo(f"{'time':>8} {'hitter':<8} {'km/h':>9} {'net m':>6} {'landing (x, y) m':>17} call")
    for r in table.to_pylist():
        if not all_hits and not over_net(r):
            continue
        speed = num(r["speed_racket_kmh"], ".0f")
        if r["speed_sigma_kmh"] is not None:
            speed += f"±{r['speed_sigma_kmh']:.0f}"
        land = "-"
        if r["landing_x"] is not None:
            land = f"({r['landing_x']:.2f}, {r['landing_y']:.2f})"
        typer.echo(
            f"{r['t_contact']:8.2f} {r['hitter'] or '-':<8} {speed:>9} "
            f"{num(r['net_clearance_m'], '+.2f'):>6} {land:>17} {r['outcome']}"
        )
    s = shot_summary(table)
    typer.echo(
        f"{s['n']} shots over the net, {s['in']}/{s['called']} in; racket speed median "
        f"{num(s['median'], '.0f')} km/h, fastest {num(s['max'], '.0f')} km/h "
        f"({s['n_speed']} speeds certain enough)"
    )


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


@models_app.command("list")
def models_list() -> None:
    """Show known weights, whether they are downloaded, and their licenses."""
    from swingvision.models.registry import REGISTRY, weights_dir

    typer.echo(f"# {weights_dir()}")
    for spec in REGISTRY.values():
        mark = "yes" if spec.available() else "no "
        typer.echo(
            f"{spec.name:<10} {mark}  {spec.size_mb:5.1f} MB  {spec.license}  {spec.description}"
        )


@models_app.command("download")
def models_download(names: Annotated[list[str] | None, typer.Argument()] = None) -> None:
    """Download weights (default: the person detector chosen in Settings)."""
    from swingvision.models import registry

    for name in names or [load_settings().processing.person_model]:
        path = registry.ensure(name, lambda f, n=name: sys.stdout.write(f"\r{n}: {f:5.1%}"))
        typer.echo(f"\r{name}: {path}")


@profiles_app.command("list")
def profiles_list() -> None:
    from swingvision import services

    for p in services.list_profiles(load_settings()):
        height = f"{p.height_m * 100:.0f} cm" if p.height_m else "-"
        typer.echo(f"{p.id}  {p.name:<20} {p.handedness:<6} {p.backhand:<11} {height}")


@profiles_app.command("add")
def profiles_add(
    name: str,
    handedness: Annotated[str, typer.Option(help="right | left")] = "right",
    backhand: Annotated[str, typer.Option(help="two_handed | one_handed")] = "two_handed",
    height_cm: Annotated[float | None, typer.Option(help="Height in cm")] = None,
) -> None:
    from pydantic import ValidationError

    from swingvision import services

    try:
        p = services.save_profile(
            load_settings(), name, handedness, backhand, height_cm / 100 if height_cm else None
        )
    except ValidationError as exc:
        problems = "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors())
        raise typer.BadParameter(problems) from None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    typer.echo(f"Created profile {p.id} ({p.name})")


# ---------------------------------------------------------------------------
# Ball: labels, benchmark, training, evaluation (M3)
# ---------------------------------------------------------------------------


def _label_store():
    from swingvision.training.labels import LabelStore

    settings = load_settings()
    return settings, LabelStore(settings.require_output_root())


def _gt_clips(store, split: str | None, done_only: bool = True):
    clips = [c for c in store.list_clips() if c.kind == "gt"]
    if done_only:
        clips = [c for c in clips if c.status == "done"]
    if split:
        clips = [c for c in clips if c.split == split]
    return clips


@labels_app.command("list")
def labels_list() -> None:
    """Labeled clips: frames labeled, visibility counts, events."""
    _, store = _label_store()
    for c in store.list_clips():
        vis = {"visible": 0, "none": 0, "occluded": 0}
        for f in c.frames:
            lab = c.label(f)
            if lab is not None:
                vis[lab.vis] += 1
        typer.echo(
            f"{c.session_id}/{c.clip_id:<8} {c.kind:<6} {c.split or '-':<5} {c.status:<11} "
            f"{c.t0_s:8.1f}s {len(c.frames):4d} fr  visible {vis['visible']:4d}  none "
            f"{vis['none']:4d}  occluded {vis['occluded']:4d}  events {len(c.events)}"
        )


@labels_app.command("pseudo")
def labels_pseudo(
    session_id: str,
    clips: Annotated[int, typer.Option(help="Number of sample clips to create")] = 20,
    detector: Annotated[str, typer.Option(help="Detector spec (default: the active one)")] = "auto",
    seed: int = 0,
) -> None:
    """Create training clips at hitting moments, labeled from confident detections."""
    from swingvision.ball.detectors import resolve_spec
    from swingvision.training.pseudo import make_pseudo_clips

    settings, store = _label_store()
    spec = resolve_spec(detector, settings.output_root)
    made = make_pseudo_clips(store, settings, session_id, clips, spec, seed=seed, log=typer.echo)
    typer.echo(f"{len(made)} clips, {sum(n for _, n in made)} pseudo-labels ({spec})")


@bench_app.command("ball")
def bench_ball(
    subjects: Annotated[
        list[str] | None,
        typer.Argument(help="detector[,sweep_hz], e.g. motion  motion,15  unet:ball-v1"),
    ] = None,
    split: Annotated[str, typer.Option(help="Clips to score: test | train | all")] = "test",
    refresh: Annotated[bool, typer.Option(help="Re-run detectors (ignore cached runs)")] = False,
) -> None:
    """Compare ball detectors and schedules on the ground-truth clips (HTML report)."""
    from swingvision.training.bench_ball import parse_subject, run_bench

    settings, store = _label_store()
    clips = _gt_clips(store, None if split == "all" else split)
    if not clips:
        raise typer.BadParameter("No finished ground-truth clips (label some on the Labeling page)")
    subs = [parse_subject(s) for s in (subjects or ["motion"])]
    res = run_bench(store, settings, subs, clips, refresh=refresh, log=typer.echo)
    typer.echo(f"Report: {res.report}")


@train_app.command("ball")
def train_ball_cmd(
    name: Annotated[str, typer.Argument(help="Name of the new weights")],
    epochs: int = 30,
    batch: int = 16,
    lr: float = 2e-3,
    val_split: Annotated[
        str | None, typer.Option(help="Validation clips' split (monitoring only)")
    ] = "test",
) -> None:
    """Train the slim U-Net ball detector on the labeled clips (split=train)."""
    from swingvision.training.train_ball import TrainConfig, ensure_cached, train

    settings, store = _label_store()
    clips = [c for c in store.list_clips() if c.kind in ("gt", "sample")]
    tr = [c for c in clips if c.split != "test"]
    va = [c for c in clips if val_split and c.split == val_split and c.status == "done"]
    ensure_cached(store, settings, tr + va, log=typer.echo)
    card = train(store, settings, TrainConfig(name, epochs, batch, lr), tr, va, log=typer.echo)
    typer.echo(f"Trained {card['name']} on {card['train_frames']} frames")


@train_app.command("events")
def train_events_cmd(
    name: Annotated[str, typer.Argument(help="Name of the new event model")],
    detector: Annotated[str, typer.Option(help="Detector whose tracks to learn from")] = "auto",
) -> None:
    """Train the event classifier on kinks of predicted tracks vs labeled events."""
    from swingvision.ball.detectors import resolve_spec
    from swingvision.training.bench_ball import Bench, Subject
    from swingvision.training.train_events import collect_kinks, fit, save_model

    settings, store = _label_store()
    spec = resolve_spec(detector, settings.output_root)
    clips = [c for c in _gt_clips(store, "train") if c.events_labeled]
    # The detector usually trained on these clips: fine for event features (not a score).
    bench = Bench(store, settings, honor_folds=False)
    feats, labels = collect_kinks(bench, Subject(spec), clips, log=typer.echo)
    if not feats:
        raise typer.BadParameter("No kinks found on the training clips")
    clf = fit(feats, labels)
    counts = {k: labels.count(k) for k in sorted(set(labels))}
    path = save_model(
        settings.require_output_root(),
        name,
        clf,
        {"detector": spec, "train_clips": len(clips), "kinks": len(feats), "classes": counts},
    )
    from swingvision.training.train_events import MIN_TRAIN_KINKS

    typer.echo(f"Saved {path} ({len(feats)} kinks: {counts})")
    if len(feats) < MIN_TRAIN_KINKS:
        typer.echo(
            f"Not activated: fewer than {MIN_TRAIN_KINKS} kinks; the rules stay in use. "
            "Label hits and bounces on more clips."
        )


@app.command("eval")
def eval_cmd(
    detector: Annotated[str, typer.Option(help="Detector spec (default: the active one)")] = "auto",
    sweep_hz: Annotated[float | None, typer.Option(help="Sweep rate (default: full rate)")] = None,
    split: str = "test",
) -> None:
    """Ball and event metrics on the held-out ground-truth clips vs the M3 exit criteria."""
    import json

    from swingvision.ball.detectors import resolve_spec
    from swingvision.ball.schedule import Schedule
    from swingvision.training.bench_ball import Bench, Subject

    settings, store = _label_store()
    spec = resolve_spec(detector, settings.output_root)
    clips = _gt_clips(store, split)
    r = Bench(store, settings).score(Subject(spec, Schedule(sweep_hz)), clips, log=typer.echo)
    b, ev = r["ball"], r["bounce"]
    checks = [
        ("Ball F1 near >= 0.85", b["near"]["f1"], b["near"]["f1"] >= 0.85),
        ("Ball F1 far >= 0.75", b["far"]["f1"], b["far"]["f1"] >= 0.75),
        ("Bounce F1 >= 0.85", ev["f1"], ev["f1"] >= 0.85),
        (
            "Bounce error near <= 0.15 m",
            ev["median_pos_err_near_m"],
            ev["median_pos_err_near_m"] is not None and ev["median_pos_err_near_m"] <= 0.15,
        ),
        (
            "Bounce error far <= 0.35 m",
            ev["median_pos_err_far_m"],
            ev["median_pos_err_far_m"] is not None and ev["median_pos_err_far_m"] <= 0.35,
        ),
    ]
    typer.echo(
        json.dumps({k: r[k] for k in ("subject", "clips", "ball", "bounce", "hit")}, indent=1)
    )
    for label, value, ok in checks:
        typer.echo(f"{'PASS' if ok else 'FAIL'}  {label}: {value}")


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
