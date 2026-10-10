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
practice_app = typer.Typer(
    no_args_is_help=True, help="Practice sessions: shots, targets, accuracy."
)
app.add_typer(practice_app, name="practice")
swings_app = typer.Typer(no_args_is_help=True, help="Swings: strokes, phases, kinematics.")
app.add_typer(swings_app, name="swings")
serves_app = typer.Typer(no_args_is_help=True, help="Serves: contact point vs the front toe (M7b).")
app.add_typer(serves_app, name="serves")
speed_app = typer.Typer(
    no_args_is_help=True, help="Speed calibration from serves that hit the net tape (M7c)."
)
app.add_typer(speed_app, name="speed")


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

    lib = services.open_library(load_settings())
    tags = lib.tags_by_session()
    for s in lib.list_sessions():
        when = (s.get("recorded_on") or "")[:10] or "-"
        tag = f"  [{', '.join(tags[s['id']])}]" if s["id"] in tags else ""
        typer.echo(f"{s['id']}  {when}  {s['status']:<12} {s['mode']:<9} {s['name']}{tag}")


@app.command()
def tags(
    session_id: str,
    tag: Annotated[list[str] | None, typer.Argument(help="The session's new tags")] = None,
    clear: Annotated[bool, typer.Option(help="Remove all tags")] = False,
) -> None:
    """Show or set a session's tags (they select sessions for multi-session stats)."""
    from swingvision import services

    settings = load_settings()
    if tag or clear:
        try:
            services.set_session_tags(settings, session_id, [] if clear else list(tag or []))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    current = services.open_library(settings).session_tags(session_id)
    typer.echo(", ".join(current) if current else "(no tags)")


@app.command()
def stats(
    session: Annotated[
        list[str] | None, typer.Option("--session", "-s", help="Only these sessions")
    ] = None,
    date_from: Annotated[
        str | None, typer.Option("--from", help="Recorded on or after (YYYY-MM-DD)")
    ] = None,
    date_to: Annotated[str | None, typer.Option("--to", help="Recorded on or before")] = None,
    mode: Annotated[list[str] | None, typer.Option(help="practice | match")] = None,
    practice_type: Annotated[
        list[str] | None, typer.Option("--type", help="self_feed | ball_machine | serve")
    ] = None,
    tag: Annotated[list[str] | None, typer.Option(help="Sessions with all of these tags")] = None,
    profile: Annotated[list[str] | None, typer.Option(help="Player profile ids")] = None,
    exclude: Annotated[list[str] | None, typer.Option(help="Leave these sessions out")] = None,
    user_calibration: Annotated[
        bool, typer.Option(help="Only court calibrations confirmed by hand")
    ] = False,
    view: Annotated[str | None, typer.Option(help="A view saved on the Stats page")] = None,
    stroke: Annotated[list[str] | None, typer.Option(help="Only these strokes")] = None,
    trend: Annotated[
        str | None, typer.Option(help="Also print a KPI per session (e.g. in_pct)")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Also export the shots shown (.csv or .parquet)")
    ] = None,
) -> None:
    """Statistics over a selection of sessions: the Stats page's numbers (M7a)."""
    from swingvision import services
    from swingvision.analysis import aggregate as agg
    from swingvision.analysis import export as ex
    from swingvision.analysis import stats as st

    lib = services.open_library(load_settings())
    if view:
        found = [v for v in lib.list_views() if view.lower() in (v["id"], v["name"].lower())]
        if not found:
            raise typer.BadParameter(f"No saved view {view!r}")
        flt = agg.SessionFilter.from_query(found[0]["query"])
    else:
        flt = agg.SessionFilter.make(
            date_from=date_from, date_to=date_to, modes=mode, practice_types=practice_type,
            profiles=profile, tags=tag, include=session, exclude=exclude,
            user_calibration=user_calibration,
        )  # fmt: skip
    sel = agg.select(lib, flt)
    data = sel.data
    rows = st.filter_records(data.records, stroke or None, None)

    def num(v, fmt: str, unit: str = "") -> str:
        return "-" if v is None else format(v, fmt) + unit

    def pct(v) -> str:
        return "-" if v is None else f"{v:.0%}"

    for s in data.sessions:
        typer.echo(f"{s['session_id']}  {(s['recorded_on'] or '-')[:10]}  {s['name']}")
    for note in data.notes:
        typer.echo(f"Note: {note}")
    c = sel.counts()
    typer.echo(f"\n{c['sessions']} sessions, {c['shots']} shots, {c['serves']} serves")
    s = st.summarize_shots(rows)
    typer.echo(
        f"Shots {s['n']} ({s['n_seen']} with the contact seen); in {pct(s['in_pct'])} of "
        f"{s['n_called']} called, net {pct(s['net_pct'])}; speed median "
        f"{num(s['speed_median'], '.0f', ' km/h')}, fastest {num(s['speed_max'], '.0f', ' km/h')} "
        f"({s['n_speed']} speeds, uncalibrated); on target {pct(s['target_pct'])} of "
        f"{s['n_targeted']}"
    )
    swings = st.summarize_swings(data.swings)
    typer.echo(
        f"\n{'stroke':<16}{'shots':>6}{'in':>6}{'net':>6}{'km/h':>7}{'depth m':>9}{'swings':>8}"
    )
    for g, gs in st.by_group(rows).items():
        typer.echo(
            f"{st.GROUP_LABELS[g]:<16}{gs['n']:>6}{pct(gs['in_pct']):>6}{pct(gs['net_pct']):>6}"
            f"{num(gs['speed_median'], '.0f'):>7}{num(gs['depth_mean'], '.2f'):>9}"
            f"{swings.get(g, {}).get('n', '-'):>8}"
        )
    m = st.summarize_movement(data)
    if m:
        typer.echo(
            f"\nMoved {m['distance_m']:,.0f} m in {m['tracked_s'] / 60:.0f} min tracked "
            f"({m['coverage']:.0%} coverage); top speed {m['max_speed_mps']:.1f} m/s"
        )
    if trend:
        if trend not in st.TREND_KPIS:
            raise typer.BadParameter(f"Choose a trend from {', '.join(st.TREND_KPIS)}")
        typer.echo(f"\n{st.TREND_KPIS[trend][0]} per session:")
        sw = [x for x in data.swings if not stroke or x["stroke_type"] in stroke]
        for p in st.trend(data, trend, rows, sw):
            ci = "" if p["lo"] is None else f"  [{p['lo']:.3g}, {p['hi']:.3g}]"
            typer.echo(f"  {p['label']:<40} n={p['n']:<5} {num(p['value'], '.3g')}{ci}")
    if out is not None:
        fmt = "parquet" if out.suffix.lower() == ".parquet" else "csv"
        out.write_bytes(ex.selection_bytes(data, "shots", fmt, rows))
        typer.echo(f"\nWrote {out}")


@app.command()
def relink(session_id: str, path: Path) -> None:
    """Point a session at its source video's new location (same file, moved or renamed).

    Other sessions whose videos are missing are looked for in the same folder.
    """
    from swingvision import services

    settings = load_settings()
    try:
        relinked = services.relink_session(settings, session_id, path)
    except services.RelinkError as exc:
        raise typer.BadParameter(str(exc)) from exc
    lib = services.open_library(settings)
    for sid in relinked:
        row = lib.get_session(sid)
        typer.echo(f"Relinked {sid} ({row['name']}) -> {row['source_path']}")


@app.command()
def export(
    session_id: str,
    what: Annotated[str, typer.Option(help="shots | practice | swings")] = "shots",
    fmt: Annotated[str, typer.Option("--format", help="csv | parquet")] = "csv",
    out: Annotated[Path | None, typer.Option(help="Output file (default: in this folder)")] = None,
) -> None:
    """Export a session's shots (with practice results), practice shots or swings."""
    from swingvision import services
    from swingvision.analysis import export as ex

    session = services.session_by_id(load_settings(), session_id)
    if session is None:
        raise typer.BadParameter(f"Unknown session {session_id}")
    try:
        data = ex.export_bytes(session, what, fmt)
    except ex.ExportError as exc:
        raise typer.BadParameter(str(exc)) from exc
    out = out or Path(ex.filename(session, what, fmt))
    out.write_bytes(data)
    typer.echo(f"Wrote {out} ({len(data) / 1024:.0f} KB)")


@app.command()
def shots(
    session_id: str,
    all_hits: Annotated[
        bool, typer.Option("--all", help="Also hits that stayed on the hitter's side")
    ] = False,
) -> None:
    """List a session's shots: speed, net clearance, landing and line call (M4)."""
    from swingvision import services
    from swingvision.analysis.shots import speed_error_kmh, speed_error_text
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
        err = speed_error_kmh(r["speed_racket_kmh"], r["speed_sigma_kmh"])
        if err is not None:
            speed += f"±{err:.0f}"
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
    typer.echo(speed_error_text())


def _practice_session(session_id: str):
    from swingvision import services

    settings = load_settings()
    session = services.session_by_id(settings, session_id)
    if session is None:
        raise typer.BadParameter(f"Unknown session {session_id}")
    return settings, session


@practice_app.command("show")
def practice_show(
    session_id: str,
    refresh: Annotated[
        bool, typer.Option(help="Rerun segmentation and accuracy first if they're stale")
    ] = True,
) -> None:
    """A practice session's blocks and shots: kind, landing, call, targets (M5)."""
    from swingvision import services
    from swingvision.analysis import practice as pr
    from swingvision.analysis.segmentation import blocks_of
    from swingvision.storage import tables

    settings, session = _practice_session(session_id)
    if refresh:
        status, job = services.refresh_practice(settings, session_id)
        if status != "ran":
            typer.echo(f"Processing needed first ({status}, job {job}).")
    if not session.practice_path.exists():
        raise typer.BadParameter("No practice results yet: process the session first")
    rows = tables.read_table(session.practice_path).to_pylist()
    blocks = blocks_of(tables.read_table(session.segments_path))

    def num(v, fmt: str) -> str:
        return "-" if v is None else format(v, fmt)

    def pct(v) -> str:
        return "-" if v is None else f"{v:.0%}"

    for b in blocks:
        mine = [r for r in rows if r["block_id"] == b["block_id"]]
        s = pr.summarize(mine)
        end = {-1: "near", 1: "far"}.get(b["side"], "?")
        typer.echo(
            f"\nBlock {b['block_id'] + 1}: {b['start_t']:.0f}-{b['end_t']:.0f} s, "
            f"{b['n_shots']} {b['shot_kind']} shots from the {end} end; in {pct(s['in_pct'])}, "
            f"net {pct(s['net_pct'])}, target {pct(s['target_pct'])}"
        )
        for r in mine:
            land = "-"
            if r["landing_x"] is not None:
                land = f"({r['landing_x']:.2f}, {r['landing_y']:.2f})"
            tgt = "" if r["in_target"] is None else ("  HIT " if r["in_target"] else "  miss")
            if r["target_dist_m"] is not None:
                tgt += f" {r['target_dist_m']:.2f} m"
            kind = r["shot_kind"] + (f"/{r['serve_side']}" if r["serve_side"] else "")
            typer.echo(
                f"  {r['t_contact']:8.2f} {kind:<13} {land:>17} {r['outcome']:<9}"
                f"{num(r['speed_kmh'], '.0f'):>5} km/h{tgt}{'  (excluded)' if r['excluded'] else ''}"
            )
    s = pr.summarize(rows)
    typer.echo(
        f"\n{s['n']} practice shots in {len(blocks)} blocks; {s['n_called']} called: "
        f"in {pct(s['in_pct'])}, net {pct(s['net_pct'])}; target hits {pct(s['target_pct'])} "
        f"of {s['n_targeted']}; median distance to target {num(s['dist_median'], '.2f')} m"
    )


@practice_app.command("eval")
def practice_eval(
    session_ids: Annotated[
        list[str] | None, typer.Argument(help="Sessions (default: every labeled one)")
    ] = None,
) -> None:
    """Score practice segmentation against shots labeled by eye (M5 exit criterion: 95%)."""
    from swingvision import services
    from swingvision.analysis.segmentation import practice_shots
    from swingvision.storage import tables
    from swingvision.training.practice_labels import (
        SEGMENT_LABELS_DIR,
        load_labels,
        score_segments,
    )

    settings = load_settings()
    root = settings.require_output_root()
    if not session_ids:
        d = root.joinpath(*SEGMENT_LABELS_DIR)
        session_ids = sorted(p.stem for p in d.glob("*.json")) if d.exists() else []
    if not session_ids:
        raise typer.BadParameter("No labeled sessions (training/segments/<id>.json)")
    total = [0, 0, 0]  # correct, labeled, spurious
    for sid in session_ids:
        labels = load_labels(root, sid)
        session = services.session_by_id(settings, sid)
        if labels is None or session is None:
            typer.echo(f"{sid}: no labels or no session")
            continue
        services.refresh_practice(settings, sid)
        sc = score_segments(practice_shots(tables.read_table(session.segments_path)), labels)
        total[0] += sc.covered
        total[1] += sc.n_labeled
        total[2] += len(sc.false)
        typer.echo(
            f"{sid}: {sc.covered}/{sc.n_labeled} shots segmented correctly, "
            f"{len(sc.false)} spurious -> {sc.accuracy:.1%} "
            f"(recall {sc.recall:.1%}, precision {sc.precision:.1%}, net/landing agree "
            f"{sc.end_agree}/{sc.matched})"
        )
        for name, ts in (("missed", sc.missed), ("spurious", sc.false), ("cut", sc.uncovered)):
            if ts:
                typer.echo(f"  {name}: " + ", ".join(f"{t:.1f}" for t in ts))
    acc = total[0] / (total[1] + total[2]) if total[1] + total[2] else 1.0
    typer.echo(f"{'PASS' if acc >= 0.95 else 'FAIL'}  all sessions: {acc:.1%} (>= 95%)")


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
            f"{spec.name:<20} {mark}  {spec.size_mb:5.1f} MB  {spec.license}  {spec.description}"
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


@train_app.command("strokes")
def train_strokes_cmd(
    name: Annotated[str, typer.Argument(help="Name of the new stroke model")],
    epochs: Annotated[int, typer.Option(help="Training epochs")] = 60,
) -> None:
    """Train the stroke classifier on labeled swings (training/strokes + Swings-page edits)."""
    from swingvision import services
    from swingvision.storage import edits as ed
    from swingvision.training.stroke_labels import load_labels
    from swingvision.training.train_strokes import Examples, collect, train_and_save

    settings = load_settings()
    root = settings.require_output_root()
    ex = Examples()
    for row in services.open_library(settings).list_sessions():
        session = services.session_by_id(settings, row["id"])
        if session is None or not session.swings_path.exists():
            continue
        labels = load_labels(root, row["id"])
        edits = [(e.t, e.stroke) for e in ed.load(session).swings]
        if labels is None and not edits:
            continue
        n = collect(session, labels, edits, ex)
        typer.echo(f"{row['id']}: {n} labeled swings")
    if not len(ex):
        raise typer.BadParameter("No labeled swings (training/strokes/<id>.json or edits)")
    path, card = train_and_save(root, name, ex, epochs)
    cv = card["cross_validation"]
    for f in cv.get("folds", []):
        typer.echo(
            f"  held out {f['session']}: model {f['model_acc']:.1%}, rules {f['rules_acc']:.1%} "
            f"({f['n']} swings)"
        )
    typer.echo(f"Saved {path}: {card['counts']}")
    typer.echo(
        "Validated: the swings stage uses it (stroke model 'auto')."
        if card["validated"]
        else "Not validated (too few labels or classes, or it doesn't beat the rules on "
        "held-out sessions): the rules stay in use."
    )


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


@swings_app.command("show")
def swings_show(
    session_id: str,
    all_swings: Annotated[bool, typer.Option("--all", help="Also non-strokes (other)")] = False,
) -> None:
    """A session's swings: contact, stroke, phases and the main kinematics (M6)."""
    from swingvision import services
    from swingvision.pose.strokes import STROKE_LABELS
    from swingvision.storage import tables
    from swingvision.storage.fsutil import read_json

    settings = load_settings()
    session = services.session_by_id(settings, session_id)
    if session is None or not session.swings_path.exists():
        raise typer.BadParameter("No swings yet: process the session first")
    summary = read_json(session.swings_summary_path)
    typer.echo(
        f"{summary['swings']} swings, racket hand {summary['racket_hand']} "
        f"({summary['hand']['source']}), strokes {summary['strokes']}"
    )

    def num(v, fmt: str) -> str:
        return "-" if v is None else format(v, fmt)

    for r in tables.read_table(session.swings_path).to_pylist():
        if r["stroke_type"] in (None, "other") and not all_swings:
            continue
        end = {-1: "near", 1: "far"}.get(r["side"], "?")
        stroke = STROKE_LABELS.get(r["stroke_type"] or "", r["stroke_type"] or "-")
        typer.echo(
            f"#{r['swing_id']:<4} {r['t_contact']:8.2f} s  {end:<4} {stroke:<16} "
            f"{num(r['stroke_conf'], '.2f'):>4} ({r['stroke_source'] or '-'}, contact "
            f"{r['contact_source']})  prep {num(r['prep_s'], '.2f')} fwd "
            f"{num(r['forward_s'], '.2f')} follow {num(r['follow_s'], '.2f')} s  wrist "
            f"{num(r['wrist_speed_peak'], '.1f')} m/s  contact {num(r['contact_height_m'], '.2f')} m"
            f"  {' '.join(r['flags'] or [])}"
        )


@swings_app.command("eval")
def swings_eval(
    session_ids: Annotated[
        list[str] | None, typer.Argument(help="Sessions (default: every labeled one)")
    ] = None,
    include_dark: Annotated[
        bool, typer.Option(help="Also score swings flagged dark (dusk)")
    ] = False,
) -> None:
    """Score strokes, contact timing and phase consistency against labeled strokes
    (training/strokes/<id>.json; M6 exit criteria)."""
    import numpy as np

    from swingvision import services
    from swingvision.storage import tables
    from swingvision.training.stroke_labels import STROKE_DIR, SideReport, evaluate, load_labels

    settings = load_settings()
    root = settings.require_output_root()
    if not session_ids:
        d = root / "training" / STROKE_DIR
        session_ids = sorted(p.stem for p in d.glob("*.json")) if d.exists() else []
    if not session_ids:
        raise typer.BadParameter("No labeled sessions (training/strokes/<id>.json)")
    total = {"near": SideReport(), "far": SideReport()}
    exclude = frozenset() if include_dark else frozenset({"dark"})
    for sid in session_ids:
        labels = load_labels(root, sid)
        session = services.session_by_id(settings, sid)
        if labels is None or session is None or not session.swings_path.exists():
            typer.echo(f"{sid}: no labels, session or swings")
            continue
        config = session.load_config()
        events = tables.read_table(session.events_path).to_pylist()
        rep = evaluate(
            tables.read_table(session.swings_path),
            labels,
            config.video.fps_avg if config.video else 60.0,
            hit_times={e["event_id"]: e["t_s"] for e in events},
            exclude_flags=exclude,
        )
        for side in ("near", "far"):
            r, t = rep[side], total[side]
            t.strokes += r.strokes
            t.strokes_correct += r.strokes_correct
            t.others += r.others
            t.others_correct += r.others_correct
            t.contact_errors += r.contact_errors
            t.contact_errors_audio += r.contact_errors_audio
            for truth, row in r.confusion.items():
                for pred, n in row.items():
                    t.confusion.setdefault(truth, {})
                    t.confusion[truth][pred] = t.confusion[truth].get(pred, 0) + n
            if r.total:
                typer.echo(
                    f"{sid} {side}: strokes {r.strokes_correct}/{r.strokes}, other "
                    f"{r.others_correct}/{r.others} -> {r.accuracy:.1%}"
                )
        for key, g in rep["phases"].items():
            parts = [
                f"{ph} {g[ph]['median_s']:.2f} s (CV {g[ph]['cv']:.0%}, n={g[ph]['n']})"
                for ph in ("prep_s", "forward_s", "follow_s")
                if ph in g and g[ph]["cv"] is not None
            ]
            order = "-" if g["in_order"] is None else f"{g['in_order']:.0%}"
            typer.echo(f"  phases {key} (n={g['n']}, in order {order}): " + "; ".join(parts))
    targets = {"near": 0.90, "far": 0.80}
    for side, r in total.items():
        if not r.total:
            continue
        ok = r.accuracy >= targets[side]
        typer.echo(
            f"{'PASS' if ok else 'FAIL'}  {side}: {r.accuracy:.1%} (>= {targets[side]:.0%}) "
            f"over {r.strokes} strokes and {r.others} other swings"
        )
        for truth, row in sorted(r.confusion.items()):
            typer.echo(
                f"    {truth:<8} -> " + ", ".join(f"{k} {v}" for k, v in sorted(row.items()))
            )
        e = np.abs(np.array(r.contact_errors))
        if len(e):
            within = float(np.mean(e <= 2))
            typer.echo(
                f"{'PASS' if within >= 0.9 else 'FAIL'}  {side} contact from the pose: "
                f"{within:.0%} within ±2 frames of the hit (median {np.median(e):.0f}, "
                f"n={len(e)})"
            )
        a = np.abs(np.array(r.contact_errors_audio))
        if len(a):
            typer.echo(
                f"      {side} contact from the pose vs the impact sound (unseen contacts): "
                f"{np.mean(a <= 2):.0%} within ±2 frames (median {np.median(a):.0f}, n={len(a)})"
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


@serves_app.command("contact")
def serves_contact(
    session_id: str,
    all_serves: Annotated[bool, typer.Option("--all", help="Also far and flagged serves")] = False,
) -> None:
    """A session's serve contact points: contact frame, toe, offsets with σ, flags."""
    from swingvision import services
    from swingvision.analysis import serve_stats as ss
    from swingvision.storage import tables

    settings = load_settings()
    session = services.session_by_id(settings, session_id)
    if session is None or not session.serve_contact_path.exists():
        raise typer.BadParameter("No serve contacts yet: process the session first")

    def cm(v, s=None) -> str:
        if v is None:
            return "    -   "
        return f"{100 * v:+5.0f}" + (f"±{100 * s:<2.0f}" if s is not None else "   ")

    rows = tables.read_table(session.serve_contact_path).to_pylist()
    shown = 0
    for r in rows:
        flags = set(r["flags"] or [])
        if not all_serves and (r["side"] != -1 or ss.SERIOUS_FLAGS & flags):
            continue
        shown += 1
        end = {-1: "near", 1: "far"}.get(r["side"], "?")
        typer.echo(
            f"#{r['swing_id']:<4} {r['t_contact']:8.2f} s  frame {r['frame_contact'] or '-':>6} "
            f"({r['contact_source'] or '-':<9}) {end:<4} {r['serve_side'] or '-':<5} "
            f"fwd {cm(r['forward_m'], r['forward_sigma_m'])} lat {cm(r['lateral_m'], r['lateral_sigma_m'])} "
            f"cm  height {'-' if r['height_m'] is None else format(r['height_m'], '.2f')} m  "
            f"toe {r['toe_source'] or '-'}{'' if r['toe_on_ground'] is not False else ' (lifted)'}  "
            f"{' '.join(sorted(flags))}"
        )
    typer.echo(f"{shown} of {len(rows)} serves shown")


def _refresh_refs(settings, session_id: str) -> None:
    from swingvision import services

    status, job = services.refresh_practice(settings, session_id, target="speed_refs")
    if status != "ran":
        raise typer.BadParameter(
            f"The session needs processing first ({status}{f', job #{job}' if job else ''})"
        )


@speed_app.command("refs")
def speed_refs(
    session_id: str,
    accept: Annotated[list[int] | None, typer.Option(help="Accept these references (ids)")] = None,
    reject: Annotated[list[int] | None, typer.Option(help="Reject these references")] = None,
    reset: Annotated[list[int] | None, typer.Option(help="Forget these references' review")] = None,
    mark: Annotated[
        list[float] | None, typer.Option(help="Mark the serve at this time (s) as a tape hit")
    ] = None,
    temp: Annotated[
        float | None, typer.Option(help="Air temperature during the recording, °C")
    ] = None,
) -> None:
    """A session's net-tape reference serves: Δt from the sounds, reference vs fitted speed,
    review status. Reviews (--accept / --reject / --mark) are saved like the app's."""
    from swingvision import services
    from swingvision.storage import tables
    from swingvision.storage.fsutil import read_json

    settings = load_settings()
    session = services.session_by_id(settings, session_id)
    if session is None:
        raise typer.BadParameter(f"Unknown session {session_id}")
    if temp is not None:
        services.set_air_temperature(settings, session_id, temp)
    _refresh_refs(settings, session_id)
    rows = tables.read_table(session.speed_refs_path).to_pylist()
    by_id = {r["ref_id"]: r for r in rows}
    changed = False
    for ids, status in ((accept, "accepted"), (reject, "rejected"), (reset, None)):
        for i in ids or []:
            if i not in by_id:
                raise typer.BadParameter(f"No reference {i}")
            services.edit_speed_ref(settings, session_id, by_id[i]["t_contact"], status=status,
                                    **({"marked": False, "t_racket": None, "t_tape": None}
                                       if status is None else {}))  # fmt: skip
            changed = True
    for t in mark or []:
        services.edit_speed_ref(settings, session_id, t, marked=True)
        changed = True
    if changed:
        _refresh_refs(settings, session_id)
        rows = tables.read_table(session.speed_refs_path).to_pylist()
    summary = read_json(session.speed_refs_summary_path)

    def f(v, fmt):
        return "-" if v is None else format(v, fmt)

    for r in rows:
        typer.echo(
            f"#{r['ref_id']:<3} {r['t_contact']:8.2f} s  shot {f(r['shot_id'], '')}  "
            f"{r['end_kind'] or '-':<6} clear {f(r['clearance_m'], '+.3f')} m  "
            f"SNR {f(r['snr_racket_db'], '.0f')}/{f(r['snr_tape_db'], '.0f')} dB  "
            f"Δt {f(r['dt_s'] and 1000 * r['dt_s'], '.1f')} ms  "
            f"ref {f(r['v_ref_kmh'], '.1f')} fit {f(r['v_fit_kmh'], '.1f')} km/h  "
            f"ratio {f(r['ratio'], '.4f')} ± {f(r['ratio_sigma'], '.4f')}  "
            f"{r['status']:<9} {r['source']:<4} {' '.join(r['flags'] or [])}"
        )
    typer.echo(
        f"{summary['references']} references ({summary['accepted']} accepted, "
        f"{summary['rejected']} rejected) from {summary['near_serves']} near-end serves; "
        f"audio/video offset {1000 * summary['av_offset_s']:.1f} ms "
        f"({summary['av_offset_serves']} serves), sound {summary['sound_speed_mps']:.1f} m/s"
        + ("" if summary["air_temp_c"] is not None else " (20 °C assumed: set --temp)")
    )


def _device_arg(settings, device: str | None, session_id: str | None) -> str:
    from swingvision import services

    if session_id:
        row = services.open_library(settings).get_session(session_id)
        if row is None:
            raise typer.BadParameter(f"Unknown session {session_id}")
        device = row.get("device_key")
        if not device:
            raise typer.BadParameter("The session's recording device isn't known")
    if not device:
        raise typer.BadParameter("Give a device key (sv speed show) or --session")
    return device


@speed_app.command("calibrate")
def speed_calibrate(
    device: Annotated[str | None, typer.Argument(help="Device key (sv speed show)")] = None,
    session_id: Annotated[
        str | None, typer.Option("--session", help="The device this session was recorded with")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Fit and show, don't store")] = False,
) -> None:
    """Fit a device's speed calibration from its accepted references and make it the active
    one; its sessions' shots are brought up to date."""
    from swingvision import services

    settings = load_settings()
    device = _device_arg(settings, device, session_id)
    cal, row, refs = services.calibrate_device(settings, device, save=not dry_run)
    if cal is None:
        raise typer.BadParameter(
            f"{len(refs)} accepted references for {device}: at least 3 are needed"
        )
    _print_calibration(cal.as_dict(), refs)
    if row is None:
        typer.echo("(not stored: --dry-run)")
        return
    typer.echo(f"Stored as version {row['version']} ({row['id']}), active.")
    done = services.refresh_device(settings, device)
    typer.echo(
        f"Sessions updated: {len(done['ran'])}; queued: {len(done['queued'])}; "
        f"busy: {len(done['busy'])}"
    )


def _print_calibration(c: dict, refs: list[dict]) -> None:
    d = c.get("diagnostics") or {}
    rel = c["k_sigma"] / c["k"]
    typer.echo(
        f"model {c['model']}: k = {c['k']:.4f} ± {c['k_sigma']:.4f} ({rel:.2%})"
        + (
            f", readout τ = {1000 * c['tau_s']:.1f} ± {1000 * (c.get('tau_sigma_s') or 0):.1f} ms"
            if c.get("tau_s") is not None
            else ""
        )
        + f"  from {c['n_refs']} references"
    )
    if c.get("loo_sd") is not None:
        typer.echo(f"  leave-one-out spread {c['loo_sd']:.2%} (1 SD); χ²/dof {d.get('chi2_dof')}")
    for name in ("speed", "rolling_shutter"):
        t = d.get(f"trend_{name}")
        if t:
            typer.echo(
                f"  trend vs {name.replace('_', ' ')}: slope {t['slope']:.3g} ± "
                f"{t['slope_sigma']:.2g}, p = {t['p']:.2f}"
                + (" (significant)" if t["significant"] else "")
            )
    bs = d.get("by_session")
    if bs:
        per = ", ".join(f"{k} {v['k']:.4f} (n {v['n']})" for k, v in bs["sessions"].items())
        typer.echo(f"  by session: {per}; p = {bs['p']:.2f}")
    sessions = sorted({r.get("session_id") for r in refs})
    typer.echo(f"  sessions: {', '.join(s for s in sessions if s)}")


@speed_app.command("show")
def speed_show(
    device: Annotated[str | None, typer.Argument(help="Device key (default: list devices)")] = None,
) -> None:
    """Recording devices and their speed calibrations."""
    from swingvision import services

    settings = load_settings()
    services.backfill_devices(settings)
    lib = services.open_library(settings)
    if device is None:
        for d in lib.list_devices():
            cal = lib.get_calibration(d["calibration_id"]) if d["calibration_id"] else None
            state = (
                f"× {cal['k']:.4f} ± {cal['k_sigma'] / cal['k']:.2%} (v{cal['version']}, "
                f"{cal['n_refs']} refs)"
                if cal
                else "uncalibrated"
            )
            typer.echo(
                f"{d['device_key']:<44} {services.device_name(d):<44} "
                f"{d['n_sessions']:>3} sessions  {state}"
            )
        return
    d = lib.get_device(device)
    if d is None:
        raise typer.BadParameter(f"Unknown device {device}")
    typer.echo(f"{device}: {services.device_name(d)}")
    for row in lib.sessions_of_device(device):
        typer.echo(f"  session {row['id']}  {(row.get('recorded_on') or '')[:10]}  {row['name']}")
    refs = services.device_references(settings, device)
    typer.echo(f"  {len(refs)} accepted references")
    for c in lib.list_calibrations(device):
        typer.echo(f"version {c['version']} ({c['id']}){' active' if c['active'] else ''}:")
        _print_calibration(c, c["refs"])


@serves_app.command("eval")
def serves_eval(
    session_ids: Annotated[
        list[str] | None, typer.Argument(help="Sessions (default: every labeled one)")
    ] = None,
) -> None:
    """Score contact frames, toes and contact points against labeled serves
    (training/serve_contact/<id>.json plus the Swings page's corrections; M7b exit criteria)."""
    import json

    from swingvision import services
    from swingvision.court import calibration as calib
    from swingvision.storage import edits as ed
    from swingvision.storage import tables
    from swingvision.training import serve_labels as sl

    settings = load_settings()
    root = settings.require_output_root()
    if not session_ids:
        d = root / "training" / sl.SERVE_DIR
        session_ids = sorted(p.stem for p in d.glob("*.json")) if d.exists() else []
    if not session_ids:
        raise typer.BadParameter("No labeled sessions (training/serve_contact/<id>.json)")
    total = sl.ServeReport()
    for sid in session_ids:
        session = services.session_by_id(settings, sid)
        if session is None or not session.serve_contact_path.exists():
            typer.echo(f"{sid}: no session or serve contacts")
            continue
        labels = sl.with_edits(sl.load_labels(root, sid), ed.load(session), sid)
        cal = calib.load(session.calibration_path)
        config = session.load_config()
        rep = sl.evaluate(
            tables.read_table(session.serve_contact_path),
            labels,
            lambda t, cal=cal: calib.camera_at(cal, t),
            config.video.fps_avg if config.video else 60.0,
        )
        typer.echo(f"{sid}: {json.dumps(rep.summary())}")
        total.merge(rep)
    s = total.summary()

    def ok(v, test) -> str:
        return "----" if v is None else ("PASS" if test(v) else "FAIL")

    def pct(v) -> str:
        return "-" if v is None else f"{v:.1%}"

    def cmv(v) -> str:
        return "-" if v is None else f"{100 * v:.1f} cm"

    checks = [
        ("contact frame exact (>= 80%)", s["frame_exact"], lambda v: v >= 0.8, pct(s["frame_exact"])),
        ("contact frame within ±1 (>= 95%)", s["frame_within_1"], lambda v: v >= 0.95,
         pct(s["frame_within_1"])),
        ("toe median (<= 3 cm)", s["toe_median_m"], lambda v: v <= 0.03, cmv(s["toe_median_m"])),
        ("toe 90% (<= 6 cm)", s["toe_p90_m"], lambda v: v <= 0.06, cmv(s["toe_p90_m"])),
        ("on-ground test agrees (>= 95%)", s["on_ground_agree"], lambda v: v >= 0.95,
         pct(s["on_ground_agree"])),
        ("contact point reprojected (median <= 4 px)", s["ball_median_px"], lambda v: v <= 4,
         "-" if s["ball_median_px"] is None else f"{s['ball_median_px']:.1f} px"),
        ("forward σ (median <= 8 cm)", s["forward_sigma_median_m"], lambda v: v <= 0.08,
         cmv(s["forward_sigma_median_m"])),
        ("coverage, near serves with the contact in frame (>= 85%)",
         s["coverage_contact_in_frame"], lambda v: v >= 0.85, pct(s["coverage_contact_in_frame"])),
    ]  # fmt: skip
    for label, v, test, text in checks:
        typer.echo(f"{ok(v, test)}  {label}: {text}")
    typer.echo(
        f"n: {s['contact_frames']} contact frames, {s['toes']} toes, {s['balls']} balls; "
        f"{s['near']} near serves, {s['above_frame']} with the contact above the picture"
    )


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
