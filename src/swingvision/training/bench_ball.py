"""Ball detector benchmark (``sv bench ball``, PLAN.md §7.4.1).

Every *subject* is a detector and a frame-rate schedule. Each one runs over the same
ground-truth clips (with some margin around them), and its raw candidates are stored as
Parquet, so linking and events can be re-scored without re-running models:

``<output_root>/training/bench/runs/<subject>/<session>_<clip>.parquet`` (+ ``_frames``,
``.json`` with timings).

Scores, split by near/far half:

* detection precision / recall / F1 of the linked track (tolerance in
  :mod:`swingvision.training.evaluate`), and the raw top candidate per frame;
* bounce and hit event F1 (±2 frames) and the bounce landing error on the ground (m);
* **cost**: detector GPU-seconds per footage hour, measured on the same runs.

A subject's detector may list several weights separated by ``|`` (cross-validation folds):
each clip is then scored with the first model that didn't train on it, so the U-Net is only
ever scored on footage it hasn't seen.

The report (``report.html``) has the comparison table and an accuracy-vs-cost Pareto chart.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from swingvision.ball import events as ev
from swingvision.ball.schedule import Schedule, Window, merge_windows, target_frames
from swingvision.ball.trajectory import PlayerBoxes, link
from swingvision.court import calibration as calib
from swingvision.storage import tables
from swingvision.training.evaluate import EventScore, match_events, score_ball
from swingvision.training.labels import BallClip, LabelStore

MARGIN_S = 0.75


@dataclass(frozen=True)
class Subject:
    detector: str  # spec, or several specs joined by "|" (folds)
    schedule: Schedule = field(default_factory=Schedule)

    @property
    def key(self) -> str:
        d = self.detector.replace(":", "-").replace("@", "_").replace("|", "+")
        return f"{d}__{self.schedule.label()}".replace("/", "_")

    @property
    def label(self) -> str:
        return f"{self.detector} · {self.schedule.label()}"


def parse_subject(text: str) -> Subject:
    """``detector[,sweep_hz]`` e.g. ``motion``, ``motion,15``, ``unet:a|unet:b@1920,15``."""
    det, _, rate = text.partition(",")
    return Subject(det.strip(), Schedule(float(rate)) if rate.strip() else Schedule())


def spec_for_clip(
    subject: Subject, clip: BallClip, output_root: Path, honor_folds: bool = True
) -> str | None:
    """The detector spec to score ``clip`` with (None: every fold trained on it).

    ``honor_folds=False`` uses the first spec even on its own training clips (to compute
    features for training something downstream, never for scoring)."""
    if not honor_folds:
        return subject.detector.split("|")[0]
    from swingvision.ball.detectors.base import DetectorSpec
    from swingvision.ball.detectors.unet import card_path

    key = f"{clip.session_id}/{clip.clip_id}"
    for spec in subject.detector.split("|"):
        ds = DetectorSpec.parse(spec)
        if ds.name == "unet" and ds.weights:
            p = card_path(output_root, ds.weights)
            if p.exists():
                card = json.loads(p.read_text(encoding="utf-8"))
                if key in card.get("train_clips", []):
                    continue
        return spec
    return None


@dataclass
class ClipContext:
    session: object
    config: object
    cal: object
    boxes: PlayerBoxes | None
    onsets: pa.Table | None


class Bench:
    def __init__(
        self, store: LabelStore, settings, out_dir: Path | None = None, honor_folds: bool = True
    ):
        self.honor_folds = honor_folds
        self.store = store
        self.settings = settings
        self.root = Path(settings.require_output_root())
        self.out = out_dir or self.root / "training" / "bench"
        self._ctx: dict[str, ClipContext] = {}

    def context(self, session_id: str) -> ClipContext:
        if session_id not in self._ctx:
            from swingvision import services

            s = services.session_by_id(self.settings, session_id)
            if s is None:
                raise KeyError(session_id)
            onsets = (
                tables.read_table(s.audio_onsets_path) if s.audio_onsets_path.exists() else None
            )
            boxes = (
                PlayerBoxes.from_movement(tables.read_table(s.movement_path))
                if s.movement_path.exists()
                else None
            )
            self._ctx[session_id] = ClipContext(
                s, s.load_config(), calib.load(s.calibration_path), boxes, onsets
            )
        return self._ctx[session_id]

    # -- running -----------------------------------------------------------------------
    def run_dir(self, subject: Subject) -> Path:
        return self.out / "runs" / subject.key

    def candidates(
        self, subject: Subject, clip: BallClip, refresh: bool = False, log=print
    ) -> tuple[pa.Table, pa.Table, dict] | None:
        spec = spec_for_clip(subject, clip, self.root, self.honor_folds)
        if spec is None:
            return None
        d = self.run_dir(subject)
        base = d / f"{clip.session_id}_{clip.clip_id}"
        cp, fp, jp = (
            base.with_suffix(".parquet"),
            Path(f"{base}_frames.parquet"),
            base.with_suffix(".json"),
        )
        if cp.exists() and fp.exists() and jp.exists() and not refresh:
            return pq.read_table(cp), pq.read_table(fp), json.loads(jp.read_text("utf-8"))
        log(f"  running {subject.label} ({spec}) on {clip.session_id}/{clip.clip_id}")
        cand, frames, info = self._run(subject, spec, clip)
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(cand, cp)
        pq.write_table(frames, fp)
        jp.write_text(json.dumps(info, indent=1), "utf-8")
        return cand, frames, info

    def _run(self, subject: Subject, spec: str, clip: BallClip):
        from swingvision.ball.detect import ball_crop, run_detector
        from swingvision.ball.detectors import make_ball_detector
        from swingvision.io.frames import open_source

        cx = self.context(clip.session_id)
        p = self.settings.processing
        det = make_ball_detector(spec)
        det.prepare(cx.config.video, ball_crop(cx.cal, p.roi_behind_m, p.roi_beside_m))
        sched = subject.schedule
        with open_source(Path(cx.config.source.path), cx.config.video, p.decode_backend) as src:
            times = src.table.times()
            t0 = max(0.0, float(times[clip.frame0]) - MARGIN_S)
            t1 = float(times[min(clip.frame1, len(times) - 1)]) + MARGIN_S
            wall = time.perf_counter()
            if sched.full_rate:
                res = run_detector(src, det, target_frames(src.table, [Window(t0, t1)]))
                cand, frames, dsec = res.candidates, res.frames, res.detect_seconds
                n_refine = 0
                refine_s = 0.0
            else:
                sw = run_detector(
                    src, det, target_frames(src.table, [Window(t0, t1, sched.sweep_hz)])
                )
                track = self._link(clip, sw.candidates, sw.frames)
                evs, _ = ev.detect_events(track, self._event_ctx(clip))
                from swingvision.pipeline.stages.ball import refine_moments

                onsets = cx.onsets
                if onsets is not None:
                    t_on = onsets.column("t_s").to_numpy()
                    onsets = onsets.filter(pa.array((t_on >= t0) & (t_on <= t1)))
                moments = [m for m in refine_moments(track, evs, onsets) if t0 <= m <= t1]
                windows = merge_windows(moments, sched.before_s, sched.after_s, t1)
                windows = [Window(max(w.t0, t0), min(w.t1, t1)) for w in windows]
                rf = run_detector(src, det, target_frames(src.table, windows))
                refined = pa.array(np.unique(rf.frames.column("frame").to_numpy()))
                c0 = sw.candidates.filter(
                    pc.invert(pc.is_in(sw.candidates.column("frame"), refined))
                )
                f0 = sw.frames.filter(pc.invert(pc.is_in(sw.frames.column("frame"), refined)))
                cand = pa.concat_tables([c0, rf.candidates]).sort_by("frame")
                frames = pa.concat_tables([f0, rf.frames]).sort_by("frame")
                dsec = sw.detect_seconds + rf.detect_seconds
                n_refine = rf.frames.num_rows
                refine_s = sum(w.t1 - w.t0 for w in windows)
            wall = time.perf_counter() - wall
        info = {
            "spec": spec,
            "span_s": t1 - t0,
            "detect_seconds": dsec,
            "wall_seconds": wall,
            "frames_processed": frames.num_rows,
            "refine_frames": n_refine,
            "refine_seconds_covered": refine_s,
        }
        return cand, frames, info

    # -- scoring -----------------------------------------------------------------------
    def _link(self, clip: BallClip, cand: pa.Table, frames: pa.Table) -> pa.Table:
        from swingvision.io.frames import frame_table

        cx = self.context(clip.session_id)
        ft = frame_table(str(cx.config.source.path))
        times = ft.times()
        f = frames.column("frame").to_numpy()
        lo, hi = (int(f.min()), int(f.max()) + 1) if len(f) else (0, 0)
        allf = np.arange(lo, hi, dtype=np.int64)
        track, _ = link(
            cand,
            frames,
            cx.config.video.display_width,
            all_frames=(allf, times[allf]),
            player_boxes=cx.boxes,
        )
        return track

    def _event_ctx(self, clip: BallClip) -> ev.EventContext:
        from swingvision.pipeline.stages.ball import event_context

        cx = self.context(clip.session_id)
        return event_context(cx.session, cx.config, cx.cal, cx.onsets)

    def score(
        self,
        subject: Subject,
        clips: list[BallClip],
        refresh: bool = False,
        log: Callable[[str], None] = print,
    ) -> dict:
        preds: dict[str, dict] = {}
        raw: dict[str, dict] = {}
        cams = {}
        scored: list[BallClip] = []
        bounce, hit = EventScore(), EventScore()
        cost_det, span = 0.0, 0.0
        refine_cov = 0.0
        for clip in clips:
            got = self.candidates(subject, clip, refresh, log)
            if got is None:
                continue
            cand, frames, info = got
            scored.append(clip)
            cx = self.context(clip.session_id)
            cams[clip.session_id] = calib.camera_at(cx.cal, clip.t0_s)
            track = self._link(clip, cand, frames)
            key = f"{clip.session_id}/{clip.clip_id}"
            preds[key] = {
                int(f): (float(x), float(y))
                for f, x, y in zip(
                    track.column("frame").to_numpy(),
                    track.column("x").to_numpy(),
                    track.column("y").to_numpy(),
                    strict=True,
                )
            }
            raw[key] = top_candidates(cand)
            evs, _ = ev.detect_events(track, self._event_ctx(clip))
            score_events(clip, evs, cams[clip.session_id], bounce, hit)
            cost_det += info["detect_seconds"]
            span += info["span_s"]
            refine_cov += info.get("refine_seconds_covered", 0.0)
        ball = score_ball(scored, preds, cams)
        strict = score_ball(scored, preds, cams, motion_aware=False)
        raw_score = score_ball(scored, raw, cams)
        return {
            "subject": subject.label,
            "key": subject.key,
            "clips": len(scored),
            "ball": ball.summary(),
            "ball_strict": strict.summary(),
            "raw_top1": raw_score.summary(),
            "per_clip": ball.per_clip,
            "bounce": bounce.summary(),
            "hit": hit.summary(),
            "gpu_s_per_hour": round(cost_det / max(span, 1e-6) * 3600, 1),
            "refine_share": round(refine_cov / max(span, 1e-6), 3),
        }


def top_candidates(cand: pa.Table, min_score: float = 0.3) -> dict[int, tuple[float, float]]:
    """Best candidate per frame (no linking)."""
    out: dict[int, tuple[float, float, float]] = {}
    for f, x, y, s in zip(
        cand.column("frame").to_numpy(),
        cand.column("x").to_numpy(),
        cand.column("y").to_numpy(),
        cand.column("score").to_numpy(),
        strict=True,
    ):
        if s >= min_score and (int(f) not in out or s > out[int(f)][2]):
            out[int(f)] = (float(x), float(y), float(s))
    return {k: (v[0], v[1]) for k, v in out.items()}


def gt_contact(clip: BallClip, frame: int) -> tuple[float, float, float] | None:
    """Sub-frame contact (t in frames, x, y) of a labeled event from the labels around it."""
    fs, xs, ys = [], [], []
    for f in range(frame - 7, frame + 8):
        lab = clip.label(f)
        if lab is not None and lab.vis == "visible" and lab.x is not None:
            fs.append(f)
            xs.append(lab.x)
            ys.append(lab.y)
    if frame not in fs:
        return None
    t = np.array(fs, dtype=np.float64)
    i = fs.index(frame)
    return ev.refine_contact(t, np.array(xs), np.array(ys), i)


def score_events(clip: BallClip, evs: pa.Table, cam, bounce: EventScore, hit: EventScore):
    """Add one clip's event matches to the running scores (only labeled-clip frames count)."""
    if not clip.events_labeled:
        return
    kinds = evs.column("kind").to_pylist() if evs.num_rows else []
    frames = evs.column("frame").to_numpy() if evs.num_rows else np.zeros(0, dtype=np.int64)
    inside = (frames >= clip.frame0) & (frames <= clip.frame1)
    for kind, score in (("bounce", bounce), ("hit", hit)):
        gt = [e for e in clip.events if e.kind == kind]
        pi = [i for i, k in enumerate(kinds) if k == kind and inside[i]]
        pf = [int(frames[i]) for i in pi]
        m = match_events([e.frame for e in gt], pf)
        score.tp += len(m)
        score.fp += len(pf) - len(m)
        score.fn += len(gt) - len(m)
        for gi, pj in m:
            score.frame_errors.append(pf[pj] - gt[gi].frame)
            if kind != "bounce":
                continue
            c = gt_contact(clip, gt[gi].frame)
            if c is None:
                continue
            g = cam.image_to_ground(np.array([[c[1], c[2]]]), ev.BALL_RADIUS_M)[0]
            row = pi[pj]
            px = evs.column("court_x")[row].as_py()
            py = evs.column("court_y")[row].as_py()
            if px is None or py is None or not np.isfinite(g).all() or not np.isfinite(px):
                continue
            err = float(np.hypot(px - g[0], py - g[1]))
            (score.pos_errors_far if g[1] > 0 else score.pos_errors_near).append(err)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(results: list[dict], path: Path, title: str = "Ball detector benchmark") -> None:
    import plotly.graph_objects as go

    rows = []
    for r in results:
        b = r["ball"]
        rows.append(
            "<tr>"
            f"<td>{r['subject']}</td><td>{r['clips']}</td>"
            f"<td>{b['all']['f1']:.3f}</td><td>{b['near']['f1']:.3f}</td><td>{b['far']['f1']:.3f}</td>"
            f"<td>{b['all']['precision']:.3f}</td><td>{b['all']['recall']:.3f}</td>"
            f"<td>{r['ball_strict']['near']['f1']:.3f}</td>"
            f"<td>{r['ball_strict']['far']['f1']:.3f}</td>"
            f"<td>{r['raw_top1']['all']['f1']:.3f}</td>"
            f"<td>{r['bounce']['f1']:.3f}</td><td>{r['hit']['f1']:.3f}</td>"
            f"<td>{_fmt(r['bounce']['median_pos_err_near_m'])}</td>"
            f"<td>{_fmt(r['bounce']['median_pos_err_far_m'])}</td>"
            f"<td>{r['gpu_s_per_hour']:.0f}</td><td>{r['refine_share']:.2f}</td>"
            "</tr>"
        )
    fig = go.Figure()
    xs = np.array([r["gpu_s_per_hour"] / 60 for r in results])
    ys = np.array([r["ball"]["all"]["f1"] for r in results])
    front = pareto(xs, ys)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=[r["subject"] for r in results],
            textposition="top center",
            marker={"size": 11, "color": ["#0b7285" if f else "#adb5bd" for f in front]},
            name="subjects",
        )
    )
    order = np.argsort(xs)
    fx = [xs[i] for i in order if front[i]]
    fy = [ys[i] for i in order if front[i]]
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line={"dash": "dot"}, name="Pareto front"))
    fig.update_layout(
        xaxis_title="GPU minutes per footage hour (detector only)",
        yaxis_title="Ball F1 (linked track, all clips)",
        template="plotly_white",
        height=520,
    )
    chart = fig.to_html(full_html=False, include_plotlyjs=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}th{{background:#f1f3f5}}</style></head><body>
<h1>{title}</h1>
<p>Ground-truth clips, full frame rate. F1 of the linked ball track (tolerance max(4 px, 0.4 ×
ball diameter, 0.25 × per-frame motion); "strict" without the motion term); "raw" is the best
candidate per frame without linking. Events match within
±2 frames; landing error on the ground (m). Cost: detector GPU-seconds per footage hour.</p>
<table><tr><th>Subject</th><th>Clips</th><th>F1</th><th>F1 near</th><th>F1 far</th>
<th>Precision</th><th>Recall</th><th>Strict near</th><th>Strict far</th><th>Raw F1</th><th>Bounce F1</th><th>Hit F1</th>
<th>Bounce err near (m)</th><th>Bounce err far (m)</th><th>GPU s/h</th><th>Refine share</th></tr>
{"".join(rows)}</table>
<h2>Accuracy vs cost</h2>{chart}
<h2>Raw results</h2><pre>{json.dumps(results, indent=1)}</pre>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def pareto(cost: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """True where no other subject is at least as cheap and at least as accurate (and better
    in one)."""
    n = len(cost)
    out = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (
                i != j
                and cost[j] <= cost[i]
                and acc[j] >= acc[i]
                and (cost[j] < cost[i] or acc[j] > acc[i])
            ):
                out[i] = False
                break
    return out


@dataclass
class BenchResult:
    results: list[dict] = field(default_factory=list)
    report: Path | None = None


def run_bench(
    store: LabelStore,
    settings,
    subjects: list[Subject],
    clips: list[BallClip],
    refresh: bool = False,
    log: Callable[[str], None] = print,
) -> BenchResult:
    bench = Bench(store, settings)
    out = BenchResult()
    for s in subjects:
        log(f"{s.label}")
        out.results.append(bench.score(s, clips, refresh, log))
        r = out.results[-1]
        log(
            f"  F1 {r['ball']['all']['f1']:.3f} (near {r['ball']['near']['f1']:.3f}, far "
            f"{r['ball']['far']['f1']:.3f}), bounce F1 {r['bounce']['f1']:.3f}, "
            f"{r['gpu_s_per_hour']:.0f} GPU s/h"
        )
    bench.out.mkdir(parents=True, exist_ok=True)
    (bench.out / "results.json").write_text(json.dumps(out.results, indent=1), "utf-8")
    out.report = bench.out / "report.html"
    write_report(out.results, out.report)
    return out
