"""M7 stage: ``stats`` (session aggregates → ``stats.json``; PLAN.md §6 stage 20).

CPU-only and fast, so the app reruns it with ``practice_eval`` after the user's edits. Match
statistics (serve and return points, rally lengths) join it in Phase 2.
"""

from __future__ import annotations

from swingvision.analysis import stats as st
from swingvision.pipeline.stage import Stage, StageContext
from swingvision.storage.fsutil import atomic_write_json


class StatsStage(Stage):
    name = "stats"
    title = "Statistics"
    version = 1
    depends_on = ("shots", "movement", "swings", "practice_eval")
    modes = frozenset({"practice"})
    weight = 0.05

    def config(self, session, config, settings):
        p = settings.processing
        return {"version": st.STATS_VERSION, "dark_luma": p.dark_luma, "view_min": p.view_min}

    def outputs(self, session):
        return [session.stats_path]

    def run(self, ctx: StageContext):
        p = ctx.settings.processing
        data = st.load(ctx.session)
        stats = st.session_stats(data, p.dark_luma, p.view_min)
        atomic_write_json(ctx.session.stats_path, stats)
        shots = stats["shots"]
        return {
            "n_shots": shots["n"],
            "in_pct": shots["in_pct"],
            "speed_median": shots["speed_median"],
            "distance_m": stats["movement"].get("distance_m"),
        }
