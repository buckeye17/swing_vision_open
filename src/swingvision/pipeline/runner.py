"""Resolve the stage DAG for a session and run the stale stages in order."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from loguru import logger

from swingvision.pipeline.errors import describe, is_gpu_oom
from swingvision.pipeline.stage import (
    Cancelled,
    NeedsUserAction,
    Stage,
    StageContext,
    build_manifest,
    clear_manifest,
    read_manifest,
    stable_hash,
    write_manifest,
)
from swingvision.settings import AppSettings
from swingvision.storage.schemas import SessionConfig
from swingvision.storage.session import Session


class Registry:
    def __init__(self, stages: Iterable[Stage] = ()):
        self._stages: dict[str, Stage] = {}
        for s in stages:
            self.add(s)

    def add(self, stage: Stage) -> None:
        if stage.name in self._stages:
            raise ValueError(f"Duplicate stage {stage.name}")
        for dep in (*stage.depends_on, *stage.after):
            if dep not in self._stages:
                raise ValueError(f"{stage.name} depends on unknown/later stage {dep}")
        self._stages[stage.name] = stage

    def __getitem__(self, name: str) -> Stage:
        return self._stages[name]

    def __contains__(self, name: str) -> bool:
        return name in self._stages

    def names(self) -> list[str]:
        return list(self._stages)

    def for_mode(self, mode: str) -> list[Stage]:
        return [s for s in self._stages.values() if mode in s.modes]


@dataclass
class PlannedStage:
    stage: Stage
    config_hash: str
    inputs: dict[str, str]
    fingerprint: str
    fresh: bool
    reason: str


def plan(
    registry: Registry,
    session: Session,
    config: SessionConfig,
    settings: AppSettings,
    targets: list[str] | None = None,
    force: Iterable[str] = (),
) -> list[PlannedStage]:
    """Stages needed for ``targets`` (default: every stage for this mode), in run order."""
    applicable = {s.name: s for s in registry.for_mode(config.mode)}
    wanted = set(targets) if targets else set(applicable)
    unknown = wanted - set(applicable)
    if unknown:
        raise ValueError(f"Stages not applicable to {config.mode} sessions: {sorted(unknown)}")

    needed: set[str] = set()

    def visit(name: str) -> None:
        if name in needed:
            return
        needed.add(name)
        for dep in (*applicable[name].depends_on, *applicable[name].after):
            if dep not in applicable:
                raise ValueError(f"{name} depends on {dep}, which is not applicable")
            visit(dep)

    for name in wanted:
        visit(name)

    force = set(force)
    fingerprints: dict[str, str] = {}
    rerun: set[str] = set()
    result: list[PlannedStage] = []
    for stage in registry.for_mode(config.mode):  # registration order is topological
        if stage.name not in needed:
            continue
        p = _plan_stage(stage, session, config, settings, fingerprints, force, rerun)
        fingerprints[stage.name] = p.fingerprint
        if not p.fresh:
            rerun.add(stage.name)
        result.append(p)
    return result


def _plan_stage(
    stage: Stage,
    session: Session,
    config: SessionConfig,
    settings: AppSettings,
    fingerprints: dict[str, str],
    force: set[str],
    rerun: set[str] = frozenset(),  # type: ignore[assignment]
) -> PlannedStage:
    """``rerun``: stages that run (or will) in this job. Their outputs may change even when
    their fingerprint doesn't (a forced rerun), so their dependents rerun too."""
    config_hash = stable_hash(stage.config(session, config, settings))
    inputs = {dep: fingerprints[dep] for dep in stage.depends_on}
    fingerprint = stable_hash([stage.name, stage.version, config_hash, inputs])
    manifest = read_manifest(session, stage.name)
    missing = [p for p in stage.outputs(session) if not p.exists()]
    if stage.name in force:
        fresh, reason = False, "forced"
    elif manifest is None:
        fresh, reason = False, "never run"
    elif manifest.get("fingerprint") != fingerprint:
        fresh, reason = False, _stale_reason(manifest, stage, config_hash, inputs)
    elif missing:
        fresh, reason = False, f"missing output {missing[0].name}"
    elif upstream := [d for d in stage.depends_on if d in rerun]:
        fresh, reason = False, f"upstream reruns ({', '.join(upstream)})"
    else:
        fresh, reason = True, "up to date"
    return PlannedStage(stage, config_hash, inputs, fingerprint, fresh, reason)


def _stale_reason(manifest: dict, stage: Stage, config_hash: str, inputs: dict) -> str:
    if manifest.get("version") != stage.version:
        return f"stage version {manifest.get('version')} → {stage.version}"
    if manifest.get("config_hash") != config_hash:
        return "settings changed"
    changed = [k for k, v in inputs.items() if manifest.get("inputs", {}).get(k) != v]
    return f"upstream changed ({', '.join(changed)})" if changed else "fingerprint changed"


@dataclass
class RunHooks:
    on_plan: Callable[[list[PlannedStage]], None] = lambda planned: None
    on_stage_start: Callable[[str], None] = lambda name: None
    on_stage_progress: Callable[[str, float, str | None], None] = lambda n, f, m: None
    on_stage_end: Callable[[str, str, str | None], None] = lambda name, status, msg: None
    is_cancelled: Callable[[], bool] = lambda: False


@dataclass
class RunResult:
    status: str  # done | cancelled | needs_action | failed
    message: str | None = None
    error: str | None = None
    action: str | None = None  # needs_action: what for (NeedsUserAction.reason)
    ran: list[str] = field(default_factory=list)


def run(
    registry: Registry,
    session: Session,
    settings: AppSettings,
    targets: list[str] | None = None,
    force: Iterable[str] = (),
    hooks: RunHooks | None = None,
) -> RunResult:
    hooks = hooks or RunHooks()
    config = session.load_config()
    planned = plan(registry, session, config, settings, targets, force)
    hooks.on_plan(planned)
    force = set(force)
    fingerprints: dict[str, str] = {}
    ran: list[str] = []
    for initial in planned:
        name = initial.stage.name
        # Re-plan just before running: a stage's config may depend on files that upstream
        # stages (or the user, e.g. calibration edits) wrote after the initial plan.
        config = session.load_config()
        p = _plan_stage(initial.stage, session, config, settings, fingerprints, force, set(ran))
        fingerprints[name] = p.fingerprint
        if p.fresh:
            hooks.on_stage_end(name, "skipped", p.reason)
            continue
        if hooks.is_cancelled():
            return RunResult("cancelled", "Cancelled", ran=ran)
        hooks.on_stage_start(name)
        logger.info("Stage {} starting ({})", name, p.reason)
        clear_manifest(session, name)
        ctx = StageContext(
            session=session,
            config=config,
            settings=settings,
            fingerprint=p.fingerprint,
            stage_name=name,
            report=lambda frac, msg, name=name: hooks.on_stage_progress(name, frac, msg),
            is_cancelled=hooks.is_cancelled,
            log=logger.bind(stage=name),
        )
        started = time.time()
        try:
            extra = _run_stage(p.stage, ctx)
        except Cancelled:
            hooks.on_stage_end(name, "cancelled", None)
            return RunResult("cancelled", f"Cancelled during {name}", ran=ran)
        except NeedsUserAction as exc:
            hooks.on_stage_end(name, "needs_action", exc.message)
            return RunResult("needs_action", exc.message, action=exc.reason, ran=ran)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Stage {} failed:\n{}", name, tb)
            why = describe(exc)
            hooks.on_stage_end(name, "failed", why)
            title = p.stage.title or name
            return RunResult("failed", f"{title} ({name}) failed: {why}", error=tb, ran=ran)
        finally:
            _release_gpu_memory()
        write_manifest(
            session,
            name,
            build_manifest(
                p.stage, session, p.config_hash, p.inputs, p.fingerprint, started, extra
            ),
        )
        ran.append(name)
        logger.info("Stage {} done in {:.1f}s", name, time.time() - started)
        hooks.on_stage_end(name, "done", None)
    return RunResult("done", "Complete", ran=ran)


#: Before the one retry after a GPU out-of-memory error.
OOM_RETRY_WAIT_S = 30.0


def _run_stage(stage: Stage, ctx: StageContext):
    """Run a stage, retrying once after a GPU out-of-memory error (another program may
    have held the memory for a while; chunked stages resume from their checkpoints)."""
    try:
        return stage.run(ctx)
    except Exception as exc:
        if not (stage.uses_gpu and is_gpu_oom(exc)):
            raise
        ctx.log.warning("GPU out of memory in {}; retrying once", stage.name)
    _release_gpu_memory()
    time.sleep(OOM_RETRY_WAIT_S)
    return stage.run(ctx)


def _release_gpu_memory() -> None:
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
