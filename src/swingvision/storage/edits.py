"""User overrides layered over derived data (``edits.json``, PLAN.md §9.3).

Derived files are never mutated: stages that honor edits (``swings`` for strokes,
``practice_eval`` for practice shots) read this file and fold the part they use into their
fingerprint, so an edit reruns just them (and what depends on them).

Writes use optimistic versioning: :func:`update` takes the version the caller last read
and refuses to overwrite a newer file (two browser tabs editing the same session).
"""

from __future__ import annotations

from collections.abc import Callable

from swingvision.storage.fsutil import atomic_write_text
from swingvision.storage.schemas import (
    PracticeShotEdit,
    ServeEdit,
    SessionEdits,
    SpeedRefEdit,
    SwingEdit,
)
from swingvision.storage.session import Session

#: An edit applies to the practice shot whose anchor time is within this of its ``t``.
MATCH_TOL_S = 0.25


class EditConflict(RuntimeError):
    """The edits file changed since the caller read it."""


def load(session: Session) -> SessionEdits:
    path = session.edits_path
    if not path.exists():
        return SessionEdits()
    return SessionEdits.model_validate_json(path.read_text(encoding="utf-8"))


def save(session: Session, edits: SessionEdits) -> None:
    atomic_write_text(session.edits_path, edits.model_dump_json(indent=2))


def update(
    session: Session,
    change: Callable[[SessionEdits], None],
    expected_version: int | None = None,
) -> SessionEdits:
    """Apply ``change`` to the current edits and save them with the version bumped."""
    edits = load(session)
    if expected_version is not None and edits.version != expected_version:
        raise EditConflict(
            f"The session's edits changed (version {edits.version}, expected "
            f"{expected_version}). Reload the page."
        )
    change(edits)
    edits.version += 1
    save(session, edits)
    return edits


def practice_shot_edit(edits: SessionEdits, t: float) -> PracticeShotEdit | None:
    """The edit for the practice shot anchored at ``t`` (nearest within the tolerance)."""
    best, best_d = None, MATCH_TOL_S
    for e in edits.practice_shots:
        d = abs(e.t - t)
        if d <= best_d:
            best, best_d = e, d
    return best


def set_practice_shot(
    edits: SessionEdits,
    t: float,
    *,
    exclude: bool | None = None,
    landing: list[float] | bool | None = False,
    confirmed: bool | None = None,
) -> None:
    """Change one practice shot's edit in place; ``landing=None`` clears a placed landing,
    ``False`` (the default) leaves it. An edit back at the defaults is dropped."""
    e = practice_shot_edit(edits, t)
    if e is None:
        e = PracticeShotEdit(t=round(t, 3))
        edits.practice_shots.append(e)
    if exclude is not None:
        e.exclude = exclude
    if landing is not False:
        e.landing = None if landing is None else [round(float(v), 3) for v in landing]  # type: ignore[union-attr]
    if confirmed is not None:
        e.confirmed = confirmed
    if not e.exclude and e.landing is None and not e.confirmed:
        edits.practice_shots.remove(e)
    edits.practice_shots.sort(key=lambda x: x.t)


def swing_edit(edits: SessionEdits, t: float) -> SwingEdit | None:
    """The stroke correction for the swing whose contact is at ``t``."""
    best, best_d = None, MATCH_TOL_S
    for e in edits.swings:
        d = abs(e.t - t)
        if d <= best_d:
            best, best_d = e, d
    return best


def set_swing_stroke(edits: SessionEdits, t: float, stroke: str | None) -> None:
    """Set (or with ``None`` clear) the user's stroke for the swing at ``t``."""
    e = swing_edit(edits, t)
    if stroke is None:
        if e is not None:
            edits.swings.remove(e)
        return
    if e is None:
        edits.swings.append(SwingEdit(t=round(t, 3), stroke=stroke))
    else:
        e.stroke = stroke
    edits.swings.sort(key=lambda x: x.t)


def serve_edit(edits: SessionEdits, t: float) -> ServeEdit | None:
    """The contact correction for the serve whose contact is at ``t`` (M7b)."""
    best, best_d = None, MATCH_TOL_S
    for e in edits.serves:
        d = abs(e.t - t)
        if d <= best_d:
            best, best_d = e, d
    return best


def set_serve(
    edits: SessionEdits,
    t: float,
    *,
    frame: int | bool | None = False,
    toe: list[float] | bool | None = False,
) -> None:
    """Change one serve's contact correction: the contact frame and the toe tip's pixel in
    it. ``None`` clears a field, ``False`` (the default) leaves it; an edit with neither is
    dropped. ``t`` is the serve's contact time *before* any frame correction (the swing's
    estimate), so the edit finds the serve again after it moved."""
    e = serve_edit(edits, t)
    if e is None:
        e = ServeEdit(t=round(t, 3))
        edits.serves.append(e)
    if frame is not False:
        e.frame = None if frame is None else int(frame)  # type: ignore[arg-type]
    if toe is not False:
        e.toe = None if toe is None else [round(float(v), 1) for v in toe]  # type: ignore[union-attr]
    if e.frame is None and e.toe is None:
        edits.serves.remove(e)
    edits.serves.sort(key=lambda x: x.t)


def speed_ref_edit(edits: SessionEdits, t: float) -> SpeedRefEdit | None:
    """The review of the reference serve whose contact is at ``t`` (M7c)."""
    best, best_d = None, MATCH_TOL_S
    for e in edits.speed_refs:
        d = abs(e.t - t)
        if d <= best_d:
            best, best_d = e, d
    return best


def set_speed_ref(
    edits: SessionEdits,
    t: float,
    *,
    status: str | bool | None = False,
    marked: bool | None = None,
    t_racket: float | bool | None = False,
    t_tape: float | bool | None = False,
) -> None:
    """Change one reference serve's review: ``status`` accepted | rejected (``None`` clears
    the decision), ``marked`` as a tape hit, onsets placed on the waveform (audio clock;
    ``None`` back to the detected one). ``False`` leaves a field; an edit back at the
    defaults is dropped."""
    e = speed_ref_edit(edits, t)
    if e is None:
        e = SpeedRefEdit(t=round(t, 3))
        edits.speed_refs.append(e)
    if status is not False:
        e.status = status  # type: ignore[assignment]
    if marked is not None:
        e.marked = marked
    if t_racket is not False:
        e.t_racket = None if t_racket is None else round(float(t_racket), 5)  # type: ignore[arg-type]
    if t_tape is not False:
        e.t_tape = None if t_tape is None else round(float(t_tape), 5)  # type: ignore[arg-type]
    if e.status is None and not e.marked and e.t_racket is None and e.t_tape is None:
        edits.speed_refs.remove(e)
    edits.speed_refs.sort(key=lambda x: x.t)
