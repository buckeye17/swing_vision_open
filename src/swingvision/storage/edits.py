"""User overrides layered over derived data (``edits.json``, PLAN.md §9.3).

Derived files are never mutated: stages that honor edits (``practice_eval`` for now) read
this file and fold its hash into their fingerprint, so an edit reruns just them.

Writes use optimistic versioning: :func:`update` takes the version the caller last read
and refuses to overwrite a newer file (two browser tabs editing the same session).
"""

from __future__ import annotations

from collections.abc import Callable

from swingvision.storage.fsutil import atomic_write_text
from swingvision.storage.schemas import PracticeShotEdit, SessionEdits
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
