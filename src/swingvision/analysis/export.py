"""Export a session's tables as CSV or Parquet (PLAN.md §5.1: ``pyarrow.csv`` for CSV).

``shots`` is ``shots.parquet`` with, for practice sessions, each shot's practice result
(call after your corrections, target, excluded) in ``practice_*`` columns. ``practice`` and
``swings`` are the session's files as they are. Parquet keeps the units in the field
metadata; CSV joins list values with ``|``.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from swingvision.storage import tables
from swingvision.storage.session import Session

EXPORTS = {
    "shots": "Shots",
    "practice": "Practice shots",
    "swings": "Swings",
}
FORMATS = ("csv", "parquet")

#: Practice columns added to the shots export (as ``practice_<name>``).
PRACTICE_COLUMNS = (
    "segment_id",
    "block_id",
    "excluded",
    "shot_kind",
    "serve_side",
    "landing_x",
    "landing_y",
    "landing_source",
    "landing_confirmed",
    "outcome",
    "call_area",
    "margin_m",
    "close_call",
    "target_id",
    "in_target",
    "target_dist_m",
    "depth_err_m",
    "width_err_m",
)


class ExportError(ValueError):
    pass


def _path(session: Session, what: str):
    return {
        "shots": session.shots_path,
        "practice": session.practice_path,
        "swings": session.swings_path,
    }[what]


def export_table(session: Session, what: str) -> pa.Table:
    if what not in EXPORTS:
        raise ExportError(f"Unknown export {what!r}; choose one of {', '.join(EXPORTS)}")
    path = _path(session, what)
    if not path.exists():
        raise ExportError(f"{EXPORTS[what]} aren't available yet: process the session first.")
    table = tables.read_table(path)
    if what == "shots" and session.practice_path.exists():
        table = _with_practice(table, tables.read_table(session.practice_path))
    return table


def _with_practice(shots: pa.Table, practice: pa.Table) -> pa.Table:
    """Left-join the practice result of each shot (by ``shot_id``)."""
    by_shot = {r["shot_id"]: r for r in practice.select(["shot_id", *PRACTICE_COLUMNS]).to_pylist()}
    ids = shots.column("shot_id").to_pylist()
    for name in PRACTICE_COLUMNS:
        f = practice.schema.field(name)
        values = [by_shot[i][name] if i in by_shot else None for i in ids]
        shots = shots.append_column(
            pa.field(f"practice_{name}", f.type, metadata=f.metadata), pa.array(values, f.type)
        )
    return shots


def to_csv(table: pa.Table) -> bytes:
    cols = []
    for name in table.column_names:
        col = table.column(name)
        if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
            col = pc.binary_join(pc.cast(col, pa.list_(pa.string())), "|")
        cols.append(col)
    flat = pa.table(cols, names=table.column_names)
    buf = io.BytesIO()
    pa_csv.write_csv(flat, buf)
    return buf.getvalue()


def to_parquet(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression=tables.COMPRESSION)
    return buf.getvalue()


def export_bytes(session: Session, what: str, fmt: str) -> bytes:
    if fmt not in FORMATS:
        raise ExportError(f"Unknown format {fmt!r}; choose csv or parquet")
    table = export_table(session, what)
    return to_csv(table) if fmt == "csv" else to_parquet(table)


def filename(session: Session, what: str, fmt: str) -> str:
    return f"{session.path.name}_{what}.{fmt}"
