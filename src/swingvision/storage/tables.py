"""PyArrow-only Parquet helpers (PLAN.md §5.1).

``pyarrow.Table`` is the only tabular type passed between modules. Every file is
written against an explicit schema, zstd-compressed, and atomically renamed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from swingvision.storage import cache
from swingvision.storage.fsutil import atomic_write

COMPRESSION = "zstd"
PART_GLOB = "part-*.parquet"


class SchemaMismatchError(ValueError):
    pass


def conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Return ``table`` with exactly ``schema`` (column order, types, metadata) or raise."""
    names = schema.names
    missing = [n for n in names if n not in table.column_names]
    extra = [n for n in table.column_names if n not in names]
    if missing or extra:
        raise SchemaMismatchError(f"columns missing={missing} extra={extra}")
    table = table.select(names)
    try:
        return table.cast(schema)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise SchemaMismatchError(str(exc)) from exc


def empty_table(schema: pa.Schema) -> pa.Table:
    return schema.empty_table()


def write_table(table: pa.Table, path: Path, schema: pa.Schema, row_group_size: int | None = None):
    table = conform(table, schema)
    atomic_write(
        path,
        lambda tmp: pq.write_table(
            table, tmp, compression=COMPRESSION, row_group_size=row_group_size
        ),
        suffix=".parquet.tmp",
    )


def read_table(path: Path, columns: list[str] | None = None, filters: Any = None) -> pa.Table:
    return pq.read_table(cache.local(path), columns=columns, filters=filters)


def part_path(directory: Path, index: int) -> Path:
    return directory / f"part-{index:05d}.parquet"


def write_part(table: pa.Table, directory: Path, index: int, schema: pa.Schema) -> Path:
    path = part_path(directory, index)
    write_table(table, path, schema)
    return path


def read_parts(directory: Path, schema: pa.Schema) -> pa.Table:
    files = sorted(directory.glob(PART_GLOB))
    if not files:
        return empty_table(schema)
    local = [str(cache.local(f)) for f in files]
    return ds.dataset(local, format="parquet", schema=schema).to_table()


def consolidate_parts(
    directory: Path, out_path: Path, schema: pa.Schema, sort_by: str | None = None
) -> pa.Table:
    """Merge chunk part files into one file, then remove the part directory."""
    table = read_parts(directory, schema)
    if sort_by and table.num_rows:
        table = table.sort_by(sort_by)
    write_table(table, out_path, schema)
    shutil.rmtree(directory, ignore_errors=True)
    return table


def to_rows(table: pa.Table) -> list[dict[str, Any]]:
    return table.to_pylist()
