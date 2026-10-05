from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swingvision.storage import tables
from swingvision.storage.schemas import AUDIO_ONSETS


def _onsets(times):
    n = len(times)
    return pa.table(
        {
            "flatness": pa.array([0.1] * n),  # deliberately out of order + float64
            "t_s": pa.array(times),
            "strength": pa.array([5.0] * n),
            "rms_db": pa.array([-40.0] * n),
            "centroid_hz": pa.array([2000.0] * n),
        }
    )


def test_conform_reorders_and_casts():
    t = tables.conform(_onsets([1.0, 2.0]), AUDIO_ONSETS)
    assert t.schema.names == AUDIO_ONSETS.names
    assert t.schema.field("strength").type == pa.float32()


def test_conform_rejects_missing_or_extra_columns():
    with pytest.raises(tables.SchemaMismatchError):
        tables.conform(_onsets([1.0]).drop_columns(["rms_db"]), AUDIO_ONSETS)
    with pytest.raises(tables.SchemaMismatchError):
        tables.conform(_onsets([1.0]).append_column("x", pa.array([1])), AUDIO_ONSETS)


def test_write_read_roundtrip_keeps_metadata(tmp_path):
    path = tmp_path / "o.parquet"
    tables.write_table(_onsets([1.0, 2.0]), path, AUDIO_ONSETS)
    back = tables.read_table(path)
    assert back.num_rows == 2
    assert back.schema.metadata[b"name"] == b"audio_onsets"
    assert back.schema.field("t_s").metadata[b"unit"] == b"s"
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"
    assert not list(tmp_path.glob(".*"))  # no temp files left behind


def test_parts_consolidate_sorted(tmp_path):
    parts = tmp_path / "parts"
    tables.write_part(_onsets([3.0, 4.0]), parts, 1, AUDIO_ONSETS)
    tables.write_part(_onsets([1.0]), parts, 0, AUDIO_ONSETS)
    out = tmp_path / "all.parquet"
    merged = tables.consolidate_parts(parts, out, AUDIO_ONSETS, sort_by="t_s")
    assert merged.column("t_s").to_pylist() == [1.0, 3.0, 4.0]
    assert not parts.exists()
    assert tables.read_table(out).num_rows == 3


def test_read_parts_of_empty_dir_is_empty_table(tmp_path):
    (tmp_path / "p").mkdir()
    assert tables.read_parts(tmp_path / "p", AUDIO_ONSETS).num_rows == 0
