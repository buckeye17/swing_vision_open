"""Local read cache for a network output folder (storage.cache)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pyarrow as pa

from swingvision.storage import cache, tables
from swingvision.storage.fsutil import atomic_write_text


def _cache(tmp_path: Path, max_bytes: int = 10**9) -> tuple[Path, cache.LocalCache]:
    remote = tmp_path / "remote"
    (remote / "sessions" / "s1").mkdir(parents=True)
    return remote, cache.LocalCache(remote, tmp_path / "local", max_bytes)


def _bump_mtime(path: Path, seconds: float = 5) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(seconds * 1e9)))


def test_copies_on_first_read_and_reuses(tmp_path):
    remote, c = _cache(tmp_path)
    f = remote / "sessions" / "s1" / "a.json"
    f.write_text("1")
    got = c.local(f)
    assert got == tmp_path / "local" / "sessions" / "s1" / "a.json"
    assert got.read_text() == "1"
    assert got.stat().st_mtime_ns == f.stat().st_mtime_ns
    assert c.local(f) == got


def test_refreshes_when_the_original_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "STAT_TTL_S", 0.0)  # the worker's writes, after the TTL
    remote, c = _cache(tmp_path)
    f = remote / "sessions" / "s1" / "a.json"
    f.write_text("1")
    c.local(f)
    f.write_text("22")  # new size
    assert c.local(f).read_text() == "22"
    f.write_text("33")  # same size: the modification time tells
    _bump_mtime(f)
    assert c.local(f).read_text() == "33"


def test_the_apps_own_writes_show_at_once(tmp_path):
    remote, c = _cache(tmp_path)
    f = remote / "sessions" / "s1" / "session.json"
    atomic_write_text(f, "1")
    cache.activate(c)
    try:
        assert cache.local(f).read_text() == "1"
        atomic_write_text(f, "22")  # within the stat TTL, but this process wrote it
        assert cache.local(f).read_text() == "22"
    finally:
        cache.activate(None)


def test_outside_the_root_and_missing_files_pass_through(tmp_path):
    remote, c = _cache(tmp_path)
    other = tmp_path / "elsewhere.txt"
    other.write_text("x")
    assert c.local(other) == other
    missing = remote / "nope.parquet"
    assert c.local(missing) == missing


def test_big_files_copy_in_the_background(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "INLINE_MAX_BYTES", 10)
    remote, c = _cache(tmp_path)
    f = remote / "sessions" / "s1" / "proxy_720p.mp4"
    f.write_bytes(b"x" * 1000)
    assert c.local_if_ready(f) == f  # not cached yet: served from the share
    deadline = time.time() + 5
    while c.local_if_ready(f) == f and time.time() < deadline:
        time.sleep(0.02)
    local = c.local_if_ready(f)
    assert local != f and local.read_bytes() == f.read_bytes()
    assert c.local(f) == local


def test_trim_drops_least_recently_used(tmp_path):
    remote, c = _cache(tmp_path, max_bytes=250)
    files = []
    for k in range(3):
        f = remote / "sessions" / "s1" / f"{k}.bin"
        f.write_bytes(b"x" * 100)
        files.append(f)
    c.local(files[0])
    c.local(files[1])
    old = time.time_ns() - 10**10
    local0 = c.local(files[0])
    os.utime(local0, ns=(old, local0.stat().st_mtime_ns))  # 0 is the least recently used
    c.local(files[2])  # 300 bytes > 250: one goes
    left = {p.name for p, _ in c.files()}
    assert left == {"1.bin", "2.bin"}
    assert c.size_bytes() == 200


def test_read_table_goes_through_the_active_cache(tmp_path):
    remote, c = _cache(tmp_path)
    f = remote / "sessions" / "s1" / "t.parquet"
    t = pa.table({"a": [1, 2]})
    tables.write_table(t, f, t.schema)
    cache.activate(c)
    try:
        assert tables.read_table(f).column("a").to_pylist() == [1, 2]
        assert (tmp_path / "local" / "sessions" / "s1" / "t.parquet").exists()
    finally:
        cache.activate(None)
    assert cache.local(f) == f  # inactive (the worker): no copies


def test_network_paths():
    assert cache.is_network_path(Path("//nas/share/out"))
    assert cache.is_network_path(Path(r"\\nas\share\out"))
