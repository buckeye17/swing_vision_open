"""Local read cache for an output folder on a network share (NAS).

Reading a session's data over SMB is 10-50x slower than from a local disk (the ball track:
~230 ms vs ~5 ms), and the app reads it on every page and many callbacks. The app process
activates a :class:`LocalCache` (see ``app.state``); readers in ``storage`` then go through
:func:`local`, which returns a local copy of a file under the output folder, refreshed when
the original's size or modification time differs (one ``stat`` over the network, ~5 ms).
The worker never activates it: it writes the originals, and the app picks the new versions
up on its next read.

Big files (the playback proxy) are copied in the background (:func:`local_if_ready`); until
the copy is complete they're served from the share. The cache stays under a size limit by
dropping the least recently used files.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from loguru import logger

#: Copies smaller than this are made inline by :func:`local`; bigger ones in the background.
INLINE_MAX_BYTES = 64 * 1024 * 1024
_MTIME_TOLERANCE_NS = 10_000  # filesystems keep 100 ns ticks; allow for rounding
#: How long a remote stat is trusted. A page reads the same files several times in a burst;
#: the app's own writes forget theirs at once (:func:`forget`), the worker's show up after this.
STAT_TTL_S = 2.0
_TMP_PREFIX = ".part-"


def is_network_path(path: Path) -> bool:
    """True for a UNC path or a mapped network drive (Windows), or a network mount (Linux)."""
    s = str(path)
    if s.startswith(("\\\\", "//")):
        return True
    if sys.platform == "win32":
        drive = os.path.splitdrive(os.path.abspath(s))[0]
        if not drive:
            return False
        DRIVE_REMOTE = 4
        return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == DRIVE_REMOTE
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return False
    target = os.path.realpath(s)
    best, fstype = "", ""
    for line in mounts:
        parts = line.split()
        if len(parts) < 3 or len(parts[1]) <= len(best):
            continue
        if target == parts[1] or target.startswith(parts[1].rstrip("/") + "/"):
            best, fstype = parts[1], parts[2]
    return fstype in {"cifs", "smb3", "smbfs", "nfs", "nfs4", "fuse.sshfs", "9p"}


class LocalCache:
    def __init__(self, remote_root: Path, cache_dir: Path, max_bytes: int):
        self.remote_root = Path(remote_root)
        self.dir = Path(cache_dir)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._path_locks: dict[Path, threading.Lock] = {}
        self._background: set[Path] = set()
        self._stats: dict[Path, tuple[float, os.stat_result]] = {}

    # -- lookups -------------------------------------------------------------------
    def _local_for(self, remote: Path) -> Path | None:
        try:
            rel = Path(remote).relative_to(self.remote_root)
        except ValueError:
            return None
        return self.dir / rel

    def _remote_stat(self, remote: Path) -> os.stat_result:
        now = time.monotonic()
        hit = self._stats.get(remote)
        if hit is not None and now - hit[0] < STAT_TTL_S:
            return hit[1]
        st = remote.stat()
        self._stats[remote] = (now, st)
        return st

    def forget(self, remote: Path) -> None:
        self._stats.pop(Path(remote), None)

    @staticmethod
    def _fresh(local: Path, st: os.stat_result) -> bool:
        try:
            ls = local.stat()
        except OSError:
            return False
        return (
            ls.st_size == st.st_size and abs(ls.st_mtime_ns - st.st_mtime_ns) < _MTIME_TOLERANCE_NS
        )

    @staticmethod
    def _touch(local: Path, st: os.stat_result) -> None:
        """Record a use (access time) while keeping the original's modification time."""
        with contextlib.suppress(OSError):
            os.utime(local, ns=(time.time_ns(), st.st_mtime_ns))

    def _path_lock(self, local: Path) -> threading.Lock:
        with self._lock:
            return self._path_locks.setdefault(local, threading.Lock())

    # -- copying -------------------------------------------------------------------
    def _copy(self, remote: Path, local: Path, st: os.stat_result) -> bool:
        with self._path_lock(local):
            if self._fresh(local, st):
                return True
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_name(f"{_TMP_PREFIX}{uuid.uuid4().hex[:8]}-{local.name}")
            try:
                shutil.copyfile(remote, tmp)
                # The original changed while we copied: don't keep a torn copy.
                after = remote.stat()
                if after.st_size != st.st_size or after.st_mtime_ns != st.st_mtime_ns:
                    tmp.unlink(missing_ok=True)
                    return False
                os.utime(tmp, ns=(time.time_ns(), st.st_mtime_ns))
                os.replace(tmp, local)
            except OSError as exc:  # a file being served can't be replaced on Windows
                logger.debug("Cache copy of {} failed: {}", remote, exc)
                tmp.unlink(missing_ok=True)
                return False
        self.trim()
        return True

    def local(self, remote: Path) -> Path:
        """A fresh local copy of ``remote`` (copied now if needed), or ``remote`` itself."""
        local = self._local_for(remote)
        if local is None:
            return remote
        try:
            st = self._remote_stat(Path(remote))
        except OSError:
            return remote  # let the caller raise its usual error
        if self._fresh(local, st):
            self._touch(local, st)
            return local
        if st.st_size > INLINE_MAX_BYTES:
            self.prefetch(remote)
            return remote
        return local if self._copy(Path(remote), local, st) else remote

    def local_if_ready(self, remote: Path) -> Path:
        """The local copy if it's complete and fresh; otherwise start copying it in the
        background and return ``remote`` (for big files that are streamed)."""
        local = self._local_for(remote)
        if local is None:
            return remote
        try:
            st = self._remote_stat(Path(remote))
        except OSError:
            return remote
        if self._fresh(local, st):
            self._touch(local, st)
            return local
        self.prefetch(remote)
        return remote

    def prefetch(self, remote: Path) -> None:
        """Copy ``remote`` into the cache on a background thread (once at a time)."""
        remote = Path(remote)
        local = self._local_for(remote)
        if local is None:
            return
        with self._lock:
            if remote in self._background:
                return
            self._background.add(remote)

        def run():
            try:
                st = remote.stat()
                if st.st_size > self.max_bytes // 2:
                    return  # would evict most of the cache
                if not self._fresh(local, st):
                    t0 = time.perf_counter()
                    if self._copy(remote, local, st):
                        logger.info(
                            "Cached {} locally ({:.0f} MB, {:.0f} s)",
                            remote.name,
                            st.st_size / 1e6,
                            time.perf_counter() - t0,
                        )
            except OSError as exc:
                logger.debug("Prefetch of {} failed: {}", remote, exc)
            finally:
                with self._lock:
                    self._background.discard(remote)

        threading.Thread(target=run, name=f"cache:{remote.name}", daemon=True).start()

    # -- size ----------------------------------------------------------------------
    def files(self) -> list[tuple[Path, os.stat_result]]:
        out = []
        for dirpath, _, names in os.walk(self.dir):
            for name in names:
                p = Path(dirpath) / name
                with contextlib.suppress(OSError):
                    out.append((p, p.stat()))
        return out

    def size_bytes(self) -> int:
        return sum(st.st_size for _, st in self.files())

    def trim(self) -> None:
        """Drop the least recently used copies until the cache fits its limit."""
        with self._lock:
            files = [(p, st) for p, st in self.files() if not p.name.startswith(_TMP_PREFIX)]
            total = sum(st.st_size for _, st in files)
            for p, st in sorted(files, key=lambda f: f[1].st_atime_ns):
                if total <= self.max_bytes:
                    break
                try:
                    p.unlink()
                    total -= st.st_size
                except OSError:  # in use (being served); try the next one
                    pass

    def clear(self) -> None:
        with self._lock:
            for p, _ in self.files():
                with contextlib.suppress(OSError):
                    p.unlink()


_active: LocalCache | None = None


def activate(cache: LocalCache | None) -> None:
    global _active
    _active = cache


def active() -> LocalCache | None:
    return _active


def local(path: Path) -> Path:
    """Where to read ``path`` from: a local copy when a cache is active, else ``path``."""
    return _active.local(path) if _active is not None else path


def forget(path: Path) -> None:
    """``path`` was just written: check it again on the next read."""
    if _active is not None:
        _active.forget(path)


def local_if_ready(path: Path) -> Path:
    return _active.local_if_ready(path) if _active is not None else path
