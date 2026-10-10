"""Multi-session statistics (PLAN.md §7.13): the same statistics as one session, over any
selection of sessions.

A :class:`SessionFilter` (recording dates, mode, practice type, profile, tags, explicit
include / exclude, quality options) is resolved against ``library.sqlite``. The selected
sessions' ``stats_records.parquet`` files (written by the ``stats`` stage) are read with one
``pyarrow.dataset`` into the :class:`~swingvision.analysis.stats.StatsData` a single session
uses, so every summary in :mod:`swingvision.analysis.stats` runs on it unchanged.

Results are cached in memory by (filter, the selected sessions' records files and library
rows), so a reprocessed session invalidates only the selections it's in.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from urllib.parse import parse_qsl, urlencode

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from swingvision.analysis import stats as st
from swingvision.storage import cache
from swingvision.storage.library import Library
from swingvision.storage.schemas import PRACTICE_SUBMODE_LABELS, STATS_RECORDS
from swingvision.storage.session import Session

MODES = ("practice", "match")
PRACTICE_TYPES = tuple(PRACTICE_SUBMODE_LABELS)

#: URL query key per filter field (lists are comma-separated).
QUERY_KEYS = {
    "date_from": "from",
    "date_to": "to",
    "modes": "mode",
    "practice_types": "type",
    "profiles": "profile",
    "tags": "tag",
    "include": "sessions",
    "exclude": "exclude",
    "user_calibration": "cal",
    "calibrated_speeds": "speeds",
}
_FLAG_VALUES = {"user_calibration": "user", "calibrated_speeds": "calibrated"}


@dataclass(frozen=True)
class SessionFilter:
    """Which sessions a statistic covers. Empty fields don't filter.

    Dates are ``YYYY-MM-DD`` (inclusive, local recording date); a session needs *all* of
    ``tags``; ``include`` limits the selection to those sessions, ``exclude`` drops some.
    """

    date_from: str | None = None
    date_to: str | None = None
    modes: tuple[str, ...] = ()
    practice_types: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    #: Only sessions whose court calibration the user confirmed (not auto-accepted).
    user_calibration: bool = False
    #: Only sessions with calibrated speeds (speed calibration arrives with M7c).
    calibrated_speeds: bool = False

    @classmethod
    def make(cls, **kw) -> SessionFilter:
        """Build from loose values (lists, comma strings, ``None``)."""
        out = {}
        for f in fields(cls):
            if f.name not in kw:
                continue
            v = kw[f.name]
            if f.type.startswith("tuple"):
                if isinstance(v, str):
                    v = v.split(",")
                v = tuple(x.strip() for x in (v or ()) if x and str(x).strip())
            elif f.type == "bool":
                v = bool(v)
            else:
                v = (str(v).strip()[:10] or None) if v else None
            out[f.name] = v
        return cls(**out)

    def to_query(self) -> str:
        """The URL query string (no ``?``) that :meth:`from_query` reads back."""
        pairs = []
        for f in fields(self):
            v = getattr(self, f.name)
            if not v:
                continue
            if isinstance(v, bool):
                v = _FLAG_VALUES[f.name]
            elif isinstance(v, tuple):
                v = ",".join(v)
            pairs.append((QUERY_KEYS[f.name], v))
        return urlencode(pairs)

    @classmethod
    def from_query(cls, query: str | dict | None) -> SessionFilter:
        if not query:
            return cls()
        if isinstance(query, str):
            query = dict(parse_qsl(query.lstrip("?")))
        kw = {}
        for name, key in QUERY_KEYS.items():
            if key in query and query[key] not in (None, ""):
                v = query[key]
                kw[name] = (v == _FLAG_VALUES[name]) if name in _FLAG_VALUES else v
        return cls.make(**kw)

    def is_empty(self) -> bool:
        return not any(getattr(self, f.name) for f in fields(self))


def _date(row: dict) -> str:
    return (row.get("recorded_on") or row.get("created_at") or "")[:10]


def resolve(library: Library, flt: SessionFilter) -> list[dict]:
    """The library rows (plus ``tags``) the filter selects, in recording order. Quality
    options are applied later, on the records (they're facts of the processed data)."""
    tags = library.tags_by_session()
    include, exclude = set(flt.include), set(flt.exclude)
    want_tags = {t.lower() for t in flt.tags}
    out = []
    for row in library.list_sessions():
        row = {**row, "tags": tags.get(row["id"], [])}
        sid, day = row["id"], _date(row)
        if (include and sid not in include) or sid in exclude:
            continue
        if flt.date_from and day < flt.date_from:
            continue
        if flt.date_to and day > flt.date_to:
            continue
        if flt.modes and row["mode"] not in flt.modes:
            continue
        if flt.practice_types and row["submode"] not in flt.practice_types:
            continue
        if flt.profiles and row.get("profile_id") not in flt.profiles:
            continue
        if want_tags and not want_tags <= {t.lower() for t in row["tags"]}:
            continue
        out.append(row)
    return sorted(out, key=lambda r: (r.get("recorded_on") or r["created_at"] or "", r["id"]))


@dataclass
class Selection:
    """A resolved filter: the statistics, and which practice sessions matched but have no
    records yet (not processed through ``stats`` with this version)."""

    filter: SessionFilter
    data: st.StatsData
    rows: list[dict]
    missing: list[dict] = field(default_factory=list)
    #: Matched but left out by a quality option.
    filtered_out: list[dict] = field(default_factory=list)

    def counts(self) -> dict:
        recs = [r for r in self.data.records if not r["excluded"]]
        return {
            "sessions": len(self.data.sessions),
            "shots": len(recs),
            "serves": sum(r["group"] == "serve" for r in recs),
            "swings": sum(s["stroke_type"] in st.GROUPS for s in self.data.swings),
            "serve_contacts": sum(s["forward_m"] is not None for s in self.data.serves),
            "missing": len(self.missing),
            "filtered_out": len(self.filtered_out),
        }


_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()
CACHE_SIZE = 16


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _stamp(path) -> tuple | None:
    try:
        s = path.stat()
    except OSError:
        return None
    return (s.st_mtime_ns, s.st_size)


def _info(row: dict, rec: dict) -> dict:
    """A session's description: names and dates from the library, processing facts from
    its records."""
    return {
        "session_id": row["id"],
        "name": row["name"],
        "recorded_on": row.get("recorded_on"),
        "duration_s": row.get("duration_s") or 0.0,
        "mode": row["mode"],
        "practice_type": row["submode"],
        "profile_id": row.get("profile_id"),
        "device_key": rec.get("device_key"),
        "calibration_by": rec.get("calibration_by"),
        "speeds_calibrated": bool(rec.get("speeds_calibrated")),
        "tags": list(row["tags"]),
    }


def _rows(table: pa.Table, kind: str, cols: tuple[str, ...]) -> list[dict]:
    return table.filter(pc.equal(table.column("kind"), kind)).select(list(cols)).to_pylist()


def select(library: Library, flt: SessionFilter, include_excluded: bool = False) -> Selection:
    """Resolve ``flt`` and read the selected sessions' records (cached)."""
    rows = resolve(library, flt)
    paths, stamps, missing = {}, [], []
    for row in rows:
        path = Session.open(library.root, row["dir_name"]).stats_records_path
        stamp = _stamp(path)
        if stamp is None:
            if row["mode"] == "practice":  # match statistics arrive with Phase 2
                missing.append(row)
            continue
        paths[row["id"]] = path
        stamps.append((row["id"], stamp, row["name"], row.get("recorded_on"), tuple(row["tags"])))
    key = (str(library.root), flt, include_excluded, tuple(stamps), tuple(r["id"] for r in missing))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    selection = _read(rows, paths, missing, flt, include_excluded)
    with _cache_lock:
        _cache[key] = selection
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return selection


def _read(rows, paths, missing, flt: SessionFilter, include_excluded: bool) -> Selection:
    if paths:
        files = [str(cache.local(p)) for p in paths.values()]
        table = ds.dataset(files, format="parquet", schema=STATS_RECORDS).to_table()
    else:
        table = STATS_RECORDS.empty_table()
    sess_cols = table.select(list(st.SESSION_FIELDS))
    first: dict[str, dict] = {}
    for r in sess_cols.to_pylist():
        first.setdefault(r["session_id"], r)
    keep, filtered_out = [], []
    for row in rows:
        if row["id"] not in paths:
            continue
        rec = first.get(row["id"], {})
        if (flt.user_calibration and rec.get("calibration_by") != "user") or (
            flt.calibrated_speeds and not rec.get("speeds_calibrated")
        ):
            filtered_out.append(row)
            continue
        keep.append((row, rec))
    order = {row["id"]: i for i, (row, _rec) in enumerate(keep)}
    table = table.filter(pc.is_in(table.column("session_id"), pa.array(list(order), pa.string())))

    def ordered(recs: list[dict]) -> list[dict]:
        return sorted(recs, key=lambda r: (order[r["session_id"]], r["t"] or 0.0))

    base = ("session_id", "t", "side")
    shots = _rows(table, "shot", (*base, *st.SHOT_FIELDS))
    if not include_excluded:
        shots = [r for r in shots if not r["excluded"]]
    swings = _rows(table, "swing", (*base, *st.SWING_FIELDS))
    serves = _rows(table, "serve", (*base, *st.SERVE_FIELDS))
    if not include_excluded:
        serves = [r for r in serves if not r["excluded"]]
    movement = _rows(table, "movement", ("session_id", *st.MOVEMENT_FIELDS))
    sessions = [_info(row, rec) for row, rec in keep]
    data = st.StatsData(
        records=ordered(shots),
        swings=ordered(swings),
        serves=ordered(serves),
        movement=sorted(movement, key=lambda m: order[m["session_id"]]),
        sessions=sessions,
        is_practice=any(s["mode"] == "practice" for s in sessions),
        duration_s=sum(s["duration_s"] for s in sessions),
    )
    if missing:
        names = ", ".join(f"“{r['name']}”" for r in missing[:3])
        more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
        data.notes.append(
            f"{len(missing)} matching session{'s have' if len(missing) > 1 else ' has'} no "
            f"statistics yet ({names}{more}): process {'them' if len(missing) > 1 else 'it'} "
            "to include them."
        )
    return Selection(flt, data, [r for r, _ in keep], missing, filtered_out)
