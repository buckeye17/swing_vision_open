"""Serve contact point statistics (PLAN.md §7.11, M7b): how serve speed and serve-in % change
with where the ball was struck relative to the front toe.

Works on the ``serves`` records of a :class:`~swingvision.analysis.stats.StatsData`, so the
same numbers come out for one session or any selection of sessions (M7a).

* **Effect charts**: per axis (forward, lateral, height), quantile bins of at least
  ``MIN_BIN`` serves: mean speed with a 95% interval and in % with a Wilson interval.
* **Grid**: forward × lateral cells of 10 cm with in % and the median speed; cells with
  fewer than ``MIN_CELL`` serves are left out.
* **Summary**: a sentence only where a bin's intervals separate from the rest's.
* **Regression** (details): speed as a quadratic in the three offsets (least squares), in as
  a logistic in them; effects per 10 cm at the average contact, with 95% intervals.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from swingvision.analysis.stats import CALLED, Z95, mean_ci, wilson

AXES = ("forward_m", "lateral_m", "height_m")
AXIS_LABELS = {
    "forward_m": "In front of the toe",
    "lateral_m": "To the racket side",
    "height_m": "Contact height",
}
#: Flags that make a serve's offsets unreliable: hidden unless the user asks for them.
SERIOUS_FLAGS = frozenset(
    {
        "far",
        "toss_not_tracked",
        "contact_above_frame",
        "contact_not_seen",
        "poor_toss_fit",
        "no_toe",
        "dark",
        "low_conf",
    }
)
MIN_BIN = 15
MIN_CELL = 8
CELL_M = 0.10


def select(
    serves: list[dict],
    serve_side: str | None = None,
    near_only: bool = True,
    hide_flagged: bool = True,
) -> list[dict]:
    """Serves with offsets, filtered as on the Stats page."""
    out = []
    for r in serves:
        if r.get("forward_m") is None or r.get("excluded"):
            continue
        if serve_side and r.get("serve_side") != serve_side:
            continue
        if near_only and r.get("side") != -1:
            continue
        if hide_flagged and SERIOUS_FLAGS & set(r.get("serve_flags") or []):
            continue
        out.append(r)
    return out


def _is_in(r: dict) -> bool | None:
    return None if r.get("outcome") not in CALLED else r["outcome"] == "in"


def _speed(r: dict) -> float | None:
    return r["speed_kmh"] if r.get("speed_ok") and r.get("speed_kmh") is not None else None


def group_stats(rows: list[dict]) -> dict:
    """n, mean speed with its 95% interval, in % with its Wilson interval."""
    speeds = [v for v in (_speed(r) for r in rows) if v is not None]
    calls = [v for v in (_is_in(r) for r in rows) if v is not None]
    lo, hi = mean_ci(speeds)
    k = sum(calls)
    plo, phi = wilson(k, len(calls))
    return {
        "n": len(rows),
        "n_speed": len(speeds),
        "speed_mean": float(np.mean(speeds)) if speeds else None,
        "speed_lo": lo,
        "speed_hi": hi,
        "n_called": len(calls),
        "in_pct": k / len(calls) if calls else None,
        "in_lo": plo,
        "in_hi": phi,
    }


def quantile_bins(rows: list[dict], axis: str, min_n: int = MIN_BIN) -> list[dict]:
    """As many equal-count bins of ``axis`` as hold ``min_n`` serves each (at most 6)."""
    rows = sorted((r for r in rows if r.get(axis) is not None), key=lambda r: r[axis])
    k = min(6, len(rows) // min_n)
    if k < 1:
        return []
    edges = np.linspace(0, len(rows), k + 1).round().astype(int)
    out = []
    for a, b in itertools.pairwise(edges):
        part = rows[a:b]
        v = [r[axis] for r in part]
        out.append({"lo": v[0], "hi": v[-1], "mid": float(np.median(v)), **group_stats(part)})
    return out


def grid(rows: list[dict], cell: float = CELL_M, min_n: int = MIN_CELL) -> list[dict]:
    """Forward × lateral cells with at least ``min_n`` serves: in % and the median speed."""
    cells: dict[tuple[int, int], list[dict]] = {}
    for r in rows:
        key = (math.floor(r["lateral_m"] / cell), math.floor(r["forward_m"] / cell))
        cells.setdefault(key, []).append(r)
    out = []
    for (i, j), part in sorted(cells.items()):
        if len(part) < min_n:
            continue
        g = group_stats(part)
        speeds = [v for v in (_speed(r) for r in part) if v is not None]
        out.append(
            {
                "lateral_lo": i * cell,
                "forward_lo": j * cell,
                "cell": cell,
                "n": g["n"],
                "in_pct": g["in_pct"],
                "n_called": g["n_called"],
                "speed_median": float(np.median(speeds)) if speeds else None,
            }
        )
    return out


def _diff_mean(a: list[float], b: list[float]) -> tuple[float, float, float] | None:
    if len(a) < 3 or len(b) < 3:
        return None
    d = float(np.mean(a) - np.mean(b))
    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return d, d - Z95 * se, d + Z95 * se


def _diff_rate(ka: int, na: int, kb: int, nb: int) -> tuple[float, float, float] | None:
    """Difference of two proportions with Newcombe's hybrid score interval."""
    if na < 3 or nb < 3:
        return None
    pa, pb = ka / na, kb / nb
    la, ua = wilson(ka, na)
    lb, ub = wilson(kb, nb)
    d = pa - pb
    lo = d - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2)
    hi = d + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2)
    return d, lo, hi


def _strength(d: tuple[float, float, float] | None) -> float:
    """How far a difference's 95% interval stays from 0, in half-widths (0: it includes 0)."""
    if d is None or not (d[1] > 0 or d[2] < 0):
        return 0.0
    return abs(d[0]) / max(1e-9, (d[2] - d[1]) / 2)


def findings(rows: list[dict], min_n: int = MIN_BIN) -> list[dict]:
    """Per axis, the contact bin whose speed or in % differs most clearly from the rest's
    (separated 95% intervals); none for an axis where no bin does."""
    out = []
    for axis in AXES:
        ordered = sorted((r for r in rows if r.get(axis) is not None), key=lambda r: r[axis])
        best, best_s = None, 0.0
        for b in quantile_bins(ordered, axis, min_n):
            inside = [r for r in ordered if b["lo"] <= r[axis] <= b["hi"]]
            rest = [r for r in ordered if not b["lo"] <= r[axis] <= b["hi"]]
            if len(rest) < min_n:
                continue
            sa = [v for v in (_speed(r) for r in inside) if v is not None]
            sb = [v for v in (_speed(r) for r in rest) if v is not None]
            ca = [v for v in (_is_in(r) for r in inside) if v is not None]
            cb = [v for v in (_is_in(r) for r in rest) if v is not None]
            sp = _diff_mean(sa, sb)
            ip = _diff_rate(sum(ca), len(ca), sum(cb), len(cb))
            strength = max(_strength(sp), _strength(ip))
            if strength > best_s:
                best_s = strength
                best = {
                    "axis": axis,
                    "lo": b["lo"],
                    "hi": b["hi"],
                    "n": len(inside),
                    "n_rest": len(rest),
                    "speed": sp if _strength(sp) else None,
                    "in": ip if _strength(ip) else None,
                }
        if best is not None:
            out.append(best)
    return out


def summary_text(found: dict, fmt_range, fmt_speed) -> str:
    """One plain sentence for a finding, e.g. "Contacts 20–40 cm in front of the toe: +6 km/h
    and +12 points of in % vs. the rest (n = 84 / 213)". ``fmt_range(lo, hi, axis)`` and
    ``fmt_speed(kmh)`` format in the user's units."""
    where = {
        "forward_m": "in front of the toe",
        "lateral_m": "to the racket side of the toe",
        "height_m": "high",
    }[found["axis"]]
    parts = []
    if found["speed"] is not None:
        parts.append(f"{fmt_speed(found['speed'][0])}")
    if found["in"] is not None:
        parts.append(f"{100 * found['in'][0]:+.0f} points of in %")
    return (
        f"Contacts {fmt_range(found['lo'], found['hi'], found['axis'])} {where}: "
        f"{' and '.join(parts)} vs. the rest (n = {found['n']} / {found['n'] + found['n_rest']})"
    )


# ---------------------------------------------------------------------------
# Regression (details panel)
# ---------------------------------------------------------------------------


def _design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[r[a] for a in AXES] for r in rows], dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    return np.column_stack([np.ones(len(X)), Xc, Xc**2]), mu


def _effects(beta: np.ndarray, cov: np.ndarray, scale: float = 1.0) -> dict[str, tuple]:
    """Slope per 10 cm at the average contact (the quadratic term is 0 there) with 95%."""
    out = {}
    for k, a in enumerate(AXES):
        b, s = beta[1 + k] * 0.1 * scale, math.sqrt(max(cov[1 + k, 1 + k], 0.0)) * 0.1 * scale
        out[a] = (float(b), float(b - Z95 * s), float(b + Z95 * s))
    return out


def regression(rows: list[dict]) -> dict:
    """Speed: least squares on a quadratic in the offsets. In: logistic (Newton), same
    terms. Effects per 10 cm at the average contact; ``None`` with too few serves."""
    out: dict = {"speed": None, "in": None}
    sp = [r for r in rows if _speed(r) is not None]
    if len(sp) >= 20:
        X, _ = _design(sp)
        y = np.array([_speed(r) for r in sp])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        dof = max(1, len(y) - X.shape[1])
        cov = np.linalg.pinv(X.T @ X) * float(res @ res) / dof
        out["speed"] = {
            "n": len(sp),
            "effects": _effects(beta, cov),
            "rmse": float(np.sqrt(res @ res / dof)),
        }
    cl = [r for r in rows if _is_in(r) is not None]
    ys = np.array([1.0 if _is_in(r) else 0.0 for r in cl])
    if len(cl) >= 30 and 0 < ys.sum() < len(ys):
        X, _ = _design(cl)
        beta = np.zeros(X.shape[1])
        ridge = np.eye(X.shape[1]) * 1e-3
        ridge[0, 0] = 0.0
        for _ in range(50):
            p = 1 / (1 + np.exp(-(X @ beta)))
            W = p * (1 - p)
            Hm = X.T @ (X * W[:, None]) + ridge
            step = np.linalg.solve(Hm, X.T @ (ys - p) - ridge @ beta)
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        p = 1 / (1 + np.exp(-(X @ beta)))
        cov = np.linalg.pinv(X.T @ (X * (p * (1 - p))[:, None]))
        # Marginal effect on P(in) at the average contact: slope × p(1 − p), in points.
        p0 = 1 / (1 + math.exp(-beta[0]))
        out["in"] = {
            "n": len(cl),
            "effects": _effects(beta, cov, 100 * p0 * (1 - p0)),
            "p_mean": p0,
        }
    return out


def overview(rows: list[dict]) -> dict:
    """Medians of the offsets and their σ, for the KPI line."""

    def med(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.median(v)) if v else None

    return {
        "n": len(rows),
        **{f"{a}_median": med(a) for a in AXES},
        "forward_sigma_median": med("forward_sigma_m"),
        "toe_moved_median": med("toe_moved_m"),
        "on_ground_pct": (
            sum(bool(r.get("toe_on_ground")) for r in rows)
            / max(1, sum(r.get("toe_on_ground") is not None for r in rows))
        )
        if rows
        else None,
        **group_stats(rows),
    }
