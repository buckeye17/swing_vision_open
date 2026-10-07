"""Display units: metric (m, cm, km/h) or imperial (ft, in, mph), per the Units setting.

Everything is computed and stored in SI units (metres, m/s; ball speeds in km/h); only
what the app shows changes. Court plots keep metres internally (their axes are hidden);
their hover text goes through :meth:`Units.len` like everything else. Exports stay metric.

Use :func:`current` once per layout or callback and pass the result down::

    u = units.current()
    f"{u.len_str(r['net_clearance_m'], 2, sign=True)} over the net"
    go.Scatter(y=u.speed(speeds_kmh), hovertemplate=f"%{{y:.0f}} {u.speed_unit}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

M_TO_FT = 3.280839895
M_TO_IN = 39.37007874
KMH_TO_MPH = 0.621371192
MPS_TO_KMH = 3.6

System = Literal["metric", "imperial"]


@dataclass(frozen=True)
class Units:
    system: System = "metric"

    @property
    def imperial(self) -> bool:
        return self.system == "imperial"

    # -- lengths: metres → m or ft ------------------------------------------------------
    @property
    def len_unit(self) -> str:
        return "ft" if self.imperial else "m"

    @property
    def len_factor(self) -> float:
        """Display length per metre."""
        return M_TO_FT if self.imperial else 1.0

    def len(self, m):
        """Metres (a number, array or None) in display length units."""
        return _scale(m, self.len_factor)

    def len_str(self, m: float | None, nd: int = 1, sign: bool = False) -> str:
        """``1.2 m`` / ``3.9 ft`` (``–`` for a missing value)."""
        return _fmt(self.len(m), nd, sign, self.len_unit)

    # -- small lengths: metres → cm or in -----------------------------------------------
    @property
    def small_unit(self) -> str:
        return "in" if self.imperial else "cm"

    @property
    def small_factor(self) -> float:
        return M_TO_IN if self.imperial else 100.0

    def small(self, m):
        return _scale(m, self.small_factor)

    def small_str(self, m: float | None, nd: int = 0, sign: bool = False) -> str:
        """``109 cm`` / ``43 in``."""
        return _fmt(self.small(m), nd, sign, self.small_unit)

    # -- speeds: km/h (ball) or m/s (people) → km/h or mph -------------------------------
    @property
    def speed_unit(self) -> str:
        return "mph" if self.imperial else "km/h"

    @property
    def speed_factor(self) -> float:
        """Display speed per km/h."""
        return KMH_TO_MPH if self.imperial else 1.0

    def speed(self, kmh):
        """km/h (a number, array or None) in display speed units."""
        return _scale(kmh, self.speed_factor)

    def speed_from_mps(self, mps):
        return _scale(mps, MPS_TO_KMH * self.speed_factor)

    def speed_str(self, kmh: float | None, nd: int = 0) -> str:
        """``139 km/h`` / ``86 mph``."""
        return _fmt(self.speed(kmh), nd, False, self.speed_unit)

    def speed_str_mps(self, mps: float | None, nd: int = 1) -> str:
        return _fmt(self.speed_from_mps(mps), nd, False, self.speed_unit)

    # -- fast body parts (wrist): m/s, or mph in imperial --------------------------------
    @property
    def limb_speed_unit(self) -> str:
        return "mph" if self.imperial else "m/s"

    def limb_speed(self, mps):
        return _scale(mps, MPS_TO_KMH * KMH_TO_MPH if self.imperial else 1.0)

    def limb_speed_str(self, mps: float | None, nd: int = 1) -> str:
        return _fmt(self.limb_speed(mps), nd, False, self.limb_speed_unit)

    # -- a person's height ----------------------------------------------------------------
    def height_str(self, m: float | None) -> str:
        """``180 cm`` / ``5′11″``."""
        if m is None:
            return "–"
        if not self.imperial:
            return f"{m * 100:.0f} cm"
        inches = round(m * M_TO_IN)
        return f"{inches // 12}′{inches % 12}″"

    @property
    def height_input_unit(self) -> str:
        return "in" if self.imperial else "cm"

    def height_to_input(self, m: float) -> int:
        return round(m * (M_TO_IN if self.imperial else 100))

    def height_from_input(self, value: float) -> float:
        """A height typed in :attr:`height_input_unit`, in metres."""
        return float(value) / (M_TO_IN if self.imperial else 100.0)

    # -- ball speed bands for practice breakdowns ------------------------------------------
    @property
    def speed_bands_kmh(self) -> tuple[float, ...]:
        """Band edges in km/h: 80/110/140 km/h, or 50/70/90 mph."""
        if self.imperial:
            return tuple(v / KMH_TO_MPH for v in (50, 70, 90))
        return (80.0, 110.0, 140.0)

    def js_config(self) -> dict:
        """For per-frame JS overlays (``assets/review_frame.js``)."""
        return {
            "lenFactor": self.len_factor,
            "lenUnit": self.len_unit,
            "speedFactor": self.speed_factor,
            "speedUnit": self.speed_unit,
        }


def _scale(v, factor: float):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v * factor
    if isinstance(v, (list, tuple)):
        return [None if x is None else x * factor for x in v]
    return v * factor


def _fmt(v: float | None, nd: int, sign: bool, unit: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:{'+' if sign else ''}.{nd}f} {unit}"


def current() -> Units:
    """The units the Settings page chose."""
    from swingvision.app import state

    return Units(state.settings().units)
