"""Display units (metric / imperial): conversions, formatting and the settings-backed pick."""

from __future__ import annotations

import numpy as np
import pytest

from swingvision.analysis import practice as pr
from swingvision.app import units
from swingvision.app.units import KMH_TO_MPH, M_TO_FT, Units

METRIC = Units("metric")
IMPERIAL = Units("imperial")


def test_lengths():
    assert METRIC.len(2.5) == 2.5 and METRIC.len_unit == "m" and METRIC.len_factor == 1.0
    assert IMPERIAL.len(1.0) == pytest.approx(3.2808, abs=1e-4) and IMPERIAL.len_unit == "ft"
    assert METRIC.len_str(2.74) == "2.7 m"
    assert IMPERIAL.len_str(2.74) == "9.0 ft"
    assert METRIC.len_str(0.25, 2, sign=True) == "+0.25 m"
    assert IMPERIAL.len_str(-1.0, 1, sign=True) == "-3.3 ft"
    assert METRIC.small_str(1.09) == "109 cm" and IMPERIAL.small_str(1.09) == "43 in"


def test_speeds():
    assert METRIC.speed_str(139) == "139 km/h" and IMPERIAL.speed_str(139) == "86 mph"
    assert METRIC.speed_unit == "km/h" and IMPERIAL.speed_unit == "mph"
    assert METRIC.speed_from_mps(5.0) == pytest.approx(18.0)
    assert IMPERIAL.speed_from_mps(5.0) == pytest.approx(18.0 * KMH_TO_MPH)
    assert METRIC.speed_str_mps(5.0) == "18.0 km/h" and IMPERIAL.speed_str_mps(5.0) == "11.2 mph"
    assert IMPERIAL.limb_speed_str(10.0) == "22.4 mph" and METRIC.limb_speed_str(10.0) == "10.0 m/s"


def test_none_lists_and_arrays():
    for u in (METRIC, IMPERIAL):
        assert u.len(None) is None and u.speed(None) is None and u.speed_from_mps(None) is None
        assert u.len_str(None) == "–" and u.speed_str(None) == "–" and u.height_str(None) == "–"
        assert u.len_str(float("nan")) == "–"
    assert IMPERIAL.speed([100.0, None, 50.0]) == pytest.approx([62.1371192, None, 31.0685596])
    assert IMPERIAL.len((1.0, None))[1] is None
    arr = IMPERIAL.len(np.array([0.0, 1.0, 2.0]))
    assert isinstance(arr, np.ndarray) and arr == pytest.approx([0.0, M_TO_FT, 2 * M_TO_FT])
    assert METRIC.speed_from_mps(np.array([1.0, np.nan]))[0] == pytest.approx(3.6)


def test_height():
    assert METRIC.height_str(1.80) == "180 cm"
    assert IMPERIAL.height_str(1.80) == "5′11″"
    assert IMPERIAL.height_str(1.83) == "6′0″"
    assert (METRIC.height_input_unit, IMPERIAL.height_input_unit) == ("cm", "in")
    assert METRIC.height_to_input(1.8) == 180 and IMPERIAL.height_to_input(1.8) == 71
    assert METRIC.height_from_input(180) == pytest.approx(1.8)
    assert IMPERIAL.height_from_input(71) == pytest.approx(1.8034, abs=1e-4)
    # Round trip through the input stays within half an inch.
    assert IMPERIAL.height_from_input(IMPERIAL.height_to_input(1.75)) == pytest.approx(
        1.75, abs=0.0127
    )


def test_speed_bands_and_js_config():
    assert METRIC.speed_bands_kmh == (80.0, 110.0, 140.0) == pr.SPEED_BANDS_KMH
    assert [v * KMH_TO_MPH for v in IMPERIAL.speed_bands_kmh] == pytest.approx([50, 70, 90])
    assert METRIC.js_config() == {
        "lenFactor": 1.0,
        "lenUnit": "m",
        "speedFactor": 1.0,
        "speedUnit": "km/h",
    }
    cfg = IMPERIAL.js_config()
    assert cfg["lenUnit"] == "ft" and cfg["speedUnit"] == "mph"
    assert cfg["lenFactor"] == pytest.approx(M_TO_FT)
    assert cfg["speedFactor"] == pytest.approx(KMH_TO_MPH)


def test_current_follows_the_setting(settings):
    from swingvision.settings import save_settings

    assert units.current() == METRIC
    settings.units = "imperial"
    save_settings(settings)
    assert units.current() == IMPERIAL


def test_speed_band_imperial():
    edges, f = IMPERIAL.speed_bands_kmh, IMPERIAL.speed_factor
    assert pr.speed_band(None, edges, f) is None
    assert pr.speed_band(70.0, edges, f) == "< 50"
    assert pr.speed_band(100.0, edges, f) == "50–70"
    assert pr.speed_band(130.0, edges, f) == "70–90"
    assert pr.speed_band(150.0, edges, f) == "≥ 90"
    # Metric defaults unchanged.
    assert pr.speed_band(100.0) == "80–110" and pr.speed_band(150.0) == "≥ 140"


def _row(speed_kmh, flags=None):
    return {
        "shot_kind": "groundstroke",
        "serve_side": None,
        "stroke_type": None,
        "speed_kmh": speed_kmh,
        "flags": flags or [],
        "excluded": False,
        "outcome": "in",
        "in_target": None,
        "rel_x": None,
        "rel_y": None,
        "target_dist_m": None,
        "feed_speed_kmh": None,
        "feed_land_x": None,
        "feed_land_y": None,
        "landing_confirmed": False,
        "t_contact": 0.0,
    }


def test_breakdown_imperial():
    rows = [_row(100.0), _row(105.0), _row(150.0), _row(60.0, ["speed_uncertain"])]
    u = IMPERIAL
    names = dict(pr.breakdown(rows, u.speed_bands_kmh, u.speed_factor, u.speed_unit))
    assert names["50–70 mph"]["n"] == 2
    assert names["≥ 90 mph"]["n"] == 1
    assert "< 50 mph" not in names  # uncertain speeds stay out of the bands
    assert not any("km/h" in k for k in names)
    metric = dict(pr.breakdown(rows))
    assert metric["80–110 km/h"]["n"] == 2 and "≥ 140 km/h" in metric


def test_settings_roi_beside_conversion(settings):
    from swingvision.app.main import create_app

    create_app()
    from swingvision.app.pages import settings_page as sp

    # Metric: exactly as typed.
    assert sp._roi_to_input(3.5, METRIC) == 3.5
    assert sp._roi_from_input(2.5, 3.5, METRIC) == 2.5
    assert sp._roi_from_input(None, 3.5, METRIC) == 3.5
    # Imperial: shown in feet, saved in metres; an untouched value doesn't drift.
    assert sp._roi_to_input(3.5, IMPERIAL) == 11.5
    assert sp._roi_from_input(11.5, 3.5, IMPERIAL) == 3.5
    assert sp._roi_from_input(10, 3.5, IMPERIAL) == pytest.approx(3.048)


def test_profile_height_bounds(settings):
    from swingvision.app.main import create_app

    create_app()
    from swingvision.app.pages import profiles

    assert profiles._height_bounds(METRIC) == (100, 250)
    assert profiles._height_bounds(IMPERIAL) == (40, 98)
