"""CorrDiff output helpers: wind speed added, a curvilinear field put on a regular grid."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
WORKER = Path(__file__).parents[1] / "src/weather_mcp/workers/earth2/worker.py"
spec = importlib.util.spec_from_file_location("earth2_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(worker)


def test_wind_speed_is_added_from_its_components() -> None:
    lat, lon = np.array([20.0, 21.0]), np.array([120.0, 121.0])
    u, v = np.full((2, 2), 3.0), np.full((2, 2), 4.0)
    ds = worker._fields(["u10m", "v10m"], [u, v], lat, lon)
    assert float(ds["wind_speed"].max()) == pytest.approx(5.0)
    assert ds["wind_speed"].attrs["units"] == "m s-1"


def test_a_curvilinear_field_is_put_on_a_regular_grid() -> None:
    """A rotated grid (as CorrDiff's) keeps its values on a regular lat-lon grid."""
    j, i = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    lat = 20 + 0.02 * j + 0.003 * i
    lon = 120 + 0.02 * i - 0.003 * j
    field = lon - 120  # varies with longitude only
    ds = worker._fields(["speed"], [field], lat, lon)
    regular = worker._regular(ds, 0.05)
    assert regular["speed"].dims == ("lat", "lon")
    inside = regular["speed"].sel(lat=20.4, lon=120.4, method="nearest")
    assert float(inside) == pytest.approx(0.4, abs=0.03)
