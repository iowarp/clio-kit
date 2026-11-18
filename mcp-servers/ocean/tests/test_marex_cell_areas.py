"""Heatwave areas are in km^2, not cells (marEx counts cells without cell areas)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

WORKER = Path(__file__).parents[1] / "src/ocean_mcp/workers/marex/worker.py"
spec = importlib.util.spec_from_file_location("marex_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(worker)


def test_a_quarter_degree_cell_at_the_equator_is_about_770_km2() -> None:
    lat = xr.DataArray(np.arange(-1, 1.01, 0.25), dims="lat", name="lat")
    lat = lat.assign_coords(lat=lat)
    lon = xr.DataArray(np.arange(0, 2.01, 0.25), dims="lon", name="lon")
    lon = lon.assign_coords(lon=lon)
    areas = worker.cell_areas_km2(lat, lon)
    assert areas.dims == ("lat", "lon")
    at_equator = float(areas.sel(lat=0.0, lon=1.0))
    assert abs(at_equator - 773.0) < 2  # (0.25 * 111.19 km)^2


def test_cells_shrink_with_latitude_and_the_sphere_sums_right() -> None:
    lat = xr.DataArray(np.arange(-89.5, 90, 1.0), dims="lat", name="lat")
    lat = lat.assign_coords(lat=lat)
    lon = xr.DataArray(np.arange(0.5, 360, 1.0), dims="lon", name="lon")
    lon = lon.assign_coords(lon=lon)
    areas = worker.cell_areas_km2(lat, lon)
    total = float(areas.sum())
    assert abs(total - 4 * np.pi * worker.EARTH_RADIUS_KM**2) / total < 1e-6
    assert float(areas.sel(lat=60.5).mean()) < float(areas.sel(lat=0.5).mean()) * 0.51


def test_descending_latitudes_give_the_same_areas() -> None:
    up = xr.DataArray(np.arange(30, 40.1, 0.25), dims="lat", name="lat")
    up = up.assign_coords(lat=up)
    down = up[::-1]
    lon = xr.DataArray(np.arange(0, 5, 0.25), dims="lon", name="lon")
    lon = lon.assign_coords(lon=lon)
    a = worker.cell_areas_km2(up, lon)
    b = worker.cell_areas_km2(down, lon)
    assert np.allclose(a.sel(lat=35.0), b.sel(lat=35.0))
