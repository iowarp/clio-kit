# SPDX-License-Identifier: AGPL-3.0-or-later
"""marEx worker: detect and track marine extremes in a gridded daily field.

Runs in its own environment (see pyproject.toml beside this file); the ocean
MCP server starts it as ``python worker.py request.json`` and reads the JSON
it prints. It writes:

- ``<stem>.events.nc``: the event ID field for the kept events, with per-event
  start, end and daily area and centroid;
- ``<stem>.event_series.csv``: per event and day, the area-weighted mean
  anomaly and the mean threshold (both relative to marEx's climatology), the
  peak anomaly, the area and the centroid. The server computes the Hobday
  metrics and categories from this table.

This file is licensed AGPL-3.0-or-later because it imports marEx.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: Mean Earth radius (IUGG), km.
EARTH_RADIUS_KM = 6371.0088


def cell_areas_km2(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """Area of each cell of a regular latitude-longitude grid, km^2, on a sphere.

    marEx reports areas in cells unless it is given cell areas, so an event's area is
    summed here from its cells instead. Spacing comes from the coordinates; a cell is
    bounded halfway to its neighbours and clipped at the poles.
    """
    lat_deg = np.asarray(lat, dtype=float)
    lon_deg = np.asarray(lon, dtype=float)
    dlon = (
        np.deg2rad(float(np.median(np.abs(np.diff(lon_deg)))))
        if lon_deg.size > 1
        else 0.0
    )
    order = np.argsort(lat_deg)
    sorted_lat = lat_deg[order]
    edges = np.empty(sorted_lat.size + 1)
    edges[1:-1] = (sorted_lat[1:] + sorted_lat[:-1]) / 2
    step = np.median(np.diff(sorted_lat)) if sorted_lat.size > 1 else 0.0
    edges[0], edges[-1] = sorted_lat[0] - step / 2, sorted_lat[-1] + step / 2
    edges = np.clip(edges, -90.0, 90.0)
    band = np.abs(np.diff(np.sin(np.deg2rad(edges))))
    per_lat = np.empty_like(band)
    per_lat[order] = EARTH_RADIUS_KM**2 * dlon * band
    row = xr.DataArray(per_lat, coords={lat.name: lat}, dims=[lat.name])
    return row * xr.ones_like(lon, dtype=float)


def main(request_path: str) -> int:
    request = json.loads(Path(request_path).read_text())
    import marEx
    from dask.distributed import Client

    marEx.set_quiet_mode()
    client = Client(
        processes=False, n_workers=1, threads_per_worker=int(request.get("threads", 4))
    )
    try:
        dims = request["dimensions"]  # {"time": ..., "x": ..., "y": ...}
        with xr.open_dataset(request["input"]) as ds:
            da = ds[request["variable"]]
            if request.get("box"):
                x0, x1, y0, y1 = request["box"]
                da = da.sel({dims["x"]: slice(x0, x1)})
                ylat = da[dims["y"]]
                da = da.sel(
                    {dims["y"]: slice(y0, y1) if ylat[0] < ylat[-1] else slice(y1, y0)}
                )
            da = da.load().chunk({dims["time"]: -1, dims["y"]: 50, dims["x"]: 50})
        pre = marEx.preprocess_data(
            da,
            method_anomaly=request["method_anomaly"],
            method_extreme=request["method_extreme"],
            threshold_percentile=float(request["percentile"]),
            reference_period=tuple(request["baseline"])
            if request.get("baseline")
            else None,
            window_years=int(request.get("window_years", 15)),
            window_days=int(request.get("window_days", 11)),
            smooth_days=int(request.get("smooth_days", 21)),
            dimensions=dims,
            coordinates=dims,
            quiet=True,
        )
        pre = pre.persist()
        tracker = marEx.regional_tracker(
            pre.extreme_events,
            pre.mask,
            R_fill=int(request.get("R_fill", 2)),
            T_fill=int(request.get("T_fill", 2)),
            area_filter_quartile=float(request.get("area_filter_quartile", 0.5)),
            dimensions=dims,
            coordinates=dims,
            coordinate_units="degrees",
            quiet=True,
            temp_dir=tempfile.mkdtemp(prefix="marex-"),
        )
        events = tracker.run().load()
        anomaly = pre.dat_anomaly.load()
        thresholds = pre.thresholds.load()
    finally:
        client.close()

    time_dim, x_dim, y_dim = dims["time"], dims["x"], dims["y"]
    starts = pd.to_datetime(events["time_start"].to_numpy())
    ends = pd.to_datetime(events["time_end"].to_numpy())
    duration = (ends - starts).days + 1
    keep = events["ID"].to_numpy()[duration >= int(request.get("min_duration_days", 5))]
    weights = np.cos(np.deg2rad(anomaly[y_dim]))
    areas = cell_areas_km2(anomaly[y_dim], anomaly[x_dim])
    doy_index = anomaly[time_dim].dt.dayofyear
    threshold_daily = thresholds.sel(dayofyear=doy_index)
    id_field = events["ID_field"]
    rows = []
    for event_id in keep:
        present = events["presence"].sel(ID=event_id).to_numpy()
        for t_index in np.nonzero(present)[0]:
            cells = id_field.isel({time_dim: t_index}) == event_id
            if not bool(cells.any()):
                continue
            w = weights.where(cells, 0)
            a = anomaly.isel({time_dim: t_index})
            th = threshold_daily.isel({time_dim: t_index})
            rows.append(
                {
                    "event": int(event_id),
                    "time": pd.Timestamp(anomaly[time_dim].values[t_index]).isoformat(),
                    "anomaly_mean": float((a * w).sum() / w.sum()),
                    "threshold_mean": float((th * w).sum() / w.sum()),
                    "anomaly_max": float(a.where(cells).max()),
                    "area_km2": float(areas.where(cells, 0.0).sum()),
                    "centroid_lat": float(
                        events["centroid"]
                        .isel(component=0, **{time_dim: t_index})
                        .sel(ID=event_id)
                    ),
                    "centroid_lon": float(
                        events["centroid"]
                        .isel(component=1, **{time_dim: t_index})
                        .sel(ID=event_id)
                    ),
                    "cells": int(cells.sum()),
                }
            )
    out_dir = Path(request["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = request["stem"]
    series_path = out_dir / f"{stem}.event_series.csv"
    pd.DataFrame(rows).to_csv(series_path, index=False)
    kept = events[
        ["ID_field", "time_start", "time_end", "area", "centroid", "presence"]
    ].sel(ID=keep)
    kept["ID_field"] = id_field.where(id_field.isin(keep), 0)
    kept = kept.drop_vars(
        [
            v
            for v in kept.coords
            if v not in kept.dims and v not in (time_dim, x_dim, y_dim)
        ],
        errors="ignore",
    )
    kept.attrs = {
        k: (str(v) if not isinstance(v, int | float) else v)
        for k, v in events.attrs.items()
    }
    kept.attrs.update(
        {
            "marEx_version": marEx.__version__,
            "min_duration_days": int(request.get("min_duration_days", 5)),
        }
    )
    events_path = out_dir / f"{stem}.events.nc"
    kept.to_netcdf(events_path)
    print(
        json.dumps(
            {
                "status": "success",
                "events_path": str(events_path),
                "series_path": str(series_path),
                "events_tracked": int(events.sizes["ID"]),
                "events_kept": len(keep),
                "marex_version": marEx.__version__,
                "preprocess_attrs": {k: str(v) for k, v in pre.attrs.items()},
                "tracker_attrs": {k: str(v) for k, v in events.attrs.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
