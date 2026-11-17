"""Fetch what no maintainer ships keyless: ARCO ERA5 fields, ENS open data,
IBTrACS tracks. Station observations are in :mod:`.stations`.

ERA5 point series, deterministic open-data forecasts and CDS/Polytope
requests are left to the ECMWF skills and copernicus-mcp, used as shipped.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .common import Source, WeatherToolError
from .grids import select_box

ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_DOI = "https://doi.org/10.24381/cds.adbb2d47"
ERA5_LICENCE = (
    "Contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al. 2020); "
    "Copernicus licence / CC-BY 4.0; ARCO copy by Google Research (Carver et al. 2023)"
)
ECMWF_OPEN_LICENCE = "ECMWF open data, CC-BY 4.0: Copyright European Centre for Medium-Range Weather Forecasts (ECMWF)"
IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.{subset}.list.v04r01.csv"
IBTRACS_DOI = "https://doi.org/10.25921/82ty-9e16"


def era5_arco(
    *,
    variables: list[str],
    box: list[float],
    start: str,
    end: str,
    levels: list[int] | None,
    hours: list[int] | None,
    max_values: float,
) -> tuple[xr.Dataset, Source]:
    """Subset of ERA5 from the analysis-ready copy on Google Cloud (anonymous)."""
    try:
        ds = xr.open_zarr(ARCO_ERA5, storage_options={"token": "anon"}, chunks=None)
    except Exception as exc:
        raise WeatherToolError(
            "backend_unavailable", f"could not open ARCO ERA5: {exc}"
        ) from exc
    missing = [v for v in variables if v not in ds]
    if missing:
        raise WeatherToolError(
            "bad_input",
            f"ARCO ERA5 has no {', '.join(missing)} (use the long CF-style names, e.g. 2m_temperature)",
        )
    sub = ds[variables].sel(time=slice(start, end))
    if hours is not None:
        sub = sub.sel(time=sub.time.dt.hour.isin(hours))
    if levels is not None and "level" in sub.dims:
        sub = sub.sel(level=levels)
    pieces = {v: select_box(sub[v], box) for v in variables}
    sub = xr.Dataset(pieces)
    size = sum(
        float(np.prod([sub[v].sizes[d] for d in sub[v].dims])) for v in variables
    )
    if size > max_values:
        raise WeatherToolError(
            "too_large",
            f"the subset holds {size:.3g} values (limit {max_values:.3g}); narrow the box, period, levels or hours",
        )
    sub = sub.load()
    sub.attrs["arco_valid_time_stop_era5t"] = str(
        ds.attrs.get("valid_time_stop_era5t", "")
    )
    source = Source(
        name="ERA5 hourly (ARCO-ERA5, Google Cloud)",
        kind="product",
        reference=ERA5_DOI,
        version=ARCO_ERA5,
        subset={
            "variables": variables,
            "box": box,
            "start": start,
            "end": end,
            "levels": levels,
            "hours": hours,
        },
        licence=ERA5_LICENCE,
    )
    return sub, source


def empty_times(ds: xr.Dataset) -> list[str]:
    """Time steps with no value in any variable (gaps in the store, or past its end)."""
    if "time" not in ds.dims:
        return []
    filled = None
    for name in ds.data_vars:
        other = [d for d in ds[name].dims if d != "time"]
        has = ds[name].notnull().any(other) if other else ds[name].notnull()
        filled = has if filled is None else (filled | has)
    if filled is None:
        return []
    times = ds["time"].to_index()[~filled.to_numpy()]
    return [t.isoformat() for t in times]


def ensemble_open_data(
    *,
    param: str,
    steps: list[int],
    box: list[float] | None,
    point: list[float] | None,
    date: str | None,
    time: int | None,
    model: str,
) -> tuple[xr.Dataset, Source]:
    """ENS members (control + perturbed) from ECMWF open data, cut to a box or point."""
    from ecmwf.opendata import Client

    client = Client(source="ecmwf", model=model)
    request: dict[str, Any] = {
        "stream": "enfo",
        "type": ["cf", "pf"],
        "param": param,
        "step": steps,
    }
    if date:
        request["date"] = date
    if time is not None:
        request["time"] = time
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "ens.grib2"
        try:
            reply = client.retrieve(target=str(target), **request)
        except Exception as exc:
            raise WeatherToolError(
                "backend_unavailable", f"ECMWF open data: {exc}"
            ) from exc
        pieces = []
        kinds = []
        for kind in ("cf", "pf"):
            part = xr.open_dataset(
                target,
                engine="cfgrib",
                backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": kind}},
            )
            # A run may publish no control member for a parameter: the index has no
            # entry for it and this piece is empty. Its absence is reported in the
            # source name, not hidden.
            if not part.data_vars:
                continue
            if "number" not in part.dims:
                part = part.expand_dims("number")
            pieces.append(part)
            kinds.append({"cf": "control", "pf": "perturbed"}[kind])
        if not pieces:
            raise WeatherToolError("no_data", f"ECMWF open data has no {param} members")
        ds = xr.concat(pieces, dim="number").sortby("number")
        if point is not None:
            lon = point[0] % 360 if float(ds.longitude.max()) > 180 else point[0]
            ds = ds.sel(longitude=lon, latitude=point[1], method="nearest")
        elif box is not None:
            ds = xr.Dataset({v: select_box(ds[v], box) for v in ds.data_vars})
        ds = ds.load()
    run = pd.Timestamp(reply.datetime) if hasattr(reply, "datetime") else None
    source = Source(
        name=(
            f"ECMWF {model.upper()} ENS open data ({ds.sizes.get('number', 0)} members: "
            f"{' and '.join(kinds)})"
        ),
        kind="service",
        reference="https://www.ecmwf.int/en/forecasts/datasets/open-data",
        version=f"run {run.isoformat() if run is not None else 'latest'}",
        subset={"param": param, "steps": steps, "box": box, "point": point},
        licence=ECMWF_OPEN_LICENCE,
    )
    return ds, source


def _agency_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def ibtracs(
    subset: str, storm: str | None, season: int | None, name: str | None
) -> tuple[pd.DataFrame, Source]:
    """Best tracks from IBTrACS v04r01 (NOAA NCEI CSV)."""
    url = IBTRACS_URL.format(subset=subset)
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
    except Exception as exc:
        raise WeatherToolError(
            "backend_unavailable", f"IBTrACS download failed: {exc}"
        ) from exc
    frame = pd.read_csv(
        io.StringIO(response.text),
        skiprows=[1],
        low_memory=False,
        keep_default_na=False,
        na_values=[" ", ""],
    )
    if storm:
        frame = frame[frame["SID"] == storm]
    if season:
        frame = frame[frame["SEASON"].astype(int) == season]
    if name:
        frame = frame[frame["NAME"].str.upper() == name.upper()]
    if frame.empty:
        raise WeatherToolError(
            "no_data",
            f"IBTrACS {subset} has no storm matching {storm or name} {season or ''}",
        )
    out = pd.DataFrame(
        {
            "track": frame["SID"],
            "name": frame["NAME"],
            "time": pd.to_datetime(frame["ISO_TIME"], utc=True),
            "latitude": pd.to_numeric(frame["LAT"], errors="coerce"),
            "longitude": pd.to_numeric(frame["LON"], errors="coerce"),
        }
    )
    for kind, unit in (("PRES", "mslp_hpa"), ("WIND", "wind_kt")):
        wmo = _agency_column(frame, f"WMO_{kind}")
        usa = _agency_column(frame, f"USA_{kind}")
        out[f"wmo_{unit}"] = wmo
        out[f"usa_{unit}"] = usa
        # The merged column takes the WMO agency's value and, where it has none
        # (provisional tracks), JTWC's; the agency of each row is kept beside it.
        out[unit] = wmo.where(wmo.notna(), usa)
        out[f"{unit.split('_')[0]}_agency"] = np.select(
            [wmo.notna(), usa.notna()], ["wmo", "usa"], default=""
        )
    source = Source(
        name=f"IBTrACS v04r01 ({subset})",
        kind="product",
        reference=IBTRACS_DOI,
        version="v04r01",
        subset={"storm": storm, "season": season, "name": name},
        licence="NOAA NCEI public domain; cite Knapp et al. (2010) and doi:10.25921/82ty-9e16",
    )
    return out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True), source
