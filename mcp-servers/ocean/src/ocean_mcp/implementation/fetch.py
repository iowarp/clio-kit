"""Fetch observations from public services into profile tables.

Argo through argopy (Ifremer ERDDAP or the GDAC), gliders from the IOOS
Glider DAC over ERDDAP, SWOT swaths through earthaccess. Each fetch states
its service, query and QC mode so the provenance record can repeat it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .common import OceanToolError, Source, cache_dir

ARGO_DOI = "https://doi.org/10.17882/42182"
ARGO_LICENCE = "Argo data are freely available without restriction (Argo data policy); cite doi:10.17882/42182"
GLIDER_ERDDAP = "https://gliders.ioos.us/erddap"


def argo_fetch(
    *,
    box: list[float] | None,
    floats: list[int] | None,
    dataset: Literal["phy", "bgc"],
    mode: Literal["standard", "research", "expert"],
    source: Literal["erddap", "gdac", "argovis"],
    params: list[str] | None,
) -> tuple[pd.DataFrame, Source]:
    """Argo profiles as a profile table.

    ``box`` is ``[lon_min, lon_max, lat_min, lat_max, p_min, p_max, start, end]``
    (dates as YYYY-MM-DD). ``mode`` is argopy's user mode: ``standard`` keeps
    only good (QC 1/2) adjusted data, ``research`` delayed-mode high quality.
    """
    import argopy

    kwargs: dict[str, Any] = {
        "src": source,
        "mode": mode,
        "ds": dataset,
        "progress": False,
    }
    if dataset == "bgc":
        kwargs["params"] = params or "all"
    # argopy caches under ~/.cache by default, which a sandboxed server may not be
    # allowed to write; keep it beside the server's outputs instead.
    argopy.set_options(cachedir=str(cache_dir("argopy")))
    try:
        fetcher = argopy.DataFetcher(**kwargs)
        if box is not None:
            if len(box) != 8:
                raise OceanToolError(
                    "bad_input",
                    "box is [lon_min, lon_max, lat_min, lat_max, p_min, p_max, start, end]",
                )
            fetcher = fetcher.region(box)
            subset: dict[str, Any] = {"box": box}
        elif floats:
            fetcher = fetcher.float(floats)
            subset = {"floats": floats}
        else:
            raise OceanToolError(
                "bad_input", "give a box or a list of float WMO numbers"
            )
        points = fetcher.to_dataframe()
    except OceanToolError:
        raise
    except Exception as exc:
        message = str(exc)
        if "No data" in message or "DataNotFound" in type(exc).__name__:
            raise OceanToolError(
                "no_data", f"Argo returned no profiles for {kwargs} and the query"
            ) from exc
        raise OceanToolError(
            "backend_unavailable", f"argopy ({source}) failed: {message[:300]}"
        ) from exc
    if points.empty:
        raise OceanToolError("no_data", "Argo returned no profiles for that query")
    rename = {
        "TIME": "time",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "PRES": "pressure",
        "TEMP": "temperature",
        "PSAL": "salinity",
        "DOXY": "oxygen",
        "CHLA": "chlorophyll",
        "NITRATE": "nitrate",
    }
    table = points.reset_index(drop=True).rename(columns=rename)
    table["profile"] = (
        table["PLATFORM_NUMBER"].astype(str)
        + "_"
        + table["CYCLE_NUMBER"].astype(int).astype(str).str.zfill(3)
    )
    keep = ["profile", "time", "latitude", "longitude", "pressure"] + [
        c
        for c in ("temperature", "salinity", "oxygen", "chlorophyll", "nitrate")
        if c in table
    ]
    table = table[keep].sort_values(["profile", "pressure"]).reset_index(drop=True)
    src = Source(
        name=f"Argo {dataset} ({mode} mode) via argopy {argopy.__version__} / {source}",
        kind="service",
        reference=ARGO_DOI,
        version=pd.Timestamp.now(tz="UTC").strftime("snapshot fetched %Y-%m-%d"),
        subset=subset,
        licence=ARGO_LICENCE,
    )
    return table, src


# --------------------------------------------------------------------- gliders


def glider_search(box: list[float], start: str, end: str) -> list[dict[str, Any]]:
    """Glider deployments overlapping a box and period (IOOS Glider DAC)."""
    from erddapy import ERDDAP

    e = ERDDAP(server=GLIDER_ERDDAP, protocol="tabledap")
    url = e.get_search_url(
        search_for="glider",
        response="csv",
        min_lon=box[0],
        max_lon=box[1],
        min_lat=box[2],
        max_lat=box[3],
        min_time=start,
        max_time=end,
    )
    try:
        found = pd.read_csv(url)
    except Exception as exc:
        if "404" in str(exc):
            return []
        raise OceanToolError(
            "backend_unavailable", f"Glider DAC search failed: {exc}"
        ) from exc
    return [
        {"dataset_id": row["Dataset ID"], "title": row["Title"]}
        for _, row in found.iterrows()
        if row["Dataset ID"] != "allDatasets"
    ]


def glider_fetch(dataset_id: str) -> tuple[pd.DataFrame, Source]:
    """One deployment's profiles (time, position, depth, T, S and QC)."""
    from erddapy import ERDDAP

    e = ERDDAP(server=GLIDER_ERDDAP, protocol="tabledap", response="csv")
    e.dataset_id = dataset_id
    wanted = [
        "profile_id",
        "time",
        "latitude",
        "longitude",
        "depth",
        "pressure",
        "temperature",
        "salinity",
        "temperature_qc",
        "salinity_qc",
    ]
    try:
        info = pd.read_csv(e.get_info_url(dataset_id=dataset_id, response="csv"))
        available = set(info.loc[info["Row Type"] == "variable", "Variable Name"])
        e.variables = [v for v in wanted if v in available]
        frame = e.to_pandas(parse_dates=True)
    except Exception as exc:
        raise OceanToolError(
            "backend_unavailable", f"Glider DAC fetch of {dataset_id} failed: {exc}"
        ) from exc
    frame.columns = [c.split(" (")[0] for c in frame.columns]
    frame = frame.rename(columns={"profile_id": "profile", "time": "time"})
    if "pressure" not in frame and "depth" in frame:
        import gsw

        frame["pressure"] = gsw.p_from_z(
            -frame["depth"].to_numpy(), frame["latitude"].to_numpy()
        )
    frame["profile"] = dataset_id + "-" + frame["profile"].astype(str)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    src = Source(
        name=f"IOOS Glider DAC deployment {dataset_id}",
        kind="service",
        reference=f"{GLIDER_ERDDAP}/tabledap/{dataset_id}.html",
        version=pd.Timestamp.now(tz="UTC").strftime("fetched %Y-%m-%d"),
        subset={"dataset_id": dataset_id, "variables": list(frame.columns)},
        licence="IOOS Glider DAC data are public (U.S. IOOS data policy); credit the deploying institution",
    )
    return frame.sort_values(["profile", "pressure"]).reset_index(drop=True), src


def split_transects(
    table: pd.DataFrame, turn_degrees: float, min_profiles: int
) -> pd.DataFrame:
    """Assign each profile a transect number by heading reversals.

    The heading between consecutive profile positions is smoothed over three
    profiles; a new transect starts where it turns by more than
    ``turn_degrees``. Transects with fewer than ``min_profiles`` are marked -1.
    """
    heads = (
        table.groupby("profile", sort=False)[["time", "latitude", "longitude"]]
        .first()
        .sort_values("time")
    )
    lon, lat = heads["longitude"].to_numpy(), heads["latitude"].to_numpy()
    dx = np.gradient(lon) * np.cos(np.deg2rad(lat))
    dy = np.gradient(lat)
    heading = np.degrees(np.arctan2(dy, dx))
    smooth = (
        pd.Series(np.exp(1j * np.deg2rad(heading)))
        .rolling(3, center=True, min_periods=1)
        .mean()
    )
    angle = np.degrees(np.angle(smooth.to_numpy()))
    change = np.abs((np.diff(angle) + 180) % 360 - 180)
    transect = np.concatenate([[0], np.cumsum(change > turn_degrees)])
    counts = pd.Series(transect).value_counts()
    transect = np.where(pd.Series(transect).map(counts) >= min_profiles, transect, -1)
    mapping = dict(zip(heads.index, transect, strict=True))
    out = table.copy()
    out["transect"] = out["profile"].map(mapping).astype(int)
    return out


def flag_suspect(table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Flag samples: provider QC not good, spikes, or density inversions.

    Returns the table with a ``suspect`` reason column ('' when fine) and a
    list of (profile, reason, count) rows for the result.
    """
    out = table.copy()
    reason = pd.Series("", index=out.index)
    for column in ("temperature", "salinity"):
        qc = f"{column}_qc"
        if qc in out:
            bad = ~out[qc].isin([1, 2]) & out[qc].notna()
            reason = reason.where(~bad, f"{column} provider QC")
        if column in out:
            grouped = out.groupby("profile")[column]
            median = grouped.transform(
                lambda s: s.rolling(5, center=True, min_periods=3).median()
            )
            mad = grouped.transform(
                lambda s: (
                    (s - s.rolling(5, center=True, min_periods=3).median())
                    .abs()
                    .median()
                )
            )
            spike = (out[column] - median).abs() > 6 * mad.clip(lower=1e-6)
            reason = reason.where(~(spike & (reason == "")), f"{column} spike")
    if "sigma0" in out:
        inversion = (
            out.sort_values(["profile", "pressure"]).groupby("profile")["sigma0"].diff()
            < -0.03
        )
        reason = reason.where(
            ~(inversion.reindex(out.index, fill_value=False) & (reason == "")),
            "density inversion",
        )
    out["suspect"] = reason
    listing = (
        out[out["suspect"] != ""]
        .groupby(["profile", "suspect"])
        .size()
        .reset_index(name="samples")
    )
    return out, listing.rename(columns={"suspect": "reason"}).to_dict("records")


# ---------------------------------------------------------------------- swaths

SWOT_SHORT_NAME = "SWOT_L2_LR_SSH_2.0"


def earthdata_login() -> Any:
    """Log in to NASA Earthdata from the environment, or say what is missing."""
    import earthaccess

    if not (
        os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")
    ) and not os.environ.get("EARTHDATA_TOKEN"):
        raise OceanToolError(
            "credential_missing",
            "NASA Earthdata Login is needed to download SWOT granules. Create a free account at "
            "https://urs.earthdata.nasa.gov and enter it in the Ocean connector's card "
            "(EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or EARTHDATA_TOKEN).",
        )
    return earthaccess.login(strategy="environment")


def swath_search(
    box: list[float], start: str, end: str, short_name: str
) -> list[dict[str, Any]]:
    """SWOT granules over a box and period (CMR search needs no login)."""
    import earthaccess

    try:
        granules = earthaccess.search_data(
            short_name=short_name,
            bounding_box=(box[0], box[2], box[1], box[3]),
            temporal=(start, end),
        )
    except Exception as exc:
        raise OceanToolError(
            "backend_unavailable", f"CMR search failed: {exc}"
        ) from exc
    rows = []
    for g in granules:
        umm = g["umm"]
        rows.append(
            {
                "granule": umm.get("GranuleUR"),
                "start": umm.get("TemporalExtent", {})
                .get("RangeDateTime", {})
                .get("BeginningDateTime"),
                "end": umm.get("TemporalExtent", {})
                .get("RangeDateTime", {})
                .get("EndingDateTime"),
                "size_mb": g.size() if callable(g.size) else g.size,
            }
        )
    return rows


def swath_download(
    box: list[float], start: str, end: str, short_name: str, folder: Path
) -> list[Path]:
    import earthaccess

    earthdata_login()
    granules = earthaccess.search_data(
        short_name=short_name,
        bounding_box=(box[0], box[2], box[1], box[3]),
        temporal=(start, end),
    )
    if not granules:
        raise OceanToolError(
            "no_data", f"no {short_name} granules over that box and period"
        )
    folder.mkdir(parents=True, exist_ok=True)
    return [Path(p) for p in earthaccess.download(granules, local_path=str(folder))]


def swath_compare(
    swath_paths: list[Path],
    gridded: Any,
    *,
    swath_variable: str,
    grid_variable: str,
    box: list[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Swath SSH against the gridded field at each pass time.

    The gridded field is interpolated to every swath pixel at the pass's mean
    time; the result table holds pixel, both values and the difference.
    """
    import xarray as xr

    rows = []
    lon_name = "longitude" if "longitude" in gridded.coords else "lon"
    lat_name = "latitude" if "latitude" in gridded.coords else "lat"
    for path in swath_paths:
        with xr.open_dataset(path) as sw:
            lat = sw["latitude"].to_numpy().ravel()
            lon = sw["longitude"].to_numpy().ravel()
            lon = np.where(lon > 180, lon - 360, lon)
            value = sw[swath_variable].to_numpy().ravel()
            inside = (
                (lon >= box[0])
                & (lon <= box[1])
                & (lat >= box[2])
                & (lat <= box[3])
                & np.isfinite(value)
            )
            if not inside.any():
                continue
            when = pd.to_datetime(sw["time"].to_numpy().ravel()).dropna()
            pass_time = when[len(when) // 2] if len(when) else None
        field = gridded[grid_variable]
        if pass_time is not None and "time" in field.dims:
            field = field.sel(
                time=np.datetime64(
                    pass_time.tz_localize(None) if pass_time.tzinfo else pass_time
                ),
                method="nearest",
            )
        grid_lon = field[lon_name].to_numpy()
        query_lon = (
            np.where(lon[inside] < 0, lon[inside] + 360, lon[inside])
            if grid_lon.max() > 180
            else lon[inside]
        )
        on_grid = field.interp(
            {
                lon_name: xr.DataArray(query_lon, dims="pixel"),
                lat_name: xr.DataArray(lat[inside], dims="pixel"),
            }
        ).to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "pass": path.stem,
                    "time": pass_time,
                    "longitude": lon[inside],
                    "latitude": lat[inside],
                    "swath": value[inside],
                    "gridded": on_grid,
                }
            )
        )
    if not rows:
        raise OceanToolError("no_data", "no swath pixels fall inside the box")
    table = pd.concat(rows, ignore_index=True)
    # Remove each pass's mean offset: the products use different references.
    table["difference"] = (
        table["swath"] - table.groupby("pass")["swath"].transform("mean")
    ) - (table["gridded"] - table.groupby("pass")["gridded"].transform("mean"))
    numbers = {
        "passes": int(table["pass"].nunique()),
        "pixels": len(table),
        "rms_difference": float(np.sqrt(np.nanmean(table["difference"] ** 2))),
        "note": "each pass's mean is removed from both fields before differencing",
    }
    return table, numbers
