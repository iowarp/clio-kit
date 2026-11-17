"""Climate indices, composites, run evaluation and CMIP6 regional change.

ENSO: Niño 3.4 SST anomaly (5°S-5°N, 170°W-120°W) from a stated base period;
the ONI is its 3-month running mean and a phase holds when it is beyond
±0.5 °C for at least five consecutive overlapping seasons (NOAA CPC rule).
NAO: the leading EOF of seasonal-mean sea-level pressure anomalies over the
North Atlantic sector (20-80°N, 90°W-40°E), sqrt(cos lat) weighted; its
principal component standardised, the sign set so a positive index has low
pressure over Iceland (Hurrell 1995 / Hurrell & Deser 2009 convention).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from .common import WeatherToolError
from .grids import LAT, LON, TIME, area_mean, pick, select_box

NINO34 = [-170.0, -120.0, -5.0, 5.0]
NAO_BOX = [-90.0, 40.0, 20.0, 80.0]
PANGEO_CMIP6 = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"


def monthly(da: xr.DataArray) -> xr.DataArray:
    """Monthly means. Times out of order (stores joined from several datasets) are
    sorted; duplicate times are refused rather than one of them dropped silently."""
    t = pick(da, TIME)
    if not da.indexes[t].is_monotonic_increasing:
        da = da.sortby(t)
    duplicated = int(da.indexes[t].duplicated().sum())
    if duplicated:
        raise WeatherToolError(
            "bad_input", f"{duplicated} duplicate times in {da.name or 'the series'}"
        )
    return da.resample({t: "MS"}).mean()


def nino34(sst: xr.DataArray, base: tuple[int, int]) -> pd.DataFrame:
    box = monthly(area_mean(select_box(sst, NINO34)))
    t = pick(box, TIME)
    s = box.to_series()
    s.index = pd.to_datetime(s.index)
    clim = (
        s[(s.index.year >= base[0]) & (s.index.year <= base[1])]
        .groupby(lambda d: d.month)
        .mean()
    )
    anomaly = s - s.index.month.map(clim).to_numpy()
    oni = anomaly.rolling(3, center=True).mean()
    phase = pd.Series("neutral", index=oni.index)
    for sign, label in ((1, "el_nino"), (-1, "la_nina")):
        beyond = (sign * oni) >= 0.5
        run_id = (beyond != beyond.shift()).cumsum()
        lengths = beyond.groupby(run_id).transform("size")
        phase[beyond & (lengths >= 5)] = label
    del t
    return pd.DataFrame(
        {
            "time": s.index,
            "nino34": s.to_numpy(),
            "anomaly": anomaly.to_numpy(),
            "oni": oni.to_numpy(),
            "phase": phase.to_numpy(),
        }
    )


def season_mean(da: xr.DataArray, season: str) -> xr.DataArray:
    """Mean over a season ('DJF', 'JJA', ...) labelled by the year of its last month."""
    t = pick(da, TIME)
    months = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
        "JFM": [1, 2, 3],
        "NDJFM": [11, 12, 1, 2, 3],
    }[season]
    m = monthly(da)
    m = m.where(m[t].dt.month.isin(months), drop=True)
    year = m[t].dt.year + xr.where(m[t].dt.month >= months[0], 1, 0) * (
        1 if months[0] > months[-1] else 0
    )
    return m.groupby(year.rename("year")).mean()


def nao_eof(slp: xr.DataArray, season: str) -> tuple[pd.DataFrame, xr.DataArray, float]:
    """Leading EOF and standardised PC of seasonal SLP anomalies."""
    sub = season_mean(select_box(slp, NAO_BOX), season)
    lon, lat = pick(sub, LON), pick(sub, LAT)
    anomaly = sub - sub.mean("year")
    w = np.sqrt(np.cos(np.deg2rad(anomaly[lat])))
    x = (
        (anomaly * w)
        .transpose("year", lat, lon)
        .to_numpy()
        .reshape(anomaly.sizes["year"], -1)
    )
    valid = np.isfinite(x).all(axis=0)
    u, s, vt = np.linalg.svd(x[:, valid], full_matrices=False)
    pc = u[:, 0] * s[0]
    pattern = np.full(x.shape[1], np.nan)
    pattern[valid] = vt[0]
    xr.DataArray(
        pattern.reshape(anomaly.sizes[lat], anomaly.sizes[lon]),
        coords={lat: anomaly[lat], lon: anomaly[lon]},
        dims=[lat, lon],
    )
    # Regress the PC back onto unweighted anomalies so the pattern is in hPa per standard deviation.
    pcs = (pc - pc.mean()) / pc.std()
    regression = (
        anomaly * xr.DataArray(pcs, dims="year", coords={"year": anomaly["year"]})
    ).mean("year")
    iceland = regression.sel(
        {lat: 65.0, lon: (-20.0 if float(regression[lon].min()) < 0 else 340.0)},
        method="nearest",
    )
    if float(iceland) > 0:
        pcs, regression = -pcs, -regression
    explained = float(s[0] ** 2 / (s**2).sum())
    table = pd.DataFrame({"year": anomaly["year"].to_numpy(), "nao": pcs})
    return table, regression.rename("nao_pattern"), explained


def composite(
    field: xr.DataArray, phases: pd.DataFrame, *, season: str, phase_column: str
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Seasonal-mean anomaly composites per phase with a Welch t-test against the rest."""
    seasonal = season_mean(field, season)
    anomaly = seasonal - seasonal.mean("year")
    out = xr.Dataset()
    counts = {}
    for phase, group in phases.groupby(phase_column):
        years = [int(y) for y in group["year"] if int(y) in anomaly["year"].to_numpy()]
        if not years:
            continue
        inside = anomaly.sel(year=years)
        rest = anomaly.drop_sel(year=years)
        out[f"{phase}_composite"] = inside.mean("year")
        if len(years) > 1 and rest.sizes["year"] > 1:
            _, p = stats.ttest_ind(
                inside.to_numpy(),
                rest.to_numpy(),
                axis=0,
                equal_var=False,
                nan_policy="omit",
            )
            out[f"{phase}_pvalue"] = (inside.mean("year").dims, np.asarray(p))
        counts[str(phase)] = years
    if not counts:
        raise WeatherToolError("no_data", "no phase year overlaps the field's years")
    return out, {
        "years_by_phase": counts,
        "season": season,
        "test": "Welch t-test, phase years against all others",
    }


def phase_years(
    index: pd.DataFrame, *, season: str, kind: Literal["enso", "nao"], threshold: float
) -> pd.DataFrame:
    """Classify years for compositing: ENSO by the ONI phase in the season's
    central month, NAO by the seasonal index beyond ±threshold."""
    if kind == "enso":
        central = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10, "JFM": 2, "NDJFM": 1}[
            season
        ]
        sel = index[pd.to_datetime(index["time"]).dt.month == central]
        return pd.DataFrame(
            {
                "year": pd.to_datetime(sel["time"]).dt.year.to_numpy(),
                "phase": sel["phase"].to_numpy(),
            }
        )
    phase = np.where(
        index["nao"] >= threshold,
        "nao_positive",
        np.where(index["nao"] <= -threshold, "nao_negative", "neutral"),
    )
    return pd.DataFrame({"year": index["year"], "phase": phase})


def evaluate_run(
    run: xr.DataArray,
    reference: xr.DataArray,
    *,
    regions: dict[str, list[float]],
    grid_step: float | None,
) -> tuple[xr.Dataset, pd.DataFrame, dict[str, Any]]:
    """Run and reference on a common grid (the coarser, or a stated step):
    time-mean bias map, zonal-mean bias (with a vertical level if present),
    monthly climatologies per region and their RMSE."""
    _rlon, rlat = pick(run, LON), pick(run, LAT)
    if grid_step:
        lons = np.arange(-180 + grid_step / 2, 180, grid_step)
        lats = np.arange(-90 + grid_step / 2, 90, grid_step)
    else:
        steps = [
            abs(float(run[rlat].diff(rlat).median())),
            abs(
                float(
                    reference[pick(reference, LAT)].diff(pick(reference, LAT)).median()
                )
            ),
        ]
        coarse = run if steps[0] >= steps[1] else reference
        lons, lats = (
            coarse[pick(coarse, LON)].to_numpy(),
            coarse[pick(coarse, LAT)].to_numpy(),
        )

    def regrid(da: xr.DataArray) -> xr.DataArray:
        lon, lat = pick(da, LON), pick(da, LAT)
        da = (
            da.assign_coords({lon: ((da[lon] + 180) % 360) - 180})
            .sortby(lon)
            .rename({lon: "longitude", lat: "latitude"})
        )
        return da.interp(longitude=((lons + 180) % 360) - 180, latitude=lats)

    a, b = regrid(run), regrid(reference)
    ta, tb = pick(a, TIME), pick(b, TIME)
    ma, mb = monthly(a), monthly(b)
    common = np.intersect1d(ma[ta].to_numpy(), mb[tb].to_numpy())
    if common.size == 0:
        raise WeatherToolError("no_data", "the run and the reference share no months")
    ma, mb = (
        ma.sel({ta: common}),
        mb.sel({tb: common}).rename({tb: ta}) if tb != ta else mb.sel({tb: common}),
    )
    bias = (ma - mb).mean(ta)
    out = xr.Dataset({"bias": bias, "zonal_mean_bias": bias.mean("longitude")})
    rows = []
    for name, box in regions.items():
        ca = area_mean(select_box(ma, box)).groupby(f"{ta}.month").mean()
        cb = area_mean(select_box(mb, box)).groupby(f"{ta}.month").mean()
        for month in range(1, 13):
            rows.append(
                {
                    "region": name,
                    "month": month,
                    "run": float(ca.sel(month=month)),
                    "reference": float(cb.sel(month=month)),
                }
            )
    cycles = pd.DataFrame(rows)
    cycles["difference"] = cycles["run"] - cycles["reference"]
    w = xr.DataArray(
        np.cos(np.deg2rad(bias["latitude"].to_numpy())),
        coords={"latitude": bias["latitude"]},
        dims=["latitude"],
    )
    numbers = {
        "months": int(common.size),
        "global_mean_bias": float(bias.weighted(w).mean()),
        "global_rmse": float(np.sqrt((bias**2).weighted(w).mean())),
        "seasonal_cycle_rmse_by_region": cycles.groupby("region")["difference"]
        .apply(lambda d: float(np.sqrt(np.mean(d**2))))
        .to_dict(),
        "regridding": "bilinear onto "
        + (f"a {grid_step}° grid" if grid_step else "the coarser grid"),
    }
    return out, cycles, numbers


def cmip_change(
    *,
    models: list[str],
    experiment: str,
    variable: str,
    table: str,
    member: str,
    box: list[float],
    historical: tuple[int, int],
    future: tuple[int, int],
    season: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, xr.Dataset, list[dict[str, Any]]]:
    """Regional change (future minus historical period) per CMIP6 model from the Pangeo catalog."""
    import intake

    try:
        catalog = intake.open_esm_datastore(PANGEO_CMIP6)
    except Exception as exc:
        raise WeatherToolError(
            "backend_unavailable", f"Pangeo CMIP6 catalog: {exc}"
        ) from exc
    changes, series, maps, sources = [], [], [], []
    for model in models:
        rows = {}
        for exp in ("historical", experiment):
            q = catalog.search(
                source_id=model,
                experiment_id=exp,
                variable_id=variable,
                table_id=table,
                member_id=member,
            )
            if len(q.df) == 0:
                raise WeatherToolError(
                    "no_data",
                    f"{model} has no {exp} {table}.{variable} {member} in the Pangeo catalog",
                )
            store = q.df["zstore"].iloc[0]
            ds = xr.open_zarr(
                store, storage_options={"token": "anon"}, consolidated=True
            )
            rows[exp] = ds[variable]
            sources.append(
                {
                    "model": model,
                    "experiment": exp,
                    "zstore": store,
                    "version": str(q.df.get("version", pd.Series([""])).iloc[0]),
                }
            )
        joined = xr.concat([rows["historical"], rows[experiment]], dim="time")
        region = area_mean(select_box(joined, box))
        if season:
            values = season_mean(region, season)
            yearly = values.to_series()
        else:
            yearly = region.groupby("time.year").mean().to_series()
        hist = yearly[
            (yearly.index >= historical[0]) & (yearly.index <= historical[1])
        ].mean()
        fut = yearly[(yearly.index >= future[0]) & (yearly.index <= future[1])].mean()
        changes.append(
            {
                "model": model,
                "historical_mean": float(hist),
                "future_mean": float(fut),
                "change": float(fut - hist),
            }
        )
        series.append(
            pd.DataFrame(
                {
                    "model": model,
                    "year": yearly.index,
                    "value": yearly.to_numpy(),
                    "anomaly": yearly.to_numpy() - hist,
                }
            )
        )
        field = select_box(joined, [box[0] - 10, box[1] + 10, box[2] - 10, box[3] + 10])
        tf = field["time"].dt.year
        delta = field.where((tf >= future[0]) & (tf <= future[1])).mean(
            "time"
        ) - field.where((tf >= historical[0]) & (tf <= historical[1])).mean("time")
        target_lon = np.arange(box[0] - 10, box[1] + 10.01, 1.0)
        dataset_lon = (
            target_lon % 360 if float(delta["lon"].max()) > 180 else target_lon
        )
        common = delta.rename("change").interp(
            lat=np.arange(box[2] - 10, box[3] + 10.01, 1.0), lon=dataset_lon
        )
        maps.append(common.assign_coords(lon=target_lon).expand_dims(model=[model]))
    change_table = pd.DataFrame(changes)
    combined = xr.concat(maps, dim="model")
    field_ds = xr.Dataset(
        {
            "change": combined,
            "multi_model_mean": combined.mean("model"),
            "spread": combined.std("model"),
        }
    )
    return change_table, pd.concat(series, ignore_index=True), field_ds, sources
