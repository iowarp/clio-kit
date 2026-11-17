"""Write tables and grids with their provenance sidecars."""

from __future__ import annotations

from typing import Any

import pandas as pd
import xarray as xr

from .implementation.common import Output, media_type, output_path, write_provenance


def save_table(
    table: pd.DataFrame, name: str, role: str, output_dir: str | None, **prov: Any
) -> Output:
    path = output_path(name, ".parquet", output_dir)
    frame = table.copy()
    for column in frame.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(frame[column])
            and getattr(frame[column].dt, "tz", None) is None
        ):
            frame[column] = frame[column].dt.tz_localize("UTC")
    frame.to_parquet(path, index=False)
    return Output(
        path, role, media_type(path), write_provenance(path, **prov), rows=len(frame)
    )


def save_grid(
    ds: xr.Dataset, name: str, role: str, output_dir: str | None, **prov: Any
) -> Output:
    path = output_path(name, ".nc", output_dir)
    clean = ds.copy()
    for var in list(clean.variables):
        clean[var].attrs = {
            k: (str(v) if isinstance(v, (dict, list, tuple)) else v)
            for k, v in clean[var].attrs.items()
        }
        clean[var].encoding.pop("filters", None)
    clean.attrs = {
        k: (str(v) if isinstance(v, (dict, list, tuple, bool)) else v)
        for k, v in clean.attrs.items()
    }
    clean.to_netcdf(path)
    return Output(path, role, media_type(path), write_provenance(path, **prov))
