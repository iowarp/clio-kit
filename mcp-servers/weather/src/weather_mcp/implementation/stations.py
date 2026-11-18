"""Hourly station observations from NOAA's GHCN-hourly (GHCNh), keyless.

GHCNh replaced the Integrated Surface Database (ISD) in 2025: ISD's access
files stop in 2025 and its old URL now redirects to an index page. Stations
are named by their GHCNh id, ICAO code or WMO id, resolved through NCEI's
station list.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import requests

from .common import Source, WeatherToolError

GHCNH = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly"
GHCNH_FILE = GHCNH + "/access/by-year/{year}/parquet/GHCNh_{station}_{year}.parquet"
GHCNH_STATIONS = GHCNH + "/doc/ghcnh-station-list.csv"
GHCNH_REFERENCE = (
    "https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly"
)
#: Observation columns read, with their output names.
VARIABLES = {
    "temperature": "t2m_c",
    "dew_point_temperature": "td2m_c",
    "sea_level_pressure": "mslp_hpa",
}
#: Quality codes that mark a value suspect or erroneous: ISD's 2, 3, 6 and 7 for
#: values carried over from ISD, and any letter (a failed GHCNh check).
_FAILED = re.compile(r"^(?:[2367]|[A-Za-z]+)$")
_GHCNH_ID = re.compile(r"^[A-Z]{2}[A-Z0-9_-]{9}$")


def _station_list() -> pd.DataFrame:
    try:
        response = requests.get(GHCNH_STATIONS, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherToolError(
            "backend_unavailable", f"GHCNh station list: {exc}"
        ) from exc
    return pd.read_csv(io.StringIO(response.text), dtype=str, keep_default_na=False)


def resolve(stations: list[str]) -> dict[str, str]:
    """Each requested name mapped to its GHCNh id; unknown names are refused together."""
    resolved: dict[str, str] = {}
    pending = []
    for name in stations:
        text = name.strip().upper()
        if _GHCNH_ID.match(text) and not text.isdigit():
            resolved[name] = text
        else:
            pending.append(name)
    if not pending:
        return resolved
    table = _station_list()
    unknown = []
    for name in pending:
        text = name.strip().upper()
        if re.fullmatch(r"[A-Z]{4}", text):
            hits = table[table["ICAO"] == text]
        else:
            # A WMO id, or an ISD id whose first five digits are the WMO id.
            wmo = text[:5] if re.fullmatch(r"\d{11}", text) else text
            hits = table[table["WMO_ID"] == wmo.lstrip("0")]
            if hits.empty:
                hits = table[table["WMO_ID"] == wmo]
        if hits.empty:
            unknown.append(name)
        else:
            resolved[name] = str(hits.iloc[0]["GHCN_ID"])
    if unknown:
        raise WeatherToolError(
            "bad_input",
            f"not in the GHCNh station list: {', '.join(unknown)} "
            "(give a GHCNh id, a four-letter ICAO code or a WMO id)",
        )
    return resolved


def _year(station: str, year: int) -> pd.DataFrame | None:
    url = GHCNH_FILE.format(station=station, year=year)
    try:
        response = requests.get(url, timeout=180)
    except requests.RequestException as exc:
        raise WeatherToolError(
            "backend_unavailable", f"GHCNh {station} {year}: {exc}"
        ) from exc
    if response.status_code == 404:
        return None
    if not response.ok:
        raise WeatherToolError(
            "backend_unavailable",
            f"GHCNh {station} {year}: HTTP {response.status_code}",
        )
    if not response.content.startswith(b"PAR1"):
        raise WeatherToolError(
            "backend_unavailable",
            f"GHCNh {station} {year}: the response is not a Parquet file",
        )
    return pd.read_parquet(io.BytesIO(response.content))


def ghcnh_stations(
    stations: list[str], start: str, end: str
) -> tuple[pd.DataFrame, Source, dict[str, object]]:
    """Hourly 2 m temperature, dew point and sea-level pressure; returns the table,
    its source and what was done (resolved ids, values removed by quality code)."""
    ids = resolve(stations)
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    rows = []
    removed: dict[str, int] = {}
    for name, station in ids.items():
        for year in years:
            frame = _year(station, year)
            if frame is None:
                continue
            out = pd.DataFrame(
                {
                    "station": station,
                    "requested": name,
                    "name": frame.get("Station_name"),
                    "time": pd.to_datetime(frame["DATE"], utc=True),
                    "latitude": frame["LATITUDE"].astype(float),
                    "longitude": frame["LONGITUDE"].astype(float),
                }
            )
            for column, label in VARIABLES.items():
                values = pd.to_numeric(frame.get(column), errors="coerce")
                codes = frame.get(f"{column}_Quality_Code")
                if codes is not None:
                    failed = codes.astype(str).str.strip().str.match(_FAILED)
                    removed[label] = removed.get(label, 0) + int(
                        (failed & values.notna()).sum()
                    )
                    values = values.where(~failed)
                out[label] = values.to_numpy() if values is not None else np.nan
            rows.append(out)
    if not rows:
        raise WeatherToolError(
            "no_data", f"no GHCNh records for {stations} in {start}..{end}"
        )
    table = pd.concat(rows, ignore_index=True)
    table = table[
        (table["time"] >= pd.Timestamp(start, tz="UTC"))
        & (table["time"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
    ].dropna(subset=list(VARIABLES.values()), how="all")
    if table.empty:
        raise WeatherToolError(
            "no_data", f"no GHCNh records for {stations} in {start}..{end}"
        )
    source = Source(
        name="NOAA Global Historical Climatology Network hourly (GHCNh)",
        kind="service",
        reference=GHCNH_REFERENCE,
        version="by-year Parquet",
        subset={"stations": ids, "start": start, "end": end},
        licence="NOAA NCEI: free use; WMO Resolution 40 restrictions may apply to some non-US stations",
    )
    return (
        table.reset_index(drop=True),
        source,
        {"stations": ids, "removed_by_quality_code": removed},
    )
