"""Read hydrographic casts into one long-form profile table.

The profile table is the exchange format between ocean tools and the views:
one row per sample, with at least ``profile``, ``time``, ``latitude``,
``longitude`` and ``pressure`` (dbar), plus whatever variables were measured
(``temperature`` in-situ ITS-90 °C, ``salinity`` PSS-78, ``oxygen``, ...).
It is written as Parquet so the profile stack and tables can read it by
reference.

Readers: Sea-Bird ``.cnv``, CCHDO WHP-exchange CTD (``*_ct1.csv`` and zips of
them), CF NetCDF casts (CCHDO and similar), and plain CSV with named columns.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .common import OceanToolError

REQUIRED = ("profile", "time", "latitude", "longitude", "pressure")

# Sea-Bird SBE Data Processing short names -> profile-table columns. The
# primary sensor (0) is preferred; the secondary is used only if primary absent.
_CNV_NAMES = {
    "pressure": ("prDM", "prdM", "prSM", "pr"),
    "temperature": ("t090C", "t090", "tv290C", "t068C", "t190C"),
    "salinity": ("sal00", "sal11"),
    "conductivity": ("c0S/m", "c0mS/cm", "c1S/m"),
    "oxygen": ("sbox0Mm/Kg", "sbeox0Mm/Kg", "sbeox0ML/L", "sbox1Mm/Kg"),
    "fluorescence": ("flECO-AFL", "flC", "wetStar"),
    "depth": ("depSM", "depS"),
}

# CCHDO WHP-exchange parameter names -> profile-table columns.
_WHP_NAMES = {
    "CTDPRS": "pressure",
    "CTDTMP": "temperature",
    "CTDSAL": "salinity",
    "CTDOXY": "oxygen",
    "CTDFLUOR": "fluorescence",
}


def _parse_position(text: str) -> float:
    """``'32 30.12 N'`` or ``'-32.5'`` -> decimal degrees."""
    text = text.strip()
    match = re.match(r"^(-?\d+)\s+(\d+(?:\.\d+)?)\s*([NSEW])?$", text)
    if match:
        deg, minutes, hemi = float(match[1]), float(match[2]), match[3]
        value = abs(deg) + minutes / 60.0
        if deg < 0 or hemi in ("S", "W"):
            value = -value
        return value
    match = re.match(r"^(-?\d+(?:\.\d+)?)\s*([NSEW])?$", text)
    if match:
        value = float(match[1])
        return -abs(value) if match[2] in ("S", "W") else value
    raise ValueError(f"unreadable position {text!r}")


def read_cnv(path: Path, profile: str | None = None) -> pd.DataFrame:
    """Read one Sea-Bird ``.cnv`` cast."""
    names: dict[int, str] = {}
    meta: dict[str, str] = {}
    bad_flag = -9.99e-29
    lines = path.read_text(encoding="latin-1").splitlines()
    data_start = None
    for index, line in enumerate(lines):
        if line.startswith("*END*"):
            data_start = index + 1
            break
        if match := re.match(r"^# name (\d+) = ([^:]+):", line):
            names[int(match[1])] = match[2].strip()
        elif match := re.match(r"^# bad_flag = (.+)$", line):
            bad_flag = float(match[1])
        elif match := re.match(r"^\*\s*NMEA (Latitude|Longitude)\s*=\s*(.+)$", line):
            meta[match[1].lower()] = match[2]
        elif match := re.match(r"^\*\s*NMEA UTC \(Time\)\s*=\s*(.+)$", line):
            meta["time"] = match[1]
        elif match := re.match(r"^# start_time = (.+?)(\s*\[.*)?$", line):
            meta.setdefault("time", match[1])
        elif match := re.match(
            r"^\*\*\s*(Latitude|Longitude|Lat|Lon)\s*[:=]\s*(.+)$", line, re.IGNORECASE
        ):
            key = "latitude" if match[1].lower().startswith("lat") else "longitude"
            meta.setdefault(key, match[2])
    if data_start is None or not names:
        raise OceanToolError(
            "bad_input", f"{path.name} is not a Sea-Bird .cnv file (no *END* or names)"
        )
    raw = np.loadtxt(io.StringIO("\n".join(lines[data_start:])), ndmin=2)
    raw[np.isclose(raw, bad_flag)] = np.nan
    frame = pd.DataFrame(raw, columns=[names[i] for i in sorted(names)][: raw.shape[1]])
    out = pd.DataFrame()
    for column, candidates in _CNV_NAMES.items():
        for candidate in candidates:
            if candidate in frame.columns:
                out[column] = frame[candidate]
                break
    if "pressure" not in out:
        raise OceanToolError(
            "bad_input", f"{path.name} has no pressure channel (prDM/prSM)"
        )
    for key in ("latitude", "longitude"):
        if key not in meta:
            raise OceanToolError(
                "bad_input",
                f"{path.name} has no {key} in its header; pass a CSV with positions",
            )
    out["latitude"] = _parse_position(meta["latitude"])
    out["longitude"] = _parse_position(meta["longitude"])
    out["time"] = pd.to_datetime(meta.get("time"), errors="coerce")
    out["profile"] = profile or path.stem
    return out


def _read_whp_text(text: str, name: str) -> pd.DataFrame:
    """Read one CCHDO WHP-exchange CTD file (``*_ct1.csv``)."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines or not lines[0].upper().startswith("CTD"):
        raise OceanToolError("bad_input", f"{name} is not a WHP-exchange CTD file")
    header: dict[str, str] = {}
    index = 1
    if lines[index].upper().startswith("NUMBER_HEADERS"):
        count = int(lines[index].split("=")[1])
        for line in lines[index + 1 : index + count]:
            key, _, value = line.partition("=")
            header[key.strip().upper()] = value.strip()
        index += count
    columns = [c.strip() for c in lines[index].split(",")]
    body = []
    for line in lines[index + 2 :]:
        if line.startswith("END_DATA"):
            break
        body.append(line)
    frame = pd.read_csv(
        io.StringIO("\n".join(body)), names=columns, skipinitialspace=True
    )
    out = pd.DataFrame()
    for whp, column in _WHP_NAMES.items():
        if whp in frame.columns:
            values = frame[whp].astype(float).where(frame[whp].astype(float) > -900)
            flag = f"{whp}_FLAG_W"
            if flag in frame.columns:
                # WOCE flags: 2 acceptable; 3 questionable, 4 bad, 9 not sampled.
                values = values.where(frame[flag].isin([2, 6]))
            out[column] = values
    expocode = header.get("EXPOCODE", "")
    station = header.get("STNNBR", "")
    cast = header.get("CASTNO", "")
    out["profile"] = (
        "-".join(p for p in (expocode, station.zfill(3), cast) if p) or name
    )
    out["latitude"] = float(header["LATITUDE"])
    out["longitude"] = float(header["LONGITUDE"])
    date, clock = header.get("DATE", ""), header.get("TIME", "0000").zfill(4)
    out["time"] = pd.to_datetime(f"{date}{clock}", format="%Y%m%d%H%M", errors="coerce")
    out["station"] = station
    return out


def read_whp(path: Path) -> pd.DataFrame:
    """Read a WHP-exchange CTD file or a zip of them."""
    if zipfile.is_zipfile(path):
        frames = []
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if member.lower().endswith(".csv"):
                    text = archive.read(member).decode("latin-1")
                    frames.append(_read_whp_text(text, member))
        if not frames:
            raise OceanToolError(
                "bad_input", f"{path.name} holds no WHP-exchange CSV files"
            )
        return pd.concat(frames, ignore_index=True)
    return _read_whp_text(path.read_text(encoding="latin-1"), path.name)


_NC_ALIASES = {
    "pressure": ("pressure", "PRES", "CTDPRS", "pres"),
    "temperature": ("ctd_temperature", "temperature", "TEMP", "CTDTMP", "temp"),
    "salinity": ("ctd_salinity", "salinity", "PSAL", "CTDSAL", "psal"),
    "oxygen": ("ctd_oxygen", "oxygen", "DOXY", "CTDOXY"),
    "latitude": ("latitude", "LATITUDE", "lat"),
    "longitude": ("longitude", "LONGITUDE", "lon"),
    "time": ("time", "TIME", "JULD"),
}


def read_netcdf_casts(path: Path) -> pd.DataFrame:
    """Read CF NetCDF casts (one profile per file or an N_PROF x N_LEVELS file)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        found = {}
        for column, aliases in _NC_ALIASES.items():
            for alias in aliases:
                if alias in ds.variables:
                    found[column] = alias
                    break
        if "pressure" not in found:
            raise OceanToolError("bad_input", f"{path.name} has no pressure variable")
        frame = ds[list(found.values())].to_dataframe().reset_index()
    frame = frame.rename(columns={v: k for k, v in found.items()})
    profile_dims = [
        c for c in frame.columns if c.upper() in ("N_PROF", "PROFILE", "STATION")
    ]
    if profile_dims:
        frame["profile"] = path.stem + "-" + frame[profile_dims[0]].astype(str)
    else:
        frame["profile"] = path.stem
    return frame


def read_csv_casts(path: Path, columns: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a CSV with one row per sample; ``columns`` renames to table names."""
    frame = pd.read_csv(path)
    if columns:
        frame = frame.rename(columns={v: k for k, v in columns.items()})
    if "time" in frame:
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    if "profile" not in frame:
        frame["profile"] = path.stem
    return frame


def read_casts(
    paths: Iterable[Path], columns: dict[str, str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Read casts of any supported format; returns the table and formats seen."""
    frames, formats = [], []
    for path in paths:
        if not path.exists():
            raise OceanToolError("bad_input", f"{path} does not exist")
        lower = path.name.lower()
        if lower.endswith(".cnv"):
            frames.append(read_cnv(path))
            formats.append("sbe-cnv")
        elif lower.endswith((".nc", ".nc4", ".netcdf")):
            frames.append(read_netcdf_casts(path))
            formats.append("netcdf")
        elif lower.endswith(".zip") or "_ct1" in lower:
            frames.append(read_whp(path))
            formats.append("whp-exchange")
        elif lower.endswith(".csv"):
            head = path.read_text(encoding="latin-1")[:20].upper()
            if head.startswith("CTD"):
                frames.append(read_whp(path))
                formats.append("whp-exchange")
            else:
                frames.append(read_csv_casts(path, columns))
                formats.append("csv")
        else:
            raise OceanToolError(
                "bad_input",
                f"{path.name}: unsupported cast format (use .cnv, .nc, _ct1.csv, .zip, .csv)",
            )
    if not frames:
        raise OceanToolError("bad_input", "no cast files given")
    table = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED if c not in table]
    if missing:
        raise OceanToolError(
            "bad_input", f"casts lack required columns: {', '.join(missing)}"
        )
    table = table.dropna(subset=["pressure"]).sort_values(["profile", "pressure"])
    return table.reset_index(drop=True), sorted(set(formats))


def write_profiles(table: pd.DataFrame, path: Path) -> Path:
    """Write the profile table as Parquet (times as UTC timestamps)."""
    table = table.copy()
    if "time" in table:
        table["time"] = pd.to_datetime(table["time"], utc=True, errors="coerce")
    table.to_parquet(path, index=False)
    return path


def read_profiles(path: str | Path) -> pd.DataFrame:
    """Read a profile table written by any ocean tool (Parquet or CSV)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise OceanToolError("bad_input", f"profile table {path} does not exist")
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        table = pd.read_csv(path, parse_dates=["time"])
    else:
        raise OceanToolError(
            "bad_input", f"{path.name}: profile tables are .parquet or .csv"
        )
    missing = [c for c in REQUIRED if c not in table]
    if missing:
        raise OceanToolError(
            "bad_input", f"{path.name} lacks columns: {', '.join(missing)}"
        )
    return table


def profile_summary(table: pd.DataFrame) -> dict[str, object]:
    """Counts and extents for a result's numbers."""
    times = pd.to_datetime(table["time"], utc=True, errors="coerce")
    return {
        "profiles": int(table["profile"].nunique()),
        "samples": len(table),
        "latitude_range": [
            float(table["latitude"].min()),
            float(table["latitude"].max()),
        ],
        "longitude_range": [
            float(table["longitude"].min()),
            float(table["longitude"].max()),
        ],
        "pressure_max_dbar": float(table["pressure"].max()),
        "time_range": [
            times.min().isoformat() if times.notna().any() else None,
            times.max().isoformat() if times.notna().any() else None,
        ],
        "variables": [
            c for c in table.columns if c not in (*REQUIRED, "station", "depth")
        ],
    }
