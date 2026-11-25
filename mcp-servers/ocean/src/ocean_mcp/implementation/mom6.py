"""Pre-flight checks for a regional MOM6 run directory.

Checks what people get wrong by hand before a run (plan O18). Parameter
names are MOM6's own, confirmed in the MOM6 source (mom-ocean/MOM6 at
0e9a561): ``OBC_NUMBER_OF_SEGMENTS`` and ``OBC_SEGMENT_%03d`` /
``OBC_SEGMENT_%03d_DATA`` (src/core/MOM_open_boundary.F90),
``MINIMUM_DEPTH`` (src/initialization/MOM_shared_initialization.F90),
``MASKING_DEPTH``, ``TOPO_FILE`` (default ``topog.nc``), ``TOPO_VARNAME``
(default ``depth``), ``LAYOUT``, ``MASKTABLE``, ``NIGLOBAL``, ``NJGLOBAL``
(src/framework/MOM_domains.F90).

Every check returns ``passed``/``failed``/``not_checked`` with what it read,
so a check that could not run says so instead of passing silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .common import OceanToolError


def parse_mom_params(paths: list[Path]) -> dict[str, str]:
    """MOM_input then MOM_override: ``KEY = value ! comment``, ``#override KEY = value``."""
    params: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.split("!", 1)[0].strip()
            line = re.sub(r"^#override\s+", "", line)
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            params[key.strip()] = value.strip().strip('"').strip("'")
    return params


def parse_namelist(path: Path) -> dict[str, dict[str, str]]:
    """A Fortran namelist file as {group: {key: raw value}}."""
    groups: dict[str, dict[str, str]] = {}
    if not path.exists():
        return groups
    current = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if line.startswith("&"):
            current = line[1:].strip().lower()
            groups[current] = {}
        elif line.startswith("/"):
            current = None
        elif current and "=" in line:
            key, _, value = line.partition("=")
            groups[current][key.strip().lower()] = value.strip().rstrip(",")
    return groups


@dataclass
class Check:
    name: str
    status: str  # passed | failed | not_checked
    detail: str
    items: list[Any] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
            "items": self.items[:100],
            "count": len(self.items),
        }


def wet_mask(
    depth: np.ndarray, minimum_depth: float, masking_depth: float | None
) -> np.ndarray:
    """Wet where depth exceeds MASKING_DEPTH (default: MINIMUM_DEPTH, as MOM6 does)."""
    cut = masking_depth if masking_depth is not None else minimum_depth
    return np.nan_to_num(depth, nan=0.0) > cut


def narrow_cells(wet: np.ndarray) -> tuple[list[list[int]], list[list[int]]]:
    """Isolated wet cells (no wet 4-neighbour) and one-cell-wide channel cells.

    A channel cell is wet with land on both sides across one axis (west and
    east, or south and north) and water along the other axis. Indices are
    0-based [j, i] into the (y, x) array.
    """
    padded = np.pad(wet, 1, constant_values=False)
    n = padded[2:, 1:-1]
    s = padded[:-2, 1:-1]
    e = padded[1:-1, 2:]
    w = padded[1:-1, :-2]
    isolated = wet & ~n & ~s & ~e & ~w
    ns_channel = wet & ~e & ~w & (n | s)
    ew_channel = wet & ~n & ~s & (e | w)
    channel = (ns_channel | ew_channel) & ~isolated
    return np.argwhere(isolated).tolist(), np.argwhere(channel).tolist()


def _times(path: Path) -> pd.DatetimeIndex | None:
    try:
        with xr.open_dataset(path) as ds:
            for name in ("time", "Time", "valid_time"):
                if name in ds.variables:
                    return pd.DatetimeIndex(pd.to_datetime(ds[name].to_numpy()))
    except Exception:  # noqa: BLE001 - an unreadable file is reported by the caller
        return None
    return None


def _varies_in_time(path: Path, fields: list[str]) -> bool:
    """Whether any field a segment reads from this file has a time dimension.

    Tidal harmonics (amplitude and phase) and other fixed fields carry no time
    dimension and apply to the whole run, so their files are not date-checked.
    An unreadable file, or one whose fields are not found, counts as varying so
    the date check reports it.
    """
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            # MOM6 reads field "v" of segment N as "v" or "v_segment_00N".
            found = [
                ds[name]
                for name in ds.variables
                if any(
                    name == v or str(name).startswith(f"{v}_segment") for v in fields
                )
            ]
            if not found:
                return True
            return any(
                d in ("time", "Time", "valid_time") or "time" in str(d).lower()
                for da in found
                for d in da.dims
            )
    except Exception:  # noqa: BLE001 - the date check reports an unreadable file
        return True


def run_window(
    namelists: dict[str, dict[str, str]],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    """Run start and end from input.nml (ocean_solo_nml or coupler_nml)."""
    for group in ("ocean_solo_nml", "coupler_nml"):
        nml = namelists.get(group, {})
        stamp = nml.get("date_init") or nml.get("current_date")
        if not stamp:
            continue
        parts = [int(float(x)) for x in re.split(r"[,\s]+", stamp) if x]
        parts += [0] * (6 - len(parts))
        start = pd.Timestamp(
            year=parts[0],
            month=max(parts[1], 1),
            day=max(parts[2], 1),
            hour=parts[3],
            minute=parts[4],
            second=parts[5],
        )
        months = int(float(nml.get("months", 0) or 0))
        days = int(float(nml.get("days", 0) or 0))
        hours = int(float(nml.get("hours", 0) or 0))
        end = (
            start + pd.DateOffset(months=months) + pd.Timedelta(days=days, hours=hours)
        )
        return start, end, f"input.nml &{group}"
    return None, None, "input.nml has no date_init/current_date"


#: Parameters that turn on a surface boundary-layer scheme in MOM6.
PBL_SCHEMES = ("ENERGETICS_SFC_PBL", "USE_KPP", "BULKMIXEDLAYER")


def _true(value: str | None) -> bool:
    return str(value or "").strip().upper() in ("TRUE", "T", ".TRUE.")


def surface_mixing_check(params: dict[str, str]) -> Check:
    """Surface buoyancy forcing on, with a scheme to carry it below the top layer."""

    buoy = params.get("BUOY_CONFIG", "").strip().lower()
    if buoy in ("", "zero", "none"):
        return Check(
            "surface_mixing", "not_checked", "no surface buoyancy forcing (BUOY_CONFIG)"
        )
    on = [name for name in PBL_SCHEMES if _true(params.get(name))]
    if on:
        return Check(
            "surface_mixing", "passed", f"surface boundary layer: {', '.join(on)}"
        )
    pen = params.get("PEN_SW_NBANDS", "")
    detail = (
        f"BUOY_CONFIG = {params.get('BUOY_CONFIG')} applies heat and fresh-water fluxes, "
        f"but no surface boundary-layer scheme is on ({', '.join(PBL_SCHEMES)} all unset or "
        f"False); vertical mixing is the background KD = {params.get('KD', 'default')}, so the "
        "fluxes stay in the top layer and the surface drifts (warms under summer heating)"
    )
    if not pen or pen.strip() in ("0",):
        detail += "; shortwave is absorbed at the surface (no PEN_SW_NBANDS)"
    return Check("surface_mixing", "failed", detail, [name for name in PBL_SCHEMES])


def preflight(
    run_dir: Path,
    *,
    input_dir: Path | None,
    kept_cells: list[list[int]],
) -> tuple[list[Check], dict[str, Any]]:
    """Run every check on a MOM6 run directory."""
    if not run_dir.is_dir():
        raise OceanToolError("bad_input", f"{run_dir} is not a directory")
    params = parse_mom_params([run_dir / "MOM_input", run_dir / "MOM_override"])
    if not params:
        raise OceanToolError("bad_input", f"no MOM_input or MOM_override in {run_dir}")
    namelists = parse_namelist(run_dir / "input.nml")
    inputs = input_dir or (run_dir / params.get("INPUTDIR", "INPUT").strip("/"))
    checks: list[Check] = []
    facts: dict[str, Any] = {"input_dir": str(inputs)}

    # 1. Open boundaries: every declared segment has a data string and its files exist.
    n_seg = int(float(params.get("OBC_NUMBER_OF_SEGMENTS", 0)))
    missing: list[str] = []
    segments: list[dict[str, Any]] = []
    for k in range(1, n_seg + 1):
        key = f"OBC_SEGMENT_{k:03d}"
        if key not in params:
            missing.append(f"{key} not set")
            continue
        data = params.get(f"{key}_DATA", "")
        if not data:
            missing.append(f"{key}_DATA not set")
        refs = re.findall(r"file:([^\s(,]+)\(([^)\s]+)\)", data)
        files = sorted({f for f, _ in refs})
        fields = {f: sorted({v for g, v in refs if g == f}) for f in files}
        segments.append(
            {"segment": key, "position": params[key], "files": files, "fields": fields}
        )
        missing.extend(f"{key}: {f} absent" for f in files if not (inputs / f).exists())
    extra = (
        sorted(
            p.name
            for p in inputs.glob("*segment*")
            if not any(p.name in s["files"] for s in segments)
        )
        if inputs.is_dir()
        else []
    )
    checks.append(
        Check(
            "open_boundaries",
            "failed" if missing else ("passed" if n_seg else "not_checked"),
            f"{n_seg} segments declared (OBC_NUMBER_OF_SEGMENTS); unused segment files: {extra or 'none'}",
            missing,
        )
    )
    facts["segments"] = segments

    # 2. Dates: boundary and initial files cover the run.
    start, end, source = run_window(namelists)
    if start is None:
        checks.append(Check("dates_cover_run", "not_checked", source))
    else:
        gaps = []
        constant: list[str] = []
        for seg in segments:
            for f in seg["files"]:
                if not _varies_in_time(inputs / f, seg["fields"].get(f, [])):
                    constant.append(f)  # harmonics or fixed fields apply to every step
                    continue
                times = _times(inputs / f)
                if times is None or len(times) == 0:
                    gaps.append(f"{f}: no readable time axis")
                elif times.min() > start or times.max() < end:
                    gaps.append(
                        f"{f}: covers {times.min()} to {times.max()}, run needs {start} to {end}"
                    )
        init_name = params.get("TEMP_SALT_Z_INIT_FILE") or params.get("TS_FILE")
        if init_name and (inputs / init_name).exists():
            times = _times(inputs / init_name)
            if (
                times is not None
                and len(times)
                and abs(times[0] - start) > pd.Timedelta(days=1)
            ):
                gaps.append(
                    f"{init_name}: initial state at {times[0]}, run starts {start}"
                )
        checks.append(
            Check(
                "dates_cover_run",
                "failed" if gaps else "passed",
                f"run {start} to {end} ({source})"
                + (
                    f"; constant in time, so not date-checked: {', '.join(sorted(set(constant)))}"
                    if constant
                    else ""
                ),
                gaps,
            )
        )
    facts["run_window"] = [str(start), str(end)]

    # 3 and 4. Bathymetry: land mask and minimum depth.
    topo = inputs / params.get("TOPO_FILE", "topog.nc")
    varname = params.get("TOPO_VARNAME", "depth")
    min_depth = float(params.get("MINIMUM_DEPTH", "0").split()[0].replace("d", "e"))
    masking = params.get("MASKING_DEPTH")
    mask_depth = float(masking.split()[0].replace("d", "e")) if masking else None
    if not topo.exists():
        checks.append(Check("land_mask", "not_checked", f"{topo.name} not found"))
        checks.append(Check("minimum_depth", "not_checked", f"{topo.name} not found"))
        wet = None
    else:
        with xr.open_dataset(topo) as ds:
            if varname not in ds:
                raise OceanToolError(
                    "bad_input", f"{topo.name} has no '{varname}' (TOPO_VARNAME)"
                )
            depth = ds[varname].to_numpy().astype(float)
        wet = wet_mask(depth, min_depth, mask_depth)
        isolated, channels = narrow_cells(wet)
        kept = {tuple(c) for c in kept_cells}
        isolated = [c for c in isolated if tuple(c) not in kept]
        channels = [c for c in channels if tuple(c) not in kept]
        problems = [
            {"j": j, "i": i, "kind": "isolated wet cell"} for j, i in isolated
        ] + [{"j": j, "i": i, "kind": "one-cell channel"} for j, i in channels]
        checks.append(
            Check(
                "land_mask",
                "failed" if problems else "passed",
                f"{int(wet.sum())} wet cells of {wet.size}; {len(kept)} cells kept by the scientist are not flagged",
                problems,
            )
        )
        shallow = np.argwhere(
            (np.nan_to_num(depth) > 0) & (np.nan_to_num(depth) < min_depth)
        ).tolist()
        checks.append(
            Check(
                "minimum_depth",
                "passed" if not shallow or masking else "failed",
                f"MINIMUM_DEPTH {min_depth:g} m; MASKING_DEPTH {masking or 'unset'}; {len(shallow)} wet cells shallower than MINIMUM_DEPTH will be deepened to it"
                + (
                    ""
                    if masking
                    else " (set MASKING_DEPTH or edit the bathymetry if they should be land)"
                ),
                [{"j": j, "i": i, "depth": float(depth[j, i])} for j, i in shallow],
            )
        )
        facts["bathymetry"] = {
            "file": topo.name,
            "variable": varname,
            "shape": list(depth.shape),
            "wet_cells": int(wet.sum()),
        }

    # 5. Processor layout against the mask table.
    layout = params.get("LAYOUT")
    table_name = params.get("MASKTABLE")
    if not layout:
        checks.append(
            Check(
                "layout", "not_checked", "LAYOUT not set (MOM6 chooses one at run time)"
            )
        )
    else:
        ni, nj = (int(x) for x in re.split(r"[,\s]+", layout.strip())[:2])
        issues = []
        if wet is not None:
            ny, nx = wet.shape
            if nx % ni or ny % nj:
                issues.append(
                    f"grid {nx}x{ny} does not divide evenly by LAYOUT {ni},{nj} (allowed, but PEs differ in size)"
                )
        if table_name:
            table_path = inputs / table_name
            if not table_path.exists():
                issues.append(f"MASKTABLE {table_name} not found")
            else:
                rows = [
                    r.strip() for r in table_path.read_text().splitlines() if r.strip()
                ]
                count = int(rows[0])
                tl = [int(x) for x in re.split(r"[,\s]+", rows[1])[:2]]
                if tl != [ni, nj]:
                    issues.append(
                        f"mask table layout {tl} differs from LAYOUT {ni},{nj}"
                    )
                masked = [
                    [int(x) for x in re.split(r"[,\s]+", r)[:2]]
                    for r in rows[2 : 2 + count]
                ]
                if wet is not None:
                    ny, nx = wet.shape
                    for pi, pj in masked:
                        block = wet[
                            (pj - 1) * ny // nj : pj * ny // nj,
                            (pi - 1) * nx // ni : pi * nx // ni,
                        ]
                        if block.any():
                            issues.append(
                                f"PE ({pi},{pj}) is masked out but holds {int(block.sum())} wet cells"
                            )
        checks.append(
            Check(
                "layout",
                "failed"
                if any("wet" in i or "differs" in i or "not found" in i for i in issues)
                else "passed",
                f"LAYOUT {ni},{nj}; MASKTABLE {table_name or 'none'}",
                issues,
            )
        )
    # 6. Surface forcing that nothing mixes down: with heat and fresh-water fluxes
    # applied and no surface boundary-layer scheme, every flux stays in the top layer.
    checks.append(surface_mixing_check(params))

    facts["parameters_read"] = {
        k: params[k]
        for k in sorted(params)
        if k.startswith(
            (
                "OBC_",
                "MINIMUM_DEPTH",
                "MASKING_DEPTH",
                "TOPO_",
                "LAYOUT",
                "MASKTABLE",
                "NIGLOBAL",
                "NJGLOBAL",
            )
        )
    }
    return checks, facts
