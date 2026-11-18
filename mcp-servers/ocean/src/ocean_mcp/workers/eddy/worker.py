# SPDX-License-Identifier: GPL-3.0-or-later
"""py-eddy-tracker worker: identify eddies in daily sea level and track them.

Runs in its own Python 3.10 environment (py-eddy-tracker 3.6.1 pins
numpy<1.23); the ocean MCP server starts it as ``python worker.py
request.json`` and reads the JSON it prints. For each day it high-pass
filters the height field (Bessel filter at a stated wavelength), computes
geostrophic velocities and identifies closed-contour eddies (Mason et al.
2014 / Pegliasco et al. 2022 method as implemented by py-eddy-tracker);
then it links days into tracks with ``Correspondances``.

The one-cell border of the domain is masked before identification: an eddy
touching the border cannot have a closed contour, and py-eddy-tracker needs
a mask array, which a field without missing cells does not carry.

This file is licensed GPL-3.0-or-later because it imports py-eddy-tracker.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

FIELDS = (
    "lon",
    "lat",
    "time",
    "amplitude",
    "radius_s",
    "radius_e",
    "speed_average",
    "effective_area",
)


def _mask_border(grid, name: str) -> None:
    values = grid.grid(name)
    mask = np.ma.getmaskarray(values).copy()
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    grid.vars[name] = np.ma.array(np.ma.getdata(values), mask=mask)


def identify(request: dict, work: Path) -> tuple[dict[str, list[str]], int, int]:
    from py_eddy_tracker.dataset.grid import RegularGridDataset

    variable = request["variable"]
    x_name, y_name, t_name = request["x"], request["y"], request["time"]
    with xr.open_dataset(request["input"]) as ds:
        times = pd.to_datetime(ds[t_name].to_numpy())
        if request.get("start"):
            times = times[times >= pd.Timestamp(request["start"])]
        if request.get("end"):
            times = times[times <= pd.Timestamp(request["end"])]
        all_times = pd.to_datetime(ds[t_name].to_numpy())
    files: dict[str, list[str]] = {"anticyclonic": [], "cyclonic": []}
    n_a = n_c = 0
    for when in times:
        index = int(np.nonzero(all_times == when)[0][0])
        grid = RegularGridDataset(
            request["input"], x_name, y_name, indexs={t_name: index}
        )
        _mask_border(grid, variable)
        if request.get("wavelength_km"):
            grid.bessel_high_filter(variable, float(request["wavelength_km"]), order=1)
        grid.add_uv(variable)
        for name in ("u", "v"):
            values = grid.grid(name)
            grid.vars[name] = np.ma.array(
                np.ma.getdata(values), mask=np.ma.getmaskarray(values)
            )
        a, c = grid.eddy_identification(
            variable,
            "u",
            "v",
            datetime(when.year, when.month, when.day),
            step=float(request.get("step_m", 0.002)),
            shape_error=float(request.get("shape_error", 55)),
            pixel_limit=tuple(request.get("pixel_limit", (5, 2000))),
        )
        stamp = when.strftime("%Y%m%d")
        for kind, obs in (("anticyclonic", a), ("cyclonic", c)):
            path = work / f"{kind}_{stamp}.nc"
            obs.write_file(filename=str(path))
            files[kind].append(str(path))
        n_a += len(a)
        n_c += len(c)
    return files, n_a, n_c


def track(paths: list[str], virtual: int):
    from py_eddy_tracker.tracking import Correspondances

    corr = Correspondances(datasets=paths, virtual=virtual)
    corr.track()
    corr.prepare_merging()
    eddies = corr.merge(raw_data=False)
    # Days an eddy was missed (up to ``virtual`` of them) are slots left at zero by
    # the merge: time 0 (1 January 1950), 0°E 0°N. As py-eddy-tracker's own tracking
    # script does, flag them virtual and fill them by interpolation between the
    # observations either side, so no track jumps to the Gulf of Guinea in 1950.
    if len(eddies):
        eddies["virtual"][:] = eddies["time"] == 0
        eddies.filled_by_interpolation(eddies["virtual"] == 1)
    return eddies


def main(request_path: str) -> int:
    request = json.loads(Path(request_path).read_text())
    out_dir = Path(request["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="eddies-") as tmp:
        files, n_a, n_c = identify(request, Path(tmp))
        for polarity, paths in files.items():
            if len(paths) < 2:
                continue
            tracks = track(paths, int(request.get("virtual", 4)))
            if len(tracks) == 0:
                continue
            frame = pd.DataFrame(
                {
                    name: np.asarray(tracks[name])
                    for name in FIELDS
                    if name in tracks.fields
                }
            )
            frame["track"] = np.asarray(tracks["track"])
            frame["virtual"] = np.asarray(tracks["virtual"]).astype(bool)
            frame["polarity"] = polarity
            rows.append(frame)
    if rows:
        table = pd.concat(rows, ignore_index=True)
        # py-eddy-tracker times are days since 1950-01-01.
        table["time"] = pd.Timestamp("1950-01-01") + pd.to_timedelta(
            table["time"], unit="D"
        )
        table["track"] = table["polarity"].str[0].str.upper() + table["track"].astype(
            str
        )
        table = table.rename(
            columns={
                "lon": "longitude",
                "lat": "latitude",
                "radius_s": "speed_radius_m",
                "radius_e": "effective_radius_m",
            }
        )
        table["longitude"] = ((table["longitude"] + 180) % 360) - 180
    else:
        table = pd.DataFrame(
            columns=["track", "polarity", "time", "longitude", "latitude", "amplitude"]
        )
    path = out_dir / f"{request['stem']}.tracks.csv"
    table.to_csv(path, index=False)
    import py_eddy_tracker

    print(
        json.dumps(
            {
                "status": "success",
                "tracks_path": str(path),
                "identified": {"anticyclonic": n_a, "cyclonic": n_c},
                "tracks": int(table["track"].nunique()) if len(table) else 0,
                "py_eddy_tracker_version": getattr(
                    py_eddy_tracker, "__version__", "3.6.1"
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
