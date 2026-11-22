# SPDX-License-Identifier: Apache-2.0
"""Earth2Studio worker: AI forecasts and CorrDiff downscaling.

Runs in its own environment (torch and the model packages are large); the
weather MCP server starts it as ``python worker.py request.json`` and reads
the JSON line it prints. ``task`` is ``forecast`` or ``downscale``. The
device is CUDA when available; on CPU the run proceeds only if the request
says ``allow_cpu`` (CPU inference of 0.25° models is slow), and the result
always reports the device it used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # the worker imports its libraries where it uses them
    import xarray as xr

MODELS = {
    "fcn": (
        "earth2studio.models.px",
        "FCN",
        "FourCastNet (AFNO), Pathak et al. 2022; NVIDIA, Apache-2.0 checkpoint via Hugging Face nvidia/fourcastnet1",
    ),
    "dlwp": (
        "earth2studio.models.px",
        "DLWP",
        "DLWP (cubed sphere), Weyn et al. 2021; NVIDIA checkpoint, Apache-2.0",
    ),
    "persistence": ("earth2studio.models.px", "Persistence", "persistence baseline"),
}
SOURCES = {
    "arco": ("earth2studio.data", "ARCO"),
    "ifs": ("earth2studio.data", "IFS"),
    "gfs": ("earth2studio.data", "GFS"),
}


def _device(request: dict):
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if not request.get("allow_cpu"):
        raise RuntimeError("no CUDA device; pass allow_cpu to run on the CPU (slow)")
    return torch.device("cpu")


def forecast(request: dict) -> dict:
    import importlib

    from earth2studio.io import NetCDF4Backend
    from earth2studio.run import deterministic

    name = request["model"]
    module, cls, citation = MODELS[name]
    model_cls = getattr(importlib.import_module(module), cls)
    if name == "persistence":
        raise RuntimeError(
            "persistence needs explicit variables and grid; use a trained model"
        )
    model = model_cls.load_model(model_cls.load_default_package())
    smod, scls = SOURCES[request["source"]]
    data = getattr(importlib.import_module(smod), scls)()
    device = _device(request)
    out = Path(request["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    io = NetCDF4Backend(str(out), backend_kwargs={"mode": "w"})
    from collections import OrderedDict

    import numpy as np

    output_coords = OrderedDict()
    if request.get("variables"):
        output_coords["variable"] = np.array(request["variables"])
    deterministic(
        [request["time"]],
        int(request["steps"]),
        model,
        data,
        io,
        output_coords=output_coords,
        device=device,
        verbose=False,
    )
    io.close() if hasattr(io, "close") else None
    return {
        "status": "success",
        "output": str(out),
        "model": name,
        "citation": citation,
        "device": str(device),
        "source": request["source"],
    }


def _fields(names, values, lat, lon) -> xr.Dataset:
    """Variables on a grid, with 10 m wind speed added when both components are there."""
    import numpy as np
    import xarray as xr

    coords: dict[str, object]
    if np.ndim(lat) == 1:
        dims, coords = ("lat", "lon"), {"lat": np.asarray(lat), "lon": np.asarray(lon)}
    else:
        dims = ("y", "x")
        coords = {"lat": (dims, np.asarray(lat)), "lon": (dims, np.asarray(lon))}
    ds = xr.Dataset(
        {str(n): (dims, v) for n, v in zip(names, values, strict=True)}, coords=coords
    )
    if "u10m" in ds and "v10m" in ds:
        # np.hypot of two DataArrays returns a DataArray (__array_ufunc__); numpy's
        # stub only knows the ndarray result.
        speed = cast("xr.DataArray", np.hypot(ds["u10m"], ds["v10m"]))
        ds["wind_speed"] = speed.assign_attrs(
            units="m s-1", long_name="10 m wind speed"
        )
    return ds


def _regular(ds: xr.Dataset, step: float) -> xr.Dataset:
    """A curvilinear field on a regular lat-lon grid of ``step`` degrees (linear)."""
    import numpy as np
    import xarray as xr
    from scipy.interpolate import griddata

    lat2, lon2 = ds["lat"].values, ds["lon"].values
    lats = np.arange(np.ceil(lat2.min() / step) * step, lat2.max(), step)
    lons = np.arange(np.ceil(lon2.min() / step) * step, lon2.max(), step)
    glon, glat = np.meshgrid(lons, lats)
    points = np.column_stack([lon2.ravel(), lat2.ravel()])
    out = xr.Dataset(coords={"lat": lats, "lon": lons})
    for name, var in ds.data_vars.items():
        values = griddata(points, var.values.ravel(), (glon, glat), method="linear")
        out[name] = (("lat", "lon"), values.astype("float32"), var.attrs)
    return out


def downscale(request: dict) -> dict:
    """CorrDiff Taiwan from a coarse source; the coarse input and the 2 km output.

    The input is fetched at lead 0 and interpolated onto the model's own input grid,
    as the model expects (the global field was passed before, and the model refused
    it: "Coordinate systems for required dim lon are not the same"). Three files: the
    model's curvilinear output, the same on a regular grid for maps, and the coarse
    input it was given, so the two can be compared on one map.
    """
    import importlib
    from collections import OrderedDict

    import numpy as np
    import torch
    from earth2studio.data import fetch_data
    from earth2studio.models.dx import CorrDiffTaiwan

    device = _device(request)
    model = CorrDiffTaiwan.load_model(CorrDiffTaiwan.load_default_package()).to(device)
    model.number_of_samples = int(request.get("samples", 1))
    smod, scls = SOURCES[request["source"]]
    data = getattr(importlib.import_module(smod), scls)()
    ic = model.input_coords()
    x, _ = fetch_data(
        source=data,
        time=np.array([np.datetime64(request["time"])]),
        variable=ic["variable"],
        lead_time=np.array([np.timedelta64(0, "h")]),
        device=device,
        # Earth2Studio 0.20 reads the target grid from ``_lat``/``_lon``.
        interp_to=OrderedDict(_lat=ic["lat"], _lon=ic["lon"]),
        interp_method="linear",
    )
    x = x[:, 0]  # (time as batch, variable, lat, lon)
    coords = OrderedDict(
        batch=np.arange(x.shape[0]),
        variable=ic["variable"],
        lat=ic["lat"],
        lon=ic["lon"],
    )
    with torch.no_grad():
        y, yc = model(x, coords)
    fine = y.reshape(-1, *y.shape[-3:]).cpu().numpy()
    out = Path(request["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    native = _fields(list(yc["variable"]), fine[0], yc["lat"], yc["lon"])
    if fine.shape[0] > 1:  # more samples: their spread, beside the first
        for i, name in enumerate(yc["variable"]):
            native[f"{name}_spread"] = (native[str(name)].dims, fine[:, i].std(axis=0))
    native.to_netcdf(out)
    mapped = out.with_name(out.stem + "-map.nc")
    _regular(native, 0.02).to_netcdf(mapped)
    coarse = out.with_name(out.stem + "-coarse.nc")
    _fields(list(ic["variable"]), x[0].cpu().numpy(), ic["lat"], ic["lon"]).to_netcdf(
        coarse
    )
    return {
        "status": "success",
        "output": str(out),
        "map_output": str(mapped),
        "coarse_output": str(coarse),
        "model": "CorrDiffTaiwan",
        "citation": "Mardani et al. 2023 (arXiv:2309.15214); NVIDIA CorrDiff inference package (NGC)",
        "device": str(device),
        "samples": model.number_of_samples,
    }


def main(path: str) -> int:
    request = json.loads(Path(path).read_text())
    task = request["task"]
    try:
        reply = forecast(request) if task == "forecast" else downscale(request)
    except Exception as exc:  # noqa: BLE001 - reported as JSON so the server can type it
        reply = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(reply))
    return 0 if reply.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
