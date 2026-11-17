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


def downscale(request: dict) -> dict:
    import importlib

    import numpy as np
    import torch
    import xarray as xr
    from earth2studio.data import fetch_data
    from earth2studio.models.dx import CorrDiffTaiwan

    device = _device(request)
    model = CorrDiffTaiwan.load_model(CorrDiffTaiwan.load_default_package()).to(device)
    model.number_of_samples = int(request.get("samples", 1))
    smod, scls = SOURCES[request["source"]]
    data = getattr(importlib.import_module(smod), scls)()
    coords = model.input_coords()
    x, c = fetch_data(
        source=data,
        time=np.array([np.datetime64(request["time"])]),
        variable=coords["variable"],
        device=device,
    )
    with torch.no_grad():
        y, yc = model(x, c)
    da = xr.DataArray(
        y.cpu().numpy(),
        coords={k: v for k, v in yc.items() if k not in ("lat", "lon")},
        dims=list(yc.keys()),
    )
    da = (
        da.assign_coords(
            lat=(("ilat", "ilon"), yc["lat"]), lon=(("ilat", "ilon"), yc["lon"])
        )
        if np.ndim(yc["lat"]) == 2
        else da
    )
    out = Path(request["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset(name="downscaled").to_netcdf(out)
    return {
        "status": "success",
        "output": str(out),
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
