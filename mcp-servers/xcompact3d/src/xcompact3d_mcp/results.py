"""Read what a run has produced: its progress and the forces on the body.

The solver writes ``t  drag  lift`` to ``forces.dat`` at every time step. In the
periodic-span (2D) control volume these are already coefficients per unit span
on the reference length (the body length L = 1): ``Cd = 2 F_x / (rho U^2 L)``.
In the 3D control volume they are raw forces (rho = U = 1), so the coefficients
divide ``2 F`` by the frontal area.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .case import load_case
from .errors import CaseError

_STEP = re.compile(r"Time step =\s*(\d+)\s*/\s*(\d+)")
_TIME = re.compile(r"Time unit =\s*([-+0-9.Ee]+)")
_WALL = re.compile(r"Time for this time step \(s\):\s*([-+0-9.Ee]+)")
_GEOM = re.compile(r"clio: geometry mask\s+(\d+)\s+(\d+)\s+(\d+), solid voxels (\d+)")


def _forces(run_dir: Path) -> np.ndarray:
    path = run_dir / "forces.dat"
    if not path.is_file():
        return np.zeros((0, 3))
    rows = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def case_status(run_dir: str) -> dict[str, Any]:
    """How far a run is: the last time step in its log, snapshots written, failures."""
    case = load_case(run_dir)
    plan = case["plan"]
    folder = Path(plan["run_dir"])
    log = folder / "xcompact3d.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    steps = _STEP.findall(text)
    step = int(steps[-1][0]) if steps else 0
    times = _TIME.findall(text)
    geometry = _GEOM.search(text)
    walls = [float(v) for v in _WALL.findall(text)[-50:]]
    per_step = sum(walls) / len(walls) if walls else None
    tail = text.strip().splitlines()[-12:]
    failed = any(
        word in text for word in ("Error termination", "MPI_ABORT", "SIGSEGV", "Floating-point")
    )
    nan = bool(re.search(r"\bNaN\b", text, re.IGNORECASE))
    snapshots = step // plan["ioutput"] if step else 0
    done = step >= plan["ilast"] and not failed
    bp = folder / "data.bp5"
    size = sum(p.stat().st_size for p in bp.rglob("*") if p.is_file()) if bp.is_dir() else 0
    return {
        "run_dir": str(folder),
        "started": bool(text),
        "step": step,
        "last_step": plan["ilast"],
        "time": float(times[-1]) if times else step * plan["dt"],
        "end_time": plan["end_time"],
        "snapshots_written": snapshots,
        "snapshots": plan["snapshots"],
        "data_bytes": size,
        "seconds_per_step": per_step,
        "seconds_left": per_step * (plan["ilast"] - step) if per_step and step else None,
        "geometry_loaded": bool(geometry),
        "solid_points_in_mask": int(geometry.group(4)) if geometry else None,
        "finished": done,
        "failed": failed or nan,
        "diverged": nan,
        "log_tail": tail,
        "view": case["view"],
    }


def _strouhal(t: np.ndarray, lift: np.ndarray) -> tuple[float | None, float | None]:
    """Dominant lift frequency from an FFT on a uniform resample, and its Strouhal number."""
    if len(t) < 64 or t[-1] - t[0] <= 0:
        return None, None
    uniform = np.linspace(t[0], t[-1], len(t))
    signal = np.interp(uniform, t, lift)
    signal = (signal - signal.mean()) * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=uniform[1] - uniform[0])
    spectrum[0] = 0.0
    if spectrum.max() <= 0:
        return None, None
    k = int(spectrum.argmax())
    # A peak must hold at least two periods in the window to count.
    if freqs[k] * (t[-1] - t[0]) < 2.0:
        return None, None
    return float(freqs[k]), float(freqs[k])


def read_forces(
    run_dir: str, discard_fraction: float = 0.4, frontal_length: float | None = None
) -> dict[str, Any]:
    """Drag and lift coefficients over the run, after the start-up is discarded."""
    case = load_case(run_dir)
    plan = case["plan"]
    data = _forces(Path(plan["run_dir"]))
    if len(data) == 0:
        raise CaseError(
            "no_forces",
            "The run has written no forces yet (forces.dat); check_case says how far it is.",
        )
    if not 0.0 <= discard_fraction < 1.0:
        raise CaseError("bad_parameters", "discard_fraction is from 0 up to, not including, 1.")
    t, fx, fy = data[:, 0], data[:, 1], data[:, 2]
    keep = t >= t[0] + discard_fraction * (t[-1] - t[0])
    if keep.sum() < 2:
        keep = np.ones_like(t, dtype=bool)
    if plan["span"] == "free":
        area = float(plan["body"]["frontal_area"])
        cd, cl = 2.0 * fx / area, 2.0 * fy / area
        basis = f"2 F / (rho U^2 A), A = {area:.4g} L^2 (frontal area)"
    else:
        cd, cl = fx, fy
        basis = "per unit span on the body length L (chord for a wing section)"
    frequency, strouhal = _strouhal(t[keep], cl[keep])
    height = frontal_length or float(plan["body"]["frontal_height"])
    series_step = max(1, len(t) // 400)
    result: dict[str, Any] = {
        "run_dir": plan["run_dir"],
        "samples": len(t),
        "time_range": [float(t[0]), float(t[-1])],
        "averaged_from": float(t[keep][0]),
        "basis": basis,
        "cd_mean": float(cd[keep].mean()),
        "cd_rms": float(cd[keep].std()),
        "cl_mean": float(cl[keep].mean()),
        "cl_rms": float(cl[keep].std()),
        "cl_min": float(cl[keep].min()),
        "cl_max": float(cl[keep].max()),
        "lift_to_drag": float(cl[keep].mean() / cd[keep].mean()) if cd[keep].mean() else None,
        "shedding_frequency": frequency,
        "strouhal_length": strouhal,
        "strouhal_frontal": (frequency * height) if frequency is not None else None,
        "frontal_height": height,
        "reynolds": plan["reynolds"],
        "angle_deg": plan["angle_deg"],
        "series": {
            "t": [float(v) for v in t[::series_step]],
            "cd": [float(v) for v in cd[::series_step]],
            "cl": [float(v) for v in cl[::series_step]],
        },
    }
    return result
