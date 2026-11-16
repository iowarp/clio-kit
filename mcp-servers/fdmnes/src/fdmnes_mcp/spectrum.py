"""Read, normalize and compare XANES spectra.

Normalization adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0),
``tools/xanes_core.py`` ``get_normalized_xanes``: a line fitted below the edge is
subtracted and the result divided by the edge step, the difference between a line
fitted above the edge and the pre-edge line at E0. Changed: E0 is always the steepest
rise (ChemGraph defaulted to 0), the windows are relative to it, and when a window
holds too few points the spectrum is returned unnormalized with the reason.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from .errors import FdmnesError

PRE_GAP, POST_GAP, MIN_POINTS = 10.0, 20.0, 3


def read_two_columns(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Energy and intensity from an FDMNES ``*_conv.txt`` or a two-column text/CSV."""
    text = path.read_text(errors="replace").splitlines()
    rows: list[tuple[float, float]] = []
    header = ""
    for line in text:
        parts = [p for p in re.split(r"[,\s;]+", line.strip()) if p]
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            if not rows:
                header = line.strip()
            continue
    if len(rows) < 5:
        raise FdmnesError(
            "unreadable_spectrum",
            f"{path.name} has fewer than 5 rows of two numbers (energy, intensity).",
            path=str(path),
        )
    arr = np.array(sorted(rows))
    return arr[:, 0], arr[:, 1], header


def edge_energy(energy: np.ndarray, mu: np.ndarray) -> float:
    """E0: the energy of the steepest rise (largest dμ/dE)."""
    return float(energy[int(np.argmax(np.gradient(mu, energy)))])


def normalize(energy: np.ndarray, mu: np.ndarray) -> dict[str, Any]:
    e0 = edge_energy(energy, mu)
    pre = energy <= e0 - PRE_GAP
    post = energy >= e0 + POST_GAP
    out: dict[str, Any] = {"e0_eV": e0, "normalized": False}
    if pre.sum() < MIN_POINTS or post.sum() < MIN_POINTS:
        out["reason"] = (
            f"too few points {PRE_GAP:g} eV below or {POST_GAP:g} eV above the edge "
            f"(E0 {e0:.2f} eV; {int(pre.sum())} below, {int(post.sum())} above); "
            "extend the energy range to normalize"
        )
        out["mu_norm"] = None
        return out
    m_pre, b_pre = np.polyfit(energy[pre], mu[pre], 1)
    m_post, b_post = np.polyfit(energy[post], mu[post], 1)
    step = (m_post * e0 + b_post) - (m_pre * e0 + b_pre)
    if not np.isfinite(step) or abs(step) < 1e-12:
        out["reason"] = "the edge step is zero; there is no edge in this range"
        out["mu_norm"] = None
        return out
    out.update(
        normalized=True,
        edge_step=float(step),
        mu_norm=(mu - (m_pre * energy + b_pre)) / step,
        pre_edge_window_eV=[float(energy[pre][0]), float(energy[pre][-1])],
        post_edge_window_eV=[float(energy[post][0]), float(energy[post][-1])],
    )
    return out


def peaks(energy: np.ndarray, y: np.ndarray, limit: int = 4) -> list[dict[str, float]]:
    """The largest local maxima above the edge: energy and height."""
    idx = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1]]
    idx.sort(key=lambda i: -y[i])
    return [{"energy_eV": float(energy[i]), "value": float(y[i])} for i in sorted(idx[:limit])]


def write_csv(path: Path, columns: dict[str, Any]) -> None:
    names = list(columns)
    n = len(next(iter(columns.values())))
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(n):
            w.writerow(["" if columns[k] is None else f"{float(columns[k][i]):.6g}" for k in names])


def spectrum_summary(energy: np.ndarray, mu: np.ndarray, norm: dict[str, Any]) -> dict[str, Any]:
    y = norm["mu_norm"] if norm["normalized"] else mu
    above = energy >= norm["e0_eV"]
    return {
        "points": len(energy),
        "energy_range_eV": [float(energy[0]), float(energy[-1])],
        "e0_eV": norm["e0_eV"],
        "normalized": norm["normalized"],
        **({"normalization_skipped": norm["reason"]} if not norm["normalized"] else {}),
        **({"edge_step": norm["edge_step"]} if norm["normalized"] else {}),
        "white_line": peaks(energy[above], y[above], 1)[0] if above.sum() > 2 else None,
        "peaks": peaks(energy[above], y[above]),
    }


def compare(
    spectra: list[tuple[str, np.ndarray, np.ndarray]], align: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize each spectrum, optionally shift each so its E0 matches the first, put
    them on the first spectrum's grid over the shared range, and measure differences."""
    prepared = []
    for label, e, mu in spectra:
        norm = normalize(e, mu)
        if not norm["normalized"]:
            raise FdmnesError(
                "invalid_input",
                f"{label} cannot be normalized: {norm['reason']}. Comparisons are of "
                "normalized spectra.",
            )
        prepared.append((label, e, norm))
    ref_label, ref_e, ref_n = prepared[0]
    shifts = {label: (ref_n["e0_eV"] - n["e0_eV"]) if align else 0.0 for label, _, n in prepared}
    lo = max(e[0] + shifts[label] for label, e, _ in prepared)
    hi = min(e[-1] + shifts[label] for label, e, _ in prepared)
    if hi - lo < 5:
        raise FdmnesError(
            "invalid_input",
            "The spectra share less than 5 eV of energy range"
            + ("" if align else "; try align true if they are on different energy scales")
            + ".",
        )
    grid = ref_e[(ref_e >= lo) & (ref_e <= hi)]
    columns: dict[str, Any] = {"energy_eV": grid}
    curves = {}
    for label, e, n in prepared:
        curves[label] = np.interp(grid, e + shifts[label], n["mu_norm"])
        columns[label] = curves[label]
    ref = curves[ref_label]
    rows = []
    for label, _, n in prepared:
        y = curves[label]
        rows.append(
            {
                "label": label,
                "e0_eV": n["e0_eV"],
                "shift_applied_eV": shifts[label],
                "r_factor_vs_first": float(np.sum((y - ref) ** 2) / np.sum(ref**2)),
                "correlation_vs_first": float(np.corrcoef(y, ref)[0, 1]),
            }
        )
    return {"range_eV": [float(lo), float(hi)], "points": len(grid), "spectra": rows}, columns
