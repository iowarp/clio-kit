"""Read the loading from gRASPA's output.

Two output flavours are recognized and the result says which one was read:

- ``graspa`` (snurr-group/gRASPA, CUDA or HIP): block-average sections per component,
  ``=== BLOCK AVERAGES (LOADING: mol/kg) ===`` then ``COMPONENT [i] (<name>)`` and
  ``Overall: Average: X, ErrorBar: Y``; the same for ``# MOLECULES``. The adsorbate's
  block is taken by its name, never by position.
- ``graspa-sycl`` (the SYCL port ChemGraph runs): a ``UnitCells`` line and one
  ``Overall: Average: N`` molecule count, converted to mol/kg with the framework mass
  (adapted from ChemGraph's ``_read_graspa_sycl_output``, Apache-2.0).
"""

from __future__ import annotations

import math
import re
from typing import Any

from .errors import GraspaError

SECTION = re.compile(r"=+\s*BLOCK AVERAGES \(LOADING: ([^)]+)\)\s*=+")
COMPONENT = re.compile(r"COMPONENT \[(\d+)\] \((.+)\)")
OVERALL = re.compile(r"Overall: Average:\s*([-+\w.]+),?\s*(?:ErrorBar:\s*([-+\w.]+))?")


def _number(token: str | None) -> float | None:
    """A printed number; gRASPA prints ``-nan`` for an undefined error bar."""
    if token is None:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _sections(text: str) -> dict[str, dict[str, tuple[float, float | None]]]:
    """{unit: {component name: (average, error bar)}} from the block-average sections."""
    out: dict[str, dict[str, tuple[float, float | None]]] = {}
    unit, comp = None, None
    for line in text.splitlines():
        if m := SECTION.search(line):
            unit, comp = m.group(1).strip(), None
            out.setdefault(unit, {})
            continue
        if line.startswith("=") and "BLOCK AVERAGES" in line:
            unit = None  # an energy or heat section
            continue
        if unit is None:
            continue
        if m := COMPONENT.match(line.strip()):
            comp = m.group(2).strip()
            continue
        if comp and (m := OVERALL.search(line)) and comp not in out[unit]:
            avg = _number(m.group(1))
            if avg is not None:
                out[unit][comp] = (avg, _number(m.group(2)))
    return out


def _finite(value: float, what: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise GraspaError("unreadable_output", f"The {what} read is {value}, not a loading.")
    return value


def read_loading(
    text: str, adsorbate: str, unit_cells: list[int] | None, framework_mass_g_mol: float | None
) -> dict[str, Any]:
    sections = _sections(text)
    molkg = sections.get("mol/kg", {})
    if molkg:
        if adsorbate not in molkg:
            raise GraspaError(
                "unreadable_output",
                f"gRASPA's loading table has components {sorted(molkg)} but not {adsorbate}.",
            )
        avg, err = molkg[adsorbate]
        mol = sections.get("# MOLECULES", {}).get(adsorbate)
        cells = _unit_cells(text) or unit_cells
        out: dict[str, Any] = {
            "flavour": "graspa",
            "uptake_mol_kg": _finite(avg, "uptake"),
            "uptake_error_mol_kg": err,
        }
        if mol:
            out["molecules_in_box"] = mol[0]
            out["molecules_error"] = mol[1]
            if cells:
                out["molecules_per_unit_cell"] = mol[0] / math.prod(cells)
        if cells:
            out["unit_cells"] = cells
        for unit in ("mg/g", "g/L"):
            if adsorbate in sections.get(unit, {}):
                out[f"uptake_{unit.replace('/', '_per_')}"] = sections[unit][adsorbate][0]
        if m := re.search(r"Averaged Henry Coefficient \[mol/kg/Pa\]:\s*([-+\deE.]+)", text):
            out["henry_mol_kg_Pa"] = float(m.group(1))
        return out
    return _read_sycl(text, unit_cells, framework_mass_g_mol)


def _unit_cells(text: str) -> list[int] | None:
    m = re.search(r"Number of Unit Cells:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    return [int(float(v)) for v in m.groups()] if m else None


def _read_sycl(
    text: str, unit_cells: list[int] | None, framework_mass_g_mol: float | None
) -> dict[str, Any]:
    cell_line = overall = None
    for line in text.splitlines():
        if "UnitCells" in line:
            cell_line = line
        elif "Overall: Average:" in line:
            overall = line
    if overall is None:
        raise GraspaError(
            "unreadable_output",
            "gRASPA printed no loading (no block-average table and no 'Overall: Average:' "
            "line); the run may have stopped early.",
        )
    cells = unit_cells
    if cell_line is not None:
        tokens = cell_line.split()[-3:]
        try:
            parsed = [float(t) for t in tokens]
            if all(v > 0 and v.is_integer() for v in parsed):
                cells = [int(v) for v in parsed]
        except ValueError:
            pass
    if not cells or framework_mass_g_mol is None:
        raise GraspaError(
            "unreadable_output",
            "The SYCL output gives a molecule count; converting it to mol/kg needs the unit "
            "cells and the framework (read the run folder, which has both).",
        )
    molecules = _finite(
        float(overall.split("Overall: Average:", 1)[1].split()[0].rstrip(",;")), "molecule count"
    )
    mass = framework_mass_g_mol * math.prod(cells)
    return {
        "flavour": "graspa-sycl",
        "uptake_mol_kg": molecules / mass * 1000.0,
        "uptake_error_mol_kg": None,
        "molecules_in_box": molecules,
        "molecules_per_unit_cell": molecules / math.prod(cells),
        "unit_cells": cells,
    }
