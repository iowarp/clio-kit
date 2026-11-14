"""Shared fixtures: a slice record built from the checked-in G-code, no slicer needed."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from slicer_mcp.gcode.parser import parse_gcode
from slicer_mcp.store import new_slice_id, save_record, slice_dir

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GCODE = FIXTURES / "tube_mk4_pla_0.20.gcode"
MODEL = FIXTURES / "tube.stl"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty server home (saved slices, the slicer's settings folder)."""
    folder = tmp_path / "slicer-home"
    monkeypatch.setenv("SLICER_MCP_HOME", str(folder))
    return folder


def make_record(home: Path, model: Path = MODEL) -> dict[str, Any]:
    """Save a slice of the fixture G-code as ``slice`` would, without running a slicer."""
    del home  # the store reads SLICER_MCP_HOME
    slice_id = new_slice_id()
    folder = slice_dir(slice_id)
    folder.mkdir(parents=True)
    gcode = folder / "slice.gcode"
    shutil.copyfile(GCODE, gcode)
    parsed = parse_gcode(gcode, folder / "layers.bin")
    record = {
        "slice_id": slice_id,
        "model": str(model),
        "model_name": model.name,
        "slicer": {"name": "PrusaSlicer", "version": "2.7.2", "program": "/usr/bin/prusa-slicer"},
        "printer": "Original Prusa MK4 0.4 nozzle",
        "quality": "0.20mm QUALITY @MK4 0.4",
        "material": "Prusament PLA @PG",
        "material_type": "PLA",
        "request": {
            "printer": "Prusa MK4",
            "material": "PLA",
            "quality": "0.2 mm",
            "overrides": {},
            "rotate": {},
        },
        "matched": [],
        "alternatives": {},
        "overrides": [],
        "warnings": ["Detected print stability issues: tube. Long bridging extrusions."],
        "estimates": parsed.estimates.as_dict(),
        "filament_m": 0.197,
        "layer_height_mm": [0.2],
        "z_max_mm": parsed.layers[-1]["z"],
        "bed": parsed.bed,
        "bounds": parsed.bounds,
        "features": parsed.features,
        "time_basis": parsed.time_basis,
        "gcode": str(gcode),
        "gcode_bytes": gcode.stat().st_size,
        "gcode_lines": parsed.lines,
        "vertices": parsed.vertices,
        "seconds": 0.1,
        "layers": parsed.layers,
        "view": {},
    }
    save_record(record)
    return record


@pytest.fixture
def saved(home: Path) -> dict[str, Any]:
    return make_record(home)
