"""Reading G-code: layers, feature types, arcs, times and the slicer's own estimates."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from slicer_mcp.gcode.estimates import Estimates, bed_outline, parse_duration, read_comment
from slicer_mcp.gcode.features import feature_key, label, legend
from slicer_mcp.gcode.parser import GcodeError, _arc_points, parse_gcode, read_paths

from .conftest import GCODE


@pytest.fixture
def parsed(tmp_path: Path):
    return parse_gcode(GCODE, tmp_path / "layers.bin"), tmp_path / "layers.bin"


def test_the_fixture_reads_into_layers_with_the_slicers_estimates(parsed) -> None:
    result, _ = parsed
    e = result.estimates
    assert (e.print_time_s, e.first_layer_time_s, e.filament_g, e.filament_mm) == (
        201.0,
        29.0,
        0.59,
        197.23,
    )
    assert len(result.layers) == e.layer_count == 15
    assert [layer["layer"] for layer in result.layers] == list(range(15))
    assert result.layers[0]["z"] == pytest.approx(0.2)
    assert result.layers[-1]["z"] == pytest.approx(3.0)
    assert all(layer["height"] == pytest.approx(0.2) for layer in result.layers)
    assert result.bed == [[0.0, 0.0], [250.0, 0.0], [250.0, 210.0], [0.0, 210.0]]


def test_times_add_up_to_the_slicers_total_and_grams_to_its_weight(parsed) -> None:
    result, _ = parsed
    assert sum(layer["time_s"] for layer in result.layers) == pytest.approx(201.0, abs=0.05)
    grams = sum(layer["filament_g"] for layer in result.layers)
    # The purge line is extrusion too; the slicer's total includes it.
    assert grams == pytest.approx(0.59, abs=0.02)
    assert result.time_basis == "feedrate estimate scaled to the slicer's total"


def test_feature_types_and_the_print_bounds(parsed) -> None:
    result, _ = parsed
    assert set(result.features) >= {
        "external_perimeter",
        "perimeter",
        "internal_infill",
        "solid_infill",
        "top_solid_infill",
        "custom",
    }
    first = result.layers[0]["features"]
    assert first["external_perimeter"]["e_mm"] > 0
    assert first["travel"].get("grams") is None
    # Bounds are the print's: the tube is centred on the bed, the purge line is left out.
    x0, y0, x1, y1 = result.bounds
    assert x1 - x0 == pytest.approx(16.0, abs=0.6)
    assert y1 - y0 == pytest.approx(16.0, abs=0.6)
    assert y0 > 0


def test_arcs_are_split_and_paths_round_trip(parsed) -> None:
    result, binary = parsed
    arcs = sum(1 for line in GCODE.read_text().splitlines() if line.startswith(("G2 ", "G3 ")))
    assert arcs > 100
    assert result.vertices > arcs
    paths = read_paths(binary, result.layers[3])
    outer = paths["external_perimeter"]
    assert sum(outer["counts"]) * 2 == len(outer["xy"])
    assert len(outer["lines"]) == len(outer["counts"])
    xs = outer["xy"][0::2]
    ys = outer["xy"][1::2]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    radii = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys, strict=True)]
    # Two outer walls, drawn through arcs: the rim (radius about 7.8 mm) and the hole's
    # (about 3.2 mm). Every point lies on one of the two circles.
    assert all(abs(r - 7.78) < 0.3 or abs(r - 3.2) < 0.3 for r in radii)
    assert max(radii) > 7.5 and min(radii) < 3.5
    assert outer["lines"][0] > 0


def test_arc_points_follow_the_direction() -> None:
    ccw = _arc_points(1, 0, 0, 1, -1, 0, 3)
    assert ccw[0] == (1, 0) and ccw[-1] == (0, 1)
    assert all(math.hypot(x, y) == pytest.approx(1) for x, y in ccw)
    assert all(x >= -1e-9 and y >= -1e-9 for x, y in ccw)
    cw = _arc_points(1, 0, 0, 1, -1, 0, 2)
    assert any(x < -0.5 for x, _ in cw)  # the long way round


ORCA_GCODE = """\
; HEADER_BLOCK_START
; total layer number: 2
G90
M82
G92 E0
; CHANGE_LAYER
; Z_HEIGHT: 0.2
; LAYER_HEIGHT: 0.2
;LAYER_CHANGE
;Z:0.2
G1 X0 Y0 Z0.2 F3000
; FEATURE: Outer wall
G1 X10 Y0 E1 F1200
G1 X10 Y10 E2
G1 E1.2 ; retract
G1 X0 Y10 F6000
G1 E2
; FEATURE: Sparse infill
G1 X0 Y0 E3
; CHANGE_LAYER
; Z_HEIGHT: 0.4
G92 E0
G1 Z0.4
; FEATURE: Top surface
G1 X5 Y5 E0.5
; total estimated time: 1m 30s
; filament used [g] = 0.12
; filament_diameter = 1.75
; filament_density = 1.24
"""


def test_orca_markers_absolute_extrusion_and_g92(tmp_path: Path) -> None:
    gcode = tmp_path / "orca.gcode"
    gcode.write_text(ORCA_GCODE)
    result = parse_gcode(gcode, tmp_path / "orca.bin")
    assert [layer["z"] for layer in result.layers] == [0.2, 0.4]
    assert result.estimates.print_time_s == 90.0
    assert result.estimates.layer_count == 2
    first = result.layers[0]["features"]
    assert first["external_perimeter"]["e_mm"] == pytest.approx(2.0)
    assert first["internal_infill"]["e_mm"] == pytest.approx(1.0)
    assert first["travel"]["length_mm"] == pytest.approx(10.0)
    assert result.layers[1]["features"]["top_solid_infill"]["e_mm"] == pytest.approx(0.5)
    assert result.grams_per_mm == pytest.approx(math.pi * 0.875**2 / 1000 * 1.24)


def test_a_file_without_extrusion_is_refused(tmp_path: Path) -> None:
    gcode = tmp_path / "empty.gcode"
    gcode.write_text("G28\nG1 X10 Y10 F3000\n")
    with pytest.raises(GcodeError) as caught:
        parse_gcode(gcode, tmp_path / "empty.bin")
    assert caught.value.reason == "gcode_empty"


def test_durations_and_estimate_comments() -> None:
    assert parse_duration("49m 53s") == 2993
    assert parse_duration("1d 2h 3m 4s") == 93784
    assert parse_duration("") is None
    assert parse_duration("soon") is None
    e = Estimates()
    for line in (
        "; estimated printing time (normal mode) = 1h 2m 3s",
        "; filament used [g] = 1.5, 2.5",
        "; filament used [g] for wipe tower = 9",
        "; filament used [cm3] = 3.2",
        "; layer_height = 0.2",
        "; not_kept = 1",
    ):
        read_comment(line, e)
    assert e.print_time_s == 3723
    assert e.filament_g == 4.0
    assert e.filament_cm3 == 3.2
    assert e.settings == {"layer_height": "0.2"}
    assert bed_outline({"printable_area": "0x0,180x0,180x180,0x180"}) == [
        [0.0, 0.0],
        [180.0, 0.0],
        [180.0, 180.0],
        [0.0, 180.0],
    ]
    assert bed_outline({"bed_shape": "nonsense"}) is None


def test_feature_names_of_both_slicers_map_to_one_key() -> None:
    assert feature_key("External perimeter") == feature_key("Outer wall") == "external_perimeter"
    assert feature_key("Skirt/Brim") == "skirt"
    assert feature_key("Support material interface") == "support_interface"
    assert feature_key("Laser etching") == "other"
    assert label("internal_infill") == "Sparse infill"
    keys = [row["key"] for row in legend(["travel", "support"])]
    assert keys == ["support", "travel"]
