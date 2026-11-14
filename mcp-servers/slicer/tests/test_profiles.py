"""Vendor presets: inheritance, compatibility conditions, and choosing from the person's words."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from slicer_mcp.backends.orca import (
    append_estimates,
    apply_orca_overrides,
    extract_gcode,
    read_slice_info,
)
from slicer_mcp.backends.prusa import apply_overrides, merged_settings, read_warnings
from slicer_mcp.errors import SlicerError
from slicer_mcp.profiles.catalog import PrusaCatalog
from slicer_mcp.profiles.condition import ConditionError, evaluate
from slicer_mcp.profiles.ini_bundle import read_bundle, settings_text
from slicer_mcp.profiles.orca_bundle import OrcaCatalog
from slicer_mcp.profiles.select import choose

VENDOR = """
# A small vendor bundle in PrusaSlicer's format.
[vendor]
name = Acme
config_version = 1.0.0

[printer_model:A1]
name = Acme One
default_materials = Generic PLA @A1; Generic PETG @A1

[printer:*common*]
bed_shape = 0x0,200x0,200x200,0x200
max_print_height = 180
printer_model = A1
printer_notes = PRINTER_VENDOR_ACME\\nPRINTER_MODEL_A1
binary_gcode = 1

[printer:Acme One 0.4 nozzle]
inherits = *common*
nozzle_diameter = 0.4
printer_variant = 0.4
default_print_profile = 0.20mm QUALITY @A1
default_filament_profile = "Generic PLA @A1"

[printer:Acme One 0.6 nozzle]
inherits = *common*
nozzle_diameter = 0.6
printer_variant = 0.6

[print:*common*]
perimeters = 2
fill_density = 15%

[print:0.20mm QUALITY @A1]
inherits = *common*
layer_height = 0.2
compatible_printers_condition = printer_model=="A1" and nozzle_diameter[0]==0.4

[print:0.20mm SPEED @A1]
inherits = *common*
layer_height = 0.2
compatible_printers_condition = printer_notes=~/.*MODEL_A1.*/ && nozzle_diameter[0]==0.4

[print:0.30mm DRAFT @A1 0.6]
inherits = *common*
layer_height = 0.3
compatible_printers_condition = printer_model=="A1" and nozzle_diameter[0]==0.6

[filament:*PLA*]
filament_type = PLA
filament_density = 1.24

[filament:Generic PLA @A1]
inherits = *PLA*
alias = Generic PLA
compatible_printers_condition = printer_model=="A1" and nozzle_diameter[0]!=0.6

[filament:Generic PLA @A1 0.6]
inherits = Generic PLA @A1
alias = Generic PLA
compatible_printers_condition = printer_model=="A1" and nozzle_diameter[0]==0.6

[filament:Fancy PLA @A1]
inherits = *PLA*
compatible_printers_condition = printer_model=="A1"

[filament:Generic PETG @A1]
filament_type = PETG
compatible_printers_condition = printer_model=="A1" and ! (nozzle_diameter[0]==0.6)
"""


@pytest.fixture
def catalog(tmp_path: Path) -> PrusaCatalog:
    (tmp_path / "Acme.ini").write_text(VENDOR)
    (tmp_path / "Other.ini").write_text(
        "[vendor]\nname = Other\n[printer:Other Box]\nprinter_model = BOX\nnozzle_diameter = 0.4\n"
    )
    return PrusaCatalog(tmp_path)


def test_inheritance_flattens_in_order(tmp_path: Path) -> None:
    (tmp_path / "Acme.ini").write_text(VENDOR)
    bundle = read_bundle(tmp_path / "Acme.ini")
    assert bundle.vendor == "Acme"
    flat = bundle.resolve("filament", "Generic PLA @A1 0.6")
    assert flat["filament_type"] == "PLA" and flat["filament_density"] == "1.24"
    assert "inherits" not in flat
    text = settings_text({**flat, "alias": "x", "renamed_from": "y"})
    assert "alias" not in text and "filament_type = PLA" in text


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("", True),
        ('printer_model=="A1"', True),
        ("nozzle_diameter[0]>=0.4 and nozzle_diameter[0]<0.5", True),
        ("printer_notes=~/.*MODEL_A1.*/", True),
        ("printer_notes!~/.*MODEL_A1.*/", False),
        ('not (printer_model=="B") || num_extruders>1', True),
        ("! single_extruder_multi_material", True),
        ('num_extruders==2 or (printer_model=="A1" and nozzle_diameter[0]==0.4)', True),
    ],
)
def test_conditions(expression: str, expected: bool) -> None:
    config = {
        "printer_model": "A1",
        "nozzle_diameter": "0.4,0.4",
        "printer_notes": "PRINTER_VENDOR_ACME\\nPRINTER_MODEL_A1",
        "num_extruders": "1",
        "single_extruder_multi_material": "0",
    }
    assert evaluate(expression, config) is expected


def test_a_condition_outside_the_subset_raises() -> None:
    with pytest.raises(ConditionError):
        evaluate("printer_model == ", {"printer_model": "A"})
    with pytest.raises(ConditionError):
        evaluate("unknown_key == 1", {})


def test_words_choose_presets_and_say_how(catalog: PrusaCatalog) -> None:
    choice = choose(catalog, "acme one", "PLA", "0.2 mm")
    assert choice.printer.name == "Acme One 0.4 nozzle"  # the 0.4 nozzle by default
    assert choice.print_preset.name == "0.20mm QUALITY @A1"  # the printer's default first
    assert choice.filament.name == "Generic PLA @A1"  # the printer's default material
    assert choice.alternatives["printer"] == ["Acme One 0.6 nozzle"]
    assert "0.20mm SPEED @A1" in choice.alternatives["quality"]
    assert any("matched 'Acme One 0.4 nozzle'" in note for note in choice.notes)
    speed = choose(catalog, "Acme One 0.4 nozzle", "Fancy PLA @A1", "speed")
    assert (speed.print_preset.name, speed.filament.name) == ("0.20mm SPEED @A1", "Fancy PLA @A1")
    assert not any("printer" in note for note in speed.notes)


def test_a_nozzle_in_the_words_and_its_material_variant(catalog: PrusaCatalog) -> None:
    choice = choose(catalog, "Acme 0.6 nozzle", None, None)
    assert choice.printer.name == "Acme One 0.6 nozzle"
    assert choice.print_preset.name == "0.30mm DRAFT @A1 0.6"
    # The default "Generic PLA @A1" names a material; this nozzle's variant is chosen.
    assert choice.filament.name == "Generic PLA @A1 0.6"
    assert any("no material given" in note for note in choice.notes)


def test_no_match_is_typed_with_close_names(catalog: PrusaCatalog) -> None:
    with pytest.raises(SlicerError) as caught:
        choose(catalog, "Bambu X1", "PLA", None)
    assert caught.value.reason == "profile_not_found"
    assert "list_printers" in str(caught.value)
    with pytest.raises(SlicerError) as caught:
        choose(catalog, "Acme One", "PLA", "0.07 mm")
    assert "layer heights 0.2 mm" in str(caught.value)
    with pytest.raises(SlicerError) as caught:
        choose(catalog, "Acme One", "nylon", None)
    assert "Materials for Acme One 0.4 nozzle: PETG, PLA." in str(caught.value)


def test_prusa_settings_overrides_and_warnings(catalog: PrusaCatalog) -> None:
    choice = choose(catalog, "Acme One", "PLA", "0.2")
    settings = merged_settings(choice)
    assert "compatible_printers_condition" not in settings
    assert settings["print_settings_id"] == "0.20mm QUALITY @A1"
    applied = apply_overrides(settings, {"fill_density": "40%", "perimeters": 4})
    assert applied == ["fill_density = 40%", "perimeters = 4"]
    with pytest.raises(SlicerError) as caught:
        apply_overrides(settings, {"fill_densty": "40%"})
    assert caught.value.reason == "override_unknown"
    assert "fill_density" in str(caught.value)
    output = (
        "65 => Searching support spots\n"
        "print warning: Detected print stability issues:\n\nwhistle\nLong bridging extrusions\n\n"
        "Consider enabling supports.\n88 => Estimating curled extrusions\n"
    )
    assert read_warnings(output) == [
        (
            "Detected print stability issues: whistle. Long bridging extrusions. "
            "Consider enabling supports."
        )
    ]


def write_orca_tree(root: Path) -> None:
    vendor = {
        "name": "Acme",
        "machine_list": [
            {"name": "fdm_machine_common", "sub_path": "machine/common.json"},
            {"name": "Acme One 0.4 nozzle", "sub_path": "machine/one.json"},
        ],
        "process_list": [
            {"name": "fdm_process_common", "sub_path": "process/common.json"},
            {"name": "0.20mm Standard @Acme", "sub_path": "process/020.json"},
        ],
        "filament_list": [{"name": "Acme PLA", "sub_path": "filament/pla.json"}],
    }
    (root / "Acme.json").write_text(json.dumps(vendor))
    files = {
        "machine/common.json": {
            "name": "fdm_machine_common",
            "instantiation": "false",
            "printable_area": ["0x0", "220x0", "220x220", "0x220"],
        },
        "machine/one.json": {
            "name": "Acme One 0.4 nozzle",
            "inherits": "fdm_machine_common",
            "instantiation": "true",
            "printer_model": "Acme One",
            "nozzle_diameter": ["0.4"],
            "default_print_profile": "0.20mm Standard @Acme",
            "default_filament_profile": ["Acme PLA"],
        },
        "process/common.json": {
            "name": "fdm_process_common",
            "instantiation": "false",
            "wall_loops": "2",
            "sparse_infill_density": "15%",
        },
        "process/020.json": {
            "name": "0.20mm Standard @Acme",
            "inherits": "fdm_process_common",
            "instantiation": "true",
            "layer_height": "0.2",
            "compatible_printers": ["Acme One 0.4 nozzle"],
        },
        "filament/pla.json": {
            "name": "Acme PLA",
            "instantiation": "true",
            "filament_type": ["PLA"],
            "compatible_printers": [],
        },
    }
    for rel, data in files.items():
        path = root / "Acme" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))


def test_orca_tree_flattens_and_chooses(tmp_path: Path) -> None:
    write_orca_tree(tmp_path)
    catalog = OrcaCatalog(tmp_path)
    assert [p.name for p in catalog.printers()] == ["Acme One 0.4 nozzle"]
    choice = choose(catalog, "acme one", "PLA", "0.2 mm")
    assert (choice.print_preset.name, choice.filament.name) == ("0.20mm Standard @Acme", "Acme PLA")
    flat = catalog.flat("process", "0.20mm Standard @Acme")
    assert flat["wall_loops"] == "2" and "inherits" not in flat
    machine = catalog.flat("machine", "Acme One 0.4 nozzle")
    applied = apply_orca_overrides([dict(machine), dict(flat)], {"wall_loops": 3})
    assert applied == ["wall_loops = 3"]
    with pytest.raises(SlicerError):
        apply_orca_overrides([dict(flat)], {"walls": 3})


def test_orca_3mf_gcode_and_slice_info(tmp_path: Path) -> None:
    info = (
        '<?xml version="1.0"?><config><plate><metadata key="index" value="1"/>'
        '<metadata key="prediction" value="3723"/><metadata key="weight" value="4.20"/>'
        '<filament id="1" type="PLA" used_m="1.40" used_g="4.20"/></plate></config>'
    )
    assert read_slice_info(info) == {"prediction": 3723.0, "weight": 4.2, "used_m": 1.4}
    archive = tmp_path / "slice.3mf"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Metadata/plate_1.gcode", "; FEATURE: Outer wall\nG1 X1 Y1 E1\n")
        zf.writestr("Metadata/slice_info.config", info)
    gcode = tmp_path / "slice.gcode"
    estimates = extract_gcode(archive, gcode)
    append_estimates(gcode, estimates)
    text = gcode.read_text()
    assert "; estimated printing time (normal mode) = 1h 2m 3s" in text
    assert "; total filament used [g] = 4.2" in text
    empty = tmp_path / "empty.3mf"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("3D/3dmodel.model", "")
    with pytest.raises(SlicerError) as caught:
        extract_gcode(empty, tmp_path / "none.gcode")
    assert caught.value.reason == "slice_failed"
