"""What the tools do: find presets, slice, read a slice's numbers, export, compare.

Each function returns plain data (the tool's ``structuredContent``) and raises
:class:`~.errors.SlicerError` with a reason code and words for a person. The agent-facing
text is written by :mod:`.present`.
"""

from __future__ import annotations

import csv
import io
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .backends.locate import ORCA, PRUSA, Slicer, locate
from .errors import SlicerError
from .gcode.estimates import setting_numbers
from .gcode.features import TRAVEL, label, legend
from .gcode.parser import GcodeError, parse_gcode
from .profiles.catalog import Catalog, PrusaCatalog
from .profiles.select import choose, choose_print
from .store import datadir, load_record, new_slice_id, save_record, slice_dir

#: The row of time no toolpath type owns: retracting, priming and Z moves.
OTHER_MOVES = "other_moves"
OTHER_LABEL = "Retracts and Z moves"
MODEL_SUFFIXES = {".stl", ".3mf", ".step", ".stp", ".obj", ".amf"}
LIST_LIMIT = 60


@lru_cache(maxsize=4)
def _catalog_for(kind: str, folder: str) -> Catalog:
    if kind == PRUSA:
        return PrusaCatalog(Path(folder))
    from .profiles.orca_bundle import OrcaCatalog

    return OrcaCatalog(Path(folder))


def catalog(slicer: Slicer) -> Catalog:
    if slicer.profiles is None:
        raise SlicerError(
            "profiles_missing",
            f"{slicer.describe()} was found at {slicer.binary}, but not the folder of "
            "printer presets it ships (resources/profiles). Set SLICER_PROFILES to that "
            "folder.",
        )
    return _catalog_for(slicer.kind, str(slicer.profiles))


# ------------------------------------------------------------------ list_printers


def list_printers(vendor: str | None = None, printer: str | None = None) -> dict[str, Any]:
    slicer = locate()
    cat = catalog(slicer)
    printers = cat.printers()
    out: dict[str, Any] = {"slicer": _slicer_info(slicer)}
    if printer:
        from .profiles.select import choose_printer

        chosen, others = choose_printer(cat, printer)
        prints = cat.prints_for(chosen)
        default_print, _ = choose_print(cat, chosen, None)
        filaments = cat.filaments_for(chosen, default_print)
        out["printer"] = {
            "name": chosen.name,
            "vendor": chosen.vendor,
            "model": chosen.model_name or chosen.model,
            "nozzle_mm": chosen.nozzle,
            "bed": chosen.settings.get("bed_shape") or chosen.settings.get("printable_area"),
            "max_height_mm": chosen.settings.get("max_print_height")
            or chosen.settings.get("printable_height"),
        }
        out["also_matched"] = others
        out["qualities"] = [
            {"name": p.name, "layer_height_mm": p.settings.get("layer_height")}
            for p in sorted(
                prints, key=lambda p: (float(p.settings.get("layer_height") or 0), p.name)
            )
        ]
        out["default_quality"] = default_print.name
        types: dict[str, list[str]] = {}
        for f in filaments:
            types.setdefault(f.settings.get("filament_type", "other") or "other", []).append(f.name)
        out["materials"] = {k: sorted(set(v))[:12] for k, v in sorted(types.items())}
        out["material_counts"] = {k: len(set(v)) for k, v in sorted(types.items())}
        return out
    vendors: dict[str, int] = {}
    for p in printers:
        vendors[p.vendor] = vendors.get(p.vendor, 0) + 1
    out["vendors"] = [{"vendor": v, "printers": n} for v, n in sorted(vendors.items())]
    if vendor:
        wanted = vendor.strip().lower()
        chosen_list = [p for p in printers if wanted in p.vendor.lower()]
        if not chosen_list:
            raise SlicerError(
                "vendor_not_found",
                f"The installed slicer has no printers from {vendor!r}. Its vendors: "
                + ", ".join(sorted(vendors))
                + ".",
            )
        models: dict[str, dict[str, Any]] = {}
        for p in chosen_list:
            entry = models.setdefault(
                p.model_name or p.model or p.name,
                {"model": p.model_name or p.model or p.name, "vendor": p.vendor, "presets": []},
            )
            entry["presets"].append({"name": p.name, "nozzle_mm": p.nozzle})
        out["models"] = list(models.values())[:LIST_LIMIT]
        out["model_count"] = len(models)
    return out


def _slicer_info(slicer: Slicer) -> dict[str, Any]:
    return {"name": slicer.name, "version": slicer.version, "program": str(slicer.binary)}


# ------------------------------------------------------------------ slice


def _model_path(model: str) -> Path:
    path = Path(model).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise SlicerError("model_not_found", f"There is no model file at {path}.")
    if path.suffix.lower() not in MODEL_SUFFIXES:
        raise SlicerError(
            "model_unsupported",
            f"{path.name} is not a model the slicer reads; give an STL, 3MF, STEP, OBJ or "
            "AMF file.",
        )
    return path


def _check_rotation(rotation: dict[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for axis, value in (rotation or {}).items():
        if axis not in {"x", "y", "z"}:
            raise SlicerError(
                "rotation_invalid", "rotate takes degrees about x, y and z: {x, y, z}."
            )
        try:
            out[axis] = float(value)
        except (TypeError, ValueError) as exc:
            raise SlicerError("rotation_invalid", f"rotate {axis} must be a number.") from exc
    return out


def _fit_warnings(parsed: Any, settings: dict[str, str]) -> list[str]:
    """Warnings the slicer's command line does not give: the print leaves the bed."""
    warnings = []
    bed = parsed.bed
    bounds = parsed.bounds
    if bed and bounds:
        xs, ys = [p[0] for p in bed], [p[1] for p in bed]
        if (
            bounds[0] < min(xs) - 0.5
            or bounds[2] > max(xs) + 0.5
            or bounds[1] < min(ys) - 0.5
            or bounds[3] > max(ys) + 0.5
        ):
            warnings.append(
                f"The toolpaths reach outside the bed: they span x {bounds[0]:.0f} to "
                f"{bounds[2]:.0f} mm and y {bounds[1]:.0f} to {bounds[3]:.0f} mm on a bed of "
                f"{max(xs) - min(xs):.0f} by {max(ys) - min(ys):.0f} mm. Scale or rotate the "
                "model, or choose a larger printer."
            )
    heights = setting_numbers(settings.get("max_print_height") or settings.get("printable_height"))
    if heights and parsed.layers and parsed.layers[-1]["z"] > heights[0] + 0.01:
        warnings.append(
            f"The print is {parsed.layers[-1]['z']:.1f} mm tall; this printer prints at most "
            f"{heights[0]:.0f} mm."
        )
    return warnings


def slice_model(
    model: str,
    printer: str,
    material: str | None,
    quality: str | None,
    overrides: dict[str, Any] | None = None,
    rotate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slicer = locate()
    path = _model_path(model)
    rotation = _check_rotation(rotate)
    cat = catalog(slicer)
    choice = choose(cat, printer, material, quality)
    slice_id = new_slice_id()
    folder = slice_dir(slice_id)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        if slicer.kind == PRUSA:
            from .backends.prusa import run_prusa

            run, applied = run_prusa(
                slicer, choice, path, folder, datadir(slicer.kind), overrides, rotation
            )
        else:
            from .backends.orca import run_orca

            run, applied = run_orca(
                slicer, choice, path, folder, datadir(slicer.kind), overrides, rotation
            )
        parsed = parse_gcode(run.gcode, folder / "layers.bin")
    except GcodeError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise SlicerError(exc.reason, f"The slicer's output could not be read: {exc}") from exc
    except SlicerError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    settings = parsed.estimates.settings
    estimates = parsed.estimates.as_dict()
    if estimates["filament_g"] is None and parsed.grams_per_mm is not None:
        estimates["filament_g"] = round(sum(l["filament_g"] or 0 for l in parsed.layers), 2)
    filament_m = (
        round(estimates["filament_mm"] / 1000.0, 3) if estimates.get("filament_mm") else None
    )
    warnings = run.warnings + _fit_warnings(parsed, {**choice.printer.settings, **settings})
    record = {
        "slice_id": slice_id,
        "model": str(path),
        "model_name": path.name,
        "slicer": _slicer_info(slicer),
        "printer": choice.printer.name,
        "quality": choice.print_preset.name,
        "material": choice.filament.name,
        "material_type": choice.filament.settings.get("filament_type", ""),
        "request": {
            "printer": printer,
            "material": material,
            "quality": quality,
            "overrides": overrides or {},
            "rotate": rotation,
        },
        "matched": choice.notes,
        "alternatives": choice.alternatives,
        "overrides": applied,
        "warnings": warnings,
        "estimates": estimates,
        "filament_m": filament_m,
        "layer_height_mm": setting_numbers(settings.get("layer_height"))[:1] or None,
        "z_max_mm": parsed.layers[-1]["z"] if parsed.layers else None,
        "bed": parsed.bed,
        "bounds": parsed.bounds,
        "features": parsed.features,
        "time_basis": parsed.time_basis,
        "gcode": str(run.gcode),
        "gcode_bytes": run.gcode.stat().st_size,
        "gcode_lines": parsed.lines,
        "vertices": parsed.vertices,
        "seconds": round(time.monotonic() - started, 2),
        "layers": parsed.layers,
        "view": {},
    }
    save_record(record)
    return record


# ------------------------------------------------------------------ layer_stats


def _range(record: dict[str, Any], first: int | None, last: int | None) -> tuple[int, int]:
    count = len(record["layers"])
    a = 0 if first is None else int(first)
    b = count - 1 if last is None else int(last)
    if not (0 <= a <= b < count):
        raise SlicerError(
            "layer_out_of_range",
            f"Layers are numbered 0 to {count - 1} in slice {record['slice_id']}; "
            f"{a} to {b} is outside that.",
        )
    return a, b


def layer_stats(slice_id: str, first: int | None = None, last: int | None = None) -> dict[str, Any]:
    record = load_record(slice_id)
    a, b = _range(record, first, last)
    rows = record["layers"][a : b + 1]
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        for key, f in row["features"].items():
            t = totals.setdefault(key, {"time_s": 0.0, "filament_g": 0.0, "length_mm": 0.0})
            t["time_s"] += f.get("time_s", 0.0)
            t["filament_g"] += f.get("grams", 0.0) or 0.0
            t["length_mm"] += f.get("length_mm", 0.0)
    other = sum(r.get("other_time_s", 0.0) for r in rows)
    if other > 0:
        totals[OTHER_MOVES] = {"time_s": other, "filament_g": 0.0, "length_mm": 0.0}
    time_total = sum(r["time_s"] for r in rows)
    by_feature = [
        {
            "feature": key,
            "label": OTHER_LABEL if key == OTHER_MOVES else label(key),
            "time_s": round(v["time_s"], 1),
            "time_share": round(v["time_s"] / time_total, 4) if time_total else 0.0,
            "filament_g": round(v["filament_g"], 3) if key not in (TRAVEL, OTHER_MOVES) else None,
            "length_mm": round(v["length_mm"], 1),
        }
        for key, v in sorted(totals.items(), key=lambda kv: -kv[1]["time_s"])
    ]
    slowest = max(rows, key=lambda r: r["time_s"])
    return {
        "slice_id": slice_id,
        "first_layer": a,
        "last_layer": b,
        "z_from_mm": rows[0]["z"],
        "z_to_mm": rows[-1]["z"],
        "time_s": round(time_total, 1),
        "filament_g": round(sum(r["filament_g"] or 0 for r in rows), 3),
        "by_feature": by_feature,
        "slowest_layer": {
            "layer": slowest["layer"],
            "z_mm": slowest["z"],
            "time_s": slowest["time_s"],
        },
        "layers": [
            {
                "layer": r["layer"],
                "z_mm": r["z"],
                "time_s": r["time_s"],
                "filament_g": r["filament_g"],
                "features": sorted(k for k in r["features"] if k != TRAVEL),
                # Where on the bed each type prints, [x0, y0, x1, y1] in mm: which part
                # of the model a bridge or an overhang on this layer belongs to.
                "extents": {
                    k: f["bounds"] for k, f in sorted(r["features"].items()) if f.get("bounds")
                },
            }
            for r in rows
        ],
        "time_basis": record.get("time_basis"),
    }


# ------------------------------------------------------------------ export


def _layers_csv(record: dict[str, Any]) -> str:
    keys = [k for k in (e["key"] for e in legend()) if k in set(record["features"]) or k == TRAVEL]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["layer", "z_mm", "height_mm", "time_s", "filament_g"]
        + [f"time_s_{k}" for k in keys]
        + ["time_s_retracts_and_z_moves"]
    )
    for row in record["layers"]:
        writer.writerow(
            [row["layer"], row["z"], row["height"], row["time_s"], row["filament_g"]]
            + [row["features"].get(k, {}).get("time_s", 0) for k in keys]
            + [row.get("other_time_s", 0)]
        )
    return buffer.getvalue()


def export(
    slice_id: str, fmt: str, path: str | None = None, overwrite: bool = False
) -> dict[str, Any]:
    record = load_record(slice_id)
    fmt = fmt.lower().strip()
    if fmt not in {"gcode", "csv"}:
        raise SlicerError(
            "format_unsupported",
            "export writes gcode (the printer's file) or csv (time and filament per layer).",
        )
    model = Path(record["model"])
    suffix = ".gcode" if fmt == "gcode" else ".layers.csv"
    default_name = f"{model.stem}{suffix}"
    target = Path(path).expanduser() if path else model.parent / default_name
    if not target.is_absolute():
        target = Path.cwd() / target
    if target.is_dir():
        target = target / default_name
    if target.exists() and not overwrite:
        raise SlicerError(
            "export_exists",
            f"{target} already exists. Pass overwrite=true to replace it, or another path.",
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "gcode":
            shutil.copyfile(record["gcode"], target)
        else:
            target.write_text(_layers_csv(record), encoding="utf-8")
    except OSError as exc:
        raise SlicerError("export_failed", f"{target} could not be written: {exc}") from exc
    return {
        "slice_id": slice_id,
        "format": fmt,
        "path": str(target),
        "bytes": target.stat().st_size,
        "rows": len(record["layers"]) if fmt == "csv" else None,
    }


# ------------------------------------------------------------------ compare


def compare_slices(slice_ids: list[str]) -> dict[str, Any]:
    if len(slice_ids) < 2:
        raise SlicerError("compare_needs_two", "compare_slices needs two or more slice ids.")
    rows = []
    for sid in slice_ids:
        record = load_record(sid)
        stats = layer_stats(sid)
        rows.append(
            {
                "slice_id": sid,
                "model": record["model_name"],
                "printer": record["printer"],
                "quality": record["quality"],
                "material": record["material"],
                "overrides": record.get("overrides") or [],
                "print_time_s": record["estimates"].get("print_time_s"),
                "filament_g": record["estimates"].get("filament_g"),
                "layer_count": len(record["layers"]),
                "warnings": len(record.get("warnings") or []),
                "time_share": {f["feature"]: f["time_share"] for f in stats["by_feature"]},
            }
        )
    base = rows[0]
    for row in rows[1:]:
        for key in ("print_time_s", "filament_g"):
            if base.get(key) and row.get(key) is not None:
                row[f"{key}_change"] = round((row[key] - base[key]) / base[key], 4)
    return {"slices": rows, "baseline": base["slice_id"]}


__all__ = [
    "ORCA",
    "PRUSA",
    "catalog",
    "compare_slices",
    "export",
    "layer_stats",
    "list_printers",
    "slice_model",
]
