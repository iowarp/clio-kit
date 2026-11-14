"""The agent-facing text of each tool: lean, in words, never the toolpaths.

Big data stays in ``structuredContent`` or in the slice's folder, referenced by slice id.
"""

from __future__ import annotations

from typing import Any


def duration(seconds: float | None) -> str:
    """``2993`` -> ``49 min 53 s``; ``None`` -> ``unknown``."""
    if seconds is None:
        return "unknown"
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def grams(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f} g"


def _list(items: list[str]) -> str:
    return "; ".join(items)


def printers_text(data: dict[str, Any]) -> str:
    slicer = data["slicer"]
    head = f"{slicer['name']} {slicer['version']} at {slicer['program']}."
    if "printer" in data:
        p = data["printer"]
        lines = [
            head,
            (
                f"Printer preset {p['name']} ({p['vendor']} {p['model']}, {p['nozzle_mm']} mm "
                f"nozzle, bed {p['bed']}, up to {p['max_height_mm']} mm tall)."
            ),
        ]
        if data.get("also_matched"):
            lines.append("Also matched: " + _list(data["also_matched"]) + ".")
        lines.append(
            "Qualities: "
            + _list([f"{q['name']} ({q['layer_height_mm']} mm)" for q in data["qualities"]])
            + f". Default: {data['default_quality']}."
        )
        lines.append(
            "Materials by type (preset count): "
            + ", ".join(f"{k} ({n})" for k, n in data["material_counts"].items())
            + ". Examples: "
            + _list([f"{k}: {', '.join(v[:3])}" for k, v in data["materials"].items()][:8])
            + "."
        )
        return "\n".join(lines)
    lines = [
        head,
        "Vendors (printer presets): "
        + ", ".join(f"{v['vendor']} ({v['printers']})" for v in data["vendors"])
        + ".",
    ]
    for model in data.get("models", []):
        nozzles = sorted({p["nozzle_mm"] for p in model["presets"] if p["nozzle_mm"]})
        lines.append(
            f"- {model['model']}: "
            + _list([p["name"] for p in model["presets"][:4]])
            + (f" … nozzles {', '.join(nozzles)} mm" if len(model["presets"]) > 4 else "")
        )
    if "models" in data and data.get("model_count", 0) > len(data["models"]):
        lines.append(f"({data['model_count'] - len(data['models'])} more models not listed.)")
    if "models" not in data:
        lines.append(
            "Pass vendor to list its printers, or printer to see its qualities and materials."
        )
    return "\n".join(lines)


def slice_text(record: dict[str, Any]) -> str:
    e = record["estimates"]
    metres = f", {record['filament_m']:.2f} m" if record.get("filament_m") else ""
    lines = [
        (
            f"Sliced {record['model_name']} as slice {record['slice_id']} with "
            f"{record['slicer']['name']} {record['slicer']['version']}: "
            f"{duration(e.get('print_time_s'))}, {grams(e.get('filament_g'))}{metres} of filament, "
            f"{len(record['layers'])} layers up to Z {record.get('z_max_mm')} mm."
        ),
        (
            f"Presets: printer {record['printer']}; quality {record['quality']}; material "
            f"{record['material']}."
        ),
    ]
    if record.get("matched"):
        lines.append("Chosen: " + _list(record["matched"]) + ".")
    alternatives = {k: v for k, v in (record.get("alternatives") or {}).items() if v}
    if alternatives:
        lines.append(
            "Other close presets: "
            + _list([f"{k}: {', '.join(v[:3])}" for k, v in alternatives.items()])
            + "."
        )
    if record.get("overrides"):
        lines.append("Overrides: " + _list(record["overrides"]) + ".")
    if record["request"].get("rotate"):
        lines.append(
            "Rotated: "
            + ", ".join(f"{a} {v:g}°" for a, v in record["request"]["rotate"].items())
            + "."
        )
    warnings = record.get("warnings") or []
    lines.append(
        ("Warnings: " + " | ".join(warnings)) if warnings else "The slicer gave no warnings."
    )
    lines.append(
        f"show_slice slice_id={record['slice_id']!r} opens the layer view; layer_stats gives "
        "time and filament by feature type; export writes the G-code or a per-layer CSV."
    )
    return "\n".join(lines)


def show_text(payload: dict[str, Any]) -> str:
    z = payload["layers"]["z"]
    layer = payload["layer"]
    lines = [
        (
            f"Showing slice {payload['slice_id']} of {payload['model_name']} on layer {layer} "
            f"(Z {z[layer]} mm) of {payload['layer_count']} (layers 0 to "
            f"{payload['layer_count'] - 1}, Z {z[0]} to {z[-1]} mm): "
            f"{duration(payload.get('print_time_s'))}, {grams(payload.get('filament_g'))} of "
            f"{payload['material']} on {payload['printer']} at {payload['quality']}."
        ),
        "Time by feature: "
        + ", ".join(
            f"{row['label']} {row['time_share'] * 100:.0f}%"
            for row in sorted(payload["legend"], key=lambda r: -r["time_share"])
            if row["time_share"] >= 0.01
        )
        + ".",
    ]
    if payload.get("warnings"):
        lines.append("Warnings shown in the view: " + " | ".join(payload["warnings"]))
    if payload.get("stride", 1) > 1:
        lines.append(
            f"The print has {payload['vertices']:,} toolpath points, more than a browser "
            f"draws smoothly: the view draws every {payload['stride']}th layer below the shown "
            "one (the shown layer always whole) and says so; the numbers use every move."
        )
    lines.append(
        "The person can move the layer slider, toggle feature types and travel, and click "
        "a toolpath; the view reports layer, z_mm, visible_features and selection. Its "
        "tools: go_to_layer (layer or z_mm), set_feature_visibility, pick_segment. "
        f"show_slice slice_id={payload['slice_id']!r} shows it again."
    )
    return "\n".join(lines)


def stats_text(data: dict[str, Any]) -> str:
    lines = [
        (
            f"Layers {data['first_layer']} to {data['last_layer']} (Z {data['z_from_mm']} to "
            f"{data['z_to_mm']} mm) of slice {data['slice_id']}: {duration(data['time_s'])}, "
            f"{data['filament_g']:.2f} g."
        ),
        "By feature type: "
        + "; ".join(
            f"{f['label']} {duration(f['time_s'])} ({f['time_share'] * 100:.0f}%"
            + (f", {f['filament_g']:.2f} g" if f["filament_g"] else "")
            + ")"
            for f in data["by_feature"]
        )
        + ".",
        (
            f"Slowest layer: {data['slowest_layer']['layer']} at Z {data['slowest_layer']['z_mm']} "
            f"mm, {duration(data['slowest_layer']['time_s'])}."
        ),
    ]
    rows = data["layers"]
    if len(rows) <= 12:
        lines.append(
            "Per layer: "
            + "; ".join(
                f"{r['layer']} (Z {r['z_mm']}): {duration(r['time_s'])}, "
                + ", ".join(r["features"])
                for r in rows
            )
            + "."
        )
    else:
        lines.append(
            f"Per-layer rows for all {len(rows)} layers are in the structured result "
            "(layers: layer, z_mm, time_s, filament_g, features, and extents: where on the "
            "bed each type prints, [x0, y0, x1, y1] mm)."
        )
    lines.append(f"Times are a {data['time_basis']}.")
    return "\n".join(lines)


def export_text(data: dict[str, Any]) -> str:
    what = "G-code" if data["format"] == "gcode" else f"per-layer CSV ({data['rows']} rows)"
    return (
        f"Wrote the {what} of slice {data['slice_id']} to {data['path']} ({data['bytes']:,} bytes)."
    )


def compare_text(data: dict[str, Any]) -> str:
    lines = [f"Compared {len(data['slices'])} slices against {data['baseline']}:"]
    for row in data["slices"]:
        change = ""
        if "print_time_s_change" in row:
            change = (
                f" ({row['print_time_s_change'] * 100:+.0f}% time, "
                f"{row.get('filament_g_change', 0) * 100:+.0f}% filament)"
            )
        top = sorted(row["time_share"].items(), key=lambda kv: -kv[1])[:3]
        lines.append(
            f"- {row['slice_id']} {row['model']}: {duration(row['print_time_s'])}, "
            f"{grams(row['filament_g'])}, {row['layer_count']} layers{change}; "
            f"{row['printer']} / {row['quality']} / {row['material']}"
            + (f"; overrides {', '.join(row['overrides'])}" if row["overrides"] else "")
            + "; most time in "
            + ", ".join(f"{k} {v * 100:.0f}%" for k, v in top)
            + (f"; {row['warnings']} warnings" if row["warnings"] else "")
        )
    return "\n".join(lines)


__all__ = [
    "compare_text",
    "duration",
    "export_text",
    "grams",
    "printers_text",
    "show_text",
    "slice_text",
    "stats_text",
]
