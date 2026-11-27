"""Read the slicer's own estimates and settings from the comments of its G-code.

PrusaSlicer writes, near the end of the file::

    ; filament used [mm] = 3561.97
    ; filament used [cm3] = 8.57
    ; filament used [g] = 10.62
    ; estimated printing time (normal mode) = 49m 53s

and then its whole configuration as ``; key = value`` lines. OrcaSlicer writes
``; total estimated time: 1h 2m 3s`` (or ``; estimated printing time (normal mode)``),
``; filament used [g] = …`` and ``; total layer number: N``. These are the numbers the
person's slicer reports, so they are what this server reports; nothing is recomputed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Settings worth keeping from the configuration block, by both slicers' names.
KEPT_SETTINGS = frozenset(
    {
        "bed_shape",
        "printable_area",
        "printable_height",
        "max_print_height",
        "filament_diameter",
        "filament_density",
        "filament_type",
        "filament_settings_id",
        "print_settings_id",
        "printer_settings_id",
        "printer_model",
        "layer_height",
        "first_layer_height",
        "initial_layer_print_height",
        "nozzle_diameter",
        "support_material",
        "enable_support",
        "fill_density",
        "sparse_infill_density",
        "perimeters",
        "wall_loops",
        "brim_width",
    }
)

_DURATION = re.compile(r"(?:(\d+)d\s*)?(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?")


def parse_duration(text: str) -> float | None:
    """Seconds in a slicer duration such as ``1d 2h 3m 4s`` or ``49m 53s``."""
    text = text.strip()
    match = _DURATION.fullmatch(text)
    if not text or not match or not any(match.groups()):
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


@dataclass
class Estimates:
    """What the slicer itself says about a sliced print."""

    print_time_s: float | None = None
    first_layer_time_s: float | None = None
    filament_mm: float | None = None
    filament_cm3: float | None = None
    filament_g: float | None = None
    layer_count: int | None = None
    settings: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "print_time_s": self.print_time_s,
            "first_layer_time_s": self.first_layer_time_s,
            "filament_mm": self.filament_mm,
            "filament_cm3": self.filament_cm3,
            "filament_g": self.filament_g,
            "layer_count": self.layer_count,
        }


def read_comment(line: str, into: Estimates) -> None:
    """Fold one ``;`` comment line into the estimates (other lines are ignored)."""
    text = line.lstrip(";").strip()
    if not text:
        return
    lower = text.lower()
    if lower.startswith(("estimated printing time (normal mode)", "total estimated time")):
        value = text.split("=", 1)[1] if "=" in text else text.split(":", 1)[-1]
        into.print_time_s = parse_duration(value) or into.print_time_s
        return
    if lower.startswith("model printing time:") and "total estimated time:" in lower:
        into.print_time_s = parse_duration(lower.rsplit("total estimated time:", 1)[1].strip())
        return
    if lower.startswith("estimated first layer printing time (normal mode)"):
        into.first_layer_time_s = parse_duration(text.split("=", 1)[1])
        return
    if lower.startswith(("total layer number:", "total layers count")):
        number = _first_number(text.split(":", 1)[-1] if ":" in text else text)
        into.layer_count = int(number) if number is not None else into.layer_count
        return
    for unit, attr in (("[mm]", "filament_mm"), ("[cm3]", "filament_cm3"), ("[g]", "filament_g")):
        if (
            lower.startswith((f"filament used {unit}", f"total filament used {unit}"))
        ) and "wipe tower" not in lower:
            # Several extruders are listed comma-separated: the print uses their sum.
            values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text.split("=", 1)[-1])]
            if values:
                setattr(into, attr, round(sum(values), 3))
            return
    if "=" in text:
        key, _, value = text.partition("=")
        key = key.strip()
        if key in KEPT_SETTINGS:
            into.settings[key] = value.strip()


def setting_numbers(value: str | None) -> list[float]:
    """The numbers in a setting such as ``1.75`` or ``1.75,1.75``."""
    if not value:
        return []
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", value)]


def bed_outline(settings: dict[str, str]) -> list[list[float]] | None:
    """The bed polygon ``[[x, y], …]`` from ``bed_shape`` or ``printable_area``."""
    text = settings.get("bed_shape") or settings.get("printable_area")
    if not text:
        return None
    points = []
    for pair in text.split(","):
        parts = pair.strip().lower().split("x")
        if len(parts) != 2:
            return None
        try:
            points.append([float(parts[0]), float(parts[1])])
        except ValueError:
            return None
    return points if len(points) >= 3 else None


__all__ = ["KEPT_SETTINGS", "Estimates", "bed_outline", "parse_duration", "read_comment"]
