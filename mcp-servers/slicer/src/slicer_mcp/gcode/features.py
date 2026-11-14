"""The feature types a slicer writes in its G-code, under one set of names.

PrusaSlicer marks each extrusion with ``;TYPE:<name>``; OrcaSlicer with
``; FEATURE: <name>`` (and often ``;TYPE:`` too). Both name the same things
differently ("External perimeter" and "Outer wall"). Each name maps to one stable key
the view, the statistics and the state use; the label is what a person reads.
"""

from __future__ import annotations

#: Stable key -> (label, colour). Colours are a legend aid only: the view always shows
#: the label beside them, and a person can toggle each type by name.
FEATURES: dict[str, tuple[str, str]] = {
    "external_perimeter": ("Outer wall", "#ff7d38"),
    "perimeter": ("Inner wall", "#ffd23f"),
    "overhang_perimeter": ("Overhang wall", "#2f6bff"),
    "internal_infill": ("Sparse infill", "#b0303c"),
    "solid_infill": ("Solid infill", "#9a3fd1"),
    "top_solid_infill": ("Top surface", "#f0455a"),
    "bottom_surface": ("Bottom surface", "#7a5af8"),
    "bridge_infill": ("Bridge", "#4d9fe0"),
    "gap_fill": ("Gap fill", "#e8e8e8"),
    "skirt": ("Skirt and brim", "#00a88f"),
    "support": ("Support", "#3fb950"),
    "support_interface": ("Support interface", "#1f7a3a"),
    "wipe_tower": ("Wipe tower", "#b3e3ab"),
    "ironing": ("Ironing", "#ff9fa8"),
    "custom": ("Custom G-code", "#5ed6e0"),
    "other": ("Other extrusion", "#9aa0a6"),
    "travel": ("Travel", "#8a93a0"),
}

#: The non-extrusion key: moves without extrusion (and wipes).
TRAVEL = "travel"

_NAMES = {
    # PrusaSlicer
    "perimeter": "perimeter",
    "external perimeter": "external_perimeter",
    "overhang perimeter": "overhang_perimeter",
    "internal infill": "internal_infill",
    "solid infill": "solid_infill",
    "top solid infill": "top_solid_infill",
    "bridge infill": "bridge_infill",
    "internal bridge infill": "bridge_infill",
    "gap fill": "gap_fill",
    "skirt": "skirt",
    "skirt/brim": "skirt",
    "brim": "skirt",
    "support material": "support",
    "support material interface": "support_interface",
    "wipe tower": "wipe_tower",
    "ironing": "ironing",
    "custom": "custom",
    # OrcaSlicer / Bambu Studio
    "inner wall": "perimeter",
    "outer wall": "external_perimeter",
    "overhang wall": "overhang_perimeter",
    "sparse infill": "internal_infill",
    "internal solid infill": "solid_infill",
    "top surface": "top_solid_infill",
    "bottom surface": "bottom_surface",
    "bridge": "bridge_infill",
    "internal bridge": "bridge_infill",
    "gap infill": "gap_fill",
    "support": "support",
    "support interface": "support_interface",
    "support transition": "support_interface",
    "prime tower": "wipe_tower",
}


def feature_key(name: str) -> str:
    """The stable key for a slicer's feature name; unknown names are ``other``."""
    return _NAMES.get(name.strip().lower(), "other")


def label(key: str) -> str:
    return FEATURES.get(key, (key.replace("_", " ").capitalize(), ""))[0]


def legend(keys: list[str] | None = None) -> list[dict[str, str]]:
    """The legend rows for the given keys (all, in display order, when None)."""
    chosen = list(FEATURES) if keys is None else [k for k in FEATURES if k in set(keys)]
    return [{"key": k, "label": FEATURES[k][0], "color": FEATURES[k][1]} for k in chosen]


__all__ = ["FEATURES", "TRAVEL", "feature_key", "label", "legend"]
