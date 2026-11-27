"""Read the vendor preset bundles an installed PrusaSlicer ships, and flatten presets.

PrusaSlicer installs one INI file per vendor (``resources/profiles/<Vendor>.ini``) with
sections ``[printer_model:<id>]``, ``[printer:<name>]``, ``[print:<name>]`` and
``[filament:<name>]``. A preset names its parents in ``inherits`` (``a; b``: later
parents win, then the preset's own keys); names written ``*like this*`` are abstract
parents only. This module reads those files as written and resolves a preset to the
flat ``key = value`` set the slicer's ``--load`` takes. It never writes the vendor
files and never copies them anywhere: each slice writes only its own flattened config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: Keys a vendor bundle uses for bookkeeping; they are not slicer settings.
BOOKKEEPING = frozenset({"inherits", "alias", "renamed_from"})
KINDS = ("printer", "print", "filament")


class BundleError(ValueError):
    """A preset cannot be resolved; ``reason`` is a stable code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class Bundle:
    """One vendor's presets as written, before inheritance."""

    vendor: str
    path: Path
    models: dict[str, dict[str, str]] = field(default_factory=dict)
    presets: dict[str, dict[str, dict[str, str]]] = field(
        default_factory=lambda: {kind: {} for kind in KINDS}
    )

    def resolve(self, kind: str, name: str) -> dict[str, str]:
        """The flat settings of one preset, parents applied in order."""
        return _resolve(self, kind, name, ())


_SECTION = re.compile(r"^\[(?P<kind>[a-z_]+)(?::(?P<name>.*))?\]\s*$")


def read_bundle(path: Path) -> Bundle:
    """Parse one vendor INI file."""
    bundle = Bundle(vendor=path.stem, path=path)
    current: dict[str, str] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        section = _SECTION.match(line)
        if section:
            kind, name = section.group("kind"), (section.group("name") or "").strip()
            if kind == "vendor":
                current = {}
                bundle.models.setdefault("__vendor__", current)
            elif kind == "printer_model":
                current = bundle.models.setdefault(name, {})
            elif kind in KINDS:
                current = bundle.presets[kind].setdefault(name, {})
            else:
                current = None
            continue
        if current is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        current[key.strip()] = value.strip()
    vendor_name = bundle.models.get("__vendor__", {}).get("name")
    if vendor_name:
        bundle.vendor = vendor_name
    return bundle


def _resolve(bundle: Bundle, kind: str, name: str, seen: tuple[str, ...]) -> dict[str, str]:
    presets = bundle.presets.get(kind, {})
    if name not in presets:
        raise BundleError(
            "preset_not_found", f"The {kind} preset {name!r} is not in {bundle.path.name}."
        )
    if name in seen:
        raise BundleError(
            "preset_cycle",
            f"The {kind} preset {name!r} inherits from itself in {bundle.path.name}.",
        )
    own = presets[name]
    flat: dict[str, str] = {}
    for parent in _parents(own):
        flat.update(_resolve(bundle, kind, parent, (*seen, name)))
    flat.update({k: v for k, v in own.items() if k != "inherits"})
    return flat


def _parents(section: dict[str, str]) -> list[str]:
    text = section.get("inherits", "")
    return [p.strip().strip('"') for p in text.split(";") if p.strip()]


def is_abstract(name: str) -> bool:
    return name.startswith("*") and name.endswith("*")


def settings_text(flat: dict[str, str]) -> str:
    """The ``--load`` file body for a flat setting set, bookkeeping keys left out."""
    lines = [f"{k} = {v}" for k, v in sorted(flat.items()) if k not in BOOKKEEPING]
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=64)
def _cached(path: str, mtime: int) -> Bundle:
    del mtime  # part of the cache key only
    return read_bundle(Path(path))


def load_bundles(folder: Path) -> list[Bundle]:
    """Every vendor bundle in a profiles folder, cached by file and modification time."""
    out = []
    for path in sorted(folder.glob("*.ini")):
        try:
            out.append(_cached(str(path), path.stat().st_mtime_ns))
        except OSError:
            continue
    return out


__all__ = [
    "BOOKKEEPING",
    "Bundle",
    "BundleError",
    "is_abstract",
    "load_bundles",
    "read_bundle",
    "settings_text",
]
