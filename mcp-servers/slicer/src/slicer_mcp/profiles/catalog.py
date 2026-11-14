"""The installed slicer's own presets, behind one interface for both slicers.

A :class:`Preset` is a printer, print (quality) or filament (material) preset with its
settings flattened. A catalogue answers which printers exist, which print presets suit a
printer and which filaments suit a printer and print, using the vendor's own
compatibility conditions. PrusaSlicer's catalogue reads its vendor INI bundles; the
OrcaSlicer catalogue (:mod:`.orca_bundle`) reads its JSON preset tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .condition import ConditionError, evaluate
from .ini_bundle import Bundle, BundleError, is_abstract, load_bundles

#: Vendor bundles holding generic presets every printer may use.
GENERIC_BUNDLES = frozenset({"Templates"})


@dataclass
class Preset:
    """One preset with its flattened settings."""

    kind: str  # printer, print, filament
    name: str
    vendor: str
    settings: dict[str, str] = field(repr=False)
    model: str = ""
    model_name: str = ""
    nozzle: str = ""
    bundle: str = ""


class Catalog(Protocol):
    """What a slicer's preset tree answers."""

    def printers(self) -> list[Preset]: ...

    def prints_for(self, printer: Preset) -> list[Preset]: ...

    def filaments_for(self, printer: Preset, print_preset: Preset) -> list[Preset]: ...

    def default_materials(self, printer: Preset) -> list[str]: ...


def printer_variables(settings: dict[str, str]) -> dict[str, str]:
    """A printer's settings plus the derived variables the conditions use."""
    out = dict(settings)
    nozzles = [p for p in re.split(r"[;,]", settings.get("nozzle_diameter", "")) if p.strip()]
    out.setdefault("num_extruders", str(max(1, len(nozzles))))
    out.setdefault("single_extruder_multi_material", "0")
    return out


def _compatible(condition: str, names: str, subject: Preset, variables: dict[str, str]) -> bool:
    """A preset's ``compatible_*`` list wins over its condition, as in the slicer."""
    listed = [n.strip().strip('"') for n in re.split(r";", names or "") if n.strip()]
    if listed:
        return subject.name in listed
    try:
        return evaluate(condition or "", variables)
    except ConditionError:
        # A condition this reader cannot evaluate never matches: no guessed preset.
        return False


class PrusaCatalog:
    """PrusaSlicer's vendor bundles (``resources/profiles/*.ini``)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.bundles: list[Bundle] = load_bundles(folder)
        self._cache: dict[tuple[str, str, str], Preset] = {}

    def _preset(self, bundle: Bundle, kind: str, name: str) -> Preset:
        key = (bundle.path.name, kind, name)
        found = self._cache.get(key)
        if found is None:
            settings = bundle.resolve(kind, name)
            model = settings.get("printer_model", "") if kind == "printer" else ""
            model_name = bundle.models.get(model, {}).get("name", model) if model else ""
            found = Preset(
                kind=kind,
                name=name,
                vendor=bundle.vendor,
                settings=settings,
                model=model,
                model_name=model_name,
                nozzle=settings.get("printer_variant", "") if kind == "printer" else "",
                bundle=bundle.path.stem,
            )
            self._cache[key] = found
        return found

    def _all(self, bundle: Bundle, kind: str) -> list[Preset]:
        out = []
        for name in bundle.presets.get(kind, {}):
            if is_abstract(name):
                continue
            try:
                out.append(self._preset(bundle, kind, name))
            except BundleError:
                continue
        return out

    def printers(self) -> list[Preset]:
        out: list[Preset] = []
        for bundle in self.bundles:
            if bundle.path.stem in GENERIC_BUNDLES:
                continue
            for preset in self._all(bundle, "printer"):
                if preset.settings.get("printer_technology", "FFF").upper() == "FFF":
                    out.append(preset)
        return out

    def _bundles_for(self, printer: Preset) -> list[Bundle]:
        return [
            b
            for b in self.bundles
            if b.path.stem == printer.bundle or b.path.stem in GENERIC_BUNDLES
        ]

    def prints_for(self, printer: Preset) -> list[Preset]:
        variables = printer_variables(printer.settings)
        return [
            p
            for bundle in self._bundles_for(printer)
            for p in self._all(bundle, "print")
            if _compatible(
                p.settings.get("compatible_printers_condition", ""),
                p.settings.get("compatible_printers", ""),
                printer,
                variables,
            )
        ]

    def filaments_for(self, printer: Preset, print_preset: Preset) -> list[Preset]:
        variables = printer_variables(printer.settings)
        out = []
        for bundle in self._bundles_for(printer):
            for f in self._all(bundle, "filament"):
                if not _compatible(
                    f.settings.get("compatible_printers_condition", ""),
                    f.settings.get("compatible_printers", ""),
                    printer,
                    variables,
                ):
                    continue
                if not _compatible(
                    f.settings.get("compatible_prints_condition", ""),
                    f.settings.get("compatible_prints", ""),
                    print_preset,
                    print_preset.settings,
                ):
                    continue
                out.append(f)
        return out

    def default_materials(self, printer: Preset) -> list[str]:
        names: list[str] = []
        default = printer.settings.get("default_filament_profile", "").strip().strip('"')
        if default:
            names.append(default)
        for bundle in self.bundles:
            if bundle.path.stem == printer.bundle:
                listed = bundle.models.get(printer.model, {}).get("default_materials", "")
                names.extend(n.strip() for n in listed.split(";") if n.strip())
        return list(dict.fromkeys(names))


__all__ = ["Catalog", "Preset", "PrusaCatalog", "printer_variables"]
