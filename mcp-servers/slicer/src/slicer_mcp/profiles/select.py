"""Choose the printer, quality and material presets a person's words name.

The person says "a Prusa MK4 in PLA at 0.2 mm"; the slicer needs three exact preset
names. Each choice is made from the installed slicer's own presets and the vendor's own
compatibility rules (:mod:`.catalog`), and every choice is reported with the words it
matched and the close alternatives, so nothing is picked silently:

* printer: an exact preset name, else the presets whose name, model id or model name
  contain every word of the query; a 0.4 mm nozzle unless a nozzle is named; then the
  shortest name.
* quality: an exact print preset name, else the compatible presets with that layer
  height (``0.2``, ``0.20mm``) and any quality word ("quality", "speed", "draft",
  "detail", "structural"); the printer's own default first, then "QUALITY".
* material: an exact filament preset name, else compatible presets of that filament
  type (``PLA``, ``PETG`` …) or whose name contains it; the printer's default
  materials first, then "Generic" ones.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from ..errors import SlicerError
from .catalog import Catalog, Preset

ALTERNATIVES = 5
QUALITY_WORDS = ("quality", "speed", "draft", "detail", "ultradetail", "structural", "fine")


@dataclass
class Choice:
    """The presets a slice uses and how they were chosen."""

    printer: Preset
    print_preset: Preset
    filament: Preset
    notes: list[str] = field(default_factory=list)
    alternatives: dict[str, list[str]] = field(default_factory=dict)
    #: The catalogue the presets came from (the OrcaSlicer run flattens from it).
    catalog: Catalog | None = None


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9.]+", text.lower()) if t]


def _nozzle(text: str) -> str | None:
    match = re.search(r"(\d(?:\.\d+)?)\s*(?:mm)?\s*nozzle", text.lower())
    return match.group(1) if match else None


def _same_number(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def _not_found(kind: str, query: str, names: list[str], hint: str) -> SlicerError:
    close = difflib.get_close_matches(query, names, n=ALTERNATIVES, cutoff=0.3)
    listed = "; ".join(close) if close else "none close"
    return SlicerError(
        "profile_not_found",
        f"No {kind} preset of the installed slicer matches {query!r}. Close names: "
        f"{listed}. {hint}",
        kind=kind,
        query=query,
        close=close,
    )


def choose_printer(catalog: Catalog, query: str) -> tuple[Preset, list[str]]:
    printers = catalog.printers()
    exact = [p for p in printers if p.name.lower() == query.strip().lower()]
    if exact:
        return exact[0], []
    nozzle = _nozzle(query)
    words = [
        w
        for w in _tokens(
            re.sub(r"\d(?:\.\d+)?\s*(?:mm)?\s*nozzle", " ", query, flags=re.IGNORECASE)
        )
    ]
    found = []
    for p in printers:
        haystack = set(_tokens(f"{p.name} {p.model} {p.model_name} {p.vendor}"))
        if all(w in haystack for w in words) and (
            nozzle is None or _same_number(p.nozzle or "0", nozzle)
        ):
            found.append(p)
    if not found:
        raise _not_found(
            "printer",
            query,
            [p.name for p in printers],
            "list_printers shows the printers this slicer knows.",
        )

    def rank(p: Preset) -> tuple[int, int, str]:
        default_nozzle = 0 if nozzle is not None or _same_number(p.nozzle or "0", "0.4") else 1
        return default_nozzle, len(_tokens(p.name)), p.name

    found.sort(key=rank)
    return found[0], _unique([p.name for p in found[1:]])


def _alias(preset: Preset) -> str:
    """The material a filament preset is a variant of ("Prusament PLA @PG 0.6" is
    "Prusament PLA"), as the vendor's ``alias`` says or the name before ``@``."""
    return (preset.settings.get("alias") or preset.name.split(" @")[0]).strip().lower()


def _unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))[:ALTERNATIVES]


def _layer_height(text: str) -> str | None:
    match = re.search(r"(\d*\.\d+|\d+)\s*(?:mm)?", text)
    if not match:
        return None
    value = float(match.group(1))
    return str(value) if 0.01 <= value <= 2.0 else None


def choose_print(catalog: Catalog, printer: Preset, query: str | None) -> tuple[Preset, list[str]]:
    prints = catalog.prints_for(printer)
    if not prints:
        raise SlicerError(
            "profile_not_found",
            f"The installed slicer has no print preset for {printer.name}.",
            kind="print",
        )
    default = printer.settings.get("default_print_profile", "").strip().strip('"')
    if not query or not query.strip():
        chosen = next((p for p in prints if p.name == default), prints[0])
        return chosen, []
    exact = [p for p in prints if p.name.lower() == query.strip().lower()]
    if exact:
        return exact[0], []
    height = _layer_height(query)
    words = [w for w in QUALITY_WORDS if w in query.lower()]
    found = [
        p
        for p in prints
        if (height is None or _same_number(p.settings.get("layer_height", ""), height))
        and all(w in p.name.lower() for w in words)
    ]
    if not found:
        heights = sorted({p.settings.get("layer_height", "") for p in prints})
        raise _not_found(
            "quality",
            query,
            [p.name for p in prints],
            f"{printer.name} has print presets at layer heights {', '.join(heights)} mm.",
        )

    def rank(p: Preset) -> tuple[int, int, str]:
        return (0 if p.name == default else 1, 0 if "quality" in p.name.lower() else 1, p.name)

    found.sort(key=rank)
    return found[0], _unique([p.name for p in found[1:]])


def choose_filament(
    catalog: Catalog, printer: Preset, print_preset: Preset, query: str | None
) -> tuple[Preset, list[str]]:
    filaments = catalog.filaments_for(printer, print_preset)
    if not filaments:
        raise SlicerError(
            "profile_not_found",
            f"The installed slicer has no filament preset for {printer.name} with "
            f"{print_preset.name}.",
            kind="material",
        )
    # A default names a material ("Prusament PLA @PG"); the preset for this nozzle may
    # be a variant of it ("Prusament PLA @PG 0.6"), as in the slicer's own window.
    defaults = catalog.default_materials(printer)
    order: dict[str, int] = {}
    for i, name in enumerate(defaults):
        order.setdefault(name.split(" @")[0].strip().lower(), i)

    def default_rank(f: Preset) -> tuple[int, int, int]:
        # The printer's own vendor presets before the generic templates.
        return (
            order.get(_alias(f), len(order)),
            0 if f.name in defaults else 1,
            0 if f.bundle == printer.bundle else 1,
        )

    if not query or not query.strip():
        return min(filaments, key=lambda f: (default_rank(f), len(f.name), f.name)), []
    wanted = query.strip().lower()
    exact = [f for f in filaments if f.name.lower() == wanted]
    if exact:
        return exact[0], []
    found = [
        f for f in filaments if f.settings.get("filament_type", "").strip().lower() == wanted
    ] or [f for f in filaments if wanted in f.name.lower()]
    if not found:
        types = sorted({f.settings.get("filament_type", "") for f in filaments} - {""})
        raise _not_found(
            "material",
            query,
            [f.name for f in filaments],
            f"Materials for {printer.name}: {', '.join(types)}.",
        )

    def rank(f: Preset) -> tuple[tuple[int, int, int], int, int, str]:
        return (
            default_rank(f),
            0 if f.name.lower().startswith("generic") else 1,
            len(f.name),
            f.name,
        )

    found.sort(key=rank)
    return found[0], _unique([f.name for f in found[1:] if f.name != found[0].name])


def choose(catalog: Catalog, printer: str, material: str | None, quality: str | None) -> Choice:
    """The three presets for the person's words, with notes on how each was matched."""
    printer_preset, other_printers = choose_printer(catalog, printer)
    print_preset, other_prints = choose_print(catalog, printer_preset, quality)
    filament, other_filaments = choose_filament(catalog, printer_preset, print_preset, material)
    notes = []
    if printer_preset.name.lower() != printer.strip().lower():
        notes.append(f"printer {printer!r} matched {printer_preset.name!r}")
    if not quality:
        notes.append(f"no quality given: the printer's default {print_preset.name!r}")
    elif print_preset.name.lower() != quality.strip().lower():
        notes.append(f"quality {quality!r} matched {print_preset.name!r}")
    if not material:
        notes.append(f"no material given: the printer's default {filament.name!r}")
    elif filament.name.lower() != material.strip().lower():
        notes.append(f"material {material!r} matched {filament.name!r}")
    return Choice(
        printer=printer_preset,
        print_preset=print_preset,
        filament=filament,
        notes=notes,
        alternatives={
            "printer": other_printers,
            "quality": other_prints,
            "material": other_filaments,
        },
        catalog=catalog,
    )


__all__ = ["Choice", "choose", "choose_filament", "choose_print", "choose_printer"]
