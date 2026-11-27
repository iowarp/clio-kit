"""OrcaSlicer's preset tree, read as the slicer ships it, behind :class:`.catalog.Catalog`.

OrcaSlicer installs ``resources/profiles/<Vendor>.json`` naming every preset and the file
under ``resources/profiles/<Vendor>/`` that holds it (``machine_list``, ``process_list``,
``filament_list``, ``machine_model_list``). A preset names one parent in ``inherits``;
user-visible presets carry ``"instantiation": "true"``; process and filament presets list
the printers they suit in ``compatible_printers`` (or a ``compatible_printers_condition``).
Orca's command line wants each preset flattened (no ``inherits``), which
:meth:`OrcaCatalog.flat` gives; nothing is written into the slicer's own folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalog import Preset, _compatible, printer_variables
from .ini_bundle import BundleError

KIND = {"printer": "machine", "print": "process", "filament": "filament"}
LISTS = {"machine": "machine_list", "process": "process_list", "filament": "filament_list"}


def _text(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return "" if value is None else str(value)


class OrcaCatalog:
    """OrcaSlicer's vendor JSON presets."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        #: (orca kind, name) -> (vendor, file)
        self.index: dict[tuple[str, str], tuple[str, Path]] = {}
        self.models: dict[tuple[str, str], dict[str, Any]] = {}
        for vendor_file in sorted(folder.glob("*.json")):
            try:
                vendor = json.loads(vendor_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            name = vendor_file.stem
            base = folder / name
            for kind, key in LISTS.items():
                for row in vendor.get(key) or []:
                    if isinstance(row, dict) and row.get("name") and row.get("sub_path"):
                        self.index.setdefault((kind, row["name"]), (name, base / row["sub_path"]))
            for row in vendor.get("machine_model_list") or []:
                if isinstance(row, dict) and row.get("sub_path"):
                    self.models[(name, str(row.get("name")))] = {"file": base / row["sub_path"]}
        self._raw: dict[tuple[str, str], dict[str, Any]] = {}
        self._flat: dict[tuple[str, str], dict[str, Any]] = {}

    def _load(self, kind: str, name: str) -> dict[str, Any]:
        key = (kind, name)
        if key not in self._raw:
            found = self.index.get(key)
            if found is None:
                raise BundleError("preset_not_found", f"OrcaSlicer has no {kind} preset {name!r}.")
            try:
                data = json.loads(found[1].read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise BundleError("preset_unreadable", f"{found[1]} cannot be read: {exc}") from exc
            self._raw[key] = data if isinstance(data, dict) else {}
        return self._raw[key]

    def flat(self, kind: str, name: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
        """The preset with every parent applied and ``inherits`` removed."""
        key = (kind, name)
        if key in self._flat:
            return self._flat[key]
        if name in seen:
            raise BundleError("preset_cycle", f"OrcaSlicer preset {name!r} inherits from itself.")
        own = self._load(kind, name)
        parent = str(own.get("inherits") or "").strip()
        out: dict[str, Any] = dict(self.flat(kind, parent, (*seen, name))) if parent else {}
        out.update({k: v for k, v in own.items() if k != "inherits"})
        self._flat[key] = out
        return out

    def _preset(self, kind: str, name: str) -> Preset | None:
        orca_kind = KIND[kind]
        try:
            data = self.flat(orca_kind, name)
        except BundleError:
            return None
        if str(data.get("instantiation", "true")).lower() != "true":
            return None
        vendor = self.index[(orca_kind, name)][0]
        settings = {k: _text(v) for k, v in data.items()}
        nozzle = settings.get("nozzle_diameter", "").split(",")[0] if kind == "printer" else ""
        model = settings.get("printer_model", "") if kind == "printer" else ""
        return Preset(
            kind=kind,
            name=name,
            vendor=vendor,
            settings=settings,
            model=model,
            model_name=model,
            nozzle=nozzle,
            bundle=vendor,
        )

    def _all(self, kind: str, vendors: set[str] | None = None) -> list[Preset]:
        orca_kind = KIND[kind]
        out = []
        for (k, name), (vendor, _path) in self.index.items():
            if k != orca_kind or (vendors is not None and vendor not in vendors):
                continue
            preset = self._preset(kind, name)
            if preset is not None:
                out.append(preset)
        return out

    def printers(self) -> list[Preset]:
        return self._all("printer")

    def _suits(self, preset: Preset, printer: Preset) -> bool:
        return _compatible(
            preset.settings.get("compatible_printers_condition", ""),
            ";".join(self.flat(KIND[preset.kind], preset.name).get("compatible_printers") or []),
            printer,
            printer_variables(printer.settings),
        )

    def prints_for(self, printer: Preset) -> list[Preset]:
        return [p for p in self._all("print") if self._suits(p, printer)]

    def filaments_for(self, printer: Preset, print_preset: Preset) -> list[Preset]:
        return [f for f in self._all("filament") if self._suits(f, printer)]

    def default_materials(self, printer: Preset) -> list[str]:
        raw = self.flat("machine", printer.name).get("default_filament_profile") or []
        names = raw if isinstance(raw, list) else [raw]
        return [str(n) for n in names if n]


__all__ = ["OrcaCatalog"]
