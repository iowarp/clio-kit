"""Run OrcaSlicer's command line, and take the G-code out of the 3MF it writes.

The command is::

    orca-slicer --datadir <server's own folder> \\
        --load-settings "machine.json;process.json" --load-filaments filament.json \\
        [--rotate-x A --rotate-y B --rotate C] --slice 0 --export-3mf slice.3mf \\
        --outputdir <slice folder> MODEL

Each preset is flattened (:meth:`..profiles.orca_bundle.OrcaCatalog.flat`) with the
person's overrides on the preset that has the key. OrcaSlicer puts the G-code of plate N
in the 3MF as ``Metadata/plate_N.gcode`` and its estimates in
``Metadata/slice_info.config`` (``prediction`` in seconds, ``weight`` in grams); the
G-code is copied out as ``slice.gcode`` and those estimates fill any the G-code's own
comments lack.
"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from ..errors import SlicerError
from ..profiles.select import Choice
from .locate import Slicer
from .prusa import _PROGRESS, SLICE_TIMEOUT_S, SliceRun, coerce, read_warnings


def _flat(choice: Choice, kind: str, name: str) -> dict[str, Any]:
    from ..profiles.orca_bundle import OrcaCatalog

    catalog = getattr(choice, "catalog", None)
    if not isinstance(catalog, OrcaCatalog):
        raise SlicerError("profiles_missing", "The OrcaSlicer presets are not loaded.")
    data = dict(catalog.flat(kind, name))
    data.pop("inherits", None)
    data["from"] = "User"
    data["is_custom_defined"] = "1"
    return data


def apply_orca_overrides(
    presets: list[dict[str, Any]], overrides: dict[str, Any] | None
) -> list[str]:
    import difflib

    applied = []
    keys = {k for p in presets for k in p}
    for key, value in (overrides or {}).items():
        target = next((p for p in reversed(presets) if key in p), None)
        if target is None:
            close = difflib.get_close_matches(key, sorted(keys), n=5, cutoff=0.5)
            raise SlicerError(
                "override_unknown",
                f"{key!r} is not an OrcaSlicer setting of these presets"
                + (f"; close names: {', '.join(close)}" if close else "")
                + ".",
                key=key,
                close=close,
            )
        current = target[key]
        target[key] = [coerce(value)] if isinstance(current, list) else coerce(value)
        applied.append(f"{key} = {coerce(value)}")
    return applied


def read_slice_info(text: str) -> dict[str, float]:
    """``prediction`` (s) and ``weight`` (g) of the first plate in slice_info.config."""
    out: dict[str, float] = {}
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return out
    plate = root.find("plate")
    if plate is None:
        return out
    for meta in plate.findall("metadata"):
        key, value = meta.get("key"), meta.get("value")
        if key in {"prediction", "weight"} and value:
            try:
                out[key] = float(value)
            except ValueError:
                continue
    used_m = [str(f.get("used_m")) for f in plate.findall("filament") if f.get("used_m")]
    if used_m:
        out["used_m"] = float(sum(float(v) for v in used_m))
    return out


def extract_gcode(archive: Path, target: Path) -> dict[str, float]:
    """Copy plate 1's G-code out of an OrcaSlicer 3MF; returns its slice_info estimates."""
    try:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            plate = next((n for n in names if n.lower() == "metadata/plate_1.gcode"), None) or next(
                (
                    n
                    for n in names
                    if n.lower().startswith("metadata/plate_") and n.endswith(".gcode")
                ),
                None,
            )
            if plate is None:
                raise SlicerError(
                    "slice_failed",
                    "OrcaSlicer wrote a 3MF without G-code (no Metadata/plate_N.gcode).",
                )
            with zf.open(plate) as src, target.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            info = (
                read_slice_info(zf.read("Metadata/slice_info.config").decode("utf-8", "replace"))
                if "Metadata/slice_info.config" in names
                else {}
            )
    except zipfile.BadZipFile as exc:
        raise SlicerError("slice_failed", f"OrcaSlicer's 3MF cannot be read: {exc}") from exc
    return info


def append_estimates(gcode: Path, info: dict[str, float]) -> None:
    """Write slice_info's numbers as the comments the G-code reader takes, if absent."""
    lines = []
    if "prediction" in info:
        seconds = int(info["prediction"])
        lines.append(
            f"; estimated printing time (normal mode) = {seconds // 3600}h "
            f"{seconds % 3600 // 60}m {seconds % 60}s"
        )
    if "weight" in info:
        lines.append(f"; total filament used [g] = {info['weight']}")
    if "used_m" in info:
        lines.append(f"; filament used [mm] = {info['used_m'] * 1000:.2f}")
    if lines:
        with gcode.open("a", encoding="utf-8") as handle:
            handle.write("\n" + "\n".join(lines) + "\n")


def run_orca(
    slicer: Slicer,
    choice: Choice,
    model: Path,
    folder: Path,
    datadir: Path,
    overrides: dict[str, Any] | None,
    rotation: dict[str, float] | None,
) -> tuple[SliceRun, list[str]]:
    machine = _flat(choice, "machine", choice.printer.name)
    process = _flat(choice, "process", choice.print_preset.name)
    filament = _flat(choice, "filament", choice.filament.name)
    applied = apply_orca_overrides([machine, process, filament], overrides)
    files = {}
    for kind, data in (("machine", machine), ("process", process), ("filament", filament)):
        files[kind] = folder / f"{kind}.json"
        files[kind].write_text(json.dumps(data, indent=1), encoding="utf-8")
    archive = folder / "slice.3mf"
    command = [
        str(slicer.binary),
        "--datadir",
        str(datadir),
        "--load-settings",
        f"{files['machine']};{files['process']}",
        "--load-filaments",
        str(files["filament"]),
    ]
    for axis, flag in (("x", "--rotate-x"), ("y", "--rotate-y"), ("z", "--rotate")):
        angle = float((rotation or {}).get(axis) or 0.0)
        if angle:
            command += [flag, f"{angle:g}"]
    command += [
        "--slice",
        "0",
        "--export-3mf",
        archive.name,
        "--outputdir",
        str(folder),
        str(model),
    ]
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SLICE_TIMEOUT_S,
            check=False,
            stdin=subprocess.DEVNULL,
            cwd=folder,
        )
    except subprocess.TimeoutExpired as exc:
        raise SlicerError(
            "slice_timeout",
            f"{slicer.describe()} did not finish slicing in {SLICE_TIMEOUT_S // 60} minutes.",
        ) from exc
    except OSError as exc:
        raise SlicerError("slicer_missing", f"{slicer.binary} could not be started: {exc}") from exc
    log = (done.stdout or "") + ("\n" + done.stderr if done.stderr else "")
    (folder / "slicer.log").write_text(log, encoding="utf-8")
    if done.returncode != 0 or not archive.is_file():
        tail = [
            line.strip()
            for line in log.splitlines()
            if line.strip() and not _PROGRESS.match(line.strip())
        ][-6:]
        raise SlicerError(
            "slice_failed",
            f"{slicer.describe()} could not slice {model.name}"
            + (": " + " ".join(tail) if tail else f" (exit code {done.returncode})."),
            exit_code=done.returncode,
        )
    gcode = folder / "slice.gcode"
    info = extract_gcode(archive, gcode)
    append_estimates(gcode, info)
    return SliceRun(gcode=gcode, warnings=read_warnings(log), log=log), applied


__all__ = [
    "append_estimates",
    "apply_orca_overrides",
    "extract_gcode",
    "read_slice_info",
    "run_orca",
]
