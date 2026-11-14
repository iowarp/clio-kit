"""Run PrusaSlicer's command line on a model with flattened presets.

The command is::

    prusa-slicer --datadir <server's own folder> --loglevel 2 --load settings.ini \\
        [--rotate-x A --rotate-y B --rotate C] --export-gcode --output slice.gcode MODEL

``settings.ini`` is the printer, print and filament presets flattened
(:mod:`..profiles.ini_bundle`), with the person's overrides on top and
``binary_gcode = 0`` so the output is plain text G-code. ``--datadir`` points at the
server's own folder, so the person's PrusaSlicer settings are never read or written.
The slicer prints its warnings ("Detected print stability issues …") on its output;
they are returned in words.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import SlicerError
from ..profiles.ini_bundle import settings_text
from ..profiles.select import Choice
from .locate import Slicer

#: Keys only the preset tree uses, never loaded as settings.
PRESET_ONLY = frozenset(
    {
        "compatible_printers",
        "compatible_printers_condition",
        "compatible_prints",
        "compatible_prints_condition",
        "inherits",
        "alias",
        "renamed_from",
    }
)
#: A slice that runs this long has stopped answering; its process is ended.
SLICE_TIMEOUT_S = 1800


@dataclass
class SliceRun:
    """What one slicer run produced."""

    gcode: Path
    warnings: list[str] = field(default_factory=list)
    log: str = ""


def coerce(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return ",".join(coerce(v) for v in value)
    return str(value)


def merged_settings(choice: Choice) -> dict[str, str]:
    settings: dict[str, str] = {}
    for preset in (choice.printer, choice.print_preset, choice.filament):
        settings.update({k: v for k, v in preset.settings.items() if k not in PRESET_ONLY})
    settings["printer_settings_id"] = choice.printer.name
    settings["print_settings_id"] = choice.print_preset.name
    settings["filament_settings_id"] = f'"{choice.filament.name}"'
    return settings


def apply_overrides(settings: dict[str, str], overrides: dict[str, Any] | None) -> list[str]:
    """Set the person's overrides; unknown keys are refused with the nearest names."""
    import difflib

    applied = []
    for key, value in (overrides or {}).items():
        if key not in settings:
            close = difflib.get_close_matches(key, list(settings), n=5, cutoff=0.5)
            raise SlicerError(
                "override_unknown",
                f"{key!r} is not a setting of this slicer's presets"
                + (f"; close names: {', '.join(close)}" if close else "")
                + ". Overrides use the slicer's own setting names, for example "
                "support_material, fill_density, brim_width, perimeters.",
                key=key,
                close=close,
            )
        settings[key] = coerce(value)
        applied.append(f"{key} = {settings[key]}")
    return applied


_PROGRESS = re.compile(r"^\d+\s*=>")


def read_warnings(output: str) -> list[str]:
    """The slicer's warnings, each as one sentence-like line."""
    warnings: list[str] = []
    block: list[str] | None = None
    for raw in output.splitlines():
        line = raw.strip()
        lowered = line.lower()
        if "warning:" in lowered and not lowered.startswith("["):
            if block:
                warnings.append(" ".join(block))
            block = [line.split(":", 1)[1].strip()]
            continue
        if block is not None:
            if _PROGRESS.match(line) or lowered.startswith("slicing result exported"):
                warnings.append(" ".join(block))
                block = None
            elif line:
                block.append(line if line.endswith((".", ":")) else line + ".")
    if block:
        warnings.append(" ".join(block))
    return [re.sub(r"\s+", " ", w).replace(":.", ":").strip() for w in warnings if w]


def run_prusa(
    slicer: Slicer,
    choice: Choice,
    model: Path,
    folder: Path,
    datadir: Path,
    overrides: dict[str, Any] | None,
    rotation: dict[str, float] | None,
) -> tuple[SliceRun, list[str]]:
    """Slice ``model`` into ``folder/slice.gcode``; returns the run and the overrides set."""
    settings = merged_settings(choice)
    applied = apply_overrides(settings, overrides)
    settings["binary_gcode"] = "0"
    (folder / "settings.ini").write_text(settings_text(settings), encoding="utf-8")
    output = folder / "slice.gcode"
    command = [
        str(slicer.binary),
        "--datadir",
        str(datadir),
        "--loglevel",
        "2",
        "--load",
        str(folder / "settings.ini"),
    ]
    for axis, flag in (("x", "--rotate-x"), ("y", "--rotate-y"), ("z", "--rotate")):
        angle = float((rotation or {}).get(axis) or 0.0)
        if angle:
            command += [flag, f"{angle:g}"]
    command += ["--export-gcode", "--output", str(output), str(model)]
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SLICE_TIMEOUT_S,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise SlicerError(
            "slice_timeout",
            f"{slicer.describe()} did not finish slicing in {SLICE_TIMEOUT_S // 60} minutes "
            "and was stopped. A simpler model or a larger layer height slices faster.",
        ) from exc
    except OSError as exc:
        raise SlicerError("slicer_missing", f"{slicer.binary} could not be started: {exc}") from exc
    log = (done.stdout or "") + ("\n" + done.stderr if done.stderr else "")
    (folder / "slicer.log").write_text(log, encoding="utf-8")
    if done.returncode != 0 or not output.is_file():
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
    return SliceRun(gcode=output, warnings=read_warnings(log), log=log), applied


__all__ = ["SliceRun", "apply_overrides", "merged_settings", "read_warnings", "run_prusa"]
