"""Read a sliced G-code file once, line by line, into per-layer typed toolpaths.

The file is streamed; only the layer being read is held in memory. Each finished layer
is appended to one binary file of toolpaths (``layers.bin``): for every feature type
in the layer, its polylines as ``float32`` x/y pairs, a ``uint32`` vertex count per
polyline and the ``uint32`` G-code line number each polyline starts on. An index
(returned, and saved as JSON by the caller) records each layer's Z, height, time and
extrusion by type and where its toolpaths sit in the binary file.

Times: the slicer reports one total. Per-move time is the move's length over its
feedrate (no acceleration), and every layer's and type's time is then scaled so the
times add up to the slicer's own total; ``time_basis`` says so. Extrusion is the
filament length pushed (E), turned into grams with the file's own filament diameter
and density.

Both PrusaSlicer's markers (``;LAYER_CHANGE``, ``;Z:``, ``;TYPE:``) and OrcaSlicer's
(``; CHANGE_LAYER``, ``; Z_HEIGHT:``, ``; FEATURE:``) are read. Arcs (``G2``/``G3``
with ``I``/``J``) are split into short straight pieces.
"""

from __future__ import annotations

import itertools
import math
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from .estimates import Estimates, bed_outline, read_comment, setting_numbers
from .features import TRAVEL, feature_key

#: Arcs are split into pieces of about this length (mm), at most ARC_MAX_PIECES.
ARC_PIECE_MM = 1.0
ARC_MAX_PIECES = 64
DEFAULT_FEED_MM_MIN = 1500.0


class GcodeError(ValueError):
    """The file is not G-code this reader understands; ``reason`` is a stable code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class _Polylines:
    xy: array = field(default_factory=lambda: array("f"))
    counts: array = field(default_factory=lambda: array("I"))
    lines: array = field(default_factory=lambda: array("I"))


@dataclass
class _Stats:
    length_mm: float = 0.0
    e_mm: float = 0.0
    time_s: float = 0.0
    vertices: int = 0


class _Layer:
    def __init__(self, index: int, z: float | None, height: float | None) -> None:
        self.index, self.z, self.height = index, z, height
        self.paths: dict[str, _Polylines] = {}
        self.stats: dict[str, _Stats] = {}
        self.other_time_s = 0.0
        self.open: str | None = None  # feature of the polyline being extended
        self.end: tuple[float, float] | None = None
        self.bounds = [math.inf, math.inf, -math.inf, -math.inf]
        self.extents: dict[str, list[float]] = {}

    def add(self, feature: str, x0: float, y0: float, x1: float, y1: float, line: int) -> None:
        poly = self.paths.get(feature)
        if poly is None:
            poly = self.paths[feature] = _Polylines()
        if self.open != feature or self.end != (x0, y0):
            poly.xy.extend((x0, y0))
            poly.counts.append(1)
            poly.lines.append(line)
        poly.xy.extend((x1, y1))
        poly.counts[-1] += 1
        self.open, self.end = feature, (x1, y1)
        # The bounds are the print's: travel and the start G-code's purge line are not.
        if feature != TRAVEL:
            # Where each type prints on the layer (a bridge's span, an overhang's side).
            f = self.extents.setdefault(feature, [math.inf, math.inf, -math.inf, -math.inf])
            f[0], f[1] = min(f[0], x0, x1), min(f[1], y0, y1)
            f[2], f[3] = max(f[2], x0, x1), max(f[3], y0, y1)
        if feature not in (TRAVEL, "custom"):
            b = self.bounds
            b[0], b[1] = min(b[0], x0, x1), min(b[1], y0, y1)
            b[2], b[3] = max(b[2], x0, x1), max(b[3], y0, y1)

    def stat(self, feature: str) -> _Stats:
        found = self.stats.get(feature)
        if found is None:
            found = self.stats[feature] = _Stats()
        return found

    def raw_time(self) -> float:
        return self.other_time_s + sum(s.time_s for s in self.stats.values())

    def extruded(self) -> bool:
        return any(k != TRAVEL and s.e_mm > 0 for k, s in self.stats.items())


@dataclass
class ParsedSlice:
    """The index of a parsed G-code file."""

    layers: list[dict[str, Any]]
    estimates: Estimates
    bed: list[list[float]] | None
    bounds: list[float] | None
    features: list[str]
    lines: int
    vertices: int
    time_basis: str
    grams_per_mm: float | None


class _Writer:
    def __init__(self, handle: IO[bytes]) -> None:
        self.handle, self.offset = handle, 0

    def put(self, values: array) -> int:
        if sys.byteorder != "little":
            values = array(values.typecode, values)
            values.byteswap()
        start = self.offset
        data = values.tobytes()
        self.handle.write(data)
        self.offset += len(data)
        return start


def _words(code: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for word in code.split()[1:]:
        if len(word) < 2:
            continue
        try:
            out[word[0].upper()] = float(word[1:])
        except ValueError:
            continue
    return out


def _marker_value(text: str) -> float | None:
    try:
        return float(text.split(":", 1)[1].strip())
    except (IndexError, ValueError):
        return None


def parse_gcode(path: Path, binary_out: Path) -> ParsedSlice:
    """Stream ``path`` once; write toolpaths to ``binary_out``; return the index.

    Raises:
        GcodeError: ``gcode_unreadable`` when the file cannot be read, ``gcode_empty``
            when it has no extruding move.
    """
    try:
        source = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise GcodeError("gcode_unreadable", f"The G-code {path} cannot be read: {exc}") from exc
    with source, binary_out.open("wb") as sink:
        return _Reader(_Writer(sink)).run(source)


class _Reader:
    def __init__(self, writer: _Writer) -> None:
        self.writer = writer
        self.estimates = Estimates()
        self.layers: list[dict[str, Any]] = []
        self.layer = _Layer(0, None, None)
        self.feature = "custom"
        self.x = self.y = self.z = 0.0
        self.e_abs = 0.0
        self.feed = DEFAULT_FEED_MM_MIN
        self.relative_xyz = False
        self.relative_e = False
        self.wiping = False
        self.vertices = 0
        self.seen: set[str] = set()
        self.bounds = [math.inf, math.inf, -math.inf, -math.inf]
        self.lineno = 0

    # -- comments ------------------------------------------------------------
    def comment(self, text: str) -> None:
        body = text.lstrip(";").strip()
        upper = body.upper()
        if upper in {"LAYER_CHANGE", "CHANGE_LAYER"}:
            self.layer.open = None
            return
        if upper.startswith(("Z:", "Z_HEIGHT:")):
            z = _marker_value(body)
            if z is not None:
                self.new_layer(z)
            return
        if upper.startswith(("HEIGHT:", "LAYER_HEIGHT:")):
            height = _marker_value(body)
            if height is not None and self.layer.height is None:
                self.layer.height = height
            return
        if upper.startswith(("TYPE:", "FEATURE:")):
            self.feature = feature_key(body.split(":", 1)[1])
            self.layer.open = None
            return
        if upper == "WIPE_START":
            self.wiping = True
            return
        if upper == "WIPE_END":
            self.wiping = False
            return
        if text.startswith(("; ", ";")):
            read_comment(text, self.estimates)

    def new_layer(self, z: float) -> None:
        """A Z marker starts a new layer when Z changes (duplicate markers are one)."""
        current = self.layer
        if current.z is None:
            current.z = z
            return
        if abs(z - current.z) < 1e-6:
            return
        self.flush()
        self.layer = _Layer(current.index + 1, z, None)

    # -- moves ---------------------------------------------------------------
    def move(self, words: dict[str, float], arc: int) -> None:
        if "F" in words and words["F"] > 0:
            self.feed = words["F"]
        x0, y0, z0 = self.x, self.y, self.z
        if self.relative_xyz:
            x1, y1 = x0 + words.get("X", 0.0), y0 + words.get("Y", 0.0)
            z1 = z0 + words.get("Z", 0.0)
        else:
            x1, y1, z1 = words.get("X", x0), words.get("Y", y0), words.get("Z", z0)
        if "E" in words:
            de = words["E"] if self.relative_e else words["E"] - self.e_abs
            if not self.relative_e:
                self.e_abs = words["E"]
        else:
            de = 0.0
        speed = self.feed / 60.0
        points = [(x0, y0), (x1, y1)]
        if arc and ("I" in words or "J" in words):
            points = _arc_points(x0, y0, x1, y1, words.get("I", 0.0), words.get("J", 0.0), arc)
        planar = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(points))
        dz = z1 - z0
        distance = math.hypot(planar, dz)
        layer = self.layer
        if planar > 1e-9:
            extruding = de > 1e-9 and not self.wiping
            kind = self.feature if extruding else TRAVEL
            stat = layer.stat(kind)
            stat.length_mm += planar
            stat.time_s += distance / speed
            if extruding:
                stat.e_mm += de
                self.seen.add(kind)
            for a, b in itertools.pairwise(points):
                layer.add(kind, a[0], a[1], b[0], b[1], self.lineno)
        else:
            # Retracting and priming in place: time only. A prime pushes back what the
            # retraction pulled, so it is not extrusion.
            layer.other_time_s += (distance if distance > 1e-9 else abs(de)) / speed
        self.x, self.y, self.z = x1, y1, z1

    def code(self, line: str) -> None:
        code = line.split(";", 1)[0].strip()
        if not code:
            return
        head = code.split(None, 1)[0].upper()
        if head in {"G0", "G1", "G00", "G01"}:
            self.move(_words(code), 0)
        elif head in {"G2", "G02"}:
            self.move(_words(code), 2)
        elif head in {"G3", "G03"}:
            self.move(_words(code), 3)
        elif head == "G90":
            self.relative_xyz = False
        elif head == "G91":
            self.relative_xyz = True
        elif head == "M82":
            self.relative_e = False
        elif head == "M83":
            self.relative_e = True
        elif head == "G92":
            words = _words(code)
            self.e_abs = words.get("E", self.e_abs)
            self.x, self.y, self.z = (
                words.get("X", self.x),
                words.get("Y", self.y),
                words.get("Z", self.z),
            )

    # -- output --------------------------------------------------------------
    def flush(self) -> None:
        layer = self.layer
        entry: dict[str, Any] = {
            "layer": layer.index,
            "z": round(layer.z if layer.z is not None else self.z, 4),
            "height": layer.height,
            "raw_time_s": layer.raw_time(),
            "raw_other_s": layer.other_time_s,
            "features": {},
            "paths": {},
        }
        for key, stat in layer.stats.items():
            entry["features"][key] = {
                "length_mm": round(stat.length_mm, 3),
                "e_mm": round(stat.e_mm, 4),
                "raw_time_s": stat.time_s,
            }
            if key in layer.extents:
                entry["features"][key]["bounds"] = [round(v, 2) for v in layer.extents[key]]
        for key, poly in layer.paths.items():
            n = len(poly.xy) // 2
            self.vertices += n
            entry["features"].setdefault(key, {})["vertices"] = n
            entry["paths"][key] = [
                self.writer.put(poly.xy),
                n,
                self.writer.put(poly.counts),
                len(poly.counts),
                self.writer.put(poly.lines),
            ]
        if all(math.isfinite(v) for v in layer.bounds):
            b = self.bounds
            b[0], b[1] = min(b[0], layer.bounds[0]), min(b[1], layer.bounds[1])
            b[2], b[3] = max(b[2], layer.bounds[2]), max(b[3], layer.bounds[3])
        if layer.paths or layer.stats:
            self.layers.append(entry)

    def run(self, source: IO[str]) -> ParsedSlice:
        for raw in source:
            self.lineno += 1
            line = raw.strip()
            if not line:
                continue
            if line[0] == ";":
                self.comment(line)
            else:
                self.code(line)
        self.flush()
        if not any(
            any(k != TRAVEL and f.get("e_mm", 0) > 0 for k, f in layer["features"].items())
            for layer in self.layers
        ):
            raise GcodeError("gcode_empty", "The G-code has no extruding move.")
        return self.finish()

    def finish(self) -> ParsedSlice:
        layers = [entry for entry in self.layers]
        for i, entry in enumerate(layers):
            entry["layer"] = i
            if entry["height"] is None:
                below = layers[i - 1]["z"] if i else 0.0
                entry["height"] = round(entry["z"] - below, 4)
        raw_total = sum(entry["raw_time_s"] for entry in layers)
        total = self.estimates.print_time_s
        if total and raw_total > 0:
            scale, basis = total / raw_total, "feedrate estimate scaled to the slicer's total"
        else:
            scale, basis = 1.0, "feedrate estimate (the G-code carries no slicer total)"
        diameters = setting_numbers(self.estimates.settings.get("filament_diameter"))
        densities = setting_numbers(self.estimates.settings.get("filament_density"))
        grams_per_mm = (
            math.pi * (diameters[0] / 2) ** 2 / 1000.0 * densities[0]
            if diameters and densities and densities[0] > 0
            else None
        )
        for entry in layers:
            entry["time_s"] = round(entry.pop("raw_time_s") * scale, 3)
            # Retracting, priming and Z moves: time that no toolpath type owns.
            entry["other_time_s"] = round(entry.pop("raw_other_s") * scale, 3)
            grams = 0.0
            for key, feature in entry["features"].items():
                feature["time_s"] = round(feature.pop("raw_time_s", 0.0) * scale, 3)
                if grams_per_mm is not None and key != TRAVEL:
                    feature["grams"] = round(feature.get("e_mm", 0.0) * grams_per_mm, 4)
                    grams += feature["grams"]
            entry["filament_g"] = round(grams, 4) if grams_per_mm is not None else None
        if self.estimates.layer_count is None:
            self.estimates.layer_count = len(layers)
        bounds = self.bounds if all(math.isfinite(v) for v in self.bounds) else None
        return ParsedSlice(
            layers=layers,
            estimates=self.estimates,
            bed=bed_outline(self.estimates.settings),
            bounds=[round(v, 3) for v in bounds] if bounds else None,
            features=sorted(self.seen),
            lines=self.lineno,
            vertices=self.vertices,
            time_basis=basis,
            grams_per_mm=grams_per_mm,
        )


def _arc_points(
    x0: float, y0: float, x1: float, y1: float, i: float, j: float, direction: int
) -> list[tuple[float, float]]:
    """Straight pieces of a G2 (clockwise) or G3 (counter-clockwise) arc."""
    cx, cy = x0 + i, y0 + j
    radius = math.hypot(i, j)
    start = math.atan2(y0 - cy, x0 - cx)
    end = math.atan2(y1 - cy, x1 - cx)
    sweep = end - start
    if direction == 2 and sweep >= 0:
        sweep -= 2 * math.pi
    elif direction == 3 and sweep <= 0:
        sweep += 2 * math.pi
    pieces = max(1, min(ARC_MAX_PIECES, math.ceil(abs(sweep) * radius / ARC_PIECE_MM)))
    points = [(x0, y0)]
    for k in range(1, pieces):
        angle = start + sweep * k / pieces
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    points.append((x1, y1))
    return points


def read_paths(binary: Path, entry: dict[str, Any]) -> dict[str, dict[str, array]]:
    """The toolpaths of one layer from the binary file, by feature."""
    out: dict[str, dict[str, array]] = {}
    with binary.open("rb") as handle:
        for key, (xy_off, n, counts_off, polylines, lines_off) in entry["paths"].items():
            xy, counts, lines = array("f"), array("I"), array("I")
            handle.seek(xy_off)
            xy.frombytes(handle.read(n * 2 * 4))
            handle.seek(counts_off)
            counts.frombytes(handle.read(polylines * 4))
            handle.seek(lines_off)
            lines.frombytes(handle.read(polylines * 4))
            if sys.byteorder != "little":
                for values in (xy, counts, lines):
                    values.byteswap()
            out[key] = {"xy": xy, "counts": counts, "lines": lines}
    return out


__all__ = ["GcodeError", "ParsedSlice", "parse_gcode", "read_paths"]
