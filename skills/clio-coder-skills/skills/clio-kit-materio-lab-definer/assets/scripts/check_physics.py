#!/usr/bin/env python3
"""
check_physics; scan Materio task outputs for physically impossible or
implausible numeric values.

It reads generated protocol / results / input files, extracts (value, unit)
pairs, and checks them against physical bounds:

  * absolute temperature below 0 K (or below -273.15 C / -459.67 F)  -> ERROR
  * density <= 0, or implausibly high (> 30 g/cm3)                   -> ERROR / WARN
  * a fraction unit (wt%, at%, mol%, vol%) outside 0-100             -> ERROR
  * a plain % tied to a bounded quantity (porosity, purity, ...)     -> ERROR
  * an alloy composition block whose wt%/at% values do not sum ~100  -> WARN

DESIGN; to keep false positives low, v1 only asserts things that are
unconditionally true. It does not flag a value merely for being negative:
many materials quantities are legitimately negative (DFT total / formation
energies, binding enthalpies, stresses, thermal-expansion mismatch,
coordinates), as are temperature differences, rates, and gradients. Only
quantities that are non-negative by definition (absolute temperature,
density, a bounded fraction) are sign-checked.

Exit status: non-zero if any ERROR. Zero deps (Python stdlib only).

Usage:
  python3 assets/scripts/check_physics.py .research/tasks/task-01/*.md
  python3 assets/scripts/check_physics.py --json protocol.md
"""
import argparse
import json
import re
import sys

# A signed number that is NOT the tail of a range like "250-300" or "5.0-6".
# The lookbehind rejects a preceding digit/dot/comma so range internals and
# en-dash ranges ("250–300", en-dash is not a sign) are not read as negative.
_NUM = r"(?<![\d.,])([-−]?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?)"


def _to_float(s):
    return float(s.replace(",", "").replace("−", "-"))


# ---- findings -------------------------------------------------------------

class Findings:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, path, line, rule, msg):
        self.errors.append((path, line, rule, msg))

    def warn(self, path, line, rule, msg):
        self.warnings.append((path, line, rule, msg))


# ---- individual checks ----------------------------------------------------

def _iter_matches(pattern, text):
    for m in re.finditer(pattern, text):
        line_no = text.count("\n", 0, m.start()) + 1
        yield m, line_no


# A temperature UNIT also appears in signed quantities that are legitimately
# negative: differences (ΔT), rates (K/s), and gradients (K/mm). We only
# sign-check a value that reads as an ABSOLUTE temperature; one not followed
# by "/" (a rate/gradient) and not on a line about a delta/rate/gradient.
_TEMP_SIGNED_CTX = re.compile(
    r"Δ|∆|\bdelta\b|\bgradient\b|\brate\b|\bchange\b|\bdifference\b",
    re.IGNORECASE)


def check_temperature(text, path, fnd):
    for ln, line in enumerate(text.splitlines(), 1):
        signed = _TEMP_SIGNED_CTX.search(line)
        for m in re.finditer(_NUM + r"\s*K\b(?!\s*/)", line):
            v = _to_float(m.group(1))
            if v < 0 and not signed:
                fnd.error(path, ln, "temp-kelvin",
                          f"{v:g} K is below absolute zero (0 K)")
            elif v > 100000:
                fnd.warn(path, ln, "temp-kelvin",
                         f"{v:g} K is implausibly high; check units")
        for m in re.finditer(_NUM + r"\s*(?:°\s*C|degC|℃)(?!\s*/)", line):
            v = _to_float(m.group(1))
            if v < -273.15 and not signed:
                fnd.error(path, ln, "temp-celsius",
                          f"{v:g} C is below absolute zero (-273.15 C)")
        for m in re.finditer(_NUM + r"\s*(?:°\s*F|degF|℉)(?!\s*/)", line):
            v = _to_float(m.group(1))
            if v < -459.67 and not signed:
                fnd.error(path, ln, "temp-fahrenheit",
                          f"{v:g} F is below absolute zero (-459.67 F)")


# density units -> factor to convert into g/cm^3
_DENSITY = [
    (r"g\s*/\s*cm\s*(?:\^?3|³)", 1.0),
    (r"g\s*/\s*cc\b", 1.0),
    (r"g\s*/\s*mL\b", 1.0),
    (r"kg\s*/\s*m\s*(?:\^?3|³)", 0.001),
]


def check_density(text, path, fnd):
    for unit_re, factor in _DENSITY:
        for m, ln in _iter_matches(_NUM + r"\s*(?:" + unit_re + r")", text):
            v = _to_float(m.group(1)) * factor
            if v <= 0:
                fnd.error(path, ln, "density",
                          f"density {v:g} g/cm^3 must be positive")
            elif v > 30:
                fnd.warn(path, ln, "density",
                         f"density {v:g} g/cm^3 exceeds any known solid "
                         "(osmium ~22.6); check value/units")


# Fraction units are bounded 0-100 by definition.
_FRACTION_UNITS = r"(?:wt|at|mol|vol)\s*\.?\s*%"
# A bare % counts as a bounded fraction only near one of these words.
_BOUNDED_KEYWORDS = re.compile(
    r"porosit|relative density|rel\.?\s*density|densit|purit|"
    r"fraction|content|concentrat|abundance|yield|efficien|"
    r"conversion|selectivit|recover", re.IGNORECASE)
# ...but not when the % is a relative change (which can exceed 100%), e.g.
# "efficiency improved by 250%".
_RELATIVE_CHANGE = re.compile(
    r"\b(?:improv\w*|increas\w*|decreas\w*|reduc\w*|gain\w*|higher|lower\w*|"
    r"relativ\w*|baselin\w*|faster|slower|growth)\b|×|\bx\d",
    re.IGNORECASE)


def check_fractions(text, path, fnd):
    for m, ln in _iter_matches(_NUM + r"\s*(?:" + _FRACTION_UNITS + r")", text):
        v = _to_float(m.group(1))
        if v < 0 or v > 100:
            fnd.error(path, ln, "fraction",
                      f"{v:g}% is outside the valid 0-100% range")
    for line_no, line in enumerate(text.splitlines(), 1):
        if not _BOUNDED_KEYWORDS.search(line) or _RELATIVE_CHANGE.search(line):
            continue
        for m in re.finditer(_NUM + r"\s*%(?![a-zA-Z])", line):
            v = _to_float(m.group(1))
            if v < 0 or v > 100:
                fnd.error(path, line_no, "fraction",
                          f"{v:g}% is outside 0-100% for a bounded quantity")


def check_composition(text, path, fnd):
    """
    Flag an alloy composition group whose wt%/at%/mol% values do not sum to
    ~100%. A group is either an inline list (>=2 same-unit values on one line,
    e.g. "Ti 90 wt%, Al 6 wt%, V 4 wt%") or a contiguous run of lines each
    carrying exactly one same-unit value (a vertical table). Conservative:
    needs >=2 values and a total in a sane window, so unrelated scattered
    percentages are not summed together.
    """
    unit = re.compile(_NUM + r"\s*(" + _FRACTION_UNITS + r")")
    lines = text.splitlines()

    def check_group(values, kind, ref_line, span):
        if len(values) >= 2:
            total = sum(values)
            if 80 <= total <= 120 and abs(total - 100) > 1.0:
                fnd.warn(path, ref_line, "composition",
                         f"{kind} composition {span} sums to {total:g}% "
                         f"(expected ~100%)")

    run, start_line, run_kind = [], None, None

    def flush(end_line):
        span = (f"block (lines {start_line}-{end_line})"
                if end_line != start_line else f"on line {start_line}")
        check_group(run, run_kind, start_line, span)

    for i, line in enumerate(lines, 1):
        by_kind = {}
        for val, u in unit.findall(line):
            k = re.sub(r"[\s.]", "", u).lower()
            by_kind.setdefault(k, []).append(_to_float(val))

        inline = {k: v for k, v in by_kind.items() if len(v) >= 2}
        if inline:                                   # inline composition list
            if run:
                flush(i - 1)
                run, start_line, run_kind = [], None, None
            for k, vals in inline.items():
                check_group(vals, k, i, f"on line {i}")
            continue

        if len(by_kind) == 1 and sum(len(v) for v in by_kind.values()) == 1:
            k = next(iter(by_kind))                  # single value: vertical run
            if run and k != run_kind:
                flush(i - 1)
                run, start_line, run_kind = [], None, None
            if not run:
                start_line, run_kind = i, k
            run.append(by_kind[k][0])
        else:
            if run:
                flush(i - 1)
            run, start_line, run_kind = [], None, None
    if run:
        flush(len(lines))


CHECKS = [check_temperature, check_density, check_fractions, check_composition]


# ---- driver ---------------------------------------------------------------

def scan(paths):
    fnd = Findings()
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"WARN: cannot read {p}: {e}", file=sys.stderr)
            continue
        for check in CHECKS:
            check(text, p, fnd)
    return fnd


def main():
    ap = argparse.ArgumentParser(description="Physical sanity checks for "
                                             "Materio task outputs.")
    ap.add_argument("paths", nargs="+", help="files to scan (.md, input files)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    unreadable = []
    for path in args.paths:
        try:
            with open(path, encoding="utf-8") as source:
                source.read()
        except (OSError, UnicodeError) as error:
            unreadable.append({"file": path, "reason": str(error)})
    readable = [path for path in args.paths if path not in {item["file"] for item in unreadable}]
    fnd = scan(readable)

    if args.json:
        print(json.dumps({
            "coverage": {"files_requested": len(args.paths), "files_read": len(readable), "unreadable": unreadable},
            "errors": [{"file": p, "line": l, "rule": r, "message": m}
                       for p, l, r, m in fnd.errors],
            "warnings": [{"file": p, "line": l, "rule": r, "message": m}
                         for p, l, r, m in fnd.warnings],
        }, indent=2))
    else:
        def emit(items, label):
            if not items:
                return
            print(f"\n{label} ({len(items)}):")
            for p, l, r, m in sorted(items):
                print(f"  {p}:{l}  [{r}]  {m}")
        emit(fnd.errors, "IMPOSSIBLE (errors)")
        emit(fnd.warnings, "IMPLAUSIBLE (warnings)")
        n = len(fnd.errors) + len(fnd.warnings)
        print(f"\ncheck-physics: {len(fnd.errors)} error(s), "
              f"{len(fnd.warnings)} warning(s)."
              + ("" if n else " No physical-sanity issues found."))

    return 2 if unreadable else 1 if fnd.errors else 0


if __name__ == "__main__":
    sys.exit(main())
