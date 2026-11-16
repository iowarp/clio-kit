"""Lean words for the agent: what was computed, the key numbers with units, the files.
The full typed data is the tool's structured content; spectra stay in CSV files."""

from __future__ import annotations

from typing import Any


def _absorber(a: dict[str, Any]) -> str:
    if "sites" in a:
        return f"sites {', '.join(str(s) for s in a['sites'])} ({', '.join(a['elements'])})"
    return f"{a['element']} (all {a['sites_in_file']} {a['element']} atoms in the file)"


def _spec_line(s: dict[str, Any]) -> str:
    wl = s.get("white_line")
    parts = [f"E0 {s['e0_eV']:.2f} eV"]
    if wl:
        parts.append(f"white line {wl['energy_eV']:.2f} eV")
    if s["normalized"]:
        parts.append(f"edge step {s['edge_step']:.3g}")
    else:
        parts.append(f"not normalized: {s['normalization_skipped']}")
    return ", ".join(parts)


def run_text(r: dict[str, Any]) -> str:
    st, cfg, log = r["structure"], r["settings"], r.get("log", {})
    method = "multiple scattering (Green)" if cfg["method"] == "green" else "finite differences"
    head = (
        f"FDMNES {cfg['edge']}-edge XANES of {st['formula']} ({st['kind']}, {st['n_atoms']} "
        f"atoms), absorber {_absorber(st['absorber'])}: {method}, cluster radius "
        f"{cfg['radius_A']:g} Å"
        + (", SCF" if cfg["scf"] else "")
        + (", quadrupole" if cfg["quadrupole"] else "")
        + (", magnetic" if cfg["magnetism"] else "")
        + f"; {r['fdmnes']['wall_time_s']:g} s."
    )
    lines = [head]
    facts = []
    if "threshold" in log:
        facts.append(f"{log['threshold']} at {log.get('edge_energy_eV', 0):g} eV")
    if "inequivalent_absorbers" in log:
        facts.append(f"{log['inequivalent_absorbers']} inequivalent absorbing site(s)")
    if "point_group" in log:
        facts.append(f"point group {log['point_group']}")
    if "cluster" in log:
        facts.append(f"{log['cluster']['atoms']} atoms in the cluster")
    if "scf_cycles" in log:
        facts.append(f"{log['scf_cycles']} SCF cycles")
    if facts:
        lines.append("FDMNES: " + "; ".join(facts) + ".")
    for s in r["spectra"]:
        lines.append(
            f"Spectrum {s['csv']} ({s['points']} points, {s['energy_range_eV'][0]:g} to "
            f"{s['energy_range_eV'][1]:g} eV from the Fermi level): {_spec_line(s)}."
        )
    lines += [f"Note: {n}" for n in r["notes"]]
    lines.append(f"Result: {r['results_file']}; inputs and FDMNES logs in {r['run_dir']}.")
    return "\n".join(lines)


def read_text(d: dict[str, Any]) -> str:
    shift = f", energies shifted by {d['energy_shift_eV']:g} eV" if d["energy_shift_eV"] else ""
    return (
        f"Read {d['points']} points from {d['source']} ({d['energy_range_eV'][0]:g} to "
        f"{d['energy_range_eV'][1]:g} eV{shift}): {_spec_line(d)}.\nWrote {d['csv']}."
    )


def compare_text(d: dict[str, Any]) -> str:
    lines = [
        f"Compared {len(d['spectra'])} normalized spectra over {d['range_eV'][0]:g} to "
        f"{d['range_eV'][1]:g} eV ({d['points']} points"
        + (", each shifted so its E0 matches the first" if d["align"] else "")
        + "):"
    ]
    for s in d["spectra"]:
        lines.append(
            f"- {s['label']}: E0 {s['e0_eV']:.2f} eV"
            + (f", shifted {s['shift_applied_eV']:+.2f} eV" if d["align"] else "")
            + f", R-factor {s['r_factor_vs_first']:.4f}, correlation "
            f"{s['correlation_vs_first']:.3f} vs the first"
        )
    lines.append(f"Columns on one energy grid: {d['csv']}.")
    return "\n".join(lines)


def status_text(d: dict[str, Any]) -> str:
    ex = d["executable"]
    if ex["found"]:
        return f"FDMNES is ready: {ex['command']} (found via {ex['source']})."
    return f"FDMNES is not available: {ex['problem']}. {d['install']}"
