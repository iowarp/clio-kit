"""Lean words for the agent: what was done, key numbers with units, where the files are.
The full typed data is the tool's structured content; big arrays stay in files."""

from __future__ import annotations

from typing import Any


def _calc_label(calc: dict[str, Any]) -> str:
    parts = [calc["type"]]
    for key in ("model", "method", "basis"):
        if calc.get(key):
            parts.append(str(calc[key]))
    if calc.get("solvation"):
        parts.append(f"solvation {calc['solvation']}")
    return " ".join(parts[:1]) + (f" ({', '.join(parts[1:])})" if len(parts) > 1 else "")


def _energy(e: dict[str, float]) -> str:
    return f"{e['eV']:.6f} eV ({e['kJ_mol']:.2f} kJ/mol)"


def _freqs(modes: list[dict[str, Any]], limit: int = 12) -> str:
    labels = [m["label"] for m in modes]
    if len(labels) <= limit:
        return ", ".join(labels)
    return ", ".join(labels[:6]) + f", … , {', '.join(labels[-3:])}"


def structure_line(rec: dict[str, Any]) -> str:
    return f"{rec['formula']} ({rec['n_atoms']} atoms) → {rec['path']}"


def smiles_text(d: dict[str, Any]) -> str:
    lines = [
        (
            f"Built {d['formula']} from SMILES {d['canonical_smiles']} (canonical): "
            f"{d['embedding']} embedding (seed {d['seed']}), hydrogens added, "
            f"{d['force_field']} relaxed."
        ),
        f"Structure: {d['structure']['path']} (XYZ); SDF: {d['sdf']['path']}.",
        (
            f"Formal charge {d['formal_charge']}; {d['radical_electrons']} radical electron(s) "
            f"in the SMILES, i.e. multiplicity {d['multiplicity_from_smiles']} if they are "
            "parallel."
        ),
    ]
    lines += [f"Note: {n}" for n in d["notes"]]
    return "\n".join(lines)


def run_text(r: dict[str, Any]) -> str:
    calc = r["calculator"]
    head = (
        f"{r['driver']} of {r['input_structure']['formula']} with {_calc_label(calc)}, "
        f"charge {r['charge']}, multiplicity {r['multiplicity']}"
        + (
            ""
            if calc["models_spin"] or r["multiplicity"] == 1
            else f" (spin not modelled by {calc['type']})"
        )
        + f": E = {_energy(r['energy'])}."
    )
    lines = [head]
    if r.get("worker_environment", {}) and r["worker_environment"].get("built_now"):
        w = r["worker_environment"]
        lines.append(
            f"First use: built the {w['extra']} environment in {w.get('build_seconds')} s."
        )
    opt = r.get("optimization")
    if opt:
        state = "converged" if opt["converged"] else "did NOT converge"
        lines.append(
            f"Optimization ({opt['optimizer']}) {state} in {opt['steps']} steps "
            f"(fmax {opt['final_fmax_eV_per_A']} eV/Å, target "
            f"{opt['fmax_target_eV_per_A']})."
        )
    for c in r.get("final_constraint_values", []) or []:
        lines.append(f"Constrained {c['kind']} {c['indices']} = {c['value']} {c['unit']}.")
    vib = r.get("vibrations")
    if vib:
        if vib["n_imaginary"]:
            imag = ", ".join(f"{v}i" for v in vib["imaginary_cm1"])
            lines.append(
                f"Vibrations: {vib['n_modes']} modes, {vib['n_imaginary']} imaginary ({imag} cm⁻¹)."
            )
        else:
            lines.append(f"Vibrations: {vib['n_modes']} modes, no imaginary frequency.")
        lines.append(
            f"Frequencies (cm⁻¹): {_freqs(vib['modes'])}. ZPE "
            f"{vib['zpe']['eV']:.4f} eV ({vib['zpe']['kJ_mol']:.2f} kJ/mol)."
        )
    th = r.get("thermochemistry")
    if th:
        lines.append(
            f"Thermochemistry at {th['temperature_K']} K, {th['pressure_Pa']:.0f} Pa "
            f"({th['geometry']}, point group {th['point_group']}, symmetry number "
            f"{th['symmetry_number']}, spin S = {th['spin_S']}): "
            f"H = {_energy(th['enthalpy'])}, "
            f"S = {th['entropy']['eV_per_K']:.6g} eV/K ({th['entropy']['J_per_mol_K']:.2f} "
            f"J/(mol·K)), G = {_energy(th['gibbs_free_energy'])}."
        )
    ir = r.get("ir")
    if ir:
        peaks = ", ".join(
            f"{p['frequency_cm1']:.0f} ({p['intensity_km_mol']:.1f})" for p in ir["strongest"]
        )
        lines.append(f"Strongest IR peaks, cm⁻¹ (km/mol): {peaks}.")
    if r.get("dipole"):
        lines.append(f"Dipole {r['dipole']['debye']} D.")
    files = ["result.json"]
    if r.get("final_structure"):
        files.append("final.xyz")
    if opt:
        files += [
            "opt_trajectory.xyz (extended XYZ, energy per frame)",
            "opt_steps.csv (step, energy_eV, fmax_eV_per_A)",
        ]
    if vib:
        files += ["frequencies.csv", "normal_modes.npz", "modes/mode_NNN.xyz (animations)"]
    if ir:
        files += ["ir_spectrum.csv", "ir_peaks.csv"]
    lines.append(f"Files in {r['run_dir']}: " + ", ".join(files) + ".")
    lines += [f"Warning: {w}" for w in r.get("warnings", [])]
    lines += [f"Note: {n}" for n in r.get("notes", [])]
    return "\n".join(lines)


def inspect_text(d: dict[str, Any]) -> str:
    bonds = ", ".join(f"{b['atoms']} {b['length_A']:.3f}" for b in d["bonds"][:24])
    more = f" … ({d['n_bonds']} in all)" if d["n_bonds"] > 24 else ""
    lines = [
        f"{d['formula']}: {d['n_atoms']} atoms, {d['n_bonds']} bonds ({d['bond_rule']}), "
        f"{d['n_fragments']} fragment(s), extent {d['extent_A']} Å"
        + (f", frame {d['frame']} of {d['n_frames']}" if d.get("frame") is not None else "")
        + (f", energy {d['energy_eV']} eV" if d.get("energy_eV") is not None else "")
        + ".",
        f"Bonds (Å, 0-based indices): {bonds}{more}." if d["n_bonds"] else "No bonds.",
    ]
    for m in d.get("measurements") or []:
        lines.append(f"{m['kind']} {m['indices']} = {m['value']} {m['unit']}.")
    return "\n".join(lines)


def set_text(d: dict[str, Any]) -> str:
    lines = [
        f"{c['kind']} {c['indices']}: {c['before']} → {c['after']} {c['unit']} "
        f"(moved {len(c['moved_atoms'])} atoms)."
        for c in d["changes"]
    ]
    lines.append(f"New structure: {structure_line(d['structure'])}.")
    return "\n".join(lines)


def displace_text(d: dict[str, Any]) -> str:
    f = d.get("frequency_cm1")
    label = f"{abs(f)}i" if d.get("imaginary") and f is not None else f"{f}"
    lines = [
        (
            f"Displaced along mode {d.get('mode')} ({label} cm⁻¹) by {d['amplitude_A']} Å "
            "(largest atomic displacement):"
        )
    ]
    lines += [f"{s['direction']}: {s['structure']['path']}" for s in d["structures"]]
    return "\n".join(lines)


def frame_text(d: dict[str, Any]) -> str:
    extra = f", E = {d['energy_eV']} eV" if d.get("energy_eV") is not None else ""
    return (
        f"Frame {d['frame']} of {d['n_frames']} from {d['trajectory']}{extra} → "
        f"{d['structure']['path']}."
    )


def calculators_text(d: dict[str, Any]) -> str:
    lines = []
    for c in d["calculators"]:
        lines.append(f"{c['type']}: {c['state']} — {c['detail']}")
    if d.get("prepared_now"):
        lines.append("Prepared now: " + ", ".join(d["prepared_now"]) + ".")
    return "\n".join(lines)
