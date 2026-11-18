"""Lean words for the agent: the condition, the uptake with its error bar, the files."""

from __future__ import annotations

from typing import Any


def _cond(cfg: dict[str, Any]) -> str:
    return f"{cfg['adsorbate']} at {cfg['temperature_K']:g} K and {cfg['pressure_Pa']:g} Pa"


def prepare_text(r: dict[str, Any]) -> str:
    st, cfg = r["structure"], r["settings"]
    cells = "×".join(map(str, st["unit_cells"]))
    lines = [
        (
            f"Prepared a gRASPA GCMC run of {_cond(cfg)} in {st['framework']} ({st['formula']}, "
            f"{st['n_atoms']} atoms per cell, {cells} unit cells for a {cfg['cutoff_A']:g} Å "
            f"cutoff; {cfg['init_cycles']} initialization + {cfg['prod_cycles']} production "
            f"cycles) in {r['run_dir']}."
        ),
    ]
    lines += [f"Note: {n}" for n in r["notes"]]
    if "graspa" not in r:
        lines.append(
            "Run gRASPA in that folder on a GPU node with its output saved to "
            "graspa_stdout.txt (e.g. nvc_main.x > graspa_stdout.txt), then read_graspa it."
        )
    return "\n".join(lines)


def result_text(r: dict[str, Any]) -> str:
    st, cfg, ld = r["structure"], r["settings"], r["loading"]
    err = f" ± {ld['uptake_error_mol_kg']:.3g}" if ld.get("uptake_error_mol_kg") is not None else ""
    line = f"{st['framework']}: {_cond(cfg)}, uptake {ld['uptake_mol_kg']:.4g}{err} mol/kg"
    if "molecules_per_unit_cell" in ld:
        line += f" ({ld['molecules_per_unit_cell']:.3g} molecules per unit cell)"
    if "graspa" in r:
        line += f"; {r['graspa']['wall_time_s']:g} s"
    lines = [line + f". Read from {ld['flavour']} output."]
    lines += [f"Note: {n}" for n in r["notes"]]
    lines.append(f"Result: {r['results_file']}.")
    return "\n".join(lines)


def rank_text(d: dict[str, Any]) -> str:
    def cond(c: dict[str, float]) -> str:
        return f"{c['temperature_K']:g} K/{c['pressure_Pa']:g} Pa"

    what = (
        f"working capacity {cond(d['adsorption'])} → {cond(d['desorption'])}"
        if d["desorption"]
        else f"uptake at {cond(d['adsorption'])}"
    )
    lines = [f"Ranked {d['ranked']} frameworks for {d['adsorbate']} by {what} (mol/kg):"]
    for r in d["top"][:10]:
        lines.append(f"{r['rank']}. {r['framework']}: {r[d['metric']]:.4g}")
    if len(d["top"]) > 10:
        lines.append(f"… {len(d['top']) - 10} more selected in the CSV.")
    for e in d["excluded"]:
        lines.append(f"Excluded {e['structure']}: {e['reason']}.")
    lines.append(f"Ranking: {d['csv']}.")
    return "\n".join(lines)


def status_text(d: dict[str, Any]) -> str:
    ex = d["executable"]
    if ex["found"]:
        return f"gRASPA is ready: {ex['command']} (found via {ex['source']})."
    return f"gRASPA is not available here: {ex['problem']}. {d['install']}"
