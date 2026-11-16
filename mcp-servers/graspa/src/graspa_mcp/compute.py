"""Prepare, run, read and rank gRASPA adsorption runs. Each run folder keeps
``run.json`` (what was asked) and, once read, ``result.json`` (what came out)."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import engine, parse, records
from .errors import GraspaError
from .inputs import GcmcSpec, read_framework, write_run

SCHEMA = "graspa-mcp/run@1"


def prepare(
    structure: str, spec: GcmcSpec, run_name: str | None, output_dir: str | None
) -> dict[str, Any]:
    source = records.existing_file(structure, "CIF file")
    atoms = read_framework(source)
    name = run_name or (
        f"{records.slug(source.stem)}-{spec.adsorbate}-{spec.temperature_K:g}K-"
        f"{spec.pressure_Pa:g}Pa"
    )
    run_dir = records.new_run_dir(output_dir, name)
    described = write_run(source, atoms, spec, run_dir)
    run = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "structure": {"path": str(source), "sha256": records.sha256(source), **described},
        "settings": spec.as_dict(),
        "notes": described.pop("notes"),
    }
    records.write_json(run_dir / "run.json", run)
    return run


async def run(
    structure: str,
    spec: GcmcSpec,
    run_name: str | None,
    output_dir: str | None,
    timeout_s: float | None,
    report: engine.Report,
) -> dict[str, Any]:
    command = engine.require_executable()
    prepared = prepare(structure, spec, run_name, output_dir)
    run_dir = Path(prepared["run_dir"])
    await report(0, None, f"gRASPA started in {run_dir.name}")
    proc = await engine.run(command, run_dir, timeout_s, report)
    prepared["graspa"] = {"command": command, **proc}
    records.write_json(run_dir / "run.json", prepared)
    if proc["returncode"] != 0:
        failure = {
            **prepared,
            "status": "failed",
            "error": {"reason": "graspa_failed", "returncode": proc["returncode"]},
        }
        records.write_json(run_dir / "result.json", failure)
        raise GraspaError(
            "graspa_failed",
            f"gRASPA exited with code {proc['returncode']}. Last output:\n"
            f"{engine.tail(Path(proc['stdout']))}\n{engine.tail(Path(proc['stderr']))}\n"
            f"{run_dir} keeps the inputs and logs.",
            run_dir=str(run_dir),
        )
    return read(str(run_dir))


def read(path: str) -> dict[str, Any]:
    """Read a finished run folder (or its run.json) and write result.json."""
    p = Path(path).expanduser()
    if p.is_file() and p.name in ("run.json", "result.json"):
        p = p.parent
    if not p.is_dir() and not p.is_absolute():
        p = records.output_root(None) / path
    run_file = p / "run.json"
    if not run_file.is_file():
        raise GraspaError(
            "not_found",
            f"{path!r} is not a gRASPA run folder made by prepare_graspa or run_graspa "
            "(it has no run.json).",
        )
    run_info = json.loads(run_file.read_text())
    out = p / "graspa_stdout.txt"
    if not out.is_file():
        candidates = sorted(p.glob("*.out")) + sorted(p.glob("output*.txt"))
        if not candidates:
            raise GraspaError(
                "unreadable_output",
                f"{p} has no gRASPA output yet (graspa_stdout.txt). Run gRASPA in the folder "
                "with its standard output saved to graspa_stdout.txt.",
            )
        out = candidates[0]
    st, cfg = run_info["structure"], run_info["settings"]
    loading = parse.read_loading(
        out.read_text(errors="replace"),
        cfg["adsorbate"],
        st["unit_cells"],
        st["framework_mass_g_mol"],
    )
    result = {
        **run_info,
        "status": "completed",
        "output_file": str(out),
        "loading": loading,
        "condition": {"temperature_K": cfg["temperature_K"], "pressure_Pa": cfg["pressure_Pa"]},
    }
    if loading["flavour"] == "graspa-sycl":
        result["notes"] = [*result["notes"], "gRASPA-SYCL prints no error bar for the loading."]
    records.write_json(p / "result.json", result)
    result["results_file"] = str(p / "result.json")
    return result


def _gather(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            direct = p / "result.json"
            found += [direct] if direct.is_file() else sorted(p.rglob("result.json"))
        elif p.is_file():
            found.append(p)
        else:
            raise GraspaError("not_found", f"{raw!r} is neither a result.json nor a folder.")
    if not found:
        raise GraspaError("not_found", "No result.json found under the given paths.")
    return found


def rank(
    paths: list[str],
    adsorption: dict[str, float],
    desorption: dict[str, float] | None,
    top_fraction: float,
    output_dir: str | None,
) -> dict[str, Any]:
    """Rank frameworks by uptake at one condition, or by working capacity between two.

    Adapted from ChemGraph's ``rank_records`` (``tools/graspa_analysis.py``): repeats at a
    condition are averaged; a framework missing a requested condition, or with a failed
    run at it, is excluded with the reason, never ranked on partial data."""
    if not 0 < top_fraction <= 1:
        raise GraspaError("invalid_input", "top_fraction: greater than 0, at most 1.")

    def key(c: dict[str, float]) -> tuple[float, float]:
        try:
            return (float(c["temperature_K"]), float(c["pressure_Pa"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise GraspaError(
                "invalid_input", "A condition is {temperature_K, pressure_Pa}."
            ) from exc

    ads, des = key(adsorption), key(desorption) if desorption else None
    if des == ads:
        raise GraspaError("invalid_input", "Adsorption and desorption conditions must differ.")
    groups: dict[str, dict[tuple[float, float], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    adsorbates: set[str] = set()
    for f in _gather(paths):
        r = json.loads(f.read_text())
        if r.get("schema") != SCHEMA:
            continue
        adsorbates.add(r["settings"]["adsorbate"])
        cond = (r["settings"]["temperature_K"], r["settings"]["pressure_Pa"])
        groups[r["structure"]["path"]][cond].append(r)
    if len(adsorbates) > 1:
        raise GraspaError(
            "invalid_input",
            f"The results mix adsorbates ({', '.join(sorted(adsorbates))}); rank one at a time.",
        )
    ranked, excluded = [], []
    for source, by_cond in groups.items():
        needed = [ads] + ([des] if des else [])
        missing = [c for c in needed if not by_cond.get(c)]
        failed = [c for c in needed if any(r.get("status") != "completed" for r in by_cond[c])]
        if missing or failed:
            why = []
            if missing:
                why.append("no run at " + ", ".join(f"{t:g} K/{p:g} Pa" for t, p in missing))
            if failed:
                why.append("a failed run at " + ", ".join(f"{t:g} K/{p:g} Pa" for t, p in failed))
            excluded.append({"structure": source, "reason": "; ".join(why)})
            continue

        def mean(c: tuple[float, float], rows: dict = by_cond) -> float:
            vals = [r["loading"]["uptake_mol_kg"] for r in rows[c]]
            return math.fsum(vals) / len(vals)

        row: dict[str, Any] = {"structure": source, "framework": Path(source).stem}
        row["repeats"] = len(by_cond[ads])
        if des:
            row.update(uptake_ads=mean(ads), uptake_des=mean(des))
            row["working_capacity"] = row["uptake_ads"] - row["uptake_des"]
        else:
            row["uptake"] = mean(ads)
        ranked.append(row)
    metric = "working_capacity" if des else "uptake"
    ranked.sort(key=lambda r: (-r[metric], r["structure"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    selected = ranked[: math.ceil(top_fraction * len(ranked))] if ranked else []
    folder = records.output_root(output_dir) / "rankings"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "ranking.csv"
    n = 2
    while out.exists():
        out = folder / f"ranking-{n}.csv"
        n += 1
    cols = [
        "rank",
        "framework",
        *([k for k in ("uptake_ads", "uptake_des", "working_capacity")] if des else ["uptake"]),
        "repeats",
        "structure",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ranked)
    return {
        "adsorbate": next(iter(adsorbates), None),
        "metric": metric,
        "units": "mol/kg",
        "adsorption": {"temperature_K": ads[0], "pressure_Pa": ads[1]},
        "desorption": {"temperature_K": des[0], "pressure_Pa": des[1]} if des else None,
        "ranked": len(ranked),
        "excluded": excluded,
        "top": selected,
        "csv": str(out),
    }
