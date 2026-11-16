"""One XANES calculation: inputs, FDMNES, normalized spectrum CSVs and result.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import engine, records, spectrum
from .errors import FdmnesError
from .inputs import XanesSpec, read_structure, write_inputs

SCHEMA = "fdmnes-mcp/run@1"


async def run_xanes(
    structure: str,
    spec: XanesSpec,
    run_name: str | None,
    output_dir: str | None,
    timeout_s: float | None,
    report: engine.Report,
) -> dict[str, Any]:
    path = records.existing_file(structure, "structure file")
    atoms = read_structure(path)
    command = engine.require_executable()
    label = spec.absorber if not isinstance(spec.absorber, list) else "sites"
    run_dir = records.new_run_dir(
        output_dir, run_name or f"{records.slug(path.stem)}-{label}-{spec.edge}"
    )
    stem = "xanes"
    described = write_inputs(atoms, spec, run_dir, stem)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "run_dir": str(run_dir),
        "structure": {"path": str(path), "sha256": records.sha256(path), **described},
        "settings": spec.as_dict(),
        "fdmnes": {"command": command},
        "notes": [],
    }
    await report(0, None, f"FDMNES started in {run_dir.name}")
    try:
        proc = await engine.run(command, run_dir, timeout_s, report)
        result["fdmnes"].update(proc)
        result["log"] = engine.read_log(run_dir, stem)
        files = engine.conv_files(run_dir)
        if proc["returncode"] != 0 or not files:
            why = (
                f"exited with code {proc['returncode']}"
                if proc["returncode"] != 0
                else "finished without writing a convoluted spectrum (*_conv.txt)"
            )
            raise FdmnesError(
                "fdmnes_failed",
                f"FDMNES {why}. Last output:\n"
                f"{engine.tail(Path(proc['stdout']))}\n{engine.tail(Path(proc['stderr']))}\n"
                f"The run folder {run_dir} keeps the inputs and logs.",
                run_dir=str(run_dir),
            )
        result["spectra"] = [_spectrum(f, run_dir, result["log"]) for f in files]
        if result["log"].get("scf_not_converged"):
            result["notes"].append(
                "FDMNES reports the SCF did not converge (see xanes_bav.txt); the "
                "spectrum uses the last potential. Check it before relying on fine features."
            )
        result["status"] = "completed"
    except FdmnesError as exc:
        result["error"] = {"reason": exc.reason, "message": str(exc)}
        records.write_json(run_dir / "result.json", result)
        raise
    records.write_json(run_dir / "result.json", result)
    result["results_file"] = str(run_dir / "result.json")
    return result


def _spectrum(conv: Path, run_dir: Path, log: dict[str, Any]) -> dict[str, Any]:
    energy, mu, _ = spectrum.read_two_columns(conv)
    norm = spectrum.normalize(energy, mu)
    name = conv.name.removesuffix("_conv.txt").removeprefix("xanes")
    csv_path = run_dir / f"spectrum{name}.csv"
    columns: dict[str, Any] = {"energy_eV": energy, "mu": mu}
    edge = log.get("edge_energy_eV")
    if edge is not None:
        columns["photon_energy_eV"] = energy + edge
    if norm["normalized"]:
        columns["mu_norm"] = norm["mu_norm"]
    spectrum.write_csv(csv_path, columns)
    return {
        "conv_file": str(conv),
        "csv": str(csv_path),
        "energy_reference": "relative to the Fermi level (FDMNES default)",
        **spectrum.spectrum_summary(energy, mu, norm),
    }


def read(path: str, energy_shift: float, output_dir: str | None) -> dict[str, Any]:
    src = records.existing_file(path, "spectrum file")
    energy, mu, header = spectrum.read_two_columns(src)
    energy = energy + energy_shift
    norm = spectrum.normalize(energy, mu)
    folder = records.output_root(output_dir) / "spectra"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{records.slug(src.stem)}.normalized.csv"
    n = 2
    while out.exists():
        out = folder / f"{records.slug(src.stem)}.normalized-{n}.csv"
        n += 1
    columns: dict[str, Any] = {"energy_eV": energy, "mu": mu}
    if norm["normalized"]:
        columns["mu_norm"] = norm["mu_norm"]
    spectrum.write_csv(out, columns)
    return {
        "source": str(src),
        "header": header,
        "energy_shift_eV": energy_shift,
        "csv": str(out),
        **spectrum.spectrum_summary(energy, mu, norm),
    }


def compare(
    paths: list[str], labels: list[str] | None, align: bool, output_dir: str | None
) -> dict[str, Any]:
    if len(paths) < 2:
        raise FdmnesError("invalid_input", "Give at least two spectra to compare.")
    if labels and len(labels) != len(paths):
        raise FdmnesError("invalid_input", "labels: one per spectrum.")
    loaded = []
    for i, p in enumerate(paths):
        src = records.existing_file(p, "spectrum file")
        e, mu = _load_for_compare(src)
        loaded.append(((labels[i] if labels else src.stem), e, mu))
    if len({lab for lab, _, _ in loaded}) != len(loaded):
        raise FdmnesError("invalid_input", "Two spectra have the same label; pass labels.")
    summary, columns = spectrum.compare(loaded, align)
    folder = records.output_root(output_dir) / "spectra"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "comparison.csv"
    n = 2
    while out.exists():
        out = folder / f"comparison-{n}.csv"
        n += 1
    spectrum.write_csv(out, columns)
    return {"csv": str(out), "align": align, **summary}


def _load_for_compare(src: Path) -> tuple[Any, Any]:
    """A spectrum file; a CSV written by this server gives its raw ``mu`` column."""
    if src.suffix.lower() == ".csv":
        import csv

        with src.open() as fh:
            rows = list(csv.DictReader(fh))
        if rows and "energy_eV" in rows[0] and "mu" in rows[0]:
            import numpy as np

            e = np.array([float(r["energy_eV"]) for r in rows])
            mu = np.array([float(r["mu"]) for r in rows])
            return e, mu
    e, mu, _ = spectrum.read_two_columns(src)
    return e, mu
