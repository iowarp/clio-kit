"""Calculators: what each needs, whether it is ready here, and building it in a worker.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), tools/ase_core.py
(``load_calculator``) and schemas/calculators/*. EMT runs in the server's own
environment; xTB (tblite), MACE (mace-torch) and UMA (fairchem-core) run in a worker
environment built from the matching extra on first use (:mod:`.envs`); NWChem and ORCA
are external programs found through environment variables or PATH. A calculator that
cannot run says why and how to fix it with a typed error, before any environment is built.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .errors import AseError

EXTRA = {"emt": None, "nwchem": None, "orca": None, "xtb": "xtb", "mace": "mace", "uma": "uma"}

NWCHEM_ENV = ("ASE_NWCHEM_COMMAND", "NWCHEM_COMMAND")
ORCA_ENV = ("ORCA_COMMAND", "ASE_ORCA_COMMAND")
HF_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

DESCRIPTION = {
    "emt": "Effective medium theory (ASE built-in). Instant; parameters only for Al, Cu, Ag, "
    "Au, Ni, Pd, Pt and (for tests) H, C, N, O. Not a molecular method: use it to "
    "test a workflow, not for chemistry.",
    "xtb": "GFN2-xTB / GFN1-xTB tight binding (tblite). Fast semi-empirical quantum "
    "chemistry for molecules up to thousands of atoms; models charge and spin "
    "(unpaired electrons = multiplicity − 1) and gives dipoles (IR).",
    "mace": "MACE machine-learned potential. Default MACE-OFF23 small (organic molecules: H, "
    "C, N, O, F, P, S, Cl, Br, I; trained on ωB97M-D3BJ; ASL academic licence). "
    "MACE-OFF/MP do not model charge or spin and give no dipole (no IR).",
    "uma": "Meta FAIR UMA universal model (fairchem-core), task omol for molecules; models "
    "charge and spin. Weights are gated on Hugging Face.",
    "nwchem": "NWChem DFT/HF (external program, run by ASE). Default B3LYP/6-31G*.",
    "orca": "ORCA quantum chemistry (external program, run by ASE). Default B3LYP/def2-SVP.",
}


def models_spin(resolved: dict[str, Any]) -> bool:
    kind = resolved["type"]
    if kind == "mace":
        return str(resolved.get("model", "")).startswith("mace-omol")
    return kind in {"xtb", "uma", "nwchem", "orca"}


def models_charge(resolved: dict[str, Any]) -> bool:
    return models_spin(resolved)


def _env_value(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return name, value
    return None, None


def nwchem_command() -> dict[str, Any]:
    """How ASE will start NWChem, or why it cannot."""
    name, value = _env_value(NWCHEM_ENV)
    if value:
        return {"found": True, "source": name, "command": value}
    path = shutil.which("nwchem")
    if path:
        return {"found": True, "source": "PATH", "command": path}
    return {"found": False}


def orca_command() -> dict[str, Any]:
    name, value = _env_value(ORCA_ENV)
    if value:
        return {"found": True, "source": name, "command": value}
    path = shutil.which("orca")
    if path:
        return {"found": True, "source": "PATH", "command": path}
    return {"found": False}


def hf_token() -> dict[str, Any]:
    name, value = _env_value(HF_ENV)
    if value:
        return {"found": True, "source": name}
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if (home / "token").is_file():
        return {"found": True, "source": str(home / "token")}
    return {"found": False}


def gate(resolved: dict[str, Any], driver: str = "energy") -> None:
    """Raise the typed error that says how to make this calculator runnable here."""
    kind = resolved["type"]
    if driver == "ir" and kind in ("emt", "mace"):
        raise AseError(
            "dipole_unavailable",
            f"{kind} gives no dipole moment, so it cannot compute IR intensities. Use xtb, "
            "nwchem or orca for driver 'ir', or driver 'vib' for frequencies only.",
        )
    if kind == "nwchem" and not nwchem_command()["found"]:
        raise AseError(
            "binary_missing",
            "NWChem was not found: set ASE_NWCHEM_COMMAND to the full command (e.g. "
            "'mpirun -np 4 /path/to/nwchem PREFIX.nwi > PREFIX.nwo') or NWCHEM_COMMAND to "
            "the nwchem executable, or put nwchem on PATH. The nwchem plugin's install "
            "skill installs it with Spack.",
            env=list(NWCHEM_ENV),
        )
    if kind == "orca" and not orca_command()["found"]:
        raise AseError(
            "binary_missing",
            "ORCA was not found: download ORCA from the ORCA forum (free registration, "
            "academic licence), unpack it and set ORCA_COMMAND (or ASE_ORCA_COMMAND) to the "
            "full path of the orca executable, or put orca on PATH.",
            env=list(ORCA_ENV),
        )
    if kind == "uma" and not hf_token()["found"]:
        raise AseError(
            "hf_token_missing",
            "UMA weights are gated on Hugging Face: accept the licence of facebook/UMA with "
            "a Hugging Face account, create an access token and set HF_TOKEN to it (or run "
            "'hf auth login').",
            env=list(HF_ENV),
        )


# ---------------------------------------------------------------------------
# Worker side: build the ASE calculator (imports the calculator's packages)
# ---------------------------------------------------------------------------


def build(
    resolved: dict[str, Any], charge: int, multiplicity: int, workdir: Path
) -> tuple[Any, dict[str, Any], list[str], dict[str, Any]]:
    """Return ``(calculator, atoms_info, notes, versions)`` for a resolved spec."""
    kind = resolved["type"]
    notes: list[str] = []
    info: dict[str, Any] = {}
    versions: dict[str, Any] = {}
    if multiplicity != 1 and not models_spin(resolved):
        notes.append(
            f"{kind.upper()} does not model electronic spin: multiplicity {multiplicity} "
            "is used only for the electronic entropy in thermochemistry, not by the energy."
        )
    if charge != 0 and not models_charge(resolved):
        notes.append(
            f"{kind.upper()} does not model the total charge: charge {charge} is recorded "
            "but does not change the energy."
        )

    if kind == "emt":
        from ase.calculators.emt import EMT

        return EMT(asap_cutoff=bool(resolved["asap_cutoff"])), info, notes, versions

    if kind == "xtb":
        from importlib.metadata import version

        from tblite.ase import TBLite

        solvation = resolved.get("solvation")
        solv: tuple[Any, ...] | None = None
        if solvation:
            model, _, solvent = str(solvation).partition(":")
            if not solvent:
                raise AseError(
                    "invalid_input",
                    f"xTB solvation is 'model:solvent', e.g. 'alpb:water', got {solvation!r}.",
                )
            solv = (model.lower(), solvent)
        versions["tblite"] = version("tblite")
        calc = TBLite(
            method=resolved["method"],
            charge=float(charge),
            multiplicity=int(multiplicity),
            accuracy=float(resolved["accuracy"]),
            electronic_temperature=float(resolved["electronic_temperature"]),
            max_iterations=int(resolved["max_iterations"]),
            solvation=solv,
            verbosity=0,
        )
        return calc, info, notes, versions

    if kind == "mace":
        return _mace(resolved, charge, multiplicity, info, notes, versions)

    if kind == "uma":
        from importlib.metadata import version

        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        versions["fairchem-core"] = version("fairchem-core")
        try:
            unit = pretrained_mlip.get_predict_unit(
                resolved["model"],
                inference_settings=resolved["inference_settings"],
                device="cpu",
            )
        except Exception as exc:
            if type(exc).__name__ not in ("GatedRepoError", "RepositoryNotFoundError") and getattr(
                getattr(exc, "response", None), "status_code", None
            ) not in (401, 403):
                raise
            raise AseError(
                "hf_token_missing",
                f"Hugging Face refused the UMA weights ({type(exc).__name__}): the token in "
                f"{hf_token().get('source')} is invalid or its account has not accepted the "
                "facebook/UMA licence. Accept it on huggingface.co/facebook/UMA and set "
                "HF_TOKEN to a token of that account.",
            ) from exc
        info.update({"charge": int(charge), "spin": int(multiplicity)})
        return FAIRChemCalculator(unit, task_name=resolved["task"]), info, notes, versions

    if kind == "nwchem":
        from ase.calculators.nwchem import NWChem

        found = nwchem_command()
        command = found["command"]
        if found["source"] != "ASE_NWCHEM_COMMAND" and "PREFIX" not in command:
            command = f"{command} PREFIX.nwi > PREFIX.nwo"
        if int(resolved.get("nprocs") or 1) > 1 and not command.lstrip().startswith("mpi"):
            command = f"mpirun -np {int(resolved['nprocs'])} {command}"
        return (
            NWChem(**nwchem_kwargs(resolved, charge, multiplicity, workdir, command)),
            info,
            notes,
            versions,
        )

    if kind == "orca":
        from ase.calculators.orca import ORCA, OrcaProfile

        found = orca_command()
        blocks = str(resolved.get("blocks") or "")
        if int(resolved.get("nprocs") or 1) > 1:
            blocks = f"%pal nprocs {int(resolved['nprocs'])} end\n{blocks}"
        return (
            ORCA(
                profile=OrcaProfile(command=found["command"]),
                directory=str(workdir),
                charge=int(charge),
                mult=int(multiplicity),
                orcasimpleinput=f"{resolved['method']} {resolved['basis']} EnGrad",
                orcablocks=blocks,
            ),
            info,
            notes,
            versions,
        )

    raise AseError("invalid_input", f"Unknown calculator type {kind!r}.")


def nwchem_kwargs(
    resolved: dict[str, Any], charge: int, multiplicity: int, workdir: Path, command: str
) -> dict[str, Any]:
    """The keyword arguments of ASE's NWChem calculator (input-file writing, testable)."""
    theory = resolved.get("theory") or "dft"
    kwargs: dict[str, Any] = {
        "label": str(workdir / "nwchem"),
        "command": command,
        "theory": theory,
        "basis": resolved["basis"],
        "charge": int(charge),
    }
    blocks = dict(resolved.get("blocks") or {})
    if theory in ("dft", "tddft"):
        dft = {"xc": resolved["method"], "mult": int(multiplicity)}
        dft.update(blocks.pop("dft", {}) or {})
        kwargs["dft"] = dft
    else:  # HF-based theories (scf, mp2, ccsd, ...) take open shells in the scf block
        scf: dict[str, Any] = {"nopen": int(multiplicity) - 1}
        if multiplicity > 1:
            scf["uhf"] = None
        scf.update(blocks.pop("scf", {}) or {})
        kwargs["scf"] = scf
    kwargs.update(blocks)
    return kwargs


def _mace(
    resolved: dict[str, Any],
    charge: int,
    multiplicity: int,
    info: dict[str, Any],
    notes: list[str],
    versions: dict[str, Any],
) -> tuple[Any, ...]:
    from importlib.metadata import version

    model = str(resolved["model"])
    if resolved.get("dispersion") and not model.startswith("mace-mp-"):
        raise AseError("invalid_input", "dispersion applies only to MACE-MP models.")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    from mace.calculators import mace_mp, mace_off, mace_omol
    from mace.calculators.mace import MACECalculator

    versions["mace-torch"] = version("mace-torch")
    versions["torch"] = version("torch")
    dtype = str(resolved.get("dtype") or "float64")
    if model.startswith("mace-off-"):
        calc = mace_off(model=model.removeprefix("mace-off-"), device="cpu", default_dtype=dtype)
    elif model.startswith("mace-mp-"):
        calc = mace_mp(
            model=model.removeprefix("mace-mp-"),
            device="cpu",
            default_dtype=dtype,
            dispersion=bool(resolved.get("dispersion")),
        )
    elif model.startswith("mace-omol-"):
        calc = mace_omol(
            model=model.removeprefix("mace-omol-").replace("-", "_"),
            device="cpu",
            default_dtype=dtype,
        )
        info.update({"charge": int(charge), "spin": int(multiplicity)})
    elif Path(model).expanduser().is_file():
        calc = MACECalculator(
            model_paths=str(Path(model).expanduser()), device="cpu", default_dtype=dtype
        )
    else:
        raise AseError(
            "invalid_input",
            f"Unknown MACE model {model!r}: use mace-off-small/medium/large, "
            "mace-mp-small/medium/medium-mpa-0, mace-omol-extra-large, or a .model path.",
        )
    return calc, info, notes, versions
