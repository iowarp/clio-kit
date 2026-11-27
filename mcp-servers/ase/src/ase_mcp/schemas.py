"""Typed inputs of a calculation: the calculator and its settings, constraints, the run.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), schemas/ase_input.py and
schemas/calculators/{emt,tblite,mace,fairchem,nwchem,orca}_calc.py. The calculator is
one object ``{type, model?, method?, basis?, ...settings}``; settings a calculator does
not know are refused rather than ignored. Charge and multiplicity are run arguments,
passed to every calculator that models them and to the thermochemistry.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import AseError

CalculatorType = Literal["emt", "xtb", "mace", "uma", "nwchem", "orca"]
Driver = Literal["energy", "opt", "vib", "thermo", "ir"]
Optimizer = Literal["bfgs", "lbfgs", "fire", "gpmin", "mdmin"]

# Settings each calculator accepts besides type/model/method/basis, with defaults.
SETTINGS: dict[str, dict[str, Any]] = {
    "emt": {"asap_cutoff": False},
    "xtb": {
        "accuracy": 1.0,
        "electronic_temperature": 300.0,
        "max_iterations": 250,
        "solvation": None,  # e.g. "alpb:water", "gbsa:water", "cpcm:water"
    },
    "mace": {"dtype": "float64", "dispersion": False},
    "uma": {"task": "omol", "inference_settings": "default"},
    "nwchem": {"theory": "dft", "nprocs": 1, "blocks": None},
    "orca": {"nprocs": 1, "blocks": ""},
}

DEFAULT_MODEL = {"mace": "mace-off-small", "uma": "uma-s-1p1"}
DEFAULT_METHOD = {"xtb": "GFN2-xTB", "nwchem": "B3LYP", "orca": "B3LYP"}
DEFAULT_BASIS = {"nwchem": "6-31G*", "orca": "def2-SVP"}

XTB_METHODS = ("GFN2-xTB", "GFN1-xTB", "IPEA1-xTB")
MACE_FAMILIES = ("mace-off-", "mace-mp-", "mace-omol-")

# Which calculators model the electronic spin state and the total charge.
MODELS_SPIN = {"emt": False, "xtb": True, "mace": False, "uma": True, "nwchem": True, "orca": True}
MODELS_CHARGE = {
    "emt": False,
    "xtb": True,
    "mace": False,
    "uma": True,
    "nwchem": True,
    "orca": True,
}


class CalculatorSpec(BaseModel):
    """A calculator: ``type`` plus optional ``model``, ``method``, ``basis`` and settings."""

    model_config = ConfigDict(extra="allow")

    type: CalculatorType = Field(description="emt, xtb, mace, uma, nwchem or orca.")
    model: str | None = Field(
        default=None,
        description=(
            "MACE: mace-off-small (default), mace-off-medium, mace-off-large, mace-mp-small, "
            "mace-mp-medium, mace-mp-medium-mpa-0, mace-omol-extra-large, or a path to a "
            ".model file. UMA: uma-s-1p1 (default) or uma-m-1p1."
        ),
    )
    method: str | None = Field(
        default=None,
        description=(
            "xTB: GFN2-xTB (default), GFN1-xTB, IPEA1-xTB. NWChem: the DFT functional "
            "(default B3LYP). ORCA: the method keywords (default B3LYP)."
        ),
    )
    basis: str | None = Field(
        default=None, description="NWChem (default 6-31G*) or ORCA (default def2-SVP) basis."
    )

    @model_validator(mode="after")
    def _check(self) -> CalculatorSpec:
        extra = set((self.model_extra or {}).keys())
        if "settings" in extra and isinstance(self.model_extra, dict):
            nested = self.model_extra.pop("settings") or {}
            if not isinstance(nested, dict):
                raise ValueError("settings must be an object of calculator settings")
            self.model_extra.update(nested)
            extra = set(self.model_extra.keys())
        known = set(SETTINGS[self.type])
        unknown = extra - known
        if unknown:
            raise ValueError(
                f"{self.type} does not take {sorted(unknown)}; its settings are "
                f"{sorted(known) or 'none'} besides model/method/basis"
            )
        if self.type == "xtb" and self.method and self.method not in XTB_METHODS:
            raise ValueError(f"xTB method must be one of {XTB_METHODS}, got {self.method!r}")
        for field, allowed in (
            ("model", DEFAULT_MODEL),
            ("method", DEFAULT_METHOD),
            ("basis", DEFAULT_BASIS),
        ):
            if getattr(self, field) is not None and self.type not in allowed:
                raise ValueError(f"{self.type} takes no {field} (only {', '.join(allowed)} do)")
        return self

    def resolved(self) -> dict[str, Any]:
        """Every setting with its default filled in: what the run records."""
        out: dict[str, Any] = {"type": self.type}
        if self.type in DEFAULT_MODEL:
            out["model"] = self.model or DEFAULT_MODEL[self.type]
        if self.type in DEFAULT_METHOD:
            out["method"] = self.method or DEFAULT_METHOD[self.type]
        if self.type in DEFAULT_BASIS:
            out["basis"] = self.basis or DEFAULT_BASIS[self.type]
        settings = dict(SETTINGS[self.type])
        settings.update(self.model_extra or {})
        out.update(settings)
        return out


class Constraint(BaseModel):
    """fix_atoms {indices}, bond {indices: [a, b], value_A?} or dihedral
    {indices: [a, b, c, d], value_deg?}. Without a value the current one is held."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["fix_atoms", "bond", "angle", "dihedral"]
    indices: list[int] = Field(min_length=1)
    value_deg: float | None = None
    value_A: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Constraint:
        need = {"bond": 2, "angle": 3, "dihedral": 4}.get(self.type)
        if need and len(self.indices) != need:
            raise ValueError(f"a {self.type} constraint needs {need} indices")
        if self.type == "bond" and self.value_deg is not None:
            raise ValueError("a bond constraint takes value_A, not value_deg")
        if self.type in ("angle", "dihedral") and self.value_A is not None:
            raise ValueError(f"a {self.type} constraint takes value_deg, not value_A")
        if self.type == "fix_atoms" and (self.value_A is not None or self.value_deg is not None):
            raise ValueError("fix_atoms takes only indices")
        return self


class RunInput(BaseModel):
    """Everything a worker needs to run one calculation."""

    model_config = ConfigDict(extra="forbid")

    structure: str
    driver: Driver = "energy"
    calculator: CalculatorSpec
    charge: int = 0
    multiplicity: int = Field(default=1, ge=1)
    optimizer: Optimizer = "bfgs"
    fmax: float = Field(default=0.05, gt=0)
    steps: int = Field(default=200, ge=0)
    optimize: bool = True
    temperature: float = Field(default=298.15, gt=0)
    pressure: float = Field(default=101325.0, gt=0)
    constraints: list[Constraint] = Field(default_factory=list)
    run_name: str = "run"
    run_dir: str = ""


def parse_calculator(value: Any) -> CalculatorSpec:
    """Validate a calculator object, raising a typed ``invalid_input`` error."""
    from pydantic import ValidationError

    if isinstance(value, str):
        value = {"type": value}
    try:
        return CalculatorSpec.model_validate(value)
    except ValidationError as exc:
        raise AseError(
            "invalid_input",
            "The calculator is not valid: "
            + "; ".join(e["msg"] for e in exc.errors())
            + ". Example: {type: 'xtb', method: 'GFN2-xTB'} or {type: 'mace'}.",
        ) from exc


def parse_constraints(values: list[Any] | None) -> list[Constraint]:
    from pydantic import ValidationError

    out = []
    for v in values or []:
        try:
            out.append(Constraint.model_validate(v))
        except ValidationError as exc:
            raise AseError(
                "invalid_input",
                f"The constraint {v} is not valid: "
                + "; ".join(e["msg"] for e in exc.errors())
                + ". Use {type: 'fix_atoms', indices: [...]}, {type: 'bond', indices: "
                "[a, b], value_A?}, {type: 'angle', indices: [a, b, c], value_deg?} or "
                "{type: 'dihedral', indices: [a, b, c, d], value_deg?}.",
            ) from exc
    return out
