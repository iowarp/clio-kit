"""The rules a skill must satisfy to earn its place in the marketplace.

A skill is not like an MCP server. Tool definitions and skill descriptions consume client context according
to the client's loading policy. Skill bodies load on invocation, so concise,
distinct descriptions help discovery without loading every procedure. That asymmetry is why these rules exist and why they are checked
mechanically rather than left to a reviewer's attention.

The checks split by whether a machine can settle them:

``problems``
    Objective and blocking. Frontmatter that does not parse, a name that
    disagrees with its folder, absent eval scenarios, a description that does
    not say when to fire. Each has exactly one correct answer.

``advisories``
    Real but a judgement call. Whether a skill declares boundaries against the
    skills it could be confused with, and how long its discovery description is.
    Reported so a reviewer sees them, never used to reject a contribution,
    because a first skill with nothing to collide against is legitimately
    boundary-free.

Lives beside ``community.py`` rather than inside the generator so the same
rules back manifest generation, ``clio-kit plugin validate``, and a contributor
checking their own directory before opening anything.
"""

from __future__ import annotations

import yaml

from dataclasses import dataclass, field
from pathlib import Path

# Keep discovery descriptions concise; put procedural detail in the body.
DESCRIPTION_BUDGET = 500

# What a description has to establish before it can be trusted to fire at the
# right moment. "Use when" forces the author to name the situation rather than
# restate the body; "Triggers on" forces the literal phrases a user actually
# types, which is what the matcher has to work with.
REQUIRED_OPENING = "Use when"
REQUIRED_TRIGGER_CLAUSE = "Triggers on"

# A boundary tells one skill to stand down so another can fire. Without it,
# skills covering neighbouring ground hijack each other and the user cannot
# tell why the wrong one answered.
BOUNDARY_MARKER = "Not for"

# Recorded scenarios are what separate a skill that was tested from a skill
# that was merely written. Either shape is accepted: one file, or a directory
# holding several.
EVAL_FILENAMES = ("evals.md", "EVALS.md")
EVAL_DIRNAME = "evals"

# How far a skill has actually been checked, weakest first. The field is what a
# reader trusts when deciding whether to rely on a skill, so an unrecognised
# value has to be refused rather than read as "some kind of tested".
EVAL_LADDER = (
    "untested",
    "scenarios-recorded",
    "trigger-checked",
    "smoke-checked",
    "eval-run",
)


class SkillProblem(Exception):
    """A skill directory could not be read well enough to check."""


@dataclass
class SkillReport:
    """Everything the rules found in one skill directory."""

    name: str
    problems: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    description_chars: int = 0

    @property
    def ok(self) -> bool:
        """Whether this skill may publish. Advisories never block."""
        return not self.problems


def read_skill_frontmatter(skill_dir: Path) -> dict[str, str]:
    """Return one SKILL.md's frontmatter fields, or raise if it is unloadable.

    Parse YAML scalar fields and the supported CLIO metadata fields.
    Folded/literal descriptions retain YAML semantics; nested metadata cannot
    override the skill name or description.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise SkillProblem(f"{skill_dir} has no SKILL.md")
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise SkillProblem(f"{skill_md} must open with YAML frontmatter")
    _, _, rest = text.partition("---\n")
    frontmatter, separator, _ = rest.partition("\n---\n")
    if not separator:
        raise SkillProblem(f"{skill_md} has unterminated frontmatter")

    try:
        loaded = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise SkillProblem(f"{skill_md} has invalid YAML frontmatter: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SkillProblem(f"{skill_md} frontmatter must be a mapping")
    fields = {
        key: str(value)
        for key, value in loaded.items()
        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
    }
    metadata = loaded.get("clio-kit", {})
    if isinstance(metadata, dict):
        fields.update(
            {
                key: str(value)
                for key, value in metadata.items()
                if key in {"bundle", "servers", "provenance", "eval-status"}
                and isinstance(value, (str, int, float, bool))
            }
        )
    for required in ("name", "description"):
        if not fields.get(required):
            raise SkillProblem(f"{skill_md} frontmatter needs a {required}")
    if fields["name"] != skill_dir.name:
        raise SkillProblem(
            f"{skill_md} declares name {fields['name']!r} "
            f"but lives in {skill_dir.name!r}"
        )
    return fields


def has_recorded_scenarios(skill_dir: Path) -> bool:
    """Whether this skill records the scenarios it was checked against."""
    if any((skill_dir / name).is_file() for name in EVAL_FILENAMES):
        return True
    evals_dir = skill_dir / EVAL_DIRNAME
    return evals_dir.is_dir() and any(evals_dir.iterdir())


def check_skill(skill_dir: Path) -> SkillReport:
    """Run every rule against one skill directory and report what it found.

    Raises ``SkillProblem`` only when the directory cannot be read at all;
    everything a contributor can fix comes back in the report so they see the
    whole list at once rather than one failure per run.
    """
    fields = read_skill_frontmatter(skill_dir)
    description = fields["description"]
    report = SkillReport(name=fields["name"], description_chars=len(description))

    status = fields.get("eval-status")
    if status is not None and status not in EVAL_LADDER:
        report.problems.append(
            f"{skill_dir.name} has eval-status {status!r}, which is not on the "
            f"ladder: {', '.join(EVAL_LADDER)}."
        )

    if not has_recorded_scenarios(skill_dir):
        report.problems.append(
            f"{skill_dir.name} records no eval scenarios; add evals.md with the "
            "situations this skill was checked against. A skill with none is "
            "untested by definition."
        )

    if not description.startswith(REQUIRED_OPENING):
        report.problems.append(
            f"{skill_dir.name} description must open with {REQUIRED_OPENING!r} and "
            "name the situation that should fire it. A description restating what "
            "the body says is context paid for in every session and returned in none."
        )

    if REQUIRED_TRIGGER_CLAUSE not in description:
        report.problems.append(
            f"{skill_dir.name} description has no {REQUIRED_TRIGGER_CLAUSE!r} clause; "
            "quote the literal phrases a user types, which is what the match runs "
            'against -- for example: Triggers on "why is my job pending", "sbatch".'
        )

    if BOUNDARY_MARKER not in description:
        report.advisories.append(
            f"{skill_dir.name} declares no boundary. If another skill covers "
            "neighbouring ground they will hijack each other, and the user cannot "
            f"tell why the wrong one answered. Add: {BOUNDARY_MARKER} <the other "
            "job>; use <that skill>."
        )

    if len(description) > DESCRIPTION_BUDGET:
        report.advisories.append(
            f"{skill_dir.name} description is {len(description)} characters against a "
            f"{DESCRIPTION_BUDGET} budget. Move procedural detail into the body."
        )

    return report


def check_skill_collection(skills_root: Path) -> list[SkillReport]:
    """Check every skill under one ``skills/`` directory, name-sorted."""
    if not skills_root.is_dir():
        return []
    reports: list[SkillReport] = []
    for skill_dir in sorted(path for path in skills_root.iterdir() if path.is_dir()):
        try:
            reports.append(check_skill(skill_dir))
        except SkillProblem as exc:
            reports.append(SkillReport(name=skill_dir.name, problems=[str(exc)]))
    return reports


def always_on_cost(reports: list[SkillReport]) -> int:
    """Count discovery-description characters, not billed tokens.

    Kept under the existing API name for compatibility. Client loading policies
    and token estimates vary; skill bodies load when invoked.
    """
    return sum(report.description_chars for report in reports)
