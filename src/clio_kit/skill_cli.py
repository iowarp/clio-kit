"""Install portable Agent Skills independently of any client's plugin format."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import click

from clio_kit.component_cache import guarded
from clio_kit.install_transaction import InstallTransaction
from clio_kit.skills import SkillProblem, check_skill, read_skill_frontmatter


def _local_skill_inventory() -> dict[str, Path]:
    """Find canonical skills in a checkout or the installed wheel's shared data."""
    from clio_kit import MODULE_DIR
    from clio_kit.component_store import INDEX_FILE

    if INDEX_FILE.is_file():
        return {}
    roots = [MODULE_DIR.parent.parent / "skills"]
    try:
        distribution = metadata.distribution("clio-kit")
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        for record in distribution.files or ():
            if "clio-kit-skills" not in Path(str(record)).parts:
                continue
            located = Path(str(distribution.locate_file(record))).resolve()
            for parent in located.parents:
                if parent.name == "clio-kit-skills":
                    roots.append(parent)
                    break
            if len(roots) > 1:
                break
    for root in roots:
        skills = sorted(root.glob("*/skills/*/SKILL.md"))
        if root == MODULE_DIR.parent.parent / "skills":
            for kind in ("plugins", "agents", "hooks"):
                skills += sorted((root.parent / kind).glob("*/skills/*/SKILL.md"))
        if not skills:
            continue
        inventory = {}
        for skill in skills:
            fields = read_skill_frontmatter(skill.parent)
            if fields["name"] in inventory:
                raise SkillProblem(f"Duplicate skill name: {fields['name']}")
            inventory[fields["name"]] = skill.parent
        return inventory
    return {}


def skill_inventory() -> dict[str, Path]:
    """Materialize all skills when explicitly requested by a Python caller."""
    return selected_skills((), None)


def skill_records() -> dict[str, dict]:
    local = _local_skill_inventory()
    if local:
        return {
            name: {
                "bundle": path.parents[1].name,
                "servers": "unspecified",
                **read_skill_frontmatter(path),
            }
            for name, path in local.items()
        }
    from clio_kit.component_store import catalogue

    return catalogue()["skills"]


def selected_skills(names: tuple[str, ...], bundle: str | None) -> dict[str, Path]:
    records = skill_records()
    selected = _select_records(records, names, bundle)
    local = _local_skill_inventory()
    if local:
        return {name: local[name] for name in selected}
    from clio_kit.component_store import fetch

    return {name: fetch(records[name]["artifact"]) for name in selected}


def _select_records(records: dict, names: tuple[str, ...], bundle: str | None) -> dict:
    unknown = set(names) - records.keys()
    if unknown:
        raise SkillProblem(f"Unknown skills: {', '.join(sorted(unknown))}")
    selected = {
        name: record
        for name, record in records.items()
        if (not names or name in names) and (not bundle or record["bundle"] == bundle)
    }
    if not selected:
        raise SkillProblem(f"No skills match bundle {bundle!r}")
    if names and set(names) != selected.keys():
        raise SkillProblem("Some named skills do not belong to the selected bundle")
    return selected


def _contents(directory: Path) -> dict[str, bytes]:
    result = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise SkillProblem(f"Refusing linked skill content: {path}")
        if path.is_file():
            result[str(path.relative_to(directory))] = path.read_bytes()
    return result


def stage_skills(
    skills: dict[str, Path],
    target: Path,
    replace: bool,
    transaction: InstallTransaction,
) -> None:
    """Validate all conflicts, then stage only changed complete skill folders."""
    for name, source in skills.items():
        if target.resolve().is_relative_to(source.resolve()):
            raise SkillProblem(f"Target cannot be inside a source skill: {source}")
        report = check_skill(source)
        if report.problems:
            raise SkillProblem("; ".join(report.problems))
        source_files = _contents(source)
        destination = target / name
        if destination.is_symlink() or (
            destination.exists() and not destination.is_dir()
        ):
            raise SkillProblem(
                f"Skill destination is not a regular directory: {destination}"
            )
        existing_files = _contents(destination) if destination.exists() else None
        if (
            existing_files is not None
            and not replace
            and existing_files != source_files
        ):
            raise SkillProblem(
                f"{destination} contains different files; review them before using --replace"
            )
    for name, source in skills.items():
        destination = target / name
        if destination.exists() and _contents(destination) == _contents(source):
            continue
        transaction.directory(destination, source)


def install_skills(skills: dict[str, Path], target: Path, replace: bool) -> list[str]:
    """Install complete skill folders, rolling back the whole set on failure."""
    with InstallTransaction() as transaction:
        stage_skills(skills, target, replace, transaction)
        transaction.commit()
    return list(skills)


@click.group("skill")
def skill_group() -> None:
    """List, validate and install standard SKILL.md folders for compatible agents."""


@skill_group.command("list")
@click.option("--bundle", help="Workflow bundle, such as clio-scientific-io.")
@click.option("--json", "as_json", is_flag=True)
def list_skills(bundle: str | None, as_json: bool) -> None:
    """List available skills and their required MCP servers."""
    try:
        records = list(_select_records(skill_records(), (), bundle).values())
    except (SkillProblem, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(records, indent=2))
    else:
        for record in records:
            click.echo(
                f"{record['name']} ({record['bundle']}; servers: {record['servers']})"
            )


@skill_group.command("validate")
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def validate_skill(directory: Path) -> None:
    """Validate a standalone skill without requiring a plugin manifest."""
    try:
        report = check_skill(directory)
        if report.problems:
            raise SkillProblem("; ".join(report.problems))
    except (SkillProblem, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"OK: {report.name}")
    for advisory in report.advisories:
        click.echo(f"Note: {advisory}")


@skill_group.command("install")
@click.argument("names", nargs=-1)
@click.option("--bundle", help="Install only this workflow's skills.")
@click.option(
    "--target",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="The agent's skill discovery directory.",
)
@click.option(
    "--replace",
    is_flag=True,
    help="Replace conflicting skill folders after reviewing local changes.",
)
@guarded
def install(
    names: tuple[str, ...], bundle: str | None, target: Path, replace: bool
) -> None:
    """Install named skills, one bundle, or all skills when no selector is given."""
    try:
        installed = install_skills(
            selected_skills(names, bundle), target.expanduser().absolute(), replace
        )
    except (SkillProblem, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Installed {len(installed)} skill(s) in {target}")
    click.echo(
        "Configure required MCP servers in your agent separately; skill folders contain instructions."
    )
