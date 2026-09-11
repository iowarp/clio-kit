"""Install portable Agent Skills independently of any client's plugin format."""

from __future__ import annotations

import json
import shutil
import tempfile
from importlib import metadata
from pathlib import Path

import click

from clio_kit.skills import SkillProblem, check_skill, read_skill_frontmatter


def skill_inventory() -> dict[str, Path]:
    """Find canonical skills in a checkout or the installed wheel's shared data."""
    from clio_kit import MODULE_DIR

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
        if not skills:
            continue
        inventory = {}
        for skill in skills:
            fields = read_skill_frontmatter(skill.parent)
            if fields["name"] in inventory:
                raise SkillProblem(f"Duplicate skill name: {fields['name']}")
            inventory[fields["name"]] = skill.parent
        return inventory
    raise SkillProblem("No bundled skills found; reinstall CLIO Kit with skill assets")


def selected_skills(names: tuple[str, ...], bundle: str | None) -> dict[str, Path]:
    inventory = skill_inventory()
    unknown = set(names) - inventory.keys()
    if unknown:
        raise SkillProblem(f"Unknown skills: {', '.join(sorted(unknown))}")
    selected = {
        name: path
        for name, path in inventory.items()
        if (not names or name in names)
        and (not bundle or read_skill_frontmatter(path)["bundle"] == bundle)
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


def install_skills(skills: dict[str, Path], target: Path, replace: bool) -> list[str]:
    """Copy each complete skill folder; preflight conflicts before writing skills."""
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
    target.mkdir(parents=True, exist_ok=True)
    # Stage complete folders before replacing any existing installation.
    with tempfile.TemporaryDirectory(prefix=".clio-skills-", dir=target) as temporary:
        staging = Path(temporary)
        for name, source in skills.items():
            shutil.copytree(source, staging / name)
        for name, source in skills.items():
            destination = target / name
            if destination.exists():
                if _contents(destination) == _contents(source):
                    continue
                backup = staging / f"backup-{name}"
                destination.rename(backup)
                try:
                    (staging / name).rename(destination)
                except OSError:
                    backup.rename(destination)
                    raise
            else:
                (staging / name).rename(destination)
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
        records = [
            read_skill_frontmatter(path)
            for path in selected_skills((), bundle).values()
        ]
    except (SkillProblem, OSError) as exc:
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
    except (SkillProblem, OSError) as exc:
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
def install(
    names: tuple[str, ...], bundle: str | None, target: Path, replace: bool
) -> None:
    """Install named skills, one bundle, or all skills when no selector is given."""
    try:
        installed = install_skills(
            selected_skills(names, bundle), target.expanduser().absolute(), replace
        )
    except (SkillProblem, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Installed {len(installed)} skill(s) in {target}")
    click.echo(
        "Configure required MCP servers in your agent separately; skill folders contain instructions."
    )
