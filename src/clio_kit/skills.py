"""Discovery, listing, and install for clio-kit's shipped agent skills.

A "skill" here is the same open shape CLIO's own runtime already reads
(`clio_agent.gact.skills.SkillCatalog`, see that module's docstring in the
clio-agent repository): a directory under ``skills/`` holding a ``SKILL.md``
with a flat YAML-style frontmatter block (``name``, ``title``,
``description``) followed by a markdown procedure body.

**How this reaches the default CLIO agent.** clio-agent's skill catalog scans
three tiers on every session — ``pack`` (an agent blueprint's own
``skills/``), ``workspace`` (``<cwd>/.claude|.codex|.agents/skills``), and
``global`` (``~/.claude|.codex|.agents/skills``) — plus a package-local
``builtin`` tier it resolves from ITS OWN installation directory only. clio-kit
is a separate package, so a skill shipped here is never auto-discovered by
clio-agent's ``builtin`` tier; :func:`install_skills` (``clio-kit
skills-install``) closes that gap the same way Claude Code's own skill
directories are populated: by copying the shipped ``SKILL.md`` into a real
``workspace`` or ``global`` scan root. Installed to project scope
(``--scope project``, the default), clio-agent's default-registry root
expert (``base-agent``) auto-declares every workspace-scope skill it finds —
so once installed into a workspace, the skill is available there with no
blueprint edit. Installed to user scope (``--scope user``), the skill is
discoverable but NOT auto-declared for the default agent; an agent (or the
user) must still declare it under `skills:` to use it — see this package's
README/CONTRIBUTING for that tradeoff and the open follow-up to make this
`builtin` instead.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import click

SKILL_FILENAME = "SKILL.md"
REQUIRED_FRONTMATTER = ("name", "title", "description")

#: Where an installed skill lands, keyed by ``--scope``. Both are scan roots
#: clio-agent's ``SkillCatalog`` already reads (``_skill_search_roots``); the
#: kit does not invent a fourth location.
SKILL_INSTALL_SCOPES = {
    "user": Path.home() / ".claude" / "skills",
    "project": Path(".claude") / "skills",
}


def get_skills_path() -> Path:
    """Resolve the shipped skills directory (dev checkout or installed wheel).

    Mirrors :func:`clio_kit.get_prompts_path`'s search order: the repository
    copy wins in a source checkout, then the locations the wheel's
    ``shared-data`` can land a ``skills/`` directory in.
    """
    module_dir = Path(__file__).resolve().parent
    dev_path = module_dir.parent.parent / "skills"
    if dev_path.is_dir():
        return dev_path

    candidates = [
        module_dir.parent / "skills",
        module_dir / "skills",
        Path(sys.prefix) / "share" / "clio-kit" / "skills",
        Path.home() / ".local" / "share" / "clio-kit" / "skills",
        Path(sys.executable).parent.parent / "skills",
        Path(sys.executable).parent.parent / "share" / "skills",
        Path(sys.executable).parent.parent / "purelib" / "skills",
        Path(sys.executable).parent.parent / "data" / "skills",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return dev_path


def parse_frontmatter(text: str) -> dict[str, str]:
    """Return the leading ``---``-delimited flat ``key: value`` block, or ``{}``.

    Deliberately not a YAML parser -- the frontmatter every shipped skill
    (here and in clio-agent's own builtins) uses is flat scalars, and pulling
    in a YAML dependency for the launcher just to read three fields is not
    worth it. Matches the parsing clio-agent's ``SkillCatalog`` itself does.
    """
    if not text.startswith("---"):
        return {}
    _, _, rest = text.partition("\n")
    block, sep, _ = rest.partition("\n---")
    if not sep:
        return {}
    fields: dict[str, str] = {}
    for line in block.splitlines():
        key, delimiter, value = line.partition(":")
        if delimiter and key.strip() and not key.startswith((" ", "\t", "#")):
            fields[key.strip()] = value.strip().strip("\"'")
    return fields


def discover_skills(skills_root: Path) -> dict[str, Path]:
    """Map skill directory name to its ``SKILL.md``, one level under ``skills_root``."""
    if not skills_root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for child in sorted(skills_root.iterdir()):
        if child.name.startswith(".") or not child.is_dir():
            continue
        manifest = child / SKILL_FILENAME
        if manifest.is_file():
            found[child.name] = manifest
    return found


def skill_field(manifest: Path, field: str) -> str:
    """Return one frontmatter field from ``manifest``, or ``""``."""
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:
        return ""
    return parse_frontmatter(text).get(field, "")


def validate_skill(manifest: Path) -> list[str]:
    """Return every problem with one skill's ``SKILL.md``, empty when sound.

    Holds a skill to the same shape clio-agent's own catalog expects: the
    frontmatter carries ``name``/``title``/``description``, and the directory
    name matches the declared ``name`` (clio-agent falls back to the
    directory name when ``name`` is absent, but a mismatch between the two is
    almost always a copy/paste slip worth catching here rather than shipping
    an id nobody can declare correctly).
    """
    problems: list[str] = []
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"could not read {manifest}: {exc}"]
    fields = parse_frontmatter(text)
    for key in REQUIRED_FRONTMATTER:
        if not fields.get(key):
            problems.append(f"missing frontmatter '{key}'")
    if fields.get("name") and fields["name"] != manifest.parent.name:
        problems.append(
            f"frontmatter name {fields['name']!r} does not match directory {manifest.parent.name!r}"
        )
    body = text.split("---", 2)[-1] if text.startswith("---") else text
    if not body.strip():
        problems.append("body is empty after frontmatter")
    return problems


def format_skill_listing(skills_root: Path) -> list[str]:
    """Render ``clio-kit skills``."""
    skills = discover_skills(skills_root)
    if not skills:
        return ["No skills found."]
    lines: list[str] = ["Available skills:"]
    for name, manifest in sorted(skills.items()):
        lines.append(f"  {name}")
        description = skill_field(manifest, "description")
        if description:
            lines.append(f"      {description}")
    lines.append("")
    lines.append("Usage: clio-kit skill <skill-name>")
    return lines


def install_skills(skills_root: Path, destination: Path) -> list[str]:
    """Copy every shipped skill into ``destination``, replacing it wholesale.

    Each skill directory is replaced entirely (not merged), so a stale file
    from an older clio-kit release cannot survive an upgrade. Anything else
    already in ``destination`` (a user's own skill, or one from another
    source) is left untouched -- installing never clears the directory.
    """
    skills = discover_skills(skills_root)
    if not skills:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    for name, manifest in skills.items():
        target = destination / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(manifest.parent, target)
        installed.append(name)
    return installed


def format_install_result(skills_root: Path, scope: str) -> list[str]:
    """Install for one scope and render the outcome as output lines."""
    destination = SKILL_INSTALL_SCOPES[scope].expanduser().resolve()
    installed = install_skills(skills_root, destination)
    if not installed:
        return ["No skills found to install."]
    lines = [f"Installed {len(installed)} skill(s) to {destination}:"]
    lines.extend(f"  - {name}" for name in installed)
    if scope == "user":
        lines.append(
            "Note: user-scope skills are discoverable by clio-agent but not "
            "auto-declared for the default agent; declare the skill id under "
            "this agent's `skills:` to use it, or install with --scope project "
            "instead (auto-declared for the default-registry root expert)."
        )
    return lines


def _shipped_root() -> Path:
    return get_skills_path()


@click.command("skills")
def list_skills_command() -> None:
    """List the shipped clio-kit skills."""
    click.echo("\n".join(format_skill_listing(_shipped_root())))


@click.command("skill")
@click.argument("skill_name")
def show_skill_command(skill_name: str) -> None:
    """Print one shipped skill's SKILL.md to stdout."""
    skills = discover_skills(_shipped_root())
    manifest = skills.get(skill_name)
    if manifest is None:
        click.echo(f"Error: Unknown skill '{skill_name}'")
        click.echo(f"Available skills: {', '.join(sorted(skills)) or 'none'}")
        sys.exit(1)
    click.echo(manifest.read_text(encoding="utf-8"))


@click.command("skills-install")
@click.option(
    "--scope",
    type=click.Choice(sorted(SKILL_INSTALL_SCOPES)),
    default="project",
    help=(
        "Install into this project's workspace (./.claude/skills, auto-declared "
        "for CLIO's default agent) or this user's global skills "
        "(~/.claude/skills, discoverable but not auto-declared). Default: project."
    ),
)
def install_skills_command(scope: str) -> None:
    """Copy clio-kit's shipped skills to where CLIO (or Claude Code) discovers them."""
    click.echo("\n".join(format_install_result(_shipped_root(), scope)))


@click.command("skills-validate")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
)
def validate_skills_command(path: Optional[Path]) -> None:
    """Check a skill collection's SKILL.md shape. Defaults to the shipped collection."""
    root = path or _shipped_root()
    skills = discover_skills(root)
    if not skills:
        click.echo(f"No skills found in {root}")
        sys.exit(1)
    failures = 0
    for name, manifest in sorted(skills.items()):
        problems = validate_skill(manifest)
        if problems:
            failures += 1
            click.echo(f"FAIL: {name}")
            for problem in problems:
                click.echo(f"  - {problem}")
        else:
            click.echo(f"OK:   {name}")
    if failures:
        click.echo(f"\n{failures} of {len(skills)} skill(s) have problems.")
        sys.exit(1)
    click.echo(f"\nAll {len(skills)} skill(s) validate.")


SKILL_COMMANDS: tuple[click.Command, ...] = (
    list_skills_command,
    show_skill_command,
    install_skills_command,
    validate_skills_command,
)


__all__ = [
    "SKILL_COMMANDS",
    "SKILL_INSTALL_SCOPES",
    "discover_skills",
    "format_install_result",
    "format_skill_listing",
    "get_skills_path",
    "install_skills",
    "parse_frontmatter",
    "skill_field",
    "validate_skill",
]
