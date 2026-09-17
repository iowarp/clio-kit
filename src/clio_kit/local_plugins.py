"""Discover repository-owned native plugins without executing their components."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from clio_kit.plugins import PluginProblem, validate_plugin
from clio_kit.skills import read_skill_frontmatter

COMPONENT_ROOTS = ("plugins", "skills", "agents", "hooks")

NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def discover_local_plugins(
    root: Path, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return validated entries for folders not owned by another generator.

    Existing generated sources are authoritative. New local manifests remain
    handwritten; indexing never rewrites them or runs their commands/hooks.
    """
    known = {entry["name"]: entry for entry in entries}
    if len(known) != len(entries):
        raise ValueError("Duplicate plugin names in marketplace inputs")
    generated_paths = {
        (root / entry["source"]).resolve()
        for entry in entries
        if isinstance(entry.get("source"), str)
        and any(
            entry["source"].startswith(f"./{folder}/") for folder in COMPONENT_ROOTS
        )
    }
    result = []
    manifests = {}
    for kind in COMPONENT_ROOTS:
        folder = root / kind
        if not folder.exists():
            continue
        if folder.is_symlink():
            raise ValueError(f"{kind}/ must not be a symlink")
        for directory in sorted(folder.iterdir()):
            if directory.name.startswith(".") or not directory.is_dir():
                continue
            if directory.is_symlink() or any(
                path.is_symlink() for path in directory.rglob("*")
            ):
                raise ValueError(f"{directory}: linked plugin content is not supported")
            if directory.resolve() in generated_paths:
                continue
            try:
                manifest, problems = validate_plugin(directory, allow_reserved=True)
            except (PluginProblem, OSError, ValueError) as error:
                raise ValueError(f"{directory}: {error}") from error
            name = manifest.get("name")
            if (
                not isinstance(name, str)
                or not NAME.fullmatch(name)
                or name != directory.name
            ):
                problems.append("plugin name must match its kebab-case folder name")
            if isinstance(name, str) and (name in known or name in manifests):
                problems.append(
                    f"plugin name {name!r} collides with an indexed component"
                )
            version = manifest.get("version")
            if not isinstance(version, str) or not VERSION.fullmatch(version):
                problems.append("plugin needs a semantic version, for example 1.0.0")
            if (
                not isinstance(manifest.get("description"), str)
                or not manifest["description"].strip()
            ):
                problems.append("plugin needs a nonempty description")
            dependencies = manifest.get("dependencies", [])
            if not isinstance(dependencies, list) or not all(
                isinstance(dep, str) and NAME.fullmatch(dep) for dep in dependencies
            ):
                problems.append("dependencies must be a list of local plugin names")
            elif len(set(dependencies)) != len(dependencies):
                problems.append("duplicate plugin dependencies")
            if problems:
                raise ValueError(f"{directory}: " + "; ".join(problems))
            manifests[name] = manifest
            entry = {
                "name": name,
                "source": f"./{folder.name}/{name}",
                "description": manifest["description"],
                "version": version,
                "category": folder.name,
            }
            if isinstance(manifest.get("license"), str):
                entry["license"] = manifest["license"]
            result.append(entry)
    # Validate the entire local dependency graph, including generated parents.
    combined = {**known, **{entry["name"]: entry for entry in result}}
    for name, entry in combined.items():
        source = entry.get("source")
        if isinstance(source, str) and name not in manifests:
            path = root / source / ".claude-plugin/plugin.json"
            if path.is_file():
                manifests[name] = json.loads(path.read_text())
    complete: set[str] = set()

    def visit(name: str, trail: set[str]) -> None:
        if name in trail:
            raise ValueError(f"Cyclic local plugin dependency: {name}")
        if name in complete:
            return
        for dependency in manifests[name].get("dependencies", []):
            if dependency not in manifests:
                raise ValueError(
                    f"{name}: dependency {dependency!r} must be a known local plugin"
                )
            visit(dependency, trail | {name})
        complete.add(name)

    for entry in result:
        visit(entry["name"], set())
    # Portable installation uses the skill name, not its native plugin namespace.
    # Reject ambiguity before writing the catalogue or a wheel.
    skills: dict[str, Path] = {}
    for collection in COMPONENT_ROOTS:
        for skill in sorted((root / collection).glob("*/skills/*/SKILL.md")):
            name = read_skill_frontmatter(skill.parent)["name"]
            if name in skills:
                raise ValueError(
                    f"Duplicate portable skill {name!r}: {skills[name]} and {skill}"
                )
            skills[name] = skill
    return result
