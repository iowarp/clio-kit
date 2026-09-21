"""Generate marketplace-only components without modifying scientific servers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def imported_skill_entries(root: Path) -> list[dict[str, Any]]:
    """Publish optional imported collections independently of default bundles."""
    entries = []
    for lock in sorted((root / "skills").glob("*/import-lock.json")):
        directory = lock.parent
        manifest = json.loads(
            (directory / ".claude-plugin" / "plugin.json").read_text()
        )
        entries.append(
            {
                "name": manifest["name"],
                "source": f"./skills/{directory.name}",
                "description": manifest["description"],
                "version": manifest["version"],
                "category": "skills",
            }
        )
    return entries


def write_extra_plugins(root: Path, skill_packages: list[str]) -> list[dict[str, Any]]:
    """Index declared collections, preserving their authored manifest metadata.

    Only a collection opting into primary skills has its dependencies refreshed.
    Ordinary contributed folders still need no inventory entry.
    """
    from clio_kit.discovery import tomllib
    from clio_kit.local_plugins import NAME, VERSION
    from clio_kit.plugins import validate_plugin

    inventory = tomllib.loads((root / "mcp-server-versions.toml").read_text())
    collections = inventory.get("collections", {})
    if not isinstance(collections, dict):
        raise ValueError("collections must be a table")
    planned = []
    for name, spec in collections.items():
        if not NAME.fullmatch(name) or not isinstance(spec, dict):
            raise ValueError(f"Invalid collection: {name!r}")
        if set(spec) - {"category", "primary-skills"}:
            raise ValueError(f"Unknown collection fields: {name!r}")
        if spec.get("category") not in {"skills", "agents", "hooks", "plugins"}:
            raise ValueError(f"Invalid collection category: {name!r}")
        if not isinstance(spec.get("primary-skills", False), bool):
            raise ValueError(f"primary-skills must be boolean: {name!r}")
        directory = root / "plugins" / name
        if any(path.is_symlink() for path in (root / "plugins", directory)) or any(
            path.is_symlink() for path in directory.rglob("*")
        ):
            raise ValueError(f"Linked collection content is not supported: {name!r}")
        manifest, problems = validate_plugin(directory, allow_reserved=True)
        if manifest.get("name") != name:
            problems.append("collection name must match its folder")
        version = manifest.get("version")
        if not isinstance(version, str) or not VERSION.fullmatch(version):
            problems.append("collection needs a semantic version")
        if problems:
            raise ValueError(f"{directory}: " + "; ".join(problems))
        if spec.get("primary-skills"):
            if not skill_packages:
                raise ValueError(f"Collection {name!r} would contain no primary skills")
            manifest["dependencies"] = sorted(set(skill_packages))
        planned.append((name, spec, directory, manifest))

    entries = []
    for name, spec, directory, manifest in planned:
        if spec.get("primary-skills"):
            (directory / ".claude-plugin/plugin.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
        entries.append(
            {
                "name": name,
                "source": f"./plugins/{name}",
                "description": manifest["description"],
                "version": manifest["version"],
                "category": spec["category"],
            }
        )
    return entries
