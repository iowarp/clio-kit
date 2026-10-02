"""Resolve published packages without cloning the catalogue's source repository."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from clio_kit.component_cache import guarded
from clio_kit.component_store import artifact_path, catalogue, fetch


def validate_index(index: dict) -> None:
    """Enforce canonical skill ownership and references before planning downloads."""
    owners: dict[str, str] = {}
    for package, record in index["packages"].items():
        if record["artifact"] not in index["artifacts"]:
            raise ValueError(f"Missing package artifact: {package}")
        for name in record["skills"]:
            if name in owners:
                raise ValueError(f"Conflicting skill definitions: {name}")
            owners[name] = package
            skill = index["skills"].get(name)
            if skill is None:
                raise ValueError(f"Unknown skill: {name}")
            if skill["package"] != package or skill["name"] != name:
                raise ValueError(f"Conflicting skill definitions: {name}")
            if skill["artifact"] not in index["artifacts"]:
                raise ValueError(f"Missing skill artifact: {name}")
    if set(owners) != set(index["skills"]):
        raise ValueError("Release contains skills without an owning package")


def package_closure(name: str, index: dict) -> list[str]:
    validate_index(index)
    visited: set[str] = set()
    ordered = []

    def visit(package: str, trail: set[str]) -> None:
        if package in trail:
            raise ValueError(f"Cyclic dependency: {package}")
        if package in visited:
            return
        if package in index.get("external", []):
            raise ValueError(
                f"{package} is indexed externally; use its publisher's installation route"
            )
        if package not in index["packages"]:
            raise ValueError(f"Unknown package: {package}")
        for dependency in index["packages"][package]["manifest"].get(
            "dependencies", []
        ):
            visit(dependency, trail | {package})
        visited.add(package)
        ordered.append(package)

    visit(name, set())
    return ordered


def release_components(name: str) -> dict:
    """Plan portable installation entirely from trusted, installed metadata."""
    index = catalogue()
    skills = {}
    servers: dict = {}
    unsupported: list[str] = []
    artifacts: set[str] = set()
    for package in package_closure(name, index):
        record = index["packages"][package]
        manifest = record["manifest"]
        if manifest.get("skills"):
            raise ValueError(
                f"{package}: this installer requires skills/*/SKILL.md layout"
            )
        unsupported.extend(f"{package}: {field}" for field in record["unsupported"])
        for skill in record["skills"]:
            key = index["skills"][skill]["artifact"]
            skills[skill] = artifact_path(key, index)
            artifacts.add(key)
        for server, settings in record["servers"].items():
            if "${CLAUDE_PLUGIN_ROOT}" in json.dumps(settings):
                artifacts.add(record["artifact"])

                def expand(value):
                    if isinstance(value, str):
                        return value.replace(
                            "${CLAUDE_PLUGIN_ROOT}",
                            str(artifact_path(record["artifact"], index)),
                        )
                    if isinstance(value, list):
                        return [expand(item) for item in value]
                    if isinstance(value, dict):
                        return {key: expand(item) for key, item in value.items()}
                    return value

                settings = expand(settings)
            if server in servers and servers[server] != settings:
                raise ValueError(f"Conflicting MCP definitions: {server}")
            servers[server] = settings
    return {
        "skills": skills,
        "servers": servers,
        "unsupported": unsupported,
        "artifacts": sorted(artifacts),
    }


@guarded
def fetch_native_package(name: str | tuple[str, ...], target: Path) -> dict:
    """Create a selected local Claude marketplace, including native components.

    No repository clone, hook execution, or client profile edits. Refuse existing
    output so a refresh cannot overwrite user files or an active installation.
    """
    index = catalogue()
    selections = (name,) if isinstance(name, str) else name
    names = list(
        dict.fromkeys(
            package
            for selection in selections
            for package in package_closure(selection, index)
        )
    )
    target = target.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise ValueError(
            "Target already exists; choose a new directory for this version"
        )
    if any(parent.is_symlink() for parent in target.parents):
        raise ValueError("Refusing linked destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".clio-marketplace-", dir=target.parent
    ) as temporary:
        staging = Path(temporary) / "marketplace"
        staging.mkdir()
        entries = []
        for package in names:
            record = index["packages"][package]
            destination = staging / "plugins" / package
            shutil.copytree(fetch(record["artifact"], index), destination)
            for skill in record["skills"]:
                shutil.copytree(
                    fetch(index["skills"][skill]["artifact"], index),
                    destination / "skills" / skill,
                )
            entries.append(
                {
                    "name": package,
                    "source": f"./plugins/{package}",
                    "description": record["manifest"]["description"],
                }
            )
        (staging / ".claude-plugin").mkdir()
        (staging / ".claude-plugin/marketplace.json").write_text(
            json.dumps(
                {
                    "name": "clio-kit",
                    "owner": {"name": "IOWarp"},
                    "metadata": {
                        "description": "Selected CLIO Kit components",
                        "version": index["version"],
                    },
                    "plugins": entries,
                },
                indent=2,
            )
            + "\n"
        )
        staging.rename(target)
    return {
        "package": name,
        "packages": names,
        "marketplace": str(target),
        "version": index["version"],
    }
