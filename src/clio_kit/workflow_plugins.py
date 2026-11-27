"""Generate optional task plugins independently of the primary bundle partition."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


def write_workflow_plugins(
    root: Path, entries: list[dict[str, Any]], *, author: dict[str, str]
) -> list[dict[str, Any]]:
    """Validate all task definitions before writing dependency-only manifests.

    Dependencies name already indexed, maintained plugins (servers, skills,
    agents or primary bundles). Task-to-task and upstream references are not
    accepted here: they need an explicit version/cycle policy first.
    """
    inventory = tomllib.loads((root / "mcp-server-versions.toml").read_text())
    workflows = inventory.get("workflows", {})
    if not isinstance(workflows, dict):
        raise ValueError("workflows must be a table")
    occupied = {entry["name"] for entry in entries}
    available = {
        entry["name"]
        for entry in entries
        if isinstance(entry.get("source"), str)
        and entry["source"].startswith(("./plugins/", "./skills/"))
    }
    for name, spec in workflows.items():
        if not re.fullmatch(r"clio-[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError(f"Invalid workflow plugin name: {name!r}")
        if name in occupied:
            raise ValueError(f"Workflow {name!r} collides with an existing plugin")
        if not isinstance(spec, dict) or set(spec) != {
            "version",
            "description",
            "dependencies",
        }:
            raise ValueError(
                f"Workflow {name!r} requires only version, description, dependencies"
            )
        if not isinstance(spec["version"], str) or not re.fullmatch(
            r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", spec["version"]
        ):
            raise ValueError(f"Workflow {name!r} needs a stable semantic version")
        if not isinstance(spec["description"], str) or not spec["description"].strip():
            raise ValueError(f"Workflow {name!r} needs a description")
        dependencies = spec["dependencies"]
        if (
            not isinstance(dependencies, list)
            or not dependencies
            or not all(isinstance(dep, str) and dep for dep in dependencies)
        ):
            raise ValueError(f"Workflow {name!r} needs a nonempty dependencies list")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"Workflow {name!r} has duplicate dependencies")
        unknown = sorted(set(dependencies) - available)
        if unknown:
            raise ValueError(
                f"Workflow {name!r} requires existing maintained plugins: {unknown}"
            )

    result = []
    for name, spec in workflows.items():
        manifest = {
            "name": name,
            **spec,
            "author": author,
            "license": "BSD-3-Clause",
        }
        path = root / "plugins" / name / ".claude-plugin" / "plugin.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        result.append(
            {
                "name": name,
                "source": f"./plugins/{name}",
                "description": spec["description"],
                "version": spec["version"],
                "category": "workflow",
                "license": "BSD-3-Clause",
            }
        )
    return result
