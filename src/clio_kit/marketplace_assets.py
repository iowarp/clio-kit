"""Generate marketplace-only components without modifying scientific servers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_extra_plugins(root: Path, skill_packages: list[str]) -> list[dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "clio-skills": {
            "description": "All CLIO scientific workflow skills, without MCP servers.",
            "dependencies": sorted(skill_packages),
        },
        "clio-agents": {
            "description": "Scientific workflow planning and independent evidence review agents."
        },
    }
    entries = []
    for name, spec in specs.items():
        directory = root / "plugins" / name
        manifest = directory / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "name": name,
                    "version": "1.0.0",
                    "author": {"name": "IoWarp Team"},
                    **spec,
                },
                indent=2,
            )
            + "\n"
        )
        entries.append(
            {
                "name": name,
                "source": f"./plugins/{name}",
                "description": spec["description"],
                "version": "1.0.0",
                "category": "agents" if name == "clio-agents" else "skills",
            }
        )
    return entries
