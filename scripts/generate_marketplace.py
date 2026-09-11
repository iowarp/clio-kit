#!/usr/bin/env python3
"""Regenerate skills/bundles/community entries without touching MCP servers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clio_kit.community import read_community_entries
from clio_kit.federation import LOCK_NAME, refresh_marketplace
from clio_kit.marketplace_assets import write_extra_plugins
from generate_server_json import read_bundles, write_bundle_plugin, write_skills_plugin


def generate(root: Path, *, refresh: bool = False) -> None:
    path = root / ".claude-plugin" / "marketplace.json"
    marketplace = json.loads(path.read_text())
    bundles = read_bundles(root)
    owned = set(bundles) | {f"{name}-skills" for name in bundles}
    extra_names = {
        "clio-skills",
        "clio-agents",
    }
    community = read_community_entries(root)
    replacement_names = owned | extra_names | {entry["name"] for entry in community}
    entries = [
        entry
        for entry in marketplace["plugins"]
        if entry["name"] not in replacement_names
    ]
    skills = []
    for name, spec in bundles.items():
        skill = write_skills_plugin(root, name, spec)
        if skill:
            entries.append(skill)
            skills.append(skill["name"])
        entries.append(write_bundle_plugin(root, name, spec))
    entries.extend(write_extra_plugins(root, skills))
    entries.extend(community)
    marketplace["plugins"] = entries
    path.write_text(json.dumps(marketplace, indent=2) + "\n")
    if refresh:
        refresh_marketplace(root)
    elif path.with_name(LOCK_NAME).exists():
        print(
            "Kept the last federation snapshot; use --refresh to fetch external updates."
        )
    print(f"Generated {len(entries)} entries without modifying MCP server files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    generate(args.root, refresh=args.refresh)
