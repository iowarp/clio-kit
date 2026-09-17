#!/usr/bin/env python3
"""Regenerate skills/bundles/community entries without touching MCP servers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clio_kit.community import (
    read_community_entries,
    read_federated_marketplaces,
    write_shipped_marketplaces,
)
from clio_kit.federation import LOCK_NAME, refresh_marketplace, read_snapshot
from clio_kit.marketplace_assets import imported_skill_entries, write_extra_plugins
from clio_kit.local_plugins import discover_local_plugins
from clio_kit.workflow_plugins import write_workflow_plugins
from generate_server_json import (
    PLUGIN_AUTHOR,
    read_bundles,
    read_server_versions,
    write_bundle_plugin,
    write_skills_plugin,
)


def generate(root: Path, *, refresh: bool = False) -> None:
    path = root / ".claude-plugin" / "marketplace.json"
    marketplace = json.loads(path.read_text())
    bundles = read_bundles(root)
    community = read_community_entries(root)
    imported = imported_skill_entries(root)
    # Only server records require runtime probes. Rebuild every other entry
    # from its source so removed/renamed local packages cannot linger.
    servers = {f"clio-{name}" for name in read_server_versions(root)}
    entries = [entry for entry in marketplace["plugins"] if entry["name"] in servers]
    if {entry["name"] for entry in entries} != servers:
        raise ValueError("Missing server entries; run generate_server_json.py first")
    skills = []
    for name, spec in bundles.items():
        skill = write_skills_plugin(root, name, spec)
        if skill:
            entries.append(skill)
            skills.append(skill["name"])
        entries.append(write_bundle_plugin(root, name, spec))
    entries.extend(write_extra_plugins(root, skills))
    entries.extend(imported)
    entries.extend(community)
    entries.extend(read_snapshot(root))
    entries.extend(write_workflow_plugins(root, entries, author=PLUGIN_AUTHOR))
    entries.extend(discover_local_plugins(root, entries))
    marketplace["plugins"] = entries
    path.write_text(json.dumps(marketplace, indent=2) + "\n")
    if refresh:
        refresh_marketplace(root)
    elif path.with_name(LOCK_NAME).exists():
        print(
            "Kept the last federation snapshot; use --refresh to fetch external updates."
        )
    write_shipped_marketplaces(
        root / "src" / "clio_kit", read_federated_marketplaces(root)
    )
    print(f"Generated {len(entries)} entries without modifying MCP server files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--website", action="store_true")
    args = parser.parse_args()
    generate(args.root, refresh=args.refresh)
    if args.website:
        from generate_website_catalogue import generate as website_catalogue

        output = args.root / "clio-kit-website/src/data/catalogue.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(website_catalogue(args.root), indent=2, ensure_ascii=False)
            + "\n"
        )
        print(f"Updated website catalogue: {output}")
