#!/usr/bin/env python3
"""Build the website catalogue from shipped manifests, skills and federation data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from clio_kit.hooks import hook_components
from clio_kit.client_install import CLIENTS, server_settings

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/iowarp/clio-kit"
WORKFLOWS = {
    "clio-scientific-io": (
        "Explore a scientific dataset",
        "Scientific data",
        "A bounded view of your data, its structure and storage choices.",
    ),
    "clio-analysis": (
        "From results to insight",
        "Analysis & visualization",
        "Summary statistics, clear figures and guidance for interpreting them.",
    ),
    "clio-hpc": (
        "Run work on a cluster",
        "HPC & infrastructure",
        "A resource-aware path from software setup to workload monitoring.",
    ),
    "clio-performance": (
        "Understand a slow job",
        "Performance",
        "An investigation of I/O behavior, logs and potential bottlenecks.",
    ),
    "clio-geoscience": (
        "Explore the physical world",
        "Geoscience",
        "Workflows for geographic features, terrain and seismic data.",
    ),
    "clio-research": (
        "Find the research behind it",
        "Research & discovery",
        "Papers and datasets, with guidance for discovery and staging.",
    ),
}


def frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Missing frontmatter: {path}")
    header, body = text[4:].split("\n---", 1)
    return yaml.safe_load(header), body.strip()


def title(name: str) -> str:
    if name == "clio-coder-skills":
        return "Clio Coder skills"
    acronyms = {
        "hpc": "HPC",
        "io": "I/O",
        "mcp": "MCP",
        "arxiv": "arXiv",
        "hdf5": "HDF5",
        "tdd": "TDD",
        "ndp": "NDP",
    }
    words = name.removeprefix("clio-kit-").removeprefix("clio-").split("-")
    return " ".join(acronyms.get(word, word.capitalize()) for word in words)


def source_url(source: dict) -> str:
    repo = source.get("repo")
    url = f"https://github.com/{repo}" if repo else source.get("url", "")
    url = url.removesuffix(".git")
    if source.get("path") and source.get("sha"):
        url += f"/tree/{source['sha']}/{source['path']}"
    if not url and source.get("package"):
        url = f"https://www.npmjs.com/package/{source['package']}"
    if not url.startswith("https://"):
        raise ValueError(f"No HTTPS repository for indexed entry: {source}")
    return url


def classify_records(records: list[dict], entries: dict, root: Path) -> None:
    """Separate catalogue identity from the client's installation package.

    IDs remain stable for existing links. Classification follows local contents
    and dependencies; an external index alone cannot establish package contents.
    """
    by_id = {record["id"]: record for record in records}
    resolved: dict[str, set[str]] = {}

    def component_types(record: dict, visiting: frozenset = frozenset()) -> set[str]:
        key = record["id"]
        if key in visiting:
            raise ValueError(f"Cyclic catalogue membership at {key}")
        if key in resolved:
            return resolved[key]
        kind = record["kind"]
        result = {kind} if kind in {"mcp", "skill", "agent", "hook"} else set()
        if kind in {"plugin", "workflow"} and record["origin"] != "Indexed":
            source = entries[record["name"]]["source"]
            directory = root / source
            manifest = json.loads(
                (directory / ".claude-plugin/plugin.json").read_text()
            )
            if (directory / ".mcp.json").is_file() or manifest.get("mcpServers"):
                result.add("mcp")
            # Declared custom locations count too, even without a default folder.
            for field, component in (
                ("skills", "skill"),
                ("agents", "agent"),
                ("commands", "command"),
            ):
                if manifest.get(field) or (
                    field == "commands" and any(directory.glob("commands/*.md"))
                ):
                    result.add(component)
        for member in record["members"]:
            result.update(component_types(by_id[member], visiting | {key}))
        resolved[key] = result
        return result

    # Resolve before mutating kinds, so dependency order cannot affect results.
    for record in records:
        component_types(record)
    for record in records:
        previous = record["kind"]
        record["componentTypes"] = sorted(resolved[record["id"]])
        record["installation"] = (
            "portable-skill"
            if previous == "skill"
            else "launcher"
            if previous == "mcp"
            else "service"
            if previous == "service"
            else "native-package"
        )
        if previous == "mcp":
            record["nativePackage"] = f"clio-{record['name']}"
        if previous in {"mcp", "skill"}:
            record["clients"] = [*CLIENTS, "other"]
        elif previous in {"plugin", "workflow"}:
            record["nativePackage"] = record["name"]
            if previous == "workflow":
                record["kind"] = "plugin"
                record["role"] = "Workflow plugin"
            elif record["origin"] == "Indexed":
                record["kind"] = "package"
                record["role"] = "External package"
            elif len(record["componentTypes"]) > 1:
                record["role"] = "Workflow plugin"
            elif record["componentTypes"] == ["mcp"]:
                record["kind"] = "mcp"
            else:
                record["kind"] = "collection"
                record["role"] = "Component collection"
        if previous == "plugin" and record["origin"] != "Indexed":
            record["docs"] = "/docs/plugins"
        if previous in {"plugin", "workflow"} and record["origin"] != "Indexed":
            if resolved[record["id"]] & {"mcp", "skill"}:
                record["clients"] = [
                    "claude-code",
                    *[c for c in CLIENTS if c != "claude-code"],
                ]
                record["projectInstall"] = True


def generate(root: Path) -> dict:
    marketplace = json.loads((root / ".claude-plugin/marketplace.json").read_text())
    inventory = tomllib.loads((root / "mcp-server-versions.toml").read_text())
    icons = inventory.get("icons", {})
    missing_icons = {
        name
        for name in inventory["servers"]
        if not isinstance(icons.get(name), str) or not icons[name].strip()
    }
    if missing_icons:
        raise ValueError(
            f"Maintained servers need explicit icons: {sorted(missing_icons)}"
        )
    entries = {entry["name"]: entry for entry in marketplace["plugins"]}
    records: list[dict] = []
    publishers = {
        "clio-kit": {
            "id": "clio-kit",
            "name": "CLIO Kit",
            "organization": "IoWarp",
            "description": "Scientific tools and reusable workflows maintained by the Gnosis Research Center. Includes attributed adaptations from Clio Coder.",
            "repository": REPOSITORY,
            "origin": "Maintained collection",
        }
    }
    for path in sorted((root / "community/entries").glob("*.toml")):
        entry = tomllib.loads(path.read_text())
        publisher = entry["name"].removeprefix("iowarp-")
        publishers[publisher] = {
            "id": publisher,
            "name": "Clio Coder" if publisher == "clio-coder" else title(publisher),
            "organization": entry.get("maintainer", "Community"),
            "description": entry["description"],
            "repository": source_url(entry["source"]),
            "origin": "Indexed upstream collection",
        }

    def add(kind: str, name: str, **values) -> dict:
        record = {
            "id": f"{kind}/{name}",
            "kind": kind,
            "name": name,
            "title": title(name),
            "description": "",
            "publisher": "clio-kit",
            "origin": "Maintained",
            "version": "",
            "category": "Scientific computing",
            "source": REPOSITORY,
            "path": "",
            "members": [],
            "servers": [],
            "tags": [],
            "clients": ["claude-code"],
            "license": "BSD-3-Clause",
            "docs": "/docs/marketplace",
            "evidence": "See validation coverage",
            **values,
        }
        records.append(record)
        return record

    skill_paths = [
        path
        for folder in ("skills", "plugins", "agents", "hooks")
        for path in (root / folder).glob("*/skills/*/SKILL.md")
    ]
    for path in sorted(skill_paths):
        meta, body = frontmatter(path)
        metadata = meta.get("metadata", {})
        bundle = metadata.get("bundle", "")
        heading = re.search(r"^# (.+)$", body, re.MULTILINE)
        adapted = metadata.get("provenance") == "adapted"
        collection = json.loads(
            (path.parents[2] / ".claude-plugin/plugin.json").read_text()
        )
        add(
            "skill",
            meta["name"],
            title=heading.group(1) if heading else title(meta["name"]),
            description=meta["description"],
            publisher="clio-kit",
            origin="Adapted" if adapted else "Maintained",
            version=collection["version"],
            category=WORKFLOWS.get(bundle, ("", "Coding & research", ""))[1],
            source=metadata.get("source", REPOSITORY),
            path=str(path.relative_to(root)),
            servers=[
                s.strip()
                for s in metadata.get("servers", "").split(",")
                if s.strip() not in ("", "none")
            ],
            tags=[bundle, "Clio Coder" if adapted else "Scientific workflow"],
            clients=["claude-code", "codex", "antigravity", "other"],
            license=meta.get("license", "BSD-3-Clause"),
            evidence="Scenarios recorded"
            if (path.parent / "evals.md").is_file() or (path.parent / "evals").is_dir()
            else "No local scenarios recorded",
            docs="/docs/marketplace#clio-coder-integration"
            if adapted
            else "/docs/marketplace#installable-components",
        )

    for name, spec in inventory["bundles"].items():
        display, category, outcome = WORKFLOWS.get(
            name, (title(name), "Scientific computing", spec["description"])
        )
        add(
            "workflow",
            name,
            title=display,
            category=category,
            description=spec["description"],
            outcome=outcome,
            version=spec["version"],
            members=[f"mcp/{s}" for s in spec["servers"]]
            + [r["id"] for r in records if r["kind"] == "skill" and name in r["tags"]],
            servers=[f"clio-{s}" for s in spec["servers"]],
            path=f"plugins/{name}/.claude-plugin/plugin.json",
            evidence="Membership from manifest",
        )

    for name, version in inventory["servers"].items():
        entry = entries[f"clio-{name}"]
        bundle = next(
            key for key, spec in inventory["bundles"].items() if name in spec["servers"]
        )
        add(
            "mcp",
            name,
            icon=icons[name],
            title={"hdf5": "HDF5", "adios": "ADIOS2", "ndp": "NDP"}.get(
                name, title(name)
            ),
            description=entry["description"],
            version=version,
            category=WORKFLOWS.get(bundle, ("", "Scientific computing", ""))[1],
            tags=entry.get("keywords", []),
            path=f"mcp-servers/{name}",
            clients=["claude-code", "codex", "antigravity", "other"],
            docs=f"/docs/mcps/{name.replace('-', '_')}",
            evidence="Locked runtime",
        )

    for name, entry in entries.items():
        if (
            name in inventory["bundles"]
            or name in inventory.get("workflows", {})
            or name.removeprefix("clio-") in inventory["servers"]
        ):
            continue
        source = entry["source"]
        indexed = isinstance(source, dict)
        values = {}
        if indexed:
            key = entry.get("metadata", {}).get("clioFederation", name)
            publisher = key.removeprefix("iowarp-")
            if publisher not in publishers:
                publishers[publisher] = {
                    "id": publisher,
                    "name": "Clio Coder"
                    if publisher == "clio-coder"
                    else title(publisher),
                    "organization": "Community",
                    "description": entry["description"],
                    "repository": source_url(source),
                    "origin": "Indexed upstream collection",
                }
            values.update(
                publisher=publisher,
                origin="Indexed",
                source=source_url(source),
                revision=source.get("sha", ""),
                evidence="Pinned upstream source"
                if source.get("sha")
                else "Indexed source",
            )
        else:
            values["path"] = source
            directory = root / source
            manifest = json.loads(
                (directory / ".claude-plugin/plugin.json").read_text()
            )
            values["members"] = [
                f"skill/{frontmatter(path)[0]['name']}"
                for path in sorted(directory.glob("skills/*/SKILL.md"))
            ] + [
                f"agent/{directory.name}/{frontmatter(path)[0]['name']}"
                for path in sorted(directory.glob("agents/*.md"))
            ]
            if hook_components(directory, manifest)[0]:
                values["members"].append(f"hook/{directory.name}")
            for dependency in manifest.get("dependencies", []):
                kind, component = "plugin", dependency
                if dependency.removeprefix("clio-") in inventory["servers"]:
                    kind, component = "mcp", dependency.removeprefix("clio-")
                elif dependency in inventory["bundles"] or dependency in inventory.get(
                    "workflows", {}
                ):
                    kind = "workflow"
                values["members"].append(f"{kind}/{component}")
            mcp_path = directory / ".mcp.json"
            mcp_config = manifest.get("mcpServers", {})
            if mcp_path.is_file():
                mcp_config = json.loads(mcp_path.read_text())
            if isinstance(mcp_config, dict):
                values["servers"] = [
                    f"plugin:{name}:{server}"
                    for server in mcp_config.get("mcpServers", mcp_config)
                ]
        add(
            "plugin",
            name,
            description=entry["description"],
            version=entry.get("version", ""),
            category=title(entry.get("category", "Plugin collection")),
            license=entry.get("license", "See source licence"),
            **values,
        )

    def installed_servers(
        name: str, visiting: frozenset[str] = frozenset()
    ) -> set[str]:
        if name in visiting:
            raise ValueError(f"Cyclic plugin dependencies at {name}")
        if name.removeprefix("clio-") in inventory["servers"]:
            return {name}
        source = entries[name]["source"]
        if not isinstance(source, str):
            raise ValueError(f"Workflow dependency must be maintained: {name}")
        manifest = json.loads(
            (root / source / ".claude-plugin/plugin.json").read_text()
        )
        return {
            server
            for dependency in manifest.get("dependencies", [])
            for server in installed_servers(dependency, visiting | {name})
        }

    for name, spec in inventory.get("workflows", {}).items():
        entry = entries[name]
        manifest = json.loads(
            (root / entry["source"] / ".claude-plugin/plugin.json").read_text()
        )
        if any(manifest.get(key) != value for key, value in spec.items()):
            raise ValueError(f"Stale workflow manifest: {name}")
        members = []
        for dependency in spec["dependencies"]:
            kind = "workflow" if dependency in inventory["bundles"] else "plugin"
            component = dependency
            if dependency.removeprefix("clio-") in inventory["servers"]:
                kind, component = "mcp", dependency.removeprefix("clio-")
            members.append(f"{kind}/{component}")
        component_root = root / entry["source"]
        members.extend(
            f"skill/{frontmatter(path)[0]['name']}"
            for path in sorted(component_root.glob("skills/*/SKILL.md"))
        )
        if hook_components(component_root, manifest)[0]:
            members.append(f"hook/{name}")
        add(
            "workflow",
            name,
            description=spec["description"],
            version=spec["version"],
            members=members,
            servers=sorted(installed_servers(name)),
            path=f"plugins/{name}/.claude-plugin/plugin.json",
            docs="/docs/plugins#task-plugins",
            evidence="Membership from manifest",
        )

    agent_paths = [
        path
        for folder in ("plugins", "skills", "agents", "hooks")
        for path in (root / folder).glob("*/agents/*.md")
    ]
    for path in sorted(agent_paths):
        meta, _ = frontmatter(path)
        add(
            "agent",
            meta["name"],
            id=f"agent/{path.parents[1].name}/{meta['name']}",
            description=meta["description"],
            path=str(path.relative_to(root)),
            plugin=path.parents[1].name,
            category="Planning & review",
            evidence="Agent definition",
        )

    manifest_paths = [
        path
        for folder in ("plugins", "skills", "agents", "hooks")
        for path in (root / folder).glob("*/.claude-plugin/plugin.json")
    ]
    for path in sorted(manifest_paths):
        manifest = json.loads(path.read_text())
        active, problems = hook_components(path.parents[1], manifest)
        if problems:
            raise ValueError(f"Invalid hook metadata in {path}: {problems}")
        if active:
            add(
                "hook",
                manifest["name"],
                description=f"Event hooks included with {manifest['name']}. Review the configuration before enabling this package.",
                path=str(path.relative_to(root)),
                plugin=manifest["name"],
                category="Automation",
                evidence="Configuration present",
            )

    add(
        "service",
        "agentic-search",
        title="Agentic Search",
        description="Hybrid retrieval across scientific document collections: lexical, vector, graph and metadata search.",
        category="Research & discovery",
        clients=[],
        docs="/docs/agentic-search",
        path="clio-agentic-search",
        evidence="Standalone service",
    )
    ids = {r["id"] for r in records}
    if len(ids) != len(records):
        raise ValueError("Duplicate catalogue ids")
    for record in records:
        if set(record["members"]) - ids:
            raise ValueError(f"Unresolved members in {record['id']}")
    classify_records(records, entries, root)
    for record in records:
        if record["installation"] == "launcher":
            settings = {"command": "clio-kit", "args": ["mcp-server", record["name"]]}
            record["mcpSettings"] = {
                client: server_settings(settings, client) for client in CLIENTS
            }
            record["mcpSettings"]["other"] = settings
    # Feature only available collections; a rename cannot create an undefined card.
    featured = [f"workflow/{name}" for name in WORKFLOWS if f"workflow/{name}" in ids][
        :3
    ]
    return {
        "schema": 2,
        "featured": featured,
        "clientProfiles": {
            client: {"skills": paths[0], "config": paths[1], "key": paths[2]}
            for client, paths in CLIENTS.items()
        }
        | {"other": {"skills": "/path/to/agent/skills", "key": "mcpServers"}},
        "version": marketplace["metadata"]["version"],
        "publishers": list(publishers.values()),
        "items": sorted(records, key=lambda r: r["id"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.root / "clio-kit-website/src/data/catalogue.json"
    rendered = json.dumps(generate(args.root), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not output.is_file() or output.read_text() != rendered:
            raise SystemExit("Website catalogue is stale; regenerate it.")
    else:
        output.write_text(rendered)
        print(f"Generated {output}")
