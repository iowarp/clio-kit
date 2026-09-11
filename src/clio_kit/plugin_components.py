"""Validate native plugin components and scaffold optional agents/MCP wrappers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def mcp_problems(config: Any, label: str) -> list[str]:
    if not isinstance(config, dict):
        return [f"{label} must contain an object"]
    servers = config.get("mcpServers", config)
    if not isinstance(servers, dict) or not servers:
        return [f"{label} declares no servers"]
    problems = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            problems.append(f"{label} server {name!r} must be an object")
            continue
        if not any(
            isinstance(entry.get(key), str) and entry[key] for key in ("command", "url")
        ):
            problems.append(f"{label} server {name!r} needs a command or a url")
        arguments = entry.get("args", [])
        if not isinstance(arguments, list) or not all(
            isinstance(arg, str) for arg in arguments
        ):
            problems.append(f"{label} server {name!r} args must be a string array")
        environment = entry.get("env", {})
        if not isinstance(environment, dict) or not all(
            isinstance(value, str) for value in environment.values()
        ):
            problems.append(f"{label} server {name!r} env must contain string values")
    return problems


def component_problems(directory: Path, manifest: dict[str, Any]) -> list[str]:
    problems = []
    inline = manifest.get("mcpServers")
    if isinstance(inline, dict):
        problems.extend(mcp_problems(inline, "mcpServers"))
    elif isinstance(inline, str) and inline.startswith("./"):
        target = (directory / inline).resolve()
        if target.is_relative_to(directory.resolve()) and target.is_file():
            try:
                problems.extend(mcp_problems(json.loads(target.read_text()), inline))
            except (ValueError, OSError) as exc:
                problems.append(f"{inline}: {exc}")
    for field in ("skills", "agents", "commands", "hooks", "mcpServers"):
        value = manifest.get(field)
        paths = value if isinstance(value, list) else [value]
        for path in paths:
            if isinstance(path, str) and path.startswith("./"):
                target = (directory / path).resolve()
                if (
                    not target.is_relative_to(directory.resolve())
                    or not target.exists()
                ):
                    problems.append(
                        f"{field} path {path!r} is missing or leaves the plugin"
                    )
    for agent in (directory / "agents").glob("*.md"):
        try:
            text = agent.read_text()
            if not text.startswith("---\n"):
                raise ValueError("needs YAML frontmatter")
            header, sep, body = text[4:].partition("\n---\n")
            data = yaml.safe_load(header)
            if not sep or not isinstance(data, dict) or not body.strip():
                raise ValueError("needs frontmatter and instructions")
            if not all(
                isinstance(data.get(k), str) and data[k].strip()
                for k in ("name", "description")
            ):
                raise ValueError("needs a name and description")
        except (ValueError, yaml.YAMLError) as exc:
            problems.append(f"{agent.name}: {exc}")
    return problems


def write_mcp_wrapper(directory: Path, command: str, args: tuple[str, ...]) -> None:
    (directory / ".mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"server": {"command": command, "args": list(args)}}},
            indent=2,
        )
        + "\n"
    )


def write_agent(directory: Path) -> None:
    agents = directory / "agents"
    agents.mkdir(exist_ok=True)
    (agents / "workflow-reviewer.md").write_text(
        "---\nname: workflow-reviewer\n"
        "description: Review a scientific workflow's assumptions, prerequisites, and evidence before reporting its results.\n"
        "tools: Read, Glob, Grep\n---\n\n"
        "Review the supplied workflow and its recorded outputs. Identify missing inputs, "
        "unverified units, unavailable prerequisites, and claims unsupported by artifacts. "
        "Distinguish observed results from proposals. Do not execute jobs or modify data.\n"
    )
