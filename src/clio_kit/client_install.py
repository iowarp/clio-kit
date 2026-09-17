"""Install shared plugin components using each client's project configuration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import tomli_w

from clio_kit.hooks import hook_components
from clio_kit.skill_cli import install_skills
from clio_kit.skills import read_skill_frontmatter

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


# These are project-local paths; installation never changes the global profile.
CLIENTS = {
    "codex": (".agents/skills", ".codex/config.toml", "mcp_servers"),
    "opencode": (".opencode/skills", "opencode.json", "mcp"),
    "cursor": (".cursor/skills", ".cursor/mcp.json", "mcpServers"),
    "antigravity": (".agents/skills", ".agents/mcp_config.json", "mcpServers"),
    "claude-code": (".claude/skills", ".mcp.json", "mcpServers"),
    "vscode": (".github/skills", ".vscode/mcp.json", "servers"),
}


def collect_components(root: Path, name: str) -> dict[str, Any]:
    """Resolve local packages and dependencies without running their code."""
    from clio_kit.local_plugins import discover_local_plugins
    from clio_kit.plugins import validate_plugin

    marketplace = json.loads((root / ".claude-plugin/marketplace.json").read_text())
    entries = marketplace["plugins"]
    # New handwritten folders are usable even before a catalogue build.
    entries = entries + discover_local_plugins(root, entries)
    index = {entry["name"]: entry for entry in entries}
    skills: dict[str, Path] = {}
    servers: dict[str, dict] = {}
    unsupported: list[str] = []
    visited: set[str] = set()

    def visit(package: str, trail: set[str]) -> None:
        if package in trail:
            raise ValueError(f"Cyclic dependency: {package}")
        if package in visited:
            return
        entry = index.get(package)
        if not entry:
            raise ValueError(f"Unknown package: {package}")
        source = entry["source"]
        if not isinstance(source, str):
            raise ValueError(
                f"{package} is indexed externally; use its publisher's client installation route"
            )
        directory = (root / source).resolve()
        if not directory.is_relative_to(root):
            raise ValueError(f"Package leaves checkout: {package}")
        manifest, problems = validate_plugin(directory, allow_reserved=True)
        if problems:
            raise ValueError("; ".join(problems))
        for dependency in manifest.get("dependencies", []):
            visit(dependency, trail | {package})
        for field in ("agents", "commands"):
            if manifest.get(field) or any(directory.glob(f"{field}/*.md")):
                unsupported.append(f"{package}: {field}")
        if hook_components(directory, manifest)[0]:
            unsupported.append(f"{package}: hooks")
        if manifest.get("skills"):
            # Do not silently ignore custom paths or pretend they were copied.
            raise ValueError(
                f"{package}: this installer requires skills/*/SKILL.md layout"
            )
        for path in directory.glob("skills/*/SKILL.md"):
            skill = read_skill_frontmatter(path.parent)["name"]
            if skill in skills and skills[skill] != path.parent:
                raise ValueError(f"Duplicate skill: {skill}")
            skills[skill] = path.parent
        configurations = []
        if (directory / ".mcp.json").is_file():
            configurations.append(json.loads((directory / ".mcp.json").read_text()))
        inline = manifest.get("mcpServers")
        if isinstance(inline, str):
            inline = json.loads((directory / inline).read_text())
        if inline:
            configurations.append(inline)
        for configuration in configurations:
            for server, settings in configuration.get(
                "mcpServers", configuration
            ).items():
                # A source checkout must remain available for local server scripts.
                def expand(value):
                    if isinstance(value, str):
                        return value.replace("${CLAUDE_PLUGIN_ROOT}", str(directory))
                    if isinstance(value, list):
                        return [expand(item) for item in value]
                    if isinstance(value, dict):
                        return {key: expand(item) for key, item in value.items()}
                    return value

                settings = expand(settings)
                if server in servers and servers[server] != settings:
                    raise ValueError(f"Conflicting MCP definitions: {server}")
                servers[server] = settings
        visited.add(package)

    visit(name, set())
    return {"skills": skills, "servers": servers, "unsupported": unsupported}


def server_settings(settings: dict, client: str) -> dict:
    """Convert the shared stdio subset; reject host-specific options explicitly."""
    allowed = {"command", "args", "env", "type"}
    if set(settings) - allowed or settings.get("type", "stdio") != "stdio":
        raise ValueError(
            "Project installation currently supports stdio command/args/env only; "
            "use native client configuration for remote or host-specific MCP options"
        )
    if not isinstance(settings.get("command"), str):
        raise ValueError("MCP command must be a string")
    if "${" in json.dumps(settings):
        raise ValueError("Unresolved MCP variables need client-specific configuration")
    result = {key: value for key, value in settings.items() if key != "type"}
    if client == "opencode":
        result = {
            "type": "local",
            "command": [settings["command"], *settings.get("args", [])],
            "enabled": True,
            "timeout": 120000,
        }
        if settings.get("env"):
            result["environment"] = settings["env"]
    elif client in {"vscode", "claude-code"}:
        result["type"] = "stdio"
    return result


def safe_destination(project: Path, relative: str) -> Path:
    path = project / relative
    for part in (path, *path.parents):
        if part == project:
            break
        if part.is_symlink():
            raise ValueError(f"Refusing linked installation destination: {part}")
    return path


def install_for_client(
    root: Path,
    name: str,
    client: str,
    project: Path,
    *,
    components_only: bool = False,
    replace: bool = False,
    dry_run: bool = False,
) -> dict:
    root, project = root.resolve(), project.resolve()
    components = collect_components(root, name)
    if components["unsupported"] and not components_only:
        raise ValueError(
            "This package also needs native client adapters for "
            + ", ".join(components["unsupported"])
            + ". Use the native Claude package, or explicitly select --components-only "
            "to install just skills and MCPs."
        )
    if not components["skills"] and not components["servers"]:
        raise ValueError("No portable skills or stdio MCPs to install")
    skill_path, config_path, key = CLIENTS[client]
    target = safe_destination(project, skill_path)
    config = safe_destination(project, config_path)
    if client == "opencode" and (project / "opencode.jsonc").exists():
        raise ValueError(
            "opencode.jsonc exists; merge configuration manually to preserve its comments and precedence"
        )
    original = config.read_bytes() if config.exists() else None
    data = (
        (
            tomllib.loads(original.decode())
            if client == "codex"
            else json.loads(original)
        )
        if original is not None
        else {}
    )
    if not isinstance(data, dict) or not isinstance(data.get(key, {}), dict):
        raise ValueError(f"Expected an object/table in {config}: {key}")
    configured = data.setdefault(key, {})
    for server, settings in components["servers"].items():
        converted = server_settings(settings, client)
        if server in configured and configured[server] != converted and not replace:
            raise ValueError(
                f"{server} has different configuration; review before using --replace"
            )
        configured[server] = converted
    result = {
        "client": client,
        "package": name,
        "skills": sorted(components["skills"]),
        "servers": sorted(components["servers"]),
        "skill_directory": str(target),
        "config": str(config) if components["servers"] else None,
        "not_installed": components["unsupported"],
    }
    if dry_run:
        return result
    # Preflight skill conflicts before writing configuration. The skill installer
    # stages complete folders and refuses local changes unless --replace is set.
    if components["skills"]:
        install_skills(components["skills"], target, replace)
    if components["servers"]:
        config.parent.mkdir(parents=True, exist_ok=True)
        rendered = (
            tomli_w.dumps(data)
            if client == "codex"
            else json.dumps(data, indent=2) + "\n"
        ).encode()
        if original != rendered:
            # Keep a recovery copy, including TOML comments lost on serialization.
            if original is not None:
                with tempfile.NamedTemporaryFile(
                    prefix=config.name + ".backup-", dir=config.parent, delete=False
                ) as backup:
                    backup.write(original)
                    result["backup"] = backup.name
            with tempfile.NamedTemporaryFile(
                dir=config.parent, delete=False
            ) as staging:
                staging.write(rendered)
            Path(staging.name).replace(config)
    return result
