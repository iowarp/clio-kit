"""Native components with explicit host semantics; unsupported parts stay visible."""

from __future__ import annotations

import json
from pathlib import Path
import re


def native_components(directory: Path, manifest: dict) -> dict:
    """Read declarations only. Hooks are executed by the selected client later."""
    result: dict = {"agents": {}, "commands": {}, "hooks": {}}
    for kind in ("agents", "commands"):
        if manifest.get(kind):
            result["custom_" + kind] = True
        for path in sorted((directory / kind).glob("*.md")):
            result[kind][path.stem] = path.read_text()
    for client, name in (
        ("claude-code", "hooks.json"),
        ("codex", "codex.json"),
        ("opencode", "opencode.js"),
    ):
        path = directory / "hooks" / name
        if path.is_file():
            result["hooks"][client] = path.read_text()
    if manifest.get("hooks"):
        result["custom_hooks"] = True
    return result


def expand(value, directory: Path):
    if isinstance(value, str):
        return value.replace("${CLAUDE_PLUGIN_ROOT}", str(directory)).replace(
            "${CLIO_PLUGIN_ROOT}", str(directory)
        )
    if isinstance(value, dict):
        return {key: expand(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, directory) for item in value]
    return value


def plan_native(packages: dict, client: str, servers: dict | None = None) -> dict:
    import tomli_w
    import yaml

    files: dict[str, bytes] = {}
    configuration: dict = {}
    hooks: dict = {}
    unsupported = []
    agents = []
    warnings = []
    seen_agents: set[str] = set()
    seen_commands: set[str] = set()
    for package, record in packages.items():
        native = record["native"]
        directory = record["installed"]
        for kind in ("agents", "commands", "hooks"):
            if native.get("custom_" + kind):
                unsupported.append(f"{package}: custom {kind} paths")
        for name, text in native["agents"].items():
            if name in seen_agents:
                raise ValueError(f"Duplicate native agent: {name}")
            seen_agents.add(name)
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
                raise ValueError(f"Invalid agent name: {name}")
            parts = text.split("---", 2)
            if len(parts) != 3 or not isinstance(
                meta := yaml.safe_load(parts[1]), dict
            ):
                raise ValueError(f"Invalid agent frontmatter: {package}/{name}")
            body = parts[2].strip()
            if client == "claude-code":
                files[f".claude/agents/{name}.md"] = expand(text, directory).encode()
            elif client in {"codex", "opencode"}:
                # Only this permission subset has a reviewed translation.
                tools = {v.strip() for v in str(meta.get("tools", "")).split(",")}
                if (
                    not tools
                    or not tools <= {"Read", "Glob", "Grep"}
                    or set(meta) - {"name", "description", "tools", "model", "effort"}
                ):
                    unsupported.append(f"{package}: agent {name} permissions/options")
                    continue
                if meta.get("model") or meta.get("effort"):
                    warnings.append(
                        f"{name}: uses {client}'s selected model and effort; Claude model overrides are not copied"
                    )
                if client == "codex":
                    from clio_kit.client_install import server_settings

                    role = {
                        "name": name,
                        "description": meta["description"],
                        "developer_instructions": expand(body, directory),
                        "sandbox_mode": "read-only",
                        "approval_policy": "never",
                        "mcp_servers": {
                            # Codex validates transports even for disabled servers.
                            name: {
                                **server_settings(settings, client),
                                "enabled": False,
                            }
                            for name, settings in (servers or {}).items()
                        },
                    }
                    files[f".codex/agents/{name}.toml"] = tomli_w.dumps(role).encode()
                    warnings.append(
                        f"{name}: Codex uses a read-only sandbox; its tool vocabulary differs from Claude's Read/Glob/Grep. If custom-agent discovery is unavailable, use plugin run-agent with --client codex."
                    )
                else:
                    configuration.setdefault("agent", {})[name] = {
                        "description": meta["description"],
                        "mode": "subagent",
                        "prompt": expand(body, directory),
                        "permission": {
                            "*": "deny",
                            **{tool.lower(): "allow" for tool in tools},
                        },
                    }
            else:
                unsupported.append(f"{package}: agents")
                continue
            agents.append(name)
        for name, text in native["commands"].items():
            if name in seen_commands:
                raise ValueError(f"Duplicate native command: {name}")
            seen_commands.add(name)
            if client == "claude-code":
                files[f".claude/commands/{name}.md"] = expand(text, directory).encode()
            else:
                unsupported.append(f"{package}: commands")
        declared = native["hooks"]
        if declared and client not in declared:
            unsupported.append(f"{package}: hooks (no {client} adapter)")
        elif client in {"claude-code", "codex"} and client in declared:
            from clio_kit.hooks import _event_problems, CODEX_HOOK_EVENTS, HOOK_EVENTS

            event_map = json.loads(declared[client]).get("hooks")
            _, problems = _event_problems(
                event_map,
                f"{package}: {client} hooks",
                allowed_events=CODEX_HOOK_EVENTS if client == "codex" else HOOK_EVENTS,
            )
            if problems:
                raise ValueError("; ".join(problems))
            if client == "codex" and any(
                handler["type"] != "command"
                for groups in event_map.values()
                for group in groups
                for handler in group["hooks"]
            ):
                raise ValueError(
                    f"{package}: Codex adapter supports command hooks only"
                )
            if not isinstance(event_map, dict):
                raise ValueError(f"Invalid {client} hooks in {package}")
            for event, groups in expand(event_map, directory).items():
                hooks.setdefault(event, []).extend(groups)
        elif client == "opencode" and client in declared:
            files[f".opencode/plugins/{package}.js"] = (
                f"const CLIO_PLUGIN_ROOT = {json.dumps(str(directory))};\n"
                + declared[client]
            ).encode()
    if client == "codex" and hooks:
        configuration["hooks"] = hooks
        warnings.append(
            "Codex hooks require review and trust through /hooks before they run."
        )
    return {
        "files": files,
        "configuration": configuration,
        "claude_hooks": hooks if client == "claude-code" else {},
        "unsupported": sorted(set(unsupported)),
        "agents": agents,
        "warnings": warnings,
    }


def merge_configuration(current: dict, incoming: dict, *, replace: bool) -> None:
    """Merge named settings, preserving unrelated values and refusing conflicts."""
    for key, value in incoming.items():
        if key not in current or current[key] == value:
            current[key] = value
        elif isinstance(value, dict) and isinstance(current[key], dict):
            merge_configuration(current[key], value, replace=replace)
        elif isinstance(value, list) and isinstance(current[key], list):
            current[key].extend(item for item in value if item not in current[key])
        elif replace:
            current[key] = value
        else:
            raise ValueError(
                f"Conflicting client setting: {key}; review before using --replace"
            )
