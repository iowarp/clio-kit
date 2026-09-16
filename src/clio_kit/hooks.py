"""Structural validation of Claude plugin hooks; never execute submitted code.

The native client remains authoritative for optional, version-specific fields.
See https://code.claude.com/docs/en/hooks and /docs/en/plugins-reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HOOK_EVENTS = frozenset(
    "SessionStart Setup UserPromptSubmit UserPromptExpansion PreToolUse "
    "PermissionRequest PermissionDenied PostToolUse PostToolUseFailure PostToolBatch "
    "Notification MessageDisplay SubagentStart SubagentStop TaskCreated TaskCompleted "
    "Stop StopFailure TeammateIdle InstructionsLoaded ConfigChange CwdChanged "
    "DirectoryAdded FileChanged WorktreeCreate WorktreeRemove PreCompact PostCompact "
    "PreModelSwitch PostModelSwitch Elicitation ElicitationResult SessionEnd".split()
)
REQUIRED_FIELDS = {
    "command": ("command",),
    "http": ("url",),
    "prompt": ("prompt",),
    "agent": ("prompt",),
    "mcp_tool": ("server", "tool"),
}


def _event_problems(events: Any, label: str) -> tuple[bool, list[str]]:
    if not isinstance(events, dict):
        return False, [f"{label} must be a hook event object"]
    problems: list[str] = []
    active = False
    for event, groups in events.items():
        where = f"{label}.{event}"
        if event not in HOOK_EVENTS:
            problems.append(f"{where}: unknown hook event")
        if not isinstance(groups, list):
            problems.append(f"{where} must be an array of matcher groups")
            continue
        for index, group in enumerate(groups):
            location = f"{where}[{index}]"
            if not isinstance(group, dict):
                problems.append(f"{location} must be an object")
                continue
            if "matcher" in group and not isinstance(group["matcher"], str):
                problems.append(f"{location}.matcher must be a string")
            handlers = group.get("hooks")
            if not isinstance(handlers, list) or not handlers:
                problems.append(f"{location}.hooks needs a nonempty handler array")
                continue
            active = True
            for number, handler in enumerate(handlers):
                item = f"{location}.hooks[{number}]"
                if not isinstance(handler, dict):
                    problems.append(f"{item} must be an object")
                    continue
                kind = handler.get("type")
                if not isinstance(kind, str) or kind not in REQUIRED_FIELDS:
                    problems.append(f"{item}: unsupported hook type {kind!r}")
                    continue
                for field in REQUIRED_FIELDS[kind]:
                    if (
                        not isinstance(handler.get(field), str)
                        or not handler[field].strip()
                    ):
                        problems.append(f"{item}.{field} needs a nonempty string")
                if "timeout" in handler and (
                    type(handler["timeout"]) not in (int, float)
                    or handler["timeout"] <= 0
                ):
                    problems.append(f"{item}.timeout must be a positive number")
                if "async" in handler and (
                    kind != "command" or not isinstance(handler["async"], bool)
                ):
                    problems.append(f"{item}.async must be a command-hook boolean")
                if "args" in handler and (
                    kind != "command"
                    or not isinstance(handler["args"], list)
                    or not all(isinstance(arg, str) for arg in handler["args"])
                ):
                    problems.append(f"{item}.args must be a command-hook string array")
    return active, problems


def hook_components(
    directory: Path, manifest: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate default/file/inline hooks and report whether any handlers exist."""
    active = False
    problems: list[str] = []
    paths: list[str] = []
    default = directory / "hooks/hooks.json"
    if default.exists() or default.is_symlink():
        paths.append("./hooks/hooks.json")
    value = manifest.get("hooks")
    if isinstance(value, dict):
        active, inline_problems = _event_problems(value, "hooks")
        problems.extend(inline_problems)
    elif isinstance(value, str):
        paths.append(value)
    elif isinstance(value, list):
        for entry in value:
            if isinstance(entry, str):
                paths.append(entry)
            else:
                problems.append("hooks paths must be strings")
    elif value is not None:
        problems.append("hooks must be an event object, path, or array of paths")
    seen: set[Path] = set()
    for name in paths:
        path = (directory / name).resolve()
        if not name.startswith("./") or not path.is_relative_to(directory.resolve()):
            problems.append(f"hooks path {name!r} must stay inside the plugin")
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or "hooks" not in config:
                problems.append(f"{name} needs a top-level hooks object")
                continue
            present, found = _event_problems(config["hooks"], name)
            active |= present
            problems.extend(found)
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: cannot read hook JSON: {exc}")
    return active, problems


def write_hook(directory: Path) -> None:
    """Scaffold a fast, read-only SessionStart command with no external services."""
    hooks = directory / "hooks"
    hooks.mkdir(exist_ok=True)
    (hooks / "session_start.py").write_text(
        '"""Example SessionStart hook: supply context without changing files."""\n'
        "import json\nimport sys\n\n"
        "event = json.load(sys.stdin)\n"
        'if event.get("hook_event_name") == "SessionStart":\n'
        '    print(json.dumps({"hookSpecificOutput": {\n'
        '        "hookEventName": "SessionStart",\n'
        '        "additionalContext": "This plugin provides workflow guidance. Check prerequisites and report actual evidence."\n'
        "    }}))\n"
    )
    (hooks / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": 'python3 "${CLAUDE_PLUGIN_ROOT}/hooks/session_start.py"',
                                    "timeout": 10,
                                }
                            ]
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n"
    )
