"""Hook-only plugins must validate honestly without executing their commands."""

import json
import subprocess
import sys

import pytest
from click.testing import CliRunner

from clio_kit.plugins import plugin_group, validate_plugin


def plugin(tmp_path, events=None, layout="default"):
    root = tmp_path / "plugin"
    (root / ".claude-plugin").mkdir(parents=True)
    manifest = {"name": "hook-test", "description": "Hook test"}
    if layout == "inline":
        manifest["hooks"] = events
    elif events is not None:
        target = "hooks/hooks.json" if layout == "default" else "config/events.json"
        p = root / target
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"hooks": events}))
        if layout == "custom":
            manifest["hooks"] = "./" + target
        elif layout == "array":
            manifest["hooks"] = ["./" + target]
    (root / ".claude-plugin/plugin.json").write_text(json.dumps(manifest))
    return root


def events(handler=None):
    return {
        "SessionStart": [
            {"hooks": [handler or {"type": "command", "command": "echo ready"}]}
        ]
    }


@pytest.mark.parametrize("layout", ["default", "inline", "custom", "array"])
def test_hook_only_layouts_and_validation_never_executes(tmp_path, layout):
    sentinel = tmp_path / "must-not-exist"
    root = plugin(
        tmp_path, events({"type": "command", "command": f"touch {sentinel}"}), layout
    )
    assert validate_plugin(root)[1] == []
    assert not sentinel.exists()


@pytest.mark.parametrize("layout", ["default", "custom", "array"])
def test_malformed_hook_files_are_rejected(tmp_path, layout):
    root = plugin(tmp_path, events(), layout)
    p = root / ("hooks/hooks.json" if layout == "default" else "config/events.json")
    p.write_text("{broken")
    assert any("cannot read hook JSON" in p for p in validate_plugin(root)[1])


@pytest.mark.parametrize(
    "event_map",
    [
        {"FakeEvent": [{"hooks": [{"type": "command", "command": "true"}]}]},
        {"SessionStart": {}},
        {"SessionStart": [False]},
        {
            "SessionStart": [
                {"matcher": 7, "hooks": [{"type": "command", "command": "true"}]}
            ]
        },
        {"SessionStart": [{"hooks": []}]},
        {"SessionStart": [{"hooks": ["bad"]}]},
        events({"type": "command"}),
        events({"type": "command", "command": " "}),
        events({"type": "unknown"}),
        events({"type": []}),
        events({"type": "command", "command": "true", "timeout": False}),
        events({"type": "command", "command": "true", "timeout": -1}),
        events({"type": "command", "command": "true", "args": [4]}),
        events({"type": "command", "command": "true", "async": "true"}),
        events({"type": "http"}),
        events({"type": "prompt"}),
        events({"type": "agent"}),
        events({"type": "mcp_tool", "server": "s"}),
    ],
)
def test_broken_hook_definitions_are_rejected(tmp_path, event_map):
    assert validate_plugin(plugin(tmp_path, event_map))[1]


@pytest.mark.parametrize(
    "handler",
    [
        {
            "type": "command",
            "command": "python3",
            "args": ["hook.py"],
            "async": False,
            "timeout": 1,
        },
        {"type": "http", "url": "http://localhost:8080/"},
        {"type": "prompt", "prompt": "Check $ARGUMENTS"},
        {"type": "agent", "prompt": "Check $ARGUMENTS"},
        {"type": "mcp_tool", "server": "plugin:lab:numerics", "tool": "check"},
    ],
)
def test_handler_required_fields(tmp_path, handler):
    assert (
        validate_plugin(plugin(tmp_path, {"PreToolUse": [{"hooks": [handler]}]}))[1]
        == []
    )


def test_empty_hooks_directory_is_not_a_component(tmp_path):
    root = plugin(tmp_path)
    (root / "hooks").mkdir()
    assert any("do nothing" in p for p in validate_plugin(root)[1])


@pytest.mark.parametrize("layout", ["default", "custom"])
def test_linked_external_hook_file_is_rejected(tmp_path, layout):
    root = plugin(tmp_path, events(), layout)
    config = root / (
        "hooks/hooks.json" if layout == "default" else "config/events.json"
    )
    outside = tmp_path / "outside.json"
    config.rename(outside)
    config.symlink_to(outside)
    assert any("inside the plugin" in p for p in validate_plugin(root)[1])


def test_optional_hook_scaffold_runs_and_does_not_modify_project(tmp_path):
    root = tmp_path / "lab"
    result = CliRunner().invoke(plugin_group, ["init", str(root), "--hook"])
    assert result.exit_code == 0, result.output
    assert validate_plugin(root)[1] == []
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    run = subprocess.run(
        [sys.executable, str(root / "hooks/session_start.py")],
        input=json.dumps({"hook_event_name": "SessionStart"}),
        text=True,
        capture_output=True,
        check=True,
        cwd=tmp_path,
    )
    output = json.loads(run.stdout)["hookSpecificOutput"]
    assert output["hookEventName"] == "SessionStart" and output["additionalContext"]
    assert before == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
