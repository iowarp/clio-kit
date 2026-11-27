"""Mistakes a contributor can make that validation used to wave through."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from clio_kit import main
from clio_kit.doctor import prerequisite_checks
from clio_kit.local_plugins import discover_local_plugins
from clio_kit.plugins import plugin_group, validate_plugin
from clio_kit.server_cli import server_group
from clio_kit.skill_cli import skill_records


def scaffold(directory: Path, *options: str) -> Path:
    result = CliRunner().invoke(plugin_group, ["init", str(directory), *options])
    assert result.exit_code == 0, result.output
    return directory


def test_doctor_resolves_commands_the_way_the_servers_do(tmp_path, monkeypatch):
    tool = tmp_path / "lmod"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    declared = {
        "executables": ["absent-clio-tool"],
        "executable-environment": ["CLIO_TEST_COMMAND"],
        "executable-paths": [str(tmp_path / "elsewhere"), "$CLIO_TEST_ROOT/lmod"],
        "environment-files": ["CLIO_TEST_FILE"],
    }
    for name in ("CLIO_TEST_COMMAND", "CLIO_TEST_ROOT", "CLIO_TEST_FILE"):
        monkeypatch.delenv(name, raising=False)

    def available() -> list[bool]:
        return [check["available"] for check in prerequisite_checks(declared)]

    assert available() == [False, False]
    monkeypatch.setenv("CLIO_TEST_ROOT", str(tmp_path))  # a known location
    monkeypatch.setenv("CLIO_TEST_FILE", str(tmp_path / "missing.json"))
    assert available() == [True, False]
    monkeypatch.setenv("CLIO_TEST_COMMAND", str(tmp_path / "missing"))
    monkeypatch.setenv("CLIO_TEST_FILE", str(tool))
    assert available() == [False, True]  # a configured command replaces the search
    monkeypatch.setenv("CLIO_TEST_COMMAND", str(tool))
    assert available() == [True, True]
    tool.chmod(0o644)
    assert available() == [False, True]  # a file is not necessarily executable
    monkeypatch.delenv("CLIO_TEST_COMMAND")
    assert available() == [False, True]  # the fallback must check permissions too


def test_launcher_reports_its_version_and_unknown_contracts_cleanly():
    runner = CliRunner()
    assert runner.invoke(main, ["--version"]).output.startswith("clio-kit, version ")
    banner = runner.invoke(main, []).output
    assert "doctor" in banner and "prompts" not in banner
    result = runner.invoke(main, ["mcp-contract", "no-such-contract"])
    assert result.exit_code == 1
    assert result.output.startswith("Error: unknown MCP user contract")
    assert "valid ids:" in result.output and "Traceback" not in result.output


def test_hook_handler_file_must_ship_with_the_plugin(tmp_path):
    plugin = scaffold(tmp_path / "lab", "--hook")
    assert validate_plugin(plugin)[1] == []
    (plugin / "hooks/session_start.py").unlink()
    assert any("session_start.py" in p for p in validate_plugin(plugin)[1])
    # A system executable or a path only the shell can resolve is not judged.
    config = plugin / "hooks/hooks.json"
    config.write_text(config.read_text().replace("${CLAUDE_PLUGIN_ROOT}", "$HOME"))
    assert validate_plugin(plugin)[1] == []


@pytest.mark.parametrize(
    "handler",
    [
        {
            "command": 'python3 "${CLAUDE_PLUGIN_ROOT}/hook files/check.py" --output "${CLAUDE_PLUGIN_ROOT}/new.json"'
        },
        {"command": r"python3 ${CLAUDE_PLUGIN_ROOT}/hook\ files/check.py"},
        {"command": '"${CLAUDE_PLUGIN_ROOT}/hook files/check.py"'},
        {"command": "python3", "args": ["${CLAUDE_PLUGIN_ROOT}/hook files/check.py"]},
    ],
)
def test_hook_validation_respects_paths_and_ignores_output_arguments(tmp_path, handler):
    from clio_kit.hooks import hook_components

    script = tmp_path / "hook files/check.py"
    script.parent.mkdir()
    script.write_text("raise RuntimeError('validation must not execute this')\n")
    manifest = {
        "hooks": {"SessionStart": [{"hooks": [{"type": "command", **handler}]}]}
    }
    assert hook_components(tmp_path, manifest) == (True, [])
    script.unlink()
    assert any(
        "hook files/check.py" in p for p in hook_components(tmp_path, manifest)[1]
    )


def test_linked_content_is_rejected_by_plugin_validation(tmp_path):
    plugin = scaffold(tmp_path / "lab")
    (plugin / "notes.md").symlink_to(plugin / ".claude-plugin/plugin.json")
    assert any("notes.md is a link" in p for p in validate_plugin(plugin)[1])


def test_submit_refuses_placeholders_and_prints_only_the_entry(tmp_path):
    plugin = scaffold(tmp_path / "lab")
    arguments = ["submit", str(plugin), "--repo", "owner/lab"]
    result = CliRunner().invoke(plugin_group, arguments)
    assert result.exit_code != 0 and "placeholder" in result.output
    path = plugin / ".claude-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest.update(description="Crystal checks.", author={"name": "Lab"})
    path.write_text(json.dumps(manifest))
    result = CliRunner().invoke(plugin_group, arguments)
    assert result.exit_code == 0, result.output
    assert "Add this as" in result.stderr
    import tomllib

    assert tomllib.loads(result.stdout)["source"] == {
        "type": "github",
        "repo": "owner/lab",
    }


def test_two_scaffolds_index_together_and_a_real_duplicate_names_both(tmp_path):
    for name in ("one", "two"):
        scaffold(tmp_path / "plugins" / name)
    assert len(discover_local_plugins(tmp_path, [])) == 2
    copy = tmp_path / "plugins/two/skills/one-workflow"
    (tmp_path / "plugins/two/skills/two-workflow").rename(copy)
    skill = copy / "SKILL.md"
    skill.write_text(
        skill.read_text().replace("name: two-workflow", "name: one-workflow")
    )
    with pytest.raises(ValueError, match="plugins/one and plugins/two; rename"):
        discover_local_plugins(tmp_path, [])


def test_scaffolded_mcp_wrapper_is_named_after_its_plugin(tmp_path):
    plugin = scaffold(tmp_path / "lab", "--mcp-command", "node")
    assert list(json.loads((plugin / ".mcp.json").read_text())["mcpServers"]) == ["lab"]


def test_bundle_folder_decides_the_bundle_when_metadata_is_silent(
    monkeypatch, tmp_path
):
    skill = tmp_path / "skills/clio-lab-skills/skills/probe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: probe\ndescription: Use when probing.\n---\n"
    )
    monkeypatch.setattr(
        "clio_kit.skill_cli._local_skill_inventory", lambda: {"probe": skill}
    )
    assert skill_records()["probe"]["bundle"] == "clio-lab"


def test_server_inspect_reports_descriptor_problems_itself(tmp_path):
    (tmp_path / "clio-server.toml").write_text(
        'name = "crystal"\nruntime = "node"\nentry = "bundle/server.js"\n'
        '[registry]\nregistryType = "cargo"\nidentifier = "x"\nversion = "1.0.0"\n'
    )
    result = CliRunner().invoke(server_group, ["inspect", str(tmp_path)])
    assert result.exit_code == 1
    assert result.output.startswith("Error: registryType must be one of")
    assert "TaskGroup" not in result.output
