"""Native components and ownership survive reinstall, sharing and failed removal."""

import json
from pathlib import Path

import pytest

from clio_kit.client_install import CLIENTS, install_for_client, tomllib
from clio_kit.install_receipts import uninstall_for_client

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("client", ["codex", "claude-code", "opencode"])
def test_full_workflow_has_native_agents_and_hooks_and_uninstalls(tmp_path, client):
    result = install_for_client(ROOT, "clio-dataset-report", client, tmp_path)
    assert len(result["agents"]) == 2
    assert result["not_installed"] == []
    skill_path, config_path, key = CLIENTS[client]
    config = tmp_path / config_path
    before = config.read_bytes()
    install_for_client(ROOT, "clio-dataset-report", client, tmp_path)
    assert config.read_bytes() == before
    if client == "codex":
        data = tomllib.loads(config.read_text())
        assert "PostToolUse" in data["hooks"]
        role = tomllib.loads(
            (tmp_path / ".codex/agents/scientific-evidence-reviewer.toml").read_text()
        )
        assert role["sandbox_mode"] == "read-only"
        assert all(not record["enabled"] for record in role["mcp_servers"].values())
        assert all(record["command"] for record in role["mcp_servers"].values())
    elif client == "opencode":
        assert (tmp_path / ".opencode/plugins/clio-dataset-report.js").is_file()
        assert (
            json.loads(config.read_text())["agent"]["scientific-evidence-reviewer"][
                "permission"
            ]["*"]
            == "deny"
        )
    else:
        assert json.loads((tmp_path / ".claude/settings.json").read_text())["hooks"][
            "PostToolUse"
        ]
    uninstall_for_client("clio-dataset-report", client, tmp_path)
    assert not (tmp_path / skill_path / "dataset-report").exists()
    data = (
        tomllib.loads(config.read_text())
        if client == "codex"
        else json.loads(config.read_text())
    )
    assert not data.get(key)
    assert not data.get("agents", data.get("agent"))
    assert not data.get("hooks")


def test_shared_dependencies_and_unrelated_configuration_survive(tmp_path):
    for package in ("clio-scientific-io", "clio-scientific-io-skills"):
        install_for_client(ROOT, package, "codex", tmp_path)
    config = tmp_path / ".codex/config.toml"
    config.write_text('model = "keep-me"\n' + config.read_text())
    uninstall_for_client("clio-scientific-io", "codex", tmp_path)
    skill = tmp_path / ".agents/skills/dataset-explore/SKILL.md"
    assert skill.exists()
    uninstall_for_client("clio-scientific-io-skills", "codex", tmp_path)
    assert not skill.exists()
    assert tomllib.loads(config.read_text())["model"] == "keep-me"


def test_same_server_two_owners_removed_in_either_order(tmp_path):
    for package in ("clio-hdf5", "clio-scientific-io"):
        install_for_client(ROOT, package, "codex", tmp_path)
    config = tmp_path / ".codex/config.toml"
    uninstall_for_client("clio-hdf5", "codex", tmp_path)
    assert "clio-hdf5" in tomllib.loads(config.read_text())["mcp_servers"]
    uninstall_for_client("clio-scientific-io", "codex", tmp_path)
    assert not tomllib.loads(config.read_text()).get("mcp_servers")


def test_cross_client_shared_skills_have_multiple_owners(tmp_path):
    for client in ("codex", "antigravity"):
        install_for_client(ROOT, "clio-scientific-io", client, tmp_path)
    skill = tmp_path / ".agents/skills/dataset-explore/SKILL.md"
    uninstall_for_client("clio-scientific-io", "codex", tmp_path)
    assert skill.exists()
    uninstall_for_client("clio-scientific-io", "antigravity", tmp_path)
    assert not skill.exists()


def test_edited_skill_aborts_uninstall_without_losing_config(tmp_path):
    install_for_client(ROOT, "clio-scientific-io", "codex", tmp_path)
    config = tmp_path / ".codex/config.toml"
    original = config.read_bytes()
    skill = tmp_path / ".agents/skills/dataset-explore/SKILL.md"
    skill.write_text(skill.read_text() + "\nMy local instruction.\n")
    with pytest.raises(ValueError, match="edited"):
        uninstall_for_client("clio-scientific-io", "codex", tmp_path)
    assert config.read_bytes() == original
    assert skill.exists()


def test_uninstall_rolls_back_if_config_write_fails(tmp_path, monkeypatch):
    install_for_client(ROOT, "clio-scientific-io", "codex", tmp_path)
    config = tmp_path / ".codex/config.toml"
    original = config.read_bytes()
    real_replace = Path.replace

    def fail(source, target):
        if target == config:
            raise PermissionError("failure injection")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(PermissionError, match="injection"):
        uninstall_for_client("clio-scientific-io", "codex", tmp_path)
    assert config.read_bytes() == original
    assert (tmp_path / ".agents/skills/dataset-explore/SKILL.md").exists()


def test_full_to_partial_removes_native_hook_and_agent_settings(tmp_path):
    install_for_client(ROOT, "clio-dataset-report", "claude-code", tmp_path)
    install_for_client(
        ROOT, "clio-dataset-report", "claude-code", tmp_path, components_only=True
    )
    assert not json.loads((tmp_path / ".claude/settings.json").read_text()).get("hooks")
    assert not (tmp_path / ".claude/agents/scientific-evidence-reviewer.md").exists()
    assert (tmp_path / ".claude/skills/dataset-report/SKILL.md").exists()


def test_tampered_receipt_cannot_escape_project(tmp_path):
    from clio_kit.client_install import safe_destination

    for path in ("../../important", "/tmp/important"):
        with pytest.raises(ValueError, match="leaves project"):
            safe_destination(tmp_path, path)


def test_codex_report_hook_consumes_observed_apply_patch_event(tmp_path):
    import subprocess
    import sys

    (tmp_path / "clio-dataset-report.json").write_text("{}")
    event = {
        "cwd": str(tmp_path),
        "tool_name": "apply_patch",
        "tool_input": {
            "command": "*** Begin Patch\n*** Add File: dataset-report.md\n+# Draft\n*** End Patch"
        },
    }
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "plugins/clio-dataset-report/hooks/codex_report.py"),
        ],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=True,
    )
    feedback = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert feedback.startswith("CLIO_DATASET_REPORT_CHECK ")
    assert json.loads(feedback.split(" ", 1)[1]) == {
        "status": "FAIL",
        "reason": "Expected dataset-report schema 1",
    }


@pytest.mark.parametrize("first", ["codex", "opencode"])
def test_codex_opencode_share_skills_and_preserve_other_owner(tmp_path, first):
    second = "opencode" if first == "codex" else "codex"
    for client in (first, second):
        install_for_client(ROOT, "clio-scientific-io", client, tmp_path)
    assert not (tmp_path / ".opencode/skills").exists()
    skill = tmp_path / ".agents/skills/dataset-explore/SKILL.md"
    assert skill.is_file()
    uninstall_for_client("clio-scientific-io", first, tmp_path)
    assert skill.is_file()
    uninstall_for_client("clio-scientific-io", second, tmp_path)
    assert not skill.exists()


def test_opencode_managed_legacy_skills_migrate_transactionally(tmp_path, monkeypatch):
    monkeypatch.setitem(
        CLIENTS, "opencode", (".opencode/skills", "opencode.json", "mcp")
    )
    install_for_client(ROOT, "clio-scientific-io", "opencode", tmp_path)
    legacy = tmp_path / ".opencode/skills/dataset-explore"
    assert legacy.is_dir()
    monkeypatch.setitem(CLIENTS, "opencode", (".agents/skills", "opencode.json", "mcp"))
    install_for_client(ROOT, "clio-scientific-io", "opencode", tmp_path)
    assert not legacy.exists()
    assert (tmp_path / ".agents/skills/dataset-explore/SKILL.md").is_file()
