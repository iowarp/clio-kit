"""Project client adapters preserve existing settings and explicit capability limits."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from clio_kit.client_install import (
    CLIENTS,
    install_for_client,
    server_settings,
    tomllib,
)
from clio_kit.plugins import plugin_group

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("client", CLIENTS)
def test_install_bundle_and_reinstall_preserving_unrelated_settings(tmp_path, client):
    skill_path, config_path, key = CLIENTS[client]
    config = tmp_path / config_path
    config.parent.mkdir(parents=True, exist_ok=True)
    original = (
        '# retain backup of comments\nmodel = "local-model"\n'
        if client == "codex"
        else '{"custom": true}\n'
    )
    config.write_text(original)
    result = install_for_client(ROOT, "clio-scientific-io", client, tmp_path)
    assert len(result["skills"]) == 3
    assert len(result["servers"]) == 4
    assert result["not_installed"] == []
    assert len(list((tmp_path / skill_path).glob("*/SKILL.md"))) == 3
    assert Path(result["backup"]).read_text() == original
    data = (
        tomllib.loads(config.read_text())
        if client == "codex"
        else json.loads(config.read_text())
    )
    assert data["model" if client == "codex" else "custom"] == (
        "local-model" if client == "codex" else True
    )
    assert set(data[key]) == set(result["servers"])
    assert data[key]["clio-hdf5"]["command"] == (
        ["clio-kit", "mcp-server", "hdf5"] if client == "opencode" else "clio-kit"
    )
    before = config.read_bytes()
    install_for_client(ROOT, "clio-scientific-io", client, tmp_path)
    assert config.read_bytes() == before


def test_native_features_require_explicit_partial_install(tmp_path):
    with pytest.raises(ValueError, match="--components-only"):
        install_for_client(ROOT, "clio-dataset-report", "codex", tmp_path)
    assert list(tmp_path.iterdir()) == []
    result = install_for_client(
        ROOT, "clio-dataset-report", "codex", tmp_path, components_only=True
    )
    assert result["not_installed"] == [
        "clio-agents: agents",
        "clio-dataset-report: hooks",
    ]
    assert len(result["servers"]) == 3


def test_dry_run_and_conflicts_never_rewrite_settings(tmp_path):
    result = install_for_client(
        ROOT, "clio-scientific-io", "cursor", tmp_path, dry_run=True
    )
    assert len(result["servers"]) == 4
    assert list(tmp_path.iterdir()) == []
    config = tmp_path / ".cursor/mcp.json"
    config.parent.mkdir()
    original = '{"mcpServers":{"clio-hdf5":{"command":"my-hdf5"}}}'
    config.write_text(original)
    with pytest.raises(ValueError, match="different configuration"):
        install_for_client(ROOT, "clio-scientific-io", "cursor", tmp_path)
    assert config.read_text() == original
    assert not (tmp_path / ".cursor/skills").exists()


def test_linked_destinations_and_jsonc_are_not_overwritten(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (project / ".cursor").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        install_for_client(ROOT, "clio-hdf5", "cursor", project)
    assert list(other.iterdir()) == []
    (project / "opencode.jsonc").write_text("{ /* local comments */ }")
    with pytest.raises(ValueError, match="jsonc"):
        install_for_client(ROOT, "clio-hdf5", "opencode", project)


def test_host_specific_options_are_not_silently_dropped():
    with pytest.raises(ValueError, match="stdio"):
        server_settings({"url": "https://example.test/mcp"}, "codex")
    with pytest.raises(ValueError, match="Unresolved"):
        server_settings({"command": "test", "env": {"TOKEN": "${TOKEN}"}}, "cursor")


def test_cli_exposes_plan_and_external_boundary(tmp_path):
    args = [
        "install",
        "clio-scientific-io",
        "--root",
        str(ROOT),
        "--client",
        "opencode",
        "--project",
        str(tmp_path),
        "--dry-run",
    ]
    result = CliRunner().invoke(plugin_group, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["client"] == "opencode"
    args[1] = "scientific-debugging"
    result = CliRunner().invoke(plugin_group, args)
    assert result.exit_code != 0
    assert "indexed externally" in result.output
