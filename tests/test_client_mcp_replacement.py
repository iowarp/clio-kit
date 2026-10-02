"""MCP transports replace as a unit and shared owners cannot silently diverge."""

import json

import pytest
import tomli_w

from clio_kit.client_install import (
    CLIENTS,
    install_for_client,
    server_settings,
    tomllib,
)
from clio_kit.install_receipts import uninstall_for_client


def package(root, name, settings):
    directory = root / "plugins" / name
    (directory / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (directory / ".claude-plugin/plugin.json").write_text(
        json.dumps(
            {"name": name, "version": "1.0.0", "description": "Shared MCP fixture"}
        )
    )
    (directory / ".mcp.json").write_text(json.dumps({"mcpServers": {"lab": settings}}))


@pytest.fixture
def publisher(tmp_path):
    root = tmp_path / "publisher"
    (root / ".clio-kit").mkdir(parents=True)
    (root / ".clio-kit/catalogue.json").write_text(
        json.dumps({"schema": 1, "packages": []})
    )
    for name in ("lab-a", "lab-b"):
        package(root, name, {"command": "python", "args": ["old.py"]})
    return root


def read_config(path):
    return (
        tomllib.loads(path.read_text())
        if path.suffix == ".toml"
        else json.loads(path.read_text())
    )


def snapshot(project):
    return {
        str(p.relative_to(project)): p.read_bytes()
        for p in project.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("client", CLIENTS)
def test_replace_transport_and_restore_preexisting_configuration(
    publisher, tmp_path, client
):
    project = tmp_path / "project"
    _, relative, key = CLIENTS[client]
    config = project / relative
    config.parent.mkdir(parents=True)
    original = {
        key: {
            "lab": server_settings(
                {
                    "command": "previous-python",
                    "args": ["old.py", "--flag", "--flag"],
                    "env": {"STALE": "old"},
                },
                client,
            ),
            "unrelated": {"command": "keep"},
        }
    }
    config.write_text(
        tomli_w.dumps(original) if client == "codex" else json.dumps(original)
    )
    settings = {"command": "python", "args": ["new.py", "--new", "--new"]}
    package(publisher, "lab-a", settings)
    install_for_client(publisher, "lab-a", client, project, replace=True)
    expected = server_settings(settings, client)
    assert read_config(config)[key]["lab"] == expected
    # Reinstall must preserve exact argument order/duplicates too.
    install_for_client(publisher, "lab-a", client, project, replace=True)
    assert read_config(config)[key]["lab"] == expected
    uninstall_for_client("lab-a", client, project)
    assert read_config(config) == original


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize(
    "field,value", [("command", "python3"), ("args", ["new.py"]), ("env", {"NEW": "1"})]
)
def test_conflicting_shared_transport_update_is_rejected(
    publisher, tmp_path, client, field, value
):
    project = tmp_path / "project"
    for name in ("lab-a", "lab-b"):
        install_for_client(publisher, name, client, project)
    before = snapshot(project)
    settings = {"command": "python", "args": ["old.py"], field: value}
    package(publisher, "lab-a", settings)
    with pytest.raises(ValueError, match="lab-b.*different MCP configuration: lab"):
        install_for_client(publisher, "lab-a", client, project, replace=True)
    assert snapshot(project) == before
    # The refused update leaves both owners usable; releasing the other owner
    # then permits the update and eventual removal.
    uninstall_for_client("lab-b", client, project)
    install_for_client(publisher, "lab-a", client, project, replace=True)
    _, relative, key = CLIENTS[client]
    assert read_config(project / relative)[key]["lab"] == server_settings(
        settings, client
    )
    uninstall_for_client("lab-a", client, project)
    assert not read_config(project / relative).get(key)


@pytest.mark.parametrize("client", CLIENTS)
def test_new_owner_cannot_replace_shared_transport(publisher, tmp_path, client):
    project = tmp_path / "project"
    install_for_client(publisher, "lab-a", client, project)
    before = snapshot(project)
    package(publisher, "lab-b", {"command": "different"})
    with pytest.raises(ValueError, match="lab-a.*different MCP configuration: lab"):
        install_for_client(publisher, "lab-b", client, project, replace=True)
    assert snapshot(project) == before


def test_user_edited_arguments_block_removal(publisher, tmp_path):
    project = tmp_path / "project"
    install_for_client(publisher, "lab-a", "cursor", project)
    config = project / CLIENTS["cursor"][1]
    data = read_config(config)
    data["mcpServers"]["lab"]["args"] = ["user.py"]
    config.write_text(json.dumps(data))
    before = snapshot(project)
    with pytest.raises(ValueError, match="Installed setting lab was edited"):
        uninstall_for_client("lab-a", "cursor", project)
    assert snapshot(project) == before
