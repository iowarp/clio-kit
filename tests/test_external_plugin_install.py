"""Publisher revision locks, safe extraction and transactional project installation."""

import io
import json
import subprocess
import tarfile

import pytest
from click.testing import CliRunner

from clio_kit.client_install import install_for_client
from clio_kit.external_plugins import unpack_package
from clio_kit.plugins import plugin_group, validate_plugin


def git(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_external_revision_is_locked_until_explicit_update(tmp_path, monkeypatch):
    publisher = tmp_path / "publisher"
    skill = publisher / "skills/lab-read"
    skill.mkdir(parents=True)
    (publisher / ".claude-plugin").mkdir()
    (publisher / ".claude-plugin/plugin.json").write_text(
        json.dumps({"name": "lab", "description": "Read laboratory results"})
    )
    # External Agent Skills need valid metadata, not our editorial conventions.
    original = "---\nname: lab-read\ndescription: Inspect laboratory output.\n---\nReport the units.\n"
    (skill / "SKILL.md").write_text(original)
    git(publisher, "init")
    git(publisher, "add", ".")
    git(
        publisher,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "first",
    )
    root = tmp_path / "catalogue"
    (root / ".clio-kit").mkdir(parents=True)
    (root / ".clio-kit/catalogue.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "name": "science",
                "packages": [
                    {
                        "name": "lab",
                        "source": {"source": "url", "url": publisher.as_uri()},
                    }
                ],
            }
        )
    )
    project = tmp_path / "project"
    monkeypatch.chdir(root)
    preview = CliRunner().invoke(
        plugin_group,
        ["install", "lab", "--client", "codex", "--project", str(project), "--dry-run"],
    )
    assert preview.exit_code == 0, preview.output
    assert not project.exists()
    first = install_for_client(root, "lab", "codex", project)
    revision = first["sources"]["lab"]["revision"]
    target = project / ".agents/skills/lab-read/SKILL.md"
    assert target.read_text() == original
    (skill / "SKILL.md").write_text(original + "Check the sample size.\n")
    git(publisher, "add", ".")
    git(
        publisher,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "second",
    )
    monkeypatch.setenv("CLIO_KIT_OFFLINE", "1")
    assert (
        install_for_client(root, "lab", "codex", project)["sources"]["lab"]["revision"]
        == revision
    )
    assert target.read_text() == original
    monkeypatch.delenv("CLIO_KIT_OFFLINE")
    updated = install_for_client(
        root, "lab", "codex", project, update=True, replace=True
    )
    assert updated["sources"]["lab"]["revision"] != revision
    assert "sample size" in target.read_text()
    payload = project / ".clio-kit/packages/lab" / updated["sources"]["lab"]["sha256"]
    (payload / "skills/lab-read/SKILL.md").write_text("tampered")
    with pytest.raises(ValueError, match="Modified publisher"):
        install_for_client(root, "lab", "codex", project)


@pytest.mark.parametrize(
    "client", ["codex", "claude-code", "opencode", "cursor", "antigravity", "vscode"]
)
def test_update_removes_last_server_without_removing_skills(tmp_path, client):
    from clio_kit.client_install import CLIENTS, tomllib

    root = tmp_path / "catalogue"
    package = root / "plugins/lab"
    (package / ".claude-plugin").mkdir(parents=True)
    (package / ".claude-plugin/plugin.json").write_text(
        '{"name":"lab","description":"Laboratory tools"}'
    )
    # Use an existing, validated skill so this tests lifecycle, not authoring rules.
    import shutil
    from pathlib import Path

    shutil.copytree(
        Path(__file__).resolve().parents[1]
        / "skills/clio-scientific-io-skills/skills/dataset-explore",
        package / "skills/dataset-explore",
    )
    mcp = package / ".mcp.json"
    mcp.write_text('{"mcpServers":{"lab":{"command":"python3","args":["server.py"]}}}')
    (root / ".clio-kit").mkdir()
    (root / ".clio-kit/catalogue.json").write_text(
        '{"schema":1,"name":"lab","packages":[{"name":"lab","source":"./plugins/lab"}]}'
    )
    project = tmp_path / "project"
    install_for_client(root, "lab", client, project)
    skill_path, config_path, key = CLIENTS[client]
    mcp.unlink()
    install_for_client(root, "lab", client, project, update=True, replace=True)
    config = project / config_path
    data = (
        tomllib.loads(config.read_text())
        if client == "codex"
        else json.loads(config.read_text())
    )
    assert not data.get(key), "Removed MCP must not remain configured after update"
    assert (project / skill_path / "dataset-explore/SKILL.md").exists()


@pytest.mark.parametrize(
    "name,kind",
    [
        ("package/../../outside", tarfile.REGTYPE),
        ("/outside", tarfile.REGTYPE),
        ("package/C:/outside", tarfile.REGTYPE),
        ("package/link", tarfile.SYMTYPE),
        ("package/device", tarfile.CHRTYPE),
        ("package", tarfile.REGTYPE),
    ],
)
def test_unsafe_npm_members_are_rejected(tmp_path, name, kind):
    archive = tmp_path / "package.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "/outside"
        stream.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match="Unsafe"):
        unpack_package(archive, tmp_path / "extracted")


def test_codex_only_hook_package_and_explicit_agent_invocation(tmp_path, monkeypatch):
    package = tmp_path / "hook-only"
    (package / ".claude-plugin").mkdir(parents=True)
    (package / ".claude-plugin/plugin.json").write_text(
        '{"name":"lab-hooks","description":"Read-only session guidance"}'
    )
    (package / "hooks").mkdir()
    (package / "hooks/codex.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "python3 -V"}]}
                    ]
                }
            }
        )
    )
    assert validate_plugin(package)[1] == []
    project = tmp_path / "project"
    role = project / ".codex/agents/reviewer.toml"
    role.parent.mkdir(parents=True)
    role.write_text(
        'name="reviewer"\nsandbox_mode="read-only"\ndeveloper_instructions="Inspect supplied scientific evidence."\n[mcp_servers.lab]\nenabled=false\n'
    )
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("clio_kit.client_agent.subprocess.run", run)
    result = CliRunner().invoke(
        plugin_group,
        [
            "run-agent",
            "reviewer",
            "--client",
            "codex",
            "--project",
            str(project),
            "--prompt",
            "Compare supplied results.",
        ],
    )
    assert result.exit_code == 0, result.output
    args, options = calls[0]
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert 'developer_instructions="Inspect supplied scientific evidence."' in args
    assert 'mcp_servers."lab".enabled=false' in args
    assert options["input"] == "Compare supplied results."
