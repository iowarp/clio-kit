"""Regressions from actual installation and contribution audit failures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from clio_kit.community import read_community_entries
from clio_kit.federation import refresh_marketplace
from clio_kit.plugins import build_community_entry, plugin_group, validate_plugin
from clio_kit.registry import registry_package
from clio_kit.skills import read_skill_frontmatter


def test_contribution_roundtrips_quotes_newlines_and_unicode(tmp_path: Path) -> None:
    manifest = {
        "name": "crystal",
        "description": 'Tools for "crystals".\nPath C:\\data; λ',
        "author": {"name": 'A "Lab"'},
        "keywords": ['a"b', "x\ny"],
    }
    entries = tmp_path / "community" / "entries"
    entries.mkdir(parents=True)
    (entries / "crystal.toml").write_text(
        build_community_entry(manifest, "lab/crystal")
    )
    result = read_community_entries(tmp_path)[0]
    assert result["description"] == manifest["description"]
    assert result["keywords"] == manifest["keywords"]
    assert result["metadata"]["maintainer"] == manifest["author"]["name"]


@pytest.mark.parametrize("style", [">", "|"])
def test_real_yaml_multiline_description(tmp_path: Path, style: str) -> None:
    skill = tmp_path / "crystal"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        f'---\nname: crystal\ndescription: {style}\n  Use when examining crystals.\n  Triggers on "crystals".\nclio-kit:\n  eval-status: eval-run\n---\nBody.\n'
    )
    fields = read_skill_frontmatter(skill)
    assert 'Triggers on "crystals".' in fields["description"]
    assert fields["eval-status"] == "eval-run"


def test_scaffold_can_wrap_a_real_command_and_include_an_agent(tmp_path: Path) -> None:
    directory = tmp_path / "crystal"
    result = CliRunner().invoke(
        plugin_group,
        [
            "init",
            str(directory),
            "--agent",
            "--mcp-command",
            "node",
            "--mcp-arg",
            "${CLAUDE_PLUGIN_ROOT}/server.js",
        ],
    )
    assert result.exit_code == 0, result.output
    assert validate_plugin(directory)[1] == []
    config = json.loads((directory / ".mcp.json").read_text())
    assert config["mcpServers"]["server"]["args"] == ["${CLAUDE_PLUGIN_ROOT}/server.js"]
    assert (directory / "agents" / "workflow-reviewer.md").is_file()


def test_missing_explicit_component_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "crystal"
    assert CliRunner().invoke(plugin_group, ["init", str(directory)]).exit_code == 0
    manifest_path = directory / ".claude-plugin" / "plugin.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["agents"] = "./missing"
    manifest_path.write_text(json.dumps(manifest))
    assert any("missing" in problem for problem in validate_plugin(directory)[1])


def test_inline_mcp_component_is_validated(tmp_path: Path) -> None:
    manifest = tmp_path / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir()
    data = {
        "name": "crystal",
        "description": "Crystal tools",
        "mcpServers": {"crystal": {"command": "node", "args": ["server.js"]}},
    }
    manifest.write_text(json.dumps(data))
    assert validate_plugin(tmp_path)[1] == []
    data["mcpServers"]["crystal"]["args"] = "server.js"
    manifest.write_text(json.dumps(data))
    assert any("string array" in problem for problem in validate_plugin(tmp_path)[1])


def _git(directory: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def _publish(directory: Path, plugins: list[str]) -> str:
    for name in plugins:
        manifest = directory / "plugins" / name / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps({"name": name, "description": name, "version": "1.0.0"})
        )
        (manifest.parent.parent / "agents").mkdir(exist_ok=True)
        (manifest.parent.parent / "agents" / "review.md").write_text(
            "---\nname: review\ndescription: Review data\n---\nReview the provided data.\n"
        )
    index = directory / ".claude-plugin" / "marketplace.json"
    index.parent.mkdir(exist_ok=True)
    index.write_text(
        json.dumps(
            {
                "name": "lab",
                "owner": {"name": "Lab"},
                "plugins": [
                    {"name": name, "source": f"./plugins/{name}"} for name in plugins
                ],
            }
        )
    )
    _git(directory, "add", ".")
    _git(
        directory,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "Update catalogue",
    )
    return _git(directory, "rev-parse", "HEAD")


def test_federation_fetch_update_removal_and_atomic_failure(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q")
    revision = _publish(upstream, ["crystal"])
    root = tmp_path / "kit"
    entries = root / "community" / "entries"
    entries.mkdir(parents=True)
    (entries / "lab.toml").write_text(
        f'name="lab"\nkind="marketplace"\ndescription="Lab"\n[source]\ntype="url"\nurl="{upstream.as_uri()}"\n'
    )
    index = root / ".claude-plugin" / "marketplace.json"
    index.parent.mkdir()
    index.write_text(
        json.dumps({"name": "clio-kit", "owner": {"name": "Kit"}, "plugins": []})
    )
    lock = refresh_marketplace(root)
    plugin = json.loads(index.read_text())["plugins"][0]
    assert lock["imported_names"] == ["crystal"]
    assert plugin["source"]["sha"] == revision
    assert plugin["source"]["path"] == "plugins/crystal"
    _publish(upstream, ["wave"])
    refresh_marketplace(root)
    assert [p["name"] for p in json.loads(index.read_text())["plugins"]] == ["wave"]
    before = index.read_bytes()
    lock_path = index.with_name("federation.lock.json")
    before_lock = lock_path.read_bytes()
    _publish(upstream, ["clio-hpc"])
    with pytest.raises(ValueError, match="reserved"):
        refresh_marketplace(root)
    assert index.read_bytes() == before
    assert lock_path.read_bytes() == before_lock
    (entries / "lab.toml").unlink()
    refresh_marketplace(root)
    assert json.loads(index.read_text())["plugins"] == []


@pytest.mark.parametrize("kind", ["npm", "oci", "pypi", "nuget", "mcpb"])
def test_registry_coordinate_does_not_depend_on_implementation_language(
    kind: str,
) -> None:
    result = registry_package(
        {"registryType": kind, "identifier": "lab/crystal", "version": "1.0.0"}
    )
    assert result["registryType"] == kind
    assert result["transport"] == {"type": "stdio"}


def test_registry_rejects_invented_go_package_type() -> None:
    with pytest.raises(ValueError, match="registryType"):
        registry_package(
            {"registryType": "go", "identifier": "lab/crystal", "version": "1.0.0"}
        )


def test_nested_metadata_cannot_replace_skill_identity(tmp_path: Path) -> None:
    skill = tmp_path / "crystal"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: crystal\ndescription: Use when examining crystals.\n"
        "clio-kit:\n  name: different\n  description: Hidden override\n"
        "  eval-status: scenarios-recorded\n---\nInspect the crystal.\n"
    )
    fields = read_skill_frontmatter(skill)
    assert fields["name"] == "crystal"
    assert fields["description"] == "Use when examining crystals."
    assert fields["eval-status"] == "scenarios-recorded"


@pytest.mark.parametrize(
    "source", [{"source": "github"}, {"source": "npm", "package": 3}]
)
def test_federation_rejects_malformed_remote_source(tmp_path, source):
    from clio_kit.federation import compile_catalogue

    with pytest.raises(ValueError, match="External source needs"):
        compile_catalogue(
            {"name": "lab", "plugins": [{"name": "crystal", "source": source}]},
            url="https://example.invalid/lab.git",
            revision="a" * 40,
            checkout=tmp_path,
        )
