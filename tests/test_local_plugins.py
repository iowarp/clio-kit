"""Discover standalone component packages without hand-editing the catalogue."""

import importlib
import json
from pathlib import Path
import shutil

import pytest

from clio_kit.local_plugins import discover_local_plugins
from clio_kit.plugin_components import write_agent
from clio_kit.plugins import plugin_group, validate_plugin
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


def package(root, kind="plugins", name="lab-tools", component="mcp"):
    folder = root / kind / name
    (folder / ".claude-plugin").mkdir(parents=True)
    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": "A standalone component package.",
        "author": {"name": "Test"},
    }
    (folder / ".claude-plugin/plugin.json").write_text(json.dumps(manifest))
    if component == "mcp":
        (folder / ".mcp.json").write_text(
            json.dumps(
                {"mcpServers": {"lab": {"command": "python3", "args": ["server.py"]}}}
            )
        )
    elif component == "agent":
        write_agent(folder)
    elif component == "hook":
        (folder / "hooks").mkdir()
        (folder / "hooks/hooks.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": f"touch {root / 'must-not-run'}",
                                    }
                                ]
                            }
                        ]
                    }
                }
            )
        )
    elif component == "skill":
        skill = folder / "skills/lab-procedure"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: lab-procedure\ndescription: 'Use when inspecting a lab result. Triggers on \"lab result\". Not for publishing.'\n---\nRead the input and report its units.\n"
        )
        (skill / "evals.md").write_text(
            "# Scenarios\nInput with no units: report units unspecified.\n"
        )
    return folder


@pytest.mark.parametrize(
    "kind,component",
    [("plugins", "mcp"), ("skills", "skill"), ("agents", "agent"), ("hooks", "hook")],
)
def test_standalone_component_needs_no_dependencies_or_inventory(
    tmp_path, kind, component
):
    folder = package(tmp_path, kind=kind, component=component)
    before = (folder / ".claude-plugin/plugin.json").read_bytes()
    entries = discover_local_plugins(tmp_path, [])
    assert entries == [
        {
            "name": "lab-tools",
            "source": f"./{kind}/lab-tools",
            "description": "A standalone component package.",
            "version": "1.0.0",
            "category": kind,
        }
    ]
    assert (folder / ".claude-plugin/plugin.json").read_bytes() == before
    assert not (tmp_path / "must-not-run").exists()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"name": "different"}, "folder name"),
        ({"name": ["bad"]}, "folder name"),
        ({"version": "latest"}, "semantic version"),
        ({"version": "1.0.0-01"}, "semantic version"),
        ({"version": "1.0.0-alpha..1"}, "semantic version"),
        ({"description": []}, "description"),
        ({"dependencies": ["missing"]}, "known local"),
        ({"dependencies": ["lab-tools"]}, "Cyclic"),
        ({"dependencies": ["missing", "missing"]}, "duplicate"),
        ({"dependencies": {}}, "list"),
    ],
)
def test_invalid_package_cannot_enter_index(tmp_path, change, match):
    folder = package(tmp_path)
    path = folder / ".claude-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest.update(change)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        discover_local_plugins(tmp_path, [])


def test_duplicate_names_across_roots_and_indexed_sources_fail(tmp_path):
    package(tmp_path)
    package(tmp_path, kind="hooks", component="hook")
    with pytest.raises(ValueError, match="collides"):
        discover_local_plugins(tmp_path, [])
    shutil.rmtree(tmp_path / "hooks")
    with pytest.raises(ValueError, match="collides"):
        discover_local_plugins(
            tmp_path,
            [
                {
                    "name": "lab-tools",
                    "source": {"source": "github", "repo": "lab/tools"},
                }
            ],
        )


def test_local_dependencies_resolve_and_cycles_fail(tmp_path):
    one = package(tmp_path, name="first")
    two = package(tmp_path, kind="agents", name="second", component="agent")

    def depends(folder, name):
        path = folder / ".claude-plugin/plugin.json"
        manifest = json.loads(path.read_text())
        manifest["dependencies"] = [name]
        path.write_text(json.dumps(manifest))

    depends(one, "second")
    assert len(discover_local_plugins(tmp_path, [])) == 2
    depends(two, "first")
    with pytest.raises(ValueError, match="Cyclic"):
        discover_local_plugins(tmp_path, [])


def test_missing_manifest_and_linked_resources_fail(tmp_path):
    folder = tmp_path / "plugins/incomplete"
    folder.mkdir(parents=True)
    with pytest.raises(ValueError, match="plugin.json"):
        discover_local_plugins(tmp_path, [])
    folder.rmdir()
    folder = package(tmp_path)
    (folder / "outside").symlink_to(tmp_path / "sensitive")
    with pytest.raises(ValueError, match="linked"):
        discover_local_plugins(tmp_path, [])


def test_empty_component_folders_are_not_installable_content(tmp_path):
    folder = package(tmp_path, component="empty")
    (folder / "agents").mkdir()
    with pytest.raises(ValueError, match="installing it would do nothing"):
        discover_local_plugins(tmp_path, [])


def test_generated_manifests_are_not_indexed_twice_and_reserved_names_are_explicit(
    tmp_path,
):
    folder = package(tmp_path, name="clio-lab")
    entry = {"name": "clio-lab", "source": "./plugins/clio-lab"}
    assert discover_local_plugins(tmp_path, [entry]) == []
    assert discover_local_plugins(tmp_path, [])[0]["name"] == "clio-lab"
    assert validate_plugin(folder)[1]
    assert validate_plugin(folder, allow_reserved=True)[1] == []
    assert (
        CliRunner()
        .invoke(plugin_group, ["validate", str(folder), "--maintained"])
        .exit_code
        == 0
    )


def test_duplicate_portable_skill_ids_fail(tmp_path):
    package(tmp_path, component="skill")
    package(tmp_path, kind="skills", name="other", component="skill")
    with pytest.raises(ValueError, match="Duplicate portable skill"):
        discover_local_plugins(tmp_path, [])


def test_fast_generator_add_update_delete_and_failed_index_is_unchanged(
    tmp_path, monkeypatch
):
    for kind in ("plugins", "skills", ".claude-plugin"):
        shutil.copytree(ROOT / kind, tmp_path / kind)
    shutil.copy2(
        ROOT / "mcp-server-versions.toml", tmp_path / "mcp-server-versions.toml"
    )
    shutil.copytree(ROOT / "community", tmp_path / "community")
    (tmp_path / "src/clio_kit").mkdir(parents=True)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    generator = importlib.import_module("generate_marketplace")
    folder = package(tmp_path, kind="hooks", component="hook")
    manifest = folder / ".claude-plugin/plugin.json"
    original = manifest.read_bytes()
    output = tmp_path / ".claude-plugin/marketplace.json"
    generator.generate(tmp_path)
    assert (
        next(
            e
            for e in json.loads(output.read_text())["plugins"]
            if e["name"] == "lab-tools"
        )["version"]
        == "1.0.0"
    )
    assert manifest.read_bytes() == original
    data = json.loads(original)
    data["version"] = "1.1.0"
    manifest.write_text(json.dumps(data))
    generator.generate(tmp_path)
    assert (
        next(
            e
            for e in json.loads(output.read_text())["plugins"]
            if e["name"] == "lab-tools"
        )["version"]
        == "1.1.0"
    )
    before = output.read_bytes()
    manifest.write_text("{")
    with pytest.raises(ValueError):
        generator.generate(tmp_path)
    assert output.read_bytes() == before
    shutil.rmtree(folder)
    generator.generate(tmp_path)
    assert not any(
        e["name"] == "lab-tools" for e in json.loads(output.read_text())["plugins"]
    )


def test_collections_preserve_authored_metadata_and_refresh_only_dependencies(tmp_path):
    from clio_kit.marketplace_assets import write_extra_plugins

    folder = package(tmp_path, name="lab-collection", component="agent")
    manifest_path = folder / ".claude-plugin/plugin.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(version="7.2.1", description="Reviewed lab instructions.")
    manifest_path.write_text(json.dumps(manifest))
    inventory = tmp_path / "mcp-server-versions.toml"
    inventory.write_text('[collections.lab-collection]\ncategory="agents"\n')
    before = manifest_path.read_bytes()
    entries = write_extra_plugins(tmp_path, ["lab-skills"])
    assert manifest_path.read_bytes() == before
    assert entries[0]["version"] == "7.2.1"
    assert entries[0]["description"] == "Reviewed lab instructions."
    inventory.write_text(
        '[collections.lab-collection]\ncategory="skills"\nprimary-skills=true\n'
    )
    write_extra_plugins(tmp_path, ["second-skills", "first-skills"])
    assert json.loads(manifest_path.read_text()) == {
        **manifest,
        "dependencies": ["first-skills", "second-skills"],
    }


def test_invalid_collection_does_not_rewrite_earlier_manifests(tmp_path):
    from clio_kit.marketplace_assets import write_extra_plugins

    folder = package(tmp_path, name="lab-collection", component="agent")
    path = folder / ".claude-plugin/plugin.json"
    before = path.read_bytes()
    (tmp_path / "mcp-server-versions.toml").write_text(
        '[collections.lab-collection]\ncategory="skills"\nprimary-skills=true\n'
        '[collections."../escape"]\ncategory="agents"\n'
    )
    with pytest.raises(ValueError, match="Invalid collection"):
        write_extra_plugins(tmp_path, ["replacement-skills"])
    assert path.read_bytes() == before
