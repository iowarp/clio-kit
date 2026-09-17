"""The public catalogue must reflect installable sources without inventing evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "website_catalogue", ROOT / "scripts/generate_website_catalogue.py"
)
assert spec and spec.loader
catalogue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalogue)


def test_every_installable_marketplace_entry_and_local_skill_is_browsable():
    result = catalogue.generate(ROOT)
    entries = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    represented = {
        item["nativePackage"] for item in result["items"] if "nativePackage" in item
    }
    assert represented == {entry["name"] for entry in entries["plugins"]}
    expected_skills = {
        path.parent.name
        for folder in ("skills", "plugins", "agents", "hooks")
        for path in (ROOT / folder).glob("*/skills/*/SKILL.md")
    }
    assert {
        r["name"] for r in result["items"] if r["kind"] == "skill"
    } == expected_skills
    publishers = {p["id"] for p in result["publishers"]}
    assert all(item["publisher"] in publishers for item in result["items"])


def test_upstream_and_adapted_skills_have_distinct_identity_and_provenance():
    items = {item["id"]: item for item in catalogue.generate(ROOT)["items"]}
    upstream = items["plugin/scientific-debugging"]
    adapted = items["skill/clio-kit-scientific-debugging"]
    assert upstream["publisher"] == "clio-coder"
    assert upstream["origin"] == "Indexed"
    assert upstream["revision"] in upstream["source"]
    assert adapted["publisher"] == "clio-kit"
    assert adapted["origin"] == "Adapted"
    assert adapted["evidence"] == "Scenarios recorded"
    assert all("verified" not in r["evidence"].lower() for r in items.values())


def test_workflow_members_resolve_and_service_is_not_an_mcp():
    items = {item["id"]: item for item in catalogue.generate(ROOT)["items"]}
    workflow = items["workflow/clio-scientific-io"]
    assert "mcp/hdf5" in workflow["members"]
    assert "skill/exploring-an-unfamiliar-dataset" in workflow["members"]
    assert all(member in items for member in workflow["members"])
    assert items["service/agentic-search"]["clients"] == []
    assert "mcp/agentic-search" not in items


def test_product_types_do_not_count_installation_wrappers_as_workflow_plugins():
    items = {item["id"]: item for item in catalogue.generate(ROOT)["items"]}
    assert items["mcp/hdf5"]["kind"] == "mcp"
    assert items["mcp/hdf5"]["installation"] == "launcher"
    assert "plugin/clio-hdf5" not in items
    assert items["workflow/clio-scientific-io"]["kind"] == "plugin"
    assert items["workflow/clio-scientific-io"]["componentTypes"] == ["mcp", "skill"]
    report = items["workflow/clio-dataset-report"]
    assert report["kind"] == "plugin"
    assert report["componentTypes"] == ["agent", "hook", "mcp", "skill"]
    for name in ("clio-skills", "clio-coder-skills", "clio-agents"):
        assert items[f"plugin/{name}"]["kind"] == "collection"
    # Dependencies resolve transitively, not just from the package's own folder.
    assert items["plugin/clio-skills"]["componentTypes"] == ["skill"]
    assert items["plugin/clio-agents"]["componentTypes"] == ["agent"]
    assert items["plugin/scientific-debugging"]["kind"] == "package"
    assert items["plugin/scientific-debugging"]["componentTypes"] == []
    assert (
        items["skill/clio-kit-scientific-debugging"]["installation"] == "portable-skill"
    )


def test_local_mcp_package_never_gets_an_invented_launcher_command(tmp_path):
    package = tmp_path / "plugins/lab-mcp"
    (package / ".claude-plugin").mkdir(parents=True)
    (package / ".claude-plugin/plugin.json").write_text('{"name": "lab-mcp"}')
    (package / ".mcp.json").write_text('{"lab": {"command": "lab-server"}}')
    entry = {"name": "lab-mcp", "source": "./plugins/lab-mcp"}
    records = [
        {
            "id": "plugin/lab-mcp",
            "name": "lab-mcp",
            "kind": "plugin",
            "origin": "Maintained",
            "members": [],
        }
    ]
    catalogue.classify_records(records, {"lab-mcp": entry}, tmp_path)
    assert records[0]["kind"] == "mcp"
    assert records[0]["installation"] == "native-package"
    assert records[0]["nativePackage"] == "lab-mcp"


@pytest.mark.parametrize("config", ["inline", "custom-file", "default-file"])
def test_local_hooks_are_discovered_without_executing_them(tmp_path, config):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin/marketplace.json").write_text(
        json.dumps(
            {
                "metadata": {"version": "1"},
                "plugins": [
                    {
                        "name": "lab-plugin",
                        "description": "Lab plugin",
                        "source": "./plugins/lab-plugin",
                    }
                ],
            }
        )
    )
    (tmp_path / "mcp-server-versions.toml").write_text("[bundles]\n[servers]\n")
    plugin = tmp_path / "plugins/lab-plugin"
    manifest_dir = plugin / ".claude-plugin"
    manifest_dir.mkdir(parents=True)
    marker = tmp_path / "must-not-exist"
    hooks = {
        "SessionStart": [{"hooks": [{"type": "command", "command": f"touch {marker}"}]}]
    }
    manifest = {"name": "lab-plugin"}
    if config == "inline":
        manifest["hooks"] = hooks
    else:
        location = "config/lab.json" if config == "custom-file" else "hooks/hooks.json"
        path = plugin / location
        path.parent.mkdir()
        path.write_text(json.dumps({"hooks": hooks}))
        if config == "custom-file":
            manifest["hooks"] = f"./{location}"
    (manifest_dir / "plugin.json").write_text(json.dumps(manifest))
    result = catalogue.generate(tmp_path)
    assert any(r["id"] == "hook/lab-plugin" for r in result["items"])
    assert not marker.exists()


def test_generated_catalogue_is_current_and_deterministic():
    actual = json.loads((ROOT / "clio-kit-website/src/data/catalogue.json").read_text())
    assert actual == catalogue.generate(ROOT) == catalogue.generate(ROOT)


def test_indexed_source_cannot_be_a_script_url():
    with pytest.raises(ValueError, match="HTTPS"):
        catalogue.source_url({"url": "javascript:alert(1)"})
