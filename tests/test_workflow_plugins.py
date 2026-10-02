"""Task composition must not weaken the primary server coverage rule."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import tomli_w

from clio_kit.local_plugins import COMPONENT_ROOTS
from clio_kit.workflow_plugins import tomllib, write_workflow_plugins

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = {"name": "IoWarp Team"}
ENTRIES = [
    {"name": name, "source": f"./plugins/{name}"}
    for name in ("clio-hdf5", "clio-pandas", "clio-plot", "clio-agents")
] + [{"name": "upstream", "source": {"source": "github", "repo": "lab/plugin"}}]
SPEC = {
    "version": "1.0.0",
    "description": "Inspect a dataset and plot a bounded summary.",
    "dependencies": ["clio-hdf5", "clio-pandas", "clio-plot"],
}


def configure(root, workflows):
    (root / "mcp-server-versions.toml").write_text(
        tomli_w.dumps({"workflows": workflows})
    )


def test_overlapping_workflows_reuse_components_and_preserve_existing_files(tmp_path):
    configure(tmp_path, {"clio-inspect-plot": SPEC, "clio-check-plot": SPEC})
    readme = tmp_path / "plugins/clio-inspect-plot/README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("Reviewed task instructions\n")
    entries = write_workflow_plugins(tmp_path, ENTRIES, author=AUTHOR)
    assert [e["name"] for e in entries] == ["clio-inspect-plot", "clio-check-plot"]
    path = readme.parent / ".claude-plugin/plugin.json"
    original = path.read_bytes()
    assert json.loads(original)["dependencies"] == SPEC["dependencies"]
    assert readme.read_text() == "Reviewed task instructions\n"
    assert not (readme.parent / ".mcp.json").exists()
    write_workflow_plugins(tmp_path, ENTRIES, author=AUTHOR)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"dependencies": ["missing"]}, "existing maintained"),
        ({"dependencies": ["upstream"]}, "existing maintained"),
        ({"dependencies": ["clio-task"]}, "existing maintained"),
        ({"dependencies": ["clio-hdf5", "clio-hdf5"]}, "duplicate"),
        ({"dependencies": []}, "nonempty"),
        ({"dependencies": "clio-hdf5"}, "nonempty"),
        ({"version": "latest"}, "semantic version"),
        ({"description": " "}, "description"),
        ({"servers": ["hdf5"]}, "requires only"),
    ],
)
def test_invalid_tasks_fail_before_writing_any_manifests(tmp_path, changes, match):
    configure(tmp_path, {"clio-valid": SPEC, "clio-task": {**SPEC, **changes}})
    with pytest.raises(ValueError, match=match):
        write_workflow_plugins(tmp_path, ENTRIES, author=AUTHOR)
    assert not (tmp_path / "plugins").exists()


@pytest.mark.parametrize("name", ["../escape", "clio-hdf5", "clio-agents", "upstream"])
def test_invalid_or_colliding_names_are_rejected(tmp_path, name):
    configure(tmp_path, {name: SPEC})
    with pytest.raises(ValueError, match="name|collides"):
        write_workflow_plugins(tmp_path, ENTRIES, author=AUTHOR)
    assert not (tmp_path / "plugins").exists()


def test_task_cycles_are_rejected(tmp_path):
    configure(
        tmp_path,
        {
            "clio-one": {**SPEC, "dependencies": ["clio-two"]},
            "clio-two": {**SPEC, "dependencies": ["clio-one"]},
        },
    )
    with pytest.raises(ValueError, match="existing maintained"):
        write_workflow_plugins(tmp_path, ENTRIES, author=AUTHOR)


def test_workflows_are_optional(tmp_path):
    (tmp_path / "mcp-server-versions.toml").write_text("[bundles]\n")
    assert write_workflow_plugins(tmp_path, ENTRIES, author=AUTHOR) == []


def test_cross_bundle_workflow_is_browsable_with_exact_installed_servers(tmp_path):
    for folder in (*COMPONENT_ROOTS, ".claude-plugin"):
        if (ROOT / folder).is_dir():
            shutil.copytree(ROOT / folder, tmp_path / folder)
    # Added to whatever tasks the checkout already defines, so a newly
    # contributed [workflows.*] table cannot collide with this test.
    inventory = tomllib.loads((ROOT / "mcp-server-versions.toml").read_text())
    inventory.setdefault("workflows", {}).update(
        {
            "clio-inspect-plot": SPEC,
            "clio-review-data": {
                **SPEC,
                "dependencies": [
                    "clio-scientific-io",
                    "clio-analysis-skills",
                    "clio-agents",
                ],
            },
        }
    )
    (tmp_path / "mcp-server-versions.toml").write_text(tomli_w.dumps(inventory))
    marketplace_path = tmp_path / ".claude-plugin/marketplace.json"
    marketplace = json.loads(marketplace_path.read_text())
    task_names = set(inventory["workflows"])
    base = [e for e in marketplace["plugins"] if e["name"] not in task_names]
    marketplace["plugins"] = base + write_workflow_plugins(
        tmp_path, base, author=AUTHOR
    )
    marketplace_path.write_text(json.dumps(marketplace))
    spec = importlib.util.spec_from_file_location(
        "task_catalogue", ROOT / "scripts/generate_website_catalogue.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    items = {item["id"]: item for item in module.generate(tmp_path)["items"]}
    task = items["workflow/clio-inspect-plot"]
    assert task["members"] == ["mcp/hdf5", "mcp/pandas", "mcp/plot"]
    assert task["servers"] == SPEC["dependencies"]
    assert "mcp/hdf5" in items["workflow/clio-scientific-io"]["members"]
    assert "mcp/pandas" in items["workflow/clio-analysis"]["members"]
    assert "plugin/clio-inspect-plot" not in items
    review = items["workflow/clio-review-data"]
    assert review["members"] == [
        "workflow/clio-scientific-io",
        "plugin/clio-analysis-skills",
        "plugin/clio-agents",
    ]
    # Referenced skills suggest tools but do not install their MCP servers.
    assert review["servers"] == [
        "clio-adios",
        "clio-compression",
        "clio-hdf5",
        "clio-parquet",
    ]
    path = tmp_path / "plugins/clio-inspect-plot/.claude-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest["dependencies"] = ["clio-web"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Stale workflow"):
        module.generate(tmp_path)
