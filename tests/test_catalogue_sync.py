"""Folder-only submissions reach both catalogues through the public sync command."""

import json
from pathlib import Path
import shutil

from click.testing import CliRunner

from clio_kit.marketplace_cli import marketplace_group

ROOT = Path(__file__).resolve().parents[1]


def test_sync_discovers_updates_and_removes_folder_without_manual_generators(tmp_path):
    for folder in ("plugins", "skills", "community", ".claude-plugin", "scripts"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    shutil.copy2(ROOT / "mcp-server-versions.toml", tmp_path)
    (tmp_path / "src/clio_kit").mkdir(parents=True)
    package = tmp_path / "hooks/lab-sync"
    (package / ".claude-plugin").mkdir(parents=True)
    (package / "hooks").mkdir()
    manifest = package / ".claude-plugin/plugin.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "lab-sync",
                "version": "1.0.0",
                "description": "Local hook",
                "author": {"name": "Lab"},
            }
        )
    )
    (package / "hooks/hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "false"}]}
                    ]
                }
            }
        )
    )
    runner = CliRunner()
    for present in (True, False):
        if not present:
            shutil.rmtree(package)
        result = runner.invoke(marketplace_group, ["sync", "--root", str(tmp_path)])
        assert result.exit_code == 0, result.output
        native = json.loads((tmp_path / ".claude-plugin/marketplace.json").read_text())
        website = json.loads((tmp_path / "website/src/data/catalogue.json").read_text())
        assert any(e["name"] == "lab-sync" for e in native["plugins"]) is present
        assert any(e["id"] == "hook/lab-sync" for e in website["items"]) is present


def test_sync_requires_checkout(tmp_path):
    result = CliRunner().invoke(marketplace_group, ["sync", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "source checkout" in result.output
