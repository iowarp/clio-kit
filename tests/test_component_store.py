"""Partial payload selection, integrity, offline reuse and safe unpacking."""

import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile

import pytest

from clio_kit import component_store as store
from clio_kit.client_install import install_for_client
from clio_kit.release_components import fetch_native_package

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "scripts/package_components.py"))["build_components"]


@pytest.fixture
def published(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nversion="1.0.0"\n')
    package = source / "plugins/lab"
    (package / ".claude-plugin").mkdir(parents=True)
    (package / ".claude-plugin/plugin.json").write_text(
        json.dumps(
            {"name": "lab", "description": "Laboratory workflow", "version": "1.0.0"}
        )
    )
    (package / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "lab": {"command": "clio-kit", "args": ["mcp-server", "hdf5"]}
                }
            }
        )
    )
    for name in ("reading-lab-data", "unrelated-skill"):
        skill = package / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f'---\nname: {name}\ndescription: Use when reading lab data. Triggers on "lab". Not for running experiments.\n---\nInspect the data.\n'
        )
        (skill / "evals.md").write_text(
            "## Case\nInspect input.\nExpected: report fields.\n"
        )
    server = source / "mcp-servers/hdf5"
    server.mkdir(parents=True)
    (server / "clio-server.toml").write_text(
        'name="hdf5"\nruntime="python"\nentry="hdf5-mcp"\n'
    )
    (server / "uv.lock").write_text("version = 1\n")
    output = tmp_path / "assets"
    index = BUILD(source, output)
    monkeypatch.setattr(store, "INDEX_FILE", output / "index.json")
    monkeypatch.setenv("CLIO_KIT_COMPONENT_BASE_URL", output.as_uri())
    monkeypatch.setenv("CLIO_KIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CLIO_KIT_OFFLINE", raising=False)
    return index, output, source


def test_selected_skill_fetches_one_artifact_and_reuses_offline(published, monkeypatch):
    index, output, _ = published
    key = "skill/reading-lab-data"
    target = store.fetch(key)
    assert (target / "SKILL.md").is_file()
    assert not store.artifact_path("server/hdf5").exists()
    assert not store.artifact_path("skill/unrelated-skill").exists()
    monkeypatch.setenv("CLIO_KIT_OFFLINE", "1")
    (output / index["artifacts"][key]["file"]).unlink()
    assert store.fetch(key) == target
    with pytest.raises(ValueError, match="offline"):
        store.fetch("skill/unrelated-skill")
    (target / "SKILL.md").write_text("changed")
    with pytest.raises(ValueError, match="offline"):
        store.fetch(key)


def test_checksum_rejection_leaves_no_install(published):
    index, output, _ = published
    key = "server/hdf5"
    path = output / index["artifacts"][key]["file"]
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="checksum"):
        store.fetch(key)
    assert not store.artifact_path(key).exists()


def test_plugin_dry_run_needs_no_payload_then_installs_selected_skills(
    published, tmp_path
):
    project = tmp_path / "project"
    plan = install_for_client(None, "lab", "codex", project, dry_run=True)
    assert plan["servers"] == ["lab"]
    assert not project.exists()
    assert not (tmp_path / "cache").exists()
    install_for_client(None, "lab", "codex", project)
    assert (project / ".codex/config.toml").is_file()
    assert len(list((project / ".agents/skills").glob("*/SKILL.md"))) == 2
    assert not store.artifact_path("server/hdf5").exists()
    assert not store.artifact_path("package/lab").exists()


def test_native_selection_retains_payload_without_server_download(published, tmp_path):
    destination = tmp_path / "native"
    result = fetch_native_package("lab", destination)
    assert result["packages"] == ["lab"]
    assert (destination / "plugins/lab/.mcp.json").is_file()
    assert (destination / "plugins/lab/skills/reading-lab-data/SKILL.md").is_file()
    assert not store.artifact_path("server/hdf5").exists()
    with pytest.raises(ValueError, match="already exists"):
        fetch_native_package("lab", destination)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", "file"),
        ("C:/escape", "file"),
        ("/escape", "file"),
        ("linked", "link"),
        ("duplicate", "duplicate"),
    ],
)
def test_rejects_unsafe_archives_even_with_matching_archive_hash(
    published, tmp_path, name, kind
):
    index, output, _ = published
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.size = 1
        if kind == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "../escape"
        tar.addfile(member, io.BytesIO(b"x"))
        if kind == "duplicate":
            tar.addfile(member, io.BytesIO(b"x"))
    content = buffer.getvalue()
    record = index["artifacts"]["server/hdf5"]
    record.update(
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        files={
            name: {"size": 1, "sha256": hashlib.sha256(b"x").hexdigest(), "mode": 0o644}
        },
    )
    (output / record["file"]).write_bytes(content)
    with pytest.raises(ValueError, match="Unsafe|duplicate"):
        store.fetch("server/hdf5", index)
    assert not (tmp_path / "escape").exists()


def test_version_update_preserves_reusable_unchanged_components(published, tmp_path):
    first, _, source = published
    (source / "pyproject.toml").write_text('[project]\nversion="1.1.0"\n')
    second = BUILD(source, tmp_path / "second")
    assert first["artifacts"] == second["artifacts"]
    changed = source / "plugins/lab/skills/reading-lab-data/SKILL.md"
    changed.write_text(changed.read_text() + "\nNew guidance.\n")
    third = BUILD(source, tmp_path / "third")
    assert (
        third["artifacts"]["skill/reading-lab-data"]
        != second["artifacts"]["skill/reading-lab-data"]
    )
    assert third["artifacts"]["server/hdf5"] == second["artifacts"]["server/hdf5"]


def test_concurrent_requests_share_one_completed_download(published, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    original = store._download
    requests = []

    def record(*args):
        requests.append(args[0]["file"])
        return original(*args)

    monkeypatch.setattr(store, "_download", record)
    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda _: store.fetch("server/hdf5"), range(4)))
    assert len(set(paths)) == 1
    assert len(requests) == 1


def test_multiple_native_selections_reuse_shared_dependencies(published, tmp_path):
    _, output, source = published
    manifest = source / "plugins/lab/.claude-plugin/plugin.json"
    data = json.loads(manifest.read_text())
    data["dependencies"] = ["tools"]
    manifest.write_text(json.dumps(data))
    tools = source / "plugins/tools/.claude-plugin"
    tools.mkdir(parents=True)
    (tools / "plugin.json").write_text(
        json.dumps({"name": "tools", "version": "1.0.0", "description": "Tools"})
    )
    BUILD(source, output)
    result = fetch_native_package(("lab", "tools"), tmp_path / "multi")
    assert result["packages"] == ["tools", "lab"]
    assert (
        json.loads(
            (tmp_path / "multi/plugins/tools/.claude-plugin/plugin.json").read_text()
        )["name"]
        == "tools"
    )


def test_release_script_paths_preserve_windows_separators(published, monkeypatch):
    from pathlib import PureWindowsPath
    from clio_kit import release_components as release

    index, output, _ = published
    index["packages"]["lab"]["servers"] = {
        "lab": {"command": "python", "args": ["${CLAUDE_PLUGIN_ROOT}/server.py"]}
    }
    (output / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(
        release, "artifact_path", lambda *_: PureWindowsPath("C:/Users/lab/cache")
    )
    plan = release.release_components("lab")
    assert plan["servers"]["lab"]["args"] == ["C:\\Users\\lab\\cache/server.py"]
    assert "package/lab" in plan["artifacts"]
