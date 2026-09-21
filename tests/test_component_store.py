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


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "owner", "artifact"])
def test_release_rejects_ambiguous_or_missing_skills_before_download(published, fault):
    from clio_kit.release_components import release_components

    index, output, _ = published
    package = index["packages"]["lab"]
    if fault == "duplicate":
        package["skills"].append(package["skills"][0])
    elif fault == "unknown":
        package["skills"].append("missing")
    elif fault == "owner":
        index["skills"]["reading-lab-data"]["package"] = "other"
    else:
        index["skills"]["reading-lab-data"]["artifact"] = "missing"
    (output / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="skill"):
        release_components("lab")
    assert not store.artifact_path("server/hdf5", index).exists()


def test_release_and_checkout_resolve_same_skill_and_server_selection(published):
    from clio_kit.client_install import collect_components
    from clio_kit.release_components import release_components

    _, _, source = published
    (source / ".claude-plugin").mkdir()
    (source / ".claude-plugin/marketplace.json").write_text(
        json.dumps({"plugins": [{"name": "lab", "source": "./plugins/lab"}]})
    )
    checkout = collect_components(source, "lab")
    released = release_components("lab")
    assert checkout["skills"].keys() == released["skills"].keys()
    assert checkout["servers"] == released["servers"]
    assert checkout["unsupported"] == released["unsupported"]


def test_failed_install_rolls_back_project_pins_with_client_files(
    published, tmp_path, monkeypatch
):
    from clio_kit.component_cache import root_path

    _, output, source = published
    config = source / "plugins/lab/.mcp.json"
    config.write_text(
        json.dumps(
            {"lab": {"command": "python", "args": ["${CLAUDE_PLUGIN_ROOT}/server.py"]}}
        )
    )
    (config.parent / "server.py").write_text("# fixture\n")
    BUILD(source, output)
    project = tmp_path / "project"
    original_replace = Path.replace

    def fail_config(path, target):
        if Path(target) == project / ".codex/config.toml":
            raise OSError("injected config failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_config)
    with pytest.raises(OSError, match="injected"):
        install_for_client(None, "lab", "codex", project)
    assert not list((root_path() / ".references").glob("*.json"))
    assert not list(project.rglob("SKILL.md"))
    assert not (project / ".codex/config.toml").exists()


@pytest.mark.parametrize("reference", ["absolute", "parent", "linked"])
def test_release_build_rejects_mcp_configuration_outside_package(
    published, tmp_path, reference
):
    _, _, source = published
    package = source / "plugins/lab"
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"outside": {"command": "private-tool"}}))
    manifest_path = package / ".claude-plugin/plugin.json"
    manifest = json.loads(manifest_path.read_text())
    if reference == "absolute":
        manifest["mcpServers"] = str(outside)
    elif reference == "parent":
        manifest["mcpServers"] = "../../../outside.json"
    else:
        (package / "linked.json").symlink_to(outside)
        manifest["mcpServers"] = "./linked.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside|Linked|linked"):
        BUILD(source, tmp_path / "invalid-build")


def test_release_build_rejects_duplicate_server_identity(published, tmp_path):
    import shutil

    _, _, source = published
    shutil.copytree(source / "mcp-servers/hdf5", source / "mcp-servers/duplicate")
    with pytest.raises(ValueError, match="Duplicate server"):
        BUILD(source, tmp_path / "invalid-build")


def test_released_prompt_selects_markdown_instead_of_bundled_license(
    published, tmp_path, monkeypatch
):
    from click.testing import CliRunner
    from clio_kit import prompts_cli

    _, output, source = published
    (source / "LICENSE").write_text("License text, not the prompt")
    prompt_dir = source / "prompts/testing"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "review.md").write_text("Review the observed test results.")
    BUILD(source, output)
    monkeypatch.setattr(prompts_cli, "get_prompts_path", lambda: tmp_path / "absent")
    result = CliRunner().invoke(prompts_cli.prompt, ["testing/review"])
    assert result.exit_code == 0, result.output
    assert "Review the observed test results." in result.output
    assert "License text" not in result.output


def test_cached_prompt_content_is_verified_on_every_use(
    published, tmp_path, monkeypatch
):
    from click.testing import CliRunner
    from clio_kit import prompts_cli

    _, output, source = published
    (source / "prompts").mkdir()
    (source / "prompts/review.md").write_text("Reviewed instructions")
    BUILD(source, output)
    monkeypatch.setattr(prompts_cli, "get_prompts_path", lambda: tmp_path / "absent")
    runner = CliRunner()
    assert runner.invoke(prompts_cli.prompt, ["review"]).exit_code == 0
    (store.artifact_path("prompt/review") / "review.md").write_text("Unreviewed change")
    result = runner.invoke(prompts_cli.prompt, ["review"])
    assert result.exit_code != 0 and "damaged" in result.output
    assert "Unreviewed change" not in result.output
