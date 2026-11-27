"""Reclaim only superseded payloads whose consumers no longer reference them."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from clio_kit import component_cache as cache
from clio_kit import component_store as store


@pytest.fixture
def versions(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_KIT_CACHE_DIR", str(tmp_path / "cache"))
    paths = []
    for number in range(1, 6):
        directory = cache.root_path() / f"{number:064x}"
        directory.mkdir(parents=True)
        (directory / "payload").write_text(f"Version {number}")
        (directory / ".component.json").write_text(
            json.dumps(
                {
                    "key": "package/lab",
                    "digest": directory.name,
                    "used": number,
                    "legacy": number == 1,
                }
            )
        )
        paths.append(directory)
    return paths


def test_gc_preserves_current_catalogues_project_references_and_legacy(
    versions, tmp_path, monkeypatch
):
    first, second, third, fourth, fifth = versions
    index = tmp_path / "index.json"
    data = {"schema": 1, "artifacts": {"package/lab": {"sha256": second.name}}}
    index.write_text(json.dumps(data))
    cache._write(cache.root_path() / ".catalogues/current.json", {"path": str(index)})
    monkeypatch.setattr(store, "INDEX_FILE", index)
    project = tmp_path / "config.json"
    project.write_text(json.dumps({"command": str(second / "payload")}))
    cache.register_project(project, ["package/lab"])
    # Upgrade the installed launcher, keeping an old project's direct cache path.
    data["artifacts"]["package/lab"]["sha256"] = third.name
    index.write_text(json.dumps(data))
    dry = cache.prune_component_cache(keep=1)
    assert [r["digest"] for r in dry["removed"]] == [fourth.name]
    assert all(path.exists() for path in versions)
    applied = cache.prune_component_cache(keep=1, dry_run=False)
    assert applied["removed"] == dry["removed"]
    assert not fourth.exists()
    assert all(path.exists() for path in (first, second, third, fifth))
    # Deletion or indirection cannot silently release a pin.
    project.unlink()
    assert second.name in cache._protected(cache.root_path())
    cache.forget_project(project, dry_run=False)
    assert second.name in {
        r["digest"] for r in cache.prune_component_cache(keep=1)["removed"]
    }


def test_gc_preserves_untracked_versions_and_fails_closed_on_bad_metadata(versions):
    (versions[1] / ".component.json").unlink()
    result = cache.prune_component_cache(keep=1, dry_run=False)
    assert {r["reason"] for r in result["protected"]} == {
        "legacy",
        "untracked",
        "recent",
    }
    assert versions[1].exists()
    (versions[-1] / ".component.json").write_text("broken")
    with pytest.raises(ValueError, match="nothing removed"):
        cache.prune_component_cache(keep=1, dry_run=False)
    assert versions[0].exists() and versions[-1].exists()


def test_gc_waits_for_active_component_installation(versions):
    started, finished = threading.Event(), threading.Event()

    def collect():
        started.set()
        result = cache.prune_component_cache(keep=1, dry_run=False)
        finished.set()
        return result

    with ThreadPoolExecutor(max_workers=1) as executor:
        with cache.component_operation():
            with cache.component_operation():  # Nested fetches must not deadlock.
                task = executor.submit(collect)
                assert started.wait(2)
                assert not finished.wait(0.05)
                assert all(path.exists() for path in versions)
        assert task.result(timeout=3)["removed"]


def test_registering_failed_upgrade_retains_previous_config_reference(
    versions, tmp_path, monkeypatch
):
    index = tmp_path / "index.json"
    monkeypatch.setattr(store, "INDEX_FILE", index)
    project = tmp_path / "config.json"
    project.write_text(str(versions[1]))
    for directory in versions[1:3]:
        index.write_text(
            json.dumps(
                {"schema": 1, "artifacts": {"package/lab": {"sha256": directory.name}}}
            )
        )
        cache.register_project(project, ["package/lab"])
    assert versions[1].name in cache._protected(cache.root_path())


def test_indirect_paths_and_missing_configs_do_not_release_pins(versions, tmp_path):
    from hashlib import sha256

    config = tmp_path / "config.json"
    config.write_text('{"args": ["${LAB_ROOT}/server.py"]}')
    record = (
        cache.root_path()
        / ".references"
        / f"{sha256(str(config).encode()).hexdigest()}.json"
    )
    cache._write(record, {"path": str(config), "digests": [versions[1].name]})
    for present in (True, False):
        if not present:
            config.unlink()
        report = cache.prune_component_cache(keep=1)
        assert versions[1].name not in {entry["digest"] for entry in report["removed"]}
    assert cache.forget_project(config)["digests"] == [versions[1].name]
    assert record.exists()
    cache.forget_project(config, dry_run=False)
    assert not record.exists()


def test_legacy_removal_is_explicit_and_never_overrides_known_pins(versions, tmp_path):
    untracked = versions[2]
    (untracked / ".component.json").unlink()
    cache._write(
        cache.root_path() / ".references/active.json",
        {"path": str(tmp_path / "removed-config"), "digests": [versions[0].name]},
    )
    preview = cache.prune_component_cache(keep=1, include_legacy=True)
    assert untracked.name in {entry["digest"] for entry in preview["removed"]}
    assert versions[0].name not in {entry["digest"] for entry in preview["removed"]}
    assert all(path.exists() for path in versions)
    cache.prune_component_cache(keep=1, include_legacy=True, dry_run=False)
    assert not untracked.exists()
    assert versions[0].exists()


def test_component_cli_previews_by_default_and_shares_retention_policy(
    versions, monkeypatch
):
    from click.testing import CliRunner
    from types import SimpleNamespace
    from clio_kit import cache_cli

    monkeypatch.setenv("CLIO_KIT_COMPONENT_KEEP", "3")
    monkeypatch.setenv("CLIO_KIT_ENV_KEEP", "7")
    eviction = SimpleNamespace(evicted=[], skipped_in_use=[], bytes_freed=0)
    prune = SimpleNamespace(ran=False, ok=True, reason="test")
    monkeypatch.setattr(
        cache_cli, "collect_cache_gc", lambda *args, **kwargs: (eviction, prune)
    )
    runner = CliRunner()
    standalone = runner.invoke(cache_cli.cache_group, ["components"])
    combined = runner.invoke(cache_cli.cache_group, ["gc", "--dry-run"])
    assert standalone.exit_code == combined.exit_code == 0, (
        standalone.output,
        combined.output,
    )
    first, second = json.loads(standalone.output), json.loads(combined.output)
    assert first["dry_run"] is True
    assert first["keep"] == second["components"]["keep"] == 3
    assert first["removed"] == second["components"]["removed"]
    assert all(path.exists() for path in versions)
    applied = runner.invoke(cache_cli.cache_group, ["components", "--apply"])
    assert applied.exit_code == 0, applied.output
    assert not versions[1].exists()


def test_invalid_component_policy_is_rejected_before_runtime_gc(versions, monkeypatch):
    from click.testing import CliRunner
    from clio_kit import cache_cli

    monkeypatch.setenv("CLIO_KIT_COMPONENT_KEEP", "invalid")

    def forbidden(*args, **kwargs):
        pytest.fail("Runtime GC must not run with an invalid component policy")

    monkeypatch.setattr(cache_cli, "collect_cache_gc", forbidden)
    result = CliRunner().invoke(cache_cli.cache_group, ["gc"])
    assert result.exit_code != 0 and "positive integer" in result.output
    assert all(path.exists() for path in versions)
