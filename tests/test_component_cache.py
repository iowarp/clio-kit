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
    dry = cache.collect_components(keep=1)
    assert [r["digest"] for r in dry["removed"]] == [fourth.name]
    assert all(path.exists() for path in versions)
    applied = cache.collect_components(keep=1, dry_run=False)
    assert applied["removed"] == dry["removed"]
    assert not fourth.exists()
    assert all(path.exists() for path in (first, second, third, fifth))
    # Removing the consumer releases its old artifact on the next collection.
    project.unlink()
    assert second.name in {
        r["digest"] for r in cache.collect_components(keep=1)["removed"]
    }


def test_gc_preserves_untracked_versions_and_fails_closed_on_bad_metadata(versions):
    (versions[1] / ".component.json").unlink()
    result = cache.collect_components(keep=1, dry_run=False)
    assert {r["reason"] for r in result["protected"]} == {
        "legacy",
        "untracked",
        "recent",
    }
    assert versions[1].exists()
    (versions[-1] / ".component.json").write_text("broken")
    with pytest.raises(ValueError, match="nothing removed"):
        cache.collect_components(keep=1, dry_run=False)
    assert versions[0].exists() and versions[-1].exists()


def test_gc_waits_for_active_component_installation(versions):
    started, finished = threading.Event(), threading.Event()

    def collect():
        started.set()
        result = cache.collect_components(keep=1, dry_run=False)
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
