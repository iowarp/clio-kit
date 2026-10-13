"""The clio-hdf5 Agent Plugin, checked the way a CLIO loader would use it.

CLIO has no plugin loader yet, so these tests stand in for one: the manifest
and mcp.json conform to Agent Plugins 1.0, the `ai.iowarp.clio` entry's import
paths resolve, its data route serves once the host supplies a resolver, and
each convention entry identifies a file written by that convention's own
library.
"""

from __future__ import annotations

import importlib
import json
import posixpath
import re
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import jsonschema
import pytest
import yaml
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from hdf5_mcp.provider import StructuredFileProvider
from hdf5_mcp.provider.routes import RESOLVER_STATE_KEY, base_dir_resolver

PLUGIN = Path(__file__).resolve().parents[3] / "plugins" / "clio-hdf5"
FIXTURES = Path(__file__).parent / "fixtures"
CONVENTION_FILES = FIXTURES / "conventions"
NAMESPACE = "ai.iowarp.clio"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return _json(PLUGIN / "plugin.json")


@pytest.fixture(scope="module")
def entry(manifest: dict[str, Any]) -> dict[str, str]:
    (only,) = manifest["extensions"][NAMESPACE]["structured_files"]
    return only


def _load(reference: str) -> Any:
    module, _, attribute = reference.partition(":")
    return getattr(importlib.import_module(module), attribute)


def _plugin_path(value: str) -> Path:
    """Resolve a plugin-relative path the way Agent Plugins §4.1 requires."""
    assert value.startswith("./"), f"{value!r} must start with './'"
    resolved = (PLUGIN / value).resolve()
    assert resolved.is_relative_to(PLUGIN.resolve()), f"{value!r} leaves the plugin"
    return resolved


# -- manifest -------------------------------------------------------------


def test_manifest_conforms_to_agent_plugins_1_0(manifest):
    schema = _json(FIXTURES / "agent_plugins_1_0_0" / "plugin.schema.json")
    jsonschema.validate(manifest, schema)


def test_mcp_config_conforms_and_targets_the_same_version(manifest):
    config = _json(PLUGIN / "mcp.json")
    jsonschema.validate(
        config, _json(FIXTURES / "agent_plugins_1_0_0" / "mcp.schema.json")
    )
    version = re.compile(r"/schemas/([^/]+)/")
    assert (
        version.search(config["$schema"])[1] == version.search(manifest["$schema"])[1]
    )


def test_manifest_agrees_with_the_generated_claude_manifest(manifest):
    """The Claude Code manifest is generated from the server; this one is not."""
    claude = _json(PLUGIN / ".claude-plugin" / "plugin.json")
    for field in ("name", "version", "description", "license"):
        assert manifest[field] == claude[field], field


def test_extension_files_live_in_the_namespace_directory(entry):
    """Agent Plugins §8: client files belong in a directory named for the namespace."""
    namespace_dir = (PLUGIN / NAMESPACE).resolve()
    for field in ("viewer", "conventions"):
        assert _plugin_path(entry[field]).is_relative_to(namespace_dir), field


# -- entry points ---------------------------------------------------------


def test_provider_reference_names_a_structured_file_provider(entry):
    provider = _load(entry["provider"])()
    assert isinstance(provider, StructuredFileProvider)
    assert provider.detect(CONVENTION_FILES / "n.nxs")
    assert [
        node["name"] for node in provider.children(CONVENTION_FILES / "n.nxs", "/")
    ] == ["entry"]


def _app(entry: dict[str, str], resolver: Any = None) -> TestClient:
    router = _load(entry["data_routes"])
    assert isinstance(router, APIRouter)
    app = FastAPI()
    if resolver is not None:
        setattr(app.state, RESOLVER_STATE_KEY, resolver)
    app.include_router(router, prefix="/files/h5grove")
    return TestClient(app)


def test_data_route_serves_once_the_host_supplies_a_resolver(entry):
    client = _app(entry, base_dir_resolver(CONVENTION_FILES))
    response = client.get(
        "/files/h5grove/meta", params={"file": "n.nxs", "path": "/entry/data"}
    )
    assert response.status_code == 200
    assert [child["name"] for child in response.json()["children"]] == ["I", "q"]


def test_data_route_refuses_without_a_resolver(entry):
    response = _app(entry).get("/files/h5grove/meta", params={"file": "n.nxs"})
    assert response.status_code == 503
    assert "resolver" in response.json()["message"]


def test_skill_is_discoverable_and_its_guides_exist():
    skill = PLUGIN / "skills" / "hdf5"
    frontmatter = yaml.safe_load(
        (skill / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
    )
    assert frontmatter["name"] == skill.name
    body = (skill / "SKILL.md").read_text(encoding="utf-8")
    guides = set(re.findall(r"`(references/[a-z0-9_]+\.md)`", body))
    assert guides
    assert all((skill / guide).is_file() for guide in guides)


# -- conventions ----------------------------------------------------------
#
# A small reference evaluator for the entry format documented in
# ai.iowarp.clio/README.md. CLIO owns the real one; this pins down what the
# entries are meant to say.


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _attribute(f: h5py.File, reference: str, cursor: str) -> Any:
    """`/@a` on the root, `@a` on the cursor, `/path@a` on a fixed object."""
    path, _, name = reference.rpartition("@")
    path = path or cursor
    if path not in f:
        return None
    return f[path].attrs.get(name)


def _shallow(f: h5py.File) -> list[Any]:
    """The root and the objects directly under it."""
    return [f, *(f[name] for name in f)]


def _detect_item(f: h5py.File, item: dict[str, Any]) -> bool:
    if "any" in item:
        return any(_detect_item(f, inner) for inner in item["any"])
    if "path" in item:
        return item["path"] in f
    reference = item["attr"]
    if "@" in reference:
        values = [_attribute(f, reference, "/")]
    else:
        values = [obj.attrs.get(reference) for obj in _shallow(f)]
    for value in values:
        if value is None:
            continue
        if "equals" in item and _text(value) != item["equals"]:
            continue
        if "contains" in item and item["contains"] not in _text(value):
            continue
        return True
    return False


def _matches(obj: Any, rule: dict[str, str]) -> bool:
    value = obj.attrs.get(rule["attr"])
    if value is None:
        return False
    return "equals" not in rule or _text(value) == rule["equals"]


def _locate(f: h5py.File, steps: list[dict[str, Any]]) -> str | list[str]:
    cursor = "/"
    for step in steps:
        ((kind, argument),) = step.items()
        if kind in ("follow_attr", "signal"):
            value = _attribute(f, argument, cursor)
            base = "/" if argument.startswith("/@") else cursor
            target = posixpath.join(base, _text(value)) if value is not None else None
            if target is not None and target in f:
                cursor = target
        elif kind == "find":
            if not _matches(f[cursor], argument):
                queue = deque([f[cursor]])
                while queue:
                    group = queue.popleft()
                    found = next(
                        (group[n] for n in group if _matches(group[n], argument)), None
                    )
                    if found is not None:
                        cursor = found.name
                        break
                    queue.extend(
                        group[n] for n in group if isinstance(group[n], h5py.Group)
                    )
        elif kind == "path":
            if argument in f:
                cursor = argument
        elif kind == "datasets_with_attr":
            group = f[cursor]
            return sorted(
                group[n].name
                for n in group
                if isinstance(group[n], h5py.Dataset) and argument in group[n].attrs
            )
    return cursor


@pytest.fixture(scope="module")
def conventions(entry) -> dict[str, dict[str, Any]]:
    pattern = _plugin_path(entry["conventions"])
    files = sorted(pattern.parent.glob(pattern.name))
    assert files
    entries = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in files]
    by_id = {e["id"]: e for e in entries}
    assert len(by_id) == len(entries), "duplicate convention ids"
    return by_id


def test_convention_entries_are_well_formed(conventions):
    for convention_id, e in conventions.items():
        assert e["container"] == "hdf5", convention_id
        assert e["detect"] and e["locate_primary"], convention_id
        assert e.get("extends") in (None, *conventions), convention_id


FILES = {
    "n.nxs": ({"nexus"}, {"nexus": "/entry/data/I"}),
    "cf.nc": ({"netcdf4", "cf"}, {"netcdf4": ["/tas"], "cf": ["/tas"]}),
    "n.nwb": ({"nwb"}, {"nwb": "/acquisition"}),
    "a.h5ad": ({"anndata"}, {"anndata": "/X"}),
    "pmd_100.h5": ({"openpmd"}, {"openpmd": "/data"}),
    "img.vtkhdf": ({"vtkhdf"}, {"vtkhdf": "/VTKHDF/PointData"}),
    "store.h5": ({"pytables", "pandas"}, {"pytables": "/df/table", "pandas": "/df"}),
}


@pytest.mark.parametrize("name", sorted(FILES))
def test_each_library_file_is_recognized(conventions, name):
    expected_ids, expected_primary = FILES[name]
    with h5py.File(CONVENTION_FILES / name, "r") as f:
        detected = {
            convention_id
            for convention_id, e in conventions.items()
            if all(_detect_item(f, item) for item in e["detect"])
        }
        assert detected == expected_ids
        for convention_id in detected:
            assert (
                _locate(f, conventions[convention_id]["locate_primary"])
                == expected_primary[convention_id]
            ), convention_id


def test_a_plain_file_matches_no_convention(conventions, tmp_path):
    with h5py.File(tmp_path / "plain.h5", "w") as f:
        f["x"] = [1, 2, 3]
        f["x"].attrs["units"] = "m"
    with h5py.File(tmp_path / "plain.h5", "r") as f:
        assert not [
            convention_id
            for convention_id, e in conventions.items()
            if all(_detect_item(f, item) for item in e["detect"])
        ]
