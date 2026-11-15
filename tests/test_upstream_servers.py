"""Servers that run one third-party MCP server exactly as published ([upstream]).

Each such server is a thin wrapper: its manifest depends on the upstream release
named in mcp-server-versions.toml, pinned exactly, and its lock resolves that
release; no upstream code is copied into the repository. These checks keep the
inventory, the manifest, the lock and the documentation saying the same thing.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = tomllib.loads(
    (ROOT / "mcp-server-versions.toml").read_text(encoding="utf-8")
)
UPSTREAM: dict[str, dict] = INVENTORY.get("upstream", {})
FIELDS = {
    "package",
    "registry",
    "version",
    "license",
    "repository",
    "probe-environment",
}
# A wrapper's own code stays small; anything bigger is a copy of the upstream.
WRAPPER_SOURCE_LIMIT = 8_000


def test_upstream_servers_are_registered_servers() -> None:
    assert UPSTREAM
    assert set(UPSTREAM) <= set(INVENTORY["servers"])


@pytest.mark.parametrize("server", sorted(UPSTREAM))
def test_the_record_is_complete(server: str) -> None:
    record = UPSTREAM[server]
    assert set(record) <= FIELDS, sorted(set(record) - FIELDS)
    assert {"package", "registry", "version", "license", "repository"} <= set(record)
    assert record["registry"] in {"pypi", "npm"}
    assert record["repository"].startswith("https://")


@pytest.mark.parametrize("server", sorted(UPSTREAM))
def test_the_manifest_pins_exactly_that_release_and_the_lock_resolves_it(
    server: str,
) -> None:
    record, folder = UPSTREAM[server], ROOT / "mcp-servers" / server
    package, version = record["package"], record["version"]
    if record["registry"] == "pypi":
        manifest = tomllib.loads(
            (folder / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = [Requirement(d) for d in manifest["project"]["dependencies"]]
        assert [(d.name, str(d.specifier)) for d in dependencies] == [
            (package, f"=={version}")
        ]
        lock = tomllib.loads((folder / "uv.lock").read_text(encoding="utf-8"))
        locked = {row["name"]: row["version"] for row in lock["package"]}
        assert locked[package] == version
    else:
        manifest = json.loads((folder / "package.json").read_text(encoding="utf-8"))
        assert manifest["dependencies"] == {package: version}
        lock = json.loads((folder / "package-lock.json").read_text(encoding="utf-8"))
        assert lock["packages"][f"node_modules/{package}"]["version"] == version


@pytest.mark.parametrize("server", sorted(UPSTREAM))
def test_no_upstream_code_is_copied_in(server: str) -> None:
    folder = ROOT / "mcp-servers" / server
    code = [
        path
        for pattern in ("src/**/*.py", "bundle/**/*.js")
        for path in folder.glob(pattern)
    ]
    assert code, f"{server} has no wrapper entry"
    size = sum(path.stat().st_size for path in code)
    assert size < WRAPPER_SOURCE_LIMIT, f"{server}: {size} bytes of code"


@pytest.mark.parametrize("server", sorted(UPSTREAM))
def test_the_readme_names_the_release_and_its_licence(server: str) -> None:
    record = UPSTREAM[server]
    readme = (ROOT / "mcp-servers" / server / "README.md").read_text(encoding="utf-8")
    assert record["package"] in readme
    assert f"**{record['version']}**" in readme
    words = " ".join(readme.replace("**", "").split())
    assert f"is {record['license']}" in words, record["license"]
