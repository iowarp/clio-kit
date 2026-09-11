"""Compile external catalogues into native, installable marketplace entries.

Only metadata is read here. Plugin code remains in the publisher's repository
and is fetched by the client when installed. A failed refresh leaves the last
working catalogue intact. Source revisions and imported names live alongside
the native manifest, not in unsupported manifest fields.
"""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from clio_kit.community import (
    COMMUNITY_SOURCE_FIELDS,
    read_federated_marketplaces,
    write_live_marketplaces,
)

LOCK_NAME = "federation.lock.json"


def read_snapshot(root: Path) -> list[dict[str, Any]]:
    path = root / ".claude-plugin" / LOCK_NAME
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    current = {entry["name"] for entry in read_federated_marketplaces(root)}
    live_names = {
        name
        for entry in payload.get("marketplaces", [])
        if entry["name"] in current
        for name in entry["plugins"]
    }
    return [
        entry for entry in payload.get("plugins", []) if entry["name"] in live_names
    ]


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false", *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return result.stdout.strip()


def source_url(source: dict[str, Any]) -> str:
    return (
        f"https://github.com/{source['repo']}.git"
        if source["source"] == "github"
        else source["url"]
    )


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"External plugin path leaves its repository: {value!r}")
    return str(path)


def compile_catalogue(
    catalogue: dict[str, Any], *, url: str, revision: str, checkout: Path
) -> list[dict[str, Any]]:
    """Translate relative plugin paths; preserve names and dependency identities."""
    if not isinstance(catalogue.get("name"), str) or not isinstance(
        catalogue.get("plugins"), list
    ):
        raise ValueError("External marketplace needs a name and a plugins array")
    entries = []
    names: set[str] = set()
    for original in catalogue["plugins"]:
        if not isinstance(original, dict) or not isinstance(original.get("name"), str):
            raise ValueError("External marketplace contains an unnamed plugin")
        if original["name"] in names:
            raise ValueError(f"Duplicate external plugin name: {original['name']}")
        names.add(original["name"])
        entry = copy.deepcopy(original)
        tags = entry.pop("tags", [])
        if tags:
            entry["keywords"] = sorted(set(entry.get("keywords", []) + tags))
        source = entry.get("source")
        if isinstance(source, str):
            relative = _relative_path(source)
            plugin = checkout / relative
            if not plugin.resolve().is_relative_to(checkout.resolve()):
                raise ValueError(f"Linked plugin escapes external repository: {source}")
            manifest = plugin / ".claude-plugin" / "plugin.json"
            if manifest.is_file():
                content = json.loads(manifest.read_text())
                if content.get("name") != entry["name"]:
                    raise ValueError(f"Plugin name differs from its manifest: {source}")
            elif entry.get("strict", True):
                raise ValueError(f"External plugin manifest missing: {source}")
            entry["source"] = {
                "source": "git-subdir",
                "url": url,
                "path": relative,
                "sha": revision,
            }
        elif not isinstance(source, dict) or source.get("source") not in {
            "github",
            "url",
            "git-subdir",
            "npm",
        }:
            raise ValueError(f"Unsupported external plugin source: {source!r}")
        if isinstance(source, dict):
            required, optional = COMMUNITY_SOURCE_FIELDS[source["source"]]
            if set(source) - {"source", *required, *optional}:
                raise ValueError(f"Unknown external source fields: {source!r}")
            for field in required:
                if not isinstance(source.get(field), str) or not source[field].strip():
                    raise ValueError(f"External source needs {field}: {source!r}")
            for field in optional:
                if field in source and (
                    not isinstance(source[field], str) or not source[field].strip()
                ):
                    raise ValueError(
                        f"External source {field} must be a nonempty string"
                    )
            if source["source"] == "git-subdir":
                _relative_path(source["path"])
        entries.append(entry)
    return entries


def _same_source(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return left == right
    # An explicitly indexed plugin can also occur in its publisher's catalogue.
    return {k: v for k, v in left.items() if k not in {"sha", "ref"}} == {
        k: v for k, v in right.items() if k not in {"sha", "ref"}
    }


def refresh_marketplace(root: Path) -> dict[str, Any]:
    manifest_path = root / ".claude-plugin" / "marketplace.json"
    lock_path = manifest_path.with_name(LOCK_NAME)
    marketplace = json.loads(manifest_path.read_text())
    previous = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    imported_names = set(previous.get("imported_names", []))
    entries = {
        e["name"]: e
        for e in marketplace["plugins"]
        if e["name"] not in imported_names
        and not (e.get("metadata") or {}).get("clioFederation")
    }
    imports: list[str] = []
    provenance = []
    referrals = read_federated_marketplaces(root)
    for referral in referrals:
        source = referral["source"]
        url = source_url(source)
        with tempfile.TemporaryDirectory(prefix="clio-federation-") as temporary:
            checkout = Path(temporary) / "repository"
            git_output("clone", "--depth", "1", "--", url, str(checkout))
            ref = source.get("sha") or source.get("ref")
            if ref:
                git_output(
                    "-C", str(checkout), "fetch", "--depth", "1", "--", "origin", ref
                )
                git_output("-C", str(checkout), "checkout", "--detach", "FETCH_HEAD")
            revision = git_output("-C", str(checkout), "rev-parse", "HEAD")
            catalogue_path = checkout / ".claude-plugin" / "marketplace.json"
            if catalogue_path.stat().st_size > 1_048_576:
                raise ValueError(f"External catalogue exceeds 1 MiB: {url}")
            catalogue = json.loads(catalogue_path.read_text())
            plugins = compile_catalogue(
                catalogue, url=url, revision=revision, checkout=checkout
            )
        names = []
        for entry in plugins:
            name = entry["name"]
            names.append(name)
            if name in entries:
                if not _same_source(entries[name]["source"], entry["source"]):
                    raise ValueError(
                        f"Federated plugin name collision: {name!r} from {url}"
                    )
                continue
            if name.startswith("clio-"):
                raise ValueError(f"External plugin claims reserved prefix: {name}")
            entry.setdefault("metadata", {}).update(
                indexed=True, clioFederation=referral["name"]
            )
            entries[name] = entry
            imports.append(name)
        provenance.append(
            {
                "name": referral["name"],
                "url": url,
                "revision": revision,
                "plugins": names,
            }
        )
    marketplace["plugins"] = list(entries.values())
    lock = {
        "schema": "clio-kit.federation.v1",
        "imported_names": imports,
        "marketplaces": provenance,
        "plugins": [entries[name] for name in imports],
    }
    # Validate all fetches, names and sources before replacing either file.
    pending = []
    for path, content in ((manifest_path, marketplace), (lock_path, lock)):
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(content, indent=2) + "\n")
        pending.append((temporary_path, path))
    for temporary_path, path in pending:
        temporary_path.replace(path)
    write_live_marketplaces(root, referrals)
    return lock
