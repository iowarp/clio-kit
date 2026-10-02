"""Record project ownership and remove only unchanged, installed components."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import tomli_w

from clio_kit.external_plugins import package_digest
from clio_kit.component_cache import guarded
from clio_kit.install_transaction import InstallTransaction


def fingerprint(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"Linked installed component: {path}")
    return (
        package_digest(path)
        if path.is_dir()
        else hashlib.sha256(path.read_bytes()).hexdigest()
    )


def remove_owned(current: dict, owned: dict, before: dict, shared: list[dict]) -> None:
    """Inverse a named merge, preserving unrelated settings and shared owners."""
    for key, value in owned.items():
        if any(other.get(key) == value for other in shared):
            continue
        if key not in current:
            continue
        previous = before.get(key)
        if isinstance(value, dict) and isinstance(current[key], dict):
            remove_owned(
                current[key],
                value,
                previous if isinstance(previous, dict) else {},
                [other[key] for other in shared if isinstance(other.get(key), dict)],
            )
            if not current[key] and key not in before:
                del current[key]
        elif isinstance(value, list) and isinstance(current[key], list):
            keep = (previous if isinstance(previous, list) else []) + [
                v for other in shared for v in other.get(key, [])
            ]
            current[key] = [v for v in current[key] if v not in value or v in keep]
            if not current[key] and key not in before:
                del current[key]
        elif current[key] == value:
            if key in before:
                current[key] = copy.deepcopy(previous)
            else:
                del current[key]
        else:
            raise ValueError(
                f"Installed setting {key} was edited; restore it before uninstalling/updating"
            )


class Receipt:
    def __init__(self, project: Path, client: str, name: str):
        from clio_kit.client_install import safe_destination

        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError("Invalid package name")
        self.project = project
        self.path = safe_destination(
            project, f".clio-kit/installed/{client}/{name}.json"
        )
        self.old = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.others = [
            json.loads(
                safe_destination(project, str(p.relative_to(project))).read_text()
            )
            for p in self.path.parent.parent.glob("*/*.json")
            if p != self.path
        ]
        self.data: dict = {
            "package": name,
            "client": client,
            "files": {},
            "configs": {},
        }

    def configuration(self, relative: str, current: dict, owned: dict) -> None:
        old = self.old.get("configs", {}).get(relative)
        before = copy.deepcopy(current)
        if not old:
            for other in self.others:
                previous = other.get("configs", {}).get(relative)
                if previous:
                    remove_owned(before, previous["owned"], previous["before"], [])
        if old:
            shared = [
                r["configs"][relative]["owned"]
                for r in self.others
                if relative in r.get("configs", {})
            ]
            remove_owned(current, old["owned"], old["before"], shared)
            before = old["before"]
        self.data["configs"][relative] = {
            "before": before,
            "owned": copy.deepcopy(owned),
        }

    def file(self, relative: str, content: Path | bytes) -> None:
        from clio_kit.client_install import safe_destination

        path = safe_destination(self.project, relative)
        digest = (
            fingerprint(content)
            if isinstance(content, Path)
            else hashlib.sha256(content).hexdigest()
        )
        old = self.old.get("files", {}).get(relative)
        other = next(
            (
                r["files"][relative]
                for r in self.others
                if relative in r.get("files", {})
            ),
            None,
        )
        if other and other["digest"] != digest:
            raise ValueError(
                f"Another installed package owns different content: {relative}"
            )
        self.data["files"][relative] = {
            "digest": digest,
            "created": old["created"]
            if old
            else other["created"]
            if other
            else not path.exists(),
        }

    def cleared_configuration(
        self, relative: str, record: dict
    ) -> tuple[Path, bytes] | None:
        from clio_kit.client_install import safe_destination, tomllib

        path = safe_destination(self.project, relative)
        if not path.exists():
            return None
        is_toml = path.suffix == ".toml"
        current = (
            tomllib.loads(path.read_text()) if is_toml else json.loads(path.read_text())
        )
        shared = [
            r["configs"][relative]["owned"]
            for r in self.others
            if relative in r.get("configs", {})
        ]
        remove_owned(current, record["owned"], record["before"], shared)
        return path, (
            tomli_w.dumps(current) if is_toml else json.dumps(current, indent=2) + "\n"
        ).encode()

    def stage(self, transaction: InstallTransaction) -> None:
        from clio_kit.client_install import safe_destination

        for relative, old in self.old.get("files", {}).items():
            if relative in self.data["files"] or any(
                relative in r.get("files", {}) for r in self.others
            ):
                continue
            path = safe_destination(self.project, relative)
            if path.exists() and old["created"]:
                if fingerprint(path) != old["digest"]:
                    raise ValueError(f"Installed component was edited: {relative}")
                transaction.remove(path)
        for relative, record in self.old.get("configs", {}).items():
            if relative not in self.data["configs"]:
                cleared = self.cleared_configuration(relative, record)
                if cleared:
                    transaction.file(*cleared)
        transaction.file(self.path, (json.dumps(self.data, indent=2) + "\n").encode())


@guarded
def uninstall_for_client(
    name: str, client: str, project: Path, *, dry_run: bool = False
) -> dict:
    from clio_kit.client_install import safe_destination

    project = project.resolve()
    receipt = Receipt(project, client, name)
    if not receipt.old:
        raise ValueError(f"No {client} installation receipt for {name}")
    removed, retained = [], []
    writes: dict[Path, bytes | None] = {}
    for relative, record in receipt.old.get("files", {}).items():
        path = safe_destination(project, relative)
        if (
            any(relative in other.get("files", {}) for other in receipt.others)
            or not record["created"]
        ):
            retained.append(relative)
        elif path.exists():
            if fingerprint(path) != record["digest"]:
                raise ValueError(
                    f"Installed component was edited: {relative}; nothing removed"
                )
            writes[path] = None
            removed.append(relative)
    for relative, record in receipt.old.get("configs", {}).items():
        cleared = receipt.cleared_configuration(relative, record)
        if cleared:
            writes[cleared[0]] = cleared[1]
    if not dry_run:
        with InstallTransaction() as transaction:
            for path, content in writes.items():
                if content is None:
                    transaction.remove(path)
                else:
                    transaction.file(path, content)
            transaction.remove(receipt.path)
            transaction.commit()
    return {
        "package": name,
        "client": client,
        "removed": removed,
        "retained": retained,
        "note": "Publisher payloads and source locks are retained for other clients and offline reinstall.",
    }
