"""Provenance sidecars, byte-compatible with clio-kit's ocean and weather servers.

Every output file gets ``<output>.provenance.json`` (schema ``clio.provenance/1``):
the output's name, when it was made, by which server and tool, from which sources
(product, service, file or output, with the subset that was read), with which
parameters and software, and any notes. A tool result lists each output as
``{path, role, media_type, provenance}``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA = "clio.provenance/1"
SERVER_NAME = "geoservices"
SERVER_VERSION = "1.0.0"


def package_versions(names: Iterable[str]) -> dict[str, str]:
    """Installed versions of the named distributions (``"absent"`` if missing)."""
    versions: dict[str, str] = {"geoservices-mcp": SERVER_VERSION}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def file_digest(path: Path) -> str:
    """sha256 of a file, for provenance of local inputs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Source:
    """One input to a result: a product, a service query or a local file."""

    name: str
    kind: str  # "product" | "service" | "file" | "output"
    reference: str = ""  # DOI, URL, dataset id or path
    version: str = ""
    subset: Mapping[str, Any] = field(default_factory=dict)
    licence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "reference": self.reference,
            "version": self.version,
            "subset": dict(self.subset),
            "licence": self.licence,
        }


def local_file_source(path: Path, name: str | None = None) -> Source:
    """A Source for a file on disk, with its digest as the version."""
    return Source(
        name=name or path.name,
        kind="file",
        reference=str(path),
        version=f"sha256:{file_digest(path)}" if path.is_file() else "",
    )


def write_provenance(
    output: Path,
    *,
    tool: str,
    sources: Iterable[Source],
    method: Mapping[str, Any],
    packages: Iterable[str] = (),
    notes: Iterable[str] = (),
) -> Path:
    """Write ``<output>.provenance.json`` and return its path."""
    source_list = [s.as_dict() for s in sources]
    licences = sorted({s["licence"] for s in source_list if s["licence"]})
    record = {
        "schema": PROVENANCE_SCHEMA,
        "output": output.name,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "server": f"{SERVER_NAME} {SERVER_VERSION}",
        "tool": tool,
        "sources": source_list,
        "method": dict(method),
        "software": package_versions(packages),
        "licences": licences,
        "notes": list(notes),
    }
    sidecar = output.with_name(output.name + ".provenance.json")
    sidecar.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return sidecar


def output_entry(
    path: Path, role: str, media_type: str, provenance: Path
) -> dict[str, Any]:
    """One output as tool results list it."""
    return {
        "path": str(path),
        "role": role,
        "media_type": media_type,
        "provenance": str(provenance),
    }
