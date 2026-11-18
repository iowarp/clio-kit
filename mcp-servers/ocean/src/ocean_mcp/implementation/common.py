"""Shared pieces every ocean tool uses: where outputs go, provenance, results.

Every file a tool writes gets a provenance sidecar next to it
(``<file>.provenance.json``) recording the product, subset, method and the
versions of the code that produced it, in the spirit of copernicus-mcp's own
records. Science log entries cite that sidecar.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA = "clio.provenance/1"
SERVER_NAME = "ocean"
SERVER_VERSION = "1.0.0"

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class OceanToolError(Exception):
    """A failure the tool reports to the agent, with a typed reason.

    ``reason`` is a short machine-readable code (``credential_missing``,
    ``no_data``, ``bad_input``, ``backend_unavailable``, ...) and the message
    says what happened and what would fix it.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.message = message


def output_root() -> Path:
    """Directory outputs go to: ``OCEAN_MCP_OUTPUT_DIR`` or ``./ocean-outputs``."""
    configured = os.environ.get("OCEAN_MCP_OUTPUT_DIR", "").strip()
    root = Path(configured).expanduser() if configured else Path.cwd() / "ocean-outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_dir(name: str) -> Path:
    """A writable cache folder: ``OCEAN_MCP_CACHE_DIR/<name>`` or ``<output root>/.cache/<name>``."""
    configured = os.environ.get("OCEAN_MCP_CACHE_DIR", "").strip()
    base = Path(configured).expanduser() if configured else output_root() / ".cache"
    folder = base / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def output_path(name: str, suffix: str, output_dir: str | None = None) -> Path:
    """A path for a new output file, under ``output_dir`` or the output root."""
    stem = _SAFE.sub("-", name).strip("-") or "output"
    base = Path(output_dir).expanduser() if output_dir else output_root()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{stem}{suffix}"


def package_versions(names: Iterable[str]) -> dict[str, str]:
    """Installed versions of the named distributions (``"absent"`` if missing)."""
    versions: dict[str, str] = {"ocean-mcp": SERVER_VERSION}
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
    """Write ``<output>.provenance.json`` and return its path.

    Data licences of every source are collected into ``licences`` so they
    travel with the result (plan Rule 4).
    """
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


@dataclass
class Output:
    """A file a tool wrote, with what it holds."""

    path: Path
    role: str  # "profiles", "section", "tracks", "series", "grid", ...
    media_type: str
    provenance: Path | None = None
    rows: int | None = None
    #: The component that shows this output (``implementation.views``), if one fits.
    view: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "path": str(self.path),
            "role": self.role,
            "media_type": self.media_type,
        }
        if self.provenance is not None:
            record["provenance"] = str(self.provenance)
        if self.rows is not None:
            record["rows"] = self.rows
        if self.view is not None:
            record["view"] = self.view
        return record


MEDIA_TYPES = {
    ".parquet": "application/vnd.apache.parquet",
    ".nc": "application/x-netcdf",
    ".csv": "text/csv",
    ".json": "application/json",
    ".geojson": "application/geo+json",
}


def media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def result(
    summary: str, outputs: Iterable[Output] = (), **fields: Any
) -> dict[str, Any]:
    """The structured result every tool returns.

    ``summary`` is the human sentence the row and the agent both read;
    ``outputs`` lists the files written; other fields carry the numbers.
    """
    payload: dict[str, Any] = {"status": "success", "summary": summary}
    payload.update(fields)
    payload["outputs"] = [o.as_dict() for o in outputs]
    return payload


def iso(value: Any) -> str:
    """ISO-8601 text for a numpy/pandas/python datetime-like."""
    import pandas as pd

    return pd.Timestamp(value).isoformat()
