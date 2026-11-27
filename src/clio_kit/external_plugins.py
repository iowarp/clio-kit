"""Fetch selected publisher packages without executing their installation scripts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from clio_kit.community import validate_source_location
from clio_kit.federation import git_output, source_url

MAX_BYTES = 512 * 1024 * 1024
MAX_FILES = 20000


def unpack_package(archive: Path, destination: Path) -> None:
    """Extract npm's package/ tree; reject links, escapes, duplicates and bombs."""
    seen: set[str] = set()
    total = 0
    with tarfile.open(archive) as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or ":" in member.name
                or len(path.parts) < 2
                and not member.isdir()
                or not path.parts
                or path.parts[0] != "package"
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"Unsafe npm archive member: {member.name}")
            if member.isdir():
                continue
            name = str(PurePosixPath(*path.parts[1:]))
            total += member.size
            if name in seen or len(seen) >= MAX_FILES or total > MAX_BYTES:
                raise ValueError("Duplicate or oversized npm package")
            seen.add(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            assert source is not None
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def fetch_source(source: dict, temporary: Path) -> tuple[Path, dict]:
    """Return a temporary package and resolved provenance, with no lifecycle code."""
    validate_source_location(source)
    if os.getenv("CLIO_KIT_OFFLINE") == "1":
        raise ValueError("Offline mode forbids fetching external plugins")
    if source["source"] == "npm":
        coordinate = source["package"] + "@" + source.get("version", "latest")
        args = [
            "npm",
            "pack",
            coordinate,
            "--ignore-scripts",
            "--json",
            "--pack-destination",
            str(temporary),
        ]
        if source.get("registry"):
            args.extend(["--registry", source["registry"]])
        result = subprocess.run(
            args, capture_output=True, text=True, check=True, timeout=120
        )
        record = json.loads(result.stdout)[0]
        archive = temporary / Path(record["filename"]).name
        if archive.stat().st_size > MAX_BYTES:
            raise ValueError("External archive exceeds download limit")
        package = temporary / "package"
        unpack_package(archive, package)
        revision = record["version"]
    else:
        checkout = temporary / "repository"
        git_output(
            "clone",
            "--no-checkout",
            "--depth",
            "1",
            "--",
            source_url(source),
            str(checkout),
        )
        ref = source.get("sha") or source.get("ref")
        if ref:
            git_output("-C", str(checkout), "fetch", "--depth", "1", "origin", ref)
        revision = git_output(
            "-C", str(checkout), "rev-parse", "FETCH_HEAD" if ref else "HEAD"
        )
        if source.get("sha") and revision != source["sha"]:
            raise ValueError("Publisher revision differs from the indexed SHA")
        git_output("-C", str(checkout), "checkout", "--detach", revision)
        package = checkout / source.get("path", ".")
        if not package.is_dir() or not package.resolve().is_relative_to(
            checkout.resolve()
        ):
            raise ValueError("External package path leaves repository or is missing")
    digest = package_digest(package)
    clean = temporary / "payload"
    shutil.copytree(package, clean, ignore=shutil.ignore_patterns(".git"))
    return clean, {"source": source, "revision": revision, "sha256": digest}


def package_digest(package: Path) -> str:
    """Bound and hash regular content, including executable bits and file paths."""
    if not package.is_dir() or package.is_symlink():
        raise ValueError("Publisher payload is missing or linked")
    digest = hashlib.sha256()
    total = count = 0
    for base, directories, files in os.walk(package):
        directories[:] = sorted(d for d in directories if d != ".git")
        for name in [*directories, *files]:
            if (Path(base) / name).is_symlink():
                raise ValueError("Linked external package content is not supported")
        for name in sorted(files):
            path = Path(base) / name
            if not path.is_file():
                raise ValueError("External package contains a non-regular file")
            total += path.stat().st_size
            count += 1
            if total > MAX_BYTES or count > MAX_FILES:
                raise ValueError("External package exceeds extraction limits")
            item = [
                path.relative_to(package).as_posix(),
                bool(path.stat().st_mode & 0o111),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
            digest.update(json.dumps(item).encode() + b"\n")
    return digest.hexdigest()
