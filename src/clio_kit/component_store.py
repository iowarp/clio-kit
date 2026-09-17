"""Fetch only selected release artifacts, anchored to the installed catalogue."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen

import click

from clio_kit.environment_locks import _FileLock

INDEX_FILE = Path(__file__).with_name("_components.json")
MAX_BYTES = 512 * 1024 * 1024


def catalogue() -> dict:
    if not INDEX_FILE.is_file():
        raise ValueError(
            "No release component catalogue; install a built CLIO Kit release or use --root CHECKOUT"
        )
    data = json.loads(INDEX_FILE.read_text())
    if data.get("schema") != 1:
        raise ValueError("Unsupported component catalogue schema; upgrade CLIO Kit")
    return data


def artifact_path(key: str, index: dict | None = None) -> Path:
    from clio_kit.cache_cli import clio_cache_root

    data = index if index is not None else catalogue()
    digest = data["artifacts"][key]["sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid component digest")
    # The original directory name is preserved for runtime environment identities.
    name = key.rsplit("/", 1)[-1]
    if name in {"", ".", ".."} or "\\" in name:
        raise ValueError("Invalid component name")
    return clio_cache_root() / "components" / digest / name


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and name != "."
        and ":" not in name
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and str(path) == name
    )


def _verified(directory: Path, record: dict) -> bool:
    if not directory.is_dir() or directory.is_symlink():
        return False
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            return False
        if not path.is_file():
            continue
        name = path.relative_to(directory).as_posix()
        expected = record["files"].get(name)
        if not expected or path.stat().st_size != expected["size"]:
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            return False
        if os.name != "nt" and path.stat().st_mode & 0o777 != expected["mode"]:
            return False
        actual.add(name)
    return actual == set(record["files"])


def _download(record: dict, index: dict, destination: Path) -> None:
    base = os.environ.get("CLIO_KIT_COMPONENT_BASE_URL", index["base_url"])
    url = base.rstrip("/") + "/" + record["file"]
    parsed = urlparse(url)
    if not (
        parsed.scheme in {"https", "file"}
        or parsed.scheme == "http"
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError(
            "Component mirror must use HTTPS, a local file URI, or loopback HTTP"
        )
    digest = hashlib.sha256()
    total = 0
    with urlopen(url, timeout=60) as response, destination.open("wb") as output:
        final = urlparse(response.geturl())
        if parsed.scheme == "https" and final.scheme != "https":
            raise ValueError("Refusing insecure component redirect")
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > record["size"] or total > MAX_BYTES:
                raise ValueError("Component exceeds its declared download size")
            digest.update(chunk)
            output.write(chunk)
    if total != record["size"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("Component checksum/size mismatch; refusing installation")


def _unpack(archive: Path, target: Path, record: dict) -> None:
    expected = record["files"]
    if len(expected) > 20000 or sum(f["size"] for f in expected.values()) > MAX_BYTES:
        raise ValueError("Component exceeds extraction limits")
    seen = set()
    target.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if (
                not _safe_name(member.name)
                or not member.isfile()
                or member.name in seen
            ):
                raise ValueError("Unsafe or duplicate component archive member")
            metadata = expected.get(member.name)
            if not metadata or member.size != metadata["size"]:
                raise ValueError("Component archive differs from catalogue")
            seen.add(member.name)
            source = tar.extractfile(member)
            assert source is not None
            destination = target / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            destination.chmod(metadata["mode"])
    if not _verified(target, record):
        raise ValueError("Extracted component differs from catalogue")


def fetch(key: str, index: dict | None = None) -> Path:
    data = index if index is not None else catalogue()
    record = data["artifacts"][key]
    if not _safe_name(record["file"]) or "/" in record["file"]:
        raise ValueError("Invalid component artifact filename")
    target = artifact_path(key, data)
    if _verified(target, record):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _FileLock(target.parent / ".download.lock")
    deadline = time.monotonic() + 120
    while not lock.try_acquire():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Another process is downloading {key}")
        time.sleep(0.1)
    try:
        if _verified(target, record):
            return target
        if os.environ.get("CLIO_KIT_OFFLINE") == "1":
            raise ValueError(
                f"{key} is not cached or is damaged; offline mode forbids downloading it"
            )
        if target.exists() or target.is_symlink():
            raise ValueError(
                f"Cached component is damaged: {target}. Remove that component directory and retry"
            )
        click.echo(f"Downloading {key} ({record['size']} bytes)", err=True)
        with tempfile.TemporaryDirectory(
            prefix=".fetch-", dir=target.parent
        ) as temporary:
            staging = Path(temporary)
            try:
                _download(record, data, staging / "archive.tar.gz")
                _unpack(staging / "archive.tar.gz", staging / target.name, record)
            except OSError as exc:
                raise ValueError(
                    f"Cannot download {key} for CLIO Kit {data['version']}: {exc}. Check that this release's component assets are published or configure a trusted mirror"
                ) from exc
            (staging / target.name).rename(target)
        return target
    finally:
        lock.release()


def server_project(name: str, source_root: Path) -> Path:
    local = source_root / name
    if local.is_dir() and not INDEX_FILE.is_file():
        return local
    data = catalogue()
    # Names and physical directory names may differ.
    record = next(
        (
            record
            for key, record in data["servers"].items()
            if key == name or record["directory"] == name
        ),
        None,
    )
    if record is None:
        raise ValueError(f"Unknown server: {name}")
    return fetch(record["artifact"], data)
