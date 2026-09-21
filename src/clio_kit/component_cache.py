"""Conservative component reclamation with catalogue and project references."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time

from clio_kit.environment_locks import _FileLock

_local = threading.local()
_mutex = threading.RLock()


def root_path() -> Path:
    from clio_kit.cache_cli import clio_cache_root

    return clio_cache_root() / "components"


@contextmanager
def component_operation():
    """Serialize cache consumption/registration with GC, including nested calls."""
    with _mutex:
        depth = getattr(_local, "depth", 0)
        lock = None
        if not depth:
            root = root_path()
            root.mkdir(parents=True, exist_ok=True)
            lock = _FileLock(root / ".gc.lock")
            deadline = time.monotonic() + 120
            while not lock.try_acquire():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Another component operation is still active")
                time.sleep(0.1)
        _local.depth = depth + 1
        try:
            yield
        finally:
            _local.depth = depth
            if lock is not None:
                lock.release()


def guarded(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if kwargs.get("dry_run"):
            return function(*args, **kwargs)
        with component_operation():
            return function(*args, **kwargs)

    return wrapped


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(data, sort_keys=True).encode())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_use(
    key: str, index: dict, target: Path, index_file: Path, *, legacy: bool
) -> None:
    root = root_path()
    receipt = target.parent / ".component.json"
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    _write(
        receipt,
        {
            "key": key,
            "digest": target.parent.name,
            "used": time.time(),
            "legacy": previous.get("legacy", legacy),
        },
    )
    if index_file.is_file():
        path = str(index_file.resolve())
        token = hashlib.sha256(path.encode()).hexdigest()
        _write(root / ".catalogues" / f"{token}.json", {"path": path})


def register_project(config: Path, artifacts: list[str]) -> None:
    """Persist a conservative union; GC checks the actual current config text."""
    from clio_kit.component_store import catalogue

    index = catalogue()
    digests = {index["artifacts"][key]["sha256"] for key in artifacts}
    token = hashlib.sha256(str(config).encode()).hexdigest()
    path = root_path() / ".references" / f"{token}.json"
    if path.exists():
        digests.update(json.loads(path.read_text())["digests"])
    _write(path, {"path": str(config), "digests": sorted(digests)})


def _protected(root: Path) -> set[str]:
    protected: set[str] = set()
    for record in (root / ".catalogues").glob("*.json"):
        path = Path(json.loads(record.read_text())["path"])
        if path.exists():
            index = json.loads(path.read_text())
            protected.update(value["sha256"] for value in index["artifacts"].values())
    for record in (root / ".references").glob("*.json"):
        data = json.loads(record.read_text())
        path = Path(data["path"])
        if path.exists():
            content = path.read_text()
            protected.update(digest for digest in data["digests"] if digest in content)
    return protected


def collect_components(*, keep: int = 2, dry_run: bool = True) -> dict:
    """Keep installed-catalogue, project-referenced, legacy and newest artifacts."""
    if keep < 1:
        raise ValueError("--keep must be >= 1")
    root = root_path()
    result: dict = {
        "dry_run": dry_run,
        "removed": [],
        "protected": [],
        "bytes_freed": 0,
    }
    if not root.exists():
        return result
    with component_operation():
        try:
            protected = _protected(root)
            groups: dict[str, list] = {}
            for directory in sorted(root.iterdir()):
                if not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                    continue
                receipt = directory / ".component.json"
                if directory.is_symlink() or not receipt.is_file():
                    result["protected"].append(
                        {"digest": directory.name, "reason": "untracked"}
                    )
                    continue
                data = json.loads(receipt.read_text())
                if data["digest"] != directory.name:
                    raise ValueError(f"Invalid component receipt: {receipt}")
                groups.setdefault(data["key"], []).append(
                    (data["used"], directory, data)
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Cannot establish safe cache references; nothing removed: {exc}"
            ) from exc
        for values in groups.values():
            for position, (_, directory, data) in enumerate(
                sorted(values, reverse=True)
            ):
                reason = (
                    "legacy"
                    if data.get("legacy")
                    else "referenced"
                    if directory.name in protected
                    else "recent"
                    if position < keep
                    else None
                )
                if reason:
                    result["protected"].append(
                        {"digest": directory.name, "reason": reason}
                    )
                    continue
                size = sum(
                    path.stat().st_size
                    for path in directory.rglob("*")
                    if path.is_file() and not path.is_symlink()
                )
                if not dry_run:
                    shutil.rmtree(directory)
                result["removed"].append(
                    {"digest": directory.name, "key": data["key"], "bytes": size}
                )
                result["bytes_freed"] += size
    return result
