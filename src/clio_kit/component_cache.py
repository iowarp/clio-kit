"""Conservative component reclamation with catalogue and project references."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
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


def register_project(config: Path, artifacts: list[str], transaction=None) -> None:
    """Pin payloads explicitly, including versions usable through config backups.

    Do not infer non-use from config text: clients can use variables or indirect
    paths. Pins remain until the operator explicitly forgets the project.
    Installation stages this receipt in the same rollback unit as client files.
    """
    from clio_kit.component_store import catalogue

    index = catalogue()
    digests = {
        index["artifacts"][key]["sha256"]
        for key in artifacts
        if key.startswith("package/")
    }
    token = hashlib.sha256(str(config).encode()).hexdigest()
    path = root_path() / ".references" / f"{token}.json"
    if path.exists():
        digests.update(json.loads(path.read_text())["digests"])
    data = {"path": str(config), "digests": sorted(digests)}
    if transaction is None:
        _write(path, data)
    else:
        transaction.file(path, json.dumps(data, sort_keys=True).encode())


def _protected(root: Path) -> set[str]:
    protected: set[str] = set()
    for record in (root / ".catalogues").glob("*.json"):
        path = Path(json.loads(record.read_text())["path"])
        if path.exists():
            index = json.loads(path.read_text())
            protected.update(value["sha256"] for value in index["artifacts"].values())
    for record in (root / ".references").glob("*.json"):
        data = json.loads(record.read_text())
        if not isinstance(data["digests"], list) or any(
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in data["digests"]
        ):
            raise ValueError(f"Invalid project receipt: {record}")
        protected.update(data["digests"])
    return protected


def component_retention(keep: int | None = None) -> int:
    """One policy for component payloads, independent of runtime environments."""
    value = keep if keep is not None else os.getenv("CLIO_KIT_COMPONENT_KEEP", "2")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CLIO_KIT_COMPONENT_KEEP must be a positive integer") from exc
    if result < 1:
        raise ValueError("Component retention must be >= 1")
    return result


def forget_project(config: Path, *, dry_run: bool = True) -> dict:
    """Explicitly release pins after the operator has retired all consumers."""
    config = config.expanduser().resolve()
    token = hashlib.sha256(str(config).encode()).hexdigest()
    with component_operation():
        path = root_path() / ".references" / f"{token}.json"
        data = json.loads(path.read_text()) if path.exists() else {"digests": []}
        result = {"config": str(config), "digests": data["digests"], "dry_run": dry_run}
        if not dry_run:
            path.unlink(missing_ok=True)
        return result


def prune_component_cache(
    *, keep: int | None = None, dry_run: bool = True, include_legacy: bool = False
) -> dict:
    """Reclaim superseded payloads; project pins never depend on config syntax."""
    keep = component_retention(keep)
    root = root_path()
    result: dict = {
        "dry_run": dry_run,
        "keep": keep,
        "include_legacy": include_legacy,
        "removed": [],
        "protected": [],
        "bytes_freed": 0,
    }
    if not root.exists():
        return result
    with component_operation():
        # Build the complete plan before deleting anything; malformed metadata
        # must fail closed even if its directory sorts after an eligible payload.
        try:
            protected = _protected(root)
            groups: dict[str | None, list] = {}
            for directory in sorted(root.iterdir()):
                if not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                    continue
                receipt = directory / ".component.json"
                if directory.is_symlink() or not directory.is_dir():
                    result["protected"].append(
                        {"digest": directory.name, "reason": "unsafe-path"}
                    )
                    continue
                if not receipt.is_file():
                    data: dict = {
                        "key": None,
                        "used": 0,
                        "legacy": True,
                        "untracked": True,
                    }
                else:
                    data = json.loads(receipt.read_text())
                    if (
                        data["digest"] != directory.name
                        or not isinstance(data["key"], str)
                        or not isinstance(data["used"], (int, float))
                    ):
                        raise ValueError(f"Invalid component receipt: {receipt}")
                groups.setdefault(data["key"], []).append(
                    (data["used"], directory, data)
                )
            candidates = []
            for values in groups.values():
                for position, (_, directory, data) in enumerate(
                    sorted(values, key=lambda row: (row[0], row[1].name), reverse=True)
                ):
                    reason = (
                        "referenced"
                        if directory.name in protected
                        else "untracked"
                        if data.get("untracked") and not include_legacy
                        else "legacy"
                        if data.get("legacy") and not include_legacy
                        else "recent"
                        if not data.get("untracked") and position < keep
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
                    result["removed"].append(
                        {
                            "digest": directory.name,
                            "key": data["key"],
                            "bytes": size,
                            "legacy": bool(data.get("legacy")),
                        }
                    )
                    result["bytes_freed"] += size
                    candidates.append(directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Cannot establish safe cache references; nothing removed: {exc}"
            ) from exc
        if not dry_run:
            for directory in candidates:
                shutil.rmtree(directory)
    return result
