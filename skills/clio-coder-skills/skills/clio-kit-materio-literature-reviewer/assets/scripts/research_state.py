#!/usr/bin/env python3
"""Deterministic local research state operations. Python 3.10+, stdlib only.

This helper never initializes git, tags a checkpoint, or writes paper state.
It does not run experiments or generated analysis programs.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

DEFAULTS = {"web_search": False, "auto_checkpoint": True, "commit_research": False}
PRESERVED = {"data", "checkpoints"}
ARCHIVE_MANIFEST = ".snapshot-manifest.json"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def research_root(project):
    project = Path(project).resolve(strict=True)
    root = project / ".research"
    if root.is_symlink():
        raise ValueError(".research must be a real directory, not a symlink")
    return root


def config(root):
    path = root / "config.json"
    result = dict(DEFAULTS)
    if path.is_symlink():
        raise ValueError("config.json must not be a symlink")
    if path.exists():
        value = json.loads(path.read_text(), object_pairs_hook=unique_object)
        if not isinstance(value, dict):
            raise ValueError("research config must be a JSON object")
        for key in DEFAULTS:
            if key in value and type(value[key]) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
        result.update(value)
    return result


def contained(root, relative, allow_missing=False):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"invalid relative research path: {relative}")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink is not permitted: {path}")
    if not allow_missing and not path.exists():
        raise ValueError(f"missing research path: {path}")
    return path


def task_number(value):
    if not re.fullmatch(r"[0-9]+", str(value)) or int(value) < 1:
        raise ValueError("task must be a positive decimal integer")
    return f"{int(value):02d}"


def workflow(root):
    path = root / "WORKFLOW.md"
    if not path.exists():
        return [], []
    text = contained(root, "WORKFLOW.md").read_text()
    active_text, _, archive_text = text.partition("## Archived Tasks")
    def parse(text):
        matches = list(re.finditer(r"^### Task ([0-9]+):[^\n]*", text, re.M))
        return [(task_number(m.group(1)), text[m.start():matches[i + 1].start() if i + 1 < len(matches) else len(text)]) for i, m in enumerate(matches)]
    active, archived = parse(active_text), parse(archive_text)
    ids = [number for number, _ in active + archived]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task identity in WORKFLOW.md")
    return active, archived


def next_task(root):
    active, archived = workflow(root)
    ids = [int(number) for number, _ in active + archived]
    tasks = root / "tasks"
    if tasks.exists():
        contained(root, "tasks")
        for path in tasks.iterdir():
            m = re.fullmatch(r"task-([0-9]+)", path.name)
            if m:
                ids.append(int(m.group(1)))
    return task_number(max(ids, default=0) + 1)


def validate_workflow(root):
    active, archived = workflow(root)
    ids = {number for number, _ in active}
    archived_ids = {number for number, _ in archived}
    graph = {}
    for number, block in active:
        match = re.search(r"^[-*] \*\*Dependencies\*\*:\s*(.*)$", block, re.M)
        refs = [task_number(value) for value in re.findall(r"Task\s+([0-9]+)", match.group(1) if match else "")]
        for ref in refs:
            if ref not in ids:
                kind = "archived" if ref in archived_ids else "missing"
                raise ValueError(f"Task {number} depends on {kind} Task {ref}")
        graph[number] = refs
    visiting, visited = set(), set()
    def visit(number):
        if number in visiting:
            raise ValueError(f"dependency cycle at Task {number}")
        if number in visited:
            return
        visiting.add(number)
        for ref in graph[number]:
            visit(ref)
        visiting.remove(number)
        visited.add(number)
    for number in graph:
        visit(number)
    return {"active": list(graph), "archived": sorted(archived_ids), "dependencies": graph}


def next_ready_task(root):
    """Choose a pending/in-progress task by workflow order and completed inputs."""
    graph = validate_workflow(root)["dependencies"]
    active, _ = workflow(root)
    statuses = {}
    for number, block in active:
        match = re.search(r"^[-*] \*\*Status\*\*:\s*(.*)$", block, re.M)
        status = re.sub(r"^[^a-zA-Z]+", "", match.group(1) if match else "").lower()
        statuses[number] = status.split()[0] if status else "unspecified"
    blocked = []
    for number, _ in active:
        if statuses[number] not in {"pending", "in-progress"}:
            continue
        unsettled = [dependency for dependency in graph[number] if statuses[dependency] != "complete"]
        if not unsettled:
            return {"id": number, "status": statuses[number], "blocked": blocked}
        blocked.append({"id": number, "dependencies": unsettled})
    return {"id": None, "reason": "no dependency-ready pending task", "blocked": blocked}


def snapshot_files(root):
    files = {}
    for base, directories, names in os.walk(root, followlinks=False):
        base = Path(base)
        if base == root:
            directories[:] = [name for name in directories if name not in PRESERVED]
        for name in directories:
            if (base / name).is_symlink():
                raise ValueError(f"snapshot refuses symlink: {base / name}")
        for name in names:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if base == root and name in PRESERVED:
                continue
            if path == root / ARCHIVE_MANIFEST:
                raise ValueError(f"reserved snapshot filename: {ARCHIVE_MANIFEST}")
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"snapshot needs regular files: {path}")
            files[relative] = path.read_bytes()
    return files


def checkpoint_save(root, label):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", label):
        raise ValueError("checkpoint label must be 1–64 path-safe characters")
    if not root.is_dir():
        raise ValueError("research state is not initialized")
    directory = contained(root, "checkpoints", allow_missing=True)
    directory.mkdir(exist_ok=True)
    files = snapshot_files(root)
    directories = []
    for base, names, _ in os.walk(root, followlinks=False):
        if Path(base) == root:
            names[:] = [name for name in names if name not in PRESERVED]
        directories.extend((Path(base) / name).relative_to(root).as_posix() for name in names)
    modes = {name: (root / name).stat().st_mode & 0o777 for name in files}
    manifest = {"version": 1, "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                "directories": sorted(directories), "modes": modes}
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + label + "-" + uuid4().hex[:8] + ".tgz"
    destination = directory / name
    created = False
    try:
        with destination.open("xb") as output:
            created = True
            with tarfile.open(fileobj=output, mode="w:gz") as archive:
                for relative, data in sorted({**files, ARCHIVE_MANIFEST: json.dumps(manifest).encode()}.items()):
                    member = tarfile.TarInfo(relative)
                    member.size, member.mode = len(data), 0o600
                    archive.addfile(member, io.BytesIO(data))
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return {"checkpoint": name, "files": len(files), "excluded": sorted(PRESERVED)}


def checkpoint_read(root, name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*\.tgz", name):
        raise ValueError("checkpoint must be a filename listed by checkpoint list")
    path = contained(root, "checkpoints/" + name)
    files = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            relative = Path(member.name)
            if member.name in files or relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"invalid or duplicate archive member: {member.name}")
            if relative.parts[0] in PRESERVED or not member.isfile():
                raise ValueError(f"archive member is not permitted: {member.name}")
            if relative.as_posix() != member.name:
                raise ValueError(f"noncanonical archive member: {member.name}")
            files[member.name] = archive.extractfile(member).read()
    if ARCHIVE_MANIFEST not in files:
        raise ValueError("checkpoint has no integrity manifest")
    manifest = json.loads(files.pop(ARCHIVE_MANIFEST), object_pairs_hook=unique_object)
    observed = {key: hashlib.sha256(value).hexdigest() for key, value in files.items()}
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("files") != observed:
        raise ValueError("checkpoint content integrity mismatch")
    directories = manifest.get("directories", [])
    modes = manifest.get("modes", {})
    if not isinstance(directories, list) or not isinstance(modes, dict):
        raise ValueError("invalid snapshot directory or mode inventory")
    for directory in directories:
        if not isinstance(directory, str):
            raise ValueError("invalid snapshot directory")
        relative = Path(directory)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] in PRESERVED or relative.as_posix() != directory:
            raise ValueError(f"invalid snapshot directory: {directory}")
    if any(name not in files or type(mode) is not int or mode < 0 or mode > 0o777 for name, mode in modes.items()):
        raise ValueError("invalid snapshot mode")
    return files, directories, modes


def checkpoint_restore(root, name, confirmed):
    if not confirmed:
        raise ValueError("restore requires --confirmed after researcher confirmation")
    archive_name = name
    files, directories, modes = checkpoint_read(root, name)
    # Validate and stage before touching current state. Data and checkpoints are
    # moved, never traversed or copied, and rollback restores the original root.
    stage = Path(tempfile.mkdtemp(prefix=".research-restore-", dir=root.parent))
    backup = Path(tempfile.mkdtemp(prefix=".research-previous-", dir=root.parent))
    backup.rmdir()
    moved = []
    swapped = False
    try:
        for directory in directories:
            (stage / directory).mkdir(parents=True, exist_ok=True)
        for relative, data in files.items():
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(modes.get(relative, 0o600))
        root.rename(backup)
        swapped = True
        for name in sorted(PRESERVED):
            if (backup / name).exists() or (backup / name).is_symlink():
                (backup / name).rename(stage / name)
                moved.append(name)
        stage.rename(root)
    except Exception:
        if swapped:
            for name in reversed(moved):
                (stage / name).rename(backup / name)
            backup.rename(root)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    shutil.rmtree(backup)
    return {"restored": archive_name, "files": len(files), "preserved": sorted(PRESERVED)}


def record(root, message, paths):
    if not config(root)["commit_research"]:
        return {"recorded": False, "reason": "commit_research is false"}
    project = root.parent
    def git(*argv, check=True):
        return subprocess.run(["git", "-C", str(project), *argv], text=True, capture_output=True, check=check)
    git("rev-parse", "--show-toplevel")
    if git("diff", "--cached", "--name-only").stdout.strip():
        raise ValueError("optional record refused: unrelated or preexisting staged changes")
    selected = []
    for value in paths:
        relative = Path(value)
        if not relative.parts or relative.parts[0] != ".research":
            raise ValueError("optional record accepts only explicit .research files")
        path = contained(root, Path(*relative.parts[1:]))
        if not path.is_file():
            raise ValueError("record needs named files, not directory staging")
        selected.append(relative.as_posix())
    if not selected:
        return {"recorded": False, "reason": "no changed files supplied"}
    git("add", "--", *selected)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return {"recorded": False, "reason": "no staged changes"}
    result = git("commit", "-m", message)
    return {"recorded": True, "files": selected, "detail": result.stdout.strip()}


def register_copy(root, source):
    source = Path(source).resolve(strict=True)
    if not source.is_file():
        raise ValueError("data registration needs an actual file")
    directory = contained(root, "data", allow_missing=True)
    directory.mkdir(exist_ok=True)
    target = directory / source.name
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if target.exists() or target.is_symlink():
        if target.is_file() and not target.is_symlink() and target.read_bytes() == data:
            return {"path": str(target), "source": str(source), "sha256": digest, "copied": False}
        raise ValueError(f"data filename collision: {target.name}; choose a distinct name")
    with target.open("xb") as output:
        output.write(data)
    return {"path": str(target), "source": str(source), "sha256": digest, "copied": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    cfg = sub.add_parser("config")
    cfg.add_argument("--key", choices=sorted(DEFAULTS))
    sub.add_parser("next-task-id")
    sub.add_parser("next-task")
    sub.add_parser("validate-workflow")
    sub.add_parser("task-dirs")
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("operation", choices=["save", "restore", "list"])
    checkpoint.add_argument("name", nargs="?", default="manual")
    checkpoint.add_argument("--confirmed", action="store_true")
    rec = sub.add_parser("record")
    rec.add_argument("--message", required=True)
    rec.add_argument("--files", nargs="+", required=True)
    copy = sub.add_parser("copy-data")
    copy.add_argument("source")
    args = parser.parse_args(argv)
    try:
        root = research_root(args.project)
        if args.action == "init":
            config(root)
            root.mkdir(exist_ok=True)
            for name in ("tasks", "data", "checkpoints"):
                contained(root, name, allow_missing=True).mkdir(exist_ok=True)
            if not (root / "config.json").exists():
                with (root / "config.json").open("x") as output:
                    output.write(json.dumps(DEFAULTS, indent=2) + "\n")
            result = {"initialized": True, "config": config(root)}
        elif args.action == "config":
            result = config(root)
            if args.key:
                result = result[args.key]
        elif args.action == "next-task-id":
            result = {"id": next_task(root)}
        elif args.action == "next-task":
            result = next_ready_task(root)
        elif args.action in ("validate-workflow", "task-dirs"):
            result = validate_workflow(root)
            if args.action == "task-dirs":
                for number in result["active"]:
                    contained(root, f"tasks/task-{number}", allow_missing=True).mkdir(parents=True, exist_ok=True)
        elif args.action == "checkpoint":
            if args.operation == "save":
                result = checkpoint_save(root, args.name)
            elif args.operation == "restore":
                result = checkpoint_restore(root, args.name, args.confirmed)
            else:
                directory = contained(root, "checkpoints", allow_missing=True)
                result = {"checkpoints": sorted(p.name for p in directory.glob("*.tgz") if p.is_file() and not p.is_symlink())}
        elif args.action == "record":
            result = record(root, args.message, args.files)
        else:
            result = register_copy(root, args.source)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, json.JSONDecodeError, tarfile.TarError, subprocess.CalledProcessError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
