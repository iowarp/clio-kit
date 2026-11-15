"""Take a component from the source this launcher was built from.

A launcher built from a Git commit or a source directory carries a catalogue whose
components may not be published as release assets yet: the commit is ahead of the
release its version names. The installation records its origin (PEP 610
``direct_url.json``) and the catalogue records each artifact's source folder, so an
unpublished component's files can be read from that exact origin. They are accepted
only when every file matches the catalogue's size and SHA-256, as a downloaded
archive is; a branch name or a moving reference is never followed.
"""

from __future__ import annotations

import json
import os
from importlib import metadata
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from typing import NamedTuple, TypeGuard
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname, urlopen
import tarfile

from clio_kit.environment_locks import _FileLock

MAX_SOURCE_BYTES = 256 * 1024 * 1024
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GITHUB = re.compile(
    r"https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?"
)


class Source(NamedTuple):
    """Where this installation was built from: a local folder or a Git commit."""

    kind: str  # "directory" or "git"
    location: str  # a folder path, or the repository URL
    commit: str | None = None

    def describe(self) -> str:
        if self.kind == "git":
            return f"its source commit {self.commit} of {self.location}"
        return f"its source folder {self.location}"


def source_from_direct_url(text: str | None) -> Source | None:
    """Read a PEP 610 record; only a pinned Git commit or a non-editable folder counts."""
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("url"), str):
        return None
    url = data["url"]
    vcs = data.get("vcs_info")
    if isinstance(vcs, dict):
        commit = vcs.get("commit_id")
        if (
            vcs.get("vcs") == "git"
            and isinstance(commit, str)
            and _COMMIT.fullmatch(commit)
        ):
            return Source("git", url, commit)
        return None
    directory = data.get("dir_info")
    parsed = urlparse(url)
    if (
        isinstance(directory, dict)
        and not directory.get("editable")
        and parsed.scheme == "file"
    ):
        return Source("directory", url2pathname(unquote(parsed.path)))
    return None


def installed_source() -> Source | None:
    """The origin of the installed ``clio-kit`` distribution, when it was a source build."""
    try:
        text = metadata.distribution("clio-kit").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    return source_from_direct_url(text)


def archive_url(source: Source) -> str | None:
    """GitHub's archive of the commit; ``None`` for another Git host (read with git)."""
    match = _GITHUB.fullmatch(source.location)
    if match is None:
        return None
    return (
        f"https://codeload.github.com/{match['owner']}/{match['repo']}"
        f"/tar.gz/{source.commit}"
    )


def _git_archive(source: Source, destination: Path) -> None:
    """Fetch just the commit with git and write its archive (any Git host or local repo)."""
    git = shutil.which("git")
    if git is None:
        raise ValueError(
            f"Reading unpublished components from {source.location} needs git on PATH; "
            "publish the release assets or configure a mirror"
        )
    with tempfile.TemporaryDirectory(prefix=".git-", dir=destination.parent) as work:
        commands = (
            [git, "init", "-q", work],
            [
                git,
                "-C",
                work,
                "fetch",
                "-q",
                "--depth",
                "1",
                source.location,
                str(source.commit),
            ],
            [
                git,
                "-C",
                work,
                "archive",
                "--format=tar.gz",
                f"--prefix=source-{source.commit}/",
                "-o",
                str(destination),
                "FETCH_HEAD",
            ],
        )
        for command in commands:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=600, check=False
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip().splitlines()
                raise ValueError(
                    f"Cannot fetch commit {source.commit} from {source.location}: "
                    f"{detail[-1] if detail else 'git failed'}"
                )
    if destination.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Source archive exceeds its download limit")


def _safe_folder(folder: object) -> TypeGuard[str]:
    if not isinstance(folder, str) or not folder or "\\" in folder or ":" in folder:
        return False
    path = PurePosixPath(folder)
    return not path.is_absolute() and ".." not in path.parts and str(path) == folder


def source_archive(source: Source, cache_root: Path) -> Path:
    """Download the commit's archive once; only the installed commit's archive is kept."""
    assert source.commit is not None
    directory = cache_root / "sources"
    target = directory / f"{source.commit}.tar.gz"
    if target.is_file():
        return target
    directory.mkdir(parents=True, exist_ok=True)
    lock = _FileLock(directory / ".download.lock")
    deadline = time.monotonic() + 600
    while not lock.try_acquire():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Another process is downloading {source.commit}")
        time.sleep(0.1)
    try:
        if target.is_file():
            return target
        url = archive_url(source)
        if url is None:
            partial = directory / f"{source.commit}.part"
            try:
                _git_archive(source, partial)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            return _keep_only(partial, target)
        parsed = urlparse(url)
        if parsed.scheme not in {"https", "file"}:
            raise ValueError("A source archive must use HTTPS or a local file URI")
        total = 0
        with tempfile.NamedTemporaryFile(
            dir=directory, suffix=".part", delete=False
        ) as out:
            partial = Path(out.name)
            try:
                with urlopen(url, timeout=120) as response:
                    if (
                        parsed.scheme == "https"
                        and urlparse(response.geturl()).scheme != "https"
                    ):
                        raise ValueError("Refusing insecure source archive redirect")
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_SOURCE_BYTES:
                            raise ValueError(
                                "Source archive exceeds its download limit"
                            )
                        out.write(chunk)
            except BaseException:
                out.close()
                partial.unlink(missing_ok=True)
                raise
        return _keep_only(partial, target)
    finally:
        lock.release()


def _keep_only(partial: Path, target: Path) -> Path:
    """Install the new archive; one installation uses one commit, older ones are obsolete."""
    for old in target.parent.glob("*.tar.gz"):
        old.unlink(missing_ok=True)
    os.replace(partial, target)
    return target


def _write(destination: Path, name: str, content: bytes, expected: dict) -> None:
    metadata_ = expected[name]
    if len(content) != metadata_["size"]:
        raise ValueError(f"Source file {name} differs from the catalogue")
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(metadata_["mode"])


def materialize(
    record: dict, source: Source, destination: Path, cache_root: Path
) -> None:
    """Write the artifact's files from ``source`` into the new folder ``destination``.

    The caller verifies the folder against the catalogue before using it.
    """
    from clio_kit.component_store import _safe_name

    folder = record.get("source")
    if not _safe_folder(folder):
        raise ValueError(
            "This catalogue does not name the component's source folder; publish the "
            "release assets or configure a mirror"
        )
    expected = record["files"]
    if not all(_safe_name(name) for name in expected):
        raise ValueError("Unsafe component file name in the catalogue")
    destination.mkdir()
    written: set[str] = set()
    if source.kind == "directory":
        root = Path(source.location)
        for name in expected:
            path = root / folder / name
            if name == "LICENSE" and not path.is_file():
                path = root / "LICENSE"  # The builder adds the repository licence.
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Source folder lacks {folder}/{name}")
            _write(destination, name, path.read_bytes(), expected)
            written.add(name)
        return
    root_licence: bytes | None = None
    with tarfile.open(source_archive(source, cache_root), "r:gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 2:
                continue
            relative = PurePosixPath(
                *parts[1:]
            ).as_posix()  # Drop the archive's top folder.
            if relative == "LICENSE" and "LICENSE" in expected:
                stream = tar.extractfile(member)
                assert stream is not None
                root_licence = stream.read(MAX_SOURCE_BYTES)
                continue
            if not relative.startswith(folder + "/"):
                continue
            name = relative[len(folder) + 1 :]
            if name not in expected or name in written:
                continue
            if member.size != expected[name]["size"]:
                raise ValueError(f"Source file {name} differs from the catalogue")
            stream = tar.extractfile(member)
            assert stream is not None
            _write(destination, name, stream.read(), expected)
            written.add(name)
    if "LICENSE" in expected and "LICENSE" not in written and root_licence is not None:
        _write(destination, "LICENSE", root_licence, expected)
        written.add("LICENSE")
    missing = sorted(set(expected) - written)
    if missing:
        raise ValueError(f"Source commit lacks {folder}/{missing[0]}")
