"""A ParaView of a given version for this machine, fetched when none is installed.

To join a pvserver, the ParaView here must be the server's version. When the
connector names a version and this machine has no ParaView of that version,
the front fetches Kitware's official build once, checks its SHA-256 against the
pinned value, and keeps it in clio-kit's cache folder: ``CLIO_KIT_CACHE_DIR``
when the host sets one, else clio-kit's user cache. It is
``<cache>/paraview-client/<version>/``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .errors import ParaViewError


@dataclass(frozen=True)
class Build:
    url: str
    sha256: str
    size: int


#: Kitware's builds, by version and machine. Kitware publishes no checksums, so
#: each is pinned from a download checked against the server it joined.
BUILDS: dict[tuple[str, str], Build] = {
    ("5.13.1", "windows-x86_64"): Build(
        "https://www.paraview.org/files/v5.13/"
        "ParaView-5.13.1-Windows-Python3.10-msvc2017-AMD64.zip",
        "047720086bde6db7a63a236ed7bd19c6f41dc1ba5b557bfd24e256018350d890",
        1139829394,
    ),
}

_lock = threading.Lock()


def machine() -> str:
    system = platform.system().lower()
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    return f"{system}-{arch}"


def cache_root() -> Path:
    configured = os.environ.get("CLIO_KIT_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser() / "paraview-client"
    import platformdirs

    base = platformdirs.user_cache_dir("clio-kit", appauthor=False)
    return Path(base) / "paraview-client"


def _pvpython_in(folder: Path) -> Path | None:
    name = "pvpython.exe" if os.name == "nt" else "pvpython"
    found = sorted(folder.glob(f"*/bin/{name}")) or sorted(folder.glob(f"bin/{name}"))
    return found[0] if found else None


def installed(version: str) -> str | None:
    """The fetched pvpython of ``version``, if it is here."""
    folder = cache_root() / version
    if not (folder / "READY").is_file():
        return None
    found = _pvpython_in(folder)
    return str(found) if found else None


class Fetch:
    """One fetch of a ParaView build, run in the background.

    A build is a gigabyte or more, longer than a tool call should wait, so the
    first attach starts the fetch and says how far it has got; a later attach
    finds it done.
    """

    def __init__(self, version: str, build: Build) -> None:
        self.version = version
        self.build = build
        self.done_bytes = 0
        self.phase = "downloading"
        self.error: ParaViewError | None = None
        self.pvpython: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def describe(self) -> str:
        if self.phase == "downloading":
            return (
                f"downloading ParaView {self.version}: {self.done_bytes >> 20} of "
                f"{self.build.size >> 20} MB"
            )
        return f"unpacking ParaView {self.version}"

    def _download(self, target: Path) -> None:
        build = self.build
        digest = hashlib.sha256()
        part = target.with_suffix(".part")
        request = urllib.request.Request(build.url, headers={"User-Agent": "clio-kit"})
        with (
            urllib.request.urlopen(request, timeout=60) as reply,
            open(part, "wb") as out,
        ):
            while True:
                chunk = reply.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
                self.done_bytes += len(chunk)
        if digest.hexdigest() != build.sha256:
            part.unlink(missing_ok=True)
            raise ParaViewError(
                "client_checksum",
                f"The ParaView download from {build.url} did not match its pinned "
                "SHA-256, so it was not used.",
            )
        part.replace(target)

    def _run(self) -> None:
        root = cache_root()
        archive = root / f"{self.version}.zip"
        building = root / f"{self.version}.extracting"
        try:
            self._download(archive)
            self.phase = "unpacking"
            shutil.rmtree(building, ignore_errors=True)
            with zipfile.ZipFile(archive) as unpacked:
                unpacked.extractall(building)
            folder = root / self.version
            shutil.rmtree(folder, ignore_errors=True)
            building.rename(folder)
            (folder / "READY").write_text(self.build.sha256, "utf-8")
            self.pvpython = installed(self.version)
            if not self.pvpython:
                raise ParaViewError(
                    "client_fetch_failed",
                    f"The ParaView {self.version} fetched into {folder} has no pvpython.",
                )
        except ParaViewError as exc:
            self.error = exc
        except (OSError, zipfile.BadZipFile) as exc:
            self.error = ParaViewError(
                "client_fetch_failed",
                f"Could not fetch ParaView {self.version} from {self.build.url} into "
                f"{root}: {type(exc).__name__}: {exc}",
            )
        finally:
            shutil.rmtree(building, ignore_errors=True)
            archive.unlink(missing_ok=True)
            archive.with_suffix(".part").unlink(missing_ok=True)
            self.phase = "done"


_fetches: dict[str, Fetch] = {}


def _build_for(version: str) -> Build:
    build = BUILDS.get((version, machine()))
    if build is None:
        known = ", ".join(f"{v} ({m})" for v, m in sorted(BUILDS)) or "none"
        raise ParaViewError(
            "client_unavailable",
            f"This machine has no ParaView {version}, and clio-kit has no ParaView "
            f"{version} build to fetch for {machine()} (it has {known}). Install "
            f"ParaView {version} here and put its pvpython on PATH.",
        )
    return build


def start(version: str) -> Fetch | None:
    """Start fetching ParaView ``version`` unless it is here or on its way."""
    with _lock:
        if installed(version):
            return None
        fetch = _fetches.get(version)
        if fetch is not None and fetch.phase != "done":
            return fetch
        build = _build_for(version)
        root = cache_root()
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        if free < 3 * build.size:
            raise ParaViewError(
                "client_no_space",
                f"Fetching ParaView {version} needs about {(3 * build.size) >> 30} GB "
                f"free in {root}, and it has {free >> 30} GB.",
            )
        fetch = Fetch(version, build)
        _fetches[version] = fetch
        fetch.thread.start()
        return fetch


def ensure(version: str) -> str:
    """pvpython of ``version`` for this machine. The first call starts fetching it
    and raises ``client_fetching``; once it is here, its pvpython is returned."""
    found = installed(version)
    if found:
        return found
    with _lock:
        fetch = _fetches.get(version)
        if fetch is not None and fetch.phase == "done":
            del _fetches[version]
            if fetch.error is not None:
                raise fetch.error
            if fetch.pvpython:
                return fetch.pvpython
    fetch = start(version)
    if fetch is None:
        found = installed(version)
        if found:
            return found
        raise ParaViewError("client_fetch_failed", f"ParaView {version} is not here.")
    raise ParaViewError(
        "client_fetching",
        f"This machine has no ParaView {version}, which the ParaView server it joins "
        f"runs, so clio-kit is fetching Kitware's build into {cache_root()}: "
        f"{fetch.describe()}. Try again in a minute.",
    )
