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
import time
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

#: How often the fetching front touches its lock, and how long an untouched lock
#: lasts before another front takes the fetch over.
BEAT_S = 5.0
STALE_S = 60.0


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

    A connector change restarts the front, so a fetch outlives the process that
    started it: the bytes land in a ``.part`` file that the next front resumes
    with an HTTP range, and a lock file the fetching front keeps fresh lets one
    front fetch while the others follow its progress. A lock nobody has touched
    for :data:`STALE_S` seconds belongs to a front that is gone, so another
    takes it over.
    """

    def __init__(self, version: str, build: Build) -> None:
        self.version = version
        self.build = build
        self.done_bytes = 0
        self.phase = "downloading"
        self.error: ParaViewError | None = None
        self.pvpython: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._beat = 0.0

    def describe(self) -> str:
        if self.phase == "downloading":
            return (
                f"downloading ParaView {self.version}: {self.done_bytes >> 20} of "
                f"{self.build.size >> 20} MB"
            )
        return f"unpacking ParaView {self.version}"

    # -- the lock one front holds while it fetches -------------------------------
    def _lock(self) -> Path:
        return cache_root() / f"{self.version}.lock"

    def _take(self) -> bool:
        lock = self._lock()
        for _ in range(2):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - lock.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age < STALE_S:
                    return False
                lock.unlink(missing_ok=True)  # its front is gone
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(str(os.getpid()))
            self._beat = time.monotonic()
            return True
        return False

    def _heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._beat >= BEAT_S:
            self._beat = now
            try:
                os.utime(self._lock())
            except OSError:
                pass

    def _follow(self) -> bool:
        """Follow another front's fetch until it ends; True when its build is here."""
        part = cache_root() / f"{self.version}.part"
        while True:
            if installed(self.version):
                return True
            try:
                self.done_bytes = part.stat().st_size
            except FileNotFoundError:
                pass
            try:
                age = time.time() - self._lock().stat().st_mtime
            except FileNotFoundError:
                return bool(installed(self.version))
            if age >= STALE_S:
                return False
            time.sleep(1.0)

    # -- fetching ----------------------------------------------------------------
    def _download(self, target: Path) -> None:
        build = self.build
        digest = hashlib.sha256()
        part = target.with_suffix(".part")
        have = 0
        if part.is_file():
            # Resume what an earlier front fetched; its bytes count toward the check.
            with open(part, "rb") as earlier:
                while chunk := earlier.read(1 << 20):
                    digest.update(chunk)
                    have += len(chunk)
                    self._heartbeat()
            if have > build.size:
                part.unlink()
                digest, have = hashlib.sha256(), 0
        self.done_bytes = have
        if have < build.size:
            headers = {"User-Agent": "clio-kit"}
            if have:
                headers["Range"] = f"bytes={have}-"
            request = urllib.request.Request(build.url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as reply:
                if have and reply.status != 206:
                    # The server sent the whole build: start the file over.
                    digest, have = hashlib.sha256(), 0
                    self.done_bytes = 0
                with open(part, "ab" if have else "wb") as out:
                    while chunk := reply.read(1 << 20):
                        digest.update(chunk)
                        out.write(chunk)
                        self.done_bytes += len(chunk)
                        self._heartbeat()
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
        try:
            while not self._take():
                if self._follow():
                    self.pvpython = installed(self.version)
                    return
            try:
                self._fetch(root)
            finally:
                self._lock().unlink(missing_ok=True)
        finally:
            self.phase = "done"

    def _fetch(self, root: Path) -> None:
        archive = root / f"{self.version}.zip"
        building = root / f"{self.version}.extracting"
        if installed(self.version):
            self.pvpython = installed(self.version)
            return
        if archive.is_file():
            # An earlier front stopped while unpacking: check the archive again.
            archive.replace(archive.with_suffix(".part"))
        try:
            self._download(archive)
            self.phase = "unpacking"
            shutil.rmtree(building, ignore_errors=True)
            with zipfile.ZipFile(archive) as unpacked:
                for member in unpacked.infolist():
                    unpacked.extract(member, building)
                    self._heartbeat()
            folder = root / self.version
            shutil.rmtree(folder, ignore_errors=True)
            building.rename(folder)
            (folder / "READY").write_text(self.build.sha256, "utf-8")
            archive.unlink(missing_ok=True)
            self.pvpython = installed(self.version)
            if not self.pvpython:
                raise ParaViewError(
                    "client_fetch_failed",
                    f"The ParaView {self.version} fetched into {folder} has no pvpython.",
                )
        except ParaViewError as exc:
            self.error = exc
        except zipfile.BadZipFile as exc:
            archive.unlink(missing_ok=True)
            self.error = ParaViewError(
                "client_fetch_failed",
                f"The ParaView {self.version} archive from {self.build.url} would not "
                f"unpack: {exc}",
            )
        except OSError as exc:
            # The bytes so far stay in the .part file, and the next attempt resumes.
            self.error = ParaViewError(
                "client_fetch_failed",
                f"Could not fetch ParaView {self.version} from {self.build.url} into "
                f"{root}: {type(exc).__name__}: {exc}. Fetched so far: "
                f"{self.done_bytes >> 20} of {self.build.size >> 20} MB, kept for the "
                "next try.",
            )
        finally:
            shutil.rmtree(building, ignore_errors=True)


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
