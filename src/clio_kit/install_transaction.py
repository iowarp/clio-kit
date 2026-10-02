"""Stage project changes and restore every previous target on installation errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile


@dataclass
class Replacement:
    target: Path
    staging: Path
    previous: Path
    installed: bool = False
    remove: bool = False


class InstallTransaction:
    """Rollback-capable multi-file update, using same-filesystem staging.

    Atomic rename protects each target. Exceptions and interrupts roll back the
    whole operation; abrupt process termination or power loss is not atomic
    across files. If restoration fails, retain backups and report their paths.
    """

    def __init__(self) -> None:
        self.changes: list[Replacement] = []
        self.directories: list[Path] = []
        self.created_parents: list[Path] = []
        self.retain_backups = False

    def __enter__(self):
        return self

    def _stage(self, target: Path) -> Path:
        missing = []
        parent = target.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        target.parent.mkdir(parents=True, exist_ok=True)
        self.created_parents.extend(reversed(missing))
        directory = Path(tempfile.mkdtemp(prefix=".clio-install-", dir=target.parent))
        self.directories.append(directory)
        staging = directory / "next"
        self.changes.append(Replacement(target, staging, directory / "previous"))
        return staging

    def directory(self, target: Path, source: Path) -> None:
        shutil.copytree(source, self._stage(target))

    def file(self, target: Path, content: bytes) -> None:
        staging = self._stage(target)
        staging.write_bytes(content)
        staging.chmod(target.stat().st_mode & 0o777 if target.exists() else 0o600)

    def remove(self, target: Path) -> None:
        self._stage(target)
        self.changes[-1].remove = True

    def commit(self) -> None:
        try:
            for change in self.changes:
                if change.target.exists():
                    change.target.rename(change.previous)
                if not change.remove:
                    change.staging.replace(change.target)
                    change.installed = True
        except BaseException:
            failures = []
            for change in reversed(self.changes):
                try:
                    if change.installed:
                        if change.target.is_dir():
                            shutil.rmtree(change.target)
                        else:
                            change.target.unlink()
                    if change.previous.exists():
                        change.previous.rename(change.target)
                except OSError as exc:
                    failures.append(
                        f"{change.target}: {exc}; backup: {change.previous}"
                    )
            if failures:
                self.retain_backups = True
                raise RuntimeError(
                    "Installation rollback needs recovery; staging retained: "
                    + "; ".join(failures)
                )
            raise

    def __exit__(self, *_exc) -> None:
        if self.retain_backups:
            return
        for directory in self.directories:
            shutil.rmtree(directory)
        for parent in reversed(self.created_parents):
            try:
                parent.rmdir()  # Only remove empty directories created by this install.
            except OSError:
                pass
