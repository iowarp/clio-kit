"""Find Avogadro 2 and the programs the streamed view needs. This server never installs
anything: the avogadro plugin's install-avogadro skill does, with the person's consent,
and points the connector's "Where Avogadro is" (AVOGADRO_COMMAND) at it.

Lookup order: ``AVOGADRO_COMMAND`` (an ``avogadro2`` executable or an unpacked AppImage's
``AppRun``), then ``avogadro2`` on PATH, then where the install skill unpacks it
(``~/.local/opt/avogadro-<version>/squashfs-root/AppRun``, newest first), so a release
installed in one conversation is found in every other. The release the skill installs is
``VERSION``.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any

VERSION = "2.1.0"
ENV = "AVOGADRO_COMMAND"


def find() -> dict[str, Any]:
    """Where Avogadro is, how it was found, or that it is missing."""
    value = os.environ.get(ENV, "").strip()
    if value:
        path = shutil.which(value) or (value if Path(value).is_file() else None)
        if path and os.access(path, os.X_OK):
            return {"found": True, "command": str(Path(path).resolve()), "source": ENV}
        return {
            "found": False,
            "source": ENV,
            "problem": f"Where Avogadro is ({ENV}={value!r}) is not an executable file",
        }
    path = shutil.which("avogadro2")
    if path:
        return {"found": True, "command": path, "source": "PATH"}
    installed = skill_installs()
    if installed:
        return {"found": True, "command": str(installed[0]), "source": "install-avogadro"}
    return {"found": False, "source": None, "problem": "no avogadro2 found on this machine"}


def missing_text(found: dict[str, Any]) -> str:
    """Where Avogadro was looked for and the connector setting that points at it."""
    where = (
        f"Avogadro 2 was not found: {found['problem']}"
        if found.get("source")
        else f"Avogadro 2 was not found in Where Avogadro is ({ENV}), PATH (avogadro2) or "
        "~/.local/opt/avogadro-*"
    )
    return (
        f"{where}. If it is installed elsewhere, set the connector's Where Avogadro is; "
        f"otherwise see the avogadro plugin's install-avogadro skill (Avogadro {VERSION}, "
        "with the person's consent)."
    )


def skill_installs(home: Path | None = None) -> list[Path]:
    """Unpacked releases the install skill left in ``~/.local/opt``, newest version first."""

    def version(p: Path) -> tuple[int, ...]:
        tail = p.parents[1].name.removeprefix("avogadro-")
        return tuple(int(x) if x.isdigit() else 0 for x in tail.split("."))

    root = (home or Path.home()) / ".local" / "opt"
    runs = [p for p in root.glob("avogadro-*/squashfs-root/AppRun") if os.access(p, os.X_OK)]
    return sorted(runs, key=version, reverse=True)


def platform_supported() -> tuple[bool, str]:
    system, machine = platform.system(), platform.machine().lower()
    if system != "Linux":
        return False, (
            f"The streamed Avogadro view runs on Linux (a virtual display streamed to the "
            f"view); this machine is {system}. Open structures in Avogadro on the desktop "
            "instead, or run CLIO's service on a Linux machine."
        )
    if machine not in ("x86_64", "amd64"):
        return False, f"Avogadro's Linux build is for x86-64; this machine is {machine}."
    return True, ""


def display_tools() -> dict[str, Any]:
    """The virtual display and VNC programs the streamed view needs on Linux."""
    found = {name: shutil.which(name) for name in ("Xvfb", "x11vnc")}
    missing = [n for n, p in found.items() if not p]
    return {
        "found": {n: p for n, p in found.items() if p},
        "missing": missing,
        "install": (
            "Install them with the system's package manager (the avogadro plugin's "
            "install-avogadro skill), for example `apt-get install -y xvfb x11vnc` (Debian, "
            "Ubuntu) or `dnf install -y xorg-x11-server-Xvfb x11vnc` (Fedora)."
            if missing
            else ""
        ),
    }
