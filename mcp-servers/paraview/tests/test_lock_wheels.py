"""The server's first provision installs wheels only, on every OS.

A package with no wheel for a platform makes uv build it from source there: adios2
has no Windows wheels, and building it took a first provision past seven minutes.
ADIOS2 BP files are read by ParaView's own reader (ADIOS2VTXReader, shipped in
Kitware's builds for Windows, macOS and Linux), so the front never needs it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

LOCK = Path(__file__).resolve().parents[1] / "uv.lock"
PLATFORMS = {
    "windows": r"win_amd64",
    "macos": r"macosx_",
    "linux": r"manylinux|musllinux",
}
#: Packages the lock installs on one OS only, by their environment marker.
ONE_OS = {"pywin32": "windows"}


def _packages() -> list[dict]:
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    return [
        package
        for package in lock["package"]
        if not {"editable", "virtual"} & set(package.get("source", {}))
    ]


def test_every_locked_package_has_a_wheel_on_every_os() -> None:
    missing: dict[str, list[str]] = {}
    for package in _packages():
        wheels = [wheel["url"].rsplit("/", 1)[-1] for wheel in package.get("wheels", [])]
        universal = any(name.endswith("-none-any.whl") for name in wheels)
        wanted = [ONE_OS[package["name"]]] if package["name"] in ONE_OS else PLATFORMS
        gaps = [
            system
            for system in wanted
            if not universal
            and not any(re.search(PLATFORMS[system], w) for w in wheels)
        ]
        if gaps:
            missing[f"{package['name']} {package['version']}"] = gaps
    assert not missing, f"these would build from source on first provision: {missing}"


def test_the_front_does_not_install_adios2() -> None:
    names = {package["name"] for package in _packages()}
    assert "adios2" not in names
