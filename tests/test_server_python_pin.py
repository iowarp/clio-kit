"""Every Python server pins the interpreter its environment is built with.

The launcher builds a server with ``uv sync --frozen`` and no ``--python``, so
without a pin uv takes the newest managed Python it has, or downloads the
latest stable one. A Python newer than a lock's wheels (3.15 when h5py, numpy
or pandas had no cp315 wheels) turns the frozen sync into source builds, which
need a compiler: on a Mac without the Command Line Tools, and inside CLIO's
sandbox, the server never starts. ``.python-version`` in the project is what
``uv sync --project`` honours, and it travels with the server's files into the
materialised project and the wheel's shared data.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]
SERVERS = REPO / "clio-kit-mcp-servers"
QUALITY_CONTROL = REPO / ".github" / "workflows" / "quality_control.yml"

PYTHON_SERVERS = sorted(
    p.parent
    for p in SERVERS.glob("*/pyproject.toml")
    if (p.parent / "uv.lock").is_file()
)


def _pin(server: Path) -> str:
    return (server / ".python-version").read_text(encoding="utf-8").strip()


def test_python_servers_were_found() -> None:
    assert PYTHON_SERVERS


@pytest.mark.parametrize("server", PYTHON_SERVERS, ids=lambda p: p.name)
def test_server_pins_a_python_its_manifest_and_lock_accept(server: Path) -> None:
    assert (server / ".python-version").is_file(), (
        f"{server.name} has no .python-version: uv would build it with whatever "
        "Python is newest, including one its locked wheels do not cover"
    )
    pin = Version(_pin(server))
    manifest = tomllib.loads((server / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((server / "uv.lock").read_text(encoding="utf-8"))
    assert pin in SpecifierSet(manifest["project"]["requires-python"])
    assert pin in SpecifierSet(lock["requires-python"])


@pytest.mark.parametrize("server", PYTHON_SERVERS, ids=lambda p: p.name)
def test_server_pin_is_a_python_ci_tests_it_on(server: Path) -> None:
    # The server test matrix is the widest python-version list in the workflow.
    workflow = QUALITY_CONTROL.read_text(encoding="utf-8")
    lanes = re.findall(r"python-version: \[([^\]]*)\]", workflow)
    tested = max((re.findall(r"\"([^\"]+)\"", lane) for lane in lanes), key=len)
    assert _pin(server) in tested, (
        f"{server.name} pins Python {_pin(server)}, which no CI lane runs its tests on"
    )
