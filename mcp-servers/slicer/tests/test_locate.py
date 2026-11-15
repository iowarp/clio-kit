"""Finding the installed slicer, and the typed ``slicer_missing`` answer when there is none."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from slicer_mcp.backends import locate as locate_module
from slicer_mcp.backends.locate import ORCA, PRUSA, find_profiles, locate
from slicer_mcp.errors import SlicerError
from slicer_mcp.server import mcp


def fake_slicer(folder: Path, name: str, banner: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(f"#!/bin/sh\necho '{banner}'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def bare_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A PATH with no slicer on it, no SLICER_BIN, no known install folders."""
    empty = tmp_path / "bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("SLICER_BIN", raising=False)
    monkeypatch.delenv("SLICER_PROFILES", raising=False)
    monkeypatch.setattr(locate_module, "KNOWN_LOCATIONS", ())
    locate_module._describe.cache_clear()
    return empty


def test_no_slicer_is_a_typed_slicer_missing_in_words(bare_path: Path) -> None:
    with pytest.raises(SlicerError) as caught:
        locate()
    error = caught.value
    assert error.reason == "slicer_missing"
    text = str(error)
    for words in ("PrusaSlicer", "OrcaSlicer", "SLICER_BIN", "never downloads", "prusa-slicer"):
        assert words in text
    assert error.details["looked_for"] == [
        "prusa-slicer",
        "PrusaSlicer",
        "orca-slicer",
        "OrcaSlicer",
    ]


async def test_every_tool_says_slicer_missing_to_the_agent(bare_path: Path) -> None:
    async with Client(mcp) as client:
        for name, args in (
            ("list_printers", {}),
            ("slice", {"model": "/nowhere/part.stl", "printer": "Prusa MK4"}),
        ):
            with pytest.raises(ToolError) as caught:
                await client.call_tool(name, args)
            assert "[slicer_missing]" in str(caught.value)
            assert "PrusaSlicer or OrcaSlicer" in str(caught.value)


def test_a_wrong_slicer_bin_is_not_silently_replaced_by_path(
    bare_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_slicer(bare_path, "prusa-slicer", "PrusaSlicer-2.7.2+UNKNOWN based on Slic3r")
    monkeypatch.setenv("SLICER_BIN", "/opt/nothing/prusa-slicer")
    with pytest.raises(SlicerError) as caught:
        locate()
    assert caught.value.reason == "slicer_missing"
    assert "/opt/nothing/prusa-slicer" in str(caught.value)


def test_slicer_bin_must_answer_as_a_slicer(
    bare_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = fake_slicer(bare_path / "other", "cura", "UltiMaker Cura 5")
    monkeypatch.setenv("SLICER_BIN", str(other))
    with pytest.raises(SlicerError) as caught:
        locate()
    assert caught.value.reason == "slicer_missing"
    assert "did not answer as PrusaSlicer or OrcaSlicer" in str(caught.value)


def test_path_order_prefers_prusaslicer_and_reads_the_version(bare_path: Path) -> None:
    fake_slicer(bare_path, "orca-slicer", "OrcaSlicer-02.01.01 based on BambuStudio")
    fake_slicer(bare_path, "prusa-slicer", "PrusaSlicer-2.7.2+UNKNOWN based on Slic3r")
    found = locate()
    assert (found.kind, found.version, found.binary.name) == (PRUSA, "2.7.2", "prusa-slicer")
    os.remove(bare_path / "prusa-slicer")
    locate_module._describe.cache_clear()
    found = locate()
    assert (found.kind, found.version) == (ORCA, "02.01.01")
    assert found.describe() == "OrcaSlicer 02.01.01"


def test_slicer_bin_and_profiles_folder(bare_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = bare_path.parent / "PrusaSlicer"
    binary = fake_slicer(root / "bin", "prusa-slicer", "PrusaSlicer-2.8.1")
    profiles = root / "resources" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "Vendor.ini").write_text("[vendor]\nname = V\n")
    monkeypatch.setenv("SLICER_BIN", str(binary))
    found = locate()
    assert found.profiles == profiles
    monkeypatch.setenv("SLICER_PROFILES", str(bare_path / "missing"))
    assert find_profiles(PRUSA, binary) is None
