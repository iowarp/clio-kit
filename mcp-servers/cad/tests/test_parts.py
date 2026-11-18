"""Building parts in their own process and reading the meshes the view draws."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from cad_mcp import parts
from cad_mcp.examples import EXAMPLES

pytest.importorskip("build123d")


@pytest.fixture(autouse=True)
def parts_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CAD_PARTS_DIR", str(tmp_path / "parts"))
    return tmp_path / "parts"


def test_box_builds_with_its_size_volume_and_files(parts_home: Path) -> None:
    rec = parts.build("box", "result = Box(2, 1, 0.5)")
    assert rec["valid"] and rec["solids"] == 1
    assert rec["size"] == pytest.approx([2.0, 1.0, 0.5])
    assert rec["volume"] == pytest.approx(1.0)
    assert Path(rec["step"]).stat().st_size > 0
    assert rec["triangles"] == 12
    corners = parts.read_stl(Path(rec["stl"])).reshape(-1, 3)
    assert corners.min(axis=0) == pytest.approx([-1.0, -0.5, -0.25])
    assert parts.resolve("box") == Path(rec["part"])
    assert [p["name"] for p in parts.listing()] == ["box"]


def test_a_failing_script_reports_its_error() -> None:
    with pytest.raises(parts.PartError) as error:
        parts.build("broken", "result = Box(1, 1)\n")
    assert error.value.code == "script_failed"
    assert "Box" in str(error.value)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_examples_build_one_valid_solid(name: str) -> None:
    rec = parts.build(name, EXAMPLES[name]["code"])
    assert rec["valid"] and rec["solids"] == 1
    assert rec["volume"] > 0 and rec["triangles"] > 0


def test_ascii_and_binary_stl_read_the_same(tmp_path: Path) -> None:
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    binary = tmp_path / "one.stl"
    binary.write_bytes(
        b" " * 80 + struct.pack("<I", 1) + np.zeros(3, "<f4").tobytes() + tri.tobytes() + b"\0\0"
    )
    ascii_stl = tmp_path / "one-ascii.stl"
    ascii_stl.write_text(
        "solid one\nfacet normal 0 0 1\nouter loop\n"
        + "".join(f"vertex {x} {y} {z}\n" for x, y, z in tri)
        + "endloop\nendfacet\nendsolid one\n"
    )
    assert parts.read_stl(binary) == pytest.approx(tri.reshape(-1))
    assert parts.read_stl(ascii_stl) == pytest.approx(tri.reshape(-1))
    assert parts.stl_triangles(binary) == parts.stl_triangles(ascii_stl) == 1
