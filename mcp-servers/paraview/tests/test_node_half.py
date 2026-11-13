"""The node half under a real pvpython, attached the way the front attaches.

These tests need ParaView 6's ``pvpython``: on PATH, or named by
``PARAVIEW_TEST_PVPYTHON``. Without one they skip with the reason
``no_pvpython``. The first attach installs trame and trame-vtk, which needs
the package index; ``PARAVIEW_TEST_SITE`` may name a prepared site folder
to link instead.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from paraview_mcp import config
from paraview_mcp.attach import Attacher
from paraview_mcp.config import Settings, Target
from paraview_mcp.errors import ParaViewError

PVPYTHON = os.environ.get("PARAVIEW_TEST_PVPYTHON") or shutil.which("pvpython")
pytestmark = [
    pytest.mark.pvpython,
    pytest.mark.skipif(
        PVPYTHON is None, reason="no_pvpython: ParaView 6 pvpython not found"
    ),
]


def write_vti(path: Path, shift: float = 0.0, n: int = 8) -> None:
    """A small ASCII VTK image with point field ``p`` = x + y + z + shift."""
    values = " ".join(
        f"{x + y + z + shift:g}" for z in range(n) for y in range(n) for x in range(n)
    )
    extent = f"0 {n - 1} 0 {n - 1} 0 {n - 1}"
    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
<ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">
<Piece Extent="{extent}"><PointData Scalars="p">
<DataArray type="Float32" Name="p" format="ascii">{values}</DataArray>
</PointData></Piece></ImageData></VTKFile>
"""
    )


def _local(home: Path) -> Path:
    """The node half's folder on this machine for ``home`` (clio-kit's data folder)."""
    return home / ".local" / "share" / "clio-kit" / "paraview" / config.SERVER_VERSION


def _home(base: Path) -> Path:
    home = base / "home"
    folder = _local(home)
    folder.mkdir(parents=True)
    site = os.environ.get("PARAVIEW_TEST_SITE")
    if site:
        (folder / "site").symlink_to(site)
    return home


@pytest.fixture(scope="module")
def attached(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Attacher, Path]]:
    base = tmp_path_factory.mktemp("pv")
    home = _home(base)
    data = base / "data"
    data.mkdir()
    write_vti(data / "cube.vti")
    saved = {k: os.environ.get(k) for k in ("HOME", "PATH")}
    os.environ["HOME"] = str(home)
    assert PVPYTHON is not None
    os.environ["PATH"] = f"{Path(PVPYTHON).parent}:{os.environ['PATH']}"
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    attacher = Attacher(Settings(target=Target(), start_timeout=600))
    try:
        yield attacher, data
    finally:
        attacher.close()
        _stop(home)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _stop(home: Path) -> None:
    record = _local(home) / "node.json"
    if record.exists():
        pid = json.loads(record.read_text())["pid"]
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass


def _folder() -> Path:
    return _local(Path(os.environ["HOME"]))


def test_open_measure_and_rollback(attached: tuple[Attacher, Path]) -> None:
    attacher, data = attached
    link = attacher.ensure()
    client = link.client
    opened = client.command("open_view", {"path": str(data / "cube.vti"), "new": True})
    scene = opened["scene"]
    assert opened["objects"][0]["data"]["arrays"][0]["range"] == [0.0, 21.0]
    added = client.command(
        "create_filter",
        {
            "filter": "Contour",
            "properties": {"ContourBy": ["POINTS", "p"], "Isosurfaces": [10.5]},
        },
    )
    assert added["object"]["id"] == "o2"
    manifest_path = _folder() / "scenes" / scene / "manifest.json"
    before = json.loads(manifest_path.read_text())
    assert [o["id"] for o in before["objects"]] == ["o1", "o2"]

    with pytest.raises(ParaViewError) as caught:
        client.command(
            "create_filter", {"filter": "Contour", "properties": {"Bogus": 1}}
        )
    assert caught.value.code == "unknown_property"
    after = client.command("scene_state", {})
    assert [o["id"] for o in after["objects"]] == ["o1", "o2"]
    assert json.loads(manifest_path.read_text())["objects"] == before["objects"]

    measured = client.command("measure_field", {"field": "p", "bins": 4})
    assert measured["min"] == 0.0 and measured["max"] == 21.0
    assert measured["mean"] == pytest.approx(10.5)
    assert sum(measured["histogram"]["counts"]) == 512
    probe = client.command("inspect_selection", {"position": [1, 2, 3]})
    assert probe["selection"]["point_values"]["p"] == 6.0
    captured = client.command("capture", {"width": 120, "height": 90})
    assert captured["png_base64"].startswith("iVBOR")


def test_token_record_and_loopback(attached: tuple[Attacher, Path]) -> None:
    attacher, _ = attached
    link = attacher.ensure()
    record = _folder() / "node.json"
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    port = json.loads(record.read_text())["port"]
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/clio/health", timeout=10)
    assert refused.value.code == 401
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=10)
    # Bound to loopback only: another address of this machine cannot connect.
    outward = socket.gethostbyname(socket.gethostname())
    if not outward.startswith("127."):
        with socket.socket() as probe, pytest.raises(OSError):
            probe.settimeout(3)
            probe.connect((outward, port))
    assert link.client.health()["ok"]


def test_reattach_finds_the_running_node(attached: tuple[Attacher, Path]) -> None:
    attacher, _ = attached
    first = attacher.ensure().node["pid"]
    attacher.link.transport.close()  # type: ignore[union-attr]
    attacher.link = None
    started = time.monotonic()
    again = attacher.ensure()
    assert again.node["pid"] == first
    assert time.monotonic() - started < 60


def test_growing_series_follows_latest_and_parks(
    attached: tuple[Attacher, Path],
) -> None:
    attacher, data = attached
    client = attacher.ensure().client
    series = data / "series"
    series.mkdir()
    for step in range(2):
        write_vti(series / f"step_{step:03d}.vti", shift=step)
    opened = client.command(
        "open_view", {"path": str(series / "step_*.vti"), "growing": True, "new": True}
    )
    assert opened["time"]["steps"] == 2 and opened["time"]["follow_latest"]
    write_vti(series / "step_002.vti", shift=2)
    state = _wait_steps(client, 3)
    assert state["time"]["index"] == 2 and state["time"]["follow_latest"]
    parked = client.command("set_timestep", {"index": 0})
    assert parked["time"]["follow_latest"] is False
    write_vti(series / "step_003.vti", shift=3)
    state = _wait_steps(client, 4)
    assert state["time"]["index"] == 0
    assert state["time"]["newer_steps"] == 3


def _wait_steps(client, steps: int) -> dict:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = client.command("scene_state", {})
        if state["time"]["steps"] == steps:
            return state
        time.sleep(1)
    raise AssertionError(f"the series never reached {steps} steps")


def test_restart_restores_the_scene(attached: tuple[Attacher, Path]) -> None:
    attacher, data = attached
    client = attacher.ensure().client
    opened = client.command("open_view", {"path": str(data / "cube.vti")})
    scene = opened["scene"]
    client.command(
        "set_representation", {"object": "o2", "representation": "Wireframe"}
    )
    record = _folder() / "node.json"
    pid = json.loads(record.read_text())["pid"]
    os.kill(pid, 9)
    record.unlink()
    attacher.link = None
    client = attacher.ensure().client
    restored = client.command("open_view", {"scene": scene})
    assert restored["restored"] is True
    assert [o["id"] for o in restored["objects"]] == ["o1", "o2"]
    assert restored["objects"][1]["display"]["representation"] == "Wireframe"


def test_remote_mode_above_threshold(tmp_path: Path) -> None:
    home = _home(tmp_path)
    data = tmp_path / "cube.vti"
    write_vti(data, n=12)
    saved = os.environ["HOME"]
    os.environ["HOME"] = str(home)
    attacher = Attacher(
        Settings(target=Target(), threshold_mb=0.001, idle_seconds=4, start_timeout=600)
    )
    try:
        opened = attacher.ensure().client.command("open_view", {"path": str(data)})
        assert opened["view"]["mode"] == "remote"
        assert opened["view"]["geometry_mb"] > 0.001
        # Idle shutdown: with no link the node half saves and leaves.
        folder = _local(home)
        attacher.close()
        deadline = time.monotonic() + 120
        while (folder / "node.json").exists() and time.monotonic() < deadline:
            time.sleep(1)
        assert not (folder / "node.json").exists()
        assert (folder / "scenes" / opened["scene"] / "manifest.json").exists()
        assert "no link for 4 s" in (folder / "node.log").read_text()
    finally:
        attacher.close()
        _stop(home)
        os.environ["HOME"] = saved
