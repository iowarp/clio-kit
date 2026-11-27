"""The node half under a real pvpython, attached the way the front attaches.

These tests need ParaView 6's ``pvpython``: on PATH, or named by
``PARAVIEW_TEST_PVPYTHON``. Without one they skip with the reason
``no_pvpython``. The first attach installs trame and trame-vtk, which needs
the package index; ``PARAVIEW_TEST_SITE`` may name a prepared site folder
to link instead.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
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
from paraview_mcp.node import ranges

PVPYTHON = os.environ.get("PARAVIEW_TEST_PVPYTHON") or shutil.which("pvpython")
# The fixtures write each step whole before the follow looks, so the node reads a
# file as soon as it is there rather than waiting for it to settle.
os.environ.setdefault("PARAVIEW_FILE_SETTLE", "0")
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


def write_vtr(path: Path, n: int = 6) -> None:
    """A small ASCII rectilinear grid with uneven z and point field ``p`` = x + y + z."""
    values = " ".join(
        f"{x + y + z:g}" for z in range(n) for y in range(n) for x in range(n)
    )
    axis = " ".join(str(i) for i in range(n))
    z = " ".join(f"{i * i:g}" for i in range(n))
    extent = f"0 {n - 1} 0 {n - 1} 0 {n - 1}"
    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
<RectilinearGrid WholeExtent="{extent}"><Piece Extent="{extent}">
<PointData Scalars="p"><DataArray type="Float32" Name="p" format="ascii">{values}</DataArray>
</PointData><Coordinates>
<DataArray type="Float64" Name="x" format="ascii">{axis}</DataArray>
<DataArray type="Float64" Name="y" format="ascii">{axis}</DataArray>
<DataArray type="Float64" Name="z" format="ascii">{z}</DataArray>
</Coordinates></Piece></RectilinearGrid></VTKFile>
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
    write_vtr(data / "grid.vtr")
    saved = {k: os.environ.get(k) for k in ("HOME", "PATH")}
    os.environ["HOME"] = str(home)
    assert PVPYTHON is not None
    os.environ["PATH"] = f"{Path(PVPYTHON).parent}:{os.environ['PATH']}"
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    attacher = Attacher(Settings(target=Target(), start_stall=600))
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
    # An array the property's own domain does not offer is refused, the way a
    # Contour given a vector would be, rather than read past its end.
    with pytest.raises(ParaViewError) as caught:
        client.command(
            "create_filter",
            {"filter": "Contour", "properties": {"ContourBy": ["POINTS", "q"]}},
        )
    assert caught.value.code == "bad_value" and "'p'" in str(caught.value)
    after = client.command("scene_state", {})
    assert [o["id"] for o in after["objects"]] == ["o1", "o2"]
    assert json.loads(manifest_path.read_text())["objects"] == before["objects"]

    changed = client.command(
        "set_filter_property", {"object": "o2", "properties": {"Isosurfaces": [15.5]}}
    )
    assert changed["before"] == {"Isosurfaces": [10.5]}
    edited = json.loads(manifest_path.read_text())["objects"][1]
    assert edited["properties"]["Isosurfaces"] == [15.5]
    with pytest.raises(ParaViewError) as refused:
        client.command("set_filter_property", {"object": "o1", "properties": {"X": 1}})
    assert refused.value.code == "not_a_filter"
    with pytest.raises(ParaViewError) as refused:
        client.command(
            "set_filter_property", {"object": "o2", "properties": {"Bogus": 1}}
        )
    assert refused.value.code == "unknown_property"
    assert json.loads(manifest_path.read_text())["objects"][1]["properties"][
        "Isosurfaces"
    ] == [15.5]
    measured = client.command("measure_field", {"field": "p", "bins": 4})
    assert measured["min"] == 0.0 and measured["max"] == 21.0
    assert measured["mean"] == pytest.approx(10.5)
    assert sum(measured["histogram"]["counts"]) == 512
    probe = client.command("inspect_selection", {"position": [1, 2, 3]})
    assert probe["selection"]["point_values"]["p"] == 6.0
    captured = client.command("capture", {"width": 120, "height": 90})
    assert captured["png_base64"].startswith("iVBOR")
    # The capture was timed, and so were its frames, so a slow view can say so.
    assert isinstance(captured["seconds"], float)
    assert isinstance(client.command("scene_state", {})["view"]["frame_seconds"], float)


def test_a_second_front_shares_the_node_half(attached: tuple[Attacher, Path]) -> None:
    """A connector change can leave the superseded server running beside the new
    one while its view is open: the new front, even with other knobs, links to the
    same node half instead of stopping it, so the open view keeps drawing."""
    attacher, data = attached
    first = attacher.ensure(opening=True)
    opened = first.client.command("open_view", {"path": str(data / "cube.vti")})
    other = Attacher(Settings(target=Target(), start_stall=600, idle_seconds=999))
    try:
        second = other.ensure(opening=True)
        assert second.node["pid"] == first.node["pid"]
        assert second.client.command("scene_state", {})["scene"] == opened["scene"]
    finally:
        other.close()
    again = attacher.ensure()
    assert again.node["pid"] == first.node["pid"]
    assert again.client.command("scene_state", {})["scene"] == opened["scene"]


def test_a_rectilinear_grid_is_measured_whole(attached: tuple[Attacher, Path]) -> None:
    attacher, data = attached
    client = attacher.ensure().client
    client.command("open_view", {"path": str(data / "grid.vtr"), "new": True})
    measured = client.command("measure_field", {"field": "p", "bins": 3})
    # Every one of the 216 values, not the half a fetched rectilinear grid carried.
    assert sum(measured["histogram"]["counts"]) == 216
    assert measured["max"] == 15.0
    assert measured["mean"] == pytest.approx(7.5)


def test_slice_plane_is_set_through_its_part(attached: tuple[Attacher, Path]) -> None:
    attacher, data = attached
    client = attacher.ensure().client
    # Its own file, so its own scene: the cube's scene is kept for the restart test.
    write_vti(data / "slice.vti")
    client.command("open_view", {"path": str(data / "slice.vti"), "new": True})
    dotted = client.command(
        "create_filter",
        {
            "filter": "Slice",
            "input": "o1",
            "properties": {
                "SliceType": "Plane",
                "SliceType.Origin": [3.5, 3.5, 3.5],
                "SliceType.Normal": [0, 0, 1],
            },
        },
    )
    # A plane normal to z through z = 3.5 crosses p = x + y + z at 3.5 + x + y.
    assert dotted["object"]["data"]["arrays"][0]["range"] == [3.5, 17.5]
    nested = client.command(
        "create_filter",
        {
            "filter": "Slice",
            "input": "o1",
            "properties": {"SliceType": {"Origin": [2, 2, 2], "Normal": [1, 0, 0]}},
        },
    )
    assert nested["object"]["data"]["arrays"][0]["range"] == [2.0, 16.0]
    with pytest.raises(ParaViewError) as caught:
        client.command(
            "create_filter",
            {"filter": "Slice", "input": "o1", "properties": {"SliceType.Bogus": 1}},
        )
    assert caught.value.code == "unknown_property"
    assert "Slice.SliceType has no property 'Bogus'" in str(caught.value)
    scene = client.command("scene_state", {})
    assert [o["id"] for o in scene["objects"]] == ["o1", "o2", "o3"]


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


def test_a_view_descriptor_names_the_reader_and_steps_report_growth(
    attached: tuple[Attacher, Path],
) -> None:
    """RL44: the descriptor's reader reads a growing series; ``steps`` re-reads its
    time list at once, and a field is measured at every step afterwards."""
    attacher, data = attached
    client = attacher.ensure().client
    series = data / "described"
    series.mkdir()
    for step in range(2):
        write_vti(series / f"out_{step:03d}.vti", shift=step)
    pattern = str(series / "out_*.vti")
    with pytest.raises(ParaViewError) as caught:
        client.command(
            "open_view",
            {"path": pattern, "reader": "XMLImageDataRaeder", "new": True},
        )
    assert caught.value.code == "unknown_reader"
    assert "XMLImageDataReader" in str(caught.value)
    opened = client.command(
        "open_view",
        {
            "path": pattern,
            "growing": True,
            "new": True,
            "format": "vti",
            "reader": "XMLImageDataReader",
            "reader_properties": {"PointArrayStatus": ["p"]},
        },
    )
    assert opened["dataset"]["reader"] == "XMLImageDataReader"
    assert opened["dataset"]["format"] == "vti"
    assert opened["time"]["steps"] == 2
    first = client.command("steps", {"after": 0})
    assert [s["index"] for s in first["steps"]] == [0, 1]
    assert all(s["seen"] for s in first["steps"]) and first["growing"]
    write_vti(series / "out_002.vti", shift=2)
    grown = client.command("steps", {"after": 2})
    assert [s["index"] for s in grown["steps"]] == [2]
    assert grown["time"]["index"] == 2 and grown["time"]["follow_latest"]
    assert grown["last_growth"] is not None
    client.command("set_timestep", {"index": 0})
    write_vti(series / "out_003.vti", shift=3)
    parked = client.command("steps", {"after": 3})
    assert [s["index"] for s in parked["steps"]] == [3]
    assert parked["time"]["index"] == 0 and parked["time"]["newer_steps"] == 3
    measured = client.command("measure_over_time", {"field": "p"})
    means = [round(row["mean"], 3) for row in measured["series"]]
    assert means == [10.5, 11.5, 12.5, 13.5]
    assert [row["index"] for row in measured["series"]] == [0, 1, 2, 3]
    assert client.command("scene_state", {})["time"]["index"] == 0
    strided = client.command(
        "measure_over_time", {"field": "p", "start": 1, "stride": 2}
    )
    assert strided["series"] and [row["index"] for row in strided["series"]] == [1, 3]


def test_a_long_measure_reports_its_steps_and_is_cancelled_on_the_node(
    attached: tuple[Attacher, Path],
) -> None:
    """A call has no time limit: the node runs it as a job, the front asks after it
    and hears how many steps are measured, and a cancel stops it on the node, which
    then serves the next call as before."""
    import threading

    attacher, data = attached
    client = attacher.ensure().client
    series = data / "long"
    series.mkdir()
    for step in range(30):
        write_vti(series / f"out_{step:03d}.vti", shift=step)
    client.command("open_view", {"path": str(series / "out_*.vti"), "new": True})
    client.command("set_timestep", {"index": 4})
    seen: list[dict] = []
    measured = client.command(
        "measure_over_time", {"field": "p"}, progress=seen.append, wait=0.01
    )
    assert len(measured["series"]) == 30
    counts = [note["done"] for note in seen if "done" in note]
    assert counts and counts == sorted(counts) and seen[0]["total"] == 30
    stop = threading.Event()

    def cancel_once_started(note: dict) -> None:
        if note.get("done"):
            stop.set()

    with pytest.raises(ParaViewError) as caught:
        client.command(
            "measure_over_time",
            {"field": "p"},
            stop=stop,
            progress=cancel_once_started,
            wait=0.01,
        )
    assert caught.value.code == "cancelled"
    # The cancelled measure put the view back on its step, and the node goes on.
    assert client.command("scene_state", {})["time"]["index"] == 4
    again = client.command("measure_over_time", {"field": "p", "stride": 10})
    assert [row["index"] for row in again["series"]] == [0, 10, 20]
    # An object a script made is measured by its name, even when the script step
    # has the same label.
    client.command(
        "run_scene_script",
        {
            "code": "PassArrays(registrationName='copy', Input=GetActiveSource())",
            "label": "copy",
        },
    )
    copied = client.command(
        "measure_over_time", {"field": "p", "object": "copy", "stride": 10}
    )
    assert copied["label"] == "copy" and len(copied["series"]) == 3
    # Each step is measured at its own time downstream of a filter too.
    assert [round(row["mean"], 3) for row in copied["series"]] == [10.5, 20.5, 30.5]


def test_a_series_longer_than_a_thousand_steps_comes_back_whole(
    attached: tuple[Attacher, Path],
) -> None:
    """measure_over_time returns every picked step, however many there are."""
    attacher, data = attached
    client = attacher.ensure().client
    series = data / "many"
    series.mkdir()
    for step in range(1100):
        write_vti(series / f"out_{step:04d}.vti", shift=step)
    client.command("open_view", {"path": str(series / "out_*.vti"), "new": True})
    measured = client.command("measure_over_time", {"field": "p"}, wait=1.0)
    assert [row["index"] for row in measured["series"]] == list(range(1100))
    means = [row["mean"] for row in measured["series"]]
    assert all(abs(b - a - 1) < 1e-6 for a, b in itertools.pairwise(means))


def test_a_descriptor_display_gives_the_first_view(
    attached: tuple[Attacher, Path],
) -> None:
    """The display part of a view descriptor: a slice of the data, coloured by its
    array over a fixed range, with the data itself as an outline."""
    attacher, data = attached
    client = attacher.ensure().client
    write_vti(data / "shown.vti")
    opened = client.command(
        "open_view",
        {
            "path": str(data / "shown.vti"),
            "new": True,
            "descriptor": {
                "name": "pressure",
                "display": {
                    "array": "p",
                    "association": "POINTS",
                    "representation": "Slice",
                    "slice_normal": [0, 0, 1],
                    "color_map": "Viridis (matplotlib)",
                    "range": [0, 30],
                },
            },
        },
    )
    objects = opened["objects"]
    assert [o["id"] for o in objects] == ["o1", "o2"]
    assert objects[0]["display"]["representation"] == "Outline"
    assert objects[1]["display"]["color_by"] == ["POINTS", "p"]
    assert objects[1]["display"]["color_range"] == [0.0, 30.0]
    # ParaView 6 names the preset without its "(matplotlib)" qualifier; 5.13 keeps it.
    assert objects[1]["display"]["colormap"] in ("Viridis", "Viridis (matplotlib)")
    assert objects[1]["data"]["arrays"][0]["range"] == [3.5, 17.5]
    assert opened["dataset"]["descriptor"]["name"] == "pressure"
    assert "display_note" not in opened["dataset"]
    write_vti(data / "unknown_map.vti")
    partly = client.command(
        "open_view",
        {
            "path": str(data / "unknown_map.vti"),
            "new": True,
            "descriptor": {
                "display": {"array": "p", "color_map": "No Such Map", "range": [1, 2]}
            },
        },
    )
    # The rest of the display holds; the unknown preset is left out and said so.
    assert partly["objects"][0]["display"]["color_range"] == [1.0, 2.0]
    assert not partly["objects"][0]["display"]["colormap"]
    assert "No Such Map" in partly["dataset"]["display_note"]


def test_a_descriptor_geometry_opens_beside_the_data(
    attached: tuple[Attacher, Path],
) -> None:
    """A descriptor's geometry opens as scene objects of their own beside the data;
    one that is not there is left out and said so, and the data's view holds."""
    attacher, data = attached
    client = attacher.ensure().client
    write_vti(data / "flow.vti")
    (data / "body.stl").write_text(
        "solid body\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"
        "vertex 0 1 0\nendloop\nendfacet\nendsolid body\n"
    )
    opened = client.command(
        "open_view",
        {
            "path": str(data / "flow.vti"),
            "new": True,
            "descriptor": {
                "geometry": [
                    {"path": str(data / "body.stl"), "label": "body"},
                    {"path": str(data / "gone.stl"), "visible": False},
                ]
            },
        },
    )
    objects = opened["objects"]
    assert [(o["id"], o["kind"]) for o in objects] == [
        ("o1", "reader"),
        ("o2", "geometry"),
    ]
    assert objects[1]["label"] == "body"
    assert objects[1]["display"]["visible"] is True
    assert objects[1]["data"]["points"] == 3
    assert "gone.stl" in opened["dataset"]["geometry_note"]


def test_a_descriptor_slice_opens_face_on(attached: tuple[Attacher, Path]) -> None:
    """A slice cut across y is looked at along y, not edge-on along the default z."""
    attacher, data = attached
    client = attacher.ensure().client
    write_vti(data / "side.vti")
    client.command(
        "open_view",
        {
            "path": str(data / "side.vti"),
            "new": True,
            "descriptor": {
                "display": {"representation": "Slice", "slice_normal": [0, 1, 0]}
            },
        },
    )
    camera = client.command("fit_camera", {})["camera"]
    offset = [p - f for p, f in zip(camera["position"], camera["focal_point"])]
    assert offset[1] > 0
    assert abs(offset[0]) < 1e-6 and abs(offset[2]) < 1e-6
    assert camera["view_up"] == [0.0, 0.0, 1.0]


def test_a_followed_source_grows_free_colour_maps_and_keeps_fixed_ones(
    attached: tuple[Attacher, Path],
) -> None:
    _, data = attached
    report = _pvpython_script(
        "growth_colour_check.py", str(data / "growth"), prefix="GROWTH="
    )
    assert report["new_steps"] >= 1
    low, high = report["before"]["free"]
    assert report["after"]["free"][0] == low
    assert report["after"]["free"][1] > 1.5 * high
    assert report["before"]["fixed"] == report["after"]["fixed"] == [0.0, 100.0]


def test_a_growing_source_opened_late_catches_up_from_its_first_step(
    attached: tuple[Attacher, Path],
) -> None:
    _, data = attached
    report = _pvpython_script("replay_check.py", str(data / "replay"), prefix="REPLAY=")
    assert report["opened"] == [2, False, True]
    assert report["armed"] and report["started"]
    assert report["first"] == [0, True, True]
    assert report["going"] and report["second"] == [1, True, True]
    assert report["ended"] and report["last"] == [2, False, True]
    # It plays once per opening; a chosen step ends a catch-up and parks the view.
    assert report["again"] is False
    assert report["chosen"] == [1, False, False]


def test_a_colour_range_set_by_hand_holds_and_is_restored(
    attached: tuple[Attacher, Path],
) -> None:
    """A volume's colour map and opacity ramp take a range by hand, kept on restart."""
    attacher, data = attached
    client = attacher.ensure().client
    write_vti(data / "ranged.vti")
    scene = client.command("open_view", {"path": str(data / "ranged.vti")})["scene"]
    shown = client.command(
        "set_representation",
        {
            "object": "o1",
            "representation": "Volume",
            "color_by": "p",
            "color_range": [2.0, 6.0],
        },
    )
    assert shown["object"]["display"]["color_range"] == [2.0, 6.0]
    with pytest.raises(ParaViewError) as caught:
        client.command(
            "set_representation", {"object": "o1", "color_range": [5.0, 1.0]}
        )
    assert caught.value.code == "bad_color_range"
    record = _folder() / "node.json"
    os.kill(json.loads(record.read_text())["pid"], 9)
    record.unlink()
    attacher.link = None
    # Only open_view starts a node half again; other tools say it went.
    with pytest.raises(ParaViewError) as caught:
        attacher.ensure()
    assert caught.value.code == "node_lost"
    client = attacher.ensure(opening=True).client
    restored = client.command("open_view", {"scene": scene})
    assert restored["objects"][0]["display"]["color_range"] == [2.0, 6.0]
    ranges = _pvpython_script("color_range_check.py", str(data), prefix="RANGE=")
    assert ranges == {
        "color": [100.0, 200.0],
        "opacity": [100.0, 200.0],
        "kept": [100.0, 200.0],
        "solid": [1.0, 0.5, 0.0],
        "solid_coloured_by": "",
    }
    with pytest.raises(ParaViewError) as caught:
        client.command(
            "set_representation", {"object": "o1", "solid_color": "sunset glow"}
        )
    assert caught.value.code == "unknown_color"


def test_a_scene_script_builds_in_the_scene_and_replays(tmp_path: Path) -> None:
    """A script adds a second dataset and a coloured filter to the scene the view shows;
    a failing one or one that makes its own view leaves nothing; a restore replays it."""
    seen = _pvpython_script("scripts_check.py", str(tmp_path), prefix="SCRIPTS=")
    kinds = {o["type"]: o for o in seen["added"]}
    assert set(kinds) >= {"Contour"} and all(o["kind"] == "made" for o in seen["added"])
    assert kinds["Contour"]["display"]["solid_color"] == [1.0, 0.0, 0.0]
    hidden = [o for o in seen["added"] if o["type"] != "Contour"]
    assert hidden and hidden[0]["display"]["visible"] is False
    assert (
        seen["scripts"][0]["label"] == "shell" and len(seen["scripts"][0]["made"]) == 2
    )
    assert seen["failed"] == "script_error" and seen["sources_after_failure"] == 0
    assert seen["own_view"] == "script_made_a_view" and seen["views_after_refusal"] == 0
    # Restored: the dataset, and the second dataset the script made; the shell the person
    # removed stays removed when the script replays.
    assert [o["kind"] for o in seen["restored"]] == ["reader", "made"]
    assert seen["restored"][1]["visible"] is False
    assert len(seen["restored_scripts"][0]["made"]) == 1


@pytest.mark.parametrize(("encoder", "fmt"), [("ffmpeg", "mp4"), ("vtk", "ogv")])
def test_an_animation_is_rendered_in_the_scene_and_encoded(
    tmp_path: Path, encoder: str, fmt: str
) -> None:
    """Each time step is rendered in the scene's view; ffmpeg makes an MP4, and without it
    VTK's writer makes an OGG; the scene's own time step and pipeline are left as they were."""
    if encoder == "ffmpeg" and shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg on this machine")
    seen = _pvpython_script(
        "animation_check.py", str(tmp_path), encoder, prefix="ANIMATION="
    )
    assert seen["format"] == fmt and seen["suffix"] == f".{fmt}" and seen["exists"]
    assert seen["frames"] == 3 and seen["width"] == 320  # an even size, for H.264
    assert seen["time_index_after"] == 1
    assert not [n for n in seen["sources_left"] if n.startswith("Text")]
    assert seen["bad_labels"] == "bad_labels"


def test_a_capture_says_how_much_it_drew(attached: tuple[Attacher, Path]) -> None:
    """A frame with the data on it is measured as drawn; one with it all hidden is
    measured as blank, so nobody is told about a picture that is not there."""
    attacher, data = attached
    client = attacher.ensure().client
    write_vti(data / "drawn.vti")
    opened = client.command("open_view", {"path": str(data / "drawn.vti"), "new": True})
    assert opened["drawn"]["blank"] is False and opened["drawn"]["fraction"] > 0.01
    shown = client.command("capture", {"width": 320, "height": 200})
    assert shown["drawn"]["blank"] is False
    for item in opened["objects"]:
        client.command("set_representation", {"object": item["id"], "visible": False})
    hidden = client.command("capture", {"width": 320, "height": 200})
    assert hidden["drawn"]["blank"] is True


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
    client = attacher.ensure(opening=True).client
    restored = client.command("open_view", {"scene": scene})
    assert restored["restored"] is True
    assert [o["id"] for o in restored["objects"]] == ["o1", "o2"]
    assert restored["objects"][1]["display"]["representation"] == "Wireframe"


def test_a_pvserver_that_goes_stops_the_view(tmp_path: Path) -> None:
    """Once the joined pvserver ends, the node half says so, leaves a note and exits;
    the next tool hears that the pvserver went, not a blank picture."""
    import subprocess

    from paraview_mcp import attach

    pvserver = os.environ.get("PARAVIEW_TEST_PVSERVER")
    if not pvserver:
        pytest.skip("set PARAVIEW_TEST_PVSERVER to a pvserver of the same ParaView")
    home = _home(tmp_path)
    data = tmp_path / "cube.vti"
    write_vti(data, n=12)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    served = subprocess.Popen(
        [pvserver, f"--server-port={port}", "--force-offscreen-rendering"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    saved = {k: os.environ.get(k) for k in ("HOME", "PATH")}
    os.environ["HOME"] = str(home)
    assert PVPYTHON is not None
    os.environ["PATH"] = f"{Path(PVPYTHON).parent}:{os.environ['PATH']}"
    address = f"127.0.0.1:{port}"
    attacher = Attacher(Settings(target=Target(), pvserver=address, start_stall=600))
    try:
        # Connecting would use up the pvserver's one handshake, so look at the table.
        listening = f":{port:04X} 00000000:0000 0A"
        deadline = time.monotonic() + 120
        while listening not in Path("/proc/net/tcp").read_text():
            assert time.monotonic() < deadline and served.poll() is None
            time.sleep(0.5)
        client = attacher.ensure(opening=True).client
        opened = client.command("open_view", {"path": str(data)})
        assert opened["drawn"]["blank"] is False
        served.kill()
        served.wait()
        slot = attach.node_base(address)
        folder = _local(home)
        deadline = time.monotonic() + 120
        while (folder / f"{slot}.json").exists() and time.monotonic() < deadline:
            time.sleep(1)
        assert not (folder / f"{slot}.json").exists()
        note = json.loads((folder / f"{slot}.lost.json").read_text())
        assert address in note["pvserver"] and address not in note["reason"]
        assert note["scene"] == opened["scene"]
        assert "closed the connection" in (folder / f"{slot}.log").read_text()
        attacher.link = None
        with pytest.raises(ParaViewError) as caught:
            attacher.ensure()
        assert caught.value.code == "pvserver_lost"
        assert f"open_view(scene={opened['scene']!r})" in caught.value.message
    finally:
        attacher.close()
        if served.poll() is None:
            served.kill()
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def test_a_call_held_up_by_a_pvserver_that_goes_fails_at_once(tmp_path: Path) -> None:
    """The live failure: a capture waits on a pvserver that stopped answering, which
    then ends. ParaView holds Python's lock while it waits, so only the watchdog can
    end the node half: the call fails with the reason in seconds, not after its wait."""
    import signal
    import subprocess
    import threading

    from paraview_mcp import attach
    from paraview_mcp.link import LinkError

    pvserver = os.environ.get("PARAVIEW_TEST_PVSERVER")
    if not pvserver:
        pytest.skip("set PARAVIEW_TEST_PVSERVER to a pvserver of the same ParaView")
    home = _home(tmp_path)
    data = tmp_path / "cube.vti"
    write_vti(data, n=12)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    served = subprocess.Popen(
        [pvserver, f"--server-port={port}", "--force-offscreen-rendering"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    saved = {k: os.environ.get(k) for k in ("HOME", "PATH")}
    os.environ["HOME"] = str(home)
    assert PVPYTHON is not None
    os.environ["PATH"] = f"{Path(PVPYTHON).parent}:{os.environ['PATH']}"
    address = f"127.0.0.1:{port}"
    attacher = Attacher(Settings(target=Target(), pvserver=address, start_stall=600))
    try:
        listening = f":{port:04X} 00000000:0000 0A"
        deadline = time.monotonic() + 120
        while listening not in Path("/proc/net/tcp").read_text():
            assert time.monotonic() < deadline and served.poll() is None
            time.sleep(0.5)
        client = attacher.ensure(opening=True).client
        client.command("open_view", {"path": str(data)})
        slot = attach.node_base(address)
        folder = _local(home)
        assert (
            "pvserver watchdog did not start"
            not in (folder / f"{slot}.log").read_text()
        )
        served.send_signal(signal.SIGSTOP)
        outcome: list[BaseException | None] = []

        def capture() -> None:
            try:
                client.command("capture", {"width": 640, "height": 400})
                outcome.append(None)
            except BaseException as exc:  # noqa: BLE001 - checked below
                outcome.append(exc)

        waiting = threading.Thread(target=capture, daemon=True)
        waiting.start()
        waiting.join(8)
        assert waiting.is_alive(), "the capture should wait on the stopped pvserver"
        served.kill()
        served.wait()
        ended = time.monotonic()
        waiting.join(120)
        assert not waiting.is_alive() and time.monotonic() - ended < 60
        # Held in ParaView (Windows) the watchdog ends the node half and the link
        # drops; let go with a failed send (Linux) the node half refuses the result.
        failed = outcome[0]
        assert isinstance(failed, LinkError) or (
            isinstance(failed, ParaViewError) and failed.code == "pvserver_lost"
        ), failed
        deadline = time.monotonic() + 30
        while not (folder / f"{slot}.lost.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.5)
        note = json.loads((folder / f"{slot}.lost.json").read_text())
        assert address in note["pvserver"] and "closed the connection" in note["reason"]
        state = attacher.status()
        assert state["state"] == "lost" and state["code"] == "pvserver_lost"
    finally:
        attacher.close()
        if served.poll() is None:
            served.kill()
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _kill_serving(served: subprocess.Popen[bytes], port: int) -> None:
    """Kill a test pvserver and every rank of it: mpiexec's ranks may leave its
    process group, so each process serving ``port`` is found by its command line."""
    with contextlib.suppress(ProcessLookupError):
        served.kill()
    marker = f"--server-port={port}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            line = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if marker in line and b"pvserver" in line:
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(entry.name), signal.SIGKILL)
    served.wait()


def test_a_pvserver_measures_on_its_ranks(tmp_path: Path) -> None:
    """Measuring through a pvserver brings back a row per rank, never the dataset:
    the numbers are the whole field's, every step is measured at its own time while
    the view stays put, and the view's own check tells running from gone.

    ``PARAVIEW_TEST_MPIEXEC`` (for example ``mpiexec -launcher fork``) runs the
    pvserver on two ranks, so the ranks' parts are joined too."""
    import shlex
    import subprocess

    pvserver = os.environ.get("PARAVIEW_TEST_PVSERVER")
    if not pvserver:
        pytest.skip("set PARAVIEW_TEST_PVSERVER to a pvserver of the same ParaView")
    mpiexec = shlex.split(os.environ.get("PARAVIEW_TEST_MPIEXEC", ""))
    ranks = 2 if mpiexec else 1
    home = _home(tmp_path)
    for index in range(3):
        write_vti(tmp_path / f"step_{index}.vti", shift=100.0 * index, n=12)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    launch = [*mpiexec, "-n", "2"] if mpiexec else []
    served = subprocess.Popen(
        [*launch, pvserver, f"--server-port={port}", "--force-offscreen-rendering"],
        stdout=subprocess.DEVNULL,
        # What a script on the ranks printed is read here when the test fails.
        stderr=(tmp_path / "pvserver.log").open("w"),
    )
    saved = {k: os.environ.get(k) for k in ("HOME", "PATH")}
    os.environ["HOME"] = str(home)
    assert PVPYTHON is not None
    os.environ["PATH"] = f"{Path(PVPYTHON).parent}:{os.environ['PATH']}"
    address = f"127.0.0.1:{port}"
    attacher = Attacher(Settings(target=Target(), pvserver=address, start_stall=600))
    try:
        listening = f":{port:04X} 00000000:0000 0A"
        deadline = time.monotonic() + 120
        while listening not in Path("/proc/net/tcp").read_text():
            assert time.monotonic() < deadline and served.poll() is None
            time.sleep(0.5)
        assert attacher.status()["state"] == "closed"
        client = attacher.ensure(opening=True).client
        client.command("open_view", {"path": str(tmp_path / "step_*.vti")})
        assert attacher.status()["state"] == "running"
        # The view follows the latest step, so a measurement there sees its shift.
        before = client.command("scene_state", {})["time"]["index"]
        shift = 100.0 * before
        measured = client.command("measure_field", {"field": "p", "bins": 3})
        assert (measured["count"], measured["finite"]) == (1728, 1728)
        assert (measured["min"], measured["max"]) == (shift, shift + 33.0)
        assert measured["mean"] == pytest.approx(shift + 16.5)
        values = [x + y + z for z in range(12) for y in range(12) for x in range(12)]
        spread = (sum((v - 16.5) ** 2 for v in values) / len(values)) ** 0.5
        assert measured["std"] == pytest.approx(spread)
        assert measured["histogram"]["edges"] == pytest.approx(
            [shift, shift + 11.0, shift + 22.0, shift + 33.0]
        )
        counts = [sum(1 for v in values if lo <= v < lo + 11) for lo in (0, 11)]
        assert measured["histogram"]["counts"] == [*counts, 1728 - sum(counts)]
        if ranks == 1:
            assert measured["median"] == shift + 16.5
        else:
            assert measured["median"] is None
            low, high = measured["median_between"]
            assert low <= shift + 16.5 <= high
        probed = client.command("inspect_selection", {"position": [1, 2, 3]})
        assert probed["selection"]["point_values"]["p"] == shift + 6.0
        series = client.command("measure_over_time", {"field": "p"})
        assert [row["mean"] for row in series["series"]] == pytest.approx(
            [16.5, 116.5, 216.5]
        )
        assert [row["max"] for row in series["series"]] == [33.0, 133.0, 233.0]
        assert all(row["count"] == 1728 for row in series["series"])
        # A field's default colour range is measured on the ranks too.
        client.command(
            "run_scene_script",
            {
                "code": "c = Calculator(Input=objects['o1'], ResultArrayName='q')\n"
                "c.Function = 'p'\n"
                "ColorBy(Show(c, view), ('POINTS', 'q'))\n",
                "label": "colour",
            },
        )
        coloured = client.command("scene_state", {})["objects"][-1]["display"]
        ordered = sorted(v + shift for v in values)
        expected = [ranges.percentile(ordered, p) for p in ranges.PERCENTILES]
        assert coloured["auto_range"] == pytest.approx(
            expected, abs=1e-9 if ranks == 1 else 0.1
        )
        assert coloured["data_range"] == [shift, shift + 33.0]
        # The view stayed on its step; only the measurement moved through time.
        assert client.command("scene_state", {})["time"]["index"] == before
        _kill_serving(served, port)
        deadline = time.monotonic() + 120
        while attacher.status()["state"] != "lost":
            assert time.monotonic() < deadline, attacher.status()
            time.sleep(1)
        assert "pvserver" in attacher.status()["text"]
    finally:
        attacher.close()
        _kill_serving(served, port)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def test_remote_mode_above_threshold(tmp_path: Path) -> None:
    home = _home(tmp_path)
    data = tmp_path / "cube.vti"
    write_vti(data, n=12)
    saved = os.environ["HOME"]
    os.environ["HOME"] = str(home)
    attacher = Attacher(
        Settings(target=Target(), threshold_mb=0.001, idle_seconds=4, start_stall=600)
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


def _pvpython_script(name: str, *args: str, prefix: str = "COLORS=") -> dict:
    """Run tests/pvpython/<name> with pvpython; return its JSON report."""
    import subprocess

    assert PVPYTHON is not None
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    done = subprocess.run(
        [
            PVPYTHON,
            "--force-offscreen-rendering",
            str(Path(__file__).with_name("pvpython") / name),
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    line = next((x for x in done.stdout.splitlines() if x.startswith(prefix)), None)
    assert line, done.stdout[-2000:] + done.stderr[-2000:]
    return json.loads(line.split("=", 1)[1])


def test_a_fields_default_colour_range_represents_it(tmp_path: Path) -> None:
    """A field that is a ramp from 0 to 1 but for a small body at 100 is coloured over
    its 2nd to 98th percentile, not 0 to 100. A range a script sets is kept; one
    ParaView picks in a script becomes the default; Rescale gives the whole range."""
    got = _pvpython_script("robust_range_check.py", str(tmp_path), prefix="ROBUST=")
    low, high = got["expected"]
    assert high <= 1.0, got
    assert got["default"] == pytest.approx(got["expected"], abs=1e-5), got
    assert got["display"]["auto_range"] == pytest.approx([low, high], abs=1e-5)
    assert got["display"]["data_range"] == [0.0, 100.0]
    assert got["by_script"] == [0.0, 0.5] and got["kept"] == [0.0, 0.5]
    assert got["script_colour"] == pytest.approx([low, high], abs=1e-5)
    assert got["script_display"]["color_range"] is None
    assert got["script_display"]["auto_range"] == pytest.approx([low, high], abs=1e-5)
    assert got["rescaled"] == [0.0, 100.0] and got["rescaled_mode"] == "data"


def test_colour_maps_survive_a_restart_and_do_not_leak(tmp_path: Path) -> None:
    built = _pvpython_script("colors_check.py", str(tmp_path), "build")
    # A new scene starts with ParaView's own colour map, not the last scene's Jet.
    assert built.pop("jet_after_new_scene") is False
    restored = _pvpython_script("colors_check.py", str(tmp_path), "restore")
    assert restored == built
    # One colour map per field: the Slice's Inferno is the Contour's too, and both say so.
    # ParaView 5.13 names it "Inferno (matplotlib)"; 6 names it "Inferno".
    inferno = restored["recorded"]["o3"]
    assert inferno in ("Inferno", "Inferno (matplotlib)")
    assert restored["preset"] in (inferno, None) and restored["legend"] is True
    assert restored["recorded"] == {"o1": None, "o2": inferno, "o3": inferno}
    assert restored["reps"]["o2"] == [["POINTS", "RTData"], 0.7]


def test_the_camera_comes_back_exactly_after_a_restart(tmp_path: Path) -> None:
    """paraview.simple resets the camera on a process's first render; a restored
    scene's saved camera must survive it (FN-O3)."""
    built = _pvpython_script(
        "camera_check.py", str(tmp_path), "build", prefix="CAMERA="
    )
    restored = _pvpython_script(
        "camera_check.py", str(tmp_path), "restore", prefix="CAMERA="
    )
    assert restored == built


def test_a_volume_is_drawn_on_the_paraview_machine(tmp_path: Path) -> None:
    """The browser cannot draw a volume, so showing one switches the view to remote."""
    seen = _pvpython_script("view_mode_check.py", str(tmp_path), prefix="VIEW=")
    assert seen["opened"]["mode"] == "local"
    assert seen["Volume"]["mode"] == "remote" and seen["Volume"]["reason"] == "volume"
    assert seen["Surface"]["mode"] == "local" and seen["Surface"]["reason"] is None
    # A flat grid opens as an image slice, which the browser cannot draw either.
    assert seen["flat"]["representation"] == "Slice"
    assert seen["flat"]["mode"] == "remote" and seen["flat"]["reason"] == "slice"
    # A growing source is followed as images, one per new step.
    assert seen["growing"]["mode"] == "remote"
    assert seen["growing"]["reason"] == "growing"


def test_the_panel_runs_the_same_commands(tmp_path: Path) -> None:
    """Every panel action is an engine command by the person (pvpython)."""
    import subprocess

    assert PVPYTHON is not None
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    done = subprocess.run(
        [
            PVPYTHON,
            "--force-offscreen-rendering",
            str(Path(__file__).with_name("pvpython") / "panel_check.py"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    line = next(
        (x for x in done.stdout.splitlines() if x.startswith("PANEL_REPORT=")), None
    )
    assert line, done.stdout[-2000:] + done.stderr[-2000:]
    report = json.loads(line.split("=", 1)[1])
    assert set(report["offered"]) == {
        "Contour",
        "Slice",
        "Clip",
        "Threshold",
        "Glyph",
        "StreamTracer",
        "Calculator",
    }
    assert all(not m["error"] for m in report["made"].values()), report["made"]
    assert [t for _, t in report["objects"]] == [
        "XMLImageDataReader",
        "Contour",
        "Slice",
        "Clip",
        "Threshold",
        "Glyph",
        "StreamTracer",
        "Calculator",
    ]
    display = report["display"]
    assert display["color_by"] == ["POINTS", "Gradient"]
    assert display["color_component"] == 2 and display["scalar_bar"] is True
    # The preset ParaView really holds: 5.13 names it "Viridis (matplotlib)".
    assert display["colormap"] in ("Viridis", "Viridis (matplotlib)")
    assert display["opacity"] == 0.6
    assert display["representation"] == "Surface With Edges"
    assert "unknown_colormap" in report["bad_colormap"]
    assert report["display_after_refusal"] == display["colormap"]
    assert report["camera"]["parallel_projection"] is True
    assert report["camera"]["view_up"] == [0.0, 1.0, 0.0]
    assert report["camera"]["position"][2] > report["camera"]["focal_point"][2]
    assert report["after_remove"] == ["o1", "o2", "o4", "o5", "o6", "o7", "o8"]
    assert report["visible_o2"] is False
    assert report["manifest_objects"] == report["after_remove"]
    assert report["manifest_parallel"] is True
    texts = [c["text"] for c in report["changes"] if c["by"] == "person"]
    assert "removed o3" in texts and "parallel projection" in texts
    assert report["scene_state_changes"] == report["changes"]


def test_a_pdb_molecule_takes_a_representation(tmp_path: Path) -> None:
    """JR-P13: a PDB reader has no colour association; set_representation works."""

    seen = _pvpython_script("pdb_check.py", str(tmp_path), prefix="PDB=")
    assert seen == {"points": "Points", "surface_opacity": "Surface"}
