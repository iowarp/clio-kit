"""show_on_globe and its view: lean results, declarations, saving and reopening."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from geo_mcp import globe
from geo_mcp.server import mcp

POINTS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"station": f"S{i}", "note": "x" * 50},
            "geometry": {"type": "Point", "coordinates": [-121.0 - i * 0.123456, 46.0 + i * 0.1]},
        }
        for i in range(5)
    ],
}
AREA = {
    "type": "Polygon",
    "coordinates": [[[-122.5, 45.5], [-120.5, 45.5], [-120.5, 47.0], [-122.5, 45.5]]],
}


@pytest.fixture(autouse=True)
def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = tmp_path / "views"
    monkeypatch.setenv("GEO_MCP_GLOBE_DIR", str(store))
    return store


def _layers() -> list[dict[str, Any]]:
    return [
        {"name": "stations", "geojson": POINTS, "id_property": "station"},
        {"name": "area", "geojson": AREA, "style": {"facecolor": "orange", "alpha": 0.2}},
    ]


async def _call(name: str, args: dict[str, Any]) -> Any:
    async with Client(mcp) as client:
        return await client.call_tool(name, args)


async def test_tool_declares_its_view_state_keys_and_paths() -> None:
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    meta = tools["show_on_globe"].meta
    assert meta["ui"]["resourceUri"] == "ui://geo/globe"
    keys = {entry["key"]: entry for entry in meta["ai.iowarp.clio/state"]}
    assert keys["selected_feature"]["tool"] == "select_feature"
    assert {"selection", "camera", "layers", "view"} <= set(keys)
    assert meta["ai.iowarp.clio/paths"] == {"files": {"machine": "local", "access": "read"}}
    assert tools["show_on_globe"].annotations.read_only_hint is True
    # A host naming a view by its first short text argument reads the title.
    assert next(iter(tools["show_on_globe"].input_schema["properties"])) == "title"
    edit = tools["edit_globe_view"].meta["ui"]
    assert edit == {"visibility": ["app"]}  # only the view calls it; it opens no view


async def test_view_resource_declares_cesium_and_tile_domains() -> None:
    async with Client(mcp) as client:
        listed = {str(r.uri): r for r in await client.list_resources()}
        read = await client.read_resource("ui://geo/globe")
    assert listed["ui://geo/globe"].mime_type == "text/html;profile=mcp-app"
    csp = read[0].meta["ui"]["csp"]
    assert csp["resourceDomains"] == ["https://cesium.com", "https://tile.openstreetmap.org"]
    assert csp["connectDomains"] == csp["resourceDomains"]
    html = read[0].text
    assert "releases/1.123/Build/Cesium" in html  # pinned
    assert "tile.openstreetmap.org/{z}/{x}/{y}.png" in html
    assert "Ion.defaultAccessToken = undefined" in html  # no Cesium ion token


async def test_result_is_lean_and_the_geojson_goes_to_the_view_only() -> None:
    result = await _call("show_on_globe", {"layers": _layers(), "title": "Cascadia"})
    text = result.content[0].text
    assert len(result.content) == 1
    assert "stations: 5 points; ids from property 'station'" in text
    assert "area: 1 polygon" in text
    assert "bounds [-122.5, 45.5, -120.5, 47]" in text
    assert "0.123456" not in text and "xxxxx" not in text  # no coordinates, no properties
    view = result.structured_content
    assert view["title"] == "Cascadia"
    assert [layer["name"] for layer in view["layers"]] == ["stations", "area"]
    assert view["layers"][0]["geojson"]["features"][0]["id"] == "S0"
    assert result.meta["ai.iowarp.clio/reopen"] == {
        "arguments": {"title": "Cascadia", "view": view["view"]}
    }
    assert f"view={view['view']!r}" in text


async def test_reopening_by_id_re_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "stations.geojson"
    path.write_text(json.dumps(POINTS))
    first = await _call("show_on_globe", {"files": [str(path)]})
    view_id = first.structured_content["view"]
    assert first.structured_content["layers"][0]["name"] == "stations"
    grown = {"type": "FeatureCollection", "features": POINTS["features"] * 2}
    path.write_text(json.dumps(grown))
    again = await _call("show_on_globe", {"view": view_id})
    assert again.structured_content["view"] == view_id
    assert again.structured_content["layers"][0]["features"] == 10
    assert again.content[0].text.startswith("Restored 1 layer")
    saved = json.loads((Path(os.environ["GEO_MCP_GLOBE_DIR"]) / f"{view_id}.json").read_text())
    assert saved["layers"][0]["geojson"] == str(path.resolve())  # the path, not the content


async def test_a_reopened_view_shows_what_it_can_when_a_file_is_gone(tmp_path: Path) -> None:
    path = tmp_path / "gone.geojson"
    path.write_text(json.dumps(POINTS))
    first = await _call("show_on_globe", {"files": [str(path)], "layers": [_layers()[1]]})
    path.unlink()
    again = await _call("show_on_globe", {"view": first.structured_content["view"]})
    layers = {layer["name"]: layer for layer in again.structured_content["layers"]}
    assert "path not found" in layers["gone"]["error"]
    assert layers["area"]["features"] == 1
    assert "- gone: not shown, " in again.content[0].text


async def test_reopen_can_move_the_camera_and_retitle() -> None:
    first = await _call("show_on_globe", {"layers": _layers()})
    again = await _call(
        "show_on_globe",
        {
            "view": first.structured_content["view"],
            "camera": {"lon": -121, "lat": 46},
            "title": "T",
        },
    )
    assert again.structured_content["camera"]["height_m"] == 2_000_000
    assert again.structured_content["title"] == "T"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({}, "Give layers"),
        ({"view": "g0000000000"}, "view_not_found"),
        ({"view": "../etc/passwd"}, "view_id_invalid"),
        ({"layers": [{"geojson": {"type": "FeatureCollection", "features": []}}]}, "No layer has"),
        ({"layers": [{"geojson": AREA, "name": "a"}, {"geojson": AREA, "name": "a"}]}, "distinct"),
        ({"layers": [{"geojson": AREA}], "bbox": [1, 2, 3]}, "bbox must be"),
        ({"layers": [{"geojson": AREA}], "camera": {"lat": 1}}, "camera must be"),
        ({"layers": [{"geojson": AREA}], "camera": {"lon": 1, "lat": 99}}, "out of range"),
        ({"layers": [{"geojson": "/missing.geojson", "name": "m"}]}, "path not found"),
    ],
)
async def test_refusals_say_why(args: dict[str, Any], message: str) -> None:
    with pytest.raises(ToolError, match=message):
        await _call("show_on_globe", args)


async def test_view_and_layers_together_are_refused() -> None:
    first = await _call("show_on_globe", {"layers": _layers()})
    with pytest.raises(ToolError, match="not both"):
        await _call(
            "show_on_globe", {"view": first.structured_content["view"], "layers": _layers()}
        )


async def test_too_much_geojson_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(globe, "MAX_VIEW_BYTES", 500)
    with pytest.raises(ToolError, match="more than a browser globe draws"):
        await _call("show_on_globe", {"layers": _layers()})


async def test_an_unwritable_store_still_shows_the_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    monkeypatch.setenv("GEO_MCP_GLOBE_DIR", str(blocker / "views"))
    result = await _call("show_on_globe", {"layers": _layers()})
    assert result.structured_content["unsaved_reason"] == "view_not_saved"
    assert "ai.iowarp.clio/reopen" not in (result.meta or {})
    assert "could not be saved (view_not_saved)" in result.content[0].text


async def test_edits_are_saved_and_reopen_shows_them() -> None:
    first = await _call("show_on_globe", {"layers": _layers()})
    view_id = first.structured_content["view"]
    track = {"name": "track", "geojson": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}
    added = await _call("edit_globe_view", {"view": view_id, "action": "add", "layer": track})
    assert added.structured_content["layer"]["features"] == 1
    assert added.content[0].text.startswith("Added layer")
    styled = await _call(
        "edit_globe_view",
        {
            "view": view_id,
            "action": "style",
            "name": "area",
            "style": {"facecolor": "red", "alpha": None},
        },
    )
    assert styled.structured_content["layer"]["style"]["fill"] == "#ff0000"
    assert styled.structured_content["layer"]["style"]["alpha"] == 0.6  # removed key: default
    removed = await _call(
        "edit_globe_view", {"view": view_id, "action": "remove", "name": "stations"}
    )
    assert removed.structured_content == {"view": view_id, "removed": "stations"}
    again = await _call("show_on_globe", {"view": view_id})
    names = [layer["name"] for layer in again.structured_content["layers"]]
    assert names == ["area", "track"]
    assert again.structured_content["layers"][0]["style"]["fill"] == "#ff0000"


async def test_edit_refusals() -> None:
    first = await _call("show_on_globe", {"layers": _layers()})
    view_id = first.structured_content["view"]
    with pytest.raises(ToolError, match="No layer 'nope'"):
        await _call("edit_globe_view", {"view": view_id, "action": "remove", "name": "nope"})
    with pytest.raises(ToolError, match="add needs layer"):
        await _call("edit_globe_view", {"view": view_id, "action": "add"})
    with pytest.raises(ToolError, match="not a colour"):
        await _call(
            "edit_globe_view",
            {"view": view_id, "action": "style", "name": "area", "style": {"color": "zz"}},
        )
    with pytest.raises(ToolError, match="view_not_found"):
        await _call("edit_globe_view", {"view": "g1111111111", "action": "remove", "name": "a"})


def test_capabilities_list_the_globe() -> None:
    from geo_mcp.server import capabilities

    assert "show_on_globe" in capabilities()["tools"]


@pytest.mark.skipif(
    os.environ.get("GEO_GLOBE_BROWSER") != "1",
    reason="browser_check_not_requested: set GEO_GLOBE_BROWSER=1 (needs node, Playwright, "
    "Chromium and network access to cesium.com and tile.openstreetmap.org)",
)
def test_the_view_in_a_stand_in_host(tmp_path: Path) -> None:
    """Draws, clicks, host-set selection, fly_to and edits in headless Chromium."""
    script = Path(__file__).parent / "browser" / "run_standin.py"
    done = subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    report = json.loads(done.stdout)
    checks = report["checks"]
    assert report["consoleErrors"] == [] and report["failedRequests"] == []
    assert checks["drawn"]["stations"] == {"features": 8, "entities": 8, "points": 8, "labels": 8}
    assert checks["drawn"]["study area"]["entities"] == 1
    assert checks["clickReported"] == "R03"
    assert checks["hostSelectReported"] == "R07"
    after = checks["hostSelect"]["after"]
    assert abs(after["lon"] - -122.45) < 0.05 and abs(after["lat"] - 46.70) < 0.05
    assert checks["areaClickReported"] == "study-area"
    west, south, east, north = checks["flyTo"]["view"]
    assert west < -121.85 < east and south < 46.875 < north
    assert "survey track" in checks["addLayer"]["drawn"] and not checks["setStyle"]["isError"]
    assert "survey track" not in checks["removeLayer"]["drawn"]
    assert checks["cleared"] == 0
    assert report["reopened_layers"] == ["study area", "stations"]
