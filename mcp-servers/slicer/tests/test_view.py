"""The layer view: what show_slice sends, the toolpath payload, the declared state and reopening."""

from __future__ import annotations

import base64
import struct
from array import array
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from slicer_mcp import view as view_module
from slicer_mcp.server import mcp
from slicer_mcp.view import VIEW_URI

from .conftest import make_record


def decode(b64: str, typecode: str) -> array:
    values = array(typecode)
    values.frombytes(base64.b64decode(b64))
    return values


async def test_show_slice_payload_reopen_and_lean_text(saved: dict[str, Any]) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("show_slice", {"slice_id": saved["slice_id"]})
    data = result.structured_content
    assert data["slice_id"] == saved["slice_id"]
    assert data["layer_count"] == 15 and data["layer"] == 14
    assert data["print_time_s"] == 201.0 and data["filament_g"] == 0.59
    assert len(data["layers"]["z"]) == len(data["layers"]["time_s"]) == 15
    keys = [row["key"] for row in data["legend"]]
    assert keys[0] == "external_perimeter" and keys[-1] == "travel"
    assert all(row["label"] and row["color"].startswith("#") for row in data["legend"])
    shares = sum(row["time_share"] for row in data["legend"]) + data["other_moves"]["time_share"]
    assert shares == pytest.approx(1.0, abs=0.01)
    assert data["other_moves"]["label"] == "Retracts and Z moves"
    assert "travel" not in data["visible_features"]
    assert data["warnings"] == saved["warnings"]
    assert data["stride"] == 1
    assert result.meta["ai.iowarp.clio/reopen"] == {
        "arguments": {"title": "tube.stl", "slice_id": saved["slice_id"]}
    }
    text = result.content[0].text
    assert "layer 14 (Z 3.0 mm) of 15" in text
    assert "go_to_layer" in text and "Long bridging extrusions" in text
    assert len(text) < 1500 and "xy" not in text


async def test_the_tools_declare_state_paths_and_app_visibility() -> None:
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        resource = (await client.read_resource(VIEW_URI))[0]
    assert set(tools) == {
        "list_printers",
        "slice",
        "show_slice",
        "layer_stats",
        "export",
        "compare_slices",
        "slice_layers",
        "remember_slice_view",
    }
    show_meta = tools["show_slice"].meta or {}
    assert show_meta["ui"]["resourceUri"] == VIEW_URI
    state = {entry["key"]: entry for entry in show_meta["ai.iowarp.clio/state"]}
    assert set(state) == {
        "slice_id",
        "layer",
        "layer_count",
        "z_mm",
        "print_time_s",
        "filament_g",
        "printer",
        "material",
        "visible_features",
        "selection",
    }
    assert state["layer"]["tool"] == "go_to_layer"
    assert state["visible_features"]["tool"] == "set_feature_visibility"
    assert (tools["slice"].meta or {})["ai.iowarp.clio/paths"] == {
        "model": {"machine": "local", "access": "read"}
    }
    assert (tools["export"].meta or {})["ai.iowarp.clio/paths"]["path"]["access"] == "write"
    for name in ("slice_layers", "remember_slice_view"):
        assert (tools[name].meta or {})["ui"]["visibility"] == ["app"]
    csp = ((resource.meta or {}).get("ui") or {}).get("csp") or {}
    assert csp["resourceDomains"] == ["https://cdn.jsdelivr.net"]
    assert "three@0.170.0" in resource.text and "go_to_layer" in resource.text


async def test_slice_layers_payload_decodes_to_the_toolpaths(saved: dict[str, Any]) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "slice_layers", {"slice_id": saved["slice_id"], "first": 0, "last": 14}
        )
        with_travel = await client.call_tool(
            "slice_layers", {"slice_id": saved["slice_id"], "first": 3, "last": 3, "travel": True}
        )
    data = result.structured_content
    assert [layer["layer"] for layer in data["layers"]] == list(range(15))
    assert data["next"] is None
    layer = data["layers"][3]
    assert "travel" not in layer["paths"]
    outer = layer["paths"]["external_perimeter"]
    xy, counts, lines = (
        decode(outer["xy"], "f"),
        decode(outer["counts"], "I"),
        decode(outer["lines"], "I"),
    )
    assert len(xy) == 2 * sum(counts) and len(lines) == len(counts)
    assert layer["z"] == pytest.approx(0.8)
    assert data["vertices"] == sum(
        f.get("vertices", 0)
        for row in saved["layers"]
        for k, f in row["features"].items()
        if k != "travel"
    )
    assert "travel" in with_travel.structured_content["layers"][0]["paths"]
    assert struct.calcsize("f") == 4


async def test_large_answers_come_in_chunks_and_strides(
    saved: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(view_module, "CHUNK_VERTICES", 1000)
    monkeypatch.setattr(view_module, "VIEW_VERTICES", 2000)
    async with Client(mcp) as client:
        first = await client.call_tool(
            "slice_layers", {"slice_id": saved["slice_id"], "first": 0, "last": 14}
        )
        shown = await client.call_tool("show_slice", {"slice_id": saved["slice_id"]})
    assert first.structured_content["next"] is not None
    assert 0 < len(first.structured_content["layers"]) < 15
    assert shown.structured_content["stride"] > 1
    assert "draws every" in shown.content[0].text


async def test_remembered_view_is_what_a_reopen_shows(saved: dict[str, Any]) -> None:
    sid = saved["slice_id"]
    async with Client(mcp) as client:
        await client.call_tool(
            "remember_slice_view",
            {
                "slice_id": sid,
                "layer": 6,
                "visible_features": ["external_perimeter", "travel", "bogus"],
            },
        )
        reopened = await client.call_tool("show_slice", {"slice_id": sid})
        explicit = await client.call_tool("show_slice", {"slice_id": sid, "layer": 2})
        with pytest.raises(ToolError) as caught:
            await client.call_tool("show_slice", {"slice_id": sid, "layer": 99})
        with pytest.raises(ToolError) as missing:
            await client.call_tool("show_slice", {"slice_id": "s0000000000"})
    assert reopened.structured_content["layer"] == 6
    assert reopened.structured_content["visible_features"] == ["external_perimeter", "travel"]
    assert explicit.structured_content["layer"] == 2
    assert "[layer_out_of_range]" in str(caught.value)
    assert "[slice_not_found]" in str(missing.value)


async def test_layer_stats_export_and_compare(
    saved: dict[str, Any], home: Path, tmp_path: Path
) -> None:
    other = make_record(home)
    async with Client(mcp) as client:
        stats = await client.call_tool(
            "layer_stats", {"slice_id": saved["slice_id"], "first_layer": 0, "last_layer": 2}
        )
        out = tmp_path / "out"
        out.mkdir()
        csv_result = await client.call_tool(
            "export", {"slice_id": saved["slice_id"], "format": "csv", "path": str(out)}
        )
        gcode_result = await client.call_tool(
            "export",
            {"slice_id": saved["slice_id"], "format": "gcode", "path": str(out / "t.gcode")},
        )
        with pytest.raises(ToolError) as exists:
            await client.call_tool(
                "export",
                {"slice_id": saved["slice_id"], "format": "gcode", "path": str(out / "t.gcode")},
            )
        compared = await client.call_tool(
            "compare_slices", {"slice_ids": [saved["slice_id"], other["slice_id"]]}
        )
    data = stats.structured_content
    assert (data["first_layer"], data["last_layer"], len(data["layers"])) == (0, 2, 3)
    assert data["by_feature"][0]["time_s"] >= data["by_feature"][-1]["time_s"]
    assert sum(f["time_share"] for f in data["by_feature"]) == pytest.approx(1.0, abs=0.01)
    assert "Per layer: 0 (Z 0.2)" in stats.content[0].text
    x0, y0, x1, y1 = data["layers"][2]["extents"]["external_perimeter"]
    assert x1 - x0 == pytest.approx(16.0, abs=0.6) and y1 - y0 == pytest.approx(16.0, abs=0.6)
    assert "travel" not in data["layers"][2]["extents"]
    rows = (out / "tube.layers.csv").read_text().splitlines()
    assert rows[0].startswith("layer,z_mm,height_mm,time_s,filament_g,time_s_external_perimeter")
    assert len(rows) == 16 and csv_result.structured_content["rows"] == 15
    assert (out / "t.gcode").read_bytes() == Path(saved["gcode"]).read_bytes()
    assert gcode_result.structured_content["bytes"] > 100_000
    assert "[export_exists]" in str(exists.value)
    assert compared.structured_content["slices"][1]["print_time_s_change"] == 0.0


async def test_the_plan_print_prompt_walks_the_tools() -> None:
    async with Client(mcp) as client:
        prompts = {prompt.name for prompt in await client.list_prompts()}
        result = await client.get_prompt(
            "plan_print", {"model": "bracket.stl", "printer": "Prusa MK4"}
        )
    assert prompts == {"plan_print"}
    text = result.messages[0].content.text
    assert "bracket.stl" in text and "Prusa MK4" in text and "PLA" in text
    for name in ("list_printers", "slice", "show_slice", "layer_stats", "compare_slices"):
        assert name in text
