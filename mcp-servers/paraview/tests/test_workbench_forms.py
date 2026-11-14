"""The view's panel without ParaView: filter forms, the change log, the new tool inputs."""

from __future__ import annotations

import pytest
from fastmcp import Client

from paraview_mcp import present
from paraview_mcp.node import changes, filters, panel, theme
from paraview_mcp.server import mcp

WAVELET = {
    "data_type": "vtkImageData",
    "points": 9261,
    "cells": 8000,
    "bounds": [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0],
    "memory_kb": 300,
    "arrays": [
        {
            "name": "RTData",
            "association": "POINTS",
            "components": 1,
            "range": [37.4, 276.8],
        },
        {"name": "velocity", "association": "POINTS", "components": 3, "range": [0, 2]},
        {"name": "id", "association": "CELLS", "components": 1, "range": [0, 7999]},
    ],
}


def test_every_offered_filter_has_a_form_that_becomes_properties() -> None:
    offered = filters.offered(WAVELET)
    assert [f["name"] for f in offered] == [e["name"] for e in filters.CATALOG]
    assert not any(f["disabled"] for f in offered)
    for entry in offered:
        fields = filters.form(entry["name"], WAVELET)
        values = {f["key"]: f["value"] for f in fields}
        if entry["name"] == "Calculator":
            values["Function"] = "RTData*2"
        props = filters.properties(fields, values)
        assert set(props) == {f["key"] for f in fields}


def test_form_defaults_come_from_the_data() -> None:
    contour = {f["key"]: f for f in filters.form("Contour", WAVELET)}
    assert contour["ContourBy"]["value"] == "POINTS:RTData"
    assert float(contour["Isosurfaces"]["value"]) == pytest.approx(157.1)
    # Contour by a point field only.
    assert [i["value"] for i in contour["ContourBy"]["items"]] == [
        "POINTS:RTData",
        "POINTS:velocity",
    ]
    slice_ = {f["key"]: f["value"] for f in filters.form("Slice", WAVELET)}
    assert slice_ == {
        "SliceType.Origin": [0.0, 0.0, 0.0],
        "SliceType.Normal": [1.0, 0.0, 0.0],
    }
    threshold = filters.properties(
        filters.form("Threshold", WAVELET),
        {"Scalars": "CELLS:id", "LowerThreshold": "10", "UpperThreshold": 20},
    )
    assert threshold == {
        "Scalars": ["CELLS", "id"],
        "LowerThreshold": 10.0,
        "UpperThreshold": 20.0,
    }
    assert filters.properties(
        filters.form("Contour", WAVELET),
        {"ContourBy": "POINTS:RTData", "Isosurfaces": "100, 200"},
    )["Isosurfaces"] == [100.0, 200.0]


def test_vector_filters_are_off_without_a_vector_field() -> None:
    scalars_only = {**WAVELET, "arrays": WAVELET["arrays"][:1]}
    off = {f["name"]: f for f in filters.offered(scalars_only) if f["disabled"]}
    assert set(off) == {"Glyph", "StreamTracer"}
    assert "3-component" in off["Glyph"]["tip"]
    assert all(f["disabled"] for f in filters.offered(None))


@pytest.mark.parametrize(
    ("name", "values", "message"),
    [
        ("Contour", {"Isosurfaces": "abc"}, "is not a number"),
        ("Slice", {"SliceType.Normal": [0, 0, 0]}, "cannot be zero"),
        ("Slice", {"SliceType.Origin": [1, 2]}, "three numbers"),
        ("Calculator", {"Function": "  "}, "required"),
        ("Threshold", {"Scalars": ""}, "choose a field"),
    ],
)
def test_bad_form_values_are_refused_before_any_command(
    name: str, values: dict, message: str
) -> None:
    fields = filters.form(name, WAVELET)
    merged = {f["key"]: f["value"] for f in fields} | values
    with pytest.raises(filters.FormError, match=message):
        filters.properties(fields, merged)


def test_change_lines_name_who_changed_what() -> None:
    scene = {
        "objects": [
            {"id": "o1", "label": "wavelet.vti"},
            {"id": "o2", "label": "Slice2"},
        ]
    }
    _, text = changes.describe(
        "create_filter",
        {
            "filter": "Slice",
            "input": "o1",
            "properties": {"SliceType.Normal": [0, 0, 1]},
        },
        {"object": {"id": "o2", "type": "Slice", "input": "o1"}},
        scene,
    )
    assert text == "added o2 Slice on o1 wavelet.vti (SliceType.Normal (0, 0, 1))"
    _, text = changes.describe(
        "set_representation",
        {"object": "o2", "color_by": "Gradient", "component": 2},
        {},
        scene,
    )
    assert text == "o2 Slice2: coloured by Gradient component 2"
    _, text = changes.describe(
        "set_camera", {"direction": "-z", "parallel_projection": True}, {}, scene
    )
    assert text == "camera looks along -Z, parallel projection"
    assert changes.describe("scene_state", {}, {}, scene) is None


def test_a_persons_run_of_one_change_is_one_line() -> None:
    log = changes.ChangeLog(keep=3)
    log.record("person", "display:o2:opacity,object", "o2: opacity 0.9")
    log.record("person", "display:o2:opacity,object", "o2: opacity 0.4")
    log.record("agent", "time", "time step 2 of 5")
    log.record("agent", "time", "time step 3 of 5")
    assert [e["text"] for e in log.recent()] == [
        "o2: opacity 0.4",
        "time step 2 of 5",
        "time step 3 of 5",
    ]
    log.record("person", "camera", "camera turned in the view")
    assert len(log.recent()) == 3 and log.recent()[0]["text"] == "time step 2 of 5"
    assert changes.lines(log.recent())[-1].endswith(
        "by the person: camera turned in the view"
    )
    assert present.changes_lines(log.recent())[0] == "Recent changes, oldest first:"


async def test_new_tool_inputs_are_listed() -> None:
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    rep = tools["set_representation"].input_schema["properties"]
    assert {"component", "scalar_bar", "rescale"} <= set(rep)
    camera = tools["set_camera"].input_schema["properties"]
    assert {"direction", "parallel_projection"} <= set(camera)
    assert "+z" in str(camera["direction"])


def test_present_mentions_component_legend_and_projection() -> None:
    line = present.object_line(
        {
            "id": "o2",
            "label": "grad",
            "type": "Gradient",
            "display": {
                "visible": True,
                "representation": "Surface",
                "color_by": ["POINTS", "Gradient"],
                "color_component": 1,
                "colormap": "Viridis",
                "scalar_bar": True,
                "opacity": 0.5,
            },
        }
    )
    assert "coloured by Gradient component 1 (Viridis)" in line
    assert "colour legend shown" in line and "opacity 0.5" in line
    camera = {"position": [0, 0, 1], "focal_point": [0, 0, 0], "view_up": [0, 1, 0]}
    assert present.camera_text({**camera, "parallel_projection": True}).endswith(
        "Parallel projection."
    )


def test_panel_icons_have_names_and_the_theme_follows_the_host() -> None:
    labels = [label for _, _, label in panel.PANELS]
    assert labels == ["Pipeline", "Add filter", "Display", "Camera", "Information"]
    assert "--color-background-primary" in theme.FALLBACK_CSS
    assert "clio.paraview.theme" in theme.THEME_JS
    assert "light-dark(" in theme.FALLBACK_CSS


async def test_view_page_passes_the_host_theme_on() -> None:
    async with Client(mcp) as client:
        content = await client.read_resource("ui://paraview/view")
    html = content[0].text
    assert "styles.variables" in html or "styles.variables" in html.replace(" ", "")
    assert "clio.paraview.theme" in html
    assert "var(--color-background-primary)" in html
    # The context tells the agent what the person changed.
    assert "Recent changes in the view" in html


def test_slice_clip_and_threshold_replace_their_input() -> None:
    assert [
        e["name"] for e in filters.CATALOG if filters.replaces_input(e["name"])
    ] == [
        "Slice",
        "Clip",
        "Threshold",
    ]


def test_camera_presets_are_separate_lines() -> None:
    log = changes.ChangeLog()
    for args in (
        {"direction": "+y"},
        {"parallel_projection": True},
        {"parallel_projection": True},
    ):
        log.record("person", *changes.describe("set_camera", args, {}, None))
    assert [e["text"] for e in log.recent()] == [
        "camera looks along +Y",
        "parallel projection",
    ]


class _Legend:
    def __init__(self, title: str) -> None:
        self.title = title

    def GetScalarBarActor(self) -> _Legend:
        return self

    def GetTitle(self) -> str:
        return self.title


def test_the_browser_shows_the_selected_objects_legend() -> None:
    from paraview_mcp.node.serializers import chosen_legend

    legends = [_Legend("RTData"), _Legend("velocity Magnitude"), _Legend("temperature")]
    assert chosen_legend(legends, "velocity") is legends[1]
    assert chosen_legend(legends, None) is legends[2]
    assert chosen_legend(legends, "pressure") is legends[2]
    assert chosen_legend([], "RTData") is None


def test_information_shows_short_property_values() -> None:
    from paraview_mcp.node.workbench import selected_view

    item = {
        "id": "o2",
        "kind": "filter",
        "type": "Threshold",
        "properties": {"Scalars": ["POINTS", "p"], "LowerThreshold": 1.7587922e-25},
    }
    shown = selected_view(item, WAVELET)["properties"]
    assert shown == [
        {"key": "Scalars", "value": "POINTS, p"},
        {"key": "LowerThreshold", "value": "1.759e-25"},
    ]


def test_the_pipeline_lists_each_object_under_its_input() -> None:
    from paraview_mcp.node.workbench import pipeline_order

    objects = [
        {"id": "o1"},
        {"id": "o2", "input": "o1"},
        {"id": "o3", "input": "o1"},
        {"id": "o4", "input": "o2"},
        {"id": "o5", "input": "o3"},
    ]
    assert [(o["id"], d) for o, d in pipeline_order(objects)] == [
        ("o1", 0),
        ("o2", 1),
        ("o4", 2),
        ("o3", 1),
        ("o5", 2),
    ]


def test_colour_map_and_legend_are_per_field() -> None:
    from paraview_mcp.node import colors

    scene = {
        "objects": [
            {"id": "o1", "display": {"color_by": None}},
            {"id": "o2", "display": {"color_by": ["POINTS", "p"], "scalar_bar": True}},
            {"id": "o3", "display": {"color_by": ["POINTS", "p"], "scalar_bar": False}},
            {"id": "o4", "display": {"color_by": ["CELLS", "q"], "scalar_bar": False}},
        ]
    }
    assert colors.fields(scene) == ["p", "q"]
    assert colors.legend_wanted(scene, ["POINTS", "p"]) is True
    assert colors.legend_wanted(scene, ["CELLS", "q"]) is False
    colors.share(scene, ["POINTS", "p"], "colormap", "Inferno")
    assert [o["display"].get("colormap") for o in scene["objects"]] == [
        None,
        "Inferno",
        "Inferno",
        None,
    ]


def test_a_filter_is_edited_with_its_own_form_filled_in() -> None:
    fields = filters.filled(
        filters.form("Contour", WAVELET),
        {"ContourBy": ["POINTS", "RTData"], "Isosurfaces": [120.0, 200.5]},
    )
    by_key = {f["key"]: f for f in fields}
    assert by_key["ContourBy"]["value"] == "POINTS:RTData"
    assert by_key["Isosurfaces"]["value"] == "120, 200.5"
    assert by_key["Isosurfaces"]["range"] == [37.4, 276.8]  # the slider's span
    values = {f["key"]: f["value"] for f in fields}
    assert filters.properties(fields, values) == {
        "ContourBy": ["POINTS", "RTData"],
        "Isosurfaces": [120.0, 200.5],
    }


def test_a_changed_filter_is_one_line_naming_old_and_new_values() -> None:
    scene = {
        "objects": [
            {"id": "o1", "label": "clouds.pvd"},
            {"id": "o2", "label": "Contour2"},
        ]
    }
    key, text = changes.describe(
        "set_filter_property",
        {"object": "o2", "properties": {"Isosurfaces": [0.0003]}},
        {"object": {"id": "o2"}, "before": {"Isosurfaces": [0.0001]}},
        scene,
    )
    assert text == "o2 Contour2: Isosurfaces (0.0001) -> (0.0003)"
    again, _ = changes.describe(
        "set_filter_property",
        {"object": "o2", "properties": {"Isosurfaces": [0.0004]}},
        {"object": {"id": "o2"}, "before": {"Isosurfaces": [0.0003]}},
        scene,
    )
    assert again == key  # a slider's run of drags is one line, updated in place
