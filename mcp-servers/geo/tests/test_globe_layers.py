"""Layer normalization for the globe: GeoJSON forms, stable ids, styles and bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geo_mcp.implementation.globe_layers import (
    LABEL_LIMIT,
    GlobeLayerError,
    merged_bounds,
    normalize_layer,
    view_layer,
)


def _pt(lon: float, lat: float, **props: object) -> dict:
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


STATIONS = {
    "type": "FeatureCollection",
    "features": [
        _pt(-121.7, 46.8, station="A1", kind="gnss", aqi=40),
        _pt(-122.1, 47.1, station="B2", kind="seismic", aqi=180),
        _pt(-121.2, 46.5, station="C3", kind="gnss", aqi=420),
    ],
}


def test_every_accepted_geojson_form(tmp_path: Path) -> None:
    feature = STATIONS["features"][0]
    path = tmp_path / "s.geojson"
    path.write_text(json.dumps(STATIONS))
    forms = {
        "collection": (STATIONS, 3),
        "feature": (feature, 1),
        "geometry": (feature["geometry"], 1),
        "list": (STATIONS["features"], 3),
        "string": (json.dumps(STATIONS), 3),
        "path": (str(path), 3),
    }
    for form, (geojson, count) in forms.items():
        layer = normalize_layer({"geojson": geojson, "name": form}, 0)
        assert layer["features"] == count, form
        assert layer["geojson"]["type"] == "FeatureCollection"
    assert normalize_layer({"geojson": str(path)}, 0)["source"] == str(path.resolve())
    assert normalize_layer({"geojson": STATIONS}, 0)["source"] is None


def test_ids_come_from_the_declared_property_then_the_feature() -> None:
    by_property = normalize_layer({"geojson": STATIONS, "id_property": "station"}, 0)
    assert [f["id"] for f in by_property["geojson"]["features"]] == ["A1", "B2", "C3"]
    assert by_property["ids"] == "property 'station'"

    with_ids = {
        "type": "FeatureCollection",
        "features": [{**_pt(0, 0), "id": 7}, _pt(1, 1, id="x")],
    }
    layer = normalize_layer({"geojson": with_ids, "name": "w"}, 0)
    assert [f["id"] for f in layer["geojson"]["features"]] == ["7", "x"]

    generated = normalize_layer({"geojson": [_pt(0, 0), _pt(1, 1)], "name": "pts"}, 0)
    assert [f["id"] for f in generated["geojson"]["features"]] == ["pts-0", "pts-1"]
    assert generated["ids"] == "generated"


def test_a_declared_id_property_must_be_present() -> None:
    with pytest.raises(GlobeLayerError, match="no 'code' property"):
        normalize_layer({"geojson": STATIONS, "id_property": "code", "name": "s"}, 0)


def test_duplicate_ids_are_made_distinct_and_reported() -> None:
    layer = normalize_layer({"geojson": [_pt(0, 0, id="a"), _pt(1, 1, id="a")]}, 0)
    assert [f["id"] for f in layer["geojson"]["features"]] == ["a", "a~1"]
    assert any("duplicate" in note for note in layer["notes"])


def test_unusable_features_are_skipped_and_counted() -> None:
    bad = [
        _pt(0, 0),
        {"type": "Feature", "properties": {}, "geometry": None},
        {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": ["x"]}},
        {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": [0, 95]},
        },
    ]
    layer = normalize_layer({"geojson": bad}, 0)
    assert layer["features"] == 1
    assert layer["skipped"] == {"no geometry": 1, "malformed geometry": 2}


def test_missing_path_and_bad_json_say_why() -> None:
    with pytest.raises(GlobeLayerError, match="path not found"):
        normalize_layer({"geojson": "/nope/x.geojson", "name": "n"}, 0)
    with pytest.raises(GlobeLayerError, match="Invalid GeoJSON"):
        normalize_layer({"geojson": "{not json", "name": "n"}, 0)
    with pytest.raises(GlobeLayerError, match="no 'geojson'"):
        normalize_layer({"name": "n"}, 0)
    with pytest.raises(GlobeLayerError, match="not an object"):
        normalize_layer(["x"], 3)


def test_colours_resolve_like_matplotlib() -> None:
    layer = normalize_layer(
        {"geojson": STATIONS, "style": {"color": "tab:red", "edgecolor": "C0", "alpha": 0.3}}, 0
    )
    assert layer["style"]["fill"] == "#d62728"
    assert layer["style"]["stroke"] == "#1f77b4"
    assert layer["style"]["alpha"] == 0.3
    assert layer["legend"] == {
        "kind": "single",
        "entries": [{"label": "layer_0", "color": "#d62728"}],
    }
    with pytest.raises(GlobeLayerError, match="not a colour"):
        normalize_layer({"geojson": STATIONS, "style": {"color": "notacolour"}}, 0)
    with pytest.raises(GlobeLayerError, match="alpha"):
        normalize_layer({"geojson": STATIONS, "style": {"alpha": 3}}, 0)


def test_category_colours_and_legend() -> None:
    style = {"color_by": "kind", "category_colors": {"gnss": "red", "seismic": "blue"}}
    layer = normalize_layer({"geojson": STATIONS, "id_property": "station", "style": style}, 0)
    assert layer["colors"] == {"A1": "#ff0000", "B2": "#0000ff", "C3": "#ff0000"}
    assert layer["legend"]["kind"] == "categories"
    assert [e["label"] for e in layer["legend"]["entries"]] == ["gnss", "seismic"]
    assert layer["color_by"] == "kind"


def test_epa_aqi_and_colormap_scales() -> None:
    aqi = normalize_layer(
        {
            "geojson": STATIONS,
            "id_property": "station",
            "style": {"color_by": "aqi", "scale": "epa_aqi"},
        },
        0,
    )
    assert aqi["colors"] == {"A1": "#00e400", "B2": "#ff0000", "C3": "#7e0023"}
    ramp = normalize_layer(
        {
            "geojson": STATIONS,
            "id_property": "station",
            "style": {"color_by": "aqi", "scale": "viridis"},
        },
        0,
    )
    assert ramp["legend"]["kind"] == "ramp"
    assert (ramp["legend"]["min"], ramp["legend"]["max"]) == (40, 420)
    assert ramp["colors"]["A1"] == "#440154" and ramp["colors"]["C3"] == "#fde725"
    categorical = normalize_layer(
        {"geojson": STATIONS, "id_property": "station", "style": {"color_by": "kind"}}, 0
    )
    assert categorical["legend"]["kind"] == "categories"
    with pytest.raises(GlobeLayerError, match="colormap"):
        normalize_layer({"geojson": STATIONS, "style": {"color_by": "aqi", "scale": "nope"}}, 0)


def test_unused_colour_property_and_unknown_style_keys_are_reported() -> None:
    layer = normalize_layer({"geojson": STATIONS, "style": {"color_by": "missing", "glow": 1}}, 0)
    assert layer["colors"] == {} and layer["color_by"] is None
    assert any("not a property" in note for note in layer["notes"])
    assert any("glow" in note for note in layer["notes"])


def test_few_points_are_labelled_many_are_not() -> None:
    few = normalize_layer({"geojson": STATIONS}, 0)
    assert few["style"]["label_property"] == "station"
    many = normalize_layer({"geojson": [_pt(i * 0.01, 0) for i in range(LABEL_LIMIT + 1)]}, 0)
    assert many["style"]["label_property"] is None
    forced = normalize_layer({"geojson": STATIONS, "style": {"labels": False}}, 0)
    assert forced["style"]["label_property"] is None
    named = normalize_layer({"geojson": STATIONS, "style": {"label": "kind"}}, 0)
    assert named["style"]["label_property"] == "kind"


def test_bounds_cross_the_antimeridian_when_narrower() -> None:
    pacific = normalize_layer({"geojson": [_pt(170, -10), _pt(-170, 10)]}, 0)
    assert pacific["bounds"] == [170, -10, -170, 10]
    europe = normalize_layer({"geojson": [_pt(-5, 40), _pt(20, 60)]}, 1)
    assert europe["bounds"] == [-5, 40, 20, 60]
    # The widest gap is 170°W to 5°W, so the narrowest arc runs east from 5°W to 170°W.
    assert merged_bounds([pacific, europe]) == [-5, -10, -170, 60]
    assert "extent" not in view_layer(pacific)


def test_a_dot_is_never_too_small_to_see_or_click() -> None:
    tiny = normalize_layer({"geojson": STATIONS, "style": {"markersize": 4}}, 0)
    assert tiny["style"]["point_size"] == 7.0
    default = normalize_layer({"geojson": STATIONS}, 0)
    assert 9 < default["style"]["point_size"] < 10
