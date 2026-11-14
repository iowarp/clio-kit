"""The interactive globe: ``show_on_globe`` and its MCP Apps view (``ui://geo/globe``).

``show_on_globe`` takes the layers ``render_feature_map`` takes and shows them on a
CesiumJS globe the person can turn, zoom, toggle and click. The agent reads a compact
summary (layer names, feature counts, bounds); the GeoJSON travels to the view only, in
the result's ``structuredContent``.

The view reports the camera and the person's pick with ``ui/update-model-context``
(state keys ``selected_feature``, ``selection``, ``camera``, ``layers``) and offers app
tools (``fly_to``, ``select_feature``, ``add_layer``, ``remove_layer``, ``set_style``,
``get_view``). Layer edits go through ``edit_globe_view``, a tool only the view calls,
so the saved view follows them and a reopen shows what the person last saw.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.apps import AppConfig, ResourceCSP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from .implementation.globe_layers import (
    GlobeLayerError,
    merged_bounds,
    normalize_layer,
    view_layer,
)
from .implementation.globe_store import (
    GlobeStoreError,
    load_view,
    new_view_id,
    save_view,
)

GLOBE_URI = "ui://geo/globe"
#: What a host replays to reopen the view: the saved view id, never the GeoJSON.
REOPEN_META = "ai.iowarp.clio/reopen"
PATHS_META = "ai.iowarp.clio/paths"
STATE_META = "ai.iowarp.clio/state"
#: CesiumJS, pinned, from Cesium's own CDN; OpenStreetMap's standard raster tiles.
CESIUM_ORIGIN = "https://cesium.com"
TILE_ORIGIN = "https://tile.openstreetmap.org"
#: The GeoJSON one view carries, as JSON. Past this a browser globe stalls.
MAX_VIEW_BYTES = 24 * 1024 * 1024

VIEW_STATE = [
    {
        "key": "selected_feature",
        "tool": "select_feature",
        "description": "The id of the feature the person picked; set it to fly to a feature.",
    },
    {
        "key": "selection",
        "description": "The picked feature: layer, id, name, properties and position.",
    },
    {"key": "camera", "description": "Where the camera is and the area it shows."},
    {"key": "layers", "description": "Each layer's name, feature count and visibility."},
    {"key": "view", "description": "The saved view id; show_on_globe view= reopens it."},
]
APP_TOOLS = ["fly_to", "select_feature", "add_layer", "remove_layer", "set_style", "get_view"]

_LAYERS_DESCRIPTION = (
    "Ordered layers (later layers draw on top), the render_feature_map schema: "
    "{'geojson': FeatureCollection|Feature|geometry|list|JSON-string|path, 'name': str, "
    "'id_property': property naming each feature's stable id, 'style': {facecolor, "
    "edgecolor, alpha, linewidth, color, markersize, zorder, color_by, scale, "
    "category_colors, legend, label (property shown as a point's label), labels (bool)}}."
)


def _text(text: str) -> TextContent:
    return TextContent(type="text", text=text)


def _fmt_bounds(bounds: list[float] | None) -> str:
    if not bounds:
        return "no bounds"
    return "[" + ", ".join(f"{v:.4g}" for v in bounds) + "]"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _geometry_words(layer: dict[str, Any]) -> str:
    kinds = layer["geometry"]
    noun = {"Point": "point", "LineString": "line", "Polygon": "polygon"}
    if len(kinds) == 1 and kinds[0] in noun:
        return _plural(layer["features"], noun[kinds[0]])
    detail = ", ".join(k.lower() for k in kinds) or "no geometry"
    return f"{_plural(layer['features'], 'feature')} ({detail})"


def layer_line(layer: dict[str, Any]) -> str:
    """One compact line about a layer, for the agent."""
    if layer.get("error"):
        return f"- {layer['name']}: not shown, {layer['error']}"
    parts = [_geometry_words(layer), f"ids from {layer['ids']}"]
    legend = layer.get("legend") or {}
    if layer.get("color_by"):
        kind = legend.get("kind")
        extent = (
            f"{len(legend.get('entries') or [])} categories"
            if kind == "categories"
            else f"{legend.get('min'):.4g} to {legend.get('max'):.4g}"
            if kind == "ramp"
            else "data-driven"
        )
        parts.append(f"coloured by {layer['color_by']} ({extent})")
    if layer["style"].get("label_property"):
        parts.append(f"labelled by {layer['style']['label_property']}")
    parts.append(f"bounds {_fmt_bounds(layer['bounds'])}")
    skipped = layer.get("skipped") or {}
    if skipped:
        parts.append("skipped " + ", ".join(f"{n} ({why})" for why, n in sorted(skipped.items())))
    parts.extend(layer.get("notes") or [])
    if layer.get("source"):
        parts.append(f"from {layer['source']}")
    return f"- {layer['name']}: " + "; ".join(parts)


def _spec(layer: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
    """What the saved view keeps of a layer: the path for a file, else the GeoJSON."""
    spec: dict[str, Any] = {
        "name": normalized["name"],
        "geojson": normalized["source"] or layer.get("geojson"),
    }
    if layer.get("style"):
        spec["style"] = layer["style"]
    if normalized.get("id_property"):
        spec["id_property"] = normalized["id_property"]
    return spec


def _check_bbox(bbox: list[float] | None) -> list[float] | None:
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ToolError("bbox must be [min_lon, min_lat, max_lon, max_lat].")
    west, south, east, north = (float(v) for v in bbox)
    if not (-90 <= south < north <= 90) or not (-540 <= west <= 540 and -540 <= east <= 540):
        raise ToolError("bbox must be [min_lon, min_lat, max_lon, max_lat] with south < north.")
    return [west, south, east, north]


def _check_camera(camera: dict[str, Any] | None) -> dict[str, float] | None:
    if camera is None:
        return None
    try:
        lon, lat = float(camera["lon"]), float(camera["lat"])
        height = float(camera.get("height_m", 2_000_000))
        heading = float(camera.get("heading_deg", 0))
        pitch = float(camera.get("pitch_deg", -90))
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError(
            "camera must be {lon, lat, height_m?, heading_deg?, pitch_deg?} in degrees and metres."
        ) from exc
    if not (-90 <= lat <= 90 and -540 <= lon <= 540 and 1 <= height <= 1e8 and -90 <= pitch <= 0):
        raise ToolError(
            "camera is out of range (lat -90..90, height 1 m..100,000 km, pitch -90..0)."
        )
    return {"lon": lon, "lat": lat, "height_m": height, "heading_deg": heading, "pitch_deg": pitch}


def resolve_layers(
    specs: list[Any], *, strict: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize layer specs; returns (normalized layers, the specs to save).

    With ``strict`` a bad layer fails the call; otherwise (a reopen) it is reported as a
    layer with an ``error`` and the others are shown.
    """
    normalized: list[dict[str, Any]] = []
    saved: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        try:
            layer = normalize_layer(spec, index)
        except GlobeLayerError as exc:
            if strict:
                raise ToolError(str(exc)) from exc
            name = str(spec.get("name") or f"layer_{index}") if isinstance(spec, dict) else ""
            normalized.append({"name": name or f"layer_{index}", "error": str(exc)})
            if isinstance(spec, dict):
                saved.append(spec)
            continue
        normalized.append(layer)
        saved.append(_spec(spec, layer))
    names = [layer["name"] for layer in normalized]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ToolError("Layer names must be distinct; repeated: " + ", ".join(dupes))
    return normalized, saved


def _view_payload(
    view_id: str, title: str, layers: list[dict[str, Any]], record: dict[str, Any]
) -> dict[str, Any]:
    shown = [layer for layer in layers if not layer.get("error")]
    payload = {
        "view": view_id,
        "title": title,
        "bounds": merged_bounds(shown),
        "bbox": record.get("bbox"),
        "camera": record.get("camera"),
        "layers": [view_layer(layer) for layer in layers],
    }
    size = len(json.dumps(payload["layers"]))
    if size > MAX_VIEW_BYTES:
        raise ToolError(
            f"These layers are {size / 1e6:.1f} MB of GeoJSON, more than a browser globe "
            f"draws ({MAX_VIEW_BYTES / 1e6:.0f} MB). Filter or simplify the features first."
        )
    return payload


def summary_text(payload: dict[str, Any], *, saved: str | None, restored: bool) -> str:
    """The agent-facing description of the view: never the GeoJSON itself."""
    layers = payload["layers"]
    title = f' "{payload["title"]}"' if payload.get("title") else ""
    head = "Restored" if restored else "Showing"
    lines = [
        f"{head} {_plural(len(layers), 'layer')} on the globe{title} (view {payload['view']}):"
    ]
    lines.extend(layer_line(layer) for layer in layers)
    target = (
        "the requested camera"
        if payload.get("camera")
        else (f"bbox {_fmt_bounds(payload['bbox'])}" if payload.get("bbox") else "these bounds")
    )
    lines.append(
        f"All layers: bounds {_fmt_bounds(payload['bounds'])}; the view opens on {target}."
    )
    lines.append(
        "The person can toggle layers and click a feature; their pick reaches you as "
        "selected_feature (the feature id) and selection (its properties). The view's tools: "
        + ", ".join(APP_TOOLS)
        + "."
    )
    if saved:
        lines.append(f"This view could not be saved ({saved}), so it cannot be reopened by id.")
    else:
        lines.append(
            f"To show these layers again, for example in a dashboard, call show_on_globe with "
            f"view={payload['view']!r} (and a title) instead of sending the GeoJSON again."
        )
    return "\n".join(lines)


def show_on_globe(
    layers: list[dict[str, Any]] | None = None,
    files: list[str] | None = None,
    view: str | None = None,
    bbox: list[float] | None = None,
    camera: dict[str, Any] | None = None,
    title: str = "",
) -> ToolResult:
    """Build the view, save it and return the lean result."""
    restored = view is not None
    if restored:
        if layers or files:
            raise ToolError("Pass view to show a saved view again, or layers/files, not both.")
        try:
            record = load_view(str(view))
        except GlobeStoreError as exc:
            raise ToolError(f"{exc} [{exc.reason}]") from exc
        if bbox is not None or camera is not None or title:
            record = {**record}
            record["bbox"] = _check_bbox(bbox) if bbox is not None else record.get("bbox")
            record["camera"] = _check_camera(camera) if camera is not None else record.get("camera")
            record["title"] = title or record.get("title", "")
        specs = list(record["layers"])
    else:
        specs = list(layers or []) + [
            {"geojson": path, "name": Path(path).stem} for path in (files or [])
        ]
        if not specs:
            raise ToolError("Give layers (GeoJSON with a style), files (GeoJSON paths) or view.")
        record = {
            "id": new_view_id(),
            "title": title,
            "bbox": _check_bbox(bbox),
            "camera": _check_camera(camera),
        }
    normalized, saved_specs = resolve_layers(specs, strict=not restored)
    if not any(layer.get("features") for layer in normalized):
        reasons = "; ".join(layer_line(layer) for layer in normalized)
        raise ToolError(f"No layer has a feature to show: {reasons}")
    record["layers"] = saved_specs
    payload = _view_payload(record["id"], str(record.get("title") or ""), normalized, record)
    unsaved: str | None = None
    try:
        save_view(record)
    except GlobeStoreError as exc:
        unsaved = exc.reason
    reopen: dict[str, Any] = {"view": record["id"]}
    if record.get("title"):
        reopen = {"title": record["title"], **reopen}
    meta = {} if unsaved else {REOPEN_META: {"arguments": reopen}}
    if unsaved:
        payload["unsaved_reason"] = unsaved
    return ToolResult(
        content=[_text(summary_text(payload, saved=unsaved, restored=restored))],
        structured_content=payload,
        meta=meta,
    )


def edit_globe_view(
    view: str,
    action: Literal["add", "remove", "style"],
    layer: dict[str, Any] | None = None,
    name: str | None = None,
    style: dict[str, Any] | None = None,
) -> ToolResult:
    """Apply one layer edit to a saved view and return the layer as the view draws it."""
    try:
        record = load_view(view)
    except GlobeStoreError as exc:
        raise ToolError(f"{exc} [{exc.reason}]") from exc
    specs: list[dict[str, Any]] = [s for s in record["layers"] if isinstance(s, dict)]
    names = [str(s.get("name") or "") for s in specs]
    if action == "add":
        if not isinstance(layer, dict):
            raise ToolError("add needs layer: {geojson, name, style?, id_property?}.")
        layer_name = str(layer.get("name") or f"layer_{len(specs)}")
        layer = {**layer, "name": layer_name}
        index = names.index(layer_name) if layer_name in names else len(specs)
    else:
        if not name or name not in names:
            known = ", ".join(n for n in names if n) or "none"
            raise ToolError(f"No layer {name!r} in view {view}; its layers: {known}.")
        index = names.index(name)
        if action == "remove":
            del specs[index]
            record["layers"] = specs
            _save(record)
            return ToolResult(
                content=[_text(f"Removed layer {name!r} from view {view}.")],
                structured_content={"view": view, "removed": name},
            )
        merged = {**(specs[index].get("style") or {})}
        for key, value in (style or {}).items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        layer = {**specs[index], "style": merged}
    try:
        normalized = normalize_layer(layer, index)
    except GlobeLayerError as exc:
        raise ToolError(str(exc)) from exc
    if index < len(specs):
        specs[index] = _spec(layer, normalized)
    else:
        specs.append(_spec(layer, normalized))
    record["layers"] = specs
    _save(record)
    drawn = view_layer(normalized)
    if len(json.dumps(drawn)) > MAX_VIEW_BYTES:
        raise ToolError("This layer is too large for a browser globe; simplify it first.")
    verb = "Added" if action == "add" else "Restyled"
    return ToolResult(
        content=[_text(f"{verb} layer in view {view}:\n" + layer_line(normalized))],
        structured_content={"view": view, "layer": drawn},
    )


def _save(record: dict[str, Any]) -> None:
    try:
        save_view(record)
    except GlobeStoreError as exc:
        raise ToolError(f"{exc} [{exc.reason}]") from exc


def register_globe(mcp: FastMCP) -> None:
    """Register the globe tools and the view resource on the geo server."""

    @mcp.tool(
        name="show_on_globe",
        title="Show On Globe",
        description=(
            "Show GeoJSON layers (points, lines, polygons: stations, regions, footprints, "
            "tracks) on an interactive 3D globe the person can turn, zoom and click. Takes "
            "render_feature_map's layer schema and styles (color_by, category_colors, "
            "legend). For GeoJSON already in a file, pass its path in files. Returns layer "
            "names, feature counts and bounds, and a view id that shows the same layers "
            "again. The person's clicked feature is reported back as selected_feature."
        ),
        app=AppConfig(resource_uri=GLOBE_URI),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False},
        tags={"geospatial", "map", "visualization", "geojson", "globe"},
        meta={
            STATE_META: VIEW_STATE,
            PATHS_META: {"files": {"machine": "local", "access": "read"}},
        },
    )
    async def show_on_globe_tool(
        # The title comes first: a host that names a view by its first short text
        # argument (CLIO's view label) then reads the title, not a view id.
        title: Annotated[str, Field(description="A title shown on the view.")] = "",
        layers: Annotated[
            list[dict[str, Any]] | None, Field(description=_LAYERS_DESCRIPTION)
        ] = None,
        files: Annotated[
            list[str] | None,
            Field(description="GeoJSON file paths, one layer each, named after the file."),
        ] = None,
        view: Annotated[
            str | None,
            Field(description="A view id from an earlier show_on_globe, to show it again."),
        ] = None,
        bbox: Annotated[
            list[float] | None,
            Field(description="Optional area to fly to: [min_lon, min_lat, max_lon, max_lat]."),
        ] = None,
        camera: Annotated[
            dict[str, Any] | None,
            Field(description="Optional camera: {lon, lat, height_m, heading_deg, pitch_deg}."),
        ] = None,
    ) -> ToolResult:
        return show_on_globe(layers, files, view, bbox, camera, title)

    @mcp.tool(
        name="edit_globe_view",
        title="Edit Globe View",
        description=(
            "Add, remove or restyle one layer of a saved globe view. Called by the globe "
            "view itself when the person or the agent changes its layers."
        ),
        app=AppConfig(visibility=["app"]),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
        tags={"geospatial", "globe"},
    )
    async def edit_globe_view_tool(
        view: Annotated[str, Field(description="The view id.")],
        action: Annotated[Literal["add", "remove", "style"], Field(description="The edit.")],
        layer: Annotated[
            dict[str, Any] | None, Field(description="For add: the layer, as in show_on_globe.")
        ] = None,
        name: Annotated[
            str | None, Field(description="For remove and style: the layer's name.")
        ] = None,
        style: Annotated[
            dict[str, Any] | None,
            Field(description="For style: style keys to set; a null value removes a key."),
        ] = None,
    ) -> ToolResult:
        return edit_globe_view(view, action, layer, name, style)

    @mcp.resource(
        GLOBE_URI,
        name="geo_globe",
        description="An interactive CesiumJS globe showing the layers of show_on_globe.",
        mime_type="text/html;profile=mcp-app",
        app=AppConfig(
            csp=ResourceCSP(
                resource_domains=[CESIUM_ORIGIN, TILE_ORIGIN],
                connect_domains=[CESIUM_ORIGIN, TILE_ORIGIN],
            ),
            prefers_border=False,
        ),
    )
    def globe_resource() -> str:
        return resources.files("geo_mcp").joinpath("globe_view.html").read_text("utf-8")


__all__ = [
    "GLOBE_URI",
    "MAX_VIEW_BYTES",
    "VIEW_STATE",
    "edit_globe_view",
    "register_globe",
    "show_on_globe",
    "summary_text",
]
