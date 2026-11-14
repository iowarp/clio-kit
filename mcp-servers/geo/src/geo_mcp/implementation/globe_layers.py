"""Layers for the interactive globe: the ``render_feature_map`` schema, resolved for a view.

A layer is ``{"geojson": ..., "name": str, "style": {...}, "id_property": str}``,
the same schema :func:`geo_mcp.implementation.map_render.render_map` takes. Here it is
normalized for a browser globe rather than drawn into a PNG:

- the GeoJSON (FeatureCollection, Feature, geometry, list, JSON string or path) becomes
  one FeatureCollection of the features whose geometry is well formed, each with a
  stable string ``id`` (the ``id_property`` value, else the feature's ``id``, else its
  ``properties.id``, else ``<layer>-<n>``), so a person's pick and a linked table name
  the same feature;
- the style resolves to concrete colours with matplotlib's colour parser, so ``tab:red``
  and ``C2`` mean what they mean on the PNG, and data-driven colouring (``color_by``
  with ``category_colors``, the ``epa_aqi`` scale or a matplotlib colormap) becomes one
  colour per feature id plus the legend entries;
- bounds are computed across the antimeridian: a layer spanning 170°E to 170°W reports
  ``[170, s, -170, n]`` (west greater than east) rather than the whole world.

Nothing here draws or imports a geometry engine: positions are read with the standard
library helpers of :mod:`geo_mcp.implementation.geojson_inspect`.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex

from .geojson_inspect import (
    GeoJSONError,
    _validate_geometry,
    iter_features,
    iter_positions,
    load_geojson,
)
from .map_render import _AQI_FALLBACK, _EPA_AQI_BANDS, _PALETTE, _epa_aqi_color

#: Style keys the globe reads. ``zorder`` orders layers; ``label`` names the property a
#: point's label shows and ``labels`` forces labels on or off.
STYLE_KEYS = frozenset(
    {
        "facecolor",
        "edgecolor",
        "alpha",
        "linewidth",
        "color",
        "markersize",
        "zorder",
        "color_by",
        "scale",
        "category_colors",
        "legend",
        "label",
        "labels",
    }
)

#: Points in a layer up to which each point is labelled by default.
LABEL_LIMIT = 40
#: Properties tried, in order, for a point's label when the style names none.
_LABEL_KEYS = ("name", "title", "label", "station", "site", "id", "code")
_RAMP_STOPS = 7


class GlobeLayerError(ValueError):
    """Raised when a layer cannot be shown on the globe; the message says why."""


def _hex(value: Any, what: str) -> str:
    try:
        return str(to_hex(value, keep_alpha=False))
    except (ValueError, TypeError) as exc:
        raise GlobeLayerError(f"{what} {value!r} is not a colour.") from exc


def _number(value: Any, what: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GlobeLayerError(f"{what} must be a number, not {value!r}.") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise GlobeLayerError(f"{what} must be between {low:g} and {high:g}, not {value!r}.")
    return number


def _source(geojson: Any) -> tuple[Any, str | None]:
    """The parsed GeoJSON and, when it was read from a file, the file's absolute path."""
    if isinstance(geojson, str):
        text = geojson.strip()
        if text and text[0] not in "{[":
            path = Path(text).expanduser()
            if not path.is_file():
                raise GlobeLayerError(f"Layer geojson path not found: {geojson}")
            try:
                return load_geojson(str(path)), str(path.resolve())
            except GeoJSONError as exc:
                raise GlobeLayerError(str(exc)) from exc
    if geojson is None:
        raise GlobeLayerError("Layer has no 'geojson'.")
    try:
        return load_geojson(geojson), None
    except GeoJSONError as exc:
        raise GlobeLayerError(str(exc)) from exc


def _position_problem(geometry: dict[str, Any]) -> str | None:
    for position in iter_positions(geometry):
        if not -90.0 <= position[1] <= 90.0:
            return "latitude outside -90..90"
        if not -540.0 <= position[0] <= 540.0:
            return "longitude outside -540..540"
    return None


def _wrap(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


class Extent:
    """A lon/lat extent whose longitudes span the narrowest arc, across 180° if need be.

    Longitudes are kept per one-degree bin (the smallest and largest in each), so the
    widest empty gap around the circle can be found and extents can be merged.
    """

    def __init__(self) -> None:
        self.south = math.inf
        self.north = -math.inf
        self.bins: dict[int, tuple[float, float]] = {}

    @property
    def empty(self) -> bool:
        return not self.bins

    def add(self, lon: float, lat: float) -> None:
        lon = _wrap(lon)
        self.south = min(self.south, lat)
        self.north = max(self.north, lat)
        key = min(int(math.floor(lon + 180.0)), 359)
        low, high = self.bins.get(key, (lon, lon))
        self.bins[key] = (min(low, lon), max(high, lon))

    def merge(self, other: Extent) -> None:
        if other.empty:
            return
        self.south = min(self.south, other.south)
        self.north = max(self.north, other.north)
        for key, (low, high) in other.bins.items():
            mine = self.bins.get(key, (low, high))
            self.bins[key] = (min(mine[0], low), max(mine[1], high))

    def bbox(self) -> list[float] | None:
        """``[west, south, east, north]``; west > east when it crosses the antimeridian."""
        if self.empty:
            return None
        keys = sorted(self.bins)
        # The gap after each occupied bin, the last one wrapping round to the first.
        gaps = [(keys[(i + 1) % len(keys)] - k) % 360 or 360 for i, k in enumerate(keys)]
        widest = max(range(len(keys)), key=lambda i: gaps[i])
        if widest == len(keys) - 1:
            west, east = self.bins[keys[0]][0], self.bins[keys[-1]][1]
        else:
            west, east = self.bins[keys[widest + 1]][0], self.bins[keys[widest]][1]
        return [round(v, 6) for v in (west, self.south, east, self.north)]


def _feature_id(
    feature: dict[str, Any], props: dict[str, Any], id_property: str | None
) -> tuple[Any, str]:
    if id_property:
        value = props.get(id_property)
        return value, f"property {id_property!r}"
    if feature.get("id") not in (None, ""):
        return feature["id"], "the features' id"
    if props.get("id") not in (None, ""):
        return props["id"], "property 'id'"
    return None, "generated"


def _label_property(features: list[dict[str, Any]], style: dict[str, Any]) -> str | None:
    named = style.get("label")
    if named:
        return str(named)
    for key in _LABEL_KEYS:
        if any(key in (f.get("properties") or {}) for f in features):
            return key
    return None


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _data_colours(
    name: str,
    features: list[dict[str, Any]],
    style: dict[str, Any],
    want_legend: bool,
) -> tuple[dict[str, str], dict[str, Any] | None, list[str]]:
    """Per-feature colours, the legend and notes for ``color_by``; empty when not used."""
    color_by = style.get("color_by")
    notes: list[str] = []
    if not color_by:
        return {}, None, notes
    color_by = str(color_by)
    values = {f["id"]: (f.get("properties") or {}).get(color_by) for f in features}
    if all(v is None for v in values.values()):
        notes.append(f"color_by {color_by!r} is not a property of this layer; drawn in one colour")
        return {}, None, notes
    category_colors = style.get("category_colors")
    scale = str(style.get("scale") or "").strip()
    legend: dict[str, Any]
    if isinstance(category_colors, dict):
        mapping = {str(k): _hex(v, f"category_colors[{k!r}]") for k, v in category_colors.items()}
        colours = {fid: mapping.get(str(v), _AQI_FALLBACK) for fid, v in values.items()}
        unmatched = sorted({str(v) for v in values.values() if str(v) not in mapping})
        if unmatched:
            notes.append(
                f"{len(unmatched)} value(s) of {color_by!r} have no category colour; drawn grey"
            )
        legend = {
            "kind": "categories",
            "field": color_by,
            "entries": [{"label": k, "color": c} for k, c in mapping.items()],
        }
    elif scale.lower() == "epa_aqi":
        colours = {fid: _epa_aqi_color(v) for fid, v in values.items()}
        legend = {
            "kind": "categories",
            "field": color_by,
            "entries": [{"label": label, "color": c} for _u, c, label in _EPA_AQI_BANDS],
        }
    else:
        cmap_name = scale or "viridis"
        try:
            cmap = colormaps[cmap_name]
        except KeyError as exc:
            raise GlobeLayerError(
                f"Layer {name!r}: scale {cmap_name!r} is neither 'epa_aqi' nor a matplotlib "
                "colormap."
            ) from exc
        numeric = [float(cast(float, v)) for v in values.values() if _is_numeric(v)]
        present = [v for v in values.values() if v is not None]
        if numeric and len(numeric) == len(present):
            low, high = min(numeric), max(numeric)
            norm = Normalize(vmin=low, vmax=high if high > low else low + 1.0)
            colours = {
                fid: (
                    to_hex(cmap(norm(float(cast(float, v))))) if _is_numeric(v) else _AQI_FALLBACK
                )
                for fid, v in values.items()
            }
            stops = [to_hex(cmap(i / (_RAMP_STOPS - 1))) for i in range(_RAMP_STOPS)]
            legend = {"kind": "ramp", "field": color_by, "min": low, "max": high, "stops": stops}
        else:
            # As geopandas does for a non-numeric column: one colour per category.
            categories = sorted({str(v) for v in present})
            span = max(len(categories) - 1, 1)
            mapping = {c: to_hex(cmap(i / span)) for i, c in enumerate(categories)}
            colours = {
                fid: mapping.get(str(v), _AQI_FALLBACK) if v is not None else _AQI_FALLBACK
                for fid, v in values.items()
            }
            legend = {
                "kind": "categories",
                "field": color_by,
                "entries": [{"label": k, "color": c} for k, c in mapping.items()],
            }
    missing = sum(1 for v in values.values() if v is None)
    if missing:
        notes.append(f"{missing} feature(s) have no {color_by!r}; drawn grey")
    return colours, (legend if want_legend else None), notes


def normalize_layer(layer: Any, index: int) -> dict[str, Any]:
    """Resolve one ``render_feature_map`` layer for the globe view.

    Returns ``{name, source, geojson, features, geometry, skipped, ids, bounds, style,
    colors, legend, notes}``. ``source`` is the absolute path a layer was read from, or
    ``None`` for inline GeoJSON.

    Raises:
        GlobeLayerError: The layer is not an object, its GeoJSON cannot be read, or its
            style names something that is not a colour or colormap.
    """
    if not isinstance(layer, dict):
        raise GlobeLayerError(f"Layer {index} is not an object.")
    name = str(layer.get("name") or f"layer_{index}")
    style = layer.get("style") or {}
    if not isinstance(style, dict):
        raise GlobeLayerError(f"Layer {name!r}: style must be an object.")
    id_property = layer.get("id_property")
    id_property = str(id_property) if id_property not in (None, "") else None
    try:
        payload, source = _source(layer.get("geojson"))
    except GlobeLayerError as exc:
        raise GlobeLayerError(f"Layer {name!r}: {exc}") from exc

    features: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    extent = Extent()
    geometry_types: set[str] = set()
    seen: set[str] = set()
    id_sources: Counter[str] = Counter()
    duplicates = 0
    for n, raw in enumerate(iter_features(payload)):
        if not isinstance(raw, dict) or raw.get("type") not in ("Feature", None):
            skipped["not a feature"] += 1
            continue
        geometry = raw.get("geometry")
        if geometry is None:
            skipped["no geometry"] += 1
            continue
        errors: list[str] = []
        _validate_geometry(geometry, "geometry", errors)
        problem = errors[0] if errors else _position_problem(geometry)
        if problem:
            skipped["malformed geometry"] += 1
            continue
        raw_props = raw.get("properties")
        props: dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
        value, id_source = _feature_id(raw, props, id_property)
        fid = str(value) if value not in (None, "") else f"{name}-{n}"
        if value in (None, ""):
            id_source = "generated"
        if fid in seen:
            duplicates += 1
            fid = f"{fid}~{n}"
        seen.add(fid)
        id_sources[id_source] += 1
        for position in iter_positions(geometry):
            extent.add(position[0], position[1])
        geometry_types.add(str(geometry.get("type")).replace("Multi", ""))
        features.append({"type": "Feature", "id": fid, "geometry": geometry, "properties": props})

    if id_property and id_sources.get("generated"):
        raise GlobeLayerError(
            f"Layer {name!r}: {id_sources['generated']} feature(s) have no {id_property!r} "
            "property, so they have no stable id."
        )

    want_legend = bool(style.get("legend", True))
    fixed = style.get("color") or style.get("facecolor") or _PALETTE[index % len(_PALETTE)]
    fill = _hex(fixed, f"Layer {name!r}: colour")
    is_point = geometry_types <= {"Point"} and bool(geometry_types)
    stroke_default = "#000000" if is_point else fill
    stroke = _hex(style.get("edgecolor", stroke_default), f"Layer {name!r}: edgecolor")
    if str(style.get("edgecolor", "")).lower() == "none":
        stroke = fill
    colours, legend, notes = _data_colours(name, features, style, want_legend)
    if not colours and want_legend:
        legend = {"kind": "single", "entries": [{"label": name, "color": fill}]}
    points = sum(1 for f in features if str(f["geometry"].get("type")).endswith("Point"))
    label_on = style.get("labels")
    labelled = bool(label_on) if label_on is not None else 0 < points <= LABEL_LIMIT
    resolved = {
        "fill": fill,
        "stroke": stroke,
        "alpha": _number(style.get("alpha", 0.6), "alpha", 0.0, 1.0),
        # A dot must read on any basemap: opaque unless the style asks otherwise.
        "point_alpha": _number(style.get("alpha", 0.95), "alpha", 0.0, 1.0),
        "line_width": max(1.0, 2.0 * _number(style.get("linewidth", 1.0), "linewidth", 0, 50)),
        # A dot must stay visible and clickable on a globe, whatever the PNG marker size.
        "point_size": max(
            7.0,
            min(40.0, 1.4 * math.sqrt(_number(style.get("markersize", 45), "markersize", 0, 4e4))),
        ),
        "zorder": int(_number(style.get("zorder", 5 + index), "zorder", -1e6, 1e6)),
        "label_property": _label_property(features, style) if labelled else None,
    }
    unknown = sorted(set(style) - STYLE_KEYS)
    if unknown:
        notes.append("style keys the globe does not use: " + ", ".join(unknown))
    if duplicates:
        notes.append(f"{duplicates} duplicate id(s) got a '~n' suffix")
    ids_from = (
        "mixed: " + ", ".join(sorted(id_sources))
        if len(id_sources) > 1
        else next(iter(id_sources), "none")
    )
    return {
        "name": name,
        "source": source,
        "id_property": id_property,
        "geojson": {"type": "FeatureCollection", "features": features},
        "features": len(features),
        "geometry": sorted(geometry_types),
        "skipped": dict(skipped),
        "ids": ids_from,
        "bounds": extent.bbox(),
        "extent": extent,
        "style": resolved,
        "color_by": str(style["color_by"]) if style.get("color_by") and colours else None,
        "colors": colours,
        "legend": legend,
        "notes": notes,
    }


def merged_bounds(layers: list[dict[str, Any]]) -> list[float] | None:
    """The bounds of every layer together, across the antimeridian when narrower."""
    total = Extent()
    for layer in layers:
        extent = layer.get("extent")
        if isinstance(extent, Extent):
            total.merge(extent)
    return total.bbox()


def view_layer(layer: dict[str, Any]) -> dict[str, Any]:
    """The layer as the view receives it (no internal extent)."""
    return {k: v for k, v in layer.items() if k != "extent"}


__all__ = [
    "LABEL_LIMIT",
    "STYLE_KEYS",
    "Extent",
    "GlobeLayerError",
    "merged_bounds",
    "normalize_layer",
    "view_layer",
]
