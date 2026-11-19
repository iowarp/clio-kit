# Geoservices MCP

Install the launcher using the [CLIO Kit setup guide](../../setup.md) before
using the commands below. See [agent integrations](../../README.md#agent-integrations)
for MCP and skill configuration.

Reach geospatial data where it is served and bring back only the part inside a
region. Much public Earth-science data is not a plain file: it sits behind ArcGIS
REST services, OGC WFS and WCS endpoints, STAC catalogs, tiled rasters with VRT
mosaics, or lidar point clouds in Entwine Point Tiles. This server turns each of
those into a local GeoJSON, GeoTIFF or LAZ file bounded by
`[west, south, east, north]` in degrees.

Every result says how complete it is. Page limits, cell budgets and octree depth
limits are reported with the reason. A declared extent that cannot be trusted is
reported as such, never used silently.

## Tools

### `describe_geo_service`

Identifies a URL and describes it: ArcGIS service layers, or one layer's fields,
record count, declared extent and the extent measured from the data; WFS feature
types; WCS coverages (filter with `match`) or one coverage's CRS, grid and
envelope; a remote raster's header; an EPT or COPC point cloud's extent and point
count; a STAC catalog. With `bbox`, adds `coverage`: `overlaps`, `disjoint` or
`undetermined`, with the reason when undetermined. Entwine viewer links
(`…/view.html?r="…"`) are followed to their `ept.json`.

### `fetch_features`

Queries an ArcGIS FeatureServer or MapServer layer (bbox, SQL `where`,
`out_fields`) or a WFS 2.0 feature type (bbox), paging past the server's record
limit up to `max_features` (at most 50,000), and saves EPSG:4326 GeoJSON. Older
MapServer layers that cannot return GeoJSON are read as Esri JSON and converted;
the result's `paging.format` says which happened. `count_only` returns only the
number of matching features, the reliable coverage test when a layer's declared
extent is wrong.

### `read_raster_window`

Mosaics the part of one or more rasters inside the region into one GeoTIFF:
GeoTIFF/COG, VRT, ArcGrid (`.adf`) and IMG, local or over HTTP through GDAL range
reads, or an OGC WCS 2.0.1 coverage subset (`coverage_id`). Tiles in different
CRSs are warped to one CRS (`dst_crs`, or the first tile's). The output is
decimated to `max_cells` and the factor is reported. Units come from the raster
when it declares them; many services do not, in which case `units` is null.

### `read_point_cloud`

Reads the points inside the region from an EPT dataset (for example USGS 3DEP's
`usgs-lidar-public` bucket) or a COPC file and saves LAZ. It reads the finest
octree level that fits `max_points` and returns `depth`, `spacing` and
`complete`; when `complete` is false, `limit_reason` says why. `classifications`
filters ASPRS classes (2 is ground).

### `search_stac`

Searches a STAC API (`/search`, with paging) or walks a static catalog (at most
200 documents) by bbox, `datetime` interval and collections, and saves the items
as GeoJSON with their asset links.

## Run

```sh
clio-kit mcp-server geoservices   # via the clio-kit launcher
geoservices-mcp                  # direct entry point
```

Outputs are written under `CLIO_KIT_ARTIFACTS`, or a per-user temporary
directory when it is unset.

## Test

```sh
uv run pytest -v
```

Unit tests use local fixtures and a local HTTP server. `GEOSERVICES_LIVE=1`
also runs the live checks against public services.
