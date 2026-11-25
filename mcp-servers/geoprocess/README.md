# Geoprocess MCP

Install the launcher using the [CLIO Kit setup guide](../../setup.md) before
using the commands below. See [agent integrations](../../README.md#agent-integrations)
for MCP and skill configuration.

The core geospatial operations between getting data and doing science with it:
reprojection, clipping, dissolve and overlay with honest areas, zonal statistics,
point sampling and raster differencing. They run on local files, such as those
the `geoservices` server writes.

Areas are computed in an equal-area CRS (EPSG:6933) and reported in hectares,
with that CRS named. Overlapping polygons are dissolved before totals, so a total
is the area covered rather than a sum of layers. Co-registration methods and
datum, unit and resolution mismatches are reported, never assumed away.

## Tools

### `reproject`

Changes a vector's or raster's CRS. Rasters take `resolution` and `resampling`
(nearest for categories; bilinear or average for continuous values) and report
any declared vertical datum.

### `clip_features`

Cuts features to a polygon layer (`mask`) or a `bbox`, and reports how many
remain and the covered area of polygons.

### `overlay_features`

`dissolve` merges overlapping polygons, optionally per attribute (`by`), and
reports the overlap removed. `intersection`, `union`, `difference`,
`symmetric_difference` and `identity` combine two layers. Every result carries a
covered-area table.

### `zonal_statistics`

Summarises a raster inside polygons (cells, min, max, mean, std, sum, and the
share of each zone with data) or samples it at points (nearest cell, or the mean
within `buffer_m` metres). Polygons are rasterized on the raster's own grid.

### `raster_difference`

Computes first minus second on the first raster's grid, saves the difference and
reports statistics, the overlap, net volume for projected grids, and caveats when
vertical datums, units or resolutions differ.

## Run

```sh
clio-kit mcp-server geoprocess   # via the clio-kit launcher
geoprocess-mcp                  # direct entry point
```

## Test

```sh
uv run pytest -v
```
