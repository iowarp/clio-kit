---
title: Geoprocess MCP
description: "Core geospatial operations: reproject, clip, dissolve and overlay, zonal statistics and raster differencing"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Geoprocess"
  icon="📐"
  category="Utilities"
  description="Core geospatial operations: reproject, clip, dissolve and overlay, zonal statistics and raster differencing"
  version="1.0.0"
  actions={["reproject", "clip_features", "overlay_features", "zonal_statistics", "raster_difference"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "geospatial", "gis", "reprojection", "overlay", "zonal-statistics", "raster"]}
  license="BSD-3-Clause"
  tools={[{"name": "reproject", "description": "Reproject a vector file (GeoJSON, shapefile, GeoPackage, zipped shapefile) or a raster (GeoTIFF, VRT, IMG) to another CRS, e.g. EPSG:32610 or EPSG:4326. Rasters take an optional resolution and resampling (nearest for categories, bilinear or average for continuous values). Reports source and target CRS and the raster's vertical datum when it declares one.", "function_name": "reproject"}, {"name": "clip_features", "description": "Clip vector features to a polygon layer (mask) or a bbox [west, south, east, north] in degrees, save GeoJSON, and report how many features remain and, for polygons, the covered area in hectares.", "function_name": "clip_features"}, {"name": "overlay_features", "description": "Polygon overlay with honest areas. operation=dissolve merges overlapping polygons (optionally per attribute 'by'), so the total is the area covered, and reports how much overlap was removed. intersection, union, difference, symmetric_difference and identity combine 'source' with 'other'. Areas are in hectares computed in an equal-area CRS.", "function_name": "overlay_features"}, {"name": "zonal_statistics", "description": "Summarise a raster inside polygons (count, min, max, mean, std, sum and the share of each zone with data), optionally grouped by an attribute, or sample it at point features (nearest cell, or the mean within buffer_m metres). Zones are rasterized on the raster's own grid.", "function_name": "zonal_statistics"}, {"name": "raster_difference", "description": "Subtract one raster from another (first minus second) on the first raster's grid, e.g. two DEMs for elevation change or snow depth, save the difference GeoTIFF and report its statistics, overlap, net volume for projected grids, and caveats when vertical datums, units or resolutions differ.", "function_name": "raster_difference"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `reproject`, `clip_features`, `overlay_features`, `zonal_statistics`, `raster_difference`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
