---
title: Geoservices MCP
description: "Fetch geospatial data where it is served: ArcGIS REST, OGC WFS and WCS, STAC, remote rasters and EPT/COPC lidar, by region"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Geoservices"
  icon="🗺️"
  category="Data Processing"
  description="Fetch geospatial data where it is served: ArcGIS REST, OGC WFS and WCS, STAC, remote rasters and EPT/COPC lidar, by region"
  version="1.0.0"
  actions={["describe_geo_service", "fetch_features", "read_raster_window", "read_point_cloud", "search_stac"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "geospatial", "arcgis", "wfs", "wcs", "stac", "cog", "lidar", "ept", "copc"]}
  license="BSD-3-Clause"
  tools={[{"name": "describe_geo_service", "description": "Identify a geospatial URL and describe it: ArcGIS FeatureServer/MapServer/ImageServer layers, fields, record count and real data extent; OGC WFS feature types; WCS coverages (filter by words) or one coverage's grid; a remote raster's CRS, size, resolution and units; an Entwine/EPT or COPC point cloud's extent and point count; a STAC catalog. With bbox, reports whether the declared or measured extent overlaps it. Entwine viewer links are followed to their point cloud.", "function_name": "describe_geo_service"}, {"name": "fetch_features", "description": "Fetch vector features from an ArcGIS FeatureServer/MapServer layer (bbox and SQL where clause) or an OGC WFS feature type (bbox), paging through the service's record limit, and save them as a GeoJSON FeatureCollection in EPSG:4326. count_only returns just the number of matching features, which is the reliable way to test coverage when a layer's declared extent is wrong.", "function_name": "fetch_features"}, {"name": "read_raster_window", "description": "Read only the part of one or more rasters inside a region and save it as one GeoTIFF: remote or local GeoTIFF/COG, VRT mosaics, ArcGrid (.adf) and IMG tiles through ranged HTTP reads, or an OGC WCS coverage subset (give coverage_id). Tiles in different CRSs are warped to one. The output is decimated to max_cells and the decimation is reported. Returns CRS, shape, resolution, units, valid fraction and statistics.", "function_name": "read_raster_window"}, {"name": "read_point_cloud", "description": "Read the lidar points inside a region from an Entwine Point Tiles (ept.json) dataset, such as USGS 3DEP's public bucket (Entwine viewer links are accepted), or a COPC .copc.laz file, and save them as LAZ. Reads the finest octree level that fits max_points and reports the level, point spacing and whether the result is full density. Optional classification filter (2 = ground).", "function_name": "read_point_cloud"}, {"name": "search_stac", "description": "Find STAC items by region, time interval (e.g. 2020-01-01/2021-12-31) and collection, through a STAC API search or a bounded walk of a static catalog. Saves the items as GeoJSON and returns each item's id, time, bbox and asset links.", "function_name": "search_stac"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `describe_geo_service`, `fetch_features`, `read_raster_window`, `read_point_cloud`, `search_stac`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
