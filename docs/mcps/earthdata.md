---
title: Earthdata MCP
description: "Search NASA Earthdata for Earth-science datasets and their files, and download them with an Earthdata Login."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Earthdata"
  icon="🛰️"
  category="Data Processing"
  description="Search NASA Earthdata for Earth-science datasets and their files, and download them with an Earthdata Login."
  version="1.0.0"
  actions={["download_earth_data_granules", "search_earth_datagranules", "search_earth_datasets"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "NASA", "Earthdata", "CMR", "earthaccess", "remote sensing", "datasets"]}
  license="BSD-3-Clause"
  tools={[{"name": "download_earth_data_granules", "description": "Search Earthdata granules, then describe, download or script them.\n\nModes:\n- manifest: return searchable granule metadata, no download performed.\n- download: download files immediately to folder_name on this server.\n  Needs EARTHDATA_USERNAME and EARTHDATA_PASSWORD where the server runs.\n- script: return a Python script that performs the same search and\n  download, to run in a notebook or a code sandbox.\n\nStart with manifest to check what a query matches before downloading.\n\nArgs:\n    folder_name: In download mode, a path relative to this server's safe output\n        directory. In script mode, the path is embedded unchanged and resolves\n        where the generated script runs. In manifest mode, it is reported as\n        the intended download folder.\n    short_name: Short name of the dataset.\n    count: Number of data granules to return.\n    temporal: (Optional) Temporal range in the format (date_from, date_to).\n    bounding_box: (Optional) Bounding box in the format\n        (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).\n    mode: Operation to perform: manifest, download, or script.\n    max_manifest_items: Maximum number of granules returned in manifest mode;\n        ignored in download and script modes.", "function_name": "download_earth_data_granules"}, {"name": "search_earth_datagranules", "description": "Search for data granules on NASA Earthdata.\n\nArgs:\n    short_name: Short name of the dataset.\n    count: Number of data granules to return.\n    temporal: (Optional) Temporal range in the format (date_from, date_to).\n    bounding_box: (Optional) Bounding box in the format\n        (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).\n\nReturns:\n    The data granules found.", "function_name": "search_earth_datagranules"}, {"name": "search_earth_datasets", "description": "Search for datasets on NASA Earthdata.\n\nArgs:\n    search_keywords: Keywords to search for in the dataset titles.\n    count: Number of datasets to return.\n    temporal: (Optional) Temporal range in the format (date_from, date_to).\n    bounding_box: (Optional) Bounding box in the format\n        (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).\n\nReturns:\n    The datasets found: title, short name, abstract, data type, DOI,\n    landing page, visualization and data links.", "function_name": "search_earth_datasets"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `download_earth_data_granules`, `search_earth_datagranules`, `search_earth_datasets`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
