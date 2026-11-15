---
title: Globe MCP
description: "Show regions, places and data footprints on an interactive 3D globe with OpenStreetMap, and find places by name (MCP Apps)"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Globe"
  icon="🗺️"
  category="Data Processing"
  description="Show regions, places and data footprints on an interactive 3D globe with OpenStreetMap, and find places by name (MCP Apps)"
  version="1.0.0"
  actions={["show-map", "geocode"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "globe", "map", "CesiumJS", "OpenStreetMap", "geocoding", "MCP Apps"]}
  license="BSD-3-Clause"
  tools={[{"name": "show-map", "description": "Display an interactive world map zoomed to a specific bounding box. Use the GeoCode tool to find the bounding box of a location. The widget is interactive and exposes tools for navigation (fly to locations) and querying the current view.", "function_name": "show-map"}, {"name": "geocode", "description": "Search for places using OpenStreetMap. Returns coordinates and bounding boxes for up to 5 matches.", "function_name": "geocode"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `show-map`, `geocode`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
