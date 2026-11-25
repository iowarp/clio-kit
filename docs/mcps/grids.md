---
title: Grids MCP
description: "MCP server for hand edits to 2-D model grids (bathymetry, land masks, topography): an interactive editor, an ordered edit list located by longitude and latitude, and replay onto a regenerated grid with content hashes"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Grids"
  icon="🗺️"
  category="Analysis & Visualization"
  description="MCP server for hand edits to 2-D model grids (bathymetry, land masks, topography): an interactive editor, an ordered edit list located by longitude and latitude, and replay onto a regenerated grid with content hashes"
  version="1.0.0"
  actions={["grid_open", "grid_edit", "grid_replay", "grid_window"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "grid", "bathymetry", "land mask", "model", "MCP Apps"]}
  license="BSD-3-Clause"
  tools={[{"name": "grid_open", "description": "Open a 2-D gridded field (model bathymetry, land mask, topography) in an editor the person can draw on. Returns its shape, extent, value range and content hash, and the default output path for edits.", "function_name": "grid_open"}, {"name": "grid_edit", "description": "Apply an ordered list of edits to a base grid and write the result beside it with an edit log (<output>.edits.json) recording the base hash, each edit with the cells it changed, and the result hash. The base file is never changed. The grid editor calls this for every change; undo is the same list without its last edit.", "function_name": "grid_edit"}, {"name": "grid_replay", "description": "Apply a saved edit log to a grid, usually one regenerated after the edits were made. Edits are located by longitude and latitude, so they land on the same places; the result says whether the base grid is the logged one and, edit by edit, whether it changed the same number of cells.", "function_name": "grid_replay"}, {"name": "grid_window", "description": "Values of a grid strided for display. Called by the grid editor.", "function_name": "grid_window"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `grid_open`, `grid_edit`, `grid_replay`, `grid_window`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
