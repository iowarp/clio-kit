---
title: Cad MCP
description: "Model solid parts with build123d code, export STL and STEP, and rotate them in an interactive 3D view."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Cad"
  icon="📐"
  category="Utilities"
  description="Model solid parts with build123d code, export STL and STEP, and rotate them in an interactive 3D view."
  version="1.0.0"
  actions={["build_part", "show_part", "part_mesh", "remember_part_view", "list_parts", "part_examples"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "CAD", "build123d", "OpenCascade", "STEP", "STL", "parametric"]}
  license="BSD-3-Clause"
  tools={[{"name": "build_part", "description": "Model a part: run build123d code (from build123d import * is already done; assign the finished shape to result), export part.stl and part.step to <out_dir>/<name>, and open it in an interactive 3D view the person can rotate. Returns the size, bounds, volume, area, solid count and validity, and the file paths. Building the same name again replaces that part. See part_examples for worked scripts.", "function_name": "build_part"}, {"name": "show_part", "description": "Open a part build_part made in the interactive 3D view again.", "function_name": "show_part"}, {"name": "part_mesh", "description": "Triangles of a part for the 3D view to draw. Called by the view itself.", "function_name": "part_mesh"}, {"name": "remember_part_view", "description": "Save the direction the part view looks from. Called by the view itself.", "function_name": "remember_part_view"}, {"name": "list_parts", "description": "The parts built in a folder (default the parts folder), with size and volume.", "function_name": "list_parts"}, {"name": "part_examples", "description": "Worked build123d scripts to start from: naca_wing_section, ahmed_car_body, cylinder, pipe_bend. Give name for one script; empty lists them all.", "function_name": "part_examples"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `build_part`, `show_part`, `part_mesh`, `remember_part_view`, `list_parts`, `part_examples`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
