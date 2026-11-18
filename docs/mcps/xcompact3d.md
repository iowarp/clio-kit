---
title: Xcompact3D MCP
description: "Set up Xcompact3d flow simulations around a CAD body, follow them live and read the forces back."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Xcompact3D"
  icon="🌀"
  category="Data Processing"
  description="Set up Xcompact3d flow simulations around a CAD body, follow them live and read the forces back."
  version="1.0.0"
  actions={["prepare_case", "check_case", "read_forces", "build_instructions"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "CFD", "Xcompact3d", "Incompact3d", "aerodynamics", "immersed boundary", "ADIOS2"]}
  license="BSD-3-Clause"
  tools={[{"name": "prepare_case", "description": "Write an Xcompact3d run directory for the flow past a body given as an STL. The body is scaled so reference_length (default its length along x) is 1, turned nose-up by angle_deg about +z, and placed upstream in a box with a uniform inflow along +x, a convective outflow and periodic sides. span 'periodic' treats an extruded section (a wing section, a cylinder) as spanning the periodic z direction; 'free' puts a 3D body (a car, a finite wing) in the middle; 'auto' decides from the shape. Returns the grid, time step, snapshot size, the run.sh to start it, relay_job (a job spec for a batch launcher such as clio-relay run) and view (a clio.view-descriptor.v1 for ParaView's open_view).", "function_name": "prepare_case"}, {"name": "check_case", "description": "How far a run is: the time step and simulated time from its log, the snapshots written, the seconds per step and left, whether the body mask was loaded and whether the run failed or diverged, with the log's last lines and the view descriptor.", "function_name": "check_case"}, {"name": "read_forces", "description": "Drag and lift coefficients on the body over the run: mean, rms, extremes, lift to drag, the dominant lift frequency and its Strouhal number on the body length and on the frontal height, plus the time series (at most 400 points) for a chart. The first discard_fraction of the run (the start-up) is left out of the statistics.", "function_name": "read_forces"}, {"name": "build_instructions", "description": "How to build the patched Xcompact3d this server prepares cases for: the needs, the two patch scripts (written to patch_dir when given) and the build commands.", "function_name": "build_instructions"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `prepare_case`, `check_case`, `read_forces`, `build_instructions`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
