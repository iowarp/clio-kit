---
title: ParaView MCP
description: "ParaView MCP v1.0.0 - Part of Agent Toolkit (IoWarp Platform). 12 tools for scientific 3D visualization: load scientific data, generate isosurfaces, create data slices, volume rendering, flow streamlines. Enables AI agents to create autonomous scientific visualizations with native ADIOS2/BP5 support."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail 
  name="ParaView"
  icon="🔬"
  category="Scientific Visualization"
  description="ParaView MCP v1.0.0 - Part of Agent Toolkit (IoWarp Platform). 12 tools for scientific 3D visualization: load scientific data, generate isosurfaces, create data slices, volume rendering, flow streamlines. Enables AI agents to create autonomous scientific visualizations with native ADIOS2/BP5 support."
  version="1.0.0"
  actions={["load_scientific_data", "generate_isosurface", "create_data_slice", "configure_volume_display", "generate_flow_streamlines", "take_viewport_screenshot", "apply_field_coloring", "set_representation_type", "get_available_arrays", "rotate_camera", "reset_camera"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["paraview", "scientific-visualization", "3d-visualization", "adios2", "bp5", "vtk", "scientific-computing", "visualization-pipeline", "autonomous-visualization"]}
  license="BSD-3-Clause"
  tools={[{"name": "load_scientific_data", "description": "Load scientific datasets from various file formats into ParaView for visualization and analysis. Supports VTK, EXODUS, CSV, RAW, BP5, and other scientific data formats.", "function_name": "load_scientific_data"}, {"name": "generate_isosurface", "description": "Create an isosurface visualization for extracting surfaces of constant value from 3D scalar data.", "function_name": "generate_isosurface"}, {"name": "create_data_slice", "description": "Create a slice through volume data to examine cross-sections of 3D datasets.", "function_name": "create_data_slice"}, {"name": "configure_volume_display", "description": "Toggle volume rendering for direct 3D visualization of volumetric data.", "function_name": "configure_volume_display"}, {"name": "generate_flow_streamlines", "description": "Create streamlines from vector field data for visualizing flow patterns and trajectories.", "function_name": "generate_flow_streamlines"}, {"name": "take_viewport_screenshot", "description": "Capture a screenshot of the current ParaView viewport.", "function_name": "take_viewport_screenshot"}, {"name": "apply_field_coloring", "description": "Color the visualization by a specific data field for enhanced analysis.", "function_name": "apply_field_coloring"}, {"name": "set_representation_type", "description": "Set the representation type (surface, wireframe, points, volume) for visualization.", "function_name": "set_representation_type"}, {"name": "get_available_arrays", "description": "Get a list of available data arrays in the active source.", "function_name": "get_available_arrays"}, {"name": "rotate_camera", "description": "Rotate the camera to adjust the viewing perspective.", "function_name": "rotate_camera"}, {"name": "reset_camera", "description": "Reset the camera to show all data with optimal viewing parameters.", "function_name": "reset_camera"}]}
>

### 1. Scientific Data Visualization
```
Load /data/simulation_output.vtk with temperature data, create an isosurface at temperature 300, and take a screenshot.
```

**Tools called:**
- `load_scientific_data` - Load VTK simulation data
- `generate_isosurface` - Create isosurface at temperature 300
- `take_viewport_screenshot` - Capture visualization

### 2. Volume Visualization with Flow Analysis
```
Using /data/fluid_dynamics.bp5, create volume rendering of pressure field and add streamlines to visualize flow patterns.
```

**Tools called:**
- `load_scientific_data` - Load ADIOS2/BP5 fluid dynamics data
- `configure_volume_display` - Enable volume rendering
- `generate_flow_streamlines` - Create flow streamlines
- `take_viewport_screenshot` - Document result

### 3. Multi-Slice Data Exploration
```
Load /data/medical_scan.vti and create three orthogonal slices through the center, color by density field.
```

**Tools called:**
- `load_scientific_data` - Load medical imaging data
- `create_data_slice` - Create XY, XZ, YZ slices
- `apply_field_coloring` - Color by density
- `take_viewport_screenshot` - Capture multi-slice view

### 4. Comparative Isosurface Analysis
```
From /data/pressure_field.vtk, create multiple isosurfaces at pressure values 100, 200, and 300 as wireframe for comparison.
```

**Tools called:**
- `load_scientific_data` - Load pressure field data
- `generate_isosurface` - Create multiple isosurfaces
- `set_representation_type` - Set wireframe representation
- `apply_field_coloring` - Apply pressure-based coloring

### 5. Advanced Flow Visualization Pipeline
```
Using /data/turbulent_flow.bp5, examine available fields, create volume rendering of velocity magnitude, and add 50 streamlines.
```

**Tools called:**
- `load_scientific_data` - Load turbulent flow simulation
- `get_available_arrays` - Examine available data fields
- `configure_volume_display` - Enable volume rendering
- `generate_flow_streamlines` - Create 50 streamlines

### 6. Data Quality Assessment and Visualization
```
Load /data/experimental_results.csv, check available fields, and create the most appropriate 3D visualization approach.
```

**Tools called:**
- `load_scientific_data` - Load experimental CSV data
- `get_available_arrays` - Analyze available data fields
- `apply_field_coloring` - Apply appropriate field coloring
- `take_viewport_screenshot` - Document visualization

</MCPDetail>