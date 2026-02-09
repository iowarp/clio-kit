---
title: ParaView MCP
description: "ParaView MCP v1.0.0 - Part of CLIO Kit (IoWarp Platform). 29 tools for scientific 3D visualization: load scientific data, generate isosurfaces, create slices, volume rendering, flow streamlines, color mapping, histogram analysis, ADIOS2/BP5 support. Enables AI agents to create autonomous scientific visualizations."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail 
  name="ParaView"
  icon="🔬"
  category="Scientific Visualization"
  description="ParaView MCP v1.0.0 - Part of CLIO Kit (IoWarp Platform). 29 tools for scientific 3D visualization: load scientific data, generate isosurfaces, create slices, volume rendering, flow streamlines, color mapping, histogram analysis, ADIOS2/BP5 support. Enables AI agents to create autonomous scientific visualizations."
  version="1.0.0"
  actions={["load_scientific_data", "generate_isosurface", "create_data_slice", "configure_volume_display", "generate_flow_streamlines", "take_viewport_screenshot", "apply_field_coloring", "set_color_map", "set_color_map_preset", "get_histogram", "adjust_volume_opacity", "query_adios2_metadata", "convert_bp5_to_vtk", "export_data", "get_data_info", "list_arrays", "get_array_range", "apply_threshold_filter", "apply_clip_filter", "apply_calculator", "apply_contour", "apply_warp_by_vector", "toggle_object_visibility", "set_background_color", "set_representation", "rotate_camera", "reset_camera", "set_camera_position", "adjust_camera_zoom"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["paraview", "scientific-visualization", "3d-visualization", "adios2", "bp5", "vtk", "scientific-computing", "visualization-pipeline", "autonomous-visualization", "hpc-visualization", "color-mapping", "histogram-analysis"]}
  license="BSD-3-Clause"
  tools={[
    {"name": "load_scientific_data", "description": "Load scientific datasets from various file formats into ParaView for visualization and analysis. Supports VTK, EXODUS, CSV, RAW, BP5/ADIOS2, and other scientific data formats.", "function_name": "load_scientific_data"},
    {"name": "generate_isosurface", "description": "Create an isosurface visualization for extracting surfaces of constant value from 3D scalar data.", "function_name": "generate_isosurface"},
    {"name": "create_data_slice", "description": "Create a slice through volume data to examine cross-sections of 3D datasets. Supports X, Y, Z planes.", "function_name": "create_data_slice"},
    {"name": "configure_volume_display", "description": "Toggle volume rendering for direct 3D visualization of volumetric data.", "function_name": "configure_volume_display"},
    {"name": "generate_flow_streamlines", "description": "Create streamlines from vector field data for visualizing flow patterns and trajectories.", "function_name": "generate_flow_streamlines"},
    {"name": "take_viewport_screenshot", "description": "Capture high-resolution screenshots of the current ParaView viewport.", "function_name": "take_viewport_screenshot"},
    {"name": "apply_field_coloring", "description": "Color the visualization by a specific data field for enhanced analysis.", "function_name": "apply_field_coloring"},
    {"name": "set_color_map", "description": "Set custom color map (lookup table) for data visualization with configurable value ranges.", "function_name": "set_color_map"},
    {"name": "set_color_map_preset", "description": "Apply preset color maps: Rainbow, Blue to Red White, Cool to Warm, Viridis, Plasma, Inferno, Turbo, Jet, Grayscale, X Ray, Black-Body Radiation.", "function_name": "set_color_map_preset"},
    {"name": "get_histogram", "description": "Get histogram data for field values with automatic binning and distribution statistics.", "function_name": "get_histogram"},
    {"name": "adjust_volume_opacity", "description": "Edit volume rendering opacity transfer function for better visibility control.", "function_name": "adjust_volume_opacity"},
    {"name": "query_adios2_metadata", "description": "Query metadata from ADIOS2/BP5 files including variables, timesteps, and attributes.", "function_name": "query_adios2_metadata"},
    {"name": "convert_bp5_to_vtk", "description": "Convert BP5/ADIOS2 files to VTK format for broader compatibility.", "function_name": "convert_bp5_to_vtk"},
    {"name": "export_data", "description": "Export data to VTK, CSV, or other supported formats.", "function_name": "export_data"},
    {"name": "get_data_info", "description": "Get detailed information about loaded datasets including bounds, cells, points.", "function_name": "get_data_info"},
    {"name": "list_arrays", "description": "List all available data arrays in the current dataset with type information.", "function_name": "list_arrays"},
    {"name": "get_array_range", "description": "Get value range (min/max) for a specific data array.", "function_name": "get_array_range"},
    {"name": "apply_threshold_filter", "description": "Apply threshold filter to extract data within specified value range.", "function_name": "apply_threshold_filter"},
    {"name": "apply_clip_filter", "description": "Clip data using planes or other geometric shapes.", "function_name": "apply_clip_filter"},
    {"name": "apply_calculator", "description": "Create derived fields using mathematical expressions on existing data arrays.", "function_name": "apply_calculator"},
    {"name": "apply_contour", "description": "Create contours at specific data values for level-set visualization.", "function_name": "apply_contour"},
    {"name": "apply_warp_by_vector", "description": "Warp geometry using vector field data for displacement visualization.", "function_name": "apply_warp_by_vector"},
    {"name": "toggle_object_visibility", "description": "Show or hide visualization objects in the pipeline.", "function_name": "toggle_object_visibility"},
    {"name": "set_background_color", "description": "Set viewport background color for better contrast and presentation.", "function_name": "set_background_color"},
    {"name": "set_representation", "description": "Change visualization representation: Surface, Wireframe, Points, Volume.", "function_name": "set_representation"},
    {"name": "rotate_camera", "description": "Rotate camera around center of rotation to adjust viewing perspective.", "function_name": "rotate_camera"},
    {"name": "reset_camera", "description": "Reset camera to show all data with optimal viewing parameters.", "function_name": "reset_camera"},
    {"name": "set_camera_position", "description": "Set specific camera position, focal point, and view up vector.", "function_name": "set_camera_position"},
    {"name": "adjust_camera_zoom", "description": "Adjust camera zoom level for detailed or overview examination.", "function_name": "adjust_camera_zoom"}
  ]}
>

### 1. Basic Scientific Data Visualization
```
Load /data/simulation_output.vtk with temperature data, create an isosurface at temperature 300, 
apply Blue to Red color map preset, and take a high-resolution screenshot.
```

**Tools called:**
- `load_scientific_data` - Load VTK simulation data
- `generate_isosurface` - Create isosurface at temperature 300
- `set_color_map_preset` - Apply "Blue to Red White" preset
- `take_viewport_screenshot` - Capture visualization

### 2. Volume Visualization with Flow Analysis
```
Using /data/fluid_dynamics.bp5, create volume rendering of pressure field with Rainbow color map, 
add streamlines to visualize flow patterns, and adjust opacity for better visibility.
```

**Tools called:**
- `load_scientific_data` - Load ADIOS2/BP5 fluid dynamics data
- `configure_volume_display` - Enable volume rendering
- `set_color_map_preset` - Apply "Rainbow" preset
- `generate_flow_streamlines` - Create flow streamlines
- `adjust_volume_opacity` - Edit opacity transfer function
- `take_viewport_screenshot` - Document result

### 3. Multi-Slice Data Exploration
```
Load /data/medical_scan.vti, get array info to identify density field, create three orthogonal slices 
through the center, color by density field using Viridis preset, and export slices to VTK format.
```

**Tools called:**
- `load_scientific_data` - Load medical imaging data
- `list_arrays` - Get available data arrays
- `create_data_slice` - Create XY, XZ, YZ slices
- `apply_field_coloring` - Color by density
- `set_color_map_preset` - Apply "Viridis" preset
- `export_data` - Export to VTK format
- `take_viewport_screenshot` - Capture multi-slice view

### 4. Advanced ADIOS2/BP5 Analysis
```
Query metadata from /data/checkpoint.bp5 to list available timesteps and variables, convert to VTK format, 
create histogram of temperature distribution, apply threshold filter to extract hot regions (>500K), 
and visualize with appropriate color mapping.
```

**Tools called:**
- `query_adios2_metadata` - Query BP5 file metadata
- `convert_bp5_to_vtk` - Convert to VTK format
- `load_scientific_data` - Load converted data
- `get_histogram` - Analyze temperature distribution
- `apply_threshold_filter` - Extract hot regions
- `set_color_map_preset` - Apply "Inferno" preset
- `take_viewport_screenshot` - Document analysis

### 5. Comparative Isosurface Analysis
```
From /data/pressure_field.vtk, get array range for pressure, create multiple isosurfaces at pressure 
values 100, 200, and 300 as wireframe for comparison.
```

**Tools called:**
- `load_scientific_data` - Load pressure field data
- `get_array_range` - Get pressure value range
- `generate_isosurface` - Create multiple isosurfaces
- `set_representation` - Set wireframe representation
- `apply_field_coloring` - Apply pressure-based coloring
- `set_color_map_preset` - Apply "Cool to Warm" preset

### 6. Interactive Camera Control
```
Load /data/molecule.vtk, create isosurface of electron density, rotate camera 45 degrees around Y axis, 
zoom in to focus on binding site, set camera position for optimal viewing angle, and save multiple viewpoints.
```

**Tools called:**
- `load_scientific_data` - Load molecular data
- `generate_isosurface` - Create electron density isosurface
- `rotate_camera` - Rotate 45° around Y axis
- `adjust_camera_zoom` - Zoom to binding site
- `set_camera_position` - Set optimal view
- `take_viewport_screenshot` - Save multiple angles

### 7. Advanced Filtering Pipeline
```
Load /data/turbulent_flow.bp5, examine available fields, create a calculator to compute velocity magnitude, 
apply clip filter to focus on region of interest, warp by velocity vector, and visualize results.
```

**Tools called:**
- `load_scientific_data` - Load turbulent flow simulation
- `list_arrays` - Examine available data fields
- `apply_calculator` - Compute derived velocity magnitude
- `apply_clip_filter` - Focus on specific region
- `apply_warp_by_vector` - Warp by velocity
- `set_color_map_preset` - Apply "Plasma" preset
- `take_viewport_screenshot` - Document pipeline result

### 8. Data Quality Assessment
```
Load /data/experimental_results.csv, check data info and available fields, create histogram to assess 
distribution, toggle visibility of different components, and create the most appropriate 3D visualization.
```

**Tools called:**
- `load_scientific_data` - Load experimental CSV data
- `get_data_info` - Get dataset information
- `list_arrays` - Analyze available data fields
- `get_histogram` - Assess data distribution
- `toggle_object_visibility` - Control component visibility
- `apply_field_coloring` - Apply appropriate field coloring
- `set_background_color` - Set presentation background
- `take_viewport_screenshot` - Document visualization

</MCPDetail>