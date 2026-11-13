---
title: Paraview MCP
description: "See and measure 3D scientific data in ParaView, on this machine or over SSH, in a live view."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Paraview"
  icon="🧊"
  category="Data Processing"
  description="See and measure 3D scientific data in ParaView, on this machine or over SSH, in a live view."
  version="3.0.0"
  actions={["open_view", "create_filter", "remove_scene_object", "set_representation", "set_timestep", "set_camera", "fit_camera", "inspect_selection", "measure_field", "capture", "export_artifact", "export_scene", "scene_state"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "ParaView", "trame", "visualization", "scientific", "3D", "SSH"]}
  license="BSD-3-Clause"
  tools={[{"name": "open_view", "description": "Open a dataset or restore a saved scene, and show it in a live ParaView view.\n\nWith neither path nor scene, it shows the ParaView session already running\nwhere ParaView is, or the current scene.", "function_name": "open_view"}, {"name": "create_filter", "description": "Add one filter to the scene's pipeline. It commits whole or not at all.", "function_name": "create_filter"}, {"name": "remove_scene_object", "description": "Remove an object and everything built on it from the pipeline.", "function_name": "remove_scene_object"}, {"name": "set_representation", "description": "Change an object's representation, colouring, opacity or visibility.", "function_name": "set_representation"}, {"name": "set_timestep", "description": "Show another time step, or follow the latest one.", "function_name": "set_timestep"}, {"name": "set_camera", "description": "Place the camera, or turn and zoom it from where it is.", "function_name": "set_camera"}, {"name": "fit_camera", "description": "Point the camera so everything, or one object, fills the view.", "function_name": "fit_camera"}, {"name": "inspect_selection", "description": "Exact field values at the point the person picked in the view.", "function_name": "inspect_selection"}, {"name": "measure_field", "description": "Exact statistics and a histogram of a field at the current time step.", "function_name": "measure_field"}, {"name": "capture", "description": "A PNG rendered by ParaView at the camera the view last reported.", "function_name": "capture"}, {"name": "export_artifact", "description": "Write a figure of the current view as a PNG file.", "function_name": "export_artifact"}, {"name": "export_scene", "description": "Write the scene manifest, and a ParaView state file beside it.", "function_name": "export_scene"}, {"name": "scene_state", "description": "Where ParaView is attached, the dataset, the pipeline, time and view.", "function_name": "scene_state"}]}
>

{/* clio-kit:usage:start */}

### Basic Scientific Data Visualization
```
Load /data/simulation_output.vtk with temperature data, create an isosurface at temperature 300, 
apply Blue to Red color map preset, and take a high-resolution screenshot.
```

### Volume Visualization with Flow Analysis
```
Using /data/fluid_dynamics.bp5, create volume rendering of pressure field with Rainbow color map,
add streamlines to visualize flow patterns, and adjust opacity for better visibility.
```

### Multi-Slice Data Exploration
```
Load /data/medical_scan.vti, get array info to identify density field, create three orthogonal slices 
through the center, color by density field using Viridis preset, and export slices to VTK format.
```

### Advanced ADIOS2/BP5 Analysis
```
Query metadata from /data/checkpoint.bp5 to list available timesteps and variables, 
convert to VTK format, create histogram of temperature distribution, apply threshold filter 
to extract hot regions (>500K), and visualize with appropriate color mapping.
```

### Interactive Camera Control
```
Load /data/molecule.vtk, create isosurface of electron density, rotate camera 45 degrees around Y axis,
zoom in to focus on binding site, set camera position for optimal viewing angle, and save multiple viewpoints.
```

### Native backend configuration

Use a Python interpreter matching the ParaView build, with native modules on `PYTHONPATH` and dependency libraries on `LD_LIBRARY_PATH`. Rendering tools run on the same MCP thread to preserve the graphics context. Headless GLX builds may require Xvfb; the local acceptance test rendered a sphere through native pvserver.

{/* clio-kit:usage:end */}

</MCPDetail>
