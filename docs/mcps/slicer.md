---
title: Slicer MCP
description: "Slice 3D models (STL, 3MF, STEP) with the PrusaSlicer or OrcaSlicer installed here, and see the toolpaths layer by layer."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Slicer"
  icon="🖨️"
  category="Utilities"
  description="Slice 3D models (STL, 3MF, STEP) with the PrusaSlicer or OrcaSlicer installed here, and see the toolpaths layer by layer."
  version="1.0.0"
  actions={["list_printers", "slice", "layer_stats", "export", "compare_slices", "show_slice", "slice_layers", "remember_slice_view"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "3D printing", "slicer", "G-code", "PrusaSlicer", "OrcaSlicer", "toolpaths"]}
  license="BSD-3-Clause"
  tools={[{"name": "list_printers", "description": "List the printer presets the installed slicer knows. With no argument: its vendors. With vendor: that vendor's printer models and presets. With printer (words such as 'Prusa MK4' or an exact preset name): the matching preset and its quality (layer height) and material presets.", "function_name": "list_printers"}, {"name": "slice", "description": "Slice a 3D model (STL, 3MF, STEP, OBJ) for a printer with the slicer installed here. printer, material and quality are words ('Prusa MK4', 'PLA', '0.2 mm') or exact preset names; the presets chosen are reported. overrides sets slicer settings by their own names (support_material: true, fill_density: '20%', brim_width: 5); rotate turns the model in degrees about x, y, z before slicing. Returns a slice id, print time, filament in grams and metres, layer count, and the slicer's warnings in words. The G-code stays with the server; export writes it out.", "function_name": "slice"}, {"name": "layer_stats", "description": "Time and filament of a slice by feature type (outer wall, inner wall, infill, support, travel \u2026) over a range of layers (all when omitted), the slowest layer, and one row per layer (layer, z_mm, time_s, filament_g, the types it prints and where on the bed each prints).", "function_name": "layer_stats"}, {"name": "export", "description": "Write a slice out: format 'gcode' (the plain-text G-code for the printer) or 'csv' (one row per layer: z_mm, height_mm, time_s, filament_g and time per feature type, for a chart). path defaults to the model's folder.", "function_name": "export"}, {"name": "compare_slices", "description": "Compare two or more slices: print time, filament, layers, presets, overrides and where the time goes, each against the first.", "function_name": "compare_slices"}, {"name": "show_slice", "description": "Open a slice in an interactive view the person can turn and zoom: the toolpaths layer by layer, coloured by feature type (walls, infill, support, travel), with a layer slider, a toggle per type, the print time, filament (grams and metres), layer count and height, and the slicer's warnings. The view reports the shown layer as layer and z_mm and the person's pick as selection; its tools go_to_layer, set_feature_visibility and pick_segment move it: to show another layer in a view that is already open, call go_to_layer rather than opening another view. Pass layer to open on a layer.", "function_name": "show_slice"}, {"name": "slice_layers", "description": "Toolpaths of a range of layers for the slice view to draw. Called by the view itself.", "function_name": "slice_layers"}, {"name": "remember_slice_view", "description": "Save the layer and the feature types the slice view shows, so reopening it shows them again. Called by the view itself.", "function_name": "remember_slice_view"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `list_printers`, `slice`, `layer_stats`, `export`, `compare_slices`, `show_slice`, `slice_layers`, `remember_slice_view`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
