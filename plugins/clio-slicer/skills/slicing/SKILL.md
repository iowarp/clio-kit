---
name: slicing
description: 'Use when a 3D model (STL, 3MF, STEP, OBJ) is to be sliced for a 3D printer, or the person asks how long a print takes, how much filament it uses, whether it needs supports, or to see its layers. Triggers on "slice this for my printer", "how long will it take to print", "show me the layers", "does it need supports". Not for slicing a simulation volume; use simulation-visualize.'
metadata:
  bundle: clio-fabrication
  servers: clio-slicer
  provenance: designed
  eval-status: scenarios-recorded
---

# Slicing for 3D printing

The `clio-slicer` server runs the PrusaSlicer or OrcaSlicer already installed on this
machine. It never downloads a slicer and never talks to a printer: you slice, read the
numbers and show the layers; the person prints.

If a tool answers `slicer_missing`, tell the person in their words what it says: no
supported slicer was found, they can install PrusaSlicer or OrcaSlicer, or point
`SLICER_BIN` at the one they have. Do not look for another way to slice.

## Choose the presets

`slice` takes the printer, material and quality as the person says them ("a Prusa MK4",
"PLA", "0.2 mm") and picks the installed slicer's own presets with the vendor's rules.
Its result names the presets it chose, how each word matched and the close
alternatives. Say which presets were used. If the words matched more than one printer
that differ in a way that matters (nozzle size, input shaper), say which was taken and
offer the others.

`list_printers` with `printer` shows a printer's qualities (layer heights) and
materials; with `vendor` a vendor's printers. Use it when the person names something the
slice does not match, or asks what is available. Never guess a preset name.

## Orientation and supports

- Read the slicer's warnings in the slice result and pass them on in words. "Long
  bridging extrusions" or "consider enabling supports" means an unsupported span.
- A flat, wide face down prints best; `rotate` turns the model before slicing
  (degrees about x, y, z). A part that leaves the bed or is too tall is reported as a
  warning; rotate or scale it rather than ignore it.
- Supports, brim, walls and infill are `overrides` with the slicer's own setting names,
  for example `support_material: true`, `support_material_auto: true`, `brim_width: 5`,
  `perimeters: 3`, `fill_density: "20%"`. An unknown name is refused with the close
  names; nothing is guessed.
- When the person weighs choices (supports or not, two orientations, two qualities),
  slice each and compare them with `compare_slices`: it reports time, filament and where
  the time goes for each, against the first.

## Read the estimates

- Print time and filament (grams and metres) are the slicer's own numbers. Quote them
  as estimates from that slicer and preset.
- `layer_stats` gives time and filament by feature type (outer wall, inner wall,
  sparse and solid infill, top surface, bridge, support, travel) over a layer range, the
  slowest layer and one row per layer with the types it prints. Per-layer and per-type
  times are feedrate estimates scaled to the slicer's total; say so if the person relies
  on them.

## Check a layer

- `show_slice` opens the layer view where the host shows MCP Apps views: toolpaths
  coloured by type with a named legend, a layer slider, the time and filament, and the
  warnings. It reports the shown layer as `layer` and `z_mm`, the drawn types as
  `visible_features` and the person's pick as `selection`. A host without views gets
  the same numbers as text.
- To show a feature at a height ("where the roof starts", "the first layer of the
  overhang"), find it in the numbers first: the per-layer rows of `layer_stats` say
  which types each layer prints, so the first layer with `bridge_infill` above a cavity
  is where a roof starts. A part can have several spans (a hole, a channel, a window):
  each row's `extents` say where on the bed each type prints, so match the span to the
  part the person means. If the host lets you look at an open view (CLIO's
  `look_at_view`), look at the layer as the person sees it. Then move the view the
  person has open with its own tool `go_to_layer` (`layer`, or `z_mm` for a height);
  `show_slice` with `layer` is for when no view is open. Say which layer and Z it is.
- The view's other tools: `set_feature_visibility` hides or shows types (travel
  included); `pick_segment` picks the toolpath nearest a point on the bed and reports
  its type, Z and G-code line.
- Describe what a layer prints from the numbers and the view's reported state, never
  from colours in an image.

## A dashboard

In a host with dashboards (CLIO), the view can sit beside a chart: a `clio.mcp-app.v1`
component with server `clio-slicer`, tool `show_slice` and arguments `{slice_id}`, linked
to the chart by the state channel `{key: "layer", path: "/selection/layer", tool:
"go_to_layer"}`. For the chart, `export` with format `csv` writes one row per layer
(`layer`, `z_mm`, `time_s`, `filament_g` and time per type) into the person's folder;
chart `time_s` against `layer` from that file, with `layer` as the selection field.

## Files

`export` writes the plain-text G-code (format `gcode`) or the per-layer CSV next to the
model unless the person names a path. The G-code is the printer's file: hand it to the
person; do not send it anywhere.
