---
name: simulation-visualize
description: Use when inspecting and rendering simulation fields with ParaView. Triggers on "isosurface", "slice this volume", "render the field". Not for tabular charts; use results-summary.
metadata:
  bundle: clio-analysis
  servers: clio-paraview
  provenance: designed
  eval-status: scenarios-recorded
---

# Simulation Visualization

ParaView holds a **scene**: a dataset, the filters built on it, how each one is
shown, a time step and a camera. Every scene tool is one command that commits
whole or not at all, and the scene is saved after each commit. Objects have ids
(`o1`, `o2`, ...); name the object you mean instead of relying on "the last one".

## Where ParaView runs

Paths are paths on the machine where ParaView runs, which may not be this one.
The server reaches ParaView on this machine, or over SSH on the host its
configuration names. There is no connect tool: when the person names another
machine, that is a configuration change, not a tool call. If the first scene
call says no ParaView was found, report what it says about where it looked.

## Steps

**1. Open.** `clio-paraview:open_view` with the dataset path. A glob such as
`/data/run/step_*.vti` opens a file series; `growing: true` follows a series
that is still being written. The result names the scene id, every object, its
fields with their exact ranges, and the time steps. Keep the scene id: passing
`scene` to `open_view` later reopens exactly that scene.

**2. Read the fields before choosing a filter.** Field names are exact and the
ranges are in the `open_view` result. An isosurface needs a scalar; streamlines
need a vector.

**3. Find a value worth contouring.** `clio-paraview:measure_field` gives the
exact minimum, maximum, mean, median and a histogram. An isovalue picked
without looking is usually outside the data and gives an empty surface.

**4. Apply the filter that matches the question.**
`clio-paraview:create_filter` takes ParaView's own filter and property names:

| Want | Filter | Properties |
|---|---|---|
| Surface at a constant value | `Contour` | `ContourBy: ["POINTS", field]`, `Isosurfaces: [value]` |
| Cut plane through the volume | `Slice` | the plane's origin and normal |
| Keep cells in a range | `Threshold` | the field and the range |
| Flow paths | `StreamTracer` | a vector field and seeds |
| Displacement by a vector | `WarpByVector` | a vector field |

A wrong property name is refused with the names that exist; nothing changes.
`clio-paraview:remove_scene_object` removes an object and what is built on it.

**5. Show it.** `clio-paraview:set_representation` sets Surface, Wireframe,
Points, Outline or Volume, the field to colour by and a colour preset such as
Viridis. Outline cannot be coloured by a field. A rainbow map invents banding
that is not in the data; a perceptually uniform one does not. See
`chart-select`.

**6. Time.** `clio-paraview:set_timestep` moves to a step or a time, or follows
the latest step of a growing source. The person may park the view on an
earlier step; the scene says when newer steps exist.

**7. Look before you claim.** The person sees a live view and may rotate it or
pick a point. `clio-paraview:capture` renders a PNG at the camera on their
screen; inspect it before describing what is shown.
`clio-paraview:inspect_selection` gives the exact values at the point they
picked. Read numbers from these tools, never from a colour bar or the image.
`clio-paraview:set_camera` and `fit_camera` move the camera.

**8. Keep the result.** `clio-paraview:export_artifact` writes a figure;
`clio-paraview:export_scene` writes the scene manifest and a ParaView state
file. `clio-paraview:scene_state` says where ParaView is attached and what the
scene holds.

## When ParaView is the wrong tool

If the data is a table, this is the wrong server — `clio-plot` reads CSV and
Excel directly. See `results-summary`.

## What not to do

- Do not guess field names; read them from the scene.
- Do not pick an isovalue without `measure_field`.
- Do not use a rainbow colour map for continuous scalar data.
- Do not describe a render you have not captured and looked at.
- Do not read a value off the colour bar; measure it.

## Tool discovery across agents

Names such as `clio-hdf5:open_file` identify a server and its tool in this
guide. Your agent may expose a different prefix. Match the server and tool
against its live MCP inventory, then use the advertised name and input schema.
If a required server is unavailable, report it before attempting the workflow.

## Completion check

Verify opening and rendering through the configured MCP tools; a connection
alone is insufficient. Report the machine, source, time step, filters, field
association (point/cell), units, colour range and the captured image. Inspect
the image before declaring rendering complete; do not infer its colours from
the preset name alone.
