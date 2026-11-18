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
  version="3.2.0"
  actions={["open_view", "run_scene_script", "set_timestep", "watch_steps", "set_camera", "fit_camera", "inspect_selection", "measure_field", "measure_over_time", "capture", "export_artifact", "export_animation", "export_scene", "scene_state"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "ParaView", "trame", "visualization", "scientific", "3D", "SSH"]}
  license="BSD-3-Clause"
  tools={[{"name": "open_view", "description": "Open a dataset or restore a saved scene, and show it in a live ParaView view.\n\nWith neither path nor scene, it shows the ParaView session already running\nwhere ParaView is, or the current scene. A growing source is followed: the view\nmoves to each new step unless the person has parked it on one.", "function_name": "open_view"}, {"name": "run_scene_script", "description": "Build or change the scene with a ParaView script: readers (a second dataset too),\nfilters, colours, anything paraview.simple can do. It runs in the scene the person\nsees, so the live view, captures and the saved scene include what it makes. It applies\nwhole or not at all; a failing script leaves nothing behind. The step is saved with the\nscene and replayed when the scene is restored.", "function_name": "run_scene_script"}, {"name": "set_timestep", "description": "Show another time step, or follow the latest one.", "function_name": "set_timestep"}, {"name": "watch_steps", "description": "Report each new time step of the open scene's growing source as it is written,\nas progress: run as a background task, each step is the task's status; called\ndirectly, each step is a progress notification and the call lasts until it ends,\nso give max_seconds. Progress counts the steps written, against the view\ndescriptor's expected_steps when it gives them. It ends once those are written,\nwhen no step comes for idle_seconds, after max_seconds, or when the source is\nnot growing, and returns every step it saw. The view follows the newest step\nunless the person parked it.", "function_name": "watch_steps"}, {"name": "set_camera", "description": "Place the camera, or turn and zoom it from where it is.", "function_name": "set_camera"}, {"name": "fit_camera", "description": "Point the camera so everything, or one object, fills the view.", "function_name": "fit_camera"}, {"name": "inspect_selection", "description": "Exact field values at the point the person picked in the view.", "function_name": "inspect_selection"}, {"name": "measure_field", "description": "Exact statistics and a histogram of a field at the current time step.", "function_name": "measure_field"}, {"name": "measure_over_time", "description": "Exact statistics (min, max, mean, std, median) of a field at each time step: the\nseries to plot or compare over a run. The view stays on its step.", "function_name": "measure_over_time"}, {"name": "capture", "description": "A PNG rendered by ParaView at the camera the view last reported.", "function_name": "capture"}, {"name": "export_artifact", "description": "Write a figure of the current view as a PNG file.", "function_name": "export_artifact"}, {"name": "export_animation", "description": "Write a video of the scene's time steps as the live view shows them: rendered in the\nscene's own view, at its camera, colours and filters. Use it for a clip to send around,\nand put it beside a live view on a dashboard so the dashboard still shows the motion\nwhen ParaView is not running.", "function_name": "export_animation"}, {"name": "export_scene", "description": "Write the scene manifest, and a ParaView state file beside it.", "function_name": "export_scene"}, {"name": "scene_state", "description": "Where ParaView is attached, the dataset, the pipeline, time and view.", "function_name": "scene_state"}]}
>

{/* clio-kit:usage:start */}

### Open and measure

```
Open /data/run42/output.pvd and show an isosurface of pressure at 0.2.
What is the temperature range on that surface?
```

The view opens in a live window you can rotate. The agent reads numbers with
`measure_field` and `inspect_selection` and sees the view with `capture`.

### ParaView on another machine

```
Connect to the ParaView on node08, through login. It needs `module load paraview` first.
Then open /scratch/me/run42/output.pvd.
```

Where ParaView runs is a setting of the server, not a tool: `PARAVIEW_ADDRESS` (an SSH
host), `PARAVIEW_VIA` (a jump host) and `PARAVIEW_PREPARE` (a line run first). The host
must accept your SSH key or agent; a host that asks for a password or a one-time code
fails at once. ParaView 6.0 or later is required there.

### A series still being written

```
Open /scratch/me/run42/*.vti as a growing series and follow the latest step.
```

### In CLIO

In CLIO this server is the ParaView connector: enable it in Settings, Extensions, and
the agent can set the three values itself. See
[Connectors](https://clio.iowarp.ai/docs/connectors/#paraview) and
[Apps](https://clio.iowarp.ai/docs/apps/).

{/* clio-kit:usage:end */}

</MCPDetail>
