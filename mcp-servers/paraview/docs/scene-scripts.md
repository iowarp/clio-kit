# Scene scripts: freedom to build a scene, tools to manage its view

## Why

A scene was built from fixed tools (`create_filter`, `set_filter_property`,
`set_representation`, `remove_scene_object`). Each new need, such as a hand-set colour
range, a solid colour or a second dataset, became one more tool or argument. That path
leads to dozens of tools, and a long tool list is itself what makes smaller models fail.
When the tools ran out, the agent wrote its own `pvpython` through the shell. Those scripts
ran in another process, outside the scene: the live view, captures and the saved scene
knew nothing of them. In the Helene demo, both videos were made that way.

ParaView's own language for building scenes is `paraview.simple`. The split is:

- **Creating the scene is free:** a script, run inside the scene.
- **Managing its visualization is tooled:** showing it, time, camera, looking and
  measuring, exporting, saving and restoring.

## Creating: `run_scene_script`

`run_scene_script(code, label)` runs Python in the node half: the pvpython process that
owns the scene and serves its live view. The script sees:

| Name | What it is |
| --- | --- |
| `simple` | `paraview.simple` (also as `from paraview.simple import *`) |
| `view` | the scene's render view, the one the person sees |
| `objects` | the scene's objects by id and by label, e.g. `objects["o1"]` |
| `folder` | the scene's own folder, for files the script writes |

**What happens after the script runs:**
- The node reads the pipeline back.
- Every source the script created becomes a scene object (`kind: "made"`), named by its
  pipeline registration name, with the display it was left with: representation,
  colouring, opacity, visibility, solid colour.
- Colour maps are snapshotted as for any commit.
- The view mode is chosen from what is shown: a volume or an image slice is drawn by
  ParaView, and so is a growing source, so each new step arrives as one image.
- The live view redraws the same proxies with no reload.

**Failure:** a script that raises removes every source and view it created, the scene
rolls back to before the step, and the result is `script_error` with the last lines of
the traceback.

**Rules:**
- A script that creates another render view is refused (`script_made_a_view`). It builds
  into `view`.
- Loops over time steps or frames belong to a management tool (`export_animation`), not
  to a script.

**Recording and replay:**
- The step is saved in the manifest: `{kind: "script", code, label, made: [...]}`.
- Restoring a scene replays its scripts in order, finds each made object by name, and
  reapplies the recorded displays on top.
- So the person's later edits in the view's panel survive a restore as edits layered on
  the script's result.
- The change log records the step as one change, by the agent.

**Live data:** `open_view(path, growing=True)` still opens a series that is still being
written. A script can build on `objects["o1"]` (coastlines, isobars, a cloud column), and
ParaView's time-aware pipeline carries the setup to each new step as it lands.

## Managing: the tools that stay

| Tool | Role |
| --- | --- |
| `open_view` | Open a dataset (or a growing series) or restore a scene, and show the live view |
| `run_scene_script` | Build or change the scene: readers, filters, appearance |
| `set_timestep` | Move in time, or follow the latest step |
| `set_camera`, `fit_camera` | Place the camera |
| `inspect_selection`, `measure_field`, `capture`, `scene_state` | Read exact values, look, report |
| `export_artifact` | A still figure |
| `export_animation` | A video of the time steps, as the live view shows them |
| `export_scene` | The manifest and a ParaView state file |

`create_filter`, `set_filter_property`, `set_representation` and `remove_scene_object` leave
the agent's surface. They stay as engine commands, because the person's panel in the view
uses them.

## `export_animation`

`export_animation(path, width, height, fps, start, end, stamp)` renders the scene's own time
steps in the scene's own view, so the video is what the live view shows.

**How it encodes:**
1. It writes frames with `SaveScreenshot`.
2. It encodes them with `ffmpeg` when the ParaView machine has it (H.264 MP4).
3. Otherwise it writes an OGG Theora `.ogv` with ParaView's own writer.

The result names the format and why. A machine with neither is refused, and the frames
are kept.

**Options:** `stamp` puts the step's time in a corner; the colour legend is whatever the
scene shows.

**Durable dashboards:** a dashboard that embeds a live view should carry the exported
video beside it. `clio.mcp-app.v1` takes a `recordingUri`, the video file, which the
web app plays when the live view cannot run: ParaView stopped, an archive, or a shared
copy. It says that the panel is a recording. The ParaView skill says so too.

## Step-by-step series

`hdf5-ptc`'s `write_subset(format="vtk")` rewrites the `.pvd` index after each step,
atomically, so a view opened as `growing` follows the series while it is written.

The order of work for a large series:
1. Write step 1.
2. Open the view.
3. Write the rest in the background.

The person watches from the first seconds.

## Acceptance

All of it starts from a plain request, with no tool named:

1. **The Helene view:** the cloud water seen from above on the map, isobars on top,
   animated through landfall, and a video to send around. The live view draws throughout.
2. **The durable dashboard:** published to a dashboard, then ParaView is stopped and the
   dashboard reopened. The 3D panel plays the recording.
3. **The growing series:** the view shows step 1 within seconds and fills in as steps land.
