# ParaView MCP

See and measure 3D scientific data in ParaView, on this machine or on another
one over SSH, in a live view the person can rotate and the agent can drive.

The server has two halves:

- **The front** is plain Python and speaks MCP over stdio. It needs no
  ParaView, so its tools are listed before any ParaView is known.
- **The node half** runs inside ParaView's own Python (`pvpython`) on the
  machine where ParaView is. It holds the scene engine and a
  [trame](https://kitware.github.io/trame/) application on one loopback port
  protected by one token. MCP never crosses the network: the front passes each
  scene tool to the node half as one command over that port.

```
this machine                                   where ParaView is
the MCP host --stdio--> the front --one ssh, one forwarded port--> the node half
                                                                    (pvpython + trame)
```

When ParaView is on this machine there is no `ssh`.

## Configuration

The front reads where ParaView runs from its environment. A connector sets
these from the values the person (or the agent) gives:

| Variable | Meaning |
| --- | --- |
| `PARAVIEW_ADDRESS` | Empty for this machine; otherwise an SSH host name (an alias from `~/.ssh/config` works) |
| `PARAVIEW_VIA` | A jump host, used when `~/.ssh/config` gives the host no `ProxyJump` |
| `PARAVIEW_PREPARE` | A shell line run first, for example `module load paraview`, so that `pvpython` is on the PATH |

`ssh ares; ssh ares-compute-08` is the address `ares-compute-08` via `ares`.
The front runs the system `ssh` with `BatchMode=yes` and your own keys, agent
and config. A host that asks for a password or a one-time code fails at once
with a message saying interactive sign-in is needed; nothing waits.

Test knobs: `PARAVIEW_IDLE_SECONDS` (node half idle shutdown, default 1800),
`PARAVIEW_REMOTE_RENDER_THRESHOLD_MB` (default: ParaView's own setting, 20),
`PARAVIEW_PVSERVER_PORT` (default 11111), `PARAVIEW_START_TIMEOUT`.

## The first attach on a machine

The front prepares `~/.clio/paraview/<server version>/` there (on this machine,
clio-kit's own data folder, `<user data>/clio-kit/paraview/<server version>/`,
which a host that confines the servers it launches lets them write):

- `site/`: trame, trame-vtk and trame-vuetify (pinned in `config.py`),
  installed for the Python of that machine's
  `pvpython` (ParaView 6.0 or later, Python 3.12). If the machine cannot reach
  the package index, the front builds it from the jump host, which shares the
  home folder on a cluster. If neither can, the message names the command to
  run from a machine that can.
- `code/`: the node half, copied over the same connection.
- `env.json`: what the front learned about that `pvpython`, including whether
  it renders offscreen as it is. Login and compute nodes often have no X
  server, EGL or system OSMesa; the front then starts the node half with the
  ParaView launcher's `--mesa` switch, which uses the OSMesa that the Kitware
  binaries ship. A `prepare` line that brings its own GL libraries is probed
  on its own.
- `node.json`: the running node half's port and token, readable only by you.
  A node half answers only inside its own network namespace. When a host runs
  each server in its own sandbox and another sandbox's node half still holds
  the folder, the front leaves it alone and runs its own as
  `node-<namespace>.json`.
- `scenes/<id>/manifest.json`: each scene, written after every committed
  command.

Later attaches reuse the folder and take seconds. A new server version prepares
a new folder (its site may differ) and carries over the saved scenes of the
newest older one. The node half runs detached:
when the link drops it keeps the scene, and with no link for thirty minutes it
writes the scene manifest and exits. If a `pvserver` listens on port 11111 of
that machine, the node half joins it as a client and shows that session.

## Tools

| Tool | What it does |
| --- | --- |
Building a scene is free; managing its view is tooled
([design](docs/scene-scripts.md)).

| Tool | What it does |
| --- | --- |
| `open_view` | Opens a dataset (a glob opens a file series; `growing` follows one still being written), or a view descriptor (`view`), or restores a scene by id, and returns the live view |
| `run_scene_script` | Builds or changes the scene with `paraview.simple` code run inside it: readers (a second dataset too), filters, colours. It is saved with the scene and replayed on restore |
| `set_timestep` | Moves in time or follows the latest step |
| `watch_steps` | Watches a growing source: reports each new step as progress and ends when the run stops or reaches its expected steps (an MCP task where the host supports them) |
| `set_camera`, `fit_camera` | Move the camera |
| `inspect_selection`, `measure_field` | Exact values: what the person picked, a field's statistics and histogram |
| `measure_over_time` | A field's statistics at each time step, the series to plot over a run |
| `capture` | A PNG rendered by ParaView at the camera the view last reported |
| `export_artifact`, `export_animation`, `export_scene` | A figure; a video of the time steps as the view shows them (MP4 with ffmpeg, else OGG); or the scene manifest and a ParaView state file |
| `scene_state` | Where it is attached, the dataset, the pipeline, time and view |

Each tool is one command, applied whole: it commits or it rolls back. A failing
scene script leaves nothing behind. Paths are paths on the machine where
ParaView runs. There is no connect tool: where ParaView runs is the connector's
setting.

## A source that grows

A solver still writing its output (an ADIOS2 BP read with `ADIOS2VTXReader`, or a
file series) opens with `growing`. A view descriptor says the same in one object,
`clio.view-descriptor.v1` ([schema](src/paraview_mcp/schemas/view-descriptor.v1.json)):
the path and reader, `follow` (how often to look and how many steps the run will
write) and `display` (the first view: a slice or contour, coloured by an array over
a fixed range with a colour map). A display part ParaView cannot show is left out
and named in the dataset's note; the open stands.

The server re-reads the reader's time list as the source grows. A view that
follows the latest step moves to each new one; a view the person or agent moved
to an earlier step stays there and says how many newer steps exist, until it
follows again (`set_timestep` with `latest`). Each step reaches the view's model
context, and `watch_steps` reports it as task progress, "Step k of N", so any MCP
host can show a run advancing. When the run ends, `measure_over_time` gives the
statistics of a field at every step.

A growing source can be opened as soon as the run starts. While it is not there
yet, or an ADIOS2 BP5 folder has no whole step on disk, `open_view` waits as long
as the folder keeps changing, and says why it cannot be read once nothing has
changed for 90 seconds. ParaView's ADIOS2 reader takes the process down on a BP5
folder whose index (`md.idx`) names a step that its metadata (`md.0`) does not
hold yet, and waits forever for data that is not on disk. A writer adds data,
metadata and index in that order, but a reader on another machine of a shared
file system can see them land in any order. So the server reads a growing BP5
folder only when its index, metadata, meta-metadata and data agree, and keeps the
steps it has until they do. The reader gets a copy of the folder whose index
holds only the steps whose data is surely on disk: each step once the next one
starts, and the last once the folder stops changing. A run that writes a step
every second is followed one step behind, without waiting for it to pause.

Scene scripts run on ParaView 6, which drops the " (matplotlib)" qualifier from
colour preset names: `ApplyPreset("Viridis (matplotlib)")` applies "Viridis", and
`ColorBy(display, None)` on a display that colours by nothing does nothing.

## The view

`open_view` carries the MCP Apps view `ui://paraview/view`. It is a small page
that frames the trame application through the host's service route and asks
for fullscreen. trame-vtk's `VtkRemoteLocalView` draws locally with vtk.js by
default, so rotating needs no server images; above the remote render
threshold, or when the person chooses, ParaView renders and sends images. The
view reports the time step, the picked point, the pipeline with each object's
representation and colouring, the camera and the recent changes (and whether
the person or the agent made them) through `ui/update-model-context`, at most
ten times a second.

The view carries ParaView's own controls, icon-led and compact. A left tool
rail opens one panel at a time (a sheet over the lower view at phone width):

| Icon | Panel |
| --- | --- |
| Pipeline | Each object, indented under its input: show or hide, select, remove |
| Add filter | Contour, Slice, Clip, Threshold, Glyph, Stream Tracer, Calculator on the selected object, with a short form whose defaults come from its data |
| Display | Representation, colour by field and component, colour map, rescale, legend, opacity |
| Camera | Reset, look along ±X, ±Y or ±Z, parallel projection |
| Information | Data type, points, cells, memory, bounds, each field's range, filter properties |

The top bar holds the dataset name, time (previous, play, next, a slider,
follow latest) and the drawing toggle. Every control that changes the scene
sends an engine command (`create_filter`, `set_representation`, `set_camera`,
...): it commits with its manifest or rolls back and shows why. Edits the person
makes to objects a scene script built are kept on top of the script, so a
restore replays the script and then their edits. The view follows the host's `theme` and
`styles.variables` (background, text, border, font), and draws the render view
on a neutral between the host's background and text; a host that sends none
gets light and dark fallbacks.

Colour maps are part of the scene: the manifest keeps each field's colour map
(preset, range, vector component) and a reopened scene gets it back exactly; a
new scene starts with ParaView's defaults, not the last scene's maps. As in
ParaView, one colour map and one legend belong to a field and are shared by
every object coloured by it. When the browser draws, it shows one legend (the
selected object's, else the newest); images ParaView renders show them all.

The result that opens the view declares the service in its private `_meta`:
`"ai.iowarp.clio/service": {"port": <forwarded port>, "token": <token>}`. The
host proxies the service under the app's own route and adds the token as
`Authorization: Bearer`; the browser and the agent never see it. A host that
does not carry a view's service shows a message, and `capture` still works.

It also declares what a host should replay to reopen the view:
`"ai.iowarp.clio/reopen": {"arguments": {"scene": <scene id>}}`. A reopened view
restores the scene with its filters, representation and time step, even when
the first call started a fresh scene with `new: true`.

## Development

```bash
cd mcp-servers/paraview
uv sync --all-extras --dev
uv run pytest tests/test_front.py                        # no ParaView needed
PARAVIEW_TEST_PVPYTHON=/path/to/pvpython uv run pytest   # the node half too
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

Without a `pvpython` the node-half tests skip with the reason `no_pvpython`.
`tests/browser/run_standin.py` drives the view in headless Chromium through a
stand-in MCP Apps host (`tests/browser/standin.mjs`) and checks rotation,
picking, capture and the reported context. The SSH cluster fixture is the repository's
`tests/fixtures/paraview-cluster/`. It stands for a site, and its pvserver form
starts a pvserver the way a person would; the server itself never does.
