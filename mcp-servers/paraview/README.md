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

- `site/`: trame and trame-vtk, installed for the Python of that machine's
  `pvpython` (ParaView 6.0 or later, Python 3.12). If the machine cannot reach
  the package index, the front builds it from the jump host, which shares the
  home folder on a cluster. If neither can, the message names the command to
  run from a machine that can.
- `code/`: the node half, copied over the same connection.
- `node.json`: the running node half's port and token, readable only by you.
- `scenes/<id>/manifest.json`: each scene, written after every committed
  command.

Later attaches reuse the folder and take seconds. The node half runs detached:
when the link drops it keeps the scene, and with no link for thirty minutes it
writes the scene manifest and exits. If a `pvserver` listens on port 11111 of
that machine, the node half joins it as a client and shows that session.

## Tools

| Tool | What it does |
| --- | --- |
| `open_view` | Opens a dataset (a glob opens a file series; `growing` follows one still being written), or restores a scene by id, and returns the live view |
| `create_filter`, `remove_scene_object` | Change the pipeline |
| `set_representation`, `set_timestep` | Change what is shown |
| `set_camera`, `fit_camera` | Move the camera |
| `inspect_selection`, `measure_field` | Exact values: what the person picked, a field's statistics and histogram |
| `capture` | A PNG rendered by ParaView at the camera the view last reported |
| `export_artifact`, `export_scene` | A figure, or the scene manifest and a ParaView state file |
| `scene_state` | Where it is attached, the dataset, the pipeline, time and view |

Each tool is one command, applied whole: it commits or it rolls back. Paths
are paths on the machine where ParaView runs. There is no connect tool: where
ParaView runs is the connector's setting.

## The view

`open_view` carries the MCP Apps view `ui://paraview/view`. It is a small page
that frames the trame application through the host's service route and asks
for fullscreen. trame-vtk's `VtkRemoteLocalView` draws locally with vtk.js by
default, so rotating needs no server images; above the remote render
threshold, or when the person chooses, ParaView renders and sends images. The
view reports the time step, the picked point, the representation and the
camera through `ui/update-model-context`, at most ten times a second.

The result that opens the view declares the service in its private `_meta`:
`"ai.iowarp.clio/service": {"port": <forwarded port>, "token": <token>}`. The
host proxies the service under the app's own route and adds the token as
`Authorization: Bearer`; the browser and the agent never see it. A host that
does not carry a view's service shows a message, and `capture` still works.

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
picking, capture and the reported context. The SSH cluster fixture is in
`cluster/`.
